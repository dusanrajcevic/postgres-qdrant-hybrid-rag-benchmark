from __future__ import annotations
import argparse, json, time
import psycopg
from qdrant_client import QdrantClient, models
from common import ROOT, load_config, write_json

TUNING_DIR = ROOT / "tuning"

def pg_conn(cfg):
    p = cfg["postgres"]
    return psycopg.connect(
        host=p["host"], port=p["port"], dbname=p["dbname"],
        user=p["user"], password=p["password"]
    )

def assert_pg_size(conn, size):
    n = int(conn.execute("SELECT count(*) FROM chunks").fetchone()[0])
    if n != size:
        raise RuntimeError(f"PostgreSQL has {n:,} chunks, expected {size:,}")

def assert_filter_columns(conn):
    bad = int(conn.execute("""
        SELECT count(*) FROM chunks
        WHERE labels IS NULL OR cardinality(labels) <> 5
    """).fetchone()[0])
    if bad:
        raise RuntimeError(
            f"{bad:,} chunks do not have prepared labels. "
            "Run prepare_filtered_schema.py first."
        )

def build_pg_hnsw(cfg, size):
    with pg_conn(cfg) as conn:
        assert_pg_size(conn, size)
        assert_filter_columns(conn)
        conn.execute("DROP INDEX IF EXISTS chunks_embedding_diskann")
        conn.execute("DROP INDEX IF EXISTS chunks_embedding_hnsw")
        conn.commit()
        # Conservative serial tuning build for Docker Desktop's 1 GB shm.
        conn.execute("SET maintenance_work_mem = '512MB'")
        conn.execute("SET max_parallel_maintenance_workers = 0")
        t0 = time.perf_counter()
        conn.execute("""
            CREATE INDEX chunks_embedding_hnsw
            ON chunks USING hnsw (embedding vector_cosine_ops)
            WITH (m=16, ef_construction=64)
        """)
        conn.commit()
        sec = time.perf_counter() - t0
        b = int(conn.execute(
            "SELECT pg_relation_size('chunks_embedding_hnsw')"
        ).fetchone()[0])
    return sec, b

def build_pg_diskann(cfg, size):
    with pg_conn(cfg) as conn:
        assert_pg_size(conn, size)
        assert_filter_columns(conn)
        conn.execute("DROP INDEX IF EXISTS chunks_embedding_hnsw")
        conn.execute("DROP INDEX IF EXISTS chunks_embedding_diskann")
        conn.commit()
        conn.execute("SET maintenance_work_mem = '512MB'")
        # Labeled DiskANN builds are intentionally not forced parallel.
        conn.execute("SELECT set_config('diskann.force_parallel_workers','-1',false)")
        t0 = time.perf_counter()
        conn.execute("""
            CREATE INDEX chunks_embedding_diskann
            ON chunks USING diskann (embedding vector_cosine_ops, labels)
        """)
        conn.commit()
        sec = time.perf_counter() - t0
        b = int(conn.execute(
            "SELECT pg_relation_size('chunks_embedding_diskann')"
        ).fetchone()[0])
    return sec, b

def build_qdrant(cfg, size):
    q = cfg["qdrant"]
    c = QdrantClient(url=q["url"], timeout=120)
    name = q["collection"]
    n = int(c.count(collection_name=name, exact=True).count)
    if n != size:
        raise RuntimeError(f"Qdrant has {n:,} points, expected {size:,}")
    info = c.get_collection(name)
    if int(info.config.hnsw_config.m) != 0:
        raise RuntimeError(
            "Qdrant HNSW is already enabled. For a clean tuning build, "
            "reload Qdrant for this size first."
        )
    t0 = time.perf_counter()
    c.update_collection(
        collection_name=name,
        hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100),
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=20000),
    )
    deadline = time.monotonic() + 3600
    while True:
        info = c.get_collection(name)
        indexed = int(info.indexed_vectors_count or 0)
        status = str(info.status).lower()
        print(f"\rQdrant indexing: {status}, {indexed:,}/{size:,}", end="", flush=True)
        if "green" in status and indexed >= int(size * 0.99):
            break
        if time.monotonic() > deadline:
            raise TimeoutError("Qdrant indexing did not finish within one hour.")
        time.sleep(2)
    print()
    return time.perf_counter() - t0, indexed, status

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--engine", choices=["pg-hnsw","pg-diskann","qdrant"], required=True)
    args = ap.parse_args()
    cfg = load_config()
    TUNING_DIR.mkdir(parents=True, exist_ok=True)

    if args.engine == "pg-hnsw":
        sec,b = build_pg_hnsw(cfg,args.size)
        r = {
            "purpose":"TUNING BUILD ONLY - NOT PAPER RESULT",
            "engine":args.engine, "dataset_size":args.size,
            "build_seconds":sec, "index_bytes":b,
            "build_parameters":{"m":16,"ef_construction":64},
            "filter_metadata_prepared":True
        }
    elif args.engine == "pg-diskann":
        sec,b = build_pg_diskann(cfg,args.size)
        r = {
            "purpose":"TUNING BUILD ONLY - NOT PAPER RESULT",
            "engine":args.engine, "dataset_size":args.size,
            "build_seconds":sec, "index_bytes":b,
            "build_parameters":{
                "storage_layout":"memory_optimized default",
                "num_neighbors":50,
                "construction_search_list_size":100,
                "max_alpha":1.2,
                "labels_column":"labels",
                "parallel_build_forced":False
            },
            "filter_metadata_prepared":True,
            "important":"This labeled DiskANN index is the index configuration intended for W1-W3."
        }
    else:
        sec,indexed,status = build_qdrant(cfg,args.size)
        r = {
            "purpose":"TUNING BUILD ONLY - NOT PAPER RESULT",
            "engine":args.engine, "dataset_size":args.size,
            "build_seconds":sec, "indexed_vectors_count":indexed,
            "status":status,
            "build_parameters":{"m":16,"ef_construct":100,"indexing_threshold_kb":20000}
        }
    out = TUNING_DIR / f"build_{args.engine}_{args.size}.json"
    write_json(out,r)
    print(json.dumps(r,indent=2))
    print(f"Wrote {out}")

if __name__ == "__main__":
    main()
