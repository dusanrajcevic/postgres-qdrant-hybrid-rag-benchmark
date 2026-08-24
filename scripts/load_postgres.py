from __future__ import annotations

import argparse
from datetime import datetime

import numpy as np
import pyarrow.parquet as pq
import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector

from common import PROCESSED, load_config, write_json

def pg_conn(cfg):
    p = cfg["postgres"]
    return psycopg.connect(
        host=p["host"], port=p["port"], dbname=p["dbname"],
        user=p["user"], password=p["password"]
    )

def load_documents(conn, size: int, chunks_per_doc: int) -> int:
    n_docs = (size + chunks_per_doc - 1) // chunks_per_doc
    pf = pq.ParquetFile(PROCESSED / "documents.parquet")
    loaded = 0
    with conn.cursor() as cur:
        with cur.copy("""
            COPY documents
            (document_id, tenant_id, owner_id, category_id, language, status, created_at)
            FROM STDIN WITH (FORMAT BINARY)
        """) as copy:
            copy.set_types(["bigint", "integer", "integer", "integer", "text", "text", "timestamptz"])
            for batch in pf.iter_batches(batch_size=10000):
                rows = batch.to_pylist()
                for r in rows:
                    if int(r["document_id"]) >= n_docs:
                        break
                    copy.write_row([
                        int(r["document_id"]),
                        int(r["tenant_id"]),
                        int(r["owner_id"]),
                        int(r["category_id"]),
                        r["language"],
                        r["status"],
                        datetime.fromisoformat(r["created_at"].replace("Z", "+00:00")),
                    ])
                    loaded += 1
                if loaded >= n_docs:
                    break
    conn.commit()
    return loaded

def load_chunks(conn, size: int) -> int:
    emb = np.load(PROCESSED / "passage_embeddings.npy", mmap_mode="r")
    pf = pq.ParquetFile(PROCESSED / "passages.parquet")
    chunks_per_doc = int(load_config()["chunks_per_document"])
    loaded = 0
    with conn.cursor() as cur:
        with cur.copy("""
            COPY chunks (chunk_id, document_id, text, embedding)
            FROM STDIN WITH (FORMAT BINARY)
        """) as copy:
            copy.set_types(["bigint", "bigint", "text", "vector"])
            for batch in pf.iter_batches(batch_size=4096, columns=["sample_rank", "chunk_id", "text"]):
                ranks = batch.column("sample_rank").to_numpy(zero_copy_only=False).astype(np.int64)
                pids = batch.column("chunk_id").to_numpy(zero_copy_only=False).astype(np.int64)
                texts = batch.column("text").to_pylist()
                for rank, pid, text in zip(ranks, pids, texts):
                    rank = int(rank)
                    if rank >= size:
                        continue
                    copy.write_row([
                        int(pid),
                        rank // chunks_per_doc,
                        text,
                        Vector(emb[rank]),
                    ])
                    loaded += 1
                    if loaded % 25000 == 0:
                        print(f"\rPostgreSQL chunks loaded: {loaded:,}/{size:,}", end="", flush=True)
    print()
    conn.commit()
    return loaded

def load_permissions(conn, size: int, chunks_per_doc: int) -> int:
    n_docs = (size + chunks_per_doc - 1) // chunks_per_doc
    pf = pq.ParquetFile(PROCESSED / "permissions.parquet")
    loaded = 0
    with conn.cursor() as cur:
        with cur.copy("""
            COPY document_permissions (document_id, principal_id, permission_type)
            FROM STDIN WITH (FORMAT BINARY)
        """) as copy:
            copy.set_types(["bigint", "integer", "text"])
            for batch in pf.iter_batches(batch_size=10000):
                for r in batch.to_pylist():
                    if int(r["document_id"]) >= n_docs:
                        break
                    copy.write_row([
                        int(r["document_id"]),
                        int(r["principal_id"]),
                        r["permission_type"],
                    ])
                    loaded += 1
                if loaded >= n_docs:
                    break
    conn.commit()
    return loaded

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, required=True)
    args = ap.parse_args()

    cfg = load_config()
    allowed = set(map(int, cfg["dataset_sizes"]))
    if args.size not in allowed:
        raise SystemExit(f"--size must be one of {sorted(allowed)}")
    chunks_per_doc = int(cfg["chunks_per_document"])

    with pg_conn(cfg) as conn:
        register_vector(conn)
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute("CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE")
        # Never ingest through an ANN index: later index construction must be timed separately.
        conn.execute("DROP INDEX IF EXISTS chunks_embedding_hnsw")
        conn.execute("DROP INDEX IF EXISTS chunks_embedding_diskann")
        conn.execute("TRUNCATE document_permissions, chunks, documents CASCADE")
        conn.commit()

        docs = load_documents(conn, args.size, chunks_per_doc)
        chunks = load_chunks(conn, args.size)
        perms = load_permissions(conn, args.size, chunks_per_doc)
        conn.execute("ANALYZE documents")
        conn.execute("ANALYZE chunks")
        conn.execute("ANALYZE document_permissions")
        conn.commit()

        counts = conn.execute("""
            SELECT
              (SELECT count(*) FROM documents),
              (SELECT count(*) FROM chunks),
              (SELECT count(*) FROM document_permissions)
        """).fetchone()

    expected_docs = (args.size + chunks_per_doc - 1) // chunks_per_doc
    if chunks != args.size or docs != expected_docs or perms != expected_docs:
        raise RuntimeError(
            f"Load mismatch: docs={docs}, chunks={chunks}, perms={perms}; "
            f"expected docs/perms={expected_docs}, chunks={args.size}"
        )

    manifest = {
        "dataset_size": args.size,
        "documents_loaded": docs,
        "chunks_loaded": chunks,
        "permissions_loaded": perms,
        "database_counts": list(map(int, counts)),
        "ann_index_present": False,
    }
    write_json(PROCESSED / f"postgres_load_{args.size}.json", manifest)
    print(manifest)
    print("PostgreSQL base-data load complete. No ANN index was built.")

if __name__ == "__main__":
    main()
