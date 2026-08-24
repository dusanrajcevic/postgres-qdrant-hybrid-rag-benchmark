from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from qdrant_client import QdrantClient, models

from common import GROUND_TRUTH, PROCESSED, ROOT, load_config, write_json

FROZEN_PATH = ROOT / "config" / "final_100k_measurement.json"
RESULTS_ROOT = ROOT / "results" / "measurements" / "100000"
WORKLOADS = ["w1","w2_50","w2_10","w2_1","w2_0_1","w3","w4_acl"]
EXACT_PATHS = {"scalar_first_exact", "scalar_acl_exact", "filtered_exact"}

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def recall10(ids, truth):
    return len(set(map(int, ids)) & set(map(int, truth))) / 10.0

def pg_conn(cfg):
    p = cfg["postgres"]
    c = psycopg.connect(
        host=p["host"], port=p["port"], dbname=p["dbname"],
        user=p["user"], password=p["password"]
    )
    register_vector(c)
    return c

def ann_indexes(c):
    return [r[0] for r in c.execute("""
        SELECT indexname
        FROM pg_indexes
        WHERE tablename='chunks'
          AND indexname IN ('chunks_embedding_hnsw','chunks_embedding_diskann')
        ORDER BY indexname
    """).fetchall()]

def load_truths(size):
    out = {}
    for workload in WORKLOADS:
        name = "w4_acl" if workload == "w4_acl" else workload
        p = GROUND_TRUTH / str(size) / f"{name}.npz"
        if not p.exists():
            raise SystemExit(f"Missing ground truth: {p}")
        z = np.load(p)
        out[workload] = z["top10_ids"]
    return out

def scalar_filter(workload, qrank):
    if workload == "w2_50":
        return "language = %s", ["en" if qrank % 2 == 0 else "de"]
    if workload == "w2_10":
        return "category_id = %s", [qrank % 10]
    if workload == "w2_1":
        return "tenant_id = %s", [1 + (qrank % 80)]
    if workload == "w2_0_1":
        return "owner_id = %s", [qrank % 1000]
    if workload == "w3":
        start = qrank % 10
        cats = [(start + i) % 10 for i in range(5)]
        return (
            "tenant_id = 0 AND status = 'active' "
            "AND category_id = ANY(%s) "
            "AND created_at >= TIMESTAMPTZ '2022-01-01T00:00:00Z'",
            [cats],
        )
    raise ValueError(workload)

def diskann_label(workload, qrank):
    if workload == "w2_50":
        return [100 + (qrank % 2)]
    if workload == "w2_10":
        return [1000 + (qrank % 10)]
    raise ValueError(workload)

def acl_values(qrank):
    return 1 + (qrank % 80), qrank % 5

def exact_sql(workload, qrank):
    where, vals = scalar_filter(workload, qrank)
    sql = f"""
        WITH eligible AS MATERIALIZED (
            SELECT chunk_id, embedding
            FROM chunks
            WHERE {where}
        )
        SELECT chunk_id
        FROM eligible
        ORDER BY embedding <=> %s
        LIMIT 10
    """
    return sql, vals

def acl_exact_sql(qrank):
    tenant, principal = acl_values(qrank)
    sql = """
        WITH eligible AS MATERIALIZED (
            SELECT ch.chunk_id, ch.embedding
            FROM chunks AS ch
            WHERE ch.tenant_id = %s
              AND ch.status = 'active'
              AND EXISTS (
                  SELECT 1
                  FROM document_permissions AS p
                  WHERE p.document_id = ch.document_id
                    AND p.principal_id = %s
                    AND p.permission_type = 'read'
              )
        )
        SELECT chunk_id
        FROM eligible
        ORDER BY embedding <=> %s
        LIMIT 10
    """
    return sql, [tenant, principal]

def setup_pg_workload(c, engine, workload, strategy):
    if engine == "pg-hnsw":
        if strategy["path"] == "hnsw":
            c.execute("SELECT set_config('hnsw.ef_search',%s,false)",
                      (str(strategy["ef_search"]),))
            c.execute("SELECT set_config('hnsw.iterative_scan','off',false)")
        elif strategy["path"].startswith("planner_filtered_query_"):
            c.execute("SELECT set_config('hnsw.ef_search',%s,false)",
                      (str(strategy["ef_search"]),))
            c.execute("SELECT set_config('hnsw.iterative_scan',%s,false)",
                      (strategy["iterative_scan"],))
            c.execute("SELECT set_config('hnsw.max_scan_tuples',%s,false)",
                      (str(strategy["max_scan_tuples"]),))
            c.execute("SELECT set_config('hnsw.scan_mem_multiplier',%s,false)",
                      (str(strategy["scan_mem_multiplier"]),))
    elif engine == "pg-diskann":
        if strategy["path"] in {"diskann","label_aware_diskann"}:
            c.execute("SELECT set_config('diskann.query_search_list_size',%s,false)",
                      (str(strategy["query_search_list_size"]),))
            c.execute("SELECT set_config('diskann.query_rescore',%s,false)",
                      (str(strategy["query_rescore"]),))

def pg_query(c, engine, workload, qrank, qvec, strategy):
    path = strategy["path"]
    if workload == "w1":
        rows = c.execute("""
            SELECT chunk_id FROM chunks
            ORDER BY embedding <=> %s
            LIMIT 10
        """, (Vector(qvec),)).fetchall()
        return [int(r[0]) for r in rows]

    if path == "label_aware_diskann":
        labels = diskann_label(workload, qrank)
        rows = c.execute("""
            SELECT chunk_id FROM chunks
            WHERE labels && %s::smallint[]
            ORDER BY embedding <=> %s
            LIMIT 10
        """, (labels, Vector(qvec))).fetchall()
        return [int(r[0]) for r in rows]

    if path.startswith("planner_filtered_query_"):
        where, vals = scalar_filter(workload, qrank)
        rows = c.execute(f"""
            SELECT chunk_id FROM chunks
            WHERE {where}
            ORDER BY embedding <=> %s
            LIMIT 10
        """, [*vals, Vector(qvec)]).fetchall()
        return [int(r[0]) for r in rows]

    if path == "scalar_first_exact":
        sql, vals = exact_sql(workload, qrank)
        rows = c.execute(sql, [*vals, Vector(qvec)]).fetchall()
        return [int(r[0]) for r in rows]

    if path == "scalar_acl_exact":
        sql, vals = acl_exact_sql(qrank)
        rows = c.execute(sql, [*vals, Vector(qvec)]).fetchall()
        return [int(r[0]) for r in rows]

    raise ValueError(f"Unsupported PostgreSQL path {path} for {workload}")

def pg_plan(c, engine, workload, qvec, strategy):
    setup_pg_workload(c, engine, workload, strategy)
    path = strategy["path"]

    if workload == "w1":
        sql = """EXPLAIN (COSTS OFF)
                 SELECT chunk_id FROM chunks
                 ORDER BY embedding <=> %s LIMIT 10"""
        params = [Vector(qvec)]
    elif path == "label_aware_diskann":
        labels = diskann_label(workload, 0)
        sql = """EXPLAIN (COSTS OFF)
                 SELECT chunk_id FROM chunks
                 WHERE labels && %s::smallint[]
                 ORDER BY embedding <=> %s LIMIT 10"""
        params = [labels, Vector(qvec)]
    elif path.startswith("planner_filtered_query_"):
        where, vals = scalar_filter(workload, 0)
        sql = f"""EXPLAIN (COSTS OFF)
                  SELECT chunk_id FROM chunks
                  WHERE {where}
                  ORDER BY embedding <=> %s LIMIT 10"""
        params = [*vals, Vector(qvec)]
    elif path == "scalar_first_exact":
        raw, vals = exact_sql(workload, 0)
        sql = "EXPLAIN (COSTS OFF) " + raw
        params = [*vals, Vector(qvec)]
    elif path == "scalar_acl_exact":
        raw, vals = acl_exact_sql(0)
        sql = "EXPLAIN (COSTS OFF) " + raw
        params = [*vals, Vector(qvec)]
    else:
        raise ValueError(path)

    text = "\n".join(str(r[0]) for r in c.execute(sql, params).fetchall())

    if engine == "pg-hnsw":
        if path in {"hnsw","planner_filtered_query_hnsw_expected"}:
            if "chunks_embedding_hnsw" not in text:
                raise RuntimeError(
                    f"{workload}: frozen strategy expects HNSW, but plan is:\n{text}"
                )
        elif path == "planner_filtered_query_scalar_expected":
            if "chunks_embedding_hnsw" in text:
                raise RuntimeError(
                    f"{workload}: frozen tuning selected a scalar planner path, "
                    f"but HNSW is now used:\n{text}"
                )
        elif path in {"scalar_first_exact","scalar_acl_exact"}:
            if "chunks_embedding_hnsw" in text or "chunks_embedding_diskann" in text:
                raise RuntimeError(
                    f"{workload}: exact fallback unexpectedly uses ANN:\n{text}"
                )
    else:
        if path in {"diskann","label_aware_diskann"}:
            if "chunks_embedding_diskann" not in text:
                raise RuntimeError(
                    f"{workload}: frozen strategy expects DiskANN, but plan is:\n{text}"
                )
        elif path in {"scalar_first_exact","scalar_acl_exact"}:
            if "chunks_embedding_hnsw" in text or "chunks_embedding_diskann" in text:
                raise RuntimeError(
                    f"{workload}: exact fallback unexpectedly uses ANN:\n{text}"
                )
    return text

def qdrant_filter(workload, qrank):
    if workload == "w2_50":
        return models.Filter(must=[models.FieldCondition(
            key="language",
            match=models.MatchValue(value="en" if qrank % 2 == 0 else "de")
        )])
    if workload == "w2_10":
        return models.Filter(must=[models.FieldCondition(
            key="category_id",
            match=models.MatchValue(value=qrank % 10)
        )])
    if workload == "w2_1":
        return models.Filter(must=[models.FieldCondition(
            key="tenant_id",
            match=models.MatchValue(value=1 + (qrank % 80))
        )])
    if workload == "w2_0_1":
        return models.Filter(must=[models.FieldCondition(
            key="owner_id",
            match=models.MatchValue(value=qrank % 1000)
        )])
    if workload == "w3":
        start = qrank % 10
        cats = [(start + i) % 10 for i in range(5)]
        return models.Filter(must=[
            models.FieldCondition(
                key="tenant_id", match=models.MatchValue(value=0)
            ),
            models.FieldCondition(
                key="status", match=models.MatchValue(value="active")
            ),
            models.FieldCondition(
                key="category_id", match=models.MatchAny(any=cats)
            ),
            models.FieldCondition(
                key="created_at",
                range=models.DatetimeRange(gte="2022-01-01T00:00:00Z")
            ),
        ])
    if workload == "w4_acl":
        tenant, _ = acl_values(qrank)
        return models.Filter(must=[
            models.FieldCondition(
                key="tenant_id", match=models.MatchValue(value=tenant)
            ),
            models.FieldCondition(
                key="status", match=models.MatchValue(value="active")
            ),
        ])
    return None

def validate_acl_candidates(c, candidate_ids, principal):
    if not candidate_ids:
        return set()
    rows = c.execute("""
        SELECT ch.chunk_id
        FROM chunks AS ch
        JOIN document_permissions AS p
          ON p.document_id = ch.document_id
        WHERE ch.chunk_id = ANY(%s::bigint[])
          AND p.principal_id = %s
          AND p.permission_type = 'read'
    """, (list(map(int, candidate_ids)), int(principal))).fetchall()
    return {int(r[0]) for r in rows}

def qdrant_query(client, pgc, collection, workload, qrank, qvec, strategy):
    path = strategy["path"]

    if path == "approx":
        params = models.SearchParams(hnsw_ef=int(strategy["hnsw_ef"]), exact=False)
        limit = 10
        qf = None
    elif path == "filtered_approx":
        params = models.SearchParams(hnsw_ef=int(strategy["hnsw_ef"]), exact=False)
        limit = 10
        qf = qdrant_filter(workload, qrank)
    elif path == "filtered_exact":
        params = models.SearchParams(exact=True)
        limit = 10
        qf = qdrant_filter(workload, qrank)
    elif path == "qdrant_candidates_plus_postgres_acl":
        params = models.SearchParams(hnsw_ef=int(strategy["hnsw_ef"]), exact=False)
        limit = int(strategy["candidate_limit"])
        qf = qdrant_filter(workload, qrank)
    else:
        raise ValueError(path)

    resp = client.query_points(
        collection_name=collection,
        query=np.asarray(qvec, dtype=np.float32).tolist(),
        query_filter=qf,
        search_params=params,
        limit=limit,
        with_payload=False,
        with_vectors=False,
    )
    ids = [int(p.id) for p in resp.points]

    if path == "qdrant_candidates_plus_postgres_acl":
        _, principal = acl_values(qrank)
        allowed = validate_acl_candidates(pgc, ids, principal)
        return [pid for pid in ids if pid in allowed][:10], len(ids)

    return ids, len(ids)

def deterministic_orders(frozen, repetition, workload):
    seed = int(frozen["measurement_order_seed"])
    workload_offset = WORKLOADS.index(workload) * 10000
    rng = np.random.default_rng(seed + repetition * 1000 + workload_offset)
    measure = np.arange(100, 1000, dtype=int)
    rng.shuffle(measure)

    warm = np.tile(np.arange(0, 100, dtype=int), 2)
    rng2 = np.random.default_rng(seed + repetition * 1000 + workload_offset + 777)
    rng2.shuffle(warm)
    return warm.tolist(), measure.tolist()

def summarize(rows):
    lat = np.array([r["latency_ms"] for r in rows], dtype=float)
    rec = np.array([r["recall10"] for r in rows], dtype=float)
    returned = np.array([r["returned_count"] for r in rows], dtype=int)
    return {
        "queries": len(rows),
        "median_latency_ms": float(np.median(lat)),
        "p95_latency_ms": float(np.percentile(lat, 95)),
        "mean_latency_ms": float(np.mean(lat)),
        "mean_recall10": float(np.mean(rec)),
        "min_returned": int(np.min(returned)),
    }

def write_rows(path, rows):
    fields = [
        "engine","dataset_size","repetition","workload","query_rank",
        "latency_ms","recall10","returned_count","candidate_count",
        "execution_path"
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

def ensure_output_clean(engine, overwrite):
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    targets = list(RESULTS_ROOT.glob(f"{engine}_rep*.csv"))
    targets += list(RESULTS_ROOT.glob(f"{engine}_rep*_summary.json"))
    targets += [RESULTS_ROOT / f"{engine}_plans.json"]
    existing = [p for p in targets if p.exists()]
    if existing and not overwrite:
        names = "\n".join(str(p) for p in existing)
        raise SystemExit(
            "Measurement output already exists. Refusing to overwrite paper data.\n"
            "Use --overwrite only if you intentionally invalidate the prior run:\n" + names
        )
    if overwrite:
        for p in existing:
            p.unlink()

def validate_dataset(cfg, engine, size):
    with pg_conn(cfg) as c:
        pg_n = int(c.execute("SELECT count(*) FROM chunks").fetchone()[0])
        if pg_n != size:
            raise RuntimeError(f"PostgreSQL has {pg_n:,} chunks, expected {size:,}")
        indexes = ann_indexes(c)
        expected = ["chunks_embedding_hnsw"] if engine == "pg-hnsw" else ["chunks_embedding_diskann"]
        if engine.startswith("pg-") and indexes != expected:
            raise RuntimeError(f"{engine}: expected ANN index state {expected}, found {indexes}")

    if engine == "qdrant":
        q = cfg["qdrant"]
        client = QdrantClient(url=q["url"], timeout=120)
        n = int(client.count(collection_name=q["collection"], exact=True).count)
        if n != size:
            raise RuntimeError(f"Qdrant has {n:,} points, expected {size:,}")
        info = client.get_collection(q["collection"])
        if int(info.config.hnsw_config.m) == 0:
            raise RuntimeError("Qdrant HNSW is disabled.")

def run_postgres(engine, cfg, frozen, qemb, truths, repetitions):
    strategies = frozen["strategies"][engine]
    plans = {}
    with pg_conn(cfg) as c:
        for workload in WORKLOADS:
            plans[workload] = pg_plan(c, engine, workload, qemb[0], strategies[workload])
    write_json(RESULTS_ROOT / f"{engine}_plans.json", plans)

    for rep in range(1, repetitions + 1):
        print(f"\n=== {engine} measured repetition {rep}/{repetitions} ===")
        rep_rows = []
        with pg_conn(cfg) as c:
            for workload in WORKLOADS:
                strategy = strategies[workload]
                setup_pg_workload(c, engine, workload, strategy)
                warm_order, measure_order = deterministic_orders(frozen, rep, workload)

                print(f"{workload}: 200 unrecorded warmups")
                for qi in warm_order:
                    pg_query(c, engine, workload, qi, qemb[qi], strategy)

                print(f"{workload}: measuring 900 held-out queries")
                workload_rows = []
                for pos, qi in enumerate(measure_order, start=1):
                    t0 = time.perf_counter_ns()
                    ids = pg_query(c, engine, workload, qi, qemb[qi], strategy)
                    ms = (time.perf_counter_ns() - t0) / 1e6
                    row = {
                        "engine": engine,
                        "dataset_size": int(frozen["dataset_size"]),
                        "repetition": rep,
                        "workload": workload,
                        "query_rank": qi,
                        "latency_ms": ms,
                        "recall10": recall10(ids, truths[workload][qi]),
                        "returned_count": len(ids),
                        "candidate_count": "",
                        "execution_path": strategy["path"],
                    }
                    workload_rows.append(row)
                    if pos % 100 == 0:
                        print(f"  {workload}: {pos}/900", flush=True)

                s = summarize(workload_rows)
                print(json.dumps({workload: s}, indent=2))
                if strategy["path"] in {"scalar_first_exact","scalar_acl_exact"}:
                    if s["mean_recall10"] < 0.999999:
                        raise RuntimeError(
                            f"{workload}: exact path disagrees with exact ground truth: {s}"
                        )
                if s["min_returned"] != 10:
                    raise RuntimeError(f"{workload}: fewer than 10 results on held-out data: {s}")
                rep_rows.extend(workload_rows)

        path = RESULTS_ROOT / f"{engine}_rep{rep}.csv"
        write_rows(path, rep_rows)
        summary = {
            "engine": engine,
            "dataset_size": int(frozen["dataset_size"]),
            "repetition": rep,
            "workloads": {
                w: summarize([r for r in rep_rows if r["workload"] == w])
                for w in WORKLOADS
            },
            "warning": "These ARE held-out measurement results; do not replace them with tuning values."
        }
        write_json(RESULTS_ROOT / f"{engine}_rep{rep}_summary.json", summary)
        print(f"Wrote {path}")

def run_qdrant(cfg, frozen, qemb, truths, repetitions):
    engine = "qdrant"
    strategies = frozen["strategies"][engine]
    q = cfg["qdrant"]
    client = QdrantClient(url=q["url"], timeout=120)
    with pg_conn(cfg) as pgc:
        for rep in range(1, repetitions + 1):
            print(f"\n=== qdrant measured repetition {rep}/{repetitions} ===")
            rep_rows = []
            for workload in WORKLOADS:
                strategy = strategies[workload]
                warm_order, measure_order = deterministic_orders(frozen, rep, workload)

                print(f"{workload}: 200 unrecorded warmups")
                for qi in warm_order:
                    qdrant_query(
                        client, pgc, q["collection"], workload, qi, qemb[qi], strategy
                    )

                print(f"{workload}: measuring 900 held-out queries")
                workload_rows = []
                for pos, qi in enumerate(measure_order, start=1):
                    t0 = time.perf_counter_ns()
                    ids, candidate_count = qdrant_query(
                        client, pgc, q["collection"], workload, qi, qemb[qi], strategy
                    )
                    ms = (time.perf_counter_ns() - t0) / 1e6
                    row = {
                        "engine": engine,
                        "dataset_size": int(frozen["dataset_size"]),
                        "repetition": rep,
                        "workload": workload,
                        "query_rank": qi,
                        "latency_ms": ms,
                        "recall10": recall10(ids, truths[workload][qi]),
                        "returned_count": len(ids),
                        "candidate_count": candidate_count,
                        "execution_path": strategy["path"],
                    }
                    workload_rows.append(row)
                    if pos % 100 == 0:
                        print(f"  {workload}: {pos}/900", flush=True)

                s = summarize(workload_rows)
                print(json.dumps({workload: s}, indent=2))
                if strategy["path"] == "filtered_exact" and s["mean_recall10"] < 0.999999:
                    print(
                        f"WARNING: {workload}: Qdrant exact mode differs slightly "
                        f"from offline ground truth: {s}. "
                        "Recording held-out result without retuning.",
                        flush=True,
                    )
                if s["min_returned"] != 10:
                    raise RuntimeError(f"{workload}: fewer than 10 results on held-out data: {s}")
                rep_rows.extend(workload_rows)

            path = RESULTS_ROOT / f"{engine}_rep{rep}.csv"
            write_rows(path, rep_rows)
            summary = {
                "engine": engine,
                "dataset_size": int(frozen["dataset_size"]),
                "repetition": rep,
                "workloads": {
                    w: summarize([r for r in rep_rows if r["workload"] == w])
                    for w in WORKLOADS
                },
                "warning": "These ARE held-out measurement results; do not replace them with tuning values."
            }
            write_json(RESULTS_ROOT / f"{engine}_rep{rep}_summary.json", summary)
            print(f"Wrote {path}")

def write_manifest(engine, frozen):
    manifest_path = RESULTS_ROOT / "measurement_manifest.json"
    data = {
        "purpose": "HELD-OUT PAPER MEASUREMENT MANIFEST",
        "dataset_size": int(frozen["dataset_size"]),
        "measurement_query_ranks": frozen["measurement_query_ranks"],
        "measurement_query_count": int(frozen["measurement_query_count"]),
        "warmup_executions_per_workload_per_repetition":
            int(frozen["warmup_executions_per_workload_per_repetition"]),
        "warmup_source_query_ranks": frozen["warmup_source_query_ranks"],
        "repetitions": int(frozen["repetitions"]),
        "config_sha256": sha256_file(FROZEN_PATH),
        "runner_sha256": sha256_file(Path(__file__)),
        "python": sys.version,
        "platform": platform.platform(),
        "engines_started": [],
        "note": "No tuning/query-time parameter selection is performed by this runner."
    }
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old.get("config_sha256") != data["config_sha256"]:
            raise RuntimeError("Existing measurement manifest uses a different frozen config.")
        data = old
        data["runner_sha256"] = sha256_file(Path(__file__))
    if engine not in data["engines_started"]:
        data["engines_started"].append(engine)
    write_json(manifest_path, data)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True,
                    choices=["pg-hnsw","pg-diskann","qdrant"])
    ap.add_argument("--repetitions", type=int, default=3)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    frozen = json.loads(FROZEN_PATH.read_text())
    if int(frozen["dataset_size"]) != 100000:
        raise SystemExit("This runner is frozen for 100k only.")
    if args.repetitions != int(frozen["repetitions"]):
        raise SystemExit(
            f"Frozen methodology requires {frozen['repetitions']} repetitions."
        )

    ensure_output_clean(args.engine, args.overwrite)
    cfg = load_config()
    validate_dataset(cfg, args.engine, 100000)

    qemb = np.load(PROCESSED / "query_embeddings.npy", mmap_mode="r")
    if qemb.shape[0] < 1000:
        raise RuntimeError(f"Expected at least 1000 query embeddings, got {qemb.shape}")
    truths = load_truths(100000)

    write_manifest(args.engine, frozen)
    print("Frozen config:", FROZEN_PATH)
    print("Config SHA256:", sha256_file(FROZEN_PATH))
    print("Measurement ranks: 100-999 only")
    print("Warmups: 200 executions/workload/repetition from ranks 0-99 only")
    print("No tuning is performed.")

    if args.engine.startswith("pg-"):
        run_postgres(args.engine, cfg, frozen, qemb, truths, args.repetitions)
    else:
        run_qdrant(cfg, frozen, qemb, truths, args.repetitions)

    print(f"\nCompleted held-out measurements for {args.engine}.")
    print(f"Results: {RESULTS_ROOT}")

if __name__ == "__main__":
    main()
