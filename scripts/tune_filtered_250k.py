from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from qdrant_client import QdrantClient, models

from common import GROUND_TRUTH, PROCESSED, ROOT, load_config, write_json

SIZE = 250000
TARGET = 0.95
QIDX = np.arange(0, 100, dtype=int)
WORKLOADS = ["w2_50", "w2_10", "w2_1", "w2_0_1", "w3"]
TUNING_DIR = ROOT / "tuning"
FROZEN = json.loads((ROOT / "config" / "frozen_w1_250000.json").read_text())

HNSW_CAPS = [5000, 10000, 20000, 40000, 80000, 125000, 250000]
DISKANN_GRID = [
    (75, 100),
    (100, 100),
    (100, 200),
    (150, 200),
    (200, 400),
    (300, 600),
    (400, 800),
    (600, 1000),
]
QDRANT_EF = [40, 60, 80, 100, 150, 200, 300, 500, 800]

def pg_conn(cfg):
    p = cfg["postgres"]
    c = psycopg.connect(
        host=p["host"], port=p["port"], dbname=p["dbname"],
        user=p["user"], password=p["password"]
    )
    register_vector(c)
    return c

def recall10(ids, truth):
    return len(set(map(int, ids)) & set(map(int, truth))) / 10.0

def load_truths():
    return {
        w: np.load(GROUND_TRUTH / str(SIZE) / f"{w}.npz")["top10_ids"]
        for w in WORKLOADS
    }

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

def diskann_where(workload, qrank):
    if workload == "w2_50":
        return "labels && %s::smallint[]", [[100 + (qrank % 2)]]
    if workload == "w2_10":
        return "labels && %s::smallint[]", [[1000 + (qrank % 10)]]
    if workload == "w2_1":
        return "labels && %s::smallint[]", [[2000 + 1 + (qrank % 80)]]
    if workload == "w2_0_1":
        return "labels && %s::smallint[]", [[3000 + (qrank % 1000)]]
    if workload == "w3":
        start = qrank % 10
        cats = [(start + i) % 10 for i in range(5)]
        return (
            "labels && %s::smallint[] AND status = 'active' "
            "AND category_id = ANY(%s) "
            "AND created_at >= TIMESTAMPTZ '2022-01-01T00:00:00Z'",
            [[2000], cats],
        )
    raise ValueError(workload)

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

def explain(c, sql, params):
    rows = c.execute("EXPLAIN (COSTS OFF) " + sql, params).fetchall()
    return "\n".join(str(r[0]) for r in rows)

def summarize(latencies, recalls, returned):
    return {
        "mean_recall10": float(np.mean(recalls)),
        "median_latency_ms_tuning_only": float(np.median(latencies)),
        "p95_latency_ms_tuning_only": float(np.percentile(latencies, 95)),
        "min_returned": int(np.min(returned)),
    }

def valid(s):
    return s["mean_recall10"] >= TARGET and s["min_returned"] == 10

def run_pg_exact(c, qemb, truths):
    out, plans = {}, {}
    for w in WORKLOADS:
        sql0, vals0 = exact_sql(w, 0)
        plan = explain(c, sql0, [*vals0, Vector(qemb[0])])
        if "chunks_embedding_hnsw" in plan or "chunks_embedding_diskann" in plan:
            raise RuntimeError(f"{w}: exact fallback unexpectedly uses ANN:\n{plan}")
        plans[w] = plan

        for qi in QIDX[:10]:
            sql, vals = exact_sql(w, int(qi))
            c.execute(sql, [*vals, Vector(qemb[qi])]).fetchall()

        lats, recs, ret = [], [], []
        for qi in QIDX:
            sql, vals = exact_sql(w, int(qi))
            t0 = time.perf_counter_ns()
            rows = c.execute(sql, [*vals, Vector(qemb[qi])]).fetchall()
            lats.append((time.perf_counter_ns() - t0) / 1e6)
            ids = [int(r[0]) for r in rows]
            recs.append(recall10(ids, truths[w][qi]))
            ret.append(len(ids))
        out[w] = {
            "access_path": "scalar-index-first exact vector ranking",
            **summarize(lats, recs, ret),
        }
        if out[w]["mean_recall10"] < 0.999999 or out[w]["min_returned"] != 10:
            raise RuntimeError(f"{w}: exact fallback failed: {out[w]}")
        print(f"\nExact fallback {w}:")
        print(json.dumps(out[w], indent=2))
    return out, plans

def tune_pg_hnsw(cfg, qemb, truths):
    frozen_ef = int(FROZEN["w1"]["pg-hnsw"]["ef_search"])
    with pg_conn(cfg) as c:
        idx = [r[0] for r in c.execute("""
            SELECT indexname FROM pg_indexes
            WHERE tablename='chunks'
              AND indexname IN ('chunks_embedding_hnsw','chunks_embedding_diskann')
            ORDER BY indexname
        """).fetchall()]
        if idx != ["chunks_embedding_hnsw"]:
            raise RuntimeError(f"Expected only chunks_embedding_hnsw; found {idx}")

        exact, exact_plans = run_pg_exact(c, qemb, truths)
        grids, plans = [], {}

        for cap in HNSW_CAPS:
            c.execute("SELECT set_config('hnsw.ef_search',%s,false)", (str(frozen_ef),))
            c.execute("SELECT set_config('hnsw.iterative_scan','strict_order',false)")
            c.execute("SELECT set_config('hnsw.max_scan_tuples',%s,false)", (str(cap),))
            c.execute("SELECT set_config('hnsw.scan_mem_multiplier','2',false)")

            row = {"max_scan_tuples": cap, "by_workload": {}}
            plans[str(cap)] = {}
            print(f"\nPG-HNSW max_scan_tuples={cap}")

            for w in WORKLOADS:
                where, vals = scalar_filter(w, 0)
                sql = f"""
                    SELECT chunk_id FROM chunks
                    WHERE {where}
                    ORDER BY embedding <=> %s
                    LIMIT 10
                """
                plan = explain(c, sql, [*vals, Vector(qemb[0])])
                plans[str(cap)][w] = plan

                for qi in QIDX[:10]:
                    where_i, vals_i = scalar_filter(w, int(qi))
                    c.execute(f"""
                        SELECT chunk_id FROM chunks
                        WHERE {where_i}
                        ORDER BY embedding <=> %s
                        LIMIT 10
                    """, [*vals_i, Vector(qemb[qi])]).fetchall()

                lats, recs, ret = [], [], []
                for qi in QIDX:
                    where_i, vals_i = scalar_filter(w, int(qi))
                    t0 = time.perf_counter_ns()
                    rows = c.execute(f"""
                        SELECT chunk_id FROM chunks
                        WHERE {where_i}
                        ORDER BY embedding <=> %s
                        LIMIT 10
                    """, [*vals_i, Vector(qemb[qi])]).fetchall()
                    lats.append((time.perf_counter_ns() - t0) / 1e6)
                    ids = [int(r[0]) for r in rows]
                    recs.append(recall10(ids, truths[w][qi]))
                    ret.append(len(ids))

                s = summarize(lats, recs, ret)
                s["plan_uses_hnsw"] = "chunks_embedding_hnsw" in plan
                row["by_workload"][w] = s

            grids.append(row)
            print(json.dumps(row, indent=2))

    strategy = {}
    for w in WORKLOADS:
        candidates = []
        for g in grids:
            s = g["by_workload"][w]
            if valid(s):
                candidates.append({
                    "access_path": (
                        "filtered query with HNSW plan"
                        if s["plan_uses_hnsw"]
                        else "planner-selected scalar path with HNSW available"
                    ),
                    "ef_search": frozen_ef,
                    "iterative_scan": "strict_order",
                    "scan_mem_multiplier": 2,
                    "max_scan_tuples": g["max_scan_tuples"],
                    **s,
                })
        candidates.append(exact[w])
        strategy[w] = min(candidates, key=lambda x: x["median_latency_ms_tuning_only"])

    report = {
        "purpose": "PG-HNSW 250K FILTERED STRATEGY SELECTION ONLY",
        "dataset_size": SIZE,
        "target_recall10": TARGET,
        "tuning_query_ranks": [0,99],
        "measurement_query_ranks_reserved": [100,999],
        "frozen_w1_ef_search": frozen_ef,
        "strategy": strategy,
        "warning": "All latency values are tuning-only, not paper results.",
    }
    write_json(TUNING_DIR / "filtered_pg-hnsw_250000_selected.json", report)
    write_json(TUNING_DIR / "filtered_pg-hnsw_250000_grid.json", grids)
    write_json(TUNING_DIR / "filtered_pg-hnsw_250000_plans.json", plans)
    write_json(TUNING_DIR / "filtered_pg-hnsw_250000_exact_plans.json", exact_plans)
    print("\nSELECTED PG-HNSW 250K FILTERED STRATEGY")
    print(json.dumps(report, indent=2))

def tune_pg_diskann(cfg, qemb, truths):
    with pg_conn(cfg) as c:
        idx = [r[0] for r in c.execute("""
            SELECT indexname FROM pg_indexes
            WHERE tablename='chunks'
              AND indexname IN ('chunks_embedding_hnsw','chunks_embedding_diskann')
            ORDER BY indexname
        """).fetchall()]
        if idx != ["chunks_embedding_diskann"]:
            raise RuntimeError(f"Expected only chunks_embedding_diskann; found {idx}")

        exact, exact_plans = run_pg_exact(c, qemb, truths)
        grids, plans = [], {}

        for search_list, rescore in DISKANN_GRID:
            c.execute(
                "SELECT set_config('diskann.query_search_list_size',%s,false)",
                (str(search_list),)
            )
            c.execute(
                "SELECT set_config('diskann.query_rescore',%s,false)",
                (str(rescore),)
            )
            row = {
                "query_search_list_size": search_list,
                "query_rescore": rescore,
                "by_workload": {}
            }
            plans[f"{search_list}/{rescore}"] = {}
            print(f"\nPG-DiskANN search_list={search_list}, rescore={rescore}")

            for w in WORKLOADS:
                where, vals = diskann_where(w, 0)
                sql = f"""
                    SELECT chunk_id FROM chunks
                    WHERE {where}
                    ORDER BY embedding <=> %s
                    LIMIT 10
                """
                plan = explain(c, sql, [*vals, Vector(qemb[0])])
                plans[f"{search_list}/{rescore}"][w] = plan

                for qi in QIDX[:10]:
                    where_i, vals_i = diskann_where(w, int(qi))
                    c.execute(f"""
                        SELECT chunk_id FROM chunks
                        WHERE {where_i}
                        ORDER BY embedding <=> %s
                        LIMIT 10
                    """, [*vals_i, Vector(qemb[qi])]).fetchall()

                lats, recs, ret = [], [], []
                for qi in QIDX:
                    where_i, vals_i = diskann_where(w, int(qi))
                    t0 = time.perf_counter_ns()
                    rows = c.execute(f"""
                        SELECT chunk_id FROM chunks
                        WHERE {where_i}
                        ORDER BY embedding <=> %s
                        LIMIT 10
                    """, [*vals_i, Vector(qemb[qi])]).fetchall()
                    lats.append((time.perf_counter_ns() - t0) / 1e6)
                    ids = [int(r[0]) for r in rows]
                    recs.append(recall10(ids, truths[w][qi]))
                    ret.append(len(ids))
                s = summarize(lats, recs, ret)
                s["plan_uses_diskann"] = "chunks_embedding_diskann" in plan
                row["by_workload"][w] = s

            grids.append(row)
            print(json.dumps(row, indent=2))

    strategy = {}
    for w in WORKLOADS:
        candidates = []
        for g in grids:
            s = g["by_workload"][w]
            if valid(s):
                candidates.append({
                    "access_path": (
                        "label-aware DiskANN"
                        if s["plan_uses_diskann"]
                        else "planner-selected non-DiskANN path with DiskANN available"
                    ),
                    "query_search_list_size": g["query_search_list_size"],
                    "query_rescore": g["query_rescore"],
                    **s,
                })
        candidates.append(exact[w])
        strategy[w] = min(candidates, key=lambda x: x["median_latency_ms_tuning_only"])

    report = {
        "purpose": "PG-DISKANN 250K FILTERED STRATEGY SELECTION ONLY",
        "dataset_size": SIZE,
        "target_recall10": TARGET,
        "tuning_query_ranks": [0,99],
        "measurement_query_ranks_reserved": [100,999],
        "frozen_w1": FROZEN["w1"]["pg-diskann"],
        "strategy": strategy,
        "warning": "All latency values are tuning-only, not paper results.",
    }
    write_json(TUNING_DIR / "filtered_pg-diskann_250000_selected.json", report)
    write_json(TUNING_DIR / "filtered_pg-diskann_250000_grid.json", grids)
    write_json(TUNING_DIR / "filtered_pg-diskann_250000_plans.json", plans)
    write_json(TUNING_DIR / "filtered_pg-diskann_250000_exact_plans.json", exact_plans)
    print("\nSELECTED PG-DISKANN 250K FILTERED STRATEGY")
    print(json.dumps(report, indent=2))

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
            models.FieldCondition(key="tenant_id", match=models.MatchValue(value=0)),
            models.FieldCondition(key="status", match=models.MatchValue(value="active")),
            models.FieldCondition(key="category_id", match=models.MatchAny(any=cats)),
            models.FieldCondition(
                key="created_at",
                range=models.DatetimeRange(gte="2022-01-01T00:00:00Z")
            ),
        ])
    raise ValueError(workload)

def tune_qdrant(cfg, qemb, truths):
    q = cfg["qdrant"]
    client = QdrantClient(url=q["url"], timeout=120)
    n = int(client.count(collection_name=q["collection"], exact=True).count)
    if n != SIZE:
        raise RuntimeError(f"Qdrant has {n:,} points, expected {SIZE:,}")
    info = client.get_collection(q["collection"])
    if int(info.config.hnsw_config.m) == 0:
        raise RuntimeError("Qdrant HNSW is disabled.")

    modes = []
    for ef in QDRANT_EF:
        row = {"mode": "approx", "hnsw_ef": ef, "by_workload": {}}
        params = models.SearchParams(hnsw_ef=ef, exact=False)
        print(f"\nQdrant hnsw_ef={ef}")
        for w in WORKLOADS:
            for qi in QIDX[:10]:
                client.query_points(
                    collection_name=q["collection"],
                    query=np.asarray(qemb[qi], dtype=np.float32).tolist(),
                    query_filter=qdrant_filter(w, int(qi)),
                    search_params=params,
                    limit=10, with_payload=False, with_vectors=False,
                )
            lats, recs, ret = [], [], []
            for qi in QIDX:
                t0 = time.perf_counter_ns()
                resp = client.query_points(
                    collection_name=q["collection"],
                    query=np.asarray(qemb[qi], dtype=np.float32).tolist(),
                    query_filter=qdrant_filter(w, int(qi)),
                    search_params=params,
                    limit=10, with_payload=False, with_vectors=False,
                )
                lats.append((time.perf_counter_ns() - t0) / 1e6)
                ids = [int(p.id) for p in resp.points]
                recs.append(recall10(ids, truths[w][qi]))
                ret.append(len(ids))
            row["by_workload"][w] = summarize(lats, recs, ret)
        modes.append(row)
        print(json.dumps(row, indent=2))

    exact = {"mode": "exact", "by_workload": {}}
    params = models.SearchParams(exact=True)
    print("\nQdrant exact")
    for w in WORKLOADS:
        for qi in QIDX[:10]:
            client.query_points(
                collection_name=q["collection"],
                query=np.asarray(qemb[qi], dtype=np.float32).tolist(),
                query_filter=qdrant_filter(w, int(qi)),
                search_params=params,
                limit=10, with_payload=False, with_vectors=False,
            )
        lats, recs, ret = [], [], []
        for qi in QIDX:
            t0 = time.perf_counter_ns()
            resp = client.query_points(
                collection_name=q["collection"],
                query=np.asarray(qemb[qi], dtype=np.float32).tolist(),
                query_filter=qdrant_filter(w, int(qi)),
                search_params=params,
                limit=10, with_payload=False, with_vectors=False,
            )
            lats.append((time.perf_counter_ns() - t0) / 1e6)
            ids = [int(p.id) for p in resp.points]
            recs.append(recall10(ids, truths[w][qi]))
            ret.append(len(ids))
        exact["by_workload"][w] = summarize(lats, recs, ret)
    print(json.dumps(exact, indent=2))

    strategy = {}
    for w in WORKLOADS:
        candidates = []
        for m in modes:
            s = m["by_workload"][w]
            if valid(s):
                candidates.append({
                    "access_path": "Qdrant filter-aware approximate search",
                    "hnsw_ef": m["hnsw_ef"],
                    **s,
                })
        ex = exact["by_workload"][w]
        if ex["min_returned"] == 10:
            candidates.append({
                "access_path": "Qdrant filtered exact search",
                **ex,
            })
        if not candidates:
            raise RuntimeError(f"{w}: no valid Qdrant path")
        strategy[w] = min(candidates, key=lambda x: x["median_latency_ms_tuning_only"])

    report = {
        "purpose": "QDRANT 250K FILTERED STRATEGY SELECTION ONLY",
        "dataset_size": SIZE,
        "target_recall10": TARGET,
        "tuning_query_ranks": [0,99],
        "measurement_query_ranks_reserved": [100,999],
        "frozen_w1_hnsw_ef": FROZEN["w1"]["qdrant"]["hnsw_ef"],
        "strategy": strategy,
        "warning": "All latency values are tuning-only, not paper results.",
    }
    write_json(TUNING_DIR / "filtered_qdrant_250000_selected.json", report)
    write_json(TUNING_DIR / "filtered_qdrant_250000_grid.json", {
        "approximate": modes,
        "exact": exact,
    })
    print("\nSELECTED QDRANT 250K FILTERED STRATEGY")
    print(json.dumps(report, indent=2))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True,
                    choices=["pg-diskann", "pg-hnsw", "qdrant"])
    args = ap.parse_args()

    cfg = load_config()
    qemb = np.load(PROCESSED / "query_embeddings.npy", mmap_mode="r")
    truths = load_truths()
    TUNING_DIR.mkdir(parents=True, exist_ok=True)

    if args.engine == "pg-diskann":
        tune_pg_diskann(cfg, qemb, truths)
    elif args.engine == "pg-hnsw":
        tune_pg_hnsw(cfg, qemb, truths)
    else:
        tune_qdrant(cfg, qemb, truths)

if __name__ == "__main__":
    main()
