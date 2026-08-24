from __future__ import annotations

import argparse
import csv
import json
import time

import numpy as np
import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector

from common import GROUND_TRUTH, PROCESSED, ROOT, load_config, write_json

TUNING_DIR = ROOT / "tuning"
TARGET = 0.95
QIDX = np.arange(0, 100, dtype=int)
BROAD = ["w2_50", "w2_10"]
SPARSE_OR_COMPOUND = ["w2_1", "w2_0_1", "w3"]
GRID = [(100,100), (100,200), (150,200), (200,400)]

def pg_conn(cfg):
    p = cfg["postgres"]
    conn = psycopg.connect(
        host=p["host"], port=p["port"], dbname=p["dbname"],
        user=p["user"], password=p["password"]
    )
    register_vector(conn)
    return conn

def recall10(ids, truth):
    return len(set(map(int, ids)) & set(map(int, truth))) / 10.0

def load_truth(size):
    return {
        w: np.load(GROUND_TRUTH / str(size) / f"{w}.npz")["top10_ids"]
        for w in BROAD + SPARSE_OR_COMPOUND
    }

def label_for(workload, qrank):
    if workload == "w2_50":
        return [100 + (qrank % 2)]
    if workload == "w2_10":
        return [1000 + (qrank % 10)]
    raise ValueError(workload)

def exact_filter(workload, qrank):
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

def assert_diskann_only(conn):
    rows = conn.execute("""
        SELECT indexname
        FROM pg_indexes
        WHERE tablename='chunks'
          AND indexname IN ('chunks_embedding_hnsw','chunks_embedding_diskann')
        ORDER BY indexname
    """).fetchall()
    names = [r[0] for r in rows]
    if names != ["chunks_embedding_diskann"]:
        raise RuntimeError(
            f"Expected only chunks_embedding_diskann among ANN indexes; found {names}"
        )

def explain_label(conn, qvec, workload, qrank):
    labels = label_for(workload, qrank)
    rows = conn.execute("""
        EXPLAIN (COSTS OFF)
        SELECT chunk_id FROM chunks
        WHERE labels && %s::smallint[]
        ORDER BY embedding <=> %s
        LIMIT 10
    """, (labels, Vector(qvec))).fetchall()
    text = "\n".join(r[0] for r in rows)
    if "chunks_embedding_diskann" not in text:
        raise RuntimeError(
            f"{workload}: expected DiskANN plan, got:\n{text}"
        )
    return text

def explain_exact(conn, qvec, workload, qrank):
    where, vals = exact_filter(workload, qrank)
    rows = conn.execute(f"""
        EXPLAIN (COSTS OFF)
        WITH eligible AS MATERIALIZED (
            SELECT chunk_id, embedding
            FROM chunks
            WHERE {where}
        )
        SELECT chunk_id
        FROM eligible
        ORDER BY embedding <=> %s
        LIMIT 10
    """, [*vals, Vector(qvec)]).fetchall()
    text = "\n".join(r[0] for r in rows)
    if "chunks_embedding_diskann" in text or "chunks_embedding_hnsw" in text:
        raise RuntimeError(
            f"{workload}: exact fallback unexpectedly used an ANN index:\n{text}"
        )
    return text

def run_label_grid(conn, qemb, truths):
    summaries = []
    plans = {}
    for search_list, rescore in GRID:
        print(f"\nLabel-aware DiskANN: search_list={search_list}, rescore={rescore}")
        conn.execute(
            "SELECT set_config('diskann.query_search_list_size',%s,false)",
            (str(search_list),)
        )
        conn.execute(
            "SELECT set_config('diskann.query_rescore',%s,false)",
            (str(rescore),)
        )
        rows_all = []
        for workload in BROAD:
            plans.setdefault(f"{search_list}/{rescore}", {})[workload] = explain_label(
                conn, qemb[QIDX[0]], workload, int(QIDX[0])
            )
            for qi in QIDX[:10]:
                labels = label_for(workload, int(qi))
                conn.execute("""
                    SELECT chunk_id FROM chunks
                    WHERE labels && %s::smallint[]
                    ORDER BY embedding <=> %s
                    LIMIT 10
                """, (labels, Vector(qemb[qi]))).fetchall()
            for qi in QIDX:
                labels = label_for(workload, int(qi))
                t0 = time.perf_counter_ns()
                result = conn.execute("""
                    SELECT chunk_id FROM chunks
                    WHERE labels && %s::smallint[]
                    ORDER BY embedding <=> %s
                    LIMIT 10
                """, (labels, Vector(qemb[qi]))).fetchall()
                ms = (time.perf_counter_ns() - t0) / 1e6
                ids = [r[0] for r in result]
                rows_all.append((workload, ms, recall10(ids, truths[workload][qi]), len(ids)))

        by = {}
        for workload in BROAD:
            r = [x for x in rows_all if x[0] == workload]
            lats = np.array([x[1] for x in r])
            rec = np.array([x[2] for x in r])
            returned = np.array([x[3] for x in r])
            by[workload] = {
                "mean_recall10": float(rec.mean()),
                "median_latency_ms_tuning_only": float(np.median(lats)),
                "p95_latency_ms_tuning_only": float(np.percentile(lats,95)),
                "min_returned": int(returned.min()),
            }
        s = {
            "query_search_list_size": search_list,
            "query_rescore": rescore,
            "by_workload": by,
            "median_latency_ms_tuning_only": float(np.median([x[1] for x in rows_all])),
        }
        summaries.append(s)
        print(json.dumps(s, indent=2))
    eligible = [
        s for s in summaries
        if all(
            s["by_workload"][w]["mean_recall10"] >= TARGET
            and s["by_workload"][w]["min_returned"] == 10
            for w in BROAD
        )
    ]
    if not eligible:
        raise SystemExit("No label-aware DiskANN config passed both 50% and 10% workloads.")
    selected = min(eligible, key=lambda s: s["median_latency_ms_tuning_only"])
    return summaries, selected, plans

def run_exact_fallbacks(conn, qemb, truths):
    out = {}
    plans = {}
    for workload in SPARSE_OR_COMPOUND:
        plans[workload] = explain_exact(conn, qemb[QIDX[0]], workload, int(QIDX[0]))
        for qi in QIDX[:10]:
            where, vals = exact_filter(workload, int(qi))
            conn.execute(f"""
                WITH eligible AS MATERIALIZED (
                    SELECT chunk_id, embedding FROM chunks WHERE {where}
                )
                SELECT chunk_id FROM eligible
                ORDER BY embedding <=> %s LIMIT 10
            """, [*vals, Vector(qemb[qi])]).fetchall()

        latencies, recalls, returned = [], [], []
        for qi in QIDX:
            where, vals = exact_filter(workload, int(qi))
            t0 = time.perf_counter_ns()
            result = conn.execute(f"""
                WITH eligible AS MATERIALIZED (
                    SELECT chunk_id, embedding FROM chunks WHERE {where}
                )
                SELECT chunk_id FROM eligible
                ORDER BY embedding <=> %s LIMIT 10
            """, [*vals, Vector(qemb[qi])]).fetchall()
            latencies.append((time.perf_counter_ns() - t0) / 1e6)
            ids = [r[0] for r in result]
            recalls.append(recall10(ids, truths[workload][qi]))
            returned.append(len(ids))

        out[workload] = {
            "strategy": "scalar_index_first_exact_vector_ranking",
            "mean_recall10": float(np.mean(recalls)),
            "median_latency_ms_tuning_only": float(np.median(latencies)),
            "p95_latency_ms_tuning_only": float(np.percentile(latencies,95)),
            "min_returned": int(np.min(returned)),
        }
        print(f"\nExact fallback {workload}:")
        print(json.dumps(out[workload], indent=2))

        if out[workload]["mean_recall10"] < 0.999999 or out[workload]["min_returned"] != 10:
            raise RuntimeError(
                f"{workload}: exact fallback did not produce exact top-10."
            )
    return out, plans

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, required=True)
    args = ap.parse_args()
    if args.size != 100000:
        raise SystemExit("Validate the crossover at 100k before scaling.")

    cfg = load_config()
    qemb = np.load(PROCESSED / "query_embeddings.npy", mmap_mode="r")
    truths = load_truth(args.size)
    TUNING_DIR.mkdir(parents=True, exist_ok=True)

    with pg_conn(cfg) as conn:
        assert_diskann_only(conn)
        grid, label_selected, label_plans = run_label_grid(conn, qemb, truths)
        exact, exact_plans = run_exact_fallbacks(conn, qemb, truths)

    report = {
        "purpose": "PG-DISKANN FILTERED STRATEGY SELECTION ONLY",
        "dataset_size": args.size,
        "target_recall10": TARGET,
        "tuning_query_ranks": [0,99],
        "measurement_query_ranks_reserved": [100,999],
        "strategy": {
            "w2_50": {
                "access_path": "label-aware DiskANN",
                "query_search_list_size": label_selected["query_search_list_size"],
                "query_rescore": label_selected["query_rescore"],
                **label_selected["by_workload"]["w2_50"],
            },
            "w2_10": {
                "access_path": "label-aware DiskANN",
                "query_search_list_size": label_selected["query_search_list_size"],
                "query_rescore": label_selected["query_rescore"],
                **label_selected["by_workload"]["w2_10"],
            },
            "w2_1": exact["w2_1"],
            "w2_0_1": exact["w2_0_1"],
            "w3": exact["w3"],
        },
        "methodological_note": (
            "Sparse/compound predicates use scalar-index-first exact ranking because "
            "the labeled DiskANN graph did not satisfy the predeclared Recall@10 and "
            "result-completeness target at 1% and 0.1% selectivity on tuning queries."
        ),
        "warning": "All latency values are tuning-only, not paper results."
    }

    write_json(TUNING_DIR / "pg_diskann_crossover_100000_selected.json", report)
    write_json(TUNING_DIR / "pg_diskann_crossover_100000_label_plans.json", label_plans)
    write_json(TUNING_DIR / "pg_diskann_crossover_100000_exact_plans.json", exact_plans)

    grid_csv = TUNING_DIR / "pg_diskann_crossover_100000_label_grid.csv"
    with grid_csv.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "query_search_list_size","query_rescore",
            "w2_50_recall","w2_50_median_ms","w2_50_min_returned",
            "w2_10_recall","w2_10_median_ms","w2_10_min_returned",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in grid:
            w.writerow({
                "query_search_list_size": s["query_search_list_size"],
                "query_rescore": s["query_rescore"],
                "w2_50_recall": s["by_workload"]["w2_50"]["mean_recall10"],
                "w2_50_median_ms": s["by_workload"]["w2_50"]["median_latency_ms_tuning_only"],
                "w2_50_min_returned": s["by_workload"]["w2_50"]["min_returned"],
                "w2_10_recall": s["by_workload"]["w2_10"]["mean_recall10"],
                "w2_10_median_ms": s["by_workload"]["w2_10"]["median_latency_ms_tuning_only"],
                "w2_10_min_returned": s["by_workload"]["w2_10"]["min_returned"],
            })

    print("\nSELECTED PG-DISKANN CROSSOVER STRATEGY")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
