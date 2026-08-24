from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from qdrant_client import QdrantClient, models

from common import GROUND_TRUTH, PROCESSED, ROOT, load_config, write_json

TUNING_DIR = ROOT / "tuning"
FROZEN_W1 = ROOT / "config" / "frozen_w1_100000.json"
TARGET = 0.95
WORKLOADS = ["w2_50", "w2_10", "w2_1", "w2_0_1", "w3"]

PG_HNSW_SCAN_CAPS = [5000, 10000, 20000, 40000, 80000, 100000]
PG_DISKANN_GRID = [
    (100, 100),
    (100, 200),
    (150, 200),
    (200, 400),
    (300, 600),
    (400, 800),
]
QDRANT_EF = [40, 60, 80, 100, 150, 200, 300]

def recall10(ids, truth):
    return len(set(map(int, ids)) & set(map(int, truth))) / 10.0

def load_truth(size):
    out = {}
    for workload in WORKLOADS:
        p = GROUND_TRUTH / str(size) / f"{workload}.npz"
        if not p.exists():
            raise SystemExit(f"Missing ground truth: {p}")
        out[workload] = np.load(p)["top10_ids"]
    return out

def pg_conn(cfg):
    p = cfg["postgres"]
    conn = psycopg.connect(
        host=p["host"], port=p["port"], dbname=p["dbname"],
        user=p["user"], password=p["password"]
    )
    register_vector(conn)
    return conn

def pg_filter_sql(workload, qrank):
    if workload == "w2_50":
        language = "en" if qrank % 2 == 0 else "de"
        return "language = %s", [language]
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

def diskann_label_and_extra(workload, qrank):
    if workload == "w2_50":
        code = qrank % 2
        return [100 + code], "", []
    if workload == "w2_10":
        return [1000 + (qrank % 10)], "", []
    if workload == "w2_1":
        return [2000 + 1 + (qrank % 80)], "", []
    if workload == "w2_0_1":
        return [3000 + (qrank % 1000)], "", []
    if workload == "w3":
        start = qrank % 10
        cats = [(start + i) % 10 for i in range(5)]
        return (
            [2000],  # tenant 0, handled by the DiskANN label-aware index
            " AND status = 'active' "
            "AND category_id = ANY(%s) "
            "AND created_at >= TIMESTAMPTZ '2022-01-01T00:00:00Z'",
            [cats],
        )
    raise ValueError(workload)

def qdrant_filter(workload, qrank):
    if workload == "w2_50":
        value = "en" if qrank % 2 == 0 else "de"
        must = [models.FieldCondition(
            key="language", match=models.MatchValue(value=value)
        )]
    elif workload == "w2_10":
        must = [models.FieldCondition(
            key="category_id", match=models.MatchValue(value=qrank % 10)
        )]
    elif workload == "w2_1":
        must = [models.FieldCondition(
            key="tenant_id", match=models.MatchValue(value=1 + (qrank % 80))
        )]
    elif workload == "w2_0_1":
        must = [models.FieldCondition(
            key="owner_id", match=models.MatchValue(value=qrank % 1000)
        )]
    elif workload == "w3":
        start = qrank % 10
        cats = [(start + i) % 10 for i in range(5)]
        must = [
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
        ]
    else:
        raise ValueError(workload)
    return models.Filter(must=must)

def explain_pg_hnsw(conn, qvec, workload, qrank):
    where, vals = pg_filter_sql(workload, qrank)
    sql = f"""
        EXPLAIN (COSTS OFF)
        SELECT chunk_id
        FROM chunks
        WHERE {where}
        ORDER BY embedding <=> %s
        LIMIT 10
    """
    rows = conn.execute(sql, [*vals, Vector(qvec)]).fetchall()
    return "\n".join(str(r[0]) for r in rows)

def explain_pg_diskann(conn, qvec, workload, qrank):
    labels, extra, vals = diskann_label_and_extra(workload, qrank)
    sql = f"""
        EXPLAIN (COSTS OFF)
        SELECT chunk_id
        FROM chunks
        WHERE labels && %s::smallint[]{extra}
        ORDER BY embedding <=> %s
        LIMIT 10
    """
    rows = conn.execute(sql, [labels, *vals, Vector(qvec)]).fetchall()
    return "\n".join(str(r[0]) for r in rows)

def run_pg_hnsw(cfg, qemb, truths, qidx, max_scan):
    all_rows, plans = [], {}
    with pg_conn(cfg) as conn:
        conn.execute("SELECT set_config('hnsw.ef_search','60',false)")
        conn.execute("SELECT set_config('hnsw.iterative_scan','strict_order',false)")
        conn.execute("SELECT set_config('hnsw.max_scan_tuples',%s,false)", (str(max_scan),))
        conn.execute("SELECT set_config('hnsw.scan_mem_multiplier','2',false)")

        for workload in WORKLOADS:
            plans[workload] = explain_pg_hnsw(
                conn, qemb[qidx[0]], workload, int(qidx[0])
            )
            # unrecorded tuning warm-up
            for qi in qidx[:10]:
                where, vals = pg_filter_sql(workload, int(qi))
                conn.execute(
                    f"""SELECT chunk_id FROM chunks WHERE {where}
                        ORDER BY embedding <=> %s LIMIT 10""",
                    [*vals, Vector(qemb[qi])]
                ).fetchall()

            for qi in qidx:
                where, vals = pg_filter_sql(workload, int(qi))
                t0 = time.perf_counter_ns()
                rows = conn.execute(
                    f"""SELECT chunk_id FROM chunks WHERE {where}
                        ORDER BY embedding <=> %s LIMIT 10""",
                    [*vals, Vector(qemb[qi])]
                ).fetchall()
                ms = (time.perf_counter_ns() - t0) / 1e6
                ids = [r[0] for r in rows]
                all_rows.append({
                    "workload": workload,
                    "query_rank": int(qi),
                    "latency_ms": ms,
                    "recall10": recall10(ids, truths[workload][qi]),
                    "returned": len(ids),
                })
    return all_rows, plans

def run_pg_diskann(cfg, qemb, truths, qidx, search_list, rescore):
    all_rows, plans = [], {}
    with pg_conn(cfg) as conn:
        conn.execute(
            "SELECT set_config('diskann.query_search_list_size',%s,false)",
            (str(search_list),)
        )
        conn.execute(
            "SELECT set_config('diskann.query_rescore',%s,false)",
            (str(rescore),)
        )

        for workload in WORKLOADS:
            plans[workload] = explain_pg_diskann(
                conn, qemb[qidx[0]], workload, int(qidx[0])
            )

            if workload.startswith("w2") and "chunks_embedding_diskann" not in plans[workload]:
                raise RuntimeError(
                    f"{workload}: expected chunks_embedding_diskann, "
                    "but PostgreSQL chose another plan:\n" + plans[workload]
                )
            for qi in qidx[:10]:
                labels, extra, vals = diskann_label_and_extra(workload, int(qi))
                conn.execute(
                    f"""SELECT chunk_id FROM chunks
                        WHERE labels && %s::smallint[]{extra}
                        ORDER BY embedding <=> %s LIMIT 10""",
                    [labels, *vals, Vector(qemb[qi])]
                ).fetchall()

            for qi in qidx:
                labels, extra, vals = diskann_label_and_extra(workload, int(qi))
                t0 = time.perf_counter_ns()
                rows = conn.execute(
                    f"""SELECT chunk_id FROM chunks
                        WHERE labels && %s::smallint[]{extra}
                        ORDER BY embedding <=> %s LIMIT 10""",
                    [labels, *vals, Vector(qemb[qi])]
                ).fetchall()
                ms = (time.perf_counter_ns() - t0) / 1e6
                ids = [r[0] for r in rows]
                all_rows.append({
                    "workload": workload,
                    "query_rank": int(qi),
                    "latency_ms": ms,
                    "recall10": recall10(ids, truths[workload][qi]),
                    "returned": len(ids),
                })
    return all_rows, plans

def run_qdrant(cfg, qemb, truths, qidx, ef):
    q = cfg["qdrant"]
    client = QdrantClient(url=q["url"], timeout=60)
    info = client.get_collection(q["collection"])
    if int(info.config.hnsw_config.m) == 0:
        raise RuntimeError("Qdrant HNSW is disabled.")
    params = models.SearchParams(hnsw_ef=ef, exact=False)
    all_rows = []
    for workload in WORKLOADS:
        for qi in qidx[:10]:
            client.query_points(
                collection_name=q["collection"],
                query=np.asarray(qemb[qi], dtype=np.float32).tolist(),
                query_filter=qdrant_filter(workload, int(qi)),
                search_params=params,
                limit=10,
                with_payload=False,
                with_vectors=False,
            )
        for qi in qidx:
            t0 = time.perf_counter_ns()
            resp = client.query_points(
                collection_name=q["collection"],
                query=np.asarray(qemb[qi], dtype=np.float32).tolist(),
                query_filter=qdrant_filter(workload, int(qi)),
                search_params=params,
                limit=10,
                with_payload=False,
                with_vectors=False,
            )
            ms = (time.perf_counter_ns() - t0) / 1e6
            ids = [p.id for p in resp.points]
            all_rows.append({
                "workload": workload,
                "query_rank": int(qi),
                "latency_ms": ms,
                "recall10": recall10(ids, truths[workload][qi]),
                "returned": len(ids),
            })
    return all_rows, {}

def summarize(engine, params, rows):
    out = {
        "engine": engine,
        **params,
        "queries_total": len(rows),
    }
    by_workload = {}
    for workload in WORKLOADS:
        wr = [r for r in rows if r["workload"] == workload]
        rec = np.array([r["recall10"] for r in wr], dtype=float)
        lat = np.array([r["latency_ms"] for r in wr], dtype=float)
        returned = np.array([r["returned"] for r in wr], dtype=int)
        by_workload[workload] = {
            "mean_recall10": float(np.mean(rec)),
            "median_latency_ms_tuning_only": float(np.median(lat)),
            "p95_latency_ms_tuning_only": float(np.percentile(lat, 95)),
            "min_returned": int(np.min(returned)),
        }
    out["by_workload"] = by_workload
    out["min_mean_recall10_across_workloads"] = min(
        x["mean_recall10"] for x in by_workload.values()
    )
    out["median_latency_ms_all_tuning_queries"] = float(
        np.median([r["latency_ms"] for r in rows])
    )
    return out

def flatten_summary(s):
    row = {
        "engine": s["engine"],
        "min_mean_recall10": s["min_mean_recall10_across_workloads"],
        "median_all_ms_tuning_only": s["median_latency_ms_all_tuning_queries"],
    }
    for k, v in s.items():
        if k not in {"engine","by_workload",
                     "min_mean_recall10_across_workloads",
                     "median_latency_ms_all_tuning_queries",
                     "queries_total"}:
            row[k] = v
    for w, vals in s["by_workload"].items():
        row[f"{w}_recall"] = vals["mean_recall10"]
        row[f"{w}_median_ms"] = vals["median_latency_ms_tuning_only"]
        row[f"{w}_p95_ms"] = vals["p95_latency_ms_tuning_only"]
        row[f"{w}_min_returned"] = vals["min_returned"]
    return row

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--engine", choices=["pg-hnsw","pg-diskann","qdrant"], required=True)
    args = ap.parse_args()

    if args.size != 100000:
        raise SystemExit(
            "This phase is intentionally limited to 100000. "
            "Freeze the 100k methodology before scaling."
        )

    cfg = load_config()
    frozen = json.loads(FROZEN_W1.read_text())
    qidx = np.arange(0, 100, dtype=int)
    qemb = np.load(PROCESSED / "query_embeddings.npy", mmap_mode="r")
    truths = load_truth(args.size)
    TUNING_DIR.mkdir(parents=True, exist_ok=True)

    configs = []
    if args.engine == "pg-hnsw":
        configs = [{"hnsw_max_scan_tuples": x} for x in PG_HNSW_SCAN_CAPS]
    elif args.engine == "pg-diskann":
        configs = [
            {"query_search_list_size": s, "query_rescore": r}
            for s, r in PG_DISKANN_GRID
        ]
    else:
        configs = [{"hnsw_ef": x} for x in QDRANT_EF]

    summaries = []
    all_plans = {}
    for i, params in enumerate(configs, start=1):
        print(f"\n=== {args.engine} filtered config {i}/{len(configs)}: {params} ===")
        if args.engine == "pg-hnsw":
            rows, plans = run_pg_hnsw(
                cfg, qemb, truths, qidx, params["hnsw_max_scan_tuples"]
            )
            params = {
                "ef_search": frozen["w1"]["pg-hnsw"]["ef_search"],
                "iterative_scan": "strict_order",
                "scan_mem_multiplier": 2,
                **params,
            }
        elif args.engine == "pg-diskann":
            rows, plans = run_pg_diskann(
                cfg, qemb, truths, qidx,
                params["query_search_list_size"], params["query_rescore"]
            )
        else:
            rows, plans = run_qdrant(
                cfg, qemb, truths, qidx, params["hnsw_ef"]
            )

        s = summarize(args.engine, params, rows)
        summaries.append(s)
        if plans:
            all_plans[str(params)] = plans
        print(json.dumps(s, indent=2))

    eligible = [
        s for s in summaries
        if s["min_mean_recall10_across_workloads"] >= TARGET
        and all(
            vals["min_returned"] == 10
            for vals in s["by_workload"].values()
        )
    ]

    grid_path = TUNING_DIR / f"filtered_{args.engine}_{args.size}_grid.csv"
    flattened = [flatten_summary(s) for s in summaries]
    fieldnames = sorted({k for r in flattened for k in r})
    with grid_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(flattened)

    if all_plans:
        write_json(
            TUNING_DIR / f"filtered_{args.engine}_{args.size}_plans.json",
            all_plans
        )

    if not eligible:
        print(
            f"\nNo tested {args.engine} filtered configuration reached "
            f"mean Recall@10 >= {TARGET} for EVERY W2/W3 workload "
            "while returning 10 results for every tuning query."
        )
        print(f"Grid saved to {grid_path}")
        raise SystemExit(2)

    selected = min(
        eligible,
        key=lambda s: s["median_latency_ms_all_tuning_queries"]
    )
    report = {
        "purpose": "FILTERED QUERY-TIME PARAMETER SELECTION ONLY",
        "dataset_size": args.size,
        "workloads": WORKLOADS,
        "target_recall10_each_workload": TARGET,
        "tuning_query_ranks": [0, 99],
        "measurement_query_ranks_reserved": [100, 999],
        "selected": selected,
        "warning": "All latency values are tuning-only and must not be used as paper results."
    }
    out = TUNING_DIR / f"filtered_{args.engine}_{args.size}_selected.json"
    write_json(out, report)
    print("\nSELECTED FILTERED CONFIG")
    print(json.dumps(report, indent=2))
    print(f"Grid: {grid_path}")
    print(f"Selection: {out}")

if __name__ == "__main__":
    main()
