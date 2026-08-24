from __future__ import annotations

import argparse
import json
import time

import numpy as np
import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from qdrant_client import QdrantClient, models

from common import GROUND_TRUTH, PROCESSED, ROOT, load_config, write_json

SIZE = 250000
TARGET = 0.95
QIDX = np.arange(0, 100, dtype=int)
TUNING_DIR = ROOT / "tuning"

HNSW_CAPS = [20000, 40000, 80000, 125000, 250000]
DISKANN_GRID = [
    (75, 100),
    (100, 100),
    (100, 200),
    (150, 200),
    (200, 400),
    (300, 600),
    (400, 800),
]
QDRANT_EF = [60, 80, 100, 150, 200, 300, 500]
QDRANT_CANDIDATES = [64, 128, 256, 512]

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

def load_truth():
    p = GROUND_TRUTH / str(SIZE) / "w4_acl.npz"
    if not p.exists():
        raise SystemExit(f"Missing W4 ground truth: {p}")
    z = np.load(p)
    top = z["top10_ids"]
    eligible = z["eligible_count"] if "eligible_count" in z.files else None
    if top.shape != (1000, 10):
        raise RuntimeError(f"Expected W4 top10 shape (1000,10), got {top.shape}")
    if eligible is not None and int(np.min(eligible[QIDX])) < 10:
        raise RuntimeError("At least one W4 tuning query has fewer than 10 eligible vectors.")
    return top, eligible

def acl_values(qrank):
    return 1 + (qrank % 80), qrank % 5

def ann_indexes(c):
    return [r[0] for r in c.execute("""
        SELECT indexname
        FROM pg_indexes
        WHERE tablename='chunks'
          AND indexname IN ('chunks_embedding_hnsw','chunks_embedding_diskann')
        ORDER BY indexname
    """).fetchall()]

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

def pg_direct_sql(engine):
    if engine == "pg-diskann":
        return """
            SELECT ch.chunk_id
            FROM chunks AS ch
            WHERE ch.labels && %s::smallint[]
              AND ch.status = 'active'
              AND EXISTS (
                  SELECT 1
                  FROM document_permissions AS p
                  WHERE p.document_id = ch.document_id
                    AND p.principal_id = %s
                    AND p.permission_type = 'read'
              )
            ORDER BY ch.embedding <=> %s
            LIMIT 10
        """
    return """
        SELECT ch.chunk_id
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
        ORDER BY ch.embedding <=> %s
        LIMIT 10
    """

def pg_direct_params(engine, qrank, qvec):
    tenant, principal = acl_values(qrank)
    if engine == "pg-diskann":
        return [[2000 + tenant], principal, Vector(qvec)]
    return [tenant, principal, Vector(qvec)]

def pg_exact_sql():
    return """
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

def validate_acl_counts(c):
    vals = []
    for qi in QIDX:
        tenant, principal = acl_values(int(qi))
        n = int(c.execute("""
            SELECT count(*)
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
        """, (tenant, principal)).fetchone()[0])
        vals.append(n)
    if min(vals) < 10:
        raise RuntimeError(f"W4 ACL eligible count below 10: min={min(vals)}")
    return {"min": min(vals), "max": max(vals), "mean": float(np.mean(vals))}

def run_pg_exact(c, qemb, truth):
    sql = pg_exact_sql()
    tenant, principal = acl_values(0)
    plan = explain(c, sql, [tenant, principal, Vector(qemb[0])])
    if "chunks_embedding_hnsw" in plan or "chunks_embedding_diskann" in plan:
        raise RuntimeError("Exact W4 fallback unexpectedly uses ANN:\n" + plan)

    for qi in QIDX[:10]:
        tenant, principal = acl_values(int(qi))
        c.execute(sql, [tenant, principal, Vector(qemb[qi])]).fetchall()

    lats, recs, ret = [], [], []
    for qi in QIDX:
        tenant, principal = acl_values(int(qi))
        t0 = time.perf_counter_ns()
        rows = c.execute(sql, [tenant, principal, Vector(qemb[qi])]).fetchall()
        lats.append((time.perf_counter_ns() - t0) / 1e6)
        ids = [int(r[0]) for r in rows]
        recs.append(recall10(ids, truth[qi]))
        ret.append(len(ids))

    out = {
        "access_path": "scalar-metadata + live relational ACL + exact vector ranking",
        **summarize(lats, recs, ret),
    }
    if out["mean_recall10"] < 0.999999 or out["min_returned"] != 10:
        raise RuntimeError(f"Exact W4 fallback failed: {out}")
    return out, plan

def run_pg_direct(c, engine, qemb, truth, config):
    if engine == "pg-hnsw":
        c.execute("SELECT set_config('hnsw.ef_search','60',false)")
        c.execute("SELECT set_config('hnsw.iterative_scan','strict_order',false)")
        c.execute("SELECT set_config('hnsw.scan_mem_multiplier','2',false)")
        c.execute(
            "SELECT set_config('hnsw.max_scan_tuples',%s,false)",
            (str(config["max_scan_tuples"]),)
        )
    else:
        c.execute(
            "SELECT set_config('diskann.query_search_list_size',%s,false)",
            (str(config["query_search_list_size"]),)
        )
        c.execute(
            "SELECT set_config('diskann.query_rescore',%s,false)",
            (str(config["query_rescore"]),)
        )

    sql = pg_direct_sql(engine)
    plan = explain(c, sql, pg_direct_params(engine, 0, qemb[0]))

    for qi in QIDX[:10]:
        c.execute(sql, pg_direct_params(engine, int(qi), qemb[qi])).fetchall()

    lats, recs, ret = [], [], []
    for qi in QIDX:
        t0 = time.perf_counter_ns()
        rows = c.execute(sql, pg_direct_params(engine, int(qi), qemb[qi])).fetchall()
        lats.append((time.perf_counter_ns() - t0) / 1e6)
        ids = [int(r[0]) for r in rows]
        recs.append(recall10(ids, truth[qi]))
        ret.append(len(ids))

    expected = "chunks_embedding_hnsw" if engine == "pg-hnsw" else "chunks_embedding_diskann"
    return {
        **config,
        "access_path": (
            "permission-aware SQL with ANN plan"
            if expected in plan
            else "planner-selected non-ANN permission-aware SQL"
        ),
        **summarize(lats, recs, ret),
        "plan_uses_expected_ann": expected in plan,
    }, plan

def tune_pg(cfg, engine, qemb, truth):
    with pg_conn(cfg) as c:
        expected = ["chunks_embedding_hnsw"] if engine == "pg-hnsw" else ["chunks_embedding_diskann"]
        found = ann_indexes(c)
        if found != expected:
            raise RuntimeError(f"Expected ANN index state {expected}; found {found}")

        counts = validate_acl_counts(c)
        print(json.dumps({"w4_acl_eligible_chunks_tuning_queries": counts}, indent=2))

        exact, exact_plan = run_pg_exact(c, qemb, truth)
        print("\nExact relational fallback:")
        print(json.dumps(exact, indent=2))

        if engine == "pg-hnsw":
            configs = [{"max_scan_tuples": x} for x in HNSW_CAPS]
        else:
            configs = [
                {"query_search_list_size": s, "query_rescore": r}
                for s, r in DISKANN_GRID
            ]

        rows = []
        plans = {}
        for config in configs:
            print(f"\n{engine} W4 direct config: {config}")
            row, plan = run_pg_direct(c, engine, qemb, truth, config)
            rows.append(row)
            plans[str(config)] = plan
            print(json.dumps(row, indent=2))

    candidates = [r for r in rows if valid(r)]
    candidates.append(exact)
    selected = min(candidates, key=lambda r: r["median_latency_ms_tuning_only"])

    report = {
        "purpose": f"{engine.upper()} 250K W4 ACL STRATEGY SELECTION ONLY",
        "dataset_size": SIZE,
        "workload": "w4_acl",
        "target_recall10": TARGET,
        "tuning_query_ranks": [0,99],
        "measurement_query_ranks_reserved": [100,999],
        "selected": selected,
        "warning": "All latency values are tuning-only, not paper results.",
    }
    write_json(TUNING_DIR / f"w4_{engine}_250000_selected.json", report)
    write_json(TUNING_DIR / f"w4_{engine}_250000_plans.json", {
        "direct": plans,
        "exact": exact_plan,
    })
    print(f"\nSELECTED {engine.upper()} 250K W4 STRATEGY")
    print(json.dumps(report, indent=2))

def qdrant_filter(qrank):
    tenant, _ = acl_values(qrank)
    return models.Filter(must=[
        models.FieldCondition(
            key="tenant_id", match=models.MatchValue(value=tenant)
        ),
        models.FieldCondition(
            key="status", match=models.MatchValue(value="active")
        ),
    ])

def validate_permissions(c, candidate_ids, principal):
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

def run_qdrant_mode(cfg, qemb, truth, exact, ef, candidate_limit):
    q = cfg["qdrant"]
    client = QdrantClient(url=q["url"], timeout=120)
    params = (
        models.SearchParams(exact=True)
        if exact else
        models.SearchParams(hnsw_ef=int(ef), exact=False)
    )

    with pg_conn(cfg) as c:
        for qi in QIDX[:10]:
            _, principal = acl_values(int(qi))
            resp = client.query_points(
                collection_name=q["collection"],
                query=np.asarray(qemb[qi], dtype=np.float32).tolist(),
                query_filter=qdrant_filter(int(qi)),
                search_params=params,
                limit=int(candidate_limit),
                with_payload=False,
                with_vectors=False,
            )
            ids = [int(p.id) for p in resp.points]
            validate_permissions(c, ids, principal)

        lats, recs, ret, candidate_counts = [], [], [], []
        for qi in QIDX:
            _, principal = acl_values(int(qi))
            t0 = time.perf_counter_ns()
            resp = client.query_points(
                collection_name=q["collection"],
                query=np.asarray(qemb[qi], dtype=np.float32).tolist(),
                query_filter=qdrant_filter(int(qi)),
                search_params=params,
                limit=int(candidate_limit),
                with_payload=False,
                with_vectors=False,
            )
            candidate_ids = [int(p.id) for p in resp.points]
            allowed = validate_permissions(c, candidate_ids, principal)
            final_ids = [pid for pid in candidate_ids if pid in allowed][:10]
            lats.append((time.perf_counter_ns() - t0) / 1e6)
            recs.append(recall10(final_ids, truth[qi]))
            ret.append(len(final_ids))
            candidate_counts.append(len(candidate_ids))

    return {
        "access_path": (
            "Qdrant filtered exact candidates + PostgreSQL ACL validation"
            if exact else
            "Qdrant filtered approximate candidates + PostgreSQL ACL validation"
        ),
        "exact": bool(exact),
        "hnsw_ef": None if exact else int(ef),
        "candidate_limit": int(candidate_limit),
        **summarize(lats, recs, ret),
        "min_candidates_returned_by_qdrant": int(np.min(candidate_counts)),
    }

def tune_qdrant(cfg, qemb, truth):
    q = cfg["qdrant"]
    client = QdrantClient(url=q["url"], timeout=120)
    n = int(client.count(collection_name=q["collection"], exact=True).count)
    if n != SIZE:
        raise RuntimeError(f"Qdrant has {n:,} points, expected {SIZE:,}")
    info = client.get_collection(q["collection"])
    if int(info.config.hnsw_config.m) == 0:
        raise RuntimeError("Qdrant HNSW is disabled.")

    rows = []
    for ef in QDRANT_EF:
        for cand in QDRANT_CANDIDATES:
            print(f"\nQdrant W4 approximate: ef={ef}, candidates={cand}")
            row = run_qdrant_mode(cfg, qemb, truth, False, ef, cand)
            rows.append(row)
            print(json.dumps(row, indent=2))

    for cand in QDRANT_CANDIDATES:
        print(f"\nQdrant W4 exact: candidates={cand}")
        row = run_qdrant_mode(cfg, qemb, truth, True, None, cand)
        rows.append(row)
        print(json.dumps(row, indent=2))

    candidates = [r for r in rows if valid(r)]
    if not candidates:
        raise SystemExit(
            "No Qdrant+PostgreSQL W4 configuration met Recall@10 >= 0.95 "
            "and returned 10 ACL-valid results for every tuning query."
        )
    selected = min(candidates, key=lambda r: r["median_latency_ms_tuning_only"])

    report = {
        "purpose": "QDRANT+POSTGRESQL 250K W4 ACL STRATEGY SELECTION ONLY",
        "dataset_size": SIZE,
        "workload": "w4_acl",
        "target_recall10": TARGET,
        "tuning_query_ranks": [0,99],
        "measurement_query_ranks_reserved": [100,999],
        "permissions_denormalized_into_qdrant": False,
        "latency_scope": (
            "End-to-end: Qdrant candidate search plus PostgreSQL permission "
            "validation inside the same timed request."
        ),
        "selected": selected,
        "warning": "All latency values are tuning-only, not paper results.",
    }
    write_json(TUNING_DIR / "w4_qdrant_250000_selected.json", report)
    write_json(TUNING_DIR / "w4_qdrant_250000_grid.json", rows)
    print("\nSELECTED QDRANT+POSTGRESQL 250K W4 STRATEGY")
    print(json.dumps(report, indent=2))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--engine",
        required=True,
        choices=["pg-hnsw", "pg-diskann", "qdrant"],
    )
    args = ap.parse_args()

    cfg = load_config()
    qemb = np.load(PROCESSED / "query_embeddings.npy", mmap_mode="r")
    truth, _ = load_truth()
    TUNING_DIR.mkdir(parents=True, exist_ok=True)

    if args.engine in {"pg-hnsw", "pg-diskann"}:
        tune_pg(cfg, args.engine, qemb, truth)
    else:
        tune_qdrant(cfg, qemb, truth)

if __name__ == "__main__":
    main()
