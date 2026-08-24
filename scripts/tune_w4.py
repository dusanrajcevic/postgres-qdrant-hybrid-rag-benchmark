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

TUNING_DIR = ROOT / "tuning"
TARGET = 0.95
QIDX = np.arange(0, 100, dtype=int)

HNSW_CAPS = [20000, 40000, 80000, 100000]
DISKANN_GRID = [(100,100), (150,200), (200,400), (300,600)]
QDRANT_EF = [60, 100, 150, 200, 300]
QDRANT_CANDIDATES = [64, 128, 256]

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

def load_truth(size):
    p = GROUND_TRUTH / str(size) / "w4_acl.npz"
    if not p.exists():
        raise SystemExit(f"Missing W4 ground truth: {p}")
    z = np.load(p)
    return z["top10_ids"], z["eligible_count"]

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
    return "\n".join(r[0] for r in rows)

def pg_acl_sql_direct(engine):
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

def pg_acl_params(engine, qrank, qvec):
    tenant, principal = acl_values(qrank)
    if engine == "pg-diskann":
        return [[2000 + tenant], principal, Vector(qvec)]
    return [tenant, principal, Vector(qvec)]

def pg_acl_exact_sql():
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

def validate_pg_acl_counts(c):
    bad = 0
    mins = []
    for qi in QIDX:
        tenant, principal = acl_values(int(qi))
        n = int(c.execute("""
            SELECT count(*)
            FROM chunks AS ch
            WHERE ch.tenant_id = %s
              AND ch.status = 'active'
              AND EXISTS (
                  SELECT 1 FROM document_permissions AS p
                  WHERE p.document_id = ch.document_id
                    AND p.principal_id = %s
                    AND p.permission_type = 'read'
              )
        """, (tenant, principal)).fetchone()[0])
        mins.append(n)
        if n < 10:
            bad += 1
    if bad:
        raise RuntimeError(f"{bad} W4 tuning queries have fewer than 10 eligible chunks.")
    return min(mins), max(mins), float(np.mean(mins))

def run_pg_direct(c, engine, qemb, truth, config):
    if engine == "pg-hnsw":
        c.execute("SELECT set_config('hnsw.ef_search','60',false)")
        c.execute("SELECT set_config('hnsw.iterative_scan','strict_order',false)")
        c.execute("SELECT set_config('hnsw.scan_mem_multiplier','2',false)")
        c.execute("SELECT set_config('hnsw.max_scan_tuples',%s,false)",
                  (str(config["max_scan_tuples"]),))
    else:
        c.execute("SELECT set_config('diskann.query_search_list_size',%s,false)",
                  (str(config["query_search_list_size"]),))
        c.execute("SELECT set_config('diskann.query_rescore',%s,false)",
                  (str(config["query_rescore"]),))

    sql = pg_acl_sql_direct(engine)
    plan = explain(c, sql, pg_acl_params(engine, 0, qemb[0]))

    for qi in QIDX[:10]:
        c.execute(sql, pg_acl_params(engine, int(qi), qemb[qi])).fetchall()

    lats, recs, returned = [], [], []
    for qi in QIDX:
        t0 = time.perf_counter_ns()
        rows = c.execute(sql, pg_acl_params(engine, int(qi), qemb[qi])).fetchall()
        lats.append((time.perf_counter_ns() - t0) / 1e6)
        ids = [r[0] for r in rows]
        recs.append(recall10(ids, truth[qi]))
        returned.append(len(ids))

    expected_index = "chunks_embedding_hnsw" if engine == "pg-hnsw" else "chunks_embedding_diskann"
    return {
        **config,
        "access_path": "permission-aware SQL with ANN index available",
        "mean_recall10": float(np.mean(recs)),
        "median_latency_ms_tuning_only": float(np.median(lats)),
        "p95_latency_ms_tuning_only": float(np.percentile(lats, 95)),
        "min_returned": int(np.min(returned)),
        "plan_uses_expected_ann": expected_index in plan,
        "plan": plan,
    }

def run_pg_exact(c, qemb, truth):
    sql = pg_acl_exact_sql()
    tenant, principal = acl_values(0)
    plan = explain(c, sql, [tenant, principal, Vector(qemb[0])])
    if "chunks_embedding_hnsw" in plan or "chunks_embedding_diskann" in plan:
        raise RuntimeError("W4 exact fallback unexpectedly used an ANN index:\n" + plan)

    for qi in QIDX[:10]:
        tenant, principal = acl_values(int(qi))
        c.execute(sql, [tenant, principal, Vector(qemb[qi])]).fetchall()

    lats, recs, returned = [], [], []
    for qi in QIDX:
        tenant, principal = acl_values(int(qi))
        t0 = time.perf_counter_ns()
        rows = c.execute(sql, [tenant, principal, Vector(qemb[qi])]).fetchall()
        lats.append((time.perf_counter_ns() - t0) / 1e6)
        ids = [r[0] for r in rows]
        recs.append(recall10(ids, truth[qi]))
        returned.append(len(ids))

    out = {
        "access_path": "scalar-metadata + live relational ACL + exact vector ranking",
        "mean_recall10": float(np.mean(recs)),
        "median_latency_ms_tuning_only": float(np.median(lats)),
        "p95_latency_ms_tuning_only": float(np.percentile(lats, 95)),
        "min_returned": int(np.min(returned)),
        "plan": plan,
    }
    if out["mean_recall10"] < 0.999999 or out["min_returned"] != 10:
        raise RuntimeError("W4 exact PostgreSQL path failed exactness/completeness: " + json.dumps(out))
    return out

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
    params = models.SearchParams(exact=True) if exact else models.SearchParams(
        hnsw_ef=ef, exact=False
    )

    with pg_conn(cfg) as c:
        # Warmups include both Qdrant candidate generation and PostgreSQL ACL validation.
        for qi in QIDX[:10]:
            _, principal = acl_values(int(qi))
            resp = client.query_points(
                collection_name=q["collection"],
                query=np.asarray(qemb[qi], dtype=np.float32).tolist(),
                query_filter=qdrant_filter(int(qi)),
                search_params=params,
                limit=candidate_limit,
                with_payload=False,
                with_vectors=False,
            )
            ids = [int(p.id) for p in resp.points]
            validate_permissions(c, ids, principal)

        lats, recs, returned = [], [], []
        candidate_counts = []
        for qi in QIDX:
            _, principal = acl_values(int(qi))
            t0 = time.perf_counter_ns()
            resp = client.query_points(
                collection_name=q["collection"],
                query=np.asarray(qemb[qi], dtype=np.float32).tolist(),
                query_filter=qdrant_filter(int(qi)),
                search_params=params,
                limit=candidate_limit,
                with_payload=False,
                with_vectors=False,
            )
            candidate_ids = [int(p.id) for p in resp.points]
            allowed = validate_permissions(c, candidate_ids, principal)
            final_ids = [pid for pid in candidate_ids if pid in allowed][:10]
            lats.append((time.perf_counter_ns() - t0) / 1e6)
            recs.append(recall10(final_ids, truth[qi]))
            returned.append(len(final_ids))
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
        "mean_recall10": float(np.mean(recs)),
        "median_latency_ms_tuning_only": float(np.median(lats)),
        "p95_latency_ms_tuning_only": float(np.percentile(lats, 95)),
        "min_returned": int(np.min(returned)),
        "min_candidates_returned_by_qdrant": int(np.min(candidate_counts)),
    }

def tune_pg(cfg, engine, size, qemb, truth):
    with pg_conn(cfg) as c:
        expected = ["chunks_embedding_hnsw"] if engine == "pg-hnsw" else ["chunks_embedding_diskann"]
        found = ann_indexes(c)
        if found != expected:
            raise RuntimeError(f"Expected ANN indexes {expected}; found {found}")

        mn, mx, avg = validate_pg_acl_counts(c)
        print(json.dumps({
            "w4_acl_eligible_chunks_tuning_queries": {
                "min": mn, "max": mx, "mean": avg
            }
        }, indent=2))

        direct = []
        if engine == "pg-hnsw":
            configs = [{"max_scan_tuples": x} for x in HNSW_CAPS]
        else:
            configs = [
                {"query_search_list_size": s, "query_rescore": r}
                for s, r in DISKANN_GRID
            ]

        plans = {}
        for config in configs:
            print(f"\n{engine} W4 direct config: {config}")
            row = run_pg_direct(c, engine, qemb, truth, config)
            plans[str(config)] = row.pop("plan")
            direct.append(row)
            print(json.dumps(row, indent=2))

        print("\nPostgreSQL W4 exact relational fallback")
        exact = run_pg_exact(c, qemb, truth)
        exact_plan = exact.pop("plan")
        print(json.dumps(exact, indent=2))

    eligible = [
        r for r in direct
        if r["mean_recall10"] >= TARGET and r["min_returned"] == 10
    ]
    eligible.append(exact)
    selected = min(eligible, key=lambda r: r["median_latency_ms_tuning_only"])

    report = {
        "purpose": f"{engine.upper()} W4 ACL STRATEGY SELECTION ONLY",
        "dataset_size": size,
        "workload": "w4_acl",
        "target_recall10": TARGET,
        "tuning_query_ranks": [0,99],
        "measurement_query_ranks_reserved": [100,999],
        "selected": selected,
        "warning": "All latency values are tuning-only, not paper results."
    }
    write_json(TUNING_DIR / f"w4_{engine}_100000_selected.json", report)
    write_json(TUNING_DIR / f"w4_{engine}_100000_plans.json", {
        "direct": plans, "exact": exact_plan
    })
    print(f"\nSELECTED {engine.upper()} W4 STRATEGY")
    print(json.dumps(report, indent=2))

def tune_qdrant(cfg, size, qemb, truth):
    q = cfg["qdrant"]
    client = QdrantClient(url=q["url"], timeout=120)
    n = int(client.count(collection_name=q["collection"], exact=True).count)
    if n != size:
        raise RuntimeError(f"Qdrant has {n:,} points, expected {size:,}")
    info = client.get_collection(q["collection"])
    if int(info.config.hnsw_config.m) == 0:
        raise RuntimeError("Qdrant HNSW is disabled.")

    rows = []
    for ef in QDRANT_EF:
        for cand in QDRANT_CANDIDATES:
            print(f"\nQdrant W4 approximate: ef={ef}, candidates={cand}")
            r = run_qdrant_mode(cfg, qemb, truth, False, ef, cand)
            rows.append(r)
            print(json.dumps(r, indent=2))

    for cand in QDRANT_CANDIDATES:
        print(f"\nQdrant W4 exact candidates: candidates={cand}")
        r = run_qdrant_mode(cfg, qemb, truth, True, None, cand)
        rows.append(r)
        print(json.dumps(r, indent=2))

    eligible = [
        r for r in rows
        if r["mean_recall10"] >= TARGET and r["min_returned"] == 10
    ]
    if not eligible:
        raise SystemExit(
            "No Qdrant+PostgreSQL W4 configuration met Recall@10 >= 0.95 "
            "and returned 10 ACL-valid results for every tuning query."
        )
    selected = min(eligible, key=lambda r: r["median_latency_ms_tuning_only"])

    report = {
        "purpose": "QDRANT+POSTGRESQL W4 ACL STRATEGY SELECTION ONLY",
        "dataset_size": size,
        "workload": "w4_acl",
        "target_recall10": TARGET,
        "tuning_query_ranks": [0,99],
        "measurement_query_ranks_reserved": [100,999],
        "permissions_denormalized_into_qdrant": False,
        "latency_scope": (
            "End-to-end: Qdrant candidate search plus PostgreSQL permission validation "
            "inside the same timed request."
        ),
        "candidate_pagination": (
            "Single Qdrant request per query; candidate_limit is tuned. "
            "No approximate offset pagination is used."
        ),
        "selected": selected,
        "warning": "All latency values are tuning-only, not paper results."
    }
    write_json(TUNING_DIR / "w4_qdrant_100000_selected.json", report)
    print("\nSELECTED QDRANT+POSTGRESQL W4 STRATEGY")
    print(json.dumps(report, indent=2))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument(
        "--engine", required=True,
        choices=["pg-hnsw", "pg-diskann", "qdrant"]
    )
    args = ap.parse_args()
    if args.size != 100000:
        raise SystemExit("Freeze W4 at 100k before scaling.")

    cfg = load_config()
    qemb = np.load(PROCESSED / "query_embeddings.npy", mmap_mode="r")
    truth, eligible_count = load_truth(args.size)
    if int(np.min(eligible_count[QIDX])) < 10:
        raise RuntimeError("W4 ground truth has a tuning query with fewer than 10 eligible vectors.")

    TUNING_DIR.mkdir(parents=True, exist_ok=True)
    if args.engine in {"pg-hnsw","pg-diskann"}:
        tune_pg(cfg, args.engine, args.size, qemb, truth)
    else:
        tune_qdrant(cfg, args.size, qemb, truth)

if __name__ == "__main__":
    main()
