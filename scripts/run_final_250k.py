from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from qdrant_client import QdrantClient, models

from common import GROUND_TRUTH, PROCESSED, ROOT, load_config, write_json

SIZE = 250000
FROZEN_PATH = ROOT / "config" / "final_250k_measurement.json"
RESULTS_ROOT = ROOT / "results" / "measurements" / str(SIZE)
WORKLOADS = ["w1", "w2_50", "w2_10", "w2_1", "w2_0_1", "w3", "w4_acl"]

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
        SELECT indexname FROM pg_indexes
        WHERE tablename='chunks'
          AND indexname IN ('chunks_embedding_hnsw','chunks_embedding_diskann')
        ORDER BY indexname
    """).fetchall()]

def load_truths():
    out = {}
    for workload in WORKLOADS:
        p = GROUND_TRUTH / str(SIZE) / f"{workload}.npz"
        if not p.exists():
            raise RuntimeError(f"Missing ground truth: {p}")
        z = np.load(p)
        top = z["top10_ids"]
        if top.shape != (1000, 10):
            raise RuntimeError(f"{p}: expected (1000,10), got {top.shape}")
        out[workload] = top
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

def diskann_filter(workload, qrank):
    if workload == "w2_50":
        return "labels && %s::smallint[]", [[100 + (qrank % 2)]]
    if workload == "w2_10":
        return "labels && %s::smallint[]", [[1000 + (qrank % 10)]]
    if workload == "w3":
        start = qrank % 10
        cats = [(start + i) % 10 for i in range(5)]
        return (
            "labels && %s::smallint[] "
            "AND status = 'active' "
            "AND category_id = ANY(%s) "
            "AND created_at >= TIMESTAMPTZ '2022-01-01T00:00:00Z'",
            [[2000], cats],
        )
    raise ValueError(workload)

def acl_values(qrank):
    return 1 + (qrank % 80), qrank % 5

def exact_sql(workload, qrank):
    where, vals = scalar_filter(workload, qrank)
    return f"""
        WITH eligible AS MATERIALIZED (
            SELECT chunk_id, embedding
            FROM chunks
            WHERE {where}
        )
        SELECT chunk_id
        FROM eligible
        ORDER BY embedding <=> %s
        LIMIT 10
    """, vals

def acl_exact_sql(qrank):
    tenant, principal = acl_values(qrank)
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
    """, [tenant, principal]

def acl_direct_sql(qrank):
    tenant, principal = acl_values(qrank)
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
    """, [tenant, principal]

def setup_pg(c, engine, strategy):
    path = strategy["path"]
    if engine == "pg-hnsw":
        if path == "hnsw":
            c.execute("SELECT set_config('hnsw.ef_search',%s,false)",
                      (str(strategy["ef_search"]),))
            c.execute("SELECT set_config('hnsw.iterative_scan','off',false)")
        elif path in {"planner_filtered_query_scalar_expected", "planner_acl_scalar_expected"}:
            c.execute("SELECT set_config('hnsw.ef_search',%s,false)",
                      (str(strategy["ef_search"]),))
            c.execute("SELECT set_config('hnsw.iterative_scan',%s,false)",
                      (strategy["iterative_scan"],))
            c.execute("SELECT set_config('hnsw.max_scan_tuples',%s,false)",
                      (str(strategy["max_scan_tuples"]),))
            c.execute("SELECT set_config('hnsw.scan_mem_multiplier',%s,false)",
                      (str(strategy["scan_mem_multiplier"]),))
    else:
        if path in {"diskann", "label_aware_diskann", "label_aware_diskann_w3"}:
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

    if path in {"label_aware_diskann", "label_aware_diskann_w3"}:
        where, vals = diskann_filter(workload, qrank)
        rows = c.execute(f"""
            SELECT chunk_id FROM chunks
            WHERE {where}
            ORDER BY embedding <=> %s
            LIMIT 10
        """, [*vals, Vector(qvec)]).fetchall()
        return [int(r[0]) for r in rows]

    if path == "planner_filtered_query_scalar_expected":
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

    if path == "planner_acl_scalar_expected":
        sql, vals = acl_direct_sql(qrank)
        rows = c.execute(sql, [*vals, Vector(qvec)]).fetchall()
        return [int(r[0]) for r in rows]

    raise ValueError(f"Unsupported PostgreSQL path: {path}")

def pg_plan(c, engine, workload, qvec, strategy):
    setup_pg(c, engine, strategy)
    path = strategy["path"]

    if workload == "w1":
        sql = """SELECT chunk_id FROM chunks
                 ORDER BY embedding <=> %s LIMIT 10"""
        params = [Vector(qvec)]
    elif path in {"label_aware_diskann", "label_aware_diskann_w3"}:
        where, vals = diskann_filter(workload, 0)
        sql = f"""SELECT chunk_id FROM chunks
                  WHERE {where}
                  ORDER BY embedding <=> %s LIMIT 10"""
        params = [*vals, Vector(qvec)]
    elif path == "planner_filtered_query_scalar_expected":
        where, vals = scalar_filter(workload, 0)
        sql = f"""SELECT chunk_id FROM chunks
                  WHERE {where}
                  ORDER BY embedding <=> %s LIMIT 10"""
        params = [*vals, Vector(qvec)]
    elif path == "scalar_first_exact":
        sql, vals = exact_sql(workload, 0)
        params = [*vals, Vector(qvec)]
    elif path == "scalar_acl_exact":
        sql, vals = acl_exact_sql(0)
        params = [*vals, Vector(qvec)]
    elif path == "planner_acl_scalar_expected":
        sql, vals = acl_direct_sql(0)
        params = [*vals, Vector(qvec)]
    else:
        raise ValueError(path)

    rows = c.execute("EXPLAIN (COSTS OFF) " + sql, params).fetchall()
    text = "\n".join(str(r[0]) for r in rows)

    if engine == "pg-hnsw":
        if path == "hnsw" and "chunks_embedding_hnsw" not in text:
            raise RuntimeError(f"{workload}: expected HNSW plan, got:\n{text}")
        if path in {
            "scalar_first_exact",
            "planner_filtered_query_scalar_expected",
            "planner_acl_scalar_expected",
        } and "chunks_embedding_hnsw" in text:
            raise RuntimeError(
                f"{workload}: frozen strategy expects a non-HNSW path, got:\n{text}"
            )
        if path == "scalar_first_exact" and "chunks_embedding_diskann" in text:
            raise RuntimeError(f"{workload}: exact path unexpectedly uses DiskANN:\n{text}")
    else:
        if path in {"diskann", "label_aware_diskann", "label_aware_diskann_w3"}:
            if "chunks_embedding_diskann" not in text:
                raise RuntimeError(f"{workload}: expected DiskANN plan, got:\n{text}")
        if path in {"scalar_first_exact", "scalar_acl_exact"}:
            if "chunks_embedding_hnsw" in text or "chunks_embedding_diskann" in text:
                raise RuntimeError(f"{workload}: exact path unexpectedly uses ANN:\n{text}")
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
            models.FieldCondition(key="tenant_id", match=models.MatchValue(value=0)),
            models.FieldCondition(key="status", match=models.MatchValue(value="active")),
            models.FieldCondition(key="category_id", match=models.MatchAny(any=cats)),
            models.FieldCondition(
                key="created_at",
                range=models.DatetimeRange(gte="2022-01-01T00:00:00Z")
            ),
        ])
    if workload == "w4_acl":
        tenant, _ = acl_values(qrank)
        return models.Filter(must=[
            models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tenant)),
            models.FieldCondition(key="status", match=models.MatchValue(value="active")),
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
        limit, qf = 10, None
    elif path == "filtered_approx":
        params = models.SearchParams(hnsw_ef=int(strategy["hnsw_ef"]), exact=False)
        limit, qf = 10, qdrant_filter(workload, qrank)
    elif path == "qdrant_candidates_plus_postgres_acl":
        params = models.SearchParams(hnsw_ef=int(strategy["hnsw_ef"]), exact=False)
        limit, qf = int(strategy["candidate_limit"]), qdrant_filter(workload, qrank)
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

def orders(frozen, repetition, workload):
    seed = int(frozen["measurement_order_seed"])
    offset = WORKLOADS.index(workload) * 10000
    rng = np.random.default_rng(seed + repetition * 1000 + offset)
    measured = np.arange(100, 1000, dtype=int)
    rng.shuffle(measured)
    warm = np.tile(np.arange(0, 100, dtype=int), 2)
    rng2 = np.random.default_rng(seed + repetition * 1000 + offset + 777)
    rng2.shuffle(warm)
    return warm.tolist(), measured.tolist()

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
        "queries_with_fewer_than_10": int(np.sum(returned < 10)),
        "queries_below_recall_0_95": int(np.sum(rec < 0.95)),
    }

def write_rows(path, rows):
    fields = [
        "engine", "dataset_size", "repetition", "workload", "query_rank",
        "latency_ms", "recall10", "returned_count", "candidate_count",
        "execution_path"
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

def ensure_clean(engine, overwrite):
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    paths = list(RESULTS_ROOT.glob(f"{engine}_rep*.csv"))
    paths += list(RESULTS_ROOT.glob(f"{engine}_rep*_summary.json"))
    paths += [RESULTS_ROOT / f"{engine}_plans.json"]
    existing = [p for p in paths if p.exists()]
    if existing and not overwrite:
        raise SystemExit(
            "Existing paper-measurement files found; refusing to overwrite:\n" +
            "\n".join(map(str, existing))
        )
    if overwrite:
        for p in existing:
            p.unlink()

def validate_state(cfg, engine):
    with pg_conn(cfg) as c:
        n = int(c.execute("SELECT count(*) FROM chunks").fetchone()[0])
        if n != SIZE:
            raise RuntimeError(f"PostgreSQL has {n:,} chunks; expected {SIZE:,}")
        if engine.startswith("pg-"):
            expected = ["chunks_embedding_hnsw"] if engine == "pg-hnsw" else ["chunks_embedding_diskann"]
            found = ann_indexes(c)
            if found != expected:
                raise RuntimeError(f"{engine}: expected {expected}, found {found}")

    if engine == "qdrant":
        q = cfg["qdrant"]
        client = QdrantClient(url=q["url"], timeout=120)
        n = int(client.count(collection_name=q["collection"], exact=True).count)
        if n != SIZE:
            raise RuntimeError(f"Qdrant has {n:,} points; expected {SIZE:,}")
        info = client.get_collection(q["collection"])
        if int(info.config.hnsw_config.m) == 0:
            raise RuntimeError("Qdrant HNSW is disabled.")

def write_manifest(engine, frozen):
    p = RESULTS_ROOT / "measurement_manifest.json"
    data = {
        "purpose": "HELD-OUT 250K PAPER MEASUREMENT MANIFEST",
        "dataset_size": SIZE,
        "measurement_query_ranks": frozen["measurement_query_ranks"],
        "measurement_query_count": frozen["measurement_query_count"],
        "warmup_executions_per_workload_per_repetition":
            frozen["warmup_executions_per_workload_per_repetition"],
        "warmup_source_query_ranks": frozen["warmup_source_query_ranks"],
        "repetitions": frozen["repetitions"],
        "config_sha256": sha256_file(FROZEN_PATH),
        "runner_sha256": sha256_file(Path(__file__)),
        "python": sys.version,
        "platform": platform.platform(),
        "engines_started": [],
        "note": "No tuning is performed. Held-out anomalies are recorded, not retuned."
    }
    if p.exists():
        old = json.loads(p.read_text())
        if old.get("config_sha256") != data["config_sha256"]:
            raise RuntimeError("Existing manifest uses a different frozen config.")
        data = old
        data["runner_sha256"] = sha256_file(Path(__file__))
    if engine not in data["engines_started"]:
        data["engines_started"].append(engine)
    write_json(p, data)

def print_anomaly_warning(workload, summary):
    if summary["mean_recall10"] < 0.95 or summary["min_returned"] < 10:
        print(
            f"WARNING: held-out {workload} result does not satisfy the tuning criterion. "
            f"Recording without retuning: {summary}",
            flush=True,
        )

def run_postgres(engine, cfg, frozen, qemb, truths):
    strategies = frozen["strategies"][engine]
    plans = {}
    with pg_conn(cfg) as c:
        for w in WORKLOADS:
            plans[w] = pg_plan(c, engine, w, qemb[0], strategies[w])
    write_json(RESULTS_ROOT / f"{engine}_plans.json", plans)

    for rep in range(1, 4):
        print(f"\n=== {engine} repetition {rep}/3 ===")
        all_rows = []
        with pg_conn(cfg) as c:
            for w in WORKLOADS:
                s = strategies[w]
                setup_pg(c, engine, s)
                warm, measured = orders(frozen, rep, w)

                print(f"{w}: 200 unrecorded warmups")
                for qi in warm:
                    pg_query(c, engine, w, qi, qemb[qi], s)

                rows = []
                print(f"{w}: measuring 900 held-out queries")
                for pos, qi in enumerate(measured, 1):
                    t0 = time.perf_counter_ns()
                    ids = pg_query(c, engine, w, qi, qemb[qi], s)
                    ms = (time.perf_counter_ns() - t0) / 1e6
                    rows.append({
                        "engine": engine,
                        "dataset_size": SIZE,
                        "repetition": rep,
                        "workload": w,
                        "query_rank": qi,
                        "latency_ms": ms,
                        "recall10": recall10(ids, truths[w][qi]),
                        "returned_count": len(ids),
                        "candidate_count": "",
                        "execution_path": s["path"],
                    })
                    if pos % 100 == 0:
                        print(f"  {w}: {pos}/900", flush=True)

                summary = summarize(rows)
                print(json.dumps({w: summary}, indent=2))
                print_anomaly_warning(w, summary)
                all_rows.extend(rows)

        out = RESULTS_ROOT / f"{engine}_rep{rep}.csv"
        write_rows(out, all_rows)
        rep_summary = {
            "engine": engine,
            "dataset_size": SIZE,
            "repetition": rep,
            "workloads": {
                w: summarize([r for r in all_rows if r["workload"] == w])
                for w in WORKLOADS
            },
            "warning": "These are held-out paper measurements, not tuning values."
        }
        write_json(RESULTS_ROOT / f"{engine}_rep{rep}_summary.json", rep_summary)
        print(f"Wrote {out}")

def run_qdrant(cfg, frozen, qemb, truths):
    q = cfg["qdrant"]
    strategies = frozen["strategies"]["qdrant"]
    client = QdrantClient(url=q["url"], timeout=120)

    with pg_conn(cfg) as pgc:
        for rep in range(1, 4):
            print(f"\n=== qdrant repetition {rep}/3 ===")
            all_rows = []
            for w in WORKLOADS:
                s = strategies[w]
                warm, measured = orders(frozen, rep, w)

                print(f"{w}: 200 unrecorded warmups")
                for qi in warm:
                    qdrant_query(client, pgc, q["collection"], w, qi, qemb[qi], s)

                rows = []
                print(f"{w}: measuring 900 held-out queries")
                for pos, qi in enumerate(measured, 1):
                    t0 = time.perf_counter_ns()
                    ids, candidate_count = qdrant_query(
                        client, pgc, q["collection"], w, qi, qemb[qi], s
                    )
                    ms = (time.perf_counter_ns() - t0) / 1e6
                    rows.append({
                        "engine": "qdrant",
                        "dataset_size": SIZE,
                        "repetition": rep,
                        "workload": w,
                        "query_rank": qi,
                        "latency_ms": ms,
                        "recall10": recall10(ids, truths[w][qi]),
                        "returned_count": len(ids),
                        "candidate_count": candidate_count,
                        "execution_path": s["path"],
                    })
                    if pos % 100 == 0:
                        print(f"  {w}: {pos}/900", flush=True)

                summary = summarize(rows)
                print(json.dumps({w: summary}, indent=2))
                print_anomaly_warning(w, summary)
                all_rows.extend(rows)

            out = RESULTS_ROOT / f"qdrant_rep{rep}.csv"
            write_rows(out, all_rows)
            rep_summary = {
                "engine": "qdrant",
                "dataset_size": SIZE,
                "repetition": rep,
                "workloads": {
                    w: summarize([r for r in all_rows if r["workload"] == w])
                    for w in WORKLOADS
                },
                "warning": "These are held-out paper measurements, not tuning values."
            }
            write_json(RESULTS_ROOT / f"qdrant_rep{rep}_summary.json", rep_summary)
            print(f"Wrote {out}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True,
                    choices=["pg-hnsw", "pg-diskann", "qdrant"])
    ap.add_argument("--repetitions", type=int, default=3)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    frozen = json.loads(FROZEN_PATH.read_text())
    if frozen["dataset_size"] != SIZE:
        raise RuntimeError("Wrong frozen config dataset size.")
    if args.repetitions != 3:
        raise SystemExit("Frozen methodology requires exactly 3 repetitions.")

    ensure_clean(args.engine, args.overwrite)
    cfg = load_config()
    validate_state(cfg, args.engine)
    qemb = np.load(PROCESSED / "query_embeddings.npy", mmap_mode="r")
    truths = load_truths()
    write_manifest(args.engine, frozen)

    print("Frozen config:", FROZEN_PATH)
    print("Config SHA256:", sha256_file(FROZEN_PATH))
    print("Measurement ranks: 100-999 only")
    print("Warmups: ranks 0-99 only")
    print("No tuning is performed.")

    if args.engine.startswith("pg-"):
        run_postgres(args.engine, cfg, frozen, qemb, truths)
    else:
        run_qdrant(cfg, frozen, qemb, truths)

    print(f"\nCompleted 250k held-out measurements for {args.engine}.")
    print("Results:", RESULTS_ROOT)

if __name__ == "__main__":
    main()
