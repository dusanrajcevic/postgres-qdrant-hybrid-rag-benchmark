from __future__ import annotations
import argparse, json, time
import numpy as np
import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from common import GROUND_TRUTH, PROCESSED, ROOT, load_config, write_json

TUNING_DIR = ROOT / "tuning"
WORKLOADS = ["w2_50","w2_10","w2_1","w2_0_1","w3"]
TARGET = 0.95
EF_SEARCH = 60
SCAN_CAPS = [5000,10000,20000,40000,80000,100000]
QIDX = np.arange(0,100,dtype=int)

def conn(cfg):
    p=cfg["postgres"]
    c=psycopg.connect(host=p["host"],port=p["port"],dbname=p["dbname"],
                      user=p["user"],password=p["password"])
    register_vector(c)
    return c

def recall(ids,truth):
    return len(set(map(int,ids)) & set(map(int,truth))) / 10.0

def filter_sql(w,qi):
    if w=="w2_50":
        return "language = %s", ["en" if qi%2==0 else "de"]
    if w=="w2_10":
        return "category_id = %s", [qi%10]
    if w=="w2_1":
        return "tenant_id = %s", [1+(qi%80)]
    if w=="w2_0_1":
        return "owner_id = %s", [qi%1000]
    if w=="w3":
        start=qi%10
        cats=[(start+i)%10 for i in range(5)]
        return ("tenant_id = 0 AND status='active' AND category_id = ANY(%s) "
                "AND created_at >= TIMESTAMPTZ '2022-01-01T00:00:00Z'", [cats])
    raise ValueError(w)

def load_truth(size):
    return {w:np.load(GROUND_TRUTH/str(size)/f"{w}.npz")["top10_ids"] for w in WORKLOADS}

def ann_indexes(c):
    return [r[0] for r in c.execute("""
        SELECT indexname FROM pg_indexes
        WHERE tablename='chunks'
          AND indexname IN ('chunks_embedding_hnsw','chunks_embedding_diskann')
        ORDER BY indexname
    """).fetchall()]

def explain_direct(c,qvec,w,qi):
    where,vals=filter_sql(w,qi)
    rows=c.execute(f"""EXPLAIN (COSTS OFF)
        SELECT chunk_id FROM chunks WHERE {where}
        ORDER BY embedding <=> %s LIMIT 10""",
        [*vals,Vector(qvec)]).fetchall()
    return "\n".join(r[0] for r in rows)

def explain_exact(c,qvec,w,qi):
    where,vals=filter_sql(w,qi)
    rows=c.execute(f"""EXPLAIN (COSTS OFF)
        WITH eligible AS MATERIALIZED (
          SELECT chunk_id, embedding FROM chunks WHERE {where}
        )
        SELECT chunk_id FROM eligible
        ORDER BY embedding <=> %s LIMIT 10""",
        [*vals,Vector(qvec)]).fetchall()
    return "\n".join(r[0] for r in rows)

def run_direct(c,qemb,truths,cap):
    c.execute("SELECT set_config('hnsw.ef_search',%s,false)",(str(EF_SEARCH),))
    c.execute("SELECT set_config('hnsw.iterative_scan','strict_order',false)")
    c.execute("SELECT set_config('hnsw.max_scan_tuples',%s,false)",(str(cap),))
    c.execute("SELECT set_config('hnsw.scan_mem_multiplier','2',false)")
    summary={}; plans={}
    for w in WORKLOADS:
        plans[w]=explain_direct(c,qemb[0],w,0)
        lats=[]; rec=[]; returned=[]
        for qi in QIDX[:10]:
            where,vals=filter_sql(w,int(qi))
            c.execute(f"""SELECT chunk_id FROM chunks WHERE {where}
                          ORDER BY embedding <=> %s LIMIT 10""",
                      [*vals,Vector(qemb[qi])]).fetchall()
        for qi in QIDX:
            where,vals=filter_sql(w,int(qi))
            t=time.perf_counter_ns()
            rows=c.execute(f"""SELECT chunk_id FROM chunks WHERE {where}
                               ORDER BY embedding <=> %s LIMIT 10""",
                           [*vals,Vector(qemb[qi])]).fetchall()
            lats.append((time.perf_counter_ns()-t)/1e6)
            ids=[r[0] for r in rows]
            rec.append(recall(ids,truths[w][qi]))
            returned.append(len(ids))
        summary[w]={
            "mean_recall10":float(np.mean(rec)),
            "median_latency_ms_tuning_only":float(np.median(lats)),
            "p95_latency_ms_tuning_only":float(np.percentile(lats,95)),
            "min_returned":int(np.min(returned)),
            "plan_uses_hnsw":"chunks_embedding_hnsw" in plans[w],
        }
    return summary,plans

def run_exact(c,qemb,truths):
    summary={}; plans={}
    for w in WORKLOADS:
        plans[w]=explain_exact(c,qemb[0],w,0)
        if "chunks_embedding_hnsw" in plans[w] or "chunks_embedding_diskann" in plans[w]:
            raise RuntimeError(f"{w}: exact fallback unexpectedly uses ANN:\n{plans[w]}")
        lats=[]; rec=[]; returned=[]
        for qi in QIDX[:10]:
            where,vals=filter_sql(w,int(qi))
            c.execute(f"""WITH eligible AS MATERIALIZED (
                            SELECT chunk_id, embedding FROM chunks WHERE {where}
                          )
                          SELECT chunk_id FROM eligible
                          ORDER BY embedding <=> %s LIMIT 10""",
                      [*vals,Vector(qemb[qi])]).fetchall()
        for qi in QIDX:
            where,vals=filter_sql(w,int(qi))
            t=time.perf_counter_ns()
            rows=c.execute(f"""WITH eligible AS MATERIALIZED (
                                SELECT chunk_id, embedding FROM chunks WHERE {where}
                              )
                              SELECT chunk_id FROM eligible
                              ORDER BY embedding <=> %s LIMIT 10""",
                           [*vals,Vector(qemb[qi])]).fetchall()
            lats.append((time.perf_counter_ns()-t)/1e6)
            ids=[r[0] for r in rows]
            rec.append(recall(ids,truths[w][qi]))
            returned.append(len(ids))
        summary[w]={
            "mean_recall10":float(np.mean(rec)),
            "median_latency_ms_tuning_only":float(np.median(lats)),
            "p95_latency_ms_tuning_only":float(np.percentile(lats,95)),
            "min_returned":int(np.min(returned)),
        }
        if summary[w]["mean_recall10"] < 0.999999 or summary[w]["min_returned"] != 10:
            raise RuntimeError(f"{w}: exact fallback failed exactness/completeness: {summary[w]}")
    return summary,plans

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--size",type=int,required=True)
    args=ap.parse_args()
    if args.size!=100000:
        raise SystemExit("Validate/freeze at 100k before scaling.")
    cfg=load_config()
    qemb=np.load(PROCESSED/"query_embeddings.npy",mmap_mode="r")
    truths=load_truth(args.size)
    TUNING_DIR.mkdir(parents=True,exist_ok=True)

    with conn(cfg) as c:
        names=ann_indexes(c)
        if names != ["chunks_embedding_hnsw"]:
            raise RuntimeError(f"Expected only chunks_embedding_hnsw; found {names}")

        grids=[]; all_plans={}
        for cap in SCAN_CAPS:
            print(f"\nPG-HNSW direct/iterative: max_scan_tuples={cap}")
            s,p=run_direct(c,qemb,truths,cap)
            grids.append({"max_scan_tuples":cap,"by_workload":s})
            all_plans[str(cap)]=p
            print(json.dumps(grids[-1],indent=2))

        print("\nPG-HNSW scalar-first exact fallbacks")
        exact,exact_plans=run_exact(c,qemb,truths)
        print(json.dumps(exact,indent=2))

    strategy={}
    for w in WORKLOADS:
        candidates=[]
        for g in grids:
            s=g["by_workload"][w]
            if s["mean_recall10"]>=TARGET and s["min_returned"]==10:
                candidates.append({
                    "access_path":"planner-selected filtered query with HNSW available",
                    "ef_search":EF_SEARCH,
                    "iterative_scan":"strict_order",
                    "scan_mem_multiplier":2,
                    "max_scan_tuples":g["max_scan_tuples"],
                    **s
                })
        candidates.append({
            "access_path":"scalar-index-first exact vector ranking",
            **exact[w]
        })
        strategy[w]=min(candidates,key=lambda x:x["median_latency_ms_tuning_only"])

    report={
        "purpose":"PG-HNSW FILTERED STRATEGY SELECTION ONLY",
        "dataset_size":args.size,
        "target_recall10":TARGET,
        "tuning_query_ranks":[0,99],
        "measurement_query_ranks_reserved":[100,999],
        "frozen_w1_ef_search":EF_SEARCH,
        "strategy":strategy,
        "warning":"All latency values are tuning-only, not paper results."
    }
    write_json(TUNING_DIR/"pg_hnsw_crossover_100000_selected.json",report)
    write_json(TUNING_DIR/"pg_hnsw_crossover_100000_direct_plans.json",all_plans)
    write_json(TUNING_DIR/"pg_hnsw_crossover_100000_exact_plans.json",exact_plans)
    print("\nSELECTED PG-HNSW CROSSOVER STRATEGY")
    print(json.dumps(report,indent=2))

if __name__=="__main__":
    main()
