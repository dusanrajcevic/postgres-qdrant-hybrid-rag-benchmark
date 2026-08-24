from __future__ import annotations
import argparse, json, time
import numpy as np
from qdrant_client import QdrantClient, models
from common import GROUND_TRUTH, PROCESSED, ROOT, load_config, write_json

TUNING_DIR=ROOT/"tuning"
WORKLOADS=["w2_50","w2_10","w2_1","w2_0_1","w3"]
TARGET=0.95
EF_GRID=[40,60,80,100,150,200,300,500]
QIDX=np.arange(0,100,dtype=int)

def recall(ids,truth):
    return len(set(map(int,ids)) & set(map(int,truth))) / 10.0

def filt(w,qi):
    if w=="w2_50":
        return models.Filter(must=[models.FieldCondition(
            key="language",match=models.MatchValue(value="en" if qi%2==0 else "de"))])
    if w=="w2_10":
        return models.Filter(must=[models.FieldCondition(
            key="category_id",match=models.MatchValue(value=qi%10))])
    if w=="w2_1":
        return models.Filter(must=[models.FieldCondition(
            key="tenant_id",match=models.MatchValue(value=1+(qi%80)))])
    if w=="w2_0_1":
        return models.Filter(must=[models.FieldCondition(
            key="owner_id",match=models.MatchValue(value=qi%1000))])
    if w=="w3":
        start=qi%10
        cats=[(start+i)%10 for i in range(5)]
        return models.Filter(must=[
            models.FieldCondition(key="tenant_id",match=models.MatchValue(value=0)),
            models.FieldCondition(key="status",match=models.MatchValue(value="active")),
            models.FieldCondition(key="category_id",match=models.MatchAny(any=cats)),
            models.FieldCondition(key="created_at",
                                  range=models.DatetimeRange(gte="2022-01-01T00:00:00Z")),
        ])
    raise ValueError(w)

def load_truth(size):
    return {w:np.load(GROUND_TRUTH/str(size)/f"{w}.npz")["top10_ids"] for w in WORKLOADS}

def run_mode(client,name,qemb,truths,params,label):
    summary={}
    for w in WORKLOADS:
        lats=[]; rec=[]; returned=[]
        for qi in QIDX[:10]:
            client.query_points(
                collection_name=name,query=np.asarray(qemb[qi],dtype=np.float32).tolist(),
                query_filter=filt(w,int(qi)),search_params=params,limit=10,
                with_payload=False,with_vectors=False)
        for qi in QIDX:
            t=time.perf_counter_ns()
            resp=client.query_points(
                collection_name=name,query=np.asarray(qemb[qi],dtype=np.float32).tolist(),
                query_filter=filt(w,int(qi)),search_params=params,limit=10,
                with_payload=False,with_vectors=False)
            lats.append((time.perf_counter_ns()-t)/1e6)
            ids=[p.id for p in resp.points]
            rec.append(recall(ids,truths[w][qi]))
            returned.append(len(ids))
        summary[w]={
            "mode":label,
            "mean_recall10":float(np.mean(rec)),
            "median_latency_ms_tuning_only":float(np.median(lats)),
            "p95_latency_ms_tuning_only":float(np.percentile(lats,95)),
            "min_returned":int(np.min(returned)),
        }
    return summary

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--size",type=int,required=True)
    args=ap.parse_args()
    if args.size!=100000:
        raise SystemExit("Validate/freeze at 100k before scaling.")
    cfg=load_config()
    q=cfg["qdrant"]
    client=QdrantClient(url=q["url"],timeout=120)
    n=int(client.count(collection_name=q["collection"],exact=True).count)
    if n!=args.size:
        raise RuntimeError(f"Qdrant has {n:,} points, expected {args.size:,}")
    info=client.get_collection(q["collection"])
    if int(info.config.hnsw_config.m)==0:
        raise RuntimeError("Qdrant HNSW is disabled.")

    qemb=np.load(PROCESSED/"query_embeddings.npy",mmap_mode="r")
    truths=load_truth(args.size)
    TUNING_DIR.mkdir(parents=True,exist_ok=True)

    modes=[]
    for ef in EF_GRID:
        print(f"\nQdrant filtered approximate: hnsw_ef={ef}")
        s=run_mode(client,q["collection"],qemb,truths,
                   models.SearchParams(hnsw_ef=ef,exact=False),
                   f"filter-aware approximate hnsw_ef={ef}")
        modes.append({"kind":"approx","hnsw_ef":ef,"by_workload":s})
        print(json.dumps(modes[-1],indent=2))

    print("\nQdrant filtered exact mode")
    exact=run_mode(client,q["collection"],qemb,truths,
                   models.SearchParams(exact=True),
                   "filtered exact")
    print(json.dumps(exact,indent=2))

    strategy={}
    for w in WORKLOADS:
        candidates=[]
        for m in modes:
            s=m["by_workload"][w]
            if s["mean_recall10"]>=TARGET and s["min_returned"]==10:
                candidates.append({
                    "access_path":"Qdrant filter-aware approximate search",
                    "hnsw_ef":m["hnsw_ef"],
                    **s
                })
        ex=exact[w]
        if ex["mean_recall10"] < 0.999999 or ex["min_returned"] != 10:
            raise RuntimeError(f"{w}: Qdrant exact mode failed: {ex}")
        candidates.append({
            "access_path":"Qdrant filtered exact search",
            **ex
        })
        strategy[w]=min(candidates,key=lambda x:x["median_latency_ms_tuning_only"])

    report={
        "purpose":"QDRANT FILTERED STRATEGY SELECTION ONLY",
        "dataset_size":args.size,
        "target_recall10":TARGET,
        "tuning_query_ranks":[0,99],
        "measurement_query_ranks_reserved":[100,999],
        "strategy":strategy,
        "warning":"All latency values are tuning-only, not paper results."
    }
    write_json(TUNING_DIR/"qdrant_crossover_100000_selected.json",report)
    print("\nSELECTED QDRANT CROSSOVER STRATEGY")
    print(json.dumps(report,indent=2))

if __name__=="__main__":
    main()
