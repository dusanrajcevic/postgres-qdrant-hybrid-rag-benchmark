from __future__ import annotations
import argparse, csv, json, time
import numpy as np, psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from qdrant_client import QdrantClient, models
from common import GROUND_TRUTH, PROCESSED, ROOT, load_config, write_json

TUNING_DIR = ROOT / "tuning"
TCFG_PATH = ROOT / "config" / "tuning.json"
HNSW_EF = [40,60,80,100,150,200,300,400,600,800,1000]
DISKANN = [(50,50),(75,100),(100,100),(100,200),(150,200),
           (200,400),(300,600),(400,800),(600,1000)]

def recall(ids, truth):
    return len(set(map(int,ids)) & set(map(int,truth))) / 10.0

def pg_conn(cfg):
    p=cfg["postgres"]
    conn=psycopg.connect(host=p["host"],port=p["port"],dbname=p["dbname"],
                         user=p["user"],password=p["password"])
    register_vector(conn)
    return conn

def check_plan(conn, qvec, expected):
    rows=conn.execute("""EXPLAIN (COSTS OFF)
                         SELECT chunk_id FROM chunks
                         ORDER BY embedding <=> %s LIMIT 10""",(Vector(qvec),)).fetchall()
    txt="\n".join(str(x[0]) for x in rows)
    print("Query plan:\n"+txt)
    if expected not in txt:
        raise RuntimeError(f"Expected {expected} is not in the query plan.")

def pg_hnsw(cfg,qemb,truth,qidx,ef):
    lat=[]; rec=[]
    with pg_conn(cfg) as conn:
        conn.execute("SELECT set_config('hnsw.ef_search',%s,false)",(str(ef),))
        conn.execute("SELECT set_config('hnsw.iterative_scan','off',false)")
        check_plan(conn,qemb[qidx[0]],"chunks_embedding_hnsw")
        for qi in qidx[:10]:
            conn.execute("SELECT chunk_id FROM chunks ORDER BY embedding <=> %s LIMIT 10",
                         (Vector(qemb[qi]),)).fetchall()
        for qi in qidx:
            t=time.perf_counter_ns()
            rows=conn.execute("SELECT chunk_id FROM chunks ORDER BY embedding <=> %s LIMIT 10",
                              (Vector(qemb[qi]),)).fetchall()
            lat.append((time.perf_counter_ns()-t)/1e6)
            rec.append(recall([r[0] for r in rows],truth[qi]))
    return lat,rec

def pg_diskann(cfg,qemb,truth,qidx,s,r):
    lat=[]; rec=[]
    with pg_conn(cfg) as conn:
        conn.execute("SELECT set_config('diskann.query_search_list_size',%s,false)",(str(s),))
        conn.execute("SELECT set_config('diskann.query_rescore',%s,false)",(str(r),))
        check_plan(conn,qemb[qidx[0]],"chunks_embedding_diskann")
        for qi in qidx[:10]:
            conn.execute("SELECT chunk_id FROM chunks ORDER BY embedding <=> %s LIMIT 10",
                         (Vector(qemb[qi]),)).fetchall()
        for qi in qidx:
            t=time.perf_counter_ns()
            rows=conn.execute("SELECT chunk_id FROM chunks ORDER BY embedding <=> %s LIMIT 10",
                              (Vector(qemb[qi]),)).fetchall()
            lat.append((time.perf_counter_ns()-t)/1e6)
            rec.append(recall([x[0] for x in rows],truth[qi]))
    return lat,rec

def qdrant(cfg,qemb,truth,qidx,ef):
    q=cfg["qdrant"]; c=QdrantClient(url=q["url"],timeout=60)
    if int(c.get_collection(q["collection"]).config.hnsw_config.m)==0:
        raise RuntimeError("Qdrant HNSW is disabled.")
    params=models.SearchParams(hnsw_ef=ef,exact=False)
    lat=[]; rec=[]
    for qi in qidx[:10]:
        c.query_points(collection_name=q["collection"],query=qemb[qi].tolist(),limit=10,
                       search_params=params,with_payload=False,with_vectors=False)
    for qi in qidx:
        t=time.perf_counter_ns()
        resp=c.query_points(collection_name=q["collection"],query=qemb[qi].tolist(),limit=10,
                            search_params=params,with_payload=False,with_vectors=False)
        lat.append((time.perf_counter_ns()-t)/1e6)
        rec.append(recall([p.id for p in resp.points],truth[qi]))
    return lat,rec

def summarize(engine,params,lat,rec):
    return {"engine":engine,**params,"queries":len(lat),
            "mean_recall10":float(np.mean(rec)),
            "median_latency_ms_tuning_only":float(np.median(lat)),
            "p95_latency_ms_tuning_only":float(np.percentile(lat,95))}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--size",type=int,required=True)
    ap.add_argument("--engine",choices=["pg-hnsw","pg-diskann","qdrant"],required=True)
    args=ap.parse_args()
    cfg=load_config()
    tcfg=json.loads(TCFG_PATH.read_text())
    target=float(tcfg["target_recall10"])
    start=int(tcfg["tuning_query_start"]); count=int(tcfg["tuning_query_count"])
    qidx=np.arange(start,start+count,dtype=int)
    qemb=np.load(PROCESSED/"query_embeddings.npy",mmap_mode="r")
    gt=GROUND_TRUTH/str(args.size)/"w1.npz"
    if not gt.exists(): raise SystemExit(f"Missing {gt}")
    truth=np.load(gt)["top10_ids"]
    TUNING_DIR.mkdir(parents=True,exist_ok=True)
    rows=[]

    if args.engine=="pg-hnsw":
        for ef in HNSW_EF:
            print(f"\nPG-HNSW ef_search={ef}")
            lat,rec=pg_hnsw(cfg,qemb,truth,qidx,ef)
            r=summarize(args.engine,{"ef_search":ef},lat,rec); rows.append(r); print(r)
    elif args.engine=="pg-diskann":
        for s,rscore in DISKANN:
            print(f"\nPG-DiskANN search_list={s}, rescore={rscore}")
            lat,rec=pg_diskann(cfg,qemb,truth,qidx,s,rscore)
            r=summarize(args.engine,{"query_search_list_size":s,"query_rescore":rscore},lat,rec)
            rows.append(r); print(r)
    else:
        for ef in HNSW_EF:
            print(f"\nQdrant hnsw_ef={ef}")
            lat,rec=qdrant(cfg,qemb,truth,qidx,ef)
            r=summarize(args.engine,{"hnsw_ef":ef},lat,rec); rows.append(r); print(r)

    csvp=TUNING_DIR/f"w1_{args.engine}_{args.size}_grid.csv"
    fields=sorted({k for r in rows for k in r})
    with csvp.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

    eligible=[r for r in rows if r["mean_recall10"]>=target]
    if not eligible:
        print(f"No setting reached Recall@10 >= {target}. Grid saved to {csvp}")
        raise SystemExit(2)
    selected=min(eligible,key=lambda x:x["median_latency_ms_tuning_only"])
    report={"purpose":"QUERY-TIME PARAMETER SELECTION ONLY","dataset_size":args.size,
            "workload":"w1","target_recall10":target,
            "tuning_query_ranks":[int(qidx[0]),int(qidx[-1])],
            "measurement_query_ranks_reserved":[
                int(tcfg["measurement_query_start"]),
                int(tcfg["measurement_query_start"]+tcfg["measurement_query_count"]-1)],
            "selected":selected,
            "warning":"Tuning latency values are not paper results."}
    out=TUNING_DIR/f"w1_{args.engine}_{args.size}_selected.json"
    write_json(out,report)
    print("\nSELECTED\n"+json.dumps(report,indent=2))
    print(f"Grid: {csvp}\nSelection: {out}")

if __name__=="__main__":
    main()
