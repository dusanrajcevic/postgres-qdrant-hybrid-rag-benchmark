from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import numpy as np

from common import ROOT

RESULTS = ROOT / "results" / "measurements" / "100000"
ENGINES = ["pg-hnsw","pg-diskann","qdrant"]
WORKLOADS = ["w1","w2_50","w2_10","w2_1","w2_0_1","w3","w4_acl"]

def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def main():
    missing = []
    for e in ENGINES:
        for rep in (1,2,3):
            p = RESULTS / f"{e}_rep{rep}.csv"
            if not p.exists():
                missing.append(str(p))
    if missing:
        raise SystemExit(
            "All 9 measurement files are required before final summarization:\n" +
            "\n".join(missing)
        )

    final = []
    for engine in ENGINES:
        all_engine = []
        rep_rows = {}
        for rep in (1,2,3):
            rows = read_csv(RESULTS / f"{engine}_rep{rep}.csv")
            rep_rows[rep] = rows
            all_engine.extend(rows)

        for workload in WORKLOADS:
            pooled = [r for r in all_engine if r["workload"] == workload]
            lat = np.array([float(r["latency_ms"]) for r in pooled])
            rec = np.array([float(r["recall10"]) for r in pooled])
            returned = np.array([int(r["returned_count"]) for r in pooled])

            rep_medians = []
            rep_p95 = []
            rep_recall = []
            for rep in (1,2,3):
                r = [x for x in rep_rows[rep] if x["workload"] == workload]
                l = np.array([float(x["latency_ms"]) for x in r])
                rr = np.array([float(x["recall10"]) for x in r])
                rep_medians.append(float(np.median(l)))
                rep_p95.append(float(np.percentile(l,95)))
                rep_recall.append(float(np.mean(rr)))

            final.append({
                "engine": engine,
                "dataset_size": 100000,
                "workload": workload,
                "observations": len(pooled),
                "unique_queries_per_repetition": 900,
                "repetitions": 3,
                "median_latency_ms_pooled": float(np.median(lat)),
                "p95_latency_ms_pooled": float(np.percentile(lat,95)),
                "mean_latency_ms_pooled": float(np.mean(lat)),
                "mean_recall10_pooled": float(np.mean(rec)),
                "min_returned": int(np.min(returned)),
                "rep1_median_ms": rep_medians[0],
                "rep2_median_ms": rep_medians[1],
                "rep3_median_ms": rep_medians[2],
                "median_of_rep_medians_ms": float(np.median(rep_medians)),
                "stdev_rep_medians_ms": float(statistics.stdev(rep_medians)),
                "rep1_p95_ms": rep_p95[0],
                "rep2_p95_ms": rep_p95[1],
                "rep3_p95_ms": rep_p95[2],
                "rep1_mean_recall10": rep_recall[0],
                "rep2_mean_recall10": rep_recall[1],
                "rep3_mean_recall10": rep_recall[2],
            })

    out_csv = RESULTS / "final_summary_100000.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(final[0].keys()))
        w.writeheader()
        w.writerows(final)

    out_json = RESULTS / "final_summary_100000.json"
    out_json.write_text(json.dumps(final, indent=2) + "\n")

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_json}")
    print("\nThese are held-out measurement summaries and are eligible for paper analysis.")

if __name__ == "__main__":
    main()
