from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import numpy as np

from common import ROOT

SIZE = 250000
RESULTS = ROOT / "results" / "measurements" / str(SIZE)
ENGINES = ["pg-hnsw", "pg-diskann", "qdrant"]
WORKLOADS = ["w1", "w2_50", "w2_10", "w2_1", "w2_0_1", "w3", "w4_acl"]

def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def main():
    required = [
        RESULTS / f"{e}_rep{rep}.csv"
        for e in ENGINES for rep in (1, 2, 3)
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit(
            "All 9 measurement CSVs are required:\n" + "\n".join(missing)
        )

    final = []
    for engine in ENGINES:
        reps = {rep: read_csv(RESULTS / f"{engine}_rep{rep}.csv")
                for rep in (1, 2, 3)}
        pooled_all = sum(reps.values(), [])

        for w in WORKLOADS:
            pooled = [r for r in pooled_all if r["workload"] == w]
            lat = np.array([float(r["latency_ms"]) for r in pooled])
            rec = np.array([float(r["recall10"]) for r in pooled])
            returned = np.array([int(r["returned_count"]) for r in pooled])

            rep_medians, rep_p95, rep_recall = [], [], []
            for rep in (1, 2, 3):
                rr = [r for r in reps[rep] if r["workload"] == w]
                l = np.array([float(r["latency_ms"]) for r in rr])
                rc = np.array([float(r["recall10"]) for r in rr])
                rep_medians.append(float(np.median(l)))
                rep_p95.append(float(np.percentile(l, 95)))
                rep_recall.append(float(np.mean(rc)))

            final.append({
                "engine": engine,
                "dataset_size": SIZE,
                "workload": w,
                "observations": len(pooled),
                "unique_queries_per_repetition": 900,
                "repetitions": 3,
                "median_latency_ms_pooled": float(np.median(lat)),
                "p95_latency_ms_pooled": float(np.percentile(lat, 95)),
                "mean_latency_ms_pooled": float(np.mean(lat)),
                "mean_recall10_pooled": float(np.mean(rec)),
                "min_returned": int(np.min(returned)),
                "queries_with_fewer_than_10_pooled": int(np.sum(returned < 10)),
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

    out_csv = RESULTS / "final_summary_250000.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(final[0].keys()))
        writer.writeheader()
        writer.writerows(final)

    out_json = RESULTS / "final_summary_250000.json"
    out_json.write_text(json.dumps(final, indent=2) + "\n")

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_json}")
    print("These are held-out 250k paper-measurement summaries.")

if __name__ == "__main__":
    main()
