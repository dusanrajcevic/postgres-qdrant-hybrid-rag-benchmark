from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MEASUREMENTS = RESULTS / "measurements"
OUT_CSV = RESULTS / "paper_summary.csv"
OUT_JSON = RESULTS / "paper_summary.json"

SIZES = (100000, 250000)
KEEP = (
    "engine",
    "dataset_size",
    "workload",
    "observations",
    "median_latency_ms_pooled",
    "p95_latency_ms_pooled",
    "mean_recall10_pooled",
    "min_returned",
)


def main() -> None:
    combined = []
    for size in SIZES:
        path = MEASUREMENTS / str(size) / f"final_summary_{size}.csv"
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                combined.append({k: row[k] for k in KEEP})

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=KEEP, lineterminator="\n")
        writer.writeheader()
        writer.writerows(combined)

    OUT_JSON.write_text(json.dumps(combined, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
