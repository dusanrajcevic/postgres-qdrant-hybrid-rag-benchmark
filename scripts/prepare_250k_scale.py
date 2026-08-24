from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from common import GROUND_TRUTH, ROOT, load_config, write_json

WORKLOADS = ["w1", "w2_50", "w2_10", "w2_1", "w2_0_1", "w3", "w4_acl"]

def run(cmd):
    print("\n$", " ".join(map(str, cmd)), flush=True)
    p = subprocess.run(cmd)
    if p.returncode != 0:
        raise SystemExit(p.returncode)

def verify_ground_truth(size: int):
    checks = {}
    gt_dir = GROUND_TRUTH / str(size)
    for workload in WORKLOADS:
        p = gt_dir / f"{workload}.npz"
        if not p.exists():
            raise RuntimeError(f"Missing ground truth: {p}")
        z = np.load(p)
        top = z["top10_ids"]
        if top.shape != (1000, 10):
            raise RuntimeError(
                f"{p}: expected top10_ids shape (1000,10), got {top.shape}"
            )
        checks[workload] = {
            "path": str(p),
            "top10_shape": list(top.shape),
            "eligible_min": int(np.min(z["eligible_count"]))
                if "eligible_count" in z.files else None,
            "eligible_max": int(np.max(z["eligible_count"]))
                if "eligible_count" in z.files else None,
        }
    return checks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=250000)
    ap.add_argument(
        "--skip-ground-truth",
        action="store_true",
        help="Use only if all 7 ground-truth files already exist and are known valid."
    )
    args = ap.parse_args()

    if args.size != 250000:
        raise SystemExit("This helper is intentionally frozen for the 250k scale point.")

    cfg = load_config()
    if args.size not in set(map(int, cfg["dataset_sizes"])):
        raise SystemExit(f"{args.size} is not present in config/experiment.json")

    print("""
======================================================================
250K SCALE-POINT PREPARATION
This is destructive to the CURRENT database contents, but it does not
modify results/measurements/100000.
No embeddings or metadata will be regenerated.
======================================================================
""".strip())

    # Destructive reload of database state for the new scale point.
    run([sys.executable, "scripts/load_postgres.py", "--size", str(args.size)])
    run([sys.executable, "scripts/load_qdrant.py", "--size", str(args.size)])
    run([sys.executable, "scripts/validate_loaded_data.py", "--size", str(args.size)])

    # Put the same scalar metadata beside PostgreSQL vectors and prepare
    # pgvectorscale labels. This also ensures no ANN index remains.
    run([
        sys.executable,
        "scripts/prepare_filtered_schema.py",
        "--size",
        str(args.size),
    ])

    if not args.skip_ground_truth:
        for workload in WORKLOADS:
            run([
                sys.executable,
                "scripts/build_ground_truth.py",
                "--size",
                str(args.size),
                "--workload",
                workload,
            ])

    checks = verify_ground_truth(args.size)

    state = {
        "purpose": "250K SCALE-POINT PREPARATION CHECKPOINT",
        "dataset_size": args.size,
        "postgres_loaded": True,
        "qdrant_loaded": True,
        "cross_store_validated": True,
        "postgres_filter_metadata_prepared": True,
        "ann_indexes_expected_after_prepare": [],
        "qdrant_global_hnsw_expected_after_load": "disabled (m=0)",
        "ground_truth": checks,
        "next_phase": "Independent 250k tuning using query ranks 0-99 only",
        "measurement_ranks_reserved": [100, 999],
    }
    out = ROOT / "results" / "scale_state_250000.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, state)

    print("\n250K PREPARATION PASSED")
    print(json.dumps(state, indent=2))
    print(f"\nWrote {out}")

if __name__ == "__main__":
    main()
