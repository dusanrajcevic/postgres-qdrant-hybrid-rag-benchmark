from __future__ import annotations
import argparse, json
import numpy as np
from common import GROUND_TRUTH, PROCESSED, load_config, write_json

WORKLOADS = ["w1","w2_50","w2_10","w2_1","w2_0_1","w3","w4_acl"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, required=True)
    args = ap.parse_args()
    cfg = load_config()
    if args.size not in set(map(int, cfg["dataset_sizes"])):
        raise SystemExit(f"--size must be one of {cfg['dataset_sizes']}")
    expected_queries = len(np.load(PROCESSED / "query_embeddings.npy", mmap_mode="r"))
    out_dir = GROUND_TRUTH / str(args.size)
    reports, all_ok = [], True
    for w in WORKLOADS:
        p = out_dir / f"{w}.npz"
        j = out_dir / f"{w}.json"
        item = {"workload": w, "npz": str(p), "json": str(j)}
        if not p.exists() or not j.exists():
            item.update(ok=False, error="missing file(s)")
            all_ok = False
            reports.append(item)
            continue
        d = np.load(p)
        required = {"qid","query_rank","top10_ids","top10_scores","eligible_count"}
        missing = sorted(required - set(d.files))
        if missing:
            item.update(ok=False, error=f"missing arrays: {missing}")
            all_ok = False
            reports.append(item)
            continue
        valid = (
            d["qid"].shape == (expected_queries,)
            and d["top10_ids"].shape == (expected_queries,10)
            and d["top10_scores"].shape == (expected_queries,10)
            and d["eligible_count"].shape == (expected_queries,)
            and np.all(d["top10_ids"] >= 0)
            and np.all(d["eligible_count"] >= 10)
        )
        item.update(
            queries=int(d["qid"].shape[0]),
            min_eligible=int(d["eligible_count"].min()),
            max_eligible=int(d["eligible_count"].max()),
            ok=bool(valid)
        )
        all_ok &= bool(valid)
        reports.append(item)
    report = {"dataset_size": args.size, "all_passed": all_ok, "workloads": reports}
    write_json(out_dir / "verification.json", report)
    print(json.dumps(report, indent=2))
    if not all_ok:
        raise SystemExit("Ground-truth verification FAILED.")
    print("Ground-truth verification PASSED for all workloads.")

if __name__ == "__main__":
    main()
