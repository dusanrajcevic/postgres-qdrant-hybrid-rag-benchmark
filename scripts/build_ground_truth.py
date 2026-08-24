from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from common import GROUND_TRUTH, PROCESSED, load_config, write_json

K = 10
CREATED_2022_DAY = 731  # days from 2020-01-01 to 2022-01-01

def signature_for_query(workload: str, qrank: int):
    if workload == "w1":
        return ("all",)
    if workload == "w2_50":
        return ("language", qrank % 2)
    if workload == "w2_10":
        return ("category", qrank % 10)
    if workload == "w2_1":
        return ("tenant", 1 + (qrank % 80))
    if workload == "w2_0_1":
        return ("owner", qrank % 1000)
    if workload == "w3":
        return ("compound", qrank % 10)
    if workload == "w4_acl":
        return ("acl", 1 + (qrank % 80), qrank % 5)
    raise ValueError(workload)

def eligible_indices(meta, size: int, signature):
    kind = signature[0]
    sl = slice(0, size)
    if kind == "all":
        return np.arange(size, dtype=np.int64)
    if kind == "language":
        mask = meta["language_code"][sl] == signature[1]
    elif kind == "category":
        mask = meta["category_id"][sl] == signature[1]
    elif kind == "tenant":
        mask = meta["tenant_id"][sl] == signature[1]
    elif kind == "owner":
        mask = meta["owner_id"][sl] == signature[1]
    elif kind == "compound":
        start = int(signature[1])
        cats = {(start + i) % 10 for i in range(5)}
        cat = meta["category_id"][sl]
        cat_mask = np.zeros(size, dtype=bool)
        for c in cats:
            cat_mask |= (cat == c)
        mask = (
            (meta["tenant_id"][sl] == 0)
            & (meta["status_code"][sl] == 0)
            & cat_mask
            & (meta["created_day"][sl] >= CREATED_2022_DAY)
        )
    elif kind == "acl":
        mask = (
            (meta["tenant_id"][sl] == signature[1])
            & (meta["status_code"][sl] == 0)
            & (meta["principal_id"][sl] == signature[2])
        )
    else:
        raise ValueError(signature)
    return np.flatnonzero(mask).astype(np.int64)

def update_topk(best_scores, best_ids, scores, ids, k=K):
    local_k = min(k, scores.shape[1])
    if local_k == 0:
        return best_scores, best_ids
    part = np.argpartition(scores, -local_k, axis=1)[:, -local_k:]
    local_scores = np.take_along_axis(scores, part, axis=1)
    local_ids = ids[part]

    merged_scores = np.concatenate([best_scores, local_scores], axis=1)
    merged_ids = np.concatenate([best_ids, local_ids], axis=1)
    take = np.argpartition(merged_scores, -k, axis=1)[:, -k:]
    best_scores = np.take_along_axis(merged_scores, take, axis=1)
    best_ids = np.take_along_axis(merged_ids, take, axis=1)
    return best_scores, best_ids

def compute_group(emb, qemb, pid_by_rank, eligible, q_indices, corpus_block, query_block):
    nq = len(q_indices)
    if len(eligible) < K:
        raise RuntimeError(f"Only {len(eligible)} eligible vectors for query group; need at least {K}.")
    out_scores = np.full((nq, K), -np.inf, dtype=np.float32)
    out_ids = np.full((nq, K), -1, dtype=np.int64)

    for q0 in range(0, nq, query_block):
        qsel = q_indices[q0:q0 + query_block]
        qmat = np.asarray(qemb[qsel], dtype=np.float32)
        best_scores = np.full((len(qsel), K), -np.inf, dtype=np.float32)
        best_ids = np.full((len(qsel), K), -1, dtype=np.int64)

        for c0 in range(0, len(eligible), corpus_block):
            idx = eligible[c0:c0 + corpus_block]
            cmat = np.asarray(emb[idx], dtype=np.float32)
            scores = qmat @ cmat.T
            ids = np.asarray(pid_by_rank[idx], dtype=np.int64)
            best_scores, best_ids = update_topk(best_scores, best_ids, scores, ids)

        order = np.argsort(best_scores, axis=1)[:, ::-1]
        best_scores = np.take_along_axis(best_scores, order, axis=1)
        best_ids = np.take_along_axis(best_ids, order, axis=1)
        out_scores[q0:q0 + len(qsel)] = best_scores
        out_ids[q0:q0 + len(qsel)] = best_ids

    return out_scores, out_ids

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument(
        "--workload",
        required=True,
        choices=["w1", "w2_50", "w2_10", "w2_1", "w2_0_1", "w3", "w4_acl"],
    )
    ap.add_argument("--corpus-block", type=int, default=20000)
    ap.add_argument("--query-block", type=int, default=32)
    args = ap.parse_args()

    cfg = load_config()
    if args.size not in set(map(int, cfg["dataset_sizes"])):
        raise SystemExit(f"--size must be one of {cfg['dataset_sizes']}")

    emb = np.load(PROCESSED / "passage_embeddings.npy", mmap_mode="r")
    qemb = np.load(PROCESSED / "query_embeddings.npy", mmap_mode="r")
    pid_by_rank = np.load(PROCESSED / "sampled_pids_by_rank.npy", mmap_mode="r")
    meta = np.load(PROCESSED / "aligned_metadata.npz")
    query_table = pq.read_table(PROCESSED / "queries.parquet", columns=["query_rank", "qid"])
    qids = np.asarray(query_table.column("qid").to_numpy(zero_copy_only=False), dtype=np.int64)
    qranks = np.asarray(query_table.column("query_rank").to_numpy(zero_copy_only=False), dtype=np.int64)
    order = np.argsort(qranks)
    qids = qids[order]
    n_queries = len(qemb)

    groups = defaultdict(list)
    for qrank in range(n_queries):
        groups[signature_for_query(args.workload, qrank)].append(qrank)

    final_scores = np.full((n_queries, K), -np.inf, dtype=np.float32)
    final_ids = np.full((n_queries, K), -1, dtype=np.int64)
    eligible_count = np.zeros(n_queries, dtype=np.int32)

    print(f"Ground truth: size={args.size:,}, workload={args.workload}, groups={len(groups)}")
    for group_no, (sig, q_indices_list) in enumerate(groups.items(), start=1):
        q_indices = np.asarray(q_indices_list, dtype=np.int64)
        eligible = eligible_indices(meta, args.size, sig)
        eligible_count[q_indices] = len(eligible)
        print(
            f"[{group_no}/{len(groups)}] signature={sig}; "
            f"queries={len(q_indices)}; eligible={len(eligible):,} "
            f"({len(eligible)/args.size:.4%})"
        )
        scores, ids = compute_group(
            emb, qemb, pid_by_rank, eligible, q_indices,
            args.corpus_block, args.query_block
        )
        final_scores[q_indices] = scores
        final_ids[q_indices] = ids

    out_dir = GROUND_TRUTH / f"{args.size}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.workload}.npz"
    np.savez_compressed(
        out_path,
        qid=qids,
        query_rank=np.arange(n_queries, dtype=np.int32),
        top10_ids=final_ids,
        top10_scores=final_scores,
        eligible_count=eligible_count,
    )
    report = {
        "dataset_size": args.size,
        "workload": args.workload,
        "queries": n_queries,
        "k": K,
        "mean_eligible_fraction": float(np.mean(eligible_count / args.size)),
        "min_eligible": int(eligible_count.min()),
        "max_eligible": int(eligible_count.max()),
        "ground_truth_file": str(out_path),
        "metric": "cosine similarity implemented as dot product over L2-normalized float32 embeddings",
    }
    write_json(out_dir / f"{args.workload}.json", report)
    print(report)
    print("Exact ground truth complete.")

if __name__ == "__main__":
    main()
