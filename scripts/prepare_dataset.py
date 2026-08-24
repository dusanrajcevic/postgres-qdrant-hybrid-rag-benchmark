from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from tqdm import tqdm

from common import RAW, PROCESSED, ensure_dirs, load_config, sha256_file, write_json

COLLECTION_URL = "https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz"
QUERIES_URL = "https://msmarco.z22.web.core.windows.net/msmarcoranking/queries.tar.gz"

def download_with_resume(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"Already downloaded: {dest}")
        return

    part = dest.with_suffix(dest.suffix + ".part")
    existing = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}

    with requests.get(url, stream=True, headers=headers, timeout=(15, 120)) as r:
        if existing and r.status_code == 200:
            # Server ignored Range; restart cleanly.
            existing = 0
            mode = "wb"
        elif r.status_code in (200, 206):
            mode = "ab" if existing else "wb"
        else:
            r.raise_for_status()
            return

        total = r.headers.get("Content-Length")
        total = int(total) + existing if total else None
        with part.open(mode) as f, tqdm(
            total=total,
            initial=existing,
            unit="B",
            unit_scale=True,
            desc=dest.name,
        ) as bar:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
    part.replace(dest)

def extract_member(archive: Path, output: Path, preferred_basenames: list[str]) -> str:
    if output.exists():
        return output.name

    with tarfile.open(archive, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.isfile()]
        by_base = {Path(m.name).name: m for m in members}
        selected = None
        for name in preferred_basenames:
            if name in by_base:
                selected = by_base[name]
                break
        if selected is None:
            available = ", ".join(sorted(by_base)[:50])
            raise RuntimeError(
                f"None of {preferred_basenames} found in {archive}. "
                f"First archive members: {available}"
            )
        src = tar.extractfile(selected)
        if src is None:
            raise RuntimeError(f"Could not read {selected.name} from {archive}")
        with output.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
        return selected.name

def select_passages(collection_tsv: Path, max_vectors: int, corpus_records: int, seed: int) -> None:
    out = PROCESSED / "passages.parquet"
    pid_order_path = PROCESSED / "sampled_pids_by_rank.npy"
    if out.exists() and pid_order_path.exists():
        print(f"Processed passage sample already exists: {out}")
        return

    rng = np.random.default_rng(seed)
    # Random sample without replacement from the official contiguous PID range.
    selected_pids = rng.choice(corpus_records, size=max_vectors, replace=False)
    np.save(pid_order_path, selected_pids.astype(np.int64))

    # 8,841,823 int32 values ≈ 35 MB and is much smaller than a Python dict.
    rank_by_pid = np.full(corpus_records, -1, dtype=np.int32)
    rank_by_pid[selected_pids] = np.arange(max_vectors, dtype=np.int32)

    schema = pa.schema([
        ("sample_rank", pa.int32()),
        ("chunk_id", pa.int64()),
        ("text", pa.string()),
    ])
    writer = pq.ParquetWriter(out, schema=schema, compression="zstd")
    ranks, pids, texts = [], [], []
    found = 0

    try:
        with collection_tsv.open("r", encoding="utf-8", errors="replace") as f:
            for line_no, line in enumerate(tqdm(f, desc="Scanning MS MARCO collection", unit="lines"), start=1):
                line = line.rstrip("\n")
                if not line:
                    continue
                try:
                    pid_s, text = line.split("\t", 1)
                    pid = int(pid_s)
                except ValueError as e:
                    raise RuntimeError(f"Malformed collection line {line_no}") from e
                if pid < 0 or pid >= corpus_records:
                    raise RuntimeError(f"Unexpected PID {pid} on line {line_no}")

                rank = int(rank_by_pid[pid])
                if rank >= 0:
                    ranks.append(rank)
                    pids.append(pid)
                    texts.append(text)
                    found += 1

                if len(ranks) >= 10000:
                    writer.write_table(pa.table(
                        {"sample_rank": ranks, "chunk_id": pids, "text": texts},
                        schema=schema,
                    ))
                    ranks.clear(); pids.clear(); texts.clear()

        if ranks:
            writer.write_table(pa.table(
                {"sample_rank": ranks, "chunk_id": pids, "text": texts},
                schema=schema,
            ))
    finally:
        writer.close()

    if found != max_vectors:
        out.unlink(missing_ok=True)
        raise RuntimeError(
            f"Expected {max_vectors:,} selected passages, found {found:,}. "
            "The corpus PID range/order may differ from the expected MS MARCO v1 passage collection."
        )
    print(f"Wrote {found:,} sampled passages to {out}")

def select_queries(query_tsv: Path, query_count: int, seed: int) -> None:
    out = PROCESSED / "queries.parquet"
    if out.exists():
        print(f"Processed query sample already exists: {out}")
        return

    rows = []
    with query_tsv.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                qid_s, text = line.split("\t", 1)
                rows.append((int(qid_s), text))
            except ValueError as e:
                raise RuntimeError(f"Malformed query line {line_no}") from e

    if len(rows) < query_count:
        raise RuntimeError(f"Only {len(rows)} queries found; need {query_count}")

    rng = np.random.default_rng(seed + 1)
    chosen = rng.choice(len(rows), size=query_count, replace=False)
    qids = [rows[int(i)][0] for i in chosen]
    texts = [rows[int(i)][1] for i in chosen]
    table = pa.table({
        "query_rank": pa.array(np.arange(query_count, dtype=np.int32)),
        "qid": pa.array(qids, type=pa.int64()),
        "text": pa.array(texts, type=pa.string()),
    })
    pq.write_table(table, out, compression="zstd")
    print(f"Wrote {query_count:,} fixed queries to {out}")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--redownload", action="store_true", help="Remove archives and download them again.")
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs()

    collection_archive = RAW / "collection.tar.gz"
    queries_archive = RAW / "queries.tar.gz"
    collection_tsv = RAW / "collection.tsv"
    query_tsv = RAW / "queries.dev.small.tsv"

    if args.redownload:
        for p in (collection_archive, queries_archive):
            p.unlink(missing_ok=True)
            p.with_suffix(p.suffix + ".part").unlink(missing_ok=True)

    download_with_resume(COLLECTION_URL, collection_archive)
    download_with_resume(QUERIES_URL, queries_archive)

    collection_member = extract_member(collection_archive, collection_tsv, ["collection.tsv"])
    query_member = extract_member(
        queries_archive,
        query_tsv,
        ["queries.dev.small.tsv", "queries.dev.tsv", "queries.eval.small.tsv"],
    )

    select_passages(
        collection_tsv,
        max_vectors=int(cfg["max_vectors"]),
        corpus_records=int(cfg["msmarco_corpus_records"]),
        seed=int(cfg["seed"]),
    )
    select_queries(query_tsv, int(cfg["query_count"]), int(cfg["seed"]))

    manifest = {
        "dataset": "MS MARCO v1 passage ranking collection",
        "collection_url": COLLECTION_URL,
        "queries_url": QUERIES_URL,
        "collection_archive_sha256": sha256_file(collection_archive),
        "queries_archive_sha256": sha256_file(queries_archive),
        "collection_member": collection_member,
        "query_member": query_member,
        "official_corpus_records": int(cfg["msmarco_corpus_records"]),
        "seed": int(cfg["seed"]),
        "max_vectors": int(cfg["max_vectors"]),
        "nested_dataset_sizes": cfg["dataset_sizes"],
        "query_count": int(cfg["query_count"]),
        "passages_parquet_sha256": sha256_file(PROCESSED / "passages.parquet"),
        "queries_parquet_sha256": sha256_file(PROCESSED / "queries.parquet"),
        "sampled_pids_sha256": sha256_file(PROCESSED / "sampled_pids_by_rank.npy"),
    }
    write_json(PROCESSED / "dataset_manifest.json", manifest)
    print("Dataset preparation complete.")
    print(f"Manifest: {PROCESSED / 'dataset_manifest.json'}")

if __name__ == "__main__":
    main()
