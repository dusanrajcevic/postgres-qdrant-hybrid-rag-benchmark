from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

from common import MODELS, PROCESSED, ensure_dirs, load_config, sha256_file, write_json

def open_or_create_npy(path: Path, shape: tuple[int, ...], dtype=np.float32):
    if path.exists():
        arr = np.load(path, mmap_mode="r+")
        if tuple(arr.shape) != tuple(shape) or arr.dtype != np.dtype(dtype):
            raise RuntimeError(f"Existing {path} has shape/dtype {arr.shape}/{arr.dtype}, expected {shape}/{dtype}")
        return arr
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)

def open_or_create_done(path: Path, n: int):
    if path.exists():
        arr = np.load(path, mmap_mode="r+")
        if arr.shape != (n,):
            raise RuntimeError(f"Existing progress file {path} has wrong shape")
        return arr
    arr = np.lib.format.open_memmap(path, mode="w+", dtype=np.bool_, shape=(n,))
    arr[:] = False
    arr.flush()
    return arr

def choose_device(requested: str) -> str:
    import torch
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def embed_passages(model, dims: int, batch_size: int) -> None:
    cfg = load_config()
    n = int(cfg["max_vectors"])
    emb_path = PROCESSED / "passage_embeddings.npy"
    done_path = PROCESSED / "passage_embeddings_done.npy"
    emb = open_or_create_npy(emb_path, (n, dims))
    done = open_or_create_done(done_path, n)

    pf = pq.ParquetFile(PROCESSED / "passages.parquet")
    processed = int(done.sum())
    print(f"Passage embeddings already complete: {processed:,}/{n:,}")

    for batch in pf.iter_batches(batch_size=max(batch_size * 8, 1024), columns=["sample_rank", "text"]):
        ranks = batch.column("sample_rank").to_numpy(zero_copy_only=False).astype(np.int64)
        missing_mask = ~done[ranks]
        if not np.any(missing_mask):
            continue
        missing_ranks = ranks[missing_mask]
        texts_all = batch.column("text").to_pylist()
        texts = [texts_all[i] for i, keep in enumerate(missing_mask) if keep]

        vectors = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32, copy=False)

        if vectors.shape != (len(missing_ranks), dims):
            raise RuntimeError(f"Unexpected embedding shape {vectors.shape}")
        emb[missing_ranks] = vectors
        done[missing_ranks] = True
        emb.flush(); done.flush()
        processed += len(missing_ranks)
        print(f"\rPassages embedded: {processed:,}/{n:,}", end="", flush=True)

    print()
    if not bool(done.all()):
        raise RuntimeError("Passage embedding pass ended with unfinished rows.")

def embed_queries(model, dims: int, batch_size: int) -> None:
    cfg = load_config()
    n = int(cfg["query_count"])
    table = pq.read_table(PROCESSED / "queries.parquet", columns=["query_rank", "text"])
    ranks = table.column("query_rank").to_numpy(zero_copy_only=False).astype(np.int64)
    texts = table.column("text").to_pylist()
    order = np.argsort(ranks)
    texts = [texts[int(i)] for i in order]

    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32, copy=False)
    if vectors.shape != (n, dims):
        raise RuntimeError(f"Unexpected query embedding shape {vectors.shape}")
    np.save(PROCESSED / "query_embeddings.npy", vectors)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs()
    dims = int(cfg["embedding_dimensions"])
    repo = str(cfg["embedding_model"])
    revision = str(cfg["embedding_model_revision"])
    device = choose_device(args.device)

    model_dir = MODELS / f"all-MiniLM-L6-v2-{revision[:12]}"
    snapshot_download(repo_id=repo, revision=revision, local_dir=model_dir)
    model = SentenceTransformer(str(model_dir), device=device)

    probe = model.encode(["dimension check"], normalize_embeddings=True, convert_to_numpy=True)
    if probe.shape[1] != dims:
        raise RuntimeError(f"Model emits {probe.shape[1]} dimensions, expected {dims}")

    print(f"Embedding model: {repo}@{revision}")
    print(f"Device: {device}; dimensions: {dims}; max sequence length: {model.max_seq_length}")

    embed_passages(model, dims, args.batch_size)
    embed_queries(model, dims, args.batch_size)

    passage_emb = np.load(PROCESSED / "passage_embeddings.npy", mmap_mode="r")
    query_emb = np.load(PROCESSED / "query_embeddings.npy", mmap_mode="r")
    # Sanity check normalized vectors without scanning every row.
    sample_idx = np.linspace(0, len(passage_emb) - 1, num=min(1000, len(passage_emb)), dtype=np.int64)
    norms = np.linalg.norm(passage_emb[sample_idx], axis=1)
    if not np.allclose(norms, 1.0, atol=2e-3):
        raise RuntimeError("Passage embeddings do not appear to be L2-normalized.")

    manifest = {
        "model_repo": repo,
        "model_revision": revision,
        "dimensions": dims,
        "normalize_embeddings": True,
        "model_max_seq_length": int(model.max_seq_length),
        "device_used_for_generation": device,
        "platform": platform.platform(),
        "passage_shape": list(passage_emb.shape),
        "query_shape": list(query_emb.shape),
        "passage_embeddings_sha256": sha256_file(PROCESSED / "passage_embeddings.npy"),
        "query_embeddings_sha256": sha256_file(PROCESSED / "query_embeddings.npy"),
    }
    write_json(PROCESSED / "embedding_manifest.json", manifest)
    print("Embedding generation complete.")
    print(f"Manifest: {PROCESSED / 'embedding_manifest.json'}")

if __name__ == "__main__":
    main()
