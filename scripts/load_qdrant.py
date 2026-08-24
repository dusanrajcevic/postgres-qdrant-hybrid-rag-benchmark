from __future__ import annotations

import argparse
from datetime import date, timedelta

import numpy as np
import pyarrow.parquet as pq
from qdrant_client import QdrantClient, models

from common import PROCESSED, load_config, write_json

EPOCH = date(2020, 1, 1)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--omit-text", action="store_true",
                    help="Do not copy passage text into Qdrant payload. Default keeps text for a self-contained point.")
    args = ap.parse_args()

    cfg = load_config()
    allowed = set(map(int, cfg["dataset_sizes"]))
    if args.size not in allowed:
        raise SystemExit(f"--size must be one of {sorted(allowed)}")

    qcfg = cfg["qdrant"]
    collection = qcfg["collection"]
    client = QdrantClient(url=qcfg["url"], timeout=120)

    if client.collection_exists(collection):
        client.delete_collection(collection)

    # HNSW deliberately disabled during ingestion. It will be enabled and timed later.
    client.create_collection(
        collection_name=collection,
        vectors_config=models.VectorParams(
            size=int(cfg["embedding_dimensions"]),
            distance=models.Distance.COSINE,
        ),
        hnsw_config=models.HnswConfigDiff(m=0),
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=0),
        shard_number=1,
    )

    # Create filter indexes before upload, as recommended by Qdrant for filtered HNSW.
    indexes = [
        ("tenant_id", models.PayloadSchemaType.INTEGER),
        ("owner_id", models.PayloadSchemaType.INTEGER),
        ("category_id", models.PayloadSchemaType.INTEGER),
        ("language", models.PayloadSchemaType.KEYWORD),
        ("status", models.PayloadSchemaType.KEYWORD),
        ("created_at", models.PayloadSchemaType.DATETIME),
    ]
    for field, schema in indexes:
        client.create_payload_index(
            collection_name=collection,
            field_name=field,
            field_schema=schema,
            wait=True,
        )

    emb = np.load(PROCESSED / "passage_embeddings.npy", mmap_mode="r")
    meta = np.load(PROCESSED / "aligned_metadata.npz")
    pf = pq.ParquetFile(PROCESSED / "passages.parquet")
    chunks_per_doc = int(cfg["chunks_per_document"])

    uploaded = 0
    points = []

    def flush():
        nonlocal uploaded, points
        if not points:
            return
        client.upsert(collection_name=collection, points=points, wait=True)
        uploaded += len(points)
        points = []
        print(f"\rQdrant points uploaded: {uploaded:,}/{args.size:,}", end="", flush=True)

    for batch in pf.iter_batches(batch_size=4096, columns=["sample_rank", "chunk_id", "text"]):
        ranks = batch.column("sample_rank").to_numpy(zero_copy_only=False).astype(np.int64)
        pids = batch.column("chunk_id").to_numpy(zero_copy_only=False).astype(np.int64)
        texts = batch.column("text").to_pylist()
        for rank, pid, text in zip(ranks, pids, texts):
            rank = int(rank)
            if rank >= args.size:
                continue
            day = int(meta["created_day"][rank])
            payload = {
                "document_id": rank // chunks_per_doc,
                "tenant_id": int(meta["tenant_id"][rank]),
                "owner_id": int(meta["owner_id"][rank]),
                "category_id": int(meta["category_id"][rank]),
                "language": "en" if int(meta["language_code"][rank]) == 0 else "de",
                "status": "active" if int(meta["status_code"][rank]) == 0 else "inactive",
                "created_at": (EPOCH + timedelta(days=day)).isoformat() + "T00:00:00Z",
            }
            # Permissions are intentionally NOT denormalized into Qdrant;
            # W4 validates those against PostgreSQL at request time.
            if not args.omit_text:
                payload["text"] = text
            points.append(models.PointStruct(
                id=int(pid),
                vector=np.asarray(emb[rank], dtype=np.float32).tolist(),
                payload=payload,
            ))
            if len(points) >= args.batch_size:
                flush()
    flush()
    print()

    count = client.count(collection_name=collection, exact=True).count
    if int(count) != args.size:
        raise RuntimeError(f"Qdrant count {count} != expected {args.size}")

    info = client.get_collection(collection)
    manifest = {
        "dataset_size": args.size,
        "collection": collection,
        "points_loaded": int(count),
        "payload_indexes": [f for f, _ in indexes],
        "permissions_denormalized": False,
        "text_in_payload": not args.omit_text,
        "hnsw_m_during_load": 0,
        "vector_indexing_threshold_during_load": 0,
        "collection_status": str(info.status),
    }
    write_json(PROCESSED / f"qdrant_load_{args.size}.json", manifest)
    print(manifest)
    print("Qdrant base-data load complete. Global HNSW was not built.")

if __name__ == "__main__":
    main()
