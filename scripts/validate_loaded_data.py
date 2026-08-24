from __future__ import annotations

import argparse
import json

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from qdrant_client import QdrantClient

from common import PROCESSED, load_config, write_json

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, required=True)
    args = ap.parse_args()

    cfg = load_config()
    p = cfg["postgres"]
    q = cfg["qdrant"]
    pids = np.load(PROCESSED / "sampled_pids_by_rank.npy")
    emb = np.load(PROCESSED / "passage_embeddings.npy", mmap_mode="r")
    meta = np.load(PROCESSED / "aligned_metadata.npz")

    sample_ranks = np.array(sorted(set([0, args.size // 3, (2 * args.size) // 3, args.size - 1])), dtype=int)
    sample_pids = [int(pids[r]) for r in sample_ranks]

    with psycopg.connect(
        host=p["host"], port=p["port"], dbname=p["dbname"], user=p["user"], password=p["password"]
    ) as conn:
        register_vector(conn)
        pg_counts = conn.execute("""
            SELECT
              (SELECT count(*) FROM documents),
              (SELECT count(*) FROM chunks),
              (SELECT count(*) FROM document_permissions)
        """).fetchone()
        pg_rows = {}
        for rank, pid in zip(sample_ranks, sample_pids):
            row = conn.execute("""
                SELECT c.document_id, d.tenant_id, d.owner_id, d.category_id,
                       d.language, d.status, c.embedding
                FROM chunks c JOIN documents d USING (document_id)
                WHERE c.chunk_id = %s
            """, (pid,)).fetchone()
            if row is None:
                raise RuntimeError(f"PostgreSQL missing sample PID {pid}")
            pg_rows[pid] = row

    client = QdrantClient(url=q["url"], timeout=60)
    q_count = int(client.count(collection_name=q["collection"], exact=True).count)
    q_rows = client.retrieve(
        collection_name=q["collection"],
        ids=sample_pids,
        with_payload=True,
        with_vectors=True,
    )
    q_by_id = {int(r.id): r for r in q_rows}

    checks = []
    for rank, pid in zip(sample_ranks, sample_pids):
        pg = pg_rows[pid]
        qr = q_by_id.get(pid)
        if qr is None:
            raise RuntimeError(f"Qdrant missing sample PID {pid}")
        local = np.asarray(emb[rank], dtype=np.float32)
        pgvec = np.asarray(pg[6].to_numpy(), dtype=np.float32)
        qvec = np.asarray(qr.vector, dtype=np.float32)

        expected_language = "en" if int(meta["language_code"][rank]) == 0 else "de"
        expected_status = "active" if int(meta["status_code"][rank]) == 0 else "inactive"

        ok = (
            int(pg[0]) == int(meta["document_id"][rank]) == int(qr.payload["document_id"])
            and int(pg[1]) == int(meta["tenant_id"][rank]) == int(qr.payload["tenant_id"])
            and int(pg[2]) == int(meta["owner_id"][rank]) == int(qr.payload["owner_id"])
            and int(pg[3]) == int(meta["category_id"][rank]) == int(qr.payload["category_id"])
            and pg[4] == expected_language == qr.payload["language"]
            and pg[5] == expected_status == qr.payload["status"]
            and np.allclose(pgvec, local, atol=1e-6)
            and np.allclose(qvec, local, atol=1e-6)
        )
        checks.append({"rank": int(rank), "pid": pid, "ok": bool(ok)})
        if not ok:
            raise RuntimeError(f"Cross-store validation failed for rank={rank}, pid={pid}")

    expected_docs = (args.size + int(cfg["chunks_per_document"]) - 1) // int(cfg["chunks_per_document"])
    if int(pg_counts[1]) != args.size or q_count != args.size:
        raise RuntimeError(f"Count mismatch: PostgreSQL chunks={pg_counts[1]}, Qdrant={q_count}, expected={args.size}")
    if int(pg_counts[0]) != expected_docs or int(pg_counts[2]) != expected_docs:
        raise RuntimeError(f"PostgreSQL document/permission count mismatch: {pg_counts}")

    report = {
        "dataset_size": args.size,
        "postgres_counts": list(map(int, pg_counts)),
        "qdrant_points": q_count,
        "sample_checks": checks,
        "passed": True,
    }
    write_json(PROCESSED / f"load_validation_{args.size}.json", report)
    print(json.dumps(report, indent=2))
    print("Cross-store validation PASSED.")

if __name__ == "__main__":
    main()
