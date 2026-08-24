from __future__ import annotations

import json
from datetime import date, timedelta
import math

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from common import PROCESSED, ensure_dirs, load_config, sha256_file, write_json

EPOCH = date(2020, 1, 1)
END = date(2025, 12, 31)
N_DAYS = (END - EPOCH).days + 1

def main() -> None:
    cfg = load_config()
    ensure_dirs()

    max_vectors = int(cfg["max_vectors"])
    chunks_per_doc = int(cfg["chunks_per_document"])
    n_docs = math.ceil(max_vectors / chunks_per_doc)
    rng = np.random.default_rng(int(cfg["seed"]) + 101)

    # Controlled distribution:
    # tenant 0 ≈20%; tenants 1..80 ≈1% each.
    tenant_values = np.arange(81, dtype=np.int16)
    tenant_probs = np.array([0.20] + [0.01] * 80, dtype=np.float64)
    tenant_probs /= tenant_probs.sum()
    tenant_id = rng.choice(tenant_values, size=n_docs, p=tenant_probs).astype(np.int16)

    owner_id = rng.integers(0, 1000, size=n_docs, dtype=np.int16)
    category_id = rng.integers(0, 10, size=n_docs, dtype=np.int8)
    language_code = rng.integers(0, 2, size=n_docs, dtype=np.int8)  # 0=en, 1=de
    status_code = (rng.random(n_docs) >= 0.80).astype(np.int8)      # 0=active (~80%), 1=inactive
    created_day = rng.integers(0, N_DAYS, size=n_docs, dtype=np.int16)
    principal_id = rng.integers(0, 5, size=n_docs, dtype=np.int8)   # one read principal per document

    document_id = np.arange(n_docs, dtype=np.int64)
    languages = np.where(language_code == 0, "en", "de")
    statuses = np.where(status_code == 0, "active", "inactive")
    created_at = [(EPOCH + timedelta(days=int(d))).isoformat() + "T00:00:00Z" for d in created_day]

    docs = pa.table({
        "document_id": pa.array(document_id),
        "tenant_id": pa.array(tenant_id.astype(np.int32)),
        "owner_id": pa.array(owner_id.astype(np.int32)),
        "category_id": pa.array(category_id.astype(np.int32)),
        "language": pa.array(languages),
        "status": pa.array(statuses),
        "created_at": pa.array(created_at),
    })
    pq.write_table(docs, PROCESSED / "documents.parquet", compression="zstd")

    perms = pa.table({
        "document_id": pa.array(document_id),
        "principal_id": pa.array(principal_id.astype(np.int32)),
        "permission_type": pa.array(["read"] * n_docs),
    })
    pq.write_table(perms, PROCESSED / "permissions.parquet", compression="zstd")

    # Metadata aligned by sample_rank lets ground-truth and Qdrant loaders avoid joins.
    doc_for_rank = np.arange(max_vectors, dtype=np.int64) // chunks_per_doc
    np.savez_compressed(
        PROCESSED / "aligned_metadata.npz",
        document_id=doc_for_rank,
        tenant_id=tenant_id[doc_for_rank],
        owner_id=owner_id[doc_for_rank],
        category_id=category_id[doc_for_rank],
        language_code=language_code[doc_for_rank],
        status_code=status_code[doc_for_rank],
        created_day=created_day[doc_for_rank],
        principal_id=principal_id[doc_for_rank],
    )

    workload_spec = {
        "notes": {
            "language_code": {"0": "en", "1": "de"},
            "status_code": {"0": "active", "1": "inactive"},
            "created_day_epoch": EPOCH.isoformat(),
            "permissions": "Each synthetic document is readable by one principal in 0..4."
        },
        "w2_50": "language_code = query_rank mod 2",
        "w2_10": "category_id = query_rank mod 10",
        "w2_1": "tenant_id = 1 + (query_rank mod 80)",
        "w2_0_1": "owner_id = query_rank mod 1000",
        "w3": {
            "tenant_id": 0,
            "status_code": 0,
            "category_rule": "five consecutive category IDs beginning at query_rank mod 10",
            "created_at_gte": "2022-01-01T00:00:00Z"
        },
        "w4_acl": {
            "tenant_rule": "1 + (query_rank mod 80)",
            "status_code": 0,
            "principal_rule": "query_rank mod 5",
            "permission_type": "read"
        }
    }
    write_json(PROCESSED / "workload_filters.json", workload_spec)

    summary = {
        "documents": n_docs,
        "chunks": max_vectors,
        "chunks_per_document": chunks_per_doc,
        "tenant_0_fraction": float(np.mean(tenant_id == 0)),
        "typical_tenant_1_80_mean_fraction": float(np.mean([np.mean(tenant_id == x) for x in range(1, 81)])),
        "active_fraction": float(np.mean(status_code == 0)),
        "english_fraction": float(np.mean(language_code == 0)),
        "category_mean_fraction": float(np.mean([np.mean(category_id == x) for x in range(10)])),
        "owner_mean_fraction": float(np.mean([np.mean(owner_id == x) for x in range(1000)])),
        "principal_mean_fraction": float(np.mean([np.mean(principal_id == x) for x in range(5)])),
        "date_range": [EPOCH.isoformat(), END.isoformat()],
    }
    write_json(PROCESSED / "metadata_summary.json", summary)

    manifest = {
        "seed": int(cfg["seed"]) + 101,
        "documents_parquet_sha256": sha256_file(PROCESSED / "documents.parquet"),
        "permissions_parquet_sha256": sha256_file(PROCESSED / "permissions.parquet"),
        "aligned_metadata_sha256": sha256_file(PROCESSED / "aligned_metadata.npz"),
    }
    write_json(PROCESSED / "metadata_manifest.json", manifest)
    print(json.dumps(summary, indent=2))
    print("Metadata generation complete.")

if __name__ == "__main__":
    main()
