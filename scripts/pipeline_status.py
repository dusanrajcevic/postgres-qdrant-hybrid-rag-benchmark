from __future__ import annotations

from pathlib import Path
import json

from common import PROCESSED, load_config

def check(path: Path, label: str):
    print(f"{'OK' if path.exists() else '--'}  {label}: {path}")

def main():
    cfg = load_config()
    check(PROCESSED / "dataset_manifest.json", "dataset prepared")
    check(PROCESSED / "metadata_manifest.json", "metadata generated")
    check(PROCESSED / "embedding_manifest.json", "embeddings generated")
    if (PROCESSED / "embedding_manifest.json").exists():
        print(json.dumps(json.loads((PROCESSED / "embedding_manifest.json").read_text()), indent=2))
    for size in cfg["dataset_sizes"]:
        check(PROCESSED / f"postgres_load_{size}.json", f"PostgreSQL {size:,} loaded")
        check(PROCESSED / f"qdrant_load_{size}.json", f"Qdrant {size:,} loaded")
        check(PROCESSED / f"load_validation_{size}.json", f"cross-store {size:,} validated")

if __name__ == "__main__":
    main()
