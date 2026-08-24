#!/usr/bin/env bash
set -euo pipefail

if [[ ! -d ".venv" ]]; then
  echo "Run this from the vector_benchmark_starter directory after creating .venv."
  exit 1
fi

source .venv/bin/activate

echo "=== 1/3 Prepare deterministic MS MARCO corpus + query sample ==="
python scripts/prepare_dataset.py

echo "=== 2/3 Generate deterministic relational metadata ==="
python scripts/generate_metadata.py

echo "=== 3/3 Generate fixed passage/query embeddings ==="
python scripts/generate_embeddings.py --device auto --batch-size 128

echo
echo "Data preparation complete."
echo "Next load ONE dataset size at a time, starting with 100000:"
echo "  python scripts/load_postgres.py --size 100000"
echo "  python scripts/load_qdrant.py --size 100000"
echo "  python scripts/validate_loaded_data.py --size 100000"
