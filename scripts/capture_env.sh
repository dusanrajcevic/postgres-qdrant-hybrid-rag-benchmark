#!/usr/bin/env bash
set -euo pipefail

mkdir -p results
OUT="results/environment.txt"

{
  echo "=== Timestamp ==="
  date -u +"%Y-%m-%dT%H:%M:%SZ"
  echo

  echo "=== macOS ==="
  sw_vers
  echo

  echo "=== Hardware ==="
  echo "CPU: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || true)"
  echo "Physical CPUs: $(sysctl -n hw.physicalcpu 2>/dev/null || true)"
  echo "Logical CPUs: $(sysctl -n hw.logicalcpu 2>/dev/null || true)"
  echo "Memory bytes: $(sysctl -n hw.memsize 2>/dev/null || true)"
  echo

  echo "=== Docker ==="
  docker version
  echo
  docker info
  echo

  echo "=== Images ==="
  docker image inspect timescale/timescaledb-ha:pg18.4-ts2.29.1-all-oss \
    --format 'Timescale image id={{.Id}} repo_digests={{json .RepoDigests}}' 2>/dev/null || true
  docker image inspect qdrant/qdrant:v1.18.1 \
    --format 'Qdrant image id={{.Id}} repo_digests={{json .RepoDigests}}' 2>/dev/null || true
} | tee "$OUT"

echo "Wrote $OUT"
