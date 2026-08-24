# Changelog

## 1.0.0 - 2026-08-24

Initial research release associated with the 100k and 250k benchmark study.

- Includes PostgreSQL HNSW, PostgreSQL StreamingDiskANN, and Qdrant benchmark code.
- Includes deterministic MS MARCO sampling, synthetic metadata generation, embedding generation, exact ground-truth construction, loading, tuning, and held-out measurement scripts.
- Includes frozen 100k and 250k strategies.
- Includes the complete held-out measurement CSVs used for the reported 100k and 250k results.
- Excludes the large generated `data/` directory. It can be regenerated from the official MS MARCO source with the included pipeline.
