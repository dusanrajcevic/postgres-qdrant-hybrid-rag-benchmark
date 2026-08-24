# PostgreSQL and Qdrant Hybrid RAG Benchmark

[![DOI](https://zenodo.org/badge/1344575339.svg)](https://doi.org/10.5281/zenodo.22076089)

Reproduction code and held-out measurements for the manuscript **Relational Extensions vs. Dedicated Vector Engines: An Empirical Performance Benchmark of PostgreSQL (pgvector) and Qdrant in Hybrid RAG Architectures** by Dušan Rajčević and Vladimir Šimović.

The benchmark compares three single-node retrieval configurations under pure vector, metadata-filtered, compound metadata, and live permission-aware workloads:

- PostgreSQL 18.4 with pgvector 0.8.6 using HNSW
- PostgreSQL 18.4 with pgvector 0.8.6 and pgvectorscale 0.9.0 using StreamingDiskANN
- Qdrant 1.18.1 using HNSW and indexed payload fields

The paper reports final held-out measurements at **100,000** and **250,000** vectors.

## Repository status

This repository contains the code, frozen configurations, tuning provenance, query plans, environment snapshots, and complete held-out measurement CSVs used for the reported results. The large generated `data/` directory is not included because it contains downloaded MS MARCO files, processed samples, model files, embeddings, metadata, and exact ground-truth arrays. The data pipeline required to regenerate those artifacts is included.

The original data pipeline generated a deterministic one-million-passage sample so that scale points are nested prefixes. `config/experiment.json` therefore lists 100k, 250k, 500k, and 1M candidate prefixes. The manuscript reports measurements only for 100k and 250k. The publication scope is recorded separately in `config/paper_scope.json`.

## Experimental design

- Corpus: MS MARCO v1 passage-ranking collection, 8,841,823 passages
- Deterministic passage sample seed: `20260823`
- Fixed query set: 1,000 MS MARCO development queries
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Model revision: `ea78891063587eb050ed4166b20062eaf978037c`
- Embeddings: 384-dimensional, L2-normalized float32
- Top-k: 10
- Tuning and warmup ranks: 0 to 99
- Held-out measurement ranks: 100 to 999
- Held-out queries per workload and repetition: 900
- Workloads: 7
- Measured repetitions: 3
- Unrecorded warmups per workload and repetition: 200
- Tuning Recall@10 target: 0.95

No parameter selection is performed on ranks 100 to 999. Held-out recall shortfalls are retained as results and do not trigger retuning.

## Workloads

| Workload | Description |
| --- | --- |
| W1 | Pure vector top-10 retrieval |
| W2 50% | Approximately 50% eligible by language |
| W2 10% | Approximately 10% eligible by category |
| W2 1% | Approximately 1% eligible by tenant |
| W2 0.1% | Approximately 0.1% eligible by owner |
| W3 | Compound tenant, status, category, and date predicate |
| W4 ACL | Tenant and status filtering with live PostgreSQL permission validation |

W2 and W3 retrieval metadata is denormalized beside each PostgreSQL vector and stored as indexed Qdrant payload so that the systems receive comparable vector-local metadata. W4 permissions remain relational in PostgreSQL and are not copied into Qdrant.

## Original benchmark environment

The recorded experiment ran on a MacBook Pro with Apple M1 Pro and 16 GB unified memory using Docker Desktop ARM64. Each database service was limited to 6 CPUs and 8 GB memory. PostgreSQL used 1 GB shared memory.

Key software versions:

- macOS 26.6.2
- Docker Desktop 4.87.0
- Docker Engine 29.7.2
- Docker Compose 5.4.0
- Python 3.11.11
- PostgreSQL 18.4
- pgvector 0.8.6
- pgvectorscale 0.9.0
- Qdrant 1.18.1

See `docs/ENVIRONMENT.md` for image digests and additional details.

## Quick start

### 1. Create the Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-data.txt
```

### 2. Start PostgreSQL and Qdrant

```bash
docker compose pull
docker compose up -d postgres qdrant
bash scripts/check_versions.sh
```

### 3. Generate the dataset, metadata, and embeddings

```bash
bash scripts/run_data_pipeline.sh
```

This downloads MS MARCO from the official Microsoft-hosted source, creates the deterministic sample, generates synthetic relational metadata, downloads the pinned embedding model revision, and generates the passage and query embeddings.

The original embedding files had these SHA-256 hashes:

```text
passage_embeddings.npy  3494716799612ea9c6cffa53f169ec5169597346f66f5c9310ea2a5e37fbab38
query_embeddings.npy    148335c6ed9b25d60ccb5f53662dd8054d0b8e5a07da8508c918622a7fe12c26
```

### 4. Follow the reproduction protocol

The full sequence for loading, validation, exact ground truth, tuning, final measurement, and result validation is documented in `docs/REPRODUCIBILITY.md`.

## Final measurement artifacts

Paper-eligible results are under:

```text
results/measurements/100000/
results/measurements/250000/
```

Each size contains:

- 9 raw repetition CSVs: 3 engines x 3 repetitions
- 6,300 rows per repetition CSV: 7 workloads x 900 held-out queries
- final pooled summary CSV and JSON
- measurement manifest
- PostgreSQL plan captures
- environment snapshot
- per-repetition summary JSON files

Across both sizes, the repository contains 113,400 held-out query observations. Every held-out request returned the requested ten results.

Validate the archived release with:

```bash
python analysis/validate_release.py
```

Build a compact combined summary with:

```bash
python analysis/build_paper_summary.py
```

## Important measurement semantics

Latency is client-side wall-clock time measured with `time.perf_counter_ns()` around one logical retrieval request. Database connections and the Qdrant client persist across queries. Connection setup is excluded.

For Qdrant W4, timing includes Qdrant candidate retrieval and PostgreSQL relational ACL validation. No embedding generation or LLM generation occurs inside the timed region.

The `tuning/` directory is retained to document how search effort and access paths were selected. Tuning latency and recall are not paper results. The final configurations are frozen in:

- `config/final_100k_measurement.json`
- `config/final_250k_measurement.json`

SHA-256 checksums for the frozen configurations and archived measurement artifacts are stored in `checksums/ARTIFACTS.sha256`. For CSV and JSON artifacts, the validator canonicalizes line endings to LF before hashing so validation is stable across Git checkouts on different operating systems. The release validator verifies these checksums before checking row-level measurement invariants.

## Repository layout

```text
.
├── analysis/                 release validation and combined paper summary
├── config/                   experiment provenance and frozen configurations
├── docs/                     data, environment, protocol, and release documentation
├── results/measurements/     archived held-out measurements
├── scripts/                  data, loading, tuning, and measurement programs
├── sql/                      PostgreSQL schema and index definitions
├── tuning/                   tuning grids, selected parameters, and plans
├── CITATION.cff
├── LICENSE
├── docker-compose.yml
├── requirements-data.txt
└── requirements.txt
```

## Data availability

The benchmark is derived from the official MS MARCO v1 passage-ranking collection. The repository contains scripts for deterministic passage and query sampling, synthetic relational metadata and access-control generation, embedding generation, and exact ground-truth construction.

The raw MS MARCO collection is not redistributed in this repository. The data preparation pipeline downloads it from the official Microsoft-hosted locations recorded in `config/provenance.json`. Large generated artifacts, including processed samples, model files, embeddings, metadata, and ground-truth arrays, are excluded from Git because they require several gigabytes and can be regenerated using the provided pipeline.

See [`docs/DATA.md`](docs/DATA.md) for full data provenance, sampling details, metadata distributions, embedding hashes, the expected local data tree, and regeneration instructions.

## Citation

The exact software release used for the manuscript has been archived on Zenodo.

If you use this benchmark implementation, configurations, or experimental artifacts, please cite:

Rajčević, D., & Šimović, V. (2026). *Relational Extensions vs. Dedicated Vector Engines: An Empirical Performance Benchmark of PostgreSQL (pgvector) and Qdrant in Hybrid RAG Architectures* (Version 1.0.0). Zenodo. https://doi.org/10.5281/zenodo.22076090

Machine-readable citation metadata is provided in [`CITATION.cff`](CITATION.cff).

## License

The source code and original repository documentation are released under the MIT License. Third-party datasets, models, container images, database systems, and libraries remain subject to their own licenses and terms. MS MARCO data is not covered by this repository's MIT License.
