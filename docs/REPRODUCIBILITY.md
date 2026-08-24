# Reproducing the benchmark

## Archived benchmark version

The exact repository version associated with the reported experiments is archived on Zenodo:

https://doi.org/10.5281/zenodo.22076090

The corresponding Git tag is `v1.0.0-paper`.

This document describes the shortest path from a fresh checkout to regenerated data and final measurements. The final CSVs already included under `results/measurements/` should be treated as the archived paper outputs. New runs should be written separately if they are performed on different hardware or software versions.

## 1. Host setup

Install Docker Desktop with ARM64 container support and Python 3.11. Configure Docker Desktop with at least 6 CPUs and 8 GB memory available to the benchmark. The Compose file applies a 6 CPU and 8 GB limit to each database service.

Create the Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-data.txt
```

Pull and start the services:

```bash
docker compose pull
docker compose up -d postgres qdrant
bash scripts/check_versions.sh
```

## 2. Generate the immutable data artifacts

The following command downloads MS MARCO, builds the deterministic sample and query set, generates synthetic metadata, and computes embeddings:

```bash
bash scripts/run_data_pipeline.sh
```

On Apple Silicon, embedding generation uses MPS when available. To force a device manually:

```bash
python scripts/generate_embeddings.py --device mps --batch-size 128
# or
python scripts/generate_embeddings.py --device cpu --batch-size 64
```

Compare the generated embedding hashes with `docs/DATA.md`.

## 3. Prepare the 100k scale

Load identical vectors and metadata into both stores:

```bash
python scripts/load_postgres.py --size 100000
python scripts/load_qdrant.py --size 100000
python scripts/validate_loaded_data.py --size 100000
```

Prepare PostgreSQL filter columns and indexes:

```bash
python scripts/prepare_filtered_schema.py --size 100000
```

Build exact ground truth for all seven workloads:

```bash
for w in w1 w2_50 w2_10 w2_1 w2_0_1 w3 w4_acl; do
  python scripts/build_ground_truth.py --size 100000 --workload "$w"
done
python scripts/verify_ground_truth.py --size 100000
```

## 4. Reproduce tuning at 100k

The `tuning/` directory already contains the tuning artifacts used to freeze the paper configuration. To reproduce parameter selection, run the tuning scripts using ranks 0 to 99 only. The sequence is:

```bash
python scripts/build_tuning_index.py --size 100000 --engine pg-hnsw
python scripts/tune_w1.py --size 100000 --engine pg-hnsw

python scripts/build_tuning_index.py --size 100000 --engine pg-diskann
python scripts/tune_w1.py --size 100000 --engine pg-diskann

python scripts/build_tuning_index.py --size 100000 --engine qdrant
python scripts/tune_w1.py --size 100000 --engine qdrant

python scripts/tune_pg_diskann_crossover.py --size 100000
python scripts/tune_pg_hnsw_crossover.py --size 100000
python scripts/tune_qdrant_crossover.py --size 100000
python scripts/tune_filtered.py --size 100000 --engine pg-diskann
python scripts/tune_filtered.py --size 100000 --engine pg-hnsw
python scripts/tune_filtered.py --size 100000 --engine qdrant
python scripts/tune_w4.py --size 100000 --engine pg-hnsw
python scripts/tune_w4.py --size 100000 --engine pg-diskann
python scripts/tune_w4.py --size 100000 --engine qdrant
```

The frozen paper configuration is `config/final_100k_measurement.json`. Do not use ranks 100 to 999 to change it.

## 5. Run the final 100k measurement

Capture the environment:

```bash
python scripts/capture_final_env.py
```

The original measurement sequence was PG-DiskANN, PG-HNSW, then Qdrant. The runner verifies the frozen access paths before timing.

```bash
python scripts/run_final_100k.py --engine pg-diskann --repetitions 3

python scripts/build_tuning_index.py --size 100000 --engine pg-hnsw
python scripts/run_final_100k.py --engine pg-hnsw --repetitions 3

python scripts/run_final_100k.py --engine qdrant --repetitions 3
python scripts/summarize_final_100k.py
```

The measurement runner refuses to overwrite an existing engine result by default. Keep the archived paper CSVs intact.

## 6. Prepare and tune the 250k scale

The data are nested, so the corpus and embeddings are not regenerated. Prepare the larger prefix:

```bash
python scripts/prepare_250k_scale.py --size 250000
```

`prepare_250k_scale.py` reloads both stores, validates cross-store consistency, prepares PostgreSQL filter metadata, and builds all seven exact ground-truth files unless `--skip-ground-truth` is supplied.

Reproduce 250k tuning with:

```bash
python scripts/build_tuning_index.py --size 250000 --engine pg-hnsw
python scripts/tune_w1.py --size 250000 --engine pg-hnsw

python scripts/build_tuning_index.py --size 250000 --engine pg-diskann
python scripts/tune_w1.py --size 250000 --engine pg-diskann

python scripts/build_tuning_index.py --size 250000 --engine qdrant
python scripts/tune_w1.py --size 250000 --engine qdrant

python scripts/tune_filtered_250k.py --engine pg-diskann
python scripts/build_tuning_index.py --size 250000 --engine pg-hnsw
python scripts/tune_filtered_250k.py --engine pg-hnsw
python scripts/tune_filtered_250k.py --engine qdrant

python scripts/tune_w4_250k.py --engine pg-hnsw
python scripts/build_tuning_index.py --size 250000 --engine pg-diskann
python scripts/tune_w4_250k.py --engine pg-diskann
python scripts/tune_w4_250k.py --engine qdrant
```

The selected strategies are retained in `tuning/` and are frozen in `config/final_250k_measurement.json`.

## 7. Run the final 250k measurement

```bash
python scripts/capture_final_env_250k.py
python scripts/run_final_250k.py --engine pg-diskann --repetitions 3

python scripts/build_tuning_index.py --size 250000 --engine pg-hnsw
python scripts/run_final_250k.py --engine pg-hnsw --repetitions 3

python scripts/run_final_250k.py --engine qdrant --repetitions 3
python scripts/summarize_final_250k.py
```

## 8. Validate the release artifacts

Run:

```bash
python analysis/validate_release.py
python analysis/build_paper_summary.py
```

The validator checks the expected files, row counts, held-out rank coverage, workload coverage, repetition count, returned-result completeness, and summary consistency.
