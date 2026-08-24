# Data and generated artifacts

## Data provenance

The benchmark dataset was derived from the official MS MARCO v1 passage-ranking collection. A deterministic sample of passages and development queries was selected using fixed random seeds. Synthetic relational metadata and access-control attributes were then generated programmatically, and passage and query embeddings were produced using the sentence-transformers/all-MiniLM-L6-v2 model at a fixed revision.

## Source dataset

The benchmark uses the official MS MARCO v1 passage-ranking collection. The data pipeline downloads the collection and query archives from the Microsoft-hosted URLs recorded in `config/provenance.json` and `scripts/prepare_dataset.py`.

The repository does not redistribute the raw MS MARCO collection. Users should obtain it through the official source.

## Deterministic sample

The experiment uses seed `20260823`. `scripts/prepare_dataset.py` samples 1,000,000 passage IDs without replacement from the 8,841,823-passage collection and stores one deterministic ordering. Dataset sizes are nested prefixes of that ordering. The paper measures the 100,000 and 250,000 prefixes.

The query set contains 1,000 fixed queries selected from the MS MARCO development query file with seed `20260824`, which is the experiment seed plus one.

## Synthetic relational metadata

Four sampled passages are grouped into one synthetic document. Metadata is generated deterministically with seed `20260924`, which is the experiment seed plus 101. The distributions implemented in `scripts/generate_metadata.py` are:

- tenant 0: approximately 20%
- tenants 1 to 80: approximately 1% each
- owners 0 to 999: approximately 0.1% each
- categories 0 to 9: approximately 10% each
- language: en or de, with approximately equal probability
- active status: approximately 80%
- creation dates: uniformly sampled from 2020-01-01 through 2025-12-31
- permissions: one read principal per document, with principals 0 to 4

The actual realized fractions are written to `data/processed/metadata_summary.json` when the pipeline is run.

## Embeddings

The embedding model is `sentence-transformers/all-MiniLM-L6-v2` at revision `ea78891063587eb050ed4166b20062eaf978037c`. Vectors are 384-dimensional, L2-normalized float32 arrays. The original experiment generated embeddings on Apple MPS with model maximum sequence length 256.

The original experiment produced these SHA-256 hashes:

- passage embeddings: `3494716799612ea9c6cffa53f169ec5169597346f66f5c9310ea2a5e37fbab38`
- query embeddings: `148335c6ed9b25d60ccb5f53662dd8054d0b8e5a07da8508c918622a7fe12c26`

The embedding script writes `data/processed/embedding_manifest.json`, which also records the device, platform, shapes, dimensions, and hashes.

## Files excluded from Git

The `data/` directory is intentionally ignored because the downloaded source files, processed corpus sample, models, embeddings, and ground-truth arrays require several gigabytes. All of these artifacts can be regenerated from the included scripts.

A complete local data tree contains files such as:

```text
data/
  raw/
  models/
  processed/
    passages.parquet
    queries.parquet
    sampled_pids_by_rank.npy
    documents.parquet
    permissions.parquet
    aligned_metadata.npz
    workload_filters.json
    metadata_summary.json
    passage_embeddings.npy
    query_embeddings.npy
    dataset_manifest.json
    metadata_manifest.json
    embedding_manifest.json
  ground_truth/
    100000/
    250000/
```

The final held-out measurement results are included in Git under `results/measurements/` because they are small enough to archive with the source and are central to reproducing the paper tables.
