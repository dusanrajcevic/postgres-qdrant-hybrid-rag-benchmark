# Experimental protocol

## Scope

The paper evaluates three retrieval configurations at 100,000 and 250,000 vectors:

- PG-HNSW: PostgreSQL 18.4 with pgvector 0.8.6 using HNSW
- PG-DiskANN: PostgreSQL 18.4 with pgvector 0.8.6 and pgvectorscale 0.9.0 using StreamingDiskANN
- Qdrant: Qdrant 1.18.1 using HNSW and indexed payload fields

The deterministic data pipeline creates a one-million-passage sample so that possible scale points are nested prefixes. The published benchmark reports final held-out measurements only at 100k and 250k.

## Query split

The same 1,000 fixed queries are used throughout the experiment. Query ranks are partitioned before final measurement:

- ranks 0 to 99: tuning and warmup only
- ranks 100 to 999: held-out measurement only

No parameter is selected from held-out results. A held-out recall value below the 0.95 tuning target remains a result and does not trigger retuning.

## Measurement counts

For each engine, dataset size, and workload:

- 900 held-out queries per repetition
- 3 measured repetitions
- 200 unrecorded warmup executions per workload and repetition
- warmups are drawn only from ranks 0 to 99
- top-k is fixed at 10

This produces 18,900 measured query observations per engine and 56,700 observations per dataset size.

## Workloads

| Workload | Eligibility condition |
| --- | --- |
| W1 | Unfiltered top-10 vector retrieval |
| W2 50% | `language_code = query_rank mod 2` |
| W2 10% | `category_id = query_rank mod 10` |
| W2 1% | `tenant_id = 1 + (query_rank mod 80)` |
| W2 0.1% | `owner_id = query_rank mod 1000` |
| W3 | tenant 0, active status, five consecutive categories beginning at `query_rank mod 10`, and `created_at >= 2022-01-01` |
| W4 ACL | tenant `1 + (query_rank mod 80)`, active status, principal `query_rank mod 5`, and permission type `read` |

For W2 and W3, retrieval metadata is available beside the vector in PostgreSQL and as indexed payload in Qdrant. W4 permissions remain normalized in PostgreSQL and are not copied into Qdrant.

## Ground truth

Exact top-10 ground truth is calculated from the saved normalized float32 embeddings, independently of either database. For filtered workloads, exact ranking is performed only over eligible vectors. Cosine similarity is implemented as a dot product because embeddings are L2-normalized.

## Timing boundary

Latency is measured in the benchmark client with `time.perf_counter_ns()` around one logical retrieval request. PostgreSQL connections and the Qdrant client are persistent, so connection setup is excluded.

For Qdrant W4, the timed operation includes Qdrant candidate retrieval, PostgreSQL permission validation, and selection of the first ten ACL-valid candidates.

## Frozen strategies

The paper measurement configurations are stored in:

- `config/final_100k_measurement.json`
- `config/final_250k_measurement.json`

The tuning grids, selected settings, and PostgreSQL plan captures are retained under `tuning/`. These tuning artifacts document parameter selection but are not paper measurement results.
