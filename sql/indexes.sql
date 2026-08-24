
-- PG-HNSW condition
DROP INDEX IF EXISTS chunks_embedding_hnsw;
DROP INDEX IF EXISTS chunks_embedding_diskann;

CREATE INDEX chunks_embedding_hnsw
ON chunks USING hnsw (embedding vector_cosine_ops);

-- Suggested query-time starting point; tune against Recall@10.
-- SET hnsw.ef_search = 100;
-- SET hnsw.iterative_scan = strict_order;


-- PG-DiskANN condition
-- Drop HNSW first, then:
-- CREATE INDEX chunks_embedding_diskann
-- ON chunks USING diskann (embedding vector_cosine_ops);
--
-- Suggested query-time starting points; tune against Recall@10.
-- SET diskann.query_search_list_size = 100;
-- SET diskann.query_rescore = 100;
