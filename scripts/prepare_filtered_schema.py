from __future__ import annotations
import argparse, json
import psycopg
from common import PROCESSED, load_config, write_json

def conn(cfg):
    p = cfg["postgres"]
    return psycopg.connect(
        host=p["host"], port=p["port"], dbname=p["dbname"],
        user=p["user"], password=p["password"]
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, required=True)
    args = ap.parse_args()
    cfg = load_config()
    if args.size not in set(map(int, cfg["dataset_sizes"])):
        raise SystemExit(f"--size must be one of {cfg['dataset_sizes']}")

    with conn(cfg) as c:
        n = int(c.execute("SELECT count(*) FROM chunks").fetchone()[0])
        if n != args.size:
            raise RuntimeError(f"PostgreSQL has {n:,} chunks, expected {args.size:,}")

        # Remove ANN indexes before changing benchmark row metadata.
        c.execute("DROP INDEX IF EXISTS chunks_embedding_hnsw")
        c.execute("DROP INDEX IF EXISTS chunks_embedding_diskann")

        c.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS tenant_id integer")
        c.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS owner_id integer")
        c.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS category_id integer")
        c.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS language text")
        c.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS status text")
        c.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS created_at timestamptz")
        c.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS labels smallint[]")
        c.commit()

        # Namespace labels so unrelated fields never collide.
        # 100-101: language, 1000-1009: category, 2000-2080: tenant,
        # 3000-3999: owner, 5000-5001: status.
        c.execute("""
            UPDATE chunks AS ch
            SET tenant_id   = d.tenant_id,
                owner_id    = d.owner_id,
                category_id = d.category_id,
                language    = d.language,
                status      = d.status,
                created_at  = d.created_at,
                labels = ARRAY[
                    (100  + CASE d.language WHEN 'en' THEN 0 ELSE 1 END)::smallint,
                    (1000 + d.category_id)::smallint,
                    (2000 + d.tenant_id)::smallint,
                    (3000 + d.owner_id)::smallint,
                    (5000 + CASE d.status WHEN 'active' THEN 0 ELSE 1 END)::smallint
                ]::smallint[]
            FROM documents AS d
            WHERE ch.document_id = d.document_id
        """)
        c.commit()

        indexes = [
            "CREATE INDEX IF NOT EXISTS chunks_tenant_idx ON chunks (tenant_id)",
            "CREATE INDEX IF NOT EXISTS chunks_owner_idx ON chunks (owner_id)",
            "CREATE INDEX IF NOT EXISTS chunks_category_idx ON chunks (category_id)",
            "CREATE INDEX IF NOT EXISTS chunks_language_idx ON chunks (language)",
            "CREATE INDEX IF NOT EXISTS chunks_status_idx ON chunks (status)",
            "CREATE INDEX IF NOT EXISTS chunks_created_at_idx ON chunks (created_at)",
        ]
        for sql in indexes:
            c.execute(sql)
        c.execute("ANALYZE chunks")
        c.commit()

        row = c.execute("""
            SELECT count(*) AS total,
                   count(*) FILTER (
                     WHERE tenant_id IS NULL OR owner_id IS NULL OR category_id IS NULL
                        OR language IS NULL OR status IS NULL OR created_at IS NULL
                        OR labels IS NULL OR cardinality(labels) <> 5
                   ) AS bad,
                   count(DISTINCT tenant_id),
                   count(DISTINCT owner_id),
                   count(DISTINCT category_id)
            FROM chunks
        """).fetchone()

        if int(row[0]) != args.size or int(row[1]) != 0:
            raise RuntimeError(f"Metadata augmentation validation failed: {row}")

        sample = c.execute("""
            SELECT chunk_id, tenant_id, owner_id, category_id, language, status,
                   created_at, labels
            FROM chunks ORDER BY chunk_id LIMIT 3
        """).fetchall()

    report = {
        "dataset_size": args.size,
        "chunks": int(row[0]),
        "bad_rows": int(row[1]),
        "distinct_tenants": int(row[2]),
        "distinct_owners": int(row[3]),
        "distinct_categories": int(row[4]),
        "label_namespaces": {
            "language": "100 + code",
            "category": "1000 + category_id",
            "tenant": "2000 + tenant_id",
            "owner": "3000 + owner_id",
            "status": "5000 + code"
        },
        "ann_indexes_present_after_augmentation": False,
        "sample": [list(map(str, r)) for r in sample],
    }
    out = PROCESSED / f"postgres_filter_metadata_{args.size}.json"
    write_json(out, report)
    print(json.dumps(report, indent=2))
    print("PostgreSQL filter metadata preparation PASSED.")
    print("ANN indexes were intentionally removed and must now be rebuilt.")

if __name__ == "__main__":
    main()
