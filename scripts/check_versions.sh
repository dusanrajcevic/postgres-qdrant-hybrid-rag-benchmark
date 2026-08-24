#!/usr/bin/env bash
set -euo pipefail

echo "Starting PostgreSQL container..."
docker compose up -d postgres

echo "Waiting for PostgreSQL..."
until docker compose exec -T postgres pg_isready -U benchmark -d benchmark >/dev/null 2>&1; do
  sleep 1
done

echo "PostgreSQL and available vector extensions:"
docker compose exec -T postgres psql -U benchmark -d benchmark -v ON_ERROR_STOP=1 <<'SQL'
SELECT version();
SELECT name, default_version, installed_version
FROM pg_available_extensions
WHERE name IN ('vector', 'vectorscale')
ORDER BY name;

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE;

SELECT extname, extversion
FROM pg_extension
WHERE extname IN ('vector', 'vectorscale')
ORDER BY extname;
SQL

echo
echo "Qdrant version:"
docker compose up -d qdrant
curl -s http://localhost:6333/ | python3 -m json.tool || true
