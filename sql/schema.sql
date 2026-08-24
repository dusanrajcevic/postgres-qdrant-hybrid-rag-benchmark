
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE;

DROP TABLE IF EXISTS document_permissions;
DROP TABLE IF EXISTS chunks;
DROP TABLE IF EXISTS documents;

CREATE TABLE documents (
    document_id BIGINT PRIMARY KEY,
    tenant_id INTEGER NOT NULL,
    owner_id INTEGER NOT NULL,
    category_id INTEGER NOT NULL,
    language TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE chunks (
    chunk_id BIGINT PRIMARY KEY,
    document_id BIGINT NOT NULL REFERENCES documents(document_id),
    text TEXT NOT NULL,
    embedding VECTOR(384) NOT NULL
);

CREATE TABLE document_permissions (
    document_id BIGINT NOT NULL REFERENCES documents(document_id),
    principal_id INTEGER NOT NULL,
    permission_type TEXT NOT NULL,
    PRIMARY KEY (document_id, principal_id, permission_type)
);

CREATE INDEX documents_tenant_idx ON documents (tenant_id);
CREATE INDEX documents_category_idx ON documents (category_id);
CREATE INDEX documents_language_idx ON documents (language);
CREATE INDEX documents_status_idx ON documents (status);
CREATE INDEX documents_created_at_idx ON documents (created_at);
CREATE INDEX document_permissions_principal_idx
    ON document_permissions (principal_id, permission_type, document_id);
