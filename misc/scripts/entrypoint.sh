#!/usr/bin/env bash
set -euo pipefail

echo "=========================================="
echo "🚀 Enterprise RAG Bot - Startup"
echo "=========================================="

DATABASE_HOST="${DATABASE_HOST:-postgres}"
DATABASE_PORT="${DATABASE_PORT:-5432}"
DATABASE_USER="${DATABASE_USER:-ragbot}"
MILVUS_HOST="${MILVUS_HOST:-milvus}"

# --------------------------------------------------
# Wait for PostgreSQL
# --------------------------------------------------
echo "⏳ Waiting for PostgreSQL at ${DATABASE_HOST}:${DATABASE_PORT}..."
until pg_isready -h "$DATABASE_HOST" -p "$DATABASE_PORT" -U "$DATABASE_USER" >/dev/null 2>&1; do
    sleep 2
done
echo "✅ PostgreSQL ready"

# --------------------------------------------------
# Wait for Milvus
# --------------------------------------------------
echo "⏳ Waiting for Milvus at ${MILVUS_HOST}:9091..."
MAX_ATTEMPTS=60
ATTEMPT=0

until curl -sf "http://${MILVUS_HOST}:9091/metrics" >/dev/null 2>&1; do
    ATTEMPT=$((ATTEMPT + 1))
    if [ "$ATTEMPT" -ge "$MAX_ATTEMPTS" ]; then
        echo "❌ Milvus did not become ready in time"
        exit 1
    fi
    sleep 3
done
echo "✅ Milvus ready"

# --------------------------------------------------
# Prepare directories
# --------------------------------------------------
mkdir -p /app/logs /app/uploads /app/outputs /app/backups /app/temp
chmod -R 755 /app/logs /app/uploads /app/outputs /app/backups /app/temp

echo "✅ Initialization complete"
echo "➡️  Starting main process: $*"

exec "$@"
