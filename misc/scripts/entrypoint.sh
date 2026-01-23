#!/bin/bash
set -e

echo "=========================================="
echo "🚀 Enterprise RAG Bot - Startup"
echo "=========================================="

# Wait for PostgreSQL
DB_HOST="${DATABASE_HOST:-postgres}"
DB_PORT="${DATABASE_PORT:-5432}"
DB_USER="${DATABASE_USER:-ragbot}"

until pg_isready -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" >/dev/null 2>&1; do
    echo "⏳ Waiting for PostgreSQL at $DB_HOST:$DB_PORT..."
    sleep 2
done

echo "✅ PostgreSQL ready"

# Disable Milvus
export DISABLE_MILVUS=true
unset MILVUS_HOST
unset MILVUS_PORT

# Runtime directories
for dir in /app/logs /app/uploads /app/outputs /app/backups /app/temp /var/log/supervisor; do
    mkdir -p "$dir"
    chmod 755 "$dir"
done

echo "✅ Initialization complete"
echo "🚀 Starting supervisord..."

exec /usr/bin/supervisord -c /etc/supervisord.conf
