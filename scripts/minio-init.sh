#!/usr/bin/env sh
set -eu

# MinIO MC-based initializer (idempotent)
# Environment variables expected:
# MINIO_ROOT_USER, MINIO_ROOT_PASSWORD

MC_BIN=/usr/bin/mc
ALIAS_NAME=myminio
ENDPOINT="http://minio:9000"
BUCKET="milvus-bucket"
RETRIES=${MINIO_INIT_RETRIES:-12}
SLEEP=${MINIO_INIT_SLEEP:-5}

i=0
echo "⏳ Waiting for MinIO (attempts: $RETRIES)..."

until $MC_BIN alias set $ALIAS_NAME "$ENDPOINT" "$MINIO_ROOT_USER" "$MINIO_ROOT_PASSWORD" >/dev/null 2>&1; do
  i=$((i+1))
  if [ "$i" -ge "$RETRIES" ]; then
    echo "❌ Failed to reach MinIO after $RETRIES attempts"
    exit 1
  fi
  echo " - not ready yet, sleeping $SLEEP (try $i/$RETRIES)"
  sleep "$SLEEP"
done

echo "✅ mc can talk to MinIO — ensuring bucket exists: $BUCKET"
$MC_BIN mb --ignore-existing $ALIAS_NAME/$BUCKET || true
echo "✅ MinIO bucket ensured: $ALIAS_NAME/$BUCKET"
