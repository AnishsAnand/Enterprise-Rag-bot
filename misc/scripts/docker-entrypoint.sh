#!/bin/sh
set -e

echo "🚀 Starting Enterprise RAG Bot..."

# Ensure required binaries exist
which nginx
which supervisord
which python

# Print env for debugging
echo "Environment loaded"

exec "$@"
