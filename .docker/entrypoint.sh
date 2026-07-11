#!/usr/bin/env bash
set -e

cd /workspace

# Execute the command passed into this entrypoint
exec "$@"
