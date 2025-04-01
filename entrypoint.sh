#!/bin/sh
set -e

# Start Ray head node
.venv/bin/ray start --head --port=6379 --dashboard-host=0.0.0.0 --disable-usage-stats

# Serve the model (detached=True, starts in background)
.venv/bin/python catboost_incremental/serve_ray.py

# Keep container alive indefinitely
exec tail -f /dev/null
