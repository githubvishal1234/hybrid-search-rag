#!/bin/sh
# Container entrypoint: starts the API and the UI. If either one dies, the container exits
# (so the platform restarts it) instead of serving a UI with no backend.
set -e

uv run --no-sync uvicorn hybrid_rag.api.main:app --host 0.0.0.0 --port 8000 &
API_PID=$!

uv run --no-sync streamlit run ui/app.py \
    --server.port "${PORT:-7860}" --server.address 0.0.0.0 --server.headless true &
UI_PID=$!

# Exit as soon as either process exits.
while kill -0 "$API_PID" 2>/dev/null && kill -0 "$UI_PID" 2>/dev/null; do
    sleep 5
done
exit 1
