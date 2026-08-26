#!/usr/bin/env bash
set -euo pipefail
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
trap 'kill $BACKEND_PID 2>/dev/null || true' EXIT
sleep 1
python -m streamlit run frontend/app.py --server.port 8501
