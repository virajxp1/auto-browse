#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# Prefer local virtualenv python when available.
if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  PYTHON_BIN="python"
fi

HOST="${AUTO_BROWSE_API_HOST:-0.0.0.0}"
PORT="${PORT:-${AUTO_BROWSE_API_PORT:-8000}}"
LOG_LEVEL="${AUTO_BROWSE_API_LOG_LEVEL:-info}"

# Set AUTO_BROWSE_INSTALL_PLAYWRIGHT=1 at startup if browser binaries are missing.
if [[ "${AUTO_BROWSE_INSTALL_PLAYWRIGHT:-0}" == "1" ]]; then
  "$PYTHON_BIN" -m playwright install chromium
fi

if [[ "${AUTO_BROWSE_API_RELOAD:-0}" == "1" ]]; then
  exec "$PYTHON_BIN" -m uvicorn auto_browse.api:create_app --factory --host "${HOST}" --port "${PORT}" --log-level "${LOG_LEVEL}" --reload
fi

exec "$PYTHON_BIN" -m uvicorn auto_browse.api:create_app --factory --host "${HOST}" --port "${PORT}" --log-level "${LOG_LEVEL}"
