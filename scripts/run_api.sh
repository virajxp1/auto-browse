#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# Use local virtualenv automatically when present.
if [[ -d ".venv" && -z "${VIRTUAL_ENV:-}" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

HOST="${AUTO_BROWSE_API_HOST:-0.0.0.0}"
PORT="${PORT:-${AUTO_BROWSE_API_PORT:-8000}}"

# Set AUTO_BROWSE_INSTALL_PLAYWRIGHT=1 at startup if browser binaries are missing.
if [[ "${AUTO_BROWSE_INSTALL_PLAYWRIGHT:-0}" == "1" ]]; then
  playwright install chromium
fi

if [[ "${AUTO_BROWSE_API_RELOAD:-0}" == "1" ]]; then
  exec python -m uvicorn auto_browse.api:app --host "${HOST}" --port "${PORT}" --reload
fi

exec python -m uvicorn auto_browse.api:app --host "${HOST}" --port "${PORT}"
