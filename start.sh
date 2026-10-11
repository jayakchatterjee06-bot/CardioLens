#!/usr/bin/env bash
# Starts CardioLens: the FastAPI backend (:8000) and the Vite frontend (:5173) together.
# Ctrl+C stops both. Run ./setup.sh first on a fresh clone.
#
#   ./start.sh              start both servers and open the browser
#   ./start.sh --no-open    start without opening the browser
set -euo pipefail
cd "$(dirname "$0")"
export PATH="/opt/homebrew/bin:$PATH"

API_PORT=8000
UI_PORT=5173
OPEN_BROWSER=true
[[ "${1:-}" == "--no-open" ]] && OPEN_BROWSER=false

fail() { echo "error: $*" >&2; exit 1; }

# --- preflight ---------------------------------------------------------------
[[ -x .venv/bin/uvicorn ]] || fail "Python environment missing. Run ./setup.sh first."
[[ -d frontend/node_modules ]] || fail "Frontend packages missing. Run ./setup.sh first."
if [[ ! -f ml/models/manifest.json ]]; then
  echo "No trained models found. Training now (about 2 minutes)..."
  .venv/bin/python -W ignore -m ml.train
fi
for port in $API_PORT $UI_PORT; do
  if lsof -ti:"$port" >/dev/null 2>&1; then
    fail "Port $port is already in use. Stop the other process (lsof -ti:$port | xargs kill) and try again."
  fi
done

# --- start -------------------------------------------------------------------
cleanup() {
  trap - EXIT INT TERM
  echo
  echo "Stopping CardioLens..."
  kill "${API_PID:-}" "${UI_PID:-}" 2>/dev/null || true
  sleep 1
  # npm/vite spawn their own children; the ports were free at start, so whatever listens now is ours.
  lsof -ti:"$API_PORT","$UI_PORT" 2>/dev/null | xargs kill 2>/dev/null || true
  wait 2>/dev/null || true
  echo "Stopped."
}
trap cleanup EXIT INT TERM

.venv/bin/uvicorn backend.main:app --port $API_PORT &
API_PID=$!

( cd frontend && npm run dev -- --port $UI_PORT ) &
UI_PID=$!

# Wait for the API to finish loading models (it warms up SHAP during startup).
printf "Waiting for the prediction server"
for _ in $(seq 1 60); do
  if curl -sf "http://127.0.0.1:$API_PORT/api/health" >/dev/null 2>&1; then break; fi
  printf "."
  sleep 1
done
echo
curl -sf "http://127.0.0.1:$API_PORT/api/health" >/dev/null || fail "The prediction server did not start. See the log above."

echo
echo "CardioLens is running."
echo "  App:       http://localhost:$UI_PORT"
echo "  API docs:  http://127.0.0.1:$API_PORT/docs"
echo "Press Ctrl+C to stop both servers."
echo

if $OPEN_BROWSER; then
  if command -v open >/dev/null; then open "http://localhost:$UI_PORT"
  elif command -v xdg-open >/dev/null; then xdg-open "http://localhost:$UI_PORT"
  fi
fi

wait
