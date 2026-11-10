#!/usr/bin/env bash
# One-time local setup: Python virtualenv for the backend + npm install for the frontend.
set -euo pipefail
cd "$(dirname "$0")"

echo "→ Backend (Python ≥ 3.10)"
python3 -m venv .venv
.venv/bin/pip install --upgrade pip >/dev/null
.venv/bin/pip install -r backend/requirements-dev.txt

echo "→ Frontend (Node ≥ 18)"
(cd frontend && npm install)

[ -f .env ] || cp .env.example .env

cat <<'MSG'

Setup complete. Start the app in two terminals:

  cd backend && ../.venv/bin/python main.py      # API on http://localhost:8000
  cd frontend && npm run dev                     # UI  on http://localhost:5173

Optional: add GROQ_API_KEY to .env for AI explanations (everything else works without it).
MSG
