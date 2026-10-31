#!/usr/bin/env bash
# SkillPulse AI - one-time setup (macOS / Linux). Run:  bash setup.sh
set -e
cd "$(dirname "$0")"
PY=""
for v in python3.12 python3.13 python3.11 python3; do
  if command -v "$v" >/dev/null && "$v" -c 'import sys; assert (3,11) <= sys.version_info[:2] <= (3,13)' 2>/dev/null; then PY="$v"; break; fi
done
[ -z "$PY" ] && { echo "Python 3.11-3.13 not found"; exit 1; }
command -v npm >/dev/null || { echo "Node.js (npm) not found: install Node 20 LTS"; exit 1; }
echo "Using $PY"
cd backend
[ -d .venv ] || "$PY" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip >/dev/null
pip install -r requirements-dev.txt
[ -f .env ] || cp .env.example .env
cd ../frontend
npm install
echo "Setup complete. Start the app with:  bash start.sh"
