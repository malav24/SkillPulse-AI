#!/usr/bin/env bash
# SkillPulse AI - start backend + frontend (macOS / Linux). Ctrl+C stops both.
cd "$(dirname "$0")"
(cd backend && source .venv/bin/activate && uvicorn app.main:app --reload --port 8000) &
BACK=$!
(cd frontend && npm run dev) &
FRONT=$!
trap 'kill $BACK $FRONT 2>/dev/null' INT TERM EXIT
sleep 8
(command -v open >/dev/null && open http://localhost:5173) || (command -v xdg-open >/dev/null && xdg-open http://localhost:5173) || echo "Open http://localhost:5173"
wait
