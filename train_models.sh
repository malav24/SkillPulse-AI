#!/usr/bin/env bash
# SkillPulse AI - retrain all 3 ML models (macOS / Linux). About 10 minutes.
set -e
cd "$(dirname "$0")/backend"
source .venv/bin/activate
[ -d artifacts ] && rm -rf artifacts_backup && cp -r artifacts artifacts_backup
python -m ml.train_all
echo "Done. Restart the backend to use the new models."
