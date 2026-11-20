#!/usr/bin/env bash
# The local BESSER Lab container (docker-compose.lab.yml, Dockerfile target "lab"), as the lab user.
#
# First start: lays out the workspace in the /srv/besser-lab volume (scripts/lab/setup_workspace.sh: the base
# clones, npm ci, the venvs; many minutes). Every start: the lab backend on :9000 and the lab editor's server on
# :8080. The container stops when either stops.
#
#   docker compose -f docker-compose.lab.yml run --rm besser-lab setup   # bring the workspace up to date, exit
set -euo pipefail

readonly VOLUME=/srv/besser-lab
# Written once the first setup has finished; a failed setup runs again at the next start.
readonly SETUP_DONE=$VOLUME/setup-done
readonly EDITOR_SERVER=/opt/besser-lab-editor/build/server

# The folders the lab hides from every sandbox must exist before it starts (a missing one cannot be masked).
mkdir -p "$VOLUME/state"
(umask 077 && mkdir -p "$(dirname "$OAUTH_SESSIONS_PATH")" "$(dirname "$USER_TOKENS_PATH")" \
  "$BESSER_LLM_RUN_WORKSPACE_ROOT")

if [[ ${1:-} == setup || ! -f $SETUP_DONE ]]; then
  LAB_PYTHON=python bash /app/scripts/lab/setup_workspace.sh
  touch "$SETUP_DONE"
fi
if [[ ${1:-} == setup ]]; then
  exit 0
fi

python -m besser.utilities.web_modeling_editor.backend.backend --host 0.0.0.0 --port 9000 &
backend=$!
# The editor's server needs none of the lab's keys: it gets a clean environment. It finds ../webapp from here.
(cd "$EDITOR_SERVER" && exec env -i PATH="$PATH" HOME="$HOME" DEPLOYMENT_URL="$DEPLOYMENT_URL" node bundle.js) &
editor=$!

trap 'kill -TERM "$backend" "$editor" 2>/dev/null || true' TERM INT
status=0
wait -n "$backend" "$editor" || status=$?
kill -TERM "$backend" "$editor" 2>/dev/null || true
wait || true
echo "besser-lab: a process stopped (exit $status); stopping the container" >&2
exit "$status"
