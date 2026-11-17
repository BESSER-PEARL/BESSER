#!/usr/bin/env bash
# Wait until the public editor serves what this run pushed. The editor host is
# only reachable over the LIST VPN, so nothing here logs in to it: watchtower
# on the host pulls new :latest images every 30 minutes and recreates the
# containers.
#
# Usage: TARGET=<all|backend|frontend|agent|agent_simulator> EDITOR_URL=<url>
#        GITHUB_SHA=<commit> [FRONTEND_ENTRY=assets/index-*.js] wait_for_live.sh
#
# Checks: /besser_api/ reports the commit just built (backend), and / links
# the entry bundle of the image just built (frontend). The worker, agent and
# agent simulator have no public build stamp. The workflow pushes them before
# the backend, so the backend stamp shows a watchtower pass that came after them.
set -euo pipefail

wait_minutes="${WAIT_MINUTES:-45}"
deadline=$(( $(date +%s) + wait_minutes * 60 ))
check_backend=false
check_frontend=false
case "$TARGET" in
  all)      check_backend=true; check_frontend=true ;;
  backend)  check_backend=true ;;
  frontend) check_frontend=true ;;
esac

backend_ok() {
  local served
  served="$(curl -fsS --max-time 15 "$EDITOR_URL/besser_api/" | jq -r '.besser_build // empty')" || return 1
  echo "backend: serves ${served:-nothing}, want $GITHUB_SHA"
  [[ "$served" == "$GITHUB_SHA" ]]
}

frontend_ok() {
  local served
  served="$(curl -fsS --max-time 15 "$EDITOR_URL/" | grep -oE 'assets/index-[A-Za-z0-9_-]+\.js' | head -1)" || return 1
  echo "frontend: serves ${served:-nothing}, want $FRONTEND_ENTRY"
  [[ "$served" == "$FRONTEND_ENTRY" ]]
}

while :; do
  done_all=true
  if $check_backend && ! backend_ok; then done_all=false; fi
  if $check_frontend && ! frontend_ok; then done_all=false; fi
  if $done_all; then break; fi
  if (( $(date +%s) > deadline )); then
    echo "::error::$EDITOR_URL does not serve this build after $wait_minutes min. Check watchtower on the host (docker logs watchtower), e.g. for a revoked registry token."
    exit 1
  fi
  sleep 60
done

code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 \
  -H 'Connection: Upgrade' -H 'Upgrade: websocket' -H 'Sec-WebSocket-Version: 13' \
  -H 'Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==' -H "Origin: $EDITOR_URL" "$EDITOR_URL/agent" || true)"
echo "agent WebSocket handshake: $code"
if [[ "$code" != "101" ]]; then
  echo "::error::The modeling agent WebSocket answered $code, not 101."
  exit 1
fi
if ! $check_backend && ! $check_frontend; then
  echo "::notice::The $TARGET image has no public build stamp. Watchtower installs it within 30 min; check it on the host if it matters."
fi
echo "$EDITOR_URL serves this build."
