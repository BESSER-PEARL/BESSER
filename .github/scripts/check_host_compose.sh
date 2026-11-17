#!/usr/bin/env bash
# Check, before anything is built or pushed, that the deploy host's compose
# file defines the services about to be deployed. A missing service would fail
# `docker compose pull` halfway through, and a worker still on the backend
# image would lose its toolchains and bubblewrap without any error.
#
# Usage: SERVICES="<service> ..." REGISTRY=<image prefix> check_host_compose.sh
# `ssh wme` must reach the host.
set -euo pipefail

runbook="docs/source/spec_driven_agent/production_deployment.rst"
unchanged="Nothing was built, pushed or restarted."

if ! defined="$(ssh wme "docker compose config --services")"; then
  echo "::error::Could not read the host's compose file (docker compose config failed). $unchanged"
  exit 1
fi

missing=""
for service in $SERVICES; do
  grep -qx "$service" <<< "$defined" || missing="$missing $service"
done
if [[ -n "$missing" ]]; then
  echo "::error::The host's compose file does not define:$missing. Add them first (besser-wme-smartgen: copy it from docker-compose.prod.yml, see $runbook). $unchanged"
  exit 1
fi

if [[ " $SERVICES " == *" besser-wme-smartgen "* ]]; then
  want="$REGISTRY/smartgen_worker:latest"
  image="$(ssh wme "docker compose config --images besser-wme-smartgen")" \
    || image="unknown (docker compose config --images failed)"
  if [[ "$image" != "$want" ]]; then
    echo "::error::besser-wme-smartgen uses '$image' on the host, not '$want'. The backend image no longer carries the worker's toolchains or bubblewrap; point the service at $want (see $runbook). $unchanged"
    exit 1
  fi
fi
echo "The host defines: $SERVICES"
