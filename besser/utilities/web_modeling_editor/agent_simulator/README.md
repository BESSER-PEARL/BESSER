# BESSER Agent Simulator

Isolated execution service for user-designed BAF agents (live agent testing in
the web modeling editor). The backend delegates each simulation session to this
service, which runs the agent in its own bubblewrap sandbox, under its own Unix
UID, with resource limits.

The full reference (architecture, security model, every environment variable)
is in `docs/source/utilities/agent_simulator.rst`.

## Build

From the **repository root** (the image ships this checkout's `besser`):

```bash
docker build -f besser/utilities/web_modeling_editor/agent_simulator/Dockerfile \
  -t artefacts.list.lu/besser/web_modeling_editor/agent_simulator:latest .
```

or `docker compose build besser-wme-agent-simulator`.

## Run

Use the `besser-wme-agent-simulator` service in `docker-compose.yml` /
`docker-compose.prod.yml`; it carries every setting the sandbox needs. For a
standalone container the equivalent is:

```bash
docker network create agent_simulator_network
docker run -d --name besser-wme-agent-simulator \
  --network agent_simulator_network \
  --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
  --security-opt no-new-privileges:true \
  --cap-drop ALL --cap-add CHOWN --cap-add DAC_OVERRIDE \
  --cap-add SETUID --cap-add SETGID --cap-add KILL \
  --init --pids-limit 512 --memory 6g --cpus 2 \
  --tmpfs /tmp/sessions:size=512m,mode=0711 \
  -e AGENT_SIMULATOR_API_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')" \
  artefacts.list.lu/besser/web_modeling_editor/agent_simulator:latest
```

No capability is added. `seccomp=unconfined` and `apparmor=unconfined` are what
bubblewrap needs to create an unprivileged user namespace (Docker's default
seccomp profile blocks `unshare`/`mount`/`pivot_root`, and the docker-default
AppArmor profile denies `mount`). Without them the simulator fails closed and
refuses every session.

The backend must share `agent_simulator_network`, send the same token in the
`X-Agent-Simulator-Token` header, and point `AGENT_SIMULATOR_URL` at
`http://besser-wme-agent-simulator:8001`. Do not publish port 8001 on the host.

## Checks

```bash
# Sandbox self-test, probing as the first session uid
docker exec besser-wme-agent-simulator \
  python -m besser.utilities.web_modeling_editor.agent_simulator.sandbox 20000

# Health (needs the token)
docker exec besser-wme-agent-simulator sh -c \
  'python -c "import os,urllib.request as u; print(u.urlopen(u.Request(\"http://127.0.0.1:8001/health\", headers={\"X-Agent-Simulator-Token\": os.environ[\"AGENT_SIMULATOR_API_TOKEN\"]})).read())"'
```
