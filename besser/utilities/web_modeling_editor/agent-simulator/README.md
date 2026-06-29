# BESSER Agent Simulator

Isolated execution service for user-designed BAF agents. The BESSER web modeling editor backend delegates agent simulation sessions to this service, which runs each agent in a sandboxed subprocess with resource limits and UID isolation.

## Architecture

```
Browser  ──WS──►  Backend  ──WS──►  Agent Simulator  ──spawns──►  agent.py subprocess
                             HTTP                        per session
```

- The backend sends the agent code and configuration to the simulator via REST.
- The simulator spawns one Python subprocess per session, each isolated under a dedicated Unix UID with resource limits applied via `setrlimit`.
- WebSocket traffic between the browser and the running agent is relayed through both the backend and the simulator.
- The simulator is designed to run on an internal Docker network — it should **not** be exposed to the public internet.

## Requirements

- Docker
- The backend must be able to reach the simulator at `http://besser-agent-simulator:8001` (or the URL set via `AGENT_SIMULATOR_URL`)

## Build

Run from the `agent-simulator` directory:

```bash
docker build -t besser-agent-simulator .
```

## Run

### Standalone (for local development)

```bash
docker run -d \
  --name besser-agent-simulator \
  --cap-add SYS_ADMIN \
  -p 8001:8001 \
  --tmpfs /tmp/sessions:rw,size=512m \
  besser-agent-simulator
```

`--cap-add SYS_ADMIN` is required so the entrypoint can call `chown`/`chmod` on `/tmp/sessions` and so each session subprocess can drop privileges via `setuid`/`setgid`.

### On a shared Docker network (recommended for production)

```bash
# Create the network once
docker network create besser-net

docker run -d \
  --name besser-agent-simulator \
  --network besser-net \
  --cap-add SYS_ADMIN \
  --tmpfs /tmp/sessions:rw,size=512m \
  besser-agent-simulator
```

Then start the backend on the same network and set `AGENT_SIMULATOR_URL=http://besser-agent-simulator:8001`.

### Using an env file

Create a file (e.g. `simulator.env`) with the variables you want to override (see Configuration below), then:

```bash
docker run -d \
  --name besser-agent-simulator \
  --network besser-net \
  --cap-add SYS_ADMIN \
  --tmpfs /tmp/sessions:rw,size=512m \
  --env-file simulator.env \
  besser-agent-simulator
```

## Health check

```bash
curl http://localhost:8001/health
# {"status":"ok","sessions":0}
```

## Configuration

All parameters are set via environment variables. If a variable is not set, the default value is used.

### Concurrency and session lifecycle

| Variable | Default | Description |
|---|---|---|
| `AGENT_SIMULATOR_MAX_SESSIONS` | `5` | Maximum number of agent sessions that can run concurrently. Each session consumes one port and one UID from the pool. |
| `AGENT_SIMULATOR_SESSION_LIFETIME_SECONDS` | `900` | How long (in seconds) a session is allowed to live. Expired sessions are terminated automatically by the background cleanup task, which runs every 60 seconds. |
| `AGENT_SIMULATOR_PORT_POOL_START` | `7700` | First port in the pool assigned to agent WebSocket servers. Ports `START` through `START + MAX_SESSIONS - 1` are used. These ports are internal to the container and do not need to be published. |
| `AGENT_SIMULATOR_SESSION_UID_BASE` | `20000` | Base Unix UID for subprocess isolation. Each session runs under UID `BASE + n`. These UIDs must exist in the container (they are created automatically). |

### Per-session resource limits

These limits are applied to each agent subprocess via Linux `setrlimit`. They protect the host from runaway or malicious agents.

| Variable | Default | Description |
|---|---|---|
| `AGENT_SIMULATOR_RLIMIT_AS_GB` | `4` | Virtual memory limit in **gigabytes**. Must be generous — Python and ML libraries (numpy, PyTorch, openai…) map several GB of address space via shared libraries even when actual RAM usage is low. |
| `AGENT_SIMULATOR_RLIMIT_CPU_SEC` | `120` | Maximum **CPU time** in seconds the subprocess may consume before being killed by the OS. |
| `AGENT_SIMULATOR_RLIMIT_FSIZE_MB` | `100` | Maximum size of any single **file** the subprocess may write, in megabytes. |
| `AGENT_SIMULATOR_RLIMIT_NPROC` | `64` | Maximum number of **child processes** the subprocess may spawn. |
| `AGENT_SIMULATOR_RLIMIT_NOFILE` | `1024` | Maximum number of **open file descriptors**. Python's import machinery needs ~200; 1024 is a safe minimum. |

### Backend-side variables

These are read by the BESSER **backend** (`agent_simulator_router.py`), not by the simulator container itself. Set them in the backend's environment.

| Variable | Default | Description |
|---|---|---|
| `AGENT_SIMULATOR_URL` | `http://besser-agent-simulator:8001` | URL the backend uses to reach the simulator. |
| `AGENT_SIMULATOR_REQUIRE_AUTH` | `true` | When `true`, a valid GitHub session token is required to start a simulation. Set to `false` to disable auth (development only). |
| `AGENT_SIMULATOR_RATE_LIMIT_WINDOW_SECONDS` | `60` | Sliding window duration for the per-user rate limiter. |
| `AGENT_SIMULATOR_RATE_LIMIT_MAX_REQUESTS` | `12` | Maximum requests allowed per user within the rate limit window. |

## Scaling notes

If you increase `AGENT_SIMULATOR_MAX_SESSIONS`, also:

- Increase the `--tmpfs` size proportionally (each session uses `/tmp/sessions/<uuid>/`).
- Ensure the host has enough free ports starting from `AGENT_SIMULATOR_PORT_POOL_START`.
- Increase the `--cap-add` or memory limits on the container itself so it can support more concurrent subprocesses.
