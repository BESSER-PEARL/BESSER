Agent Simulator
===============

The agent simulator runs user-designed BAF agents so they can be tried out live
from the web modeling editor. It is a separate service
(``besser/utilities/web_modeling_editor/agent_simulator/``) deployed as its own
container, ``besser-wme-agent-simulator``. Only the backend talks to it.

Architecture
------------

.. code-block:: text

   Browser --WS--> Backend --HTTP/WS--> Agent simulator --spawns--> bwrap -> python agent.py
                            X-Agent-Simulator-Token               (one per session)

* The backend generates the agent code and ``config.yaml`` and sends them to
  ``POST /sessions``.
* The simulator writes them into a per-session work directory
  (``/tmp/sessions/<session id>``, mode ``0700``, owned by the session UID) and
  starts ``python agent.py`` inside a bubblewrap sandbox.
* ``/sessions/{id}/ws`` relays messages between the backend and the agent's
  WebSocket server (bound to ``127.0.0.1`` inside the container), and forwards
  the agent's stdout as ``{"type": "stdout"}`` / ``{"type": "state_change"}``
  events.
* ``GET /sessions/{id}/files`` lists the regular files the agent wrote.
  Symlinks, FIFOs and other non-regular entries are never followed nor
  returned.
* ``DELETE /sessions/{id}`` (or expiry after
  ``AGENT_SIMULATOR_SESSION_LIFETIME_SECONDS``) kills the session.

Security model
--------------

The simulator executes code the user controls, so it is confined in layers.
The design mirrors the Spec-Driven Agent's ``run_command`` sandbox.

API authentication
   Every HTTP and WebSocket endpoint, ``/health`` included, requires the header
   ``X-Agent-Simulator-Token`` to equal ``AGENT_SIMULATOR_API_TOKEN``
   (constant-time comparison). WebSocket handshakes without it are closed with
   code ``1008``. While the variable is unset, every request is refused with
   ``503`` (fail closed).

Per-session sandbox (bubblewrap)
   Each session runs with its own user, PID, IPC, UTS and cgroup namespaces,
   ``--die-with-parent`` and ``--new-session``. The container filesystem is
   bound read-only; the top-level directory holding the sessions root is not
   bound at all and only this session's work directory is bound back
   read-write, so sibling sessions are invisible. ``/proc`` and ``/dev`` are
   private and ``/tmp``, ``/run``, ``/var/tmp`` and ``/dev/shm`` are fresh
   tmpfs. Because PID 1 inside the sandbox is bwrap's own init,
   ``/proc/1/environ`` shows the sandbox's scrubbed environment, never the
   simulator's (which holds the API token).

   If ``bwrap`` is missing or its self-test fails, sessions are refused with
   ``503``. ``AGENT_SIMULATOR_SANDBOX=off`` runs sessions unconfined; it logs a
   loud warning and is meant for single-tenant development hosts only.

Per-session UID and resource limits
   Each session runs under its own UID from a pool
   (``AGENT_SIMULATOR_SESSION_UID_BASE`` + n), with no supplementary groups and
   umask ``077``. A UID of its own keeps ``RLIMIT_NPROC`` accounting per
   session, keeps sibling work directories unreadable even with the sandbox
   off, and gives a complete kill list. ``setrlimit`` limits (address space,
   CPU time, file size, process count, open files) are applied before the
   process starts; if they cannot be applied, the session fails to start.

Environment scrub
   The agent gets an allowlisted environment (``PATH`` and locale variables,
   minus anything whose name looks like a secret) plus only the session
   settings and the LLM keys the user supplied for this session
   (``OPENAI_API_KEY``, ``HUGGINGFACEHUB_API_TOKEN``, ``REPLICATE_API_TOKEN``).

Process cleanup
   Each session starts as its own session and process group. Terminating it
   signals the whole group (``SIGTERM``, then ``SIGKILL``). Killing bwrap tears
   down the sandbox's PID namespace, which reaches children that double-forked.
   Every remaining process owned by the session UID is then killed. The port
   and UID go back to the pools only when none is left; otherwise they stay
   quarantined and an error is logged.

Container hardening
   The compose service has no ``env_file`` (only the variables listed below are
   passed), sits on its own ``agent_simulator_network`` (the backend joins it;
   the frontend and modeling agent do not), publishes no host port, and runs
   with ``cap_drop: ALL`` plus ``CHOWN``, ``DAC_OVERRIDE``, ``SETUID``,
   ``SETGID`` and ``KILL`` (all within Docker's default set; nothing is added),
   ``no-new-privileges``, ``init: true``, ``pids_limit``, ``mem_limit``,
   ``cpus``, a tmpfs sessions root and a healthcheck. The container runs as
   root only so it can switch each session to its UID; agent code never runs
   as root. ``security_opt: seccomp=unconfined, apparmor=unconfined`` is
   required by bubblewrap (see ``sandbox.py``).

Residual risks (accepted)
   The network namespace is not unshared: agents need outbound access to the
   LLM providers, and the simulator relays to the agent's loopback port. So
   agent code can reach the internet, the simulator API port (useless without
   the token), the other sessions' agent WebSocket ports on ``127.0.0.1``
   (``AGENT_SIMULATOR_PORT_POOL_START`` and up), and the backend over
   ``agent_simulator_network``, which is the same API any internet user can
   reach.

Configuration
-------------

Simulator container (passed explicitly in the compose files):

.. list-table::
   :header-rows: 1
   :widths: 40 15 45

   * - Variable
     - Default
     - Description
   * - ``AGENT_SIMULATOR_API_TOKEN``
     - *(unset)*
     - Shared secret, required in ``X-Agent-Simulator-Token``. Unset means every
       request is refused.
   * - ``AGENT_SIMULATOR_SANDBOX``
     - ``auto``
     - ``auto``: bubblewrap is mandatory. ``off``: run unconfined, for
       single-tenant development only.
   * - ``AGENT_SIMULATOR_MAX_SESSIONS``
     - ``5``
     - Concurrent sessions; sizes the port and UID pools.
   * - ``AGENT_SIMULATOR_SESSION_LIFETIME_SECONDS``
     - ``900``
     - Sessions older than this are terminated by the cleanup task (every 60 s).
   * - ``AGENT_SIMULATOR_PORT_POOL_START``
     - ``7700``
     - First loopback port for the agents' WebSocket servers. Never published.
   * - ``AGENT_SIMULATOR_SESSION_UID_BASE``
     - ``20000``
     - First session UID. The UIDs need no ``/etc/passwd`` entry.
   * - ``AGENT_SIMULATOR_RLIMIT_AS_GB``
     - ``4``
     - Address-space limit per session (Python and ML libraries map several GB).
   * - ``AGENT_SIMULATOR_RLIMIT_CPU_SEC``
     - ``120``
     - CPU-time limit per session process.
   * - ``AGENT_SIMULATOR_RLIMIT_FSIZE_MB``
     - ``100``
     - Largest file a session may write.
   * - ``AGENT_SIMULATOR_RLIMIT_NPROC``
     - ``64``
     - Processes and threads per session UID.
   * - ``AGENT_SIMULATOR_RLIMIT_NOFILE``
     - ``1024``
     - Open file descriptors per session process.

Compose-level ceilings for the simulator container. The backend also reports
these values to the editor:

.. list-table::
   :header-rows: 1
   :widths: 40 15 45

   * - Variable
     - Default
     - Description
   * - ``AGENT_SIMULATOR_MEMORY_MB``
     - ``6144``
     - Container ``mem_limit``.
   * - ``AGENT_SIMULATOR_CPU_CORES``
     - ``2``
     - Container ``cpus``.
   * - ``AGENT_SIMULATOR_DISK_MB``
     - ``512``
     - Size of the ``/tmp/sessions`` tmpfs.

``pids_limit`` is fixed at ``512`` (``MAX_SESSIONS x RLIMIT_NPROC`` plus the
API process). Raise it together with those two.

Backend (read from ``.env``):

.. list-table::
   :header-rows: 1
   :widths: 40 25 35

   * - Variable
     - Default
     - Description
   * - ``AGENT_SIMULATOR_URL``
     - ``http://besser-wme-agent-simulator:8001`` in compose
     - Where the backend reaches the simulator.
   * - ``AGENT_SIMULATOR_API_TOKEN``
     - *(unset)*
     - Same value as on the simulator; sent on every request and WS handshake.
   * - ``AGENT_SIMULATOR_REQUIRE_AUTH``
     - ``true``
     - Require a GitHub session to start a simulation.
   * - ``AGENT_SIMULATOR_RATE_LIMIT_WINDOW_SECONDS``
     - ``60``
     - Per-user rate-limit window.
   * - ``AGENT_SIMULATOR_RATE_LIMIT_MAX_REQUESTS``
     - ``12``
     - Requests per user per window.
   * - ``AGENT_SIMULATOR_RATE_LIMIT_MAX_KEYS``
     - ``10000``
     - Most actors the in-process rate limiter tracks at once.
   * - ``AGENT_SIMULATOR_MAX_SESSIONS_PER_ACTOR``
     - ``1``
     - Concurrent simulation sessions per user.
   * - ``AGENT_SIMULATOR_RESTRICT_CUSTOM_CODE``
     - ``true``
     - Refuse to simulate agents that contain custom Python code actions.
   * - ``AGENT_SIMULATOR_SESSION_LIFETIME_SECONDS``
     - ``900``
     - Same variable as on the simulator; the backend forgets session
       ownership after this delay and reports it to the editor.
   * - ``AGENT_SIMULATOR_MEMORY_MB`` / ``_CPU_CORES`` / ``_DISK_MB``
     - *(unset)*
     - Limits reported to the editor. Set them to the compose values above.
   * - ``AGENT_SIMULATOR_QUOTA_ENABLED``
     - ``false``
     - Tell the editor that quotas apply.

Operations
----------

Build the image from the repository root, so it ships this checkout's
``besser`` package. The GitHub ``Deploy WME to EC2`` workflow has an
``agent_simulator`` target that builds, pushes and restarts it:

.. code-block:: bash

   docker compose build besser-wme-agent-simulator
   # or
   docker build -f besser/utilities/web_modeling_editor/agent_simulator/Dockerfile \
     -t artefacts.list.lu/besser/web_modeling_editor/agent_simulator:latest .

After a deploy, check that the sandbox works:

.. code-block:: bash

   docker exec besser-wme-agent-simulator \
     python -m besser.utilities.web_modeling_editor.agent_simulator.sandbox 20000
   # prints "sandbox OK", or the reason sessions will be refused

``/health`` (with the token) also reports ``"sandbox": "ok"`` or the reason it
is unavailable, and ``"status": "degraded"`` in that case.
