Production deployment
=====================

A hosted editor runs the Spec-Driven Agent in its own container,
``besser-wme-smartgen``, next to the regular backend. Generation is the only
path that executes LLM-authored code, so the worker gets the compilers, the
shell sandbox and the LLM credentials, and nothing else; the backend keeps
every other secret and has no compilers at all. This page lists what a
production host needs for that split. The reference definitions are in
``docker-compose.prod.yml`` and ``.github/workflows/deploy-wme.yml``.

Images
------

Both images are built from the root ``Dockerfile``, one target each, and
stamped with the commit they were built from (``BESSER_BUILD_SHA``).

.. list-table::
   :header-rows: 1
   :widths: 22 26 52

   * - Service
     - Image (``docker build --target``)
     - Contents
   * - ``besser-wme-backend``
     - ``artefacts.list.lu/besser/web_modeling_editor/backend:latest``
       (``backend``)
     - Python 3.12, the root and backend requirements (including ``ruff``) and
       ``besser``. No Node.js, JDK, Rust, Kotlin or bubblewrap.
   * - ``besser-wme-smartgen``
     - ``artefacts.list.lu/besser/web_modeling_editor/smartgen_worker:latest``
       (``smartgen-worker``)
     - Everything in the backend image, plus Node.js 20 with ``tsc``, JDK 21,
       Rust (``cargo``), ``kotlinc``, ``build-essential`` and bubblewrap.

The ``Dockerfile`` requires BuildKit (the default builder in current Docker
releases; set ``DOCKER_BUILDKIT=1`` on older ones): the proxy certificates are
read through a ``RUN --mount=type=bind`` of ``ca-certs-extra/``, so they never
get an image layer of their own. The directory ships with a ``.gitkeep`` and
must stay in the build context.

Both targets trust a build-time proxy CA only while building: put its
certificates (``*.crt``) in ``ca-certs-extra/`` and pass ``--build-arg
TRUST_EXTRA_CAS=1``. Before the image is finished the CA is removed again, and
the build fails if a known TLS-inspection CA is still in the system trust
store or, on the worker, in the JDK keystore, which is rebuilt from the
cleaned store for that reason.

.. warning::

   The worker must use the ``smartgen_worker`` image. Left on the ``backend``
   image, it starts normally but has no compilers and no bubblewrap: every
   ``run_command`` is refused and Phase 3 cannot compile TypeScript, Rust or
   Kotlin output. The deploy workflow refuses to deploy in that state.

The worker service
------------------

Add this service to the host's compose file, unchanged apart from the image
registry if yours differs. It is the ``besser-wme-smartgen`` block of
``docker-compose.prod.yml``:

.. literalinclude:: ../../../docker-compose.prod.yml
   :language: yaml
   :start-at: besser-wme-smartgen:
   :end-before: # Agent simulator:

and, at the top level, its volume and network:

.. code-block:: yaml

   volumes:
     smartgen_workspace:
       driver: local

   networks:
     smartgen_network:
       name: smartgen_network
       driver: bridge

The backend mounts the same volume read-only (``smartgen_workspace:/workspace:ro``)
so that push-to-GitHub and import-from-GitHub, which need the backend's OAuth
session, can read the generated files.

Environment
-----------

The backend loads the whole ``./.env`` (``env_file``). The worker has no
``env_file``: it receives only the variables listed in its ``environment``
block, whose values compose substitutes from the same ``./.env``.

**Set in ./.env** (read by the worker through substitution): the LLM
credentials ``BESSER_FREE_LLM_BASE_URL`` / ``_TOKEN`` / ``_MODEL`` /
``_ALT_MODELS``, the ``BESSER_FREE_LLM_FALLBACK_*`` and
``BESSER_SPONSORED_LLM_*`` triples, ``BESSER_DEMO_TOKEN``,
``BESSER_PILOT_LLM_MODEL``, ``BESSER_TELEMETRY_ENABLED`` and, optionally, the
budget overrides (``BESSER_LLM_DEFAULT_MAX_COST_USD``,
``BESSER_LLM_DEFAULT_MAX_RUNTIME_SECONDS``,
``BESSER_LLM_MAX_RUNTIME_SECONDS_HARD_CAP``, ``BESSER_LLM_MAX_CONCURRENT_RUNS``).

**Never put in ./.env**, because the backend would pick them up:

- ``BESSER_LLM_ENABLE_SHELL_TOOLS`` -- shell tools belong to the worker only,
  which holds nothing but the LLM tokens. On the backend they would run
  model-authored commands next to the SMTP, GitHub OAuth and telemetry
  secrets. See :ref:`spec-driven-shell-tools`.
- ``BESSER_LLM_ENABLE_TOOLCHAIN_VALIDATION`` -- the backend image has no
  compilers to run.
- ``BESSER_LLM_RUN_WORKSPACE_ROOT`` and ``BESSER_LLM_RUN_STORE_PATH`` -- the
  backend's ``/workspace`` is read-only, so runs created under it fail.
- ``BESSER_LLM_ALLOW_CUSTOM_BASE_URL=true`` -- keep it off on every hosted
  service (SSRF); the worker sets it to ``false`` explicitly.
- ``BESSER_LLM_SHELL_SANDBOX=off`` -- a development opt-out that runs
  model-authored commands unconfined.

Sandbox and network
-------------------

- ``run_command`` wraps every model-authored command in bubblewrap, which
  needs an unprivileged user namespace. The worker carries four
  ``security_opt`` entries and no added capability:

  - ``seccomp=unconfined`` -- Docker's default seccomp profile blocks the
    namespace and mount calls bubblewrap makes.
  - ``apparmor=unconfined`` -- the default AppArmor profile denies ``mount``.
  - ``systempaths=unconfined`` -- Docker masks parts of ``/proc``, and on
    stock kernels such as Amazon Linux's the kernel then refuses the private
    ``/proc`` that bubblewrap mounts for each command.
  - ``no-new-privileges:true`` -- no process in the container can gain
    privileges through a setuid binary.

  Without the first three the worker fails closed and refuses every command,
  as it does on a kernel that forbids unprivileged user namespaces. The
  trade-off is plain: ``systempaths=unconfined`` unmasks ``/proc`` for the
  worker's own root process. Generated code never runs as that process; it
  runs inside bubblewrap's user namespace. Running the worker as a non-root
  user would narrow the exposure further.
- A run's commands share one long-lived sandbox, its shell session, so a
  server the agent starts in one command is still up for the next. It is torn
  down when the run ends in any way, when its folder is deleted, and when the
  worker exits (the session's control channel is a pipe to the worker
  process). Two settings bound what sessions hold:
  ``BESSER_LLM_SHELL_SESSION_IDLE_SECONDS`` (900) closes a session that has
  had no command for that long, and ``BESSER_LLM_SHELL_SESSION_MAX`` (10)
  caps live sessions per worker; past it a command gets a one-off sandbox as
  before. A session costs a bubblewrap process and a small Python supervisor
  plus whatever the run left running, so size the worker's memory for
  ``BESSER_LLM_SHELL_SESSION_MAX`` running apps rather than for one command
  at a time.
- The validators that execute generated code run in sandboxes of the same
  kind, each fresh and never the run's shell session, with
  the network cut: the import check after each file write, the import smoke
  check, the startup, create and API probes, and the ``tsc``, ``cargo check``
  and ``npm run build`` checks. When the worker cannot start the sandbox they
  are skipped and reported as unverified, never run unconfined. Tool results
  and findings then say only that the shell sandbox is unavailable; look in
  the worker's log for the cause.
- Two steps need the network and get it, but are sandboxed all the same, so
  package install scripts see read-only ``/usr/local``, no other runs and a
  stripped environment: the Phase 1 ``npm install`` of a scaffolded frontend
  and the ``pip install --dry-run`` dependency check. When the sandbox cannot
  start, the install is skipped with a logged reason, and the frontend build
  check then reports the frontend's dependencies as not installed.
- Each run gets its own writable ``$HOME`` beside its workspace
  (``<run dir>.sandbox-home``), where installs and the npm / cargo caches go;
  ``/usr/local`` and ``/root`` are read-only. The telemetry folder and the
  incident log folder (``BESSER_INCIDENT_LOG_DIR``, default
  ``/app/incidents``, which names other runs) are masked from every run.
  Deleting a run removes its sandbox ``$HOME`` along with its workspace.
- ``cargo check`` has no network, so only crates that the run already fetched
  through ``run_command`` resolve. Otherwise the check is reported as
  "dependencies could not be fetched", a skipped check rather than a compile
  failure.
- The worker is only on ``smartgen_network``, not ``besser_network``: it
  cannot reach the backend, the frontend or the modeling agent by name, and
  still has outbound internet for the LLM endpoints.
- It publishes ``127.0.0.1:9001`` only, so the host's reverse proxy must
  route ``/besser_api/spec-driven/`` to it, except
  ``/besser_api/spec-driven/push-to-github`` and
  ``/besser_api/spec-driven/import-github-run``, which stay on the backend
  with the rest of ``/besser_api/``.

Deploying
---------

The *Deploy WME to EC2* workflow (``target: backend`` or ``all``) builds and
pushes both images, then recreates ``besser-wme-backend`` and
``besser-wme-smartgen``. Before it builds anything, it reads the host's
compose file and stops, with nothing pushed or restarted, when a service it
would deploy is not defined there or when ``besser-wme-smartgen`` is not on the
``smartgen_worker`` image. After the restart it compares ``BESSER_BUILD_SHA``
in both containers with the commit it built, then checks that the API answers
``200``.

Verification
------------

Run these on the host, in the directory of its compose file:

.. code-block:: bash

   # Both containers run the commit that was deployed
   docker compose exec -T besser-wme-backend printenv BESSER_BUILD_SHA
   docker compose exec -T besser-wme-smartgen printenv BESSER_BUILD_SHA

   # The worker has its toolchains, and bubblewrap can start
   docker compose exec -T besser-wme-smartgen sh -c \
     'node --version && tsc -v && cargo --version && kotlinc -version && bwrap --version && ruff --version'
   docker compose exec -T besser-wme-smartgen python -c \
     "from besser.spec_driven_agent.execution.sandbox import sandbox_selftest_error as e; print(e() or 'sandbox OK')"

   # Shell tools and toolchain validation are on in the worker ...
   curl -s http://127.0.0.1:9001/besser_api/spec-driven/config | python3 -m json.tool | grep -E 'shell_tools_enabled|toolchain_validation_enabled'
   # ... and off in the backend
   docker compose exec -T besser-wme-backend python -c \
     "import besser.utilities.web_modeling_editor.backend.constants.constants as c; print(c.LLM_ENABLE_SHELL_TOOLS, c.LLM_ENABLE_TOOLCHAIN_VALIDATION)"

   # The API answers through the public URL
   curl -s -o /dev/null -w '%{http_code}\n' https://<editor-host>/besser_api/

Expected: the same commit twice, the tool versions, ``sandbox OK``, both
features ``true`` in the worker, ``False False`` in the backend, and ``200``.
