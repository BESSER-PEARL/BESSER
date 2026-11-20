BESSER Lab in local mode
========================

Local mode runs a whole :doc:`BESSER Lab <lab>` in one Docker container on your
own machine, for one person or to try the lab before a host exists: the lab
backend, the lab's own editor (built with ``LAB_BUILD=true``) and one preview
slot, with the same sandbox as on a host. A team instance belongs on a host
(``scripts/lab/README.md``).

What it needs
-------------

- **Docker with a Linux kernel**: Linux, or Docker Desktop with the WSL2
  backend on Windows. The container runs bubblewrap and pasta, so it gets an
  unprivileged user namespace (``seccomp``, ``apparmor`` and ``systempaths``
  set to ``unconfined``, no capability added, ``no-new-privileges``) and
  ``/dev/net/tun``.
- **Disk.** The first start clones BESSER, the web editor and the modeling
  agent into the ``besser-lab`` volume and installs their environments: many
  minutes and several GB, plus the image. Check free disk first; on Windows
  the WSL2 disk image does not shrink by itself when the space is freed.
- **A GitHub OAuth app of your own**, with Homepage URL
  ``http://localhost:8080`` and Authorization callback URL
  ``http://localhost:9000/besser_api/github/auth/callback``.
- **An API key** of the lab agent's provider (Anthropic by default). Every run
  is paid with it.

Native Windows, without Docker, has no sandbox: the lab never runs
model-written code unconfined on Linux, and on Windows it would. Use native
Windows for the lab's unit tests only.

The files
---------

All in the BESSER repository's root:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - File
     - What it is
   * - ``docker-compose.lab.yml``
     - One service, ``besser-lab``, built from the ``lab`` target of the root
       ``Dockerfile`` (the editor submodule is passed as an extra build
       context). It sets the layout and addresses of local mode: the lab on,
       the workspace, git root and state in the ``besser-lab`` volume
       (``/srv/besser-lab``), one preview slot on localhost ports, the shell's
       private network, ``DEPLOYMENT_URL=http://localhost:8080`` and the OAuth
       callback on port 9000.
   * - ``.env.lab.example``
     - The settings you fill in, copied to ``.env.lab`` (git-ignored): allowed
       logins and admins, the lab agent's key, provider and models, the OAuth
       app, the repositories (``BESSER_LAB_REPO_*``: point them at forks to
       try the lab without touching BESSER-PEARL), the cost caps, the preview
       assistant's key and, behind a proxy, ``HTTPS_PROXY``/``NO_PROXY``.
   * - ``docker/lab/entrypoint.sh``
     - The container's start: on the first start (or with ``setup``) it lays
       out the workspace with ``scripts/lab/setup_workspace.sh``; then it runs
       the lab backend on port 9000 and the editor's server on port 8080, and
       stops the container when either stops.

Every port is published on ``127.0.0.1`` only:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Port
     - Serves
   * - 8080
     - The lab's editor: open ``http://localhost:8080``.
   * - 9000
     - The lab backend (``/besser_api``), and the OAuth callback.
   * - 8180
     - The preview's editor (Vite).
   * - 9100
     - The preview's backend.
   * - 8865
     - The preview's modeling agent.

First start
-----------

From the repository root, with the editor submodule checked out:

.. code-block:: bash

   cp .env.lab.example .env.lab        # fill in the values marked "Required"; never commit .env.lab
   docker compose -f docker-compose.lab.yml up --build

The first start builds the image, then clones the three repositories of
``BESSER_LAB_REPO_*`` into the volume, with their git directories outside the
working trees, and makes the base environments (BESSER's venv, the editor's
``npm ci``, the modeling agent's venv). A setup that fails runs again at the
next start. When the backend is up, open ``http://localhost:8080``, sign in
with GitHub with a login listed in ``BESSER_LAB_ALLOWED_LOGINS``, and describe
a change on the home page.

To bring the base clones and their environments up to date, with the lab
stopped:

.. code-block:: bash

   docker compose -f docker-compose.lab.yml run --rm besser-lab setup

Open changes never move; a new change starts from the refreshed bases (the
lab also refreshes them every ``BESSER_LAB_BASE_REFRESH_MINUTES`` and when a
change opens).

What works over http
--------------------

Everything except the preview frame. Local mode has no ``https`` origin, and
the lab's editor frames only ``https:`` pages, so a change's **Try it** tab
opens the preview (``http://localhost:8180``) in a new tab instead of showing
it in the change view. The preview's state, restart and logs stay in the tab.

The preview runs without the assistant until the modeling agent declares
``BESSER_AGENT_WS_PORT`` (see :doc:`lab`, "Try it: the preview"); and without
``BESSER_LAB_PREVIEW_OPENAI_API_KEY`` the preview assistant has no model access
in any case.

There is one slot, so a second change you try takes it over once the first
has been idle for 10 minutes.

Where things are
----------------

The ``besser-lab`` volume holds the workspace (``/srv/besser-lab/work``), the
git root (``/srv/besser-lab/git``), the lab's state
(``/srv/besser-lab/state/lab``: changes, threads, audit log, team notes,
preview logs), the encrypted GitHub sessions and the run store. Removing the
volume (``docker compose -f docker-compose.lab.yml down -v``) deletes all of
it, work not yet pushed included.
