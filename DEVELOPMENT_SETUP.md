# Development Setup Guide

Complete guide to set up the BESSER development environment from scratch.

---

## Prerequisites

### 1. Python 3.11 or newer

> `setup.cfg` requires Python >= 3.11, and CI tests 3.11 and 3.12. Python 3.13 installs and passes the test suite
> since `bocl` 1.0.3, but is not covered by CI. Older checkouts pinned `bocl==1.0.1`, which fails to install on 3.13
> (see `docs/source/troubleshooting.rst`).

- Download from [python.org](https://www.python.org/downloads/)
- During installation, check **"Add Python to PATH"** (Windows)
- Verify:
  ```bash
  python --version
  ```

### 2. Node.js 20+

- Download the **LTS** version from [nodejs.org](https://nodejs.org/) (includes npm)
- Verify:
  ```bash
  node --version
  npm --version
  ```

### 3. Git

- Download from [git-scm.com](https://git-scm.com/)
- Verify:
  ```bash
  git --version
  ```

### 4. Docker Desktop (optional)

Only needed if you want to run the full stack with `docker compose up`.

- Download from [docker.com](https://www.docker.com/products/docker-desktop/)

### 5. System packages (Linux)

On Debian or Ubuntu, install these before creating the virtual environment:

```bash
sudo apt-get install python3-venv unixodbc bubblewrap
```

- `python3-venv` (`python3.12-venv` on Ubuntu 24.04): without it, `python -m venv` fails with `ensurepip is not available`.
- `unixodbc`: `pyodbc` needs it at import time.
- `bubblewrap`: the Spec-Driven Agent's shell sandbox. CI installs it; without it the sandbox tests skip.

---

## Clone the Repository

```bash
git clone --recurse-submodules https://github.com/BESSER-PEARL/BESSER.git
cd BESSER
```

If you already cloned without `--recurse-submodules`, run `git submodule update --init --recursive`.

The frontend in `besser/utilities/web_modeling_editor/frontend` is a second repository,
[BESSER-Web-Modeling-Editor](https://github.com/BESSER-PEARL/BESSER-Web-Modeling-Editor), included as a git submodule.
It is checked out at a fixed commit (a detached HEAD), not on a branch. A UI change is a branch and pull request in
that repository, plus a submodule-pointer update here: see
[Coordinating Changes Across BESSER and WME](https://besser.readthedocs.io/en/latest/contributor_guide.html#coordinating-changes-across-besser-and-wme).

---

## Start the Backend (Terminal 1)

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
pip install -e .
pip install -r besser/utilities/web_modeling_editor/backend/requirements.txt
besser-wme-backend
```

Keep the virtual environment activated in every terminal you use. `besser-wme-backend`, `pytest` and the
`django-admin` command the Django generator calls are installed into it. Without it, they are not on `PATH`, and the
Django generator tests skip.

`besser-wme-backend` (installed by `pip install -e .` or `pip install besser`)
accepts `--host`, `--port` and `--log-level`; run it with `--help` for details.
`python besser/utilities/web_modeling_editor/backend/backend.py` starts the
same server.

The backend runs on `http://localhost:9000/besser_api`. To use the
Spec-Driven Agent's Local (e.g. Ollama) provider, set
`BESSER_LLM_ALLOW_CUSTOM_BASE_URL=true` before starting it (the local
`docker-compose.yml` does this for you).

At startup the backend warns about unset optional variables such as `GITHUB_CLIENT_ID`, `GITHUB_CLIENT_SECRET` and
`OPENAI_API_KEY`. They are only needed for GitHub sign-in and deploy and for the OpenAI-based features; drawing models
and generating code work without them. See `.env.example`.

---

## Start the Frontend (Terminal 2)

```bash
cd besser/utilities/web_modeling_editor/frontend
npm ci
npm run dev
```

`npm ci` installs the exact versions in the submodule's `package-lock.json`. Run it from the submodule root, as above.

The frontend runs on `http://localhost:8080`.

Open your browser and go to **http://localhost:8080**. The editor calls the backend at `http://localhost:9000/besser_api`,
and the backend's CORS allow-list expects the `localhost` origin. If you open a different host name (for example your
machine's IP address), add it to `CORS_ORIGINS` before starting the backend.

---

## Full Stack with Docker (alternative)

Instead of running backend and frontend separately:

```bash
docker compose up --build
```

This starts both services. Open `http://localhost:8080`.

---

## Recommended Tools

### VS Code

- Download from [code.visualstudio.com](https://code.visualstudio.com/)
- During installation, check **"Add to PATH"** and **"Open with Code" context menu options** (Windows)

### Claude Code

[Claude Code](https://docs.anthropic.com/en/docs/claude-code) is an AI coding assistant that runs in your terminal. See the [official documentation](https://docs.anthropic.com/en/docs/claude-code/overview) for installation and setup.

From the BESSER repository root:

```bash
claude
```

Claude Code will read the `CLAUDE.md` file for project context and help you with code, documentation, testing, and debugging.

---

## Verify Everything Works

1. Backend is running → `http://localhost:9000/besser_api` returns JSON
2. Frontend is running → `http://localhost:8080` shows the editor
3. Create a new project, add a class diagram, click **Generate > Python Classes** → a `.py` file downloads
