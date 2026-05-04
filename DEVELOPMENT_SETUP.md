# Development Setup Guide

Complete guide to set up the BESSER development environment from scratch.

---

## Prerequisites

### 1. Python 3.10, 3.11, or 3.12

> **Python 3.13 is NOT supported** (some dependencies lack compatible wheels).

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

---

## Clone the Repository

```bash
git clone https://github.com/BESSER-PEARL/BESSER.git
cd BESSER
git submodule update --init --recursive
```

---

## Start the Backend (Terminal 1)

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
pip install -e .
python besser/utilities/web_modeling_editor/backend/backend.py
```

The backend runs on `http://localhost:9000/besser_api`.

---

## Start the Frontend (Terminal 2)

```bash
cd besser/utilities/web_modeling_editor/frontend
npm install
npm run dev
```

The frontend runs on `http://localhost:8080`.

Open your browser and go to **http://localhost:8080**.

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
