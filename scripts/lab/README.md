# BESSER Lab: setting up the server

BESSER Lab is a private instance of the BESSER web editor in which the BESSER team develops BESSER itself: a
change goes from a request or a GitHub issue to pull requests on BESSER, the web editor and the modeling agent,
and can be tried live in the browser on the way. This folder sets one up on an empty Linux host. For one machine
only, without a host, see [Local mode](#local-mode).

| File | What it is |
|---|---|
| `provision.sh` | `install` sets the host up from zero, `update` brings it to the latest code. Run as root. |
| `setup_workspace.sh` | Lays out the lab's workspace: base clones, their environments, shared stores. Run by `provision.sh` and by the local container. |
| `besser-lab.service` | systemd unit of the lab backend. |
| `besser-lab-editor.service` | systemd unit of the lab's own editor (the editor's Express server). |
| `lab.env.example` | The lab backend's environment, installed once as `/etc/besser-lab/lab.env`. |
| `nginx-lab.conf` | The nginx site: the lab origin and one origin per preview slot. |
| `../../Dockerfile` (target `lab`), `../../docker-compose.lab.yml`, `../../docker/lab/entrypoint.sh`, `../../.env.lab.example` | Local mode. |

## What runs where

Two kinds of origin, so code the lab agent writes never runs where the browser holds the lab's GitHub session:

| Origin | Path | Goes to |
|---|---|---|
| `https://<LAB_HOST>` | `/besser_api/` | lab backend `127.0.0.1:9000` (SSE unbuffered, 1 h timeout) |
| | `/` | lab editor `127.0.0.1:8080`; `X-Frame-Options: DENY` |
| `https://preview.<LAB_HOST>` (slot 0), `https://preview-2.<LAB_HOST>` (slot 1) | `/besser_api/` | the slot's backend, `9100 + 10 i` |
| | `/agent` | the slot's modeling agent, `8865 + 10 i` (WebSocket) |
| | `/` | the slot's editor dev server (Vite), `8180 + 10 i`, hot reload; `frame-ancestors https://<LAB_HOST>` |

No preview block routes to port 9000. Every block admits only `LAB_ALLOW_CIDRS`.

| Process | Listens on | Runs from |
|---|---|---|
| lab backend (`besser-lab.service`, user `besser-lab`) | `127.0.0.1:9000` | `/opt/besser-lab/app` (BESSER at `LAB_APP_REF`), venv `/opt/besser-lab/venv` |
| preview LLM proxy, inside the lab backend | `127.0.0.1:9300` | only when the preview is on and `BESSER_LAB_PREVIEW_OPENAI_API_KEY` is set |
| lab editor (`besser-lab-editor.service`, a dynamic user) | port 8080, loopback only (`IPAddressDeny`) | a copy of `/opt/besser-lab/app/besser/utilities/web_modeling_editor/frontend/build` (built with `LAB_BUILD=true`) |
| preview slot *i*: backend, editor (Vite), modeling agent | `9100+10i`, `8180+10i`, `8865+10i` on `BESSER_LAB_PREVIEW_BIND` | the trees of the change the slot shows, each service in its own bubblewrap with a private network (pasta); children of the lab backend |

A slot shows one change at a time and follows it after each build. Its modeling agent runs only once that
change's modeling agent reads its port from `BESSER_AGENT_WS_PORT` (its `src/common/settings.py`); until then the
Try it tab says the assistant is not available in the preview. The lab's runs (plan and build steps) run in the
lab backend; their shell, checks and dependency installs each run in bubblewrap too.

Files on the host:

| Path | Owner, mode | Holds |
|---|---|---|
| `/opt/besser-lab/{app,venv,node,python,uv,bin}` | root, 0755 | The lab's code, its editor build and tools. `bin/` links uv, `python3.12` and `python3.11`, from which every venv is made. |
| `/etc/besser-lab/lab.env` | root, 0600 | Keys and settings. Read by systemd only. |
| `/etc/besser-lab/build.env` | root, 0644 | `BESSER_BUILD_SHA` of the lab's checkout. |
| `/srv/besser-lab/work/` | besser-lab | `BESSER_LAB_WORKSPACE_ROOT`: `base/{besser,modeling-agent}` (the base clones, the editor's at the submodule path inside `base/besser`), `changes/<id>/...` (one worktree per repository per change), `.cache/` (uv cache, `node_modules` snapshots). Hidden in every sandbox. |
| `/srv/besser-lab/git/` | besser-lab | `BESSER_LAB_GIT_ROOT`: every git directory, each worktree's under its repository's `worktrees/`. Read-only in every sandbox. |
| `/srv/besser-lab/state/lab/` | besser-lab | `BESSER_LAB_STATE_DIR`: changes, threads, runs and costs, audit log, team notes, preview logs. Hidden in every sandbox. |
| `/srv/besser-lab/state/sessions/` | besser-lab, 0700 | Signed-in GitHub sessions (encrypted) and their key. Hidden. |
| `/srv/besser-lab/state/spec_driven_runs.sqlite3` | besser-lab | The run store: the event logs of lab runs. Its folder is hidden. |
| `/srv/besser-lab/runs/` | besser-lab, 0700 | `BESSER_LLM_RUN_WORKSPACE_ROOT` (stays empty: the lab backend refuses Spec-Driven runs). Hidden. |
| `/var/lib/besser-lab-editor/` | the editor's dynamic user | The copy of the editor build the server runs (it rewrites its bundle at start). |

## Requirements

- Ubuntu 24.04 LTS or Debian 12, x86_64 or arm64, root access.
- Sizing (estimate, to be measured on the host): 4 vCPU, 16 GB RAM, 100 GB disk for two preview slots, two runs at
  once and a dozen open changes. Each change adds about 140 MB of trees and environments.
- **The data volume (`/srv/besser-lab`) on XFS with reflink** (`mkfs.xfs -m reflink=1`, the default of recent
  xfsprogs) if you can: changes then get copy-on-write clones of the shared `node_modules` and venvs. On ext4 the lab
  hard-links them instead, makes the shared snapshot read-only and warns on its settings page that an in-place write
  into `node_modules` or a venv could reach other changes.
- DNS names pointing at the host: `LAB_HOST` and one per preview slot (`preview.<LAB_HOST>`,
  `preview-2.<LAB_HOST>`). Port 443 open to the networks that may use the lab, port 80 open to Let's Encrypt for the
  certificate (step 4 for a host the internet cannot reach).
- Outbound HTTPS to github.com, nodejs.org, pypi.org, registry.npmjs.org, the GitHub releases of
  python-build-standalone (uv downloads Python from there), the LLM provider and, for CI failure logs on the run
  card, `*.blob.core.windows.net` (where GitHub serves Actions logs). Behind a proxy, export
  `HTTPS_PROXY`/`HTTP_PROXY`/`NO_PROXY=127.0.0.1,localhost` (and the lower-case names) and pass them through `sudo`,
  whose default `env_reset` drops exported variables:
  `sudo --preserve-env=HTTPS_PROXY,HTTP_PROXY,NO_PROXY,https_proxy,http_proxy,no_proxy LAB_HOST=... bash provision.sh install`
  (the same for `update`). If apt still cannot reach its mirrors, set `Acquire::http::Proxy` and
  `Acquire::https::Proxy` in `/etc/apt/apt.conf.d/`. Set the upper-case variables in `lab.env` too.
- The lab's code pushed to GitHub: `provision.sh` checks out `feature/besser-lab` of BESSER-PEARL/BESSER
  (`LAB_APP_REF` selects another branch, tag or commit), and the editor commit it pins must be on
  BESSER-PEARL/BESSER-Web-Modeling-Editor. The lab's UI is that build; nothing of the lab has to be on `develop`.
- To try changes branched from the editor's `develop` in the preview, `develop` needs `VITE_DEV_BACKEND_URL` in
  `shared/constants/constant.ts` (a small pull request of its own); without it a preview editor calls
  `http://localhost:9000` on the visitor's machine.

## Setting it up

1. **Get this folder** on the host, from the code you will install:

   ```bash
   git clone --depth 1 --branch feature/besser-lab https://github.com/BESSER-PEARL/BESSER.git besser-lab-src
   cd besser-lab-src/scripts/lab
   ```

2. **First run.** It installs the system packages, Node.js, uv with Python 3.12 and 3.11, the `besser-lab` user and
   the lab's checkout in `/opt/besser-lab/app`, writes `/etc/besser-lab/lab.env`, and stops at the certificate:

   ```bash
   sudo LAB_HOST=lab.example.org LAB_ALLOW_CIDRS="192.0.2.0/24 198.51.100.7/32" bash provision.sh install
   ```

   `LAB_ALLOW_CIDRS` is the space-separated list of networks (VPN, office, single admin addresses) that may reach
   the lab at all; nginx answers 403 to everyone else. It is required on purpose: the preview's backend and
   modeling agent have no sign-in of their own and run code nobody has reviewed yet. Everything after the checkout
   runs from the checkout's own copy of `provision.sh`, so the deploy always matches the code it deploys.

3. **GitHub OAuth app.** In the GitHub settings of the BESSER-PEARL organization: Developer settings, OAuth
   Apps, New OAuth App.
   - Homepage URL: `https://<LAB_HOST>`
   - Authorization callback URL: `https://<LAB_HOST>/besser_api/github/auth/callback`
   - Generate a client secret. The client id and secret go into `lab.env` in step 6.

   The backend asks for the scopes `repo,gist,user`. Without the `workflow` scope GitHub refuses a push that changes
   `.github/workflows/`, so the lab refuses such a change before pushing. Each person on
   `BESSER_LAB_ALLOWED_LOGINS` needs write access to the repositories their changes touch, and when the
   organization restricts third-party OAuth apps, an owner has to approve this one, or every push fails.

4. **TLS certificate**, one for the lab and every preview host. With the distribution's default nginx site still
   serving `/var/www/html` on port 80, run the command step 2 printed, for example:

   ```bash
   sudo certbot certonly --webroot -w /var/www/html --expand --cert-name lab.example.org \
     -d lab.example.org -d preview.lab.example.org -d preview-2.lab.example.org
   ```

   On a host Let's Encrypt cannot reach (VPN only), use a DNS challenge (`certbot certonly --manual
   --preferred-challenges dns ...` or your DNS provider's certbot plugin) or your organization's CA, and put the
   files where `nginx-lab.conf` expects them: `/etc/letsencrypt/live/<LAB_HOST>/{fullchain,privkey}.pem`.
   `provision.sh` checks that the certificate names every host.

5. **Second run**, same command as step 2. It installs the nginx site, runs `setup_workspace.sh` as `besser-lab`
   (it clones the three repositories of `BESSER_LAB_REPO_*` with their git directories in `/srv/besser-lab/git`,
   makes the base environments the way each repository's docs describe: BESSER's venv with `-e ".[dev]"`, the
   backend and docs requirements and `import-linter==2.15`, the editor's `npm ci`, the modeling agent's venv with
   its vendored BAF patch, and snapshots the editor's `node_modules`), builds the lab editor with `LAB_BUILD=true`,
   and installs and enables both units. It starts the editor, but not the backend while required values are blank.

6. **Fill in `lab.env`** (`sudoedit /etc/besser-lab/lab.env`):

   | Variable | Value |
   |---|---|
   | `BESSER_LAB_ALLOWED_LOGINS` | Required. GitHub logins allowed to use the lab, comma-separated. Read when the service starts: a login removed later keeps its access until `systemctl restart besser-lab` (which cuts off a run in progress). |
   | `BESSER_LAB_LLM_API_KEY` | Required. Key of `BESSER_LAB_LLM_PROVIDER` (default `anthropic`). |
   | `GITHUB_CLIENT_ID`, `GITHUB_CLIENT_SECRET` | Required. From step 3. |
   | `BESSER_LAB_ADMIN_LOGINS` | The maintainers who may drop anyone's change and stop any run. |
   | `BESSER_LAB_PREVIEW_OPENAI_API_KEY` | A separate OpenAI project key with a monthly budget, for the preview's assistant. Only the lab's LLM proxy holds it. |

   The other values have working defaults; each is explained in the file. Then:

   ```bash
   sudo systemctl start besser-lab
   ```

7. **Check it** (a deploy is done when the running services say so, not when the script ends). On the host:

   ```bash
   systemctl status besser-lab besser-lab-editor
   curl -s http://127.0.0.1:9000/besser_api/ | grep -o '"besser_build":"[^"]*"'   # = git -C /opt/besser-lab/app rev-parse HEAD
   curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8080/                # 200: the lab editor
   # The sandbox under the unit's own restrictions:
   sudo systemd-run --wait --pipe --quiet --uid=besser-lab --gid=besser-lab \
     -p NoNewPrivileges=yes -p PrivateTmp=yes -p ProtectSystem=full -p WorkingDirectory=/opt/besser-lab/app \
     /opt/besser-lab/venv/bin/python -c \
     "from besser.spec_driven_agent.execution.sandbox import sandbox_selftest_error as e; print(e() or 'sandbox OK')"
   ```

   From a machine in `LAB_ALLOW_CIDRS` (on the host itself nginx answers 403 unless its own address is listed):

   ```bash
   curl -s -o /dev/null -w '%{http_code}\n' https://lab.example.org/                  # 200: the lab editor
   curl -s -o /dev/null -w '%{http_code}\n' https://lab.example.org/besser_api/lab/home  # 401 without a session
   curl -sI https://preview.lab.example.org/ | grep -i content-security-policy        # frame-ancestors https://lab.example.org
   ```

   Then sign in on `https://<LAB_HOST>/` with an allowed login: the home page shows the base branches, the month's
   meter and the preview slots. Open a change's Try it tab to start a slot (the first start of a change's editor
   environment takes minutes). Logs: `journalctl -u besser-lab -f`, and each preview service's log in
   `/srv/besser-lab/state/lab/preview-logs/slot-<i>/`.

## Updating

```bash
sudo bash /opt/besser-lab/app/scripts/lab/provision.sh update     # behind a proxy: sudo --preserve-env=... as above
```

It stops the lab backend (a run in progress is cut off: check the home page first), moves `/opt/besser-lab/app` to
the latest `LAB_APP_REF`, and continues with that checkout's `provision.sh`: `setup_workspace.sh` fast-forwards the
base clones to their upstream branches and refreshes the base environments (open changes never move), the lab editor
is rebuilt when its commit or `DEPLOYMENT_URL` changed, the units are reinstalled, and both services restart.

`update` never overwrites `lab.env` or the base modeling agent's `config.yaml`, so a variable that a newer
`lab.env.example` adds has to be copied into `/etc/besser-lab/lab.env` by hand. To change `LAB_HOST`,
`LAB_ALLOW_CIDRS` or `BESSER_LAB_PREVIEW_URLS`, issue the certificate for the new names (step 4), edit `lab.env`
(`GITHUB_REDIRECT_URI`, `DEPLOYMENT_URL`, `CORS_ORIGINS`, `BESSER_LAB_PREVIEW_URLS`), point the OAuth app at the new
callback (step 3), and run `install` again from `/opt/besser-lab/app/scripts/lab/`.

Back up `/srv/besser-lab/state` (changes, threads, audit log, team notes) and `/srv/besser-lab/git` with
`/srv/besser-lab/work/changes` (work not yet pushed) if the changes matter.

## Sandbox

Every shell command of the lab agent, every check, every dependency install of a change and every preview service
runs in bubblewrap, which needs an unprivileged user namespace, and gets a private network from pasta (`passt`),
which needs `/dev/net/tun`. Without them the lab refuses those commands and never runs one unconfined;
`provision.sh` runs the sandbox self-test and prints a warning when it fails.

- Debian 12: works as installed.
- Ubuntu 24.04 restricts unprivileged user namespaces through AppArmor. Allow them for bubblewrap only:

  ```bash
  sudo tee /etc/apparmor.d/bwrap >/dev/null <<'EOF'
  abi <abi/4.0>,
  include <tunables/global>

  profile bwrap /usr/bin/bwrap flags=(unconfined) {
    userns,
    include if exists <local/bwrap>
  }
  EOF
  sudo apparmor_parser -r /etc/apparmor.d/bwrap
  ```

  Then run `install` again and look for "bubblewrap sandbox: works".

What a sandbox sees, from the lab's one mount policy: its change's trees read-write; the git root read-only; the
workspace root (other changes, the base clones, the shared stores), the lab state, the session store, the run
store, the run workspaces and the lab's own checkout masked; the rest of the host read-only. Checks run with no
network; the shell, installs and preview services with a private one (NAT to the internet, no host loopback; a
preview's modeling agent may reach only the LLM proxy's port).

## Security

What the deployment does:

- nginx admits only `LAB_ALLOW_CIDRS`, on every origin. The lab routes also require a GitHub session of an allowed
  login; the allowlist is read when the service starts, so removing a login needs `systemctl restart besser-lab`.
- `lab.env` (LLM key, OAuth client secret, preview key) is readable only by root; systemd passes it to the backend.
  The editor unit gets none of it.
- The lab's own code, its editor build and its tools are root-owned, so nothing the lab runs can change the lab.
- The service's `PATH` holds only root-owned directories, so `git` (which pushes with the user's token), uv, npm and
  the Pythons never come from a directory a sandbox can write.
- The preview gets a scrubbed environment (no variable whose name contains KEY, SECRET, TOKEN, PASSWORD, SESSION
  and the like), no `GITHUB_CLIENT_ID` (sign-in is off in the preview), and talks only to its own origin.
- The preview's modeling agent gets a placeholder key and reaches OpenAI only through the lab's LLM proxy, which
  adds the preview key outside the sandbox and caps requests per day. Model-written agent code can spend the key,
  never read it.

The two gaps phase 1 documented are closed:

- **Model-written code ran outside the sandbox.** Checks (pytest, vitest, ruff, docs), dependency installs and
  the preview services now all run in bubblewrap under the same mount policy as the shell: they cannot read the
  lab backend's environment (`/proc/<pid>/environ`, private PID namespace), the session store, the lab state or
  the git directories' config (read-only, and the lab never reads a work tree's `.git` file).
- **The lab's chat was the working copy's editor, on the lab's origin.** The lab UI is now a build of the lab's own
  root-owned checkout, served on `https://<LAB_HOST>`; a change's editor runs only on a preview origin, which holds
  no session and which only the lab may frame.

What it does not protect against, and what to do about it:

- **Outbound traffic from the sandboxes.** pasta's NAT leaves the host as `besser-lab`, so model-written code can
  reach whatever the host reaches, intranet hosts included. Recommended on networks with internal services (an
  operator decision, so not scripted): drop private and link-local destinations for that user, except your DNS
  resolver and outbound proxy, for example with nftables:

  ```text
  table inet besser_lab_egress {
    chain output {
      type filter hook output priority 0;
      meta skuid "besser-lab" ip daddr { <resolver>, <proxy> } accept
      meta skuid "besser-lab" ip daddr { 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16, 169.254.0.0/16, 100.64.0.0/10 } drop
      meta skuid "besser-lab" ip6 daddr { fc00::/7, fe80::/10 } drop
    }
  }
  ```

  The rule also applies to the lab backend itself (same user), which needs only the internet and loopback.
- **CI failure logs** come from `*.blob.core.windows.net`. If the host cannot reach it, the run card shows the
  check summary and a link instead of the log's tail; check it once (`curl -sI https://<any>.blob.core.windows.net`).
- **Dependency installs share the uv cache read-write**, so a hostile install could plant a package file another
  change's install then links. Changes come only from allowed logins and their requests; review what a change
  adds to its requirements.
- **On a filesystem without reflinks** (ext4), see Requirements: the shared files are hard links.

## Local mode

One container on your own machine (Linux, or Docker Desktop with WSL2) runs the lab backend, the lab's own editor
and one preview slot, with the same sandbox. From the repository root:

```bash
cp .env.lab.example .env.lab          # fill in the four required values; never commit .env.lab
docker compose -f docker-compose.lab.yml up --build
```

Then open `http://localhost:8080`. The first start clones the three repositories into the `besser-lab` volume and
installs their environments (many minutes, several GB: check free disk first). Every port is published on
`127.0.0.1` only. The preview runs on `http://localhost:8180` (backend 9100, modeling agent 8865); the lab editor
only frames `https:` origins, so its Try it tab opens the preview in a new tab. To bring the base clones and their
environments up to date, with the lab stopped: `docker compose -f docker-compose.lab.yml run --rm besser-lab setup`. Native Windows has no sandbox, so it runs the
lab's unit tests only.

## Host checklist

What only the real host can prove, in this order; each paid step only with the go-ahead of whoever pays.

1. `provision.sh install` on the host (XFS volume if possible); both units start; `sandbox OK` as `besser-lab` under
   the unit's protections (step 7); the Linux-only sandbox, checks and preview tests as that user
   (`systemd-run` as in step 7, `python -m pytest tests/spec_driven_agent/test_sandbox_mounts.py
   tests/spec_driven_agent/test_pasta_forwarding.py tests/spec_driven_agent/lab tests/utilities/web_modeling_editor/backend/lab/test_preview.py`).
2. Red-team build: ask the agent to read `/proc/*/environ`, the sessions, the git config, and to reach
   `127.0.0.1:9000`, `169.254.169.254` and an intranet host, from the shell, a check, an install and a preview backend.
3. Two browsers, two logins, two builds at once; the caps and `runs_full`; switching a slot; a base refresh with an
   open change. Measure clone, venv and `node_modules` times and RAM per slot.
4. Try it through nginx on the preview origin: hot reload, a generator download in Chrome and Firefox, the frame
   allowed only from the lab; the preview assistant through the proxy (needs the preview key and a modeling agent
   that reads `BESSER_AGENT_WS_PORT`).
5. The review loop on forks (`BESSER_LAB_REPO_*`): a two-repository change, diff and line comment, a CI failure's
   log tail (the `*.blob.core.windows.net` egress), Fix CI, a collaborator's and an outsider's comment, update from
   base, merge, the automatic close.
6. Knowledge: a plan that reads the skill index and the repository map without re-reading `CLAUDE.md`, and one
   "work on #<open issue>" plan.
7. Local mode: `docker compose -f docker-compose.lab.yml up --build` on WSL2 Docker.
8. More systemd hardening for `besser-lab.service`, one setting at a time, each followed by the step 7 self-test and a
   preview start: `ProtectHome=yes`, `ProtectKernelTunables=yes`, `ProtectKernelModules=yes`,
   `ProtectControlGroups=yes`, `PrivateDevices=yes` (pasta needs `/dev/net/tun`: likely breaks it),
   `RestrictSUIDSGID=yes`, `LockPersonality=yes`, `SystemCallArchitectures=native`. Keep what passes;
   `systemd-analyze security besser-lab` rated phase 1's unit 8.5.

## What this was checked against

Built from the code it configures: `backend/settings.py` (every `BESSER_LAB_*` name), `services/lab/preview.py`
(slot ports, preview environment), `services/lab/service.py` (startup, hidden paths), `spec_driven_agent/lab/`
(`workspace.py` layout, `environments.py`), the editor's `vite.config.ts`, `constant.ts` and server bundle, and the
modeling agent's settings. `tests/utilities/web_modeling_editor/backend/lab/test_deploy_files.py` checks: every
`BESSER_LAB_*` of both env examples is a backend setting and `lab.env.example` sets all of them with values the
backend accepts; the nginx site as `provision.sh` renders it (lab block, one block per preview slot on
`preview.py`'s ports, `deny all` everywhere, no preview route to 9000, frame headers); the base clones
`setup_workspace.sh` makes pass `LabWorkspace.ensure_ready`, hold no `.git` directory and follow their upstream on a
second run; the local container's ports, volume and sandbox options.

Not tested, because it needs the host: everything in the host checklist above, `nginx -t` on the rendered site, a
complete `install` or `update` run and the editor build, the units under systemd (`systemd-analyze verify` was not
available where these files were written), and the local image build.
