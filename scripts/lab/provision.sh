#!/usr/bin/env bash
# Set up or update a BESSER Lab host. Ubuntu 24.04 LTS or Debian 12, run as root. See README.md beside this file.
#
#   sudo LAB_HOST=lab.example.org LAB_ALLOW_CIDRS="192.0.2.0/24 198.51.100.7/32" bash provision.sh install
#   sudo bash /opt/besser-lab/app/scripts/lab/provision.sh update
#
# install: system packages, Node.js, uv and its Pythons, the service user, the lab's own BESSER checkout
#          (/opt/besser-lab/app) and its venv. Then, from that checkout's copy of this script: /etc/besser-lab/lab.env
#          (first time only), the certificate check, the nginx site, the workspace (setup_workspace.sh), the lab
#          editor build, the two systemd units, and the start. Run it again to change LAB_HOST, LAB_ALLOW_CIDRS or
#          the preview hosts (BESSER_LAB_PREVIEW_URLS in lab.env).
# update:  stops the lab, moves the checkout to the latest LAB_APP_REF, then continues from the new checkout's copy
#          of this script: workspace, editor build, units, start.
set -euo pipefail

readonly LAB_USER=besser-lab
readonly OPT=/opt/besser-lab
readonly APP=$OPT/app
# This folder in the lab's checkout: every step after install_app runs the files of the code it installed.
readonly APP_SCRIPTS=$APP/scripts/lab
readonly LAB_HOME=/srv/besser-lab
readonly SESSIONS=$LAB_HOME/state/sessions
readonly RUNS=$LAB_HOME/runs
readonly ETC=/etc/besser-lab
readonly APP_REF=${LAB_APP_REF:-feature/besser-lab}
readonly BESSER_URL=https://github.com/BESSER-PEARL/BESSER.git
readonly NODE_VERSION=22.14.0
readonly UV_VERSION=0.5.11
readonly UV=$OPT/uv/bin/uv
# Root-owned links for the service's PATH: uv and its two Pythons (3.12 for BESSER, 3.11 for the modeling agent).
# The lab makes every venv from these; they live outside the workspace, which every sandbox hides.
readonly BIN=$OPT/bin
readonly FRONTEND_PATH=besser/utilities/web_modeling_editor/frontend
readonly EDITOR_BUILD=$APP/$FRONTEND_PATH/build
readonly LAB_BACKEND_PORT=9000
# Ports of preview slot 0; slot i adds i * SLOT_PORT_STEP. Same as services/lab/preview.py (test_deploy_files.py).
readonly PREVIEW_BACKEND_PORT=9100
readonly PREVIEW_FRONTEND_PORT=8180
readonly PREVIEW_AGENT_PORT=8865
readonly SLOT_PORT_STEP=10
# What setup_workspace.sh reads from lab.env: the layout, the repositories and the preview origins, never a key.
readonly -a WORKSPACE_SETTINGS=(BESSER_LAB_WORKSPACE_ROOT BESSER_LAB_GIT_ROOT BESSER_LAB_STATE_DIR
  BESSER_LAB_REPO_BESSER BESSER_LAB_REPO_FRONTEND BESSER_LAB_REPO_MODELING_AGENT BESSER_LAB_PREVIEW_URLS)
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
readonly HERE

die() {
  echo "provision: $*" >&2
  exit 1
}

note() {
  echo "provision: $*"
}

# uv's Python builds live under /opt, readable by every user; nothing downloads one implicitly.
uv() {
  UV_PYTHON_INSTALL_DIR=$OPT/python UV_PYTHON_DOWNLOADS=never "$UV" "$@"
}

# A command as the service user, in the lab's checkout, with the service's PATH. Proxy variables pass through.
as_lab() {
  runuser -u "$LAB_USER" -- env -C "$APP" HOME="$LAB_HOME" \
    PATH="$BIN:$OPT/node/bin:/usr/local/bin:/usr/bin:/bin" "$@"
}

# The last value of NAME in lab.env (systemd's syntax: NAME=value, no quotes in this file).
lab_env_value() {
  sed -n "s/^$1=//p" "$ETC/lab.env" | tail -n 1
}

install_packages() {
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -q
  # build-essential: wheels the modeling agent's dependencies may compile (its Docker image installs gcc/g++).
  # bubblewrap and passt: the sandbox and private network of every shell, check, install and preview service.
  apt-get install -y -q --no-install-recommends \
    git ca-certificates curl xz-utils openssl nginx certbot python3-venv bubblewrap passt build-essential
}

install_node() {
  if [[ -x $OPT/node/bin/node && $("$OPT/node/bin/node" --version) == "v$NODE_VERSION" ]]; then
    return
  fi
  local arch
  case $(uname -m) in
    x86_64) arch=x64 ;;
    aarch64) arch=arm64 ;;
    *) die "no Node.js build for $(uname -m)" ;;
  esac
  local name=node-v$NODE_VERSION-linux-$arch.tar.xz tmp
  tmp=$(mktemp -d)
  curl -fsSL --output "$tmp/$name" "https://nodejs.org/dist/v$NODE_VERSION/$name"
  curl -fsSL --output "$tmp/SHASUMS256.txt" "https://nodejs.org/dist/v$NODE_VERSION/SHASUMS256.txt"
  (cd "$tmp" && grep " $name\$" SHASUMS256.txt | sha256sum --check --quiet -) || die "checksum mismatch for $name"
  rm -rf "$OPT/node"
  install -d -m 0755 "$OPT/node"
  tar -xJf "$tmp/$name" --strip-components=1 --no-same-owner -C "$OPT/node"
  rm -rf "$tmp"
  note "Node.js $NODE_VERSION in $OPT/node"
}

# uv provides the two Python versions the repositories' own images use: 3.12 for BESSER, 3.11 for the agent.
install_python() {
  if [[ ! -x $UV ]] || [[ $("$UV" --version) != "uv $UV_VERSION"* ]]; then
    rm -rf "$OPT/uv"
    python3 -m venv "$OPT/uv"
    "$OPT/uv/bin/pip" install --quiet --disable-pip-version-check "uv==$UV_VERSION"
  fi
  # The one step that downloads Python builds; every other uv call has downloads off.
  UV_PYTHON_INSTALL_DIR=$OPT/python "$UV" python install 3.12 3.11
  install -d -m 0755 "$BIN"
  ln -sfn "$UV" "$BIN/uv"
  local version
  for version in 3.12 3.11; do
    ln -sfn "$(uv python find --python-preference only-managed "$version")" "$BIN/python$version"
  done
}

create_user() {
  if ! id -u "$LAB_USER" >/dev/null 2>&1; then
    useradd --system --user-group --home-dir "$LAB_HOME" --shell /usr/sbin/nologin "$LAB_USER"
  fi
  # setup_workspace.sh makes the workspace, git root and lab state folders under these.
  install -d -o "$LAB_USER" -g "$LAB_USER" -m 0750 "$LAB_HOME"
  install -d -o "$LAB_USER" -g "$LAB_USER" -m 0700 "$LAB_HOME/state" "$SESSIONS" "$RUNS"
}

# The lab's own code: root-owned, so neither the service nor anything it runs can change what runs as the lab.
install_app() {
  install -d -m 0755 "$OPT"
  if [[ ! -d $APP/.git ]]; then
    git clone --quiet "$BESSER_URL" "$APP"
  fi
  git -C "$APP" fetch --quiet origin "$APP_REF"
  git -C "$APP" checkout --quiet --force --detach FETCH_HEAD
  # The lab editor is built from the editor commit this checkout pins.
  git -C "$APP" submodule --quiet update --init --force "$FRONTEND_PATH"
  if [[ ! -x $OPT/venv/bin/python ]]; then
    uv venv --quiet --python 3.12 "$OPT/venv"
  fi
  uv pip install --quiet --python "$OPT/venv/bin/python" \
    -e "$APP" -r "$APP/besser/utilities/web_modeling_editor/backend/requirements.txt"
  install -d -m 0755 "$ETC"
  printf 'BESSER_BUILD_SHA=%s\n' "$(git -C "$APP" rev-parse HEAD)" > "$ETC/build.env"
  chmod 0644 "$ETC/build.env"
  note "lab code: $(git -C "$APP" log -1 --format='%h %s')"
}

install_env_file() {
  install -d -m 0755 "$ETC"
  if [[ -f $ETC/lab.env ]]; then
    return
  fi
  (umask 077 && sed "s/lab\.example\.org/$LAB_HOST/g" "$APP_SCRIPTS/lab.env.example" > "$ETC/lab.env")
  chown root:root "$ETC/lab.env"
  note "wrote $ETC/lab.env: fill in its required values (README step 6)"
}

# Names of the required lab.env values that are still blank.
missing_env() {
  local name
  for name in BESSER_LAB_ALLOWED_LOGINS BESSER_LAB_LLM_API_KEY GITHUB_CLIENT_ID GITHUB_CLIENT_SECRET; do
    grep -Eq "^$name=.+" "$ETC/lab.env" || printf '%s ' "$name"
  done
}

# The host names of a comma-separated BESSER_LAB_PREVIEW_URLS, one per line, in slot order.
preview_hosts() {
  local url host
  local -a urls=()
  IFS=, read -r -a urls <<<"$1"
  for url in "${urls[@]}"; do
    url=${url//[[:space:]]/}
    [[ -n $url ]] || continue
    host=${url#https://}
    host=${host%/}
    [[ $url == https://* && $host =~ ^[a-z0-9]([a-z0-9.-]*[a-z0-9])?$ ]] \
      || die "not a preview origin of the form https://<host name>: $url"
    printf '%s\n' "$host"
  done
}

# nginx-lab.conf for LAB_HOST: one `allow` line per network of CIDRS at each marker, and the preview server block
# once per host, slot i on its ports.   render_nginx_conf TEMPLATE LAB_HOST "CIDR ..." [PREVIEW_HOST ...]
render_nginx_conf() {
  local template=$1 host=$2 cidrs=$3
  shift 3
  awk -v lab="$host" -v cidrs="$cidrs" -v previews="$*" -v step="$SLOT_PORT_STEP" \
      -v backend="$PREVIEW_BACKEND_PORT" -v frontend="$PREVIEW_FRONTEND_PORT" -v agent="$PREVIEW_AGENT_PORT" '
    function emit(line, slot,    indent, count, k, list) {
      if (line ~ /# LAB_ALLOW_CIDRS$/) {
        match(line, /^[ \t]*/)
        indent = substr(line, 1, RLENGTH)
        count = split(cidrs, list, " ")
        for (k = 1; k <= count; k++) printf "%sallow %s;\n", indent, list[k]
        return
      }
      gsub(/__PREVIEW_HOSTS__/, previews, line)
      gsub(/__PREVIEW_HOST__/, host_of[slot + 1], line)
      gsub(/__PREVIEW_BACKEND_PORT__/, backend + slot * step, line)
      gsub(/__PREVIEW_FRONTEND_PORT__/, frontend + slot * step, line)
      gsub(/__PREVIEW_AGENT_PORT__/, agent + slot * step, line)
      gsub(/lab\.example\.org/, lab, line)
      print line
    }
    BEGIN { slots = split(previews, host_of, " ") }
    /# BEGIN PREVIEW SLOT$/ { in_slot = 1; next }
    /# END PREVIEW SLOT$/ {
      in_slot = 0
      for (slot = 0; slot < slots; slot++) for (k = 1; k <= lines; k++) emit(block[k], slot)
      next
    }
    in_slot { block[++lines] = $0; next }
    { emit($0, 0) }' "$template"
}

# One certificate, in LAB_HOST's folder, for the lab and every preview host.
check_certificate() {
  local cert=/etc/letsencrypt/live/$LAB_HOST/fullchain.pem name
  local -a missing=()
  for name in "$LAB_HOST" "$@"; do
    if [[ ! -f $cert || $(openssl x509 -in "$cert" -noout -checkhost "$name") != *"does match"* ]]; then
      missing+=("$name")
    fi
  done
  if ((${#missing[@]})); then
    die "no TLS certificate for ${missing[*]} yet. Issue one for every name, then run install again:
    certbot certonly --webroot -w /var/www/html --expand --cert-name $LAB_HOST$(printf ' -d %s' "$LAB_HOST" "$@")"
  fi
}

install_nginx_site() {
  local listed
  local -a hosts=()
  listed=$(preview_hosts "$(lab_env_value BESSER_LAB_PREVIEW_URLS)")
  [[ -z $listed ]] || mapfile -t hosts <<<"$listed"
  check_certificate "${hosts[@]}"
  render_nginx_conf "$APP_SCRIPTS/nginx-lab.conf" "$LAB_HOST" "$LAB_ALLOW_CIDRS" "${hosts[@]}" \
    > /etc/nginx/conf.d/besser-lab.conf
  nginx -t
  systemctl reload nginx
  # certbot's own timer renews the certificate; nginx has to reload to serve the new one.
  install -d -m 0755 /etc/letsencrypt/renewal-hooks/deploy
  printf '#!/bin/sh\nsystemctl reload nginx\n' > /etc/letsencrypt/renewal-hooks/deploy/besser-lab-reload-nginx
  chmod 0755 /etc/letsencrypt/renewal-hooks/deploy/besser-lab-reload-nginx
}

# The base clones, their environments and the shared stores, as the service user (setup_workspace.sh).
setup_workspace() {
  local name value
  local -a settings=()
  for name in "${WORKSPACE_SETTINGS[@]}"; do
    value=$(lab_env_value "$name")
    [[ -z $value ]] || settings+=("$name=$value")
  done
  as_lab "${settings[@]}" LAB_PYTHON="$OPT/venv/bin/python" bash "$APP_SCRIPTS/setup_workspace.sh"
}

# The lab's editor (LAB_BUILD=true) from the checkout's editor, for DEPLOYMENT_URL; rebuilt when either changes.
build_lab_editor() {
  local url stamp want
  url=$(lab_env_value DEPLOYMENT_URL)
  [[ $url == https://* ]] || die "DEPLOYMENT_URL in $ETC/lab.env is not https://<LAB_HOST>: ${url:-unset}"
  stamp=$EDITOR_BUILD/.besser-lab-build
  want="$(git -C "$APP" rev-parse "HEAD:$FRONTEND_PATH") $url"
  if [[ -f $stamp && $(cat "$stamp") == "$want" ]]; then
    return
  fi
  note "building the lab editor for $url (several minutes)"
  (
    cd "$APP/$FRONTEND_PATH"
    export PATH=$OPT/node/bin:$PATH
    npm ci --no-audit --no-fund --loglevel=error
    LAB_BUILD=true DEPLOYMENT_URL=$url BACKEND_URL=$url/besser_api npm run build
  )
  printf '%s\n' "$want" > "$stamp"
}

install_units() {
  install -m 0644 "$APP_SCRIPTS/besser-lab.service" "$APP_SCRIPTS/besser-lab-editor.service" /etc/systemd/system/
  systemctl daemon-reload
  systemctl enable --quiet besser-lab.service besser-lab-editor.service
}

# Every shell, check, install and preview service of the lab runs in bubblewrap with a private network. Without
# a user namespace the lab refuses them (it never runs one unconfined), so say so now, not at the first build.
check_sandbox() {
  if as_lab "$OPT/venv/bin/python" -c 'import sys
from besser.spec_driven_agent.execution.sandbox import sandbox_selftest_error
error = sandbox_selftest_error()
print(error or "sandbox OK")
sys.exit(1 if error else 0)'; then
    note "bubblewrap sandbox: works for $LAB_USER"
    return
  fi
  echo "provision: WARNING: the sandbox does not work for $LAB_USER, so the lab refuses every shell command," \
    "check and preview. Fix: README section 'Sandbox'." >&2
}

start_lab() {
  systemctl restart besser-lab-editor.service
  local missing
  missing=$(missing_env)
  if [[ -n $missing ]]; then
    note "not started: fill in $missing in $ETC/lab.env, then: systemctl start besser-lab"
    return
  fi
  systemctl restart besser-lab.service
  note "started. Check: curl -s http://127.0.0.1:$LAB_BACKEND_PORT/besser_api/ (besser_build) and" \
    "journalctl -u besser-lab"
}

validate_install_inputs() {
  [[ ${LAB_HOST:-} =~ ^[a-z0-9]([a-z0-9.-]*[a-z0-9])?$ ]] || die "set LAB_HOST to the lab's host name"
  [[ -n ${LAB_ALLOW_CIDRS:-} ]] || die "set LAB_ALLOW_CIDRS to the networks allowed to reach the lab"
  local cidr
  for cidr in $LAB_ALLOW_CIDRS; do
    [[ $cidr =~ ^[0-9a-fA-F:.]+(/[0-9]{1,3})?$ ]] || die "not an address or network: $cidr"
  done
}

cmd_install() {
  validate_install_inputs
  install_packages
  install_node
  install_python
  create_user
  install_app
  exec bash "$APP_SCRIPTS/provision.sh" finish-install
}

finish_install() {
  validate_install_inputs
  install_env_file
  install_nginx_site
  setup_workspace
  build_lab_editor
  install_units
  check_sandbox
  start_lab
}

cmd_update() {
  [[ -d $APP/.git && -f $ETC/lab.env ]] || die "not installed yet: run install first"
  # A lab run in progress is cut off; so are the previews while their dependencies are replaced.
  systemctl stop besser-lab.service
  install_node
  install_python
  install_app
  exec bash "$APP_SCRIPTS/provision.sh" finish-update
}

finish_update() {
  setup_workspace
  build_lab_editor
  install_units
  check_sandbox
  start_lab
}

# Sourcing the file (to check one function) defines everything and runs nothing.
if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
  [[ $(id -u) == 0 ]] || die "run as root"
  case ${1:-} in
    install) cmd_install ;;
    update) cmd_update ;;
    # The second half of install and update: run by them from the checkout install_app has just moved.
    finish-install | finish-update)
      [[ $HERE == "$APP_SCRIPTS" ]] || die "$1 runs only from $APP_SCRIPTS"
      if [[ $1 == finish-install ]]; then finish_install; else finish_update; fi
      ;;
    *) die "usage: provision.sh install|update (see README.md)" ;;
  esac
fi
