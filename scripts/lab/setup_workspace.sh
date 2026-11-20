#!/usr/bin/env bash
# Lay out a BESSER Lab workspace, or bring it up to date: the base clones with every git directory in the git
# root, their environments, the shared stores and the base modeling agent's config.yaml. Idempotent. Run as the
# lab's service user from the lab's own BESSER checkout: provision.sh does on a host, docker/lab/entrypoint.sh in
# local mode.
#
#   BESSER_LAB_WORKSPACE_ROOT=/srv/besser-lab/work LAB_PYTHON=/opt/besser-lab/venv/bin/python \
#     bash scripts/lab/setup_workspace.sh
#
# The settings are the backend's, read the backend's way (BackendSettings): BESSER_LAB_WORKSPACE_ROOT is required;
# BESSER_LAB_GIT_ROOT, BESSER_LAB_STATE_DIR, BESSER_LAB_REPO_* and BESSER_LAB_PREVIEW_URLS have its defaults. The
# layout is LabWorkspace's (repo_specs_from_settings) and the environments are made by the lab's own Environments,
# so the lab finds them current. LAB_PYTHON is a Python that imports this checkout's besser (default: python3).
# Needs git, and uv, npm, python3.12 and python3.11 on PATH, outside the workspace.
set -euo pipefail

LAB_PYTHON=${LAB_PYTHON:-python3}

die() {
  echo "setup_workspace: $*" >&2
  exit 1
}

note() {
  echo "setup_workspace: $*"
}

# Shared by the Python steps: the settings, and each repository's spec as the lab builds it.
readonly SPECS_PY='
import sys
from besser.spec_driven_agent.lab.contracts import REPO_BESSER, REPO_FRONTEND, REPO_MODELING_AGENT
from besser.spec_driven_agent.lab.workspace import repo_specs_from_settings, upstream_url
from besser.utilities.web_modeling_editor.backend.settings import BackendSettings

settings = BackendSettings()
if not settings.lab_workspace_root:
    sys.exit("setup_workspace: set BESSER_LAB_WORKSPACE_ROOT")
upstreams = {REPO_BESSER: settings.lab_repo_besser, REPO_FRONTEND: settings.lab_repo_frontend,
             REPO_MODELING_AGENT: settings.lab_repo_modeling_agent}
specs = {spec.key: spec for spec in
         repo_specs_from_settings(settings.lab_workspace_root, settings.lab_git_root, upstreams)}
'

# Tab-separated: the folders, then the base modeling agent's port, browser origin and key, then one line per
# repository (key, tree, git directory, upstream URL, base branch), BESSER before the editor inside it.
readonly LAYOUT_PY='
from besser.utilities.web_modeling_editor.backend.services.lab.llm_proxy import PREVIEW_PROXY_PLACEHOLDER_KEY
from besser.utilities.web_modeling_editor.backend.services.lab.preview import PREVIEW_AGENT_PORT, PREVIEW_FRONTEND_PORT

origin = settings.lab_preview_urls[0] if settings.lab_preview_urls else f"http://localhost:{PREVIEW_FRONTEND_PORT}"
print("\t".join((settings.lab_workspace_root, settings.lab_git_root, settings.lab_state_dir,
                 str(PREVIEW_AGENT_PORT), origin, PREVIEW_PROXY_PLACEHOLDER_KEY)))
for key in (REPO_BESSER, REPO_FRONTEND, REPO_MODELING_AGENT):
    spec = specs[key]
    print("\t".join((key, spec.path, spec.git_dir, upstream_url(spec), spec.base_branch)))
'

# Each base environment, then the node_modules snapshot changes link from (Environments.ensure / snapshot_node).
readonly ENVIRONMENTS_PY='
import logging
import os
from besser.spec_driven_agent.lab.contracts import LAB_CACHE_DIRNAME, REPO_ORDER
from besser.spec_driven_agent.lab.environments import Environments, LabEnvironmentError, probe_link_mode

logging.basicConfig(level=logging.INFO, format="setup_workspace: %(message)s")
cache = os.path.join(settings.lab_workspace_root, LAB_CACHE_DIRNAME)
environments = Environments(cache, mode=probe_link_mode(cache))
try:
    for key in REPO_ORDER:
        environments.ensure(specs[key])
    environments.snapshot_node(specs[REPO_FRONTEND])
except LabEnvironmentError as exc:
    sys.exit(f"setup_workspace: {exc}")
'

# git on a base clone through its git directory, never through the tree's .git file; no hook, no credential.
base_git() {
  local git_dir=$1 tree=$2
  shift 2
  git -c core.hooksPath=/dev/null -c core.fsmonitor=false -c credential.helper= \
    --git-dir="$git_dir" --work-tree="$tree" "$@"
}

# Clone one base repository (its tree in the workspace, its git directory in the git root), or bring an existing
# clone to the tip of its upstream branch, as the lab's refresh does.
#   ensure_base_clone KEY TREE GIT_DIR UPSTREAM_URL BRANCH
ensure_base_clone() {
  local key=$1 tree=$2 git_dir=$3 url=$4 branch=$5
  local upstream_ref=refs/remotes/upstream/$branch
  if [[ -d $tree/.git ]]; then
    die "$key: $tree/.git is a git directory inside the workspace, where model-written code could rewrite" \
      "it. Remove $tree and run this script again: it clones with the git directory in $git_dir."
  fi
  if [[ ! -d $git_dir ]]; then
    if [[ -d $tree && -n $(ls -A "$tree") ]]; then
      die "$key: $tree holds files but has no git directory $git_dir; remove $tree and run this script again"
    fi
    note "$key: cloning $url ($branch)"
    mkdir -p "$(dirname "$git_dir")" "$tree"
    git -c credential.helper= clone --quiet --single-branch --branch "$branch" \
      --separate-git-dir="$git_dir" "$url" "$tree"
  fi
  base_git "$git_dir" "$tree" fetch --quiet --no-tags "$url" "+refs/heads/$branch:$upstream_ref"
  if [[ $(base_git "$git_dir" "$tree" symbolic-ref -q HEAD || true) != "refs/heads/$branch" ]]; then
    # BESSER_LAB_REPO_* named another branch: the base moves to it; open changes keep their own.
    if base_git "$git_dir" "$tree" show-ref --verify --quiet "refs/heads/$branch"; then
      base_git "$git_dir" "$tree" checkout --quiet "$branch"
    else
      base_git "$git_dir" "$tree" checkout --quiet -b "$branch" "$upstream_ref"
    fi
  fi
  base_git "$git_dir" "$tree" merge --quiet --ff-only "$upstream_ref"
}

# The base modeling agent's config.yaml (gitignored), written once: the preview's port and origin and the LLM
# proxy's placeholder key, never a real key. A change's preview passes these by environment instead.
#   write_agent_config TREE PORT ORIGIN KEY
write_agent_config() {
  local config=$1/config.yaml port=$2 origin=$3 key=$4
  if [[ -f $config ]]; then
    return
  fi
  (umask 077 && cat > "$config") <<EOF
agent:
  check_transitions_delay: 5

nlp:
  language: en
  region: US
  timezone: Europe/Madrid
  pre_processing: True
  intent_threshold: 0.55
  openai:
    api_key: $key

platforms:
  websocket:
    host: 127.0.0.1
    port: $port
    origins:
      - "$origin"
    streamlit:
      host: localhost
      port: 5000
EOF
  note "wrote $config"
}

main() {
  command -v git >/dev/null || die "git is not installed"
  local layout root git_root state_dir agent_port agent_origin agent_key line key tree git_dir url branch
  local agent_tree=
  local -a lines=()
  layout=$("$LAB_PYTHON" -c "$SPECS_PY$LAYOUT_PY") || die "the lab's settings cannot be read (see above)"
  mapfile -t lines <<<"${layout//$'\r'/}"
  IFS=$'\t' read -r root git_root state_dir agent_port agent_origin agent_key <<<"${lines[0]}"
  mkdir -p "$root" "$git_root"
  (umask 077 && mkdir -p "$state_dir")
  for line in "${lines[@]:1}"; do
    IFS=$'\t' read -r key tree git_dir url branch <<<"$line"
    ensure_base_clone "$key" "$tree" "$git_dir" "$url" "$branch"
    [[ $key != modeling_agent ]] || agent_tree=$tree
  done
  [[ -n $agent_tree ]] || die "no modeling agent repository in the layout"
  "$LAB_PYTHON" -c "$SPECS_PY$ENVIRONMENTS_PY"
  write_agent_config "$agent_tree" "$agent_port" "$agent_origin" "$agent_key"
  note "workspace ready in $root (git directories in $git_root, state in $state_dir)"
}

# Sourcing the file (to check one function) defines everything and runs nothing.
if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
  main "$@"
fi
