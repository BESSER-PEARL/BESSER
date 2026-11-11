"""Push a finished spec-driven run to GitHub, and import a GitHub repo as a run.

Serves ``/spec-driven/push-to-github`` and ``/spec-driven/import-github-run``
(``routers/spec_driven_router.py``), which check the GitHub session and run
slot and map errors to HTTP. A refusal GitHub's answers lead to is an
``UpstreamError`` carrying its HTTP status; ``httpx.HTTPStatusError`` from the
GitHub API propagates for the router to map.
"""

import asyncio
import dataclasses
import fnmatch
import io
import json
import logging
import os
import re
import shutil
import tempfile
import stat
import time
import uuid
import zipfile
import zlib
from typing import Optional

import httpx

from besser.spec_driven_agent.providers.llm_client import FREE_TIER_PROVIDER, SPONSORED_PROVIDER
from besser.spec_driven_agent.run_recap import (
    AGENTS_POINTER,
    COMMITS_SINCE_RUN,
    RECORDS_DIR,
    build_recap,
    migrate_agents_recap,
    write_records,
)
from besser.spec_driven_agent.run_report import RECIPE_FILENAME
from besser.utilities.provenance import (
    AGENTS_FILENAME,
    PROVENANCE_FILENAME,
    add_agents_block,
    add_readme_block,
    readme_about,
    readme_besser_block,
)
from besser.utilities.secret_redaction import scrub_secret_files
from besser.utilities.workspace_fs import copytree_plain, is_plain_entry, open_plain_write
from besser.utilities.web_modeling_editor.backend.models.spec_driven import (
    ImportGitHubRunRequest,
    ImportGitHubRunResponse,
    PushSmartToGitHubRequest,
    PushSmartToGitHubResponse,
)
from besser.utilities.web_modeling_editor.backend.constants.constants import (
    MAX_REQUEST_SIZE,
    SMART_PUSH_TEMP_DIR_PREFIX,
)
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.class_diagram_converter import (
    class_buml_to_json,
)
from besser.utilities.web_modeling_editor.backend.services.deployment import render_blueprint, repo_model
from besser.utilities.web_modeling_editor.backend.services.deployment.github_service import (
    MAX_ARCHIVE_MEMBERS,
    MAX_EXTRACTED_ARCHIVE_BYTES,
    content_hash,
    create_github_service,
)
from besser.utilities.web_modeling_editor.backend.services.exceptions import UpstreamError
from besser.utilities.web_modeling_editor.backend.services.reverse_engineering import (
    AI_READ_TIMEOUT_SECONDS,
    ai_reading_available,
    analyze_repository,
    propose_model_with_ai,
    should_read_with_ai,
)
from besser.utilities.web_modeling_editor.backend.services.spec_driven.runner import (
    _EXCLUDED_OUTPUT_DIRS,
    CARRIED_RUN_FILES,
    GITHUB_SOURCE_FILENAME,
    SmartRunEntry,
    SmartRunRegistry,
    create_run_workspace,
    run_changes,
)
from besser.utilities.web_modeling_editor.backend.services.spec_driven.workspace_tree import (
    read_run_json,
    write_run_json,
)
from besser.utilities.web_modeling_editor.backend.settings import is_demo_token

logger = logging.getLogger(__name__)


def _scrub_secret_env_files(workdir: str) -> list[str]:
    """Compatibility wrapper for the GitHub-push security boundary."""
    return list(scrub_secret_files(workdir).removed_files)


# The run's own files the push never sends; the import and the seed skip build
# output too, so a repo path under one of these was never in the run.
_RUN_INTERNAL_FILES = ".besser_*"


def _hidden_from_run(path: str, scrubbed: list[str]) -> bool:
    """True for a repo path the pushed run could not hold, so its absence is no deletion.

    The seed and the push copy skip ``_EXCLUDED_OUTPUT_DIRS`` and BESSER's
    internal files, and the push removes secret ``.env`` files (``scrubbed``).
    The internal files a run carries are its own: a stale copy is deleted.
    """
    if path in scrubbed:
        return True
    if path in CARRIED_RUN_FILES:
        return False
    return any(
        fnmatch.fnmatch(part, pattern)
        for part in path.split("/")
        for pattern in (*_EXCLUDED_OUTPUT_DIRS, _RUN_INTERNAL_FILES)
    )


def _resolve_target_branch(
    requested: Optional[str],
    available: Optional[list[str]],
    default_branch: str,
) -> str:
    """Pick the push target branch.

    Honours ``requested`` when it is present in ``available`` (``available
    is None`` means the branch set is not materialised yet — e.g. a
    freshly created repo — so the explicit request is honoured there).
    Falls back to ``default_branch`` otherwise.
    """
    if requested:
        requested = requested.strip()
    if requested:
        if available is None or requested in available:
            return requested
        logger.warning(
            "spec-driven push-to-github: requested branch %r not present; using %r",
            requested, default_branch,
        )
    return default_branch


async def push_run_to_github(
    entry: SmartRunEntry,
    req: PushSmartToGitHubRequest,
    access_token: str,
    repo_name: str,
    registry: SmartRunRegistry,
) -> PushSmartToGitHubResponse:
    """Push the stored run ``entry`` to ``repo_name`` (steps 3-6 of ``push_spec_driven_to_github``).

    3. Copy the stored file tree into a fresh temp dir, skipping build/
       dependency dirs, ``.besser_*`` internals, and any ``.env`` that
       carries a real secret.
    4. Inject ``buml/`` model files + ``buml/diagrams.json``.
    5. Create (default private) or reuse the repo, resolve the branch.
    6. Commit what the run changed relative to the commit it started from:
       files the run never held (build output, ``*.db``, symlinks, secret
       files) stay as they are in the repo.
    """
    deploy_config = req.deploy_config
    github = create_github_service(access_token)

    user_info = await github.get_authenticated_user()
    login = user_info.get("login")
    if not login:
        raise UpstreamError("Could not resolve GitHub user from session.", status_code=401)
    owner = (deploy_config.use_existing and deploy_config.owner) or login

    with tempfile.TemporaryDirectory(
        prefix=f"{SMART_PUSH_TEMP_DIR_PREFIX}{uuid.uuid4().hex}_"
    ) as workdir:
        # ---- 3. Copy the stored tree, dropping build/dep dirs and
        # `.besser_*` internal files (checkpoint / snapshot), except
        # those a continue-from-GitHub run reads back: the recipe (the
        # seed's generator and run history; without it the modify run
        # loses the framework guard), the line attribution (lines a
        # person edited in between) and the generator's stored output
        # (the base a model change is merged against).
        # Symlinks are dropped, never read through.
        copytree_plain(
            entry.temp_dir, workdir, [*_EXCLUDED_OUTPUT_DIRS, _RUN_INTERNAL_FILES],
            dirs_exist_ok=True,
        )
        for _internal in CARRIED_RUN_FILES:
            _src = os.path.join(entry.temp_dir, _internal)
            if os.path.isfile(_src) and is_plain_entry(_src):
                shutil.copy2(_src, os.path.join(workdir, _internal))
        scrubbed = _scrub_secret_env_files(workdir)
        if scrubbed:
            logger.info(
                "spec-driven push-to-github: scrubbed %d secret env file(s): %s",
                len(scrubbed), scrubbed,
            )

        # ---- 4. Inject the model source into buml/ ----
        # Prefer the run's model-synced export (a MODIFY run whose
        # instruction added domain entities stores it on the registry
        # entry) so the pushed buml/ matches the code. Fall back to the
        # request's projectExport for every other run.
        run_export = getattr(entry, "updated_project_export", None)
        effective_export = run_export or req.projectExport
        repo_model.write_model_files(workdir, effective_export)

        # ---- 5. Resolve the repo + target branch ----
        is_first_push = not deploy_config.use_existing
        if deploy_config.use_existing:
            try:
                branches = await github.get_branches(owner, repo_name)
            except httpx.HTTPStatusError as exc:
                if exc.response is not None and exc.response.status_code == 404:
                    # Requested an existing repo that isn't there — surface
                    # a machine-usable code rather than silently creating.
                    raise UpstreamError("repo_missing", status_code=404) from exc
                raise
            repo_default = await github.get_default_branch(owner, repo_name)
            target_branch = _resolve_target_branch(
                deploy_config.branch, branches, repo_default
            )
        else:
            description = (
                deploy_config.description
                or "Spec-driven app + model — Generated by BESSER"
            )
            repo_info = await github.create_repository(
                repo_name=repo_name,
                description=description,
                is_private=deploy_config.is_private,
                auto_init=True,
            )
            repo_default = repo_info.get("default_branch", "main")
            target_branch = _resolve_target_branch(
                deploy_config.branch, None, repo_default
            )

        # ---- 5b. Keep the app's deploy files current ----
        deploy_warnings = (
            [] if is_first_push
            else await _refresh_deploy_files(github, owner, repo_name, target_branch, workdir)
        )

        # ---- 5c. Link the README back to BESSER and the model ----
        # Only a new repo gets a README the run lacks; an existing one keeps its own.
        # That new README says what the project is and when it was generated (#750).
        model_source = f"{repo_model.MODEL_DIR}/{repo_model.DOMAIN_SOURCE}"
        project = repo_model.project_of(effective_export) if isinstance(effective_export, dict) else {}
        add_readme_block(workdir, readme_besser_block(
            owner, repo_name, target_branch,
            model_source=model_source if os.path.isfile(os.path.join(workdir, *model_source.split("/"))) else None,
            has_provenance=os.path.isfile(os.path.join(workdir, PROVENANCE_FILENAME)),
        ), title=repo_name if is_first_push else None,
            about=readme_about(project.get("description"), "Generated"))

        # Whoever comes back to this repo, here or in another coding tool, reads what the run left:
        # the records in .besser/, and one fixed pointer to them in AGENTS.md.
        seed_sha, held_files = _run_seed(entry.temp_dir, owner, repo_name)
        recipe = read_run_json(os.path.join(workdir, RECIPE_FILENAME))
        recap = build_recap(recipe)
        if recap:
            try:
                write_records(workdir, recipe)
            except OSError:
                logger.warning("%s/%s: could not write %s/", owner, repo_name, RECORDS_DIR, exc_info=True)
                deploy_warnings.append(f"BESSER did not write its run records to {RECORDS_DIR}/: in this "
                                       "repository a path there is a link or not a folder.")
            holds_file = is_first_push or await _hold_repo_agents_md(
                github, owner, repo_name, target_branch, workdir,
                seeded=seed_sha is not None, base_sha=seed_sha or deploy_config.base_commit_sha,
            )
            if holds_file:
                try:
                    # A repo pushed before the records moved still has the recap section in AGENTS.md.
                    migrate_agents_recap(workdir)
                except OSError:
                    logger.warning("%s/%s: could not move the AGENTS.md recap", owner, repo_name, exc_info=True)
                    holds_file = False
            if not (holds_file and add_agents_block(workdir, AGENTS_POINTER)):
                logger.warning("%s/%s: could not add to AGENTS.md; the push leaves it alone", owner, repo_name)
                deploy_warnings.append(f"BESSER did not add its pointer to {RECORDS_DIR}/ to {AGENTS_FILENAME}: "
                                       "in this repository it is a link or not text, it was added after this run "
                                       "started, or GitHub did not return it.")

        # ---- 5d. Fold BESSER's run records in GitHub diffs ----
        # A run that did not start from this repo lacks the repo's own file, which the push would replace.
        repo_gitattributes = None
        if not is_first_push and not os.path.lexists(os.path.join(workdir, repo_model.GITATTRIBUTES_FILE)):
            current = await github.get_file_content(
                owner, repo_name, repo_model.GITATTRIBUTES_FILE, ref=target_branch,
            )
            repo_gitattributes = current.get("content") if current else None
        repo_model.mark_generated(workdir, CARRIED_RUN_FILES, repo_gitattributes)

        # ---- 6. Push what the run changed ----
        # The tree is compared with the commit the run started from;
        # commits made on the branch since are merged in, never reverted.
        # The run records that commit itself: the editor's link may have
        # moved on (a pull after the run started), and pushing over the
        # link's newer base would revert what was pulled.
        commit_message = repo_model.with_origin(
            deploy_config.commit_message or "Spec-driven app + model — BESSER",
            repo_model.ORIGIN_AGENT,
        )
        pull_request_url = None
        if deploy_config.push_mode == "pull_request" and not is_first_push:
            push_branch = f"besser/update-{uuid.uuid4().hex[:8]}"
            base_sha = (
                seed_sha
                or deploy_config.base_commit_sha
                or await github.get_branch_head(owner, repo_name, target_branch)
            )
        else:
            push_branch = target_branch
            base_sha = seed_sha or deploy_config.base_commit_sha
        # What the push adds over the commit it starts from is credited to whoever pushes.
        effective_export = await repo_model.attribute_new_elements(
            github, owner, repo_name, effective_export, login=login, ref=base_sha, branch=target_branch,
        )
        if effective_export:
            repo_model.write_model_files(workdir, effective_export)
        push_results = await github.push_directory_to_repo(
            owner=owner,
            repo_name=repo_name,
            directory_path=workdir,
            commit_message=commit_message,
            branch=push_branch,
            # Only a run that started from this repo held its files, so only
            # such a run can have deleted one; any other run adds and updates.
            preserve_existing_files=seed_sha is None,
            base_commit_sha=base_sha,
            keep_path=lambda path: (_hidden_from_run(path, scrubbed)
                                    or (held_files is not None and path not in held_files)),
            model_merge=repo_model.ModelMerge(effective_export) if effective_export else None,
            base_hashes={path: digest for path, digest in (held_files or {}).items() if digest},
        )
        if not push_results.get("commit_sha"):
            raise UpstreamError("Failed to create GitHub commit for pushed files.", status_code=500)
        status = push_results.get("status") or "pushed"
        deploy_warnings += push_results.get("warnings", [])
        if push_branch != target_branch:
            pull_request = await github.create_pull_request(
                owner, repo_name, head=push_branch, base=target_branch,
                title=deploy_config.commit_message or "Update from the BESSER Spec-Driven Agent",
                body=repo_model.pull_request_body(
                    _model_changes(entry), push_results["uploaded_files"] + push_results["deleted_files"],
                ),
            )
            pull_request_url = pull_request["html_url"]
            status = "pull_request"
        await github.add_topic(owner, repo_name, repo_model.BESSER_TOPIC)
        if status in ("pushed", "merged"):
            # Runs seeded from this one start from the commit holding its tree.
            _record_run_seed(entry.temp_dir, owner, repo_name, push_results["own_commit_sha"],
                             _plain_file_hashes(workdir))

        # After a merge the run's files no longer match the branch: hand
        # back the merged repo as the next modify run's base.
        # The push has landed: a failure from here on is reported, not raised,
        # or the user would see a failed push and push again.
        base_run_id = None
        if status == "merged":
            try:
                base_run_id, _ = await _import_repo_as_run(
                    github, owner, repo_name, push_results["commit_sha"], registry,
                )
            except Exception:
                logger.warning("Pushed and merged %s/%s, but could not read the merged repo back",
                               owner, repo_name, exc_info=True)
                deploy_warnings.append(
                    "Your changes were merged on GitHub, but BESSER could not read the merged code back. "
                    "Use Get latest before the next change, so it starts from the merged code."
                )

        return PushSmartToGitHubResponse(
            success=True,
            repo_url=f"https://github.com/{owner}/{repo_name}",
            owner=owner,
            repo_name=repo_name,
            is_first_push=is_first_push,
            files_uploaded=push_results.get("total_files", 0),
            branch=target_branch,
            commit_sha=push_results["commit_sha"],
            status=status,
            conflict_branch=push_results.get("conflict_branch"),
            pull_request_url=pull_request_url,
            pull_request_branch=push_branch if pull_request_url else None,
            base_run_id=base_run_id,
            warnings=deploy_warnings,
            project_export=effective_export if run_export else None,
            recap=recap,
        )


async def _hold_repo_agents_md(
    github, owner: str, repo: str, branch: str, workdir: str, *, seeded: bool, base_sha: Optional[str],
) -> bool:
    """Make sure a push adds to the repo's ``AGENTS.md`` rather than replacing it; False when it cannot.

    A run that did not start from this repo holds no copy of the file, or its own;
    the repo's text wins. So does a run whose archive left the file out. A link, a
    file that is not text, a file GitHub did not return or one added after the push's
    base stays as it is: the pointer is not worth a failed push or a lost file. In that
    last case the run's own copy goes too, or the merge would see two new files.
    """
    path = os.path.join(workdir, AGENTS_FILENAME)
    run_holds_file = os.path.lexists(path)
    try:
        head = await github.get_branch_head(owner, repo, branch)
        entry = (await github.get_tree_entries(owner, repo, head)).get(AGENTS_FILENAME) if head else None
        if entry is None:
            return True
        if entry["mode"] not in repo_model.REGULAR_FILE_MODES:
            return False
        if base_sha and base_sha != head and AGENTS_FILENAME not in await github.get_tree_entries(owner, repo, base_sha):
            if run_holds_file:
                os.remove(path)
            return False
        if seeded and run_holds_file:
            return True
        current = await github.get_file_content(owner, repo, AGENTS_FILENAME, ref=head)
    except (httpx.HTTPError, UnicodeDecodeError):
        return seeded and run_holds_file  # Its own copy of the repo's file merges without GitHub's answer here.
    if current and current.get("content"):
        with open_plain_write(path, "wb", root=workdir) as handle:
            handle.write(current["content"].encode("utf-8"))
    return True


def _model_changes(entry) -> list[str]:
    """What the run changed in the model, from its recipe (empty when it changed none)."""
    recipe = read_run_json(os.path.join(entry.temp_dir, ".besser_recipe.json"))
    return run_changes(recipe)["modelChanges"] if isinstance(recipe, dict) else []


async def _refresh_deploy_files(github, owner: str, repo: str, branch: str, workdir: str) -> list[str]:
    """Rewrite the Render files "Go live" put in the repo, for the pushed tree's layout.

    The run does not hold them, so without this the push would delete them;
    with it, the next deploy also follows a layout change the agent made. A
    render.yaml BESSER did not write is left alone (it is in the pushed tree or
    not, as the run left it). Returns warnings for the user.
    """
    current = await github.get_file_content(owner, repo, "render.yaml", ref=branch)
    text = current.get("content") if current else None
    if not render_blueprint.is_managed(text):
        return []
    try:
        layout = render_blueprint.detect_layout(workdir)
        files, _ = render_blueprint.deploy_files(layout, repo, render_blueprint.existing_suffix(text))
        warnings = list(layout.warnings)
    except render_blueprint.LayoutRefused as refused:
        logger.warning("Pushed app in %s/%s no longer deploys as before: %s", owner, repo, refused)
        files = {"render.yaml": text}
        warnings = [f"This app can no longer be deployed as before ({refused}). The previous deploy settings were kept."]
    for path, content in files.items():
        target = os.path.join(workdir, *path.split("/"))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding="utf-8") as fh:
            fh.write(content)
    return warnings


def _run_seed(run_dir: str, owner: str, repo: str) -> tuple[Optional[str], Optional[dict[str, Optional[str]]]]:
    """The commit of ``owner/repo`` the run's files were derived from, and the files it held then.

    Import writes them (``GITHUB_SOURCE_FILENAME``), a push updates them, and
    seeding copies them into every modify run that starts from that run. The
    files are ``repo path -> content_hash`` of the bytes the run held; the hash
    is None in a record written before hashes were kept (a list of paths).
    ``(None, None)`` for a run that did not start from this repo; the files are
    None for a record written before they were kept.
    """
    source = read_run_json(os.path.join(run_dir, GITHUB_SOURCE_FILENAME))
    if not isinstance(source, dict):
        return None, None
    # GitHub names are case-insensitive: "Ana/Shop" and "ana/shop" are one repo.
    if str(source.get("owner")).lower() != owner.lower() or str(source.get("repo")).lower() != repo.lower():
        return None, None
    sha = source.get("commitSha")
    if not (isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{40}", sha)):
        return None, None
    paths = source.get("paths")
    if isinstance(paths, dict) and all(isinstance(p, str) and isinstance(h, str) for p, h in paths.items()):
        return sha, dict(paths)
    if isinstance(paths, list) and all(isinstance(p, str) for p in paths):
        return sha, dict.fromkeys(paths)
    return sha, None


def _record_run_seed(run_dir: str, owner: str, repo: str, commit_sha: str, files: dict[str, str]) -> None:
    """Record that the run's files now match ``commit_sha``, as ``repo path -> content_hash`` (best effort)."""
    try:
        write_run_json(os.path.join(run_dir, GITHUB_SOURCE_FILENAME),
                       {"owner": owner, "repo": repo, "commitSha": commit_sha, "paths": files})
    except OSError:
        logger.warning("Could not record the pushed commit in run %s", run_dir, exc_info=True)


def _plain_file_hashes(root: str) -> dict[str, str]:
    """``repo path -> content_hash`` of the regular files under ``root``, BESSER's own records left
    out except those a run carries (so a push can tell when one changed or is gone)."""
    found = {}
    for directory, dirs, names in os.walk(root):
        dirs[:] = [d for d in dirs if is_plain_entry(os.path.join(directory, d))]
        for name in names:
            full = os.path.join(directory, name)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            if is_plain_entry(full) and (rel in CARRIED_RUN_FILES or not fnmatch.fnmatch(name, _RUN_INTERNAL_FILES)):
                with open(full, "rb") as fh:
                    found[rel] = content_hash(fh.read())
    return dict(sorted(found.items()))


async def _note_commits_since_run(github, owner: str, repo: str, commit_sha: str, run_dir: str) -> None:
    """Add to the imported recipe what the repo gained after the run that wrote it.

    The next run and the recap read it, and the next run's own recipe drops it.
    Without GitHub's answer the repo still opens: the recap just lists no commits.
    """
    path = os.path.join(run_dir, RECIPE_FILENAME)
    recipe = read_run_json(path)
    if not isinstance(recipe, dict):
        return
    if recipe.pop(COMMITS_SINCE_RUN, None) is not None:  # A repo's own list is not GitHub's answer.
        write_run_json(path, recipe)
    try:
        last_run = await github.get_commits(owner, repo, path=RECIPE_FILENAME, per_page=1, ref=commit_sha)
        if not last_run or last_run[0]["sha"] == commit_sha:
            return
        compared = await github.compare_commits(owner, repo, last_run[0]["sha"], commit_sha)
    except httpx.HTTPError:
        logger.warning("Could not read the commits of %s/%s since its last run", owner, repo, exc_info=True)
        return
    recipe[COMMITS_SINCE_RUN] = [
        {"sha": commit["sha"], "message": commit["message"].partition("\n")[0], "author": commit["author"]}
        for commit in compared["commits"] if len(commit["parents"]) < 2  # a merge adds nothing to read
    ]
    write_run_json(path, recipe)


async def _import_repo_as_run(
    github, owner: str, repo: str, commit_sha: str, registry: SmartRunRegistry,
) -> tuple[str, str]:
    """Register ``owner/repo`` at ``commit_sha`` as a run a modify run can seed from.

    The repo is extracted into the shared run workspace, where the Spec-Driven
    worker (a separate container) finds it by run id. Returns the run id and
    its directory.
    """
    run_id = uuid.uuid4().hex
    extract_dir = create_run_workspace(run_id)
    try:
        await github.download_repo_tarball(owner, repo, commit_sha, dest_dir=extract_dir)
        # Before the hashes below: this note is not a change the run made.
        await _note_commits_since_run(github, owner, repo, commit_sha, extract_dir)
        # The registry persists a manifest pointing at one file of the run;
        # this one also records where the run came from and which files it
        # holds: an archive leaves out export-ignore paths, links and
        # submodules, and a later push must not read their absence as deletions.
        # It hashes each file too: the archive's bytes can differ from the blob
        # (.gitattributes, LFS), so only a changed hash means the run changed it.
        source_path = os.path.join(extract_dir, GITHUB_SOURCE_FILENAME)
        _record_run_seed(extract_dir, owner, repo, commit_sha, _plain_file_hashes(extract_dir))
    except BaseException:
        shutil.rmtree(extract_dir, ignore_errors=True)
        raise
    await registry.put(run_id, SmartRunEntry(
        file_path=source_path,
        file_name=GITHUB_SOURCE_FILENAME,
        is_zip=False,
        temp_dir=extract_dir,
        created_at=time.time(),
    ))
    return run_id, extract_dir


# Cap on generator output uploaded to be pushed: the request size limit, which
# rejects a larger body before the router sees it. Its expansion is capped like
# a repo imported from GitHub.
MAX_GENERATED_CODE_BYTES = MAX_REQUEST_SIZE
# Where the uploaded file itself is kept, for the run's download: a
# ``.besser_*`` name, so a push of the run never commits it.
GENERATED_CODE_FILENAME = ".besser_generated_code"


def _extract_generated_zip(archive: zipfile.ZipFile, dest_dir: str) -> None:
    """Extract ``archive`` into ``dest_dir``; refuses unsafe paths and an archive past the caps.

    Links and other non-regular members are skipped, and so are BESSER's own
    run records (``.besser_*``): the run's are written by BESSER alone.
    Everything is checked before anything is written, so a refused archive
    writes nothing.
    """
    dest_abs = os.path.abspath(dest_dir)
    members = []
    total = 0
    for info in archive.infolist():
        if info.is_dir() or stat.S_IFMT(info.external_attr >> 16) not in (0, stat.S_IFREG):
            continue
        if any(fnmatch.fnmatch(part, _RUN_INTERNAL_FILES) for part in info.filename.replace("\\", "/").split("/")):
            continue
        target = os.path.abspath(os.path.join(dest_abs, info.filename))
        if not target.startswith(dest_abs + os.sep):
            raise ValueError(f"Unsafe path in the generated code: {info.filename!r}")
        total += info.file_size
        members.append((info, target))
        if len(members) > MAX_ARCHIVE_MEMBERS or total > MAX_EXTRACTED_ARCHIVE_BYTES:
            raise ValueError("The generated code is too large to save to GitHub.")
    try:
        for info, target in members:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            # The reader stops at the declared size, which the caps above counted.
            with archive.open(info) as source, open_plain_write(target, "wb", root=dest_abs) as handle:
                shutil.copyfileobj(source, handle)
    except OSError as exc:
        # E.g. both ``a`` and ``a/b``: the archive has no tree a run can hold.
        raise ValueError("The generated ZIP could not be unpacked.") from exc


def _generated_file_name(file_name: str) -> str:
    """The uploaded file's own name, without any directory; refuses one BESSER keeps for itself."""
    name = os.path.basename((file_name or "").replace("\\", "/")).strip()
    if not name or name in (".", "..") or fnmatch.fnmatch(name, _RUN_INTERNAL_FILES):
        raise ValueError("The generated file has no usable name.")
    return name


def _store_generated_code(content: bytes, name: str, is_zip: bool, run_dir: str) -> str:
    """Write the uploaded file into ``run_dir`` (a ZIP also unpacked there); returns the stored file's path."""
    if is_zip:
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                _extract_generated_zip(archive, run_dir)
        except (zipfile.BadZipFile, zlib.error, EOFError, NotImplementedError, RuntimeError) as exc:
            # Damaged, encrypted, or packed with a method Python cannot read.
            raise ValueError("The generated ZIP could not be read.") from exc
        stored = os.path.join(run_dir, GENERATED_CODE_FILENAME)
    else:
        stored = os.path.join(run_dir, name)
    with open_plain_write(stored, "wb", root=run_dir) as handle:
        handle.write(content)
    return stored


async def import_generated_code(content: bytes, file_name: str, registry: SmartRunRegistry) -> str:
    """Register the output of a deterministic generator as a run, so it can be pushed to GitHub.

    The editor's generators (``/generate-output``) hand their output to the
    browser and keep nothing, while ``/spec-driven/push-to-github`` pushes a
    stored run. A ZIP is unpacked as the run's file tree; any other file is the
    tree's only file. Returns the run id.
    """
    name = _generated_file_name(file_name)
    # A generator names its ZIP output ``*.zip``; any other file (a ``.jar`` or
    # ``.xlsx`` is a ZIP too) is pushed as it is.
    is_zip = name.lower().endswith(".zip")
    run_id = uuid.uuid4().hex
    run_dir = create_run_workspace(run_id)
    writing = asyncio.ensure_future(asyncio.to_thread(_store_generated_code, content, name, is_zip, run_dir))
    try:
        stored = await asyncio.shield(writing)
    except asyncio.CancelledError:
        # A thread cannot be stopped: let it finish before removing what it
        # wrote (or it recreates the run directory), and before the caller
        # gives back the run slot.
        await asyncio.wait([writing])
        shutil.rmtree(run_dir, ignore_errors=True)
        raise
    except BaseException:
        shutil.rmtree(run_dir, ignore_errors=True)
        raise
    await registry.put(run_id, SmartRunEntry(
        file_path=stored,
        file_name=name,
        is_zip=is_zip,
        temp_dir=run_dir,
        created_at=time.time(),
    ))
    return run_id


_AI_READING_UNAVAILABLE = (
    "This repo has no BESSER model and no data classes BESSER can read, and "
    "AI reading is not available on this server; you can still work on its code.")
_AI_READING_OFFER = (
    "BESSER could not read this repo's data classes directly. It can read the code with an "
    "AI model and propose a class diagram, if you allow it to send the code to that model.")


def _ai_reading_provider(req: ImportGitHubRunRequest) -> str:
    """The server-paid tier that reads a repo: ``sponsored`` for a demo tab, else ``free``.

    A wrong demo secret just means the keyless model: it opens nothing up, so
    it is not worth failing the import.
    """
    demo = req.demo_token is not None and is_demo_token(req.resolved_demo_token())
    if req.demo_token is not None and not demo:
        logger.warning("import-github-run: invalid demo token, reading with the keyless model")
    return SPONSORED_PROVIDER if demo else FREE_TIER_PROVIDER


async def _read_repo_with_ai(req: ImportGitHubRunRequest, extract_dir, code_analysis, owner, repo):
    """``(model, analysis, message)`` from the AI fallback reader; the message explains a missing model.

    The repo's code goes to the server's model chain, whose first provider may
    be a third party, so only a request that allowed it (``allow_ai_reading``) is read.
    """
    if not req.allow_ai_reading:
        raise PermissionError("AI reading of a repository needs allow_ai_reading")
    provider = _ai_reading_provider(req)
    if not ai_reading_available(provider):
        logger.warning("import-github-run: no readable data classes in %s/%s and no %s model "
                       "for an AI reading", owner, repo, provider)
        return None, code_analysis, _AI_READING_UNAVAILABLE
    # A copy: a timed-out reading keeps running in its thread and must not mark this response.
    ai_analysis = dataclasses.replace(code_analysis)
    try:
        model = await asyncio.wait_for(
            asyncio.to_thread(propose_model_with_ai, extract_dir, ai_analysis, provider),
            timeout=AI_READ_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.warning("import-github-run: AI reading of %s/%s timed out after %ss",
                       owner, repo, AI_READ_TIMEOUT_SECONDS)
        model = None
    except Exception:
        logger.warning("import-github-run: AI reading of %s/%s failed", owner, repo, exc_info=True)
        model = None
    if model is None:
        return None, code_analysis, (
            "This repo has no data classes BESSER can read directly, and the AI model "
            "could not read a model from its code; you can still work on its code.")
    return model, ai_analysis, None


async def import_run_from_github(
    req: ImportGitHubRunRequest, access_token: str, registry: SmartRunRegistry,
) -> ImportGitHubRunResponse:
    """Steps 2-5 of ``import_github_run``, under its run slot.

    ``req.owner``, ``req.repo`` and ``req.model_path`` were checked when the
    request was parsed (see ``ImportGitHubRunRequest``).
    """
    owner, repo = req.owner, req.repo
    github = create_github_service(access_token)

    # ---- 2. Resolve the target branch (repo default when unset) ----
    try:
        branches = await github.get_branches(owner, repo)
    except httpx.HTTPStatusError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            # The repo (or the caller's access to it) doesn't exist —
            # surface a machine-usable code the frontend can branch on.
            raise UpstreamError("repo_missing", status_code=404) from exc
        raise
    branch = _resolve_target_branch(
        req.branch, branches, await github.get_default_branch(owner, repo)
    )

    # ---- 3-4. Pin the branch head and register it as a run ----
    # The commit is pinned so the next push knows what it was based on.
    commit_sha = await github.get_branch_head(owner, repo, branch)
    if not commit_sha:
        raise UpstreamError("repo_missing", status_code=404)
    run_id, extract_dir = await _import_repo_as_run(github, owner, repo, commit_sha, registry)

    # ---- 5. Read the re-importable model (buml/diagrams.json) ----
    project: Optional[dict] = None
    has_model = False
    message: Optional[str] = None
    diagrams_path = os.path.join(extract_dir, *repo_model.MODEL_FILE.split("/"))
    model_path: Optional[str] = None
    if req.model_path not in (None, repo_model.MODEL_FILE):
        # The file the user picked; when it is gone, the usual order below.
        saved = await asyncio.to_thread(repo_model.find_saved_project, extract_dir, req.model_path)
        if saved is not None:
            model_path, project = saved
            has_model = True
    if not has_model and os.path.isfile(diagrams_path):
        try:
            with open(diagrams_path, "r", encoding="utf-8") as fh:
                project = json.load(fh)
            has_model = True
            model_path = repo_model.MODEL_FILE
        except (OSError, ValueError, RecursionError):
            logger.warning(
                "import-github-run: buml/diagrams.json present but "
                "unreadable for %s/%s@%s",
                owner, repo, branch, exc_info=True,
            )
            message = (
                "This repo's BESSER model (buml/diagrams.json) could not "
                "be parsed."
            )
    if not has_model and message is None:
        saved = await asyncio.to_thread(repo_model.find_saved_project, extract_dir)
        if saved is not None:
            model_path, project = saved
            has_model = True
            logger.info("import-github-run: read the model of %s/%s from %s, saved by an older editor",
                        owner, repo, model_path)
    analysis = proposed_diagram = None
    ai_reading_offered = False
    if not has_model:
        # Not a BESSER repo: read its code for a model the user can accept.
        code_analysis, domain_model = await asyncio.to_thread(analyze_repository, extract_dir)
        ai_message = None
        if should_read_with_ai(code_analysis, domain_model) and not req.allow_ai_reading:
            # The code would leave the server: the user decides, then imports again.
            ai_reading_offered = ai_reading_available(_ai_reading_provider(req))
            ai_message = _AI_READING_OFFER if ai_reading_offered else _AI_READING_UNAVAILABLE
            logger.info("import-github-run: %s/%s needs an AI reading; %s", owner, repo,
                        "offered it" if ai_reading_offered else "none available")
        elif should_read_with_ai(code_analysis, domain_model):
            ai_model, ai_analysis, ai_message = await _read_repo_with_ai(
                req, extract_dir, code_analysis, owner, repo)
            if ai_model is not None:
                domain_model, code_analysis = ai_model, ai_analysis
            elif domain_model is not None:
                logger.warning("import-github-run: proposing the %s of %s/%s, as the AI reading gave no model",
                               code_analysis.model_source, owner, repo)
        analysis = code_analysis.to_dict()
        if domain_model is not None:
            proposed_diagram = {
                "title": f"{repo} model",
                "model": {**class_buml_to_json(domain_model), "type": "ClassDiagram"},
                "diagramType": "ClassDiagram",
            }
            message = (
                "This repo has no data classes BESSER can read directly, so an AI "
                "model read its code and proposes one. Check it before accepting."
            ) if code_analysis.ai else (
                f"This repo has no BESSER model. BESSER read its "
                f"{code_analysis.model_source} and proposes one."
            )
            if code_analysis.skipped_relationships:
                message += (
                    f" {len(code_analysis.skipped_relationships)} relationship(s) could not be "
                    f"read and are not in it: {', '.join(code_analysis.skipped_relationships)}."
                )
            if ai_reading_offered:
                message += f" {_AI_READING_OFFER}"
        elif message is None:
            message = ai_message or (
                "This repo has no BESSER model and no data classes BESSER "
                "can read; you can still work on its code."
            )

    return ImportGitHubRunResponse(
        run_id=run_id,
        project=project,
        has_model=has_model,
        owner=owner,
        repo=repo,
        branch=branch,
        commit_sha=commit_sha,
        message=message,
        analysis=analysis,
        proposed_diagram=proposed_diagram,
        ai_reading_offered=ai_reading_offered,
        model_path=model_path,
        has_code=await asyncio.to_thread(repo_model.has_app_code, extract_dir, model_path),
        recap=build_recap(read_run_json(os.path.join(extract_dir, RECIPE_FILENAME))),
    )
