"""
GitHub Service for Repository Creation and Code Push.

Handles creating GitHub repositories and pushing generated code.
"""

import base64
import hashlib
import io
import logging
import os
import shutil
import stat
import tarfile
import tempfile
from typing import Any, Callable, Dict, List, Optional
from pathlib import Path

import httpx

from besser.utilities.web_modeling_editor.backend.services.deployment.render_blueprint import RENDER_DEPLOY_URL
from besser.utilities.web_modeling_editor.backend.services.deployment.repo_model import commit_origin, with_origin

logger = logging.getLogger(__name__)

# Side branch a push lands on when the target branch moved since the push's
# base commit; it is merged and deleted, or kept for a pull request on conflict.
SYNC_BRANCH_PREFIX = "besser/sync-"

# Git tree modes of a regular file; links (120000) and submodules (160000) are never pushed.
_FILE_MODE = "100644"
_EXECUTABLE_MODE = "100755"
_FILE_MODES = (_FILE_MODE, _EXECUTABLE_MODE)

# GitHub's largest page. A listing stops after this many pages, so a huge
# account or repo costs a bounded number of requests.
_PAGE_SIZE = 100
_MAX_REPO_PAGES = 10
_MAX_BRANCH_PAGES = 10


# A commit retries once when the branch moved between reading its head and
# updating it; moving again within that window is a real race, reported as one.
_COMMIT_ATTEMPTS = 2


def git_blob_sha(content: bytes) -> str:
    """The object id git gives a blob holding ``content``, to tell an unchanged file without uploading it."""
    return hashlib.sha1(b"blob %d\0" % len(content) + content).hexdigest()


def content_hash(content: bytes) -> str:
    """The hash a run records of each file it got from a repo (``push_directory_to_repo``'s ``base_hashes``)."""
    return hashlib.sha256(content).hexdigest()


def _merge_message(branch: str, commit_message: str) -> str:
    """The merge commit carries the ``BESSER-Origin`` of the change it merges, so history shows it as BESSER's."""
    message = f"Merge BESSER changes into {branch}"
    origin = commit_origin(commit_message)
    return with_origin(message, origin) if origin else message


def _branch_moved(exc: httpx.HTTPStatusError) -> bool:
    """True for GitHub refusing a ref update that is no longer a fast-forward (someone pushed meanwhile)."""
    return (exc.response is not None and exc.response.status_code == 422
            and exc.request.method == "PATCH" and "/git/refs/heads/" in exc.request.url.path)


class GitHubService:
    """Service for interacting with GitHub API."""

    def __init__(self, access_token: str):
        """
        Initialize GitHub service with user's access token.

        Args:
            access_token: GitHub OAuth access token
        """
        self.access_token = access_token
        self.base_url = "https://api.github.com"
        self.headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"
        }

    async def get_authenticated_user(self) -> Dict[str, Any]:
        """
        Get information about the authenticated user.

        Returns:
            User information dictionary
        """
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/user",
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
            return response.json()

    # Default hard cap on the downloaded (compressed) archive. A committed
    # ``node_modules`` / ``target`` could otherwise pull tens of thousands
    # of files and hundreds of MB into the process.
    _DEFAULT_MAX_ARCHIVE_BYTES = 100 * 1024 * 1024  # 100 MB

    # Companion cap on the *uncompressed* tree. The cap above bounds the
    # download, not the expansion: 100 MB of gzip holds ~100 GB. Source gzips
    # 3-5x, so 5x the compressed cap is the lowest ceiling that still admits
    # every archive the download already accepted.
    _DEFAULT_MAX_EXTRACTED_BYTES = 500 * 1024 * 1024  # 500 MB

    # Companion cap on member count. Empty files weigh nothing against the
    # byte cap but still cost an inode each, and the extracted tree is copied
    # again as a run seed and again into the packaging zip.
    _DEFAULT_MAX_ARCHIVE_MEMBERS = 50_000

    async def download_repo_tarball(
        self,
        owner: str,
        repo: str,
        ref: str,
        max_archive_bytes: int = _DEFAULT_MAX_ARCHIVE_BYTES,
        dest_dir: Optional[str] = None,
    ) -> str:
        """Download a repository tarball for ``ref`` and extract it locally.

        Calls ``GET /repos/{owner}/{repo}/tarball/{ref}``. GitHub answers
        with a 302 redirect to a pre-signed ``codeload.github.com`` URL;
        ``follow_redirects=True`` follows it and httpx correctly drops the
        ``Authorization`` header on the cross-host hop (the codeload URL is
        already signed, and forwarding the token would be rejected).

        The archive is streamed with a hard size cap (``max_archive_bytes``,
        default 100 MB) so a committed dependency directory can't blow up
        the process; extraction applies the companion uncompressed-size and
        member-count caps. GitHub wraps the whole repo under a single top-level
        ``{owner}-{repo}-{sha}/`` directory; that wrapper is stripped during
        extraction so the returned path *is* the repo root.

        Args:
            owner: Repository owner username.
            repo: Repository name.
            ref: Branch name, tag, or commit SHA.
            max_archive_bytes: Abort the download once the streamed archive
                exceeds this many bytes.
            dest_dir: Existing empty directory to extract into. When omitted a
                fresh temp directory is created.

        Returns:
            Path to the directory holding the extracted repo root. The caller
            owns cleanup of this directory.
        """
        url = f"{self.base_url}/repos/{owner}/{repo}/tarball/{ref}"

        buffer = io.BytesIO()
        total = 0
        async with httpx.AsyncClient(follow_redirects=True) as client:
            async with client.stream(
                "GET", url, headers=self.headers, timeout=60
            ) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    total += len(chunk)
                    if total > max_archive_bytes:
                        raise ValueError(
                            "Repository archive exceeds the "
                            f"{max_archive_bytes // (1024 * 1024)} MB size cap; "
                            "refusing to download."
                        )
                    buffer.write(chunk)

        buffer.seek(0)
        extract_root = dest_dir or tempfile.mkdtemp(prefix="besser_gh_import_")
        try:
            with tarfile.open(fileobj=buffer, mode="r:*") as tar:
                self._extract_tarball_stripping_root(tar, extract_root)
        except Exception:
            shutil.rmtree(extract_root, ignore_errors=True)
            raise

        return extract_root

    @staticmethod
    def _extract_tarball_stripping_root(
        tar: tarfile.TarFile,
        dest_dir: str,
        max_extracted_bytes: int = _DEFAULT_MAX_EXTRACTED_BYTES,
        max_members: int = _DEFAULT_MAX_ARCHIVE_MEMBERS,
    ) -> None:
        """Extract ``tar`` into ``dest_dir``, stripping the single top-level dir.

        GitHub tarballs nest everything under one ``{owner}-{repo}-{sha}/``
        directory; that leading path component is removed so files land at
        the repo root. Path-traversal entries (``..`` / absolute paths) are
        rejected, and non-regular members (symlinks, hardlinks, devices)
        are skipped so a crafted archive cannot write outside ``dest_dir``.

        The download cap is on *compressed* bytes, so this scan also totals
        the declared uncompressed sizes and counts the members it keeps.
        Both are checked before ``extractall``, so an over-limit archive
        writes nothing.
        """
        dest_abs = os.path.abspath(dest_dir)
        safe_members = []
        extracted_bytes = 0
        # Iterate lazily: a member-count explosion aborts before the whole
        # member list is materialised.
        for member in tar:
            # Drop the leading ``{owner}-{repo}-{sha}/`` component.
            parts = member.name.split("/", 1)
            if len(parts) == 1:
                # The wrapper directory entry itself — nothing to keep.
                continue
            relative = parts[1]
            if not relative:
                continue
            # Only extract regular files and directories; skip links/devices.
            if not (member.isfile() or member.isdir()):
                continue
            target = os.path.abspath(os.path.join(dest_abs, relative))
            if target != dest_abs and not target.startswith(dest_abs + os.sep):
                raise ValueError(
                    f"Unsafe path in repository archive: {member.name!r}"
                )
            if len(safe_members) >= max_members:
                raise ValueError(
                    f"This repository holds more than {max_members:,} files, "
                    "more than BESSER can import. Remove committed dependency "
                    "or build directories (node_modules, target, dist) and "
                    "import again."
                )
            extracted_bytes += member.size
            if extracted_bytes > max_extracted_bytes:
                raise ValueError(
                    "This repository unpacks to more than "
                    f"{max_extracted_bytes // (1024 * 1024)} MB, more than "
                    "BESSER can import (the limit was reached at "
                    f"{relative!r}). Remove large files or committed "
                    "dependency directories (node_modules, target, dist) "
                    "and import again."
                )
            member.name = relative
            safe_members.append(member)
        # The ``data`` filter (3.12+, backported to 3.8.17 / 3.9.17 / 3.10.12 /
        # 3.11.4) is defence in depth on top of the scan above: it also refuses
        # absolute paths, traversal and links, and drops special mode bits.
        # Passing it explicitly silences the 3.12 DeprecationWarning and pins
        # the behaviour 3.14 makes the default.
        if hasattr(tarfile, "data_filter"):
            tar.extractall(dest_dir, members=safe_members, filter="data")
        else:  # pragma: no cover - interpreter without PEP 706 support
            tar.extractall(dest_dir, members=safe_members)

    async def create_repository(
        self,
        repo_name: str,
        description: str = "Web application generated by BESSER",
        is_private: bool = True,
        auto_init: bool = False
    ) -> Dict[str, Any]:
        """
        Create a new GitHub repository in user's account.

        Args:
            repo_name: Name for the repository
            description: Repository description
            is_private: Whether to make the repository private

        Returns:
            Repository information dictionary
        """
        data = {
            "name": repo_name,
            "description": description,
            "private": is_private,
            "auto_init": auto_init,
            "has_issues": True,
            "has_projects": False,
            "has_wiki": False
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/user/repos",
                headers=self.headers,
                json=data,
                timeout=10
            )
            response.raise_for_status()
            return response.json()

    async def update_repository(
        self,
        owner: str,
        repo_name: str,
        description: str | None = None,
        homepage: str | None = None,
        topics: list[str] | None = None
    ) -> Dict[str, Any]:
        """
        Update repository metadata (description, homepage URL, topics).

        Args:
            owner: Repository owner username
            repo_name: Repository name
            description: Repository description
            homepage: Homepage URL shown in the About section
            topics: List of topic tags

        Returns:
            Updated repository information
        """
        data = {}
        if description is not None:
            data["description"] = description
        if homepage is not None:
            data["homepage"] = homepage

        async with httpx.AsyncClient() as client:
            if data:
                response = await client.patch(
                    f"{self.base_url}/repos/{owner}/{repo_name}",
                    headers=self.headers,
                    json=data,
                    timeout=10
                )
                response.raise_for_status()

            # Topics use a separate endpoint
            if topics is not None:
                topic_headers = {**self.headers, "Accept": "application/vnd.github+json"}
                await client.put(
                    f"{self.base_url}/repos/{owner}/{repo_name}/topics",
                    headers=topic_headers,
                    json={"names": topics},
                    timeout=10
                )

        return {"owner": owner, "repo_name": repo_name}

    async def delete_repository(self, owner: str, repo_name: str) -> bool:
        """
        Delete a repository (use with caution!).

        Args:
            owner: Repository owner username
            repo_name: Repository name

        Returns:
            True if deleted successfully
        """
        async with httpx.AsyncClient() as client:
            response = await client.delete(
                f"{self.base_url}/repos/{owner}/{repo_name}",
                headers=self.headers,
                timeout=10
            )
            return response.status_code == 204

    async def push_directory_to_repo(
        self,
        owner: str,
        repo_name: str,
        directory_path: str,
        commit_message: str = "Initial commit - Generated by BESSER",
        branch: str = "main",
        preserve_existing_files: bool = False,
        base_commit_sha: Optional[str] = None,
        keep_path: Optional[Callable[[str], bool]] = None,
        model_merge: Optional[Any] = None,
        base_hashes: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Commit a directory to ``branch`` as a change to the commit it was derived from.

        Only what differs from that commit is sent: a file whose content and
        mode match it uploads nothing, a changed file keeps its mode (an
        executable stays executable), and symlinks and submodules in the repo
        are never touched, since the directory cannot hold them. A repository
        with no commit yet gets its first one through the contents API.

        Args:
            owner: Repository owner username
            repo_name: Repository name
            directory_path: Path to directory containing files to push
            commit_message: Commit message
            branch: Target branch name
            preserve_existing_files: If True, files already in the repo that
                are not in the directory are kept. If False, they are deleted,
                except those ``keep_path`` keeps.
            base_commit_sha: The commit this directory was derived from (else
                the branch head). When the branch has moved past it, the push
                is merged into the branch (see ``commit_tree``), so commits
                made since are kept instead of reverted.
            keep_path: True for a repo path the directory never held (build
                output, a secret file it was not given), so its absence is not
                a deletion.
            model_merge: Passed on to ``commit_tree``.
            base_hashes: ``path -> content_hash`` of each file as the directory
                received it from ``base_commit_sha``. A file that still hashes
                the same is unchanged even when its bytes differ from the blob:
                the archive a run is imported from applies ``.gitattributes``
                (``eol``, ``export-subst``, ``ident``) and LFS on the way out.

        Returns:
            ``total_files`` (paths added, changed or deleted), ``uploaded_files``,
            ``deleted_files`` and the ``commit_tree`` outcome.
        """
        local = self._collect_push_files(directory_path)
        if not local:
            raise RuntimeError("No files found to upload.")
        # A plain file, so the contents API's 100644 is its mode.
        first = next((path for path in sorted(local) if not local[path][1]), min(local))
        await self.ensure_first_commit(owner, repo_name, branch, first, local[first][0], commit_message)
        head_sha = await self._get_ref_sha(owner, repo_name, branch)
        parent_sha = base_commit_sha or head_sha
        parent = await self.get_tree_entries(owner, repo_name, parent_sha) if parent_sha else {}
        base_hashes = base_hashes or {}

        tree_entries = []
        uploaded = []
        for path, (content, executable) in local.items():
            old = parent.get(path)
            kept_mode = old["mode"] if old and old["mode"] in _FILE_MODES else None
            mode = kept_mode or (_EXECUTABLE_MODE if executable else _FILE_MODE)
            unchanged = old and (old["sha"] == git_blob_sha(content) or base_hashes.get(path) == content_hash(content))
            if unchanged and old["mode"] == mode:
                continue
            blob_sha = await self.create_blob(owner=owner, repo_name=repo_name, content=content)
            if not blob_sha:
                raise RuntimeError(f"GitHub did not return a blob SHA for {path}.")
            tree_entries.append({"path": path, "mode": mode, "type": "blob", "sha": blob_sha})
            uploaded.append(path)

        deleted = []
        if not preserve_existing_files:
            for path, old in parent.items():
                if old["mode"] in _FILE_MODES and path not in local and not (keep_path and keep_path(path)):
                    tree_entries.append({"path": path, "mode": old["mode"], "type": "blob", "sha": None})
                    deleted.append(path)

        outcome = await self.commit_tree(
            owner, repo_name, branch, tree_entries, commit_message,
            layer_on_parent=True,
            base_commit_sha=parent_sha,
            model_merge=model_merge,
        )
        return {
            "uploaded_files": uploaded,
            "deleted_files": deleted,
            "total_files": len(uploaded) + len(deleted),
            **outcome,
        }

    @staticmethod
    def _collect_push_files(directory_path: str) -> Dict[str, tuple[bytes, bool]]:
        """``repo path -> (content, executable)`` for the regular files under ``directory_path``.

        The tree is a Spec-Driven workspace written by model-authored code. A
        symlink in it (``ln -s /proc/self/environ leak.txt``) would be read
        through and its target committed by a backend holding the full secret
        set, so symlinks are skipped and every file must resolve inside the tree.
        """
        directory = Path(directory_path).resolve()
        files: Dict[str, tuple[bytes, bool]] = {}
        skipped_unsafe = 0
        for candidate in directory.rglob("*"):
            if candidate.is_symlink():
                skipped_unsafe += 1
                continue
            if not candidate.is_file():
                continue
            try:
                resolved = candidate.resolve()
                if os.path.commonpath([str(directory), str(resolved)]) != str(directory):
                    skipped_unsafe += 1
                    continue
                content = candidate.read_bytes()
                executable = bool(candidate.stat().st_mode & stat.S_IXUSR)
            except (OSError, ValueError):
                skipped_unsafe += 1
                continue
            files[candidate.relative_to(directory).as_posix()] = (content, executable)
        if skipped_unsafe:
            logger.warning(
                "Skipped %d symlinked/out-of-tree entr%s while pushing %s",
                skipped_unsafe, "y" if skipped_unsafe == 1 else "ies", directory,
            )
        return files

    async def get_tree_entries(self, owner: str, repo_name: str, commit_sha: str) -> Dict[str, Dict[str, str]]:
        """``path -> {mode, type, sha}`` of every blob, symlink and submodule at ``commit_sha``."""
        tree_sha = await self._get_commit_tree_sha(owner, repo_name, commit_sha)
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/repos/{owner}/{repo_name}/git/trees/{tree_sha}",
                headers=self.headers,
                params={"recursive": "1"},
                timeout=60
            )
            response.raise_for_status()
            data = response.json()
        if data.get("truncated"):
            # GitHub caps a recursive listing (100,000 entries / 7 MB); a partial
            # listing would turn every unlisted file into a deletion.
            raise RuntimeError(f"{owner}/{repo_name} has too many files for BESSER to update it.")
        return {
            entry["path"]: {"mode": entry["mode"], "type": entry["type"], "sha": entry["sha"]}
            for entry in data.get("tree", [])
            if entry.get("type") in ("blob", "commit")
        }

    async def commit_tree(
        self,
        owner: str,
        repo_name: str,
        branch: str,
        tree_entries: List[Dict[str, Any]],
        commit_message: str,
        layer_on_parent: bool,
        base_commit_sha: Optional[str] = None,
        model_merge: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Commit ``tree_entries`` to ``branch`` without reverting newer commits.

        The commit's parent is ``base_commit_sha`` when given (the commit the
        caller's files were derived from), else the branch head. When the
        branch is still at that parent the branch is fast-forwarded. When it
        has moved on (someone pushed since), the commit goes on a side branch
        and GitHub's merge API merges it into ``branch``: both sides' changes
        are kept. A merge GitHub cannot resolve leaves the side branch in place
        for a pull request.

        Args:
            layer_on_parent: Keep the parent's files that ``tree_entries`` does
                not list. When False the tree replaces the parent's.
            tree_entries: Git tree entries; an entry with ``"sha": None``
                deletes that path (only meaningful with ``layer_on_parent``).
            model_merge: A ``repo_model.ModelMerge`` when the entries hold the
                BESSER model: on a merge its files are merged as a model rather
                than as text (see ``_merge_with_model``).

        Returns:
            ``commit_sha`` (the branch head after the push, or the side-branch
            commit on a conflict), ``status`` (``"pushed"`` fast-forward,
            ``"merged"`` merged with newer commits, ``"conflict"`` not merged),
            ``conflict_branch`` (the side branch on a conflict, else None),
            ``own_commit_sha`` (the commit holding exactly ``tree_entries``,
            which differs from ``commit_sha`` after a merge) and ``warnings``
            (what the user should know about the merge).

        When the branch moves between reading its head and updating it, the
        commit is made again on the new head, once; GitHub's 422 is raised if
        it moved again.
        """
        for attempt in range(1, _COMMIT_ATTEMPTS + 1):
            try:
                return await self._commit_tree_once(
                    owner, repo_name, branch, tree_entries, commit_message,
                    layer_on_parent, base_commit_sha, model_merge,
                )
            except httpx.HTTPStatusError as exc:
                if not _branch_moved(exc) or attempt == _COMMIT_ATTEMPTS:
                    raise
                logger.warning("%s/%s@%s moved while committing; committing again on its new head",
                               owner, repo_name, branch)
        raise AssertionError("unreachable")

    async def _commit_tree_once(
        self, owner: str, repo_name: str, branch: str, tree_entries: List[Dict[str, Any]],
        commit_message: str, layer_on_parent: bool, base_commit_sha: Optional[str], model_merge: Optional[Any],
    ) -> Dict[str, Any]:
        """One attempt of ``commit_tree``, against the branch head as it is now."""
        head_sha = await self._get_ref_sha(owner, repo_name, branch)
        parent_sha = base_commit_sha or head_sha

        if not tree_entries and layer_on_parent and head_sha:
            # Nothing differs from the parent: the branch already holds it, merged
            # with anything newer. No commit, so the history gets no empty one.
            status = "pushed" if head_sha == parent_sha else "merged"
            return {"commit_sha": head_sha, "status": status, "conflict_branch": None,
                    "own_commit_sha": parent_sha, "warnings": []}

        base_tree_sha = None
        if layer_on_parent and parent_sha:
            base_tree_sha = await self._get_commit_tree_sha(owner, repo_name, parent_sha)

        tree_sha = await self._create_tree(owner, repo_name, tree_entries, base_tree=base_tree_sha)
        commit_sha = await self._create_commit(
            owner=owner,
            repo_name=repo_name,
            message=commit_message,
            tree_sha=tree_sha,
            parent_sha=parent_sha
        )

        if not head_sha or head_sha == parent_sha:
            await self._update_or_create_ref(owner, repo_name, branch, commit_sha, head_sha)
            return {"commit_sha": commit_sha, "status": "pushed", "conflict_branch": None,
                    "own_commit_sha": commit_sha, "warnings": []}

        if model_merge is not None and any(model_merge.is_model_path(e["path"]) for e in tree_entries):
            return await self._merge_with_model(
                owner, repo_name, branch, head_sha, parent_sha, base_tree_sha,
                commit_sha, commit_message, tree_entries, model_merge,
            )

        side_branch = f"{SYNC_BRANCH_PREFIX}{commit_sha[:7]}"
        await self._update_or_create_ref(owner, repo_name, side_branch, commit_sha, None)
        merge_sha = await self.merge_branch(
            owner, repo_name, base=branch, head=side_branch,
            commit_message=_merge_message(branch, commit_message),
        )
        if merge_sha is None:
            return await self._keep_for_pull_request(owner, repo_name, branch, commit_sha)

        await self.delete_branch(owner, repo_name, side_branch)
        return {"commit_sha": merge_sha, "status": "merged", "conflict_branch": None,
                "own_commit_sha": commit_sha, "warnings": []}

    async def _merge_with_model(
        self, owner: str, repo_name: str, branch: str, head_sha: str, parent_sha: str,
        base_tree_sha: Optional[str], own_sha: str, commit_message: str,
        tree_entries: List[Dict[str, Any]], model_merge: Any,
    ) -> Dict[str, Any]:
        """Merge ``own_sha`` into ``branch``: the code by GitHub, the model by ``model_merge``.

        Git merges ``diagrams.json`` and the B-UML sources as text, so two
        people adding a class each conflict. Here GitHub merges everything but
        the model files on a scratch branch; the model is merged as data and
        its files are written from the result. The branch only moves, by a
        fast-forward, once both merges succeeded, to a merge commit whose
        parents are the branch head and ``own_sha``.
        """
        code_entries = [e for e in tree_entries if not model_merge.is_model_path(e["path"])]
        scratch = f"{SYNC_BRANCH_PREFIX}merge-{own_sha[:7]}"
        await self._update_or_create_ref(owner, repo_name, scratch, head_sha, None)
        try:
            merged_head = head_sha
            if code_entries:
                code_tree = await self._create_tree(owner, repo_name, code_entries, base_tree=base_tree_sha)
                code_sha = await self._create_commit(owner, repo_name, commit_message, code_tree, parent_sha)
                code_branch = f"{SYNC_BRANCH_PREFIX}code-{code_sha[:7]}"
                await self._update_or_create_ref(owner, repo_name, code_branch, code_sha, None)
                merged_head = await self.merge_branch(
                    owner, repo_name, base=scratch, head=code_branch, commit_message=commit_message,
                )
                await self.delete_branch(owner, repo_name, code_branch)
            merged = None
            if merged_head is not None:
                base_file = await self.get_file_content(owner, repo_name, model_merge.model_file, ref=parent_sha)
                theirs_file = await self.get_file_content(owner, repo_name, model_merge.model_file, ref=head_sha)
                merged = model_merge.merge(
                    base_file and base_file.get("content"), theirs_file and theirs_file.get("content"),
                )
            if merged is None:
                return await self._keep_for_pull_request(owner, repo_name, branch, own_sha)

            current = await self.get_tree_entries(owner, repo_name, merged_head)
            entries = [
                {"path": path, "mode": _FILE_MODE, "type": "blob",
                 "sha": await self.create_blob(owner, repo_name, content)}
                for path, content in merged.files.items()
                if (current.get(path) or {}).get("sha") != git_blob_sha(content)
            ]
            # Model files the merged model no longer produces (a removed agent's source).
            stale = [path for path, entry in current.items()
                     if model_merge.is_model_path(path) and path not in merged.files and entry["mode"] in _FILE_MODES]
            warnings = []
            if stale and not merged.sources_built:
                logger.warning("Kept %s in %s/%s: the merged model's B-UML export failed", stale, owner, repo_name)
                warnings.append("BESSER could not write the B-UML sources of the merged model, so the "
                                f"previous ones were kept and may be out of date: {', '.join(sorted(stale))}.")
                stale = []
            entries += [{"path": path, "mode": current[path]["mode"], "type": "blob", "sha": None} for path in stale]
            merged_tree = await self._get_commit_tree_sha(owner, repo_name, merged_head)
            if entries:
                merged_tree = await self._create_tree(owner, repo_name, entries, base_tree=merged_tree)
            merge_sha = await self._create_commit(
                owner, repo_name, _merge_message(branch, commit_message), merged_tree, head_sha,
                second_parent=own_sha,
            )
            # A fast-forward: it fails, rather than reverting, if the branch moved again meanwhile.
            await self._update_or_create_ref(owner, repo_name, branch, merge_sha, head_sha)
            return {"commit_sha": merge_sha, "status": "merged", "conflict_branch": None,
                    "own_commit_sha": own_sha, "warnings": warnings}
        finally:
            await self.delete_branch(owner, repo_name, scratch)

    async def _keep_for_pull_request(self, owner: str, repo_name: str, branch: str, own_sha: str) -> Dict[str, Any]:
        """Leave an unmergeable commit on its own branch, for a pull request."""
        side_branch = f"{SYNC_BRANCH_PREFIX}{own_sha[:7]}"
        await self._update_or_create_ref(owner, repo_name, side_branch, own_sha, None)
        logger.warning(
            "Could not merge %s into %s/%s@%s; left it on %s for a pull request",
            own_sha[:7], owner, repo_name, branch, side_branch,
        )
        return {"commit_sha": own_sha, "status": "conflict", "conflict_branch": side_branch,
                "own_commit_sha": own_sha, "warnings": []}

    async def _get_commit_tree_sha(self, owner: str, repo_name: str, commit_sha: str) -> Optional[str]:
        """Return the tree SHA of ``commit_sha``."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/repos/{owner}/{repo_name}/git/commits/{commit_sha}",
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
            return response.json().get("tree", {}).get("sha")

    async def get_branch_head(self, owner: str, repo_name: str, branch: str) -> Optional[str]:
        """Return the commit SHA ``branch`` points at, or None when it does not exist."""
        return await self._get_ref_sha(owner, repo_name, branch)

    async def merge_branch(
        self, owner: str, repo_name: str, base: str, head: str, commit_message: str
    ) -> Optional[str]:
        """Merge branch ``head`` into ``base`` with GitHub's merge API.

        Returns the resulting head SHA of ``base`` (the merge commit, or the
        unchanged head when there was nothing to merge), or None on a merge
        conflict.
        """
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/repos/{owner}/{repo_name}/merges",
                headers=self.headers,
                json={"base": base, "head": head, "commit_message": commit_message},
                timeout=30
            )
        if response.status_code == 409:
            return None
        if response.status_code == 204:
            return await self._get_ref_sha(owner, repo_name, base)
        response.raise_for_status()
        return response.json().get("sha")

    async def delete_branch(self, owner: str, repo_name: str, branch: str) -> None:
        """Delete ``branch``. A failure is logged, not raised: the push already landed."""
        async with httpx.AsyncClient() as client:
            response = await client.delete(
                f"{self.base_url}/repos/{owner}/{repo_name}/git/refs/heads/{branch}",
                headers=self.headers,
                timeout=10
            )
        if response.status_code not in (204, 404, 422):
            logger.warning(
                "Could not delete branch %s in %s/%s (HTTP %s)",
                branch, owner, repo_name, response.status_code,
            )

    async def create_pull_request(
        self, owner: str, repo_name: str, head: str, base: str, title: str, body: str
    ) -> Dict[str, Any]:
        """Open a pull request from ``head`` into ``base``; returns ``number`` and ``html_url``.

        When ``head`` already has an open pull request, that one is returned.
        """
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/repos/{owner}/{repo_name}/pulls",
                headers=self.headers,
                json={"head": head, "base": base, "title": title, "body": body},
                timeout=15
            )
        if response.status_code == 422:
            existing = await self.get_pull_request_for_branch(owner, repo_name, head)
            if existing and existing["state"] == "open":
                return {"number": existing["number"], "html_url": existing["html_url"]}
        response.raise_for_status()
        data = response.json()
        return {"number": data.get("number"), "html_url": data.get("html_url", "")}

    async def get_pull_request_for_branch(self, owner: str, repo_name: str, branch: str) -> Optional[Dict[str, Any]]:
        """The newest pull request from ``branch`` (open, closed or merged), or None."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/repos/{owner}/{repo_name}/pulls",
                headers=self.headers,
                params={"head": f"{owner}:{branch}", "state": "all", "per_page": 1},
                timeout=15
            )
            response.raise_for_status()
            pulls = response.json()
        if not pulls:
            return None
        pull = pulls[0]
        return {
            "number": pull.get("number"),
            "html_url": pull.get("html_url", ""),
            "state": pull.get("state", ""),
            "merged": bool(pull.get("merged_at")),
        }

    async def compare_commits(
        self, owner: str, repo_name: str, base: str, head: str
    ) -> Dict[str, Any]:
        """Compare ``base...head``: how many commits ``head`` is ahead/behind and which."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/repos/{owner}/{repo_name}/compare/{base}...{head}",
                headers=self.headers,
                params={"per_page": 100},
                timeout=15
            )
            response.raise_for_status()
            data = response.json()
        return {
            "status": data.get("status", ""),
            "ahead_by": data.get("ahead_by", 0),
            "behind_by": data.get("behind_by", 0),
            "commits": [self._simplify_commit(c) for c in data.get("commits", [])],
        }

    async def add_topic(self, owner: str, repo_name: str, topic: str) -> None:
        """Add ``topic`` to the repository's topics, keeping the existing ones.

        Best effort: a failure is logged, not raised.
        """
        url = f"{self.base_url}/repos/{owner}/{repo_name}/topics"
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=self.headers, timeout=10)
                response.raise_for_status()
                names = response.json().get("names", [])
                if topic in names:
                    return
                response = await client.put(
                    url, headers=self.headers, json={"names": [*names, topic]}, timeout=10
                )
                response.raise_for_status()
        except httpx.HTTPError:
            logger.warning("Could not add topic %r to %s/%s", topic, owner, repo_name, exc_info=True)

    async def _upload_file(
        self,
        owner: str,
        repo_name: str,
        file_path: str,
        content_b64: str,
        message: str,
        branch: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Upload a single file to GitHub repository.

        Args:
            owner: Repository owner
            repo_name: Repository name
            file_path: Path within repository
            content_b64: Base64-encoded file content
            message: Commit message
            branch: Branch to commit to (the default branch when None)

        Returns:
            Response from GitHub API
        """
        data = {
            "message": message,
            "content": content_b64
        }
        if branch:
            data["branch"] = branch

        async with httpx.AsyncClient() as client:
            response = await client.put(
                f"{self.base_url}/repos/{owner}/{repo_name}/contents/{file_path}",
                headers=self.headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()
            return response.json()

    async def create_readme(
        self,
        owner: str,
        repo_name: str,
        app_name: str,
        deploy_instructions: bool = True
    ) -> Dict[str, Any]:
        """
        Create a README.md file with deployment instructions.

        Args:
            owner: Repository owner
            repo_name: Repository name
            app_name: Application name
            deploy_instructions: Include deployment instructions

        Returns:
            GitHub API response
        """
        readme_content = self.generate_readme_content(app_name, deploy_instructions)
        content_b64 = base64.b64encode(readme_content.encode("utf-8")).decode("utf-8")

        return await self._upload_file(
            owner=owner,
            repo_name=repo_name,
            file_path="README.md",
            content_b64=content_b64,
            message="Add README with deployment instructions"
        )

    def generate_readme_content(
        self,
        app_name: str,
        deploy_instructions: bool = True,
        owner: str = "",
        repo_name: str = "",
        deployment_type: str = "webapp",
    ) -> str:
        """Generate README.md content."""
        if deployment_type == "chatbot":
            readme = f"""<div align=\"center\">

# {app_name}

[![Generated by BESSER](https://img.shields.io/badge/Generated%20by-BESSER-blue?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZD0iTTEyIDJMMiA3bDEwIDUgMTAtNS0xMC01ek0yIDE3bDEwIDUgMTAtNS0xMC01LTEwIDV6TTIgMTJsMTAgNSAxMC01LTEwLTUtMTAgNXoiIGZpbGw9IndoaXRlIi8+PC9zdmc+)](https://github.com/BESSER-PEARL/BESSER)
[![BESSER Editor](https://img.shields.io/badge/Built%20with-BESSER%20Editor-green)](https://editor.besser-pearl.org)
"""

            if deploy_instructions and owner and repo_name:
                readme += f"""[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/{owner}/{repo_name})
"""

            readme += """
A standalone chatbot generated from an Agent Diagram using the [BESSER Web Modeling Editor](https://editor.besser-pearl.org).

</div>

## Project Structure

```
<agent_name>.py    Generated BAF agent script
config.yaml        Runtime configuration (platforms, NLP, DB toggles)
render.yaml        Render deployment blueprint
buml/              Source B-UML exports
```

## Deploy

Click the **Deploy to Render** button above to deploy your chatbot.

The Render blueprint runs one Python web service with Streamlit UI and websocket runtime.

## Local Development

```bash
pip install besser-agentic-framework[llms]==4.3.2
python <agent_name>.py
```

## About BESSER

[BESSER](https://github.com/BESSER-PEARL/BESSER) is an open-source low-code platform that generates applications from visual models.

[Documentation](https://besser.readthedocs.io/) · [Web Editor](https://editor.besser-pearl.org) · [Examples](https://github.com/BESSER-PEARL/BESSER-examples)

If BESSER saved you time, a [⭐ on GitHub](https://github.com/BESSER-PEARL/BESSER) helps others find it.
"""
            return readme

        readme = f"""<div align="center">

# {app_name}

[![Generated by BESSER](https://img.shields.io/badge/Generated%20by-BESSER-blue?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZD0iTTEyIDJMMiA3bDEwIDUgMTAtNS0xMC01ek0yIDE3bDEwIDUgMTAtNS0xMC01LTEwIDV6TTIgMTJsMTAgNSAxMC01LTEwLTUtMTAgNXoiIGZpbGw9IndoaXRlIi8+PC9zdmc+)](https://github.com/BESSER-PEARL/BESSER)
[![BESSER Editor](https://img.shields.io/badge/Built%20with-BESSER%20Editor-green)](https://editor.besser-pearl.org)
"""

        if deploy_instructions and owner and repo_name:
            readme += f"""[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/{owner}/{repo_name})
"""

        readme += """
A full-stack web application generated from visual models using the [BESSER Web Modeling Editor](https://editor.besser-pearl.org), a low-code open-source platform.

</div>

"""

        if deploy_instructions:
            readme += """## Project Structure

```
frontend/          React + TypeScript frontend (Vite)
backend/           FastAPI Python backend
agents/<name>/     BESSER conversational agents (one subfolder per agent, if any)
render.yaml        Render deployment config
docker-compose.yml Local development setup
```

## Deploy

Click the **Deploy to Render** button above to launch your app for free.

Render will automatically set up your frontend, backend, and agent services.
After the first deploy, pushing updates to this repo will auto-redeploy.

## Local Development

**Frontend:**
```bash
cd frontend && npm install && npm start
```

**Backend:**
```bash
cd backend && pip install -r requirements.txt && uvicorn main_api:app --reload
```

**Full stack (Docker):**
```bash
docker-compose up
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React, TypeScript, Vite |
| Backend | FastAPI, SQLAlchemy, Python 3.11 |
| Agent | BESSER Agentic Framework (WebSocket) |
| Hosting | Render (free tier) |

## About BESSER

[BESSER](https://github.com/BESSER-PEARL/BESSER) is an open-source low-code platform that generates full-stack applications from visual models.

[Documentation](https://besser.readthedocs.io/) · [Web Editor](https://editor.besser-pearl.org) · [Examples](https://github.com/BESSER-PEARL/BESSER-examples)

If BESSER saved you time, a [⭐ on GitHub](https://github.com/BESSER-PEARL/BESSER) helps others find it.
"""

        return readme

    async def create_blob(self, owner: str, repo_name: str, content: bytes) -> str:
        """Create a blob and return its SHA."""
        data = {
            "content": base64.b64encode(content).decode("utf-8"),
            "encoding": "base64"
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/repos/{owner}/{repo_name}/git/blobs",
                headers=self.headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()
            return response.json().get("sha", "")

    async def _create_tree(self, owner: str, repo_name: str, tree: List[Dict[str, Any]], base_tree: Optional[str] = None) -> str:
        """Create a tree and return its SHA.

        Args:
            base_tree: If provided, the new tree is layered on top of this
                       existing tree SHA, preserving files not in ``tree``.
        """
        payload: Dict[str, Any] = {"tree": tree}
        if base_tree:
            payload["base_tree"] = base_tree
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/repos/{owner}/{repo_name}/git/trees",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json().get("sha", "")

    async def _create_commit(
        self,
        owner: str,
        repo_name: str,
        message: str,
        tree_sha: str,
        parent_sha: Optional[str],
        second_parent: Optional[str] = None,
    ) -> str:
        """Create a commit and return its SHA; ``second_parent`` makes it a merge commit."""
        payload = {
            "message": message,
            "tree": tree_sha,
            "parents": [sha for sha in (parent_sha, second_parent) if sha]
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/repos/{owner}/{repo_name}/git/commits",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json().get("sha", "")

    async def ensure_first_commit(
        self, owner: str, repo_name: str, branch: str, path: str, content: bytes, message: str,
    ) -> None:
        """Give a repository with no commit its first one: ``path`` holding ``content``, on ``branch``.

        GitHub's git data API (blobs, trees, commits, refs) answers 409 "Git
        Repository is empty" until a repository has a commit; the contents API
        can make that first commit. A repository that has commits is left alone.
        """
        if await self._get_ref_sha(owner, repo_name, branch):
            return
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/repos/{owner}/{repo_name}/branches",
                headers=self.headers,
                params={"per_page": 1},
                timeout=10
            )
            response.raise_for_status()
        if response.json():
            return
        logger.info("%s/%s is empty; making its first commit through the contents API", owner, repo_name)
        await self._upload_file(owner, repo_name, path, base64.b64encode(content).decode("utf-8"), message, branch)

    async def _get_ref_sha(self, owner: str, repo_name: str, branch: str) -> Optional[str]:
        """Get the current SHA for a branch ref, or None if it doesn't exist (or the repo has no commit)."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/repos/{owner}/{repo_name}/git/ref/heads/{branch}",
                headers=self.headers,
                timeout=10
            )
            # 409: "Git Repository is empty".
            if response.status_code in (404, 409):
                return None
            response.raise_for_status()
            return response.json().get("object", {}).get("sha")

    async def _update_or_create_ref(
        self,
        owner: str,
        repo_name: str,
        branch: str,
        commit_sha: str,
        parent_sha: Optional[str]
    ) -> None:
        """Update the branch ref if it exists, otherwise create it."""
        async with httpx.AsyncClient() as client:
            if parent_sha:
                response = await client.patch(
                    f"{self.base_url}/repos/{owner}/{repo_name}/git/refs/heads/{branch}",
                    headers=self.headers,
                    json={"sha": commit_sha, "force": False},
                    timeout=10
                )
                response.raise_for_status()
                return

            response = await client.post(
                f"{self.base_url}/repos/{owner}/{repo_name}/git/refs",
                headers=self.headers,
                json={"ref": f"refs/heads/{branch}", "sha": commit_sha},
                timeout=10
            )
            if response.status_code == 422:
                # Ref already exists; fall back to patch.
                response = await client.patch(
                    f"{self.base_url}/repos/{owner}/{repo_name}/git/refs/heads/{branch}",
                    headers=self.headers,
                    json={"sha": commit_sha, "force": False},
                    timeout=10
                )
            response.raise_for_status()

    def get_deployment_urls(
        self,
        owner: str,
        repo_name: str
    ) -> Dict[str, str]:
        """
        Get deployment URLs for Render platform.

        Args:
            owner: Repository owner
            repo_name: Repository name

        Returns:
            Dictionary with GitHub and Render deployment URLs
        """
        repo_url = f"https://github.com/{owner}/{repo_name}"

        return {
            "github": repo_url,
            "render": f"{RENDER_DEPLOY_URL}?repo={repo_url}"
        }

    async def _get_pages(self, url: str, params: Dict[str, Any], max_pages: int) -> tuple[List[Any], bool]:
        """Every item of a paged GitHub listing, following its ``Link: rel="next"`` headers.

        Returns the items and whether ``max_pages`` cut the listing short.
        """
        items: List[Any] = []
        next_url: Optional[str] = url
        query: Optional[Dict[str, Any]] = {**params, "per_page": _PAGE_SIZE}
        async with httpx.AsyncClient() as client:
            for _ in range(max_pages):
                response = await client.get(next_url, headers=self.headers, params=query, timeout=15)
                response.raise_for_status()
                items.extend(response.json())
                next_url = response.links.get("next", {}).get("url")
                if not next_url:
                    return items, False
                query = None  # the next link carries the query
        logger.warning("Listing %s stopped after %d pages", url, max_pages)
        return items, True

    async def get_user_repositories(self, sort: str = "updated") -> Dict[str, Any]:
        """The repositories the signed-in user can open, most recently updated first.

        Those they own, those they collaborate on and those of their
        organizations (``affiliation``), across every page up to
        ``_MAX_REPO_PAGES``.

        Returns:
            ``repositories`` (simplified repo dicts, ``can_push`` saying whether
            the user may write to it) and ``truncated``.
        """
        repos, truncated = await self._get_pages(
            f"{self.base_url}/user/repos",
            {"sort": sort, "affiliation": "owner,collaborator,organization_member"},
            _MAX_REPO_PAGES,
        )
        return {
            "repositories": [
                {
                    "id": r.get("id"),
                    "name": r.get("name"),
                    "full_name": r.get("full_name"),
                    "description": r.get("description"),
                    "html_url": r.get("html_url"),
                    "private": r.get("private"),
                    "updated_at": r.get("updated_at"),
                    "default_branch": r.get("default_branch", "main"),
                    "topics": r.get("topics", []),
                    "can_push": bool((r.get("permissions") or {}).get("push", True)),
                }
                for r in repos
            ],
            "truncated": truncated,
        }

    async def get_branches(self, owner: str, repo_name: str) -> List[str]:
        """Every branch name of a repository (up to ``_MAX_BRANCH_PAGES`` pages)."""
        branches, _ = await self._get_pages(
            f"{self.base_url}/repos/{owner}/{repo_name}/branches", {}, _MAX_BRANCH_PAGES,
        )
        return [b.get("name") for b in branches]

    async def get_default_branch(self, owner: str, repo_name: str) -> str:
        """The repository's default branch, as GitHub reports it.

        Raises ``httpx.HTTPStatusError`` (404) when the repo does not exist or is not visible.
        """
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/repos/{owner}/{repo_name}",
                headers=self.headers,
                timeout=15
            )
            response.raise_for_status()
        return response.json()["default_branch"]

    async def file_exists(
        self,
        owner: str,
        repo_name: str,
        file_path: str,
        branch: str = "main"
    ) -> bool:
        """
        Check if a file exists in the repository.

        Args:
            owner: Repository owner
            repo_name: Repository name
            file_path: Path to file in repository
            branch: Branch name

        Returns:
            True if file exists, False otherwise
        """
        params = {"ref": branch}

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/repos/{owner}/{repo_name}/contents/{file_path}",
                headers=self.headers,
                params=params,
                timeout=10
            )


        return response.status_code == 200

    async def get_repository_contents(
        self,
        owner: str,
        repo_name: str,
        path: str = "",
        branch: str = "main"
    ) -> List[Dict[str, Any]]:
        """
        Get contents of a repository path.

        Args:
            owner: Repository owner
            repo_name: Repository name
            path: Path to directory (empty for root)
            branch: Branch name

        Returns:
            List of file/directory objects
        """
        params = {"ref": branch}

        # Ensure path doesn't start with /
        if path.startswith("/"):
            path = path[1:]

        url = f"{self.base_url}/repos/{owner}/{repo_name}/contents/{path}"

        async with httpx.AsyncClient() as client:
            response = await client.get(
                url,
                headers=self.headers,
                params=params,
                timeout=15
            )
            response.raise_for_status()

            data = response.json()

        # If result is a dict, it's a single file. Wrap in list.
        if isinstance(data, dict):
            return [data]

        # Sort: directories first, then files
        return sorted(data, key=lambda x: (x.get("type") != "dir", x.get("name").lower()))

    async def get_commits(
        self,
        owner: str,
        repo_name: str,
        path: Optional[str] = None,
        per_page: int = 30,
        ref: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get commit history for a repository or specific file.

        Args:
            owner: Repository owner
            repo_name: Repository name
            path: Optional file path to filter commits
            per_page: Number of commits to return
            ref: Branch or commit to list history from (default branch when None)

        Returns:
            List of commit information dictionaries
        """
        params = {"per_page": per_page}
        if path:
            params["path"] = path
        if ref:
            params["sha"] = ref

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/repos/{owner}/{repo_name}/commits",
                headers=self.headers,
                params=params,
                timeout=15
            )
            response.raise_for_status()

            commits = response.json()
        return [self._simplify_commit(c) for c in commits]

    @staticmethod
    def _simplify_commit(c: Dict[str, Any]) -> Dict[str, Any]:
        """The commit fields the editor shows, from a GitHub commit object."""
        return {
            "sha": c.get("sha"),
            "message": c.get("commit", {}).get("message", ""),
            "author": c.get("commit", {}).get("author", {}).get("name", "Unknown"),
            "date": c.get("commit", {}).get("author", {}).get("date", ""),
            "html_url": c.get("html_url", ""),
            "parents": [p.get("sha") for p in c.get("parents", [])],
        }

    async def get_file_content(
        self,
        owner: str,
        repo_name: str,
        file_path: str,
        branch: str = "main",
        ref: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get content of a file from the repository.

        Args:
            owner: Repository owner
            repo_name: Repository name
            file_path: Path to file in repository
            branch: Branch name (ignored if ref is provided)
            ref: Specific commit SHA to get file from (optional)

        Returns:
            Dictionary with file content and metadata, or None if not found
        """
        # Use ref (commit SHA) if provided, otherwise use branch
        params = {"ref": ref if ref else branch}

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/repos/{owner}/{repo_name}/contents/{file_path}",
                headers=self.headers,
                params=params,
                timeout=15
            )

        if response.status_code == 404:
            return None

        response.raise_for_status()
        data = response.json()

        # Decode content from base64
        content = None
        if data.get("content"):
            content = base64.b64decode(data["content"]).decode("utf-8")
        elif data.get("encoding") == "none" and data.get("size"):
            # Over 1 MB the JSON response carries no content; the raw media type returns it.
            async with httpx.AsyncClient() as client:
                raw = await client.get(
                    f"{self.base_url}/repos/{owner}/{repo_name}/contents/{file_path}",
                    headers={**self.headers, "Accept": "application/vnd.github.raw+json"},
                    params=params,
                    timeout=60
                )
            raw.raise_for_status()
            content = raw.content.decode("utf-8")

        return {
            "sha": data.get("sha"),
            "content": content,
            "path": data.get("path"),
            "size": data.get("size"),
        }

    async def create_or_update_file(
        self,
        owner: str,
        repo_name: str,
        file_path: str,
        content: str,
        commit_message: str,
        branch: str = "main"
    ) -> Dict[str, Any]:
        """
        Create or update a file in the repository.

        Args:
            owner: Repository owner
            repo_name: Repository name
            file_path: Path to file in repository
            content: File content (string)
            commit_message: Commit message
            branch: Branch name

        Returns:
            API response with commit info
        """
        # Check if file exists to get its SHA
        existing = await self.get_file_content(owner, repo_name, file_path, branch)

        # Encode content
        content_b64 = base64.b64encode(content.encode("utf-8")).decode("utf-8")

        data = {
            "message": commit_message,
            "content": content_b64,
            "branch": branch
        }

        # If file exists, include SHA for update
        if existing and existing.get("sha"):
            data["sha"] = existing["sha"]

        async with httpx.AsyncClient() as client:
            response = await client.put(
                f"{self.base_url}/repos/{owner}/{repo_name}/contents/{file_path}",
                headers=self.headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()

            result = response.json()
        return {
            "commit_sha": result.get("commit", {}).get("sha"),
            "content_sha": result.get("content", {}).get("sha"),
        }

    async def create_gist(
        self,
        content: str,
        filename: str,
        description: str = "",
        is_public: bool = False
    ) -> Dict[str, Any]:
        """
        Create a GitHub Gist.

        Args:
            content: Content of the gist file
            filename: Name of the file in the gist
            description: Gist description
            is_public: Whether the gist is public

        Returns:
            Dictionary with gist URL and ID
        """
        data = {
            "description": description,
            "public": is_public,
            "files": {
                filename: {
                    "content": content
                }
            }
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/gists",
                headers=self.headers,
                json=data,
                timeout=15
            )
            response.raise_for_status()

            result = response.json()
        return {
            "gist_id": result.get("id"),
            "gist_url": result.get("html_url"),
            "raw_url": result.get("files", {}).get(filename, {}).get("raw_url"),
        }


def create_github_service(access_token: str) -> GitHubService:
    """
    Factory function to create GitHub service instance.

    Args:
        access_token: GitHub OAuth access token

    Returns:
        GitHubService instance
    """
    return GitHubService(access_token)
