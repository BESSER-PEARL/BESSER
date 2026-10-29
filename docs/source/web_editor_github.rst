GitHub integration
==================

How the Web Modeling Editor keeps a project, its model and its application code
in a GitHub repository. This page is for contributors: what lives where, which
rules the code keeps, and how to test it. The user-facing guide is the "GitHub
Integration" section of the `editor user guide
<https://besser.readthedocs.io/projects/besser-web-modeling-editor/en/latest/>`_;
the endpoint list is in :doc:`web_editor_backend`.

The pieces
----------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Where
     - What it owns
   * - ``backend/services/deployment/github_oauth.py``
     - OAuth sign-in, the server-side session store, sign-out with token
       revocation. The browser only ever holds an opaque session id.
   * - ``backend/services/deployment/github_service.py``
     - Every GitHub REST call: listings across pages, blobs, trees, commits,
       refs, merges, pull requests, tarballs. ``push_directory_to_repo`` and
       ``commit_tree`` hold the push and merge rules below.
   * - ``backend/services/deployment/repo_model.py``
     - The layout of a BESSER repository, the ``BESSER-Origin`` trailer,
       credential redaction, ``ModelMerge``.
   * - ``backend/services/deployment/model_merge.py``
     - The three-way merge of ``diagrams.json`` as data.
   * - ``backend/services/deployment/github_deploy_api.py``
     - The ``/github/*`` endpoints: save, load, create, repo summary, sync
       status, model history, pull requests, go live.
   * - ``backend/routers/spec_driven_router.py``
     - ``/spec-driven/import-github-run`` (a repository becomes a run) and
       ``/spec-driven/push-to-github`` (a run goes back to the repository).
   * - ``backend/services/reverse_engineering/``
     - Reads a repository BESSER did not create: ``code_reverse.py`` (exact,
       Python ``ast`` only, nothing is executed) and ``ai_reverse.py`` (a
       bounded, secret-free digest read by the server's model).
   * - WME ``shared/services/github/githubAuth.ts``
     - The tab's one sign-in store, read through ``shared/hooks/useGitHubAuth``.
   * - WME ``shared/services/storage/projectGitHubLink.ts``
     - The project's one link to a repository (``besser_github_link_<id>``).
   * - WME ``shared/api/github.ts``
     - Typed calls to the endpoints above, used by every feature that talks
       to GitHub.
   * - WME ``shared/components/github/``
     - The repo browser (also used by the project hub and the assistant's
       *Start from a repository*).
   * - WME ``features/github/``
     - The GitHub menu, dialogs, the chat's repository header, ``useRepoSync``
       (sync state and actions), ``useSpecDrivenGithubPush`` (pushing a run).

Repository layout
-----------------

A project is linked to one repository, and every path that writes it (the
editor's Save, the agent's push, Go live) uses the same layout, defined once in
``repo_model.py``:

- ``buml/diagrams.json``: the model, a V2 project export. The only model file
  the editor reads back (the legacy ``besser-project.json`` is still read and
  removed on the next save).
- ``buml/*.py``: the same model as B-UML Python source, for people reading the
  repository. Always written from ``diagrams.json``, never read.
- ``.besser/project.json``: which BESSER project the repository belongs to.
- The ``besser`` topic on the repository, which is how the repo browser lists
  BESSER projects first.
- Commits BESSER makes end with ``BESSER-Origin: editor`` or
  ``BESSER-Origin: spec-driven-agent``; the history uses it to tell them from a
  person's commits.

Agent credentials (LLM keys, bot tokens, database passwords in the agent
diagram's config) are replaced with ``[REDACTED]`` before ``diagrams.json`` is
written; the editor keeps the user's local values when it reads the file back
(``shared/utils/redactedSecrets.ts``). New repositories are private unless the
user asks otherwise.

Sign-in and the session
-----------------------

``/github/auth/login`` redirects to GitHub; the callback exchanges the code for
a token, keeps the token server-side and returns to the editor with a session
id in the URL fragment, which browsers never send to a server. The editor keeps
that id in ``sessionStorage`` and sends it in the ``X-GitHub-Session`` header,
never in a URL. Sessions live in the backend's memory, so a backend restart ends
them.

The editor has one sign-in store for the tab. Its states are ``checking`` (a
stored session is being verified; actions wait instead of starting a new
sign-in), ``signed-in``, ``expired`` and ``signed-out``. Any request carrying the
session header that gets a 401 dispatches ``besser:github-session-rejected``
from ``api-client.ts``, which marks the session expired everywhere at once. A
remembered username without a session also reads as expired, so a linked
project offers *Reconnect* after a reload; signing out forgets the username and
revokes the token at GitHub.

Sync state
----------

The project's link records the commit the editor last synced
(``baseCommitSha``) and a fingerprint of the model at that commit. A fingerprint
taken right after a pull would be wrong (the editor rewrites a loaded model into
its current format), so it is taken once the editor settles
(``baselinePending``). ``useRepoSync`` compares the link with
``/github/sync-status`` (commits on the branch since the base) on menu open, on
window focus and once a minute, and derives one state: in sync, local changes,
behind, diverged, reconnect, missing. Only the newest check may set the state,
so a check answered for the commit before a save cannot show that save as a new
commit.

Writing to the repository
-------------------------

Writes never revert newer commits. Every write is a commit on top of the base it
was derived from (the editor's ``baseCommitSha``, or for an agent push the
commit the run started from, recorded in the run's
``.besser_github_source.json``):

- **The branch has not moved**: the branch is fast-forwarded.
- **It has moved**: the commit is merged with the newer ones (next section).
- **The merge fails**: the commit stays on a ``besser/sync-<sha>`` branch and the
  answer is ``status: "conflict"``. Nothing is overwritten.

An agent push changes only what the run changed. ``push_directory_to_repo``
compares the run's files with GitHub's recursive tree of the base commit:

- a file whose git blob id and mode match is not uploaded;
- a changed file keeps its mode, so an executable stays executable;
- symlinks and submodules in the repository are never touched;
- a repository path the run could not hold (build and dependency output,
  ``*.db``, BESSER's ``.besser_*`` files, a scrubbed secret ``.env``) is never
  deleted;
- only a run seeded from this repository can delete a file at all; a run
  generated from the model (a first generation) only adds and updates;
- a seeded run deletes only paths it held: import and every push record the
  run's files in ``.besser_github_source.json``, and a path missing from that
  list (an ``export-ignore`` file the archive left out, say) is kept;
- a push that changes nothing makes no commit.

Merging
-------

Git merges ``diagrams.json`` and the B-UML sources line by line, so two people
who each add a class collide on the same lines. When a write holds the model,
``GitHubService._merge_with_model`` merges in two parts:

1. Everything except the model files is merged by GitHub's merge API on a
   scratch branch (``besser/sync-merge-…``).
2. ``diagrams.json`` is merged three ways as data by ``model_merge.py``: objects
   key by key, lists of objects with an ``id`` item by item (diagram elements in
   either editor engine), lists of unique values as sets. Layout and timestamps
   keep this side's value. A value both sides changed differently, or a merged
   model that no longer builds, is a conflict.
3. The B-UML sources are written again from the merged model.
4. The branch moves only by a fast-forward, to a merge commit whose parents are
   the branch head and the write's own commit. If the branch moved again
   meanwhile, the fast-forward fails rather than reverting anything.

The scratch and code branches are deleted in every case.

Conflicts and pull requests
---------------------------

A save or push that did not land (a conflict, or a push the user chose to send
as a pull request) is kept on the project's link as ``pendingChange``: its
branch, the message it was saved with, and the pull request once one exists.
The GitHub menu offers to open the pull request (``POST /github/pull-request``,
whose description lists the model changes and files, and which returns the open
one when asked twice), links to it once open, and drops the pending change when
``/github/sync-status?pending_branch=…`` reports the pull request merged or
closed, or its branch deleted. After a merge, *Get latest* brings the model in.

When the agent's run changed the model, the push response carries that model
(``project_export``). On a direct push the editor loads it and records the
pushed commit as synced; on a pull request or conflict it loads it too, counted
as unsaved against the branch until the pull request lands.

Starting from any repository
----------------------------

``/spec-driven/import-github-run`` pins the branch head, extracts the tarball
into the shared run workspace (limits: 100 MB archive, 500 MB unpacked, 50,000
files; links and devices are skipped) and registers it as a run the next modify
run seeds from. A repository with ``buml/diagrams.json`` opens with its model.
For any other, ``code_reverse.py`` reads SQLAlchemy, Django, Pydantic and
dataclass models; when there are none (or only API shapes), ``ai_reverse.py``
asks the server's model to read a digest of the code. The result is a proposed
class diagram the user accepts or discards on the canvas. A relationship the
metamodel rejects is left out on its own and named in the import message
(``skipped_relationships``); the rest of the model is kept.

The backend shares the run directory with the sandboxed worker, so it reads and
writes its own records there (``.besser_github_source.json``,
``.besser_recipe.json``) through ``read_run_json`` / ``write_run_json``: a
link is never followed, a read is size-capped, and a write replaces the file
atomically.

The Spec-Driven Agent's model fallback
--------------------------------------

Runs on the server-paid tiers (``free``, ``sponsored``) carry a fallback chain
(see :doc:`spec_driven_agent/configuration`). A run moves along it when its
endpoint is unavailable, out of quota or credits, or rejects the server's token;
a rejected token skips the other models on the same endpoint, which share it.
A user's own key has no fallback and fails with ``INVALID_KEY``.

Testing
-------

``tests/utilities/web_modeling_editor/backend/fake_github.py`` is a GitHub REST
stand-in backed by real git repositories (it needs ``git`` on ``PATH``).
Blobs, trees, commits, refs, merges (with GitHub's 204/409 answers), compares,
paging with ``Link`` headers, pull requests, tarballs and the OAuth code
exchange are real git operations, so tests run the real ``GitHubService`` and
assert on the files GitHub would hold:

.. code-block:: python

    from tests.utilities.web_modeling_editor.backend.fake_github import FakeGitHub, use_fake_github

    def test_something(monkeypatch, tmp_path):
        fake = FakeGitHub(root=str(tmp_path / "github"))
        use_fake_github(monkeypatch, fake)   # only GitHub's hosts reach the fake
        repo = fake.create_repo("library", files={"README.md": "# Library\n"})
        ...                                  # call the backend
        assert fake.files(repo)["README.md"][1] == b"# Library\n"

``test_github_round_trip.py`` covers the flows end to end this way (delta
pushes, organization repos, paging, merges, conflicts, pull requests, sign-in
and sign-out); ``test_model_merge.py`` covers the merge rules.

For a browser check, ``github_harness.py`` runs the real backend on port 9000
with GitHub replaced by the stand-in, a scripted Spec-Driven run (no LLM) and a
chat stand-in on port 8766:

.. code-block:: bash

    python -m tests.utilities.web_modeling_editor.backend.github_harness
    # then, in frontend/packages/webapp:
    UML_BOT_WS_URL=ws://localhost:8766 npx vite --port 5173

"Connect GitHub" signs in as ``ana`` through the real callback. The harness
seeds a BESSER repository, a plain FastAPI repository with build output, an
executable script and a symlink, two organization repositories (one read-only)
and enough empty repositories to need a second page. Its ``/mock`` endpoints act
as a teammate on GitHub (add or rename a class, change code, merge a pull
request) and show what a repository holds; the module docstring lists them.
