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
   * - ``backend/routers/github_oauth.py``
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
   * - ``backend/routers/github_deploy_api.py``
     - The ``/github/*`` endpoints: save, load, create, repo summary, sync
       status, model history, pull requests, go live.
   * - ``backend/services/spec_driven/github_sync.py``
     - ``/spec-driven/import-github-run`` (a repository becomes a run) and
       ``/spec-driven/push-to-github`` (a run goes back to the repository);
       the routes are in ``backend/routers/spec_driven_router.py``.
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

- ``buml/diagrams.json``: the model, a V2 project export, and the file the
  editor reads back. A repository an older editor saved has no
  ``buml/diagrams.json``: its ``besser-project.json`` or the old sidebar's
  ``<dir>/<project>.json`` is read instead, found by looking for a ``.json``
  file that holds a BESSER project. When a repository holds several, the
  editor asks which one to use. The next save writes ``buml/diagrams.json``
  and removes that older file (``legacy_file_path`` on
  ``/github/project/save``).
- ``buml/domain_model.py``, ``buml/gui_model.py``, ``buml/agent_model_<agent>.py``:
  the same model as B-UML Python source, for people reading the repository.
  Always written from ``diagrams.json``, never read. Only these names are
  BESSER's: another file under ``buml/`` (a ``seed_data.py``) is the
  repository's own and is merged like any code.
- ``.besser/project.json``: which BESSER project the repository belongs to.
- The ``besser`` topic on the repository, which is how the repo browser lists
  BESSER projects first.
- Commits BESSER makes end with ``BESSER-Origin: editor`` or
  ``BESSER-Origin: spec-driven-agent``; the history uses it to tell them from a
  person's commits.

Agent credentials (LLM keys, bot tokens, database passwords in the agent
diagram's config) are replaced with ``[REDACTED]`` before ``diagrams.json`` is
written, by every path including ``/github/deploy-webapp``; the editor keeps the user's local values when it reads the file back
(``shared/utils/redactedSecrets.ts``). New repositories are private unless the
user asks otherwise.

Sign-in and the session
-----------------------

``/github/auth/login`` redirects to GitHub; the callback exchanges the code for
a token, keeps the token server-side and returns to the editor with a session
id in the URL fragment, which browsers never send to a server. The editor keeps
that id in ``sessionStorage`` and sends it in the ``X-GitHub-Session`` header,
never in a URL. For one release, ``/github/auth/status``, the star endpoints and
logout still accept the old ``session_id`` query parameter when the header (or
logout's body) is absent, and log a deprecation warning. Sessions live in the
app's sign-in stores (``GitHubSessions``, built in the backend's lifespan): an
encrypted file (``USER_TOKENS_PATH``, else ``SESSION_STORE_PATH``, else the
temp directory) that outlives a restart, or memory, which does not, when the
``cryptography`` package is missing.

The editor has one sign-in store for the tab. Its states are ``checking`` (a
stored session is being verified; actions wait instead of starting a new
sign-in), ``signed-in``, ``expired`` and ``signed-out``. Any request carrying the
session header that gets a 401 dispatches ``besser:github-session-rejected``
from ``api-client.ts``, which marks the session expired everywhere at once. A
remembered username without a session also reads as expired, so a linked
project offers *Reconnect* after a reload; signing out forgets the username and
revokes the token at GitHub.

A GitHub API error reaches the editor with one mapping on every endpoint
(``_github_status_error`` in ``routers/github_deploy_api.py``): 401, 403 and
other 4xx keep their status, 422 (a name already taken, a branch that moved)
becomes 409, and a 5xx becomes 502, each with GitHub's own message. Two
exceptions: ``/spec-driven/import-github-run`` answers a missing repository
with 404 ``repo_missing``, and ``/github/gist/create`` answers 403/404 with how
to grant the ``gist`` scope.

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
- **The branch moves while committing** (someone pushed between reading the
  head and moving it, which GitHub refuses with 422): the commit is made again
  on the new head, once; a second race is reported as a conflict (HTTP 409).
- **The repository has no commit**: GitHub's git data API answers 409 "Git
  Repository is empty", so the first commit is made through the contents API
  (one model or run file), and the write continues on top of it.

An agent push changes only what the run changed. ``push_directory_to_repo``
compares the run's files with GitHub's recursive tree of the base commit:

- a file whose git blob id and mode match is not uploaded, nor is one whose
  bytes still hash as they did when the run got them: the import is a git
  archive, which applies ``.gitattributes`` (``eol``, ``export-subst``,
  ``ident``) and LFS, so a file the run never touched can differ from its blob;
- a changed file keeps its mode, so an executable stays executable;
- symlinks and submodules in the repository are never touched;
- a repository path the run could not hold (build and dependency output,
  ``*.db``, BESSER's other ``.besser_*`` files, a scrubbed secret ``.env``) is
  never deleted. The records a run carries (``.besser_recipe.json``,
  ``.besser_attribution.json``, ``.besser_generator_output.json``) are the run's
  own: a seeded run deletes one it no longer has;
- the push marks those three records ``linguist-generated`` in
  ``.gitattributes``, so GitHub folds their diffs. The repository's own lines are
  kept, and a record a line already sets ``linguist-generated`` for keeps that
  setting;
- only a run seeded from this repository can delete a file at all; a run
  generated from the model (a first generation) only adds and updates;
- a seeded run deletes only paths it held: import and every push record the
  run's files in ``.besser_github_source.json`` (``paths``: repo path to the
  SHA-256 of the bytes the run held; a record from before the hashes is a plain
  list and is still read), and a path missing from it (an ``export-ignore``
  file the archive left out, say) is kept;
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
3. The B-UML sources are written again from the merged model. If that export
   fails, the previous sources are kept, not deleted, and the answer's
   ``warnings`` says they may be out of date.
4. The branch moves only by a fast-forward, to a merge commit whose parents are
   the branch head and the write's own commit. If the branch moved again
   meanwhile, the fast-forward fails rather than reverting anything.

The scratch and code branches are deleted in every case.

Who added what
--------------

``project.attribution`` in ``diagrams.json`` maps the id of a class or member
(attribute, method) to who added it::

    "attribution": {"<element id>": {"addedBy": "<GitHub login>", "addedAt": "2026-11-04T12:00:00Z"}}

- **Where the author comes from**: ``repo_model.attribute_new_elements`` credits
  each element the write adds over the model at its base commit to the signed-in
  GitHub user, the same identity as the commit's author. A save needs a session,
  so nothing is credited without one, and a model that nobody saves through
  BESSER (a hand edit of the file) simply has no record. When GitHub does not
  answer, the save goes on without new records and logs a warning. The save
  returns the records it wrote as ``attribution``.
- **What it proves**: the records of the base commit stand and a client cannot
  rewrite them or claim an element it adds for someone else; a malformed record is
  dropped. Anyone with push access can still edit the file, so a record says who
  BESSER recorded, not who is proven to have added the element.
- **Why it cannot affect generation**: the block sits beside the diagrams.
  ``ProjectInput`` does not read it, so no converter or generator sees it, and
  the B-UML sources are the same with or without it.
- **Merging**: an element's record merges like any other key, so two people's
  additions both keep their credit. ``addedBy`` and ``addedAt`` keep this side's
  value rather than conflict. A sender that dropped the block (an older editor)
  does not erase it: the records of the base commit are kept. Records of removed
  elements are dropped.
- **Older repositories**: elements already in the base commit get no record, so
  a repository saved before this block existed gets records only for what is
  added from then on.

After a pull, the editor marks what the pull brought in
(``pullRepoIntoProject.ts``): elements the local project lacks and that have a
record. Only a project already in sync with the repository compares, so opening
a repository marks nothing. The mark is a list of ids in the project's link in
the browser (``newFromRepo``), never in the model, and
``ModelAttributionLayer.tsx`` draws it over the canvas from the position of each
``[data-id]`` element, so a colour someone chose for an element is not touched.

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
run seeds from. An import takes one of the ``BESSER_LLM_MAX_CONCURRENT_RUNS``
slots for its duration and answers 429 when they are all taken. A repository
with ``buml/diagrams.json`` opens with its model. For any other,
``code_reverse.py`` reads SQLAlchemy, Django, Pydantic and dataclass models.
When there are none (or only API shapes), ``ai_reverse.py`` can ask the
server's model to read a digest of the code, whose first provider may be a
third party, so only with the user's consent: without ``allow_ai_reading`` the
import calls no model, still returns the exact reading if any, and answers
``ai_reading_offered: true`` with a message offering it; the editor imports
again with ``allow_ai_reading: true`` once the user allows it. The result is a
proposed class diagram the user accepts or discards on the canvas. A relationship the
metamodel rejects is left out on its own and named in the import message
(``skipped_relationships``); the rest of the model is kept.

The backend shares the run directory with the sandboxed worker, so it reads and
writes its own records there (``.besser_github_source.json``,
``.besser_recipe.json``) through ``read_run_json`` / ``write_run_json``: a
link is never followed, a read is size-capped, and a write replaces the file
atomically.

.. _web-editor-github-coming-back:

Coming back to a repository
---------------------------

A finished run lives on the server only for ``BESSER_LLM_DOWNLOAD_TTL_SECONDS``
(30 minutes by default). The repository is what lasts, so that is what a later
request continues from.

**The base of the next run.** When the project is linked to a repository and its
last run has expired, the editor imports the branch head before the run
(``features/spec-driven/linkedRepoBase.ts``) and edits that code:

- the head is the commit the project last synced with: the run starts from it,
  and the project's model stays, with any changes made to it since;
- the head moved and the model has no unsaved changes: the project takes the
  repository's model first, as **Get latest** does;
- the head moved and the model has unsaved changes: no run starts, and the chat
  asks the user to save first, which merges both models. Running anyway would
  push the older model over the newer one;
- the repository holds only a saved model (``has_code: false`` on the import):
  there is no app yet, so the run builds one from the model;
- the code cannot be read (this tab is not connected to GitHub, the repository
  is gone, GitHub cannot be reached): the user chooses between rebuilding from
  the model and stopping. A repository known to hold only a saved model just
  builds, without asking. When the server has no free run slot, nothing runs and
  the chat asks to try again.

**What the run knows.** ``besser/spec_driven_agent/run_recap.py`` reads
``.besser_recipe.json`` once for everyone who continues the app:

- the next run's inventory gets the earlier requests, where the last run ended
  (repairs rolled back, requirements missing from the code or not checked, tasks
  left out), the commits made since, and the repository's ``AGENTS.md``;
- every push of a run with a recipe keeps a section in ``AGENTS.md``, for people
  and for other coding tools. It sits between ``<!-- besser:begin -->`` and ``<!-- besser:end -->``,
  and anything outside those markers stays as it is, also when the run never held
  the file. A seeded run works without the section, and the push renders it again.
  An ``AGENTS.md`` that is a link or not text, or that was added on GitHub after
  the run started, is left alone, with a warning;
- the import and the push return it as ``recap``. The chat shows it when a
  repository opens or a run continues from one, and the editor sends it to the
  modeling agent with each message (``context.githubRepo``).

A recipe and an ``AGENTS.md`` can be written by anyone with push access, and a
public repository's by a stranger. So a recap value is one line, capped, free of
credentials and unable to close the ``AGENTS.md`` section; the agent receives
all of it labelled as a record and not as instructions, and ``AGENTS.md``
quoted line by line and capped at 4,000 characters; the chat escapes it.

The commits since the last run are the ones after the last commit that changed
the recipe. The import notes them on the imported recipe
(``commits_since_run``), so the next run reads them and its own recipe drops
them. Without an answer from GitHub the repository still opens, with no commits
listed.

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
