Validation and auto-fix
=======================

Phase 3 attempts validation within the run's budget. Core checks need no network
and no Docker. Missing runtime prerequisites do not count as verified execution:

- Python syntax on every ``.py`` file, decoded as Python decodes it (a UTF-8
  BOM or a ``coding:`` line is valid; an undecodable file is a syntax error).
  The file is compiled, not run, so an error Python raises only on import
  (``await`` in a plain function, a late ``from __future__`` import) counts too.
  Installed dependencies and build output (``node_modules``, ``.venv``,
  ``venv``, ``dist``, ``build``) are not the app: no check, ``ruff``
  included, reads or rewrites files there.
- Python declaration contracts: invalid enum members, overwritten declarations,
  and SQLite-incompatible constant CHECK constraints.
- Isolated backend startup and create-request probes. An import or DDL failure
  is a blocker, not a skipped check. Schema changes are checked against router
  consumers as well as syntax.
- Retained ``test_api`` workflows replayed after source changes.

``.ts`` / ``.tsx`` / ``.js`` / ``.jsx`` files get a structural syntax check
(delimiter balance across comments, strings, template literals, regexes and
JSX, plus strict-JSON parsing of the ``attr={{...}}`` containers the generated
table configuration is written as) when they are written, not in the Phase 3
sweep. It needs no Node toolchain. A write reports what it finds to the model,
and an edit that breaks a file that was clean is refused; an edit to a file
that was already broken is never refused, so it can be repaired.

- Dockerfile coherence — referenced ``requirements.txt`` / ``package.json``
  must exist. A ``COPY`` source is resolved against the build context, as
  Docker does: the compose ``build.context`` when a compose file names the
  Dockerfile, else the Dockerfile's own folder. ``COPY --from=<stage>`` copies
  from a build stage and is not checked. This check only reports; it never
  writes a file. A separate repair step before validation restores a
  ``requirements.txt`` that a Phase 2 edit deleted from a FastAPI scaffold.
  Several other common mistakes are repaired outright without spending
  an LLM turn (``npm ci`` with no lockfile anywhere in the project becomes
  ``npm install``; a ``COPY`` of a non-existent ``package-lock.json`` is
  dropped).
- Known-incompatible dependencies — a table of pairs that pip resolves
  without complaint but that break at runtime (``passlib`` needs
  ``bcrypt<4.1``: bcrypt 5 rejects the >72-byte secret passlib's self-test
  hashes, so every password hash fails). Each ``requirements.txt`` that
  declares the first package gets the second pinned, unless its declared range
  is already compatible. The same table is applied when scaffold repair writes
  a ``requirements.txt`` and by ``install_dependencies`` before it installs,
  so the delivered file is pinned even when Phase 3 is skipped.
- Local-import resolution — an import naming a module the app does not ship
  where it is used. ``ruff`` is structurally blind to this (a star import
  excuses every name), and it is fatal at startup. Imports the app starts
  without are not checked: inside a ``try`` that handles ``ImportError`` or a
  ``with suppress(ImportError):``, and under ``if TYPE_CHECKING:``. Under a
  catch-all (``except Exception``, ``with suppress(Exception):``) an
  unresolved import is a warning, ``possibly missing module:``, unless it is
  inside a function, the handler goes on to the code after the ``try`` (no
  ``return`` or ``raise`` in it), and the import has a fallback: every
  catch-all handler binds the imported name (another import, ``= None``), or
  each read of it outside the ``try`` body (or its ``else``) or the ``with``
  block comes after another binding of it in the function (a parameter, an
  assignment before the guard or a later one). That is an optional fast path
  the function falls back from. Otherwise the read, after the guard, in the
  handler, in ``finally`` or in a closure, is a ``NameError`` when the import
  failed, so it is reported. A flag set in the handler (``found = False``,
  then ``if found:``) is not followed, so that read is reported too. A package
  declared in the app's manifest satisfies its import name, matched
  case-insensitively and, for names that differ (``python-socketio`` /
  ``socketio``), by name prefix, unless the app ships a module of that name.
- Frontend contract — high-precision correctness defects that leave the UI
  visibly broken: a router with no home route (blank on load), or a form whose
  submit handler is a no-op (cannot save).
- Data contract — generated code that disagrees with the domain model's
  declared identifier types or server-owned fields, or that fakes success for
  an unimplemented method.
- Framework coherence — generated code importing a rival framework into the
  chosen scaffold (Flask into a FastAPI project).
- Model conformance (modify runs whose class diagram changed, FastAPI backend
  only) — ``sql_alchemy.py`` or a ``<Class>Create`` schema that still defines
  a removed attribute, lacks an added one, declares a changed type as the old
  one, or contradicts a changed optional/required flag, and a renamed class
  still defined only under its old name. Blocker; only the changed attributes
  are checked (a renamed class under its new name), and an app without these
  files reports the check as not run. See :ref:`spec-driven-modify-mode`.
- Generated model-action contracts — actual handler paths and functions, with
  deterministic checks for missing handlers and known unimplemented bodies.
  Absence of a known stub is structural evidence, not proof of business behavior.
- Frontend/backend endpoint coherence — literal ``fetch`` and Axios URLs, and
  the generated client's ``apiRequest(<METHOD>, <path>)`` calls, that match no
  generated backend route, and calls of a typed client function
  (``bookApi.create``) the generated client does not have. The typed functions
  are resolved to their routes through the app's ``api_contract.json``.
- API call shapes (``api call:``, blocker) — a frontend call with a literal or
  template URL that resolves to exactly one route, checked against that
  route's handler, which is read without importing the app. The calls read are
  ``fetch``, ``axios.<method>`` (with or without a TypeScript type argument),
  the ``axios({method, url, data})`` config form, an ``axios.create()``
  instance defined in the file or imported from the file that defines it, and
  a ``fetch`` wrapper function such as ``api(path, options)``, defined or
  imported by name, the generated client's ``apiRequest`` (its body is the
  ``body`` option) and its typed ``create``/``update`` functions (their body is
  the last argument, cast or not; their responses are already unwrapped and are
  not checked). A wrapper is followed only when it calls
  ``fetch(BASE + path, ...)`` once, passes the caller's options on without a
  method or body of its own, and returns the parsed JSON; an instance or base
  whose URL adds a path (``${API}/v2``, or a constant declared with one) is not
  followed, nor an instance with its own ``transformRequest`` or
  ``transformResponse``.

  * the JSON body: an object literal, written in place or in a variable
    declared with it and otherwise only read, none of whose keys the route
    reads, or one that leaves out a required field. A route with one Pydantic
    model parameter reads the model's fields; otherwise the body has one
    top-level key per body parameter, so a ``/methods/<name>/`` route with
    ``params: dict = Body(embed=True)`` reads ``{"params": {...}}``, never the
    values flat. A key sent beside ``params`` is reported only when the
    handler reads it from inside ``params`` (``params.get('<key>')``); one it
    takes elsewhere, such as an id from the path, is harmless.
  * the response: when every ``return`` of the handler sends a dict literal
    that is not handed to another function, the object's top-level keys. Code
    that uses that object's data only as an array (``Array.isArray``,
    ``.length``, ``.map()``, ``.filter()``, ``Array.isArray(x) ? x : []``, or
    handing it to the setter of a ``useState([])`` whose state is used only as
    an array) or reads only keys it does not have is reported, with the keys
    it has. The generated relation routes answer
    ``{<parent>_id, <end>_count, <end>: [...]}``.

  Generated routes declare ``response_model=None``, so the app's OpenAPI has
  no response schema to compare with; the handler's ``return`` statements are
  what it sends. Only certain cases are reported. Left alone: a body with a
  spread or a computed key, or one held in a variable that is changed
  (``delete body.x`` included); a model with a model-level validator that runs
  before field validation (``root_validator(pre=True)``, ``model_validator``
  in before or wrap mode), which can fill a missing field (a field validator
  cannot); a handler that takes the ``Request`` and can read the raw body
  itself; data passed on or read by one of its real keys; Axios bodies when a
  request interceptor or ``defaults.transformRequest`` is set, and Axios
  responses when a response interceptor or ``defaults.transformResponse`` is;
  and, for ``fetch`` and its
  wrappers, a read of a key the success body lacks (the parsed body is the
  error body too). A ``fetch`` method is the options object's own top-level
  ``method``, wherever it is written. The same finding is attached to the
  write that introduces it.
- ``ruff`` lint, limited to defects that break at runtime (syntax errors, undefined
  names, redefinitions, repeated dict keys, invalid format strings and comparisons,
  ``raise NotImplemented``, a value returned from ``__init__``, a non-string in
  ``__all__``, logging calls whose arguments do not match the format).
  Style rules are not run, and a ``ruff.toml`` or ``pyproject.toml`` inside the app
  is not read. A file ``ruff`` cannot read (rule ``E902``, for example one that is
  not UTF-8) is not a lint finding: it is reported as ``validation: ruff did not
  run (it could not read N file(s), first: …)``, so the result does not claim the
  file was linted.
- Optionally ``tsc --noEmit``, ``cargo check`` and ``kotlinc``, per project.

Every import a generated frontend makes is resolved against the files it
actually ships, with no install, no bundler and no shell, so this runs in every
configuration. A relative import that resolves to no file, and a ``.jsx`` /
``.tsx`` file containing JSX in a project that imports no React and configures
no automatic JSX runtime, are blockers: both leave a blank page in the browser.
The JSX rule reports one finding per project, naming the files, since one
config line fixes them all. Import specifiers are read from code tokens, so
the word ``import`` or ``from`` inside a string literal or JSX text is not
an import.
An undeclared package is reported as a warning, because a bundler can still
satisfy it. The JSX rule matters even where build verification is enabled:
``React is not defined`` is a runtime error in a bundle that builds cleanly, so
no build check can see it. A Vite project counts as using the automatic runtime
only when its ``vite.config`` imports ``@vitejs/plugin-react`` and calls it
inside ``plugins: [...]``. Declaring the package in ``package.json``, or naming
it in a comment, does not enable it. Dependencies whose version is ``"latest"``,
``"*"``, ``"x"``, empty or another dist-tag are one warning per ``package.json``,
naming the packages: the install still works, but takes whatever npm publishes
that day, so the build is not reproducible. The warning points at the lockfile
when the project has one and never suggests a version number.

Create forms are checked statically against the backend's ``*Create`` Pydantic
schemas, because the runtime probes drive HTTP and never render a page. Each
finding names the file, the line and the change to make:

- ``api url:`` (blocker) — the API helper builds URLs as ``BASE + path`` and
  resource paths are written without a leading ``/``, so requests go to
  ``http://localhost:8000expense/``.
- ``form create target:`` (blocker) — Create reuses the edit handler with an id
  placeholder of ``'new'``, and the handler picks update whenever the id is
  truthy, so Create sends ``PUT /<entity>/new/``.
- ``form route:`` (blocker) — a New/Create link points to a path that no
  ``<Route>`` declares. The rule is skipped when a route path is held in a
  variable.
- ``form json textarea:`` (blocker) — the form is a single textarea whose text
  is passed to ``JSON.parse`` as the record.
- ``form field type:`` (warning) — a text input for a field the create schema
  types as a number, date, datetime or boolean. It is reported only when the
  field-to-schema mapping is unambiguous. It is a warning because a user who
  types a valid value can still save.

For discovered TypeScript projects and frontend applications, required checks
that are disabled, unavailable, timed out, or only partially run are recorded as
``validation unverified:`` warnings. Their effect on completion depends on the
check: some contribute to the completion gate, while others are advisory.
The editor follows the ``done`` event's ``incomplete`` flag; it may present an
application as ready with unverified checks. Read the findings to distinguish
checks that passed from checks that did not run. Optional lint checks remain
advisory.

Checks that execute generated code (the import and runtime probes and the
compiler and build checks) run in the bubblewrap sandbox described in
:ref:`spec-driven-shell-tools`, with no network. Two skip reasons come from
that. *The sandbox is unavailable* means the host could not start it; the check
is skipped rather than run unconfined. *Its dependencies could not be
fetched* means ``cargo check`` needed a crate the run had not already
downloaded through ``run_command``. Both are reported as checks that did not
run, not as defects in the generated code.

The runtime checks (the import smoke check, the boot-and-create probe and the
replay of a retained ``test_api`` workflow) report a run that did not happen
(no sandbox, a timeout, a dependency missing from the validation harness, such
as ``httpx``) as ``validation unverified: runtime check (<check>) did not run
(<reason>)``. The auto-fix loop does not spend rounds on them, and the run
ends *unverified*: neither verified complete nor incomplete. A retained
workflow that did not run is kept and replayed on the next source revision;
only a workflow that ran and failed is an ``api scenario:`` blocker; one the
agent retired as replaced (``test_api(action="retire")``) is not replayed. A repair
after which the probe can no longer measure a backend it measured before is
not accepted as "all blockers fixed": the rollback check ranks that tree
behind the measured one.

The same applies when these checks are turned off for the run
(``enable_import_smoke_check=False``): the finding is a warning, not a
blocker. One cause of a probe timeout is found without running anything: a
``main_api.py`` that calls ``uvicorn.run()`` outside
``if __name__ == "__main__":`` never finishes importing, and is reported as an
``application startup:`` blocker that names the line.

When both toolchain validation and shell tools are explicitly enabled, npm is
available, and project dependencies are missing, ``verification setup:`` is
instead actionable. The agent may use its existing authorized dependency tool,
then request fresh validation. This is a verification-only obligation: Phase 3
does not force application edits to satisfy it. Missing system tools or disabled
permissions remain unverified; validation itself never installs anything.
Authorized TypeScript checks prefer the project's installed compiler over a
potentially incompatible global version.

Frontend production builds are distinct from typechecking. A discovered app's
configured ``npm run build`` is checked only when both toolchain validation and
shell tools are explicitly enabled and its dependencies are already installed.
Validation never installs packages or enables shell access. Configured checks
execute generated build scripts under that explicit permission, inside a fresh
bubblewrap sandbox with no network. Source is checked before and after execution. Results,
failures included, are reused only for unchanged source and dependency-install
markers during the same run; source changes during a build invalidate its
result. The same holds for ``tsc`` (also keyed by the compiler it would use),
``cargo check``, ``kotlinc`` and the ``pip install --dry-run`` check: Phase 3
re-validates after rounds that edit nothing and after a rollback, and those
passes do not run the same check on the same tree again. A check that did not
finish (a timeout, no sandbox, an unreachable package index, unfetched crates)
gave no verdict and runs again on the next pass. With shell tools
disabled, a separate external build can verify the output, but raw generation
continues to state that frontend build verification is pending.

Once a frontend's build has passed on the current tree, Phase 3 opens its
screens in a headless browser (``validation/ui_probe.py``). Inside the same
no-network sandbox it first creates records through the app's own API with the
create probe described above, into a scratch SQLite database, then boots the
generated FastAPI backend on that database (on every local port the built
bundle calls), serves the build output under its base path (``vite build --base
/app/`` included; ``index.html`` for page loads, the backend for any other path
the build does not hold) and opens each screen in a fresh tab: ``/``, every path
the router declares (``<Route path>``, nested relative routes,
``createBrowserRouter`` objects and their ``children``, the scaffold's
``ui_components.json``), then same-origin links found on the opened pages that
match a declared route (any link, when a route's path is computed), which is how
a ``/orders/:id`` screen is reached. A page that is still empty once the network
is quiet is looked at again for up to 3 seconds, so a late first render is not
called empty. At most 25 screens, 15 seconds each and 150 seconds in all for
every frontend of the run together (less when the run has less time left).
Nothing is clicked or typed, so forms and buttons are not exercised.

Every finding is a ``screen warning:``; nothing from this check blocks, counts
as a blocker or is sent to repair, on a fix run too. A finding names the route
and, when the router names it, the page's source file:

- a page that showed nothing (no text and no visible control anywhere in the
  body, portals included), with what it was opened with: no signed-in user, and
  the check's sample records or an empty database;
- an uncaught exception, said apart for a page that still rendered;
- React Router's default error page, which hides a render error from the
  exception log, and console errors;
- a 5xx from the app's backend, a 4xx, a request to a local address nothing
  answers, and a page that did not settle (a request still open, or a fetch
  loop);
- a created record that the API lists but whose text is not found in the page
  text or form fields (input and textarea values, select options) of any opened
  screen, naming the screens that load its list route, or saying that none
  does. The create probe's own text samples (``p3-order-title``) are the markers
  looked for; a value drawn on a canvas or shown only as an image is not seen.

A warning that starts with ``not checked:`` says what the check could not look
at, not what the app does: a 5xx "in the sandbox (no network there)" when the
backend's log for that screen shows a connection or name-resolution error, a
screen the check could not finish, a backend that did not start for the check,
the time limit stopping it, and records it could not check (its sample was
refused or needs a record it could not create first, the record has no text
field, its list route needs a parent record, or the API lists it only to a
signed-in user).

A screen that threw and showed nothing is also written to the trace, under the
``screen_blocker_candidates`` key of the ``phase3_screens`` phase's exit, so the
precision of that rule can be measured on real runs before it blocks anything.
``screens: not run (reason)`` -- no browser, no Node.js, no sandbox, no build
output, or the check turned off -- is a note that does not withhold verified
completion.

The browser is a Chromium-family binary found on ``PATH``
(``chromium-headless-shell`` in the worker image), driven over the DevTools
protocol by a Node.js script that uses Node 22's built-in WebSocket; nothing is
downloaded. WebGL runs in software (SwiftShader). CORS is not judged, since the
deployed origins are unknown, and host names other than ``localhost`` resolve to
nothing. Results are reused for an unchanged tree; a check that did not run is
tried again, and a check stopped midway keeps the screens it opened. Its time
shows in the trace as the ``phase3_screens`` phase.

Generated create probes use distinct fixture values and native association-link
payloads. A legitimate 4xx rejection of guessed data is **unknown**, not proof
that every valid request fails. An unresolved create probe can be discharged by
a current-revision ``test_api`` scenario on the same backend in which the app
booted and a successful ``POST`` to that endpoint (a path its route template
serves: ``/order/7/line/`` for ``/order/{order_id}/line/``) is followed by a
successful ``GET`` of that path (or a path under it) returning the same record:
the same ``id`` and at least one other equal, unredacted scalar field. The
scenario as a whole need not pass; a failing negative check in it does not
discard a create that was read back. Health responses, an isolated
200, stale results, and evidence from a different route/backend do not qualify.
An ``action unverified:`` finding is discharged by a passing current scenario
whose 2xx ``POST`` called that action route, unless the response body was cut at
the report's size limit or is a failure envelope: no record ``id`` and an explicit
flag that the call did not act (``success``/``ok``/``result`` false, an
``error`` or ``detail``, ``"status": "error"``), at the top level, in a wrapper
object or in every item of a list. A record in any state, such as an offer
returned with ``"status": "rejected"`` by its reject action, shows the action ran.
Observed server crashes remain failures even if another scenario passes.
The probe drives only entity creates: a ``POST`` whose body is ``<Name>Create``
where ``<Name>`` is a mapped ORM class the model does not mark abstract. A
``*Create`` body whose handler writes no table (a signup or login form) persists
no record to read back, so it is listed in the probe report as
``skipped_non_entity`` and produces no finding. When the handler writes one
table under another name (``NoteCreate`` stored as ``NoteRow``) the create is
probed as that table; when it writes several, it is reported as
``create unverified:``.
When a request is refused with 401 or 403, the probe tries once to log in as an
account it registers itself, as any visitor could, and never overrides the
app's authentication dependency. It registers and logs in through the app's
unsecured register/login routes (OpenAPI paths such as ``/register``,
``/login`` or an OAuth2 ``tokenUrl``; a templated path such as
``/room/{room_id}/register`` is not one). It fills the username, email and
password, plus any other required field that is plain text or a number (a
name, a phone number, an age, a study group; a date of birth gets a past
date), and ticks a required consent box: a boolean whose name contains at
least one of the words terms, tos, eula, consent, privacy, gdpr, cookie(s),
legal, newsletter, marketing, remember, adult or age (or an age: "over" and
a number, or a number and "or older"), and whose other words are only
numbers, the words is, has, i, accept(ed/s), agree(d/s), to, the, of, and,
opt, in, confirm(ed), me, over, read, given, signup, user, data, processing,
acknowledge(d), agreement, confirmation, have, or, older, and the words
policy, conditions, service or use right after terms, tos, eula, consent,
privacy, gdpr, cookie(s) or legal (or after "of"/"and" following one)
(``terms_accepted``, ``termsAndConditions``, ``terms_of_service``,
``privacyPolicyAgreement``, ``consent_given``, ``over_18``,
``is18OrOlder``; not ``is_confirmed``, ``has_policy``, ``use_privacy``,
``privacy_officer`` or ``bypass_age_check``). A staff flag named by a
consent word alone, such as ``is_marketing`` or ``newsletter``, is ticked
too. It does not use a route that
requires any other boolean, an enum/``Literal``, an object or array, a number
that references another record or names a rank (a name ending in ``_id`` or
``Id``, or with the word level, tier, rank, grade, clearance or plan, such as
``group_id`` or ``access_level``), or a field named like a privilege: a field
whose name contains the whole word role, admin, administrator, superuser,
staff, permission, scope, privilege, manager, owner or moderator, or the words
account type or user type, in snake_case, camelCase or kebab-case (``is_admin``,
``userRole``, ``account-type``; not ``parole_officer``). Those choose what kind
of account it gets. If registration fails, it still tries the login route. If
its logged-in account is still refused, it stops there and the finding quotes
the app's answer: a 403 means the route needs a privilege or a step the probe
does not take, such as a CSRF token. A login form that needs a fixed value,
such as ``OAuth2PasswordRequestFormStrict``'s ``grant_type=password``, is not
filled; the finding quotes the 422.

A login counts only when the refused request then gets past the 401/403. The
report keeps the first refused request and its status under ``auth`` (evidence
that authentication is enforced), with the account used. An app that never
refuses a request receives no login requests. When no login works, the refused
creates stay ``create unverified`` and the finding gives the reason; a
``test_api`` scenario that logs in, creates and reads back the record settles
them, as above. Saved scenarios are never replayed to log the probe in, so its
result depends only on the source.

In an app with its own unsecured register and login routes, the probe then
checks the account table, the table the register route creates. Each ``PUT``,
``PATCH`` and ``DELETE`` under that table's generic create path gets a fresh
record. A write that answers 2xx to a request with no credentials is reported as
``open account writes: <backend>: PUT /user/{user_id}/, DELETE /user/{user_id}/
changed a User record for a request with no credentials, in an app with its own
login (POST /auth/login)``. When the anonymous write is refused, a second account
the probe logs in through the app's own routes sends the update with a new
password and a new value for a privilege-named field. If a later read of the
record shows the new privilege value, or logging in as that other record with the new password
succeeds, it is reported as ``open account writes: <backend>: PUT
/user/{user_id}/ let one logged-in account change password, role of another User
record``. Both are warnings. A ``POST`` is never reported, since an open create
is how a visitor signs up, and an app without register and login routes is not
checked. The second account is registered under its own username and email,
after the probe's own account, so it is never the app's first account or an
admin, and a privilege counts as changed only when a later read of the record
shows the new value. If the second account cannot log in because the app raises
an error, the report records ``account_writes_not_checked`` and keeps every
other result.

.. _spec-driven-authorization-check:

Authorization check
-------------------

For a backend with accounts, a second check
(``validation/authorization_probe.py``) asks what one account can do to
another's records. It runs when the probe got past a login wall, and also when
no create was refused but the backend's source names a registration route. It
boots the app the same way and uses only its public HTTP API: it registers a
first account (an app may make that one an administrator), then accounts A and
B, through the app's own routes, over https so that a login kept in a
``Secure`` cookie is followed. It overrides no dependency, forges no token and
never opens the database.

Every finding is a **warning** in the run's report, stated as what was observed.
None is a blocker: none is sent to repair, ranks a tree, counts for rollback or
withholds verified completion, and the check runs only in the Phase 3 sweep, so
the agent's ``validate_app`` never shows them. From outside, a designed feature
(a shared workspace, a role, a link the request asks for) can look like a hole;
the user decides.

Account A creates one record of every entity it can, through the entity's plain
create route. Account B then sends every update method (``PUT`` and ``PATCH``)
of each record; a reference to an account in B's body names B itself, since an
app may refuse a request made in another account's name. Where B is refused
(401, 403 or 404), A sends the same update with the values its record already
shows: only when A is allowed is the refusal a rule of the app, so a route that
refuses every caller is not one. B also sends the routes under each record: a
route whose tail names a relationship links another of A's records to it (since
attaching one's own record, as a like or a vote does, is what such routes are
for), and any other route is sent only when it takes the record's own create or
update body, since an action with a body of its own (a rating, a vote) is what
other accounts are meant to send. Then every other entity's create and update
with A's record in each relationship field whose foreign key is stored on A's
record, and the single and bulk deletes; a single delete that refuses B is a
rule too. A reads its record back after each request, and that read-back is the
result: a request counts only when a field now holds the value B sent under
that name, a link now points at B's record, the record is gone, or A is refused
its own record right after B's request was accepted. A timestamp, counter or
derived value that moved on its own does not count. A's own account record is
tested last. A caller who is not logged in sends every write route a valid
request and reads every ``GET`` route, and an account that deleted itself sends
its old token again.

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Observation
     - ``authorization:`` warning
   * - B changed, linked or deleted A's record through a route, and the app
       refused B on that record elsewhere (an update A may make, or for a
       delete, the single delete)
     - ``another account could change Task.project_id through POST /project/;
       PUT /task/{task_id}/ refused that account``
   * - B changed or deleted A's own account record, or A can no longer read it
       after B's accepted request; a route the probe already reported as
       ``open account writes:`` is not repeated
     - ``another account could change Member.email through ...; Member.email
       identifies the account``. Only an e-mail, username, password or phone
       field, or an owner locked out, is described as touching the account
       itself
   * - B changed or deleted a record on which the app refused B nowhere
     - nothing: shared data, or a record locked for everyone
   * - A write route answers a caller who is not logged in with success
       (register, login, session routes and the account entity's own create
       are left out, and so are the account routes the probe already reports)
     - ``a caller who is not logged in could successfully send ...``; only
       the status was seen
   * - A ``GET`` route returns A's email or phone to a caller who is not
       logged in
     - ``a caller who is not logged in could read an account's email and phone
       through ...``, unless the request makes profiles or contact details
       public
   * - A ``GET`` route returns, beside the value A logs in by (its e-mail, else
       its username), a credential field holding a credential-shaped value
       (A's password, a hash or token format, or 20 or more mixed characters)
     - ``... returned an account's `password` to other callers``. A token that
       is no password is left out when the request makes profiles public
   * - A route that refuses anonymous callers accepts the token of an account
       that deleted itself and whose login is then refused (400, 401, 403, 404)
     - ``... still accepted the token of an account that had deleted itself
       and could no longer log in``

What the check did not cover is one ``authorization coverage:`` warning, so that
no finding does not read as checked: it did not run, two accounts could not be
registered and logged in (an e-mail that must be verified first, a role the
sign-up requires), a record could not be set up (each is named with the answer
its create got), or the check stopped at its limits. It sends at most 1500
requests and stops 60 seconds after it started, boot included.

Not covered: a record created in another account's name, many-to-many lists in
another entity's create and update bodies, unlink routes, routes under a record
whose body is not the record's own (actions such as a rating, and method routes),
what an anonymous caller's accepted write changed, and data shown to other
logged-in accounts. Whether the request makes profiles public is recognised in
English only.

.. _spec-driven-model-contract:

Model contract (Phase 0)
------------------------

Before Phase 1, the run validates the domain model with ``DomainModel.validate()``
(the checks listed in :ref:`structural-design-checks`) and records findings as
``model contract:`` blockers. No edit to the generated code can repair the model
it was generated from, so these are reported, not fed to the auto-fix loop, and
they survive every Phase 3 pass.

- Every validation **error** is promoted.
- **Warnings** that make any generated CRUD app unusable are promoted at once:
  mandatory creation cycles, and class names or attribute names that differ only
  in case.
- **Warnings** about what one family of BESSER generators emits are promoted once
  Phase 1 has run and only when the scaffold came from that family: Python keyword
  names, empty enumerations, foreign key column clashes and subclasses declaring
  ``id`` (FastAPI and Django scaffolds), join table name clashes and class-typed
  attributes (FastAPI), and a root ``id`` attribute that is not the identifier
  (Django). A modify run knows its scaffold from the seed run and checks
  everything at Phase 0.
- Other warnings (an attribute and an association end sharing a name, isolated
  classes, abstract classes without a concrete subclass, parts with several
  mandatory owners, duplicate associations) describe an app that is generated but
  degraded, and stay warnings.

Severities
----------

Findings are classified into three severities:

.. list-table::
   :header-rows: 1
   :widths: 16 84

   * - Severity
     - What lands here
   * - ``blocker``
     - Syntax errors; dependency conflicts; a Dockerfile referencing a file
       that doesn't exist; unresolvable local imports; an ORM module that
       fails to import or to configure its SQLAlchemy mappers (``mapper
       config:`` — the generated ``sql_alchemy.py`` is imported in a
       subprocess and ``configure_mappers()`` is run, which is the only way
       to see a ``relationship()`` whose string arguments resolve to
       nothing); a requirement the user stated that the code does not
       implement (``requirement:`` — the verbatim request is turned into
       atomic requirements once and each is judged against the generated
       code, with every "implemented" citation re-checked by the harness;
       only behaviour the request states is listed, so a vague request
       yields few requirements or none. The judge cites a list of
       ``{"path", "line"}`` objects, one per source line, and explains in a
       separate note; every citation must be found in the file it names
       and at least one found line must enforce or perform the
       requirement (evidence stored by older versions as one
       ``path: quote`` string is still read and checked the same way).
       A citation is code copied from the
       cited file: a whole line, a statement spread over up to 25 lines, a
       complete call of at least 12 characters or a run of at least 30 from
       inside a line (never from its strings or comment), or the start of a
       line or statement, at least 20 characters, that the judge cut with
       ``...``. An abridged or listed quote ("a ... b", "a; b", "x.py: a and
       y.py: b") passes only when every part is code in the file it names,
       and each part is checked on its own text, so an invented part cannot
       supply the proof. A part that is exactly one or more whole string literals
       of a line (``"Booking end date must be after ..."``) counts beside a part that is
       code; quoted alone it proves nothing, and the note names the line it is in
       when it is a message (it holds a space) or at least 20 characters long; a
       short one-word literal (``"ok"``, ``"admin"``) is found in almost any file and
       counts as code that exists nowhere. A citation that is only a fragment of the cited
       file's code, too short for a line of its own (``<Table id=`` for
       ``<Table id="orders" ... />``, ``title=``), is found but proves nothing, like a part
       of an abridged quote: another cited line must prove the requirement, and the
       fragment does not count as code that exists nowhere. A fragment of at least eight
       characters with a word in it has its line(s) named in the note ("a cited part is part
       of ``x.tsx`` line 6; cite that whole line"), and a verdict whose citations are only
       such fragments counts as misquoted code; a shorter one (``title=``, ``({``) names no
       line. A line of punctuation alone
       (``{``, ``/>``, even followed by a comment) identifies no code and proves nothing.
       A quote that starts or ends inside a string or comment
       is not code. An explanation after a closed statement or element
       ("<Plot /> which renders ...") is not matched. Spacing is not compared, and
       quotation marks are compared by kind, not glyph: ``'``, ``“ ”``, ``‘ ’``, ``« »``
       and the fullwidth ``＂ ＇`` read as ``"`` on both sides, since some models return
       them typographic. Some providers return a citation with its double quotation
       marks left out (``<button type=submit>``): a citation with no quotation mark
       at all that no line matches is compared once more with the file's quotation
       marks removed (a quote that kept some marks is compared with them), and
       locates a line only when exactly one line of the cited file then reads as the
       whole quote; with two or more it is refused, never picked.
       Prose, a paraphrase or
       another file's code leaves the requirement unverified. A Python
       quote that matches no line is also located by syntax tree, so a
       formatter's wrapping parentheses, adjacent string literals merged or
       split, an ``if``, ``for``, ``while`` or ``with`` header quoted on one line, and a
       statement quoted on one line still match; so does a string split by a formatter
       into ``"..." f"..."`` pieces quoted as one. A quoted
       decorator is located but proves nothing alone; quoted with a ``def``,
       it must be on that definition. A line quoted from the wrong file is
       rejected, and the note names the file it is in. Every citation of a
       verdict must be found. A validation or rule requirement whose text
       states a constraint (``must not``, ``at least``, ``between`` ...) needs
       a line that enforces it, not a declaration. A constraint word counts
       as a whole word ("lesson" does not state "less"). A value the
       requirement lists ("one of: positive, mixed, negative") still states
       that the field takes only those values. A Python enumeration, cited
       itself or as the type of a field, is such an enforcement when a field
       of the same file that the requirement names is of that enumeration,
       its values are exactly the ones the requirement lists, and the
       requirement states no constraint beyond those values. With a class model and an agent model, the list also holds
       one requirement built from the models: every intent that creates,
       lists or changes the model's records reads or writes them through the
       app's code (its models, database, API or a helper), while a greeting
       or a fixed fact may keep a fixed reply; the citation is the line that
       reads or writes the records);
       partial requirements and unverified evidence (distinct from proven missing
       behavior); a checklist item whose deterministic check fails (an action
       handler still a stub, a requested route the app does not declare, a
       missing frontend; a stub already reported as an ``action contract:``
       is not reported a second time) and unimplemented action contracts;
       a method button that takes its row id from a table of another
       entity; frontend-contract and
       data-contract violations; create forms that cannot save (``api url:``,
       ``form create target:``, ``form route:``, ``form json textarea:``); frontend calls
       whose body or response use contradicts the route (``api call:``); ``ruff`` **F821** / **F822** / **F823**
       (undefined name — the classic "ships green, boots dead" bug) and
       **F811** (redefinition, e.g. an ORM model shadowed by a Pydantic model
       of the same name); per-project toolchain errors from ``tsc`` /
       ``cargo`` / ``kotlinc`` and a failing frontend production build
       (``frontend build [...]``). By prefix, the blockers are ``Syntax error
       in``, ``Dependency conflict in``, ``... but it doesn't exist``,
       ``frontend contract:``, ``api url:``, ``form create target:``,
       ``form route:``, ``form json textarea:``, ``api call:``, ``data contract:``,
       ``model conformance:``, ``model contract:``, ``missing module:``,
       ``mapper config:``, ``application startup:``, ``python contract:``
       (which also reports undefined names, those behind ``from x import *``
       included), ``runtime unverified:``, ``api scenario:``,
       ``create contract:``, ``action call:``, ``requirement:``,
       ``requirement partial:``, ``requirement unverified:``,
       ``action contract:``, ``task incomplete:``, ``verification setup:``
       (dependencies a permitted install would provide), ``tsc [``,
       ``cargo [``, ``kotlinc [`` and ``frontend build [``. These drive the
       auto-fix loop.
   * - ``style``
     - The ``(+N more ... truncated)`` notes of ``ruff``, ``tsc``, ``cargo``
       and ``kotlinc``, which are counts, not defects. The lint selects no
       cosmetic rule, so no ``ruff`` finding is ``style``.
   * - ``warning``
     - Everything else, including endpoint-coherence findings (report-only for
       now), ``task unverified:`` (a checklist item nothing closed: no
       passing check and no ``task_list`` done, or the model recorded it as
       ``blocked``, whose reason is shown; the finding says whether the files
       the item names were edited. It is also raised when the model closes
       one of the harness's own items that no other check covers (a modeled
       rule the generators could not enforce, a derived attribute, or the
       initial value of a derived enumeration attribute; they are marked when
       the checklist is seeded, so a run resumed from a checkpoint written
       before that mark existed does not report them) on its word alone. The run report lists it as not confirmed; it is not a
       blocker, not counted in the blocker number, not sent to the repair
       loop, and on a fix run it is not promoted to the reported failure. A
       missing behavior behind it is still caught by the build, startup and
       create checks and by the requirements judge when the request states
       it),
       ``form field type:``, ``agent left out:`` (the project has an
       agent diagram, and the web app generator's output has no ``agents``
       folder because it was generated when no screen held an agent
       component; read from the files, so a modify run that changes the GUI
       without regenerating keeps it; the run report lists it as not
       checked), the
       model-derived acceptance matrix (with
       a GUI model whose screens bind at least one class, a class no screen
       offers to create, through a table with action buttons, a form or a
       create button, is not asked for a create path), ``entity coverage:`` (for an ``other`` target used through screens, the
       judge also checks that every entity can be created, viewed, edited and
       deleted, since the acceptance matrix only reads web routes and pages;
       like the matrix it is reported, never a blocker. Abstract classes are
       left out, at most 20 entities are listed, and the check is skipped
       when a BESSER generator built the base, such as a BAF chatbot or
       Python classes), ``requirement out of
       scope:`` — a UI requirement on a run that neither has nor asked for a
       frontend, which a backend-only output has no files to satisfy —
       ``authorization:`` and ``authorization coverage:`` (see
       :ref:`spec-driven-authorization-check`) — and
       ``validation unverified:`` — a required check that did not run, such
       as a runtime check (``validation unverified: runtime check (...) did
       not run``) or a requirements judge that returned no verdicts. No edit
       to the generated code can settle these, so they spend no repair turns,
       but they still prevent verified completion.

One Python defect is one finding: a file that does not parse is reported as
``Syntax error in``, without the python contract's and ``ruff``'s copies of the
same parse failure, and an undefined name the python contract reports (it also
judges names behind ``from x import *``, where ``ruff`` cannot) drops ``ruff``'s
``F821`` for the same name and line. A name used many times in one file is one
finding, at its first use, ending ``(+N more uses)``: ``ruff`` ``F821`` /
``F811``, the python contract's undefined names and ``tsc``'s ``TS2304``. One
forgotten import in a 40-endpoint router is two blockers, not 105.

A finding's message is written for the agent. Where the user reads about it (the
run card, ``BESSER_GENERATION.md``, an incomplete notice) it is said in one
sentence that ``validation/user_wording.py`` builds from the same prefix, for
example ``Creating order (POST /order/) could not be tested automatically: it
needs an existing Customer first.``. A new prefix gets its sentence there;
without one it reads as a generic sentence.

Only ``blocker`` findings spend LLM turns. The ``done`` event reports
``blockerCount`` — completion-blocking defects and required verification gaps
still standing when the run ended. A non-zero value means the downloaded output
is not verified complete; it does not necessarily mean that the code is broken.

.. _spec-driven-design-fidelity:

Design fidelity
---------------

When the GUI model carries a stylesheet, the React generator writes it to
``src/design.css`` and renders every designed screen from it. To keep the agent from discarding that
design, Phase 2's system prompt gains a short section, built from the project's own ``design.css``: its ``--ds-*`` variables
and its classes grouped by role (layout, navigation, cards, forms, buttons,
tables, status, text), then the rules — never edit ``design.css`` (new rules go
in ``src/design-overrides.css``, imported after it); edit designed pages in
place with ``modify_file`` / ``replace_file_lines``, never ``write_file`` over
one; keep each page's nav, layout wrappers and generated components
(``TableBlock``, ``MethodButton``, ``CrudButton``, ``FormBlock``, charts,
metric cards); build new markup from the design classes rather than bare tags,
``style={{...}}`` objects or hex colours; plus brief design-quality guidance.
The general styling advice it would contradict is switched off for such runs:
the "loose hint" framing of the GUI model, Rule 2's ``write_file`` fallback for
pages, and the "define concrete CSS" / "one shared stylesheet" bullets. Without
``design.css`` the section falls back to the GUI model's stylesheet, embedded in
the prompt; with neither, the prompt is unchanged.

The bounded auto-fix loop
-------------------------

Blocker-level findings drive a repair/recheck loop within the remaining turn,
cost and runtime budgets. The loop stops when blockers reach zero, after two
consecutive unchanged/repeated source states, or after three consecutive rounds
that change the tree without improving it. A round is progress when the tree
scores better, the source changed, or a verification obligation was discharged
(a checklist or scenario change counts only if it changed the set of blockers)
— writing no source is not by itself a stop, because a round spent closing
checklist items or correcting a scenario can resolve blockers without touching
a file, and a round whose edits were all rejected feeds those rejections into
the next attempt's prompt. A round that wrote nothing and never called an edit
tool is the exception: it leaves the next prompt identical to its own, so it
ends the loop immediately rather than paying for the same attempt twice. A larger error count does not trigger an automatic rollback: fixing one
import may expose several previously unreachable CRUD failures. Unresolved
output remains explicitly incomplete. Startup and data-entry failures are
repaired before spending tokens on business requirement judgment.

A fix turn whose reply is cut off at the output-token limit is retried with
an instruction to emit less, up to four times per attempt, rather than
applying a half-written edit.

For concrete code defects, an attempt without a successful edit — the model explained
the fix instead of making it, or read files until its turn budget ran out — is
re-prompted once, with ``modify_file`` forced through ``tool_choice`` where the
provider honours it and a reminder in the message either way; if that still
produces no edit the attempt ends, and the log says ``ended with no successful
edit``. Phase 3 tool calls are recorded in the trace and the recipe exactly
like Phase 2's.

Evidence-only findings and generated API assertions do not force a source edit.
The agent can inspect saved scenarios, correct a mistaken expectation with a
specification-grounded reason, or cite an already-present implementation.
A correction that leaves out requests which passed in the scenario's last run
gets them back at the end of the corrected scenario, their ``{{N...}}``
references renumbered, and the result's ``restored_checks`` names them once the
scenario ran. Resending the same corrected list later reruns that scenario, the
restored requests included, with no new reason. It is refused, as before, when
such a request reads a response it would no longer have, when the scenario would
exceed 20 requests, when the app did not boot (the restored requests did not run),
or when a restored request fails at the end (it ran after the corrected requests
changed the data): the saved scenario is then kept unchanged and the result
carries the report. A resumed run keeps the corrected list it was sent as.
Accepted scenario/checklist corrections count as progress; changing a judge's
opinion alone does not. Cancellation is checked between repair and validation,
so stopping a run does not trigger another paid coverage judgment.

Repair prompts retain the original specification, current files/symbols, actual
action handlers and recent rejected operations. The parts that are fixed for the
run (the original specification, the inspection handoff, the domain model and the
requirements ledger) form the system prompt, identical in every repair round so
the provider can serve it from its prompt cache; each round's message carries the
files, the blockers and the excerpts. The action handlers and the relationship
mutation coverage are sent only when a blocker concerns behaviour, not for a round
whose blockers are all syntax, import, lint, compiler or dependency errors. The
prompt lists at most 30 blockers in repair order, counts the rest by kind, cuts a
message over 1,200 characters with a marker, and names a requirement already in
the ledger by its id. Validation still records and reports every blocker.
A fix attempt is skipped when every remaining blocker is a requirement whose
cited code exists but whose citation is malformed and already had its
citation-repair round, and that has no fragment the fix agent may complete (below):
no edit can change it. A citation of code that exists nowhere keeps its attempt. The
loop also ends when the last attempt faced the same
blockers (requirements compared by id, status and citation label, since a re-judge
rewords its note; other blockers by message), its edits were not all rejected,
and each remaining blocker is a ``requirement unverified`` verdict the
judge left without a citation (or with a malformed one) after its focused
re-judge on the current source: another attempt would get the same request. A
citation of code that exists nowhere, or one whose cited body fails its role (an
HTTP 501 scaffold, a no-op), keeps its attempts. The blockers keep their severity
and are reported. ``validate_app`` lists the requirement blockers of the last
judgment too (no new judge call), so the fix agent is never told the app is
verified while the run holds one; their message tells it to implement behaviour
only if it is really absent, since citations are re-judged automatically. When the
judge found the requirement implemented and one of its citations is a fragment found
in exactly one line, the message names that line instead and tells the agent to cite
it: ``validate_app`` takes ``citations=[{requirement, path, line}]``, and accepts only
that whole line (the judge's own citation, completed, never a line the agent picks),
checked exactly like the judge's citations against the source of the current
judgment. A verified one replaces the judge's citation in that judgment, its note
says the fix agent supplied it, and the trace records a ``requirement_cited`` event;
it holds until the cited files change. Any other citation, a requirement the judge
left unverified, partial or missing, source edited since the judgment, or a
malformed item is refused, one reason per citation. A
judgment of other source than the judge would read now (an edit to a judged file)
is listed as advisory, "judged before your latest edits; it is re-judged when you
finish", and does not count as a blocker there; an edit to a file the judge does
not read (a README) changes nothing.
Excerpts use the file/line formats emitted by the checks. Unchanged source is not
counted as repair progress merely because a judge's blocker count changed.

Requirement judgments are cached by source/configuration revision. After an
edit, a verified requirement keeps its verdict without being judged again only
while every file its citation names is byte-identical to the version it was
verified against (a verdict with no recorded file hashes is judged again). An
edit to a cited file re-judges it, even when the cited line survives; an edit
to an uncited file does not, so a change elsewhere that stops the cited code
from running (``main.py`` no longer including the cited router) keeps the
verdict. A single bounded citation-repair
judgment per revision, right after the judgment or on a later pass, may revisit
only unverified entries without a code change. Evidence must name a real application source file and quote an
executable line; traces, recipes and unrelated files cannot validate a claim.
Failure to extract requirements remains unknown and blocks verified completion.

.. _spec-driven-modify-requirements:

A modify run is answerable for its own request and for what it breaks, not for
the whole application again. When the seeded recipe records requirement
verdicts, the run keeps the previous run's requirements (same ``R`` numbers, at
most 128) and extracts only the user messages the previous run's request did not
hold: the messages after the previous run's when those start the conversation,
otherwise all later messages (a message sent to a run that was cancelled counts
as new, and so does one repeated after the previous run), with the earlier
conversation and the numbered requirements as context, so "retry", "yes" or "the
same for the other screen" resolve. Their requirements follow the previous ones.
The extraction also names the previous requirements a new message changes, asks
for again or removes. Those stay in the ledger and are judged as usual (an
extraction that names every requirement cannot empty it): met, they are met;
unmet, they are a warning, ``requirement changed by your newest message:``, never
a regression and never work for the fix loop; when it was implemented as the
previous run ended, the warning says so, which tells a deliberate change from a
break. A changed or restated requirement comes back as a new requirement of its own.
It also says whether the messages only ask to retry, continue or finish earlier
work. When that extraction fails, the whole request is extracted instead and
judged as in a run without a previous ledger. When the editor sends its own
requirements, a previous verdict is kept for
the requirement with the same trace id, else the same text.

The previous verdicts are taken as of the tree the run starts from: an
"implemented" citation is checked again against it (one that no longer matches
stays "implemented" but is judged again), and every verdict records the hashes of
the files it cites or names and of the files they rest on; a verdict that names no
file records none and is judged again. A verdict whose files this run leaves
byte-identical is not judged again, whether it was met or not. A recipe whose
verdicts cannot be read is ignored, and the whole request is judged. Then:

- a requirement that was implemented and is no longer is a blocker, and its
  note says it was implemented when the previous run ended;
- a requirement of the new messages that is not implemented is a blocker;
- a requirement the previous run had left partial, missing or unverified that
  is still so is a warning, ``requirement still open from the previous run:``.
  It does not drive the fix loop and does not make the run incomplete; the
  report lists it with "open since the previous run". When the new messages only
  ask to retry or finish earlier work, these requirements are this run's own
  instead: judged, and blockers while unmet. The incomplete-extraction gate and
  the requirements derived from the model or the agent are never carried this way.

Known limit: the files a verdict rests on are the files it cites, their local
imports and the entry files that mount them. A frontend screen that reaches the
backend only over HTTP does not rest on the backend's files, so an edit to the
backend route it calls keeps the screen's verdict.

When the editor sends the modeling assistant's requirements (the request's
``requirements`` field, see :doc:`api`), the ledger is built from them instead of
extracted. Only the kinds that can be judged from code are judged (``uniqueness``,
``validation``, ``rule``, ``computed``, ``transition``, ``action``, ``ui``,
``integration``); ``data``, ``quality`` and ``technical`` requirements are not.
Each verdict keeps the requirement's trace id as ``ref``, and the recipe's
``requirements_source`` says where the ledger came from (``editor`` or
``extracted``). The assistant extracts from the user's first request only, so a
request carrying later messages is extracted as before, with a warning in the log.

The judge reads a digest of the app of at most 200,000 characters: the schema
files first, then the backend code ranked by the share of its lines the agent
wrote (a module the generator never wrote first, untouched generator output
last), with 30% reserved for pages and components. When the whole app does not
fit, a generator file the agent edited in under half of its lines is shown as
its edits with three lines of context; the unchanged generator lines left out,
and the places where generator lines were removed, are marked. A file over 1,500
lines is not compared (its share is unknown) and is shown whole. Files cut, left
out or shown as edits are listed; a partial, missing or unverified verdict naming
one is judged once more with that file shown whole. The focused round shows up to
ten distinct files, taken from each requirement's files in turn, so one
requirement's files cannot crowd out another's. That focused answer replaces the
verdict only when it is a verified "implemented"; otherwise a missing or partial
from it proves no absence, so the checked verdict keeps its status, and it takes
the focused judge's citation (when it gave one), note and inspection paths, which
name the files that judge saw. An answer naming a file that round left out gets
one more round with that file shown, together with every other file the answer
names (its citations, inspection paths and the files its note names), including
those the round showed for another requirement. An inspection path may be a glob
(``src/**/*.css``); the files globs match fill only the slots the named files leave.
An empty citation fails as "no source line is cited", never as a citation of code
that exists nowhere.

Stylesheets (``.css``, ``.scss``, ``.sass``, ``.less``) are judged source: a
stylesheet rule is a citation like any other line, and a rule inside a comment is
not. The whole-app digest shows them, last, only when a requirement is about
appearance (its text names a colour, font, hover, cursor, border, shadow, accent,
spacing, layout, styling and the like); otherwise they are listed as omitted, so a
verdict naming one is judged once more with it shown. In a stylesheet a line
starting with ``#`` is an id selector and is kept. A stylesheet a cited page imports
is not among the files the verdict rests on, so a style edit does not re-judge it.

Two kinds of requirement cannot be shown by citing a line of code, and the
extraction labels them: ``absence``, something the app must not have at all ("no
data export", or data invented beyond a given source such as an imported file;
never a refusal the app enforces), and ``preservation``, existing behaviour a
change must keep working as it is. Neither needs an enforcing line: a citation
of where the data comes from can verify an absence. When the judge leaves one
unverified, it is settled without a citation, never as met, and an unverified
absence is not sent to the focused round:

- An absence is ``not_provable``, with a note that it cannot be shown by citing
  code. The app's files are not scanned for it: a find could not tell the excluded
  feature from a requested one, and finding nothing proves nothing. Only the
  judge's own missing or partial verdict blocks an absence.
- A preservation is ``not_provable``. Its note names the files the judge named for
  it (inspection paths and citations), and the files they import or are mounted by,
  that changed since the previous version (in a modify run the files as the run found
  them, before its model sync and regeneration; otherwise the stored generator
  output), or says that none did, which does not show the behaviour still works.

A ``not_provable`` requirement is a warning, ``requirement not provable:``, never a
blocker; the report lists it as not verified, with its note. A judge verdict of
missing or partial for either kind stays a blocker.

The recipe records, for each implemented requirement, ``verification`` and a
``verification_note`` giving its API half and its screens half apart. The API half
is exercised only when a ``test_api`` call that passed on the final tree hit a route
whose request runs the cited code: its decorator, its handler (also one registered
with ``add_api_route``), or a function of the same file the handler calls or names in
``Depends(...)`` (not a parameter, variable or nested function of the handler with
that name), under the literal ``prefix=`` of its router (a router whose prefix
is not a literal verifies nothing). An abridged citation ("a ... b") counts a route
only when the route runs every part of it; when a part is too short to locate, the
parts left may be code several routes share, so it counts no route ("API: not run
(a part of the abridged citation is too short to tell its route)"). The note names that call ("API: a passing
workflow called GET /x/, the cited route"). A call to another route of the same file
proves nothing; a citation no route runs (a model, an import) is found in the code
whatever ran: "API: not run (no cited line is in a route handler)", and cited backend
code whose routes the check cannot read (not Python) says "API: not run (the check
cannot read the routes of the cited code)". No check renders a
screen, so a cited frontend file is never exercised, and the note says why: "screens:
not run (the frontend builds)", "(the frontend build failed)", "(the frontend build
check did not run)" or "(no frontend)". ``verified`` means every cited half was
exercised, ``api_verified`` that the API half was and screens are cited,
``unverified`` anything else. It is reported, never a blocker: the verification
report counts a ``verified`` or ``api_verified`` requirement as confirmed and an
``unverified`` one as found in the code but not run. The recipe also records
``business_rules_stated``: ``false`` only when the ledger ran and extracted no
requirement at all and the model has no rule, ``null`` when the ledger did not
run. When it is ``false``, the repository recap and the end-of-run report say
"Your request stated no business rules, so none were enforced or checked", and
the ``AGENTS.md`` block carries it as a note outside its "Still open" list; the
next run's prompt does not get that line, from the recap or from ``AGENTS.md``.

Model conversion losses
-----------------------

An OCL rule that cannot be parsed or attached is not silently removed from the
spec-driven handoff. The converted model retains ``conversion_issues`` with
stable identifiers, diagram/element provenance, the original expression and
the conversion reason. Invalid rules stay out of executable model constraints.

The planner, customization prompt and repair prompt receive these diagnostics.
Each loss becomes a harness-owned recovery task and a requirement checked
against current application source. Closing or dropping a task cannot discharge
the corresponding requirement. If coverage verification is disabled or cannot
run, recovery remains explicitly unverified and blocks verified completion.
Successful evidence is tied to the current code revision; it does not rewrite
the user's model or prove end-to-end behavior. Diagnostics remain in the recipe
under ``model_conversion_issues`` for review even after code recovery succeeds.

The original natural-language request takes precedence when a rejected model
expression conflicts with it. Relationship names must be resolved from the
actual model and code, not guessed or automatically renamed.

A modify run treats rules broken by the user's own model change differently.
The converter records the element a rejected rule names but the model lacks
(``missing_element``, e.g. ``Person.phone``),
and every run's recipe records a fingerprint of each rule that converted
(``model.converted_rules``) next to the user's model (``model.user_model``). A
rejected rule is marked ``dropped_by_model_change`` only when the same rule
text converted in the previous run and the element it misses is in the previous
model but was removed from the new one (an attribute, association end or
class). Such a rule raises no recovery task or requirement, the agent sees it
listed as dropped (name and missing element only) with an instruction not to
re-implement it, and the verification report lists it as not enforced because
it was dropped by the model change. It stays dropped in later modify runs while
the element is still missing.

When the element was renamed instead (the class or attribute keeps its
identity under a new name), the rule is recovered as usual and its requirement
says to keep the rule and enforce it on the new name (``renamed_in_model``).
A new or edited rule, a misspelled name, an element that never existed, a rule
that already failed in the previous run, a recipe without the recorded model
or rule fingerprints, any other conversion failure, and every rule in a fresh
run are recovered as described above.

Validation is attempted within the run's runtime budget. If that budget is
already exhausted, the run records an unverified-completion blocker instead
of silently reporting an empty validation result. The *repair* half is a flag:
``LLMOrchestrator(auto_fix_issues=...)`` defaults to ``False`` for library
users — report by default, fix on request, the usual static-analyser
contract. The web deployment turns it on
(``BESSER_LLM_ENABLE_AUTO_FIX``, default ``true``), because shipping a
broken artifact as a green success is worse than spending a few more turns.

.. note::
   The heavy per-project compilers (``tsc`` / ``cargo`` / ``kotlinc``) are the
   one part of Phase 3 that is **opt-in** for the web backend
   (``BESSER_LLM_ENABLE_TOOLCHAIN_VALIDATION``) — they were the main driver of
   a duration and cost regression on non-Python stacks. Two further checks
   are on by default and can be switched off per deployment: executing the
   generated backend (the ORM import check, the isolated startup and create
   probes, and ``test_api`` replay) is gated by
   ``BESSER_LLM_ENABLE_IMPORT_SMOKE_CHECK``, and the requirement judgment by
   ``BESSER_LLM_ENABLE_REQUIREMENTS_LEDGER``. Turning either off makes the run
   report that check as ``unverified``, never as passed. Every other check
   above runs on every run. See :doc:`configuration`.

.. note::
   A check that could not run reports that it did not, rather than returning
   nothing. A collector that times out or cannot be launched emits a
   ``validation: <tool> did not run (...) - its checks were SKIPPED`` finding,
   because an empty result is otherwise identical to a clean one and the run
   would report "0 blockers" having verified nothing. These are ``warning``
   findings on purpose: not looking is not evidence of a defect.

   The same finding names every compiled language that nothing here compiles,
   for example ``validation: the Dart compiler did not run (the build server has
   no Dart toolchain; 12 file(s) were not compiled)``: Dart, C#, Swift, Java,
   Go, C/C++, Objective-C, GDScript and Scala always, and Rust and Kotlin when
   toolchain validation is off. An Android Kotlin module is never compiled with
   bare ``kotlinc``, which has no Android SDK; it reports the same way.

   For the same reason the acceptance matrix's ``route`` cell is *unmeasured*,
   not *missing*, on stacks its route parser does not read. That parser
   understands FastAPI decorators in ``.py`` files, so an Express, Django,
   Spring or axum backend reports no route finding at all rather than one per
   entity. The ``page`` and ``create`` cells are stack-independent and still
   apply. The ``create`` cell of an abstract class is *not expected*: its
   records are created through its subclasses.
   A create whose path is read off a metadata entry the page hands down
   through components (``<EntityPage entity={ENTITIES.order} />`` passing its
   ``entity`` prop on to a form that calls ``api.post(entity.path, ...)``) is
   credited to that entry's entity. An entry of a page's metadata object that
   declares no create form (``allowCreate: false``, ``readOnly: true``), or a
   page that renders the component with ``readOnly`` or
   ``allowCreate={false}``, is not credited with one.
   Through the generated typed client, only ``<x>Api.create`` (or ``bulkCreate``)
   credits a create, to the class whose create route it posts to; a page that only
   reads another class (``projectApi.list()``) is not credited with its create. An
   import line names a module, not an entity: importing ``../api/generated/client``
   says nothing about a class named ``Client``.

.. warning::
   Validation includes isolated backend execution, not a deployment or full
   browser acceptance run. Phase 3 does not build a container, and passing
   create probes or submitted API scenarios does not prove every workflow.
   Independently exercise the generated user interface and specification.

Phase 3 is not the only place findings surface: the per-write diagnostics guard
described in :doc:`how_it_works` parses every file the moment the model writes
it, so the cheapest defects never reach Phase 3 at all.
