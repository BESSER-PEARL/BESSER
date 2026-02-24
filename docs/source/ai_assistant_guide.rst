AI Assistant Guide
==================

This page documents how automated assistants (e.g., GitHub Apps, CI bots,
code-generation agents) should collaborate with the BESSER project. Following
these practices keeps generated contributions aligned with community
expectations and reduces maintenance overhead.

Scope and Responsibilities
--------------------------

* Respect the same `Code of Conduct <https://github.com/BESSER-PEARL/BESSER/blob/master/CODE_OF_CONDUCT.md>`_
  and governance rules
  that apply to human contributors.
* Limit changes to the scope requested in an issue or task. Avoid speculative or
  unrequested refactors.
* Provide transparent commit messages and pull request descriptions so
  maintainers can understand the rationale for automated changes.

Preparation Checklist
---------------------

Before performing any automated edits:

1. **Understand the task.** Read the associated issue, previous discussions, and
   the relevant sections of :doc:`contributor_guide`.
2. **Inspect repository instructions.** Some directories may contain
   ``AGENTS.md`` files or other automation rules. Honor the most specific
   instructions that apply to the files being modified.
3. **Plan the change.** Summarize the intended approach (files to touch, tests to
   run, documentation to update) before editing. Share the plan in the issue or
   pull request when possible.

Implementation Guidelines
-------------------------

* Prefer minimal diffs that accomplish the requested change without unrelated
  formatting churn.
* Update or add tests alongside functional changes. When tests are not feasible,
  document the manual validation that was performed.
* Keep documentation in sync with the code, especially when adding new features
  or altering user-facing behavior.
* Avoid committing generated files, local environment settings, or large binary
  assets unless explicitly requested.

Repository Boundaries
---------------------

* Treat the ``besser/`` packages as the backend for the BESSER Web Modeling
  Editor and the standalone SDK. Changes should focus on the B-UML metamodel,
  generators, utilities, and backend services.
* The web editor frontend lives in
  `BESSER-WEB-MODELING-EDITOR <https://github.com/BESSER-PEARL/BESSER-WEB-MODELING-EDITOR>`_.
  Only modify the vendored submodule under
  ``besser/utilities/web_modeling_editor/frontend`` when explicitly instructed.
* When adding backend capabilities for the editor, update corresponding
  documentation (``docs/source/web_editor.rst``) and cross-reference
  contributor guidance.

Quality Assurance
-----------------

* Run ``python -m pytest`` from the repository root to execute the automated
  tests relevant to your change.
* Build the documentation (``cd docs && make html``) when altering files under
  ``docs/`` or when the change impacts user-facing guidance.
* Surface failing tests or build issues directly in the pull request summary,
  along with any mitigation steps you attempted.

Pull Request Expectations
-------------------------

* Use clear, factual titles (e.g., ``docs: clarify generator setup``).
* In the description, reference the motivating issue, summarize the change, and
  list the tests that were executed.
* If the change introduces follow-up work or limitations, document them in a
  dedicated section ("Known limitations" or "Follow-up tasks").
* Respond to review comments with additional context or revisions. Make it easy
  for maintainers to verify updates by quoting relevant code snippets or
  pointing to documentation sections.

Security and Privacy
--------------------

* Do not expose secrets, credentials, or sensitive model data in commits, logs,
  or documentation. Redact such information if encountered.
* Avoid making network calls to third-party services during tests or
  documentation builds unless explicitly required and approved.

By following these guidelines, automated assistants can contribute effectively
while respecting the workflows that keep the BESSER project healthy.
