Maintain the documentation
==========================

Build and preview
-----------------

Use Python 3.12 for documentation tooling. From the repository root:

.. code-block:: console

   python -m pip install -r docs/requirements.txt
   python -m pip install -r requirements.txt
   python docs/check_docs.py
   python -m http.server 8001 --directory docs/build/html

Open ``http://localhost:8001``. ``--offline`` skips external inventories for
local builds without internet access. The checker rejects build errors and
warnings. All three documentation sites must build without warnings.

Put information where readers expect it
---------------------------------------

* **Start here:** a short, complete first task with an observable result.
* **Task guides:** one goal per page, prerequisites first, then numbered steps.
* **Concepts:** explain how the parts fit together.
* **Reference:** complete options, supported inputs, and API contracts.
* **Install and deploy:** environment setup, migration, and operations.

Keep editor interface instructions in the editor documentation. Link to
them from this site. Preserve established page URLs when reorganising content.

Before opening a pull request
-----------------------------

Run every new command or example. Use actual interface labels and release
screenshots. Give images useful alt text; omit screenshots that only repeat
text. Preview at desktop and phone widths, in light and dark mode.

Add pages to a toctree and use ``:doc:`` for links within this site. Keep
paragraphs short and avoid repeating explanations on several pages.
Release notes should describe user-visible changes; put breaking changes,
environment variables, and upgrade steps in the migration guide.

The shared stylesheet uses the editor's BESSER brand colours:
``hsl(193, 45%, 40%)`` in light mode and ``hsl(195, 52%, 64%)`` in dark mode.
Review the same styling change in all three documentation projects.

Maintenance handoff
-------------------

``docs/DOCUMENTATION_HANDOFF.md`` in the repository records the three-site
structure, theme decisions, build commands, verification, and publication
workflow. The editor repository contains ``docs/SCREENSHOT_HANDOFF.md`` and a
reusable Playwright screenshot tool. Refresh shared editor images in both
repositories when changing those captures.

Publish a release
-----------------

After merging, verify the Read the Docs build succeeded and the published
version matches the release. Test the tutorial, search, navigation, downloads,
and cross-project links on the hosted sites. A successful local build does
not update the public documentation.
