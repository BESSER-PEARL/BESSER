Build an application with AI
============================

**Goal:** turn a reviewed model and a written description into an application
using the experimental Spec-Driven Agent.

Before you start
----------------

Open the `editor <https://editor.besser-pearl.org>`_ and create a project.
AI availability depends on the deployment: a configured free tier, a sponsored
tier, or your own provider key can power a run. A deterministic generator
remains available when you do not need AI customisation.

Describe, review, generate
--------------------------

1. Ask the assistant to model a small system, for example:
   *"Create a book catalogue with titles, authors, and a screen for adding
   and listing books."*
2. Review the class and GUI diagrams. Correct names, relationships, and fields
   before requesting code.
3. Ask: *"Generate the web app."* Creating the model does not itself start a
   generation run.
4. Choose a provider if prompted. A saved provider key can start the requested
   run immediately and incur provider charges; set the per-run budget in the
   key dialog before starting.
5. Follow the run card. Use **Stop** to cancel or wait for the result.
6. Download the archive and read its README. Review its verification findings
   before running or deploying it.

What the run does
-----------------

The agent starts from a BESSER scaffold when one fits the request. For another
stack it plans the project structure. It then edits files, validates the result,
and attempts repairs within the run's turn, cost, and time limits.

The checks cover different things: model validation checks the diagram;
generated-code validation checks the files; authorised runtime probes check
whether a supported backend boots and creates records. Disabled or unavailable
checks are recorded as unverified. Read the result's findings even when the
editor presents the application as ready.

Continue your work
------------------

Ask for a change in the same project to modify the previous application's
files while they remain available on the server. For longer-term work, push
the result to GitHub and later choose **Continue from GitHub** in the editor.

* `Editor run guide <https://besser.readthedocs.io/projects/besser-web-modeling-editor/en/latest/user-guide/spec-driven-agent.html>`_:
  run cards, providers, download, and GitHub.
* :doc:`../spec_driven_agent/usage`: Python and REST usage.
* :doc:`../spec_driven_agent/validation`: what the checks prove and what they skip.
* :doc:`../spec_driven_agent/production_deployment`: deploy the worker.
