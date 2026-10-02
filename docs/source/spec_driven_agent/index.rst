Spec-Driven Agent
=================

.. note::
   **Experimental.** The Spec-Driven Agent is under active development.
   Its pipeline, request shape, and defaults may change between releases.

The Spec-Driven Agent (also called the *LLM-augmented generator*) is the
agentic layer that sits on top of BESSER's
:doc:`deterministic code generators <../generators>`. It bridges the gap
between template-based generation and free-form natural-language code
generation.

Where a template generator (Django, FastAPI, SQLAlchemy, …) emits a fixed
scaffold from your :doc:`structural model <../buml_language/model_types/structural>`,
the Spec-Driven Agent **starts from that deterministic scaffold, lets an LLM
customise it** to satisfy a natural-language request, and then **validates the
result and repairs what it can**. It can add features the templates don't cover
(authentication, JWT, Docker, custom middleware, migrations, tests) or target a
stack BESSER has no built-in generator for (Rails, Rust/Axum, Kotlin/Spring,
Next.js, …).

The key idea: generation is **hybrid**, and the model remains the source of
truth. Where a BESSER generator fits, the LLM edits its scaffold. For a stack
without a built-in generator, it writes the application from stack metadata
and the model's contract.

.. toctree::
   :maxdepth: 1
   :caption: In this section

   how_it_works
   usage
   api
   models
   tools
   validation
   runs
   configuration
   production_deployment

When to use it
--------------

BESSER offers two ways to turn a model into code. They are complementary, and
the agent is built on top of the generators rather than replacing them.

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * -
     - :doc:`Deterministic generator <../generators>`
     - Spec-Driven Agent
   * - You provide
     - A B-UML model, plus the generator's own options.
     - A B-UML model **and** a natural-language description of what to build.
   * - What comes out
     - One template pass over the model: same input in, same output out.
     - A deterministic scaffold that an LLM has customised, then validated and
       repaired — see :doc:`how_it_works`.
   * - Cost
     - Free and offline.
     - Paid LLM calls, or the keyless free tier when the deployment configures
       one — see :doc:`models`.
   * - Reach
     - The ~25 targets listed in :doc:`../generators`.
     - Those targets *plus* features the templates don't cover, and stacks with
       no BESSER generator at all.
   * - Reproducibility
     - Byte-for-byte reproducible.
     - Bounded, not reproducible: an LLM authored part of the output.

Reach for a plain generator when the target is one of BESSER's supported
technologies and the templated output is what you want. Reach for the agent
when you need something on top of that output, or a stack BESSER does not
template.

Limitations
-----------

- **Round-trip preservation.** Each run regenerates from the model and
  instructions. :ref:`Modify mode <spec-driven-modify-mode>` carries a previous
  run's files forward, but hand-written edits made outside BESSER are not
  re-derivable from the model and are not carried across a from-scratch
  re-generation. Closing this gap is the most significant planned improvement.
- **Validation has a defined scope.** :doc:`Phase 3 <validation>` checks
  syntax, imports, contracts and lint. Authorised sandboxed checks can also
  boot a supported backend, create records, compile code, and build a frontend.
  Checks that cannot run are reported as unverified. It does not build a
  deployment container or prove every business requirement; inspect the
  findings and test the application in your target environment.
- **Fixes land in generated code.** The auto-fix loop repairs the generated
  files, not the model. A later from-scratch regeneration starts from the model
  again and will not carry those repairs.
- **Quality depends on the model.** Output fidelity tracks the chosen LLM;
  a weaker or cheaper model produces a thinner result.
