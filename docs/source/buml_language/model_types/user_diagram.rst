User Diagram
============

A **UserDiagram** in BESSER is an instance of the **BESSER User Modeling
Language** — a dedicated, literature-backed modeling language for describing
end-users across many dimensions. It is maintained as a standalone project at
`BESSER-PEARL/User-Modeling-Language
<https://github.com/BESSER-PEARL/User-Modeling-Language>`_ and was derived from
a systematic literature review on user modeling in model-driven engineering:
*User Modeling in Model-Driven Engineering: A Systematic Literature Review*
(`arXiv:2412.15871 <https://arxiv.org/abs/2412.15871>`_).

A UserDiagram describes an end-user that the system (typically an agent) needs
to adapt to. It covers personal information, accessibility needs, competences,
culture, personality, goals, preferences, mood and emotional state. Once
instantiated, the diagram is consumed by the
:doc:`agent personalization <../../generators/agent_personalization>` pipeline
to tailor the generated agent to that specific user.

.. note::
   The full user metamodel contains 60+ classes and 20+ enumerations covering
   every dimension category below. The Web Modeling Editor exposes a
   **condensed subset** (``User``, ``Personal_Information``, ``Competence``,
   ``Accessibility``, ``Culture``, ``Language``, ``Skill``, ``Education``,
   ``Disability``) so that everyday modeling stays manageable — the full
   metamodel remains available as a B-UML Python module for advanced use.


Dimension Categories
--------------------

The User Modeling Language is organised into nine dimension categories. Every
user instance is a composition of parts from one or more of these categories,
rooted on a single ``User`` instance.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Dimension
     - Covered concepts
   * - **Personal Information**
     - First / last name, age, nationality (ISO 3166), address, gender,
       sexuality, political belief, interests, hobbies, topics.
   * - **Accessibility**
     - Anthropometric, Speech, Hearing, Cognitive, Sight, Memory, Mobility,
       Motoric, Physiological State, Disability (with ``Aspect`` enum:
       Sight, Hearing, Mobility, Cognitive, Memory, Learning, Mental Health,
       Social Relationships, Speech).
   * - **Personality**
     - Attitude, Bias, Interpersonal / Intraphysic / Instrumental Motivation,
       Trait (Big Five, Myers-Briggs), Characteristic.
   * - **Competence**
     - Education (degreeName, degreeType, fieldOfDegree, providedBy), Skill,
       Language (ISO 693-3 code, CEFR level), Knowledge, Experience, Topic.
   * - **Culture**
     - Hofstede dimensions (femininity↔masculinity,
       collectivism↔individualism, power distance, uncertainty avoidance,
       temporal orientation, restraint↔indulgence), religion.
   * - **Goal**
     - Goal with a name, description, and deadline — both inside and outside
       the system.
   * - **MoodStatus**
     - Longer-lived affective state: bad↔good, tired↔energetic,
       tense↔relaxed.
   * - **EmotionStatus**
     - Short-lived emotions tied to specific events: happiness, anxiety,
       fear, joy, relief, love, hate, pride, dread, excitement, shame,
       anger, disgust, worry, boredom, sadness, satisfaction, confusion, hope.
   * - **Preference**
     - Interaction modality (input/output), Design (colour, font size,
       contrast…), Item (ranked choices), PreferredLanguage.

Each category has its own sub-metamodel with class diagrams and a literature
mapping — see the directories under ``metamodel/`` in the User Modeling
Language repository for the full references.


Available formats
-----------------

The metamodel is published in several interchangeable formats, so you can pick
the one that fits the tool you are using:

- **Draw.io** (``usermetamodel.drawio.xml``) — editable source of the
  canonical metamodel diagram.
- **PNG** (``usermetamodel.png``, ``condensedusermetamodel.png``) — rendered
  overview, useful for quick reference.
- **B-UML Python** (``usermetamodel_buml.py``) — every class, enumeration,
  association, and constraint as a runnable :doc:`structural model <structural>`,
  ready to be consumed by any BESSER generator or validator.
- **JSON Schema** (``json_schema.json``) — schema for validating serialised
  user profile documents programmatically.


Relationship to the Object Diagram
----------------------------------

A UserDiagram is *instantiated* the same way an :doc:`object diagram <object>`
is instantiated against a structural model: the User Modeling Language is the
fixed reference domain model, and the diagram you draw or author is a set of
objects that conform to it.

The backend ships the reference domain model in
``besser/utilities/web_modeling_editor/backend/constants/user_buml_model.py``
so any conversion path (editor → BUML, BUML → JSON, validation) can resolve
the classes without the caller having to load the User Modeling Language
separately.


Normalised profile document
---------------------------

Raw UserDiagram payloads (the flat ``{objects, relationships}`` shape produced
by the editor) are collapsed into a hierarchical JSON document by
``generate_user_profile_document`` in
``besser/utilities/web_modeling_editor/backend/services/utils/user_profile_utils.py``.
The document is rooted on the ``User`` instance, with each related class
inlined (single-child) or expanded into an array (multi-child):

.. code-block:: json

    {
      "title": "Alice",
      "diagramType": "UserDiagram",
      "model": {
        "id": "user-1",
        "class": "User",
        "Personal_Information": {
          "firstName": "Alice",
          "lastName": "Example",
          "age": 68,
          "gender": "Female",
          "nationality_iso3166": "LU"
        },
        "Competence": {
          "Language": [
            {"iso693_3": "eng", "level": "C2"},
            {"iso693_3": "fra", "level": "B2"}
          ],
          "Skill": [{"name": "reading", "score": "high"}],
          "Education": {
            "degreeName": "B.Sc.",
            "degreeType": "Bachelor's_degree",
            "providedBy": "University of Luxembourg"
          }
        },
        "Accessibility": {
          "Disability": {
            "name": "low-vision",
            "description": "Reduced visual acuity",
            "affects": "Sight"
          }
        },
        "Culture": {"religion": "Irreligion"}
      }
    }

This normalised form is what the recommendation endpoints
(``/recommend-agent-config-llm``, ``/recommend-agent-config-mapping``) and the
agent generator's personalization pipeline consume. Consumers never see the
editor-internal object-and-relationships layout.


Semantic validation
-------------------

The condensed metamodel shipped with the Web Modeling Editor carries a small
set of OCL constraints that the editor evaluates each time a UserDiagram is
validated. They are intentionally lightweight first-step checks and can be
extended over time.

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Context
     - Constraint
     - Rationale
   * - ``Personal_Information``
     - ``self.age >= 0 and self.age <= 120``
     - Prevents unrealistic ages and catches common input mistakes.
   * - ``Skill``
     - ``self.name <> ''``
     - Prevents unnamed skills.
   * - ``Education``
     - ``self.degreeName <> '' and self.providedBy <> ''``
     - Avoids incomplete education records.
   * - ``Disability``
     - ``self.description <> ''``
     - Ensures disability entries include meaningful explanatory text.

The constraints are stored as ``ClassOCLConstraint`` elements directly inside
the reference metamodel JSON, so both the editor's validator and the standard
OCL evaluation path pick them up without additional wiring.


How it is used
--------------

- **Agent personalization**: the normalised document is passed to either
  ``/recommend-agent-config-llm`` or ``/recommend-agent-config-mapping`` to
  obtain a structured agent configuration, which is then fed into the
  :doc:`BAF generator <../../generators/baf>`.
- **Personalization mapping**: a ``personalizationMapping`` entry can pair a
  user profile with a specific agent configuration so the generator emits a
  dedicated variant for that user. See the
  :doc:`agent personalization guide <../../generators/agent_personalization>`
  for the full mapping shape.
- **Profile-aware UIs**: because the user profile document is a plain,
  hierarchical JSON structure rooted on ``User``, downstream generators
  (e.g. React) can consume it directly to render profile-aware UIs.


See Also
--------

- `BESSER User Modeling Language repository
  <https://github.com/BESSER-PEARL/User-Modeling-Language>`_ — the canonical
  source for the full metamodel, per-dimension README files and the
  literature mapping.
- :doc:`object` — the object-diagram mechanism UserDiagrams build on.
- :doc:`ocl` — how OCL constraints are written and evaluated in BESSER.
- :doc:`../../generators/agent_personalization` — the consumer of user
  profile documents.
