Agent Personalization
=====================

The BAF generator can produce a **personalized** agent whose behaviour, presentation
and modality are adapted to an end-user's profile. Personalization is opt-in — a
plain ``BAFGenerator`` with no config runs the normal pipeline and this page does
not apply.

Personalization has three distinct pieces, which can be combined:

- **Structured configuration** — a JSON document organised into
  ``presentation`` / ``modality`` / ``behavior`` / ``content`` / ``system``
  sections that controls agent language, style, readability, voice, platform, etc.
- **Recommendation** — given a user profile described in the
  :doc:`BESSER User Modeling Language <../buml_language/model_types/user_diagram>`
  (personal information, accessibility, competences, preferences, culture…),
  produce a structured configuration automatically. Two backends are available:
  a deterministic **rule-based** mapping, and an **LLM-based** recommender.
- **Variants** — a single agent model can be generated as several parallel
  variants (e.g. one per language, one per configuration, one per mapped user
  profile) which are bundled into the same ZIP output.

.. tip::
   If you are using the :doc:`Web Modeling Editor <../web_editor>`, the three
   pieces above are exposed through the *Agent Configuration* panel and the
   *Generate / Deploy* dialogs. The rest of this page describes the underlying
   API so you can also use the pieces programmatically.


Structured Configuration
------------------------

The configuration is a dictionary with up to five sections. Every field is
optional — unspecified fields fall back to the defaults defined in
``agent_config_recommendation_utils.load_default_agent_recommendation_config()``.

.. code-block:: python

    agent_config = {
        "presentation": {
            "agentLanguage": "english",       # see allowed values below
            "agentStyle": "formal",
            "languageComplexity": "simple",
            "sentenceLength": "concise",
            "interfaceStyle": {
                "size": 20,
                "font": "sans",
                "lineSpacing": 1.8,
                "alignment": "left",
                "color": "var(--apollon-primary-contrast)",
                "contrast": "high",
            },
            "voiceStyle": {"gender": "female", "speed": 1.0},
            "avatar": None,
            "useAbbreviations": False,
        },
        "modality": {
            "inputModalities": ["text", "speech"],
            "outputModalities": ["text"],
        },
        "behavior": {
            "responseTiming": "instant",      # or "delayed"
        },
        "content": {
            "adaptContentToUserProfile": True,
        },
        "system": {
            "agentPlatform": "streamlit",
            "intentRecognitionTechnology": "llm-based",
            "llm": {"provider": "openai", "model": "gpt-5-mini"},
        },
    }

The full list of allowed values is defined in
``RECOMMENDATION_ALLOWED_VALUES`` and includes:

- ``agentLanguage``: ``original``, ``english``, ``french``, ``german``,
  ``spanish``, ``luxembourgish``, ``portuguese``
- ``agentStyle``: ``original``, ``formal``, ``informal``
- ``languageComplexity``: ``original``, ``simple``, ``medium``, ``complex``
- ``sentenceLength``: ``original``, ``concise``, ``verbose``
- ``font``: ``sans``, ``serif``, ``monospace``, ``neutral``, ``grotesque``, ``condensed``
- ``alignment``: ``left``, ``center``, ``justify``
- ``contrast``: ``low``, ``medium``, ``high``
- ``voiceGender``: ``male``, ``female``, ``ambiguous``
- ``responseTiming``: ``instant``, ``delayed``
- ``agentPlatform``: ``websocket``, ``streamlit``, ``telegram``
- ``intentRecognitionTechnology``: ``classical``, ``llm-based``
- ``llmProvider``: ``openai``, ``huggingface``, ``huggingfaceapi``, ``replicate``
- ``openaiModels``: ``gpt-5``, ``gpt-5-mini``, ``gpt-5-nano``

.. note::

   The generator accepts both the **sectioned** shape above and the legacy
   **flat** shape (all fields at the top level). A call to
   ``flatten_agent_config_structure`` normalises any sectioned input into the
   flat shape before templates are rendered, so both are equivalent.


Personalized Code Generation
----------------------------

Pass the configuration to the :doc:`BAF Generator <baf>` via the ``config``
parameter. Provide an OpenAI key either via ``openai_api_key`` or the
``OPENAI_API_KEY`` environment variable when ``agentLanguage``,
``agentStyle``, ``languageComplexity``, ``sentenceLength`` or
``useAbbreviations`` differ from ``original`` — those fields are applied by
re-writing message content through an LLM call.

.. code-block:: python

    from besser.generators.agents.baf_generator import BAFGenerator, GenerationMode

    generator = BAFGenerator(
        model=agent,
        config=agent_config,
        openai_api_key="sk-...",           # or set OPENAI_API_KEY env var
        generation_mode=GenerationMode.FULL,
    )
    generator.generate()

In ``FULL`` mode the generator produces, in addition to the usual
``<agent>.py`` / ``config.yaml`` / ``readme.txt``:

- ``personalized_agent_model.py`` — the B-UML agent model after personalization
  has been applied, suitable for re-importing or further transformation.
- ``personalized_agent_model.json`` — the same model serialised as JSON for
  frontend preview / round-tripping.

``PERSONALIZED_ONLY`` skips the templated code and only emits the two
personalization artefacts. ``CODE_ONLY`` skips personalization entirely. See
:doc:`baf` for the full mode reference.


Recommendation Backends
-----------------------

The personalization pipeline can be fed by a manual recommendation — but the
backend also exposes two endpoints that **recommend** a configuration from a
:doc:`user profile <../buml_language/model_types/user_diagram>`.

Rule-based manual mapping
^^^^^^^^^^^^^^^^^^^^^^^^^

The manual mapping lives in
``besser/utilities/web_modeling_editor/backend/services/utils/agent_config_manual_mapping_utils.py``.
It is a literature-synthesised rule table with:

- **Conditions** that match characteristics from the user profile document
  (``age_gte``, ``profile_text_contains_any``, language codes, disabilities,
  education level…).
- **Priority** — when multiple rules match, lower numbers take precedence.
  The output configuration is the merged result of every matched rule.
- **Evidence** — each rule cites the papers the recommendation is drawn from.
- **Match mode** — ``any`` (any condition matches) or ``all``.

Examples of shipped rules: ``older_adults_readability``,
``adolescents_relatable_style``, ``low_vision_contrast_boost``,
``hearing_impairment_speech_support``.

Rules produce partial configurations (only the fields they affect), which are
merged onto the defaults via ``merge_dicts``. This backend is deterministic
and does not require an OpenAI key.

LLM-based recommendation
^^^^^^^^^^^^^^^^^^^^^^^^

The LLM backend sends the normalised user-profile document, the allowed values,
the default configuration, and an optional ``currentConfig`` to an OpenAI model
and asks it to return a valid configuration JSON. The output is parsed by
``extract_json_object`` (robust to light markdown wrapping) and validated by
``normalize_recommended_agent_config`` before being returned to the caller.

The LLM backend requires an OpenAI API key. The model name is caller-selectable
(``gpt-5``, ``gpt-5-mini``, ``gpt-5-nano``).


Variants
--------

A single generator invocation can produce **multiple variants** bundled into the
same ZIP. Three orthogonal mechanisms are supported; they are looked up in this
order and the first one that matches wins:

1. **Multi-language** — ``config["languages"]`` is a dict mapping language name
   to a per-language override. Each language becomes a sub-directory and every
   message string is translated.
2. **Variations** — ``config["baseModel"]`` + ``config["variations"]`` — each
   variation produces a separate variant starting from the base agent model.
3. **Configuration variants** — ``config["configurations"]`` is a list of
   ``{name, config}`` objects. Each entry generates an agent with that specific
   configuration.

If none of the three match but the config contains a ``personalizationMapping``
list, a **per-user-profile** variant is emitted for each mapping entry. See
the section below.

The bundling and helper dispatch is handled by
``besser/utilities/web_modeling_editor/backend/services/utils/agent_generation_utils.py``
(``handle_multi_language_generation``, ``handle_variation_generation``,
``handle_configuration_variants``, ``handle_personalized_agent``).


Personalization Mapping
-----------------------

A ``personalizationMapping`` is a list linking **user profiles** to
**configurations**:

.. code-block:: python

    config["personalizationMapping"] = [
        {
            "name": "senior-user-1",
            "user_profile": {...},     # serialised UserDiagram (see user_diagram.rst)
            "agent_config": {...},     # structured agent configuration (sectioned or flat)
        },
        ...
    ]

Each entry produces its own agent variant inside the output ZIP plus a
``user_profiles.json`` file at the root bundling every profile. At runtime the
generated agent can select the right variant for an incoming user based on
their mapped profile.

The backend normalises the mapping in-place before invoking the generator:
the raw UML JSON present in each ``user_profile`` is converted to the
normalised user-profile document (see
:doc:`../buml_language/model_types/user_diagram`) so the generator never sees
editor-internal shapes.


OpenAI API Key
--------------

Personalization features that call an LLM (language/style rewriting,
``agentLanguage`` translation, LLM-based recommendation) need an OpenAI API
key. The key is looked up in this order:

1. ``openai_api_key`` constructor argument on ``BAFGenerator``.
2. ``openai_api_key`` / ``openaiApiKey`` / ``OPENAI_API_KEY`` / ``apiKey``
   inside the ``config`` dict (top-level or under ``system``).
3. ``OPENAI_API_KEY`` environment variable.

If no key is found and a pipeline stage needs one, ``configure_agent`` raises
``RuntimeError`` with the failing stage. The BAF generator then continues with
any non-LLM fields already applied. For deployments to external hosting
(GitHub + Render), the generated ``render.yaml`` declares ``OPENAI_API_KEY``
as a secret env var the user is expected to set on Render's side.


See Also
--------

- :doc:`baf` — the BAF generator and its ``GenerationMode`` parameter.
- :doc:`../buml_language/model_types/user_diagram` — the ``UserDiagram`` model
  type that feeds the recommendation backends.
- :doc:`../web_editor_backend` — HTTP endpoints for recommendation
  (``/recommend-agent-config-llm``, ``/recommend-agent-config-mapping``,
  ``/agent-config-manual-mapping``).
