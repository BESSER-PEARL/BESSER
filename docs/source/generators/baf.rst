BESSER Agentic Framework (BAF) Generator
========================================

The BAF generator produces a BAF Agent based on a given B-UML agent model.
Let's generate the agent for Greetings Agent defined in :doc:`../buml_language/model_types/agent`. You should create a ``BAFGenerator`` object, provide the agent model, and use the ``generate`` method as follows:

.. code-block:: python
    
    from besser.generators.agents.baf_generator import BAFGenerator

    generator: BAFGenerator = BAFGenerator(model=agent)
    generator.generate()

Optional constructor parameters:

- ``output_dir``: Output directory (default: ``output/`` in the current directory).
- ``config_path``: Path to a **JSON** configuration file for the agent. It is read with
  ``json.load``, so a YAML file will fail to parse.
- ``config``: Configuration dictionary (alternative to ``config_path``).
- ``config_yaml``: Raw YAML text to write out as the agent's ``config.yaml``
  instead of the template-rendered default.
- ``openai_api_key``: OpenAI API key for LLM-powered agent features.
- ``generation_mode``: See `Generation Modes`_ below.
- ``test_mode``: When ``True``, the agent is generated to be driven headlessly
  in an isolated test environment such as the Agent Simulator:

  - every WebSocket/Streamlit platform is created with ``use_ui=False`` (the
    built-in Streamlit chat UI is not launched); without ``test_mode`` it is
    created with ``use_ui=True``. The only exception is
    ``config={"agentPlatform": "websocket"}``, which always uses ``use_ui=False``;
  - each workspace declared in the agent model is pre-created inside the output
    folder. Its ``path`` is made relative (backslashes become ``/``, drive
    letters and leading slashes are dropped, e.g. ``/tmp/data`` becomes
    ``tmp/data``); a blank path falls back to the workspace name. A path with
    ``..`` segments, or with no relative part at all (e.g. ``/``), raises
    ``ValueError`` because it would point outside the output folder.

The generated files land in the ``<<current_directory>>/output`` folder:

- ``<AgentName>.py``: the agent script, named after the agent model — not ``agent.py``.
- ``config.yaml``: the agent's configuration file.
- ``readme.txt``: how to run the generated agent.
- ``tools.py``: the agent's tool function definitions — only when the model declares tools.
- ``skills/``: one Markdown file per skill — only when the model declares skills.
- ``personalized_agent_model.py`` / ``personalized_agent_model.json``: the
  personalized agent model, written alongside the others only when a
  personalization config is supplied and the mode is not ``CODE_ONLY``.

Check out the BAF documentation for more details on how to use the generated agent: `BESSER Agentic Framework Documentation <https://besser-agentic-framework.readthedocs.io/latest/>`_.


Generation Modes
----------------

The BAF generator supports three generation modes via the ``generation_mode`` parameter:

.. code-block:: python

    from besser.generators.agents.baf_generator import BAFGenerator, GenerationMode

    # Default: full pipeline (personalization + templated code)
    generator = BAFGenerator(model=agent, generation_mode=GenerationMode.FULL)

    # Skip personalization, render templates immediately
    generator = BAFGenerator(model=agent, generation_mode=GenerationMode.CODE_ONLY)

    # Run personalization JSON/model export only (no code templates)
    generator = BAFGenerator(model=agent, generation_mode=GenerationMode.PERSONALIZED_ONLY)

- **FULL** (default): Runs personalization (if configured) followed by templated code generation.
- **CODE_ONLY**: Skips personalization helpers and renders templates immediately. Use this when you
  do not need personalization assets.
- **PERSONALIZED_ONLY**: Runs only the personalization JSON/model export. Use this to produce
  personalization artifacts without generating the agent code.


Personalization
---------------

The BAF generator can adapt the generated agent to an end-user's profile —
language, style, readability, modality, platform, LLM, and more. The
personalization flow is opt-in: with no ``config`` passed the generator behaves
identically to the classic pipeline.

See :doc:`agent_personalization` for the structured configuration schema, the
two recommendation backends (rule-based and LLM-based), the variant mechanisms
(languages, variations, configuration variants, personalization mapping), and
the ``OPENAI_API_KEY`` lookup order.


RAG Support
-----------

If the agent model includes RAG elements (see :doc:`../buml_language/model_types/agent`),
the generator produces the vector store setup (Chroma), text splitter configuration,
and ``session.run_rag()`` calls. A data folder is created for each RAG element
where you should place your PDF documents before running the agent. The folder
name is the lower-cased RAG element name (e.g. ``Knowledge_Base`` becomes
``knowledge_base/``).


Reasoning States and Multi-LLM
-------------------------------

The generator emits the multi-LLM and reasoning constructs described in
:doc:`../buml_language/model_types/agent`:

- Every LLM registered via ``agent.new_llm()`` is generated as an
  ``agent.new_llm(...)`` call, and ``agent.set_default_llm(...)`` is emitted
  when the chosen default differs from the first-registered (auto) default.
- ``ReasoningState`` states are generated through the ``new_reasoning_state``
  factory, carrying their ``llm`` reference, ``max_steps``,
  ``enable_task_planning``, ``stream_steps``, ``system_prompt`` and
  ``fallback_message``.
- Agent-level tools, skills and workspaces are emitted as ``agent.new_tool()``,
  ``agent.new_skill()`` and ``agent.new_workspace()`` calls.

Consumers (``LLMReply``, ``DBReply``, RAG, reasoning states) reference their
LLM by ``llm_name``; when omitted, the agent's default LLM is used.


Session Variables and Reply Options
-----------------------------------

Several actions can read from and write to the user session:

- **Session-variable interpolation.** When the matching flag is ``True``, every
  ``{key}`` placeholder in the text is replaced at runtime with
  ``session.get('key')`` (an empty string when unset); ``{user_message}`` is
  replaced with the last user message. The flags are ``use_session_vars`` on
  ``AgentReply``, ``WebSocketReplyMarkdown``, ``WebSocketReplyHTML`` and
  ``WebSocketReplySpeech``; ``system_prompt_use_session_vars`` on ``LLMReply``
  and ``LLMChatReply``; ``prompt_use_session_vars`` on ``RAGReply``;
  ``system_message_prefix_use_session_vars`` on ``WebCrawlLLMReply``; and
  ``custom_input_prompt_use_session_vars`` for custom input prompts.
  Without the flag, the text is emitted as-is, braces included.
- **Custom input prompt.** ``LLMReply``, ``RAGReply`` and ``DBReply`` (LLM query
  mode) use the last user message as input by default. With
  ``input_prompt_mode='custom'`` they use ``custom_input_prompt`` instead.
- **store_in_session.** ``LLMReply``, ``LLMChatReply``, ``RAGReply``,
  ``DBReply`` and ``WebCrawlLLMReply`` store their result with
  ``session.set(<key>, result)`` when ``store_in_session`` names a key, so later
  actions and states can use it (for example through ``{key}`` interpolation).
- **send_reply.** When ``False``, the result is computed (and stored if
  ``store_in_session`` is set) but not sent to the user. Defaults to ``True``.

.. code-block:: python

    from besser.BUML.metamodel.state_machine.agent import AgentReply, LLMReply
    from besser.BUML.metamodel.state_machine.state_machine import Body

    summarize_state = agent.new_state("summarize")
    summarize_state.set_body(Body("summarize_body", actions=[
        # Silently summarize the question and keep the result in the session
        LLMReply(prompt="Summarize the question in one sentence.",
                 input_prompt_mode="custom",
                 custom_input_prompt="Question about {topic}: {user_message}",
                 custom_input_prompt_use_session_vars=True,
                 store_in_session="summary", send_reply=False),
        # ...then reuse it in a templated reply
        AgentReply("You asked: {summary}", use_session_vars=True),
    ]))


GUI Generation
--------------

A ``GUIReplyAction`` sends a GUI as a chat message. The GUI is a
:class:`~besser.BUML.metamodel.gui.GUIModel` registered on the agent with
``agent.add_gui_model(gui_id, gui_model)`` (stored in ``agent.gui_models``);
the action references it through its ``gui_id`` (see
:doc:`../buml_language/model_types/agent`).

For each distinct ``gui_id`` referenced by a ``GUIReplyAction``, the generator
writes a module ``guis/<gui_id as identifier>.py`` in the output folder. The
module holds the BUML code of the GUI model (``gui_model``) and wraps it in a
BAF ``AgentGUI`` object named ``gui``. The agent imports it and the state body
calls ``platform.reply_gui(session, <module>)``. Generation raises
``ValueError`` when a ``GUIReplyAction`` references a ``gui_id`` that is not
registered in ``agent.gui_models``, or when two ids map to the same module name
(for example ``order-form`` and ``order form``).

Transitions can react to the GUI: ``state.when_form_submitted(form_id=...)``
fires when the user submits the form with that id (any form when ``form_id`` is
omitted), and ``state.when_event(GUIEvent(message_id=...))`` fires on events
sent by that GUI.

.. code-block:: python

    from besser.BUML.metamodel.gui import GUIModel, Module, Screen, Text
    from besser.BUML.metamodel.state_machine.agent import (
        Agent, AgentReply, GUIReplyAction, WebSocketPlatform
    )
    from besser.BUML.metamodel.state_machine.state_machine import Body
    from besser.generators.agents.baf_generator import BAFGenerator

    agent = Agent("order_agent")
    agent.platforms.append(WebSocketPlatform())

    # The GUI shown in the chat, registered under the id the GUIReplyAction references
    title = Text(name="title", content="Place your order")
    screen = Screen(name="order_screen", description="", view_elements={title}, is_main_page=True)
    order_gui = GUIModel(name="order_gui", package="", versionCode="1", versionName="1.0",
                         description="", modules={Module(name="order_module", screens={screen})})
    agent.add_gui_model("order_form", order_gui)

    ask = agent.new_state("ask", initial=True)
    thanks = agent.new_state("thanks")
    ask.set_body(Body("ask_body", actions=[GUIReplyAction(gui_id="order_form", is_form=True)]))
    thanks.set_body(Body("thanks_body", actions=[AgentReply("Thanks, your order is in!")]))
    ask.when_form_submitted(form_id="order_form").go_to(thanks)

    generator: BAFGenerator = BAFGenerator(model=agent)
    generator.generate()
    # output/
    #   order_agent.py    <- imports guis.order_form, calls platform.reply_gui(session, order_form)
    #   guis/
    #     __init__.py
    #     order_form.py   <- gui_model (BUML code) + gui = AgentGUI(...)


Missing BAF Features
--------------------

Currently, some features available in BAF are stil missing in the B-UML agent model and the BAF Generator. Most notably:

- **Platform Configuration**
- **Entities**
- **Processors**