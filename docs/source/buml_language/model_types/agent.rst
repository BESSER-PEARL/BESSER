Agent model
===========

.. warning::

   While you can define agents using the B-UML agent metamodel, the BAF generator still has limitations in terms of supported features. If your goal is to develop an agent in a textual notation, we'd recommend to get accustomed to the BAF.
   The B-UML agent notation is fitting if you plan to re-use agent components in different agents or if you plan to develop/use model-based techniques on the agents. 
   If you aim to develop agents graphically, then the BESSER framework provides a graphical editor for that purpose.

Agent metamodel
-----------------------

This metamodel allows the definition of agents. 
The agents follow a state machine-like behavior, where they can be in different states and transition between them based on events and conditions.
Thus, similarly to the state machine metamodel, the agent metamodel contains the main elements of a state machine:

- AgentState: Represent the different conditions or statuses that an agent can be in at any given time.
- Transitions: Define the rules for how an agent moves from one state to another, triggered by events or conditions.
- Events: External or internal stimuli (inputs) that cause a check of conditions and potentially trigger transitions between states.
- Conditions: Conditions that must be met for a transition to occur, allowing for more complex decision-making.
- Actions: Activities or responses (outputs) that occur due to transitions or when the agent is in a specific state. Each state has a **Body**, which defines the sequence of actions to be executed when an event causes the transition to a state (and a **fallback body** that defines the actions to be executed in case of error in the machine).
- AgentSession: An agent can have multiple **sessions** running simultaneously (e.g., one for each user interacting with the agent). A Session is always located in one of the states. If there are multiple sessions, each can store data privately (with respect to the other sessions). When modelling an agent, a session is only used as an argument for the events and bodies.

Beyond the state machine-like elements, the agent metamodel also includes agent specific elements. These are closely related to the agent concepts contained in the `BESSER Agentic Framework <https://github.com/BESSER-PEARL/BESSER-Agentic-Framework>`_:

- Agent
- Intent
- IntentParameter
- Entity
- IntentClassifierConfiguration
- Platform
- LLMWrapper


To read about their meaning and usage, please refer to the `documentation <https://besser-agentic-framework.readthedocs.io/latest/>`_ of the BESSER Agentic Framework.

Actions
~~~~~~~

Each state body is a sequence of actions.  The following action classes are
available in ``besser.BUML.metamodel.state_machine.agent``:

**Text and LLM replies**

- ``AgentReply(message, use_session_vars=False)`` — send a plain-text reply.
  When ``use_session_vars=True``, ``{key}`` placeholders in *message* are
  replaced at runtime with ``session.get("key")``. The special placeholder
  ``{user_message}`` resolves to the current user input.
- ``LLMReply(prompt, llm_name, input_prompt_mode, custom_input_prompt,
  custom_input_prompt_use_session_vars, system_prompt_use_session_vars,
  store_in_session, send_reply)`` — generate a reply using an LLM.

  - ``prompt``: optional system prompt.
  - ``llm_name``: selects a registered LLM (defaults to the agent default).
  - ``input_prompt_mode``: ``'last_user_message'`` (default) passes the user's
    message directly; ``'custom'`` uses ``custom_input_prompt`` instead. Any
    other value raises ``ValueError`` (the accepted values are
    ``VALID_INPUT_PROMPT_MODES``).
  - ``custom_input_prompt``: template string for the LLM input when
    ``input_prompt_mode='custom'``; required (non-empty) in that mode.
  - ``custom_input_prompt_use_session_vars`` / ``system_prompt_use_session_vars``:
    enable ``{key}`` interpolation in the respective strings.
  - ``store_in_session``: when set, the LLM reply is stored in the session under
    this key before being sent.
  - ``send_reply`` (default ``True``): set to ``False`` to suppress sending the
    reply to the user (useful when only storing the result in the session).

- ``LLMChatReply(prompt, llm_name, system_prompt_use_session_vars,
  store_in_session, send_reply)`` — like ``LLMReply`` but calls
  ``llm.chat(...)`` with the conversation history, making it suitable for
  multi-turn dialogue states. Supports the same ``store_in_session`` and
  ``send_reply`` controls.
- ``RAGReply(rag_db_name, prompt, input_prompt_mode, custom_input_prompt,
  custom_input_prompt_use_session_vars, prompt_use_session_vars,
  store_in_session, send_reply)`` — answer using a configured RAG database.
  Supports the same ``input_prompt_mode`` / ``custom_input_prompt``,
  session-var interpolation, ``store_in_session``, and ``send_reply`` controls
  as ``LLMReply`` (``prompt_use_session_vars`` applies to the RAG hint prompt).
- ``DBReply(db_selection_type, db_custom_name, db_query_mode, db_operation,
  db_sql_query, llm_name, input_prompt_mode, custom_input_prompt,
  custom_input_prompt_use_session_vars, store_in_session, send_reply)`` —
  answer from a database, either with a fixed SQL query (``db_query_mode='sql'``)
  or with a query written by an LLM (``'llm_query'``). The
  ``input_prompt_mode`` / ``custom_input_prompt`` pair, ``store_in_session`` and
  ``send_reply`` behave as in ``LLMReply``.

**GUI replies**

- ``GUIReplyAction(gui_id, persist=True, width=None, is_form=False)`` — send a
  BESSER GUI model as an interactive chat message. ``gui_id`` must be a key of
  ``agent.gui_models`` (see `GUI integration`_ below); ``agent.validate()``
  reports a reply whose GUI is not registered. When ``persist=True`` the
  submitted form field values are stored in the session. ``width`` is an
  optional CSS width for the rendered bubble. ``is_form=True`` marks the GUI as
  a form: submitting it emits the event ``when_form_submitted`` transitions
  react to.

**Web crawling**

- ``WebCrawlLLMReply(initial_url, max_depth, max_pages, crawl_format,
  base_url_prefix, run_crawl, no_crawl_error_message, system_message_prefix,
  llm_name, system_message_prefix_use_session_vars, store_in_session,
  send_reply)`` — performs a BFS web crawl starting at ``initial_url`` and
  queries an LLM with the retrieved content.  The crawl result is cached in the
  session; set ``run_crawl=False`` in subsequent states to reuse the cache
  without re-fetching. ``system_message_prefix_use_session_vars`` enables
  ``{key}`` interpolation in the system message prefix. The same
  ``store_in_session`` and ``send_reply`` controls as ``LLMReply`` are
  available.

**WebSocket rich-media replies**

The following actions map to the corresponding ``WebSocketPlatform`` methods and
require the agent to use a ``WebSocketPlatform``:

- ``WebSocketReplyMarkdown(message, use_session_vars=False)`` — send
  Markdown-formatted text. ``use_session_vars`` enables ``{key}``
  interpolation.
- ``WebSocketReplyHTML(message, use_session_vars=False)`` — send an
  HTML-formatted message.
- ``WebSocketReplySpeech(message, audio_speed, use_session_vars=False)`` —
  convert text to speech and send the audio.
- ``WebSocketReplyOptions(options)`` — present a list of selectable options.
- ``WebSocketReplyLocation(latitude, longitude)`` — send a geographic
  coordinate.
- ``WebSocketReplyFile()`` — send a file; the body must supply a ``File``
  object at runtime.
- ``WebSocketReplyImage()`` — send an image (NumPy ``ndarray``); body must
  supply the array at runtime.
- ``WebSocketReplyDataframe()`` — send a pandas ``DataFrame``; body must supply
  it at runtime.
- ``WebSocketReplyPlotly()`` — send a Plotly figure; body must supply a
  ``plotly.graph_objects.Figure`` at runtime.

RAG (Retrieval-Augmented Generation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Agents can use RAG to answer questions from a set of documents (e.g. PDFs).
A RAG element is added to an agent via ``agent.new_rag()`` and combines a
``RAGVectorStore`` (embedding config), a ``RAGTextSplitter`` (chunking strategy),
and an LLM name. Use ``RAGReply`` in a state body to trigger a RAG query.

When generated, a data folder named after the RAG element is created
(e.g. ``"Knowledge Base"`` produces ``knowledge_base/``). Place your PDF
documents in this folder before running the agent.

The optional ``llm_prompt`` parameter injects a fixed prefix instruction before
every RAG query, useful for enforcing domain-specific constraints or tone:

.. code-block:: python

    kb = agent.new_rag(
        name='Knowledge Base',
        vector_store=vector_store,
        splitter=splitter,
        llm_name='gpt-4o-mini',
        llm_prompt='Answer only from the provided documents.',
    )

Retrieval can combine the vector store with a BM25 keyword index (hybrid
retrieval). Set ``use_hybrid_rag=True`` and, optionally, ``bm25_weight`` — the
weight of the BM25 results between 0 and 1, the vector results getting the
remainder (default ``0.6``). The generated agent then uses BAF's ``HybridRAG``
instead of ``RAG``; in the web editor these are the *Hybrid RAG (BM25)* and
*BM25 weight* fields of the RAG element:

.. code-block:: python

    kb = agent.new_rag(
        name='Knowledge Base',
        vector_store=vector_store,
        splitter=splitter,
        llm_name='gpt-4o-mini',
        use_hybrid_rag=True,
        bm25_weight=0.6,
    )

Multiple LLMs
~~~~~~~~~~~~~

An agent can register more than one LLM and reference each by name. Add an
LLM with ``agent.new_llm()``:

.. code-block:: python

    fast = agent.new_llm(name='fast', provider='openai', parameters={'model': 'gpt-4o-mini'})
    big = agent.new_llm(name='big', provider='openai', parameters={'model': 'gpt-4o'})

``provider`` selects the concrete wrapper: ``openai`` → ``LLMOpenAI``,
``huggingface`` → ``LLMHuggingFace``, ``huggingface_api`` →
``LLMHuggingFaceAPI``, ``replicate`` → ``LLMReplicate``. ``parameters`` is a
free-form dict passed to the wrapper (e.g. the model id). Optional
``num_previous_messages`` (default 1) and ``global_context`` are also supported.

The first LLM registered becomes the default. Change the default with
``agent.set_default_llm('big')``. Any consumer — ``LLMReply``, ``DBReply``,
``RAGReply`` and reasoning states — uses the default unless it specifies its
own ``llm_name``. Every ``llm_name`` reference must resolve to a registered
LLM; this is checked by ``agent.validate()``.

Reasoning states
~~~~~~~~~~~~~~~~

A ``ReasoningState`` is a state whose body is an autonomous reasoning loop
driven by an LLM (using the agent's tools, skills and workspaces). Create one
with ``agent.new_reasoning_state()``:

.. code-block:: python

    assistant = agent.new_reasoning_state(
        name='assistant',
        llm='big',                  # registered LLM name; omit to use the default
        max_steps=8,                # max reasoning iterations
        enable_task_planning=True,
        stream_steps=True,
        system_prompt='You are a helpful assistant.',
        fallback_message='Sorry, I could not complete that.',
    )

The body of a reasoning state is supplied automatically by the factory; the
metamodel rejects manual ``set_body`` / ``set_fallback_body`` calls on it.

Tools, skills and workspaces
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reasoning states draw on three agent-level primitives, shared by every
reasoning state:

- **Tools** (``agent.new_tool(name, description, code)``) — callable functions the agent can invoke. ``code`` holds the Python implementation.
- **Skills** (``agent.new_skill(name, content, description)``) — reusable instruction snippets injected into the reasoning context.
- **Workspaces** (``agent.new_workspace(name, path, description, writable, max_read_bytes)``) — file-system locations the agent may read from (and write to when ``writable``).

GUI integration
~~~~~~~~~~~~~~~

An agent can send interactive GUI panels — forms, dashboards, or any BESSER
:doc:`GUI model <gui>` — directly in the chat conversation. The workflow is:

1. Register each GUI on the agent with ``agent.add_gui_model(gui_id, gui_model)``,
   where ``gui_model`` is a :class:`~besser.BUML.metamodel.gui.GUIModel`.
   ``agent.gui_models`` is the resulting ``dict[str, GUIModel]``; assigning it
   directly is validated too (keys must be non-empty strings, values
   ``GUIModel`` instances).
2. In a state body, add a ``GUIReplyAction`` referencing the ``gui_id``.
3. In the next state, add a ``when_form_submitted(form_id)`` transition so the
   agent reacts when the user submits the form.

.. code-block:: python

    agent.add_gui_model('signup', signup_gui)          # signup_gui: GUIModel
    ask.set_body(Body('ask_body', actions=[GUIReplyAction('signup', is_form=True)]))
    ask.when_form_submitted(form_id='signup').go_to(thanks)

In the web editor each GUI is an ``AgentGUI`` component of the agent diagram.
On import its GrapesJS design is converted into a ``GUIModel`` by the GUI
diagram processor (agent GUIs are not bound to a class diagram), and a GUI
that has not been designed yet becomes an empty ``GUIModel``. The B-UML code
export emits every GUI as a builder function followed by
``agent.add_gui_model(...)``, so the exported module rebuilds the same models.

The BAF generator collects all ``GUIReplyAction`` instances, creates a
``guis/`` package, and writes the code of each referenced ``GUIModel`` there.

Transitions triggered by GUI events
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Beyond the standard ``when_intent_matched`` / ``when_file_received`` builders,
``AgentState`` provides GUI-specific transition helpers:

- ``state.when_form_submitted(form_id=None)`` — triggered when the user submits
  a GUI form. If ``form_id`` is provided (must match the ``gui_id`` of a
  ``GUIReplyAction``), only submissions from that specific form trigger the
  transition; otherwise any form submission matches.

Under the hood this uses the ``GUIEvent`` event class and the
``FormSubmitMatcher`` condition, which you can also instantiate directly if you
need finer control:

.. code-block:: python

    from besser.BUML.metamodel.state_machine.agent import GUIEvent, FormSubmitMatcher

    # equivalent to state.when_form_submitted(form_id='my_form')
    state.when_event(GUIEvent(message_id='my_form')) \
        .with_condition(FormSubmitMatcher(form_id='my_form')) \
        .go_to(next_state)

``GUIEvent(message_id=None)`` on its own (``state.when_event(GUIEvent('my_form'))``)
fires on any interaction with the GUI message ``my_form``; without a
``message_id`` it fires on interactions with any GUI.

.. image:: ../../img/agent_mm.png
  :width: 1600
  :alt: Agent metamodel
  :align: center

.. note::

    The classes highlighted in green originate from the :doc:`structural metamodel <structural>` and :doc:`state machine <state_machine>` .


Example agent model
-------------------

As a simple example, we modeled the `Greetings Agent <https://besser-agentic-framework.readthedocs.io/latest/your_first_agent.html#the-greetings-agent>`_ from the BAF documentation.

.. code-block:: python

    import datetime
    from besser.BUML.metamodel.state_machine.state_machine import Body, ConfigProperty
    from besser.BUML.metamodel.state_machine.agent import Agent, AgentSession
    import operator

    agent = Agent('Generated_Agent')

    agent.add_property(ConfigProperty('websocket_platform', 'websocket.host', 'localhost'))
    agent.add_property(ConfigProperty('websocket_platform', 'websocket.port', 8765))
    agent.add_property(ConfigProperty('websocket_platform', 'streamlit.host', 'localhost'))
    agent.add_property(ConfigProperty('websocket_platform', 'streamlit.port', 5000))
    agent.add_property(ConfigProperty('nlp', 'nlp.language', 'en'))
    agent.add_property(ConfigProperty('nlp', 'nlp.region', 'US'))
    agent.add_property(ConfigProperty('nlp', 'nlp.timezone', 'Europe/Madrid'))
    agent.add_property(ConfigProperty('nlp', 'nlp.pre_processing', True))
    agent.add_property(ConfigProperty('nlp', 'nlp.intent_threshold', 0.4))

    # INTENTS
    Greeting = agent.new_intent('Greeting', [
        'Hi',
        'Hello',
        'Howdy',
    ])
    Good = agent.new_intent('Good', [
        'Good',
        'Fine',
        'I m alright',
    ])
    Bad = agent.new_intent('Bad', [
        'Bad',
        'Not so good',
        'Could be better',
    ])


    # STATES
    initial = agent.new_state('initial', initial=True)
    greeting = agent.new_state('greeting')
    bad = agent.new_state('bad')
    good = agent.new_state('good')

    # initial state
    # greeting state
    def greeting_body(session: AgentSession):
        session.reply('Hi!')
        session.reply('How are you?')

    greeting.set_body(Body('greeting_body', greeting_body))
    greeting.when_intent_matched(Good).go_to(good)
    greeting.when_intent_matched(Bad).go_to(bad)

    # bad state
    def bad_body(session: AgentSession):
        session.reply('I m sorry to hear that...')

    bad.set_body(Body('bad_body', bad_body))
    bad.go_to(initial)

    # good state
    def good_body(session: AgentSession):
        session.reply('I am glad to hear that!')

    good.set_body(Body('good_body', good_body))
    good.go_to(initial)
