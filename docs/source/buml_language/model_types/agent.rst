Agent model
===========


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

.. image:: ../../img/agent_mm.png
  :width: 1600
  :alt: Agent metamodel
  :align: center

.. note::

  The classes highlighted in green originate from the :doc:`structural metamodel <structural>` and :doc:`state machine <state_machine>` .
