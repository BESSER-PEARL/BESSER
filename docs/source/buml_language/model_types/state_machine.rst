State Machine model
====================

.. _state-machine-metamodel:

State Machine metamodel
-----------------------

This metamodel allows the definition of state machines (also known as finite state machines or FSM), which are computational
models used to design and analyze the behaviour of systems. A state machine is characterized by:

- States: A finite set of conditions or statuses that the system can be in at any given time.
- Transitions: Rules that describe how the system moves from one state to another, often triggered by events or conditions.
- Events: External or internal stimuli (inputs) that cause transitions between states.
- Actions: Activities or responses (outputs) that occur due to transitions or when the system is in a specific state. In our
  state machines vision, each state has a **Body**, which defines the sequence of actions to be executed when an event causes the transition to a state
  (and a **fallback body** that defines the actions to be executed in case of error in the machine).

A state machine can have multiple **sessions** running simultaneously (e.g., one for each user interacting with the system).
A Session is always located in one of the states. If there are multiple sessions, each can store data privately (with respect to the other sessions).
When modelling a state machine, a session is only used as an argument for the events and bodies.

.. image:: ../../img/state_machine_mm.png
  :width: 800
  :alt: State machine metamodel
  :align: center

.. note::

  The classes highlighted in green originate from the :doc:`structural metamodel <structural>`.


Conditions
^^^^^^^^^^

A ``Condition`` guards a transition — the transition fires only when the condition evaluates
to ``True``. Conditions can be created from a Python callable or from a raw source string
(useful for JSON round-trip serialization with the web editor):

.. code-block:: python

    from besser.BUML.metamodel.state_machine import Condition

    # From a callable
    cond = Condition(name="is_adult", callable=lambda session: session.get("age") >= 18)

    # From a source string (deserialized from JSON)
    cond = Condition(name="is_adult", source='lambda session: session.get("age") >= 18')


Validation
^^^^^^^^^^

Call ``StateMachine.validate()`` to check structural correctness before generation:

.. code-block:: python

    result = my_state_machine.validate()
    # result = {"success": True/False, "errors": [...], "warnings": [...]}
