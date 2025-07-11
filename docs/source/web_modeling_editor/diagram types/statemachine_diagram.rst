State Machine Diagrams
======================

State machine diagrams model the dynamic behavior of a system by showing how objects change state in response to events. They are particularly useful for modeling reactive systems, user interfaces, and protocol specifications.

Overview
--------

State machine diagrams are essential for modeling the dynamic behavior of systems. They show how objects respond to events and change their state over time, making them invaluable for understanding complex system behaviors and designing robust applications.

Key Features
------------

* **Dynamic Behavior**: Model how objects change over time
* **Event-Driven Logic**: Show responses to external and internal events
* **State Management**: Define clear states and transitions
* **Action Specification**: Include entry, exit, and do activities

Core Concepts
-------------

States
~~~~~~

States represent the different conditions or situations an object can be in:

* **Simple States**: Basic states with a name and optional activities
* **Initial State**: Starting point of the state machine (black circle)

Transitions
~~~~~~~~~~~

Transitions define how the system moves from one state to another:

* **Trigger Events**: Events that cause the transition
* **Guard Conditions**: Boolean expressions that must be true
* **Actions**: Activities performed during the transition

Activities
~~~~~~~~~~

States can contain various types of activities:

* **Entry Actions**: Executed when entering the state
* **Exit Actions**: Executed when leaving the state  
* **Do Activities**: Executed while in the state

Getting Started
---------------

Adding Initial States
~~~~~~~~~~~~~~~~~~~~~

To add an initial state:

1. Drag and drop the initial state element (black circle) from the left panel
2. This marks where the state machine begins
3. Every state machine should have exactly one initial state

Adding States
~~~~~~~~~~~~~

To add regular states:

1. Drag and drop a state element from the left panel onto the canvas
2. States represent the different conditions your object can be in
3. Give each state a meaningful, descriptive name

Editing States
~~~~~~~~~~~~~~

To edit a state:

1. Double-click on the state to open the editing popup
2. Modify the state properties:

   * **Name**: Provide a clear, descriptive state name
   * **Entry Actions**: Actions executed when entering the state
   * **Exit Actions**: Actions executed when leaving the state
   * **Do Activities**: Ongoing activities while in the state

**Activity Format Examples:**

* ``entry / startTimer()`` - Start a timer when entering
* ``exit / cleanup()`` - Clean up when leaving  
* ``do / processData()`` - Continuously process data

Creating Transitions
~~~~~~~~~~~~~~~~~~~~

To create transitions between states:

1. Select the source state with a single click
2. You'll see blue circles indicating connection points
3. Click and hold on one of these points
4. Drag to the target state to create the transition

Editing Transitions
~~~~~~~~~~~~~~~~~~~

To edit a transition:

1. Double-click on the transition arrow to open the editing popup
2. Define transition properties:

   * **Trigger Events**: Events that cause the transition
   * **Guard Conditions**: Boolean conditions that must be met
   * **Actions**: Activities performed during the transition

**Transition Format Examples:**

* ``buttonPressed`` - Simple event trigger
* ``timeout [count > 5]`` - Event with guard condition
* ``start / initialize()`` - Event with action
* ``error [severity > 3] / logError()`` - Complete transition specification

Adding Final States
~~~~~~~~~~~~~~~~~~~

To add final states:

1. Drag and drop the final state element (double circle) from the left panel
2. These mark where the state machine ends
3. A state machine can have multiple final states


Additional Resources
--------------------

For more information about state machine diagrams and the BESSER Web Modeling Editor:

* `BESSER Documentation <https://besser.readthedocs.io/en/latest/>`_
* `WME GitHub Repository <https://github.com/BESSER-PEARL/BESSER_WME_standalone>`_
* :doc:`../use_the_wme` - General editor usage guide
* :doc:`class_diagram` - Related class diagram documentation
