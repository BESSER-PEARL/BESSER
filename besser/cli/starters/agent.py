"""A starter B-UML agent: a library assistant that greets the user and answers two kinds of question.

    besser validate MODEL_FILE                    check the model
    besser info MODEL_FILE                        what it holds, and the generators that read it
    besser generate agent MODEL_FILE -o out       generate a BESSER Agentic Framework agent
    besser convert MODEL_FILE -o model.json       open it in https://editor.besser-pearl.org (import the project)

B-UML reference: https://besser.readthedocs.io/en/latest/buml_language.html
"""

from besser.BUML.metamodel.state_machine import Body
from besser.BUML.metamodel.state_machine.agent import (
    Agent,
    AgentReply,
)

library_agent = Agent(name="LibraryAgent")

# Intents: what a user may want, each with example sentences that express it
greeting = library_agent.new_intent(name="greeting", training_sentences=["Hi", "Hello", "Good morning"])
opening_hours = library_agent.new_intent(name="opening_hours", training_sentences=[
    "When are you open?",
    "What are your opening hours?",
])
borrow_book = library_agent.new_intent(name="borrow_book", training_sentences=[
    "I would like to borrow a book",
    "How do I borrow a book?",
])

# States: exactly one is initial, and it is created first
idle = library_agent.new_state(name="idle", initial=True)
welcome = library_agent.new_state(name="welcome")
hours = library_agent.new_state(name="hours")
borrowing = library_agent.new_state(name="borrowing")

# Bodies: what the agent does on entering a state, here a list of predefined actions (a reply)
welcome_body = Body(name="welcome_body", actions=[AgentReply("Hello! Ask me about opening hours or borrowing.")])
welcome.set_body(welcome_body)
hours_body = Body(name="hours_body", actions=[AgentReply("We are open from 9:00 to 18:00, Monday to Saturday.")])
hours.set_body(hours_body)


# A body can also be a function: it gets the user's session (an AgentSession), to store data and reply
def borrowing_body(session):
    session.set("asked_to_borrow", True)
    session.reply("Show your library card at the desk; you can keep a book for three weeks.")


borrowing.set_body(Body(name="borrowing_body", callable=borrowing_body))

# A fallback body: what a state does when the user's message matches none of its intents
idle.set_fallback_body(Body(name="idle_fallback_body", actions=[AgentReply("Sorry, I did not understand that.")]))

# Transitions: on a matched intent, move to another state
idle.when_intent_matched(greeting).go_to(welcome)
idle.when_intent_matched(opening_hours).go_to(hours)
idle.when_intent_matched(borrow_book).go_to(borrowing)
welcome.when_intent_matched(opening_hours).go_to(hours)
welcome.when_intent_matched(borrow_book).go_to(borrowing)

# An automatic transition: taken at once, without waiting for the user
hours.go_to(idle)
borrowing.go_to(idle)
