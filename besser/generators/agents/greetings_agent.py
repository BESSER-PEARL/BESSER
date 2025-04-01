# You may need to add your working directory to the Python path. To do so, uncomment the following lines of code
# import sys

import logging

from besser.agent.core.agent import Agent
from besser.agent.core.session import Session
from besser.agent.exceptions.logger import logger

# Configure the logging module (optional)
logger.setLevel(logging.INFO)

# Create the agent
agent = Agent('greetings_agent')
# Load agent properties stored in a dedicated file
agent.load_properties('config.ini')
# Define the platform your agent will use
websocket_platform = agent.use_websocket_platform(use_ui=True)

# STATES

initial_state = agent.new_state('initial_state', initial=True)
hello_state = agent.new_state('hello_state')
good_state = agent.new_state('good_state')
bad_state = agent.new_state('bad_state')

# INTENTS

hello_intent = agent.new_intent('hello_intent', [
    'hello',
    'hi',
])

good_intent = agent.new_intent('good_intent', [
    'good',
    'fine',
])

bad_intent = agent.new_intent('bad_intent', [
    'bad',
    'awful',
])


# STATES BODIES' DEFINITION + TRANSITIONS


initial_state.when_intent_matched_go_to(hello_intent, hello_state)
def hello_body(session: Session):
    session.reply('Hi! How are you?')

def hello_body(session: Session):
    session.reply('Hi! How are you?')


hello_state.set_body(hello_body)
hello_state.when_intent_matched_go_to(good_intent, good_state)
hello_state.when_intent_matched_go_to(bad_intent, bad_state)


def good_body(session: Session):
    session.reply('I am glad to hear that!')


good_state.set_body(good_body)
good_state.go_to(initial_state)


def bad_body(session: Session):
    session.reply('I am sorry to hear that...')


bad_state.set_body(bad_body)
bad_state.go_to(initial_state)


# RUN APPLICATION

if __name__ == '__main__':
    agent.run()
