# You may need to add your working directory to the Python path. To do so, uncomment the following lines of code
# import sys
# sys.path.append("/Path/to/directory/bot-framework") # Replace with your directory path

import logging

from besser.bot.core.bot import Bot
from besser.bot.core.session import Session
from besser.bot.nlp.intent_classifier.intent_classifier_configuration import LLMIntentClassifierConfiguration, SimpleIntentClassifierConfiguration

# Configure the logging module
logging.basicConfig(level=logging.INFO, format='{levelname} - {asctime}: {message}', style='{')

# Create the bot
bot = Bot('bot')
# Load bot properties stored in a dedicated file
bot.load_properties('config.ini')
# Define the platform your chatbot will use
websocket_platform = bot.use_websocket_platform(use_ui=True)

##############################
# INTENTS
##############################

hello_intent = bot.new_intent('hello_intent', [
    'hello',
    'hi',
    'whats up?',
    ])

good_intent = bot.new_intent('good_intent', [
    'good',
    'fine',
    ])

bad_intent = bot.new_intent('bad_intent', [
    'bad',
    'awful',
    ])

##############################
# STATES
##############################

initial_state = bot.new_state('initial_state', initial=True)
hello_state = bot.new_state('hello_state')
good_state = bot.new_state('good_state')
bad_state = bot.new_state('bad_state')


# initial_state


initial_state.when_intent_matched_go_to(hello_intent, hello_state)

# hello_state
def hello_body(session: Session):
    session.reply('Hi! How are you?')


hello_state.set_body(hello_body)


hello_state.when_intent_matched_go_to(good_intent, good_state)
hello_state.when_intent_matched_go_to(bad_intent, bad_state)

# good_state
def good_body(session: Session):
    session.reply('I am glad to hear that!')


good_state.set_body(good_body)



good_state.go_to(initial_state)

# bad_state
def bad_body(session: Session):
    session.reply('I am sorry to hear that...')


bad_state.set_body(bad_body)



bad_state.go_to(initial_state)

# RUN APPLICATION

if __name__ == '__main__':
    bot.run()