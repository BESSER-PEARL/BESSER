"""The BESSER Agentic Framework release generated agents are built and deployed against.

One place for the version: the Docker Compose images, the web-app agent images, the GitHub / Render
deployments and their READMEs all install this release. The agent simulator pins the same version in its
``requirements.txt`` (a plain file cannot import this), and a test keeps the two equal.

4.6.1 is the first release whose ``WebSocketPlatform.reply_options`` accepts a ``label``; generated agents pass
one when an options reply has a label, and an older BAF would drop the buttons.
"""

BAF_VERSION = "4.6.1"
