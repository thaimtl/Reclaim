"""Silent Voice call service (Frank): the Twilio phone call, Austin's voice (Cartesia Sonic) and the judge's words (Ink).
How it flows (Mermaid): call/README.md.

Interfaces: docs/TEAM_TASKS_AND_INTERFACES.md (topics voice.say, partner.text, cam.tick); how it fits the system:
docs/REALTIME_PIPELINE.md. Runs in venv-call (requirements-call.txt):

    venv-call/bin/python -m call.server                     # waits for calls (inbound) on PORT
    venv-call/bin/python -m call.server --dial +14155550142 # Austin calls the judge
"""
