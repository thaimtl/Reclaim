"""Reclaim call service (Frank): the Twilio phone call, Austin's voice (Cartesia Sonic) and the judge's words (Ink).

    judge's phone ── Twilio ──▶ /media websocket ──▶ Ink (ink-2, pcm_mulaw 8 kHz) ──▶ partner.text
    voice.say (Austin's decoder) + emotion (cam.tick) ──▶ Sonic websocket, one context per utterance,
                                                          pcm_mulaw 8 kHz ──▶ Twilio ──▶ judge's phone

Interfaces: Reclaim docs/TEAM_TASKS_AND_INTERFACES.md (topics voice.say, partner.text, cam.tick). Plan:
plans/call-loop/PLAN_v2.md. Runs in venv-call (requirements-call.txt):

    venv-call/bin/python -m call.server                     # waits for calls (inbound) on PORT
    venv-call/bin/python -m call.server --dial +14155550142 # Austin calls the judge
"""
