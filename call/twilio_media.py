"""Twilio Media Streams: the call's audio both ways as 8 kHz mu-law over a websocket.

Twilio -> us: JSON text messages {"event": "connected" | "start" | "media" | "mark" | "stop", ...}; media payloads are
base64 mu-law, 20 ms (160 bytes) each. Us -> Twilio: {"event": "media", "streamSid", "media": {"payload"}} plays audio,
{"event": "mark", "mark": {"name"}} comes back once everything before it has played, {"event": "clear"} drops queued
audio. Cartesia sends pcm_mulaw at 8 kHz too, so audio passes through with no conversion.
"""
import base64
import json
from xml.sax.saxutils import quoteattr

FRAME = 160                 # bytes of mu-law per 20 ms at 8 kHz


def twiml_stream(public_host, token, peer=None):
    """TwiML that connects the call's audio to our media websocket (both directions). peer: the other phone's number,
    handed back in the stream's start message (customParameters), so a text can be matched to the live call."""
    url = f'wss://{public_host}/media/{token}'
    param = f'<Parameter name="peer" value={quoteattr(peer)}/>' if peer else ''
    return ('<?xml version="1.0" encoding="UTF-8"?>'
            f'<Response><Connect><Stream url={quoteattr(url)}>{param}</Stream></Connect></Response>')


EMPTY_TWIML = '<?xml version="1.0" encoding="UTF-8"?><Response></Response>'     # a text answered with no reply


def frames(mulaw):
    """Split mu-law bytes into Twilio-sized 20 ms payloads (the last may be shorter)."""
    for i in range(0, len(mulaw), FRAME):
        yield mulaw[i:i + FRAME]


def media_msg(stream_sid, chunk):
    return json.dumps({'event': 'media', 'streamSid': stream_sid,
                       'media': {'payload': base64.b64encode(chunk).decode('ascii')}})


def mark_msg(stream_sid, name):
    return json.dumps({'event': 'mark', 'streamSid': stream_sid, 'mark': {'name': name}})


def parse(text):
    """Twilio message -> (event, dict). Media payload decoded to bytes under 'audio'."""
    msg = json.loads(text)
    ev = msg.get('event')
    if ev == 'media':
        msg['audio'] = base64.b64decode(msg['media']['payload'])
    return ev, msg


def signed(cfg, url, params, signature):
    """True if a webhook really comes from our Twilio account (X-Twilio-Signature). No token: never."""
    if not cfg.twilio_token or not signature:
        return False
    from twilio.request_validator import RequestValidator
    return RequestValidator(cfg.twilio_token).validate(url, params, signature)


def dial(cfg, to_number):
    """Austin calls the judge: Twilio rings `to_number` from our number and streams the call to us."""
    from twilio.rest import Client
    client = Client(cfg.twilio_sid, cfg.twilio_token)
    call = client.calls.create(to=to_number, from_=cfg.twilio_number,
                               twiml=twiml_stream(cfg.public_host, cfg.media_token, to_number))
    return call.sid
