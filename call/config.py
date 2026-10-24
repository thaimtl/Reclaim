"""Settings from the environment and call/keys.env (git-ignored, never committed; call/.env also works).
Secrets live only there, never in code or logs."""
import os
import secrets
from dataclasses import dataclass, field

try:
    from dotenv import load_dotenv
    for _name in ('keys.env', '.env'):     # keys.env: visible in file browsers that hide dot files (Jupyter)
        load_dotenv(os.path.join(os.path.dirname(__file__), _name))
except ImportError:  # pragma: no cover
    pass


def _env(name, default=None):
    v = os.environ.get(name)
    return v if v not in (None, '') else default


@dataclass(frozen=True)
class Config:
    cartesia_key: str = field(default_factory=lambda: _env('CARTESIA_API_KEY'), repr=False)
    voice_id: str = field(default_factory=lambda: _env('CARTESIA_VOICE_ID'))          # Austin's instant clone
    tts_model: str = field(default_factory=lambda: _env('CARTESIA_TTS_MODEL', 'sonic-3.6'))
    stt_model: str = field(default_factory=lambda: _env('CARTESIA_STT_MODEL', 'ink-2'))
    # repo plan (README, REALTIME_PIPELINE.md): each word goes to Sonic as soon as it locks, max_buffer_delay_ms low.
    # CARTESIA_PHRASE_WORDS > 1 is an opt-in experiment: group words client-side (then use CARTESIA_MAX_BUFFER_MS=0).
    max_buffer_delay_ms: int = field(default_factory=lambda: int(_env('CARTESIA_MAX_BUFFER_MS', '120')))
    phrase_words: int = field(default_factory=lambda: int(_env('CARTESIA_PHRASE_WORDS', '1')))
    twilio_sid: str = field(default_factory=lambda: _env('TWILIO_ACCOUNT_SID'))
    twilio_token: str = field(default_factory=lambda: _env('TWILIO_AUTH_TOKEN'), repr=False)
    twilio_number: str = field(default_factory=lambda: _env('TWILIO_NUMBER'))         # the number we call from
    public_host: str = field(default_factory=lambda: _env('PUBLIC_HOST'))              # ngrok / cloudflared host
    port: int = field(default_factory=lambda: int(_env('PORT', '8787')))
    noise_gate: bool = field(default_factory=lambda: _env('NOISE_GATE', '1') != '0')
    # debug only, with the caller's consent: write the caller's raw audio (mu-law 8 kHz) here to tune the gate
    record_inbound: str = field(default_factory=lambda: _env('RECORD_INBOUND'))
    # unguessable path for the media websocket: only Twilio (who gets it in our TwiML) can open it
    media_token: str = field(default_factory=lambda: _env('MEDIA_TOKEN', secrets.token_urlsafe(16)), repr=False)
    # team bus (ZeroMQ on the loopback, one TCP port per producer: docs/TEAM_TASKS_AND_INTERFACES.md, bus.team): we
    # connect to the decoder's and the camera reader's publishers and bind our own. Each env var overrides.
    dec_pub: str = field(default_factory=lambda: _env('RECLAIM_DEC_PUB', 'tcp://127.0.0.1:5557'))
    cam_pub: str = field(default_factory=lambda: _env('RECLAIM_CAM_PUB', 'tcp://127.0.0.1:5556'))
    call_pub: str = field(default_factory=lambda: _env('RECLAIM_CALL_PUB', 'tcp://127.0.0.1:5559'))

    def missing(self, *names):
        """Names of required settings that are empty, for a clear startup error."""
        return [n for n in names if not getattr(self, n)]
