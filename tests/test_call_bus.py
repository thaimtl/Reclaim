"""The call service's own bus code (call/bus.py, no numpy) and bus.team agree on the wire: voice.say from the
decoder reaches the call, partner.text from the call reaches Claude, on the same endpoints and clock."""
import pytest

pytest.importorskip('zmq.asyncio')

from bus import team  # noqa: E402
from call import bus as call_bus  # noqa: E402
from call.config import Config  # noqa: E402


def test_voice_say_reaches_the_call():
    frames = team.encode('voice.say', {'utt_id': 3, 'words': ['i', 'want'], 'final': False, 'pace': 1.2,
                                       'effort': 0.8}, seq=5)
    topic, msg = call_bus.decode(frames)
    assert topic == 'voice.say' and msg['words'] == ['i', 'want'] and msg['v'] == 2 and msg['seq'] == 5


def test_partner_text_reaches_claude():
    topic, msg = team.decode(call_bus.encode('partner.text', {'text': 'how are you', 'is_final': True, 't': 1.0}, 7))
    assert topic == 'partner.text' and msg['text'] == 'how are you' and msg['v'] == call_bus.VERSION == 2
    assert abs(msg['t_pub'] - team.now()) < 5.0          # the same clock: bus.now() = time.perf_counter()


def test_same_endpoints(monkeypatch):
    for n in ('CAM', 'DEC', 'CALL'):
        monkeypatch.delenv(f'RECLAIM_{n}_PUB', raising=False)
    c = Config()
    defaults = team._DEFAULTS
    assert (c.cam_pub, c.dec_pub, c.call_pub) == (defaults['cam'], defaults['dec'], defaults['call'])
