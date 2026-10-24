"""Call service without network: fake Sonic, fake Ink, a fake Twilio socket through aiohttp's test client.

    venv-call/bin/python -m pytest tests/call -q
"""
import asyncio
import base64
import json
import os
import sys
from types import SimpleNamespace

import pytest
import zmq
import zmq.asyncio

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from call import bus, twilio_media  # noqa: E402
from call.config import Config  # noqa: E402
from call.voice import Utterance, Voice, generation_config  # noqa: E402



# --- pure pieces -----------------------------------------------------------------------------------------------
def test_generation_config_maps_and_clamps():
    assert generation_config('happy', 0.9) == {'emotion': 'happy', 'speed': 1.1}
    assert generation_config('happy', 0.2)['emotion'] == 'neutral'         # unsure face: neutral voice
    assert generation_config('confused', 1.0)['emotion'] == 'neutral'      # unknown label
    g = generation_config('sad', 1.0, pace=3.0, effort=1.0)
    assert g == {'emotion': 'sad', 'speed': 1.5, 'volume': 1.0}             # effort 1.0 = Austin's normal
    g = generation_config('neutral', 1.0, pace=0.1, effort=0.0)
    assert g == {'emotion': 'neutral', 'speed': 0.6, 'volume': 0.5}
    assert generation_config('happy', 1.0, effort=4.0)['volume'] == 2.0    # emphatic, clamped


def test_utterance_handles_delta_and_cumulative_messages():
    u = Utterance(1)
    assert u.new_words(['I', 'GOT']) == ['I', 'GOT']
    assert u.new_words(['I', 'GOT', 'THE']) == ['THE']          # cumulative
    assert u.new_words(['PILLS']) == ['PILLS']                  # delta
    assert u.new_words(['I', 'GOT', 'THE', 'PILLS']) == []      # repeat: nothing spoken twice
    assert u.sent == ['I', 'GOT', 'THE', 'PILLS']


def test_twiml_and_frames():
    x = twilio_media.twiml_stream('abc.ngrok.app', 'tok&en')
    assert 'wss://abc.ngrok.app/media/tok&amp;en' in x and '<Connect><Stream' in x
    chunks = list(twilio_media.frames(bytes(400)))
    assert [len(c) for c in chunks] == [160, 160, 80]
    payload = base64.b64encode(b'\xff' * 160).decode()
    ev, m = twilio_media.parse(json.dumps({'event': 'media', 'media': {'payload': payload}}))
    assert ev == 'media' and m['audio'] == b'\xff' * 160


# --- fakes ---------------------------------------------------------------------------------------------------
class FakeCtx:
    def __init__(self, conn, cid, **defaults):
        self.conn, self.cid, self.defaults = conn, cid, defaults
        self.pushed, self.closed = [], False
        self.q = asyncio.Queue()

    async def push(self, transcript, generation_config=None):
        self.pushed.append((transcript, generation_config))
        await self.q.put(SimpleNamespace(type='chunk', data='x', audio=transcript.encode(), done=False))

    async def no_more_inputs(self):
        self.closed = True
        await self.q.put(SimpleNamespace(type='done', data=None, done=True))

    async def receive(self):
        while True:
            ev = await self.q.get()
            yield ev
            if ev.done:
                return


class FakeConn:
    def __init__(self):
        self.contexts = []

    def context(self, cid, **kw):
        c = FakeCtx(self, cid, **kw)
        self.contexts.append(c)
        return c


class FakeTTS:
    def __init__(self, conn):
        self.conn = conn

    def websocket_connect(self):
        conn = self.conn

        class CM:
            async def __aenter__(self):
                return conn

            async def __aexit__(self, *a):
                return None
        return CM()


def cfg(**kw):
    base = dict(cartesia_key='k', voice_id='austin', tts_model='sonic-3.6', public_host='h', media_token='tok',
                call_pub='inproc://call', dec_pub='inproc://dec', cam_pub='inproc://cam')
    base.update(kw)
    return Config(**base)


async def make_voice(sink):
    conn = FakeConn()
    v = Voice(cfg(), sink, client=SimpleNamespace(tts=FakeTTS(conn)))
    await v.start()
    return v, conn


# --- voice ---------------------------------------------------------------------------------------------------
async def test_voice_streams_words_then_closes_context():
    heard = []

    async def sink(b):
        heard.append(b)
    v, conn = await make_voice(sink)
    v.set_emotion('happy', 0.9)
    await v.say(7, ['A', 'BURRITO'])
    await v.say(7, ['A', 'BURRITO', 'IT', 'WAS'])
    await v.say(7, ['AMAZING'], final=True)
    await asyncio.sleep(0.05)
    ctx = conn.contexts[0]
    assert len(conn.contexts) == 1, 'one context per utterance'
    assert [p[0] for p in ctx.pushed] == ['A burrito', ' it was', ' amazing.']   # each word as it arrives (the plan)
    assert ctx.pushed[0][1]['emotion'] == 'happy'
    assert ctx.defaults['output_format']['encoding'] == 'pcm_mulaw' and ctx.defaults['voice'] == {'id': 'austin'}
    assert ctx.closed
    assert heard == [b'A burrito', b' it was', b' amazing.']
    await v.say(7, ['AGAIN'])                     # after final: ignored
    assert len(ctx.pushed) == 3
    await v.close()


async def test_voice_two_utterances_two_contexts():
    async def sink(b):
        pass
    v, conn = await make_voice(sink)
    await v.say(1, ['HI'], final=True)
    await v.say(2, ['THERE'], final=True)
    assert len(conn.contexts) == 2
    await v.close()


# --- whole service over a fake Twilio socket ---------------------------------------------------------------
class FakeInk:
    instances = []

    def __init__(self, on_text):
        self.on_text, self.fed, self.closed = on_text, bytearray(), False
        FakeInk.instances.append(self)

    async def start(self):
        pass

    def feed(self, b):
        self.fed += b

    async def close(self):
        self.closed = True


class ListSub:
    """Subscriber stand-in: yields queued (topic, msg) pairs."""

    def __init__(self):
        self.q = asyncio.Queue()

    async def recv(self):
        return await self.q.get()

    def close(self):
        pass


async def test_service_end_to_end(aiohttp_client):
    ctx = zmq.asyncio.Context()
    pub = bus.Publisher('inproc://call-test', ctx)
    listen = ctx.socket(zmq.SUB)
    listen.connect('inproc://call-test')
    listen.setsockopt(zmq.SUBSCRIBE, b'')
    sub = ListSub()
    from call.server import Service
    conn = FakeConn()
    c = cfg()
    voice = Voice(c, None, client=SimpleNamespace(tts=FakeTTS(conn)))
    svc = Service(c, voice=voice, ink_factory=FakeInk, pub=pub, sub=sub)
    voice.sink, voice.on_timing, voice.on_done = svc._audio, svc._timing, svc._voice_done
    client = await aiohttp_client(svc.app())

    assert (await client.get('/media/wrong')).status == 404
    assert (await client.post('/twiml', data={'CallSid': 'CA1'})).status == 403          # unsigned: refused
    ws = await client.ws_connect('/media/tok')
    await ws.send_str(json.dumps({'event': 'connected'}))
    await ws.send_str(json.dumps({'event': 'start', 'start': {'streamSid': 'MZ1', 'callSid': 'CA1'}}))
    payload = base64.b64encode(b'\x7f' * 160).decode()
    await ws.send_str(json.dumps({'event': 'media', 'media': {'track': 'inbound', 'payload': payload}}))
    await asyncio.sleep(0.05)
    ink = FakeInk.instances[-1]
    assert bytes(ink.fed) == b'\x7f' * 160, 'judge audio reaches Ink untouched'
    await ink.on_text('did you pick up your meds', True, 1.0)          # Ink -> partner.text on the bus

    sub.q.put_nowait(('cam.tick', {'emotion': {'label': 'happy', 'confidence': 0.8, 'scores': {}}}))
    sub.q.put_nowait(('voice.say', {'utt_id': 3, 'words': ['I', 'GOT', 'THE', 'PILLS'], 'final': True}))
    out = []
    while len(out) < 2:                                                  # one media frame, then the mark
        m = json.loads((await asyncio.wait_for(ws.receive(), 2)).data)
        out.append(m)
    assert out[0]['event'] == 'media' and out[0]['streamSid'] == 'MZ1'
    assert base64.b64decode(out[0]['media']['payload']) == b'I got the pills.'
    assert out[1] == {'event': 'mark', 'streamSid': 'MZ1', 'mark': {'name': 'utt-3'}}
    assert conn.contexts[0].pushed[0][1]['emotion'] == 'happy'
    await ws.send_str(json.dumps({'event': 'mark', 'mark': {'name': 'utt-3'}}))   # Twilio: it has played
    await ws.send_str(json.dumps({'event': 'stop'}))
    await asyncio.sleep(0.05)

    topics = []
    while True:
        try:
            t, m = bus.decode(await asyncio.wait_for(listen.recv_multipart(), 0.2))
            topics.append((t, m))
        except asyncio.TimeoutError:
            break
    names = [t for t, _ in topics]
    assert 'partner.text' in names and 'call.state' in names
    pt = next(m for t, m in topics if t == 'partner.text')
    assert pt['text'] == 'did you pick up your meds' and pt['is_final'] is True and pt['v'] == 2
    stages = [m['stage'] for t, m in topics if t == 'timing.voice']
    assert 'sonic_first_audio' in stages and 'played_to_judge' in stages
    states = [m['state'] for t, m in topics if t == 'call.state']
    assert states == ['connected', 'ended']
    assert ink.closed
    listen.close()


def test_twilio_signature_check():
    from twilio.request_validator import RequestValidator
    c = cfg(twilio_token='secret')
    url, params = 'https://h/twiml', {'CallSid': 'CA1', 'From': '+1415'}
    good = RequestValidator('secret').compute_signature(url, params)
    assert twilio_media.signed(c, url, params, good)
    assert not twilio_media.signed(c, url, params, 'forged')
    assert not twilio_media.signed(cfg(twilio_token=None), url, params, good)


def test_speakable_text():
    from call.voice import end_mark, speakable
    assert speakable(['I', 'GOT', 'THE', 'PILLS'], True) == 'I got the pills'
    assert speakable(["I'M", 'FINE'], False) == "I'm fine"
    assert end_mark('WHAT') == '?' and end_mark('TELL') == '.'


async def test_final_without_new_words_still_ends_the_sentence():
    async def sink(b):
        pass
    v, conn = await make_voice(sink)
    await v.say(5, ['WHERE', 'IS', 'IT'])
    await v.say(5, [], final=True)
    assert [p[0] for p in conn.contexts[0].pushed] == ['Where is it', '?']
    await v.close()


def test_noise_gate_keeps_the_caller_and_drops_the_room():
    import numpy as np
    from call.ink import FRAME, NoiseGate, level_dbfs

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        import audioop                               # reference G.711 codec (Python 3.11)

    def mulaw(x):                                   # linear float [-1, 1] -> mu-law bytes
        return audioop.lin2ulaw(np.clip(x * 32767, -32768, 32767).astype('<i2').tobytes(), 2)

    rng = np.random.default_rng(0)
    babble = [mulaw(rng.normal(0, 0.01, FRAME)) for _ in range(150)]      # room voices, ~ -40 dBFS
    caller = [mulaw(rng.normal(0, 0.2, FRAME)) for _ in range(50)]        # phone at the ear, ~ -14 dBFS
    assert level_dbfs(caller[0]) - level_dbfs(babble[0]) > 20
    g = NoiseGate()
    room = [g.push(f) for f in babble]
    talk = [g.push(f) for f in caller]
    after = [g.push(f) for f in babble[:40]]
    assert sum(f != bytes([0xFF]) * FRAME for f in room[60:]) == 0, 'background silenced once the floor is known'
    assert all(a == b for a, b in zip(talk, caller)), 'caller passes untouched'
    assert after[:14] == babble[:14] and all(f == bytes([0xFF]) * FRAME for f in after[14:]), '~300 ms hangover'


def test_mulaw_table_matches_reference_codec():
    import warnings
    import numpy as np
    from call.ink import MULAW
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        import audioop
    ref = np.frombuffer(audioop.ulaw2lin(bytes(range(256)), 2), '<i2')
    assert np.array_equal(MULAW.astype(np.int16), ref)


def test_phrase_policy():
    from call.voice import ready
    assert ready(['I'], 0) == 1 and ready(['THE'], 3) == 1  # the plan: every word as soon as it arrives
    assert ready(['I'], 0, 3) == 0                          # opt-in grouping experiment:
    assert ready(['I', 'GOT'], 0, 3) == 2                   # first phrase: 2 words
    assert ready(['THE', 'PILLS'], 2, 3) == 0               # later phrases: 3 words
    assert ready(['GOT', 'THE', 'PILLS'], 2, 3) == 3
    assert ready(['WENT', 'TO', 'THE'], 2, 3) == 0          # never end on a function word
    assert ready(['A', 'B', 'TO', 'THE', 'OF'], 2, 3) == 5  # but never hold more than 5


def test_defaults_follow_the_repo_plan():
    c = Config()
    if 'CARTESIA_PHRASE_WORDS' not in os.environ and 'CARTESIA_MAX_BUFFER_MS' not in os.environ:
        assert c.phrase_words == 1 and 0 < c.max_buffer_delay_ms <= 200


async def test_pause_speaks_waiting_words():
    import call.voice as cv
    async def sink(b):
        pass
    conn = FakeConn()
    v = Voice(cfg(phrase_words=3), sink, client=SimpleNamespace(tts=FakeTTS(conn)))   # the grouping experiment
    await v.start()
    old, cv.PAUSE_S = cv.PAUSE_S, 0.05
    try:
        await v.say(9, ['I', 'WENT'])                       # first phrase goes at once
        await v.say(9, ['I', 'WENT', 'TO', 'THE'])          # held: ends on a function word
        assert [p[0] for p in conn.contexts[0].pushed] == ['I went']
        await asyncio.sleep(0.15)                           # Austin pauses: the waiting words are spoken
        assert [p[0] for p in conn.contexts[0].pushed] == ['I went', ' to the']
    finally:
        cv.PAUSE_S = old
        await v.close()


def test_defaults_match_the_team_bus():
    import sys
    if any(f'RECLAIM_{n}_PUB' in os.environ for n in ('CAM', 'DEC', 'CALL')):
        pytest.skip('endpoints overridden in the environment')
    c = Config()
    assert (c.cam_pub, c.dec_pub, c.call_pub) == ('tcp://127.0.0.1:5556', 'tcp://127.0.0.1:5557', 'tcp://127.0.0.1:5559')
    sys.path.insert(0, ROOT)
    try:
        import bus.team as team                   # the repo's own table, when bus/ is importable (needs numpy)
    except Exception:                             # noqa: BLE001
        return
    assert (team.ENDPOINTS['cam'], team.ENDPOINTS['dec'], team.ENDPOINTS['call']) == (c.cam_pub, c.dec_pub, c.call_pub)


async def test_ipc_publisher_creates_its_folder(tmp_path):
    addr = f'ipc://{tmp_path}/sub/dir/call'
    p = bus.Publisher(addr, zmq.asyncio.Context())
    assert os.path.isdir(f'{tmp_path}/sub/dir')
    p.close()


# --- review fixes -------------------------------------------------------------------------------------------------
class SlowCtx(FakeCtx):
    """Audio arrives later than the push (as from Sonic), in two pieces."""

    async def push(self, transcript, generation_config=None):
        self.pushed.append((transcript, generation_config))

        async def later():
            await asyncio.sleep(0.03)
            await self.q.put(SimpleNamespace(type='chunk', data='x', audio=transcript.encode()[:3], done=False))
            await asyncio.sleep(0.03)
            await self.q.put(SimpleNamespace(type='chunk', data='x', audio=transcript.encode()[3:], done=False))
        asyncio.ensure_future(later())

    async def no_more_inputs(self):
        self.closed = True

        async def later():
            await asyncio.sleep(0.1)
            await self.q.put(SimpleNamespace(type='done', data=None, done=True))
        asyncio.ensure_future(later())


class SlowConn(FakeConn):
    def context(self, cid, **kw):
        c = SlowCtx(self, cid, **kw)
        self.contexts.append(c)
        return c


async def test_two_overlapping_utterances_never_interleave():
    heard, done = [], []

    async def sink(b):
        heard.append(b)

    async def on_done(u, t):
        done.append((u, len(heard)))
    conn = SlowConn()
    v = Voice(cfg(), sink, client=SimpleNamespace(tts=FakeTTS(conn)), on_done=on_done)
    await v.start()
    await v.say(1, ['GOOD', 'MORNING'], final=True)
    await v.say(2, ['HOW', 'ARE', 'YOU'], final=True)          # starts while 1 is still streaming
    await asyncio.sleep(0.5)
    assert b''.join(heard) == b'Good morning.How are you?'
    assert done == [(1, 2), (2, 4)], 'each mark only after all of its own audio'
    await v.close()


class DyingCtx(FakeCtx):
    async def push(self, transcript, generation_config=None):
        if self.conn.dead:
            raise ConnectionError('socket closed')
        await super().push(transcript, generation_config)


class DyingConn(FakeConn):
    def __init__(self):
        super().__init__()
        self.dead = False

    def context(self, cid, **kw):
        c = DyingCtx(self, cid, **kw)
        self.contexts.append(c)
        return c

    async def close(self):
        pass


class ReconnectingTTS:
    """Each websocket_connect() gives a fresh connection, as a real reconnect does."""

    def __init__(self):
        self.conns = []

    def websocket_connect(self):
        conn = DyingConn()
        self.conns.append(conn)

        class CM:
            async def __aenter__(self):
                return conn
        return CM()


async def test_reconnects_and_resends_the_phrase_when_sonic_dies():
    heard = []

    async def sink(b):
        heard.append(b)
    tts = ReconnectingTTS()
    v = Voice(cfg(), sink, client=SimpleNamespace(tts=tts))
    await v.start()
    await v.say(4, ['I', 'GOT'])
    tts.conns[0].dead = True                                     # the socket dies mid-utterance
    await v.say(4, ['I', 'GOT', 'THE', 'PILLS'], final=True)
    await asyncio.sleep(0.05)
    assert len(tts.conns) == 2, 'a fresh connection, not the cached dead one'
    assert [p[0] for p in tts.conns[1].contexts[0].pushed] == [' the pills.']
    assert heard[-1] == b' the pills.'
    await v.close()


async def test_failed_final_push_still_ends_the_utterance():
    heard = []

    async def sink(b):
        heard.append(b)
    tts = ReconnectingTTS()
    v = Voice(cfg(), sink, client=SimpleNamespace(tts=tts))
    await v.start()
    orig = tts.websocket_connect

    def always_dead():
        cm = orig()
        tts.conns[-1].dead = True
        return cm
    tts.websocket_connect = always_dead
    tts.conns[0].dead = True
    await v.say(5, ['HELLO', 'THERE'], final=True)              # both attempts fail
    await v.say(6, ['STILL', 'WORKS'], final=True)             # the next one is not stuck behind it
    assert v.utts[5].ended
    await v.close()


async def test_reused_utt_id_after_a_decoder_restart_is_spoken():
    import call.voice as cv
    async def sink(b):
        pass
    v, conn = await make_voice(sink)
    old, cv.REUSE_S = cv.REUSE_S, 0.01
    try:
        await v.say(1, ['HI', 'THERE'], final=True)
        await v.say(1, ['HI', 'THERE'], final=True)            # immediate duplicate: ignored
        assert len(conn.contexts) == 1
        await asyncio.sleep(0.05)
        await v.say(1, ['NEW', 'SENTENCE'], final=True)        # decoder restarted and counts from 1 again
        assert len(conn.contexts) == 2
    finally:
        cv.REUSE_S = old
        await v.close()


def test_offset_removes_delta_cumulative_guessing():
    u = Utterance(1)
    assert u.new_words(['NO'], offset=0) == ['NO']
    assert u.new_words(['NO'], offset=1) == ['NO']             # a real second "no", not a repeat
    assert u.new_words(['NO', 'NO', 'WAY'], offset=0) == ['WAY']


async def test_signed_twiml_returns_the_stream(aiohttp_client):
    from twilio.request_validator import RequestValidator
    from call.server import Service
    conn = FakeConn()
    c = cfg(twilio_token='secret', public_host='h.example')
    voice = Voice(c, None, client=SimpleNamespace(tts=FakeTTS(conn)))
    svc = Service(c, voice=voice, ink_factory=FakeInk, pub=bus.Publisher('inproc://twiml-test', zmq.asyncio.Context()),
                  sub=ListSub())
    voice.sink = svc._audio
    client = await aiohttp_client(svc.app())
    params = {'CallSid': 'CA9', 'From': '+14155550100'}
    sig = RequestValidator('secret').compute_signature('https://h.example/twiml', params)
    r = await client.post('/twiml', data=params, headers={'X-Twilio-Signature': sig})
    assert r.status == 200 and 'wss://h.example/media/tok' in await r.text()


async def test_whole_sentence_fallback_when_streaming_keeps_failing():
    """docs/REALTIME_PIPELINE.md: if streaming voice glitches, the voice falls back to whole sentences at the end."""
    heard, done = [], []

    async def sink(b):
        heard.append(b)

    async def on_done(u, t):
        done.append(u)

    class Resp:
        def __init__(self, text):
            self.text = text

        async def read(self):
            return b'WHOLE:' + self.text.encode()

    tts = ReconnectingTTS()
    orig = tts.websocket_connect

    def always_dead():
        cm = orig()
        tts.conns[-1].dead = True
        return cm
    calls = []

    async def generate(**kw):
        calls.append(kw)
        return Resp(kw['transcript'])
    tts.generate = generate
    v = Voice(cfg(), sink, client=SimpleNamespace(tts=tts), on_done=on_done)
    await v.start()
    tts.websocket_connect = always_dead
    tts.conns[0].dead = True
    await v.say(8, ['I', 'GOT', 'THE', 'PILLS'], final=True)
    await asyncio.sleep(0.05)
    assert heard == [b'WHOLE:I got the pills.']
    assert calls[0]['output_format']['encoding'] == 'pcm_mulaw' and done == [8]
    await v.close()
