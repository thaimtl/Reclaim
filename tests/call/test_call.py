"""Call service without network: fake Sonic, fake Ink, a fake Twilio socket through aiohttp's test client.

    venv-call/bin/python -m pytest tests/call -q
"""
import asyncio
import base64
import json
import os
import sys
from types import SimpleNamespace

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
    assert generation_config('happy', 1.0, effort=1.0)['volume'] == 1.0     # effort 1.0 = his normal = normal volume
    g = generation_config('sad', 1.0, pace=3.0, effort=3.0)
    assert g == {'emotion': 'sad', 'speed': 1.5, 'volume': 2.0}
    g = generation_config('neutral', 1.0, pace=0.1, effort=0.0)
    assert g == {'emotion': 'neutral', 'speed': 0.6, 'volume': 0.5}


def test_utterance_words_without_offset_are_new_words():
    """No guessing: without an offset every message is the next words (the decoder's voice.say), so a real repeat is
    spoken ("no no way"); with an offset, resent words are skipped (nothing spoken twice)."""
    u = Utterance(1)
    assert u.new_words(['NO']) == ['NO']
    assert u.new_words(['NO', 'WAY']) == ['NO', 'WAY']           # a second "no", not the first one resent
    assert u.sent == ['NO', 'NO', 'WAY']
    v = Utterance(2)
    assert v.new_words(['I', 'GOT'], offset=0) == ['I', 'GOT']
    assert v.new_words(['I', 'GOT', 'THE'], offset=0) == ['THE']   # cumulative, said so by the offset
    assert v.new_words(['THE', 'PILLS'], offset=2) == ['PILLS']
    assert v.sent == ['I', 'GOT', 'THE', 'PILLS']


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
    await v.say(7, ['IT', 'WAS'])                 # only the new words (voice.say)
    await v.say(7, ['AMAZING'], final=True)
    await asyncio.sleep(0.05)
    ctx = conn.contexts[0]
    assert len(conn.contexts) == 1, 'one context per utterance'
    assert [p[0] for p in ctx.pushed] == ['A burrito', ' it was', ' amazing.']   # each word as it arrives (the plan)
    assert ctx.defaults['generation_config']['emotion'] == 'happy'
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
    assert out[1] == {'event': 'mark', 'streamSid': 'MZ1', 'mark': {'name': 'utt-3-1'}}
    assert conn.contexts[0].defaults['generation_config']['emotion'] == 'happy'   # set when the context opens
    await ws.send_str(json.dumps({'event': 'mark', 'mark': {'name': 'utt-3-1'}}))   # Twilio: it has played
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
        await v.say(9, ['TO', 'THE'])                       # held: ends on a function word
        assert [p[0] for p in conn.contexts[0].pushed] == ['I went']
        await asyncio.sleep(0.15)                           # Austin pauses: the waiting words are spoken
        assert [p[0] for p in conn.contexts[0].pushed] == ['I went', ' to the']
    finally:
        cv.PAUSE_S = old
        await v.close()


def test_defaults_match_the_team_bus():
    import importlib
    import sys
    from call import config
    sys.path.insert(0, ROOT)
    team = importlib.import_module('bus')         # the repo's bus package: the camera reader's own default
    assert config.DEFAULT_CAM_PUB == team.DEFAULT_CAM_PUB
    assert bus.VERSION == team.VERSION            # the same envelope as the rest of the team
    assert len({config.DEFAULT_CAM_PUB, config.DEFAULT_DEC_PUB, config.DEFAULT_CALL_PUB}) == 3
    assert all(a.startswith('tcp://127.0.0.1:') for a in (config.DEFAULT_DEC_PUB, config.DEFAULT_CALL_PUB))


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
    await v.say(4, ['THE', 'PILLS'], final=True)
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


def test_inline_tags_only_for_changes_inside_a_context():
    from call.voice import inline_tags
    base = {'emotion': 'neutral', 'speed': 1.1}
    assert inline_tags(base, dict(base)) == ''
    assert inline_tags(base, {'emotion': 'happy', 'speed': 1.1}) == '<emotion value="happy"/> '
    assert inline_tags(base, {'emotion': 'neutral', 'speed': 1.32, 'volume': 1.8}) == \
        '<speed ratio="1.32"/><volume ratio="1.80"/> '
    assert inline_tags(None, base) == ''


async def test_a_change_mid_sentence_goes_inline_not_in_generation_config():
    conn = FakeConn()
    v = Voice(cfg(), lambda b: asyncio.sleep(0), client=SimpleNamespace(tts=FakeTTS(conn)))
    await v.start()
    v.set_emotion('neutral', 1.0)
    await v.say(3, ['I', 'GOT'])
    v.set_emotion('happy', 1.0)                                 # his face changes mid-sentence
    await v.say(3, ['IT'], effort=1.8)                          # and he stresses a word
    await v.say(3, ['TODAY'], final=True)
    pushed = [p for p, _ in conn.contexts[0].pushed]
    assert len(conn.contexts) == 1 and conn.contexts[0].defaults['generation_config']['emotion'] == 'neutral'
    assert pushed[1] == ' <emotion value="happy"/><volume ratio="1.80"/> it'
    assert pushed[2] == ' today.'                               # no effort sent: the volume stays as it is
    assert all(g is None for _, g in conn.contexts[0].pushed)   # generation_config only when the context opens
    await v.close()


async def test_fallback_after_the_audio_ended_is_still_played():
    """Words left for the whole-sentence fallback after the context's audio already ended get their own run."""
    heard = []

    async def sink(b):
        heard.append(b)

    class Gen:
        async def generate(self, **kw):
            return SimpleNamespace(read=lambda: asyncio.sleep(0, result=b'FALLBACK'))
    v = Voice(cfg(), sink, client=SimpleNamespace(tts=Gen()))
    v._player = asyncio.create_task(v._play())
    u = Utterance(11)
    u.sent, u.final, u.lost = ['HELLO'], True, ['HELLO']
    v.utts[11] = u
    v._ensure_run(u)
    u.audio.put_nowait(b'early')
    u.end_audio()                                               # the context's audio already ended
    await v._lost_then_end(u)
    await asyncio.sleep(0.05)
    assert heard == [b'early', b'FALLBACK']
    v._player.cancel()


def test_gate_floor_ignores_the_callers_speech():
    from call.ink import FRAME, NoiseGate
    import audioop  # noqa: F401  (py3.11 still has it; only to build mu-law frames)
    quiet = audioop.lin2ulaw(b'\x10\x00' * FRAME, 2)
    loud = audioop.lin2ulaw(b'\x00\x30' * FRAME, 2)
    g = NoiseGate()
    for _ in range(50):
        g.push(quiet)
    for _ in range(300):                                        # 6 s of continuous speech, no pause
        out = g.push(loud)
    assert out == loud, 'the gate must stay open through a long sentence'


def test_gate_floor_follows_a_room_that_gets_louder():
    from call.ink import FRAME, NoiseGate
    import audioop  # noqa: F401
    quiet = audioop.lin2ulaw(b'\x10\x00' * FRAME, 2)
    crowd = audioop.lin2ulaw(b'\x00\x20' * FRAME, 2)          # the expo fills up: steady noise, well above the start
    g = NoiseGate()
    for _ in range(50):
        g.push(quiet)
    outs = [g.push(crowd) for _ in range(2000)]                 # 40 s of crowd
    assert outs[-1] != crowd, 'a louder room must not hold the gate open forever'


async def test_effort_and_pace_carry_over_when_a_message_leaves_them_out():
    conn = FakeConn()
    v = Voice(cfg(), lambda b: asyncio.sleep(0), client=SimpleNamespace(tts=FakeTTS(conn)))
    await v.start()
    await v.say(4, ['SO'], pace=1.0, effort=1.0)
    await v.say(4, ['LOUD'], effort=2.0)
    await v.say(4, ['RIGHT', 'NOW'], final=True)              # no pace or effort: keep the last ones
    pushed = [p for p, _ in conn.contexts[0].pushed]
    assert pushed[1] == ' <volume ratio="2.00"/> loud' and pushed[2] == ' right now.'
    assert v.utts[4].gen['volume'] == 2.0 and v.utts[4].gen['speed'] == conn.contexts[0].defaults['generation_config']['speed']
    await v.close()


async def test_end_mark_alone_does_not_open_a_new_context():
    conn = FakeConn()
    v = Voice(cfg(), lambda b: asyncio.sleep(0), client=SimpleNamespace(tts=FakeTTS(conn)))
    await v.start()
    u = Utterance(6)
    u.sent, u.spoken, u.final = ['HI'], 1, True
    v.utts[6] = u
    await v._send(u, 0, end=True)                               # its context timed out earlier: nothing to close
    assert conn.contexts == []
    await v.close()


async def test_after_a_double_failure_the_rest_keeps_its_order():
    conn = FakeConn()
    v = Voice(cfg(), lambda b: asyncio.sleep(0), client=SimpleNamespace(tts=FakeTTS(conn)))
    await v.start()
    u = Utterance(8)
    u.sent, u.pending, u.broken, u.lost = ['A', 'B', 'C'], ['C'], True, ['A', 'B']
    v.utts[8] = u
    await v._send(u, 1)
    assert u.lost == ['A', 'B', 'C'] and conn.contexts == []
    await v.close()
