"""Austin's voice: voice.say words -> Cartesia Sonic over one long-lived websocket -> mu-law audio for the call.

One Sonic context per utterance (utt_id), continue=True ("more to come"), the final message closes it, so speech starts
while Austin is still mouthing. As the repo plans it (README: "Cartesia speaks each word ... as soon as it locks";
REALTIME_PIPELINE.md, TEAM_TASKS_AND_INTERFACES.md: words streamed in as they arrive on voice.say, max_buffer_delay_ms
kept low), each word goes to Sonic the moment it arrives, with a 120 ms server buffer (CARTESIA_MAX_BUFFER_MS). Every
chunk carries a leading space after the first and the last one closing punctuation, so Sonic joins them as one sentence.

Opt-in experiment, not the plan (CARTESIA_PHRASE_WORDS=3 with CARTESIA_MAX_BUFFER_MS=0): group words client-side, 2
first then 3, never ending on a function word, the rest after a 0.5 s pause. Measured 2026-10-24 (sonic-3.6, a word
every 300 ms): word by word with a 120 ms buffer had ~3 s of mid-sentence silence per 14-word sentence; grouping had
none and started ~0.4 s after the second word. Austin mouths about as fast as speech plays, so the voice can only trail.

Audio order: each utterance's audio goes into its own queue and one player task plays utterances in the order they
started, so two sentences never interleave on the phone, and on_done fires only after an utterance's last audio is
queued (the server puts a Twilio mark behind it).

Reconnect: the SDK's connection manager caches its first connection, so its own reconnect hands back the dead socket;
on a failed push we open a fresh connection ourselves and resend that phrase (words of other in-flight utterances whose
audio had not arrived yet are lost, and logged).

Fallback (docs/REALTIME_PIPELINE.md: "If streaming voice glitches, the voice falls back to speaking whole sentences at
the end"): words whose streaming push still fails after the reconnect are kept, and when the utterance ends they are
spoken in one non-streaming request (tts.generate), after whatever did stream.

voice.say may carry only the new words or the whole utterance so far; Utterance.new_words() handles both, and an
explicit `offset` (index of the first word in the message) removes the guesswork when the decoder sends it.
Emotion comes from the newest cam.tick (Evan's camera reader); pace and effort from voice.say. All three go through
generation_config: emotion name, speed 0.6-1.5, volume 0.5-2.0. pace and effort are relative, 1.0 = Austin's normal
(docs/TEAM_TASKS_AND_INTERFACES.md), so effort 1.0 is volume 1.0.
"""
import asyncio
import logging
import time

log = logging.getLogger(__name__)

OUTPUT = {'container': 'raw', 'encoding': 'pcm_mulaw', 'sample_rate': 8000}
# camera label (bus.EMOTIONS, lipread/) -> Sonic emotion; all five exist in Sonic's list
EMOTION = {'neutral': 'neutral', 'happy': 'happy', 'sad': 'sad', 'angry': 'angry', 'surprised': 'surprised'}
MIN_CONFIDENCE = 0.5        # below this the face label is a guess: speak neutral
SPEED_BIAS = 1.1            # speech plays about as fast as Austin mouths; 10% faster keeps the voice up with him
FIRST_WORDS, NEXT_WORDS, MAX_WORDS = 2, 3, 5
PAUSE_S = 0.5               # Austin stopped mouthing for this long: speak what is waiting
CONTEXT_TIMEOUT_S = 20.0    # a context that hears nothing from Sonic for this long is given up (dead socket)
REUSE_S = 10.0              # a message for a finished utt_id older than this starts a new utterance (decoder restart)
FUNCTION_WORDS = {'a', 'an', 'the', 'to', 'of', 'and', 'or', 'but', 'in', 'on', 'at', 'for', 'with', 'my', 'your',
                  'his', 'her', 'our', 'their', 'is', 'are', 'was', 'were', 'i', 'you', 'we', 'they', 'it', 'that',
                  'this', 'very', 'so', 'if', 'than', 'from', 'by', 'as', 'be', 'have', 'has', 'had', 'will',
                  'would', 'can'}
QUESTION_STARTS = {'what', 'who', 'where', 'when', 'why', 'how', 'which', 'do', 'does', 'did', 'is', 'are', 'can',
                   'could', 'would', 'will', 'should', 'have', 'has'}


def speakable(words, first):
    """Lip-reader words (UPPERCASE, no punctuation) -> text Sonic reads naturally. All caps can be read as letters,
    so lowercase, then capitalise the sentence start and 'I' / "I'm"-style words."""
    out = []
    for i, w in enumerate(words):
        w = w.lower()
        if w == 'i' or w.startswith("i'"):
            w = 'I' + w[1:]
        if first and i == 0:
            w = w[:1].upper() + w[1:]
        out.append(w)
    return ' '.join(out)


def end_mark(first_word):
    """Closing punctuation for a finished utterance (Sonic waits for it; without it audio starts late)."""
    return '?' if (first_word or '').lower() in QUESTION_STARTS else '.'


def generation_config(emotion=None, confidence=1.0, pace=None, effort=None):
    """Map the face label, pace (1.0 = normal, times SPEED_BIAS) and effort (1.0 = normal: soft below, emphatic above)
    onto Sonic's generation_config."""
    cfg = {'emotion': EMOTION.get(emotion, 'neutral') if confidence >= MIN_CONFIDENCE else 'neutral',
           'speed': round(min(1.5, max(0.6, SPEED_BIAS * (1.0 if pace is None else float(pace)))), 2)}
    if effort is not None:
        cfg['volume'] = round(min(2.0, max(0.5, float(effort))), 2)
    return cfg


def ready(pending, spoken, phrase_words=1):
    """How many of the pending words to send now (0 = wait for more). phrase_words=1, the repo plan: every word as soon
    as it arrives. phrase_words>1, the opt-in experiment: 2 words first, then phrase_words, never ending on a function
    word, at most MAX_WORDS."""
    n = len(pending)
    if phrase_words <= 1:
        return n
    if n >= MAX_WORDS:
        return MAX_WORDS
    need = FIRST_WORDS if not spoken else phrase_words
    if n >= need and pending[-1].lower() not in FUNCTION_WORDS:
        return n
    return 0


class Utterance:
    """One utt_id: words taken (never spoken twice), words waiting to form a phrase, its Sonic context, its audio."""

    def __init__(self, utt_id):
        self.utt_id, self.sent, self.final = utt_id, [], False
        self.pending, self.spoken = [], 0
        self.gen, self.timer, self.seq = None, None, 0
        self.ctx, self.reader = None, None
        self.audio = asyncio.Queue()        # bytes, then None when the utterance's audio is complete
        self.queued = self.ended = False
        self.lost = []                      # words streaming couldn't take: spoken whole at the end (fallback)
        self.t_first_text = self.t_first_audio = self.t_final = None

    def new_words(self, words, offset=None):
        words = [w for w in words if w]
        if offset is not None:              # explicit position from the decoder: no guessing
            new = words[max(0, len(self.sent) - int(offset)):]
        elif len(words) >= len(self.sent) and words[:len(self.sent)] == self.sent:
            new = words[len(self.sent):]    # cumulative message: the prefix we already took
        else:
            new = words                     # delta message
        self.sent += new
        return new

    def end_audio(self):
        if not self.ended:
            self.ended = True
            self.audio.put_nowait(None)


class Voice:
    """Owns the Sonic websocket. say() is called per voice.say message; audio goes to `sink(bytes)` in order."""

    def __init__(self, cfg, sink, on_timing=None, client=None, on_done=None):
        self.cfg, self.sink, self.on_timing, self.on_done = cfg, sink, on_timing, on_done
        self.client = client
        self.conn = self._cm = None
        self.utts = {}              # utt_id -> Utterance
        self.order = asyncio.Queue()    # utterances in the order their audio started
        self.emotion = ('neutral', 1.0)
        self._lock = asyncio.Lock()
        self._tasks = set()
        self._player = None

    # --- connection ---------------------------------------------------------------------------------------------
    async def start(self):
        if self.client is None:
            from cartesia import AsyncCartesia
            self.client = AsyncCartesia(api_key=self.cfg.cartesia_key)
        await self._connect()
        self._player = self._spawn(self._play())

    async def _connect(self):
        self._cm = self.client.tts.websocket_connect()
        self.conn = await self._cm.__aenter__()
        log.info('Sonic websocket open (model %s)', self.cfg.tts_model)

    def _alive(self):
        try:
            from websockets.protocol import State
            return self.conn._connection.state not in (State.CLOSED, State.CLOSING)   # SDK internals, best effort
        except Exception:  # noqa: BLE001  (fakes / SDK changes: assume alive, a failed push still reconnects)
            return True

    async def _reconnect(self):
        log.warning('Sonic websocket lost: opening a new one')
        try:
            await self.conn.close()
        except Exception:  # noqa: BLE001
            pass
        await self._connect()
        for u in self.utts.values():        # contexts on the old socket are gone
            if u.ctx is not None:
                if u.reader:
                    u.reader.cancel()
                u.ctx = u.reader = None
                if u.final:                 # nothing more will be pushed: end its audio so the player moves on
                    log.warning('utterance %s lost its remaining audio in the reconnect', u.utt_id)
                    u.end_audio()

    def _spawn(self, coro):
        t = asyncio.create_task(coro)
        self._tasks.add(t)

        def done(task):
            self._tasks.discard(task)
            if not task.cancelled() and task.exception():
                log.error('voice task failed', exc_info=task.exception())
        t.add_done_callback(done)
        return t

    def set_emotion(self, label, confidence):
        self.emotion = (label, float(confidence))

    # --- words in -------------------------------------------------------------------------------------------------
    async def say(self, utt_id, words, final=False, pace=None, effort=None, offset=None):
        async with self._lock:      # messages stay in order
            u = self.utts.get(utt_id)
            if u is not None and u.final:
                if time.perf_counter() - u.t_final < REUSE_S:
                    log.debug('late message for finished utterance %s ignored', utt_id)
                    return
                log.warning('utt_id %s reused after it finished (decoder restart?): new utterance', utt_id)
                u = None
            if u is None:
                u = self.utts[utt_id] = Utterance(utt_id)
                self._prune()
            u.seq += 1
            if u.t_first_text is None and words:
                u.t_first_text = time.perf_counter()
            u.pending += u.new_words(words, offset)
            u.gen = generation_config(*self.emotion, pace=pace, effort=effort)
            if final:
                await self._finish(u)
                return
            n = ready(u.pending, u.spoken, self.cfg.phrase_words)
            if n:
                await self._send(u, n)
            self._arm(u)

    async def _finish(self, u):
        u.final, u.t_final = True, time.perf_counter()
        if u.timer:
            u.timer.cancel()
        try:
            await self._send(u, len(u.pending), end=True)
        finally:
            if u.ctx is not None:
                try:
                    await u.ctx.no_more_inputs()
                except Exception:  # noqa: BLE001
                    log.exception('closing utterance %s failed', u.utt_id)
                    u.end_audio()
            else:                   # nothing in flight for it (empty, or lost in a reconnect)
                await self._speak_lost(u)
                u.end_audio()

    async def _send(self, u, n, end=False):
        """Send the first n pending words as one chunk; they leave `pending` only once Sonic accepted them. Sonic joins
        continued chunks exactly as written: a leading space after the first chunk, closing punctuation on the last."""
        words = u.pending[:n]
        text = (('' if not u.spoken else ' ') + speakable(words, not u.spoken)) if words else ''
        if end and (u.spoken or words):
            text += end_mark(u.sent[0])
        if not text:
            return
        for attempt in (1, 2):
            try:
                if not self._alive():
                    await self._reconnect()
                ctx = u.ctx or self._open(u)
                await ctx.push(transcript=text, generation_config=u.gen)
                break
            except Exception:  # noqa: BLE001
                if attempt == 2:
                    log.exception('Sonic streaming failed twice: "%s" will be spoken whole at the end', text)
                    u.lost += words
                    break
                try:
                    await self._reconnect()
                except Exception:  # noqa: BLE001  (no network at all: the fallback will try at the end)
                    log.exception('Sonic reconnect failed')
        u.pending = u.pending[n:]
        u.spoken += len(words)

    def _arm(self, u):
        """(Re)start the pause timer: if no new words arrive for PAUSE_S, speak what is waiting."""
        if u.timer:
            u.timer.cancel()
        if u.pending:
            seq = u.seq
            u.timer = asyncio.get_running_loop().call_later(PAUSE_S, lambda: self._spawn(self._on_pause(u, seq)))

    async def _on_pause(self, u, seq):
        async with self._lock:
            if seq == u.seq and not u.final and u.pending:  # a newer message re-armed or closed it: do nothing
                await self._send(u, len(u.pending))

    # --- audio out ------------------------------------------------------------------------------------------------
    def _open(self, u):
        if u.ended:                         # its earlier context timed out: start a fresh audio run for the rest
            u.audio, u.ended, u.queued = asyncio.Queue(), False, False
        u.ctx = self.conn.context(f'utt-{u.utt_id}-{int(time.perf_counter() * 1000)}', timeout=CONTEXT_TIMEOUT_S,
                                  model_id=self.cfg.tts_model, voice={'id': self.cfg.voice_id}, output_format=OUTPUT,
                                  generation_config=u.gen, max_buffer_delay_ms=self.cfg.max_buffer_delay_ms)
        u.reader = self._spawn(self._read(u, u.ctx))
        if not u.queued:
            u.queued = True
            self.order.put_nowait(u)
        return u.ctx

    async def _read(self, u, ctx):
        """One context's events -> the utterance's audio queue. Ends the queue on done, error or timeout; a cancel
        (reconnect) leaves it open for the replacement context."""
        try:
            async for ev in ctx.receive():
                kind = getattr(ev, 'type', None)
                if kind == 'chunk' and ev.data:
                    if u.t_first_audio is None:
                        u.t_first_audio = time.perf_counter()
                        if self.on_timing:
                            await self.on_timing({'stage': 'sonic_first_audio', 'utt_id': u.utt_id,
                                                  'ms': (u.t_first_audio - u.t_first_text) * 1000})
                    u.audio.put_nowait(ev.audio)
                elif kind == 'error':
                    log.error('Sonic error on utt %s: %s', u.utt_id, getattr(ev, 'error', ev))
                    break
                if getattr(ev, 'done', False):
                    break
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001  timeout or a dead socket
            log.exception('Sonic reader for utt %s stopped', u.utt_id)
        if u.ctx is ctx:
            u.ctx = u.reader = None
        if u.final:
            await self._speak_lost(u)
        u.end_audio()

    async def _speak_lost(self, u):
        """The fallback: the words streaming couldn't take, as one whole-sentence request, queued after the rest."""
        if not u.lost:
            return
        words, u.lost = u.lost, []
        text = speakable(words, words[0] == u.sent[0]) + end_mark(u.sent[0])
        try:
            resp = await self.client.tts.generate(model_id=self.cfg.tts_model, transcript=text,
                                                  voice={'id': self.cfg.voice_id}, output_format=OUTPUT,
                                                  generation_config=u.gen or generation_config())
            audio = await resp.read()
        except Exception:  # noqa: BLE001
            log.exception('fallback voice failed too: "%s" not spoken', text)
            return
        if not u.queued:
            u.queued = True
            self.order.put_nowait(u)
        u.audio.put_nowait(audio)
        log.warning('spoke "%s" with the whole-sentence fallback', text)

    async def _play(self):
        """Utterances one after another, each to its end, so audio never interleaves."""
        while True:
            u = await self.order.get()
            heard = False
            while (b := await u.audio.get()) is not None:
                await self.sink(b)
                heard = True
            if heard and self.on_done:
                await self.on_done(u.utt_id, u.t_first_text)

    def _prune(self):
        """Forget utterances whose audio finished a while ago."""
        now = time.perf_counter()
        for k in [k for k, u in self.utts.items() if u.final and u.ended and now - u.t_final > 3 * REUSE_S]:
            del self.utts[k]

    async def close(self):
        for u in self.utts.values():
            if u.timer:
                u.timer.cancel()
        for t in list(self._tasks):
            t.cancel()
        if self._cm is not None:
            try:
                await self.conn.close()
            except Exception:  # noqa: BLE001
                pass
