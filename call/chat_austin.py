"""Test bot: Claude plays Austin on a real call, so the voice clone and the emotion tags can be tried without the decoder.

    venv-call/bin/python -m call.chat_austin            # with call.server running; start this first, then the call

Hears the other person through partner.text (Cartesia Ink), asks Claude for a short reply, and "mouths" it the way
Austin's decoder will send it: a cam.tick with the sentence's emotion (as the camera's face label would), then word by
word at mouthing pace, each word locked LOCK_S after it ends: dec.hyp every 100 ms, dec.lock, and voice.say with only
the new word (bare, uppercase, as the lip reader gives it) and its offset; he pauses at commas. So the voice (which
ends its phrases on his pauses, from dec.hyp) and the dashboard see what they will see on the day. Binds
RECLAIM_DEC_PUB and RECLAIM_CAM_PUB, so don't run it while the real decoder or camera reader is publishing.
Needs ANTHROPIC_API_KEY in call/keys.env.
"""
import argparse
import asyncio
import os
import re
import time

from anthropic import AsyncAnthropic

from call import bus
from call.config import Config
from call.voice import EMOTION

MODEL = 'claude-haiku-5-5'      # light and fast: the reply has to start within a second or so
SYSTEM = """You are Austin, a college student at the Cal Hacks hackathon in San Francisco, talking on the phone.
Your team is building Silent Voice: silent speech for people who lost their voice box. A camera reads your lips and EMG pads
read your face muscles, and your cloned voice speaks the words on a real phone call. You are tired but excited.

This is a voice call: reply in ONE short spoken sentence (two only if asked a question that needs it), under 15 words,
casual, no lists, no emoji, no stage directions. Never apologize for the line or repeat your greeting.
Start EVERY sentence with one emotion tag, the tone it should be spoken in: <neutral>, <happy>, <sad>, <angry> or
<surprised>. Change the tag when the feeling changes, e.g. "<surprised> Wait, really? <happy> That's awesome."
If the other person asks you to show an emotion, use that tag."""
LOCK_S = 0.15       # the decoder locks a word this long after Austin finishes mouthing it
GAP_S = 0.06        # between two words he mouths
COMMA_PAUSE_S = 0.35    # he pauses at a comma
HYP_S = 0.1


def mouthing_s(word):
    """About how long Austin takes to mouth a word: longer words longer (0.2 s for "a", 0.5 s for "airport")."""
    return min(0.6, 0.14 + 0.045 * len(word.strip('.,!?')))


GREETING = '<happy> Hey, it\'s Austin! <neutral> Yeah, I can talk for a minute.'
TAG = re.compile(r'<(\w+)>')


def sentences(text):
    """'<happy> Hi! <sad> Oh no.' -> [('happy', 'Hi!'), ('sad', 'Oh no.')]; untagged text keeps the last emotion."""
    out, emotion = [], 'neutral'
    for part in re.split(r'(<\w+>)', text):
        m = TAG.fullmatch(part.strip())
        if m:
            emotion = m.group(1).lower() if m.group(1).lower() in EMOTION else 'neutral'
        elif part.strip():
            out += [(emotion, s) for s in re.split(r'(?<=[.!?])\s+', part.strip()) if s]
    return out


class Bot:
    def __init__(self, cfg, client):
        self.cfg, self.client = cfg, client
        self.dec = bus.Publisher(cfg.dec_pub)
        self.cam = bus.Publisher(cfg.cam_pub)
        self.sub = bus.Subscriber([cfg.call_pub], [bus.PARTNER_TEXT, bus.CALL_STATE])
        self.history = []
        self.utt = int(time.time()) % 100000
        self.lock = asyncio.Lock()
        self.task = None        # the reply being written; new speech from them cancels it and joins their turn

    async def speak(self, text):
        async with self.lock:
            for emotion, line in sentences(text):
                print(f'  austin [{emotion}]: {line}')
                await self.cam.send(bus.CAM_TICK, {'emotion': {'label': emotion, 'confidence': 1.0}})
                await asyncio.sleep(0.15)                   # two sockets: let the emotion land before the words
                self.utt += 1
                await self.mouth(self.utt, line.split())

    @staticmethod
    def bare(word):
        """A word the way the lip reader gives it: uppercase, no punctuation."""
        return word.strip('.,!?;:"').upper()

    async def mouth(self, utt, words):
        """One sentence the way the decoder sends it: he mouths the words back to back (GAP_S apart), each locks LOCK_S
        after he finishes it, while he is already on the next; dec.hyp every 100 ms, dec.lock and voice.say per word."""
        t, times = time.perf_counter(), []
        for w in words:
            times.append((t, t + mouthing_s(w)))
            t += mouthing_s(w) + (COMMA_PAUSE_S if w.endswith(',') else GAP_S)
        bare = [self.bare(w) for w in words]       # the lip reader's words: uppercase, no punctuation
        locked = 0
        while locked < len(words):
            now = time.perf_counter()
            while locked < len(words) and times[locked][1] + LOCK_S <= now:
                w, t_end = bare[locked], times[locked][1]
                await self.dec.send('dec.lock', {'utt_id': utt, 'words': [{'w': w, 'conf': 0.9, 'alts': [], 'cam': 0.9,
                                                                           'emg': 0.9, 't': t_end}]})
                await self.dec.send(bus.VOICE_SAY, {'utt_id': utt, 'words': [w], 'offset': locked,
                                                    'final': locked == len(words) - 1, 'pace': 1.0, 'effort': 1.0})
                locked += 1
            mouthing = [w for w, (a, _) in zip(bare[locked:], times[locked:]) if a <= now]
            await self.dec.send('dec.hyp', {'utt_id': utt, 'locked': bare[:locked], 'unlocked': mouthing,
                                            'source': ['camera'] * (locked + len(mouthing)),
                                            'confidence': [0.9] * locked + [0.5] * len(mouthing)})
            nxt = times[locked][1] + LOCK_S if locked < len(words) else now
            await asyncio.sleep(max(0.0, min(HYP_S, nxt - time.perf_counter())))

    async def reply(self, heard):
        if self.history and self.history[-1]['role'] == 'user':     # they kept talking before we answered
            self.history[-1]['content'] += ' ' + heard
        else:
            self.history.append({'role': 'user', 'content': heard})
        t0 = time.perf_counter()
        msg = await self.client.messages.create(model=MODEL, max_tokens=80, system=SYSTEM, messages=self.history,
                                                thinking={'type': 'disabled'})   # thinking ate the tokens: silence
        text = ''.join(b.text for b in msg.content if b.type == 'text').strip()
        self.history.append({'role': 'assistant', 'content': text})
        print(f'  (claude {1000 * (time.perf_counter() - t0):.0f} ms, stop={msg.stop_reason}): {text!r}')
        speaking = asyncio.create_task(self.speak(text))
        speaking.add_done_callback(
            lambda t: t.cancelled() or not t.exception() or print('  speak failed:', repr(t.exception())))
        await asyncio.shield(speaking)               # once it starts talking, new speech doesn't cut it off

    async def run(self):
        print('waiting for the call (start call.server, then dial)...')
        while True:
            topic, msg = await self.sub.recv()
            if topic == bus.CALL_STATE:
                print('call:', msg.get('state'))
                if msg.get('state') == 'connected':
                    self.history = [{'role': 'user', 'content': '(the call connects)'},
                                    {'role': 'assistant', 'content': GREETING}]
                    await asyncio.sleep(1.0)                # let the person say hello into the phone first
                    await self.speak(GREETING)
            elif topic == bus.PARTNER_TEXT and msg.get('is_final') and msg.get('text', '').strip():
                print('them:', msg['text'])
                if self.task and not self.task.done():
                    print('  (cancelling the pending reply: they kept talking)')
                    self.task.cancel()
                self.task = asyncio.create_task(self.reply(msg['text'].strip()))
                self.task.add_done_callback(
                    lambda t: t.cancelled() or not t.exception() or print('  reply failed:', repr(t.exception())))


async def main():
    argparse.ArgumentParser(description=__doc__.split('\n')[0]).parse_args()
    cfg = Config()
    if not os.environ.get('ANTHROPIC_API_KEY'):              # Config loaded call/keys.env into the environment
        raise SystemExit('ANTHROPIC_API_KEY is missing from call/keys.env')
    client = AsyncAnthropic()
    await Bot(cfg, client).run()


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
