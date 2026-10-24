"""Test bot: Claude plays Austin on a real call, so the voice clone and the emotion tags can be tried without the decoder.

    venv-call/bin/python -m call.chat_austin            # with call.server running; start this first, then the call

Hears the other person through partner.text (Cartesia Ink), asks Claude for a short reply, and speaks it sentence by
sentence: a cam.tick with the sentence's emotion (as the camera's face label would), then voice.say with its words.
Binds RECLAIM_DEC_PUB and RECLAIM_CAM_PUB, so don't run it while the real decoder or camera reader is publishing.
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
Your team is building Reclaim: silent speech for people who lost their voice box. A camera reads your lips and EMG pads
read your face muscles, and your cloned voice speaks the words on a real phone call. You are tired but excited.

This is a voice call: reply in ONE short spoken sentence (two only if asked a question that needs it), under 15 words,
casual, no lists, no emoji, no stage directions. Never apologize for the line or repeat your greeting.
Start EVERY sentence with one emotion tag, the tone it should be spoken in: <neutral>, <happy>, <sad>, <angry> or
<surprised>. Change the tag when the feeling changes, e.g. "<surprised> Wait, really? <happy> That's awesome."
If the other person asks you to show an emotion, use that tag."""
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
                await self.dec.send(bus.VOICE_SAY, {'utt_id': self.utt, 'words': line.split(), 'final': True,
                                                    'pace': 1.0, 'effort': 0.5})

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
