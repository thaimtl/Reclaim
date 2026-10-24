"""Type to speak: publish voice.say like Austin's decoder would, for tests and as the demo's typed fallback.

    venv-call/bin/python -m call.say "I got the pills this morning"      # one sentence, then exit
    venv-call/bin/python -m call.say                                       # interactive: one line = one sentence
Binds RECLAIM_DEC_PUB (default tcp://127.0.0.1:5557), so don't run it while the real decoder is publishing there.
"""
import argparse
import asyncio
import time

from call import bus
from call.config import Config


async def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('text', nargs='*')
    ap.add_argument('--emotion', help='also publish a cam.tick-style emotion first (needs RECLAIM_CAM_PUB free)')
    args = ap.parse_args()
    cfg = Config()
    pub = bus.Publisher(cfg.dec_pub)
    await asyncio.sleep(0.6)            # let the call server's SUB socket (re)connect before the first message
    utt = int(time.time()) % 100000

    async def say(line):
        nonlocal utt
        words = line.split()
        if words:
            utt += 1
            await pub.send(bus.VOICE_SAY, {'utt_id': utt, 'words': [w.lower() for w in words], 'final': True,
                                           'pace': 1.0, 'effort': 1.0})   # as the decoder sends: lowercase, 1.0 = normal
    if args.text:
        await say(' '.join(args.text))
        await asyncio.sleep(0.3)
    else:
        loop = asyncio.get_running_loop()
        while True:
            line = await loop.run_in_executor(None, input, 'say> ')
            await say(line)
    pub.close()


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, EOFError):
        pass
