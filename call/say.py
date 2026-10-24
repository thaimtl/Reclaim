"""Type to speak: publish voice.say like Austin's decoder would, for tests and as the demo's typed fallback.

    venv-call/bin/python -m call.say "I got the pills this morning"      # one sentence, then exit
    venv-call/bin/python -m call.say                                       # interactive: one line = one sentence
Binds RECLAIM_DEC_PUB (default tcp://127.0.0.1:5557), so don't run it while the real decoder is publishing there;
--emotion also binds RECLAIM_CAM_PUB (tcp://127.0.0.1:5556), so not while the camera reader runs.
"""
import argparse
import asyncio
import time

from call import bus
from call.config import Config


async def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('text', nargs='*')
    ap.add_argument('--emotion', choices=('neutral', 'happy', 'sad', 'angry', 'surprised'),
                    help='also publish this emotion as cam.tick does, before each line (needs RECLAIM_CAM_PUB free)')
    args = ap.parse_args()
    cfg = Config()
    pub = bus.Publisher(cfg.dec_pub)
    cam = bus.Publisher(cfg.cam_pub) if args.emotion else None
    await asyncio.sleep(0.6)            # let the call server's SUB socket (re)connect before the first message
    utt = int(time.time()) % 100000

    async def say(line):
        nonlocal utt
        words = line.split()
        if words:
            utt += 1
            if cam is not None:         # the camera reader's emotion, as cam.tick carries it
                await cam.send(bus.CAM_TICK, {'emotion': {'label': args.emotion, 'confidence': 1.0}})
                await asyncio.sleep(0.15)       # two sockets: the emotion lands before the words
            await pub.send(bus.VOICE_SAY, {'utt_id': utt, 'words': [w.upper() for w in words], 'final': True,
                                           'offset': 0, 'pace': 1.0, 'effort': 1.0})
    if args.text:
        await say(' '.join(args.text))
        await asyncio.sleep(0.3)
    else:
        loop = asyncio.get_running_loop()
        while True:
            line = await loop.run_in_executor(None, input, 'say> ')
            await say(line)
    pub.close()
    if cam is not None:
        cam.close()


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, EOFError):
        pass
