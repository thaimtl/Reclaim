"""Print the call service's bus messages (partner.text, timing.voice, call.state).

    venv-call/bin/python -m call.listen
"""
import asyncio

from call import bus
from call.config import Config


async def main():
    sub = bus.Subscriber([Config().call_pub], [''])
    while True:
        topic, msg = await sub.recv()
        body = {k: v for k, v in msg.items() if k not in ('v', 'seq', 't_pub')}
        print(f'{topic:13s} {body}', flush=True)


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
