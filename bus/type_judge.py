"""Stand-in for the judge on the phone: type the judge's lines and they go out as partner.text, as Cartesia Ink's would.

    venv-svr/bin/python -m bus.type_judge                 # type a line, Enter sends it; Ctrl-D or Ctrl-C ends the call
    venv-svr/bin/python -m bus.type_judge --script judge.txt --gap 8   # one line every 8 s from a file

For testing the pipeline with a conversation before the Twilio call is wired in: the Claude service predicts from
these lines, the fusion process adds them to its language model's call cache, and the dashboard shows them.
Binds the 'call' endpoint (bus/team.py), so don't run it while call.server is running. Sends call.state 'connected'
at the start and 'ended' at the end, as the call does; with --stream, each line first goes out word by word as
partial partner.text (is_final false, ~3 words per second), the way Ink streams, then once more as final.
"""
import argparse
import sys
import time

import bus
from bus.team import Pub


def send_line(pub, text, stream):
    words = text.split()
    if stream:
        for k in range(1, len(words)):
            pub.send('partner.text', {'text': ' '.join(words[:k]), 'is_final': False, 't': bus.now()})
            time.sleep(0.33)
    pub.send('partner.text', {'text': text, 'is_final': True, 't': bus.now()})


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--script', default=None, help='a text file: one judge line per line, sent every --gap seconds')
    ap.add_argument('--gap', type=float, default=8.0, help='--script: seconds between lines')
    ap.add_argument('--stream', action='store_true', help='send each line word by word first, as Ink does')
    ap.add_argument('--pub', default='call')
    args = ap.parse_args()
    pub = Pub(args.pub)
    time.sleep(0.5)                     # let subscribers connect before the first message
    pub.send('call.state', {'state': 'connected', 't': bus.now(), 'call_sid': 'typed-judge', 'ink': 'typed'})
    try:
        if args.script:
            with open(args.script, encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
            for k, line in enumerate(lines):
                print(f'judge: {line}', flush=True)
                send_line(pub, line, args.stream)
                if k < len(lines) - 1:
                    time.sleep(args.gap)
        else:
            print('type the judge\'s lines; Enter sends, Ctrl-D ends the call', file=sys.stderr)
            for line in sys.stdin:
                if line.strip():
                    send_line(pub, line.strip(), args.stream)
    except KeyboardInterrupt:
        pass
    finally:
        pub.send('call.state', {'state': 'ended', 't': bus.now()})
        pub.close()


if __name__ == '__main__':
    main()
