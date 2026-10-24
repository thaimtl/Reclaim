"""The Claude service on the bus: reads partner.text, dec.lock and dec.hyp, publishes claude.predict,
claude.repair and timing.claude (with every request and reply, for the logger).

One conversation-lane call at a time, so the conversation stays append-only (whatever arrives meanwhile goes into
the next one), and a repair-lane call for each dec.lock at once, up to brain.MAX_REPAIRS in flight (brain.py).
While Austin mouths and nothing changed, the last claude.predict goes out again with a fresh ttl_s, without a call.
The bus loop never waits on Claude, and nothing that goes wrong (a malformed
message, a failed call, a reply that breaks apply) stops it: the error goes out on timing.claude and the
captions go on without Claude.

    ANTHROPIC_API_KEY=... python -m claude_service.service      # connects to 'dec' and 'call', binds 'claude'
"""
import argparse
import signal
import sys
import threading

import bus
from bus.team import Pub, Sub
from claude_service.brain import MAX_REPAIRS, Brain

TOPICS = ("partner.text", "dec.lock", "dec.hyp")
MOUTHING_S = 0.5  # dec.hyp showed unlocked words this recently


def report(pub, error):
    pub.send("timing.claude", {"stage": "claude", "error": error})
    print(f"claude: {error}", file=sys.stderr)


class Call:
    """brain.ask(messages) on a daemon thread, so quitting never waits for an unanswered call."""

    def __init__(self, brain, target, messages=None):
        self.target, self.result, self.error = target, None, None
        self.thread = threading.Thread(target=self._work, args=(brain, messages), daemon=True)
        self.thread.start()

    def _work(self, brain, messages):
        try:
            self.result = brain.ask(messages)
        except Exception as e:  # handed to the bus loop, which reports it
            self.error = e

    def done(self):
        return not self.thread.is_alive()

    def get(self):
        if self.error is not None:
            raise self.error
        return self.result


def finish(brain, call, pub, now):
    try:
        reply, log = call.get()
        if reply is not None:
            out, rejected = brain.apply(reply, call.target, now)
            log["rejected_edits"] = rejected
            for topic, msg in out:
                pub.send(topic, msg)
        pub.send("timing.claude", log)
        if "error" in log:
            print(f"claude: {log['error']}", file=sys.stderr)
    except Exception as e:
        report(pub, f"reply failed: {type(e).__name__}: {e}")


def run(brain, sub, pub, stop=lambda: False):
    """sub: a bus.Sub on TOPICS; pub: a bus.Pub. Runs until stop() is true."""
    convo, repairs, last_mouth = None, [], -1e9  # one conversation call, up to MAX_REPAIRS repair calls
    while not stop():
        timeout = 20
        while True:  # wait up to 20 ms for the first message, then drain the rest
            try:
                topic, msg = sub.recv(timeout_ms=timeout)
                if topic is None:
                    break
                if topic == "partner.text":
                    brain.partner(msg)
                elif topic == "dec.lock":
                    if bad := brain.locked(msg):
                        report(pub, f"bad message: {bad} malformed word(s) in dec.lock {msg['utt_id']}")
                elif topic == "dec.hyp":
                    brain.hyp(msg)
                    if msg.get("unlocked"):
                        last_mouth = bus.now()
            except Exception as e:  # one bad message must not stop the service
                report(pub, f"bad message: {type(e).__name__}: {e}")
            timeout = 0
        now = bus.now()
        mouthing = now - last_mouth < MOUTHING_S
        for call in [c for c in repairs if c.done()]:
            finish(brain, call, pub, now)
            repairs.remove(call)
        if convo is not None and convo.done():
            finish(brain, convo, pub, now)
            convo = None
        while brain.review and len(repairs) < MAX_REPAIRS:
            try:
                messages, target = brain.next_review(now)
                repairs.append(Call(brain, target, messages))
            except Exception as e:
                report(pub, f"repair request failed: {type(e).__name__}: {e}")
                brain.review = {}  # drop what broke it, or it fails again every loop
        if convo is None and brain.due(now, mouthing):
            try:
                convo = Call(brain, brain.next_request(now))
            except Exception as e:
                report(pub, f"request failed: {type(e).__name__}: {e}")
                brain.events, brain.updated, brain.notes = [], [], []  # drop what broke it
        if (again := brain.refresh(now, mouthing)) is not None:
            pub.send("claude.predict", again)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", action="append", default=[],
                    help="endpoint name or address to connect to, repeatable (default: dec and call)")
    ap.add_argument("--pub", default="claude", help="endpoint name or address to bind for our topics")
    args = ap.parse_args()
    import anthropic  # here, so the tests run without the SDK
    client = anthropic.Anthropic(timeout=10.0, max_retries=1)  # key from ANTHROPIC_API_KEY, never from the repo
    signal.signal(signal.SIGTERM, signal.default_int_handler)  # stop the same way as on Ctrl-C
    sub, pub = Sub(args.sub or ["dec", "call"], TOPICS), Pub(args.pub)
    try:
        run(Brain(client), sub, pub)
    except KeyboardInterrupt:
        pass
    finally:
        sub.close()
        pub.close()  # the default linger lets the last timing.claude out


if __name__ == "__main__":
    main()
