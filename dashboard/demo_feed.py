"""A scripted call, for rehearsing the dashboard and for its tests. It walks through every state in Thai's Figma
v5: reading, a Claude repair, Please repeat, the face hidden, the muscle sensors off, and the end of the call.

    venv-live/bin/python -m dashboard.demo_feed [--speed 1.0] [--video clip.mp4] [--loop]      # onto the bus
    venv-live/bin/python -m dashboard.demo_feed --fixtures DIR      # snapshots at each state, no bus (for tests)

The call is one list of events (events()): bus messages, pauses, changes to the background producers (emotion, face,
muscle sensors), and named checkpoints. On the bus, the messages go out with real pauses and a background thread
publishes cam.tick and cam.face (10 Hz) and emg.raw (100 Hz), plus cam.preview from --video (the lip reader's camera
mirror). simulate() replays the same events into CallState on a simulated clock, so a test sees exactly the state
at each checkpoint.

On the bus it binds the producers' endpoints (bus/team.py), so run it only with those parts stopped, and on private
ports (RECLAIM_*_PUB) whenever anyone may be logging the real ones. Everything it sends is sample data.
"""
import argparse
import json
import math
import os
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FACE = json.load(open(os.path.join(HERE, 'demo_face.json')))

# (judge line, Austin's words, who settled each word, emotion, extras): repair (index, from, to), repeat
# (index, word, alternatives), face hidden, muscles off, and the candidates for one word (index, rows).
SCRIPT = [
    dict(judge='So what would you order at a coffee shop?', words="I'd love a coffee with oat milk.".split(),
         src=['camera', 'camera', 'emg', 'camera', 'camera', 'emg', 'camera'], emotion='happy',
         cands=(3, [('coffee', .36, .14, .12), ('copy', .08, 0, .03), ('toffee', .05, .02, 0)])),
    dict(judge='Nice. Did you fly in for this?', words='Yes, but they lost my pack.'.split(),
         src=['camera', 'emg', 'camera', 'emg', 'camera', 'claude'], emotion='sad', repair=(5, 'pack.', 'bag.')),
    dict(judge='Oh no! Did they find it?', words='Yes, they sent'.split(), src=['camera', 'emg', 'camera'],
         emotion='happy', repeat=(2, 'sent', ['meant', 'bent']),
         cands=(2, [('sent', .22, .12, .07), ('meant', .14, .17, .07), ('bent', .03, .01, .02)])),
    dict(judge='Want it hot or iced?', words='Iced, please.'.split(), src=['emg', 'emg'], emotion='happy',
         face_hidden=True),
    dict(judge='Anything else?', words="That's all, thanks.".split(), src=['camera', 'camera', 'camera'],
         emotion='neutral', muscles_off=True),
]


class Now:
    """A time in a scripted message: bus.now() (plus an offset) when it is sent, so a script is clock-free."""

    def __init__(self, offset=0.0):
        self.offset = offset


def resolve(v, now):
    if isinstance(v, Now):
        return now + v.offset
    if isinstance(v, dict):
        return {k: resolve(x, now) for k, x in v.items()}
    if isinstance(v, list):
        return [resolve(x, now) for x in v]
    return v


def events(script=SCRIPT):
    """The whole call as ('wait', s) / ('set', {background state}) / ('msg', endpoint, topic, msg) /
    ('checkpoint', name)."""
    yield ('set', dict(emotion='neutral', face_ok=True, emg_on=True, mouthing=False))
    yield ('wait', 1.0)
    yield ('checkpoint', 'waiting')
    yield ('msg', 'call', 'call.state', {'state': 'connected', 't': Now(), 'call_sid': 'CA-sample', 'ink': True})
    for utt, turn in enumerate(script, 1):
        words = turn['judge'].split()
        for n in range(1, len(words)):
            yield ('msg', 'call', 'partner.text', {'text': ' '.join(words[:n]), 'is_final': False, 't': Now()})
            yield ('wait', 0.22)
        yield ('msg', 'call', 'partner.text', {'text': turn['judge'], 'is_final': True, 't': Now()})
        yield ('wait', 1.0)
        yield from austin(utt, turn)
        yield ('set', dict(face_ok=True, emg_on=True))
    yield ('msg', 'call', 'call.state', {'state': 'ended', 't': Now()})
    yield ('wait', 1.0)
    yield ('checkpoint', 'end')


def austin(utt, turn):
    words, src = turn['words'], turn['src']
    yield ('set', dict(emotion=turn['emotion'], face_ok=not turn.get('face_hidden'),
                       emg_on=not turn.get('muscles_off'), mouthing=True))
    yield ('wait', 1.2)                                   # the background producers catch up (face, sensors)
    repeat, repair, cands = turn.get('repeat'), turn.get('repair'), turn.get('cands')
    locked, spoken = [], []
    for i, w in enumerate(words):
        conf = [0.92 - 0.04 * (j % 3) for j in range(len(locked))]
        hyp = {'utt_id': utt, 'locked': locked, 'unlocked': [w.strip('.,').lower() if i else w], 'confidence': conf + [0.6],
               'source': src[:len(locked)]}
        if cands and cands[0] == i:
            off = turn.get('muscles_off')
            hyp['cands'] = [{'w': c, 'cam': a, 'emg': 0 if off else b, 'claude': cl} for c, a, b, cl in cands[1]]
            hyp['threshold'] = 0.7
        yield ('msg', 'dec', 'dec.hyp', hyp)
        yield ('wait', 0.55)
        if i == 3 and utt == 1:
            yield ('checkpoint', 'reading')
        if turn.get('muscles_off') and i == 1:
            yield ('checkpoint', 'muscles_off')
        if repeat and repeat[0] == i:
            yield ('msg', 'dec', 'dec.repeat', {'utt_id': utt, 'reason': {'reader': 'camera', 'words': [
                {'index': i, 'w': repeat[1], 'alts': repeat[2]}]}})
            yield ('wait', 0.3)
            yield ('checkpoint', 'please_repeat')
            break
        locked = locked + [w]
        truth = w.strip('.,').lower()
        yield ('msg', 'dec', 'dec.lock', {'utt_id': utt, 'words': [{
            'w': w, 'conf': 0.8, 'alts': [], 'cam': 0.7, 'emg': 0.6, 't': Now(-0.3),
            'cam_top': truth if (utt + 2 * i) % 7 else 'copy', 'emg_top': truth if (utt + i) % 4 else 'toffee'}]})
        yield ('msg', 'dec', 'timing.dec', {'stage': 'lock', 'ms': 95.0 + 10 * (i % 3), 't': Now()})
        locked_hyp = {'utt_id': utt, 'locked': locked, 'unlocked': [], 'source': src[:len(locked)],
                      'confidence': [0.92 - 0.04 * (j % 3) for j in range(len(locked))]}
        if 'cands' in hyp:
            locked_hyp.update(cands=hyp['cands'], threshold=hyp['threshold'])
        yield ('msg', 'dec', 'dec.hyp', locked_hyp)
        if repair and repair[0] == i:
            yield ('msg', 'claude', 'timing.claude', {'stage': 'claude', 'ms': 310.0, 't': Now()})
            yield ('msg', 'claude', 'claude.repair', {'utt_id': utt, 'edits': [
                {'index': i, 'from': repair[1], 'to': repair[2]}], 'please_repeat': False, 't': Now()})
            w = repair[2]
        yield ('msg', 'dec', 'timing.dec', {'stage': 'window', 'ms': 300.0, 't': Now()})
        yield ('msg', 'dec', 'voice.say', {'utt_id': utt, 'words': [w], 'offset': len(spoken), 'final': False,
                                            'pace': 1.0, 'effort': 1.0})
        if not spoken:
            yield ('msg', 'call', 'timing.voice', {'stage': 'sonic_first_audio', 'utt_id': utt, 'ms': 190.0})
        spoken.append(w)
        yield ('wait', 0.2)
        if repair and repair[0] == i:
            yield ('checkpoint', 'claude_repair')
        if turn.get('face_hidden') and i == 1:
            yield ('checkpoint', 'face_hidden')
    yield ('msg', 'dec', 'voice.say', {'utt_id': utt, 'words': [], 'offset': len(spoken), 'final': True,
                                        'pace': 1.0, 'effort': 1.0})
    yield ('set', dict(mouthing=False))
    yield ('wait', 0.8)
    if spoken:          # Twilio's mark: the judge heard the sentence (call/server.py)
        yield ('msg', 'call', 'timing.voice', {'stage': 'played_to_judge', 'utt_id': utt, 'ms': 1100.0})
    yield ('wait', 0.8)


def background_messages(bg, t, k, rng):
    """What the always-on producers send at step k (10 ms): cam.tick + cam.face every 10th step, emg.raw every step."""
    out = []
    if k % 10 == 0:
        out.append(('cam', 'cam.tick', {'seg_id': 1, 'seg_start': t, 'frame_t': t - np.arange(3)[::-1] / 25.0,
                                        'face_ok': bg['face_ok'], 'timing_ms': {'T': 3, 'total': 140.0 + 15 * math.sin(t)},
                                        'emotion': {'label': bg['emotion'], 'confidence': 0.9, 'scores': {}}}))
        out.append(('cam', 'cam.face', {'t': t, 'w': FACE['w'], 'h': FACE['h'], 'face_ok': bg['face_ok'],
                                        'pts': np.array(FACE['pts'], np.float32), 'blend': None, 'pose': None}))
    if bg['emg_on']:
        amp = np.array([6, 9, 4, 11]) * (1 + 5 * bg['mouthing'] * (0.5 + 0.5 * math.sin(t * 9)))
        s = 512 + rng.normal(0, 1, (10, 4)) * amp
        out.append(('emg', 'emg.raw', {'t0': t - 0.01, 'fs': 1000, 'samples': s.astype(np.uint16), 'counter': 10 * k}))
    return out


def simulate(script=SCRIPT, number='(628) 888-6598'):
    """{checkpoint: snapshot}: the events fed straight into CallState on a simulated clock (no sockets, no sleeps),
    the background producers stepped every 10 ms, just as they run on the bus."""
    from dashboard.state import CallState
    state, t, k, rng = CallState(number), 1000.0, 0, np.random.default_rng(0)
    state.logging(t)
    bg, shots = {}, {}
    for ev in events(script):
        if ev[0] == 'set':
            bg.update(ev[1])
        elif ev[0] == 'msg':
            _, _, topic, msg = ev
            state.feed(topic, dict(resolve(msg, t), t_pub=t), t)
        elif ev[0] == 'wait':
            for _ in range(round(ev[1] / 0.01)):
                t += 0.01
                k += 1
                for _, topic, msg in background_messages(bg, t, k, rng):
                    state.feed(topic, msg, t)
                state.logging(t)
        elif ev[0] == 'checkpoint':
            shots[ev[1]] = json.loads(json.dumps(state.snapshot(t)))
    return shots


class Feed:
    """The events onto the bus, with real pauses and the background producers on their own thread."""

    def __init__(self, speed=1.0, video=None):
        from bus.team import Pub
        self.speed, self.video = speed, video
        self.pubs = {name: Pub(name) for name in ('cam', 'dec', 'claude', 'call', 'emg')}
        self.bg = dict(emotion='neutral', face_ok=True, emg_on=True, mouthing=False)
        self.stop = threading.Event()

    def background(self):
        """One thread owns the cam and emg sockets (ZeroMQ sockets are not thread-safe)."""
        import bus
        k, rng, cap = 0, np.random.default_rng(0), None
        if self.video:
            import cv2
            cap = cv2.VideoCapture(self.video)
        while not self.stop.is_set():
            for endpoint, topic, msg in background_messages(self.bg, bus.now(), k, rng):
                self.pubs[endpoint].send(topic, msg)
            if cap is not None and k % 10 == 0:
                self.preview(cap)
            k += 1
            time.sleep(0.01)

    def preview(self, cap):
        import bus
        import cv2
        ok, img = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, img = cap.read()
            if not ok:
                return
        h, w = img.shape[:2]
        img = cv2.resize(img, (640, round(h * 640 / w)), interpolation=cv2.INTER_AREA)
        ok, jpeg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            self.pubs['cam'].send('cam.preview', {'t': bus.now(), 'w': img.shape[1], 'h': img.shape[0],
                                                  'jpeg': jpeg.tobytes()})

    def run(self, loop=False):
        import bus
        threading.Thread(target=self.background, daemon=True).start()
        try:
            while True:
                for ev in events():
                    if ev[0] == 'set':
                        self.bg.update(ev[1])
                    elif ev[0] == 'msg':
                        _, endpoint, topic, msg = ev
                        self.pubs[endpoint].send(topic, resolve(msg, bus.now()))
                    elif ev[0] == 'wait':
                        time.sleep(ev[1] / self.speed)
                time.sleep(8.0 / self.speed)
                if not loop:
                    break
        finally:
            self.stop.set()
            time.sleep(0.05)
            for p in self.pubs.values():
                p.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--speed', type=float, default=1.0)
    ap.add_argument('--video', help="a video file to mirror as cam.preview (e.g. one of Austin's recordings)")
    ap.add_argument('--loop', action='store_true')
    ap.add_argument('--fixtures', help='write one snapshot JSON per checkpoint into this folder, no bus')
    args = ap.parse_args()
    if args.fixtures:
        os.makedirs(args.fixtures, exist_ok=True)
        for name, snap in simulate().items():
            with open(os.path.join(args.fixtures, f'{name}.json'), 'w') as f:
                json.dump(snap, f)
        return
    Feed(args.speed, args.video).run(args.loop)


if __name__ == '__main__':
    main()
