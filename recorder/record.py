"""Reclaim recorder: webcam + 4-channel EMG + the session's sentences, saved for SVR-1 fine-tuning.

    python recorder/record.py --session 1                       # webcam 0, Uno found automatically
    python recorder/record.py --session 1 --camera 1 --port COM4
    python recorder/record.py --session 1 --camera fake --fake-emg --allow-charger   # no hardware, to try it

A window shows the camera, the sentence to mouth and how to mouth it, live EMG bars and warnings. Keys (click the
window first):
    SPACE  start the clip / stop the clip        R  redo the last clip       S  skip this prompt
    H      show or hide help                     Q or ESC  quit (progress is saved; rerun to resume)

Each clip is saved in svr/recordings.py's layout, under <root>/session_0N/ (root: $RECLAIM_WORKDIR/recordings):
    <id>.mp4             camera video, unmirrored          <id>_video_ts.npy  laptop-clock time of every frame
    <id>_emg.npy         raw Uno readings, (samples, 4),   <id>_emg_ts.npy    laptop-clock time of every sample
                         uint16, unfiltered, A0..A3
    manifest.jsonl       one JSON line per clip (a redo replaces its line), meta.json for the session
Camera and EMG share time.monotonic(): EMG sample times come from a fit of sample count against arrival time, so they
are exact to about a millisecond even though the Uno's clock drifts from the laptop's.
"""
import argparse
import datetime
import json
import os
import platform
import queue
import random
import sys
import textwrap
import threading
import time
from collections import deque

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, 'rig'))
from filters import EmgFilter  # noqa: E402  (rig/filters.py, Thai's live filter)

VERSION = 2  # 2: delivery styles (slow pace, facial expression, effort) on some sentences
FS = 1000
CHANNELS = 4
SITES = [3, 2, 6, 1]  # Gaddy and Klein sites on rig pins A0, A1, A2, A3
PROMPTS = os.path.join(REPO, 'docs', 'recording')
SESSIONS = {1: ('session1.txt', 'C'), 2: ('session2.txt', 'C'), 3: ('session3_test.txt', 'E')}
FAST_FROM = 201  # sentences 201+ of sessions 1 and 2 are mouthed faster than natural
# Delivery styles for the expressive voice (README: the voice follows pace, effort and facial expression). The demo has
# Austin smile or frown while he mouths, which changes the EMG (site 1 sits near the smile muscles), and speak slowly,
# softly or emphatically, which sets the voice's speed and volume; neutral natural-pace mouthing alone wouldn't cover
# that. Each style goes to a fixed, seeded set of natural-pace sentences (demo-pace in session 3), never two at once.
STYLES = {1: {'slow': 20, 'smile': 10, 'frown': 10, 'surprised': 10, 'soft': 10, 'emphatic': 10},
          2: {'slow': 20, 'smile': 10, 'frown': 10, 'surprised': 10, 'soft': 10, 'emphatic': 10},
          3: {'smile': 10, 'frown': 10}}
EXPRESSIONS = ('smile', 'frown', 'surprised')
EFFORTS = ('soft', 'emphatic')
STYLE_CUES = {  # what the window tells Austin, in the instruction line above the sentence
    'slow': 'SLOW: mouth it slowly and clearly, about half your natural speed',
    'smile': 'SMILE while you mouth it: a real smile, held the whole sentence',
    'frown': 'FROWN while you mouth it: look upset or annoyed, held the whole sentence',
    'surprised': 'SURPRISED while you mouth it: brows up, eyes wide, held the whole sentence',
    'soft': 'SOFT: small, relaxed movements, as if whispering',
    'emphatic': 'EMPHATIC: big, firm movements, as if speaking up to be heard',
}
REST_CUES = ['Still face, relaxed', 'Swallow once', 'Nod your head', 'Look left, then back', 'Look right, then back',
             'Smile', 'Raise your eyebrows', 'Take a slow breath', 'Lick your lips', 'Small cough']
ARDUINO_VIDS = {0x2341, 0x2A03, 0x1A86}
WIN = 'Reclaim recorder'


# ------------------------------------------------------------------------------------------------ prompts
def assign_styles(items, session):
    """Give STYLES[session] to a seeded sample of the natural- or demo-pace sentences, one style per sentence.
    Sets pace ('slow'), expression (neutral / smile / frown / surprised) and effort (normal / soft / emphatic)."""
    eligible = sorted(it['id'] for it in items if it['pace'] in ('natural', 'demo'))
    want = STYLES.get(session, {})
    if sum(want.values()) > len(eligible):
        raise ValueError(f'session {session}: {sum(want.values())} styled sentences, only {len(eligible)} eligible')
    picks = random.Random(f'reclaim-style-{session}').sample(eligible, sum(want.values()))
    style_of, i = {}, 0
    for style, n in want.items():
        for uid in picks[i:i + n]:
            style_of[uid] = style
        i += n
    for it in items:
        if it['pace'] == 'rest':
            it['expression'] = it['effort'] = None
            continue
        style = style_of.get(it['id'])
        if style == 'slow':
            it['pace'] = 'slow'
        it['expression'] = style if style in EXPRESSIONS else 'neutral'
        it['effort'] = style if style in EFFORTS else 'normal'
        it['style'] = style
    return items


def load_items(session):
    """The session's clips in a fixed shuffled order: sentences (natural, fast or slow pace, some with an expression or
    an effort to hold) and 10 rest clips."""
    fname, set_ = SESSIONS[session]
    items = []
    with open(os.path.join(PROMPTS, fname), encoding='utf-8') as f:
        for line in f:
            if line.strip():
                uid, text = line.rstrip('\n').split('\t', 1)
                num = int(uid.split('_')[1])
                pace = 'demo' if session == 3 else ('fast' if num >= FAST_FROM else 'natural')
                items.append({'id': uid, 'set': set_, 'prompt': text.strip(), 'pace': pace})
    for k, cue in enumerate(REST_CUES, 1):
        items.append({'id': f's{session}_rest_{k:02d}', 'set': 'B', 'prompt': '', 'pace': 'rest', 'cue': cue})
    assign_styles(items, session)
    random.Random(f'reclaim-session-{session}').shuffle(items)
    return items


# ------------------------------------------------------------------------------------------------ EMG
class EmgBuffer:
    """Raw samples with a global sample counter, the (count, arrival time) pairs that place them on the laptop clock,
    and a live filtered view for the level bars."""

    def __init__(self, keep_s=180):
        self.lock = threading.Lock()
        self.chunks = deque()  # (first sample index, uint16 array)
        self.n = 0
        self.anchors = deque()  # (sample count after a chunk, its arrival time)
        self.keep = keep_s * FS
        self.dropped = 0
        self.last_arrival = None
        self.live = EmgFilter()
        self.recent_f = np.zeros((0, CHANNELS))
        self.recent_raw = np.zeros((0, CHANNELS))

    def push(self, chunk, t_arrival, dropped=0):
        chunk = np.asarray(chunk, dtype=np.uint16)
        with self.lock:
            if dropped and self.chunks:  # keep the sample clock right: fill the gap with the last reading
                fill = np.repeat(self.chunks[-1][1][-1:], dropped, axis=0)
                self.chunks.append((self.n, fill))
                self.n += dropped
                self.dropped += dropped
            self.chunks.append((self.n, chunk))
            self.n += len(chunk)
            self.anchors.append((self.n, t_arrival))
            self.last_arrival = t_arrival
            while self.chunks and self.chunks[0][0] + len(self.chunks[0][1]) < self.n - self.keep:
                self.chunks.popleft()
            while self.anchors and self.anchors[0][0] < self.n - self.keep:
                self.anchors.popleft()
            self.recent_f = np.vstack([self.recent_f, self.live(chunk.astype(float))])[-300:]
            self.recent_raw = np.vstack([self.recent_raw, chunk])[-500:]

    def clock(self, window_s=60):
        """(a, b) with sample n taken at time a + b*n: a line through the earliest arrivals (lowest USB latency)."""
        with self.lock:
            pts = np.array([p for p in self.anchors if p[0] >= self.n - window_s * FS], dtype=np.float64)
        if len(pts) < 2:
            return (pts[0][1] - pts[0][0] / FS, 1.0 / FS) if len(pts) else (time.monotonic(), 1.0 / FS)
        n, t = pts[:, 0], pts[:, 1]
        b, a = np.polyfit(n, t, 1)
        resid = t - (a + b * n)
        low = resid <= np.percentile(resid, 25)
        if low.sum() >= 2 and np.ptp(n[low]) > 0:
            b, a = np.polyfit(n[low], t[low], 1)
        return a, b

    def slice(self, t0, t1, wait_s=1.0):
        """Raw samples taken between t0 and t1 (laptop clock), and each sample's time."""
        a, b = self.clock()
        n0, n1 = int(np.ceil((t0 - a) / b)), int(np.floor((t1 - a) / b))
        deadline = time.monotonic() + wait_s
        while self.n <= n1 and time.monotonic() < deadline:  # the last samples may still be in the USB pipe
            time.sleep(0.01)
        with self.lock:
            parts = []
            for start, arr in self.chunks:
                lo, hi = max(n0, start), min(n1 + 1, start + len(arr))
                if lo < hi:
                    parts.append(arr[lo - start:hi - start])
            first = max(n0, self.chunks[0][0]) if self.chunks else n0
        raw = np.vstack(parts) if parts else np.zeros((0, CHANNELS), np.uint16)
        ts = a + b * np.arange(first, first + len(raw), dtype=np.float64)
        return raw, ts

    def levels(self):
        """Per channel: filtered level over the last 150 ms, and the share of the last 0.5 s pinned at 0 or 1023."""
        with self.lock:
            f, r = self.recent_f[-150:], self.recent_raw
        if len(f) == 0:
            return np.zeros(CHANNELS), np.zeros(CHANNELS)
        return f.std(0), ((r <= 2) | (r >= 1021)).mean(0)


class SerialEmg(threading.Thread):
    """Reads the Uno through Thai's rig/reader.py (rig/uno firmware, 4 channels, 1 kHz)."""

    def __init__(self, port, buf):
        super().__init__(daemon=True)
        self.port, self.buf, self.error = port, buf, None

    def run(self):
        from reader import frames  # rig/reader.py
        stats = {}
        seen = 0
        try:
            for chunk in frames(self.port, stats):
                d = stats.get('dropped', 0)
                self.buf.push(chunk, time.monotonic(), d - seen)
                seen = d
        except Exception as e:  # noqa: BLE001  (shown in the window)
            self.error = str(e)


class FakeEmg(threading.Thread):
    """Synthetic 4-channel stream at 1 kHz with hum, noise and bursts, delivered in USB-like chunks."""

    def __init__(self, buf, drift=1.0003):
        super().__init__(daemon=True)
        self.buf, self.drift, self.error = buf, drift, None

    def run(self):
        rng = np.random.default_rng(0)
        n, t0 = 0, time.monotonic()
        while True:
            time.sleep(0.008 + rng.uniform(0, 0.006))
            target = int((time.monotonic() - t0) * FS * self.drift)
            k = target - n
            if k <= 0:
                continue
            tt = (n + np.arange(k))[:, None] / FS
            burst = (np.sin(2 * np.pi * 0.4 * tt) > 0.3) * rng.normal(0, 25, (k, CHANNELS))
            x = 512 + 20 * np.sin(2 * np.pi * 60 * tt) + rng.normal(0, 3, (k, CHANNELS)) + burst
            n = target
            self.buf.push(np.clip(x, 0, 1023).astype(np.uint16), time.monotonic() + rng.uniform(0.001, 0.004))


def find_port():
    from serial.tools import list_ports
    ports = [p.device for p in list_ports.comports() if p.vid in ARDUINO_VIDS]
    if not ports:
        sys.exit('Uno not found: plug in its USB cable, or pass --port (COM3 on Windows, /dev/cu.usbmodem... on macOS).')
    return ports[0]


# ------------------------------------------------------------------------------------------------ camera
class Camera(threading.Thread):
    """Grabs frames as fast as the camera gives them and stamps each with time.monotonic() right after read()."""

    def __init__(self, source, width, height, fps):
        super().__init__(daemon=True)
        self.fake = source == 'fake'
        self.width, self.height, self.fps = width, height, fps
        self.lock = threading.Lock()
        self.latest = None  # (frame, t)
        self.sink = None  # a ClipWriter while recording
        self.error = None
        self.times = deque(maxlen=60)
        if not self.fake:
            src = int(source) if str(source).isdigit() else source
            backend = {'Windows': cv2.CAP_DSHOW, 'Darwin': cv2.CAP_AVFOUNDATION}.get(platform.system(), cv2.CAP_ANY)
            self.cap = cv2.VideoCapture(src, backend) if isinstance(src, int) else cv2.VideoCapture(src)
            if not self.cap.isOpened():
                sys.exit(f'Cannot open camera {source}. Try --camera 1, or --list-cameras.')
            self.file = not isinstance(src, int)
            if not self.file:
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                self.cap.set(cv2.CAP_PROP_FPS, fps)
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or width
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or height
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or fps

    def run(self):
        k = 0
        period = 1.0 / self.fps
        next_t = time.monotonic()
        while True:
            if self.fake:
                next_t += period
                time.sleep(max(0.0, next_t - time.monotonic()))
                frame = np.full((self.height, self.width, 3), 40, np.uint8)
                x = int((k * 7) % self.width)
                cv2.rectangle(frame, (x, self.height // 3), (x + 60, 2 * self.height // 3), (80, 160, 220), -1)
                cv2.putText(frame, f'fake camera {k}', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                ok = True
            else:
                ok, frame = self.cap.read()
                if not ok and self.file:  # a video file loops, paced at its own rate
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                if self.file:
                    next_t += period
                    time.sleep(max(0.0, next_t - time.monotonic()))
            t = time.monotonic()
            if not ok:
                self.error = 'camera stopped sending frames'
                time.sleep(0.1)
                continue
            k += 1
            with self.lock:
                self.latest = (frame, t)
                sink = self.sink
            self.times.append(t)
            if sink is not None:
                sink.put(frame, t)

    def measured_fps(self):
        ts = list(self.times)
        return (len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) > 2 and ts[-1] > ts[0] else 0.0


class ClipWriter(threading.Thread):
    """Writes one clip's frames to MP4 off the capture thread; a full queue drops a frame and counts it."""

    def __init__(self, path, fps, size):
        super().__init__(daemon=True)
        self.path, self.fps, self.size = path, fps, size
        self.q = queue.Queue(maxsize=240)
        self.ts, self.dropped = [], 0

    def put(self, frame, t):
        try:
            self.q.put_nowait((frame, t))
        except queue.Full:
            self.dropped += 1

    def run(self):
        out = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, self.size)
        while True:
            item = self.q.get()
            if item is None:
                break
            frame, t = item
            if (frame.shape[1], frame.shape[0]) != self.size:
                frame = cv2.resize(frame, self.size)
            out.write(frame)
            self.ts.append(t)
        out.release()

    def finish(self):
        self.q.put(None)
        self.join()
        return np.array(self.ts, dtype=np.float64)


# ------------------------------------------------------------------------------------------------ session
class Session:
    def __init__(self, root, session, speaker, emg, cam, emg_source):
        self.session = session
        self.dir = os.path.join(root, f'session_{session:02d}')
        os.makedirs(self.dir, exist_ok=True)
        self.items = load_items(session)
        self.speaker, self.emg, self.cam, self.emg_source = speaker, emg, cam, emg_source
        self.manifest = os.path.join(self.dir, 'manifest.jsonl')
        self.entries = {}
        if os.path.exists(self.manifest):
            with open(self.manifest, encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        e = json.loads(line)
                        self.entries[e['id']] = e
        self.skipped = set()
        self.writer = None
        self.current = None
        self.t_start = None
        self.last_id = None
        self.write_meta()

    def pending(self):
        return [it for it in self.items if it['id'] not in self.entries and it['id'] not in self.skipped]

    def write_meta(self):
        meta = {'session': self.session, 'speaker': self.speaker, 'created': datetime.datetime.now().isoformat(),
                'emg': {'sample_rate': FS, 'channels': CHANNELS, 'pins': ['A0', 'A1', 'A2', 'A3'], 'sites': SITES,
                        'units': 'raw 10-bit Uno readings, unfiltered', 'source': self.emg_source},
                'camera': {'width': self.cam.width, 'height': self.cam.height, 'fps_requested': self.cam.fps,
                           'mirrored': False},
                'clock': 'time.monotonic() on this laptop, shared by camera and EMG',
                'recorder_version': VERSION, 'platform': platform.platform(),
                'n_items': len(self.items), 'prompts': SESSIONS[self.session][0],
                'styles': STYLES.get(self.session, {})}
        with open(os.path.join(self.dir, 'meta.json'), 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=1)

    def start(self, item):
        self.current = item
        path = os.path.join(self.dir, f"{item['id']}.mp4")
        self.writer = ClipWriter(path, self.cam.fps, (self.cam.width, self.cam.height))
        self.writer.start()
        self.t_start = time.monotonic()
        with self.cam.lock:
            self.cam.sink = self.writer

    def stop(self):
        t_end = time.monotonic()
        with self.cam.lock:
            self.cam.sink = None
        video_ts = self.writer.finish()
        item, uid = self.current, self.current['id']
        raw, emg_ts = self.emg.slice(self.t_start, t_end)
        dur = t_end - self.t_start
        np.save(os.path.join(self.dir, f'{uid}_emg.npy'), raw)
        np.save(os.path.join(self.dir, f'{uid}_emg_ts.npy'), emg_ts)
        np.save(os.path.join(self.dir, f'{uid}_video_ts.npy'), video_ts)
        warnings = []
        if len(raw) < 0.95 * dur * FS:
            warnings.append(f'EMG short: {len(raw)} samples for {dur:.1f} s')
        if len(raw):
            pinned = ((raw <= 2) | (raw >= 1021)).mean(0)
            flat = raw.astype(float).std(0) < 0.5
            warnings += [f'A{c} pinned at 0 or 1023' for c in range(CHANNELS) if pinned[c] > 0.5]
            warnings += [f'A{c} flat' for c in range(CHANNELS) if flat[c]]
        if len(video_ts) < max(5, 0.8 * dur * self.cam.fps):
            warnings.append(f'video short: {len(video_ts)} frames for {dur:.1f} s')
        if self.writer.dropped:
            warnings.append(f'{self.writer.dropped} video frames dropped')
        entry = {'id': uid, 'session': f'session_{self.session:02d}', 'set': item['set'], 'prompt': item['prompt'],
                 'pace': item['pace'], 'expression': item.get('expression'), 'effort': item.get('effort'),
                 'cue': item.get('cue'), 'target_wpm': None, 'mouth': 'open', 'voiced': False,
                 'emg': f'{uid}_emg.npy', 'emg_ts': f'{uid}_emg_ts.npy', 'video': f'{uid}.mp4',
                 'video_ts': f'{uid}_video_ts.npy', 't_start': None, 't_end': None,
                 'clip_t0': self.t_start, 'clip_t1': t_end, 'duration_s': round(dur, 3),
                 'emg_samples': int(len(raw)), 'emg_dropped_total': int(self.emg.dropped),
                 'n_frames': int(len(video_ts)), 'video_dropped': int(self.writer.dropped),
                 'fps': round(len(video_ts) / dur, 2) if dur > 0 else 0, 'speaker': self.speaker, 'sites': SITES,
                 'recorded_at': datetime.datetime.now().isoformat(), 'recorder_version': VERSION,
                 'warnings': warnings}
        self.entries[uid] = entry
        self.save_manifest()
        self.last_id, self.current, self.writer = uid, None, None
        return entry

    def redo(self, uid):
        for suffix in ('.mp4', '_emg.npy', '_emg_ts.npy', '_video_ts.npy'):
            p = os.path.join(self.dir, uid + suffix)
            if os.path.exists(p):
                os.remove(p)
        self.entries.pop(uid, None)
        self.save_manifest()

    def save_manifest(self):
        order = {it['id']: i for i, it in enumerate(self.items)}
        tmp = self.manifest + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            for e in sorted(self.entries.values(), key=lambda e: order.get(e['id'], 1e9)):
                f.write(json.dumps(e) + '\n')
        os.replace(tmp, self.manifest)


# ------------------------------------------------------------------------------------------------ window
def charger_plugged():
    try:
        import psutil
        b = psutil.sensors_battery()
        return bool(b and b.power_plugged)
    except Exception:  # noqa: BLE001
        return False


def put(img, text, xy, scale=0.7, color=(235, 235, 235), thick=2):
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw(sess, cam, emg, emg_thread, state, item, help_on, msg, charger, real_emg):
    W, H, VW, VH = 1280, 820, 960, 540
    canvas = np.full((H, W, 3), 24, np.uint8)
    with cam.lock:
        latest = cam.latest
    if latest is not None:
        view = cv2.resize(cv2.flip(latest[0], 1), (VW, VH))  # mirrored for the speaker; files stay unmirrored
        canvas[:VH, :VW] = view
        cx, cy = VW // 2, int(VH * 0.62)
        cv2.ellipse(canvas, (cx, cy), (110, 70), 0, 0, 360, (90, 200, 120), 1)  # where the mouth should sit
    # side panel: EMG levels
    lv, pinned = emg.levels()
    put(canvas, 'EMG', (VW + 20, 40), 0.8)
    alive = emg.last_arrival is not None and time.monotonic() - emg.last_arrival < 1.0
    for c in range(CHANNELS):
        y = 80 + c * 60
        put(canvas, f'A{c} site {SITES[c]}', (VW + 20, y), 0.55)
        w = int(min(1.0, lv[c] / 12.0) * 240)
        col = (60, 60, 220) if pinned[c] > 0.05 else (90, 200, 120)
        cv2.rectangle(canvas, (VW + 20, y + 10), (VW + 20 + w, y + 30), col, -1)
        cv2.rectangle(canvas, (VW + 20, y + 10), (VW + 260, y + 30), (90, 90, 90), 1)
    put(canvas, f'camera {cam.measured_fps():4.1f} fps', (VW + 20, 345), 0.55)
    put(canvas, f'EMG dropped frames: {emg.dropped}', (VW + 20, 375), 0.55)
    done, total = len(sess.entries), len(sess.items)
    put(canvas, f'session {sess.session}: {done}/{total} saved', (VW + 20, 415), 0.6)
    if sess.skipped:
        put(canvas, f'skipped: {len(sess.skipped)}', (VW + 20, 445), 0.55)
    # warnings
    warn = []
    if charger and real_emg:
        warn.append('UNPLUG THE CHARGER: pads are on, battery only')
    if not alive:
        warn.append('EMG silent: ' + (emg_thread.error or 'no frames from the Uno'))
    if any(pinned > 0.05):
        warn.append('a channel hits 0 or 1023: press pads / REF down')
    if cam.error:
        warn.append(cam.error)
    for i, w in enumerate(warn):
        put(canvas, w, (20, VH - 20 - 32 * i), 0.75, (80, 80, 255))
    # prompt area
    y0 = VH + 40
    if state == 'done':
        put(canvas, 'Session complete. Press Q to quit.', (30, y0 + 30), 1.1, (120, 230, 140), 3)
    elif item is not None:
        tag = {'natural': 'NATURAL PACE', 'fast': 'FAST PACE', 'slow': 'SLOW PACE', 'demo': 'DEMO PACE',
               'rest': 'REST CLIP'}[item['pace']]
        style = item.get('style')
        put(canvas, f"{item['id']}  |  {tag}" + (f"  |  {style.upper()}" if style else ''), (30, y0), 0.65,
            (180, 200, 255))
        top = y0 + 55
        if style:   # the delivery instruction, in amber above the sentence
            put(canvas, STYLE_CUES[style], (30, y0 + 38), 0.75, (60, 200, 255), 2)
            top += 36
        text = item['prompt'] or item['cue']
        for i, line in enumerate(textwrap.wrap(text, 42)[:3]):
            put(canvas, line, (30, top + i * 52), 1.45, (255, 255, 255), 3)
        status = (f'RECORDING  {time.monotonic() - sess.t_start:4.1f} s   (SPACE to stop)' if state == 'rec'
                  else 'SPACE to start, mouth it once, SPACE to stop')
        put(canvas, status, (30, H - 25), 0.8, (60, 60, 230) if state == 'rec' else (200, 200, 200))
        if state == 'rec':
            cv2.circle(canvas, (VW - 40, 40), 16, (40, 40, 230), -1)
    if msg:
        put(canvas, msg, (VW + 20, H - 25), 0.55, (200, 200, 120))
    if help_on:
        lines = ['SPACE start / stop clip', 'R redo last clip', 'S skip prompt', 'H hide help', 'Q / ESC quit']
        for i, line in enumerate(lines):
            put(canvas, line, (VW + 20, 500 + i * 30), 0.6, (200, 200, 120))
    return canvas


# ------------------------------------------------------------------------------------------------ main
def list_cameras():
    backend = {'Windows': cv2.CAP_DSHOW, 'Darwin': cv2.CAP_AVFOUNDATION}.get(platform.system(), cv2.CAP_ANY)
    for i in range(6):
        cap = cv2.VideoCapture(i, backend)
        ok = cap.isOpened() and cap.read()[0]
        print(f'camera {i}: ' + (f'{int(cap.get(3))}x{int(cap.get(4))}' if ok else 'not available'))
        cap.release()


def parse(argv=None):
    p = argparse.ArgumentParser(description='Record camera + EMG for the Reclaim sessions (RECORDING.md).')
    p.add_argument('--session', type=int, choices=[1, 2, 3])
    p.add_argument('--root', default=os.path.join(os.environ.get('RECLAIM_WORKDIR', REPO), 'recordings'))
    p.add_argument('--speaker', default='austin')
    p.add_argument('--camera', default='0', help='camera index, a video file, or "fake"')
    # the live camera reader captures the C920 at 1920x1080 and shrinks to 1280x720 itself (live/run.py --size,
    # --process-size): record the same capture, so replayed sessions match the live demo
    p.add_argument('--width', type=int, default=1920)
    p.add_argument('--height', type=int, default=1080)
    p.add_argument('--fps', type=float, default=30)
    p.add_argument('--port', default=None, help='Uno serial port; found automatically if omitted')
    p.add_argument('--fake-emg', action='store_true', help='synthetic EMG, no Uno')
    p.add_argument('--allow-charger', action='store_true', help='only with --fake-emg: ignore the charger check')
    p.add_argument('--list-cameras', action='store_true')
    p.add_argument('--auto', type=int, default=None, help='record N clips without a window (testing)')
    p.add_argument('--clip-seconds', type=float, default=2.0, help='clip length in --auto mode')
    a = p.parse_args(argv)
    if not a.list_cameras and a.session is None:
        p.error('--session is required')
    if a.allow_charger and not a.fake_emg:
        p.error('--allow-charger only works with --fake-emg: with pads on skin the laptop must run on battery')
    return a


def main(argv=None):
    a = parse(argv)
    if a.list_cameras:
        return list_cameras()
    emg = EmgBuffer()
    emg_thread = FakeEmg(emg) if a.fake_emg else SerialEmg(a.port or find_port(), emg)
    emg_source = 'fake' if a.fake_emg else emg_thread.port
    cam = Camera(a.camera, a.width, a.height, a.fps)
    emg_thread.start()
    cam.start()
    sess = Session(a.root, a.session, a.speaker, emg, cam, emg_source)
    print(f'{len(sess.pending())} of {len(sess.items)} clips to record, saving to {sess.dir}', flush=True)
    t_wait = time.monotonic() + 6  # the Uno resets when the port opens
    while (emg.last_arrival is None or cam.latest is None) and time.monotonic() < t_wait:
        time.sleep(0.05)
    if a.auto is not None:  # testing: no window
        for item in sess.pending()[:a.auto]:
            sess.start(item)
            time.sleep(a.clip_seconds)
            e = sess.stop()
            print(f"saved {e['id']}: {e['emg_samples']} EMG samples, {e['n_frames']} frames, warnings {e['warnings']}")
        return sess
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 820)
    state, help_on, msg, msg_t = 'idle', True, '', 0
    charger, charger_t = charger_plugged(), time.monotonic()
    while True:
        if time.monotonic() - charger_t > 2:
            charger, charger_t = charger_plugged(), time.monotonic()
        pend = sess.pending()
        item = sess.current if state == 'rec' else (pend[0] if pend else None)
        if item is None and state != 'rec':
            state = 'done'
        if msg and time.monotonic() - msg_t > 4:
            msg = ''
        cv2.imshow(WIN, draw(sess, cam, emg, emg_thread, state, item, help_on, msg, charger, not a.fake_emg))
        key = cv2.waitKey(15) & 0xFF
        if key in (ord('q'), 27):
            if state == 'rec':
                sess.stop()
                sess.redo(sess.last_id)  # a clip cut short by quitting is not kept
            break
        elif key == ord(' ') and state == 'idle':
            if charger and not a.fake_emg:
                msg, msg_t = 'Unplug the charger first (pads are on).', time.monotonic()
            else:
                sess.start(item)
                state = 'rec'
        elif key == ord(' ') and state == 'rec':
            e = sess.stop()
            state = 'idle'
            msg = f"saved {e['id']}" + (f": {', '.join(e['warnings'])}" if e['warnings'] else '')
            msg_t = time.monotonic()
            print(msg, flush=True)
        elif key == ord('r') and state in ('idle', 'done') and sess.last_id:
            sess.redo(sess.last_id)
            msg, msg_t = f'redo {sess.last_id}', time.monotonic()
            sess.last_id, state = None, 'idle'
        elif key == ord('s') and state == 'idle' and item is not None:
            sess.skipped.add(item['id'])
            msg, msg_t = f"skipped {item['id']} (rerun to get it back)", time.monotonic()
        elif key == ord('h'):
            help_on = not help_on
    cv2.destroyAllWindows()
    print(f'{len(sess.entries)} of {len(sess.items)} clips saved in {sess.dir}')
    return sess


if __name__ == '__main__':
    main()
