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
Camera and EMG share time.perf_counter(): EMG sample times come from a fit of sample count against arrival time, so they
are exact to about a millisecond even though the Uno's clock drifts from the laptop's.
"""
import argparse
import datetime
import json
import os
import platform
import queue
import random
import shutil
import subprocess
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

VERSION = 3  # 2: delivery styles (slow pace, facial expression, effort) on some sentences
#             3: H.264 video (NVENC), MJPG camera capture asked for at open, perf_counter clock
# The clock for every camera and EMG timestamp. time.monotonic() ticks only every 15.6 ms on Windows before Python 3.13;
# perf_counter is the same steady clock at under a microsecond (on 3.13+ and on Linux/macOS the two are the same clock).
now = time.perf_counter
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
STYLE_CUES = {  # the amber line above the sentence; short enough to stay left of the key-help panel (x < 960)
    'slow': 'SLOW: about half your natural speed',
    'smile': 'SMILE: a real smile, held all sentence',
    'frown': 'FROWN: look upset, held all sentence',
    'surprised': 'SURPRISED: brows up, held all sentence',
    'soft': 'SOFT: small movements, like a whisper',
    'emphatic': 'EMPHATIC: big, firm movements',
}
CUE_SCALE = 0.75
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
            return (pts[0][1] - pts[0][0] / FS, 1.0 / FS) if len(pts) else (now(), 1.0 / FS)
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
        deadline = now() + wait_s
        while self.n <= n1 and now() < deadline:  # the last samples may still be in the USB pipe
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
                self.buf.push(chunk, now(), d - seen)
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
        n, t0 = 0, now()
        while True:
            time.sleep(0.008 + rng.uniform(0, 0.006))
            target = int((now() - t0) * FS * self.drift)
            k = target - n
            if k <= 0:
                continue
            tt = (n + np.arange(k))[:, None] / FS
            burst = (np.sin(2 * np.pi * 0.4 * tt) > 0.3) * rng.normal(0, 25, (k, CHANNELS))
            x = 512 + 20 * np.sin(2 * np.pi * 60 * tt) + rng.normal(0, 3, (k, CHANNELS)) + burst
            n = target
            self.buf.push(np.clip(x, 0, 1023).astype(np.uint16), now() + rng.uniform(0.001, 0.004))


def find_port():
    from serial.tools import list_ports
    ports = [p.device for p in list_ports.comports() if p.vid in ARDUINO_VIDS]
    if not ports:
        sys.exit('Uno not found: plug in its USB cable, or pass --port (COM3 on Windows, /dev/cu.usbmodem... on macOS).')
    return ports[0]


# ------------------------------------------------------------------------------------------------ camera
class Camera(threading.Thread):
    """Grabs frames as fast as the camera gives them and stamps each with now() (time.perf_counter) right after read()."""

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
            self.file = not isinstance(src, int)
            if self.file:
                self.cap = cv2.VideoCapture(src)
            else:
                # Ask for MJPG when opening. Set afterwards, Windows DirectShow (OpenCV 5) keeps the uncompressed YUY2
                # mode, which USB 2 limits to 5 fps at 1920x1080 and 10 fps at 1280x720; the C920 gives 30 in MJPG.
                params = [cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'), cv2.CAP_PROP_FRAME_WIDTH, width,
                          cv2.CAP_PROP_FRAME_HEIGHT, height, cv2.CAP_PROP_FPS, int(round(fps))]
                self.cap = cv2.VideoCapture(src, backend, params)
                if not self.cap.isOpened():  # a backend that rejects open parameters: open, then set
                    self.cap = cv2.VideoCapture(src, backend)
                    for prop, value in zip(params[::2], params[1::2]):
                        self.cap.set(prop, value)
            if not self.cap.isOpened():
                sys.exit(f'Cannot open camera {source}. Try --camera 1, or --list-cameras.')
            if not self.file:
                # Auto exposure on (0.75 means auto for DirectShow and V4L2): a camera keeps a manual exposure that
                # another program set until it is unplugged, and the C920 came up nearly black that way.
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
            # a property a backend can't report reads 0 or -1 (DirectShow gives fps -1 after open parameters)
            got = [self.cap.get(p) for p in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)]
            self.width = int(got[0]) if got[0] > 0 else width
            self.height = int(got[1]) if got[1] > 0 else height
            self.fps = got[2] if got[2] > 0 else fps
            code = int(self.cap.get(cv2.CAP_PROP_FOURCC))
            self.fourcc = ''.join(chr((code >> 8 * i) & 0xFF) for i in range(4)).strip('\0 ') or None
            print(f'camera {source}: {self.width}x{self.height}, {self.fourcc or "unknown format"}', flush=True)
            if not self.file and self.fourcc and self.fourcc != 'MJPG':
                print(f'warning: the camera gave {self.fourcc}, not MJPG, so it may run at 5 to 10 fps', flush=True)
        else:
            self.fourcc = None

    def run(self):
        k = 0
        period = 1.0 / self.fps
        next_t = now()
        while True:
            if self.fake:
                next_t += period
                time.sleep(max(0.0, next_t - now()))
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
                    time.sleep(max(0.0, next_t - now()))
            t = now()
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


# H.264 encoders through ffmpeg, best first: h264_nvenc runs on the GPU's video engine (not the CUDA cores Auto-AVSR and
# SVR-1 use), libx264 on the CPU. Without ffmpeg, OpenCV's MPEG-4 Part 2 ('mp4v') is the last resort.
ENCODERS = {
    'nvenc': ['-c:v', 'h264_nvenc', '-preset', 'p5', '-rc', 'vbr', '-cq', '21', '-b:v', '0'],
    'x264': ['-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20'],
}
NO_WINDOW = getattr(subprocess, 'CREATE_NO_WINDOW', 0)  # Windows: no console window per clip


def ffmpeg_cmd(ffmpeg, codec, path, fps, size):
    return [ffmpeg, '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
            '-s', f'{size[0]}x{size[1]}', '-r', f'{fps:.3f}', '-i', '-', *ENCODERS[codec],
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart', path]


def pick_codec(want, ffmpeg):
    """The codec to record with: 'nvenc' or 'x264' if a 1-second test encode works, else 'mp4v'."""
    if want == 'mp4v':
        return 'mp4v'
    if not ffmpeg:
        print('warning: ffmpeg not found (pass --ffmpeg PATH): writing MPEG-4 with OpenCV, not H.264', flush=True)
        return 'mp4v'
    frame = np.zeros((256, 256, 3), np.uint8).tobytes()
    for codec in ([want] if want != 'auto' else list(ENCODERS)):
        cmd = ffmpeg_cmd(ffmpeg, codec, '-', 30.0, (256, 256))
        cmd[-1:] = ['-f', 'null', '-']
        try:
            r = subprocess.run(cmd, input=frame * 30, capture_output=True, timeout=30, creationflags=NO_WINDOW)
        except (OSError, subprocess.TimeoutExpired) as e:
            r = type('R', (), {'returncode': 1, 'stderr': str(e).encode()})()
        if r.returncode == 0:
            return codec
        print(f'warning: {codec} test encode failed: {r.stderr.decode(errors="replace").strip()[-300:]}', flush=True)
    print('warning: no H.264 encoder worked: writing MPEG-4 with OpenCV', flush=True)
    return 'mp4v'


class ClipWriter(threading.Thread):
    """Writes one clip's frames to MP4 off the capture thread; a full queue drops a frame and counts it.

    H.264 goes through an ffmpeg process fed raw frames on stdin; the file's frame rate is nominal, the _video_ts.npy
    times are the truth."""

    QUEUE_BYTES = 600e6     # frames waiting to be encoded: ~100 at 1920x1080 (3 s), ~200 at 1280x720

    def __init__(self, path, fps, size, codec='mp4v', ffmpeg=None):
        super().__init__(daemon=True)
        self.path, self.fps, self.size, self.codec, self.ffmpeg = path, fps, size, codec, ffmpeg
        self.q = queue.Queue(maxsize=max(30, int(self.QUEUE_BYTES / (size[0] * size[1] * 3))))
        self.ts, self.dropped, self.error = [], 0, None

    def put(self, frame, t):
        try:
            self.q.put_nowait((frame, t))
        except queue.Full:
            self.dropped += 1

    def run(self):
        proc = out = None
        if self.codec == 'mp4v':
            out = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, self.size)
        else:
            proc = subprocess.Popen(ffmpeg_cmd(self.ffmpeg, self.codec, self.path, self.fps, self.size),
                                    stdin=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=NO_WINDOW)
        while True:
            item = self.q.get()
            if item is None:
                break
            frame, t = item
            if self.error:
                continue  # keep draining so the capture thread never waits
            if (frame.shape[1], frame.shape[0]) != self.size:
                frame = cv2.resize(frame, self.size)
            if out is not None:
                out.write(frame)
            else:
                try:
                    proc.stdin.write(np.ascontiguousarray(frame).data)
                except OSError:
                    self.error = 'video encoder stopped'
                    continue
            self.ts.append(t)
        if out is not None:
            out.release()
        else:
            try:
                proc.stdin.close()
            except OSError:
                pass
            err = proc.stderr.read().decode(errors='replace').strip()
            if proc.wait() != 0 or self.error:
                self.error = f'video encoder failed: {err[-200:] or proc.returncode}'

    def finish(self):
        self.q.put(None)
        self.join()
        return np.array(self.ts, dtype=np.float64)


# ------------------------------------------------------------------------------------------------ session
class Session:
    def __init__(self, root, session, speaker, emg, cam, emg_source, codec='mp4v', ffmpeg=None):
        self.session = session
        self.dir = os.path.join(root, f'session_{session:02d}')
        os.makedirs(self.dir, exist_ok=True)
        self.items = load_items(session)
        self.speaker, self.emg, self.cam, self.emg_source = speaker, emg, cam, emg_source
        self.codec, self.ffmpeg = codec, ffmpeg
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
        """meta.json for the session. On a resume it keeps the first 'created' and appends to 'runs', so a session
        recorded across recorder versions or camera sizes says so; each clip's own fields stay the truth."""
        path = os.path.join(self.dir, 'meta.json')
        old = {}
        try:
            with open(path, encoding='utf-8') as f:
                old = json.load(f)
        except FileNotFoundError:
            pass
        except ValueError:          # cut short by a crash: the clips and manifest are fine, start the history again
            print('note: meta.json was unreadable, rewriting it', flush=True)
        started = datetime.datetime.now().isoformat()
        run = {'started': started, 'recorder_version': VERSION, 'camera': [self.cam.width, self.cam.height],
               'codec': self.codec}
        unstyled = sum(1 for e in self.entries.values() if 'expression' not in e)
        if unstyled:
            print(f'note: {unstyled} clips in this session were saved by recorder v1 (no delivery styles); only the '
                  f'remaining sentences get styles', flush=True)
        meta = {'session': self.session, 'speaker': self.speaker, 'created': old.get('created', started),
                'runs': old.get('runs', []) + [run],
                'emg': {'sample_rate': FS, 'channels': CHANNELS, 'pins': ['A0', 'A1', 'A2', 'A3'], 'sites': SITES,
                        'units': 'raw 10-bit Uno readings, unfiltered', 'source': self.emg_source},
                'camera': {'width': self.cam.width, 'height': self.cam.height, 'fps_requested': self.cam.fps,
                           'capture_format': self.cam.fourcc, 'codec': self.codec, 'mirrored': False},
                'clock': 'time.perf_counter() on this laptop, shared by camera and EMG',
                'recorder_version': VERSION, 'platform': platform.platform(),
                'n_items': len(self.items), 'prompts': SESSIONS[self.session][0],
                'styles': STYLES.get(self.session, {})}
        with open(path + '.tmp', 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=1)
        os.replace(path + '.tmp', path)

    def start(self, item):
        self.current = item
        path = os.path.join(self.dir, f"{item['id']}.mp4")
        self.writer = ClipWriter(path, self.cam.fps, (self.cam.width, self.cam.height), self.codec, self.ffmpeg)
        self.writer.start()
        self.t_start = now()
        with self.cam.lock:
            self.cam.sink = self.writer

    def stop(self):
        t_end = now()
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
        if self.writer.error:
            warnings.append(self.writer.error)
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
                 'video_codec': self.codec,
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
        for attempt in range(20):   # Windows: a virus scanner or sync tool can hold the file open for a moment
            try:
                os.replace(tmp, self.manifest)
                return
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.05)


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
    alive = emg.last_arrival is not None and now() - emg.last_arrival < 1.0
    for c in range(CHANNELS):
        y = 80 + c * 60
        put(canvas, f'A{c} site {SITES[c]}', (VW + 20, y), 0.55)
        w = int(min(1.0, lv[c] / 12.0) * 240)
        col = (60, 60, 220) if pinned[c] > 0.05 else (90, 200, 120)
        cv2.rectangle(canvas, (VW + 20, y + 10), (VW + 20 + w, y + 30), col, -1)
        cv2.rectangle(canvas, (VW + 20, y + 10), (VW + 260, y + 30), (90, 90, 90), 1)
    fps = cam.measured_fps()
    slow = len(cam.times) == cam.times.maxlen and fps < 0.85 * cam.fps   # after 60 frames, under ~25 of 30 fps
    put(canvas, f'camera {fps:4.1f} fps', (VW + 20, 345), 0.55, (80, 80, 255) if slow else (235, 235, 235))
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
    if slow:
        warn.append(f'camera only {fps:.0f} fps: check the light, or that it gave MJPG')
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
        extra = f"  |  {style.upper()}" if style and style != 'slow' else ''      # SLOW is already the pace tag
        put(canvas, f"{item['id']}  |  {tag}{extra}", (30, y0), 0.65, (180, 200, 255))
        top = y0 + 55
        if style:   # the delivery instruction, in amber above the sentence
            put(canvas, STYLE_CUES[style], (30, y0 + 38), CUE_SCALE, (60, 200, 255), 2)
            top += 36
        text = item['prompt'] or item['cue']
        for i, line in enumerate(textwrap.wrap(text, 42)[:3]):
            put(canvas, line, (30, top + i * 52), 1.45, (255, 255, 255), 3)
        status = (f'RECORDING  {now() - sess.t_start:4.1f} s   (SPACE to stop)' if state == 'rec'
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
    # Evan's live camera reader (live/, branch feat/camera-live, not merged yet) captures the C920 at 1920x1080 and
    # shrinks to 1280x720 itself: record the same capture, so replayed sessions match the live demo
    p.add_argument('--width', type=int, default=1920)
    p.add_argument('--height', type=int, default=1080)
    p.add_argument('--fps', type=float, default=30)
    p.add_argument('--codec', default='auto', choices=['auto', 'nvenc', 'x264', 'mp4v'],
                   help='video codec; auto: H.264 on the GPU (nvenc), else on the CPU (x264), else mp4v')
    p.add_argument('--ffmpeg', default=os.environ.get('RECLAIM_FFMPEG') or shutil.which('ffmpeg'),
                   help='ffmpeg executable (default: $RECLAIM_FFMPEG, then ffmpeg on PATH)')
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
    codec = pick_codec(a.codec, a.ffmpeg)
    print(f'video: {codec}' + (' (H.264 on the GPU)' if codec == 'nvenc' else ''), flush=True)
    sess = Session(a.root, a.session, a.speaker, emg, cam, emg_source, codec, a.ffmpeg)
    print(f'{len(sess.pending())} of {len(sess.items)} clips to record, saving to {sess.dir}', flush=True)
    t_wait = now() + 6  # the Uno resets when the port opens
    while (emg.last_arrival is None or cam.latest is None) and now() < t_wait:
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
    charger, charger_t = charger_plugged(), now()
    while True:
        if now() - charger_t > 2:
            charger, charger_t = charger_plugged(), now()
        pend = sess.pending()
        item = sess.current if state == 'rec' else (pend[0] if pend else None)
        if item is None and state != 'rec':
            state = 'done'
        if msg and now() - msg_t > 4:
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
                msg, msg_t = 'Unplug the charger first (pads are on).', now()
            else:
                sess.start(item)
                state = 'rec'
        elif key == ord(' ') and state == 'rec':
            e = sess.stop()
            state = 'idle'
            msg = f"saved {e['id']}" + (f": {', '.join(e['warnings'])}" if e['warnings'] else '')
            msg_t = now()
            print(msg, flush=True)
        elif key == ord('r') and state in ('idle', 'done') and sess.last_id:
            sess.redo(sess.last_id)
            msg, msg_t = f'redo {sess.last_id}', now()
            sess.last_id, state = None, 'idle'
        elif key == ord('s') and state == 'idle' and item is not None:
            sess.skipped.add(item['id'])
            msg, msg_t = f"skipped {item['id']} (rerun to get it back)", now()
        elif key == ord('h'):
            help_on = not help_on
    cv2.destroyAllWindows()
    print(f'{len(sess.entries)} of {len(sess.items)} clips saved in {sess.dir}')
    return sess


if __name__ == '__main__':
    main()
