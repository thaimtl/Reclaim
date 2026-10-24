"""Recorder for the wearer sessions (E7): raw camera frames per utterance in svr/recordings.py's layout.

    recordings/<session>/<id>.mp4            H.264 (libx264, crf 18, yuv420p), the raw unmirrored full-size frames,
                                             pts = capture time - first frame's time (variable frame rate)
    recordings/<session>/<id>_video_ts.npy   float64 (n_frames,) capture times (Frame.t, time.monotonic()), 1:1 with
                                             the video's frames: a dropped frame is in neither
    recordings/<session>/<manifest>          one JSON line per utterance; a redo under the same id replaces its line

A take is written to <id>.part.mp4 / <id>_video_ts.part.npy and renamed onto the final names (by the writer thread,
both files saved before either rename) only when it closed cleanly, so a failed redo leaves the previous take's files
and manifest line as they were. So does an empty redo (no frames) of an id whose line already has a video; an empty
take of a new id still gets its line (video null).

write() never blocks the capture thread: frames go through a bounded queue to one writer thread, and a full queue
drops the frame (counted in `dropped`). A frame whose t is not after the last one (CameraSource repeats a frame to
fill the 25 fps grid) is skipped and counted in `repeats`: the files hold each captured image once.

EMG: Thai's recorder writes its own lines (`emg` set, video fields null). Both recorders appending to one
manifest.jsonl would give load_manifest two rows per id, and load_emg fails on the camera's row (emg null). So when
both run, record the camera to manifest='manifest_video.jsonl' and merge by id afterwards (merge_manifests: the
non-null field wins, video_* from the camera file); alone, the default manifest.jsonl loads as is with emg null.

    rec = Recorder('recordings', 's1')
    rec.start('s1_001', set='C', prompt="Hi, it's nice to meet you.")
    for f in source.frames(): rec.write(f)        # from the capture loop, before any mirroring or drawing
    entry = rec.stop()                            # n_frames, dropped, fps, clip_t0 / clip_t1 (monotonic s), ...
    rec.close()

    venv-live/bin/python -m live.record --session s1 --set C --prompts docs/recording/session1.txt [--cam 0]
    venv-live/bin/python -m live.record --session t --set C --prompts P.txt --root /tmp/rec \
        --replay third_party/lipflow/samples/2016-03-12.mov   # no webcam: a file stamped as if live
    venv-live/bin/python -m live.record --merge recordings/s1      # manifest_video.jsonl -> manifest.jsonl
"""
import argparse
import datetime
import json
import logging
import os
import queue
import sys
import threading
import time
from fractions import Fraction

import av
import cv2
import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from live.source import MANIFEST, VIDEO_MANIFEST, lock_controls, ts_name, v4l2_fps  # noqa: E402
from live.types import Frame  # noqa: E402

log = logging.getLogger(__name__)
CODEC, CRF, PRESET = 'libx264', 18, 'veryfast'
PIX_FMT = 'yuv420p'
TIME_BASE = Fraction(1, 90000)
QUEUE_SIZE = 90
VIDEO_KEYS = ('video', 'video_ts')
DEFAULTS = {'set': None, 'prompt': '', 'target_wpm': None, 'mouth': 'open', 'voiced': False, 'emg': None,
            't_start': None, 't_end': None}


class Recorder:
    def __init__(self, root, session, fps_hint=30, codec=CODEC, queue_size=QUEUE_SIZE, crf=CRF, preset=PRESET,
                 manifest=MANIFEST):
        self.dir = os.path.join(root, session)
        os.makedirs(self.dir, exist_ok=True)
        self.session, self.fps_hint, self.codec, self.crf, self.preset = session, fps_hint, codec, crf, preset
        self.manifest = os.path.join(self.dir, manifest)
        self.dropped = 0
        self.repeats = 0
        self.encode_ms = []
        self._q = queue.Queue(maxsize=queue_size)
        self._utt = None
        self._seq = 0
        self._last_t = None
        self._utt_dropped = self._utt_repeats = 0
        self._done = None
        self._result = None
        self._error = None
        self._thread = threading.Thread(target=self._run, name='live-recorder', daemon=True)
        self._thread.start()

    @property
    def recording(self):
        return self._utt is not None

    def start(self, utt_id, **manifest_fields):
        """Open <id>.mp4 for the next frames. A redo under an existing id overwrites its files and manifest line."""
        if self._utt is not None:
            self.stop()
        self._seq += 1
        self._last_t, self._utt_dropped, self._utt_repeats = None, 0, 0
        fields = {**DEFAULTS, **manifest_fields}
        self._q.put(('open', self._seq, utt_id))
        self._utt = (self._seq, utt_id, fields)

    def write(self, frame: Frame):
        """Queue one frame of the open utterance; never blocks. Returns False when dropped, skipped or not recording."""
        utt = self._utt
        if utt is None:
            return False
        if self._last_t is not None and frame.t <= self._last_t:
            self._utt_repeats += 1
            self.repeats += 1
            return False
        try:
            self._q.put_nowait(('frame', utt[0], frame))
        except queue.Full:
            self._utt_dropped += 1
            self.dropped += 1
            return False
        self._last_t = frame.t
        return True

    def stop(self):
        """Close the file, write <id>_video_ts.npy and the manifest line; returns the manifest entry (None if idle)."""
        utt = self._utt
        if utt is None:
            return None
        self._utt = None
        seq, utt_id, fields = utt
        self._done = threading.Event()
        self._q.put(('close', seq, None))
        self._done.wait()
        err, self._error = self._error, None
        if err is not None:
            raise RuntimeError(f'recorder failed on {utt_id}') from err
        ts, size = self._result
        fps = (len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) > 1 and ts[-1] > ts[0] else 0.0
        entry = {'id': utt_id, 'session': self.session, **fields,
                 'video': f'{utt_id}.mp4' if len(ts) else None,
                 'video_ts': ts_name(utt_id) if len(ts) else None,
                 'n_frames': int(len(ts)), 'dropped': self._utt_dropped, 'repeats': self._utt_repeats,
                 'fps': round(fps, 3), 'width': size[0], 'height': size[1],
                 'clip_t0': float(ts[0]) if len(ts) else None, 'clip_t1': float(ts[-1]) if len(ts) else None,
                 'codec': f'{self.codec} crf {self.crf} {self.preset} {PIX_FMT}',
                 'recorded_at': datetime.datetime.now().isoformat(timespec='seconds')}
        if len(ts) or not any(r.get('id') == utt_id and r.get('video') for r in read_manifest(self.manifest)):
            upsert(self.manifest, entry)
        else:
            log.warning('%s: empty redo, the previous take stays', utt_id)
        return entry

    def close(self):
        if self._utt is not None:
            self.stop()
        self._q.put(('quit', 0, None))
        self._thread.join(timeout=10)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _run(self):
        out = stream = None
        seq, ts, size = 0, [], (None, None)
        path = utt = None
        while True:
            kind, s, item = self._q.get()
            try:
                if kind == 'open':
                    seq, ts, size, self.encode_ms = s, [], (None, None), []
                    utt, final = item, os.path.join(self.dir, f'{item}.mp4')
                    path = os.path.join(self.dir, f'{item}.part.mp4')
                elif kind == 'frame' and s == seq and path is not None:
                    t0 = time.perf_counter()
                    if out is None:
                        h, w = item.image.shape[:2]
                        out, stream = self._open(path, w, h)
                        size = (w, h)
                    vf = to_video_frame(item.image)
                    vf.pts = int(round((item.t - (ts[0] if ts else item.t)) / TIME_BASE))
                    vf.time_base = TIME_BASE
                    for p in stream.encode(vf):
                        out.mux(p)
                    ts.append(item.t)
                    self.encode_ms.append((time.perf_counter() - t0) * 1000)
                elif kind in ('close', 'quit'):
                    if out is not None:
                        for p in stream.encode(None):
                            out.mux(p)
                        out.close()
                        if kind == 'close':
                            ts_part = os.path.join(self.dir, ts_name(utt, part=True))
                            np.save(ts_part, np.asarray(ts, dtype=np.float64))
                            os.replace(path, final)
                            os.replace(ts_part, os.path.join(self.dir, ts_name(utt)))
                    if kind == 'close':
                        self._result = (np.asarray(ts, dtype=np.float64), size)
                    _remove(path)
                    out = stream = path = None
                    seq = 0
            except Exception as e:
                self._error = self._error or e
                if out is not None:
                    try:
                        out.close()
                    except Exception:
                        pass
                _remove(path)
                if utt is not None:
                    _remove(os.path.join(self.dir, ts_name(utt, part=True)))
                out = stream = path = None
                if kind == 'close':
                    self._result = (np.asarray(ts, dtype=np.float64), size)
            if kind == 'close':
                self._done.set()
            if kind == 'quit':
                return

    def _open(self, path, w, h):
        out = av.open(path, 'w')
        stream = out.add_stream(self.codec, rate=self.fps_hint, options={'crf': str(self.crf), 'preset': self.preset})
        stream.width, stream.height, stream.pix_fmt = w, h, PIX_FMT
        stream.time_base = TIME_BASE
        stream.codec_context.time_base = TIME_BASE
        return out, stream


def _remove(path):
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


def to_video_frame(image):
    """BGR uint8 -> yuv420p (PIX_FMT) VideoFrame by cv2 (BT.601 limited range, rounded; PyAV's swscale path reads back
    ~0.6 grey levels darker and is slower)."""
    return av.VideoFrame.from_ndarray(cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420), format=PIX_FMT)


def read_manifest(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_manifest(path, rows):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    os.replace(tmp, path)


def upsert(path, entry):
    """Append entry to a manifest.jsonl, replacing an earlier line with the same id (a redo)."""
    rows = read_manifest(path)
    if any(r.get('id') == entry['id'] for r in rows):
        write_manifest(path, [entry if r.get('id') == entry['id'] else r for r in rows])
    else:
        with open(path, 'a') as f:
            f.write(json.dumps(entry) + '\n')


def merge_manifests(session_dir, video=VIDEO_MANIFEST, emg=MANIFEST, out=MANIFEST):
    """Merge the camera's lines into the EMG recorder's by id; returns (n_merged, n_video_only, n_emg_only).

    Per id: the EMG line's fields, then every non-null camera field on top (video, video_ts, n_frames, clip_t0...);
    a field both set (prompt, set...) keeps the EMG value except video_*. Ids only one side has are kept as they are.
    """
    v = {r['id']: r for r in read_manifest(os.path.join(session_dir, video))}
    e = read_manifest(os.path.join(session_dir, emg))
    seen, rows = set(), []
    for r in e:
        c = v.get(r['id'])
        if c is not None:
            r = {**r, **{k: x for k, x in c.items() if x is not None and (k in VIDEO_KEYS or r.get(k) is None)}}
            seen.add(r['id'])
        rows.append(r)
    e_ids = {r['id'] for r in e}
    rows += [c for i, c in v.items() if i not in e_ids]
    write_manifest(os.path.join(session_dir, out), rows)
    return len(seen), len(v) - len(seen), len(e) - len(seen)


class RawSource:
    """Frames at the camera's own rate, no resampling or repeats (unlike CameraSource's 25 fps stream), t = grab time.

    replay: a video file read at its own fps in real time and looped, stamped with time.monotonic() like a camera.
    """

    def __init__(self, device=0, width=640, height=480, fps=None, replay=None):
        self.replay = replay
        if replay:
            self.cap = cv2.VideoCapture(replay)
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        else:
            backend = cv2.CAP_V4L2 if sys.platform.startswith('linux') else cv2.CAP_ANY
            self.cap = cv2.VideoCapture(device, backend)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            offered = v4l2_fps(device, width, height)
            want = fps or next((r for r in (60, 50) if r in offered), 30)
            self.cap.set(cv2.CAP_PROP_FPS, want)
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or want
            self.locked = lock_controls(device, self.cap)
        if not self.cap.isOpened():
            raise RuntimeError(f'could not open {replay or device}')
        self.size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def frames(self):
        index, next_t = 0, time.monotonic()
        while self.cap is not None:
            if self.replay:
                time.sleep(max(0.0, next_t - time.monotonic()))
                next_t += 1.0 / self.fps
                ok, image = self.cap.read()
                if not ok:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                t = time.monotonic()
            else:
                if not self.cap.grab():
                    time.sleep(0.005)
                    continue
                t = time.monotonic()
                ok, image = self.cap.retrieve()
                if not ok:
                    continue
            yield Frame(t=t, image=image, index=index)
            index += 1

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def read_prompts(path):
    """[(id, sentence)] from `id <tab> sentence` lines (docs/recording/*.txt); a bare id has prompt ''."""
    out = []
    with open(path) as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                i, _, s = line.rstrip('\n').partition('\t')
                out.append((i.strip(), s.strip()))
    return out


def main():
    p = argparse.ArgumentParser(description='Record one camera clip per prompt: Enter starts, Enter stops.')
    p.add_argument('--session')
    p.add_argument('--set', default='C', help='A B C D E F (svr/recordings.py)')
    p.add_argument('--prompts', help='id <tab> sentence per line')
    p.add_argument('--root', default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                  'recordings'))
    p.add_argument('--cam', default='0')
    p.add_argument('--size', default='640x480')
    p.add_argument('--fps', type=int, help='camera rate to request (default: 60 or 50 if offered, else 30)')
    p.add_argument('--replay', help='video file fed as if it were the camera (real time, looped)')
    p.add_argument('--manifest', default=MANIFEST, help=f'{VIDEO_MANIFEST} when the EMG recorder writes {MANIFEST}')
    p.add_argument('--first', help='start at this prompt id')
    p.add_argument('--seconds', type=float, help='stop each clip by itself after this long (rest clips, scripts)')
    p.add_argument('--target-wpm', type=int)
    p.add_argument('--mouth', default='open')
    p.add_argument('--merge', metavar='SESSION_DIR', help=f'merge {VIDEO_MANIFEST} into {MANIFEST} by id and exit')
    a = p.parse_args()
    if a.merge:
        print('merged %d, camera only %d, EMG only %d' % merge_manifests(a.merge))
        return
    if not (a.session and a.prompts):
        p.error('--session and --prompts are required')
    prompts = read_prompts(a.prompts)
    if a.first:
        prompts = prompts[[i for i, _ in prompts].index(a.first):]
    w, h = map(int, a.size.split('x'))
    src = RawSource(int(a.cam) if a.cam.isdigit() else a.cam, w, h, a.fps, a.replay)
    rec = Recorder(a.root, a.session, fps_hint=round(src.fps), manifest=a.manifest)
    stop = threading.Event()

    def capture():
        for f in src.frames():
            if stop.is_set():
                break
            rec.write(f)

    th = threading.Thread(target=capture, name='record-capture', daemon=True)
    th.start()
    print(f'{src.size[0]}x{src.size[1]} at {src.fps:g} fps -> {rec.dir}/{a.manifest}')
    print('Enter: start / stop.  r Enter: redo the last clip.  s Enter: skip.  q Enter: quit.')
    i, last = 0, None
    try:
        while i < len(prompts):
            utt_id, prompt = prompts[i]
            cmd = input(f'\n[{i + 1}/{len(prompts)}] {utt_id}: {prompt or "(rest clip)"}\n  Enter to start > ').strip()
            if cmd == 'q':
                break
            if cmd == 's':
                i += 1
                continue
            if cmd == 'r' and last is not None:
                i = last
                continue
            rec.start(utt_id, set=a.set, prompt='' if a.set == 'B' else prompt,
                      target_wpm=a.target_wpm, mouth=a.mouth, voiced=a.set == 'F')
            if a.seconds:
                print(f'  recording {a.seconds:g} s...')
                time.sleep(a.seconds)
            else:
                input('  recording... Enter to stop > ')
            e = rec.stop()
            print(f'  {e["n_frames"]} frames, {e["fps"]:g} fps, dropped {e["dropped"]}, '
                  f'{(e["clip_t1"] or 0) - (e["clip_t0"] or 0):.2f} s')
            last, i = i, i + 1
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        stop.set()
        rec.close()
        th.join(timeout=2)
        src.close()


if __name__ == '__main__':
    main()
