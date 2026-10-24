"""Video -> 96x96 grayscale mouth crops at 25 fps, plus per-frame mouth opening, cached per utterance.

Uses Lipflow's MediaPipe face tracking and Auto-AVSR-identical mouth alignment (third_party/lipflow, MIT), so every
camera system sees exactly the same crops. Mouth opening = inner-lip gap / mouth width (Lipflow's FaceObs.mouth_open);
it also feeds our reliability gate.

    venv-camera/bin/python -m camera.mouth [--sets C,E,A] [--force]
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
WORK = os.environ.get('RECLAIM_WORKDIR', ROOT)  # datasets, checkpoints, results: outside the repo
LIPFLOW = os.path.join(WORK, 'third_party', 'lipflow')
sys.path.insert(0, LIPFLOW)
sys.path.insert(0, ROOT)
from lipflow.face import FaceTracker, mouth_rois  # noqa: E402

from svr.recordings import load_manifest  # noqa: E402

CACHE = os.path.join(WORK, 'data', 'camera_cache')
MODEL_FPS = 25
FACE_MODEL = os.path.join(LIPFLOW, 'models', 'face_landmarker.task')


def read_video(path, ts_path=None):
    """Frames as (timestamps in s, grayscale frames, BGR frames are not kept)."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    ts = np.load(ts_path).astype(np.float64) if ts_path else np.arange(len(frames)) / fps
    ts = ts[:len(frames)] - (ts[0] if len(ts) else 0.0)
    return ts, frames


def resample_idx(ts, n):
    """Indices that turn a variable-rate capture into a steady 25 fps sequence (as Lipflow's LipReader)."""
    if n == 0:
        return []
    grid = np.arange(ts[0], ts[-1] + 1e-9, 1.0 / MODEL_FPS)
    return np.clip(np.searchsorted(ts, grid), 0, n - 1).tolist()


def process(r, tracker):
    ts, frames = read_video(r['video'], r.get('video_ts'))
    anchors, opening, grays = [], [], []
    for t, frame in zip(ts, frames):
        obs = tracker.detect(frame, int(t * 1000))
        anchors.append(obs.anchors if obs else None)
        opening.append(obs.mouth_open if obs else np.nan)
        grays.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    idx = resample_idx(ts, len(frames))
    rois = mouth_rois([grays[i] for i in idx], [anchors[i] for i in idx])
    return {
        'rois': rois if rois is not None else np.zeros((0, 96, 96), np.uint8),
        't': np.asarray([ts[i] for i in idx], dtype=np.float64),
        'mouth_open': np.asarray([opening[i] for i in idx], dtype=np.float32),
        'face_found': np.asarray([anchors[i] is not None for i in idx]),
        'source_fps': np.float64(len(frames) / max(ts[-1] - ts[0], 1e-6)) if len(frames) > 1 else np.float64(0),
    }


def cache_path(uid):
    return os.path.join(CACHE, f'{uid}.npz')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sets', default='A,C,D,E')
    p.add_argument('--force', action='store_true')
    p.add_argument('--recordings', default=None)
    a = p.parse_args()
    os.makedirs(CACHE, exist_ok=True)
    root = {'root': a.recordings} if a.recordings else {}
    records = [r for r in load_manifest(sets=a.sets.split(','), **root) if r.get('video')]
    print(f'{len(records)} utterances with video', flush=True)
    for n, r in enumerate(records, 1):
        out = cache_path(r['id'])
        if os.path.exists(out) and not a.force:
            continue
        t0 = time.perf_counter()
        tracker = FaceTracker(FACE_MODEL)  # fresh tracker per clip: MediaPipe VIDEO mode needs increasing time
        d = process(r, tracker)
        tracker.close()
        d['preprocess_ms'] = np.float64((time.perf_counter() - t0) * 1000)
        np.savez_compressed(out, **d)
        if n % 25 == 0:
            print(f'  {n}/{len(records)}', flush=True)


if __name__ == '__main__':
    main()
