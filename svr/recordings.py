"""Our wearer's recordings: the manifest format and a loader shared by the EMG and camera pipelines.

Pure json + numpy so both the main venv and the camera venv can import it.

Layout (written by recorder/record.py, the Reclaim recorder; session folders are named session_01, session_02, ...):

    recordings/<session>/manifest.jsonl          one JSON object per utterance
    recordings/<session>/<id>_emg.npy            (samples, 4), 1000 Hz, raw unfiltered Uno readings (uint16, 0-1023),
                                                 columns = rig pins A0..A3 = Gaddy sites 3, 2, 6, 1
    recordings/<session>/<id>_emg_ts.npy         float64 (samples,): time.monotonic() of each EMG sample
    recordings/<session>/<id>.mp4                webcam video of the same utterance
    recordings/<session>/<id>_video_ts.npy       per-frame timestamps (s), time.monotonic(), the same clock as the EMG

Manifest fields:
    id          unique utterance id, e.g. "s1_C_0042"
    session     recording session, e.g. "s1" (electrodes re-applied between sessions)
    set         A fixed sentences | B rest/junk | C prompted free speech | D pitch script | E held-out test | F voiced
    prompt      the sentence shown to the wearer ("" for rest/junk)
    target_wpm  pace shown on screen (100, 150, 180...), null when unpaced
    mouth       "open" or "closed"
    voiced      true when read aloud (set F)
    emg         path of the EMG array, relative to the session folder
    emg_ts      path of the per-sample EMG times, relative (null for older recordings)
    pace        natural | fast | demo | rest (recorder/record.py)
    video       path of the video, relative to the session folder (null if none)
    video_ts    path of per-frame timestamps, relative (null: assume constant fps from the file)
    t_start     speech onset in seconds from the start of the clip (null: whole clip)
    t_end       speech offset in seconds (null: whole clip)
    audio       optional path of the microphone recording (set F)
"""
import json
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
WORK = os.environ.get('RECLAIM_WORKDIR', ROOT)  # datasets, checkpoints, results: outside the repo
RECORDINGS = os.path.join(WORK, 'recordings')
EMG_RATE = 1000


def load_manifest(root=RECORDINGS, sets=None, sessions=None, mouth=None):
    """All utterances under root, with absolute paths and derived timing fields."""
    out = []
    for session in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        path = os.path.join(root, session, 'manifest.jsonl')
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if sets and r.get('set') not in sets:
                    continue
                if sessions and r.get('session', session) not in sessions:
                    continue
                if mouth and r.get('mouth') != mouth:
                    continue
                base = os.path.join(root, session)
                for key in ('emg', 'emg_ts', 'video', 'video_ts', 'audio'):
                    if r.get(key):
                        r[key] = os.path.join(base, r[key])
                r.setdefault('session', session)
                out.append(r)
    return out


def speech_window(r, clip_seconds):
    t0 = r.get('t_start') if r.get('t_start') is not None else 0.0
    t1 = r.get('t_end') if r.get('t_end') is not None else clip_seconds
    return t0, max(t1, t0 + 1e-3)


def load_emg(r):
    """(samples, 6) float64 EMG of one utterance and its speech window (s)."""
    x = np.load(r['emg']).astype(np.float64)
    return x, speech_window(r, len(x) / EMG_RATE)


def spoken_wpm(r, duration_s):
    words = len(r.get('prompt', '').split())
    return 60.0 * words / duration_s if duration_s > 0 else 0.0
