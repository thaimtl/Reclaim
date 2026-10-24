"""Recorder (E7): files in svr/recordings.py's layout, 1:1 frames and timestamps, a non-blocking write, the EMG merge.

    venv-live/bin/python -m pytest tests/live/test_record.py -q -s     # -s prints the measured crop difference
"""
import json
import os
import subprocess
import sys
import time

import cv2
import numpy as np
import pytest

import live
import live.record as record
from live.record import MANIFEST, VIDEO_MANIFEST, Recorder, merge_manifests, read_prompts
from live.source import ReplaySource
from live.types import Frame
from svr.recordings import load_manifest

from .conftest import ROOT, needs_sample

FIELDS = ('id', 'session', 'set', 'prompt', 'target_wpm', 'mouth', 'voiced', 'emg', 'video', 'video_ts', 't_start',
          't_end')


def read_video(path):
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        out.append(im)
    cap.release()
    return out


def level_frame(i, t, size=(48, 64)):
    """Flat frame whose grey level encodes i, to tell which frames reached the file."""
    return Frame(t=t, image=np.full(size + (3,), 10 + 7 * (i % 34), np.uint8), index=i)


def level_of(im):
    return int(round((im.mean() - 10) / 7))


@pytest.fixture(scope='module')
def clip(sample_video):
    src = ReplaySource(sample_video, live.SAMPLE_START, live.SAMPLE_START + 3.0, realtime=False)
    out = list(src.frames())
    src.close()
    return out


@needs_sample
def test_round_trip_through_load_manifest(clip, tmp_path):
    t0 = time.monotonic()
    written = [Frame(t=t0 + f.t, image=f.image, index=f.index) for f in clip]
    with Recorder(str(tmp_path), 's1') as rec:
        rec.start('s1_001', set='C', prompt="Hi, it's nice to meet you.")
        assert all(rec.write(f) for f in written)
        entry = rec.stop()
    rows = load_manifest(str(tmp_path))
    assert len(rows) == 1
    r = rows[0]
    assert all(k in r for k in FIELDS)
    assert (r['id'], r['session'], r['set'], r['mouth'], r['voiced'], r['emg']) == ('s1_001', 's1', 'C', 'open', False,
                                                                                    None)
    assert r['video'] == str(tmp_path / 's1' / 's1_001.mp4') and os.path.exists(r['video'])
    ts = np.load(r['video_ts'])
    assert ts.dtype == np.float64 and np.all(np.diff(ts) > 0)
    np.testing.assert_array_equal(ts, [f.t for f in written])
    assert entry['n_frames'] == r['n_frames'] == len(ts) == len(written) and entry['dropped'] == 0
    assert entry['clip_t0'] == ts[0] and entry['clip_t1'] == ts[-1] and abs(entry['fps'] - 25) < 0.1
    dec = read_video(r['video'])
    assert len(dec) == len(ts) and dec[0].shape == clip[0].image.shape  # full size, as captured
    diff = np.mean([np.abs(a.astype(np.int16) - b.image).mean() for a, b in zip(dec, clip)])
    assert diff < 3.0, diff
    src = ReplaySource(r['video'], realtime=False)
    assert abs(src.src_fps - 25) < 0.5 and len(list(src.frames())) == len(ts)
    src.close()


@needs_sample
def test_crops_of_recording_match_original(clip, tmp_path, face_model):
    from live.crop import LiveCropper
    with Recorder(str(tmp_path), 's1') as rec:
        rec.start('u', set='C')
        for f in clip:
            rec.write(Frame(t=100 + f.t, image=f.image, index=f.index))
        rec.stop()

    def crops(images):
        c = LiveCropper(face_model)
        out = []
        for i, im in enumerate(images):
            out += c.push(Frame(t=i / 25, image=im, index=i))
        out += c.flush()
        c.close()
        return np.stack([x.roi96 for x in out]).astype(np.int16)

    d = np.abs(crops(read_video(str(tmp_path / 's1' / 'u.mp4'))) - crops([f.image for f in clip])).mean(axis=(1, 2))
    print(f'\ncrop mean abs diff recorded vs original: mean {d.mean():.2f} p95 {np.percentile(d, 95):.2f} (0-255)')
    assert d.mean() < 3.5 and np.percentile(d, 95) < 5.0


def test_write_never_blocks_and_drops_stay_consistent(tmp_path, monkeypatch):
    real = record.to_video_frame

    def slow(image):
        time.sleep(0.02)
        return real(image)

    monkeypatch.setattr(record, 'to_video_frame', slow)
    rec = Recorder(str(tmp_path), 's1', queue_size=2)
    rec.start('u', set='B')
    kept, worst = [], 0.0
    for i in range(60):
        f = level_frame(i, 50.0 + i / 30)
        w0 = time.perf_counter()
        if rec.write(f):
            kept.append(f)
        worst = max(worst, time.perf_counter() - w0)
        time.sleep(0.001)
    e = rec.stop()
    rec.close()
    assert worst < 0.005, worst
    assert e['dropped'] > 0 and e['dropped'] == rec.dropped and e['n_frames'] + e['dropped'] == 60
    ts = np.load(tmp_path / 's1' / 'u_video_ts.npy')
    np.testing.assert_array_equal(ts, [f.t for f in kept])
    dec = read_video(str(tmp_path / 's1' / 'u.mp4'))
    assert len(dec) == len(ts) == e['n_frames']
    assert [level_of(im) for im in dec] == [f.index % 34 for f in kept]  # the same frames in video and ts


def test_repeats_skipped_idle_ignored_empty_clip(tmp_path):
    with Recorder(str(tmp_path), 's1') as rec:
        assert not rec.write(level_frame(0, 1.0))           # not recording
        rec.start('a', set='C')
        assert rec.write(level_frame(0, 1.0))
        assert not rec.write(level_frame(0, 1.0))           # CameraSource's repeated frame
        assert rec.write(level_frame(1, 1.04))
        e = rec.stop()
        assert (e['n_frames'], e['repeats'], e['dropped']) == (2, 1, 0)
        rec.start('b', set='B')
        e = rec.stop()
        assert e['n_frames'] == 0 and e['video'] is None and e['video_ts'] is None
    rows = load_manifest(str(tmp_path))
    assert [r['id'] for r in rows] == ['a', 'b'] and rows[1]['video'] is None


def test_redo_replaces_line_and_files(tmp_path):
    with Recorder(str(tmp_path), 's1') as rec:
        for take, n in ((1, 5), (2, 8)):
            rec.start('s1_007', set='C', prompt=f'take {take}')
            for i in range(n):
                rec.write(level_frame(i, 10.0 * take + i / 30))
            rec.stop()
    rows = load_manifest(str(tmp_path))
    assert len(rows) == 1 and rows[0]['prompt'] == 'take 2' and rows[0]['n_frames'] == 8
    assert len(np.load(rows[0]['video_ts'])) == len(read_video(rows[0]['video'])) == 8


def test_merge_with_emg_manifest(tmp_path):
    d = tmp_path / 's1'
    with Recorder(str(tmp_path), 's1', manifest=VIDEO_MANIFEST) as rec:
        for k, uid in enumerate(('s1_001', 's1_002')):
            rec.start(uid, set='C', prompt='camera prompt')
            for i in range(4):
                rec.write(level_frame(i, 100.0 * (k + 1) + i / 30))
            rec.stop()
    emg = [{'id': 's1_001', 'session': 's1', 'set': 'C', 'prompt': 'emg prompt', 'target_wpm': None, 'mouth': 'open',
            'voiced': False, 'emg': 's1_001_emg.npy', 'video': None, 'video_ts': None, 't_start': None, 't_end': None},
           {'id': 's1_003', 'session': 's1', 'set': 'C', 'prompt': 'x', 'emg': 's1_003_emg.npy'}]
    (d / MANIFEST).write_text(''.join(json.dumps(r) + '\n' for r in emg))
    assert merge_manifests(str(d)) == (1, 1, 1)
    rows = {r['id']: r for r in load_manifest(str(tmp_path))}
    assert sorted(rows) == ['s1_001', 's1_002', 's1_003']
    r = rows['s1_001']
    assert r['emg'] == str(d / 's1_001_emg.npy') and r['video'] == str(d / 's1_001.mp4')
    assert r['prompt'] == 'emg prompt' and r['n_frames'] == 4 and r['clip_t0'] == 100.0
    assert rows['s1_002']['emg'] is None and rows['s1_003'].get('video') is None


def test_read_prompts(tmp_path):
    p = tmp_path / 'p.txt'
    p.write_text("s1_001\tHi, it's nice to meet you.\n\n# comment\nrest_01\n")
    assert read_prompts(str(p)) == [('s1_001', "Hi, it's nice to meet you."), ('rest_01', '')]
    # the team's prompt files: recording_prompts/ in the working folder, docs/recording/ in the repo
    real = [f for f in (os.path.join(ROOT, 'recording_prompts', 'session1.txt'),
                        os.path.join(ROOT, 'docs', 'recording', 'session1.txt')) if os.path.exists(f)]
    if real:
        assert read_prompts(real[0])[0][0] == 's1_001'


@needs_sample
def test_cli_replay(tmp_path):
    prompts = tmp_path / 'p.txt'
    prompts.write_text('s1_001\tHi there.\ns1_002\tSecond one.\n')
    cmd = [sys.executable, '-m', 'live.record', '--session', 's1', '--set', 'C', '--prompts', str(prompts),
           '--replay', live.SAMPLE_VIDEO, '--root', str(tmp_path / 'rec'), '--seconds', '0.6']
    res = subprocess.run(cmd, input='\n\n', capture_output=True, text=True, cwd=ROOT, timeout=60)
    assert res.returncode == 0, res.stderr
    rows = load_manifest(str(tmp_path / 'rec'))
    assert [r['id'] for r in rows] == ['s1_001', 's1_002']
    for r in rows:
        ts = np.load(r['video_ts'])
        assert r['n_frames'] == len(ts) == len(read_video(r['video'])) > 10 and np.all(np.diff(ts) > 0)
        assert (r['width'], r['height']) == (640, 360) and abs(r['fps'] - 29.97) < 2


def test_failed_redo_keeps_the_previous_take(tmp_path, monkeypatch):
    with Recorder(str(tmp_path), 's1') as rec:
        rec.start('s1_007', set='C', prompt='take 1')
        for i in range(5):
            rec.write(level_frame(i, 10.0 + i / 30))
        rec.stop()
        good = record.to_video_frame
        calls = []

        def flaky(image):
            calls.append(1)
            if len(calls) > 3:
                raise OSError('disk full')
            return good(image)
        monkeypatch.setattr(record, 'to_video_frame', flaky)
        rec.start('s1_007', set='C', prompt='take 2')
        for i in range(8):
            rec.write(level_frame(i, 20.0 + i / 30))
        with pytest.raises(RuntimeError):
            rec.stop()
    rows = load_manifest(str(tmp_path))
    assert len(rows) == 1 and rows[0]['prompt'] == 'take 1' and rows[0]['n_frames'] == 5
    assert len(np.load(rows[0]['video_ts'])) == len(read_video(rows[0]['video'])) == 5
    assert sorted(os.listdir(tmp_path / 's1')) == ['manifest.jsonl', 's1_007.mp4', 's1_007_video_ts.npy']


def test_an_empty_redo_keeps_the_good_take(tmp_path):
    with Recorder(str(tmp_path), 's1') as rec:
        rec.start('u1')
        for i in range(5):
            rec.write(level_frame(i, 1.0 + i / 25))
        good = rec.stop()
        rec.start('u1')                                      # redo, but the camera delivers nothing
        assert rec.stop()['n_frames'] == 0
    rows = load_manifest(str(tmp_path))
    assert len(rows) == 1 and rows[0]['n_frames'] == 5 and good['video'] == 'u1.mp4'  # load_manifest makes paths absolute
    d = tmp_path / 's1'
    assert (d / 'u1.mp4').exists() and len(np.load(d / 'u1_video_ts.npy')) == 5
    assert not list(d.glob('*.part.*'))
