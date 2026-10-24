"""Recorder tests that need no hardware: a fake camera and fake EMG.

    python recorder/test_record.py      (or: python -m pytest recorder/test_record.py)
"""
import json
import os
import sys
import tempfile

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import record  # noqa: E402
from svr.recordings import load_emg, load_manifest  # noqa: E402  (the loader fine-tuning uses)


def test_clock_fit_handles_jitter_and_drift():
    """Sample times from (count, arrival) pairs: USB latency 1-15 ms, Uno clock 0.5% fast."""
    rng = np.random.default_rng(1)
    buf = record.EmgBuffer()
    true_rate = record.FS * 1.005
    n, t = 0, 100.0
    for _ in range(3000):
        k = int(rng.integers(5, 20))
        n += k
        buf.push(np.full((k, 4), 512), t0 := 100.0 + n / true_rate + rng.uniform(0.001, 0.015))
        t = t0
    a, b = buf.clock()
    worst = max(abs((a + b * m) - (100.0 + m / true_rate)) for m in (0, n // 2, n))
    assert worst < 0.003, worst  # the line follows the earliest arrivals, within 3 ms


def test_dropped_frames_keep_the_sample_count():
    buf = record.EmgBuffer()
    buf.push(np.full((10, 4), 100), 1.0)
    buf.push(np.full((10, 4), 200), 1.03, dropped=5)
    assert buf.n == 25 and buf.dropped == 5


def test_items_are_shuffled_the_same_way_every_run():
    a, b = record.load_items(1), record.load_items(1)
    assert [i['id'] for i in a] == [i['id'] for i in b]
    assert len(a) == 260 and sum(i['set'] == 'B' for i in a) == 10
    assert {i['pace'] for i in a} == {'natural', 'fast', 'slow', 'rest'}
    assert all(i['set'] == 'E' for i in record.load_items(3) if i['pace'] != 'rest')


def test_delivery_styles():
    """Each session gets its STYLES on a fixed set of natural- (or demo-) pace sentences, one style per sentence."""
    from collections import Counter
    for session in (1, 2, 3):
        items = record.load_items(session)
        styled = [i for i in items if i.get('style')]
        assert Counter(i['style'] for i in styled) == Counter(record.STYLES[session])
        assert [i['id'] for i in styled] == [i['id'] for i in record.load_items(session) if i.get('style')]
        for i in items:
            if i['pace'] == 'rest':
                assert i['expression'] is None and i['effort'] is None and not i.get('style')
                continue
            assert i['expression'] in ('neutral',) + record.EXPRESSIONS and i['effort'] in ('normal',) + record.EFFORTS
            assert (i['pace'] == 'slow') == (i.get('style') == 'slow')
            assert sum([i['pace'] == 'slow', i['expression'] != 'neutral', i['effort'] != 'normal']) <= 1
            if i['pace'] == 'fast':
                assert not i.get('style'), 'fast sentences stay neutral'
    n1 = {i['id'].split('_')[1] for i in record.load_items(1) if i.get('style')}
    n2 = {i['id'].split('_')[1] for i in record.load_items(2) if i.get('style')}
    assert n1 != n2, 'each session seeds its own picks'
    assert set(record.STYLE_CUES) >= {s for st in record.STYLES.values() for s in st}
    for cue in record.STYLE_CUES.values():      # stays left of the key-help panel (x = 960)
        (w, _), _ = cv2.getTextSize(cue, cv2.FONT_HERSHEY_SIMPLEX, record.CUE_SCALE, 2)
        assert 30 + w < 950, (cue, w)
    import textwrap
    for s in (1, 2, 3):                         # with the cue line, a prompt has room for 2 wrapped lines
        assert max(len(textwrap.wrap(i['prompt'], 42)) for i in record.load_items(s) if i['prompt']) <= 2
    assert sum(1 for i in record.load_items(1) if i['pace'] == 'natural' and i['expression'] == 'neutral'
               and i['effort'] == 'normal') == 130


def test_record_redo_resume_and_load():
    root = tempfile.mkdtemp()
    args = ['--session', '1', '--camera', 'fake', '--fake-emg', '--allow-charger', '--root', root,
            '--width', '320', '--height', '240', '--auto', '3', '--clip-seconds', '1.5']
    sess = record.main(args)
    rows = load_manifest(root)
    assert len(rows) == 3, rows
    for r in rows:
        x, _ = load_emg(r)
        ts = np.load(r['emg_ts'])
        vts = np.load(r['video_ts'])
        assert x.shape[1] == 4 and abs(len(x) - 1500) < 60, x.shape
        assert len(ts) == len(x) and np.all(np.diff(ts) > 0)
        assert abs(np.median(np.diff(ts)) - 1e-3) < 2e-5
        assert ts[0] >= r['clip_t0'] - 0.002 and ts[-1] <= r['clip_t1'] + 0.002
        assert vts[0] >= r['clip_t0'] and vts[-1] <= r['clip_t1'] and len(vts) > 30
        cap = cv2.VideoCapture(r['video'])
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == len(vts)
        cap.release()
        assert r['warnings'] == [], r['warnings']
        assert 'expression' in r and 'effort' in r     # delivery style saved with the clip
    meta = json.load(open(os.path.join(sess.dir, 'meta.json')))
    assert meta['recorder_version'] == record.VERSION and len(meta['runs']) >= 1
    redone = sess.last_id
    sess.redo(redone)
    assert len(load_manifest(root)) == 2
    assert not os.path.exists(os.path.join(sess.dir, redone + '.mp4'))
    sess2 = record.main(args[:-4] + ['--auto', '1', '--clip-seconds', '1.0'])
    ids = {r['id'] for r in load_manifest(root)}
    assert len(ids) == 3 and redone in ids  # resume records the next pending clip, which is the one redone
    assert sess2.dir == sess.dir
    assert len(json.load(open(os.path.join(sess.dir, 'meta.json')))['runs']) == 2
    open(os.path.join(sess.dir, 'meta.json'), 'w').write('{"sess')     # cut short by a crash
    record.main(args[:-4] + ['--auto', '0'])                            # still resumes


def test_window_draws():
    """The screen renders in every state (headless: drawn to an array, not shown)."""
    root = tempfile.mkdtemp()
    sess = record.main(['--session', '2', '--camera', 'fake', '--fake-emg', '--allow-charger', '--root', root,
                        '--width', '320', '--height', '240', '--auto', '0'])
    item = sess.pending()[0]
    styled = next(i for i in sess.pending() if i.get('style'))
    claude = type('T', (), {'error': None})()
    img = record.draw(sess, sess.cam, sess.emg, claude, 'idle', styled, True, '', False, False)
    cv2.imwrite(os.path.join(root, 'screen_styled.png'), img)
    for state in ('idle', 'done'):
        img = record.draw(sess, sess.cam, sess.emg, type('T', (), {'error': None})(), state, item, True, 'msg',
                          charger=True, real_emg=True)
        assert img.shape == (820, 1280, 3)
    sess.start(item)
    img = record.draw(sess, sess.cam, sess.emg, type('T', (), {'error': None})(), 'rec', item, False, '', False, False)
    sess.stop()
    cv2.imwrite(os.path.join(root, 'screen.png'), img)


if __name__ == '__main__':
    test_window_draws()
    test_clock_fit_handles_jitter_and_drift()
    test_dropped_frames_keep_the_sample_count()
    test_items_are_shuffled_the_same_way_every_run()
    test_delivery_styles()
    test_record_redo_resume_and_load()
    print('ok')
