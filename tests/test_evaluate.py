"""fusion.evaluate on fake buslog sessions: WER, raw vs fixed, please repeat, delays, clocks, offline --decode.

    python -m pytest tests/test_evaluate.py -q
"""
import json

import numpy as np
import pytest

import buslog
from bus.team import encode
from fusion import decoder, evaluate, lock
from fusion.decoder import Decoder
from tests.test_fusion import U, frames_for, tick


def write_session(folder, records):
    """records: [(topic, msg, t_pub)] -> a buslog session folder."""
    w = buslog.Writer(folder)
    for n, (topic, msg, t_pub) in enumerate(records, 1):
        w.write(t_pub, 'dec' if topic.startswith(('dec.', 'voice.')) else topic.split('.')[0],
                encode(topic, msg, seq=n, t_pub=t_pub))
    w.close()
    return folder


def write_manifest(path, clips):
    path.write_text(''.join(json.dumps(c) + '\n' for c in clips))
    return path


CLIPS = [{'id': 'c1', 'set': 'C', 'prompt': 'I want water.', 'clip_t0': 10.0, 'clip_t1': 14.0},
         {'id': 'c2', 'set': 'C', 'prompt': 'Yes, please', 'clip_t0': 20.0, 'clip_t1': 24.0},
         {'id': 'r1', 'set': 'B', 'prompt': None, 'cue': 'swallow', 'clip_t0': 30.0, 'clip_t1': 32.0}]


def hyp(utt, locked, raw=None, offset=0, t0=None, unlocked=()):
    return ('dec.hyp', {'utt_id': utt, 'locked': locked, 'unlocked': list(unlocked), 'confidence': [],
                        'source': [], 'offset': offset, 'raw': raw or locked, 'spoken': [None] * len(locked),
                        't0': t0})


def lock_msg(utt, words):
    return ('dec.lock', {'utt_id': utt, 'words': [{'w': w, 'conf': 0.8, 'alts': [], 'cam': 0.9, 'emg': None, 't': t}
                                                  for w, t in words]})


def say(utt, words, final=False):
    return ('voice.say', {'utt_id': utt, 'words': words, 'final': final, 'pace': 1.0, 'effort': 1.0})


def logged_records():
    r = []
    add = lambda m, t: r.append((*m, t))
    add(hyp(1, [], t0=10.4, unlocked=['i']), 10.6)
    add(lock_msg(1, [('i', 10.8), ('want', 10.9)]), 11.0)
    add(hyp(1, ['i', 'want'], t0=10.4), 11.0)
    add(say(1, ['i', 'want']), 11.3)
    add(lock_msg(1, [('what', 11.2)]), 11.5)
    add(hyp(1, ['what'], offset=2, t0=10.4), 11.5)                    # only the newest locked word
    add(say(1, ['what'], final=True), 11.8)
    add(hyp(1, ['water'], raw=['what'], offset=2, t0=10.4), 12.4)     # a late repair: captions only
    add(hyp(2, [], t0=20.5), 20.7)
    add(lock_msg(2, [('yes', 20.8)]), 21.0)
    add(hyp(2, ['yes'], t0=20.5, unlocked=['fees']), 21.0)
    add(('dec.repeat', {'utt_id': 2, 'reason': {'reader': 'camera', 'words': []}}), 22.0)
    add(say(2, ['yes'], final=True), 22.1)
    add(hyp(3, ['hi'], t0=30.5), 30.9)                                # rest clip: a false word
    add(hyp(4, ['lost'], t0=50.0), 50.2)                              # outside every clip
    return r


def test_scores_logged_session(tmp_path):
    session = write_session(tmp_path / 's', logged_records())
    res = evaluate.score(evaluate.logged_events(session), evaluate.load_manifest(write_manifest(tmp_path / 'm', CLIPS)))
    c1, c2, r1 = res['clips']
    assert (c1['hyp'], c1['raw'], c1['wer'], c1['raw_wer']) == ('i want water', 'i want what', 0.0, pytest.approx(1 / 3))
    assert c2['wer'] == 0.5 and c2['repeats'] == 1 and r1['false_words'] == 1 and r1['wer'] is None
    o = res['overall']
    assert o['wer'] == pytest.approx(1 / 5) and o['raw_wer'] == pytest.approx(2 / 5)
    assert o['please_repeat_rate'] == pytest.approx(1 / 3) and o['utts'] == 3
    assert o['end_of_word_to_lock_ms']['n'] == 4 and o['end_of_word_to_lock_ms']['median'] == pytest.approx(200, abs=1e-6)
    assert o['lock_to_voice_ms']['n'] == 4 and o['lock_to_voice_ms']['mean'] == pytest.approx(500, abs=1e-6)   # .3 x 3, 1.1
    assert [u['utt_id'] for u in res['unmatched']] == [4]


def test_replayed_clock_is_offset_corrected(tmp_path):
    ft = 1000.0 + 0.04 * np.arange(10)                                # the recording's clock
    rec = [('cam.tick', {'seg_id': 7, 'frame_t': ft[:5]}, 10.5),       # first tick: offset = 10.5 - ft[4]
           ('cam.tick', {'seg_id': 7, 'frame_t': ft}, 10.7),
           (*hyp(1, ['i'], t0=float(ft[0])), 10.9),
           (*lock_msg(1, [('i', float(ft[3])), ('far', 5000.0)]), 10.9)]
    session = write_session(tmp_path / 's', rec)
    clips = [{'id': 'c', 'set': 'C', 'prompt': 'i', 'clip_t0': 999.5, 'clip_t1': 1001.0}]   # matched on t0
    res = evaluate.score(evaluate.logged_events(session), clips, evaluate.segment_clocks(session))
    o = res['overall']
    assert res['clips'][0]['wer'] == 0.0
    assert o['end_of_word_to_lock_offset_corrected'] == 1 and o['end_of_word_to_lock_clock_unknown'] == 1
    assert o['end_of_word_to_lock_ms']['median'] == pytest.approx((10.9 - (ft[3] + 10.5 - ft[4])) * 1000)


def test_edit_distance_and_norm():
    assert evaluate.edit_distance('a b c'.split(), 'a c'.split()) == 1
    assert evaluate.edit_distance([], ['x']) == 1 and evaluate.edit_distance(['x'], []) == 1
    assert evaluate.norm("Don't, STOP!") == ["don't", 'stop']


FPS_T = 0.04


def camera_records(t0=10.0):
    """cam.tick re-reads of "i want", ticks every 100 ms on the same clock as frame_t."""
    fr = frames_for([(2, 0.95), (3, 0.95)]) + [None] * 12
    out = [('cam.tick', tick(1, fr[:n], t0=t0), t0 + FPS_T * n + 0.05) for n in range(8, len(fr) + 1, 3)]
    out.append(('cam.final', {'seg_id': 1, 'nbest': [{'text': 'I WANT', 'score': -1.0}]}, out[-1][2] + 1.1))
    return out


def test_decode_offline(tmp_path):
    session = write_session(tmp_path / 's', camera_records())
    clips = [{'id': 'c', 'set': 'C', 'prompt': 'I want', 'clip_t0': 9.5, 'clip_t1': 12.0}]
    events = evaluate.decode_events(session, Decoder(U))
    res = evaluate.score(events, clips, evaluate.segment_clocks(session))
    assert res['clips'][0]['hyp'] == 'i want' and res['overall']['wer'] == 0.0
    says = [m for _, t, m in events if t == 'voice.say']
    assert says[-1]['final'] and [w for m in says for w in m['words']] == ['i', 'want']
    assert res['overall']['end_of_word_to_lock_offset_corrected'] == 0


def test_cli_logged_and_decode_with_override(tmp_path, monkeypatch, capsys):
    man = write_manifest(tmp_path / 'm.jsonl', CLIPS)
    out = tmp_path / 'r.json'
    res = evaluate.main([str(write_session(tmp_path / 's', logged_records())), '--manifest', str(man),
                         '--json', str(out)])
    assert json.loads(out.read_text())['overall']['wer'] == pytest.approx(res['overall']['wer'])
    assert 'WER 20.0%' in capsys.readouterr().out

    units = tmp_path / 'units.txt'
    units.write_text(''.join(f"{U.get(n, f'X{n}')} {n}\n" for n in range(1, 15)))
    monkeypatch.setattr(lock, 'THRESHOLD', lock.THRESHOLD)            # restored after the test
    cam = write_session(tmp_path / 'cam', camera_records())
    clips = write_manifest(tmp_path / 'c.jsonl', [{'id': 'c', 'set': 'C', 'prompt': 'I want', 'clip_t0': 9.5,
                                                   'clip_t1': 12.0}])
    res = evaluate.main([str(cam), '--manifest', str(clips), '--decode', '--units', str(units), '--no-lookalikes',
                         '--set', 'lock.THRESHOLD=5'])                 # nothing can lock
    assert res['mode'] == 'decode' and res['clips'][0]['hyp'] == '' and res['overall']['wer'] == 1.0
    with pytest.raises(SystemExit):
        evaluate.main([str(cam), '--manifest', str(clips), '--decode', '--set', 'lock.NOPE=1'])


def test_set_keeps_types_and_rejects_bad_values(monkeypatch):
    monkeypatch.setattr(decoder, 'HYP_LOCKED', decoder.HYP_LOCKED)
    monkeypatch.setattr(lock, 'THRESHOLD', lock.THRESHOLD)
    evaluate.apply_overrides(['decoder.HYP_LOCKED=3', 'lock.THRESHOLD=0.7'])
    assert decoder.HYP_LOCKED == 3 and type(decoder.HYP_LOCKED) is int and lock.THRESHOLD == 0.7
    for bad in ['decoder.HYP_LOCKED=2.5', 'lock.THRESHOLD=nan', 'lock.THRESHOLD=inf', 'lock.THRESHOLD=x',
                'lock.gate=1', 'lock.NOPE=1', 'nope.X=1', 'lock.THRESHOLD=']:
        with pytest.raises(SystemExit):
            evaluate.apply_overrides([bad])


def test_set_reaches_the_decoder(tmp_path, monkeypatch):
    monkeypatch.setattr(decoder, 'REPAIR_WINDOW_S', decoder.REPAIR_WINDOW_S)
    units = tmp_path / 'units.txt'
    units.write_text(''.join(f"{U.get(n, f'X{n}')} {n}\n" for n in range(1, 15)))
    cam = write_session(tmp_path / 'cam', camera_records())
    clips = write_manifest(tmp_path / 'c.jsonl', [{'id': 'c', 'set': 'C', 'prompt': 'I want', 'clip_t0': 9.5,
                                                   'clip_t1': 12.0}])
    args = [str(cam), '--manifest', str(clips), '--decode', '--units', str(units), '--no-lookalikes']
    fast = evaluate.main(args)['overall']['lock_to_voice_ms']
    slow = evaluate.main(args + ['--set', 'decoder.REPAIR_WINDOW_S=1.5'])['overall']['lock_to_voice_ms']
    assert fast['n'] == slow['n'] == 2 and fast['median'] < 400 and slow['median'] >= 1500


def test_utterance_with_missing_start_is_left_out_and_missed_clips(tmp_path, capsys):
    rec = [(*hyp(1, ['water'], offset=2, t0=10.4), 11.5),            # the log began after words 0 and 1
           (*hyp(2, ['yes'], t0=30.5), 30.9)]
    session = write_session(tmp_path / 's', rec)
    res = evaluate.score(evaluate.logged_events(session), CLIPS)
    assert res['skipped_utts'] == [1] and 'left out' in capsys.readouterr().err
    assert res['missed_clips'] == ['c1', 'c2'] and res['overall']['missed_clips'] == 2
    assert res['overall']['wer'] == 1.0                               # missed clips still count in full
