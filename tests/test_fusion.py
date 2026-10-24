"""Fusion decoder: word extraction, stability, locking, repair window, 8 s cap overlap, please repeat, bus round trip.

Fake cam.tick messages built with numpy, no model:  python -m pytest tests/test_fusion.py -q
"""
import os
import threading
import time

import numpy as np
import pytest
import zmq

from bench.lookalikes import Lexicon
import bus
from bus.team import Pub, Sub
from fusion import decoder, lock
from fusion.decoder import REPAIR_WINDOW_S, Decoder, cut
from fusion.run import TOPICS, newest_ticks, serve
from fusion.words import SPM, UNITS, LookAlikes, Word, extract_words, load_units, load_word_logfreq

U = {1: '<unk>', 2: '▁I', 3: '▁WANT', 4: '▁WA', 5: 'TER', 6: '▁WHAT', 7: '▁BAT', 8: '▁MAT', 9: '▁EYE',
     10: '▁YES', 11: '▁PLEASE', 12: '▁HELLO', 13: '▁THERE', 14: '▁NOW'}
FPS = 0.04


def frames_for(pieces, start=3, gap=6, run=2):
    """[(id, p) or (id, p, [(alt id, p)]), ...] -> per-frame entries: None = blank, else the piece."""
    out = [None] * start
    for pc in pieces:
        out += [pc] * run + [None] * gap
    return out


def tick(seg, frames, t0=100.0, face_ok=True, quiet_tail=0):
    """A cam.tick over these frames. Activity is high around pieces, low elsewhere."""
    frames = list(frames) + [None] * quiet_tail
    T = len(frames)
    ids = np.full((T, 32), 0, np.int32)
    logp = np.full((T, 32), np.log(1e-4), np.float32)
    blank = np.full(T, np.log(0.9), np.float32)
    act = np.full(T, 0.05, np.float32)
    for t, fr in enumerate(frames):
        cands = [(0, 0.9)] if fr is None else [(fr[0], fr[1]), (0, (1 - fr[1]) / 2)] + list(fr[2] if len(fr) > 2 else [])
        if fr is not None:
            blank[t] = np.log((1 - fr[1]) / 2)
            act[max(0, t - 3):t + 4] = 0.6
        cands.sort(key=lambda c: -c[1])
        for k, (i, p) in enumerate(cands):
            ids[t, k], logp[t, k] = i, np.log(p)
        ids[t, len(cands):] = np.arange(100, 100 + 32 - len(cands))
    return {'seg_id': seg, 'seg_start': t0, 'frame_t': t0 + FPS * np.arange(T), 'topk_ids': ids, 'topk_logp': logp,
            'blank_logp': blank, 'greedy': '', 'activity': act, 'face_ok': face_ok,
            'emotion': {'label': 'neutral', 'confidence': 1.0, 'scores': {e: float(e == 'neutral') for e in bus.EMOTIONS}},
            'timing_ms': bus.zero_timing(T)}


def topics(out, name):
    return [m for t, m in out if t == name]


def all_words(dec, utt_id=None):
    utt = dec.utts[utt_id or dec.cur.utt_id]
    return [w.w for w in utt.locked + utt.unlocked]


def feed(dec, frames, steps, seg=1, now=0.0, dt=0.1, t0=100.0, **kw):
    """Ticks re-reading frames[:n] for each n in steps; returns (all outputs, the clock after)."""
    out = []
    for n in steps:
        out += dec.on_tick(tick(seg, frames[:n], t0=t0, **kw), now)
        out += dec.pump(now)
        now += dt
    return out, now


# --- words ---------------------------------------------------------------------------------------------------

def test_words_from_tick():
    fr = frames_for([(2, 0.95), (3, 0.8, [(6, 0.1)]), (4, 0.9), (5, 0.7)])
    words = extract_words(tick(1, fr), U)
    assert [w.w for w in words] == ['i', 'want', 'water']
    i, want, water = words
    assert (i.f0, i.f1) == (3, 4) and i.t0 == pytest.approx(100 + 3 * FPS)
    assert i.cam == pytest.approx(0.95, abs=1e-5)
    assert water.cam == pytest.approx(0.7, abs=1e-5)       # least sure piece
    assert water.f0 == 19 and water.f1 == 28                # WA starts, TER ends
    assert [a for a, _ in want.alts] == ['what'] and want.alts[0][1] == pytest.approx(0.1, abs=1e-5)


def test_greedy_blank_beats_topk_and_repeats_split_by_blank():
    m = tick(1, frames_for([(2, 0.9), (2, 0.9)]))
    m['blank_logp'][3] = 0.0  # blank beats the top-k on frame 3: the first run is one frame
    words = extract_words(m, U)
    assert [w.w for w in words] == ['i', 'i'] and words[0].f0 == words[0].f1 == 4


def test_lookalikes_in_alts():
    lex = Lexicon(['bat', 'mat', 'want'], pron={'bat': ['B', 'AE', 'T'], 'mat': ['M', 'AE', 'T'], 'want': ['W']})
    words = extract_words(tick(1, frames_for([(7, 0.9)])), U, LookAlikes(lex, {}))
    assert words[0].alts == [('mat', pytest.approx(0.9, abs=1e-5))]


def test_alts_rank_camera_first_then_common_lookalikes():
    vocab = ['fine', 'vine', 'fight', 'file', 'fail', 'fan', 'van', 'feel', 'fit', 'what']
    lex = Lexicon(vocab, pron={'fine': ['F', 'AY', 'N'], 'vine': ['V', 'AY', 'N'], 'fight': ['F', 'AY', 'T'],
                               'file': ['F', 'AY', 'L'], 'fail': ['F', 'EY', 'L'], 'fan': ['F', 'AE', 'N'],
                               'van': ['V', 'AE', 'N'], 'feel': ['F', 'IY', 'L'], 'fit': ['F', 'IH', 'T'],
                               'what': ['W', 'AH', 'T']})
    freq = {'feel': -7.4, 'fit': -9.2, 'fail': -9.7, 'fight': -9.9, 'fan': -10.3, 'van': -10.5, 'vine': -12.0}
    la = LookAlikes(lex, freq)
    assert la('fine')[:3] == ('feel', 'fit', 'fail') and la('fine')[-1] == 'file'   # unknown frequency last
    units = {**U, 20: '\u2581FINE'}
    words = extract_words(tick(1, frames_for([(20, 0.8, [(6, 0.1)])])), units, la)
    assert [a for a, _ in words[0].alts] == ['what', 'feel', 'fit', 'fail', 'fight', 'fan']   # camera first, cap 6


def test_word_frequencies_from_spm_model():
    if not os.path.exists(SPM):
        pytest.skip('unigram5000.model not under RECLAIM_WORKDIR')
    f = load_word_logfreq()
    assert f['the'] > f['water'] > f['fine'] > f['van']
    try:
        import sentencepiece
    except ImportError:
        return
    m = sentencepiece.SentencePieceProcessor(model_file=SPM)
    assert f['water'] == pytest.approx(m.get_score(m.piece_to_id('\u2581WATER')))


def test_real_units_file():
    if not os.path.exists(UNITS):
        pytest.skip('units file not under RECLAIM_WORKDIR')
    u = load_units()
    assert u[4899] == '▁WATER' and u[1] == '<unk>' and 0 not in u


# --- stability and the pinned prefix -------------------------------------------------------------------------

def test_passes_carry_and_reset():
    dec = Decoder(U)
    fr = frames_for([(2, 0.5), (3, 0.5)])   # too unsure to lock: stays unlocked
    feed(dec, fr, [len(fr)] * 3)
    utt = dec.cur
    assert [w.passes for w in utt.unlocked] == [2, 2]
    fr2 = frames_for([(2, 0.5), (6, 0.5)])  # "want" re-read as "what"
    dec.on_tick(tick(1, fr2), 1.0)
    assert [(w.w, w.passes) for w in utt.unlocked] == [('i', 3), ('what', 0)]


def test_lock_after_passes_and_locked_prefix_never_rewritten():
    dec = Decoder(U)
    fr = frames_for([(2, 0.9), (3, 0.9)])
    out, now = feed(dec, fr, [12, 14, 16])
    locks = topics(out, 'dec.lock')
    assert [w['w'] for m in locks for w in m['words']] == ['i']      # 3rd pass: passes 2 -> 0.73 >= 0.72
    assert locks[0]['words'][0]['emg'] is None and locks[0]['words'][0]['t'] == pytest.approx(100 + 4 * FPS)
    rewrite = frames_for([(9, 0.9), (3, 0.9)])                       # the next pass reads "eye want"
    dec.on_tick(tick(1, rewrite), now)
    assert all_words(dec) == ['i', 'want']
    hyp = topics(dec.on_tick(tick(1, rewrite), now + 0.1), 'dec.hyp')[-1]
    assert hyp['locked'][0] == 'i' and 'eye' not in hyp['unlocked']


def test_cut_by_text_when_spike_shifts_late():
    p = Word('i', 3, 4, 1.0, 1.04, 0.9)
    shifted = [Word('i', 5, 6, 1.08, 1.12, 0.9), Word('want', 12, 13, 1.4, 1.44, 0.9)]
    assert cut([p], shifted, 0.0) == 1


def test_unfinished_word_does_not_lock():
    dec = Decoder(U)
    fr = frames_for([(2, 0.99)], gap=0)       # the last word, lips still moving after it
    m = tick(1, fr)
    m['activity'][:] = 0.6
    for k in range(6):
        dec.on_tick(m, k * 0.1)
    assert dec.cur.locked == []


# --- lock scoring --------------------------------------------------------------------------------------------

def test_claude_alone_never_locks():
    m = lock.LockModel()
    for cam in np.linspace(0, lock.SENSOR_MIN - 1e-3, 10):
        for claude in (0.0, 0.5, 1.0):
            f = {'cam': cam, 'emg': None, 'claude': claude, 'finished': True, 'passes': 99, 'face_ok': True}
            assert not m.locks(f, lock.threshold(1.0))
    f = {'cam': 0.0, 'emg': None, 'claude': 1.0, 'finished': True, 'passes': 99, 'face_ok': True}
    assert m.score(f) < lock.threshold(1.0)
    assert lock.W_STABLE + lock.CLAUDE_CAP < lock.threshold(1.0)


def test_score_shape():
    m = lock.LockModel()
    base = {'cam': 0.9, 'emg': None, 'claude': 0.0, 'finished': True, 'passes': 0, 'face_ok': True}
    s = m.score(base)
    assert m.score(dict(base, finished=False)) < s
    assert m.score(dict(base, face_ok=False)) < s
    assert m.score(dict(base, passes=3)) > s
    assert m.score(dict(base, claude=1.0)) - s == pytest.approx(lock.CLAUDE_CAP)
    assert lock.gate(base) == 1.0
    assert lock.gate(dict(base, emg=0.5)) < 1.0                       # the EMG hook


def test_threshold_follows_predictability_and_ttl():
    dec = Decoder(U)
    assert dec.threshold(0.0) == lock.THRESHOLD
    dec.on_predict({'words': [{'w': 'yes', 'weight': 0.9}], 'phrases': ['a little'], 'predictability': 1.0,
                    'ttl_s': 4.0}, 10.0)
    assert dec.threshold(11.0) == pytest.approx(lock.THRESHOLD - lock.SWING)
    assert dec._claude('yes', 11.0) == 0.9 and dec._claude('little', 11.0) == 0.5
    assert dec.threshold(14.5) == lock.THRESHOLD and dec._claude('yes', 14.5) == 0.0   # expired


def test_predicted_word_locks_on_first_pass():
    dec = Decoder(U)
    dec.on_predict({'words': [{'w': 'yes', 'weight': 1.0}], 'phrases': [], 'predictability': 0.5, 'ttl_s': 4}, 0.0)
    out = dec.on_tick(tick(1, frames_for([(10, 0.9)]), quiet_tail=5), 0.1)
    assert [w['w'] for m in topics(out, 'dec.lock') for w in m['words']] == ['yes']


def test_claude_prefers_lookalike_when_camera_tied():
    lex = Lexicon(['bat', 'mat'], pron={'bat': ['B', 'AE', 'T'], 'mat': ['M', 'AE', 'T']})
    dec = Decoder(U, lookalikes=LookAlikes(lex, {}))
    fr = frames_for([(7, 0.9)])
    dec.on_tick(tick(1, fr), 0.0)
    assert dec.cur.unlocked[0].w == 'bat'                             # no prediction: greedy word
    dec.on_predict({'words': [{'w': 'mat', 'weight': 0.9}], 'phrases': [], 'predictability': 0.5, 'ttl_s': 4}, 0.1)
    dec.on_tick(tick(1, fr), 0.2)
    w = (dec.cur.locked + dec.cur.unlocked)[0]
    assert (w.w, w.source) == ('mat', 'claude') and 'bat' in [a for a, _ in w.alts]


# --- repair window and the voice -----------------------------------------------------------------------------

BAT_MAT = Lexicon(['bat', 'mat'], pron={'bat': ['B', 'AE', 'T'], 'mat': ['M', 'AE', 'T']})


def _locked_bat():
    dec = Decoder(U, lookalikes=LookAlikes(BAT_MAT, {}))
    fr = frames_for([(7, 0.95)])
    _, now = feed(dec, fr, [len(fr)] * 3, quiet_tail=5)
    assert [w.w for w in dec.cur.locked] == ['bat']
    return dec, dec.cur.locked[0].lock_t


def test_repair_in_window_changes_voice():
    dec, t = _locked_bat()
    assert topics(dec.pump(t + REPAIR_WINDOW_S / 2), 'voice.say') == []
    dec.on_repair({'utt_id': 1, 'edits': [{'index': 0, 'from': 'bat', 'to': 'mat'}], 'please_repeat': False}, t + 0.2)
    say = topics(dec.pump(t + REPAIR_WINDOW_S + 1e-6), 'voice.say')
    assert say == [{'utt_id': 1, 'words': ['mat'], 'final': False, 'offset': 0, 'pace': 1.0, 'effort': 1.0}]


def test_late_repair_fixes_captions_only():
    dec, t = _locked_bat()
    assert topics(dec.pump(t + REPAIR_WINDOW_S + 1e-6), 'voice.say')[0]['words'] == ['bat']
    out = dec.on_repair({'utt_id': 1, 'edits': [{'index': 0, 'from': 'bat', 'to': 'mat'}], 'please_repeat': False},
                        t + 1.0)
    hyp = topics(out, 'dec.hyp')[0]
    assert hyp['locked'] == ['mat'] and hyp['raw'] == ['bat'] and hyp['spoken'] == ['bat']
    assert [w for m in topics(dec.pump(t + 2.0), 'voice.say') for w in m['words']] == []   # never said again


def test_repair_must_match_word_and_alts():
    dec, t = _locked_bat()
    bad = [{'index': 0, 'from': 'pat', 'to': 'mat'}, {'index': 5, 'from': 'bat', 'to': 'mat'},
           {'index': 0, 'from': 'bat', 'to': 'banana'}, {'index': 'x', 'from': 'bat', 'to': 'mat'}]
    out = dec.on_repair({'utt_id': 1, 'edits': bad}, t)
    assert topics(out, 'dec.hyp') == [] and dec.cur.locked[0].w == 'bat'
    assert topics(out, 'timing.dec') == [{'stage': 'dec.repair', 'ms': 0.0, 'utt_id': 1, 'applied': 0, 'rejected': 4}]
    out = dec.on_repair({'utt_id': 1, 'edits': [{'index': 0, 'from': ' BAT', 'to': 'Mat '}]}, t)
    assert dec.cur.locked[0].w == 'mat' and 'bat' in [a for a, _ in dec.cur.locked[0].alts]
    assert topics(out, 'timing.dec')[0]['applied'] == 1


# --- sentence end, please repeat, the 8 s cap ----------------------------------------------------------------

def test_sentence_end_settles_with_nbest():
    dec = Decoder(U)
    fr = frames_for([(2, 0.95), (3, 0.6)])                            # "want" too unsure to lock alone
    out, now = feed(dec, fr, [len(fr)] * 6, quiet_tail=12)
    assert [w.w for w in dec.cur.locked] == ['i']
    out += dec.pump(now + 0.5)                                        # ticks stopped, read ended quiet
    assert dec.cur.state == 'closing' and topics(out, 'dec.repeat') == []
    fin = dec.on_final({'seg_id': 1, 'nbest': [{'text': 'I WANT', 'score': -1.0}, {'text': 'I WANT', 'score': -2.0}]},
                       now + 1.6)
    assert [w['w'] for m in topics(fin, 'dec.lock') for w in m['words']] == ['want']
    assert dec.cur.state == 'done' and topics(fin, 'dec.repeat') == []
    says = topics(out + fin + dec.pump(now + 5.0), 'voice.say')
    assert [w for s in says for w in s['words']] == ['i', 'want']
    assert [s['final'] for s in says] == [False, True]
    assert [s['offset'] for s in says] == [0, 1]                     # where each message's first word sits


def test_please_repeat_instead_of_guessing():
    dec = Decoder(U)
    fr = frames_for([(2, 0.95), (7, 0.5), (3, 0.95)])                 # "i bat want", "bat" unclear
    out, now = feed(dec, fr, [len(fr)] * 6, quiet_tail=12)
    out += dec.pump(now + 0.5)
    fin = dec.on_final({'seg_id': 1, 'nbest': [{'text': 'I BAT WANT', 'score': -1.0},
                                               {'text': 'I MAT WANT', 'score': -1.0}]}, now + 1.6)
    rep = topics(fin, 'dec.repeat')
    assert len(rep) == 1 and rep[0]['utt_id'] == 1
    r = rep[0]['reason']
    assert r['reader'] == 'camera' and [(w['w'], w['index']) for w in r['words']] == [('bat', 1), ('want', 2)]
    assert r['words'][0]['nbest'] == pytest.approx(0.5) and r['words'][0]['score'] < r['threshold']
    hyp = topics(fin, 'dec.hyp')[-1]
    assert hyp['locked'] == ['i'] and hyp['unlocked'] == ['bat', 'want']   # later words stay in the captions
    says = topics(out + fin + dec.pump(now + 5.0), 'voice.say')
    assert [w for s in says for w in s['words']] == ['i'] and says[-1]['final'] is True   # voice stops at "bat"


def test_claude_please_repeat_waits_for_sentence_end_and_never_holds_voice():
    dec, t = _locked_bat()
    out = dec.on_repair({'utt_id': 1, 'edits': [], 'please_repeat': True}, t + 0.1)
    assert topics(out, 'dec.repeat') == []                            # mid-sentence: nothing yet
    assert topics(dec.pump(t + REPAIR_WINDOW_S + 1e-6), 'voice.say')[0]['words'] == ['bat']
    out = dec.pump(t + 2.0)                                           # sentence ends, all locked: settles
    rep = topics(out, 'dec.repeat')
    assert len(rep) == 1 and rep[0]['reason']['reader'] == 'claude'
    out = dec.on_repair({'utt_id': 1, 'edits': [], 'please_repeat': True}, t + 3.0)
    assert topics(out, 'dec.repeat') == []                            # reported once


def test_late_claude_please_repeat_after_sentence_end():
    dec, t = _locked_bat()
    dec.pump(t + 2.0)
    assert dec.cur.state == 'done'
    out = dec.on_repair({'utt_id': 1, 'edits': [], 'please_repeat': True}, t + 2.5)
    assert topics(out, 'dec.repeat')[0]['reason']['reader'] == 'claude'


def test_voice_final_even_with_nothing_locked():
    dec = Decoder(U)
    fr = frames_for([(7, 0.3)])                                       # far too unsure
    _, now = feed(dec, fr, [len(fr)] * 3, quiet_tail=12)
    out = dec.pump(now + 0.5) + dec.pump(now + 0.5 + 3.0)
    assert topics(out, 'voice.say') == [{'utt_id': 1, 'words': [], 'final': True, 'offset': 0, 'pace': 1.0,
                                         'effort': 1.0}]


def test_final_timeout_without_cam_final():
    dec = Decoder(U)
    fr = frames_for([(2, 0.95), (7, 0.5)])
    _, now = feed(dec, fr, [len(fr)] * 3, quiet_tail=12)
    dec.pump(now + 0.5)
    out = dec.pump(now + 0.5 + 3.0)
    assert len(topics(out, 'dec.repeat')) == 1 and dec.cur.state == 'done'


WORDS9 = [2, 3, 10, 11, 12, 13, 14, 7, 6]   # i want yes please hello there now bat what


@pytest.mark.parametrize('p', [0.95, 0.5])   # 0.95: overlap words locked; 0.5: nothing locks, words frozen
def test_cap_overlap_same_utterance_no_duplicates(p):
    dec = Decoder(U)
    timeline = frames_for([(i, p) for i in WORDS9], start=10, gap=28)   # word k at frames 30k+10..11, 270 frames
    out, now = feed(dec, timeline, range(40, 201, 10), seg=1, t0=0.0)
    seg2 = timeline[150:]
    out2, now = feed(dec, seg2, range(60, len(seg2) + 1, 10), seg=2, t0=150 * FPS, now=now)
    assert len(dec.utts) == 1 and dec.cur.segs == [1, 2]
    assert all_words(dec) == 'i want yes please hello there now bat what'.split()
    ids = {m['utt_id'] for t, m in out + out2 if t.startswith('dec.')}
    assert ids == {1}
    if p == 0.95:
        locked = [w['w'] for m in topics(out + out2, 'dec.lock') for w in m['words']]
        assert locked == 'i want yes please hello there now bat what'.split()[:len(locked)] and len(locked) >= 8


def test_pause_overlap_is_not_a_cap():
    dec = Decoder(U)
    fr = frames_for([(2, 0.95), (3, 0.95)], gap=10)                   # a short segment, T well below the cap
    _, now = feed(dec, fr, [len(fr)] * 3, t0=0.0)
    dec.on_tick(tick(2, frames_for([(10, 0.95)]), t0=FPS * (len(fr) - 8)), now)   # starts inside seg 1
    assert dec.cur.utt_id == 2 and dec.utts[1].state != 'open' and all_words(dec) == ['yes']


def test_long_utterance_ends_at_a_cap_without_losing_words():
    dec = Decoder(U)
    n = 33
    timeline = frames_for([(WORDS9[k % 9], 0.95) for k in range(n)], start=10, gap=28)  # 1000 frames, 40 s
    out, now, k = [], 0.0, 0
    while 150 * k < len(timeline) - 50:
        seg = timeline[150 * k:150 * k + 200]
        o, now = feed(dec, seg, list(range(60, len(seg), 10)) + [len(seg)], seg=k + 1, t0=150 * k * FPS, now=now)
        out += o
        k += 1
    out += dec.pump(now + 0.5) + dec.pump(now + 4.0)
    assert len(dec.utts) == 2 and dec.utts[1].segs == [1, 2, 3, 4, 5]
    words = [w for u in sorted(dec.utts) for w in all_words(dec, u)]
    assert words == [U[WORDS9[k % 9]][1:].lower() for k in range(n)]
    finals = [m['utt_id'] for m in topics(out, 'voice.say') if m['final']]
    assert finals == [1, 2]


def test_dec_hyp_sends_only_newest_locked(monkeypatch):
    monkeypatch.setattr(decoder, 'HYP_LOCKED', 3)
    dec = Decoder(U)
    fr = frames_for([(i, 0.95) for i in WORDS9[:6]])
    out, _ = feed(dec, fr, [len(fr)] * 4, quiet_tail=5)
    hyp = topics(out, 'dec.hyp')[-1]
    assert len(dec.cur.locked) == 6 and hyp['offset'] == 3 and hyp['locked'] == ['please', 'hello', 'there']
    assert len(hyp['raw']) == len(hyp['spoken']) == 3 and len(hyp['confidence']) == 3 + len(hyp['unlocked'])


def test_new_segment_after_gap_is_new_utterance():
    dec = Decoder(U)
    fr = frames_for([(2, 0.95)])
    _, now = feed(dec, fr, [len(fr)] * 3, quiet_tail=12)
    out = dec.on_tick(tick(2, fr, t0=200.0), now + 0.1)               # starts after seg 1 ended: no overlap
    assert dec.cur.utt_id == 2 and dec.utts[1].state != 'open'


def test_newest_tick_per_segment_only():
    msgs = [('cam.tick', {'seg_id': 1, 'n': 1}), ('claude.predict', {}), ('cam.tick', {'seg_id': 1, 'n': 2}),
            ('cam.tick', {'seg_id': 2, 'n': 3})]
    assert [m.get('n') for _, m in newest_ticks(msgs)] == [None, 2, 3]


def test_serve_survives_decoder_exceptions():
    class Boom(Decoder):
        calls = 0

        def on_tick(self, msg, now):
            raise RuntimeError('tick bug')

        def pump(self, now):
            Boom.calls += 1
            if Boom.calls == 1:
                raise RuntimeError('pump bug')
            return super().pump(now)

    class FakeSub:
        def __init__(self, msgs):
            self.msgs = list(msgs)

        def recv(self, timeout_ms=None):
            return self.msgs.pop(0) if self.msgs else (None, None)

    class FakePub:
        sent = []

        def send(self, topic, msg):
            self.sent.append((topic, msg))

    pub, n = FakePub(), iter(range(3))
    serve(Boom(U), FakeSub([('cam.tick', tick(1, frames_for([(2, 0.9)])))]), pub, lambda: next(n, None) is None)
    bad = [m for t, m in pub.sent if t == 'timing.dec' and m['stage'] == 'dec.bad_msg']
    errors = ' '.join(e for m in bad for e in m['errors'])
    assert 'tick bug' in errors and 'pump bug' in errors
    assert Boom.calls >= 2                                            # the loop went on after both


# --- the bus -------------------------------------------------------------------------------------------------

def test_bus_round_trip():
    ctx = zmq.Context()
    cam, claude = Pub('inproc://cam', ctx), Pub('inproc://claude', ctx)
    dec_pub = Pub('inproc://dec', ctx)
    sub = Sub(['inproc://cam', 'inproc://claude'], TOPICS, ctx)
    listen = Sub(['inproc://dec'], ['dec.', 'voice.', 'timing.dec'], ctx)
    stop = threading.Event()
    th = threading.Thread(target=serve, args=(Decoder(U, repair_window_s=0.05), sub, dec_pub, stop.is_set))
    th.start()
    try:
        time.sleep(0.2)                                               # let the subscriptions reach the publishers
        cam.sock.send_multipart([b'cam.tick', b'not msgpack'])        # malformed: counted, loop survives
        claude.send('claude.predict', {'words': [{'w': 'yes', 'weight': 1.0}], 'phrases': [], 'predictability': 0.5,
                                       'ttl_s': 4.0, 't': 0.0})
        time.sleep(0.05)
        fr = frames_for([(10, 0.95)])
        for _ in range(3):
            cam.send('cam.tick', tick(1, fr, quiet_tail=5))
            time.sleep(0.05)
        got, deadline = {}, bus.now() + 3.0
        while 'voice.say' not in got and bus.now() < deadline:
            topic, msg = listen.recv(timeout_ms=100)
            if topic:
                got.setdefault(topic, []).append(msg)
        assert got['dec.lock'][0]['words'][0]['w'] == 'yes'
        assert got['voice.say'][0]['words'] == ['yes']
        assert any(m['stage'] == 'dec.bad_msg' and m['count'] == 1 for m in got['timing.dec'])
        assert any(m['stage'] == 'dec.pass' and m['ms'] >= 0 for m in got['timing.dec'])
        assert set(got['dec.hyp'][0]) >= {'utt_id', 'locked', 'unlocked', 'confidence', 'source'}
    finally:
        stop.set()
        th.join(2)
        for s in (cam, claude, dec_pub, sub, listen):
            s.close()
        ctx.term()


# --- review regressions ----------------------------------------------------------------------------------------

def at(spec, T):
    """Frames with a piece (id, p) at the given frame indices, blank elsewhere."""
    out = [None] * T
    for f, piece in spec:
        out[f] = piece
    return out


def pump_until(dec, now, until, dt=0.1):
    out = []
    while now < until:
        out += dec.pump(now)
        now += dt
    return out, now


def test_swapped_last_word_reread_a_frame_later_is_not_locked_twice():
    dec = Decoder(U, lookalikes=LookAlikes(BAT_MAT, {}))
    dec.on_predict({'words': [{'w': 'mat', 'weight': 0.9}], 'phrases': [], 'predictability': 0.5, 'ttl_s': 30}, 0.0)
    out, now = [], 0.1
    for T in (18, 20, 22):                                    # camera 'bat', Claude makes it 'mat', it locks
        out += dec.on_tick(tick(1, at([(10, (7, 0.95))], T)), now) + dec.pump(now)
        now += 0.1
    for T in (24, 26, 30, 34):                                # the re-read moves 'bat' one frame later
        out += dec.on_tick(tick(1, at([(11, (7, 0.95)), (20, (14, 0.95))], T)), now) + dec.pump(now)
        now += 0.1
    out += pump_until(dec, now, now + 1.0)[0]
    assert [w['w'] for t, m in out if t == 'dec.lock' for w in m['words']] == ['mat', 'now']
    assert [w for t, m in out if t == 'voice.say' for w in m['words']] == ['mat', 'now']


def test_voice_speaks_sentences_in_order():
    dec = Decoder(U)
    out, now = [], 0.0
    for T in (14, 16, 18, 20):                                # 'i' sure, 'want' unsure: waits for cam.final
        out += dec.on_tick(tick(1, at([(3, (2, 0.95)), (9, (3, 0.6))], T)), now) + dec.pump(now)
        now += 0.1
    o, now = pump_until(dec, now, now + 0.6)
    out += o
    for T in (12, 14, 16, 18):                                # the next sentence starts and locks 'there'
        out += dec.on_tick(tick(2, at([(3, (13, 0.95))], T), t0=102.0), now) + dec.pump(now)
        now += 0.1
    out += dec.on_final({'seg_id': 1, 'nbest': [{'text': 'I WANT', 'score': 0.0}]}, now)
    out += pump_until(dec, now, now + 4.0)[0]
    said = [(m['utt_id'], m['words'], m['final']) for t, m in out if t == 'voice.say']
    assert said == [(1, ['i'], False), (1, ['want'], True), (2, ['there'], True)]    # 'there' waited for sentence 1


def test_camera_restart_reusing_seg_ids_starts_new_sentences():
    dec = Decoder(U)
    fr = frames_for([(2, 0.95)])
    out, now = feed(dec, fr, [len(fr)] * 3, quiet_tail=12)
    out += pump_until(dec, now, now + 4.0)[0]
    assert dec.utts[1].state == 'done'
    out2, now = feed(dec, frames_for([(10, 0.95)]), [len(fr)] * 3, now=now + 4.0, t0=150.0, quiet_tail=12)
    out2 += pump_until(dec, now, now + 4.0)[0]
    assert [(m['utt_id'], m['words']) for t, m in out2 if t == 'voice.say' and m['words']] == [(2, ['yes'])]


def test_ticks_resuming_after_a_settled_stall_go_on_as_a_new_utterance():
    dec = Decoder(U)
    fr = frames_for([(2, 0.95), (3, 0.95)])
    first = fr[:12]                                           # 'i' only, then the camera stalls
    out, now = feed(dec, first, [len(first)] * 3)
    o, now = pump_until(dec, now, now + 4.5)
    out += o
    assert dec.utts[1].state == 'done'
    o, now = feed(dec, fr, [len(fr)] * 3, now=now, quiet_tail=12)   # same segment, now with 'want'
    out += o + pump_until(dec, now, now + 4.0)[0]
    said = [w for t, m in out if t == 'voice.say' for w in m['words']]
    assert said == ['i', 'want']


def test_nbest_text_change_drops_the_old_texts_claude_weight():
    from fusion.decoder import apply_nbest
    w = Word(w='mat', f0=0, f1=1, t0=0.0, t1=0.04, cam=0.6, alts=[('bat', 0.6)], claude=0.9, source='claude')
    apply_nbest([w], [w], [{'text': 'BAT', 'score': 0.0}])
    assert (w.w, w.claude, w.source) == ('bat', 0.0, 'camera')


def test_serve_survives_an_unsendable_message():
    class OddDecoder:
        def on_tick(self, msg, now):
            return [('dec.hyp', {'bad': object()}), ('dec.hyp', {'ok': 1})]

        def pump(self, now):
            return []

    class OneTick:
        msgs = [('cam.tick', {'seg_id': [1]})]                 # unhashable seg_id too

        def recv(self, timeout_ms=None):
            return self.msgs.pop() if self.msgs else (None, None)

    sent = []

    class Pub:
        def send(self, topic, msg):
            from bus.team import encode
            encode(topic, msg)
            sent.append(topic)

    n = iter(range(3))
    serve(OddDecoder(), OneTick(), Pub(), lambda: next(n, None) is None)
    assert 'dec.hyp' in sent and 'timing.dec' in sent
