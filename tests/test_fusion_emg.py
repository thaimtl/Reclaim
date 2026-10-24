"""fusion.emg and the decoder with EMG: CTC word support, when EMG has no say, look-alike settling, the voice's pace
and effort, the run loop's ordering and the offline clock merge. Synthetic emg.read, no torch needed (two checks
use it when present)."""
import numpy as np
import pytest

from bench.lookalikes import Lexicon
from fusion import emg as E
from fusion import evaluate
from fusion.decoder import Decoder
from fusion.emg import EmgSupport, ctc_loglik, letter_ids
from fusion.run import newest_ticks
from fusion.words import LookAlikes, Word
from tests.test_fusion import U, frames_for, tick

BAT_MAT = Lexicon(['bat', 'mat'], pron={'bat': ['B', 'AE', 'T'], 'mat': ['M', 'AE', 'T']})
FR = 86.13


def spelled(text, t_word=(100.10, 100.20), t_span=(99.5, 101.0), sharp=6.0):
    """An emg.read whose letters spell `text` inside t_word and are blank elsewhere."""
    ft = np.arange(t_span[0], t_span[1], 1 / FR)
    logits = np.zeros((len(ft), 38))
    logits[:, E.BLANK] = sharp
    inside = np.flatnonzero((ft >= t_word[0]) & (ft <= t_word[1]))
    for k, part in enumerate(np.array_split(inside, len(text))):
        logits[part, E.BLANK] = 0
        logits[part, letter_ids(text[k])[0]] = sharp
    logp = logits - np.logaddexp.reduce(logits, axis=1, keepdims=True)
    return {'ok': True, 'frame_t': ft, 'letter_logp': logp.astype(np.float16), 'pace': 1.4, 'effort': 0.6,
            'seg_id': 1, 'quality': {'pinned': [0.0] * 4, 'flat': [False] * 4}}


def word(w='bat', alts=(('mat', 0.85),), cam=0.9, t0=100.10, t1=100.20):
    return Word(w=w, f0=0, f1=1, t0=t0, t1=t1, cam=cam, alts=list(alts))


def test_ctc_loglik_matches_torch():
    torch = pytest.importorskip('torch')
    rng = np.random.default_rng(0)
    for T, labels in [(20, [3, 0, 19]), (12, [11, 11, 4]), (5, [1, 2, 3, 4, 5]), (30, [7]), (8, [2, 2, 2, 2])]:
        logp = torch.log_softmax(torch.tensor(rng.standard_normal((T, 38))), -1)
        want = -torch.nn.functional.ctc_loss(logp[:, None], torch.tensor([labels]), torch.tensor([T]),
                                             torch.tensor([len(labels)]), blank=E.BLANK, reduction='sum',
                                             zero_infinity=False).item()
        got = ctc_loglik(logp.numpy(), labels)
        assert (got == E.INFEASIBLE and not np.isfinite(want)) or got == pytest.approx(want, abs=1e-6)


def test_infeasible_when_the_span_is_too_short():
    assert ctc_loglik(np.zeros((3, 38)) - np.log(38), letter_ids('pills')) == E.INFEASIBLE   # 5 letters + 1 repeat


def test_letters_match_svr_text():
    text = pytest.importorskip('svr.text')
    assert E.CHARS == text.CHARS and E.BLANK == text.BLANK
    assert letter_ids("don't Stop 42") == [i for i in text.text_to_ids("don't Stop 42") if i != text.SPACE]


def test_support_backs_the_spelled_lookalike():
    s = EmgSupport()
    s.on_read(spelled('mat'), 10.0)
    sc = s.scores(word(), 10.1)
    assert sc['mat'] > 0.9 and sc['bat'] < 0.1 and s.support(word(), 10.1) == pytest.approx(sc['bat'])
    assert s.choose(word(), 10.1) == ('mat', pytest.approx(sc['mat']))
    assert s.choose(word(alts=[('mat', 0.5)]), 10.1) is None      # the camera can tell them apart: no swap


def test_no_say_without_a_usable_read():
    s = EmgSupport()
    assert s.support(word(), 0.0) is None                          # no read yet
    s.on_read(spelled('mat'), 10.0)
    assert s.support(word(), 10.31) is None                        # stale
    assert s.support(word(alts=()), 10.1) is None                  # nothing to compare against
    assert s.support(word(t1=101.5), 10.1) is None                 # the read doesn't cover the word yet
    s.on_read({'ok': False}, 10.2)
    assert s.support(word(), 10.2) is None                         # the Uno was pulled


def test_undecided_emg_neither_swaps_nor_blocks_the_lock():
    s = EmgSupport()
    r = spelled('mat')
    r['letter_logp'] = np.full_like(r['letter_logp'], -np.log(38))  # flat: every candidate alike
    s.on_read(r, 10.0)
    assert s.scores(word(), 10.1)['bat'] == pytest.approx(0.5, abs=1e-3)
    assert s.choose(word(), 10.1) is None and s.support(word(), 10.1) is None   # undecided: no say, no veto


def test_voice_follows_emg_and_falls_back():
    s = EmgSupport()
    assert s.voice(0.0) == (1.0, 1.0)
    s.on_read(spelled('mat'), 10.0)
    assert s.voice(10.1) == (pytest.approx(1.4), pytest.approx(0.6))
    assert s.voice(11.0) == (1.0, 1.0)                              # stale
    s.on_read(dict(spelled('mat'), pace=50.0, effort=None), 12.0)
    assert s.voice(12.0) == (E.VOICE_RANGE[1], 1.0)


def run_bat(dec, read=None):
    """'bat' mouthed (camera-tied with 'mat'), three passes, then quiet; returns the decoder's outputs."""
    out = []
    tk = tick(1, frames_for([(7, 0.95)]), quiet_tail=5)    # 'mat' joins as bat's look-alike at the same prob
    for t in (0.1, 0.2, 0.3):
        if read is not None:
            out += dec.on_emg(read, t - 0.01)
        out += dec.on_tick(tk, t)
    for t in np.arange(0.32, 6.0, 0.02):            # past the cam.final wait: the sentence settles
        out += dec.pump(float(t))
    return out


def test_decoder_settles_the_lookalike_with_emg():
    dec = Decoder(U, lookalikes=LookAlikes(BAT_MAT, {}), emg=EmgSupport())
    out = run_bat(dec, spelled('mat'))
    lock = [m for t, m in out if t == 'dec.lock']
    hyp = [m for t, m in out if t == 'dec.hyp'][0]
    assert hyp['source'] == ['emg'] and (hyp['locked'] + hyp['unlocked']) == ['mat']
    assert lock and lock[0]['words'][0]['w'] == 'mat' and lock[0]['words'][0]['emg'] > 0.9
    say = [m for t, m in out if t == 'voice.say' and m['words']]
    assert say[0]['words'] == ['mat']
    assert say[0]['pace'] == pytest.approx(1.4) and say[0]['effort'] == pytest.approx(0.6)


def test_emg_choice_is_never_overridden_by_claude():
    dec = Decoder(U, lookalikes=LookAlikes(BAT_MAT, {}), emg=EmgSupport())
    dec.on_predict({'words': [{'w': 'bat', 'weight': 0.9}], 'phrases': [], 'predictability': 0.5, 'ttl_s': 9}, 0.0)
    hyp = [m for t, m in run_bat(dec, spelled('mat')) if t == 'dec.hyp'][0]
    assert hyp['source'] == ['emg'] and (hyp['locked'] + hyp['unlocked']) == ['mat']


def test_without_reads_emg_support_changes_nothing():
    strip = lambda out: [(t, {k: v for k, v in m.items() if k != 'ms'}) for t, m in out if t != 'timing.dec']
    a = strip(run_bat(Decoder(U, lookalikes=LookAlikes(BAT_MAT, {}))))
    b = strip(run_bat(Decoder(U, lookalikes=LookAlikes(BAT_MAT, {}), emg=EmgSupport())))
    assert a == b and any(t == 'voice.say' for t, _ in a)


def test_run_loop_hands_the_newest_read_over_first():
    msgs = [('cam.tick', {'seg_id': 1, 'i': 0}), ('emg.read', {'i': 1}), ('cam.tick', {'seg_id': 1, 'i': 2}),
            ('claude.predict', {'i': 3}), ('emg.read', {'i': 4})]
    assert [m['i'] for _, m in newest_ticks(msgs)] == [4, 2, 3]


def test_offline_emg_reads_move_onto_the_camera_clock():
    segs = [(1000.0, 1003.0, -900.0), (1010.0, 1012.0, -895.0)]     # replayed camera: t_pub = frame_t + offset
    ev = [(1001.0, 'emg.read', {'t_pub': 1001.0}), (1011.5, 'emg.read', {'t_pub': 1011.5}),
          (1005.0, 'emg.read', {'t_pub': 1005.0})]
    got = [m['t_pub'] for _, _, m in evaluate.emg_on_camera_clock(ev, segs)]
    assert got == [101.0, 116.5, 105.0]                             # 1005 is nearest the first segment
    assert evaluate.emg_on_camera_clock(ev, []) == ev


def test_undecided_emg_still_lets_the_camera_lock():
    r = spelled('mat')
    r['letter_logp'] = np.full_like(r['letter_logp'], -np.log(38))
    out = run_bat(Decoder(U, lookalikes=LookAlikes(BAT_MAT, {}), emg=EmgSupport()), r)
    lock = [w for t, m in out if t == 'dec.lock' for w in m['words']]
    assert [(w['w'], w['emg']) for w in lock] == [('bat', None)]


def test_decisive_emg_against_the_word_holds_the_lock(monkeypatch):
    monkeypatch.setattr(E, 'SWAP_MARGIN', 2.0)      # no swap allowed: EMG can only vote against 'bat'
    out = run_bat(Decoder(U, lookalikes=LookAlikes(BAT_MAT, {}), emg=EmgSupport()), spelled('mat'))
    assert not [m for t, m in out if t == 'dec.lock']
    assert [m['reason']['reader'] for t, m in out if t == 'dec.repeat'] == ['emg']
