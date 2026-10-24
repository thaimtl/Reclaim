"""Committer (D9) on synthetic CTC word ends.

    venv-live/bin/python -m pytest tests/live/test_commit.py -q
"""
from live.commit import Committer
from live.types import FAINT, SOLID, frame_time


def states(update):
    return [(w.w, w.state) for w in update.words]


def test_margin_and_agreement(hyp):
    c = Committer(margin_frames=4, agree=2)
    u = c.update(hyp([('WHAT', 5), ('DO', 8)], n_frames=10))
    assert states(u) == [('WHAT', FAINT), ('DO', FAINT)]
    u = c.update(hyp([('WHAT', 5), ('DO', 8)], n_frames=12))
    assert states(u) == [('WHAT', SOLID), ('DO', FAINT)]
    assert u.words[0].t_end == frame_time(0.0, 5)
    assert u.t == frame_time(0.0, 11)


def test_margin_blocks_even_when_agreeing(hyp):
    c = Committer(margin_frames=8, agree=2)
    for n in (10, 11, 12):
        u = c.update(hyp([('WHAT', 5)], n_frames=n))
    assert states(u) == [('WHAT', FAINT)]
    u = c.update(hyp([('WHAT', 5)], n_frames=14))
    assert states(u) == [('WHAT', SOLID)]


def test_right_edge_flip_does_not_commit(hyp):
    c = Committer(margin_frames=4, agree=2)
    c.update(hyp([('WHAT', 5), ('DO', 10)], n_frames=16))
    u = c.update(hyp([('WHAT', 5), ('TO', 10)], n_frames=17))
    assert states(u) == [('WHAT', SOLID), ('TO', FAINT)]
    u = c.update(hyp([('WHAT', 5), ('DO', 10)], n_frames=18))
    assert states(u) == [('WHAT', SOLID), ('DO', FAINT)]
    u = c.update(hyp([('WHAT', 5), ('DO', 10)], n_frames=19))
    assert states(u) == [('WHAT', SOLID), ('DO', SOLID)]


def test_word_behind_a_changed_word_waits(hyp):
    c = Committer(margin_frames=2, agree=2)
    c.update(hyp([('A', 3), ('B', 6)], n_frames=20))
    u = c.update(hyp([('X', 3), ('B', 6)], n_frames=21))
    assert states(u) == [('X', FAINT), ('B', FAINT)]


def test_last_word_commits_by_margin_only(hyp):
    c = Committer(margin_frames=8, agree=2)
    words = [('WHAT', 5), ('DO', 10), ('YOU', 15), ('NEED', 20)]
    u = None
    for n in range(21, 29):
        u = c.update(hyp(words, n_frames=n, trailing_blanks=n - 21))
    assert states(u)[-1] == ('NEED', FAINT)
    u = c.update(hyp(words, n_frames=29, trailing_blanks=8))
    assert [s for _, s in states(u)] == [SOLID] * 4


def test_solid_prefix_stays(hyp):
    c = Committer(margin_frames=4, agree=2)
    c.update(hyp([('WHAT', 5), ('DO', 9)], n_frames=20))
    u = c.update(hyp([('WHAT', 5), ('DO', 9)], n_frames=21))
    assert states(u) == [('WHAT', SOLID), ('DO', SOLID)]
    u = c.update(hyp([('WHY', 5), ('TO', 9), ('YOU', 18)], n_frames=22))
    assert states(u) == [('WHAT', SOLID), ('DO', SOLID), ('YOU', FAINT)]
    assert u.words[0].t_end == frame_time(0.0, 5)
    u = c.update(hyp([('WHATDO', 9), ('YOU', 18)], n_frames=23))
    assert states(u) == [('WHAT', SOLID), ('DO', SOLID), ('YOU', SOLID)]


def test_utt_change_resets(hyp):
    c = Committer(margin_frames=2, agree=2)
    c.update(hyp([('HELLO', 3)], n_frames=10))
    u = c.update(hyp([('HELLO', 3)], n_frames=11))
    assert states(u) == [('HELLO', SOLID)]
    u = c.update(hyp([('BYE', 3)], n_frames=10, utt_id=2, t0=5.0))
    assert states(u) == [('BYE', FAINT)]
    assert u.utt_id == 2 and u.words[0].t_end == frame_time(5.0, 3)
    c.reset()
    u = c.update(hyp([('BYE', 3)], n_frames=11, utt_id=2, t0=5.0))
    assert states(u) == [('BYE', FAINT)]


def test_revision_counter(hyp):
    c = Committer(margin_frames=50, agree=2)
    c.update(hyp([('WHAT', 5), ('DO', 9)], n_frames=10))
    c.update(hyp([('WHAT', 5), ('TO', 9)], n_frames=11))
    assert c.last_revisions == 1
    c.update(hyp([('WHAT', 5)], n_frames=12))
    assert c.last_revisions == 1
    c.update(hyp([('WHAT', 5), ('DO', 9)], n_frames=13))
    assert c.last_revisions == 0 and c.revisions == 2


def test_empty_hyp(hyp):
    u = Committer().update(hyp([], n_frames=3))
    assert u.words == ()
