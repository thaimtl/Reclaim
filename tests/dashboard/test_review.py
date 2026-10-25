"""The end-of-call review: what Opus is shown, how each reader is scored, and that it never blocks the end screen.

    venv-call/bin/python -m pytest tests/dashboard -q
"""
import asyncio
import json
from types import SimpleNamespace

from dashboard import review as rv
from dashboard.server import Dashboard
from dashboard.state import CallState


def finished_call():
    st = CallState()
    st.feed('call.state', {'state': 'connected'}, 0.0)
    st.feed('partner.text', {'text': 'What did you lose at the airport?', 'is_final': True}, 1.0)
    st.feed('dec.lock', {'utt_id': 1, 'words': [{'w': 'I', 'cam_top': 'I'}, {'w': 'LOST', 'cam_top': 'LOST'},
                                                 {'w': 'MY', 'cam_top': 'BY'}, {'w': 'PACK', 'cam_top': 'PACK'}]}, 2.0)
    st.feed('dec.hyp', {'utt_id': 1, 'locked': ['I', 'LOST', 'MY', 'PACK'], 'unlocked': []}, 2.0)
    st.feed('claude.repair', {'utt_id': 1, 'edits': [{'index': 3, 'from': 'PACK', 'to': 'BAG'}]}, 2.5)
    st.feed('dec.hyp', {'utt_id': 1, 'locked': ['I', 'LOST', 'MY', 'BAG'], 'unlocked': []}, 2.6)
    st.feed('dec.final', {'utt_id': 1}, 3.0)
    st.feed('call.state', {'state': 'ended'}, 4.0)
    return st


def test_opus_sees_the_call_in_order_with_each_readers_words():
    data = rv.review_input(finished_call())
    assert [list(t) for t in data['turns']] == [['judge'], ['austin']]
    s = data['sentences'][0]
    assert s['readers'] == {'lips': ['i', 'lost', 'by', 'pack'], 'lips_muscles': ['i', 'lost', 'my', 'pack'],
                            'claude': ['i', 'lost', 'my', 'bag']}
    words = data['turns'][1]['austin']['words']
    assert words[2] == {'caption': 'MY', 'lips_alone': 'BY'} and words[3] == {'caption': 'BAG', 'before_claude': 'PACK'}


def test_each_reader_is_scored_against_what_he_meant():
    data = rv.review_input(finished_call())
    out = rv.score(data['sentences'], {1: ('I lost my bag.', False)})
    assert out['wer'] == {'lips': 50.0, 'lips_muscles': 25.0, 'claude': 0.0} and out['words'] == 4
    assert out['n_wrong'] == 0 and out['unclear'] == 0
    out = rv.score(data['sentences'], {1: ('I lost my black bag.', False)})     # a word nobody caught
    assert out['wrong'] == [{'sentence': 1, 'op': 'missed', 'got': None, 'meant': 'black'}]
    for unsure in (('I lost my black bag.', True), ('', False)):     # Opus can't tell: scored as captioned
        out = rv.score(data['sentences'], {1: unsure})
        assert out['wer']['claude'] == 0.0 and out['unclear'] == 1 and out['n_wrong'] == 0


def test_edits_count_substitutions_insertions_and_deletions():
    assert rv.edits(['a', 'b', 'c'], ['a', 'x', 'c']) == [('sub', 'b', 'x')]
    assert [op for op, *_ in rv.edits(['a', 'b'], ['a'])] == ['extra']
    assert [op for op, *_ in rv.edits([], ['a', 'b'])] == ['missed', 'missed']


class FakeClient:
    """Stands in for anthropic.Anthropic: checks the request shape, answers like Opus with structured output."""
    def __init__(self, reply, stop='end_turn'):
        self.reply, self.stop, self.sent = reply, stop, None
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))

    def create(self, **kw):
        self.sent = kw
        return SimpleNamespace(stop_reason=self.stop, model=kw['model'], usage=SimpleNamespace(input_tokens=1, output_tokens=1),
                               content=[SimpleNamespace(type='thinking', text=None),
                                        SimpleNamespace(type='text', text=json.dumps(self.reply))])


def test_the_request_is_opus_with_a_fallback_and_a_schema():
    c = FakeClient({'sentences': [{'id': 1, 'meant': 'I lost my bag.', 'unclear': False}]})
    out = rv.review(rv.review_input(finished_call()), client=c)
    assert c.sent['model'] == 'claude-opus-5-5' and c.sent['fallbacks'] == 'default'
    assert c.sent['output_config']['format']['type'] == 'json_schema' and 'thinking' not in c.sent
    assert out['status'] == 'done' and out['wer']['claude'] == 0.0
    try:
        rv.review(rv.review_input(finished_call()), client=FakeClient({}, stop='refusal'))
        raise AssertionError('a refusal must not pass as a review')
    except RuntimeError:
        pass


async def test_the_review_runs_once_when_the_call_ends_and_never_blocks(aiohttp_client):
    calls = []

    def slow_reviewer(data):
        calls.append(data)
        return {'status': 'done', 'wer': {'lips': 1.0, 'lips_muscles': 1.0, 'claude': 0.0}}
    d = Dashboard(reviewer=slow_reviewer)
    d.state = finished_call()
    await aiohttp_client(d.app())
    for _ in range(40):
        await asyncio.sleep(0.05)
        if d.state.review and d.state.review['status'] == 'done':
            break
    assert len(calls) == 1 and d.state.snapshot(5.0)['review']['status'] == 'done'


async def test_a_failed_review_says_so(aiohttp_client):
    def broken(data):
        raise ConnectionError('no network')
    d = Dashboard(reviewer=broken)
    d.state = finished_call()
    await aiohttp_client(d.app())
    for _ in range(40):
        await asyncio.sleep(0.05)
        if d.state.review and d.state.review['status'] != 'pending':
            break
    assert d.state.review == {'status': 'failed'}
