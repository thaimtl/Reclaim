"""The live dashboard: CallState on the scripted call (every state in Thai's Figma v5), its edge cases, and the server.

    venv-call/bin/python -m pytest tests/dashboard -q -c tests/dashboard/pytest.ini     # runs in venv-call, like call/
"""
import asyncio
import json

import numpy as np
import pytest

pytest.importorskip('aiohttp')      # the dashboard runs in venv-call; venv-live's suite skips it

from bus.team import encode
from dashboard.demo_feed import simulate
from dashboard.server import Dashboard
from dashboard.state import CallState


@pytest.fixture(scope='module')
def shots():
    return simulate()


def words(s):
    return [w['w'] for w in s['austin']['words']] if s['austin'] else None


def test_waiting_before_the_call(shots):
    s = shots['waiting']
    assert s['mode'] == 'waiting' and s['austin'] is None and s['metrics'] == {'lips_to_voice': None, 'fixed': 0,
                                                                                 'repeats': 0}


def test_reading(shots):
    s = shots['reading']
    assert s['mode'] == 'live' and s['judge_live'] == '' and words(s) == ["I'd", 'love', 'a', 'coffee']
    assert [w['locked'] for w in s['austin']['words']] == [True, True, True, False]
    assert [w['src'] for w in s['austin']['words'][:3]] == ['camera', 'camera', 'emg']
    assert [r['w'] for r in s['austin']['cands']['rows']] == ['coffee', 'copy', 'toffee']
    assert s['emotion']['chip'] == 'Smiling' and s['face_ok'] and s['muscles']
    assert s['status']['ok'] and 'camera 25 fps, 4 of 4 pads in contact' in s['status']['text']
    assert s['metrics']['lips_to_voice'] == 300                      # first lock t (sent at -0.3 s) to Sonic's audio


def test_claude_repair(shots):
    s = shots['claude_repair']
    last = s['austin']['words'][-1]
    assert {k: last[k] for k in ('w', 'src', 'locked', 'was')} == {'w': 'bag.', 'src': 'claude', 'locked': True,
                                                                     'was': 'pack'}
    assert s['austin']['repairs'] == [{'index': 5, 'frm': 'pack', 'to': 'bag'}] and s['metrics']['fixed'] == 1
    assert s['austin']['spoken'] == 'Yes, but they lost my bag.' and s['emotion']['tone'] == 'Sounds concerned'


def test_please_repeat(shots):
    s = shots['please_repeat']
    assert s['austin']['repeat'] == {'reader': 'camera', 'words': ['sent'], 'alts': [['meant', 'bent']]}
    assert s['metrics']['repeats'] == 1 and words(s) == ['Yes,', 'they', 'sent'] and s['austin']['spoken'] == 'Yes, they'


def test_face_hidden(shots):
    s = shots['face_hidden']
    assert not s['face_ok'] and not s['status']['ok'] and s['status']['text'].startswith('Face out of view')
    assert [w['src'] for w in s['austin']['words']] == ['emg', 'emg']
    assert s['muscles'] and s['latency'][0]['label'] == 'Muscles only'


def test_muscles_off(shots):
    s = shots['muscles_off']
    assert not s['muscles'] and s['latency'][0]['label'] == 'Lips only'
    assert s['status']['text'].startswith('Muscle sensors: no signal')


def test_end_of_call_summary(shots):
    s = shots['end']
    m = s['summary']
    assert s['mode'] == 'end' and m['seconds'] > 30 and [r['to'] for r in m['repairs']] == ['bag']
    assert len(m['repeats']) == 1 and m['face_hidden_s'] > 1 and m['emg_words'] == 2
    assert m['disagree']['lips'] is not None and m['disagree']['muscles'] is not None
    who = [t['who'] for t in m['transcript']]
    assert who == ['judge', 'austin'] * 5
    austin = [t for t in m['transcript'] if t['who'] == 'austin']
    assert austin[1]['text'] == 'Yes, but they lost my bag.' and austin[1]['emotion'] == 'sad'
    assert austin[2]['repeat'] and austin[3]['face_hidden']
    assert m['latency'] == {'read': s['latency'][0]['ms'], 'lock': 105, 'claude': 300, 'claude_label': "Claude's window",
                            'sonic': 190, 'lips_to_voice': 300}
    assert m['disagree'] == {'lips': 10, 'muscles': 20}      # demo_feed: about 1 in 7 and 1 in 4 differ


# --- edge cases ------------------------------------------------------------------------------------------------
def fed(*msgs, now=10.0):
    st = CallState('(628) 888-6598')
    st.logging(now)
    for topic, msg in msgs:
        st.feed(topic, msg, now)
    return st


def test_judge_partial_then_final_and_a_new_call_resets():
    st = CallState()
    st.feed('call.state', {'state': 'connected'}, 1.0)
    st.feed('partner.text', {'text': 'so what', 'is_final': False}, 1.0)
    assert st.snapshot(10.0)['judge_live'] == 'so what'
    st.feed('partner.text', {'text': 'So what would you like?', 'is_final': True, 't': 2.0}, 10.0)
    s = st.snapshot(10.0)
    assert s['judge_live'] == '' and s['transcript'][-1]['text'] == 'So what would you like?' and s['elapsed'] == 9
    st.feed('call.state', {'state': 'connected', 't': 20.0}, 20.0)
    assert st.snapshot(20.0)['transcript'] == []


def test_voice_offset_skips_resent_words_and_keeps_real_repeats():
    st = fed(('voice.say', {'utt_id': 1, 'words': ['no'], 'offset': 0}),
             ('voice.say', {'utt_id': 1, 'words': ['no', 'no', 'way'], 'offset': 0}),
             ('voice.say', {'utt_id': 1, 'words': [], 'offset': 3, 'final': True}))
    assert st.utts[1]['spoken'] == ['no', 'no', 'way'] and st.transcript[-1]['text'] == 'no no way'


def test_repair_already_applied_by_the_decoder_is_shown_once():
    st = fed(('dec.hyp', {'utt_id': 2, 'locked': ['lost', 'my', 'bag'], 'unlocked': [], 'source': ['camera'] * 3}),
             ('claude.repair', {'utt_id': 2, 'edits': [{'index': 2, 'from': 'pack', 'to': 'bag'}]}))
    w = st.snapshot(10.0)['austin']['words'][2]
    assert (w['w'], w['src'], w['was']) == ('bag', 'claude', 'pack')


def test_muscles_go_stale_and_pads_without_contact_read_zero():
    flat = {'samples': np.full((10, 4), 1023, np.uint16)}                  # railed: pads off the skin
    st = fed(('emg.raw', flat))
    assert not st.snapshot(10.0)['muscles'] and all(not p['contact'] for p in st.snapshot(10.0)['pads'])
    live = {'samples': (512 + np.random.default_rng(0).normal(0, 20, (10, 4))).astype(np.uint16)}
    st.feed('emg.raw', live, 11.0)
    assert st.snapshot(11.0)['muscles'] and st.snapshot(12.5)['muscles'] is False   # 1 s without emg.raw: off


def test_face_hidden_time_counts_only_during_a_call():
    st = CallState()
    st.feed('call.state', {'state': 'connected'}, 0.0)
    st.feed('cam.tick', {'face_ok': False}, 5.0)
    st.feed('cam.tick', {'face_ok': True}, 7.5)
    st.feed('call.state', {'state': 'ended'}, 9.0)
    assert st.snapshot(9.0)['summary']['face_hidden_s'] == 2.5


def test_logging_and_claude_problems_show_in_the_status():
    st = CallState()
    assert 'Not logging' in st.status(10.0)['text']
    st.logging(10.0)
    st.feed('timing.claude', {'stage': 'claude', 'ms': 30.0, 'error': 'RateLimitError'}, 10.0)   # a failed call
    assert 'Claude: not answering' in st.status(10.0)['text']


def test_no_reader_picks_means_no_made_up_disagreement():
    st = fed(('call.state', {'state': 'connected', 't': 0.0}),
             ('dec.lock', {'utt_id': 1, 'words': [{'w': 'hi', 't': 1.0}]}),
             ('dec.hyp', {'utt_id': 1, 'locked': ['hi'], 'unlocked': []}),
             ('call.state', {'state': 'ended', 't': 30.0}))
    assert st.snapshot(30.0)['summary']['disagree'] == {'lips': None, 'muscles': None}


def test_odd_messages_never_break_the_state():
    st = fed(('dec.hyp', {'utt_id': 1, 'locked': None, 'unlocked': None}), ('cam.tick', {'emotion': None}),
             ('emg.raw', {'samples': np.zeros((0, 4))}), ('timing.voice', {'stage': 'sonic_first_audio', 'utt_id': 99}),
             ('call.state', {'state': 'ended'}))
    json.dumps(st.snapshot(10.0))


# --- server ----------------------------------------------------------------------------------------------------
def test_server_decodes_skips_bad_frames_and_keeps_the_newest_preview():
    d = Dashboard('(628) 888-6598', endpoints={})
    d.handle(encode('partner.text', {'text': 'hello', 'is_final': True, 't': 1.0}))
    d.handle([b'partner.text', b'\xc1 not msgpack'])
    d.handle(encode('cam.preview', {'t': 1.0, 'w': 640, 'h': 360, 'jpeg': b'\xff\xd8jpeg'}))
    assert d.state.transcript[-1]['text'] == 'hello' and d.frame.endswith(b'\xff\xd8jpeg')


async def test_websocket_sends_snapshots_and_the_camera_mirror(aiohttp_client):
    d = Dashboard('(628) 888-6598', endpoints={'none': 'inproc://dashboard-test-none'})
    client = await aiohttp_client(d.app())
    ws = await client.ws_connect('/ws')
    msg = await asyncio.wait_for(ws.receive(), 2)
    snap = json.loads(msg.data)
    assert snap['mode'] == 'waiting' and snap['number'] == '(628) 888-6598'
    d.handle(encode('cam.preview', {'t': 1.0, 'w': 640, 'h': 360, 'jpeg': b'\xff\xd8frame'}))
    got = []
    for _ in range(10):
        m = await asyncio.wait_for(ws.receive(), 2)
        got.append(m.data)
        if isinstance(m.data, bytes):
            break
    frame = next(g for g in got if isinstance(g, bytes))
    assert frame.startswith(b'RCF1') and frame.endswith(b'\xff\xd8frame')
    page = await client.get('/')
    assert page.status == 200 and 'app.js' in await page.text()
    css = await client.get('/static/style.css')
    assert css.status == 200
    await ws.close()


def test_disagreement_ignores_case_and_punctuation():
    st = fed(('call.state', {'state': 'connected', 't': 0.0}),
             ('dec.lock', {'utt_id': 1, 'words': [{'w': "I'd", 'cam_top': "i'd", 'emg_top': 'i'},
                                                  {'w': 'milk.', 'cam_top': 'milk', 'emg_top': 'milk'}]}),
             ('dec.hyp', {'utt_id': 1, 'locked': ["I'd", 'milk.'], 'unlocked': []}),
             ('call.state', {'state': 'ended', 't': 30.0}))
    assert st.snapshot(30.0)['summary']['disagree'] == {'lips': 0, 'muscles': 50}


def test_face_hidden_stops_counting_when_the_call_ends_and_before_it_starts():
    st = CallState()
    st.feed('cam.tick', {'face_ok': False}, 1.0)                          # hidden before the call
    st.feed('call.state', {'state': 'connected'}, 5.0)
    st.feed('call.state', {'state': 'ended'}, 8.0)
    assert st.snapshot(8.0)['summary']['face_hidden_s'] == 3.0
    assert st.snapshot(60.0)['summary']['face_hidden_s'] == 3.0           # the end screen doesn't keep counting


def test_late_messages_for_an_earlier_sentence_dont_steal_the_caption():
    st = fed(('dec.hyp', {'utt_id': 4, 'locked': ['and'], 'unlocked': ['then']}),
             ('voice.say', {'utt_id': 3, 'words': [], 'offset': 2, 'final': True}),
             ('claude.repair', {'utt_id': 3, 'edits': [{'index': 0, 'from': 'pack', 'to': 'bag'}]}))
    assert st.snapshot(10.0)['austin']['utt_id'] == 4


def test_candidates_clear_when_the_decoder_stops_sending_them():
    st = fed(('dec.hyp', {'utt_id': 1, 'locked': [], 'unlocked': ['coffee'], 'threshold': 0.7,
                          'cands': [{'w': 'coffee', 'cam': .3, 'emg': .2, 'claude': .1}]}),
             ('dec.hyp', {'utt_id': 1, 'locked': ['coffee'], 'unlocked': ['with']}))
    assert st.snapshot(10.0)['austin']['cands'] is None


def test_claude_status_only_from_real_calls():
    st = fed(('call.state', {'state': 'connected', 't': 0.0}))
    assert 'Claude,' not in st.status(10.0)['text'] and 'Cartesia and Twilio connected' in st.status(10.0)['text']
    st.feed('timing.claude', {'stage': 'claude', 'error': 'bad message: KeyError'}, 10.0)   # a bus complaint, no call
    assert 'not answering' not in st.status(10.0)['text']
    st.feed('timing.claude', {'stage': 'claude', 'ms': 6900.0}, 10.0)
    assert 'Claude, Cartesia and Twilio connected' in st.status(10.0)['text']
    assert st.latency()[2] == {'key': 'claude', 'label': 'Claude reply', 'ms': 6900}   # no window: the reply, named so


def test_speaking_runs_from_first_audio_to_played():
    st = fed(('dec.hyp', {'utt_id': 1, 'locked': ['hi'], 'unlocked': []}),
             ('timing.voice', {'stage': 'sonic_first_audio', 'utt_id': 1, 'ms': 190.0}))
    assert st.snapshot(10.0)['austin']['speaking']
    st.feed('timing.voice', {'stage': 'played_to_judge', 'utt_id': 1, 'ms': 900.0}, 11.0)
    assert not st.snapshot(11.0)['austin']['speaking']


def test_replay_moves_call_state_onto_its_clock():
    import msgpack
    from buslog.replay import retime
    msg = msgpack.unpackb(retime(encode('call.state', {'state': 'connected', 't': 5.0}, t_pub=5.0), 100.0)[1])
    assert msg['t'] == 105.0


def test_a_decoder_restart_starts_a_new_sentence():
    st = fed(('dec.hyp', {'utt_id': 7, 'locked': ['bye'], 'unlocked': []}))
    st.feed('dec.hyp', {'utt_id': 1, 'locked': ['hi'], 'unlocked': []}, 10.0)
    st.feed('voice.say', {'utt_id': 1, 'words': ['hi'], 'offset': 0, 'final': True}, 10.5)
    st.feed('dec.hyp', {'utt_id': 1, 'locked': ['hello'], 'unlocked': []}, 30.0)   # after a restart: utt 1 again
    st.feed('voice.say', {'utt_id': 1, 'words': ['hello'], 'offset': 0, 'final': True}, 30.5)
    assert st.snapshot(30.5)['austin']['words'][0]['w'] == 'hello'
    assert [t['text'] for t in st.transcript if t['who'] == 'austin'] == ['hi', 'hello']


def test_voice_indicator_times_out_and_lips_to_voice_counts_first_audio_once():
    st = fed(('dec.lock', {'utt_id': 1, 'words': [{'w': 'hi', 't': 9.7}]}),
             ('timing.voice', {'stage': 'sonic_first_audio', 'utt_id': 1, 'ms': 190.0, 't_pub': 10.0}))
    st.feed('timing.voice', {'stage': 'sonic_first_audio', 'utt_id': 1, 'ms': 190.0, 't_pub': 40.0}, 40.0)  # fallback
    assert list(st.lat['lips_to_voice']) == [pytest.approx(300.0)]
    st.feed('dec.hyp', {'utt_id': 1, 'locked': ['hi'], 'unlocked': []}, 40.0)
    assert st.snapshot(41.0)['austin']['speaking'] and not st.snapshot(49.0)['austin']['speaking']


def test_end_summary_is_built_once():
    st = fed(('call.state', {'state': 'connected'}), ('call.state', {'state': 'ended'}))
    first = st.snapshot(10.0)['summary']
    assert st.snapshot(20.0)['summary'] is first


def unpack(frame):
    import struct
    n = struct.unpack('>I', frame[4:8])[0]
    return json.loads(frame[8:8 + n]), frame[8 + n:]


def test_each_preview_frame_carries_the_face_points_of_that_frame():
    """The pads are drawn from the face of the very frame shown: the cam.face closest in time (within one camera
    frame), or the preview's own points if the camera reader sends them; none closer: no points (the page keeps
    its last placement rather than drawing a stale face)."""
    from dashboard.server import Dashboard
    d = Dashboard(endpoints={'none': 'inproc://dashboard-test-none'})
    a = [[0.1, 0.2]] * 478
    b = [[0.3, 0.4]] * 478
    d.handle(encode('cam.face', {'t': 10.00, 'w': 640, 'h': 360, 'face_ok': True, 'pts': a}))
    d.handle(encode('cam.face', {'t': 10.08, 'w': 640, 'h': 360, 'face_ok': True, 'pts': b}))
    d.handle(encode('cam.preview', {'t': 10.07, 'w': 640, 'h': 360, 'jpeg': b'\xff\xd8x'}))
    head, jpeg = unpack(d.frame)
    assert head['pts'][0] == [0.3, 0.4] and head['t'] == 10.07 and jpeg == b'\xff\xd8x'
    d.handle(encode('cam.preview', {'t': 10.50, 'w': 640, 'h': 360, 'jpeg': b'\xff\xd8y'}))
    assert unpack(d.frame)[0]['pts'] is None                     # no face from that moment
    own = [[0.5, 0.6]] * 478
    d.handle(encode('cam.preview', {'t': 10.51, 'w': 640, 'h': 360, 'jpeg': b'\xff\xd8z', 'pts': own}))
    assert unpack(d.frame)[0]['pts'][0] == [0.5, 0.6]            # the frame's own points win


def test_bad_face_points_never_stop_the_dashboard():
    from dashboard.server import Dashboard
    d = Dashboard(endpoints={'none': 'inproc://dashboard-test-none'})
    d.handle(encode('cam.face', {'t': 5.0, 'w': 640, 'h': 360, 'face_ok': True, 'pts': [[0.1, 0.2], [0.3]]}))
    d.handle(encode('cam.preview', {'t': 5.0, 'w': 640, 'h': 360, 'jpeg': b'\xff\xd8a', 'pts': [1, 2, 3]}))
    head, jpeg = unpack(d.frame)
    assert head['pts'] is None and jpeg == b'\xff\xd8a'        # the picture still goes out
