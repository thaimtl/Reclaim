"""svr.infer: recorded clips replayed through the live EMG reader into a buslog session, joined to a camera log by
fusion.evaluate --decode --emg; load / score / bench. Tiny random SVR-1 on CPU; needs the SVR-1 venv (skipped
without it)."""
import json

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('svr.live', reason='needs the SVR-1 venv (scipy, soundfile, unidecode)')

import buslog  # noqa: E402
from fusion import evaluate  # noqa: E402
from svr import infer  # noqa: E402
from svr.live import LiveSVR1  # noqa: E402
from tests.svr_fixture import raw_signal, tiny_checkpoint  # noqa: E402

CLIP_T0 = (5000.0, 5004.0)


@pytest.fixture(scope='module')
def ckpt(tmp_path_factory):
    return tiny_checkpoint(tmp_path_factory.mktemp('svr') / 'model.pt')


def recordings(root, session='s1'):
    folder = root / 'session_01'
    folder.mkdir(parents=True)
    rows = []
    for k, t0 in enumerate(CLIP_T0):
        raw = raw_signal(2.0, seed=k, bursts=[(0.5, 1.5)])
        np.save(folder / f'c{k}_emg.npy', raw)
        np.save(folder / f'c{k}_emg_ts.npy', t0 + np.arange(len(raw)) / 1000)
        rows.append({'id': f'c{k}', 'session': session, 'set': 'C', 'prompt': 'i want', 'emg': f'c{k}_emg.npy',
                     'emg_ts': f'c{k}_emg_ts.npy', 'clip_t0': t0, 'clip_t1': t0 + 2.0})
    (folder / 'manifest.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return root


def replayed(ckpt, tmp_path, name, session='s1'):
    from svr.recordings import load_manifest
    rec = recordings(tmp_path / f'rec_{name}', session)
    out = tmp_path / f'log_{name}'
    n_clips, n_reads = infer.replay_sessions(str(ckpt), load_manifest(str(rec)), out, device='cpu')
    return out, n_clips, n_reads


def test_replay_logs_reads_on_the_recorders_clock(ckpt, tmp_path):
    out, n_clips, n_reads = replayed(ckpt, tmp_path, 'a')
    reads = list(buslog.read(out, ['emg.read']))
    assert n_clips == 2 and n_reads == len(reads) and 30 <= n_reads <= 42   # ~19 reads per 2 s clip
    for t, topic, m in reads:
        clip = max(c for c in CLIP_T0 if c <= t)
        assert m['ok'] and m['t_pub'] == t and clip < t <= clip + 2.01
        assert clip - 0.01 <= m['frame_t'][0] and m['frame_t'][-1] <= t
    meta = json.loads((out / 'meta.json').read_text())
    assert meta['source'] == 'svr.infer' and meta['clips'] == ['c0', 'c1']


def test_replay_is_deterministic(ckpt, tmp_path):
    a = [m for _, _, m in buslog.read(replayed(ckpt, tmp_path, 'a')[0], ['emg.read'])]
    b = [m for _, _, m in buslog.read(replayed(ckpt, tmp_path, 'b')[0], ['emg.read'])]
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert np.array_equal(x['letter_logp'], y['letter_logp']) and np.array_equal(x['frame_t'], y['frame_t'])


def test_unknown_session_is_calibrated(ckpt, tmp_path, monkeypatch):
    seen = []
    real = LiveSVR1.calibrate
    monkeypatch.setattr(LiveSVR1, 'calibrate', lambda self, raw, **k: seen.append(len(raw)) or real(self, raw, **k))
    _, n_clips, n_reads = replayed(ckpt, tmp_path, 'c', session='s9')
    assert seen == [4000] and n_clips == 2 and n_reads > 0


def test_evaluate_joins_the_emg_log(ckpt, tmp_path):
    from tests.test_evaluate import CLIPS, camera_records, write_manifest, write_session
    emg_log = tmp_path / 'emg'
    w = buslog.Writer(emg_log)
    from bus.team import encode
    live = LiveSVR1(str(ckpt), device='cpu')
    reader = infer.EmgReader(live)
    raw = raw_signal(1.5, bursts=[(0.2, 1.0)])
    for t, topic, msg in infer.replay_clip(reader, raw, 9.9 + np.arange(len(raw)) / 1000):
        w.write(t, 'svr', encode(topic, msg, t_pub=t))
    w.close()
    cam = write_session(tmp_path / 'cam', camera_records())
    man = write_manifest(tmp_path / 'm.jsonl', CLIPS)
    units = tmp_path / 'units.txt'
    from tests.test_fusion import U
    units.write_text(''.join(f'{U.get(i, f"<x{i}>")} {i}\n' for i in range(1, max(U) + 1)))
    with_emg = evaluate.main([str(cam), '--manifest', str(man), '--decode', '--emg', str(emg_log),
                              '--units', str(units), '--no-lookalikes'])
    without = evaluate.main([str(cam), '--manifest', str(man), '--decode', '--no-emg',
                             '--units', str(units), '--no-lookalikes'])
    assert with_emg['emg'] == str(emg_log) and without['emg'] is None
    assert with_emg['overall']['wer'] is not None and without['overall']['wer'] is not None


def test_load_refuses_old_preprocessing_and_score_ranks(ckpt, tmp_path):
    state = torch.load(ckpt, weights_only=False)
    state['preprocess'] = 'gaddy'
    torch.save(state, tmp_path / 'old.pt')
    with pytest.raises(ValueError, match='causal'):
        infer.load(tmp_path / 'old.pt')
    assert infer.load(ckpt).training is False
    live = LiveSVR1(str(ckpt), device='cpu')
    live.feed(raw_signal(2.0), 0.0)
    live.read(camera_active=True)
    s = infer.score(live, ['i want water', 'i want what'])
    assert s.shape == (2,) and np.all(np.isfinite(s))


def test_bench_runs(ckpt):
    out = infer.bench(str(ckpt), device='cpu', windows=(1.0, 2.0), reps=2)
    assert set(out) == {1.0, 2.0} and all(v > 0 for v in out.values())
