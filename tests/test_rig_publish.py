"""rig/publish.py's batching and stamping on fake chunks, no hardware: python -m pytest tests/test_rig_publish.py"""
import numpy as np

from rig.publish import batches


def chunk(t_read, start, n, dropped=0):
    rows = (np.arange(start, start + n)[:, None] * [1, 2, 3, 4] % 1024).astype(np.uint16)
    return t_read, rows, dropped


def test_batches_of_ten_with_regular_times():
    # 25 samples arrive in bursts, each read 3 ms after its last sample
    chunks = [chunk(0.010 + 0.003, 0, 11), chunk(0.017 + 0.003, 11, 7), chunk(0.025 + 0.003, 18, 7)]
    out = list(batches(chunks))
    assert [len(m['samples']) for m in out] == [10, 10, 5]
    assert [m['counter'] for m in out] == [0, 10, 20]
    assert all(m['fs'] == 1000 and m['samples'].dtype == np.uint16 and m['samples'].shape[1] == 4 for m in out)
    assert np.array_equal(np.concatenate([m['samples'] for m in out]), chunk(0, 0, 25)[1])
    # sample k is at k ms plus the smallest delay seen (3 ms), whatever the burst sizes, give or take the leak
    for m in out:
        assert abs(m['t0'] - (0.003 + m['counter'] / 1000)) < 1e-4


def test_lower_envelope_ignores_late_reads():
    out = list(batches([chunk(0.0095, 0, 10), chunk(0.030, 10, 10)]))  # second read is 10 ms late
    assert abs(out[1]['t0'] - (out[0]['t0'] + 0.010)) < 2e-4  # 10 ms later, not 20: only the leak's 0.1 ms


def test_gap_splits_batches_and_shows_in_counter():
    # 5 frames lost before the second chunk
    out = list(batches([chunk(0.008, 0, 7), chunk(0.020, 12, 8, dropped=5)]))
    assert [(m['counter'], len(m['samples'])) for m in out] == [(0, 7), (12, 8)]
    assert out[0]['counter'] + len(out[0]['samples']) < out[1]['counter']
    assert abs(out[1]['t0'] - out[0]['t0'] - 0.012) < 1e-3
    assert out[1]['samples'][0][0] == 12


def test_leak_follows_a_slow_uno():
    # the Uno runs 0.3% slow: real time per sample is 1.003 ms; reads come right after the last sample
    chunks, t = [], 0.0
    for i in range(1000):
        t = (i * 10 + 9) * 1.003e-3
        chunks.append(chunk(t, i * 10, 10))
    out = list(batches(chunks))
    last = out[-1]
    true_t0 = last['counter'] * 1.003e-3
    assert abs(last['t0'] - true_t0) < 0.005  # without the leak this would be 30 ms off after 10 s
