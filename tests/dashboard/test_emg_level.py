"""The pad meters: rig/filters.py's filter in numpy, and levels that read 0 for a still face (hum, drift, noise)
and fill for a muscle burst."""
import numpy as np
import pytest

from dashboard.emg_level import EmgFilter, PadLevels

FS = 1000
RNG = np.random.default_rng(0)


def test_the_filter_is_rig_filters_sample_for_sample():
    ref = pytest.importorskip('rig.filters')               # needs scipy (venv-live)
    x = RNG.normal(512, 20, (3000, 4))
    want = ref.EmgFilter()(x)
    f = EmgFilter()
    got = np.vstack([f(x[i:i + 10]) for i in range(0, len(x), 10)])     # in 10 ms messages, as emg.raw
    assert np.allclose(got, want, atol=1e-9)


def raw(seconds, hum=30.0, noise=2.0, burst=None):
    """10-bit samples around 512: mains hum at 60 Hz, slow drift, sensor noise, and an optional muscle burst
    (broadband) on one channel."""
    t = np.arange(int(seconds * FS)) / FS
    x = 512 + hum * np.sin(2 * np.pi * 60 * t)[:, None] + 20 * np.sin(2 * np.pi * 0.3 * t)[:, None] \
        + RNG.normal(0, noise, (len(t), 4))
    if burst is not None:
        x[:, burst] += RNG.normal(0, 25, len(t))
    return np.clip(x, 0, 1023)


def feed(pl, x):
    out = []
    for i in range(0, len(x), 10):
        out.append(pl.push(x[i:i + 10], np.ones(4, bool)).copy())
    return np.array(out)


def test_a_still_face_reads_zero_despite_hum_and_drift():
    pl = PadLevels(4)
    lv = feed(pl, raw(6.0))
    assert (lv[100:] > 0.15).mean() < 0.02                   # hum and drift are filtered out, noise is the floor


def test_a_muscle_burst_fills_its_own_pad_only():
    pl = PadLevels(4)
    feed(pl, raw(4.0))                                        # rest first: each pad learns its floor
    lv = feed(pl, raw(1.0, burst=2))
    assert lv[-1][2] > 0.6 and max(lv[-1][[0, 1, 3]]) < 0.15
    lv = feed(pl, raw(1.0))                                   # and it falls back when he stops
    assert lv[-1].max() < 0.15


def test_a_pad_off_the_skin_reads_zero():
    pl = PadLevels(4)
    x = raw(1.0, burst=1)
    out = [pl.push(x[i:i + 10], np.array([True, False, True, True])).copy() for i in range(0, len(x), 10)]
    assert out[-1][1] == pytest.approx(0, abs=1e-6)


def test_a_pad_that_loses_contact_comes_back_at_rest_not_lit():
    pl = PadLevels(4)
    feed(pl, raw(4.0))                                        # resting floors learned
    rail = raw(3.0)
    rail[:, 1] = 1023                                         # pad A1 slips off: its channel sits on the rail
    for i in range(0, len(rail), 10):
        pl.push(rail[i:i + 10], np.array([True, False, True, True]))
    after = feed(pl, raw(2.0))                                # back on the skin, at rest
    assert after[:, 1].max() < 0.15


def test_the_dashboard_shows_the_filtered_levels():
    from dashboard.state import CallState
    st = CallState()
    x = raw(4.0)
    x[-1000:, 3] += RNG.normal(0, 25, 1000)                   # the upper lip works in the last second
    for i in range(0, len(x), 10):
        st.feed('emg.raw', {'t0': i / FS, 'fs': FS, 'samples': x[i:i + 10].astype(np.uint16), 'counter': i}, i / FS)
    pads = {p['pin']: p['level'] for p in st.snapshot(4.0)['pads']}
    assert pads['A3'] > 0.5 and max(pads['A0'], pads['A1'], pads['A2']) < 0.15
