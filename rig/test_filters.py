import numpy as np

from filters import FS, EmgFilter


def synth(drift=1.0, seconds=4, channels=6, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(seconds * FS)[:, None] / FS
    phase = rng.uniform(0, 2 * np.pi, channels)
    f = 60 * drift
    hum = 30 * np.sin(2 * np.pi * f * t + phase) + 5 * np.sin(4 * np.pi * f * t) + np.sin(6 * np.pi * f * t)
    burst = rng.normal(0, 3, (t.size, channels)) * ((t > 2) & (t < 3))
    return 512 + hum + burst


def test_removes_hum_and_keeps_bursts():
    for drift in (1.0, 0.998, 1.002):
        y = EmgFilter()(synth(drift))
        rest, burst = y[500:2000], y[2100:2900]
        assert rest.std() < 1.2, (drift, rest.std())
        assert burst.std() > 2.4, (drift, burst.std())


def test_chunked_matches_whole():
    x = synth()
    whole = EmgFilter()(x)
    live, edges = EmgFilter(), [0, 1, 7, 100, 999, 2500, len(x)]
    chunked = np.vstack([live(x[a:b]) for a, b in zip(edges, edges[1:])])
    assert np.allclose(whole, chunked)


if __name__ == "__main__":
    test_removes_hum_and_keeps_bursts()
    test_chunked_matches_whole()
    print("ok")
