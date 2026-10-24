import sys

import numpy as np

from filters import EmgFilter

WIN = 250
SETTLE = 300


def window_levels(y, a, b):
    n = (b - a) // WIN
    return y[a:a + n * WIN].reshape(n, WIN, -1).std(1)


def segments(step):
    edges = [0, *(np.flatnonzero(step[1:] != step[:-1]) + 1), len(step)]
    return [(step[a], max(a, SETTLE), b) for a, b in zip(edges, edges[1:]) if b - max(a, SETTLE) >= WIN]


def main(path, detail=False):
    d = np.load(path)
    raw, step = d["raw"].astype(float), d["step"]
    y = EmgFilter()(raw)
    segs = segments(step)
    if not any(s == "rest" for s, _, _ in segs):
        sys.exit("The recording needs at least one rest step.")
    rest_raw = np.vstack([raw[a:b] for s, a, b in segs if s == "rest"])
    rest = np.maximum(np.median(np.vstack([window_levels(y, a, b) for s, a, b in segs if s == "rest"]), 0), 1e-3)
    tasks = [(s, np.percentile(window_levels(y, a, b), 90, 0) / rest) for s, a, b in segs if s != "rest"]
    clipped = ((raw <= 2) | (raw >= 1021)).mean(0) * 100
    hum = rest_raw.std(0)

    print(f"{len(raw)} samples, {len(raw) / 1000:.0f} s")
    print("ch  clipped  hum   rest  " + "  ".join(f"{s[:10]:>10}" for s, _ in tasks) + "  verdict")
    for c in range(raw.shape[1]):
        score = np.median([r[c] for _, r in tasks])
        if clipped[c] > 1:
            verdict = "MAXES OUT: loose pad or REF"
        elif score < 1.5:
            verdict = "NO RESPONSE: pads touching, cable not seated, or no skin contact"
        elif score < 2:
            verdict = "WEAK: press pads and REF down"
        else:
            verdict = "OK" + (", high hum so press REF down" if hum[c] > 20 else "")
        print(f"A{c}  {clipped[c]:6.1f}%  {hum[c]:4.0f}  {rest[c]:4.1f}  "
              + "  ".join(f"{r[c]:9.1f}x" for _, r in tasks) + f"  {verdict}")

    if detail:
        for s, a, b in segs:
            for i, lv in enumerate(window_levels(y, a, b)):
                print(f"{(a + i * WIN) / 1000:6.2f}s {s[:10]:<10} " + " ".join(f"{v:4.1f} {'#' * min(10, int(v)):<10}" for v in lv))


if __name__ == "__main__":
    main(sys.argv[1], "--detail" in sys.argv)
