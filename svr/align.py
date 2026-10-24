"""CTC forced alignment and the two things SVR-1 trains from it (docs/SVR1_SPEC.md section 4):

speaking targets   1 from the first to the last aligned letter, widened by 50 ms each side; 0 elsewhere
partial sentences  cut a clip at frame t; the target is the letters aligned before t (a partial last word included),
                   which is what the live loop sees when it re-reads the start of a sentence every 100 ms

Alignments are computed by the trainers with the current EMA model (Viterbi over the letter CTC) and stored as, per
utterance, the first frame of each target letter (`emit`, int array, one entry per letter).
"""
import numpy as np

FRAME_RATE = 689.06 / 8
SPEAK_WIDEN = int(round(0.050 * FRAME_RATE))  # 50 ms = 4 frames

_VITERBI = None


def _compile():
    from numba import njit

    @njit(cache=True)
    def viterbi(logp, ids, blank):
        t_len = logp.shape[0]
        s_len = ids.shape[0]
        n = 2 * s_len + 1
        ext = np.full(n, blank, np.int64)
        for j in range(s_len):
            ext[2 * j + 1] = ids[j]
        neg = -1e30
        dp = np.full(n, neg)
        back = np.zeros((t_len, n), np.int8)
        dp[0] = logp[0, blank]
        if n > 1:
            dp[1] = logp[0, ext[1]]
        for t in range(1, t_len):
            new = np.full(n, neg)
            for s in range(n):
                best, arg = dp[s], 0
                if s >= 1 and dp[s - 1] > best:
                    best, arg = dp[s - 1], 1
                if s >= 2 and ext[s] != blank and ext[s] != ext[s - 2] and dp[s - 2] > best:
                    best, arg = dp[s - 2], 2
                if best > neg / 2:
                    new[s] = best + logp[t, ext[s]]
                    back[t, s] = arg
            dp = new
        s = n - 1
        if n > 1 and dp[n - 2] > dp[n - 1]:
            s = n - 2
        if dp[s] < neg / 2:
            return np.full(s_len, -1, np.int64)
        emit = np.full(s_len, -1, np.int64)
        for t in range(t_len - 1, -1, -1):
            if s % 2 == 1:
                emit[(s - 1) // 2] = t  # overwritten down to the first frame of the letter's run
            s -= back[t, s]
        return emit
    return viterbi


def ctc_align(logp, ids, blank):
    """First frame of each target letter on the best CTC path; logp (T, C) log-probs, ids (S,). None if no path."""
    global _VITERBI
    ids = np.asarray(ids, np.int64)
    if len(ids) == 0:
        return np.zeros(0, np.int64)
    if _VITERBI is None:
        _VITERBI = _compile()
    emit = _VITERBI(np.ascontiguousarray(logp, dtype=np.float64), ids, int(blank))
    return None if emit[0] < 0 else emit


def scale_emit(emit, old_frames, new_frames):
    """Alignment of a clip resampled in time from old_frames to new_frames (speed augmentation)."""
    if old_frames == new_frames:
        return emit
    return np.minimum(np.floor(emit * (new_frames / old_frames)).astype(np.int64), new_frames - 1)


def speak_target(emit, n_frames, widen=SPEAK_WIDEN):
    """(n_frames,) float32: 1 from the first to the last aligned letter, widened by `widen` frames each side."""
    y = np.zeros(n_frames, np.float32)
    if emit is not None and len(emit):
        y[max(0, int(emit[0]) - widen):min(n_frames, int(emit[-1]) + widen + 1)] = 1
    return y


def prefix_target(emit, ids, t):
    """Target letters of a clip cut at frame t: the letters whose first frame is before t."""
    return np.asarray(ids)[:int(np.searchsorted(emit, t, side='left'))]
