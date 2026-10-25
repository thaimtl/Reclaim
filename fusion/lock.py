"""The lock model (docs/SVR1_SPEC.md section 6): one probability per unlocked word that it is right and final.

    p_lock(w) = sigmoid(theta . f(w) + bias)
    a word locks when p_lock >= tau, tau = clip(tau0 - kappa * predictability, tau_min, tau_max),
    and only when every word before it is locked, and camera-EMG agreement is at least agree_floor.

Features f(w), in FEATURES order:
    joint         the joint search's posterior for the word at its position
    cam, emg      the camera-only and EMG-only posteriors (each reader rescoring the candidates alone)
    agree         min(cam, emg)
    claude        Claude's prediction weight for the word, capped at 0.3, so Claude alone can never lock a word
    next_started  1 when a later word exists in the best candidate (this word is finished), else 0
    pause         the mouth is still: 1 - mean camera activity over the last 0.3 s
    quiet         the muscles are quiet: 1 - mean EMG speaking probability over the last 0.3 s
    stable        log(1 + passes this word has stayed the best candidate's word at this position)
    face          the face is visible on the newest frame
    emg_quality   1 clean .. 0 (railed or flat pads)

theta is a logistic regression fitted on sessions 1 and 2 (python -m fusion.offline fit): every word on every pass is
an example, labelled right when it matches the true word at its position and never changes later in the segment.
tau0 is chosen on the same sessions, each scored by a model fitted on the other one, so that at most 2% of locked
words are wrong with the shortest delay. kappa can't be fitted offline (Claude has no offline replay): it is a set
value, checked live.
"""
import numpy as np

FEATURES = ('joint', 'cam', 'emg', 'agree', 'claude', 'next_started', 'pause', 'quiet', 'stable', 'face',
            'emg_quality')
CLAUDE_CAP = 0.3


def features(joint, cam, emg, claude, next_started, pause, quiet, stable, face, emg_quality):
    return np.array([joint, cam, emg, min(cam, emg), min(claude, CLAUDE_CAP), float(next_started), pause, quiet,
                     np.log1p(stable), float(face), emg_quality])


class LockModel:
    def __init__(self, theta=None, bias=0.0, tau0=0.9, kappa=0.1, tau_min=0.75, tau_max=0.97, agree_floor=0.05):
        if theta is None:   # before fitting: lean on the joint posterior and a finished word
            theta = np.zeros(len(FEATURES))
            theta[FEATURES.index('joint')] = 6.0
            theta[FEATURES.index('next_started')] = 2.0
            bias = -5.0
        self.theta, self.bias = np.asarray(theta, np.float64), float(bias)
        self.tau0, self.kappa, self.tau_min, self.tau_max = tau0, kappa, tau_min, tau_max
        self.agree_floor = agree_floor

    def prob(self, f):
        z = np.asarray(f, np.float64) @ self.theta + self.bias
        return 1.0 / (1.0 + np.exp(-z))

    def threshold(self, predictability=0.0):
        return float(np.clip(self.tau0 - self.kappa * (predictability or 0.0), self.tau_min, self.tau_max))

    def may_lock(self, f, p, tau):
        """The lock rule for one word (its predecessors already locked)."""
        return p >= tau and f[FEATURES.index('agree')] >= self.agree_floor

    def fit(self, X, y, l2=1.0, iters=200):
        """Logistic regression by Newton's method with an L2 penalty (not on the bias); features standardised inside."""
        X, y = np.asarray(X, np.float64), np.asarray(y, np.float64)
        mu, sd = X.mean(0), X.std(0) + 1e-6
        Z = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
        w = np.zeros(Z.shape[1])
        reg = np.full(Z.shape[1], l2)
        reg[-1] = 0.0
        for _ in range(iters):
            p = 1 / (1 + np.exp(-Z @ w))
            g = Z.T @ (p - y) + reg * w
            H = (Z * (p * (1 - p))[:, None]).T @ Z + np.diag(reg) + 1e-9 * np.eye(len(w))
            step = np.linalg.solve(H, g)
            w -= step
            if np.abs(step).max() < 1e-8:
                break
        self.theta = w[:-1] / sd
        self.bias = float(w[-1] - (w[:-1] / sd) @ mu)
        return self

    def to_dict(self):
        return {'theta': self.theta.tolist(), 'bias': self.bias, 'tau0': self.tau0, 'kappa': self.kappa,
                'tau_min': self.tau_min, 'tau_max': self.tau_max, 'agree_floor': self.agree_floor,
                'features': list(FEATURES)}

    @classmethod
    def from_dict(cls, d):
        return cls(d['theta'], d['bias'], d['tau0'], d['kappa'], d['tau_min'], d['tau_max'], d['agree_floor'])
