"""SVR-1 and the real-time fusion around it (revised design, docs/SVR1_SPEC.md), in the visual language of the Varest
method figure. Panels: (A)-(F) SVR-1's contributions, (G) the real-time loop, (H)-(I) the joint search's language model.

    venv-svr/bin/python figures/make_architecture.py   ->  figures/svr1_architecture.{pdf,png}
"""
import os

import matplotlib
import numpy as np
from matplotlib import font_manager as fm
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, PathPatch, Rectangle
from matplotlib.path import Path

matplotlib.use('Agg')
HERE = os.path.dirname(os.path.abspath(__file__))
for f in os.listdir(os.path.join(HERE, 'fonts')):
    if f.endswith('.ttf'):
        fm.fontManager.addfont(os.path.join(HERE, 'fonts', f))
plt.rcParams.update({
    'font.family': 'Liberation Sans', 'mathtext.fontset': 'custom', 'mathtext.rm': 'Liberation Sans',
    'mathtext.it': 'Liberation Sans:italic', 'mathtext.bf': 'Liberation Sans:bold', 'pdf.fonttype': 42,
})

INK = '#1f1f1f'
FILL = {'op': '#ffffff', 'mod': '#ddebf7', 'new': '#6fa3d8', 'sup': '#c9b8e8', 'lit': '#a9c4a2'}
EMG_TOKEN, PHON_TOKEN, DISPUTED = '#f08c84', '#fde27a', '#d62728'
PANEL = '#7ea6d9'

fig = plt.figure(figsize=(20, 14.4))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 2000)
ax.set_ylim(1440, 0)
ax.axis('off')


def flame(x, y, s=1.0):
    """Flame icon (trainable): black outer flame with a side tongue and a white inner flame."""
    P = lambda pts: [(x + a * s, y + b * s) for a, b in pts]
    outer = [(0, 0), (-9, 0), (-10, -9), (-7, -15), (-6, -12), (-4, -11), (-3, -12), (-3, -18), (-2, -21),
             (1, -27), (3, -22), (9, -18), (8, -10), (8, -3), (5, 0), (0, 0)]
    inner = [(0, -1), (-4, -1), (-4, -6), (0, -11), (4, -6), (4, -1), (0, -1)]
    ax.add_patch(PathPatch(Path(P(outer), [Path.MOVETO] + [Path.CURVE4] * 15), facecolor=INK, edgecolor=INK,
                           lw=0.4, zorder=6))
    ax.add_patch(PathPatch(Path(P(inner), [Path.MOVETO] + [Path.CURVE4] * 6), facecolor='white',
                           edgecolor='none', zorder=7))


def snow(x, y, size=15):
    ax.text(x, y, '❄', fontsize=size, fontfamily='DejaVu Sans', ha='center', va='center', color=INK,
            zorder=6)


def box(x, y, w, h, text, kind='mod', fs=12.5, icon=None, bold=False, italic=False):
    dashed = kind == 'sup'
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=7', facecolor=FILL[kind],
                                edgecolor=INK, lw=1.9, linestyle=(0, (4, 2.5)) if dashed else '-', zorder=3))
    shift = -5 if icon in ('fire', 'snow') else 0
    ax.text(x + w / 2 + shift, y + h / 2, text, ha='center', va='center', fontsize=fs, color=INK, zorder=5,
            fontweight='bold' if bold else 'normal', style='italic' if italic else 'normal', linespacing=1.15)
    if icon == 'fire':
        flame(x + w - 10, y + 21, 0.62)
    elif icon == 'snow':
        snow(x + w - 11, y + 12, size=13)
    elif icon == 'snow-left':
        snow(x + 13, y + 13)


def tokens(x, y, n, color, size=17, gap=4, disputed=(), edge=INK):
    for i in range(n):
        ax.add_patch(Rectangle((x + i * (size + gap), y), size, size, facecolor=color,
                               edgecolor=DISPUTED if i in disputed else edge,
                               lw=2.6 if i in disputed else 1.3, zorder=4))
    return x + n * (size + gap) - gap


def arrow(pts, dashed=False, color=INK, lw=1.8):
    ls = (0, (4, 3)) if dashed else '-'
    if len(pts) > 2:
        xs, ys = zip(*pts[:-1])
        ax.plot(xs, ys, color=color, lw=lw, linestyle=ls, zorder=2, solid_capstyle='butt')
    ax.annotate('', xy=pts[-1], xytext=pts[-2], zorder=2,
                arrowprops=dict(arrowstyle='-|>', color=color, lw=lw, linestyle=ls, mutation_scale=15,
                                shrinkA=0, shrinkB=0))


def label(x, y, s, fs=11, ha='center', italic=False, color=INK, rot=0):
    ax.text(x, y, s, ha=ha, va='center', fontsize=fs, color=color, style='italic' if italic else 'normal',
            zorder=5, rotation=rot)


def panel(x, y, w, h, title):
    ax.add_patch(Rectangle((x, y), w, h, facecolor='none', edgecolor=PANEL, lw=1.8, linestyle=(0, (5, 3)),
                           zorder=1))
    ax.text(x + w / 2, y + 16, title, ha='center', va='center', fontsize=13, fontweight='bold', color=INK)


rng = np.random.default_rng(3)


def emg_image(x, y, w, h, n=6, bursts=None, amp=1.0, overlap=0.0, dark=True):
    ax.add_patch(Rectangle((x, y), w, h, facecolor='#20232b' if dark else '#ffffff', edgecolor=INK, lw=1.5,
                           zorder=3))
    t = np.linspace(0, 1, 260)
    cols = ['#f08c84', '#fde27a', '#8fd19e', '#7fb8ff', '#d9a6ff', '#ffb36b']
    bursts = bursts if bursts is not None else [0.2, 0.45, 0.72]
    for c in range(n):
        env = sum(np.exp(-((t - b - 0.03 * c) ** 2) / 0.0025) for b in bursts)
        if overlap:
            env = env + overlap * np.roll(env, 18)
        sig = amp * env * rng.normal(0, 1, t.size) * 0.35 + rng.normal(0, 0.03, t.size)
        yc = y + (c + 0.5) * h / n
        ax.plot(x + 4 + t * (w - 8), yc + np.clip(sig, -1.2, 1.2) * h / (2.6 * n), color=cols[c % 6],
                lw=0.8, zorder=4)


def video_image(x, y, w, h):
    ax.add_patch(Rectangle((x, y), w, h, facecolor='#2b2d35', edgecolor=INK, lw=1.5, zorder=3))
    cx, cy = x + w / 2, y + h / 2 + 4
    ax.add_patch(matplotlib.patches.Ellipse((cx, cy), w * 0.48, h * 0.86, facecolor='#e8c4a6', edgecolor='none',
                                            zorder=4))
    for dx in (-12, 12):
        ax.add_patch(matplotlib.patches.Ellipse((cx + dx, cy - 12), 8, 4, facecolor='#3a2a20', zorder=5))
    ax.add_patch(matplotlib.patches.Ellipse((cx, cy + 15), 24, 9, facecolor='#c2504a', edgecolor='#7a2a26',
                                            lw=1, zorder=5))
    for i, px in enumerate(np.linspace(cx - 12, cx + 12, 5)):  # face landmarks on the lips
        ax.add_patch(Circle((px, cy + 15 + (4 if i % 2 else -4)), 1.6, facecolor='#5ef0a0', zorder=6))


# ---------------------------------------------------------------- legend
legend = [('Operator', 'op'), ('Module', 'mod'), ('New (ours)', 'new'), ('Supervision', 'sup'),
          ('Literal', 'lit')]
for i, (name, kind) in enumerate(legend):
    yy = 8 + i * 30
    ax.add_patch(FancyBboxPatch((8, yy), 22, 20, boxstyle='round,pad=0,rounding_size=3', facecolor=FILL[kind],
                                edgecolor=INK, lw=1.5, linestyle=(0, (3, 2)) if kind == 'sup' else '-'))
    label(38, yy + 10, name, fs=11.5, ha='left')
for i, (name, col, disp) in enumerate([('EMG state', EMG_TOKEN, ()), ('Phoneme token', PHON_TOKEN, ()),
                                       ('Disputed token', PHON_TOKEN, (0,))]):
    yy = 158 + i * 30
    tokens(8, yy + 1, 1, col, size=20, disputed=disp)
    label(38, yy + 10, name, fs=11.5, ha='left')
flame(19, 268, 0.62)
label(38, 258, 'Trained', fs=11.5, ha='left')
snow(19, 287)
label(38, 287, 'Frozen', fs=11.5, ha='left')

# ---------------------------------------------------------------- EMG path (top row)
emg_image(170, 20, 140, 80, n=4)
label(240, 110, '4-ch EMG (sites 3, 2, 6, 1)', fs=11)
box(335, 25, 130, 70, 'Live hum filter\nrig/filters.py\n→ 689 Hz', 'op', fs=10.5)
box(490, 25, 130, 70, 'Scale +\nSession\nAdapter', 'mod', fs=11.5, icon='fire')
box(645, 25, 130, 70, 'Conv Front-end\n8× down', 'mod', fs=11, icon='fire')
box(800, 15, 175, 90, 'Transformer ×12\nRoPE · SwiGLU\nFA2 (H100) / SDPA', 'mod', fs=11.5,
    icon='fire')
box(800, 130, 175, 50, 'Rate Token$^{(E)}$ (FiLM)', 'new', fs=11.5, icon='fire')
arrow([(310, 60), (335, 60)])
arrow([(465, 60), (490, 60)])
arrow([(620, 60), (645, 60)])
arrow([(775, 60), (800, 60)])
arrow([(400, 95), (400, 155), (800, 155)])
label(600, 146, 'EMG envelope', fs=10.5, italic=True)
arrow([(887, 130), (887, 105)])
label(887, 194, 'pace + effort → voice speed, volume', fs=9.5, italic=True)
tok_end = tokens(1000, 45, 8, EMG_TOKEN)
label(1081, 30, '86 frames/s EMG states', fs=10.5)
arrow([(975, 54), (1000, 54)])

# heads
ax.plot([1165, 1180], [54, 54], color=INK, lw=1.8)
ax.plot([1180, 1180], [31, 181], color=INK, lw=1.8)
heads = [(10, 'Letter CTC', 'mod'), (60, 'Phoneme CTC', 'mod'), (110, 'Articulator Head', 'mod'),
         (160, 'Speaking Head', 'new')]
for yy, name, kind in heads:
    box(1195, yy, 160, 42, name, kind, fs=10.5, icon='fire')
    arrow([(1180, yy + 21), (1195, yy + 21)])
box(1370, 110, 140, 42, 'SPARC targets', 'sup', fs=11.5, icon='snow')
label(1440, 100, 'from parallel audio', fs=10, italic=True)
arrow([(1370, 131), (1355, 131)], dashed=True)
box(1195, 222, 160, 42, 'Partial-Sentence\nTraining$^{(F)}$', 'sup', fs=11)
arrow([(1272, 222), (1272, 202)], dashed=True)
box(1420, 160, 160, 42, 'start · end · word ends', 'lit', fs=11)
arrow([(1355, 181), (1420, 181)])
label(1500, 214, 'to the joint search and the lock', fs=9.5, italic=True)

# ---------------------------------------------------------------- camera path (lower row)
video_image(170, 300, 140, 80)
label(240, 390, 'Webcam video', fs=11)
box(335, 305, 130, 70, 'Auto-AVSR\nre-read\nevery 100 ms', 'mod', fs=11, icon='snow')
box(335, 410, 130, 50, 'Mouth activity\n+ expression', 'op', fs=11)
box(160, 468, 150, 46, 'Endpoint$^{(G)}$: EMG\nspeaking + lips still', 'new', fs=10)
arrow([(335, 452), (250, 452), (250, 468)])
box(845, 222, 150, 44, 'EMG top-N\n(letter-CTC beam)', 'lit', fs=10.5)
arrow([(920, 266), (920, 331)])
arrow([(310, 340), (335, 340)])
arrow([(300, 380), (300, 435), (335, 435)])
box(490, 305, 170, 70, 'Camera CTC probs\nper frame: “grab\nthe bat” first', 'lit', fs=10.5)
arrow([(465, 340), (490, 340)])
box(685, 305, 150, 70, 'Look-alike\nExpansion$^{(C)}$\nin the joint search', 'new', fs=11)
arrow([(660, 340), (685, 340)])
tokens(860, 331, 7, PHON_TOKEN, disputed=(4, 5))
label(927, 362, 'candidate phonemes', fs=10)
arrow([(835, 340), (860, 340)])
box(1030, 305, 170, 70, 'Disputed-Span\nVerifier$^{(B)}$', 'new', fs=12.5, icon='fire')
arrow([(1003, 340), (1030, 340)])
arrow([(1080, 63), (1080, 305)])
label(1088, 250, 'K, V', fs=10, ha='left', italic=True)
box(1420, 315, 160, 50, '$S_{emg}$ = CTC + verifier', 'op', fs=11.5)
arrow([(1200, 340), (1420, 340)])
arrow([(1500, 202), (1500, 315)])
box(1208, 410, 184, 50, 'Homophene-Contrastive\nLoss$^{(A)}$', 'sup', fs=11)
arrow([(1440, 365), (1440, 390), (1300, 390), (1300, 410)], dashed=True)
box(1420, 410, 160, 50, 'Claude\npredict + repair', 'mod', fs=11.5, icon='snow')
arrow([(1580, 435), (1600, 435), (1600, 385), (1620, 385)])
label(1500, 470, 'whole conversation, judge via Ink', fs=9.5, italic=True)

# ---------------------------------------------------------------- gate and output
box(1620, 150, 160, 50, '$r$: camera conf · face\nEMG quality · rate', 'lit', fs=10.5)
box(1620, 230, 160, 60, 'Reliability\nGate$^{(D)}$', 'new', fs=12.5, icon='fire')
arrow([(1700, 200), (1700, 230)])
box(1620, 320, 160, 80, 'Joint Search\n$\\Sigma_m\\, g_m\\, s_m$ + hints', 'op', fs=11.5)
arrow([(1700, 290), (1700, 320)])
label(1708, 305, '$g_{cam}, g_{emg}, g_{lm}$', fs=10, ha='left')
arrow([(1580, 340), (1620, 340)])
arrow([(575, 375), (575, 495), (1660, 495), (1660, 400)])
label(1120, 487, 'camera log-probability', fs=10, italic=True)
box(1675, 418, 125, 48, 'Conversational\nLM$^{(H)}$ + cache$^{(I)}$', 'new', fs=9.5)
arrow([(1737, 418), (1737, 400)])
label(1737, 479, 'judge (Ink) + locked words', fs=8.5, italic=True)
box(1810, 335, 160, 50, 'Confidence\nLock$^{(G)}$', 'new', fs=11.5, icon='fire')
arrow([(1780, 360), (1810, 360)])
box(1810, 420, 160, 50, '“grab the mat”\n→ Cartesia, call', 'lit', fs=11)
arrow([(1890, 385), (1890, 420)])
arrow([(400, 460), (400, 525), (1990, 525), (1990, 175), (1780, 175)])
label(1200, 517, 'mouth activity, face visible', fs=10, italic=True)

# ---------------------------------------------------------------- detail panels
PY, PH, PW, GAP = 545, 450, 323, 10
px = [5 + i * (PW + GAP) for i in range(6)]

# (A) Homophene-contrastive training
x0 = px[0]
panel(x0, PY, PW, PH, '(A) Homophene-Contrastive Training')
cx = x0 + PW / 2
box(cx - 120, PY + 38, 240, 36, 'true: “grab the mat”', 'lit', fs=11.5)
box(cx - 120, PY + 96, 240, 40, 'Phonemes → Visemes\nlexicon + viseme map', 'op', fs=11)
box(cx - 120, PY + 158, 240, 40, 'Swap Lip Look-alikes\nm ↔ b ↔ p,  f ↔ v,  t ↔ d ↔ n', 'op',
    fs=11)
for i, s in enumerate(['grab the bat', 'grab the pat', 'crab the mat']):
    box(x0 + 14 + i * 100, PY + 222, 94, 30, s, 'lit', fs=9.5)
label(cx - 8, PY + 266, 'N look-alike negatives', fs=10, italic=True, ha='right')
box(cx - 120, PY + 280, 240, 40, 'CTC + Verifier Scores\nfor true + N negatives', 'mod', fs=11, icon='fire')
box(cx - 120, PY + 340, 240, 34, 'Softmax over N+1', 'op', fs=11.5)
box(cx - 120, PY + 394, 240, 40, 'Cross-Entropy\n(true sentence must win)', 'sup', fs=11)
for a, b in [(74, 96), (136, 158), (198, 222), (252, 280), (320, 340), (374, 394)]:
    arrow([(cx, PY + a), (cx, PY + b)], dashed=(b == 394))

# (B) Disputed-span verifier
x0 = px[1]
panel(x0, PY, PW, PH, '(B) Disputed-Span Verifier')
cx = x0 + PW / 2
label(x0 + 100, PY + 42, 'candidate phonemes (Q)', fs=10)
tokens(x0 + 27, PY + 54, 7, PHON_TOKEN, disputed=(4, 5))
label(x0 + 255, PY + 42, 'EMG states (K, V)', fs=10)
tokens(x0 + 203, PY + 54, 5, EMG_TOKEN)
box(x0 + 12, PY + 98, 186, 40, '+ Disputed Embedding\n(differs across candidates)', 'new', fs=10, icon='fire')
arrow([(x0 + 100, PY + 71), (x0 + 100, PY + 98)])
box(x0 + 20, PY + 162, 283, 46, 'Cross-Attention Decoder ×2\nphonemes attend to EMG frames', 'mod', fs=11,
    icon='fire')
arrow([(x0 + 100, PY + 138), (x0 + 100, PY + 162)])
arrow([(x0 + 255, PY + 71), (x0 + 255, PY + 162)])
box(x0 + 60, PY + 232, 203, 40, 'Pool over Disputed\nTokens', 'new', fs=11)
arrow([(cx, PY + 208), (cx, PY + 232)])
box(x0 + 60, PY + 296, 203, 36, 'Linear → Match Score', 'mod', fs=11.5, icon='fire')
arrow([(cx, PY + 272), (cx, PY + 296)])
box(x0 + 90, PY + 356, 143, 32, '$s_{ver}$ = 2.7', 'lit', fs=11.5)
arrow([(cx, PY + 332), (cx, PY + 356)])
label(cx, PY + 410, 'Only the words in dispute decide the score:', fs=10, italic=True)
label(cx, PY + 428, 'EMG checks exactly what the camera cannot see.', fs=10, italic=True)

# (C) Viseme lattice expansion
x0 = px[2]
panel(x0, PY, PW, PH, '(C) Look-Alike Expansion')
cx = x0 + PW / 2
box(cx - 125, PY + 38, 250, 34, 'a draft in the beam: “grab the bat”', 'lit', fs=11)
box(cx - 125, PY + 92, 250, 34, 'Phonemes → Visemes', 'op', fs=11.5)
arrow([(cx, PY + 72), (cx, PY + 92)])
cols = [(x0 + 40, ['grab', 'crab', 'grape']), (x0 + 145, ['the', 'a']), (x0 + 235, ['bat', 'mat', 'pat', 'bad'])]
for i, (xx, words) in enumerate(cols):
    for j, wd in enumerate(words):
        yy = PY + 148 + j * 27
        box(xx, yy, 62, 22, wd, 'lit', fs=10)
        if i < 2:
            for k in range(len(cols[i + 1][1])):
                ax.plot([xx + 62, cols[i + 1][0]], [yy + 11, PY + 159 + k * 27], color='#888', lw=0.7,
                        zorder=2)
arrow([(cx, PY + 126), (cx, PY + 146)])
label(cx, PY + 262, 'every word sequence that looks the same', fs=10, italic=True)
box(cx - 125, PY + 280, 250, 40, 'LM$^{(H, I)}$ + Claude hints prune', 'mod', fs=11.5)
arrow([(cx, PY + 256 + 12), (cx, PY + 280)])
box(cx - 125, PY + 340, 250, 40, 'EMG Verifier picks$^{(B)}$', 'new', fs=11.5, icon='fire')
arrow([(cx, PY + 320), (cx, PY + 340)])
box(cx - 90, PY + 398, 180, 34, '“grab the mat”', 'lit', fs=11.5)
arrow([(cx, PY + 380), (cx, PY + 398)])

# (D) Reliability gate
x0 = px[3]
panel(x0, PY, PW, PH, '(D) Reliability Gate')
cx = x0 + PW / 2
for i, s in enumerate(['camera\nconfidence', 'face\nvisible', 'EMG\nquality', 'speaking\nrate']):
    box(x0 + 14 + i * 75, PY + 38, 70, 40, s, 'lit', fs=9.5)
    arrow([(x0 + 49 + i * 75, PY + 78), (x0 + 49 + i * 75, PY + 98)])
box(x0 + 14, PY + 98, 295, 36, 'MLP', 'mod', fs=12, icon='fire')
box(cx - 90, PY + 156, 180, 34, 'Softmax', 'op', fs=11.5)
arrow([(cx, PY + 134), (cx, PY + 156)])
box(cx - 110, PY + 210, 220, 34, '$g_{cam}$, $g_{emg}$, $g_{lm}$', 'lit', fs=12)
arrow([(cx, PY + 190), (cx, PY + 210)])
# inset: weights vs mouth opening
gx, gy, gw, gh = x0 + 50, PY + 270, 240, 130
ax.add_patch(Rectangle((gx, gy), gw, gh, facecolor='white', edgecolor=INK, lw=1.2, zorder=3))
m = np.linspace(0, 1, 100)
g_cam = 0.05 + 0.6 / (1 + np.exp(-12 * (m - 0.35)))
g_emg = 0.9 - 0.55 / (1 + np.exp(-12 * (m - 0.35)))
ax.plot(gx + m * gw, gy + gh - g_cam * gh, color='#6fa3d8', lw=2.2, zorder=4)
ax.plot(gx + m * gw, gy + gh - g_emg * gh, color='#f08c84', lw=2.2, zorder=4)
label(gx + gw - 8, gy + gh - 0.68 * gh, '$g_{cam}$', fs=10.5, ha='right', color='#3f7fc0')
label(gx + gw - 8, gy + gh - 0.30 * gh, '$g_{emg}$', fs=10.5, ha='right', color='#c0504a')
label(gx + gw / 2, gy + gh + 14, 'face visibility →', fs=10)
label(gx - 14, gy + gh / 2, 'weight', fs=10, rot=90)
label(gx + 6, gy + 12, 'hidden', fs=9, ha='left', italic=True)
label(gx + gw - 6, gy + 12, 'clear', fs=9, ha='right', italic=True)

# (E) Fast-speech robustness
x0 = px[4]
panel(x0, PY, PW, PH, '(E) Fast-Speech Robustness')
cx = x0 + PW / 2
emg_image(x0 + 20, PY + 38, 120, 56, n=3)
label(x0 + 80, PY + 104, 'normal pace', fs=10)
emg_image(x0 + 183, PY + 38, 120, 56, n=3, bursts=[0.15, 0.3, 0.45], amp=0.6, overlap=0.5)
label(x0 + 243, PY + 104, 'fast: smaller, overlapping', fs=10)
arrow([(x0 + 140, PY + 66), (x0 + 183, PY + 66)])
box(x0 + 20, PY + 120, 283, 36, 'Speed Perturbation 0.8–1.6×', 'op', fs=11.5)
box(x0 + 20, PY + 176, 283, 46, 'Articulatory Undershoot\npeak compression + burst overlap', 'new', fs=11)
arrow([(cx, PY + 156), (cx, PY + 176)])
box(x0 + 20, PY + 248, 283, 36, 'Envelope → Rate Estimate', 'op', fs=11.5)
arrow([(cx, PY + 222), (cx, PY + 248)])
box(x0 + 20, PY + 308, 283, 46, 'FiLM: $\\gamma(\\mathrm{rate})\\, h + \\beta(\\mathrm{rate})$\non each block\'s normalised input',
    'new', fs=11, icon='fire')
arrow([(cx, PY + 284), (cx, PY + 308)])
box(x0 + 60, PY + 378, 203, 36, 'Transformer ×12', 'mod', fs=11.5, icon='fire')
arrow([(cx, PY + 354), (cx, PY + 378)])

# (F) Partial-sentence training
x0 = px[5]
panel(x0, PY, PW, PH, '(F) Partial-Sentence Training')
cx = x0 + PW / 2
emg_image(x0 + 20, PY + 38, 283, 60, n=3, bursts=[0.12, 0.32, 0.55, 0.78])
cut = x0 + 20 + 0.62 * 283
ax.plot([cut, cut], [PY + 32, PY + 104], color=DISPUTED, lw=2.4, zorder=6)
label(cut - 70, PY + 112, 'kept: frames before t', fs=9.5, italic=True)
label(cut + 48, PY + 112, 'cut', fs=9.5, italic=True)
box(x0 + 20, PY + 128, 283, 44, 'Forced alignment (EMA model)\nevery 10 epochs', 'op', fs=10.5)
arrow([(cx, PY + 118), (cx, PY + 128)])
box(x0 + 20, PY + 192, 283, 44, 'Cut 30% of rows at a random\nframe t (30–100% of length)', 'op', fs=10.5)
arrow([(cx, PY + 172), (cx, PY + 192)])
box(x0 + 20, PY + 256, 283, 44, 'Target = letters aligned before t\n“i want wa…”', 'lit', fs=10.5)
arrow([(cx, PY + 236), (cx, PY + 256)])
box(x0 + 20, PY + 320, 283, 44, 'Letter CTC + speaking loss\non the cut', 'sup', fs=10.5)
arrow([(cx, PY + 300), (cx, PY + 320)], dashed=True)
label(cx, PY + 390, 'The live loop reads beginnings of sentences', fs=10, italic=True)
label(cx, PY + 408, 'every 100 ms, so SVR-1 trains on them.', fs=10, italic=True)

# (G) Real-time loop: a full-width strip under the panels
GY, GH = 1005, 150
panel(5, GY, 1988, GH, '(G) Real-Time Loop: text and voice while you mouth')
row1 = [('Claude predicts likely words\nbefore you mouth', 'mod'), ('Every 100 ms while mouthing:\nre-read clip, camera + EMG', 'mod'),
        ('Joint search: camera + EMG\n+ LM$^{(H, I)}$ + Claude hints', 'new'), ('Confidence lock: one learned\nscore, threshold moves with context', 'new'),
        ('Locked words: grey \u2192 black\n+ Cartesia voice into the call', 'lit')]
row2 = [('Endpoint$^{(G)}$: ~0.5 s of\nEMG quiet + lips still', 'new'), ('Final pass: camera beam\n+ look-alikes$^{(C)}$ + verifier$^{(B)}$ + Claude', 'new'),
        ('Corrected text\nreplaces live text', 'lit'), ('Still unsure: \u201cPlease repeat\u201d\n+ reasons in the error log', 'new')]
bw, gap = 290, 44
for i, (t, k) in enumerate(row1):
    x = 22 + i * (bw + gap)
    box(x, GY + 34, bw, 46, t, k, fs=10.5)
    if i:
        arrow([(x - gap, GY + 57), (x, GY + 57)])
for i, (t, k) in enumerate(row2):
    x = 22 + i * (bw + gap)
    box(x, GY + 96, bw, 46, t, k, fs=10.5)
    if i:
        arrow([(x - gap, GY + 119), (x, GY + 119)])

# (H) Conversational LM and (I) call cache: a row of two panels under the real-time loop
HY, HH, HW = 1165, 265, 989
x0 = 5
panel(x0, HY, HW, HH, '(H) Conversational Language Model')
steps = [('72M words of dialogue\nsubtitles · Cornell films\nDailyDialog · Topical-Chat', 'lit'),
         ("Normalised like SVR-1\nrecording prompts removed", 'op'),
         ('Pruned word 4-gram (KenLM)\n110 MB, loads in 0.4 s', 'new'),
         ('Every hypothesis, every pass\n~2 µs per word', 'lit')]
for i, (t, k) in enumerate(steps):
    xx = x0 + 20 + i * 240
    box(xx, HY + 40, 210, 66, t, k, fs=10.5)
    if i:
        arrow([(xx - 30, HY + 73), (xx, HY + 73)])
label(x0 + HW / 2, HY + 132, 'Perplexity on our 600 recording prompts (lower is better: the LM expects how people talk)',
      fs=10.5, italic=True)
for i, (name, val, col) in enumerate([('book LM (Gaddy & Klein)', 181.2, '#b8b6ae'), ('conversational LM (ours)', 62.1, '#6fa3d8')]):
    yy = HY + 152 + i * 34
    label(x0 + 300, yy + 12, name, fs=10.5, ha='right')
    ax.add_patch(Rectangle((x0 + 315, yy), val * 2.6, 24, facecolor=col, edgecolor=INK, lw=1.2, zorder=3))
    label(x0 + 325 + val * 2.6, yy + 12, f'{val:.0f}', fs=10.5, ha='left')
label(x0 + HW / 2, HY + 240, 'size 946 MB → 110 MB · load 5.2 s → 0.4 s · words never seen 8 → 3', fs=10, italic=True)

x0 = 5 + HW + 10
panel(x0, HY, HW, HH, '(I) Call Cache: the conversation so far')
box(x0 + 20, HY + 40, 170, 48, "Judge's lines\n(Cartesia Ink)", 'lit', fs=10.5)
box(x0 + 20, HY + 100, 170, 48, "Austin's locked\nwords", 'lit', fs=10.5)
box(x0 + 235, HY + 64, 250, 60, 'Call cache: bigram + unigram\ncounts, fading (half-life 400 words)', 'new', fs=10.5)
arrow([(x0 + 190, HY + 64), (x0 + 212, HY + 64), (x0 + 212, HY + 88), (x0 + 235, HY + 88)])
arrow([(x0 + 190, HY + 124), (x0 + 212, HY + 124), (x0 + 212, HY + 100), (x0 + 235, HY + 100)])
box(x0 + 525, HY + 64, 270, 60, '$P = (1-\\lambda)\\,P_{ngram} + \\lambda\\,P_{cache}$\n$\\lambda$: 0 → 0.15 as the call grows', 'op', fs=10.5)
arrow([(x0 + 485, HY + 94), (x0 + 525, HY + 94)])
box(x0 + 835, HY + 69, 135, 50, 'to the joint\nsearch ($g_{lm}$)', 'lit', fs=10.5)
arrow([(x0 + 795, HY + 94), (x0 + 835, HY + 94)])
box(x0 + 20, HY + 176, 330, 40, 'judge: “hi, this is Priya from admissions”', 'lit', fs=10.5)
box(x0 + 385, HY + 176, 400, 40, '“i talked to priya …”: ln P −33.7 → −25.3 (×4000)', 'lit', fs=10.5)
arrow([(x0 + 350, HY + 196), (x0 + 385, HY + 196)])
label(x0 + HW / 2, HY + 240, 'A name just mentioned becomes likely at once; its look-alike “pria” does not. Claude stays separate: hints.',
      fs=10, italic=True)

out = os.path.join(HERE, 'svr1_architecture')
fig.savefig(out + '.pdf', bbox_inches='tight', pad_inches=0.05)
fig.savefig(out + '.png', dpi=110, bbox_inches='tight', pad_inches=0.05)
print('wrote', out + '.pdf', out + '.png')
