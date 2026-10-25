"""Charts for sponsors/cartesia.md and sponsors/anthropic.md, light and dark: figures/sponsors/*.png.

    python figures/make_sponsor_charts.py

Every number comes from a measurement written down in the repo: call/README.md and the call commits for Cartesia,
the claude_service commits for Claude, and rig/filters.py for the hum filter.
"""
import importlib.util
import os

import matplotlib
import numpy as np
from matplotlib import font_manager as fm
from matplotlib import pyplot as plt
from scipy import signal

matplotlib.use('Agg')
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'sponsors')
for f in os.listdir(os.path.join(HERE, 'fonts')):
    if f.endswith('.ttf'):
        fm.fontManager.addfont(os.path.join(HERE, 'fonts', f))
plt.rcParams.update({'font.family': 'Liberation Sans', 'font.size': 12})

GAP = 1.5
CARTESIA, CLAUDE, CAMERA = '#309D4B', '#D97757', '#2f6fde'
THEMES = {
    'light': dict(bg='#ffffff', ink='#1f2328', sub='#59636e', grid='#e6e9ed', muted='#b6bec7'),
    'dark': dict(bg='#0d1117', ink='#f0f6fc', sub='#9198a1', grid='#262c34', muted='#4b535d'),
}
spec = importlib.util.spec_from_file_location('rig_filters', os.path.join(HERE, '..', 'rig', 'filters.py'))
filters = importlib.util.module_from_spec(spec)
spec.loader.exec_module(filters)


def frame(t, title, note, size=(10, 4.2)):
    fig, ax = plt.subplots(figsize=size, dpi=200)
    fig.patch.set_facecolor(t['bg'])
    ax.set_facecolor(t['bg'])
    fig.text(0.02, 0.95, title, fontsize=17, fontweight='bold', color=t['ink'], va='top')
    fig.text(0.02, 0.87, note, fontsize=12, color=t['sub'], va='top')
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(t['grid'])
    ax.tick_params(colors=t['sub'], length=0)
    ax.grid(axis='x', color=t['grid'], linewidth=1)
    ax.set_axisbelow(True)
    return fig, ax


def num(x, digits):
    return f'{x:.{digits}f}'.replace('-', '\u2212')


def ranges(ax, t, rows, unit, label_x=0, pad=0.12, digits=1):
    """Horizontal range bars: (label, low, high, color, note). A single value has low == high."""
    for i, (label, lo, hi, color, note) in enumerate(rows):
        y = (len(rows) - 1 - i) * GAP
        if hi > lo:
            ax.barh(y, hi - lo, left=lo, height=0.42, color=color)
        else:
            ax.barh(y, hi, height=0.42, color=color)
        ax.text(hi + pad, y, (f'{num(lo, digits)} to {num(hi, digits)} {unit}' if hi > lo else f'{num(hi, digits)} {unit}'), va='center',
                fontsize=12, fontweight='bold', color=t['ink'])
        ax.text(lo if label_x is None else label_x, y + 0.36, label, fontsize=12, color=t['ink'], va='bottom')
        if note:
            ax.text(hi + pad, y - 0.3, note, va='center', fontsize=10.5, color=t['sub'])
    ax.set_yticks([])
    ax.set_ylim(-0.8, (len(rows) - 1) * GAP + 0.8)


def cartesia_wait(t):
    fig, ax = frame(t, 'How long the judge waits in silence',
                    'One 15-word sentence that Austin mouthed in 5.5 s, measured on Cartesia Sonic 3.6')
    ranges(ax, t, [
        ('Silent Voice, natural phrases over the Sonic WebSocket', 0.6, 0.6, CARTESIA,
         'until the first sound, and done 1.0 s after Austin'),
        ('Sonic with its default 120 ms buffer', 4.9, 4.9, t['muted'], '14 of 15 words held back to the end'),
        ('Speak once the sentence is over', 6.5, 7.5, t['muted'], 'estimate, his 5.5 s plus 1 to 2 s of decoding'),
    ], 's')
    ax.set_xlim(0, 11)
    ax.set_xlabel('seconds', color=t['sub'])
    fig.subplots_adjust(left=0.03, right=0.97, top=0.76, bottom=0.14)
    return fig


def cartesia_gate(t):
    fig, ax = frame(t, 'The call hears the judge, not the room',
                    'Audio levels on test calls at the expo. The gate only passes sound near the caller’s own level.')
    ax.axvspan(-26, -6, color=CARTESIA, alpha=0.1, linewidth=0)
    ax.text(-16, 2.2, 'gate passes', ha='center', fontsize=11, color=t['sub'])
    ranges(ax, t, [('The judge, phone at the ear', -20, -12, CARTESIA, None),
                   ('People talking nearby', -60, -40, t['muted'], None)], 'dBFS', label_x=None, pad=1, digits=0)
    ax.set_xlim(-70, 0)
    ax.set_ylim(-0.8, 2.45)
    ax.set_xlabel('dBFS, louder to the right', color=t['sub'])
    fig.subplots_adjust(left=0.03, right=0.97, top=0.76, bottom=0.14)
    return fig


def claude_latency(t):
    fig, ax = frame(t, 'Claude’s reply time on live calls',
                    'Haiku 5.5 with thinking off answers in about 2 to 3 s, so a prediction is ready before Austin '
                    'starts mouthing', size=(10, 4.8))
    ranges(ax, t, [
        ('Opus 5.5, low effort', 5.7, 6.9, t['muted'], 'can’t turn thinking off'),
        ('Haiku 5.5, prediction while the judge is still talking', 1.9, 3.9, CLAUDE,
         'two in flight, so a fresh one lands every ~0.8 s'),
        ('Haiku 5.5, prediction after the judge’s line', 1.9, 2.3, CLAUDE, None),
        ('Haiku 5.5, look-alike repair', 2.0, 2.3, CLAUDE, None),
    ], 's')
    ax.set_xlim(0, 10)
    ax.set_xlabel('seconds per call', color=t['sub'])
    fig.subplots_adjust(left=0.03, right=0.97, top=0.78, bottom=0.12)
    return fig


def hum_filter(t):
    fig, ax = frame(t, 'The hum filter on every EMG channel',
                    'Causal, so it runs live sample by sample, and training uses the exact same filter')
    f, h = signal.sosfreqz(filters.design(), worN=8192, fs=filters.FS)
    db = 20 * np.log10(np.maximum(np.abs(h), 1e-6))
    keep = f <= 300
    ax.plot(f[keep], db[keep], color=CAMERA, linewidth=2)
    ax.axvspan(0, 20, color=t['grid'], linewidth=0)
    ax.text(10, 3, 'drift', ha='center', fontsize=10.5, color=t['sub'])
    for hz, name in ((60, 'hum'), (120, 'echo'), (180, 'echo')):
        ax.text(hz + 4, -56, f'{hz} Hz {name}', fontsize=10.5, color=t['sub'])
    ax.set_xlim(0, 300)
    ax.set_ylim(-60, 8)
    ax.grid(axis='y', color=t['grid'], linewidth=1)
    ax.set_xlabel('frequency (Hz)', color=t['sub'])
    ax.set_ylabel('gain (dB)', color=t['sub'])
    ax.spines['left'].set_visible(False)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.76, bottom=0.14)
    return fig


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    for name, draw in [('cartesia_wait', cartesia_wait), ('cartesia_gate', cartesia_gate),
                       ('claude_latency', claude_latency), ('hum_filter', hum_filter)]:
        for theme, t in THEMES.items():
            fig = draw(t)
            path = os.path.join(OUT, f'{name}-{theme}.png')
            fig.savefig(path, facecolor=t['bg'])
            plt.close(fig)
            print('wrote', os.path.relpath(path))
