"""SVR-1 against Gaddy and Klein's model on our 4 electrode sites (Berkeley test set), from the benchmark summaries.

    venv-svr/bin/python figures/make_results_figure.py   ->  figures/svr1_vs_gaddy_4ch.{pdf,png}

Reads $RECLAIM_WORKDIR/benchmark_results/<model>_gpu/summary.json (bench.score). Every model is tested on sites 1, 2,
3, 6 (input slots 0, 1, 2, 5); the Gaddy-architecture models were trained on all 8 sites, SVR-1 on these 4.
Errors are greedy decoding without a language model (the model's own output). Palette: the first four categorical
slots of the data-viz reference palette, validated (CVD and normal-vision separation) in the order the bars sit;
every bar carries its value, since two of the hues are below 3:1 contrast on the surface.
"""
import json
import os

import matplotlib
from matplotlib import font_manager as fm
from matplotlib import pyplot as plt

matplotlib.use('Agg')
HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ.get('RECLAIM_WORKDIR', os.path.join(HERE, '..'))
RESULTS = os.path.join(WORK, 'benchmark_results')
for f in os.listdir(os.path.join(HERE, 'fonts')):
    if f.endswith('.ttf'):
        fm.fontManager.addfont(os.path.join(HERE, 'fonts', f))
plt.rcParams.update({'font.family': 'Liberation Sans', 'pdf.fonttype': 42})

SURFACE, INK, INK_2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e2dc'
MODELS = [  # (summary folder, legend label, colour): baselines first, ours last
    ('B1_ours_gpu', "Gaddy & Klein's released model\n(trained on 8 sites)", '#eda100'),
    ('B1T_ours_gpu', 'Gaddy architecture, our retrain\n(trained on 8 sites)', '#1baf7a'),
    ('SVR1_4ch_ours_gpu', 'SVR-1 (ours)\n(trained on these 4 sites)', '#2a78d6'),
    ('SVR1_4ch_artic_ours_gpu', 'SVR-1 + articulator targets (ours)\n(trained on these 4 sites)', '#eb6834'),
]
ERRORS = [('open_vocab', 'greedy', 'wer', 'Words wrong\n(WER)'), ('open_vocab', 'greedy', 'cer', 'Letters wrong\n(CER)')]
CHOICES = [('phrase_id_openvocab', 'Right sentence\namong 10', 10), ('homophene_10', 'Among 10 lip\nlook-alikes', 10),
           ('homophene_50', 'Among 50 lip\nlook-alikes', 2), ('nearduplicate_10', 'Among 10 near-\nduplicates', 10)]


def load():
    out = {}
    for folder, *_ in MODELS:
        with open(os.path.join(RESULTS, folder, 'summary.json')) as f:
            b = json.load(f)['benchmarks']
        m = {f'{k}.{e}': 100 * b[f'{k}__4ch']['metrics'][d][e] for k, d, e, _ in ERRORS}
        m.update({k: 100 * b[f'{k}__4ch']['metrics']['top1_accuracy'] for k, _, _ in CHOICES})
        out[folder] = m
    return out


def bars(ax, groups, data, keys):
    n, width = len(MODELS), 0.19
    for j, (folder, label, colour) in enumerate(MODELS):
        xs = [i + (j - (n - 1) / 2) * width for i in range(len(groups))]
        vals = [data[folder][k] for k in keys]
        ax.bar(xs, vals, width * 0.92, color=colour, edgecolor=SURFACE, linewidth=1.5, label=label, zorder=3)
        for x, v in zip(xs, vals):
            ax.text(x, v + 1.2, f'{v:.0f}', ha='center', va='bottom', fontsize=9, color=INK_2, zorder=4)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, fontsize=10.5, color=INK)
    ax.set_ylim(0, 108)
    ax.set_yticks(range(0, 101, 20))
    ax.set_yticklabels([f'{t}%' for t in range(0, 101, 20)], fontsize=9.5, color=INK_2)
    ax.grid(axis='y', color=GRID, linewidth=0.8, zorder=0)
    ax.set_facecolor(SURFACE)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color(INK_2)
    ax.tick_params(length=0)


def main():
    data = load()
    fig, (a, b) = plt.subplots(1, 2, figsize=(13.5, 5.6), gridspec_kw={'width_ratios': [1, 2.1], 'wspace': 0.12})
    fig.patch.set_facecolor(SURFACE)
    bars(a, [g for *_, g in ERRORS], data, [f'{k}.{e}' for k, _, e, _ in ERRORS])
    a.set_title('Transcription errors  (lower is better)', fontsize=12, color=INK, loc='left', pad=10)
    bars(b, [g for _, g, _ in CHOICES], data, [k for k, *_ in CHOICES])
    b.set_title('Picking the true sentence  (higher is better)', fontsize=12, color=INK, loc='left', pad=10)
    for i, (_, _, chance) in enumerate(CHOICES):
        b.plot([i - 0.42, i + 0.42], [chance, chance], color=INK, linewidth=1, linestyle=(0, (3, 2)), zorder=5)
    b.text(len(CHOICES) - 0.58, 10 + 1.5, 'chance', fontsize=8.5, color=INK, ha='right', va='bottom', zorder=5)
    fig.suptitle('SVR-1 vs Gaddy & Klein on our 4 electrode sites', x=0.06, ha='left', y=1.06, fontsize=15,
                 fontweight='bold', color=INK)
    fig.text(0.06, 0.975, 'Berkeley silent-speech test set (99 sentences), sites 1, 2, 3, 6. Errors: greedy decoding, '
             'no language model. Chance is 10% (2% to 5% among 50 look-alikes).', fontsize=10, color=INK_2, ha='left')
    handles, labels = a.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, -0.13),
               labelcolor=INK, handlelength=1.2, columnspacing=2.2)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(HERE, f'svr1_vs_gaddy_4ch.{ext}'), dpi=200, bbox_inches='tight',
                    facecolor=SURFACE)
    print('| Model | ' + ' | '.join(g.replace('\n', ' ') for *_, g in ERRORS) + ' | '
          + ' | '.join(g.replace('\n', ' ') for _, g, _ in CHOICES) + ' |')
    print('|' + ' --- |' * (1 + len(ERRORS) + len(CHOICES)))
    for folder, label, _ in MODELS:
        vals = [data[folder][f'{k}.{e}'] for k, _, e, _ in ERRORS] + [data[folder][k] for k, *_ in CHOICES]
        print(f"| {label.splitlines()[0]} | " + ' | '.join(f'{v:.1f}%' for v in vals) + ' |')


if __name__ == '__main__':
    main()
