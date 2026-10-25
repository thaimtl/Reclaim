"""SVR-1 against Gaddy and Klein's model on our 4 electrode sites (Berkeley test set), from the benchmark summaries.

    venv-svr/bin/python figures/make_svr1_vs_gaddy.py    ->  figures/svr1_vs_gaddy_4ch.{pdf,png} (bar chart)
                                                             figures/svr1_vs_gaddy_4ch_table.{pdf,png} (every result)

Reads $RECLAIM_WORKDIR/benchmark_results/<model>_gpu/summary.json (bench.score). Every model is tested on sites 1, 2,
3, 6 (input slots 0, 1, 2, 5); the Gaddy-architecture models were trained on all 8 sites, SVR-1 on these 4.
The bar chart's errors are greedy decoding without a language model (the model's own output); the table adds the
errors with Gaddy & Klein's language model (their 5-gram and beam-search weights, tuned for their model, not refit).
The table is also printed as Markdown, for docs/SVR1_SPEC.md section 7.
"""
import json
import os

import matplotlib

matplotlib.use('Agg')
from matplotlib import font_manager as fm  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402

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


TABLE = [  # (key, header group, header, lower is better)
    ('open_vocab.greedy.wer', 'Words wrong (WER)', 'no LM', True), ('open_vocab.beam_lm.wer', 'Words wrong (WER)', 'with LM', True),
    ('open_vocab.greedy.cer', 'Letters wrong (CER)', 'no LM', True), ('open_vocab.beam_lm.cer', 'Letters wrong (CER)', 'with LM', True),
    ('phrase_id_openvocab', 'True sentence picked among', '10 sentences', False),
    ('phrase_id_closedvocab', 'True sentence picked among', '10 (closed set)', False),
    ('homophene_10', 'True sentence picked among', '10 look-alikes', False),
    ('homophene_50', 'True sentence picked among', '50 look-alikes', False),
    ('nearduplicate_10', 'True sentence picked among', '10 near-dups', False),
]


def load():
    out = {}
    for folder, *_ in MODELS:
        with open(os.path.join(RESULTS, folder, 'summary.json')) as f:
            b = json.load(f)['benchmarks']
        m = {k: 100 * b[f'{k}__4ch']['metrics']['top1_accuracy'] for k, _, _ in CHOICES}
        m['phrase_id_closedvocab'] = 100 * b['phrase_id_closedvocab__4ch']['metrics']['top1_accuracy']
        for d in ('greedy', 'beam_lm'):
            for e in ('wer', 'cer'):
                m[f'open_vocab.{d}.{e}'] = 100 * b['open_vocab__4ch']['metrics'][d][e]
        out[folder] = m
    return out


def table(data):
    """Every result in one table: rows = models, columns = TABLE; the best value of each column in bold."""
    names = [label.split('\n')[0] for _, label, _ in MODELS]
    sites = ['8', '8', '4', '4']
    best = {k: (min if low else max)(data[f][k] for f, *_ in MODELS) for k, _, _, low in TABLE}
    fig = plt.figure(figsize=(15.5, 4.6))
    fig.patch.set_facecolor(SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 155)
    ax.set_ylim(46, 0)
    ax.axis('off')
    x_name, x_sites, x0, cw = 1.5, 37.5, 44.5, 12.2
    xs = [x0 + i * cw + (1.6 if i >= 2 else 0) + (3.2 if i >= 4 else 0) for i in range(len(TABLE))]
    ax.text(x_name, 2.5, 'SVR-1 vs Gaddy & Klein on our 4 electrode sites: every result', fontsize=14.5,
            fontweight='bold', color=INK, va='center')
    ax.text(x_name, 6.3, 'Berkeley silent-speech test set (99 sentences), tested on sites 1, 2, 3, 6. Lower is better '
            'for errors, higher for picking the true sentence (chance 10%; 2% to 5% among 50). Best in bold.',
            fontsize=10, color=INK_2, va='center')
    groups = {}
    for x, (_, g, _, _) in zip(xs, TABLE):
        groups.setdefault(g, []).append(x)
    for g, gx in groups.items():
        ax.text((gx[0] + gx[-1]) / 2, 11.5, g, ha='center', va='center', fontsize=10.5, color=INK, fontweight='bold')
        ax.plot([gx[0] - cw / 2 + 0.8, gx[-1] + cw / 2 - 0.8], [13.2, 13.2], color=INK_2, lw=0.8)
    ax.text(x_name, 15.8, 'Model', fontsize=10.5, color=INK, fontweight='bold', va='center')
    ax.text(x_sites, 15.8, 'Trained\non sites', fontsize=10, color=INK, fontweight='bold', va='center', ha='center',
            linespacing=1.0)
    for x, (_, _, h, _) in zip(xs, TABLE):
        ax.text(x, 15.8, h, ha='center', va='center', fontsize=10, color=INK)
    ax.plot([0.8, 154.2], [18.6, 18.6], color=INK, lw=1.2)
    for r, ((folder, _, colour), name, st) in enumerate(zip(MODELS, names, sites)):
        y = 22.3 + r * 5.6
        if r % 2 == 1:
            ax.add_patch(plt.Rectangle((0.8, y - 2.8), 153.4, 5.6, color='#f1f0ec', zorder=0, lw=0))
        ax.add_patch(plt.Rectangle((x_name, y - 1.1), 1.8, 2.2, color=colour, zorder=2, lw=0))
        ax.text(x_name + 2.8, y, name, fontsize=10.5, color=INK, va='center',
                fontweight='bold' if 'ours' in name else 'normal')
        ax.text(x_sites, y, st, fontsize=10.5, color=INK, va='center', ha='center')
        for x, (k, _, _, _) in zip(xs, TABLE):
            v = data[folder][k]
            ax.text(x, y, f'{v:.1f}%', ha='center', va='center', fontsize=10.5, color=INK,
                    fontweight='bold' if abs(v - best[k]) < 1e-9 else 'normal')
    ax.plot([0.8, 154.2], [22.3 + 3.5 * 5.6, 22.3 + 3.5 * 5.6], color=INK, lw=1.2)
    ax.text(x_name, 44.0, 'LM = Gaddy & Klein\'s 5-gram language model with their beam-search weights, tuned for their '
            'model and not refit for SVR-1 (it drops words from SVR-1\'s output, hence "with LM" can be worse than '
            '"no LM"). The Gaddy-architecture models were trained on all 8 sites; SVR-1 on these 4.',
            fontsize=9, color=INK_2, va='center')
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(HERE, f'svr1_vs_gaddy_4ch_table.{ext}'), dpi=200, facecolor=SURFACE,
                    bbox_inches='tight')
    print('| Model | Trained on sites | ' + ' | '.join(f'{g.split(" (")[0]} {h}' for _, g, h, _ in TABLE) + ' |')
    print('|' + ' --- |' * (2 + len(TABLE)))
    for (folder, *_), name, st in zip(MODELS, names, sites):
        print(f'| {name} | {st} | ' + ' | '.join(
            (f'**{data[folder][k]:.1f}%**' if abs(data[folder][k] - best[k]) < 1e-9 else f'{data[folder][k]:.1f}%')
            for k, *_ in TABLE) + ' |')


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
    bars(a, [g for *_, g in ERRORS], data, [f'{k}.{d}.{e}' for k, d, e, _ in ERRORS])
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
    plt.close(fig)
    table(data)


if __name__ == '__main__':
    main()
