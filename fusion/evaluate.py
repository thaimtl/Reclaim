"""Offline scores for a recorded session: WER, WER before vs after Claude's repairs, how often the system says
"Please repeat", the delay from mouthing a word to its lock, and from the lock to the voice.

Inputs: a buslog session (buslog.read) and the recorder's manifest.jsonl (one line per clip: id, prompt, set,
clip_t0 / clip_t1 on the recorder's bus.now() (time.perf_counter())). Two modes:

    logged (default)  score the dec.* and voice.say messages the live decoder published during the session.
    --decode          run a fresh Decoder over the session's cam.* and claude.* messages, with now = each message's
                      t_pub and pump() every PUMP_STEP_S between them, and score its outputs. No bus and no real
                      time, so the lock threshold and weights can be tuned on sessions 1 and 2:
                      --set lock.THRESHOLD=0.68 --set decoder.NBEST_W=0.6. Every logged tick is read (live, the
                      decoder may skip stale ones), and a logged claude.repair answered the live decoder's
                      locks, so one whose 'from' no longer matches is ignored.

Clocks. Live, frame_t (capture times) and t_pub are both the laptop's bus.now(), and so are the
recorder's clip_t0 / clip_t1. When the camera replays a recorded clip, frame_t is the recording's video_ts: the
recorder's clock, the same as clip_t0 / clip_t1, while t_pub is the replay's live clock. So clips are matched on
frame time, and only delays need t_pub (below).

Scoring. An utterance belongs to the clip whose [clip_t0, clip_t1] contains its start: dec.hyp's t0 (frame time
of its first frame), else its first message's t_pub; the others are listed as unmatched. A clip's text is its
utterances' locked words in order, rebuilt from their dec.hyp messages (each carries the newest locked words from
`offset` on): `locked` (after Claude's repairs, late ones included) and `raw` (as they locked). An utterance
whose dec.hyp skips words (the logger started mid-utterance) is left out with a warning. WER is the word edit
distance over the prompt's word count, lowercase with punctuation dropped (jiwer is not in venv-live); overall
WER sums edits and words over clips, so a prompted clip with no utterance counts in full and is also listed
in missed_clips. Rest clips (set REST_SET) count their locked words as false_words instead.

Delays. end_of_word_to_lock = dec.lock t_pub - the word's t (frame time of its last frame: a word cannot lock
before it is finished); lock_to_voice = the k-th word's voice.say t_pub - its dec.lock t_pub, per utterance.
A word whose t_pub - t is not within CLOCK_OK_S comes from a replay and is offset-corrected: the segment whose
frames contain t gives offset = t_pub - frame_t[-1] of its first cam.tick, and the delay becomes
t_pub - (t + offset). That measures from the camera's first read of the word instead of from capture, leaving
out the camera's own lag; the report counts these words separately.

    python -m fusion.evaluate SESSION --manifest PATH [--json OUT] [--decode [--set lock.THRESHOLD=0.7]]
"""
import argparse
import json
import re
import sys

import numpy as np

import buslog
from fusion import decoder as decoder_mod
from fusion import lock as lock_mod
from fusion.decoder import Decoder
from fusion.run import TOPICS as IN_TOPICS
from fusion.run import handle
from fusion.words import SPM, UNITS, load_lookalikes, load_units

OUT_TOPICS = ('dec.hyp', 'dec.lock', 'dec.repeat', 'voice.say')
CLOCK_OK_S = 5.0                                     # |t_pub - frame time| above this: different clocks
PUMP_STEP_S = 0.02                                   # the live loop pumps about every 10 to 20 ms
REST_SET = 'B'                                       # recorder sets: C training, E test, B rest (no prompt)
TUNABLE = {'lock': lock_mod, 'decoder': decoder_mod}


def settle_s(window):
    """After this long without input nothing is pending (constants read now, so --set overrides count)."""
    return decoder_mod.STALL_S + decoder_mod.FINAL_WAIT_S + window + 1.0


def logged_events(session):
    """[(t_pub, topic, msg)] of the decoder's outputs as the live decoder published them."""
    return [(m.get('t_pub', t), topic, m) for t, topic, m in buslog.read(session, OUT_TOPICS)]


def decode_events(session, dec, step=PUMP_STEP_S):
    """Run `dec` over the session's inputs offline; [(now, topic, msg)] of what it would have published."""
    out, last, settle = [], None, settle_s(dec.window)
    inputs = sorted(buslog.read(session, IN_TOPICS), key=lambda r: r[2].get('t_pub', r[0]))
    for t_recv, topic, msg in inputs:
        now = msg.get('t_pub', t_recv)
        if last is not None:
            t = last + step
            while t < min(now, last + settle):
                out += [(t, tp, m) for tp, m in dec.pump(t)]
                t += step
        try:
            out += [(now, tp, m) for tp, m in handle(dec, topic, msg, now)]
        except Exception as e:  # as fusion.run: log a bad message and go on
            print(f'evaluate: bad {topic}: {type(e).__name__}: {e}', file=sys.stderr)
        out += [(now, tp, m) for tp, m in dec.pump(now)]
        last = now
    if last is not None:
        t = last + step
        while t < last + settle:
            out += [(t, tp, m) for tp, m in dec.pump(t)]
            t += step
    return out


def segment_clocks(session):
    """[(first frame_t, last frame_t, offset)] per seg_id from the session's cam.tick, offset from its first tick."""
    segs = {}
    for t_recv, _, m in buslog.read(session, ['cam.tick']):
        ft = m['frame_t']
        if not len(ft):
            continue
        if m['seg_id'] not in segs:
            segs[m['seg_id']] = [float(ft[0]), float(ft[-1]), m.get('t_pub', t_recv) - float(ft[-1])]
        segs[m['seg_id']][1] = max(segs[m['seg_id']][1], float(ft[-1]))
    return list(segs.values())


def end_of_word_to_lock(t_pub, t_word, segs):
    """(delay s, offset-corrected?) or (None, None) when the clocks differ and no segment covers the word."""
    d = t_pub - t_word
    if abs(d) < CLOCK_OK_S:
        return d, False
    for f0, f1, off in reversed(segs):
        if f0 <= t_word <= f1:
            return t_pub - (t_word + off), True
    return None, None


def load_manifest(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def norm(text):
    return re.sub(r"[^a-z0-9' ]+", ' ', (text or '').lower()).split()


def edit_distance(ref, hyp):
    """Word-level Levenshtein distance."""
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def stats_ms(xs):
    if not xs:
        return None
    a = np.asarray(xs) * 1000
    return {'n': len(a), 'mean': float(a.mean()), 'median': float(np.median(a)), 'p90': float(np.percentile(a, 90))}


def utterances(events, segs=()):
    """utt_id -> what the decoder said about it, from its output messages in time order."""
    utts = {}
    for t, topic, m in sorted(events, key=lambda e: e[0]):
        u = utts.setdefault(m['utt_id'], {'first_t': t, 't0': None, 'locked': [], 'raw': [], 'repeats': 0,
                                          'lock_t': [], 'end_of_word_to_lock': [], 'corrected': 0, 'clock_unknown': 0,
                                          'say_t': [], 'gap': False})
        if topic == 'dec.hyp':
            off = m.get('offset', 0)
            if off > len(u['locked']):  # words before the log started: they would be misplaced
                u['gap'] = True
                continue
            raw = [r if r is not None else w for r, w in zip(m.get('raw') or m['locked'], m['locked'])]
            u['locked'][off:], u['raw'][off:] = list(m['locked']), raw
            if u['t0'] is None and m.get('t0') is not None:
                u['t0'] = m['t0']
        elif topic == 'dec.lock':
            for w in m['words']:
                u['lock_t'].append(t)
                d, corrected = end_of_word_to_lock(t, w['t'], segs)
                if d is None:
                    u['clock_unknown'] += 1
                    continue
                u['end_of_word_to_lock'].append(d)
                u['corrected'] += corrected
        elif topic == 'dec.repeat':
            u['repeats'] += 1
        elif topic == 'voice.say':
            u['say_t'] += [t] * len(m['words'])
    for u in utts.values():
        u['lock_to_voice'] = [s - l for l, s in zip(u['lock_t'], u['say_t'])]
    return utts


def score(events, manifest, segs=()):
    utts = utterances(events, segs)
    skipped = sorted(i for i, u in utts.items() if u['gap'])
    for i in skipped:
        print(f'evaluate: utterance {i} left out: its dec.hyp skips words (log started mid-utterance?)',
              file=sys.stderr)
        del utts[i]
    clips = [c for c in manifest if c.get('clip_t0') is not None and c.get('clip_t1') is not None]
    owner = {}
    for uid, u in utts.items():
        start = u['t0'] if u['t0'] is not None else u['first_t']
        clip = next((c for c in clips if c['clip_t0'] <= start <= c['clip_t1']), None)
        if clip is not None:
            owner.setdefault(clip['id'], []).append(uid)
    matched = {i for ids in owner.values() for i in ids}
    rows, tot = [], {'edits': 0, 'raw_edits': 0, 'ref': 0, 'false_words': 0, 'utts': 0, 'repeat_utts': 0}
    m2l, l2v, missed = [], [], []
    for c in clips:
        us = [utts[i] for i in sorted(owner.get(c['id'], []), key=lambda i: utts[i]['first_t'])]
        ref = norm(c.get('prompt')) if c.get('set') != REST_SET else []
        hyp = norm(' '.join(w for u in us for w in u['locked']))
        raw = norm(' '.join(w for u in us for w in u['raw']))
        row = {'id': c['id'], 'prompt': c.get('prompt'), 'hyp': ' '.join(hyp), 'raw': ' '.join(raw),
               'utts': len(us), 'repeats': sum(u['repeats'] for u in us),
               'end_of_word_to_lock_ms': stats_ms([x for u in us for x in u['end_of_word_to_lock']]),
               'lock_to_voice_ms': stats_ms([x for u in us for x in u['lock_to_voice']])}
        if ref:
            if not us:
                missed.append(c['id'])
            e, er = edit_distance(ref, hyp), edit_distance(ref, raw)
            row.update(wer=e / len(ref), raw_wer=er / len(ref))
            tot['edits'] += e
            tot['raw_edits'] += er
            tot['ref'] += len(ref)
        else:
            row.update(wer=None, raw_wer=None, false_words=len(hyp))
            tot['false_words'] += len(hyp)
        tot['utts'] += len(us)
        tot['repeat_utts'] += sum(u['repeats'] > 0 for u in us)
        m2l += [x for u in us for x in u['end_of_word_to_lock']]
        l2v += [x for u in us for x in u['lock_to_voice']]
        rows.append(row)
    overall = {'clips': len(clips), 'ref_words': tot['ref'], 'utts': tot['utts'],
               'wer': tot['edits'] / tot['ref'] if tot['ref'] else None,
               'raw_wer': tot['raw_edits'] / tot['ref'] if tot['ref'] else None,
               'please_repeat_rate': tot['repeat_utts'] / tot['utts'] if tot['utts'] else None,
               'false_words': tot['false_words'], 'missed_clips': len(missed),
               'end_of_word_to_lock_ms': stats_ms(m2l), 'lock_to_voice_ms': stats_ms(l2v),
               'end_of_word_to_lock_offset_corrected': sum(utts[i]['corrected'] for i in matched),
               'end_of_word_to_lock_clock_unknown': sum(utts[i]['clock_unknown'] for i in matched)}
    unmatched = [{'utt_id': i, 't': u['first_t'], 'text': ' '.join(u['locked'])}
                 for i, u in utts.items() if i not in matched]
    return {'overall': overall, 'clips': rows, 'unmatched': unmatched, 'missed_clips': missed, 'skipped_utts': skipped}


def apply_overrides(pairs):
    """['lock.THRESHOLD=0.7', ...] -> set those module constants: existing numeric ones, keeping their type."""
    for p in pairs:
        name, _, value = p.partition('=')
        mod, _, const = name.partition('.')
        cur = getattr(TUNABLE.get(mod), const, None) if const.isupper() else None
        if isinstance(cur, bool) or not isinstance(cur, (int, float)):
            raise SystemExit(f'--set {p}: expected lock.NAME=number or decoder.NAME=number for an existing constant')
        try:
            x = float(value)
        except ValueError:
            raise SystemExit(f'--set {p}: {value!r} is not a number')
        if not np.isfinite(x) or (isinstance(cur, int) and not x.is_integer()):
            raise SystemExit(f'--set {p}: needs a finite {type(cur).__name__}')
        setattr(TUNABLE[mod], const, type(cur)(x))


def _fmt(x, pct=True):
    return '-' if x is None else (f'{100 * x:.1f}%' if pct else f"{x['median']:.0f} ms (p90 {x['p90']:.0f})")


def report(res, out=None):
    out, o = out or sys.stdout, res['overall']
    print(f"clips {o['clips']}  utterances {o['utts']}  reference words {o['ref_words']}", file=out)
    print(f"WER {_fmt(o['wer'])}   before repairs {_fmt(o['raw_wer'])}   please repeat "
          f"{_fmt(o['please_repeat_rate'])} of utterances   false words in rest clips {o['false_words']}", file=out)
    print(f"end of word to lock {_fmt(o['end_of_word_to_lock_ms'], False)} ({o['end_of_word_to_lock_offset_corrected']} words "
          f"offset-corrected)   lock to voice {_fmt(o['lock_to_voice_ms'], False)}", file=out)
    for r in res['clips']:
        print(f"  {r['id']:<14} {_fmt(r['wer']):>6}  {r['hyp'][:60]!r} <- {(r['prompt'] or '')[:60]!r}", file=out)
    if res['missed_clips']:
        print(f"{len(res['missed_clips'])} prompted clip(s) with no utterance (counted in WER)", file=out)
    if res['unmatched']:
        print(f"{len(res['unmatched'])} utterance(s) outside every clip", file=out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('session', help="buslog session folder, or 'latest'")
    ap.add_argument('--manifest', required=True, help="the recorder's manifest.jsonl for the same session")
    ap.add_argument('--json', help='write the full result here')
    ap.add_argument('--decode', action='store_true', help='re-run the decoder offline over cam.* and claude.*')
    ap.add_argument('--set', action='append', default=[], metavar='MOD.NAME=X',
                    help='with --decode: override a constant of fusion.lock or fusion.decoder, repeatable')
    ap.add_argument('--units', default=UNITS, help='with --decode: unigram5000_units.txt (default %(default)s)')
    ap.add_argument('--spm', default=SPM, help='with --decode: unigram5000.model (default %(default)s)')
    ap.add_argument('--no-lookalikes', action='store_true')
    ap.add_argument('--window', type=float, help='with --decode: repair window, s (default decoder.REPAIR_WINDOW_S)')
    args = ap.parse_args(argv)
    if args.decode:
        apply_overrides(args.set)
        units = load_units(args.units)
        dec = Decoder(units, None if args.no_lookalikes else load_lookalikes(units, args.spm),
                      repair_window_s=args.window)
        events = decode_events(args.session, dec)
    elif args.set:
        raise SystemExit('--set only applies with --decode')
    else:
        events = logged_events(args.session)
    res = dict(score(events, load_manifest(args.manifest), segment_clocks(args.session)), session=str(buslog.resolve(args.session)),
               mode='decode' if args.decode else 'logged', overrides=args.set)
    report(res)
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(res, f, indent=1)
    return res


if __name__ == '__main__':
    main()
