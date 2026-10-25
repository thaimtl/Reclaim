"""The end-of-call review: Claude Opus 5.5 reads the whole call and says what Austin meant, so the end screen scores
this call's words, not a benchmark.

Live, there is no script to score against. After the call, the whole conversation is: the judge's lines and each of
Austin's sentences with what every reader made of each word (lips alone, lips and muscles before Claude, the captions
after Claude). Opus writes, for each sentence, what Austin most likely meant; each reader is then scored against that
with word error rate (substitutions + insertions + deletions over the words he meant), and the words the captions got
wrong are listed. It is an estimate made from context, and the end screen says so.

    from dashboard.review import review_input, review
    result = review(review_input(state))      # blocking: run it off the event loop (asyncio.to_thread)
"""
import json
import logging
import re

log = logging.getLogger('dashboard.review')

MODEL = 'claude-opus-5-5'
FALLBACK_BETA = 'server-side-fallback-2026-07-01'
MAX_TOKENS = 16000          # Opus always thinks first: room for the thinking and the JSON
TIMEOUT_S = 90.0
READERS = ('lips', 'lips_muscles', 'claude')   # the end screen's three bars, in order

SYSTEM = """You review a phone call made with Silent Voice: Austin mouths words without sound, a lip reader (camera) \
and muscle sensors read them, Claude fixes lip look-alikes, and a cloned voice speaks them. The judge's lines are exact.

For each of Austin's sentences, write what he most likely meant to say, given the whole conversation. The readers' \
mistakes are mostly lip look-alikes (p/b/m, f/v, t/d/n, s/z, k/g, words like "pack" and "bag"), dropped or extra short \
words, and words split or merged. Each word comes with what each reader saw; use them as evidence.

Rules:
- Keep the caption's word whenever it plausibly fits. Change a word only when the conversation makes another reading \
clearly more likely, preferably one a reader also saw or a lip look-alike of it.
- Never improve grammar, style or content, never add ideas, never finish a cut-off sentence. Write his words, not \
better words.
- If a sentence is too garbled to tell, keep the captions as they are and set "unclear" to true.
- "meant" is plain words with normal capitalisation and punctuation."""

SCHEMA = {
    'type': 'object',
    'properties': {
        'sentences': {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {'id': {'type': 'integer'}, 'meant': {'type': 'string'}, 'unclear': {'type': 'boolean'}},
                'required': ['id', 'meant', 'unclear'],
                'additionalProperties': False,
            },
        },
    },
    'required': ['sentences'],
    'additionalProperties': False,
}


def norm(text):
    """Words to compare: lower case, no punctuation (Auto-AVSR has neither)."""
    return [w for w in re.sub(r"[^\w' ]+", ' ', str(text).lower()).split() if w.strip("'")]


def review_input(state):
    """The call as Opus sees it, and each reader's words per sentence, from a finished CallState."""
    turns, sentences = [], []
    for line in state.transcript:
        if line['who'] == 'judge':
            turns.append({'judge': line['text']})
            continue
        u = _utt_for(state, line)
        captions = line['text']
        if u is None:                       # no word detail (e.g. a sentence closed without locks): captions only
            readers = {k: norm(captions) for k in READERS}
            words = [{'caption': w} for w in norm(captions)]
        else:
            final, before, lips = [], [], []
            for i, w in enumerate(u['locked']):
                rep = u['repaired'].get(i)
                pre = rep[0] if rep else w                  # what lips and muscles locked, before Claude
                post = rep[1] if rep else w
                cam = u['cam_top'][i] if i < len(u['cam_top']) and u['cam_top'][i] else pre
                final.append(post), before.append(pre), lips.append(cam)
            if u.get('caption'):        # Opus rewrote the caption at the sentence's end (claude.caption)
                final = [str(w) for w in u['caption']]
            readers = {'lips': norm(' '.join(lips)), 'lips_muscles': norm(' '.join(before)),
                       'claude': norm(' '.join(final))}
            words = [dict(caption=f, **({'before_claude': b} if b != f else {}), **({'lips_alone': c} if c != b else {}))
                     for f, b, c in zip(final, before, lips)]
            if u.get('caption'):        # a rewrite can change the word count: show the readers' words side by side
                words = {'caption': ' '.join(final), 'before_claude': ' '.join(before), 'lips_alone': ' '.join(lips)}
        sid = len(sentences) + 1
        sentences.append({'id': sid, 'readers': readers, 'captions': captions})
        turns.append({'austin': {'id': sid, 'captions': captions, 'words': words}})
    return {'turns': turns, 'sentences': sentences}


def _utt_for(state, line):
    """The decoder's sentence behind a transcript line: same utt_id, closed at the line's time."""
    for u in state.utts.values():
        if u['utt_id'] == line.get('utt_id') and u['final'] and u['t_final'] == line['t']:
            return u
    return None


def review(data, client=None):
    """Asks Opus, then scores. Returns the end screen's review, or raises (the caller shows the call without it)."""
    if not data['sentences']:
        return {'status': 'empty'}
    meant = ask(data['turns'], client)
    return score(data['sentences'], meant)


def ask(turns, client=None):
    """{sentence id: (meant, unclear)} from Opus 5.5 (structured output; server-side fallback on a refusal)."""
    if client is None:
        import anthropic                    # here, so the dashboard runs (and tests) without the SDK
        from claude_service.service import load_keys
        load_keys()                         # ANTHROPIC_API_KEY from call/keys.env unless the environment has it
        client = anthropic.Anthropic(timeout=TIMEOUT_S, max_retries=2)
    r = client.beta.messages.create(
        model=MODEL, max_tokens=MAX_TOKENS, betas=[FALLBACK_BETA], fallbacks='default', system=SYSTEM,
        output_config={'effort': 'high', 'format': {'type': 'json_schema', 'schema': SCHEMA}},
        messages=[{'role': 'user', 'content': 'The call, in order:\n' + json.dumps(turns, ensure_ascii=False, indent=1)}])
    text = next((b.text for b in r.content or [] if b.type == 'text' and b.text), None)
    if r.stop_reason != 'end_turn' or text is None:
        raise RuntimeError(f'no usable review (stop_reason {r.stop_reason})')
    log.info('call reviewed by %s: %s in, %s out', getattr(r, 'model', MODEL), r.usage.input_tokens, r.usage.output_tokens)
    return {int(s['id']): (s['meant'], bool(s['unclear'])) for s in json.loads(text)['sentences']}


def score(sentences, meant):
    """Word error per reader against what he meant, the words the captions got wrong, and how many sentences Opus
    couldn't tell (scored as they were captioned, so they count as right: the estimate errs on our side, and says so)."""
    errors, total, wrong, unclear = dict.fromkeys(READERS, 0), 0, [], 0
    for s in sentences:
        text, is_unclear = meant.get(s['id'], (s['captions'], True))
        is_unclear = is_unclear or not norm(text)         # an empty answer can't be the truth either
        unclear += is_unclear
        ref = norm(s['captions'] if is_unclear else text)
        total += len(ref)
        for k in READERS:
            errors[k] += len(edits(s['readers'][k], ref))
        for op, got, want in edits(s['readers']['claude'], ref):
            wrong.append({'sentence': s['id'], 'op': op, 'got': got, 'meant': want})
    wer = {k: round(100 * errors[k] / total, 1) if total else None for k in READERS}
    return {'status': 'done', 'model': MODEL, 'words': total, 'wer': wer, 'wrong': wrong[:12], 'n_wrong': len(wrong),
            'sentences': len(sentences), 'unclear': unclear,
            'meant': {s['id']: meant.get(s['id'], (s['captions'], True))[0] for s in sentences}}


def edits(hyp, ref):
    """The word edits that turn hyp into ref (Levenshtein, back-traced): [(op, hyp word, ref word)]."""
    n, m = len(hyp), len(ref)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (hyp[i - 1] != ref[j - 1]))
    out, i, j = [], n, m
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (hyp[i - 1] != ref[j - 1]):
            if hyp[i - 1] != ref[j - 1]:
                out.append(('sub', hyp[i - 1], ref[j - 1]))
            i, j = i - 1, j - 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            out.append(('extra', hyp[i - 1], None))
            i -= 1
        else:
            out.append(('missed', None, ref[j - 1]))
            j -= 1
    return out[::-1]
