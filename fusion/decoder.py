"""The decoder's state machine: camera passes -> words that lock, wait out the repair window, and leave on voice.

Pure: every handler takes a decoded message and the decoder's clock (bus.now()) and returns the bus
messages to publish as [(topic, msg)]. run.py owns the sockets; tests drive it with fake ticks.

Utterances. A segment from the camera is one utterance, except that a segment capped at 8 s continues in the
next seg_id, whose first 2 s repeat the old segment's end: that is the same utterance. It is a continuation only
when the old segment reached the cap (T >= CAP_T_MIN) and the new one starts inside it; a normal pause also
overlaps (0.5 s still before the end, 0.3 s pre-roll), and that is a new sentence. At the cap, the old
segment's unlocked words that end before the new segment starts are frozen (finished, never re-read); the rest
are dropped, and the new segment's reading of them takes over. Utterances stay bounded: once one spans
MAX_UTT_S, the next cap ends it as a sentence (closed, settled, voice final) and the new segment starts a new
utterance, which carries the old one's words in the overlap only to cut them out of its passes.

Each pass. Only the newest tick per seg_id counts (run.py drops older ones). The words of the pass are split
into the part the pinned prefix (locked words, then frozen ones) already covers and the new tail: the cut is
after the tail-side word matching the prefix's last matching word (difflib on the words, within MATCH_TOL_S),
and never before a word whose middle lies inside the prefix's time. So locked words can never be rewritten or
repeated. The tail is aligned to the previous pass's tail (difflib again): a word in an equal block survived the
pass and carries passes + 1. Then words lock left to right while their score crosses the threshold (lock.py).

Claude. The newest claude.predict (until its ttl_s runs out) gives each word Claude's weight and moves the
threshold. Before locking, a camera-tied alternative (a lip look-alike, or a CTC competitor within NEAR_TIE)
that Claude clearly prefers replaces the greedy word; the camera still has to carry the lock. A claude.repair
that reaches a locked word while it waits its REPAIR_WINDOW_S changes what the voice says; a later one changes
the captions only (dec.hyp keeps the locked text in `raw` and the spoken text in `spoken`). Like the Claude
service, the decoder only applies an edit whose `from` is the word and whose `to` is one of its alts; rejected
edits are counted on timing.dec. Claude's please_repeat never holds back the voice: it is reported on dec.repeat
(reader 'claude') when the sentence ends, or at once if it already has.

dec.hyp carries the interface fields plus `offset` (utterance index of its first locked word: only the last
HYP_LOCKED locked words are sent, so the message stays small), `raw`, `spoken` (parallel to `locked`) and `t0`
(capture time of the utterance's first frame, for matching recordings offline).

Clocks. `now` (arrival time) drives everything timed: sentence end, the repair window, ttl_s. frame_t is only
compared with other frame times (word spans, the cap), because a camera replaying a recording keeps the
recording's clock in frame_t while t_pub is live.

Sentence end. The utterance closes when its ticks stop and the newest read ends quiet (the camera closes a
segment after 0.5 s still), or ticks stall for STALL_S. Closing locks what it can with every word finished, then
waits for cam.final: its n-best settles the unlocked words. A word still below threshold, and everything after
it, is not guessed: dec.repeat says why, and voice.say final=True closes the sentence with the locked words.

EMG hook: pass emg= an object with support(word) -> 0..1 or None (it gets the Word, with t0/t1 in capture
time) and the gate in lock.py weighs it against the camera. With emg=None (today) the camera has full weight.
"""
import difflib
import time

import numpy as np

from fusion import lock
from fusion.words import extract_words

REPAIR_WINDOW_S = 0.3   # a locked word waits this long for claude.repair before voice.say
END_GAP_S = 0.35        # no tick for this long, and the newest read ends quiet: the segment closed
STALL_S = 1.5           # no tick for this long: closed whatever the activity says
FINAL_WAIT_S = 2.5      # how long a closed utterance waits for cam.final (sent ~1.1 s after the end)
ACT_QUIET = 0.12        # activity below this is still lips (live/endpoint.py STILL_THR)
QUIET_FRAMES = 5        # newest frames still, after a word's end: the lips paused (200 ms at 25 fps)
END_QUIET_FRAMES = 10   # still frames that must end the newest read before a tick gap closes the utterance
MATCH_TOL_S = 0.5       # a prefix word and a pass word only match if their middles are this close
NEAR_TIE = 0.15         # a CTC competitor this close to the greedy word's prob counts as camera-tied
CLAUDE_SWAP_MARGIN = 0.3  # Claude's weight must favour the tied alternative by this much to swap
NBEST_W = 0.7           # after cam.final, camera confidence = (1 - NBEST_W) * greedy + NBEST_W * n-best support
PHRASE_WEIGHT = 0.5     # weight of a word that appears only in a predicted phrase
PACE = 1.0              # voice.say pace and effort, relative to normal (1.0 = normal); EMG will set them
EFFORT = 1.0
KEEP_UTTS = 50          # past utterances kept for late repairs
CAP_T_MIN = 190         # frames the old segment must have reached for a new one to continue it (cap at 200)
MAX_UTT_S = 30.0        # an utterance this long ends at its next cap
HYP_LOCKED = 40         # dec.hyp sends at most this many locked words, the newest


class Utt:
    def __init__(self, utt_id, seg_id, now, t0):
        self.utt_id = utt_id
        self.segs = [seg_id]
        self.t0 = t0            # capture time of its first frame
        self.carry = []         # the previous utterance's words in the overlap: cut out, never locked here
        self.locked = []        # pinned prefix, in order; indices are claude.repair's
        self.unlocked = []      # frozen words from capped segments, then the newest pass's tail
        self.state = 'open'     # open -> closing (waiting for cam.final) -> done
        self.last_tick = now
        self.last_T = 0
        self.last_frame_t = -np.inf
        self.tail_quiet = False
        self.seg_words = {}     # seg_id -> its newest pass's words, for the n-best
        self.nbest = {}         # seg_id -> cam.final n-best
        self.closed_t = None
        self.spoken = 0         # locked words already sent on voice.say
        self.final_sent = False
        self.claude_repeat = False      # Claude answered please_repeat for this utterance
        self.claude_repeat_sent = False


class Decoder:
    def __init__(self, units, lookalikes=None, model=None, emg=None, repair_window_s=None):
        self.units = units
        self.lookalikes = lookalikes    # word -> look-alikes, most frequent first (words.LookAlikes), or None
        self.model = model or lock.LockModel()
        self.emg = emg
        self._window = repair_window_s  # None: REPAIR_WINDOW_S, read at call time so offline overrides reach it
        self.utts = {}          # utt_id -> Utt
        self.seg_utt = {}       # seg_id -> Utt
        self.cur = None         # the newest utterance
        self.next_id = 1
        self.pred, self.pred_until, self.pred_w = None, -np.inf, {}

    @property
    def window(self):
        return REPAIR_WINDOW_S if self._window is None else self._window

    # --- Claude's prediction -------------------------------------------------------------------------------

    def on_predict(self, msg, now):
        self.pred, self.pred_until = msg, now + float(msg.get('ttl_s', 0))
        w = {}
        for p in msg.get('phrases', []):
            for x in p.lower().split():
                w[x] = PHRASE_WEIGHT
        for p in msg.get('words', []):
            w[p['w'].lower()] = max(w.get(p['w'].lower(), 0.0), float(p['weight']))
        self.pred_w = w
        return []

    def _claude(self, word, now):
        return self.pred_w.get(word, 0.0) if now < self.pred_until else 0.0

    def threshold(self, now):
        return lock.threshold(self.pred.get('predictability') if now < self.pred_until else None)

    # --- camera passes -------------------------------------------------------------------------------------

    def on_tick(self, msg, now):
        t_start = time.perf_counter()
        out, seg, frame_t = [], msg['seg_id'], msg['frame_t']
        utt = self.seg_utt.get(seg)
        if utt is None:
            cur = self.cur
            capped = cur is not None and cur.state == 'open' and seg > cur.segs[-1] \
                and cur.last_T >= CAP_T_MIN and frame_t[0] <= cur.last_frame_t
            if capped and cur.last_frame_t - cur.t0 < MAX_UTT_S:
                self._cap(cur, seg, float(frame_t[0]))
                utt = cur
            else:
                carry = []
                if capped:  # long enough: end the sentence here, keep the overlap only to cut it out
                    self._cap(cur, seg, float(frame_t[0]))
                    cur.segs.pop()
                    carry = [w for w in cur.locked + cur.unlocked if w.t1 >= frame_t[0] - MATCH_TOL_S]
                if cur is not None and cur.state == 'open':
                    out += self._close(cur, now)
                utt = self._new_utt(seg, now, float(frame_t[0]))
                utt.carry = carry
            self.seg_utt[seg] = utt
        if seg != utt.segs[-1] or utt.state == 'done' or (len(frame_t) < utt.last_T):
            return out          # a late tick of a capped or settled segment, or an older read
        if utt.state == 'closing':
            utt.state = 'open'  # ticks stalled but the segment was still open
        utt.last_tick, utt.last_T, utt.last_frame_t = now, len(frame_t), float(frame_t[-1])
        act = np.asarray(msg['activity'])
        utt.tail_quiet = len(act) >= END_QUIET_FRAMES and float(act[-END_QUIET_FRAMES:].max()) < ACT_QUIET

        words = extract_words(msg, self.units, self.lookalikes)
        for w in words:
            w.seg_id = seg
        utt.seg_words[seg] = words
        prefix = utt.carry + utt.locked + [w for w in utt.unlocked if w.frozen]
        tail = words[cut(prefix, words, float(frame_t[0])):]
        for w in tail:
            w.claude = self._claude(w.w, now)
            self._choose(w, now)
            w.emg = self.emg.support(w) if self.emg is not None else None
            w.paused = len(act) - 1 - w.f1 >= QUIET_FRAMES and float(act[-QUIET_FRAMES:].max()) < ACT_QUIET
            w.face_ok = bool(msg['face_ok'])
        carry_passes([w for w in utt.unlocked if not w.frozen], tail)
        utt.unlocked = [w for w in utt.unlocked if w.frozen] + tail
        out += self._lock_run(utt, now, ended=False)
        out.append(('timing.dec', {'stage': 'dec.pass', 'ms': (time.perf_counter() - t_start) * 1000,
                                   'utt_id': utt.utt_id, 'seg_id': seg, 'T': len(frame_t), 'n_words': len(words)}))
        return out

    def _new_utt(self, seg, now, t0):
        utt = Utt(self.next_id, seg, now, t0)
        self.next_id += 1
        self.utts[utt.utt_id] = utt
        self.cur = utt
        for old in sorted(self.utts)[:-KEEP_UTTS]:
            gone = self.utts.pop(old)
            for s in gone.segs:
                self.seg_utt.pop(s, None)
        return utt

    def _cap(self, utt, seg, seg_start):
        """The old segment hit the 8 s cap; `seg` re-reads its last 2 s and goes on."""
        keep = [w for w in utt.unlocked if w.frozen or w.t1 < seg_start]
        for w in keep:
            w.frozen = True
        utt.unlocked = keep
        utt.segs.append(seg)
        utt.last_T = 0

    def _choose(self, w, now):
        """Swap the greedy word for a camera-tied alternative that Claude clearly prefers."""
        best, best_c = None, w.claude + CLAUDE_SWAP_MARGIN
        for a, p in w.alts:
            c = self._claude(a, now)
            if p >= w.cam - NEAR_TIE and c >= best_c:
                best, best_c = (a, p), c
        if best is not None:
            w.alts = [x for x in w.alts if x[0] != best[0]] + [(w.w, w.cam)]
            w.w, w.cam, w.claude, w.source = best[0], best[1], best_c, 'claude'

    def _features(self, w, finished):
        return {'cam': w.cam, 'emg': w.emg, 'claude': w.claude, 'finished': finished, 'passes': w.passes,
                'face_ok': w.face_ok}

    def _lock_run(self, utt, now, ended):
        """Lock words left to right while they cross the threshold; score the rest for the captions."""
        thr, newly = self.threshold(now), []
        blocked = False
        for i, w in enumerate(list(utt.unlocked)):
            w.claude = max(w.claude, self._claude(w.w, now))
            finished = ended or w.frozen or i < len(utt.unlocked) - 1 or w.paused
            f = self._features(w, finished)
            w.score = self.model.score(f)
            if not blocked and self.model.locks(f, thr):
                w.lock_t, w.raw = now, w.w
                newly.append(w)
            else:
                blocked = True
        if newly:
            utt.locked += newly
            utt.unlocked = utt.unlocked[len(newly):]
        out = []
        if newly:
            out.append(('dec.lock', {'utt_id': utt.utt_id, 'words': [
                {'w': w.w, 'conf': w.score, 'alts': [a for a, _ in w.alts], 'cam': w.cam, 'emg': w.emg, 't': w.t1}
                for w in newly]}))
        out.append(self._hyp(utt))
        return out

    def _hyp(self, utt):
        off = max(0, len(utt.locked) - HYP_LOCKED)
        locked = utt.locked[off:]
        words = locked + utt.unlocked
        return ('dec.hyp', {'utt_id': utt.utt_id, 'locked': [w.w for w in locked],
                            'unlocked': [w.w for w in utt.unlocked], 'confidence': [w.score for w in words],
                            'source': [w.source for w in words], 'offset': off, 'raw': [w.raw for w in locked],
                            'spoken': [w.spoken for w in locked], 't0': utt.t0})

    # --- sentence end --------------------------------------------------------------------------------------

    def _close(self, utt, now):
        utt.state, utt.closed_t = 'closing', now
        out = self._lock_run(utt, now, ended=True)
        if not utt.unlocked or utt.segs[-1] in utt.nbest:
            out += self._settle(utt, now)
        return out

    def on_final(self, msg, now):
        utt = self.seg_utt.get(msg['seg_id'])
        if utt is None:
            return []
        utt.nbest[msg['seg_id']] = msg['nbest']
        if utt.state == 'closing' and msg['seg_id'] == utt.segs[-1]:
            return self._settle(utt, now)
        return []

    def _settle(self, utt, now):
        """The n-best settles unlocked words; any still below threshold is not guessed."""
        for seg, nbest in utt.nbest.items():
            apply_nbest(utt.seg_words.get(seg, []), [w for w in utt.unlocked if w.seg_id == seg], nbest)
        out = self._lock_run(utt, now, ended=True)
        utt.state = 'done'
        utt.seg_words, utt.nbest, utt.carry = {}, {}, []   # only needed until now
        if utt.unlocked:
            u0, thr, base = utt.unlocked[0], self.threshold(now), len(utt.locked)
            out.append(('dec.repeat', {'utt_id': utt.utt_id, 'reason': {
                'reader': 'emg' if u0.emg is not None and u0.emg < u0.cam else 'camera',
                'text': f"unsure of '{u0.w}' (score {u0.score:.2f} < {thr:.2f})",
                'threshold': thr, 'claude_please_repeat': utt.claude_repeat,
                'words': [{'index': base + k, 'w': w.w, 'cam': w.cam, 'emg': w.emg, 'claude': w.claude,
                           'nbest': w.nbest, 'score': w.score, 'alts': [a for a, _ in w.alts]}
                          for k, w in enumerate(utt.unlocked)]}}))
            utt.claude_repeat_sent = utt.claude_repeat
        elif utt.claude_repeat:
            out += self._claude_repeat(utt, now)
        return out

    def _claude_repeat(self, utt, now):
        utt.claude_repeat_sent = True
        return [('dec.repeat', {'utt_id': utt.utt_id, 'reason': {
            'reader': 'claude', 'text': 'Claude could not make sense of the words under review',
            'threshold': self.threshold(now), 'claude_please_repeat': True, 'words': []}})]

    # --- Claude's repairs and the voice --------------------------------------------------------------------

    def on_repair(self, msg, now):
        utt = self.utts.get(msg['utt_id'])
        if utt is None:
            return []
        applied = rejected = 0
        for e in msg.get('edits', []):
            i, frm, to = e.get('index'), _norm(e.get('from')), _norm(e.get('to'))
            w = utt.locked[i] if isinstance(i, int) and 0 <= i < len(utt.locked) else None
            if w is None or w.w != frm or to == frm or to not in {a for a, _ in w.alts}:
                rejected += 1   # Claude may only swap a word for one of its alternatives
                continue
            w.alts = [x for x in w.alts if x[0] != to] + [(w.w, w.cam)]
            w.w, w.source = to, 'claude'   # spoken already: captions only
            applied += 1
        out = [self._hyp(utt)] if applied else []
        if msg.get('please_repeat'):  # never holds the voice: Claude alone does not decide what is said
            utt.claude_repeat = True
            if utt.state == 'done' and not utt.claude_repeat_sent:
                out += self._claude_repeat(utt, now)
        out.append(('timing.dec', {'stage': 'dec.repair', 'ms': 0.0, 'utt_id': utt.utt_id,
                                   'applied': applied, 'rejected': rejected}))
        return out

    def pump(self, now):
        """Called every loop: close quiet utterances, time out cam.final, release words from the repair window."""
        out = []
        for utt in list(self.utts.values()):
            gap = now - utt.last_tick
            if utt.state == 'open' and (gap >= STALL_S or (gap >= END_GAP_S and utt.tail_quiet)):
                out += self._close(utt, now)
            if utt.state == 'closing' and now - utt.closed_t >= FINAL_WAIT_S:
                out += self._settle(utt, now)
            out += self._release(utt, now)
        return out

    def _release(self, utt, now):
        ready = []
        while utt.spoken < len(utt.locked) and utt.locked[utt.spoken].lock_t + self.window <= now:
            w = utt.locked[utt.spoken]
            w.spoken = w.w
            ready.append(w.w)
            utt.spoken += 1
        final = utt.state == 'done' and utt.spoken == len(utt.locked) and not utt.final_sent
        if not ready and not final:
            return []
        utt.final_sent = final   # sent even with no words, so the voice always closes its context
        return [('voice.say', {'utt_id': utt.utt_id, 'words': ready, 'final': final,
                               'pace': PACE, 'effort': EFFORT})]


def _norm(x):
    return str(x or '').strip().lower()


def cut(prefix, words, seg_start):
    """Index of the first word of this pass that the pinned prefix does not already cover."""
    if not prefix:
        return 0
    k = 0
    near = [p for p in prefix if p.t1 >= seg_start - MATCH_TOL_S]
    sm = difflib.SequenceMatcher(a=[p.w for p in near], b=[w.w for w in words], autojunk=False)
    for a, b, n in sm.get_matching_blocks():
        for i, j in zip(range(a, a + n), range(b, b + n)):
            if abs(near[i].mid - words[j].mid) <= MATCH_TOL_S:
                k = max(k, j + 1)
    end = prefix[-1].t1
    while k < len(words) and words[k].mid <= end:
        k += 1
    return k


def carry_passes(prev, tail):
    """A tail word in an equal block with the previous pass's tail survived it: passes + 1."""
    sm = difflib.SequenceMatcher(a=[w.w for w in prev], b=[w.w for w in tail], autojunk=False)
    for a, b, n in sm.get_matching_blocks():
        for i, j in zip(range(a, a + n), range(b, b + n)):
            tail[j].passes = prev[i].passes + 1


def apply_nbest(seg_words, targets, nbest):
    """cam.final n-best -> support 0..1 per word position; a target word takes the best-supported text.

    The hypotheses are weighted by softmax(score) and aligned to the segment's newest pass with difflib: equal
    words and one-for-one replacements vote for that position. The camera confidence moves NBEST_W of the way
    to the n-best support, so a beam (with its LM) that agrees lifts a word and one that disagrees sinks it.
    """
    if not nbest or not targets:
        return
    scores = np.array([h['score'] for h in nbest], dtype=float)
    weights = np.exp(scores - scores.max())
    weights /= weights.sum()
    texts = [w.w for w in seg_words]
    votes = [{} for _ in seg_words]
    for h, wt in zip(nbest, weights):
        hw = h['text'].lower().split()
        for op, a0, a1, b0, b1 in difflib.SequenceMatcher(a=texts, b=hw, autojunk=False).get_opcodes():
            if op == 'equal' or (op == 'replace' and a1 - a0 == b1 - b0):
                for i, j in zip(range(a0, a1), range(b0, b1)):
                    votes[i][hw[j]] = votes[i].get(hw[j], 0.0) + wt
    pos = {id(w): i for i, w in enumerate(seg_words)}
    for w in targets:
        i = pos.get(id(w))
        if i is None:
            continue
        v = votes[i]
        best = max(v, key=v.get) if v else w.w
        if v.get(best, 0.0) > v.get(w.w, 0.0):
            w.alts = [x for x in w.alts if x[0] != best] + [(w.w, w.cam)]
            w.w = best
        w.nbest = v.get(w.w, 0.0)
        w.cam = (1 - NBEST_W) * w.cam + NBEST_W * w.nbest
