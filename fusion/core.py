"""The fusion process's logic, without sockets or a clock of its own: bus messages in, bus messages out.

    core = Fusion(JointSearch(lm, lexicon=...), Gate.from_dict(...), LockModel.from_dict(...), svr=LiveSVR1(...))
    out = core.feed(topic, msg, now)      # every cam.tick, cam.final, emg.raw, claude.*, partner.text
    out += core.step(now)                 # every ~20 ms: ends sentences, releases words from the repair window
    for topic, msg in out: pub.send(topic, msg)

`now` is bus.now() live; offline (fusion/offline.py) it is the recording's clock, which the camera's frame times
and the EMG's sample times share (lipread.run --recorder-clock).

One utterance (utt_id) is one camera segment. Every cam.tick is one pass of the joint search (fusion/search.py) over
the whole segment so far; the lock model (fusion/lock.py) then locks words left to right. Locked words go on
dec.lock at once (Claude's repair lane) and on voice.say after the repair window (0.3 s), with Claude's repairs
applied. When the segment's ticks stop (end_wait_s), the end pass reads it once more as a finished sentence; when
cam.final's n-best arrives (or final_wait_s passes) the final pass adds it to the candidates, and any word still
below the threshold ends the sentence in "Please repeat" (dec.repeat) instead of being spoken.

EMG alone never writes captions: SVR-1 transcribes about 96% of words wrong on its own (dev, svr/dev_compare.py), so
speech the muscles detect while the camera has no face ends in "Please repeat" (reader: camera), not in guesses.
"""
import time
from collections import deque

import numpy as np

from fusion import lock as lockmod
from fusion.search import CamReading, gate_features, softmax, viterbi_word_ends, word_posteriors

EMG_RAIL = (8, 1015)       # dashboard/state.py: a pad off the skin pins its channel to a rail of the 10-bit range
EMG_FLAT = 1.0             # raw-unit standard deviation below this: no contact
PHRASE_WEIGHT = 0.3        # hint weight of a word that only appears in one of Claude's predicted phrases


class Utterance:
    def __init__(self, utt_id, seg_id, seg_start):
        self.utt_id, self.seg_id, self.seg_start = utt_id, seg_id, seg_start
        self.cam = None
        self.last_tick = None          # `now` of the newest cam.tick
        self.locked = []               # dec.lock word dicts (w, conf, alts, cam, emg, t, cam_top, emg_top)
        self.unlocked, self.unlocked_conf, self.unlocked_src = [], [], []
        self.stable = {}               # position -> (word, passes it has been the best candidate's word there)
        self.queue = deque()           # (release time, index) of locked words waiting out the repair window
        self.spoken = 0                # words sent on voice.say
        self.state = 'open'            # open -> ending (waiting for cam.final) -> done
        self.final_deadline = None
        self.nbest = None
        self.final_sent = False
        self.repeat_sent = False


class Fusion:
    def __init__(self, search, gate, lock, svr=None, repair_s=0.3, end_wait_s=0.3, final_wait_s=2.5,
                 use_emg=True, use_lm=True, use_hints=True, idle_emg=True, record=False, oracle=None):
        """idle_emg: between camera segments, keep reading SVR-1 to catch speech without a face (live only);
        record: keep every pass's judge scores and lock features in self.records (fitting, evaluation);
        oracle(utterance) -> true sentence or None, scored on the final pass when recording (gate fitting)."""
        self.search, self.gate, self.lock, self.svr = search, gate, lock, svr
        self.repair_s, self.end_wait_s, self.final_wait_s = repair_s, end_wait_s, final_wait_s
        self.use_emg, self.use_lm, self.use_hints = use_emg and svr is not None, use_lm, use_hints
        self.records = [] if record else None
        self.oracle = oracle
        self.idle_emg = idle_emg
        self.utt = None                # the open utterance
        self.closing = []              # utterances still speaking or waiting for cam.final
        self.recent = {}               # utt_id -> Utterance, for late Claude repairs
        self.next_id = 1
        self.hints, self.hints_until, self.predictability = {}, -1e18, 0.0
        self.phrases = []              # Claude's predicted phrases, as word lists
        self.raw = deque(maxlen=500)   # the last 0.5 s of raw EMG, for the pads' quality
        self.emg_quality = 1.0
        self.pace, self.effort = 1.0, 1.0
        self.last_emg_read = None
        self.idle_read_t = -1e18
        self.emg_only = None           # time an EMG-detected segment started with no camera segment
        from concurrent.futures import ThreadPoolExecutor
        self._pool = ThreadPoolExecutor(1, thread_name_prefix='svr')

    # ------------------------------------------------------------------ input
    def feed(self, topic, msg, now):
        out = []
        if topic == 'emg.raw':
            x = np.asarray(msg['samples']).reshape(-1, 4)
            self.raw.extend(x)
            if self.svr is not None:
                self.svr.feed(x, msg.get('t0'))
        elif topic == 'cam.tick':
            out += self._tick(msg, now)
        elif topic == 'cam.final':
            for u in self.closing:
                if u.seg_id == int(msg['seg_id']) and u.state == 'ending':
                    u.nbest = [dict(text=str(h['text']), score=float(h['score'])) for h in msg.get('nbest') or []]
                    out += self._final(u, now)
        elif topic == 'claude.predict' and self.use_hints:
            h = {}
            for p in msg.get('phrases') or []:
                for word in self.search_words(p):
                    h[word] = max(h.get(word, 0.0), PHRASE_WEIGHT)
            for item in msg.get('words') or []:
                for word in self.search_words(str(item.get('w', ''))):
                    h[word] = max(h.get(word, 0.0), float(item.get('weight', 0.0)))
            self.hints = h
            self.phrases = [self.search_words(str(p)) for p in msg.get('phrases') or []]
            self.hints_until = now + float(msg.get('ttl_s', 10.0))
            self.predictability = float(msg.get('predictability', 0.0))
        elif topic == 'claude.repair':
            out += self._repair(msg, now)
        elif topic == 'partner.text' and msg.get('is_final'):
            self.search.lm.add_text(str(msg.get('text', '')))
        return out

    @staticmethod
    def search_words(text):
        from fusion.search import norm_words
        return norm_words(text)

    def step(self, now):
        out = []
        u = self.utt
        if u is not None and u.last_tick is not None and now - u.last_tick >= self.end_wait_s:
            out += self._end(u, now)
        for v in list(self.closing):
            if v.state == 'ending' and now >= v.final_deadline:
                out += self._final(v, now)
        for v in ([self.utt] if self.utt else []) + self.closing:
            out += self._release(v, now)
        self.closing = [v for v in self.closing if not (v.state == 'done' and v.final_sent)]
        if self.utt is None and self.use_emg and self.idle_emg and now - self.idle_read_t >= 0.1:
            out += self._idle_emg(now)
        return out

    # ------------------------------------------------------------------ passes
    def _tick(self, msg, now):
        out = []
        seg = int(msg['seg_id'])
        ended = self._ended(seg, float(msg['seg_start']))
        if ended is not None:   # the camera's last tick of a segment comes ~0.55 s late, after end_wait_s ended it:
            if ended.state == 'ending':     # still that sentence (its final pass reads it), never a new one
                ended.cam = CamReading(msg)
            return out
        if self.utt is not None and self.utt.seg_id != seg:
            out += self._end(self.utt, now)
        if self.utt is None:
            self.utt = Utterance(self.next_id, seg, float(msg['seg_start']))
            self.recent[self.next_id] = self.utt
            for k in [k for k in self.recent if k < self.next_id - 20]:
                del self.recent[k]
            self.next_id += 1
            self.emg_only = None
        u = self.utt
        u.cam, u.last_tick = CamReading(msg), now
        out += self._pass(u, now, final=False)
        return out

    def _ended(self, seg, seg_start):
        """The ended utterance of this camera segment, if any (seg_start too: a restarted camera reuses seg_ids)."""
        for v in list(self.closing) + list(self.recent.values()):
            if v is not self.utt and v.state != 'open' and v.seg_id == seg and v.seg_start == seg_start:
                return v
        return None

    def _end(self, u, now):
        """The segment's ticks stopped: read it once more as a finished sentence, then wait for cam.final."""
        out = self._pass(u, now, final=True) if u.cam is not None else []
        u.state, u.final_deadline = 'ending', now + self.final_wait_s
        self.closing.append(u)
        if self.utt is u:
            self.utt = None
        return out

    def _final(self, u, now):
        """The final pass, with the camera's attention n-best: every word left is spoken, unless the sentence as a
        whole is unreadable (LockModel.settle); then the words left end in "Please repeat"."""
        out = self._pass(u, now, final=True, nbest=u.nbest or (), settle=True)
        u.state = 'done'
        rest = list(range(len(u.locked), len(u.locked) + len(u.unlocked)))
        if rest and not u.repeat_sent:
            out.append(('dec.repeat', self._repeat_msg(u, rest)))
            u.repeat_sent = True
        words = ' '.join(w['w'] for w in u.locked)
        if words:
            self.search.lm.add_text(words)      # Austin's own words join the call cache
        return out

    def _weights(self, cam, emg_read):
        f = gate_features(cam, emg_read, self.emg_quality)
        g = self.gate.weights(f)
        return f, {'cam': g[0], 'emg': g[1] if emg_read is not None else 0.0, 'lm': g[2] if self.use_lm else 0.0,
                   'beta': self.gate.beta if self.use_hints else 0.0, 'gamma': self.gate.gamma,
                   'final': self.gate.final, 'unseen': self.gate.unseen}

    def _pass(self, u, now, final, nbest=(), settle=False):
        t_start = time.perf_counter()
        self._quality()
        # SVR-1 re-reads the segment on the GPU while the camera beam runs here; the gate uses the newest EMG read
        # it has (the previous pass's, 100 ms old)
        job = self._pool.submit(self.svr.read, True, u.seg_start) if self.use_emg else None
        gf, w = self._weights(u.cam, self.last_emg_read if self.use_emg else None)
        hints = self.hints if (self.use_hints and now < self.hints_until) else {}
        locked = [x['w'] for x in u.locked]
        cands = self.search.candidates(u.cam, w, hints, locked, [h['text'] for h in nbest], final,
                                       phrases=self.phrases if hints else (),
                                       locked_tokens=[x['tokens'] for x in u.locked])
        emg_read = job.result() if job is not None else None
        if emg_read is not None:
            self.last_emg_read = emg_read
            self.pace = emg_read['pace'] if np.isfinite(emg_read['pace']) else 1.0
            if emg_read['effort'] is not None and np.isfinite(emg_read['effort']):
                self.effort = emg_read['effort']
            if self.last_emg_read is not None and w['emg'] == 0.0:
                gf, w = self._weights(u.cam, emg_read)
            from svr.decoders import greedy_decode
            self.search.extend(cands, [greedy_decode(emg_read['letter_logp'])], locked)
        elif self.use_emg:
            w['emg'] = 0.0
        emg_fn = self.svr.verify if emg_read is not None else None
        scores = self.search.judge_scores(cands, u.cam, emg_fn, hints, final, nbest)
        P = softmax(self.search.combine(scores, w))
        Pc = softmax(self.gate.t_cam * scores[:, 0])
        Pe = softmax(self.gate.t_emg * scores[:, 1]) if emg_read is not None else np.full(len(cands), 1.0 / len(cands))
        best, post = word_posteriors(cands, P, start=len(locked))
        tau = self.lock.threshold(self.predictability if hints else 0.0, final)
        pause = 1.0 - float(u.cam.activity[-8:].mean()) if u.cam is not None and len(u.cam.activity) else 1.0
        quiet = 1.0 - float(np.mean(emg_read['speak_prob'][-26:])) if emg_read is not None else 0.0
        words = []
        n_best = len(best.words)
        for k, d in enumerate(post):
            i = len(locked) + k
            word = best.words[i]
            mc = {c.words[i]: 0.0 for c in cands if i < len(c.words)}
            me = dict(mc)
            for c, qc, qe in zip(cands, Pc, Pe):
                if i < len(c.words):
                    mc[c.words[i]] += qc
                    me[c.words[i]] += qe
            prev = u.stable.get(i)
            stable = prev[1] + 1 if prev and prev[0] == word else 0
            u.stable[i] = (word, stable)
            f = lockmod.features(d.get(word, 0.0), mc.get(word, 0.0), me.get(word, 0.0) if emg_read is not None
                                 else mc.get(word, 0.0), hints.get(word, 0.0), final or i < n_best - 1, pause, quiet,
                                 stable, u.cam.face_ok if u.cam is not None else False, self.emg_quality)
            p = float(self.lock.prob(f))
            alts = self._alts(word, i, best, cands, d)
            words.append(dict(i=i, w=word, p=p, f=f, alts=alts, cam=mc.get(word, 0.0), tokens=best.tokens[i],
                              emg=me.get(word, 0.0) if emg_read is not None else None,
                              cam_top=max(mc, key=mc.get) if mc else word,
                              emg_top=max(me, key=me.get) if (me and emg_read is not None) else None,
                              claude=hints.get(word, 0.0), cands=sorted(d.items(), key=lambda kv: -kv[1])[:3],
                              mc=mc, me=me))
        new = []
        if settle and self.lock.settle(locked + [x['w'] for x in words],
                                       [x['conf'] for x in u.locked] + [x['p'] for x in words]):
            new = list(words)   # a wrong guess beats "Please repeat"
        else:
            for x in words:   # left to right: a word locks only after every word before it
                if not self.lock.may_lock(x['f'], x['p'], tau):
                    break
                new.append(x)
        for x in new:
            u.locked.append(dict(w=x['w'], conf=x['p'], alts=x['alts'], cam=x['cam'], emg=x['emg'], t=now,
                                 cam_top=x['cam_top'], emg_top=x['emg_top'], tokens=x['tokens']))
            u.queue.append((now + self.repair_s, len(u.locked) - 1))
        rest = words[len(new):]
        u.unlocked = [x['w'] for x in rest]
        u.unlocked_conf = [x['p'] for x in rest]
        u.unlocked_src = [self._source(x) for x in rest]
        u.last_words = rest
        ms = (time.perf_counter() - t_start) * 1000
        if self.records is not None:
            rec = dict(utt_id=u.utt_id, seg_id=u.seg_id, seg_start=u.seg_start, t=now, final=final,
                       with_nbest=bool(nbest), nbest=list(nbest), gate=gf, weights=dict(w), scores=scores,
                       cands=[c.words for c in cands], sources=[c.source for c in cands], best=best.words,
                       n_locked_before=len(locked), locked_now=[x['w'] for x in new],
                       words=[dict(i=x['i'], w=x['w'], p=x['p'], f=x['f'], cam_top=x['cam_top'],
                                   emg_top=x['emg_top']) for x in words], tau=tau, ms=ms,
                       cam_frame_t=float(u.cam.frame_t[-1]) if u.cam is not None else None)
            if final and self.oracle is not None:
                truth = self.oracle(u)
                if truth:
                    from fusion.search import Cand, norm_words
                    tw = norm_words(truth)
                    tc = [Cand(tw, [self.search.pieces.word_ids(x) for x in tw], 'truth')]
                    rec['truth'] = tuple(tw)
                    rec['truth_scores'] = self.search.judge_scores(tc, u.cam, emg_fn, hints, True, nbest)[0]
                    rec['truth_in_cands'] = tuple(tw) in set(rec['cands'])
                    if u.cam is not None:   # when each true word ends on the lips: the lock delay's reference
                        rec['truth_ends'] = viterbi_word_ends(u.cam, tc[0].tokens)
            self.records.append(rec)
        out = []
        if new:
            out.append(('dec.lock', {'utt_id': u.utt_id, 'words': [dict(w=x['w'], conf=x['p'], alts=x['alts'],
                                                                        cam=x['cam'], emg=x['emg'], t=now,
                                                                        cam_top=x['cam_top'], emg_top=x['emg_top'])
                                                                   for x in new]}))
        newest = words[-1] if words else None
        hyp = {'utt_id': u.utt_id, 'locked': [x['w'] for x in u.locked], 'unlocked': u.unlocked,
               'confidence': [x['conf'] for x in u.locked] + u.unlocked_conf,
               'source': [self._source(x) for x in u.locked] + u.unlocked_src, 'threshold': tau}
        if newest is not None:
            hyp['cands'] = [dict(w=c, cam=newest['mc'].get(c, 0.0), emg=newest['me'].get(c, 0.0) if
                                 emg_read is not None else 0.0, claude=hints.get(c, 0.0)) for c, _ in newest['cands']]
        out.append(('dec.hyp', hyp))
        out.append(('timing.dec', {'stage': 'lock', 'ms': ms, 'utt_id': u.utt_id, 'final': final,
                                   'candidates': len(cands)}))
        return out

    def _alts(self, word, i, best, cands, d):
        """What word i could be instead (Claude may only swap it for one of these): words that candidates identical
        to the best one everywhere else put there, then its lip look-alikes; real words only. Position i of a
        candidate with a word dropped or added is a neighbour shifted in (43% of locked words had one on session 3)."""
        pron = getattr(getattr(self.search, 'lexicon', None), 'pron', None)
        n = len(best.words)
        swaps = {c.words[i] for c in cands if len(c.words) == n and c.words[:i] == best.words[:i]
                 and c.words[i + 1:] == best.words[i + 1:]}
        out = [x for x, _ in sorted(d.items(), key=lambda kv: -kv[1]) if x in swaps]
        out += [x for x in self.search.lookalikes(word) if x not in out]
        near = set(best.words[max(0, i - 2):i] + best.words[i + 1:i + 3])
        return [x for x in out if x != word and x not in near and (pron is None or x in pron)][:4]

    # ------------------------------------------------------------------ voice, repairs, repeats
    @staticmethod
    def _source(x):
        """Who settled the word: the camera when its own pick, else EMG when EMG's pick, else Claude if predicted."""
        if x.get('cam_top') == x['w']:
            return 'camera'
        if x.get('emg_top') == x['w']:
            return 'EMG'
        if x.get('claude'):
            return 'Claude'
        return 'camera'

    def _release(self, u, now):
        out = []
        words = []
        while u.queue and u.queue[0][0] <= now:
            _, i = u.queue.popleft()
            words.append(u.locked[i]['w'])
        last = u.state == 'done' and not u.queue
        if words or (last and not u.final_sent):
            out.append(('voice.say', {'utt_id': u.utt_id, 'words': words, 'offset': u.spoken, 'final': last,
                                      'pace': float(self.pace), 'effort': float(self.effort)}))
            if words:
                out.append(('timing.dec', {'stage': 'window', 'ms': self.repair_s * 1000, 'utt_id': u.utt_id}))
            u.spoken += len(words)
            u.final_sent = u.final_sent or last
        return out

    def _repair(self, msg, now):
        u = self.recent.get(msg.get('utt_id'))
        out = []
        if u is None:
            return out
        for e in msg.get('edits') or []:
            i = int(e.get('index', -1))
            words = self.search_words(str(e.get('to', '')))
            frm = self.search_words(str(e.get('from', '')))
            if 0 <= i < len(u.locked) and len(words) == 1 and frm[:1] == [u.locked[i]['w']]:
                u.locked[i]['w'] = words[0]     # spoken words can't change; captions and later voice can
        if msg.get('please_repeat') and not u.repeat_sent:
            out.append(('dec.repeat', {'utt_id': u.utt_id, 'reason': {
                'reader': 'Claude', 'words': [dict(index=i, w=x['w'], alts=x['alts']) for i, x in
                                              enumerate(u.locked) if i >= u.spoken]}}))
            u.repeat_sent = True
        return out

    def _repeat_msg(self, u, indices):
        rest = {x['i']: x for x in getattr(u, 'last_words', [])}
        words, cam_low = [], 0
        for i in indices:
            x = rest.get(i)
            if x is None:
                continue
            words.append(dict(index=i, w=x['w'], alts=x['alts'], p=x['p'], cam=x['cam'], emg=x['emg']))
            cam_low += (x['emg'] is not None and x['cam'] < x['emg'])
        face = u.cam.face_ok if u.cam is not None else False
        has_emg = any(x['emg'] is not None for x in words)
        reader = 'camera' if (not face or not has_emg or cam_low * 2 > len(words)) else 'EMG'
        return {'utt_id': u.utt_id, 'reason': {'reader': reader, 'words': words, 'threshold': self.lock.threshold(
            self.predictability)}}

    def _quality(self):
        """1 - the worst pad's share of railed samples over the last 0.5 s; 0 for a flat (unconnected) pad."""
        if len(self.raw) < 50:
            return
        x = np.asarray(self.raw, np.float64)
        railed = ((x <= EMG_RAIL[0]) | (x >= EMG_RAIL[1])).mean(0)
        flat = x.std(0) < EMG_FLAT
        self.emg_quality = float(1.0 - max(railed.max(), float(flat.any())))

    def _idle_emg(self, now):
        """No camera segment: keep SVR-1's own segmentation going; speech it hears without a face -> Please repeat."""
        self.idle_read_t = now
        r = self.svr.read(camera_active=False)
        if r is None:
            return []
        if r['segment_started']:
            self.emg_only = now
        if r['segment_ended'] and self.emg_only is not None:
            self.emg_only = None
            uid = self.next_id
            self.next_id += 1
            return [('dec.repeat', {'utt_id': uid, 'reason': {'reader': 'camera', 'words': []}})]
        return []
