// Silent Voice live dashboard: draws each snapshot from dashboard/server.py (dashboard/state.py) as Thai's Figma v5.
// Text frames are snapshots (JSON); binary frames are cam.preview JPEGs with their face points (server.py pack_frame).
'use strict';
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
const SRC = { camera: 'camera', emg: 'emg', claude: 'claude', lips: 'camera', muscles: 'emg' };
const TONE_COLOR = { happy: '--warm', sad: '--concerned', neutral: '--calm', surprised: '--surprised', angry: '--firm' };
// Pad = landmark + a * (nose-to-chin axis) + b * (cheek-to-cheek axis); defaults measured on session 2, C recalibrates.
const PAD_DEFAULT = {
  A0: { lm: 152, a: 0.1092, b: -0.0342, left: false }, A1: { lm: 379, a: -0.0212, b: -0.0043, left: false },
  A2: { lm: 150, a: -0.0298, b: 0.021, left: true }, A3: { lm: 411, a: 0.0126, b: -0.0134, left: false },
};
const AXIS = { top: 168, chin: 152, left: 234, right: 454 };
const PAD_ORDER = ['A0', 'A1', 'A2', 'A3'];
const MOUTH = { left: 61, right: 291, top: 13, bottom: 14 };
const W = 760, H = 1080, PREVIEW_STALE_MS = 2000;
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

let snap = null, lastPreview = -Infinity, speaking = false, toneLabel = 'neutral';
let frame = null, framePending = false, lastFace = null;

// ---- the 1920 x 1080 stage, scaled to the window -------------------------------------------------------
function fit() {
  const k = Math.min(innerWidth / 1920, innerHeight / 1080);
  $('stage').style.transform = `translate(-50%, -50%) scale(${k})`;
}
addEventListener('resize', fit); fit();

// ---- connection ------------------------------------------------------------------------------------------
function connect() {
  const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
  ws.binaryType = 'arraybuffer';
  ws.onmessage = e => {
    if (typeof e.data === 'string') { snap = JSON.parse(e.data); render(snap); return; }
    onFrame(e.data);
  };
  ws.onclose = () => setTimeout(connect, 1000);
}
// ?offline: no websocket; a test hands snapshots to window.reclaimRender (dashboard/demo_feed.py --fixtures)
window.reclaimRender = s => { snap = s; render(s); };
if (!new URLSearchParams(location.search).has('offline')) connect();

// ---- the Call / Text switch ------------------------------------------------------------------------------
// The call server owns the mode; a click (or M) asks the dashboard server to switch it. The thumb moves at once
// and settles on what the call server says; if it can't switch, the thumb goes back and says why.
let modeWant = null, noteTimer = null;
function renderMode(s) {
  const el = $('mode'), ch = modeWant || s.channel || 'call';
  el.dataset.ch = ch;
  el.classList.toggle('off', !s.switchable);
  el.classList.toggle('busy', !!modeWant);
  el.querySelectorAll('button').forEach(b => {
    b.setAttribute('aria-checked', String(b.dataset.ch === ch));
    b.disabled = !s.switchable;
  });
}
async function switchMode(ch) {
  if (!snap || !snap.switchable || modeWant || ch === snap.channel) return;
  modeWant = ch;
  renderMode(snap);
  try {
    const r = await fetch('/api/channel', { method: 'POST', body: JSON.stringify({ channel: ch }),
      headers: { 'Content-Type': 'application/json', 'X-Silent-Voice': '1' } });
    if (!r.ok) throw new Error((await r.text()) || r.statusText);
    snap.channel = (await r.json()).channel;
    modeNote('');
  } catch (e) {
    modeNote(`Couldn't switch: ${e.message}`);
  } finally {
    modeWant = null;
    renderMode(snap);
  }
}
function modeNote(text) {
  const el = $('mode-note');
  el.textContent = text;
  el.classList.toggle('show', !!text);
  clearTimeout(noteTimer);
  if (text) noteTimer = setTimeout(() => el.classList.remove('show'), 5000);
}
$('mode').addEventListener('click', e => { const b = e.target.closest('button'); if (b) switchMode(b.dataset.ch); });
addEventListener('keydown', e => {
  if ((e.key === 'm' || e.key === 'M') && !e.ctrlKey && !e.metaKey && !e.altKey && snap && snap.mode !== 'end')
    switchMode(snap.channel === 'sms' ? 'call' : 'sms');
});

// ---- live screen -----------------------------------------------------------------------------------------
const mmss = s => `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;

function render(s) {
  $('live').classList.toggle('hidden', s.mode === 'end');
  $('end').classList.toggle('hidden', s.mode !== 'end');
  if (s.mode === 'end') { renderEnd(s); return; }
  const live = s.mode === 'live';
  $('live-dot').classList.toggle('off', !live);
  const sms = s.channel === 'sms';
  $('title').textContent = !live ? 'Waiting for the call' : sms ? 'Live call, the judge is texting' : 'Live call with the judge';
  $('judge-via').textContent = sms ? '· texted from the phone on the call' : '· transcribed by Cartesia Ink';
  $('sub').textContent = live ? `${mmss(s.elapsed)}${s.number ? ' on ' + s.number : ''}` : (s.number || '');
  $('m-lips').textContent = s.metrics.lips_to_voice == null ? '--' : (s.metrics.lips_to_voice / 1000).toFixed(2) + ' s';
  $('m-fixed').textContent = s.metrics.fixed;
  $('m-repeats').textContent = s.metrics.repeats;
  renderMode(s);
  renderConversation(s);
  renderLatency(s);
  $('status').classList.toggle('warn', !s.status.ok);
  $('status').querySelector('span').textContent = s.status.text;
  renderCamera(s);
}

function renderConversation(s) {
  const t = s.transcript;
  const lastJudge = t.map(x => x.who).lastIndexOf('judge');
  const judge = s.judge_live || (lastJudge >= 0 ? t[lastJudge].text : '');
  $('judge').classList.toggle('hidden', !judge);
  $('judge-text').textContent = judge;
  $('judge-text').classList.toggle('live', !!s.judge_live);
  const a = s.austin, words = $('words');
  $('austin').classList.toggle('waiting', s.mode !== 'live');
  const current = a && !(a.final && lastJudge >= 0 && t[lastJudge].t > (a.t || 0)) ? a : null;
  const cut = s.judge_live ? t.length : Math.max(0, lastJudge);
  const past = t.slice(0, cut).filter(x => !(current && x.who === 'austin' && x.utt_id === current.utt_id));
  $('history').innerHTML = past.slice(-2).map(x => `<div class="past ${x.who}">${esc(x.text)}</div>`).join('');
  if (!current || !current.words.length) {
    words.className = 'words empty';
    words.innerHTML = `<div class="word"><span class="w">${s.mode === 'live' ? 'Listening' : 'Not on a call yet'}</span></div>`;
  } else {
    words.className = 'words';
    const thr = current.cands && current.cands.threshold;
    words.innerHTML = current.words.map(w => {
      const src = w.locked ? (SRC[w.src] || 'camera') : 'pending';
      const was = w.was ? `<span class="was">${esc(w.was)}</span>` : '';
      return `<div class="word${w.locked ? '' : ' pending'}">${was}<span class="w">${esc(w.w)}</span>` +
             `<div class="bar"><i class="src-${src}" style="width:${barWidth(w, thr)}%"></i></div></div>`;
    }).join('');
  }
  renderExplain(s, current);
}

// the bar under a word: full once locked; a pending word fills toward the lock threshold
const barWidth = (w, thr) => w.locked ? 100 : w.conf == null ? 0 : Math.round(100 * Math.min(1, Math.max(0, w.conf / (thr || 1))));

function renderExplain(s, a) {
  let html = '';
  const repair = a && a.repairs.length ? a.repairs[a.repairs.length - 1] : null;
  if (a && a.repeat) {
    const w = a.repeat.words, alts = (a.repeat.alts[0] || []).slice(0, 1);
    const reader = { camera: 'Lips', emg: 'Muscles', claude: 'Claude' }[a.repeat.reader] || 'The readers';
    html += `<div class="repeat-card"><span class="ic">!</span><b>Please repeat</b><p>${esc(reader)} read ` +
            `“${esc(w[0] || '')}”${alts.length ? `, the others lean to “${esc(alts[0])}”` : ''}. ` +
            `Too close to call, so nothing was spoken.</p></div>`;
  } else if (repair) {
    html += `<div class="note claude">✦ <span>Claude changed “${esc(repair.frm)}” to “${esc(repair.to)}”: ` +
            `they look the same on the lips, and “${esc(repair.to)}” fits the conversation.</span></div>`;
  }
  if (a && a.cands && a.cands.rows.length) {
    const rows = a.cands.rows, thr = a.cands.threshold, tot = r => r.cam + r.emg + r.claude;
    const best = k => [...rows].sort((x, y) => y[k] - x[k])[0];
    const byCam = best('cam'), byEmg = best('emg'), top = rows[0];
    if (!a.repeat && !repair) {
      html += `<div class="note">${!s.muscles ? 'The muscle sensors are off, so the camera and Claude decide alone.'
        : byCam.w !== top.w ? `Lips alone see “${esc(byCam.w)}”. With the conversation, “${esc(top.w)}” leads.`
        : byEmg.w !== top.w ? `Muscles alone lean to “${esc(byEmg.w)}”. Lips and the conversation pick “${esc(top.w)}”.`
        : `Lips and muscles agree on “${esc(top.w)}”.`}</div>`;
    }
    html += `<div class="legend"><span><i class="src-camera"></i>Lips</span><span><i class="src-emg"></i>Muscles</span>` +
            `<span><i class="src-claude"></i>Claude</span><span><i class="thr"></i>Lock threshold</span></div>`;
    html += rows.map((r, i) => `<div class="cand${i === 0 ? ' top' : ''}"><span class="cw">${esc(r.w)}</span>` +
      `<div class="track"><i class="src-camera" style="width:${100 * r.cam}%"></i><i class="src-emg" style="width:${100 * r.emg}%"></i>` +
      `<i class="src-claude" style="width:${100 * r.claude}%"></i>${thr == null ? '' : `<span class="thr" style="left:${100 * thr}%"></span>`}</div>` +
      `<span class="cv">${tot(r).toFixed(2)}</span></div>`).join('');
  }
  $('explain').innerHTML = html;
}

function renderLatency(s) {
  const a = s.austin;
  const active = a && a.repairs.length && !a.final ? 'claude' : a && a.spoken && !a.final ? 'sonic' : 'read';
  for (const st of s.latency) {
    const el = document.querySelector(`.stage[data-k="${st.key}"]`);
    el.classList.toggle('active', st.key === active);
    el.querySelector('.sl').textContent = st.label;
    el.querySelector('.sv b').textContent = st.ms == null ? '--' : st.ms;
  }
}

// ---- camera side -----------------------------------------------------------------------------------------
function coverMap(size) {           // processed-frame normalized coords -> 760 x 1080 panel (object-fit: cover)
  const [fw, fh] = size && size[0] ? size : [1280, 720];
  const k = Math.max(W / fw, H / fh), ox = (W - fw * k) / 2, oy = (H - fh * k) / 2;
  return (x, y) => [ox + x * fw * k, oy + y * fh * k];
}

// ---- face filter: the picture and the pads on it, drawn in the same animation frame -------------------------
let frameSeq = 0;
async function onFrame(buf) {
  const seq = ++frameSeq;
  try {
    const dv = new DataView(buf);
    if (dv.getUint32(0) !== 0x52434631) return;           // 'RCF1'
    const n = dv.getUint32(4), head = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, n)));
    const bitmap = await createImageBitmap(new Blob([new Uint8Array(buf, 8 + n)], { type: 'image/jpeg' }));
    if (frame && frame.seq > seq) { bitmap.close(); return; }
    if (frame) frame.bitmap.close();
    frame = { bitmap, seq, ...head };
    lastPreview = performance.now();
    if (!framePending) { framePending = true; requestAnimationFrame(drawFrame); }
  } catch (err) { /* skip a bad frame */ }
}

function drawFrame() {
  framePending = false;
  if (!frame) return;
  const f = frame, size = [f.w || f.bitmap.width, f.h || f.bitmap.height];
  const k = Math.max(W / size[0], H / size[1]);
  const g = $('mesh').getContext('2d');
  g.drawImage(f.bitmap, (W - size[0] * k) / 2, (H - size[1] * k) / 2, size[0] * k, size[1] * k);
  if (f.pts) lastFace = { pts: f.pts, size };
  placeOverlays(lastFace);
}

function faceOnPanel(face) {
  const map = coverMap(face.size), P = face.pts.map(([x, y]) => map(x, y));
  const U = [P[AXIS.chin][0] - P[AXIS.top][0], P[AXIS.chin][1] - P[AXIS.top][1]];
  const V = [P[AXIS.right][0] - P[AXIS.left][0], P[AXIS.right][1] - P[AXIS.left][1]];
  return { P, U, V };
}

function placeOverlays(face) {
  const ring = $('mouth-ring'), pads = [...$('pads').children];
  if (!face || !snap || !snap.face_ok) {
    ring.classList.add('hidden');
    layoutCallouts(pads, PAD_ORDER.map(pin => PAD_FALLBACK[pin]), 270, 490);
    return;
  }
  const { P, U, V } = faceOnPanel(face);
  const [lx, ly] = P[MOUTH.left], [rx, ry] = P[MOUTH.right], [, ty] = P[MOUTH.top], [, by] = P[MOUTH.bottom];
  const cx = (lx + rx) / 2, cy = (ty + by) / 2, w = Math.hypot(rx - lx, ry - ly) * 1.9, h = w * 82 / 183;
  Object.assign(ring.style, { width: w + 'px', height: h + 'px', left: cx - w / 2 + 'px', top: cy - h / 2 + 'px',
                              transform: `rotate(${Math.atan2(ry - ly, rx - lx)}rad)` });
  ring.classList.remove('hidden');
  const dots = PAD_ORDER.map(pin => {
    const at = padAt[pin], p = P[at.lm] || P[0];
    return [p[0] + at.a * U[0] + at.b * V[0], p[1] + at.a * U[1] + at.b * V[1]];
  });
  layoutCallouts(pads, dots, Math.min(P[AXIS.left][0], P[AXIS.right][0]), Math.max(P[AXIS.left][0], P[AXIS.right][0]));
}

const PAD_FALLBACK = { A0: [378, 672], A1: [489, 616], A2: [270, 616], A3: [489, 516] };   // no face yet

const GAP = 28, ROW = 40, LABEL_MAX_Y = 820;          // LABEL_MAX_Y keeps clear of "Heard on the call"
function layoutCallouts(pads, dots, faceLeft, faceRight) {
  for (const left of [true, false]) {
    const idx = dots.map((d, i) => i).filter(i => pads[i] && (padAt[PAD_ORDER[i]].left !== false) === left)
      .sort((a, b) => dots[a][1] - dots[b][1]);
    let prev = -Infinity;
    const ys = idx.map(i => (prev = Math.max(dots[i][1], prev + ROW)));
    const over = ys.length ? Math.max(0, ys[ys.length - 1] - LABEL_MAX_Y) : 0;
    idx.forEach((i, j) => {
      const el = pads[i], pill = el.querySelector('.pill'), [x, y] = dots[i], ly = ys[j] - over;
      const w = pill.offsetWidth || 300;
      const lx = left ? Math.max(8, faceLeft - GAP - w) : Math.min(W - 8 - w, faceRight + GAP);
      Object.assign(el.querySelector('.pdot').style, { left: `${x - 12}px`, top: `${y - 12}px` });
      Object.assign(pill.style, { left: `${lx}px`, top: `${ly - 15.5}px` });
      const ex = left ? lx + w : lx, line = el.querySelector('line');
      line.setAttribute('x1', x); line.setAttribute('y1', y); line.setAttribute('x2', ex); line.setAttribute('y2', ly);
    });
  }
}

// ---- pad calibration: press C, then click A0 to A3 -------------------------------------------------------
let padAt = loadPads(), calib = null;
function loadPads() {
  try { return { ...PAD_DEFAULT, ...JSON.parse(localStorage.getItem('reclaim.pads') || '{}') }; }
  catch { return { ...PAD_DEFAULT }; }
}
function savePads() {
  try { localStorage.setItem('reclaim.pads', JSON.stringify(padAt)); } catch { /* kept for this session only */ }
}
const PAD_PLACE = { A0: 'under his chin', A1: 'his left chin corner', A2: 'his right cheek below the mouth',
                    A3: 'his left cheek above the mouth' };

function calibHint() {
  const el = $('calib');
  el.classList.toggle('hidden', !calib);
  if (calib) el.textContent = `Click pad ${PAD_ORDER[calib.i]} (${PAD_PLACE[PAD_ORDER[calib.i]]}). Esc cancels.`;
}
addEventListener('keydown', e => {
  if (e.key === 'c' || e.key === 'C') {
    if (e.shiftKey) { padAt = { ...PAD_DEFAULT }; savePads(); calib = null; }
    else calib = lastFace ? { i: 0, next: { ...padAt } } : null;
    calibHint();
  } else if (e.key === 'Escape' && calib) { calib = null; calibHint(); }
});
$('right').addEventListener('click', e => {
  if (!calib || !lastFace) return;
  const r = $('right').getBoundingClientRect(), k = r.width / W;
  const cx = (e.clientX - r.left) / k, cy = (e.clientY - r.top) / k;
  const { P, U, V } = faceOnPanel(lastFace);
  let lm = 0;
  P.forEach((p, i) => { if (Math.hypot(p[0] - cx, p[1] - cy) < Math.hypot(P[lm][0] - cx, P[lm][1] - cy)) lm = i; });
  const dx = cx - P[lm][0], dy = cy - P[lm][1], det = U[0] * V[1] - U[1] * V[0];     // solve dx,dy = a U + b V
  if (Math.abs(det) < 1e-6) return;
  const mid = (P[AXIS.left][0] + P[AXIS.right][0]) / 2;
  calib.next[PAD_ORDER[calib.i]] = { lm, a: (dx * V[1] - dy * V[0]) / det, b: (U[0] * dy - U[1] * dx) / det,
                                     left: cx < mid };
  if (++calib.i === PAD_ORDER.length) { padAt = calib.next; savePads(); calib = null; }
  calibHint();
});

function renderCamera(s) {
  const previewOn = performance.now() - lastPreview < PREVIEW_STALE_MS;
  const face = s.face;
  const mesh = $('mesh'), g = mesh.getContext('2d');
  if (!previewOn) {                  // no video: draw the cam.face mesh
    g.clearRect(0, 0, W, H);
    if (face) {
      const map = coverMap(face.size);
      g.fillStyle = 'rgba(201,209,217,.55)';
      for (const [x, y] of face.pts) { const [px, py] = map(x, y); g.fillRect(px - 1, py - 1, 2, 2); }
      lastFace = face;
    }
  }
  $('cam-empty').classList.toggle('hidden', previewOn || !!face);
  $('cam-empty').textContent = s.status.text.startsWith('Camera') ? 'Camera: no signal' : 'Waiting for the camera';

  const chip = $('chip');
  const label = !s.face_ok ? 'Face hidden' : !s.muscles ? 'Muscle sensors off' : s.emotion.chip;
  chip.querySelector('span').textContent = label;
  chip.querySelector('i').style.background = !s.face_ok || !s.muscles ? '#8b95a1' : css(TONE_COLOR[s.emotion.label] || '--calm');
  $('lips-off').classList.toggle('hidden', s.face_ok);

  const box = $('pads');             // content per snapshot, position per video frame
  if (box.children.length !== s.pads.length) {
    box.innerHTML = s.pads.map(() => `<div class="pad"><svg width="${W}" height="${H}"><line/></svg>` +
      '<span class="pdot"></span><span class="pill"><span class="pn"></span>' +
      `<span class="segs">${'<i></i>'.repeat(20)}</span></span></div>`).join('');
  }
  s.pads.forEach((p, i) => {
    const el = box.children[i], on = s.muscles && p.contact, lit = Math.round(p.level * 20);
    el.classList.toggle('on', on && p.level > 0.15);
    el.classList.toggle('off', !on);
    el.querySelector('.pn').textContent = p.name;
    [...el.querySelectorAll('.segs i')].forEach((seg, j) => seg.classList.toggle('lit', on && j < lit));
  });
  if (!previewOn) placeOverlays(face && s.face_ok ? face : lastFace);

  const a = s.austin;
  const heard = a && a.spoken ? a.spoken : '';
  speaking = !!(a && a.speaking);           // Sonic's first audio arrived and Twilio hasn't confirmed it played yet
  toneLabel = (a && a.emotion) || s.emotion.label;   // the emotion the sentence went out with
  $('heard').textContent = heard ? `“${heard}”` : '';
  $('tone').textContent = heard ? (a.tone || '') : '';
  $('tone').style.color = css(TONE_COLOR[toneLabel] || '--calm');
}

// a "voice playing" indicator, not a level meter (the call's audio isn't on the bus): it moves from Sonic's first
// audio until Twilio says the sentence was heard (timing.voice), flat otherwise
function wave(t) {
  const c = $('wave'), g = c.getContext('2d'), n = 64, active = speaking;
  g.clearRect(0, 0, c.width, c.height);
  g.fillStyle = css(TONE_COLOR[toneLabel] || '--calm');
  for (let i = 0; i < n; i++) {
    const env = Math.sin(Math.PI * (i + 0.5) / n);
    const h = active ? 8 + 100 * env * (0.35 + 0.65 * Math.abs(Math.sin(t / 140 + i * 0.7) * Math.sin(t / 410 + i * 1.3))) : 8;
    g.fillRect(i * (c.width / n) + 4, (c.height - h) / 2, 8, h);
  }
  requestAnimationFrame(wave);
}
requestAnimationFrame(wave);

// ---- end of call ----------------------------------------------------------------------------------------
const PLACEHOLDER_WER = { lips: 22, lips_muscles: 14, claude: 4 };    // design values until a session is scored
const TONES = { warm: 'Warm', concerned: 'Concerned', calm: 'Calm', surprised: 'Surprised', firm: 'Firm' };
const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
const use = (id, cls = '') => `<svg class="${cls}" viewBox="0 0 24 24" aria-hidden="true"><use href="#${id}"/></svg>`;
const liveIcon = k => document.querySelector(`.stage[data-k="${k}"] .ib svg`)?.outerHTML || '';
const ARCHIE = '<img src="/static/icons/cartesia-archie.svg" alt="">';
let endKey = null;

function renderEnd(s) {
  const m = s.summary, key = JSON.stringify(m);
  if (key === endKey) return;                 // the summary never changes once the call is over: build once
  endKey = key;
  const L = m.latency, num = v => v == null ? '--' : v;
  $('end-judge-who').textContent = s.channel === 'sms' ? 'Judge, by text' : 'Judge, heard by Cartesia Ink';
  $('end-call').textContent = `Call ended ${mmss(m.seconds)}${s.number ? ' on ' + s.number : ''}`;
  const [n, unit] = m.seconds < 120 ? [m.seconds, 'second'] : [Math.round(m.seconds / 60), 'minute'];
  $('end-head').innerHTML = `Austin held a <span class="num">${n}</span>-${unit} phone call without making a sound.`;

  const r = m.repairs[0];
  const stats = [
    ['s-mouthed', liveIcon('read'), 'Words mouthed', String(m.words), '', 'Mouthed silently, spoken in his cloned voice.'],
    ['s-lips', ARCHIE, 'Lips to voice', L.lips_to_voice == null ? '--' : (L.lips_to_voice / 1000).toFixed(2), 's',
     `Read ${num(L.read)}, lock ${num(L.lock)}, Claude ${num(L.claude)}, voice ${num(L.sonic)} ms.`],
    ['s-claude', use('claude-mark'), 'Fixed by Claude', String(m.repairs.length), '',
     !r ? 'No look-alikes needed fixing.' : m.repairs.length === 1 ? `“${r.frm}” to “${r.to}”, a lip look-alike.`
       : `${m.repairs.length} lip look-alikes, like “${r.frm}” to “${r.to}”.`],
    ['s-repeat', use('repeat-icon'), 'Asked to repeat', String(m.repeats.length), '',
     m.repeats.length ? 'Readers disagreed, so nothing was spoken.' : 'Every word was clear enough to speak.'],
  ];
  $('end-stats').innerHTML = stats.map(([cls, icon, label, value, unit, tip]) =>
    `<div class="stat ${cls}"><span class="wm-clip"><span class="wm">${cls === 's-lips' ? liveIcon('sonic') : icon}</span></span><div class="tip">${esc(tip)}</div>` +
    `<div class="top"><span class="ic">${icon}</span><span class="k">${label}</span></div>` +
    `<b class="num">${esc(value)}${unit && value !== '--' ? `<small>${unit}</small>` : ''}</b></div>`).join('');

  const b = m.benchmark, wer = b || PLACEHOLDER_WER;
  const lab = (b && b.labels) || {};
  const rows = [[lab.lips || 'Lips alone', wer.lips, '--camera'], [lab.lips_muscles || '+ muscles', wer.lips_muscles, '--emg'],
    [lab.claude || '+ Claude', wer.claude, '--claude']];
  const max = Math.max(...rows.map(x => x[1]), 1) * 1.14;
  $('end-ph').classList.toggle('hidden', !!b);
  $('end-wer-sub').textContent = `${b && b.session ? b.session : 'Session 3'}, scored against the script he mouthed.`;
  $('end-steps').classList.toggle('placeholder', !b);
  $('end-steps').innerHTML = rows.map(([name, v, col], i) => {
    const drop = i ? Math.round(rows[i - 1][1] - v) : 0, shown = b ? `${+v.toFixed(1)}%` : `[${v}%]`;
    return `<div class="step-row"><span class="sn">${name}</span><div class="st"><i style="--w:${(100 * v / max).toFixed(1)}%;` +
      `background:var(${col})"></i></div><b>${shown}</b><span class="drop">${drop > 0 ? drop + ' fewer' : ''}</span></div>`;
  }).join('');
  const pct = v => v == null ? '--' : v + '%';
  $('end-call-line').textContent = `This call: lips alone disagreed ${pct(m.disagree.lips)} of the time, muscles ${pct(m.disagree.muscles)}.`;

  renderTones(m.tones);
  $('end-latency').innerHTML = [['read', liveIcon('read'), 'Lips and muscles', L.read], ['lock', liveIcon('lock'), 'Locking', L.lock],
    ['claude', use('claude-mark'), L.claude_label, L.claude], ['sonic', ARCHIE, 'Cartesia Sonic', L.sonic]].map(([k, icon, label, ms]) =>
    `<div class="stage"><div class="ib${k === 'claude' ? ' claude' : ''}">${icon}</div><div><div class="sl">${esc(label)}</div>` +
    `<div class="sv"><b>${num(ms)}</b><span>ms</span></div></div></div>`).join('');

  renderThread(m.transcript);
  document.fonts.ready.then(() => { if (endKey === key) renderThread(m.transcript); });   // re-trim with final font metrics
  document.querySelectorAll('#end .num').forEach((el, k) => roll(el, k));
  setTimeout(() => document.querySelectorAll('#end .st i').forEach(x => x.classList.add('go')), reduceMotion ? 0 : 700);
}

// numbers roll up from 0 like an odometer; each lower digit spins one more loop before it lands
function roll(el, k) {
  const node = el.firstChild, text = node.textContent, h = parseFloat(getComputedStyle(el).lineHeight);
  const odo = document.createElement('span');
  odo.className = 'odo'; odo.style.setProperty('--h', h + 'px'); odo.setAttribute('aria-hidden', 'true');
  let n = 0;
  for (const c of text) {
    const cell = document.createElement('span');
    if (!/\d/.test(c)) { cell.textContent = c; odo.appendChild(cell); continue; }
    const loops = n++, target = 10 * loops + Number(c), col = document.createElement('span');
    col.className = 'col';
    col.innerHTML = Array.from({ length: target + 1 }, (_, i) => `<i>${i % 10}</i>`).join('');
    cell.appendChild(col); odo.appendChild(cell);
    const go = () => { col.style.transition = reduceMotion ? 'none' : `transform ${1.5 + 0.3 * loops}s cubic-bezier(.15, .7, .2, 1)`;
                       col.style.transform = `translateY(-${target * h}px)`; };
    reduceMotion ? go() : setTimeout(() => requestAnimationFrame(go), 500 + 160 * k);
  }
  const sr = document.createElement('span'); sr.className = 'sr'; sr.textContent = text;
  el.replaceChild(odo, node); el.insertBefore(sr, odo);
}


function renderTones(tones) {
  const R = 56, C = 2 * Math.PI * R, entries = Object.entries(tones || {}), total = entries.reduce((a, [, c]) => a + c, 0);
  let off = 0;
  $('end-donut').innerHTML = total ? entries.map(([t, c]) => {
    const len = C * c / total, seg = `<circle cx="72" cy="72" r="${R}" class="seg" data-name="${t}" ` +
      `style="stroke:var(--tone-${t});stroke-dasharray:${Math.max(0, len - 4).toFixed(1)} ${C.toFixed(1)};stroke-dashoffset:${(-off).toFixed(1)}"/>`;
    off += len; return seg;
  }).join('') : `<circle cx="72" cy="72" r="${R}" class="seg empty"/>`;
  $('end-legend').innerHTML = total ? entries.map(([t, c]) =>
    `<div class="lg" data-name="${t}"><i style="background:var(--tone-${t})"></i><span>${TONES[t] || t}</span><b>${c}</b></div>`).join('')
    : '<p class="lg-empty">No tone data on this call.</p>';
  const donut = document.querySelector('#end .donut'), legend = $('end-legend');
  const show = name => {
    const seg = name && donut.querySelector(`.seg[data-name="${name}"]`);
    donut.classList.toggle('hot', !!seg);
    legend.classList.toggle('hot', !!seg);
    donut.querySelectorAll('.seg').forEach(x => x.classList.toggle('on', x === seg));
    legend.querySelectorAll('.lg').forEach(x => x.classList.toggle('on', !!seg && x.dataset.name === name));
  };
  document.querySelectorAll('#end .seg, #end .lg').forEach(el => {
    el.addEventListener('mouseenter', () => show(el.dataset.name));
    el.addEventListener('mouseleave', () => show(null));
  });
  show(null);
}

// the call as a chat; a long call keeps its newest messages and counts the rest
function renderThread(turns) {
  const bare = w => w.toLowerCase().replace(/[.,!?;:"“”]/g, '');
  const msgs = turns.map(x => {
    if (x.who === 'judge') return `<div class="msg judge"><div class="bubble">${esc(x.text)}</div></div>`;
    const ws = x.text.split(' ').map(esc);
    for (const r of x.repairs || [])
      if (r.index < ws.length && bare(ws[r.index]) === bare(r.to)) ws[r.index] = `<s>${esc(r.frm)}</s> ${ws[r.index]}`;
    const notes = [
      (x.repairs || []).length ? `<div class="cnote claude">${use('claude-mark')}Claude fixed a look-alike</div>` : '',
      x.repeat ? `<div class="cnote repeat">${use('repeat-icon')}Asked to repeat, nothing wrong was said</div>` : '',
      x.face_hidden ? `<div class="cnote face">${liveIcon('read')}Face hidden, muscles carried it</div>` : ''].join('');
    return `<div class="msg austin${x.repeat ? ' unspoken' : ''}"><div class="bubble">${ws.join(' ')}${x.repeat ? ' …' : ''}</div>${notes}</div>`;
  });
  const thread = $('end-thread');
  let from = 0;
  const fill = () => {
    thread.innerHTML = `<span class="sep">${from ? `${from} earlier message${from === 1 ? '' : 's'}` : 'Call started'}</span>` +
      msgs.slice(from).join('') + '<span class="sep">Call ended</span>';
  };
  fill();
  while (from < msgs.length - 1 && thread.scrollHeight > thread.clientHeight + 1) { from++; fill(); }
}
