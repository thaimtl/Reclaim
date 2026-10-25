// Reclaim live dashboard: draws each snapshot from dashboard/server.py (dashboard/state.py) as Thai's Figma v5.
// Text frames are snapshots (JSON); binary frames are the lip reader's camera mirror (cam.preview JPEG) with the
// face points of that same video frame (dashboard/server.py pack_frame).
'use strict';
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
const SRC = { camera: 'camera', emg: 'emg', claude: 'claude', lips: 'camera', muscles: 'emg' };
const TONE_COLOR = { happy: '--warm', sad: '--concerned', neutral: '--calm', surprised: '--surprised', angry: '--firm' };
// Where each pad sits (HARDWARE.md: A0 under the chin, A1 left chin corner, A2 right cheek below the mouth, A3 left
// cheek above the mouth; "left" is Austin's left, the image's right): a MediaPipe landmark plus a and b times his
// face's own axes (down: nose bridge to chin, across: cheek to cheek), so it scales and turns with his head. If the
// pads went on somewhere else this session, press C on the dashboard and click them (saved in this browser).
// measured on Austin's session-2 video at his electrode pairs' centres (they land within a few px on other clips)
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
let frame = null, framePending = false, lastFace = null;    // the newest video frame and the face it was drawn with

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

// ---- live screen -----------------------------------------------------------------------------------------
const mmss = s => `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;

function render(s) {
  $('live').classList.toggle('hidden', s.mode === 'end');
  $('end').classList.toggle('hidden', s.mode !== 'end');
  if (s.mode === 'end') { renderEnd(s); return; }
  const live = s.mode === 'live';
  $('live-dot').classList.toggle('off', !live);
  $('title').textContent = live ? 'Live call with the judge' : 'Waiting for the call';
  $('sub').textContent = live ? `${mmss(s.elapsed)}${s.number ? ' on ' + s.number : ''}` : (s.number || '');
  $('m-lips').textContent = s.metrics.lips_to_voice == null ? '--' : (s.metrics.lips_to_voice / 1000).toFixed(2) + ' s';
  $('m-fixed').textContent = s.metrics.fixed;
  $('m-repeats').textContent = s.metrics.repeats;
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
// A frame is decoded off the main thread (createImageBitmap) and drawn at the next screen refresh together with
// the pads placed on that frame's own face points, so the pads can't trail the video. No smoothing on top: MediaPipe's
// video mode already steadies its points (measured on his session-2 videos: 0.44 px of frame-to-frame jitter on this
// 760 px panel), and a One Euro filter on top only left the pads up to 6 px behind his face when he moved.
let frameSeq = 0;
async function onFrame(buf) {
  const seq = ++frameSeq;
  try {
    const dv = new DataView(buf);
    if (dv.getUint32(0) !== 0x52434631) return;           // 'RCF1'
    const n = dv.getUint32(4), head = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, n)));
    const bitmap = await createImageBitmap(new Blob([new Uint8Array(buf, 8 + n)], { type: 'image/jpeg' }));
    if (frame && frame.seq > seq) { bitmap.close(); return; }   // a newer frame finished decoding first
    if (frame) frame.bitmap.close();
    frame = { bitmap, seq, ...head };
    lastPreview = performance.now();
    if (!framePending) { framePending = true; requestAnimationFrame(drawFrame); }
  } catch (err) { /* a bad frame is skipped; the next one replaces it */ }
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

// the face's points on the panel, and its own axes there
function faceOnPanel(face) {
  const map = coverMap(face.size), P = face.pts.map(([x, y]) => map(x, y));
  const U = [P[AXIS.chin][0] - P[AXIS.top][0], P[AXIS.chin][1] - P[AXIS.top][1]];
  const V = [P[AXIS.right][0] - P[AXIS.left][0], P[AXIS.right][1] - P[AXIS.left][1]];
  return { P, U, V };
}

function placeOverlays(face) {
  const ring = $('mouth-ring'), pads = [...$('pads').children];
  if (!face || !snap || !snap.face_ok) {          // no face: the pads wait at their usual places
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

// Callouts: each dot sits on its electrode; its label sits in a column beside his face (each pad has its side, set
// where it was clicked), in the dots' order top to bottom, joined to it by a line. Labels never overlap each other
// or jump sides; on a close-up with no room beside his face, they keep to the panel's edges.
const GAP = 28, ROW = 40, LABEL_MAX_Y = 820;          // label rows stay clear of "Heard on the call"
function layoutCallouts(pads, dots, faceLeft, faceRight) {
  for (const left of [true, false]) {
    const idx = dots.map((d, i) => i).filter(i => pads[i] && (padAt[PAD_ORDER[i]].left !== false) === left)
      .sort((a, b) => dots[a][1] - dots[b][1]);
    let prev = -Infinity;
    const ys = idx.map(i => (prev = Math.max(dots[i][1], prev + ROW)));
    const over = ys.length ? Math.max(0, ys[ys.length - 1] - LABEL_MAX_Y) : 0;   // too low: the column moves up
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

// ---- pad calibration: press C, then click A0, A1, A2, A3 where they are on his face -------------------------
let padAt = loadPads(), calib = null;
function loadPads() {
  try { return { ...PAD_DEFAULT, ...JSON.parse(localStorage.getItem('reclaim.pads') || '{}') }; }
  catch { return { ...PAD_DEFAULT }; }
}
function savePads() {
  try { localStorage.setItem('reclaim.pads', JSON.stringify(padAt)); } catch { /* this session keeps them */ }
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
  const dx = cx - P[lm][0], dy = cy - P[lm][1], det = U[0] * V[1] - U[1] * V[0];     // dx,dy = a U + b V
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
  if (!previewOn) {                  // no mirrored video: the face mesh from cam.face, and the pads on it
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

  const box = $('pads');             // the pads' content per snapshot; where they sit is set per video frame
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

// ---- end of call -----------------------------------------------------------------------------------------
function renderEnd(s) {
  const m = s.summary;
  $('end-sub').textContent = `Call ended ${mmss(m.seconds)}${s.number ? ' on ' + s.number : ''}`;
  const dur = m.seconds < 60 ? `${m.seconds} seconds` : `${Math.floor(m.seconds / 60)} min ${m.seconds % 60} s`;
  $('end-head').textContent = `Austin held a conversation for ${dur} without making a sound.`;
  const pct = v => v == null ? '--' : v + '%';
  const claudePct = m.words ? Math.round(100 * m.repairs.length / m.words) : null;
  const rows = [['Lips alone', m.disagree.lips, '--camera', ''], ['Muscles alone', m.disagree.muscles, '--emg', ''],
                ['Words Claude changed', claudePct, '--claude', ' claude']];
  $('end-bars').innerHTML = rows.map(([l, v, col, cls]) => `<div class="brow${cls}"><span class="bl">${l}</span>` +
    `<div class="bt"><i style="width:${v || 0}%;background:var(${col})"></i></div><span class="bv">${pct(v)}</span></div>`).join('');
  $('end-bars-note').textContent = m.disagree.lips == null
    ? 'Measured live, on this call. Each reader’s own pick isn’t on the bus yet, so its share shows as --.'
    : 'Measured live, on this call: there is no answer key during a call, so this counts disagreement, not errors.';
  const L = m.latency, parts = [['read', 'Lips and muscles read it', '--camera'], ['lock', 'Locks', '--text-muted'],
    ['claude', L.claude_label, '--claude'], ['sonic', 'Cartesia Sonic starts speaking', '--emg']];
  const total = parts.reduce((a, [k]) => a + (L[k] || 0), 0) || 1;
  $('end-lat-label').textContent = `Where the ${L.lips_to_voice == null ? '' : (L.lips_to_voice / 1000).toFixed(2) + ' s '}from lips to voice goes, median per word, before the phone network`;
  $('end-split').innerHTML = parts.map(([k, , c]) => `<i style="flex:${L[k] || 0.0001};background:var(${c})"></i>`).join('');
  $('end-split-labels').innerHTML = parts.map(([k, lab]) => `<div style="flex:${L[k] || 0.0001}"><b>${L[k] == null ? '--' : L[k] + ' ms'}</b>${lab}</div>`).join('');
  const ex = m.repairs.slice(0, 2).map(r => `${r.frm} to ${r.to}`).join(', ');
  $('end-hl').innerHTML = [
    [m.repairs.length, `look-alikes fixed by Claude${ex ? ': ' + esc(ex) : ''}`, '--claude'],
    [m.repeats.length, 'Please repeat, instead of guessing', '--repeat'],
    [m.face_hidden_s + ' s', `with the face hidden; the muscles carried ${m.emg_words} words`, '--emg'],
    [m.wpm == null ? '--' : m.wpm, 'words per minute, mouthed silently', '--text-primary'],
  ].map(([v, l, c]) => `<div class="hl"><b style="color:var(${c})">${v}</b><span>${l}</span></div>`).join('');
  $('end-transcript').innerHTML = m.transcript.slice(-12).map(x => {
    if (x.who === 'judge') return `<div class="turn"><div class="who"><span>Judge</span></div><div class="tx">${esc(x.text)}</div></div>`;
    const ws = x.text.split(' ').map(esc), bare = w => w.toLowerCase().replace(/[.,!?;:"“”]/g, '');
    for (const r of x.repairs)      // the repaired word at its own position, only if it is still that word
      if (r.index < ws.length && bare(ws[r.index]) === bare(r.to)) ws[r.index] = `<s>${esc(r.frm)}</s> ${ws[r.index]}`;
    const tx = ws.join(' ');
    const an = [...x.repairs.map(r => `<div class="an claude">Claude fixed ${esc(r.frm)} to ${esc(r.to)}</div>`),
      x.repeat ? `<div class="an repeat">Please repeat on “${esc(x.repeat.words[0] || '')}”</div>` : '',
      x.face_hidden ? '<div class="an face">Face hidden: muscles only</div>' : ''].join('');
    const tone = x.emotion === 'neutral' ? '' : (x.tone || '');
    return `<div class="turn austin"><div class="who"><span>Austin</span><span style="color:var(${TONE_COLOR[x.emotion] || '--calm'})">${tone}</span></div><div class="tx">${tx}</div>${an}</div>`;
  }).join('');
}
