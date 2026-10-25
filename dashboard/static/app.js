// Reclaim live dashboard: draws each snapshot from dashboard/server.py (dashboard/state.py) as Thai's Figma v5.
// Text frames are snapshots (JSON); binary frames are the lip reader's camera mirror (cam.preview JPEG).
'use strict';
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
const SRC = { camera: 'camera', emg: 'emg', claude: 'claude', lips: 'camera', muscles: 'emg' };
const TONE_COLOR = { happy: '--warm', sad: '--concerned', neutral: '--calm', surprised: '--surprised', angry: '--firm' };
// MediaPipe landmark each pad sits on (HARDWARE.md: A0 under the chin, A1 left chin corner, A2 right cheek below
// the mouth, A3 left cheek above the mouth; "left" is Austin's left, the image's right), and which way its label goes.
const PAD_AT = { A0: { lm: 152, dy: 0.035, flip: false }, A1: { lm: 400, dy: 0, flip: false },
                 A2: { lm: 202, dy: 0, flip: true }, A3: { lm: 425, dy: 0, flip: false } };
const MOUTH = { left: 61, right: 291, top: 13, bottom: 14 };
const W = 760, H = 1080, PREVIEW_STALE_MS = 2000;
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

let snap = null, lastPreview = -Infinity, speaking = false, toneLabel = 'neutral';

// ---- the 1920 x 1080 stage, scaled to the window -------------------------------------------------------
function fit() {
  const k = Math.min(innerWidth / 1920, innerHeight / 1080);
  $('stage').style.transform = `translate(-50%, -50%) scale(${k})`;
}
addEventListener('resize', fit); fit();

// ---- connection ------------------------------------------------------------------------------------------
function connect() {
  const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
  ws.binaryType = 'blob';
  ws.onmessage = e => {
    if (typeof e.data === 'string') { snap = JSON.parse(e.data); render(snap); return; }
    const img = $('cam'), old = img.src;
    img.src = URL.createObjectURL(e.data); lastPreview = performance.now();
    if (old.startsWith('blob:')) URL.revokeObjectURL(old);
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

function renderCamera(s) {
  const previewOn = performance.now() - lastPreview < PREVIEW_STALE_MS;
  $('cam').classList.toggle('hidden', !previewOn);
  const face = s.face, map = coverMap(face && face.size);
  const mesh = $('mesh'), g = mesh.getContext('2d');
  g.clearRect(0, 0, W, H);
  if (!previewOn && face) {          // no mirrored video yet: the face mesh from cam.face
    g.fillStyle = 'rgba(201,209,217,.55)';
    for (const [x, y] of face.pts) { const [px, py] = map(x, y); g.fillRect(px - 1, py - 1, 2, 2); }
  }
  $('cam-empty').classList.toggle('hidden', previewOn || !!face);
  $('cam-empty').textContent = s.status.text.startsWith('Camera') ? 'Camera: no signal' : 'Waiting for the camera';

  const chip = $('chip');
  const label = !s.face_ok ? 'Face hidden' : !s.muscles ? 'Muscle sensors off' : s.emotion.chip;
  chip.querySelector('span').textContent = label;
  chip.querySelector('i').style.background = !s.face_ok || !s.muscles ? '#8b95a1' : css(TONE_COLOR[s.emotion.label] || '--calm');
  $('lips-off').classList.toggle('hidden', s.face_ok);

  const ring = $('mouth-ring');
  if (face && s.face_ok) {
    const p = i => map(face.pts[i][0], face.pts[i][1]);
    const [lx, ly] = p(MOUTH.left), [rx, ry] = p(MOUTH.right), [, ty] = p(MOUTH.top), [, by] = p(MOUTH.bottom);
    const w = Math.abs(rx - lx) * 1.9, h = w * 82 / 183;
    Object.assign(ring.style, { width: w + 'px', height: h + 'px', left: (lx + rx) / 2 - w / 2 + 'px', top: (ty + by) / 2 - h / 2 + 'px' });
    ring.classList.remove('hidden');
  } else ring.classList.add('hidden');

  const placed = [];                 // label rows already taken: a pad that would overlap one moves up a row
  $('pads').innerHTML = s.pads.map(p => {
    const at = PAD_AT[p.pin];
    let x, y;
    if (face) { [x, y] = map(face.pts[at.lm][0], face.pts[at.lm][1] + at.dy); }
    else { [x, y] = { A0: [378, 672], A1: [489, 616], A2: [270, 616], A3: [489, 516] }[p.pin]; }
    y = Math.min(y, 820);              // clear of "Heard on the call"
    const flip = at.flip ? x - 300 > 0 : x + 300 > W;      // the label goes to whichever side has room
    const span = flip ? [x - 300, x + 12] : [x - 12, x + 300];
    while (placed.some(q => Math.abs(q.y - y) < 38 && q.span[0] < span[1] && span[0] < q.span[1])) y -= 40;
    placed.push({ y, span });
    const lit = Math.round(p.level * 20), on = s.muscles && p.contact;
    const segs = Array.from({ length: 20 }, (_, i) => `<i class="${on && i < lit ? 'lit' : ''}"></i>`).join('');
    const cls = `pad${flip ? ' flip' : ''}${on ? (p.level > 0.15 ? ' on' : '') : ' off'}`;
    const style = flip ? `right:${W - x - 12}px;top:${y - 15}px` : `left:${x - 12}px;top:${y - 15}px`;
    return `<div class="${cls}" style="${style}"><span class="pdot"></span><span class="pill">${esc(p.name)}<span class="segs">${segs}</span></span></div>`;
  }).join('');

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
  const rows = [['Lips alone', wer.lips, '--camera'], ['+ muscles', wer.lips_muscles, '--emg'], ['+ Claude', wer.claude, '--claude']];
  const max = Math.max(...rows.map(x => x[1]), 1) * 1.14;
  $('end-ph').classList.toggle('hidden', !!b);
  $('end-wer-sub').textContent = `${b && b.session ? b.session : 'Session 3'}, scored against the script he mouthed.`;
  $('end-steps').classList.toggle('placeholder', !b);
  $('end-steps').innerHTML = rows.map(([name, v, col], i) => {
    const drop = i ? Math.round(rows[i - 1][1] - v) : 0, shown = b ? `${Math.round(v)}%` : `[${v}%]`;
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
