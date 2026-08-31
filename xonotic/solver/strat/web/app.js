'use strict';

/* j-oracle viewer — hand-rolled, zero dependencies, no CDN.
   Everything here is a read of /api/*; nothing is written back. */

const TEAM = ['#6b7684', '#e5484d', '#3b82f6', '#eab308', '#d946ef', '#22c55e', '#f97316'];
const CART = ['#4fd1c5', '#f0b429', '#d946ef', '#57d38c', '#f2555a', '#8b9dff'];
const $ = (id) => document.getElementById(id);

let LIVE = null;
let ORACLE = null;

/* ---------------------------------------------------------------- canvas */

function fit(canvas, cssHeight) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || canvas.parentElement.clientWidth;
  /* The intended CSS height is remembered on first sight: after the first
     resize canvas.height holds device pixels, and reading it back each frame
     would double the element on every repaint. */
  if (!canvas.dataset.h) canvas.dataset.h = String(cssHeight || canvas.height);
  const h = Number(canvas.dataset.h);
  canvas.width = Math.max(1, Math.round(w * dpr));
  canvas.height = Math.max(1, Math.round(h * dpr));
  canvas.style.height = h + 'px';
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h };
}

function axis(ctx, w, h, pad, yMin, yMax, label) {
  ctx.strokeStyle = '#1f2933';
  ctx.fillStyle = '#4b5a6a';
  ctx.font = '10px ui-monospace, Menlo, monospace';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad.t + (h - pad.t - pad.b) * (i / 4);
    ctx.beginPath();
    ctx.moveTo(pad.l, Math.round(y) + 0.5);
    ctx.lineTo(w - pad.r, Math.round(y) + 0.5);
    ctx.stroke();
    const v = yMax - (yMax - yMin) * (i / 4);
    ctx.fillText(v.toFixed(2), 4, y + 3);
  }
  if (label) { ctx.fillStyle = '#4b5a6a'; ctx.fillText(label, pad.l + 2, pad.t - 4); }
}

function noData(canvas, message) {
  const { ctx, w, h } = fit(canvas);
  ctx.fillStyle = '#4b5a6a';
  ctx.font = '11px ui-monospace, Menlo, monospace';
  ctx.fillText(message, 8, h / 2);
}

/* -------------------------------------------------------------- behavior */

function drawDepth(series) {
  const canvas = $('depth');
  if (!series.length) return noData(canvas, 'no telemetry frames yet');
  const { ctx, w, h } = fit(canvas);
  const pad = { l: 34, r: 8, t: 14, b: 16 };
  const j = Math.max(...series.map((s) => (s.depth || []).length), 0);
  if (!j) return noData(canvas, 'frames carry no cart rows');
  axis(ctx, w, h, pad, 0, 1, 'cart depth (0..1)');
  const px = (i) => pad.l + (w - pad.l - pad.r) * (series.length < 2 ? 0 : i / (series.length - 1));
  const py = (v) => pad.t + (h - pad.t - pad.b) * (1 - Math.max(0, Math.min(1, v)));

  /* control bands: a strip per cart, coloured by the controlling team */
  const bandH = 4, bandTop = h - pad.b + 3;
  for (let c = 0; c < j; c++) {
    for (let i = 0; i < series.length; i++) {
      const ctrl = (series[i].ctrl || [])[c];
      if (ctrl === undefined || ctrl === null) continue;
      ctx.fillStyle = ctrl > 0 ? TEAM[ctrl % TEAM.length] : '#232c36';
      const x0 = px(i), x1 = i + 1 < series.length ? px(i + 1) : x0 + 2;
      ctx.fillRect(x0, bandTop + c * (bandH + 1) - (j * (bandH + 1)) - 2, Math.max(1, x1 - x0), bandH);
    }
  }
  for (let c = 0; c < j; c++) {
    ctx.strokeStyle = CART[c % CART.length];
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    let started = false;
    series.forEach((s, i) => {
      const v = (s.depth || [])[c];
      if (v === undefined || v === null) { started = false; return; }
      if (!started) { ctx.moveTo(px(i), py(v)); started = true; } else ctx.lineTo(px(i), py(v));
    });
    ctx.stroke();
  }
  /* epoch boundaries — where the responder or the server restarted */
  ctx.strokeStyle = '#5c1f22';
  ctx.setLineDash([3, 3]);
  for (let i = 1; i < series.length; i++) {
    if (series[i].epoch !== series[i - 1].epoch || (series[i].resp_id || 0) < (series[i - 1].resp_id || 0)) {
      ctx.beginPath(); ctx.moveTo(px(i), pad.t); ctx.lineTo(px(i), h - pad.b); ctx.stroke();
    }
  }
  ctx.setLineDash([]);
  const legend = [];
  for (let c = 0; c < j; c++) legend.push(`<span style="color:${CART[c % CART.length]}">&#9644; cart ${c}</span>`);
  legend.push('<span class="dimmed">strip under the axis = controlling team; dashed red = stream restart</span>');
  $('pwlegend').innerHTML = legend.join(' &nbsp; ');
}

function drawPW(series) {
  const canvas = $('pw');
  if (!series.length) return noData(canvas, '');
  const { ctx, w, h } = fit(canvas);
  const pad = { l: 34, r: 8 };
  const px = (i) => pad.l + (w - pad.l - pad.r) * (series.length < 2 ? 0 : i / (series.length - 1));
  let flips = 0;
  for (let i = 0; i < series.length; i++) {
    const pw = series[i].PW;
    const x0 = px(i), x1 = i + 1 < series.length ? px(i + 1) : x0 + 2;
    ctx.fillStyle = (pw === null || pw === undefined) ? '#232c36' : (pw > 0 ? TEAM[pw % TEAM.length] : '#232c36');
    ctx.fillRect(x0, 14, Math.max(1, x1 - x0), h - 20);
    if (i && series[i].PW !== series[i - 1].PW) {
      flips++;
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(Math.round(x0), 10, 1, h - 12);
    }
  }
  ctx.fillStyle = '#78899b';
  ctx.font = '10px ui-monospace, Menlo, monospace';
  ctx.fillText('PW', 6, 24);
  ctx.fillText(`${flips} flip${flips === 1 ? '' : 's'} in window`, pad.l + 4, 10);
}

function drawFocus(focus, k) {
  const canvas = $('focus');
  if (!focus || !focus.length) return noData(canvas, 'strategy_focus absent from this frame');
  const { ctx, w, h } = fit(canvas);
  const n = focus.length;
  const pad = 26;
  const cell = Math.min((w - pad - 8) / n, (h - pad - 8) / n);
  let max = 0;
  focus.forEach((r) => r.forEach((v) => { if (v > max) max = v; }));
  ctx.font = '10px ui-monospace, Menlo, monospace';
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      const v = focus[r][c] || 0;
      const a = max ? v / max : 0;
      ctx.fillStyle = r === c ? '#141a21' : `rgba(79,209,197,${0.06 + 0.9 * a})`;
      ctx.fillRect(pad + c * cell, pad + r * cell, cell - 1, cell - 1);
      if (v) {
        ctx.fillStyle = a > 0.5 ? '#04110f' : '#d8e0e9';
        ctx.fillText(String(v), pad + c * cell + cell / 2 - 3, pad + r * cell + cell / 2 + 3);
      }
    }
    ctx.fillStyle = TEAM[(r + 1) % TEAM.length];
    ctx.fillText('t' + (r + 1), 4, pad + r * cell + cell / 2 + 3);
    ctx.fillText('t' + (r + 1), pad + r * cell + cell / 2 - 6, 16);
  }
  const total = focus.reduce((s, row) => s + row.reduce((a, b) => a + b, 0), 0);
  $('focusnote').textContent = total
    ? `${total} cross-team assignments this tick`
    : 'zero cross-team assignments this tick — nobody is being hunted or suppressed';
}

/* ------------------------------------------------------------- heatmaps */

function heatmap(canvas, matrix, opts) {
  opts = opts || {};
  if (!matrix || !matrix.length) return noData(canvas, opts.empty || 'absent from this frame');
  const rows = matrix.length, cols = matrix[0].length;
  const { ctx, w, h } = fit(canvas);
  let lo = Infinity, hi = -Infinity;
  for (const row of matrix) for (const v of row) { if (v === null) continue; if (v < lo) lo = v; if (v > hi) hi = v; }
  if (!isFinite(lo)) return noData(canvas, 'all values non-finite');
  const span = Math.max(Math.abs(lo), Math.abs(hi)) || 1;
  const cw = w / cols, ch = (h - 12) / rows;
  const img = ctx.createImageData(Math.max(1, Math.round(w)), Math.max(1, Math.round(h - 12)));
  for (let y = 0; y < img.height; y++) {
    const r = Math.min(rows - 1, Math.floor(y / (img.height / rows)));
    for (let x = 0; x < img.width; x++) {
      const c = Math.min(cols - 1, Math.floor(x / (img.width / cols)));
      const v = matrix[r][c];
      const t = v === null || v === undefined ? 0 : Math.max(-1, Math.min(1, v / span));
      const i = 4 * (y * img.width + x);
      /* diverging: teal negative, amber positive, near-black at zero */
      img.data[i]     = t > 0 ? 20 + 220 * t : 12 + 60 * -t;
      img.data[i + 1] = t > 0 ? 20 + 160 * t : 20 + 190 * -t;
      img.data[i + 2] = t > 0 ? 26 + 40 * t : 30 + 180 * -t;
      img.data[i + 3] = 255;
    }
  }
  const scratch = document.createElement('canvas');
  scratch.width = img.width; scratch.height = img.height;
  scratch.getContext('2d').putImageData(img, 0, 0);
  const dpr = window.devicePixelRatio || 1;
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(scratch, 0, 0, img.width / dpr, img.height / dpr);
  ctx.fillStyle = '#4b5a6a';
  ctx.font = '10px ui-monospace, Menlo, monospace';
  ctx.fillText(`${rows}×${cols}   min ${lo.toFixed(3)}   max ${hi.toFixed(3)}`, 2, h - 2);
  void cw; void ch;
}

function drawSpectrum(spectrum) {
  const canvas = $('spectrum');
  if (!spectrum || !spectrum.length) return noData(canvas, 'IR absent — no spectrum');
  const { ctx, w, h } = fit(canvas);
  const pad = { l: 4, r: 4, t: 6, b: 12 };
  const n = spectrum.length;
  const bw = (w - pad.l - pad.r) / n;
  for (let i = 0; i < n; i++) {
    const v = Math.max(0, Math.min(1, spectrum[i]));
    const bh = (h - pad.t - pad.b) * v;
    ctx.fillStyle = v > 0.05 ? '#4fd1c5' : '#243039';
    ctx.fillRect(pad.l + i * bw, h - pad.b - bh, Math.max(1, bw - 1), bh);
  }
  ctx.fillStyle = '#4b5a6a';
  ctx.font = '10px ui-monospace, Menlo, monospace';
  ctx.fillText(`${n} leading singular values`, 4, h - 2);
}

/* ---------------------------------------------------------------- tables */

function num(v, digits) {
  if (v === null || v === undefined || Number.isNaN(v)) return '<span class="absentval">absent</span>';
  return typeof v === 'number' ? v.toFixed(digits === undefined ? 3 : digits) : String(v);
}

function renderHierarchy(latest) {
  const body = $('hier').querySelector('tbody');
  if (!latest) { body.innerHTML = '<tr><td colspan="9" class="dimmed">no frame</td></tr>'; return; }
  const succ = {};
  (latest.SUCC || []).forEach(([team, denial]) => { succ[team] = denial; });
  const carts = latest.ctrl || [];
  const rows = (latest.resources || []).map((r) => {
    const controlled = carts.filter((c) => c === r.team).length;
    const pw = latest.PW === r.team;
    return `<tr>
      <td style="color:${TEAM[r.team % TEAM.length]}">team ${r.team}</td>
      <td class="num">${r.alive}/${r.players}</td>
      <td class="num">${num(r.health, 0)}</td>
      <td class="num">${num(r.armor, 0)}</td>
      <td class="num">${num(r.ammo, 1)}</td>
      <td class="num">${r.weapon_slots}</td>
      <td class="num">${controlled}</td>
      <td class="num">${succ[r.team] === undefined ? '<span class="dimmed">—</span>' : num(succ[r.team], 3)}</td>
      <td>${pw ? '<span class="ok">PW</span>' : '<span class="dimmed">·</span>'}</td>
    </tr>`;
  });
  body.innerHTML = rows.join('') || '<tr><td colspan="9" class="absentval">resources absent from this frame</td></tr>';
}

function renderAssignments(internals) {
  const body = $('assign').querySelector('tbody');
  const rows = (internals && internals.assignments) || [];
  if (!rows.length) { body.innerHTML = '<tr><td colspan="21" class="absentval">assignments absent from this frame</td></tr>'; return; }
  const diag = internals.diag_k;
  body.innerHTML = rows.map((a, i) => {
    const dk = diag && diag[i] ? diag[i][a.action] : null;
    const applied = `${a.applied_kind || '?'}:${a.applied_subject}`;
    const goal = `${a.goal_kind || '?'}:${a.goal_subject}`;
    return `<tr>
      <td class="num">${a.edict}</td>
      <td style="color:${TEAM[a.team % TEAM.length]}">t${a.team}</td>
      <td class="dimmed">${a.controller === 'human' ? 'HUMAN' : 'bot'}</td>
      <td class="${a.behavior === 'policy' ? '' : 'warn'}">${a.behavior}</td>
      <td>${a.kind}</td>
      <td class="num">${a.subject}</td>
      <td class="num">${a.target}</td>
      <td class="num">${num(a.gain)}</td>
      <td class="num">${num(a.commit)}</td>
      <td class="num">${num(a.spawn)}</td>
      <td class="num">${num(a.target_winner, 0)}</td>
      <td class="num">${num(a.target_rank)}</td>
      <td class="num">${num(a.target_nimber)}</td>
      <td class="num">${num(a.target_denial)}</td>
      <td>${applied}</td>
      <td class="${a.target_resolved ? 'ok' : 'crit'}">${a.target_resolved ? 'resolved' : 'unresolved'}</td>
      <td>${goal}</td>
      <td class="${a.goal_match ? 'ok' : 'dimmed'}">${a.goal_match ? 'match' : 'other'}</td>
      <td class="num">${num(a.goal_distance)}</td>
      <td class="${a.target_touch ? 'ok' : 'dimmed'}">${a.target_touch ? 'touch' : '·'}</td>
      <td class="num">${num(a.target_logp)}</td>
      <td class="num">${dk === null || dk === undefined ? '<span class="absentval">absent</span>' : num(dk, 4)}</td>
    </tr>`;
  }).join('');
}

function cell(value, best) {
  if (value === null || value === undefined) return '<td class="dimmed">—</td>';
  if (typeof value === 'object') {
    const v = `${value.acc.toFixed(3)}<span class="dimmed"> / maj ${value.majority.toFixed(3)}</span>`;
    return `<td class="num ${best ? 'beat' : ''}">${v}</td>`;
  }
  return `<td class="num ${best ? 'beat' : ''}">${value.toFixed(4)}</td>`;
}

function renderProbe(report) {
  const body = $('probe').querySelector('tbody');
  if (!report || !report.available) {
    body.innerHTML = `<tr><td colspan="7" class="dimmed">${(report && report.reason) || 'no probe yet'}</td></tr>`;
    $('probemeta').textContent = '— ' + ((report && report.reason) || 'accumulating');
    $('verdict').textContent = '';
    return;
  }
  const rows = [];
  const push = (r) => {
    const beat = r.control_ok && r.delta_vs_randproj !== null && r.delta_vs_randproj > 0.05 && !r.tautological;
    const bad = r.delta_vs_randproj !== null && !r.control_ok;
    rows.push(`<tr class="${r.tautological ? 'tautological' : ''}${bad ? ' degenerate' : ''}" title="${r.note.replace(/"/g, '')}">
      <td>${r.target}${bad ? ' <span class="crit">control failed</span>' : ''}</td>
      ${cell(r.ir, beat)}${cell(r.randproj)}${cell(r.shuffled)}${cell(r.raw_x)}
      <td class="num ${beat ? 'beat' : 'nobeat'}">${r.delta_vs_randproj === null ? '—' : r.delta_vs_randproj.toFixed(4)}</td>
      <td class="num dimmed">${r.n_finite === undefined ? '' : r.n_finite}</td>
    </tr>`);
  };
  rows.push('<tr><td colspan="7" class="dimmed" style="padding-top:8px">regression &nbsp;—&nbsp; R&#178; on the held-out ticks</td></tr>');
  report.regression.forEach(push);
  rows.push('<tr><td colspan="7" class="dimmed" style="padding-top:8px">classification &nbsp;—&nbsp; accuracy / majority baseline</td></tr>');
  report.classification.forEach(push);
  body.innerHTML = rows.join('');

  const v = report.verdict;
  const klass = v.shuffled_label_control_passes ? (v.n_beats ? 'ok' : 'crit') : 'crit';
  $('verdict').innerHTML =
    `<span class="${klass}">${v.reading}</span>` +
    (v.degenerate_targets && v.degenerate_targets.length
      ? `<br><span class="warn">${v.degenerate_targets.length}/${v.scored_targets} targets failed their own shuffled-label control and are not read: ${v.degenerate_targets.join(', ')}</span>`
      : '') +
    (v.beats_random_projection.length
      ? ' &nbsp;<span class="dimmed">(' + v.beats_random_projection.map((b) => `${b.target} +${b.delta}`).join(', ') + ')</span>'
      : '') +
    `<br><span class="dimmed">shuffled-label control: worst |R&#178;| ${v.worst_shuffled_r2}; ` +
    `${v.shuffled_label_control_passes ? 'most targets at chance — those probes are honest' : 'too many targets degenerate on this window'}</span>`;
  const m = report.method;
  $('method').innerHTML =
    `${m.estimator} · split ${m.split} · train ${m.train_rows} / test ${m.test_rows} rows<br>` +
    `<span class="warn">control not measured here:</span> ${m.control_not_available}`;
  $('probemeta').textContent = `— ${report.geometry.rows} rows over ${report.geometry.ticks} ticks`;
}

function renderGeometry(report, internals) {
  const g = (report && report.geometry) || null;
  const kv = $('geom');
  if (!g) { kv.innerHTML = '<div><span>window</span><span class="dimmed">empty</span></div>'; return; }
  const irNarrow = g.ir_width < g.spec_ir_width_floor;
  const items = [
    ['IR width', `<span class="${irNarrow ? 'crit' : 'ok'}">${g.ir_width}</span> <span class="dimmed">/ spec &#8805;${g.spec_ir_width_floor}</span>`],
    ['IR rank (window)', g.ir_rank],
    ['IR eff. rank', g.ir_effective_rank],
    ['input x width', g.x_width],
    ['input x rank', `<span class="${g.x_rank !== null && g.x_rank <= 5 ? 'crit' : ''}">${g.x_rank}</span>`],
    ['x nonzero cols', `<span class="${g.x_nonzero_columns <= 8 ? 'crit' : ''}">${g.x_nonzero_columns}</span> <span class="dimmed">/ ${g.x_width}</span>`],
    ['beta width', g.beta_width === null ? '<span class="absentval">absent</span>' : g.beta_width],
    ['beta rank', g.beta_rank === null ? '<span class="absentval">absent</span>' : g.beta_rank],
    ['rows in window', g.rows],
    ['ticks in window', g.ticks],
  ];
  if (internals && internals.ir_stats) {
    items.push(['IR rank (this tick)', internals.ir_stats.frame_rank]);
    items.push(['IR std', internals.ir_stats.std]);
  }
  if (internals) {
    const behind = internals.ticks_behind;
    items.push(['internals from', internals.resp_id === undefined || internals.resp_id === null
      ? '<span class="absentval">no sampled frame</span>'
      : `resp ${internals.resp_id}` + (behind ? ` <span class="warn">(${behind} ticks back)</span>` : ' <span class="ok">(this tick)</span>')]);
  }
  kv.innerHTML = items.map(([k, v]) => `<div><span>${k}</span><span>${v === null || v === undefined ? '<span class="dimmed">—</span>' : v}</span></div>`).join('');
}

function renderAlarms(report) {
  const box = $('alarms');
  const p = report && report.pathology;
  if (!p) { box.innerHTML = '<div class="clear dimmed">waiting for the first probe window…</div>'; return; }
  if (p.clear) {
    box.innerHTML = '<div class="clear">&#10003; no rank collapse, no zeroed input block, probes honest, IR beats its controls somewhere non-tautological.</div>';
    return;
  }
  box.innerHTML = p.alarms.map((a) =>
    `<div class="alarm ${a.severity}"><span class="tag">${a.severity}</span>
     <div class="body">${a.text}<br><code>${a.id}</code></div></div>`).join('');
}

function renderAudit(auditReport) {
  const body = $('audit').querySelector('tbody');
  if (!auditReport || !auditReport.available) {
    body.innerHTML = '<tr><td colspan="5" class="dimmed">no frame to audit</td></tr>';
    return;
  }
  body.innerHTML = auditReport.fields.map((f) => `<tr>
      <td title="${f.why.replace(/"/g, '')}">${f.path}</td>
      <td><span class="chip ${f.status}">${f.status}</span></td>
      <td class="num dimmed">${f.shape ? f.shape.join('×') : '—'}</td>
      <td class="num dimmed">${f.status === 'present' || f.status === 'all_zero' ? (100 * f.nonzero_fraction).toFixed(0) + '%' : '—'}</td>
      <td class="dimmed">${f.owner}</td>
    </tr>`).join('');
}

function renderXBlocks(auditReport) {
  const box = $('xblocks');
  const blocks = (auditReport && auditReport.x_blocks) || [];
  if (!blocks.length) {
    box.innerHTML = '<div class="block absent"><span>model.x</span><b class="crit">absent</b></div>';
    $('xnote').textContent = 'the model input is not in the stream — the R19 question cannot even be asked from here.';
    return;
  }
  box.innerHTML = blocks.map((b) => `<div class="block ${b.status}">
      <span>${b.label}<br><span class="dimmed">x[${b.cols[0]}:${b.cols[1]}]</span></span>
      <b class="${b.status === 'present' ? 'ok' : 'crit'}">${b.nonzero_cols}/${b.width}</b>
    </div>`).join('');
  const dead = blocks.filter((b) => b.per_player && b.status !== 'present');
  $('xnote').innerHTML = dead.length
    ? `<span class="crit">${dead.length} per-player block(s) all-zero: ${dead.map((d) => d.label).join(', ')}.</span> This is the AGENDA E9 / R19 condition — the policy is not integrating the state SPEC &#167;3 requires.`
    : '<span class="ok">every per-player resource block carries signal</span> — health, armor, ammo, position, velocity and the weapon bitset are all entering the matmul.';
}

function renderValues(internals) {
  const kv = $('values');
  if (!internals || !internals.available) { kv.innerHTML = ''; return; }
  const stat = (arr) => {
    if (!arr || !arr.length) return null;
    const finite = arr.filter((v) => typeof v === 'number');
    if (!finite.length) return null;
    const mean = finite.reduce((a, b) => a + b, 0) / finite.length;
    return { mean, min: Math.min(...finite), max: Math.max(...finite) };
  };
  const W = stat(internals.winner_value), L = stat(internals.loser_value);
  const items = [
    ['value W mean', W ? W.mean.toFixed(4) : '<span class="absentval">absent</span>'],
    ['value W range', W ? `${W.min.toFixed(3)} … ${W.max.toFixed(3)}` : '—'],
    ['value L mean', L ? L.mean.toFixed(4) : '<span class="absentval">absent</span>'],
    ['value L range', L ? `${L.min.toFixed(3)} … ${L.max.toFixed(3)}` : '—'],
    ['W &#8722; L mean', W && L ? (W.mean - L.mean).toFixed(4) : '—'],
    ['advantage', internals.advantage === null || internals.advantage === undefined
      ? '<span class="absentval">absent — see design/joracle-viewer.md hook 2</span>'
      : Number(internals.advantage).toFixed(4)],
    ['diag(K)', internals.diag_k ? 'present' : '<span class="absentval">absent — hook 1</span>'],
    ['game value', internals.game_value ? JSON.stringify(internals.game_value).slice(0, 40) : '<span class="absentval">absent — B11 unresolved</span>'],
  ];
  const u = internals.update || {};
  ['loss', 'loss_pg', 'loss_w', 'loss_l', 'importance_mean', 'updates'].forEach((k) => {
    if (u[k] !== undefined && u[k] !== null) items.push([k, typeof u[k] === 'number' ? u[k].toFixed(4) : u[k]]);
  });
  kv.innerHTML = items.map(([k, v]) => `<div><span>${k}</span><span>${v}</span></div>`).join('');
}

/* ----------------------------------------------------------------- chrome */

function renderHeader(live) {
  const f = live.follower;
  const stale = f.seconds_since_frame;
  let klass = 'crit', text = f.state;
  if (f.state === 'attached' && stale !== null && stale < 6) { klass = 'ok'; text = 'live'; }
  else if (f.state === 'attached') { klass = 'warn'; text = 'attached, no frames'; }
  else if (f.state === 'reattaching') { klass = 'warn'; text = 'reattaching'; }
  $('tap').className = 'pill ' + klass;
  $('tap').innerHTML = `<i class="dot ${klass === 'ok' ? 'live' : ''}"></i><b>${text}</b>
    <span class="dimmed">${f.frames_buffered} frames · ${stale === null ? 'never' : stale + 's ago'} · epoch ${f.epochs}${f.resp_id_resets ? ' · ' + f.resp_id_resets + ' restarts' : ''}</span>`;

  const s = live.series.length ? live.series[live.series.length - 1] : null;
  $('match').innerHTML = s
    ? `<b>k=${s.k}</b> teams · <b>j=${s.j}</b> carts · <b>l=${s.l}</b> players · ${s.mode}${s.updates ? ' · ' + s.updates + ' updates' : ''}`
    : `<span class="dimmed">${live.demo.telemetry}</span>`;
  $('connect').innerHTML = live.demo.connect
    ? `xonotic: <b>connect ${live.demo.connect}</b>${live.demo.map ? ' <span class="dimmed">· ' + live.demo.map + '</span>' : ''}`
    : '<span class="dimmed">client address not configured</span>';
  $('clock').textContent = new Date().toLocaleTimeString();
}

/* -------------------------------------------------------------- polling */

async function pull(url) {
  const response = await fetch(url, { cache: 'no-store' });
  if (!response.ok) throw new Error(url + ' -> ' + response.status);
  return response.json();
}

async function tickLive() {
  try {
    LIVE = await pull('/api/live');
    renderHeader(LIVE);
    drawDepth(LIVE.series);
    drawPW(LIVE.series);
    const latest = LIVE.series.length ? LIVE.series[LIVE.series.length - 1] : null;
    drawFocus(latest && latest.focus, latest && latest.k);
    renderHierarchy(latest);
    renderAssignments(LIVE.internals);
    renderValues(LIVE.internals);
    renderAudit(LIVE.audit);
    renderXBlocks(LIVE.audit);
    heatmap($('irmap'), LIVE.internals && LIVE.internals.ir, { empty: 'model.ir absent from this frame' });
    heatmap($('gram'), LIVE.internals && LIVE.internals.gram, { empty: 'model.gram absent from this frame' });
    drawSpectrum(LIVE.internals && LIVE.internals.ir_stats && LIVE.internals.ir_stats.spectrum);
    renderGeometry(ORACLE && ORACLE.report, LIVE.internals);
  } catch (exc) {
    $('tap').className = 'pill crit';
    $('tap').innerHTML = `<i class="dot"></i><b>viewer offline</b> <span class="dimmed">${exc.message}</span>`;
  }
}

async function tickOracle() {
  try {
    ORACLE = await pull('/api/joracle');
    renderProbe(ORACLE.report);
    renderAlarms(ORACLE.report);
    renderGeometry(ORACLE.report, LIVE && LIVE.internals);
  } catch (exc) {
    $('verdict').innerHTML = `<span class="crit">probe endpoint unreachable: ${exc.message}</span>`;
  }
}

tickLive(); tickOracle();
setInterval(tickLive, 1000);
setInterval(tickOracle, 3000);
window.addEventListener('resize', () => { if (LIVE) tickLive(); });
