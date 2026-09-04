'use strict';

const TEAM = ['#6b7684', '#e5484d', '#3b82f6', '#eab308', '#d946ef', '#22c55e', '#f97316'];
const CART = ['#4fd1c5', '#f0b429', '#d946ef', '#57d38c', '#f2555a', '#8b9dff'];
const $ = (id) => document.getElementById(id);

let LIVE = null;
let ORACLE = null;

function fit(canvas, cssHeight) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || canvas.parentElement.clientWidth;
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

  const bandH = 4, bandTop = h - pad.b + 3;
  for (let c = 0; c < j; c++) {
    for (let i = 0; i < series.length; i++) {
      const ctrl = (series[i].control_team || [])[c];
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
  if (!spectrum || !spectrum.length) return noData(canvas, 'J absent — no spectrum');
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

function num(v, digits) {
  if (v === null || v === undefined || Number.isNaN(v)) return '<span class="absentval">absent</span>';
  return typeof v === 'number' ? v.toFixed(digits === undefined ? 3 : digits) : String(v);
}

function renderHierarchy(latest) {
  const body = $('hier').querySelector('tbody');
  if (!latest) { body.innerHTML = '<tr><td colspan="9" class="dimmed">no frame</td></tr>'; return; }
  const succ = {};
  (latest.SUCC || []).forEach(([team, denial]) => { succ[team] = denial; });
  const carts = latest.control_team || [];
  const rows = (latest.resources || []).map((r) => {
    const controlled = carts.filter((c) => c === r.team).length;
    const pw = latest.PW === r.team;
    return `<tr>
      <td style="color:${TEAM[r.team % TEAM.length]}">team ${r.team}</td>
      <td class="num">${r.alive}/${r.players}</td>
      <td class="num">${num(r.health, 0)}</td>
      <td class="num">${num(r.armor, 0)}</td>
      <td class="num">${Object.entries(r.ammo || {}).map(([k, v]) => `${k}:${num(v, 0)}`).join(' ')}</td>
      <td class="num">${Object.entries(r.weapon_words || {}).map(([k, v]) => `${k}:0x${Number(v).toString(16)}`).join(' ')}</td>
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
    const diagRow = diag && Array.isArray(diag[i]) ? diag[i] : diag;
    const dk = diagRow && diagRow[a.action] !== undefined ? diagRow[a.action] : null;
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
      <td class="num">${Number(Boolean(a.target_resolved))}</td>
      <td>${goal}</td>
      <td class="num">${Number(Boolean(a.goal_match))}</td>
      <td class="num">${num(a.goal_distance)}</td>
      <td class="num">${Number(Boolean(a.target_touch))}</td>
      <td class="num">${num(a.target_logp)}</td>
      <td class="num">${dk === null || dk === undefined ? '<span class="absentval">absent</span>' : num(dk, 4)}</td>
    </tr>`;
  }).join('');
}

function renderJMeasures(report) {
  const lens = report && report.j_lens;
  const oracle = report && report.j_oracle;
  const lensBody = $('jlens').querySelector('tbody');
  const oracleBody = $('joracle').querySelector('tbody');
  const matrixFusionBody = $('matrixfusionintervention').querySelector('tbody');
  if (!lens) {
    lensBody.innerHTML = '<tr><td colspan="4" class="dimmed">no composer rows</td></tr>';
    $('jlensmeta').textContent = '— mass 0';
  } else {
    const coordinateStratum = [...(lens.coordinate_strata || [])].reverse().find(stratum =>
      JSON.stringify(stratum.input_labels || []) === JSON.stringify(lens.input_labels || [])
    ) || {};
    const featureProjection = coordinateStratum.j_to_source_feature_affine_projection || {};
    const coordinates = (coordinateStratum.input_labels || lens.input_labels || []).map((name, index) => {
      const covariance = (coordinateStratum.cross_covariance || lens.cross_covariance || [])[index] || [];
      const norm = Math.sqrt(covariance.reduce((sum, value) => sum + value * value, 0));
      return `<tr><td>${name}</td><td class="num">${num((coordinateStratum.input_variance || lens.input_variance || [])[index], 6)}</td><td class="num">${num(norm, 6)}</td><td class="num">${num((featureProjection.residual_mean_square || [])[index], 6)}</td></tr>`;
    });
    const stateStratum = [...((oracle && oracle.source_state_strata) || [])].reverse()[0] || {};
    const stateProjection = stateStratum.j_to_authoritative_state_affine_projection || {};
    const states = (stateStratum.state_labels || []).map((name, index) => {
      const covariance = (stateStratum.state_j_covariance || [])[index] || [];
      const norm = Math.sqrt(covariance.reduce((sum, value) => sum + value * value, 0));
      return `<tr><td>state.${name}</td><td class="num">${num((stateStratum.state_variance || [])[index], 6)}</td><td class="num">${num(norm, 6)}</td><td class="num">${num((stateProjection.residual_mean_square || [])[index], 6)}</td></tr>`;
    });
    const families = Object.entries(lens.composer_measures || {}).map(([name, measure]) =>
      `<tr class="dimmed"><td>composer.${name} <span class="dimmed">mass ${measure.mass}</span></td><td class="num">${num(measure.variance, 6)}</td><td class="num">—</td><td class="num">—</td></tr>`
    );
    lensBody.innerHTML = [...coordinates, ...states, ...families].join('');
    const window = report.observation_window || {};
    $('jlensmeta').textContent = `— mass ${lens.mass}, ${(coordinateStratum.input_labels || []).length} feature + ${(stateStratum.state_labels || []).length} state coordinates × ${lens.j_integral.length} J coordinates; retained ${window.retained_coordinate_row_mass || lens.mass}/${window.ingested_coordinate_row_mass || lens.mass}`;
  }
  const matrixFusion = lens && lens.matrix_fusion_intervention;
  if (!matrixFusion) {
    matrixFusionBody.innerHTML = '<tr><td colspan="5" class="dimmed">no paired matrix-fusion frames</td></tr>';
    $('matrixfusionmeta').textContent = '— frame mass 0';
  } else {
    matrixFusionBody.innerHTML = Object.entries(matrixFusion.measures || {}).map(([name, measure]) =>
      `<tr><td>${name}</td><td class="num">${measure.mass}</td><td class="num">${num(measure.integral, 6)}</td><td class="num">${num(measure.mean, 6)}</td><td class="num">${num(measure.variance, 6)}</td></tr>`
    ).join('');
    $('matrixfusionmeta').textContent = `— ${matrixFusion.left} − ${matrixFusion.right}, ${matrixFusion.frame_mass} identical-state frames`;
  }
  if (!oracle) {
    oracleBody.innerHTML = '<tr><td colspan="7" class="dimmed">no joined outcomes</td></tr>';
    $('joraclemeta').textContent = '— mass 0';
  } else {
    const oracleResiduals = new Map();
    for (const stratum of oracle.outcome_affine_projection_strata || []) {
      const projection = stratum.j_to_outcome_affine_projection || {};
      (stratum.outcome_labels || []).forEach((name, index) =>
        oracleResiduals.set(`${stratum.policy_arm}.${stratum.channel}.${name}`, (projection.residual_mean_square || [])[index])
      );
    }
    for (const stratum of oracle.state_delta_affine_projection_strata || []) {
      const projection = stratum.j_to_state_delta_affine_projection || {};
      (stratum.state_labels || []).forEach((name, index) =>
        oracleResiduals.set(`${stratum.policy_arm}.Δstate.${stratum.channel}.${name}`, (projection.residual_mean_square || [])[index])
      );
    }
    const armMeasures = Object.entries(oracle.policy_arm_measures || {}).flatMap(([arm, armMeasure]) => [
      ...Object.entries(armMeasure.outcome_measures || {}).map(([name, measure]) => [`${arm}.${name}`, measure]),
      ...Object.entries(armMeasure.state_delta_measures || {}).map(([name, measure]) => [`${arm}.Δstate.${name}`, measure]),
    ]);
    const measures = [
      ...Object.entries(oracle.outcome_measures || {}),
      ...Object.entries(oracle.state_delta_measures || {}).map(([name, measure]) => [`Δstate.${name}`, measure]),
      ...armMeasures,
    ];
    const successorStratum = [...(oracle.successor_state_strata || [])].reverse()[0] || {};
    const successorProjection = successorStratum.j_to_authoritative_successor_state_affine_projection || {};
    const successorRows = (successorStratum.state_labels || []).map((name, index) => {
      const covariance = (successorStratum.state_j_covariance || [])[index] || [];
      const norm = Math.sqrt(covariance.reduce((sum, value) => sum + value * value, 0));
      return `<tr><td>S′.${successorStratum.channel}.${name}</td><td class="num">${successorStratum.mass}</td><td class="num">${num((successorStratum.state_integral || [])[index], 6)}</td><td class="num">${num((successorStratum.state_mean || [])[index], 6)}</td><td class="num">${num((successorStratum.state_variance || [])[index], 6)}</td><td class="num">${num(norm, 6)}</td><td class="num">${num((successorProjection.residual_mean_square || [])[index], 6)}</td></tr>`;
    });
    const measureRows = measures.map(([name, measure]) => {
      const norm = Math.sqrt((measure.j_covariance || []).reduce((sum, value) => sum + value * value, 0));
      return `<tr><td>${name}</td><td class="num">${measure.mass}</td><td class="num">${num(measure.integral, 6)}</td><td class="num">${num(measure.mean, 6)}</td><td class="num">${num(measure.variance, 6)}</td><td class="num">${num(norm, 6)}</td><td class="num">${num(oracleResiduals.get(name), 6)}</td></tr>`;
    });
    oracleBody.innerHTML = [...successorRows, ...measureRows].join('') || '<tr><td colspan="7" class="dimmed">no joined outcomes</td></tr>';
    const referenceMass = Object.entries(oracle.state_reference_measures || {}).map(([name, measure]) => `${name} ${measure.joined_mass}/${measure.mass}`).join(', ');
    $('joraclemeta').textContent = `— ${referenceMass || `delivered ${oracle.delivery_joined_mass}/${oracle.delivery_mass}`}, successor states ${oracle.successor_state_atom_mass || 0}, active-route outcomes ${oracle.applied_joined_mass}/${oracle.applied_mass}, events ${oracle.event_joined_mass || 0}/${oracle.event_mass || 0}, categorical transition atoms ${oracle.state_categorical_transition_atom_mass || 0} over ${oracle.state_categorical_transition_coordinates || 0} source→target coordinates`;
  }
}

function renderGeometry(report, internals) {
  const g = (report && report.geometry) || null;
  const kv = $('geom');
  if (!g) { kv.innerHTML = '<div><span>window</span><span class="dimmed">empty</span></div>'; return; }
  const items = [
    ['J width', g.j_width],
    ['J singular mass', g.j_spectral_measure && g.j_spectral_measure.mass],
    ['J eff. rank', g.j_spectral_measure && g.j_spectral_measure.effective_rank],
    ['input x width', g.x_width],
    ['x eff. rank', g.x_spectral_measure && g.x_spectral_measure.effective_rank],
    ['x nonzero cols', `${g.x_nonzero_columns} <span class="dimmed">/ ${g.x_width}</span>`],
    ['beta width', g.beta_width === null ? '<span class="absentval">absent</span>' : g.beta_width],
    ['beta eff. rank', g.beta_spectral_measure ? g.beta_spectral_measure.effective_rank : '<span class="absentval">absent</span>'],
    ['rows in window', g.rows],
    ['ticks in window', g.ticks],
  ];
  if (internals && internals.j_stats) {
    items.push(['J finite row mass (this tick)', internals.j_stats.finite_row_mass]);
    items.push(['J std', internals.j_stats.std]);
  }
  if (internals) {
    const behind = internals.ticks_behind;
    items.push(['internals from', internals.resp_id === undefined || internals.resp_id === null
      ? '<span class="absentval">no sampled frame</span>'
      : `resp ${internals.resp_id}` + (behind ? ` <span class="warn">(${behind} ticks back)</span>` : ' <span class="ok">(this tick)</span>')]);
  }
  kv.innerHTML = items.map(([k, v]) => `<div><span>${k}</span><span>${v === null || v === undefined ? '<span class="dimmed">—</span>' : v}</span></div>`).join('');
}

function renderFieldMeasures(report) {
  const body = $('fieldmeasures').querySelector('tbody');
  if (!report || !report.frame_mass) {
    body.innerHTML = '<tr><td colspan="6" class="dimmed">frame mass 0</td></tr>';
    return;
  }
  body.innerHTML = report.fields.map((f) => `<tr>
      <td title="${f.label.replace(/"/g, '')}">${f.path}</td>
      <td class="num">${f.value_mass}/${f.key_mass}</td>
      <td class="num dimmed">${f.shape ? f.shape.join('×') : '—'}</td>
      <td class="num">${f.finite_mass}/${f.coordinate_mass}</td>
      <td class="num">${f.nonzero_mass}/${f.coordinate_mass}</td>
      <td class="dimmed">${f.owner}</td>
    </tr>`).join('');
}

function renderXBlocks(fieldReport) {
  const box = $('xblocks');
  const blocks = (fieldReport && fieldReport.x_blocks) || [];
  if (!blocks.length) {
    box.innerHTML = '<div class="block"><span>model.x</span><b>mass 0</b></div>';
    $('xnote').textContent = 'model.x coordinate mass 0';
    return;
  }
  box.innerHTML = blocks.map((b) => `<div class="block">
      <span>${b.label}<br><span class="dimmed">x[${b.cols[0]}:${b.cols[1]}]</span></span>
      <b>${b.nonzero_columns}/${b.observed_width}</b>
    </div>`).join('');
  const coordinates = blocks.reduce((sum, block) => sum + block.coordinate_mass, 0);
  const finite = blocks.reduce((sum, block) => sum + block.finite_mass, 0);
  const nonzero = blocks.reduce((sum, block) => sum + block.nonzero_mass, 0);
  $('xnote').textContent = `coordinate mass ${coordinates}, finite mass ${finite}, nonzero mass ${nonzero}`;
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
    ['game value measures', internals.game_value ? JSON.stringify(internals.game_value).slice(0, 80) : '<span class="absentval">absent</span>'],
  ];
  const u = internals.update || {};
  ['loss', 'loss_pg', 'loss_w', 'loss_l', 'importance_mean', 'updates'].forEach((k) => {
    if (u[k] !== undefined && u[k] !== null) items.push([k, typeof u[k] === 'number' ? u[k].toFixed(4) : u[k]]);
  });
  kv.innerHTML = items.map(([k, v]) => `<div><span>${k}</span><span>${v}</span></div>`).join('');
}

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
    renderFieldMeasures(LIVE.field_measures);
    renderXBlocks(LIVE.field_measures);
    heatmap($('irmap'), LIVE.internals && LIVE.internals.j, { empty: 'model.j absent from this frame' });
    heatmap($('coupling'), LIVE.internals && LIVE.internals.coupling, { empty: 'model.coupling absent from this frame' });
    drawSpectrum(LIVE.internals && LIVE.internals.j_stats && LIVE.internals.j_stats.spectrum);
    renderGeometry(ORACLE && ORACLE.report, LIVE.internals);
  } catch (exc) {
    $('tap').className = 'pill crit';
    $('tap').innerHTML = `<i class="dot"></i><b>viewer offline</b> <span class="dimmed">${exc.message}</span>`;
  }
}

async function tickOracle() {
  try {
    ORACLE = await pull('/api/joracle');
    renderJMeasures(ORACLE.report);
    renderGeometry(ORACLE.report, LIVE && LIVE.internals);
  } catch (exc) {
    $('joraclemeta').textContent = `— endpoint error: ${exc.message}`;
  }
}

tickLive(); tickOracle();
setInterval(tickLive, 1000);
setInterval(tickOracle, 3000);
window.addEventListener('resize', () => { if (LIVE) tickLive(); });
