"""python -m route COMMAND [--config route.toml] [--set section.key=value ...]: every result a JSON record a line.

  train       one rank's training: loop.train, the data flow (design/dataflow.png); a telemetry record every
              report.interval; rank 0's evaluations at evaluation.at
  evaluate    the measured cluster's table for run.theta_in (or the initial theta): each mode, kind and size
  simcheck    tests/switch.c's fixed router on the measured cluster (us): the simulator against switch.c's own table
  cases       the distribution's draws: their rate, and what they are (state sources, fabrics, windows, streams)
  dump        every refresh's tokens of one collective simulated from a zero state, the CPU forward (forward.c, as
              switch.c's router) answering: tests/switch.c ROUTE_DUMP's format (--set dump.*: see DUMP)
  compare     two dumps, refresh by refresh: the first features that differ
  deploy      the CPU forward against the GPU's on the first refresh of the 12 MB all-reduce
  check       the GPU's credit against split.h's on the CPU, scores against central differences (--graph: the int8
              network against the fp32 reference)
  bench       one forward and one learning step of the training's sizes, kernel by kernel
  xbench      the exchange alone (world > 1): N rounds (0: declare and close, a dead run's programs flushed)
  stat        telemetry records of tables (FILE...) where they lie
  cluster     python -m route cluster {build|train|xbench|stat} ...: every host of cluster.hosts (cluster.py)"""
import json
import os
import sys
import threading
import time

import numpy as np

from . import cases as C
from . import config, fabric, loop, native, telemetry
from .native import const, dtype, ffi

DUMP = {'kind': 1, 'size': 12582912, 'burst': 0, 'out': '/tmp/sim-all.bin', 'params': ''}
MODES = [('policy', 0, 0), ('policy (burst)', 0, 1), ('spt', 2, 0), ('uniform', 3, 0), ('switch.c fixed', 4, 0)]


def emit(rec):
    sys.stdout.write(json.dumps(rec) + '\n'); sys.stdout.flush()


def args(argv):
    cfgpath, sets, rest = None, [], []
    i = 0
    while i < len(argv):
        if argv[i] == '--config':
            cfgpath = argv[i + 1]; i += 2
        elif argv[i] == '--set':
            sets.append(argv[i + 1]); i += 2
        else:
            rest.append(argv[i]); i += 1
    return config.load(cfgpath, sets), rest, sets


def network(cfg, lib, runs, refreshes, rows):
    r = cfg['run']
    n = lib.route_net_new(r['metallib'].encode(), runs, refreshes, rows, r['theta_seed'], r['theta_in'].encode() if r['theta_in'] else ffi.NULL)
    if n == ffi.NULL:
        emit({'record': 'failure', 'rank': r['rank'], 'what': ffi.string(lib.route_net_failure()).decode()}); sys.exit(1)
    return n


def evaluation(cfg, lib, learner, measured, label, modes=MODES):
    """the measured cluster: each mode, kind and size (cases.measured), under the learner's theta"""
    c, rows = C.measured(measured, [(m, b) for _, m, b in modes])
    ev = network(cfg, lib, 4 * len(c), 1, 4 * len(c) * loop.CX)
    T = loop.evaluate(lib, learner, ev, c)
    lib.route_net_free(ev)
    table = {}
    for (name, m, b) in modes:
        for k, kind in enumerate(C.KINDS):
            for s in C.SIZES:
                i = rows.index((m, b, k, s)); t = float(T[i].mean())
                floor = lib.route_op_floor(ffi.from_buffer('struct route_fabric *', measured), k, float(s), int(measured['compute'][0]))
                table.setdefault(name, {}).setdefault(kind, {})[s] = {'over_floor': round(t, 4), 'us': round(t * floor / 1e3, 2)}
    emit({'record': 'evaluation', 'label': label, 'table': table})
    return table


def train(cfg, lib):
    r, cap = cfg['run'], cfg['capacity']
    measured = fabric.load(r['fabric'])
    dist = C.Distribution(config.flat(cfg), measured, lib, np.random.default_rng([r['data_seed'], r['rank']]))
    lib.route_tel_open(r['telemetry'].encode(), r['rank'], r['world']); lib.route_tel_reach(lib.RT_PHASE_START)
    runs = cap['forward_rows'] // loop.CX // 4 * 4   # the runs in flight: one forward can carry all of them
    learner = network(cfg, lib, cap['learn_runs'], cap['refreshes'], cap['forward_rows'])
    actor = network(cfg, lib, runs, 1, runs * loop.CX)
    lib.route_tel_reach(lib.RT_PHASE_NETWORKS)
    if r['world'] > 1:
        if lib.route_net_join(learner, r['rank'], r['world'], r['mesh_id'].encode()) != 0:
            emit({'record': 'failure', 'rank': r['rank'], 'what': ffi.string(lib.route_exchange_failure()).decode()}); sys.exit(1)
        lib.route_tel_reach(lib.RT_PHASE_JOINED)
    evaluating = r['rank'] in cfg['evaluation']['ranks']
    if evaluating and 'start' in cfg['evaluation']['at']:
        evaluation(cfg, lib, learner, measured, 'start')
    table = telemetry.Table(r['telemetry']); table.sample()
    done = threading.Event()
    last, lastops = {}, {}
    def report():   # the telemetry table's records (the table is the store; this only reads it); the loop's phases, s/s
        while not done.wait(cfg['report']['interval']):
            rec = table.sample()
            if rec:
                rec['pool'] = dist.pool.count
                now = dict(loop.PHASES); rec['phases'] = {k: round((now[k] - last.get(k, 0.0)) / cfg['report']['interval'], 3) for k in now}; last.update(now)
                ops = dict(loop.OPS); rec['int8_tops'] = {k: round((ops[k] - lastops.get(k, 0)) / cfg['report']['interval'] / 1e12, 4) for k in ops}; lastops.update(ops)
                emit(rec)
    threading.Thread(target=report, daemon=True).start()
    loop.train(cfg, lib, dist, learner, actor, r['steps'], r['world'])
    done.set(); lib.route_tel_reach(lib.RT_PHASE_DONE)
    failure = lib.route_net_failure()
    if failure != ffi.NULL:
        emit({'record': 'failure', 'rank': r['rank'], 'what': ffi.string(failure).decode()})
    if evaluating and 'end' in cfg['evaluation']['at']:
        evaluation(cfg, lib, learner, measured, f'end, {lib.route_net_steps(learner)} steps')
    if r['rank'] == 0 and (r['theta'] or r['theta_int8']):
        lib.route_net_save(learner, r['theta'].encode() if r['theta'] else ffi.NULL, r['theta_int8'].encode() if r['theta_int8'] else ffi.NULL)
        emit({'record': 'saved', 'theta': r['theta'], 'theta_int8': r['theta_int8']})
    if r['world'] > 1:
        lib.route_exchange_close()


def simcheck(cfg, lib):
    """tests/switch.c's fixed router on the measured cluster, us (no network involved: mode 4)"""
    measured = fabric.load(cfg['run']['fabric'])
    learner = network(cfg, lib, 4, 1, loop.CX)
    t = evaluation(cfg, lib, learner, measured, 'switch.c fixed', [('switch.c fixed', 4, 0)])
    emit({'record': 'simcheck', 'us': {k: {s: v['us'] for s, v in t['switch.c fixed'][k].items()} for k in C.KINDS}})


def describe(cfg, lib, n=4096):
    r = cfg['run']; measured = fabric.load(r['fabric'])
    rng = np.random.default_rng([r['data_seed'], r['rank']])
    t0 = time.perf_counter(); dist = C.Distribution(config.flat(cfg), measured, lib, rng); t1 = time.perf_counter()
    dist.pool.add(rng.standard_normal((64, C.D2)).astype(np.float32))   # a pool with states in it, so its third draws from it
    c, source = dist.draw(n); t2 = time.perf_counter()
    emit({'record': 'cases', 'cases': n, 'cases_per_second': round(n / (t2 - t1), 1), 'fabric_pool_seconds': round(t1 - t0, 3),
          'state_source': {k: round(float((source == v).mean()), 4) for k, v in (('zero', 0), ('gaussian', 1), ('pool', 2))},
          'state_rms_by_source': {k: round(float(np.sqrt((c['h'][source == v] ** 2).mean())) if (source == v).any() else 0, 4) for k, v in (('zero', 0), ('gaussian', 1), ('pool', 2))},
          'fabric_nodes': {int(k): int(v) for k, v in zip(*np.unique(c['f']['nodes'], return_counts=True))},
          'fabric_ranks': {int(k): int(v) for k, v in zip(*np.unique(c['f']['ncompute'], return_counts=True))},
          'ops_mean': round(float(c['ops'].mean()), 2), 'ops_max': int(c['ops'].max()), 'under_way_mean': round(float((c['op']['done'] > 0).sum(1).mean()), 3),
          'refreshes': {int(k): int(v) for k, v in zip(*np.unique(c['w']['points'], return_counts=True))},
          'dt_us_median': round(float(np.median(c['w']['dt'])) / 1e3, 1), 'burst': round(float(c['w']['burst'].mean()), 3), 'warm_spt': round(float((c['w']['warm'] == 2).mean()), 3),
          'background_lanes': round(float((c['background'] > 0).sum(1).mean() / np.maximum(c['f']['lanes'], 1).mean()), 3)})


def stream_one(lib, f, kind, size):
    s = ffi.new('struct route_stream *'); lib.route_stream_one(ffi.from_buffer('struct route_fabric *', f), kind, size, int(f['compute'][0]), s)
    return s


def dump(cfg, lib, sets):
    d = dict(DUMP)
    for s in sets:
        k, v = s.split('=', 1); d[k] = type(DUMP[k])(v)
    measured = fabric.load(cfg['run']['fabric']); lib.route_argmax = 1 if cfg['run']['argmax'] else 0
    p = ffi.new('struct route_params *')
    if lib.route_params_read(p, d['params'].encode()) != 0:
        emit({'record': 'failure', 'what': f"cannot read the int8 weights {d['params']}"}); sys.exit(1)
    s = stream_one(lib, measured, d['kind'], float(d['size']))
    w = ffi.new('struct route_window *', {'t0': 0, 'dt': 100e3, 'lag': 100e3, 'points': 0, 'warm': 2, 'burst': d['burst']})
    sim = lib.route_sim_new(); lib.route_sim_begin(sim, s, w, 0, 1, 1)
    topo = ffi.new('struct route_topology *'); lib.route_topology_make(ffi.addressof(s, 'f'), s, topo)
    CX = const('ROUTE_CONTEXTS'); pat, node, owed = ffi.new('int[]', CX), ffi.new('int[]', CX), ffi.new('uint32_t[]', CX)
    lib.route_contexts(s, topo, pat, node, owed)
    dec = ffi.new('struct route_decision *'); tok = np.zeros((CX, const('ROUTE_IN')), np.float32)
    state = np.zeros(2 * const('ROUTE_D'), np.float32); wv = np.zeros((CX, const('ROUTE_FS')), np.float32)
    scratch = ffi.new('float[]', lib.route_forward_scratch(CX))
    out, k = bytearray(), 0
    while lib.route_sim_decide(sim, dec, ffi.from_buffer('float[]', tok, require_writable=True)) == 1:
        nl, nc = dec.lanes, dec.contexts
        out += np.array([nl, nc], np.int32).tobytes() + bytes(ffi.buffer(dec.lane))
        out += np.stack([np.frombuffer(ffi.buffer(pat, 4 * nc), np.int32), np.frombuffer(ffi.buffer(node, 4 * nc), np.int32), np.frombuffer(ffi.buffer(owed, 4 * nc), np.int32)], 1).tobytes()
        out += tok[:nc].tobytes()
        lib.route_forward(p, ffi.cast('float *', dec.lane), nl, ffi.from_buffer('float[]', tok), nc, ffi.from_buffer('float[]', state, require_writable=True), ffi.from_buffer('float[]', wv, require_writable=True), scratch)
        lib.route_sim_answer(sim, ffi.from_buffer('float[]', wv)); k += 1
    cst = ffi.new('struct route_costs *'); lib.route_sim_costs(sim, cst)
    open(d['out'], 'wb').write(out)
    emit({'record': 'dump', 'out': d['out'], 'refreshes': k, 'over_floor': cst.throughput, 'us': cst.throughput * s.op[0].floor / 1e3})


def records(path):
    L, IN = const('ROUTE_L'), const('ROUTE_IN'); b = open(path, 'rb').read(); o = 0; out = []
    while o < len(b):
        nl, nc = np.frombuffer(b, np.int32, 2, o); o += 8
        lanes = np.frombuffer(b, np.float32, L * IN, o).reshape(L, IN); o += L * IN * 4 + 12 * nc
        tok = np.frombuffer(b, np.float32, nc * IN, o).reshape(nc, IN); o += nc * IN * 4
        out.append((int(nl), int(nc), lanes, tok))
    return out


def compare(a, b, tol=1e-4):
    A, B = records(a), records(b)
    rows = []
    for k in range(min(len(A), len(B))):
        diffs = []
        for name, x, y in (('lane', A[k][2][:A[k][0]], B[k][2][:A[k][0]]), ('context', A[k][3], B[k][3][:len(A[k][3])])):
            bad = np.abs(x - y) > tol * np.maximum(1, np.abs(x))
            for f in np.nonzero(bad.any(0))[0]:
                r = int(np.nonzero(bad[:, f])[0][0])
                diffs.append({'tokens': name, 'feature': int(f), 'rows': int(bad[:, f].sum()), 'row': r, 'a': float(x[r, f]), 'b': float(y[r, f])})
        rows.append({'refresh': k, 'identical': not diffs, 'differences': diffs[:8]})
        if diffs and k > 1:
            break
    emit({'record': 'compare', 'a': a, 'b': b, 'refreshes': [len(A), len(B)], 'by_refresh': rows})


def one_case(cfg, lib):
    measured = fabric.load(cfg['run']['fabric'])
    dist = C.Distribution(config.flat(cfg), measured, lib, np.random.default_rng([cfg['run']['data_seed'], cfg['run']['rank']]))
    return measured, dist


def deploy(cfg, lib):
    """the CPU forward (forward.c, as switch.c's router) against the GPU's, the first refresh of the 12 MB all-reduce"""
    measured = fabric.load(cfg['run']['fabric'])
    net = network(cfg, lib, 4, 1, 4 * loop.CX)
    path = f'/tmp/route-deploy-{os.getpid()}.i8'; lib.route_net_save(net, ffi.NULL, path.encode())
    p = ffi.new('struct route_params *'); assert lib.route_params_read(p, path.encode()) == 0; os.unlink(path)
    c, _ = C.measured(measured, [(0, 0)]); c = c[[C.KINDS.index('all-reduce') * len(C.SIZES) + C.SIZES.index(12582912)]]
    ro = loop.Rollouts(lib, net, 1, 1); ro.begin([0], c)
    while ro.inset[0] is None:   # the first forward committed (its runs at their first refresh)
        ro.poll(True)
    F, (runs, off, ncs) = ro.F[0], ro.inset[0]
    b = int(np.nonzero(runs == 0)[0][0]); nc, nl = int(ncs[b]), int(F.mask[b].sum())
    lanes, tok = F.lanes[b].copy(), F.tokens[off[b]:off[b] + nc].copy()
    F.wait()
    h, w = np.zeros(loop.D2, np.float32), np.zeros((nc, loop.FS), np.float32)
    lib.route_forward(p, ffi.from_buffer('float[]', lanes), nl, ffi.from_buffer('float[]', tok), nc, ffi.from_buffer('float[]', h, require_writable=True),
                      ffi.from_buffer('float[]', w, require_writable=True), ffi.new('float[]', lib.route_forward_scratch(loop.CX)))
    emit({'record': 'deploy', 'lanes': nl, 'contexts': nc, 'w_largest_difference': float(np.abs(F.w[off[b]:off[b] + nc] - w).max()), 'w_largest': float(np.abs(w).max()),
          'state_largest_difference': float(np.abs(F.next[b] - h).max())})
    lib.route_sims_free(ro.sims); lib.route_net_free(net)


def check(cfg, lib, graph):
    """one drawn case's group, learned on (rate 0): the GPU's credit against split.h's on the CPU (route_draw_score),
    each draw's score against central differences of its log-probability, the fused producers' codes; graph: the int8
    network against the fp32 reference (w, each task's gradients)"""
    K = cfg['capacity']['refreshes']
    measured, dist = one_case(cfg, lib)
    net = network(cfg, lib, 4, K, 4 * K * loop.CX)
    ro = loop.Rollouts(lib, net, 1, K); B = loop.Batch(lib, net, 4, K, 4 * K * loop.CX)
    score = np.zeros(loop.FS, np.float32)
    def scores(draws, w):   # each draw's score under w (its row's), split.h on the CPU
        out = np.zeros((len(draws), loop.FS), np.float32)
        for i in range(len(draws)):
            lib.route_draw_score(ffi.from_buffer('struct route_draw *', draws[i:i + 1]), ffi.from_buffer('float[]', np.ascontiguousarray(w[draws[i]['row']])), ffi.from_buffer('float[]', score, require_writable=True))
            out[i] = score
        return out
    tried = 0
    while True:   # cases drawn until one's draws had a choice (a table of more than one allowed split)
        tried += 1
        ro.begin([0], dist.draw(1)[0]); ro.run_out()
        _, m, R, ND, scale = loop.load(lib, ro, np.array([0]), B)
        if ND and np.abs(scores(loop.view(B.p.draws, loop.DRAW, ND), B.w)).max() > 0:
            break
    lib.route_net_probes(net, 1, ffi.NULL, 0)
    lib.route_net_step(net, B.p, m, K, R, ND, scale, 0.0, 1); B.wait()
    buf = ffi.new('char[]', 1 << 20); lib.route_net_probes(net, 0, buf, 1 << 20)
    draws = loop.view(B.p.draws, loop.DRAW, ND)
    gpu = np.frombuffer(ffi.buffer(lib.route_net_dw(net), 3 * R * loop.FS * 4), np.float32).reshape(3, R, loop.FS)
    cpu = np.zeros((3, R, loop.FS), np.float32)
    sc = scores(draws, B.wcur)   # under the learning forward's w (the GPU's credit's), ratio 1 at rate 0
    for t, a in enumerate((B.scale0[draws['row']], scale * draws['a1'], scale * draws['a2'])):
        np.add.at(cpu[t], draws['row'], a[:, None] * sc)
    # central differences of a row's draws' log-probability, the row with the most draws, its largest scores
    row = int(np.bincount(draws['row'], weights=np.abs(sc).sum(1), minlength=R).argmax()); mine = np.nonzero(draws['row'] == row)[0]
    total = sc[mine].sum(0)
    def logp(w):
        return sum(lib.route_draw_score(ffi.from_buffer('struct route_draw *', draws[i:i + 1]), ffi.from_buffer('float[]', w), ffi.from_buffer('float[]', score, require_writable=True)) for i in mine)
    pairs, eps = [], 1e-2
    for f in np.argsort(-np.abs(total))[:6]:
        up, down = B.wcur[row].copy(), B.wcur[row].copy(); up[f] += eps; down[f] -= eps
        pairs.append({'feature': int(f), 'score': float(total[f]), 'central_difference': float((logp(up) - logp(down)) / (2 * eps))})
    rec = {'record': 'check', 'cases_tried': tried, 'runs': m, 'rows': R, 'draws': ND, 'gpu_credit_vs_cpu': {'largest_difference': float(np.abs(gpu - cpu).max()), 'largest': float(np.abs(cpu).max())},
           'score_vs_central_difference': pairs, 'fused_codes_vs_quant_rows': json.loads(ffi.string(buf).decode())}
    # the step's statistics (route_net_stats) against numpy's: the mean's squared norms (matrix units) and the rows
    Pn = ffi.new('int *'); lib.route_net_theta(net, Pn)
    g3 = np.frombuffer(ffi.buffer(lib.route_net_grads(net), 3 * Pn[0] * 4), np.float32).reshape(3, Pn[0]).astype(np.float64)
    stv = np.frombuffer(ffi.buffer(lib.route_net_stats(net), 17 * 4), np.float32)
    rec['stats_vs_numpy'] = {'rows': [float(stv[8]), R], 'mean_sq_norms': [float(x) for x in stv[14:17]], 'numpy': [float(x) for x in (g3 ** 2).sum(1)],
                             'microbatch_sq_norms': [float(x) for x in stv[9:12]]}
    if graph:
        P = ffi.new('int *'); lib.route_net_theta(net, P); P = P[0]
        wref, gref = np.zeros((R, loop.FS), np.float32), np.zeros(3 * P, np.float32)
        lib.route_net_reference(net, B.p, m, K, R, ffi.from_buffer('float[]', wref, require_writable=True), ffi.from_buffer('float[]', gref, require_writable=True))
        gint = np.frombuffer(ffi.buffer(lib.route_net_grads(net), 3 * P * 4), np.float32).reshape(3, P); gref = gref.reshape(3, P)
        rec['w_relative_error'] = float(np.linalg.norm(B.wcur[:R] - wref) / max(np.linalg.norm(wref), 1e-30))
        rec['gradient_cosine_by_task'] = [float(gint[t] @ gref[t] / max(np.linalg.norm(gint[t]) * np.linalg.norm(gref[t]), 1e-30)) for t in range(3)]
    emit(rec)
    lib.route_sims_free(ro.sims); lib.route_net_free(net)


def bench(cfg, lib):
    """one forward and one learning step of the training's sizes, kernel by kernel on the GPU clock"""
    K, cap = cfg['capacity']['refreshes'], cfg['capacity']
    measured, dist = one_case(cfg, lib)
    groups = cap['forward_rows'] // loop.CX // 4
    actor = network(cfg, lib, groups * 4, 1, groups * 4 * loop.CX); learner = network(cfg, lib, cap['learn_runs'], K, cap['forward_rows'])
    ro = loop.Rollouts(lib, actor, groups, K)
    out = ffi.new('char[]', 1 << 22); rec = {'record': 'bench'}
    lib.route_net_profile(actor, 1, ffi.NULL, 0)
    ro.begin(np.arange(groups), dist.draw(groups)[0])
    while ro.inset[0] is None:
        ro.poll(True)
    ro.F[0].wait(); lib.route_net_profile(actor, 0, out, 1 << 22)
    rec['forward'] = {'runs': int(len(ro.inset[0][0])), 'rows': int(ro.inset[0][2].sum()), 'kernels': json.loads(ffi.string(out).decode())}
    while not len(ro.ready()):
        ro.poll(True)
    B = loop.Batch(lib, learner, cap['learn_runs'], K, cap['forward_rows'])
    _, m, R, ND, scale = loop.load(lib, ro, ro.ready(), B)
    lib.route_net_profile(learner, 1, ffi.NULL, 0); lib.route_net_step(learner, B.p, m, K, R, ND, scale, 0.0, 1); B.wait(); lib.route_net_profile(learner, 0, out, 1 << 22)
    rec['step'] = {'runs': m, 'rows': R, 'draws': ND, 'kernels': json.loads(ffi.string(out).decode())}
    t0 = time.perf_counter()   # the same step again, alone on the GPU, back to back (not profiled): its wall time
    for _ in range(20):
        lib.route_net_step(learner, B.p, m, K, R, ND, scale, 0.0, 1)
    B.wait(); rec['step']['alone_ms'] = (time.perf_counter() - t0) / 20 * 1e3
    emit(rec)


def roofline(cfg, lib):
    """random numbers through the real forward and learning step at the configured capacity (no simulator): their
    effective int8 TOP/s and each kernel's share (profiled call apart)"""
    cap, K = cfg['capacity'], cfg['capacity']['refreshes']
    B, R = cap['learn_runs'], cap['forward_rows']
    rng = np.random.default_rng(1)
    net = network(cfg, lib, B, K, R)
    LB = loop.Batch(lib, net, B, K, R)
    LB.lanes[:] = rng.standard_normal(LB.lanes.shape).astype(np.float32); LB.mask[:] = 1
    LB.h0[:] = 0.1 * rng.standard_normal(LB.h0.shape).astype(np.float32)
    LB.tokens[:] = rng.standard_normal(LB.tokens.shape).astype(np.float32)
    seg = np.repeat(np.arange(B * K), R // (B * K)); seg = np.concatenate([seg, np.full(R - len(seg), B * K - 1)])
    LB.pos[:] = seg * loop.L + loop.L - 1
    bounds = np.searchsorted(seg, np.arange(B * K + 1)); LB.seg[:] = np.stack([bounds[:-1], bounds[1:]], 1).ravel()
    LB.w[:] = 0.1 * rng.standard_normal(LB.w.shape).astype(np.float32); LB.scale0[:] = 1.0 / B
    dr = LB.draws(R)   # a draw a row: one destination, five lanes all valid and allowed
    dr['arrive'] = rng.random((R, 30)).astype(np.float32); dr['valid'] = 31; dr['mask'] = 31; dr['S'] = 1; dr['deg'] = 5; dr['ntok'] = 5
    dr['token'] = rng.integers(0, 5, R); dr['row'] = np.arange(R); dr['unit'] = 1; dr['a1'] = 0.01; dr['a2'] = 0.01
    LB.first[:] = np.arange(R + 1)
    ops_learn = 3 * 2 * (B * K * loop.L * loop.LANE_MACS + R * loop.CTX_MACS)
    lib.route_net_step(net, LB.p, B, K, R, R, 1.0 / B, 0.0, 1); LB.wait()
    t0 = time.perf_counter(); n = 5
    for _ in range(n):
        lib.route_net_step(net, LB.p, B, K, R, R, 1.0 / B, 0.0, 1)
    LB.wait(); t = (time.perf_counter() - t0) / n
    out = ffi.new('char[]', 1 << 22)
    lib.route_net_profile(net, 1, ffi.NULL, 0); lib.route_net_step(net, LB.p, B, K, R, R, 1.0 / B, 0.0, 1); LB.wait(); lib.route_net_profile(net, 0, out, 1 << 22)
    ks = json.loads(ffi.string(out).decode()); tot = sum(k['ns'] for k in ks); agg = {}
    for k in ks:
        name = k['kernel'].rsplit(' ', 1)[1]; agg[name] = agg.get(name, 0) + k['ns']
    emit({'record': 'roofline', 'call': 'learning step', 'runs': B, 'refreshes': K, 'lane_rows': B * K * loop.L, 'context_rows': R, 'seconds': round(t, 5),
          'int8_tops': round(ops_learn / t / 1e12, 2), 'kernels_share': {k: round(v / tot, 3) for k, v in sorted(agg.items(), key=lambda x: -x[1])[:12]}})
    # the forward: as many runs as its rows allow (one refresh each), the same random rows
    FB = loop.Batch(lib, net, B, 1, R)
    FB.lanes[:] = LB.lanes[:B]; FB.mask[:] = 1; FB.h0[:] = LB.h0; FB.tokens[:] = LB.tokens; FB.pos[:] = (np.arange(R) % B) * loop.L + loop.L - 1
    ops_fwd = 2 * (B * loop.L * loop.LANE_MACS + R * loop.CTX_MACS)
    lib.route_net_forward(net, FB.p, B, R); FB.wait()
    t0 = time.perf_counter()
    for _ in range(n):
        lib.route_net_forward(net, FB.p, B, R)
    FB.wait(); t = (time.perf_counter() - t0) / n
    lib.route_net_profile(net, 1, ffi.NULL, 0); lib.route_net_forward(net, FB.p, B, R); FB.wait(); lib.route_net_profile(net, 0, out, 1 << 22)
    ks = json.loads(ffi.string(out).decode()); tot = sum(k['ns'] for k in ks); agg = {}
    for k in ks:
        name = k['kernel'].rsplit(' ', 1)[1]; agg[name] = agg.get(name, 0) + k['ns']
    emit({'record': 'roofline', 'call': 'forward', 'runs': B, 'lane_rows': B * loop.L, 'context_rows': R, 'seconds': round(t, 5),
          'int8_tops': round(ops_fwd / t / 1e12, 2), 'kernels_share': {k: round(v / tot, 3) for k, v in sorted(agg.items(), key=lambda x: -x[1])[:12]}})


def xbench(cfg, lib, n):
    """the exchange alone: n sums of known values through the mesh (0: joined and closed, a dead run's programs flushed)"""
    r = cfg['run']
    net = network(cfg, lib, 4, 1, loop.CX)
    if lib.route_net_join(net, r['rank'], r['world'], r['mesh_id'].encode()) != 0:
        emit({'record': 'failure', 'rank': r['rank'], 'what': ffi.string(lib.route_exchange_failure()).decode()}); sys.exit(1)
    P = ffi.new('int *'); lib.route_net_theta(net, P); P = 3 * P[0]
    g = np.frombuffer(ffi.buffer(lib.route_net_grads(net), P * 4), np.float32)
    seconds, wrong = [], 0
    for k in range(n):
        g[:] = r['rank'] + 1 + k % 5 + np.arange(P) % 7
        t0 = time.perf_counter(); lib.route_net_exchange(net); lib.route_net_grads(net); seconds.append(time.perf_counter() - t0)
        wrong += int((g != r['world'] * (r['world'] + 1) / 2 + r['world'] * (k % 5 + np.arange(P) % 7)).sum())
    lib.route_exchange_close()
    s = np.array(seconds) if n else np.zeros(1)
    emit({'record': 'xbench', 'rank': r['rank'], 'rounds': n, 'floats': P, 'wrong': wrong, 'ms': {'median': float(np.median(s)) * 1e3, 'max': float(s.max()) * 1e3}})


def stat(files, interval, samples):
    tables = [telemetry.Table(f) for f in files]
    for t in tables:
        t.sample()
    for _ in range(samples):
        time.sleep(interval)
        for t in tables:
            rec = t.sample()
            if rec:
                emit(rec)


def main(argv):
    cfg, rest, sets = args(argv)
    cmd = rest[0] if rest else 'help'
    if cmd == 'stat':
        stat(rest[1:], cfg['report']['interval'], 1); return
    if cmd == 'compare':
        compare(rest[1], rest[2]); return
    if cmd == 'cluster':
        from . import cluster
        cluster.main(cfg, rest[1:], sets); return
    lib = native.load(cfg['run']['library'])
    lib.route_argmax = 1 if cfg['run']['argmax'] else 0
    if cmd == 'train':
        train(cfg, lib)
    elif cmd == 'evaluate':
        evaluation(cfg, lib, network(cfg, lib, 4, 1, loop.CX), fabric.load(cfg['run']['fabric']), cfg['run']['theta_in'] or 'initial')
    elif cmd == 'simcheck':
        simcheck(cfg, lib)
    elif cmd == 'cases':
        describe(cfg, lib)
    elif cmd == 'dump':
        dump(cfg, lib, rest[1:])
    elif cmd == 'deploy':
        deploy(cfg, lib)
    elif cmd == 'check':
        check(cfg, lib, '--graph' in rest)
    elif cmd == 'bench':
        bench(cfg, lib)
    elif cmd == 'roofline':
        roofline(cfg, lib)
    elif cmd == 'xbench':
        xbench(cfg, lib, int(rest[1]))
    else:
        print(__doc__)


main(sys.argv[1:])
