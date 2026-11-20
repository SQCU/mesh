"""python -m route COMMAND [--config route.toml] [--set section.key=value ...]: every result a JSON record a line.

  train       one rank's training: the cases drawn here and pushed as the trainer has room; its end states into the
              pool; a telemetry record every report.interval; rank 0's evaluations at evaluation.at
  evaluate    the measured cluster's table for run.theta_in (or the initial theta): each mode, kind and size
  simcheck    tests/switch.c's fixed router on the measured cluster (us): the simulator against switch.c's own table
  cases       the distribution's draws: their rate, and what they are (state sources, fabrics, windows, streams)
  dump        every refresh's tokens of one collective simulated from a zero state, the CPU forward (forward.c, as
              switch.c's router) answering: tests/switch.c ROUTE_DUMP's format (--set dump.*: see DUMP)
  compare     two dumps, refresh by refresh: the first features that differ
  deploy      the CPU forward against the GPU's on the first refresh of the 12 MB all-reduce
  check       the credit against central differences (--graph: the int8 network against the fp32 reference)
  bench       each call's time, kernel by kernel; the environment's on drawn cases
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
from . import config, fabric, native, telemetry
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


def open_trainer(cfg, lib, steps=None):
    r = cfg['run']; keep = []
    def s(v):
        if not v:
            return ffi.NULL
        b = ffi.new('char[]', v.encode()); keep.append(b); return b
    c = ffi.new('struct route_config *', {'rank': r['rank'], 'world': r['world'], 'steps': r['steps'] if steps is None else steps, 'rate': r['rate'],
                                          'seed': r['theta_seed'], 'forward_rows': cfg['capacity']['forward_rows'], 'learn_runs': cfg['capacity']['learn_runs'],
                                          'refreshes': cfg['capacity']['refreshes']})
    c.mesh_id, c.metallib, c.theta_in, c.telemetry = s(r['mesh_id']), s(r['metallib']), s(r['theta_in']), s(r['telemetry'])
    lib.route_argmax = 1 if r['argmax'] else 0
    t = lib.route_trainer_open(c)
    if t == ffi.NULL:
        emit({'record': 'failure', 'rank': r['rank'], 'what': ffi.string(lib.route_trainer_failure()).decode()}); sys.exit(1)
    return t


def cases_ptr(c):
    return native.records('struct route_case', c)


def evaluation(cfg, lib, t, measured, label):
    c, rows = C.measured(measured, [(m, b) for _, m, b in MODES])
    out = np.empty((len(c), 4))
    lib.route_trainer_evaluate(t, cases_ptr(c), len(c), ffi.from_buffer('double[]', out, require_writable=True))
    table = {}
    for (name, m, b) in MODES:
        for k, kind in enumerate(C.KINDS):
            for s in C.SIZES:
                i = rows.index((m, b, k, s)); T = float(out[i].mean())
                floor = lib.route_op_floor(ffi.from_buffer('struct route_fabric *', measured), k, float(s), int(measured['compute'][0]))
                table.setdefault(name, {}).setdefault(kind, {})[s] = {'over_floor': round(T, 4), 'us': round(T * floor / 1e3, 2)}
    emit({'record': 'evaluation', 'label': label, 'table': table})


def train(cfg, lib):
    r = cfg['run']; measured = fabric.load(r['fabric'])
    rng = np.random.default_rng([r['data_seed'], r['rank']])
    dist = C.Distribution(config.flat(cfg), measured, lib, rng)
    t = open_trainer(cfg, lib)
    evaluating = r['rank'] in cfg['evaluation']['ranks']
    ev = threading.Thread(target=evaluation, args=(cfg, lib, t, measured, 'start')) if evaluating and 'start' in cfg['evaluation']['at'] else None
    if ev:
        ev.start()
    table = telemetry.Table(r['telemetry']); table.sample()
    done = threading.Event()
    def report():
        while not done.wait(cfg['report']['interval']):
            rec = table.sample()
            if rec:
                rec['pool'] = dist.pool.count; emit(rec)
    reporter = threading.Thread(target=report, daemon=True); reporter.start()
    lib.route_trainer_start(t)
    room = lib.route_trainer_groups(t) * 4 * 4
    ended = np.empty((room, C.D2), np.float32)
    while True:
        n = lib.route_trainer_room(t, cfg['report']['interval'])
        got = lib.route_trainer_ended(t, ffi.from_buffer('float[]', ended, require_writable=True), room)
        dist.pool.add(ended[:got])
        if n < 0:
            break
        if n > 0:
            c, _ = dist.draw(n)
            lib.route_trainer_push(t, cases_ptr(c), n)
    lib.route_trainer_finish(t)
    done.set()
    if ev:
        ev.join()
    failure = lib.route_trainer_failure()
    if failure != ffi.NULL:
        emit({'record': 'failure', 'rank': r['rank'], 'what': ffi.string(failure).decode()})
    if evaluating and 'end' in cfg['evaluation']['at']:
        evaluation(cfg, lib, t, measured, f'end, {lib.route_trainer_learned(t)} steps')
    if r['rank'] == 0 and (r['theta'] or r['theta_int8']):
        lib.route_trainer_save(t, r['theta'].encode() if r['theta'] else ffi.NULL, r['theta_int8'].encode() if r['theta_int8'] else ffi.NULL)
        emit({'record': 'saved', 'theta': r['theta'], 'theta_int8': r['theta_int8']})
    lib.route_trainer_close(t)


def simcheck(cfg, lib):
    measured = fabric.load(cfg['run']['fabric'])
    t = open_trainer(cfg, lib, steps=0)
    c, rows = C.measured(measured, [(4, 0)])
    out = np.empty((len(c), 4))
    lib.route_trainer_evaluate(t, cases_ptr(c), len(c), ffi.from_buffer('double[]', out, require_writable=True))
    table = {}
    for i, (_, _, k, s) in enumerate(rows):
        floor = lib.route_op_floor(ffi.from_buffer('struct route_fabric *', measured), k, float(s), int(measured['compute'][0]))
        table.setdefault(C.KINDS[k], {})[s] = round(float(out[i].mean()) * floor / 1e3, 2)
    emit({'record': 'simcheck', 'router': 'switch.c fixed', 'us': table})
    lib.route_trainer_close(t)


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


def deploy(cfg, lib):
    measured = fabric.load(cfg['run']['fabric'])
    t = open_trainer(cfg, lib, steps=0)
    path = f'/tmp/route-deploy-{os.getpid()}.i8'; lib.route_trainer_save(t, ffi.NULL, path.encode())
    p = ffi.new('struct route_params *'); assert lib.route_params_read(p, path.encode()) == 0; os.unlink(path)
    s = stream_one(lib, measured, 1, 12582912.0)
    w = ffi.new('struct route_window *', {'t0': 0, 'dt': 100e3, 'lag': 100e3, 'points': 0, 'warm': 2, 'burst': 0})
    sim = lib.route_sim_new(); lib.route_sim_begin(sim, s, w, 0, 1, 1)
    CX = const('ROUTE_CONTEXTS'); dec = ffi.new('struct route_decision *'); tok = np.zeros((CX, const('ROUTE_IN')), np.float32)
    lib.route_sim_decide(sim, dec, ffi.from_buffer('float[]', tok, require_writable=True))
    nl, nc = dec.lanes, dec.contexts
    hg, hc = np.zeros(2 * const('ROUTE_D'), np.float32), np.zeros(2 * const('ROUTE_D'), np.float32)
    wg, wc = np.zeros((nc, const('ROUTE_FS')), np.float32), np.zeros((nc, const('ROUTE_FS')), np.float32)
    lib.route_trainer_forward(t, ffi.cast('float *', dec.lane), nl, ffi.from_buffer('float[]', tok), nc, ffi.from_buffer('float[]', hg, require_writable=True), ffi.from_buffer('float[]', wg, require_writable=True))
    lib.route_forward(p, ffi.cast('float *', dec.lane), nl, ffi.from_buffer('float[]', tok), nc, ffi.from_buffer('float[]', hc, require_writable=True), ffi.from_buffer('float[]', wc, require_writable=True), ffi.new('float[]', lib.route_forward_scratch(CX)))
    emit({'record': 'deploy', 'lanes': nl, 'contexts': nc, 'w_largest_difference': float(np.abs(wg - wc).max()), 'w_largest': float(np.abs(wg).max()), 'state_largest_difference': float(np.abs(hg - hc).max())})
    lib.route_trainer_close(t)


def report_call(cfg, lib, name, fn, *extra):
    measured = fabric.load(cfg['run']['fabric'])
    rng = np.random.default_rng([cfg['run']['data_seed'], cfg['run']['rank']])
    dist = C.Distribution(config.flat(cfg), measured, lib, rng)
    c, _ = dist.draw(200)
    t = open_trainer(cfg, lib, steps=0)
    buf = ffi.new('char[]', 1 << 22)
    rc = fn(t, cases_ptr(c), len(c), *extra, buf, 1 << 22)
    emit({'record': name, 'report': json.loads(ffi.string(buf).decode()) if rc == 0 else None})
    lib.route_trainer_close(t)


def xbench(cfg, lib, n):
    t = open_trainer(cfg, lib, steps=0)
    if n == 0:
        lib.route_trainer_close(t); emit({'record': 'xbench', 'rank': cfg['run']['rank'], 'rounds': 0, 'declared_and_closed': True}); return
    s = np.zeros(n); wrong = ffi.new('long *')
    ok = lib.route_trainer_xbench(t, n, ffi.from_buffer('double[]', s, require_writable=True), wrong) == 0
    emit({'record': 'xbench', 'rank': cfg['run']['rank'], 'rounds': n, 'ok': ok, 'wrong': int(wrong[0]),
          'ms': {'median': float(np.median(s)) * 1e3, 'p90': float(np.quantile(s, 0.9)) * 1e3, 'max': float(s.max()) * 1e3}})
    lib.route_trainer_close(t)


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
    if cmd == 'train':
        train(cfg, lib)
    elif cmd == 'evaluate':
        t = open_trainer(cfg, lib, steps=0); evaluation(cfg, lib, t, fabric.load(cfg['run']['fabric']), cfg['run']['theta_in'] or 'initial'); lib.route_trainer_close(t)
    elif cmd == 'simcheck':
        simcheck(cfg, lib)
    elif cmd == 'cases':
        describe(cfg, lib)
    elif cmd == 'dump':
        dump(cfg, lib, rest[1:])
    elif cmd == 'deploy':
        deploy(cfg, lib)
    elif cmd == 'check':
        report_call(cfg, lib, 'check', lib.route_trainer_check, 1e-3, 1 if '--graph' in rest else 0)
    elif cmd == 'bench':
        report_call(cfg, lib, 'bench', lib.route_trainer_bench)
    elif cmd == 'xbench':
        xbench(cfg, lib, int(rest[1]))
    else:
        print(__doc__)


main(sys.argv[1:])
