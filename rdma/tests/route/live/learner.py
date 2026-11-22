"""The live routing loop's learner (README.md): every node's agent ships its bridge's records here; each draw (a piece's
split at a hop, its context, its exact behaviour log-probability) is joined with its piece's deliveries at the
destinations it still owed (the piece's name: source, key, staging stamp), its reward the mean cost those deliveries
carried in-band (each hop's queueing and residence on its own clock, the wire's time); the advantage against a running
baseline of its (node, context, size); the policy gradient of each context's w (live.c live_grad) taken by Adam; every
node's tables sent back to its agent, which publishes them into its bridge. Memory and sockets only.

usage: python learner.py PORT AGENTS LIB [SAMPLE]   (stdin, a word a line: learn | freeze | prior | withdraw | publish | stats)"""
import ctypes, os, selectors, socket, struct, sys, time
import numpy as np

port, agents, libpath = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
SAMPLE = int(sys.argv[4]) if len(sys.argv) > 4 else 16
lib = ctypes.CDLL(libpath)
lib.live_grad.restype = ctypes.c_uint32
lib.live_fabric.restype = ctypes.c_uint32
if lib.live_attach(): sys.exit('learner: no bridge region of this build here (the fabric the floors are over)')
P = ctypes.c_void_p
STATE_F = 35   # split.h: features 35..37 the state ones (latest, mean arrival; share), 0..34 lane identities
STEP, AGE, MOST, LR, CLIP = 0.1, 1.0, 40000, float(os.environ.get('LIVE_LR', '0.02')), 4.0
DRAW = np.dtype([('arrive', '<f4', 30), ('valid', '<u4'), ('mask', '<u4'), ('S', '<u4'), ('token', '<u4'), ('deg', '<u4'),
                 ('ntok', '<u4'), ('row', '<u4'), ('unit', '<f4'), ('a1', '<f4'), ('a2', '<f4')])
REC = np.dtype([('kind', '<u4'), ('node', '<u4'), ('key', '<u8'), ('execution', '<u8'), ('at', '<u8'), ('draw', DRAW),
                ('logp', '<f4'), ('latency', '<f4'), ('queueing', '<f4'), ('floor', '<f4'), ('source', '<u4'), ('transfer', '<u4'),
                ('span', '<u4'), ('bytes', '<u4'), ('context', '<u4'), ('point', '<u4'), ('dest', '<u4'), ('padding', '<u4'),
                ('piece', '<u8')])
assert DRAW.itemsize == 160


def pid(r):
    return r['piece'] ^ (r['key'] * np.uint64(0x9E3779B97F4A7C15)) ^ (r['source'].astype(np.uint64) * np.uint64(0xC2B2AE3D27D4EB4F))


class Node:
    def __init__(self, sock, hello):
        self.sock = sock
        self.id, rb, nn = struct.unpack('<III', hello[:12])
        assert rb == REC.itemsize, f'record {rb} bytes, the learner reads {REC.itemsize}'
        self.view = list(np.frombuffer(hello[12:12 + 4 * nn], np.uint32))
        self.keys = np.frombuffer(hello[12 + 4 * nn:], np.uint32).reshape(-1, 4)
        n = len(self.keys)
        self.w = np.zeros((n, 40), np.float32); self.m = np.zeros_like(self.w); self.v = np.zeros_like(self.w); self.t = 0
        self.inbox = bytearray(); self.dropped = 0; self.point = 0; self.out = b''; self.out_ctl = b''; self.out_w = None


sel = selectors.DefaultSelector()
srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); srv.bind(('0.0.0.0', port)); srv.listen()
nodes = {}
pending = []
while len(nodes) < agents:
    c, _ = srv.accept(); c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    head = b''
    while len(head) < 5: head += c.recv(5 - len(head))
    kind, n = struct.unpack('<cI', head)
    body = b''
    while len(body) < n: body += c.recv(n - len(body))
    x = Node(c, body); nodes[x.id] = x
    c.setblocking(False); sel.register(c, selectors.EVENT_READ, x)
    print(f'agent {x.id}: {len(x.keys)} contexts, view {x.view}', flush=True)
view = next(iter(nodes.values())).view
index = {int(n): i for i, n in enumerate(view)}
_in = (ctypes.c_double * 8)(); _out = (ctypes.c_double * 8)(); _unit = ctypes.c_double(0)
lib.live_fabric(_in, _out, ctypes.byref(_unit))
CAP_IN, CAP_OUT, UNIT = np.array(_in[:len(view)]), np.array(_out[:len(view)]), _unit.value
assert all(x.view == view for x in nodes.values()), 'agents of different views'
sel.register(sys.stdin, selectors.EVENT_READ, None)

mode = 'learn'
draws, pieces = [], []          # arrays of REC, with their arrival time here
base = {}                       # (node, context, bucket) -> [mean, var, count]
stats = dict(draws=0, joined=0, pieces=0, steps=0, thr=[], lat=[], que=[], alpha=[], g=[])
run_thr = {}
last_step = last_stat = time.monotonic()


def send(x, kind, payload):
    m = struct.pack('<cI', kind, len(payload)) + payload
    if kind == b'W': x.out_w = m          # only the latest table matters: an older one not yet sent is replaced
    else: x.out_ctl += m


def flush(x):
    if not x.out and (x.out_ctl or x.out_w):
        x.out = x.out_ctl + (x.out_w or b''); x.out_ctl = b''; x.out_w = None
    while x.out:
        try: k = x.sock.send(x.out)
        except BlockingIOError: return
        x.out = x.out[k:]


def step(now):
    """The trainer's loss (rdma/tests/route/sim.c route_sim_costs, route_sim_draws; route.h ROUTE_TASKS), live:
    a transaction is a program's execution (key, execution), its rollouts the program's other executions; once its
    sampled pieces' landings are in (AGE after its first), its three costs, each over its floor (its cut bound on the
    fabric's capacities, at least the view's unit: the tasks' gradients commensurate across sizes, no size class):
      throughput  its makespan: at each destination the first landing's carried cost and the destination's own clock
                  from first to last landing, the latest destination's;
      latency     each piece's carried cost to each destination it owed (at presence);
      queueing    the queueing it carried there;
    each draw's advantage a task: throughput against the key's other executions (leave-one-out), latency and queueing
    its piece's against the other pieces of the same (transaction, source); three gradients (score x advantage, live.c
    live_grad), UPGrad's weights from their Gram matrix (live.c live_upgrad), Adam descending."""
    global draws, pieces
    if not draws or not pieces: return
    D = np.concatenate([d for d, _ in draws]); Dt = np.concatenate([np.full(len(d), t) for d, t in draws])
    Pc = np.concatenate([p for p, _ in pieces]); Pt = np.concatenate([np.full(len(p), t) for p, t in pieces])
    tid = lambda r: r['key'] * np.uint64(0x9E3779B97F4A7C15) ^ r['execution']
    Ptid, Dtid = tid(Pc), tid(D)
    # a transaction's age: since its first landing here
    ut, inv = np.unique(Ptid, return_inverse=True)
    first_seen = np.full(len(ut), np.inf); np.minimum.at(first_seen, inv, Pt)
    final_t = ut[now - first_seen >= AGE]
    if not len(final_t):
        return
    pf = np.isin(Ptid, final_t); df = np.isin(Dtid, final_t)
    P_, D_ = Pc[pf], D[df]
    # keep the rest for later steps (bounded)
    kd, kt = D[~df], Dt[~df]
    if len(kd) > 4 * MOST: kd, kt = kd[-4 * MOST:], kt[-4 * MOST:]
    draws = [(kd, now)] if len(kd) else []
    pieces = [(Pc[~pf], now)] if (~pf).any() else []
    if not len(P_) or not len(D_): return
    # each transaction's floor: its cut bound over the sampled bytes (scaled by the sample) and the fabric's capacities
    T, ti = np.unique(tid(P_), return_inverse=True)
    di = np.array([index.get(int(n), 0) for n in P_['dest']]); si = np.array([index.get(int(n), 0) for n in P_['source']])
    nv = len(view)
    bin_ = np.zeros((len(T), nv)); bout = np.zeros((len(T), nv))
    np.add.at(bin_, (ti, di), P_['bytes'] * float(SAMPLE)); np.add.at(bout, (ti, si), P_['bytes'] * float(SAMPLE))
    cut = np.maximum((bin_ / np.maximum(CAP_IN, 1e-9)).max(1), (bout / np.maximum(CAP_OUT, 1e-9)).max(1))
    floor = np.maximum(cut, UNIT)
    # throughput: the makespan, the latest destination's (first landing's carried cost + its own clock first to last)
    key_td = ti * nv + di
    g, gi = np.unique(key_td, return_inverse=True)
    at = P_['at'].astype(np.float64)
    first = np.full(len(g), np.inf); last = np.full(len(g), -np.inf); np.minimum.at(first, gi, at); np.maximum.at(last, gi, at)
    lat_first = np.zeros(len(g)); o = np.lexsort((at, gi)); firsts = o[np.r_[True, gi[o][1:] != gi[o][:-1]]]
    lat_first[gi[firsts]] = P_['latency'][firsts]
    done_d = lat_first + (last - first)
    make = np.zeros(len(T)); np.maximum.at(make, g // nv, done_d)
    c_thr = make / floor
    # a piece's latency and queueing to a destination, over its transaction's floor
    c_lat = P_['latency'] / floor[ti]; c_que = P_['queueing'] / floor[ti]
    # the draws joined with their pieces' landings at the destinations they owed
    pk = pid(P_) * np.uint64(8) + di.astype(np.uint64)
    order = np.argsort(pk, kind='stable'); pks = pk[order]
    dk = pid(D_); S = D_['draw']['S']
    lat = np.zeros(len(D_)); que = np.zeros(len(D_)); cnt = np.zeros(len(D_))
    for d in range(nv):
        has = (S >> np.uint32(d)) & np.uint32(1) == 1
        q = dk * np.uint64(8) + np.uint64(d)
        j = np.searchsorted(pks, q); ok = has & (j < len(pks))
        ok[ok] &= pks[j[ok]] == q[ok]
        lat[ok] += c_lat[order[j[ok]]]; que[ok] += c_que[order[j[ok]]]; cnt[ok] += 1
    use = cnt > 0
    D_, lat, que, cnt = D_[use], lat[use] / cnt[use], que[use] / cnt[use], cnt[use]
    if not len(D_): return
    dT = np.searchsorted(T, tid(D_))
    # throughput's advantage: against the key's other executions (leave-one-out), else the key's running mean
    a_thr = np.zeros(len(D_))
    Tkeys = np.zeros(len(T), np.uint64)
    np.maximum.at(Tkeys, ti, P_['key'])
    for k in np.unique(Tkeys):
        m = Tkeys == k; ct = c_thr[m]; n = m.sum()
        mean_other = (ct.sum() - ct) / (n - 1) if n > 1 else np.full(n, run_thr.get(int(k), ct[0]))
        adv_t = np.zeros(len(T)); adv_t[m] = ct - mean_other
        sel = Tkeys[dT] == k
        a_thr[sel] = adv_t[dT[sel]]
        run_thr[int(k)] = 0.9 * run_thr.get(int(k), float(ct.mean())) + 0.1 * float(ct.mean())
    # latency and queueing: against the other pieces of the same (transaction, source), leave-one-out
    gs = ti * nv + si
    gsu, gsi = np.unique(gs, return_inverse=True)
    s1 = np.bincount(gsi, c_lat, len(gsu)); s2 = np.bincount(gsi, c_que, len(gsu)); sn = np.bincount(gsi, minlength=len(gsu)).astype(float)
    dsrc = np.array([index.get(int(n), 0) for n in D_['source']])
    dg = np.searchsorted(gsu, dT * nv + dsrc); dg = np.minimum(dg, len(gsu) - 1)
    okg = gsu[dg] == dT * nv + dsrc
    rest = np.maximum(sn[dg] - cnt, 1)
    a_lat = np.where(okg & (sn[dg] > cnt), lat - (s1[dg] - lat * cnt) / rest, 0.0)
    a_que = np.where(okg & (sn[dg] > cnt), que - (s2[dg] - que * cnt) / rest, 0.0)
    stats['joined'] += len(D_); stats['thr'].append(c_thr); stats['lat'].append(lat); stats['que'].append(que)
    if mode != 'learn': return
    # three gradients over every node's tables, UPGrad's weights, Adam descending
    J = []
    for adv in (a_thr, a_lat, a_que):
        parts = []
        for nid, x in nodes.items():
            sel_ = D_['node'] == nid
            gx = np.zeros_like(x.w)
            if sel_.any():
                u = D_[sel_]; rows = np.ascontiguousarray(u['context'].astype(np.uint32)); ok = rows < len(x.keys)
                u, rows = u[ok], rows[ok]; a = np.ascontiguousarray(adv[sel_][ok].astype(np.float32))
                dr = np.ascontiguousarray(u['draw']); beh = np.ascontiguousarray(u['logp']); lp = np.zeros(len(u), np.float32)
                lib.live_grad(dr.ctypes.data_as(P), ctypes.c_uint32(DRAW.itemsize), beh.ctypes.data_as(P), ctypes.c_uint32(len(u)),
                              rows.ctypes.data_as(P), a.ctypes.data_as(P), x.w.ctypes.data_as(P), gx.ctypes.data_as(P), ctypes.c_float(CLIP),
                              lp.ctypes.data_as(P))
            # a context's table holds no route: only its weights on the split's state features (the latest and mean
            # arrival estimates, the lane share) learn; a lane's identity one-hots (0..34) stay 0, since a table's
            # weight on them would be a route fixed whatever the live state (the net, w from the live tokens, is what
            # may weigh them, every refresh)
            gx[:, :STATE_F] = 0
            parts.append(gx.ravel() / max(len(D_), 1))
        J.append(np.concatenate(parts))
    J = np.array(J, np.float32)
    G = np.ascontiguousarray((J @ J.T).astype(np.float32)); alpha = np.zeros(3, np.float32)
    lib.live_upgrad(G.ctypes.data_as(P), alpha.ctypes.data_as(P))
    gall = (alpha[:, None] * J).sum(0)
    stats['alpha'].append(alpha.copy())
    off = 0
    for nid, x in nodes.items():
        gx = gall[off:off + x.w.size].reshape(x.w.shape); off += x.w.size
        x.t += 1
        x.m = 0.9 * x.m + 0.1 * gx; x.v = 0.999 * x.v + 0.001 * gx * gx
        mh = x.m / (1 - 0.9 ** x.t); vh = x.v / (1 - 0.999 ** x.t)
        x.w -= (LR * mh / (np.sqrt(vh) + 1e-6)).astype(np.float32)
        np.clip(x.w, -60, 60, out=x.w)
        x.point += 1
        send(x, b'W', struct.pack('<Q', x.point) + x.w.tobytes())
    # the sign of each task's gradient on the arrival features, consistent or not (the credit's signal)
    stats['g'].append(np.array([J[t].reshape(-1, 40)[:, 35].sum() for t in range(3)]))
    stats['steps'] += 1


print(f'learner: {len(nodes)} agents, view {view}, lr {LR}; mode learn', flush=True)
try:
    while True:
        for k, _ in sel.select(timeout=0.01):
            if k.data is None:
                line = sys.stdin.readline().strip()
                if line in ('learn', 'freeze'): mode = line; print(f'mode {mode}', flush=True)
                elif line == 'prior':
                    for x in nodes.values(): x.w[:] = 0; x.m[:] = 0; x.v[:] = 0; x.t = 0; send(x, b'P', b'')
                    base.clear(); print('prior published (w 0), baselines reset', flush=True)
                elif line == 'withdraw':
                    mode = 'freeze'
                    for x in nodes.values(): send(x, b'D', b'')
                    print('tables withdrawn: the bridges\' own rule; mode freeze', flush=True)
                elif line == 'publish':
                    for x in nodes.values(): x.point += 1; send(x, b'W', struct.pack('<Q', x.point) + x.w.tobytes())
                    print('current w published', flush=True)
                elif line == 'stats':
                    for x in nodes.values():
                        print(f'  node {x.id}: |w| mean {np.abs(x.w).mean():.3f} max {np.abs(x.w).max():.2f}; '
                              f'w[35..37] (latest, mean arrival, share) mean {x.w[:, 35:38].mean(0).round(2).tolist()}', flush=True)
                elif line == '':
                    raise KeyboardInterrupt
                continue
            x = k.data
            try:
                b = x.sock.recv(1 << 22)
            except BlockingIOError:
                continue
            if not b: raise ConnectionError(f'agent {x.id} gone')
            x.inbox += b
            while len(x.inbox) >= 5:
                kind, n = struct.unpack('<cI', x.inbox[:5])
                if len(x.inbox) < 5 + n: break
                body = bytes(x.inbox[5:5 + n]); del x.inbox[:5 + n]
                if kind == b'R':
                    x.dropped = struct.unpack('<Q', body[:8])[0]
                    r = np.frombuffer(body[8:], REC)
                    now = time.monotonic()
                    d = r[(r['kind'] == 1) & (r['context'] != 0xFFFFFFFF)]
                    p = r[r['kind'] == 2]
                    if len(d): draws.append((d.copy(), now)); stats['draws'] += len(d)
                    if len(p): pieces.append((p.copy(), now)); stats['pieces'] += len(p)
        for x in nodes.values(): flush(x)
        now = time.monotonic()
        if now - last_step >= STEP:
            step(now); last_step = now
        if now - last_stat >= 10.0:
            cat = lambda k: np.concatenate(stats[k]) if stats[k] else np.zeros(1)
            A = np.array(stats['alpha']) if stats['alpha'] else np.zeros((1, 3)); Gs = np.array(stats['g']) if stats['g'] else np.zeros((1, 3))
            print(f'{time.strftime("%H:%M:%S")} {mode}: draws {stats["draws"]} landings {stats["pieces"]} joined {stats["joined"]} steps {stats["steps"]}; '
                  f'costs over floor: throughput {cat("thr").mean():.2f} latency {cat("lat").mean():.2f} queueing {cat("que").mean():.2f}; '
                  f'UPGrad weights {A.mean(0).round(2).tolist()}; arrival-gradient negative {(Gs < 0).mean(0).round(2).tolist()}; '
                  f'dropped {sum(x.dropped for x in nodes.values())}', flush=True)
            stats.update(draws=0, joined=0, pieces=0, steps=0, thr=[], lat=[], que=[], alpha=[], g=[]); last_stat = now
except (KeyboardInterrupt, ConnectionError) as e:
    print(f'learner ends: {e!r}', flush=True)
finally:
    for x in nodes.values():
        try:
            x.sock.setblocking(True); x.sock.sendall(struct.pack('<cI', b'D', 0))
        except Exception: pass
