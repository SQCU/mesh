"""The live routing loop's learner (README.md): every node's agent ships its bridge's records here; each draw (a piece's
split at a hop, its context, its exact behaviour log-probability) is joined with its piece's deliveries at the
destinations it still owed (the piece's name: source, key, staging stamp), its reward the mean cost those deliveries
carried in-band (each hop's queueing and residence on its own clock, the wire's time); the advantage against a running
baseline of its (node, context, size); the policy gradient of each context's w (live.c live_grad) taken by Adam; every
node's tables sent back to its agent, which publishes them into its bridge. Memory and sockets only.

usage: python learner.py PORT AGENTS LIB   (stdin, a word a line: learn | freeze | prior | withdraw | stats)"""
import ctypes, os, selectors, socket, struct, sys, time
import numpy as np

port, agents, libpath = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
lib = ctypes.CDLL(libpath)
lib.live_grad.restype = ctypes.c_uint32
P = ctypes.c_void_p
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
assert all(x.view == view for x in nodes.values()), 'agents of different views'
sel.register(sys.stdin, selectors.EVENT_READ, None)

mode = 'learn'
draws, pieces = [], []          # arrays of REC, with their arrival time here
base = {}                       # (node, context, bucket) -> [mean, var, count]
stats = dict(draws=0, joined=0, pieces=0, steps=0, lat=[], ent=[], g=[])
gsum = []
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
    global draws, pieces
    if not draws: return
    D = np.concatenate([d for d, _ in draws]); Dt = np.concatenate([np.full(len(d), t) for d, t in draws])
    Pc = np.concatenate([p for p, _ in pieces]) if pieces else np.zeros(0, REC)
    Pt = np.concatenate([np.full(len(p), t) for p, t in pieces]) if pieces else np.zeros(0)
    # the deliveries by (piece, destination's view index)
    pdi = np.array([index.get(int(n), 7) for n in Pc['dest']], np.uint64) if len(Pc) else np.zeros(0, np.uint64)
    pk = pid(Pc) * np.uint64(8) + pdi if len(Pc) else np.zeros(0, np.uint64)
    order = np.argsort(pk, kind='stable'); pks = pk[order]; plat = Pc['latency'][order]
    dk = pid(D)
    S = D['draw']['S']
    total = np.zeros(len(D)); count = np.zeros(len(D)); want = np.zeros(len(D))
    for d in range(len(view)):
        has = (S >> np.uint32(d)) & np.uint32(1) == 1
        want += has
        q = dk * np.uint64(8) + np.uint64(d)
        at = np.searchsorted(pks, q); ok = has & (at < len(pks))
        ok[ok] &= pks[at[ok]] == q[ok]
        total[ok] += plat[at[ok]]; count[ok] += 1
    done = (count == want) & (want > 0)
    late = ~done & (now - Dt > AGE)
    use = done | (late & (count > 0))
    lat = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    # the rest kept for the next step (their deliveries not here yet); deliveries kept AGE
    keep = ~use & ~late
    kd, kt = D[keep], Dt[keep]
    if len(kd) > 4 * MOST: kd, kt = kd[-4 * MOST:], kt[-4 * MOST:]
    draws = [(kd, t) for t in [kt.min()]] if len(kd) else []
    pkeep = now - Pt <= AGE
    pieces = [(Pc[pkeep], now)] if pkeep.any() else []
    U = D[use]; L = lat[use]
    if not len(U): return
    stats['joined'] += len(U); stats['lat'].append(L)
    if len(U) > MOST: pick = np.random.default_rng().choice(len(U), MOST, replace=False); U, L = U[pick], L[pick]
    bucket = np.minimum(np.log2(np.maximum(U['bytes'], 1)).astype(np.int64), 24)
    gk = (U['node'].astype(np.int64) << 32) | (U['context'].astype(np.int64) << 8) | bucket
    keys, inv = np.unique(gk, return_inverse=True)
    # the baselines as they stood (each group's mean and variance), the advantage against them, then each group's
    # running mean and variance moved by this step's draws of it
    mean = np.array([base.get(int(k), (np.nan, np.nan, 0))[0] for k in keys]); var = np.array([base.get(int(k), (0, 0, 0))[1] for k in keys])
    seen = np.array([base.get(int(k), (0, 0, 0))[2] for k in keys])
    cnt = np.bincount(inv, minlength=len(keys)); bm = np.bincount(inv, L, len(keys)) / cnt
    bv = np.bincount(inv, (L - bm[inv]) ** 2, len(keys)) / cnt
    fresh = seen == 0
    mean[fresh] = bm[fresh]; var[fresh] = bv[fresh] + 1.0
    sd = np.sqrt(np.maximum(var, 1.0))
    adv = np.where(seen[inv] >= 8, np.clip((mean[inv] - L) / sd[inv], -3, 3), 0.0).astype(np.float32)
    e = np.where(seen >= 50, 0.05, 1.0 / (seen + 1))
    nm = mean + e * (bm - mean); nv = (1 - e) * (var + e * (bm - mean) ** 2) + e * bv
    for k, a, b, c in zip(keys.tolist(), nm.tolist(), nv.tolist(), (seen + cnt).tolist()): base[k] = (a, b, c)
    if mode != 'learn': return
    for nid, x in nodes.items():
        sel_ = U['node'] == nid
        if not sel_.any(): continue
        u = U[sel_]; a = np.ascontiguousarray(adv[sel_])
        rows = np.ascontiguousarray(u['context'].astype(np.uint32))
        ok = rows < len(x.keys)
        if not ok.all(): u, a, rows = u[ok], a[ok], rows[ok]
        dr = np.ascontiguousarray(u['draw']); beh = np.ascontiguousarray(u['logp'])
        g = np.zeros_like(x.w); now_lp = np.zeros(len(u), np.float32)
        lib.live_grad(dr.ctypes.data_as(P), ctypes.c_uint32(DRAW.itemsize), beh.ctypes.data_as(P), ctypes.c_uint32(len(u)),
                      rows.ctypes.data_as(P), a.ctypes.data_as(P), x.w.ctypes.data_as(P), g.ctypes.data_as(P), ctypes.c_float(CLIP),
                      now_lp.ctypes.data_as(P))
        stats['ent'].append(-now_lp[np.isfinite(now_lp)])
        per = np.bincount(rows, minlength=len(x.keys)).astype(np.float32)[:, None]
        g /= np.maximum(per, 1)
        stats['g'].append(g[:, 35:38].sum(0))
        x.t += 1
        x.m = 0.9 * x.m + 0.1 * g; x.v = 0.999 * x.v + 0.001 * g * g
        mh = x.m / (1 - 0.9 ** x.t); vh = x.v / (1 - 0.999 ** x.t)
        x.w += (LR * mh / (np.sqrt(vh) + 1e-6)).astype(np.float32)
        np.clip(x.w, -60, 60, out=x.w)
        x.point += 1
        send(x, b'W', struct.pack('<Q', x.point) + x.w.tobytes())
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
        if now - last_stat >= 1.0:
            L = np.concatenate(stats['lat']) if stats['lat'] else np.zeros(0)
            E = np.concatenate(stats['ent']) if stats['ent'] else np.zeros(0)
            print(f'{time.strftime("%H:%M:%S")} {mode}: draws {stats["draws"]} pieces {stats["pieces"]} joined {stats["joined"]} '
                  f'steps {stats["steps"]}; piece cost us mean {np.nanmean(L) / 1e3 if len(L) else 0:.1f} '
                  f'p50 {np.nanpercentile(L, 50) / 1e3 if len(L) else 0:.1f} p90 {np.nanpercentile(L, 90) / 1e3 if len(L) else 0:.1f}; '
                  f'-logp {E.mean() if len(E) else 0:.2f}; dropped {sum(x.dropped for x in nodes.values())}', flush=True)
            gsum.extend(stats['g'])
            if len(gsum) >= 200:
                G = np.array(gsum)
                print(f'  gradient on (latest, mean arrival, share) over {len(G)} node-steps: mean {G.mean(0).round(4).tolist()} '
                      f'sd {G.std(0).round(4).tolist()} negative {(G < 0).mean(0).round(2).tolist()}', flush=True)
                gsum.clear()
            stats.update(draws=0, joined=0, pieces=0, steps=0, lat=[], ent=[], g=[]); last_stat = now
except (KeyboardInterrupt, ConnectionError) as e:
    print(f'learner ends: {e!r}', flush=True)
finally:
    for x in nodes.values():
        try:
            x.sock.setblocking(True); x.sock.sendall(struct.pack('<cI', b'D', 0))
        except Exception: pass
