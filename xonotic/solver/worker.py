"""The solver worker: node 1 answers the game server's bot-plan requests.

    python3 worker.py [--policy nearest|inverted] [--peer 0] [--secs 0]
    python3 worker.py --check            # validate the solve, no mesh needed
    python3 worker.py --bench 20         # time the solve, no mesh needed

Reads request blocks framed by bridge/PORT.md section 2, scores every bot row
against the objective basis, and writes one response block back per block it
completes. See bridge/PORT.md section 4 for the contract this implements.

The solve is the design's workload (design/mesh-coprocessor-demo.md): the
per-tick rows join a persistent context window of CTX recent rows, the whole
window is lifted into a RES-wide residual basis, routed through EXPERTS
experts grouped per expert, and a Gram matrix over the residual couples the
objective scores of the current rows to the recent history. CTX=4096 keeps
rows-per-expert at the measured T/E >= 512 floor and RES=2048 sits at the
measured orthogonalisation floor, so every solve is ~104 GFLOP against ~270 MB
of resident expert weights instead of a 16-row MLP.
"""
import argparse, sys, time
import numpy as np
from xonwire import (Mesh, Reassembler, TxWindow, REQ, RESP, REQ_WIDTH,
                     RESP_WIDTH, TEAMS, rows_per_slot)

SEED, EXPERTS, FF, HID = 20260828, 8, 2048, 64
CTX, RES = 4096, 2048
POLICIES = {"nearest": np.argmax, "inverted": np.argmin}

try:
    import mlx.core as mx
except Exception:
    mx = None


def _init(seed):
    g = np.random.default_rng(seed)
    return lambda *s: (g.standard_normal(s) * (1.0 / np.sqrt(s[-2]))).astype(np.float32)


def weights(D):
    """The original per-row model: small MoE (mlx) and tanh MLP (fallback)."""
    f = _init(SEED)
    return dict(R=f(D, EXPERTS), W1=f(EXPERTS, D, FF), W2=f(EXPERTS, FF, D),
                O=f(D, TEAMS), A=f(D, HID), P=f(HID, TEAMS))


def scores_mlx(X, w):
    Xm = mx.array(X)
    e = np.asarray(memoryview(mx.argmax(Xm @ mx.array(w["R"]), axis=1)))
    Y = np.zeros((X.shape[0], X.shape[1]), np.float32)
    for i in range(EXPERTS):
        sel = np.nonzero(e == i)[0]
        if sel.size:
            Yi = mx.maximum(mx.array(X[sel]) @ mx.array(w["W1"][i]), 0.0) @ mx.array(w["W2"][i])
            mx.eval(Yi)
            Y[sel] = np.asarray(memoryview(Yi))
    G = mx.array(Y) @ mx.array(w["O"])
    mx.eval(G)
    return np.nan_to_num(np.asarray(memoryview(G))).astype(np.float32)


def scores_np(X, w):
    with np.errstate(all="ignore"):
        return np.nan_to_num(np.tanh(X @ w["A"]) @ w["P"]).astype(np.float32)


def scores_moe_ref(X, w):
    """Pure-numpy reference for scores_mlx: same routing, experts, basis."""
    with np.errstate(all="ignore"):
        e = np.argmax(X @ w["R"], axis=1)
        Y = np.zeros_like(X)
        for i in range(EXPERTS):
            sel = np.nonzero(e == i)[0]
            if sel.size:
                Y[sel] = np.maximum(X[sel] @ w["W1"][i], 0.0) @ w["W2"][i]
        return np.nan_to_num(Y @ w["O"]).astype(np.float32)


def ctx_weights(D):
    f = _init(SEED)
    return dict(Ein=f(D, RES), Rt=f(RES, EXPERTS), W1=f(EXPERTS, RES, FF),
                W2=f(EXPERTS, FF, RES), Og=f(RES, TEAMS))


def ctx_flops(T, n):
    return (2 * T * REQ_WIDTH * RES + 2 * T * RES * EXPERTS + 4 * T * RES * FF
            + 2 * T * RES * RES + 2 * RES * RES * TEAMS + 2 * n * RES * TEAMS)


class CtxSolver:
    """MoE + Gram-over-residual over a rolling context window.

    Every per-tick request block is inserted into a ring of the CTX most
    recent (normalised) rows. The solve lifts the whole window to the
    residual basis, routes it through the experts grouped per expert, adds
    the residual, forms the Gram matrix of the window, and scores only the
    current tick's rows through the Gram-coupled objective basis. All state
    is a deterministic function of (seed, request history); the policy only
    selects argmax or argmin over the returned scores.
    """

    def __init__(self, T=CTX, backend="mlx"):
        self.T, self.backend = T, backend
        w = ctx_weights(REQ_WIDTH)
        if backend == "mlx":
            self.w = {k: mx.array(v) for k, v in w.items()}
            mx.eval(*self.w.values())
        else:
            self.w = w
        g = np.random.default_rng(SEED + 1)
        C0 = (g.standard_normal((T, REQ_WIDTH)) * 0.5).astype(np.float32)
        self.C = (C0 / (1.0 + np.abs(C0))).astype(np.float32)
        self.pos = 0
        self.expert_rows = np.zeros(EXPERTS, np.int64)

    def _insert(self, Xn):
        n = Xn.shape[0]
        if n >= self.T:
            self.C[:] = Xn[-self.T:]
            self.pos = 0
            return np.arange(self.T)
        idx = (self.pos + np.arange(n)) % self.T
        self.C[idx] = Xn
        self.pos = int((self.pos + n) % self.T)
        return idx

    def solve(self, X):
        Xn = (X / (1.0 + np.abs(X))).astype(np.float32)
        idx = self._insert(Xn)
        G = self._mlx(idx) if self.backend == "mlx" else self._np(idx)
        return np.nan_to_num(G).astype(np.float32)

    def _mlx(self, idx):
        w = self.w
        H = mx.array(self.C) @ w["Ein"]
        e = mx.argmax(H @ w["Rt"], axis=1)
        mx.eval(e)
        en = np.asarray(memoryview(e))
        self.expert_rows = np.bincount(en, minlength=EXPERTS)
        order = np.argsort(en, kind="stable").astype(np.uint32)
        Hs = mx.take(H, mx.array(order), axis=0)
        parts, start = [], 0
        for i in range(EXPERTS):
            c = int(self.expert_rows[i])
            if c:
                parts.append(mx.maximum(Hs[start:start + c] @ w["W1"][i], 0.0) @ w["W2"][i])
                start += c
        inv = np.argsort(order, kind="stable").astype(np.uint32)
        Z = H + mx.take(mx.concatenate(parts), mx.array(inv), axis=0)
        Gm = (Z.T @ Z) * (1.0 / self.T)
        G = mx.take(Z, mx.array(idx.astype(np.uint32)), axis=0) @ (Gm @ w["Og"])
        mx.eval(G)
        return np.asarray(memoryview(G))

    def _np(self, idx):
        w = self.w
        with np.errstate(all="ignore"):
            H = self.C @ w["Ein"]
            en = np.argmax(H @ w["Rt"], axis=1)
            self.expert_rows = np.bincount(en, minlength=EXPERTS)
            Y = np.empty_like(H)
            for i in range(EXPERTS):
                sel = np.nonzero(en == i)[0]
                if sel.size:
                    Y[sel] = np.maximum(H[sel] @ w["W1"][i], 0.0) @ w["W2"][i]
            Z = H + Y
            Gm = (Z.T @ Z) / self.T
            return Z[idx] @ (Gm @ w["Og"])


def _ticks(rows, count, seed=3):
    g = np.random.default_rng(seed)
    out = []
    for t in range(count):
        X = (g.standard_normal((rows, REQ_WIDTH)) * 2.0).astype(np.float32)
        X[:, 0] = np.arange(rows)
        X[:, 1] = np.arange(rows) % TEAMS
        out.append(X)
    return out


def check(rows):
    ok = True

    def report(name, good, detail):
        nonlocal ok
        ok = ok and good
        print(("PASS" if good else "FAIL") + f" {name}: {detail}", flush=True)

    w = weights(REQ_WIDTH)
    X = _ticks(max(rows, 256), 1, seed=5)[0]
    Xn = X / (1.0 + np.abs(X))
    if mx is not None:
        a, b = scores_mlx(Xn, w), scores_mlx(Xn, w)
        report("scores_mlx deterministic", np.array_equal(a, b),
               f"two runs bit-identical on {Xn.shape[0]} rows")
        ref = scores_moe_ref(Xn, w)
        err = float(np.max(np.abs(a - ref)) / (np.max(np.abs(ref)) + 1e-30))
        report("scores_mlx == numpy MoE reference", err < 1e-3,
               f"max rel err {err:.2e}, pick agreement "
               f"{float(np.mean(np.argmax(a, 1) == np.argmax(ref, 1))):.3f}")
        same = int(np.sum(np.argmax(a, 1) == np.argmin(a, 1)))
        report("scores_mlx policies disjoint", same == 0,
               f"{same}/{a.shape[0]} rows where nearest == inverted")
    else:
        print("skip scores_mlx checks: mlx not importable here", flush=True)
    g = scores_np(Xn, w)
    same = int(np.sum(np.argmax(g, 1) == np.argmin(g, 1)))
    report("scores_np policies disjoint", same == 0,
           f"{same}/{g.shape[0]} rows where nearest == inverted")

    stream = _ticks(rows, 3, seed=7)
    if mx is not None:
        s1, s2 = CtxSolver(), CtxSolver()
        for X in stream:
            a, b = s1.solve(X), s2.solve(X)
        report("ctx solve deterministic", np.array_equal(a, b),
               f"two fresh solvers, same 3-tick stream, bit-identical scores")
        sn = CtxSolver(backend="np")
        for X in stream:
            c = sn.solve(X)
        err = float(np.max(np.abs(a - c)) / (np.max(np.abs(c)) + 1e-30))
        agree = float(np.mean(np.argmax(a, 1) == np.argmax(c, 1)))
        report("ctx solve mlx == numpy", err < 5e-3 and agree == 1.0,
               f"max rel err {err:.2e}, pick agreement {agree:.3f}")
        same = int(np.sum(np.argmax(a, 1) == np.argmin(a, 1)))
        report("ctx policies disjoint", same == 0,
               f"{same}/{a.shape[0]} rows where nearest == inverted")
        mean = float(s1.expert_rows.mean())
        report("rows per expert at floor", mean >= 512,
               f"mean {mean:.0f} (design operating point T*k/E >= 512), "
               f"min {int(s1.expert_rows.min())}")
    else:
        print("skip ctx solve checks: mlx not importable here", flush=True)
    return ok


def bench(reps, rows, T):
    backends = (["mlx"] if mx is not None else []) + ["np"]
    fl = ctx_flops(T, rows)
    print(f"bench: T={T} RES={RES} FF={FF} experts={EXPERTS} rows/tick={rows} "
          f"flops/solve={fl/1e9:.1f} GFLOP "
          f"weights={(EXPERTS*2*RES*FF + REQ_WIDTH*RES + RES*EXPERTS + RES*TEAMS)*4/1e6:.0f} MB",
          flush=True)
    for backend in backends:
        s = CtxSolver(T, backend)
        n = reps if backend == "mlx" else max(3, reps // 4)
        stream = _ticks(rows, n + 2, seed=9)
        for X in stream[:2]:
            s.solve(X)
        times = []
        for X in stream[2:]:
            t0 = time.perf_counter()
            s.solve(X)
            times.append(time.perf_counter() - t0)
        med = float(np.median(times))
        print(f"bench: {backend:4s} median {med*1e3:8.2f} ms  min {min(times)*1e3:8.2f} ms  "
              f"{fl/med/1e9:8.1f} GFLOP/s over {n} solves", flush=True)


def attach(peer):
    while True:
        try:
            return Mesh()
        except RuntimeError as e:
            print(f"worker: {e}, retrying attach for peer {peer}", flush=True)
            time.sleep(2.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="nearest")
    ap.add_argument("--peer", type=int, default=0)
    ap.add_argument("--secs", type=float, default=0.0)
    ap.add_argument("--maxrows", type=int, default=4032)
    ap.add_argument("--ctx", type=int, default=CTX)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--trace", default="")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--bench", type=int, default=0)
    ap.add_argument("--rows", type=int, default=64)
    a = ap.parse_args()

    if a.check:
        sys.exit(0 if check(a.rows) else 1)
    if a.bench:
        bench(a.bench, a.rows, a.ctx)
        return

    pick_of = POLICIES.get(a.policy, np.argmax)
    m = attach(a.peer)
    rx = Reassembler(REQ, REQ_WIDTH, a.maxrows, m.usable)
    tx = TxWindow(m)
    w = weights(REQ_WIDTH)
    solver = CtxSolver(a.ctx) if mx is not None else None
    backend = f"mlx-ctx(T={a.ctx},RES={RES},FF={FF})" if solver else "numpy"
    print(f"worker: policy={a.policy} backend={backend} peer={a.peer} "
          f"usable={m.usable} req_rows/slot={rows_per_slot(m.usable, REQ_WIDTH)} "
          f"resp_rows/slot={rows_per_slot(m.usable, RESP_WIDTH)}", flush=True)

    tr = open(a.trace, "w", buffering=1) if a.trace else None
    if tr:
        tr.write("req,tick,live,prog,pick0,pick1,pick2,pick3,pick4,"
                 "held0,held1,held2,held3,held4,d1,d2,x1,x2,y1,y2\n")

    t0, served, blocks, short = time.time(), 0, 0, 0
    while a.secs <= 0.0 or time.time() - t0 < a.secs:
        done, src = None, a.peer
        for buf, s in m.read(dtype=np.uint8):
            d = rx.feed(buf)
            if d:
                done, src = d, s
        if done is None:
            time.sleep(0.0002)
            continue
        n = done["rows"]
        X = rx.stage[:n]
        if solver is not None:
            G = solver.solve(X)
        else:
            G = scores_np(X / (1.0 + np.abs(X)), w)
        pick = pick_of(G, axis=1).astype(np.float32)
        out = np.zeros((n, RESP_WIDTH), np.float32)
        out[:, 0] = X[:, 0]
        out[:, 1] = pick
        out[:, 2:2 + TEAMS] = G
        out[:, 7] = float(done["req_id"])
        took, chunks = tx.send(RESP, done["req_id"], done["tick"], out, src)
        short += took < chunks
        served += n
        blocks += 1
        if tr:
            hist = np.bincount(pick.astype(np.int32), minlength=TEAMS).tolist()
            back = np.bincount(np.clip(X[:, 15], 0, TEAMS - 1).astype(np.int32),
                               minlength=TEAMS).tolist()
            col = lambda t, c: float(X[X[:, 1] == t, c].mean()) if (X[:, 1] == t).any() else 0.0
            tr.write(",".join(str(v) for v in
                     [done["req_id"], done["tick"], int((X[:, 1] > 0).sum()),
                      float(X[0, 12])] + hist + back +
                     [col(1, 11), col(2, 11), col(1, 5), col(2, 5), col(1, 6), col(2, 6)]) + "\n")
        if not a.quiet:
            hist = np.bincount(pick.astype(np.int32), minlength=TEAMS).tolist()
            live = int((X[:, 1] > 0).sum())
            back = np.bincount(np.clip(X[:, 15], 0, TEAMS - 1).astype(np.int32),
                               minlength=TEAMS).tolist()
            print(f"worker: req {done['req_id']} tick {done['tick']} rows {n} live {live} "
                  f"-> node {src} chunks {took}/{chunks} picks {hist} held {back}", flush=True)
    print(f"worker: {blocks} blocks, {served} rows, short {short}, dropped {rx.dropped}", flush=True)


main()
