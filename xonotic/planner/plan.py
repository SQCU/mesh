"""Bot planning as a linear-algebra problem, solved across the mesh.

The server holds bot state. Every tick it sends one feature row per bot to the
coprocessor node, which holds the routed expert weights and the objective basis
resident, and returns each bot's chosen objective. Behaviour is therefore
conditioned on a matrix solve that does not happen on the machine running the
game.

    coprocessor:  python3 plan.py solve  <peer> [secs]
    server:       python3 plan.py play   <peer> [secs] [bots]
"""
import sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, "../../rdma")
from mesh import Mesh

ROLE = sys.argv[1]
PEER = int(sys.argv[2])
SECS = float(sys.argv[3]) if len(sys.argv) > 3 else 15.0
BOTS = int(sys.argv[4]) if len(sys.argv) > 4 else 480

TEAMS   = 5      # objectives: one payload lane per team
EXPERTS = 8      # routed experts
FF      = 2048

m = Mesh(1.0e9)
D = m.usable // 4
SEED = 20260828

def model(d):
    """Deterministic on both sides so the server can audit any row it likes."""
    g = np.random.default_rng(SEED)
    f = lambda *s: mx.array(g.standard_normal(s).astype(np.float32) * (1.0 / np.sqrt(s[-2])))
    return (f(d, EXPERTS), f(EXPERTS, d, FF), f(EXPERTS, FF, d), f(d, TEAMS))

def solve(Xn, R, W1, W2, O):
    """Route each bot to an expert, apply that expert, score the objectives.

    Experts are applied by grouping rows per expert rather than gathering a
    weight matrix per row: gathering would materialise one (D, FF) matrix per
    bot, which is gigabytes for a few hundred bots."""
    n = Xn.shape[0]
    X = mx.array(Xn)
    e = mx.argmax(X @ R, axis=1)
    mx.eval(e)
    en = np.asarray(memoryview(e))
    Y = np.zeros((n, D), dtype=np.float32)
    for i in range(EXPERTS):
        sel = np.nonzero(en == i)[0]
        if sel.size == 0:
            continue
        Yi = mx.maximum(mx.array(Xn[sel]) @ W1[i], 0.0) @ W2[i]
        mx.eval(Yi)
        Y[sel] = np.asarray(memoryview(Yi))
    G = mx.array(Y) @ O                            # (n, TEAMS) objective scores
    pick = mx.argmax(G, axis=1)
    mx.eval(pick, G)
    return pick, G

R, W1, W2, O = model(D)
mx.eval(R, W1, W2, O)
print(f"{ROLE}: D={D} bots={BOTS} teams={TEAMS} experts={EXPERTS}", flush=True)

if ROLE == "solve":
    stage = np.empty((BOTS, D), dtype=np.float32)
    ring, cur, served, t0 = (m.slots // BOTS) * BOTS, 0, 0, time.time()
    while time.time() - t0 < SECS:
        n, src = 0, None
        for buf, s in m.read():
            stage[n] = buf[:D]; src = s; n += 1
            if n >= BOTS: break
        if n == 0: continue
        pick, G = solve(stage[:n], R, W1, W2, O)
        out = np.zeros((n, D), dtype=np.float32)
        out[:, 0] = stage[:n, 0]                       # echo the bot id
        out[:, 1] = np.asarray(memoryview(pick)).astype(np.float32)
        out[:, 2:2+TEAMS] = np.asarray(memoryview(G))
        if cur + n > ring: cur = 0
        m.block(cur, n)[:, :D*4] = out.view(np.uint8).reshape(n, D*4)
        sent = 0
        while sent < n:
            sent += m.write(cur + sent, n - sent, src if src is not None else PEER)
        cur += n; served += n
    dt = time.time() - t0
    flops = 2.0 * served * D * FF * 2
    print(f"solve: {served} bots planned, {flops/dt/1e9:.1f} GFLOP/s", flush=True)

else:
    rng = np.random.default_rng(7)
    pos = rng.standard_normal((BOTS, D)).astype(np.float32) * 0.1
    pos[:, 0] = np.arange(BOTS)                        # bot id in element 0
    m.block(0, BOTS)[:, :D*4] = pos.view(np.uint8).reshape(BOTS, D*4)
    On = np.asarray(memoryview(O))                     # same basis as the solver
    objective = np.full(BOTS, -1, dtype=np.int32)
    ticks = planned = switched = 0
    t0 = time.time()
    while time.time() - t0 < SECS:
        n = 0
        while n < BOTS:
            k = m.write(n, BOTS - n, PEER)
            if k == 0: break
            n += k
        got = 0
        for buf, _ in m.read():
            b = int(buf[0])
            if 0 <= b < BOTS:
                pick = int(buf[1])
                if objective[b] != pick:
                    switched += 1
                objective[b] = pick
                # The plan moves the bot. Its new state is what gets planned on
                # next tick, so behaviour and the solve condition each other.
                pos[b, 1:] += 0.35 * On[1:, pick]
                pos[b, 1:] *= 0.98
            got += 1; planned += 1
        if got:
            m.block(0, BOTS)[:, :D*4] = pos.view(np.uint8).reshape(BOTS, D*4)
            ticks += 1
            if ticks % 2000 == 0:
                hist = np.bincount(objective[objective >= 0], minlength=TEAMS)
                print(f"  tick {ticks:5d}  bots per objective {hist.tolist()}"
                      f"  switches {switched}", flush=True)
    dt = time.time() - t0
    hist = np.bincount(objective[objective >= 0], minlength=TEAMS)
    print(f"play: {planned} plans in {ticks} ticks, {planned/dt:.0f} bot-plans/s, "
          f"final objective split {hist.tolist()}, unplanned {(objective<0).sum()}", flush=True)
