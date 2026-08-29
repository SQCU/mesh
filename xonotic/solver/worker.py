"""The solver worker: node 1 answers the game server's bot-plan requests.

    python3 worker.py [--policy nearest|inverted] [--peer 0] [--secs 0]

Reads request blocks framed by bridge/PORT.md section 2, scores every bot row
against the objective basis, and writes one response block back per block it
completes. See bridge/PORT.md section 4 for the contract this implements.
"""
import argparse, sys, time
import numpy as np
from xonwire import (Mesh, Reassembler, TxWindow, REQ, RESP, REQ_WIDTH,
                     RESP_WIDTH, TEAMS, rows_per_slot)

SEED, EXPERTS, FF, HID = 20260828, 8, 2048, 64
POLICIES = {"nearest": np.argmax, "inverted": np.argmin}

try:
    import mlx.core as mx
except Exception:
    mx = None


def weights(D):
    g = np.random.default_rng(SEED)
    f = lambda *s: (g.standard_normal(s) * (1.0 / np.sqrt(s[-2]))).astype(np.float32)
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
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--trace", default="")
    a = ap.parse_args()

    pick_of = POLICIES.get(a.policy, np.argmax)
    m = attach(a.peer)
    rx = Reassembler(REQ, REQ_WIDTH, a.maxrows, m.usable)
    tx = TxWindow(m)
    w = weights(REQ_WIDTH)
    score = scores_mlx if mx is not None else scores_np
    print(f"worker: policy={a.policy} backend={'mlx' if mx else 'numpy'} peer={a.peer} "
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
        G = score(X / (1.0 + np.abs(X)), w)
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
