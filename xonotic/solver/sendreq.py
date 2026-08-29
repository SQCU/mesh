"""A stand-in for the engine: frame request blocks, print the plans that come back.

    python3 sendreq.py [--peer 1] [--bots 64] [--ticks 4]

Rows are the width-16 request layout of bridge/PORT.md section 2, generated from
a fixed seed so two runs against different --policy workers are comparable.
"""
import argparse, json, time
import numpy as np
from xonwire import (Mesh, Reassembler, TxWindow, REQ, RESP, REQ_WIDTH,
                     RESP_WIDTH, TEAMS)


def features(bots, tick, rng):
    X = (rng.standard_normal((bots, REQ_WIDTH)) * 0.5).astype(np.float32)
    X[:, 0] = np.arange(bots)
    X[:, 1] = np.arange(bots) % TEAMS
    X[:, 2] = 100.0 - (np.arange(bots) % 7) * 5.0
    X[:, 12] = (tick % 100) / 100.0
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peer", type=int, default=1)
    ap.add_argument("--bots", type=int, default=64)
    ap.add_argument("--ticks", type=int, default=4)
    ap.add_argument("--wait", type=float, default=2.0)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    m = Mesh()
    rx = Reassembler(RESP, RESP_WIDTH, a.bots, m.usable)
    tx = TxWindow(m)
    rng = np.random.default_rng(11)
    picks, replies = {}, 0
    for t in range(a.ticks):
        X = features(a.bots, t, rng)
        took, chunks = tx.send(REQ, t + 1, t, X, a.peer)
        if not a.json:
            print(f"send: req {t+1} rows {a.bots} chunks {took}/{chunks}", flush=True)
        t1 = time.time()
        while time.time() - t1 < a.wait:
            for buf, s in m.read(dtype=np.uint8):
                d = rx.feed(buf)
                if d:
                    R = rx.stage[:d["rows"]]
                    replies += 1
                    picks[d["req_id"]] = R[:, 1].astype(int).tolist()
                    if not a.json:
                        hist = np.bincount(R[:, 1].astype(int), minlength=TEAMS).tolist()
                        print(f"recv: req {d['req_id']} tick {d['tick']} rows {d['rows']} "
                              f"picks {hist} first {R[:6, 1].astype(int).tolist()}", flush=True)
            time.sleep(0.001)
            if replies > t:
                break
    out = dict(sent=a.ticks, replies=replies, dropped=rx.dropped,
               picks={str(k): v for k, v in picks.items()})
    print(json.dumps(out) if a.json else f"send: {replies}/{a.ticks} answered, dropped {rx.dropped}",
          flush=True)


main()
