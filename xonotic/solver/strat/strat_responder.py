"""Live strategy responder: node-1 coprocessor for the Xonotic cartserver.

Reads the engine's per-tick gather streams off the mesh (obs w40, cart w12, evt w6),
runs the shared-weight strategy policy (trained policy_ckpt.npz when its (k,j) dims
match the live match, else an untrained W_all sized to the live match), and scatters
the 8-column per-player instrument weights back to node 0 where
havocbot_goalrating_strategy applies them as navigation_routerating biases.

Telemetry: one JSON line per completed strategy tick -> runs/cartserver_telemetry.jsonl
(cartstate, PW, per-player scatter actually applied).

Run on the mini:  cd ~/mesh/xonotic && ~/.venv-mesh/bin/python3 -m solver.strat.strat_responder --secs 90
"""
import argparse, json, os, sys, time
import numpy as np

# run as: cd ~/mesh/xonotic && python -m solver.strat.strat_responder
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "rdma"))          # mesh.py
sys.path.insert(0, os.path.join(_HERE, "..", ".."))                  # solver pkg parent
sys.path.insert(0, os.path.join(_HERE, "..", "..", "payload", "tools"))  # strategy_io_schema

from solver.xonwire import (Mesh, Reassembler, TxWindow, parse_hdr, pack_hdr,
                            HDRSZ, rows_per_slot, REQ, RESP)
from solver.strat.cartsim import CartSim, CartState, decode_instrument
from solver.strat.estimator import StrategyEstimator, state_from_cartsim
from strategy_io_schema import (OBS, CS, SC, encode_target, projected_winner,
                                succession, carts_from_rows)

OBS_W, CART_W, EVT_W, RESP_W = 40, 12, 6, 8
CKPT = os.path.join(_HERE, "runs", "policy_ckpt.npz")
TELEM = os.path.join(_HERE, "runs", "cartserver_telemetry.jsonl")
L_LEVELS = 8


def load_ckpt_into(est, path):
    """Load trained W_all into an estimator whose dims match the checkpoint."""
    import mlx.core as mx
    d = np.load(path, allow_pickle=True)
    m = {k: d[k] for k in d.files}
    est.qkv.W_q = mx.array(m["qkv.W_q"]); est.qkv.W_k = mx.array(m["qkv.W_k"])
    est.qkv.W_v = mx.array(m["qkv.W_v"])
    est.head.norm_weight = mx.array(m["head.norm_weight"])
    est.head.w_gate = mx.array(m["head.w_gate"]); est.head.w_up = mx.array(m["head.w_up"])
    est.head.w_down = mx.array(m["head.w_down"])
    return sorted(m.keys())


class EstCache:
    def __init__(self):
        self.key = None; self.est = None; self.trained = False

    def get(self, k, j, l):
        key = (k, j, l)
        if key == self.key:
            return self.est, self.trained
        sim = CartSim(k, j, l, team_of=[p % k for p in range(l)], L=L_LEVELS, seed=20260829)
        est = StrategyEstimator.for_cartsim(sim, seed=20260829)
        trained = False
        # checkpoint was trained at k=2,j=3 (d_x=k+3=5,d_beta=2j=6,d_z=4+k=6,M=2j+1=7)
        if os.path.exists(CKPT) and k == 2 and j == 3:
            try:
                load_ckpt_into(est, CKPT); trained = True
            except Exception as e:
                print(f"[responder] ckpt load failed ({e}); untrained W_all", flush=True)
        self.key, self.est, self.trained = key, est, trained
        print(f"[responder] estimator sized k={k} j={j} l={l} M={est.M} "
              f"{'TRAINED ckpt' if trained else 'UNTRAINED W_all'}", flush=True)
        return est, trained


def build_cartstate(cart_rows, k):
    j = cart_rows.shape[0]
    pos = np.zeros(j); control = np.full(j, -1, dtype=np.int64)
    depth_frac = np.zeros(j)
    for c in range(j):
        df = float(cart_rows[c, CS["DEPTH"]])
        depth_frac[c] = df
        pos[c] = np.clip(df, 0, 1) * L_LEVELS
        ctrl = int(round(cart_rows[c, CS["CTRL"]]))
        control[c] = (ctrl - 1) if ctrl >= 1 else -1
    cs = CartState(pos=pos, control=control, banked=np.zeros(k), L=L_LEVELS, t=0,
                   highwater=np.floor(pos).astype(np.int64))
    return cs, depth_frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--secs", type=float, default=90.0)
    ap.add_argument("--peer-node", type=int, default=0)   # engine is on node 0
    args = ap.parse_args()

    m = Mesh()
    print(f"[responder] mesh attached slots={m.slots} usable={m.usable}", flush=True)
    ra_obs = Reassembler(REQ, OBS_W, 4096, m.usable)
    ra_cart = Reassembler(REQ, CART_W, 4096, m.usable)
    ra_evt = Reassembler(REQ, EVT_W, 4096, m.usable)
    tx = TxWindow(m)

    est_cache = EstCache()
    carry_w = None; carry_key = None
    resp_id = 0
    last_obs = None; last_cart = None
    telem = open(TELEM, "w")
    nt_written = 0
    stats = dict(slots=0, obs=0, cart=0, evt=0, resp=0)

    t0 = time.time(); last_report = t0
    while time.time() - t0 < args.secs:
        got_any = False
        for buf, src in m.read(np.uint8):
            got_any = True; stats["slots"] += 1
            h = parse_hdr(buf)
            if h is None:
                continue
            w = h["width"]
            if w == OBS_W:
                r = ra_obs.feed(buf)
                if r: last_obs = (r, ra_obs.stage[:r["rows"]].copy()); stats["obs"] += 1
            elif w == CART_W:
                r = ra_cart.feed(buf)
                if r: last_cart = (r, ra_cart.stage[:r["rows"]].copy()); stats["cart"] += 1
            elif w == EVT_W:
                r = ra_evt.feed(buf)
                if r: stats["evt"] += 1

        # when we have a fresh cart+obs pair, run the policy and scatter
        if last_cart is not None and last_obs is not None:
            (ch, cart_rows) = last_cart
            (oh, obs_rows) = last_obs
            last_cart = None  # consume; obs kept until next cart to keep pairing simple
            l = obs_rows.shape[0]
            teams_present = [int(round(obs_rows[i, OBS["TEAM"]])) for i in range(l)]
            cart_ctrl = [int(round(cart_rows[c, CS["CTRL"]])) for c in range(cart_rows.shape[0])]
            k = max([2] + [t for t in teams_present if t >= 1] + [t for t in cart_ctrl if t >= 1])
            j = cart_rows.shape[0]
            team_of = [max(0, min(k - 1, (t - 1) if t >= 1 else 0)) for t in teams_present]

            est, trained = est_cache.get(k, j, l)
            key = (k, j, l)
            if carry_w is None or carry_key != key:
                carry_w = np.zeros((l, est.M), dtype=np.float32); carry_key = key

            sim = CartSim(k, j, l, team_of=team_of, L=L_LEVELS, seed=20260829)
            cs, depth_frac = build_cartstate(cart_rows, k)
            state = state_from_cartsim(sim, cs, w=carry_w)

            res = est.forward(state)
            import mlx.core as mx
            action = np.asarray(memoryview(res.action)).astype(int).reshape(l)
            w_next = np.asarray(memoryview(res.w_next)).astype(np.float32).reshape(l, est.M)
            carry_w = w_next

            # PW / SUCC over the real cartstate (canonical Game-1)
            schema_carts = carts_from_rows(cart_rows)
            pw = projected_winner(schema_carts)           # team color 1..k or 0
            succ = succession(schema_carts)

            # build the 8-col scatter, row i -> client edict i+1
            R = np.zeros((l, RESP_W), dtype=np.float32)
            applied = []
            for i in range(l):
                t = teams_present[i]
                if t < 1:                       # empty / spectator slot -> zeros (held)
                    continue
                a = int(action[i])
                kind, cart = decode_instrument(a, j)
                amp = float(np.clip(1.0 + w_next[i, a], 0.1, 3.0))
                if kind == "idle" or cart < 0:
                    tgt, gain, lane = 0.0, 0.0, 0.0
                else:
                    tgt = float(encode_target("cart", cart))
                    gain = amp
                    df = float(np.clip(depth_frac[cart], 0, 1))
                    lane = df if kind == "push_cart" else float(min(1.0, df + 0.15))
                R[i, SC["TARGET"]]  = tgt
                R[i, SC["GAIN"]]    = gain
                R[i, SC["LANE"]]    = lane
                R[i, SC["HUNT"]]    = 0.0
                R[i, SC["EXPLORE"]] = 0.0
                R[i, SC["COMMIT"]]  = 1.0
                R[i, SC["SPAWN"]]   = 0.0
                R[i, SC["LEAD"]]    = 1.0 if (t == pw) else 0.0
                if len(applied) < 8:
                    applied.append(dict(edict=i + 1, team=t, kind=kind, cart=cart,
                                        target=tgt, gain=round(gain, 4), lane=round(lane, 4),
                                        lead=int(t == pw)))

            resp_id += 1
            took, chunks = tx.send(RESP, resp_id, ch["tick"], R, args.peer_node)
            stats["resp"] += 1

            line = dict(
                t=round(time.time() - t0, 3), resp_id=resp_id, req_tick=int(ch["tick"]),
                k=k, j=j, l_rows=l, n_players=int(sum(1 for t in teams_present if t >= 1)),
                trained=bool(trained),
                carts=[dict(id=int(round(cart_rows[c, CS["ID"]])),
                            depth=round(float(cart_rows[c, CS["DEPTH"]]), 5),
                            plc_length=round(float(cart_rows[c, CS["LENGTH"]]), 2),
                            plc_s=round(float(cart_rows[c, CS["DEPTH"]]) * float(cart_rows[c, CS["LENGTH"]]), 2),
                            ctrl=int(round(cart_rows[c, CS["CTRL"]])),
                            speed=round(float(cart_rows[c, CS["SPEED"]]), 4),
                            progress=round(float(cart_rows[c, CS["PROGRESS"]]), 4))
                       for c in range(j)],
                PW=int(pw), SUCC=[[int(a), round(float(b), 4)] for a, b in succ],
                scatter=applied,
            )
            telem.write(json.dumps(line) + "\n"); telem.flush(); nt_written += 1

        if not got_any:
            time.sleep(0.003)
        if time.time() - last_report >= 5:
            print(f"[responder] {int(time.time()-t0)}s stats={stats} telem_lines={nt_written} "
                  f"resp_id={resp_id}", flush=True)
            last_report = time.time()

    telem.close()
    print(f"[responder] DONE stats={stats} telem_lines={nt_written}", flush=True)


if __name__ == "__main__":
    main()
