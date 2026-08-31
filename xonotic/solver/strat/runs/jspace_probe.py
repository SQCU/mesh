"""j-space linear-probe measurement on REAL server rows. No simulation.

Asks one question: does the TRAINED IR carry linearly decodable game semantics
that a random-initialized encoder and a random projection of the same inputs do
not? Controls: random-init encoder (same architecture, untrained), a random
projection of the raw inputs to the same width, the raw inputs themselves, and
shuffled labels (which must fail, or the probes are lying).

Provenance: one real cross-RDMA Game-2 run's telemetry, which logs the model's
OWN input `x`, its belief `beta`, its instrument descriptors `z`, its hierarchy
rows and its IR. Nothing here is simulated.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--telemetry", required=True)
ap.add_argument("--root", default=os.path.expanduser("~/dox/mesh/xonotic"))
ap.add_argument("--epochs", default="1,4,12",
                help="comma-separated training budgets; the IR is probed at each")
ap.add_argument("--out", required=True)
args = ap.parse_args()
sys.path.insert(0, args.root)
sys.path.insert(0, os.path.join(args.root, "payload", "tools"))

import mlx.core as mx
from solver.strat.estimator import StrategyEstimator, StrategyState, strategy_forward, IR_WIDTH
from solver.strat.online import OnlineLearner, architecture_fingerprint
from solver.strat.runtime import CartSnapshot, GameContext
from solver.strat.game import team_nimbers, Cart

L_LEVELS = 8
lines = [json.loads(l) for l in open(args.telemetry) if l.strip()]
# The responder logs the full model arrays only on sampled ticks; the probe
# needs the model's own input, so unsampled lines are not usable here.
_total = len(lines)
lines = [l for l in lines if "x" in l.get("model", {})]
if not lines:
    raise SystemExit(f"{args.telemetry}: none of {_total} lines carry the model arrays "
                     "(--model-sample-every produced no sampled ticks)")
if len(lines) < _total:
    print(f"[data] {len(lines)}/{_total} telemetry lines carry the model arrays", flush=True)
KINDS = sorted({a["kind"] for r in lines for a in r["assignments"]})
kix = {k: i for i, k in enumerate(KINDS)}


def rebuild(line):
    m, k = line["model"], int(line["k"])
    A = sorted(line["assignments"], key=lambda a: a["row"])
    team_of = [min(max(int(a["team"]) - 1, 0), k - 1) for a in A]
    hier = np.asarray(m["hierarchy"], dtype=np.float32)
    score = np.asarray(m["score"], dtype=np.float32)
    x = np.asarray(m["x"], dtype=np.float32)
    z = np.asarray(m["z"], dtype=np.float32)
    depth = np.array([float(c["depth"]) for c in line["carts"]])
    ctrl = np.array([int(c["ctrl"]) for c in line["carts"]], dtype=np.int64)
    control = np.where(ctrl >= 1, ctrl - 1, -1)
    snap = CartSnapshot(pos=np.clip(depth, 0, 1) * L_LEVELS, control=control,
                        banked=np.zeros(k), levels=L_LEVELS)
    ctx = GameContext(tuple(range(k)), tuple(team_of), L_LEVELS)
    state = StrategyState(
        x, np.asarray(m["beta"], dtype=np.float32), z,
        # E10 GAP: per-(player, instrument) relation rows are not in telemetry;
        # they feed the mixing head only, never the IR probed here.
        np.zeros((x.shape[0], z.shape[0], 16), dtype=np.float32),
        hier, hier[:, 5] > 0.5, np.asarray(m["w"], dtype=np.float32),
        None, ctx.teams, ctx.team_of, score > -1e8,
    )
    carts = [Cart(None if c < 0 else int(c), int(np.floor(d))) for d, c in zip(snap.pos, control)]
    nim = team_nimbers(carts, ctx.teams)
    succ = {int(t): float(v) for t, v in line.get("SUCC", [])}
    n = len(team_of)
    weapons = x[:, 24:48]
    wvar = weapons.var(0)
    wbit = int(np.argmax(wvar))
    meta = dict(
        line=np.full(n, line["resp_id"]), team=np.asarray(team_of, float),
        own_nimber=np.array([float(nim.get(t, 0)) for t in team_of]),
        max_rival=np.array([max([float(nim[o]) for o in ctx.teams if o != t], default=0.0) for t in team_of]),
        hier_margin=hier[:, 3].astype(float), is_pw=hier[:, 5].astype(float),
        pw_team=np.full(n, float(np.argmax(hier[:, 5])) if hier[:, 5].any() else -1.0),
        cart_depth_total=np.full(n, float(depth.sum())), cart0_depth=np.full(n, float(depth[0])),
        n_controlled=np.full(n, float((control >= 0).sum())),
        succ_denial=np.array([succ.get(t + 1, succ.get(t, 0.0)) for t in team_of]),
        kind=np.array([float(kix[a["kind"]]) for a in A]),
        gain=np.array([float(a["gain"]) for a in A]),
        lane=np.array([float(a["lane"]) for a in A]),
        logp=np.array([float(a["target_logp"]) for a in A]),
        # --- E9 targets: the per-player resource state that used to be zeroed ---
        health=x[:, 8].astype(float), armor=x[:, 9].astype(float), ammo=x[:, 10].astype(float),
        speed=np.linalg.norm(x[:, 14:17], axis=1).astype(float),
        dist_to_cart=x[:, 22].astype(float),
        n_weapons=weapons.sum(1).astype(float),
        weapon_bit=weapons[:, wbit].astype(float),
    )
    actions = np.asarray([int(a["action"]) for a in A], dtype=np.int32)
    blogp = np.asarray([float(a["behavior_logp"]) for a in A], dtype=np.float32)
    return dict(state=state, context=ctx, snapshot=snap, actions=actions,
                behavior_logp=blogp, meta=meta, k=k, l=n)


recs = [rebuild(l) for l in lines]
META = {key: np.concatenate([r["meta"][key] for r in recs]) for key in recs[0]["meta"]}
RAW = np.concatenate([np.c_[r["state"].x, r["state"].beta] for r in recs], 0).astype(np.float64)
print(f"[data] lines={len(recs)} rows={RAW.shape[0]} raw_dims={RAW.shape[1]} kinds={KINDS}")


def new_estimator(seed):
    return StrategyEstimator.for_runtime(recs[0]["k"], recs[0]["l"], seed=seed)


def ir_of(est):
    return np.concatenate(
        [np.asarray(strategy_forward(est, r["state"], r["state"].w)["ir"], dtype=np.float64)
         for r in recs], 0)


def train(est, epochs, learner=None):
    """Drive the SHIPPED loop (observe/flush + replay ring), not a bespoke one."""
    learner = learner or OnlineLearner(est, checkpoint=None, learning_rate=3e-4)
    history = []
    for epoch in range(epochs):
        previous = None
        for cur in recs:
            m_cur = cur["state"].z.shape[0]
            if previous is not None:
                seg_break = previous["m"] != m_cur
                if seg_break:
                    learner.flush(previous["state"], previous["snapshot"], terminal=True)
                previous["w_out"] = cur["state"].w
                got = learner.observe(previous, cur["state"], cur["snapshot"],
                                      terminal=seg_break)
                if got:
                    history.append(got["loss"])
            previous = dict(context=cur["context"], state=cur["state"],
                            snapshot=cur["snapshot"], cartstate=cur["snapshot"],
                            w_in=cur["state"].w, w_out=cur["state"].w,
                            actions=cur["actions"], behavior_logp=cur["behavior_logp"],
                            m=m_cur)
        got = learner.flush(previous["state"], previous["snapshot"], terminal=True)
        if got:
            history.append(got["loss"])
        print(f"   epoch {epoch}: updates={learner.updates} steps/transition="
              f"{learner.gradient_steps / max(1, learner.transitions):.2f} "
              f"replay={len(learner.replay)} loss={np.mean(history[-len(recs):]):.5f}", flush=True)
    return learner, history


rng = np.random.default_rng(0)
lids = np.unique(META["line"]); perm = rng.permutation(lids)
trs = set(perm[:int(0.6 * len(perm))].tolist())
tr = np.array([l in trs for l in META["line"]]); te = ~tr


def r2(F, y, lam=1e-3):
    ok = np.isfinite(y); a, b = tr & ok, te & ok
    if a.sum() < 10 or b.sum() < 10 or np.std(y[b]) < 1e-9: return None
    A = np.c_[F[a], np.ones(a.sum())]; B = np.c_[F[b], np.ones(b.sum())]
    mu, sd = A[:, :-1].mean(0), A[:, :-1].std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)   # a train-constant column carries no signal;
    A[:, :-1] = (A[:, :-1] - mu) / sd   # scaling it by 1/eps only manufactures inf
    B[:, :-1] = (B[:, :-1] - mu) / sd
    w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y[a])
    return float(round(1.0 - np.sum((y[b] - B @ w) ** 2) / np.sum((y[b] - y[b].mean()) ** 2), 4))


def acc(F, y, lam=1e-3):
    y = y.astype(int)
    cls = np.unique(y[tr])
    if len(cls) < 2: return None
    A = np.c_[F[tr], np.ones(tr.sum())]; B = np.c_[F[te], np.ones(te.sum())]
    mu, sd = A[:, :-1].mean(0), A[:, :-1].std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    A[:, :-1] = (A[:, :-1] - mu) / sd
    B[:, :-1] = (B[:, :-1] - mu) / sd
    Y = (y[tr][:, None] == cls[None, :]).astype(float)
    W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ Y)
    pred = cls[np.argmax(B @ W, 1)]
    c = np.array([(y[te] == v).sum() for v in np.unique(y[te])], float)
    return {"acc": round(float(np.mean(pred == y[te])), 4),
            "majority": round(float(c.max() / c.sum()), 4), "n_classes": int(len(cls))}


REG = ["own_nimber", "max_rival", "hier_margin", "cart_depth_total", "cart0_depth",
       "n_controlled", "succ_denial", "gain", "lane", "logp",
       "health", "armor", "ammo", "speed", "dist_to_cart", "n_weapons"]
CLS = ["is_pw", "kind", "pw_team", "team", "weapon_bit"]


def score(F, shuffled=False):
    out = {}
    for t in REG:
        y = META[t].astype(float).copy()
        if shuffled: rng.shuffle(y)
        out[t] = {"r2": r2(F, y)}
    for t in CLS:
        y = META[t].astype(float).copy()
        if shuffled: rng.shuffle(y)
        out[t] = acc(F, y)
    return out


budgets = [int(e) for e in str(args.epochs).split(",") if e.strip()]
print(f"[train] replaying the REAL transitions through the shipped loop; budgets={budgets}",
      flush=True)
est_t = new_estimator(0)
learner, history = None, []
sweep = {}
done = 0
for budget in budgets:
    learner, extra = train(est_t, budget - done, learner=learner)
    history += extra
    done = budget
    sweep[f"IR_trained_{budget}ep"] = {
        "updates": learner.updates,
        "steps_per_transition": round(learner.gradient_steps / max(1, learner.transitions), 3),
        "loss": round(float(np.mean(history[-len(recs):])), 5),
        "ir": ir_of(est_t),
    }
IR_T = sweep[f"IR_trained_{budgets[-1]}ep"]["ir"]
IR_R = ir_of(new_estimator(0))
IR_R7 = ir_of(new_estimator(7))
P = RAW @ (rng.standard_normal((RAW.shape[1], IR_WIDTH)) / np.sqrt(RAW.shape[1]))


def rank(M): return int(np.linalg.matrix_rank(M - M.mean(0), tol=1e-6))


out = {
    "provenance": {
        "telemetry": os.path.basename(args.telemetry), "n_lines": len(recs),
        "n_player_rows": int(RAW.shape[0]), "simulation_used": False,
        "environment": sorted({l.get("environment") for l in lines}),
        "training": f"replay of the real logged transitions through the shipped "
                    f"observe/flush loop; budgets={budgets} epochs, "
                    f"{learner.updates} updates at the deepest",
        "architecture": architecture_fingerprint(learner.bundle),
        "replay": {"size": len(learner.replay), "capacity": learner.replay.capacity,
                   "mb": round(learner.replay.nbytes / (1 << 20), 2),
                   "transitions": learner.transitions,
                   "gradient_steps": learner.gradient_steps,
                   "steps_per_transition": round(
                       learner.gradient_steps / max(1, learner.transitions), 3)},
        "split": "by telemetry line, 60/40, seed 0",
        "train_rows": int(tr.sum()), "test_rows": int(te.sum()),
    },
    "ir_width": {"d": IR_WIDTH, "spec_requirement": ">=128d"},
    "ranks": {"raw_inputs": rank(RAW), "raw_dims": int(RAW.shape[1]),
              "x_only": rank(RAW[:, :48]), "IR_trained": rank(IR_T),
              "IR_random_init": rank(IR_R), "n_rows": int(RAW.shape[0])},
    "ir_scale": {"trained_absmax": round(float(np.abs(IR_T).max()), 4),
                 "trained_std": round(float(IR_T.std()), 4),
                 "random_init_absmax": round(float(np.abs(IR_R).max()), 4)},
    "loss": {"first": round(float(np.mean(history[:len(recs)])), 5),
             "last": round(float(np.mean(history[-len(recs):])), 5)},
    "training_budget_sweep": {
        name: {"updates": v["updates"], "steps_per_transition": v["steps_per_transition"],
               "loss": v["loss"], "probes": score(v["ir"])}
        for name, v in sweep.items()
    },
    "probes": {
        "IR_trained": score(IR_T),
        "IR_random_init_s0": score(IR_R),
        "IR_random_init_s7": score(IR_R7),
        "control_randproj_128d": score(P),
        "control_shuffled_labels_IR_trained": score(IR_T, shuffled=True),
        "reference_raw_inputs": score(RAW),
    },
}
json.dump(out, open(args.out, "w"), indent=2)
print(json.dumps({k: out[k] for k in ("provenance", "ranks", "loss")}, indent=2))
print("WROTE", args.out)
