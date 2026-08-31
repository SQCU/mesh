"""j-space linear-probe measurement on REAL server rows. No simulation.

Asks one question: does the TRAINED IR carry linearly decodable game semantics
that a random-initialized encoder and a random projection of the same inputs do
not?  Controls, all reported side by side: a random-init encoder of the same
architecture (two seeds), a random projection of the raw inputs to the same
width, the raw inputs themselves, and shuffled labels (which MUST fail, or the
probes are lying).

Provenance.  The input is `measure.py rows` output over a real dedicated-server
log: one JSONL record per strategy tick carrying the engine's own `[PLCOBS]`
observation rows, `[PLCCART]` cart rows, `[PLCEVT]` perception events, and the
instrument descriptors `z` / relation rows built by the SAME `build_instruments`
the live operator calls.  `beta` is recomputed here by replaying the same
`LiveBelief` over the same event stream, so every array fed to the operator is
the array the operator would have been fed live.  Nothing is simulated and no
column is stubbed: R19's rank-4 input was an artifact of a probe that hardcoded
`x[8:16] = 0`, corrected in R25, and this probe reconstructs nothing -- it
builds the state through `state_from_runtime`, the responder's own constructor.

What is and is not tautological.  The IR is a function of `(x, beta)` ONLY --
`z` and the relation rows reach the mixing head, never the encoder.  So every
target is tagged `in_x` (it is literally a column of the operator's input, and
decoding it measures preservation) or not (it is not in the input at all, and
decoding it measures induced structure).  The verdict is read off the second
group.
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--rows", required=True,
                help="measure.py `rows` JSONL over a real server log")
ap.add_argument("--root", default=os.path.expanduser("~/dox/mesh/xonotic"))
ap.add_argument("--epochs", default="1,2,4",
                help="comma-separated training budgets; the IR is probed at each")
ap.add_argument("--limit", type=int, default=0, help="debug: only the first N ticks")
ap.add_argument("--cgt-labels", default=None,
                help="optional JSON {seq: {kind, nimber}} produced by `game_value."
                     "evaluate_cartstate` elsewhere. `game_value` uses `zip(strict=True)`, "
                     "which needs Python >= 3.10; the mesh-mini venv that carries mlx is "
                     "3.9, so on that host the CGT labels are precomputed and read in "
                     "rather than silently dropped. Same function, same rows.")
ap.add_argument("--out", required=True)
args = ap.parse_args()
sys.path.insert(0, args.root)
sys.path.insert(0, os.path.join(args.root, "payload", "tools"))

from strategy_io_schema import CS, EVT, OBS, OBS_WIDTH, CART_WIDTH          # noqa: E402
from solver.strat.estimator import (StrategyEstimator, state_from_runtime,  # noqa: E402
                                    strategy_forward, IR_WIDTH)
from solver.strat.instruments import RELATION_WIDTH                          # noqa: E402
from solver.strat.live_belief import LiveBelief                              # noqa: E402
from solver.strat.online import OnlineLearner, architecture_fingerprint      # noqa: E402
from solver.strat.runtime import CartSnapshot, GameContext, hierarchy_rows   # noqa: E402
from solver.strat.game import team_nimbers, Cart, succession                 # noqa: E402
try:                                                                         # noqa: E402
    from solver.strat.game_value import evaluate_cartstate
except Exception:                                # pragma: no cover - see --cgt-labels
    evaluate_cartstate = None

CGT_LABELS = {}
if args.cgt_labels:
    CGT_LABELS = {int(key): value for key, value in json.load(open(args.cgt_labels)).items()}

L_LEVELS = 8
OBS_LOG_COLUMNS = ("ID", "TEAM", "HEALTH", "ARMOR", "AMMO", "POS_X", "POS_Y", "POS_Z",
                   "VEL_X", "VEL_Y", "VEL_Z", "WEAPONS", "POWER", "TSS", "CELL",
                   "NCART", "NCART_D", "ALIVE", "CONTROL")
CART_LOG_COLUMNS = ("ID", "DEPTH", "LENGTH", "CTRL", "SPEED", "IDLE", "BANKMASK",
                    "PROGRESS", "POS_X", "POS_Y", "POS_Z")
_DISTANCE = 7          # relation column: |instrument position - actor position|


class _Batch:
    """The instrument batch as logged: descriptors, relations, eligibility, kinds.

    `state_from_runtime` needs exactly these four; rebuilding the `Instrument`
    objects would recompute what the log already carries verbatim.
    """
    __slots__ = ("instruments", "descriptors", "relations", "eligible", "kinds", "subjects")

    def __init__(self, kinds, subjects, descriptors, relations, eligible):
        self.kinds = kinds
        self.subjects = subjects
        self.instruments = kinds          # only len() is used
        self.descriptors = descriptors
        self.relations = relations
        self.eligible = eligible


def load(path, limit=0):
    """Stream the rows file into per-tick operator states. One JSON line at a time."""
    belief = LiveBelief()
    recs, kinds_seen = [], set()
    band_median, band_in, link_source = [], 0, set()
    t0 = time.time()
    with open(path) as handle:
        for n, raw in enumerate(handle):
            if limit and len(recs) >= limit:
                break
            line = json.loads(raw)
            obs = np.asarray(line["obs"], dtype=np.float32)
            cart = np.asarray(line["cart"], dtype=np.float32)
            if not len(obs) or not len(cart):
                continue
            rows = np.zeros((obs.shape[0], OBS_WIDTH), dtype=np.float32)
            for i, name in enumerate(line.get("obs_columns", OBS_LOG_COLUMNS)):
                rows[:, OBS[name]] = obs[:, i]
            cart_rows = np.zeros((cart.shape[0], CART_WIDTH), dtype=np.float32)
            for i, name in enumerate(line.get("cart_columns", CART_LOG_COLUMNS)):
                cart_rows[:, CS[name]] = cart[:, i]
            evt = np.asarray(line["evt"], dtype=np.float32)
            if len(evt):
                belief.ingest(evt, EVT)
            beta, diag = belief.beliefs(rows, OBS)
            band_median.append(diag["receptive"]["median"])
            band_in += int(diag["receptive"]["in_band"])
            link_source.add(diag.get("link_source", "n/a"))

            teams_present = np.asarray(rows[:, OBS["TEAM"]], dtype=np.int64)
            cart_ctrl = np.asarray(np.round(cart_rows[:, CS["CTRL"]]), dtype=np.int64)
            k = max([2] + teams_present.tolist() + cart_ctrl[cart_ctrl >= 1].tolist())
            team_of = np.clip(teams_present - 1, 0, k - 1).tolist()
            depth = np.clip(cart_rows[:, CS["DEPTH"]].astype(np.float64), 0, 1) * L_LEVELS
            control = np.where(cart_ctrl >= 1, cart_ctrl - 1, -1)
            snapshot = CartSnapshot(pos=depth, control=control,
                                    banked=np.zeros(k), levels=L_LEVELS)
            context = GameContext(tuple(range(k)), tuple(team_of), L_LEVELS)
            z = np.asarray(line["z"], dtype=np.float32)
            relation = np.asarray(line["relation"], dtype=np.float32)
            eligible = np.asarray(line["eligible"], dtype=bool)
            batch = _Batch(tuple(line["instrument_kinds"]), tuple(line["instrument_subjects"]),
                           z, relation, eligible)
            kinds_seen.update(batch.kinds)
            state = state_from_runtime(context, snapshot, rows, OBS, beta, batch)
            recs.append({"seq": int(line["seq"]), "state": state, "context": context,
                         "snapshot": snapshot, "batch": batch, "k": k,
                         "cart_rows": cart_rows, "rows": rows, "depth": depth,
                         "control": control, "team_of": team_of})
            if (n + 1) % 200 == 0:
                print(f"[data] {n + 1} ticks, {time.time() - t0:.1f}s", flush=True)
    return recs, sorted(kinds_seen), {
        "receptive_median_min": round(float(np.min(band_median)), 6),
        "receptive_median_median": round(float(np.median(band_median)), 6),
        "receptive_median_max": round(float(np.max(band_median)), 6),
        "ticks_with_median_in_5_15_band": int(band_in),
        "ticks": len(band_median),
        "link_source": sorted(link_source),
    }


recs, KINDS, BAND = load(args.rows, args.limit)
if not recs:
    raise SystemExit(f"{args.rows}: no usable ticks")
kix = {kind: i for i, kind in enumerate(KINDS)}
print(f"[data] ticks={len(recs)} rows={sum(len(r['team_of']) for r in recs)} "
      f"kinds={KINDS}", flush=True)
print(f"[belief] {BAND}", flush=True)


def targets(rec):
    """Per-player probe targets, straight off the engine rows and the cart game."""
    context, snapshot, rows = rec["context"], rec["snapshot"], rec["rows"]
    state, batch = rec["state"], rec["batch"]
    n = len(rec["team_of"])
    cart_list = [Cart(None if c < 0 else int(c), int(np.floor(d)))
                 for d, c in zip(rec["depth"], rec["control"])]
    nim = team_nimbers(cart_list, context.teams)
    denial = {team: amount for team, amount in succession(cart_list, context.teams)}
    hier, mask = hierarchy_rows(context, snapshot)
    if CGT_LABELS:
        label = CGT_LABELS[int(rec["seq"])]
        cgt_kind, cgt_nimber = label["kind"], label["nimber"]
    else:
        value = evaluate_cartstate([int(round(float(v))) for v in rec["depth"]],
                                   [int(v) for v in rec["control"]],
                                   list(context.teams), L_LEVELS)
        cgt_kind, cgt_nimber = value.kind, value.nimber
    weapons = state.x[:, 24:48]
    wbit = int(np.argmax(weapons.var(0))) if weapons.shape[0] > 1 else 0
    # instrument-side targets: `z` and `relation` never enter the IR, so these
    # are the least tautological quantities available.
    dist = np.where(batch.eligible, batch.relations[:, :, _DISTANCE], np.inf)
    nearest = np.argmin(dist, axis=1)
    nearest_ok = np.isfinite(dist[np.arange(n), nearest])
    return dict(
        team=np.asarray(rec["team_of"], dtype=float),
        own_nimber=np.array([float(nim.get(t, 0)) for t in rec["team_of"]]),
        max_rival=np.array([max([float(nim[o]) for o in context.teams if o != t], default=0.0)
                            for t in rec["team_of"]]),
        hier_margin=hier[:, 3].astype(float),
        is_pw=mask.astype(float),
        pw_team=np.full(n, float(np.argmax(hier[:, 5])) if hier[:, 5].any() else -1.0),
        succ_denial=np.array([float(denial.get(t, 0.0)) for t in rec["team_of"]]),
        cart_depth_total=np.full(n, float(rec["depth"].sum())),
        cart0_depth=np.full(n, float(rec["depth"][0])),
        n_controlled=np.full(n, float((rec["control"] >= 0).sum())),
        cgt_impartial=np.full(n, float(cgt_kind == "impartial")),
        cgt_nimber=np.full(n, float(cgt_nimber if cgt_nimber is not None else -1.0)),
        health=state.x[:, 8].astype(float),
        armor=state.x[:, 9].astype(float),
        ammo=state.x[:, 10].astype(float),
        speed=np.linalg.norm(state.x[:, 14:17], axis=1).astype(float),
        dist_to_cart=state.x[:, 22].astype(float),
        tss=state.x[:, 18].astype(float),
        alive=state.x[:, 19].astype(float),
        n_weapons=weapons.sum(1).astype(float),
        weapon_bit=weapons[:, wbit].astype(float),
        cell=rows[:, OBS["CELL"]].astype(float),
        n_eligible=batch.eligible.sum(1).astype(float),
        nearest_kind=np.array([float(kix[batch.kinds[int(m)]]) if ok else np.nan
                               for m, ok in zip(nearest, nearest_ok)]),
        nearest_dist=np.array([float(dist[p, int(m)]) if ok else np.nan
                               for p, (m, ok) in enumerate(zip(nearest, nearest_ok))]),
        tick=np.full(n, float(rec["seq"])),
    )


META_ROWS = [targets(r) for r in recs]
META = {key: np.concatenate([m[key] for m in META_ROWS]) for key in META_ROWS[0]}
RAW = np.concatenate([np.c_[r["state"].x, r["state"].beta] for r in recs], 0).astype(np.float64)
print(f"[data] raw input dims={RAW.shape[1]} rows={RAW.shape[0]}", flush=True)

# Which targets are literally columns of the operator's OWN input `x`.
IN_X = {"own_nimber", "max_rival", "is_pw", "health", "armor", "ammo", "speed",
        "dist_to_cart", "tss", "alive", "n_weapons", "weapon_bit", "cell",
        "cart0_depth"}


def new_estimator(seed):
    return StrategyEstimator.for_runtime(recs[0]["k"], len(recs[0]["team_of"]), seed=seed)


def ir_of(est):
    out = []
    for r in recs:
        forward = strategy_forward(est, r["state"], r["state"].w)
        out.append(np.asarray(forward["ir"], dtype=np.float64))
    return np.concatenate(out, 0)


def train(est, epochs, learner=None):
    """Drive the SHIPPED loop (observe/flush + replay ring), not a bespoke one."""
    learner = learner or OnlineLearner(est, checkpoint=None, learning_rate=3e-4)
    history = []
    for epoch in range(epochs):
        previous = None
        for cur in recs:
            m_cur = cur["state"].z.shape[0]
            l_cur = len(cur["team_of"])
            if previous is not None and previous["l"] != l_cur:
                # Roster change: close the segment, never credit across it. Same
                # rule the responder applies -- per-player arrays are not
                # comparable between two different player sets.
                got = learner.flush(previous["state"], previous["snapshot"], terminal=True)
                if got:
                    history.append(got["loss"])
                previous = None
            if previous is not None:
                seg_break = previous["m"] != m_cur
                if seg_break:
                    learner.flush(previous["state"], previous["snapshot"], terminal=True)
                previous["w_out"] = cur["state"].w
                got = learner.observe(previous, cur["state"], cur["snapshot"], terminal=seg_break)
                if got:
                    history.append(got["loss"])
            previous = dict(context=cur["context"], state=cur["state"],
                            snapshot=cur["snapshot"], cartstate=cur["snapshot"],
                            w_in=cur["state"].w, w_out=cur["state"].w,
                            actions=np.zeros(len(cur["team_of"]), dtype=np.int32),
                            behavior_logp=np.zeros(len(cur["team_of"]), dtype=np.float32),
                            m=m_cur, l=l_cur)
        got = learner.flush(previous["state"], previous["snapshot"], terminal=True)
        if got:
            history.append(got["loss"])
        print(f"   epoch {epoch}: updates={learner.updates} steps/transition="
              f"{learner.gradient_steps / max(1, learner.transitions):.2f} "
              f"replay={len(learner.replay)} loss={np.mean(history[-len(recs):]):.5f}",
              flush=True)
    return learner, history


rng = np.random.default_rng(0)
ticks = np.unique(META["tick"])
perm = rng.permutation(ticks)
train_ticks = set(perm[:int(0.6 * len(perm))].tolist())
tr = np.array([t in train_ticks for t in META["tick"]])
te = ~tr


def r2(F, y, lam=1e-3):
    ok = np.isfinite(y)
    a, b = tr & ok, te & ok
    if a.sum() < 10 or b.sum() < 10 or np.std(y[b]) < 1e-9:
        return None
    A = np.c_[F[a], np.ones(a.sum())]
    B = np.c_[F[b], np.ones(b.sum())]
    mu, sd = A[:, :-1].mean(0), A[:, :-1].std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)   # a train-constant column carries no signal;
    A[:, :-1] = (A[:, :-1] - mu) / sd   # scaling it by 1/eps only manufactures inf
    B[:, :-1] = (B[:, :-1] - mu) / sd
    w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y[a])
    return float(round(1.0 - np.sum((y[b] - B @ w) ** 2) / np.sum((y[b] - y[b].mean()) ** 2), 4))


def acc(F, y, lam=1e-3):
    ok = np.isfinite(y)
    a, b = tr & ok, te & ok
    if a.sum() < 10 or b.sum() < 10:
        return None
    yi = y.astype(int)
    cls = np.unique(yi[a])
    if len(cls) < 2:
        return None
    A = np.c_[F[a], np.ones(a.sum())]
    B = np.c_[F[b], np.ones(b.sum())]
    mu, sd = A[:, :-1].mean(0), A[:, :-1].std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    A[:, :-1] = (A[:, :-1] - mu) / sd
    B[:, :-1] = (B[:, :-1] - mu) / sd
    Y = (yi[a][:, None] == cls[None, :]).astype(float)
    W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ Y)
    pred = cls[np.argmax(B @ W, 1)]
    counts = np.array([(yi[b] == v).sum() for v in np.unique(yi[b])], float)
    return {"acc": round(float(np.mean(pred == yi[b])), 4),
            "majority": round(float(counts.max() / counts.sum()), 4),
            "n_classes": int(len(cls))}


REG = ["own_nimber", "max_rival", "hier_margin", "succ_denial", "cart_depth_total",
       "cart0_depth", "n_controlled", "cgt_nimber", "health", "armor", "ammo",
       "speed", "dist_to_cart", "tss", "n_weapons", "cell", "n_eligible",
       "nearest_dist"]
CLS = ["is_pw", "pw_team", "team", "weapon_bit", "alive", "cgt_impartial", "nearest_kind"]


def score(F, shuffled=False):
    out = {}
    local = np.random.default_rng(1234)
    for target in REG:
        y = META[target].astype(float).copy()
        if shuffled:
            local.shuffle(y)
        out[target] = {"r2": r2(F, y), "in_x": target in IN_X}
    for target in CLS:
        y = META[target].astype(float).copy()
        if shuffled:
            local.shuffle(y)
        result = acc(F, y)
        out[target] = dict(result or {}, in_x=target in IN_X)
    return out


budgets = [int(e) for e in str(args.epochs).split(",") if e.strip()]
print(f"[train] replaying the REAL transitions through the shipped loop; budgets={budgets}",
      flush=True)
est_t = new_estimator(0)
learner, history, sweep, done = None, [], {}, 0
for budget in budgets:
    t0 = time.time()
    learner, extra = train(est_t, budget - done, learner=learner)
    history += extra
    done = budget
    sweep[f"IR_trained_{budget}ep"] = {
        "updates": learner.updates,
        "steps_per_transition": round(learner.gradient_steps / max(1, learner.transitions), 3),
        "loss": round(float(np.mean(history[-len(recs):])), 5),
        "seconds": round(time.time() - t0, 1),
        "ir": ir_of(est_t),
    }
IR_T = sweep[f"IR_trained_{budgets[-1]}ep"]["ir"]
IR_R = ir_of(new_estimator(0))
IR_R7 = ir_of(new_estimator(7))
P = RAW @ (rng.standard_normal((RAW.shape[1], IR_WIDTH)) / np.sqrt(RAW.shape[1]))


def rank(M, tol=1e-6):
    return int(np.linalg.matrix_rank(M - M.mean(0), tol=tol))


def spectrum(M):
    centered = M - M.mean(0)
    sv = np.linalg.svd(centered, compute_uv=False)
    energy = sv ** 2
    total = float(energy.sum()) or 1.0
    fraction = np.cumsum(energy) / total
    participation = float((energy.sum() ** 2) / (energy ** 2).sum())
    return {"rank_1e-6": rank(M),
            "effective_rank_participation": round(participation, 3),
            "dims_for_90pct_variance": int(np.searchsorted(fraction, 0.90) + 1),
            "dims_for_99pct_variance": int(np.searchsorted(fraction, 0.99) + 1),
            "top_singular_value": round(float(sv[0]), 4)}


out = {
    "provenance": {
        "rows_file": os.path.abspath(args.rows), "n_ticks": len(recs),
        "n_player_rows": int(RAW.shape[0]), "simulation_used": False,
        "beta": "recomputed by replaying LiveBelief over the logged event stream",
        "training": f"replay of the real logged transitions through the shipped "
                    f"observe/flush loop; budgets={budgets} epochs, "
                    f"{learner.updates} updates at the deepest",
        "architecture": architecture_fingerprint(learner.bundle),
        "replay": {"size": len(learner.replay), "capacity": learner.replay.capacity,
                   "mb": round(learner.replay.nbytes / (1 << 20), 2),
                   "transitions": learner.transitions,
                   "gradient_steps": learner.gradient_steps},
        "split": "by tick, 60/40, seed 0",
        "train_rows": int(tr.sum()), "test_rows": int(te.sum()),
        "instrument_kinds": KINDS,
    },
    "belief_stage2": BAND,
    "ir_width": {"d": IR_WIDTH, "spec_requirement": ">=128d"},
    "ranks": {"raw_inputs": rank(RAW), "raw_dims": int(RAW.shape[1]),
              "x_only": rank(RAW[:, :48]), "n_rows": int(RAW.shape[0])},
    "spectra": {"raw_inputs": spectrum(RAW), "IR_trained": spectrum(IR_T),
                "IR_random_init": spectrum(IR_R), "control_randproj_128d": spectrum(P)},
    "ir_scale": {"trained_absmax": round(float(np.abs(IR_T).max()), 4),
                 "trained_std": round(float(IR_T.std()), 4),
                 "random_init_absmax": round(float(np.abs(IR_R).max()), 4)},
    "loss": {"first": round(float(np.mean(history[:len(recs)])), 5),
             "last": round(float(np.mean(history[-len(recs):])), 5)},
    "targets_in_model_input_x": sorted(IN_X),
    "training_budget_sweep": {
        name: {"updates": v["updates"], "steps_per_transition": v["steps_per_transition"],
               "loss": v["loss"], "seconds": v["seconds"],
               "effective_rank": spectrum(v["ir"])["effective_rank_participation"],
               "probes": score(v["ir"])}
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
print(json.dumps({key: out[key] for key in
                  ("provenance", "belief_stage2", "ranks", "spectra", "loss")}, indent=2))
print("WROTE", args.out)
