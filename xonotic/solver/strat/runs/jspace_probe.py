"""j-space linear-probe measurement on REAL Game-2 server rows. No simulation."""
from __future__ import annotations
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/dox/mesh/xonotic/solver"))
sys.path.insert(0, os.path.expanduser("~/dox/mesh/xonotic"))
import mlx.core as mx, mlx.nn as nn
from mlx.utils import tree_flatten
from strat.qkv import QKVProjector, QKVShapes, team_pool
from strat.relattn import RelationalEncoder, edge_features
from strat.value import StrategyValue
from strat.estimator import _hierarchy_rows, X_WIDTH, BELIEF_WIDTH, HIERARCHY_WIDTH
from strat.game import team_nimbers, Cart

RUNS = os.path.expanduser("~/dox/mesh/xonotic/solver/strat/runs")
lines = [json.loads(l) for l in open(os.path.join(RUNS, "game2_train.jsonl")) if l.strip()]
CKPT = os.path.join(RUNS, "policy_online_v3.npz")
KINDS = sorted({a["kind"] for r in lines for a in r["assignments"]})
kix = {k: i for i, k in enumerate(KINDS)}

records = []
for li, r in enumerate(lines):
    A = sorted(r["assignments"], key=lambda a: a["row"])
    k = int(r["k"]); L = len(A)
    if L == 0: continue
    team_of = [min(max(int(a["team"]) - 1, 0), k - 1) for a in A]
    teams = list(range(k))
    depth = np.array([float(c["depth"]) for c in r["carts"]])
    ctrl = np.array([int(c["ctrl"]) for c in r["carts"]])
    control = np.where(ctrl > 0, ctrl - 1, -1)
    Lf = 8.0
    carts = [Cart(None if c < 0 else int(c), int(np.floor(d * Lf))) for d, c in zip(depth, control)]
    nimbers = team_nimbers(carts, teams)
    hier, wmask = _hierarchy_rows(carts, teams, team_of, L)
    winner = int(np.argmax(hier[:, 5])) if hier[:, 5].any() else None
    winner_team = team_of[winner] if winner is not None else None
    sizes = np.bincount(np.asarray(team_of), minlength=k)
    x = np.zeros((L, X_WIDTH), dtype=np.float32)
    for p, t in enumerate(team_of):
        own_d = [depth[c] * Lf for c in range(len(carts)) if control[c] == t]
        riv = [float(v) for o, v in nimbers.items() if o != t]
        x[p] = (max(own_d, default=0.0)/Lf, (sum(own_d)/max(1,len(own_d)))/Lf,
                0.0,                                  # banked: NOT LOGGED
                float(t == winner_team),
                float(nimbers.get(t,0))/Lf, max(riv, default=0.0)/Lf,
                float(sizes[t])/max(1,L), 1.0/max(1,k),
                0,0,0,0,0,0,0,0)                      # x[8:16] health/armor/ammo/speed/power/tss/alive/control: NOT LOGGED
    beta = np.zeros((L, BELIEF_WIDTH), dtype=np.float32)
    E = edge_features(team_of, hier, wmask)
    succ = {int(t): float(v) for t, v in r.get("SUCC", [])}
    gv = r.get("game_value"); gv = float(gv) if isinstance(gv,(int,float)) else np.nan
    meta = dict(
        line=np.full(L, li), team=np.asarray(team_of),
        own_nimber=np.array([float(nimbers.get(t,0)) for t in team_of]),
        max_rival=np.array([max([float(nimbers[o]) for o in teams if o!=t], default=0.0) for t in team_of]),
        hier_margin=hier[:,3].astype(float), is_pw=hier[:,5].astype(float),
        pw_team=np.full(L, -1 if winner_team is None else winner_team),
        cart_depth_total=np.full(L, float(depth.sum())), cart0_depth=np.full(L, float(depth[0])),
        n_controlled=np.full(L, float((control>=0).sum())),
        succ_denial=np.array([succ.get(t+1, succ.get(t,0.0)) for t in team_of]),
        kind=np.array([kix[a["kind"]] for a in A]),
        gain=np.array([float(a["gain"]) for a in A]), lane=np.array([float(a["lane"]) for a in A]),
        logp=np.array([float(a["target_logp"]) for a in A]), game_value=np.full(L, gv),
    )
    records.append((x, beta, E, team_of, k, hier, meta))
print("[data] lines=%d rows=%d kinds=%s" % (len(records), sum(len(m['team']) for *_,m in records), KINDS))

D = 16
def build(seed, load=False):
    mx.random.seed(seed)
    sh = QKVShapes(k_teams=5, j_instruments=3, l_players=12, d_x=X_WIDTH, d_beta=BELIEF_WIDTH,
                   d_z=16, d=D, d_v=16) if len(QKVShapes.__dataclass_fields__)>7 else None
    if sh is None:
        sh = QKVShapes(5, 3, 12, X_WIDTH, BELIEF_WIDTH, 16, D)
    b = nn.Module(); b.qkv = QKVProjector(sh, seed=seed); b.encoder = RelationalEncoder(D)
    b.value = StrategyValue(2*D+4+HIERARCHY_WIDTH+16)
    info = {"matched": [], "mismatched": []}
    if load:
        data = np.load(CKPT, allow_pickle=False)
        flat = dict(tree_flatten(b.parameters()))
        w = []
        for key in data.files:
            if key.startswith("__"): continue
            if key in flat and tuple(flat[key].shape) == data[key].shape:
                info["matched"].append(key); w.append((key, mx.array(data[key])))
            else:
                info["mismatched"].append([key, list(data[key].shape), list(flat[key].shape) if key in flat else None])
        b.load_weights(w, strict=False)
    return b, info

def feats(b):
    Rp_all, VR_all = [], []
    for x, beta, E, team_of, k, hier, _ in records:
        R0 = b.qkv.query(mx.array(x), mx.array(beta))
        Rp = b.encoder(R0, mx.array(E))
        tc = team_pool(Rp, list(team_of), k)
        ti = mx.array(np.asarray(team_of, dtype=np.int32))
        Rp_np = np.asarray(Rp, dtype=np.float64)
        Rp_all.append(Rp_np)
        VR_all.append(np.c_[Rp_np, np.asarray(tc, dtype=np.float64)[np.asarray(team_of)], hier.astype(np.float64)])
    return np.concatenate(Rp_all,0), np.concatenate(VR_all,0)

META = {kk: np.concatenate([m[kk] for *_, m in records]) for kk in records[0][6]}
RAW = np.concatenate([np.c_[x, b] for x, b, *_ in records], 0).astype(np.float64)

rng = np.random.default_rng(0)
lids = np.unique(META["line"]); perm = rng.permutation(lids)
trs = set(perm[:int(0.6*len(perm))].tolist())
tr = np.array([l in trs for l in META["line"]]); te = ~tr

def r2(F, y, lam=1e-3):
    ok = np.isfinite(y); a, bm = tr & ok, te & ok
    if a.sum()<10 or bm.sum()<10 or np.std(y[bm])<1e-9: return None
    A=np.c_[F[a],np.ones(a.sum())]; B=np.c_[F[bm],np.ones(bm.sum())]
    mu,sd=A[:,:-1].mean(0),A[:,:-1].std(0)+1e-8
    A[:,:-1]=(A[:,:-1]-mu)/sd; B[:,:-1]=(B[:,:-1]-mu)/sd
    w=np.linalg.solve(A.T@A+lam*np.eye(A.shape[1]),A.T@y[a])
    return float(round(1.0-np.sum((y[bm]-B@w)**2)/np.sum((y[bm]-y[bm].mean())**2),4))

def acc(F, y, lam=1e-3):
    cls=np.unique(y[tr])
    if len(cls)<2: return None
    A=np.c_[F[tr],np.ones(tr.sum())]; B=np.c_[F[te],np.ones(te.sum())]
    mu,sd=A[:,:-1].mean(0),A[:,:-1].std(0)+1e-8
    A[:,:-1]=(A[:,:-1]-mu)/sd; B[:,:-1]=(B[:,:-1]-mu)/sd
    Y=(y[tr][:,None]==cls[None,:]).astype(float)
    W=np.linalg.solve(A.T@A+lam*np.eye(A.shape[1]),A.T@Y)
    pred=cls[np.argmax(B@W,1)]
    c=np.array([(y[te]==v).sum() for v in np.unique(y[te])],float)
    return {"acc":round(float(np.mean(pred==y[te])),4),"majority":round(float(c.max()/c.sum()),4),"n_classes":int(len(cls))}

REG=["own_nimber","max_rival","hier_margin","cart_depth_total","cart0_depth","n_controlled","succ_denial","gain","lane","logp","game_value"]
CLS=["is_pw","kind","pw_team","team"]
def score(F, shuffled=False):
    out={}
    for t in REG:
        y=META[t].astype(float).copy()
        if shuffled: rng.shuffle(y)
        out[t]={"r2":r2(F,y)}
    for t in CLS:
        y=META[t].astype(int).copy()
        if shuffled: rng.shuffle(y)
        out[t]=acc(F,y)
    return out

b_ck, info = build(0, load=True)
b_r0, _ = build(0); b_r1, _ = build(7)
Rp_ck, VR_ck = feats(b_ck); Rp_r0, VR_r0 = feats(b_r0); Rp_r1, VR_r1 = feats(b_r1)
P16 = RAW @ (rng.standard_normal((RAW.shape[1],16))/np.sqrt(RAW.shape[1]))
P40 = RAW @ (rng.standard_normal((RAW.shape[1],40))/np.sqrt(RAW.shape[1]))

def rank(M): return int(np.linalg.matrix_rank(M-M.mean(0), tol=1e-6))
out = {
 "provenance": {"log":"runs/game2_train.jsonl","n_lines":len(lines),"n_player_rows":int(len(META["team"])),
   "checkpoint":"runs/policy_online_v3.npz","simulation_used":False,
   "code_tree":"mesh-mini ~/dox/mesh (checkpoint-era arch: relattn RelationalEncoder, X_WIDTH=16)",
   "split":"by line, 60/40, seed 0", "train_rows":int(tr.sum()), "test_rows":int(te.sum())},
 "ir_width": {"row_width_d_trained": D, "value_head_input_width": 2*D+4+HIERARCHY_WIDTH+16,
   "value_row_dims_probed_here": VR_ck.shape[1],
   "omitted_value_row_dims": ["stats(4)","behavior_mix(16) - both need instrument descriptors z, NOT LOGGED"],
   "spec_requirement": ">=128d", "local_rewrite_IR_WIDTH": 128, "local_rewrite_has_checkpoint": False},
 "checkpoint_compat": {"matched": info["matched"], "mismatched": info["mismatched"]},
 "unavailable_targets": ["health","armor","ammo","weapon bitmask / rocket-launcher held","position","velocity",
   "distance-to-nearest-cart","per-player belief beta","instrument descriptors z","player-instrument relation rows","banked depth"],
 "zeroed_input_dims": "x[2] (banked) and x[8:16] (health,armor,ammo,speed,power,tss,alive,control) and beta[0:8] are zero - absent from the real log",
 "ranks": {"Rp_trained":rank(Rp_ck),"Rp_random":rank(Rp_r0),"value_rows_trained":rank(VR_ck),"raw_inputs":rank(RAW),"n_rows":int(RAW.shape[0])},
 "probes": {
   "Rp_trained_ckpt": score(Rp_ck), "Rp_random_init_s0": score(Rp_r0), "Rp_random_init_s7": score(Rp_r1),
   "value_rows_trained_ckpt": score(VR_ck), "value_rows_random_init": score(VR_r0),
   "control_randproj_16d": score(P16), "control_randproj_40d": score(P40),
   "control_shuffled_labels_Rp_trained": score(Rp_ck, shuffled=True),
   "reference_raw_inputs_24d": score(RAW),
 },
}
dest=os.path.join(RUNS,"jspace_probe.json")
json.dump(out, open(dest,"w"), indent=2)
print(json.dumps(out, indent=2))
print("WROTE", dest)
