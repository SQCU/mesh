"""The solver worker: node 1 answers the game server's bot-plan requests.

    python3 worker.py [--policy nearest|inverted|trained] [--peer 0] [--secs 0]
    python3 worker.py --check            # validate everything, no mesh needed
    python3 worker.py --bench 20         # time the solve, no mesh needed
    python3 worker.py --synth 400        # train against the synthetic cart

Reads request blocks framed by bridge/PORT.md section 2 (width 16, or width 20
with the dominance columns), scores every bot row, and writes one response
block back per completed block: per-bot objective weights in columns 2..6,
the hard pick in column 1, leader rank in column 7.

The heavy solve is the design's workload (design/mesh-coprocessor-demo.md):
per-tick rows join a persistent context window of CTX recent rows, the window
is lifted into a RES-wide residual basis, routed through EXPERTS experts
grouped per expert, and a Gram matrix over the residual conditions the
current rows' objective scores. CTX=4096 holds the measured T/E >= 512
rows-per-expert floor and RES=2048 the orthogonalisation floor: ~104 GFLOP
against ~270 MB of resident weights per solve.

Policies over those scores:
  nearest / inverted   frozen A/B pair, argmax / argmin, disjoint on every row
  trained              closed-form Stackelberg strategy, learnable only in a
                       small vector of named scalars

The trained policy is analytic, never a neural controller. Per team the most
dominant live bot (dominance measured from the resource columns, hysteresis
against flapping) is the leader; it commits first by quantal response over
the solve's scores plus the fictitious-play mixture measured from the context
window; followers best-respond to the leader's committed pick and the same
empirical mixture, again by quantal response. REINFORCE with a per-team EMA
baseline trains only the strategy scalars (temperatures, commitment,
mixture and score weights) against rewards measured from the request stream:
signed cart progress per episode, in a goal direction itself measured from
progress-versus-occupancy correlation, plus a small push-radius term. The
solver emits strategy at the request rate; havocbot keeps all
millisecond-scale motor control.
"""
import argparse, os, sys, time
import numpy as np
from xonwire import (Mesh, Reassembler, TxWindow, REQ, RESP, REQ_WIDTH,
                     RESP_WIDTH, TEAMS, rows_per_slot, pack_hdr, HDRSZ)

SEED, EXPERTS, FF, HID = 20260828, 8, 2048, 64
CTX, RES = 4096, 2048
EXT_WIDTH = 20
PUSH_RADIUS = 512.0
HYSTERESIS = 0.15
THETA_PATH = "~/.mesh-plc-strategy.npz"
POLICIES = {"nearest": np.argmax, "inverted": np.argmin}
THETA = dict(log_tau_leader=0.0, log_tau_follower=0.0, commit=1.0,
             fp_leader=0.5, fp_follower=0.5, score_leader=1.0, score_follower=1.0)
DOM_FORMULA = ("0.005*(health+armor) + 0.3*ammo + 0.15*weapons + 0.5*powerup"
               " - 0.5*max(0, 1 - since_spawn/10)")

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


def dominance(X):
    """Measured per-bot dominance from the resource tuple, never formalised
    game logic: health+armor, ammo, weapon-combo count, powerup flags, and a
    just-spawned penalty. Width-16 streams read the extra columns as zero."""
    n, w = X.shape
    ext = X[:, 16:20] if w >= EXT_WIDTH else np.zeros((n, 4), np.float32)
    tss = np.maximum(0.0, 1.0 - ext[:, 2] / 10.0)
    return (0.005 * (X[:, 2] + X[:, 3]) + 0.3 * X[:, 4]
            + 0.15 * ext[:, 0] + 0.5 * ext[:, 1] - 0.5 * tss).astype(np.float32)


class CtxSolver:
    """MoE + Gram-over-residual over a rolling context window.

    The window is also the fictitious-play history store: alongside the
    normalised feature ring it keeps raw team, held objective, dominance and
    position per row, whose per-team first moments (objective mixture,
    centroid, spread, mean dominance) are the empirical mixes the strategy
    layer best-responds to. The Gram the solve already computes is the same
    window's second moment. All state is a deterministic function of
    (seed, request history)."""

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
        self.Zc = None
        self.m_team = np.full(T, -1, np.int32)
        self.m_obj = np.full(T, -1, np.int32)
        self.m_dom = np.zeros(T, np.float32)
        self.m_pos = np.zeros((T, 3), np.float32)

    def _insert(self, X, Xn):
        n = Xn.shape[0]
        if n >= self.T:
            idx = np.arange(self.T)
            X, Xn = X[-self.T:], Xn[-self.T:]
            self.pos = 0
        else:
            idx = (self.pos + np.arange(n)) % self.T
            self.pos = int((self.pos + n) % self.T)
        self.C[idx] = Xn
        self.m_team[idx] = X[:, 1].astype(np.int32)
        self.m_obj[idx] = np.clip(X[:, 15], 0, TEAMS - 1).astype(np.int32)
        self.m_dom[idx] = dominance(X)
        self.m_pos[idx] = X[:, 5:8]
        return idx

    def mixes(self):
        out = {}
        for j in range(1, TEAMS + 1):
            sel = self.m_team == j
            k = int(sel.sum())
            if not k:
                continue
            m = np.bincount(self.m_obj[sel], minlength=TEAMS).astype(np.float64)
            p = self.m_pos[sel]
            out[j] = dict(mix=(m / m.sum()).astype(np.float32),
                          centroid=p.mean(axis=0),
                          spread=float(p.std()),
                          dom=float(self.m_dom[sel].mean()), rows=k)
        return out

    def solve(self, X):
        Xb = np.ascontiguousarray(X[:, :REQ_WIDTH], np.float32)
        Xn = (Xb / (1.0 + np.abs(Xb))).astype(np.float32)
        idx = self._insert(X, Xn)
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
        Zcur = mx.take(Z, mx.array(idx.astype(np.uint32)), axis=0)
        G = Zcur @ (Gm @ w["Og"])
        tr = mx.sum(Z * Z) * (1.0 / self.T)
        Zc = (Zcur @ Gm) * (RES / tr)
        mx.eval(G, Zc)
        self.Zc = np.nan_to_num(np.asarray(memoryview(Zc))).astype(np.float32)
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
            tr = float(np.sum(Z * Z)) / self.T
            Zcur = Z[idx]
            self.Zc = np.nan_to_num((Zcur @ Gm) * (RES / tr)).astype(np.float32)
            return Zcur @ (Gm @ w["Og"])


def softmax(z):
    z = z - np.max(z)
    e = np.exp(z)
    return e / e.sum()


class RewardMeter:
    """Environmental rewards measured from the request stream, not formalised.

    Per tick: the cart-progress delta (column 12) and which team's bots hold
    the push radius (column 11). Each team's goal direction is measured as an
    EMA of progress-delta times occupancy advantage; the episode return is
    the signed progress in the measured direction plus a small
    time-in-push-radius term."""

    def __init__(self, alpha=0.02):
        self.alpha = alpha
        self.dir_ema = np.zeros(TEAMS + 1, np.float64)
        self.last = None
        self._reset_episode()

    def _reset_episode(self):
        self.dprog = 0.0
        self.push = np.zeros(TEAMS + 1, np.float64)
        self.rows = np.zeros(TEAMS + 1, np.float64)

    def tick(self, X):
        prog = float(X[0, 12])
        d = 0.0 if self.last is None else prog - self.last
        self.last = prog
        if abs(d) > 0.05:
            d = 0.0
        team = X[:, 1].astype(np.int32)
        near = X[:, 11] < PUSH_RADIUS
        occ = np.zeros(TEAMS + 1)
        for j in range(1, TEAMS + 1):
            tj = team == j
            nj = float(np.sum(tj & near))
            self.rows[j] += float(np.sum(tj))
            self.push[j] += nj
            occ[j] = nj
        tot = occ.sum()
        for j in range(1, TEAMS + 1):
            share = occ[j] - (tot - occ[j]) / max(1, TEAMS - 1)
            self.dir_ema[j] = (1 - self.alpha) * self.dir_ema[j] + self.alpha * d * share
        self.dprog += d

    def dirs(self):
        return np.where(self.dir_ema >= 0, 1.0, -1.0)

    def finish(self):
        dirs = self.dirs()
        push_frac = self.push / np.maximum(1.0, self.rows)
        R = dirs * self.dprog * 10.0 + 0.1 * push_frac
        out = (R, dirs, self.dprog, push_frac)
        self._reset_episode()
        return out


class Strategy:
    """Closed-form Stackelberg strategy over the solve's objective scores.

    Per team: the most dominant live bot leads (hysteresis HYSTERESIS keeps
    leadership legible); the leader commits by quantal response over
    score_leader * standardised scores + fp_leader * enemy empirical mix at
    temperature exp(log_tau_leader); each follower best-responds by quantal
    response over score_follower * scores + fp_follower * mix +
    commit * onehot(leader pick) at exp(log_tau_follower). The emitted
    per-bot weights are those response distributions. Only the named scalars
    train, by REINFORCE over the analytic log-prob gradients."""

    def __init__(self, path, seed=SEED, lr=0.1):
        self.path = os.path.expanduser(path) if path else ""
        self.lr = lr
        self.rng = np.random.default_rng(seed)
        self.names = sorted(THETA)
        self.theta = dict(THETA)
        self.base = np.zeros(TEAMS + 1, np.float64)
        self.eps = 0
        if self.path and os.path.exists(self.path):
            z = np.load(self.path)
            for i, k in enumerate(self.names):
                self.theta[k] = float(z["theta"][i])
            self.base = z["base"].astype(np.float64)
            self.eps = int(z["eps"])
        self.leaders = {}
        self.held_leader = {}
        self.held = {}
        self._reset_episode()

    def _reset_episode(self):
        self.g = {j: {k: 0.0 for k in self.names} for j in range(1, TEAMS + 1)}
        self.ndec = np.zeros(TEAMS + 1, np.int64)

    def save(self, dir_ema=None):
        if not self.path:
            return
        tmp = self.path + ".tmp.npz"
        np.savez(tmp, theta=np.array([self.theta[k] for k in self.names]),
                 base=self.base, eps=self.eps,
                 dir_ema=dir_ema if dir_ema is not None else np.zeros(TEAMS + 1))
        os.replace(tmp, self.path)

    def _pick_leader(self, j, ids, dom):
        best = int(np.argmax(dom))
        prev = self.leaders.get(j)
        if prev is not None:
            held = np.nonzero(ids == prev)[0]
            if held.size and dom[best] < dom[held[0]] + HYSTERESIS:
                return int(held[0])
        self.leaders[j] = int(ids[best])
        return best

    def _quantal(self, comps, keys, tau_key, gacc, action=None, sample=False):
        tau = float(np.exp(self.theta[tau_key]))
        u = np.zeros(TEAMS)
        for k in keys:
            u = u + self.theta[k] * comps[k]
        z = u / tau
        pi = softmax(z)
        if action is None:
            p = (pi / pi.sum()).astype(np.float64)
            action = int(self.rng.choice(TEAMS, p=p / p.sum())) if sample else int(np.argmax(pi))
        if gacc is not None:
            for k in keys:
                gacc[k] += (comps[k][action] - float(pi @ comps[k])) / tau
            gacc[tau_key] += -(z[action] - float(pi @ z))
        return action, pi.astype(np.float32)

    def decide(self, X, G, mixes, boundary, sample):
        n = X.shape[0]
        team = X[:, 1].astype(np.int32)
        ids = X[:, 0].astype(np.int32)
        dom = dominance(X)
        Gs = G - G.mean(axis=1, keepdims=True)
        Gs = Gs / (Gs.std(axis=1, keepdims=True) + 1e-9)
        pick = np.zeros(n, np.float32)
        Wt = np.full((n, TEAMS), 1.0 / TEAMS, np.float32)
        lead = np.zeros(n, np.float32)
        present = sorted(set(int(t) for t in team) & set(range(1, TEAMS + 1)))
        for j in present:
            rows = np.nonzero(team == j)[0]
            others = [mixes[o]["mix"] for o in mixes if o != j]
            m_enemy = (np.mean(others, axis=0) if others
                       else np.full(TEAMS, 1.0 / TEAMS)).astype(np.float64)
            li = self._pick_leader(j, ids[rows], dom[rows])
            lrow = rows[li]
            lead[lrow] = 1.0
            gacc = self.g[j] if (boundary and sample) else None
            draw = sample and boundary
            cl = dict(score_leader=Gs[lrow].astype(np.float64), fp_leader=m_enemy)
            aL, piL = self._quantal(cl, ["score_leader", "fp_leader"], "log_tau_leader",
                                    gacc, None if boundary else self.held_leader.get(j),
                                    draw)
            if boundary:
                self.held_leader[j] = aL
                if gacc is not None:
                    self.ndec[j] += 1
            pick[lrow] = float(aL)
            Wt[lrow] = piL
            onehot = np.zeros(TEAMS)
            onehot[aL] = 1.0
            for r in rows:
                if r == lrow:
                    continue
                cf = dict(score_follower=Gs[r].astype(np.float64), fp_follower=m_enemy,
                          commit=onehot)
                aF, piF = self._quantal(cf, ["score_follower", "fp_follower", "commit"],
                                        "log_tau_follower", gacc,
                                        None if boundary else self.held.get(int(ids[r])),
                                        draw)
                if boundary:
                    self.held[int(ids[r])] = aF
                    if gacc is not None:
                        self.ndec[j] += 1
                pick[r] = float(aF)
                Wt[r] = piF
        return pick, Wt, lead

    def update(self, R):
        for j in range(1, TEAMS + 1):
            if self.ndec[j]:
                adv = (float(R[j]) - float(self.base[j])) / self.ndec[j]
                for k in self.names:
                    self.theta[k] += self.lr * adv * self.g[j][k]
                self.base[j] = 0.9 * self.base[j] + 0.1 * float(R[j])
        for k in ("log_tau_leader", "log_tau_follower"):
            self.theta[k] = float(np.clip(self.theta[k], -3.0, 3.0))
        for k in ("commit", "fp_leader", "fp_follower", "score_leader", "score_follower"):
            self.theta[k] = float(np.clip(self.theta[k], -4.0, 4.0))
        self.eps += 1
        self._reset_episode()

    def table(self):
        t = self.theta
        return (f"tau_L {np.exp(t['log_tau_leader']):.3f} tau_F {np.exp(t['log_tau_follower']):.3f} "
                f"commit {t['commit']:.3f} fp_L {t['fp_leader']:.3f} fp_F {t['fp_follower']:.3f} "
                f"score_L {t['score_leader']:.3f} score_F {t['score_follower']:.3f}")


class SynthEnv:
    """A stub match whose cart responds to picks in a known way.

    Each team's correct objective at any decision is the argmax of its mean
    standardised solve scores, so tracking the solve is the winning strategy
    and the strategy scalars have a real gradient to find. Progress moves in
    a team's direction with the number of its bots holding the correct pick;
    bots holding it sit inside the push radius, so reward and direction
    measurement flow through the same columns the real match uses. One bot
    per team carries a dominant resource tuple."""

    def __init__(self, seed, teams=2, bots=8):
        self.g = np.random.default_rng(seed)
        self.nteams, self.bots = teams, bots
        self.dirs = np.array([0.0] + [1.0 if j % 2 == 1 else -1.0 for j in range(1, TEAMS + 1)])
        self.prog = 0.5
        self.good = {}
        self.picks = {}

    def rows(self):
        n = self.nteams * self.bots
        X = np.zeros((n, EXT_WIDTH), np.float32)
        X[:, 0] = np.arange(n)
        X[:, 1] = np.repeat(np.arange(1, self.nteams + 1), self.bots)
        X[:, 2] = 100.0
        X[:, 4] = 0.5
        X[:, 5:8] = (self.g.standard_normal((n, 3)) * 0.5).astype(np.float32)
        X[:, 11] = np.where([self.good.get(int(i), False) for i in X[:, 0]], 100.0, 800.0)
        X[:, 11] += self.g.uniform(0, 50, n).astype(np.float32)
        X[:, 12] = self.prog
        for j in range(1, self.nteams + 1):
            sel = X[:, 1] == j
            X[sel, 15] = float(self.picks.get(j, 0))
            X[sel, 16] = 2.0
            X[sel, 18] = 20.0
            first = np.nonzero(sel)[0][0]
            X[first, 16] = 7.0
            X[first, 17] = 1.0
        return X

    def step(self, pick_by_bot, correct):
        good_count = {}
        team_pick = {}
        for bot, (j, a) in pick_by_bot.items():
            good = a == correct.get(j, -1)
            self.good[bot] = good
            good_count[j] = good_count.get(j, 0) + int(good)
            team_pick.setdefault(j, []).append(a)
        for j, ps in team_pick.items():
            self.picks[j] = int(np.bincount(ps, minlength=TEAMS).argmax())
        d = sum(0.0015 * self.dirs[j] * c / self.bots for j, c in good_count.items())
        d += float(self.g.standard_normal()) * 0.0003
        self.prog = float(np.clip(self.prog + d, 0.0, 1.0))


def run_synth(episodes, T, seed, ep_ticks, hold, lr, theta_path, log=True):
    solver = CtxSolver(T, "mlx" if mx is not None else "np")
    strat = Strategy(theta_path, seed, lr)
    meter = RewardMeter()
    env = SynthEnv(seed + 1)
    rets, accs = [], []
    follow_meas, follow_pred, recip = [], [], []
    prev_fmaj = {}
    correct, held_pairs = {}, {}
    for ep in range(episodes):
        hit = dec = 0
        for t in range(ep_ticks):
            X = env.rows()
            if t % hold == 0:
                G = solver.solve(X)
                mixes = solver.mixes()
                pick, Wt, lead = strat.decide(X, G, mixes, boundary=True, sample=True)
                team = X[:, 1].astype(np.int32)
                Gs = G - G.mean(axis=1, keepdims=True)
                Gs = Gs / (Gs.std(axis=1, keepdims=True) + 1e-9)
                for j in range(1, env.nteams + 1):
                    rows = np.nonzero(team == j)[0]
                    correct[j] = int(Gs[rows].mean(axis=0).argmax())
                    lrow = rows[np.argmax(lead[rows])]
                    aL = int(pick[lrow])
                    fmaj = []
                    for r in rows:
                        a = int(pick[r])
                        dec += 1
                        hit += int(a == correct[j])
                        if r != lrow:
                            follow_meas.append(float(a == aL))
                            follow_pred.append(float(Wt[r][aL]))
                            fmaj.append(a)
                    if j in prev_fmaj:
                        recip.append(float(aL == prev_fmaj[j]))
                    if fmaj:
                        prev_fmaj[j] = int(np.bincount(fmaj, minlength=TEAMS).argmax())
                held_pairs = {int(X[r, 0]): (int(team[r]), int(pick[r]))
                              for r in range(X.shape[0]) if team[r] >= 1}
            meter.tick(X)
            env.step(held_pairs, correct)
        R, dirs, dprog, push = meter.finish()
        strat.update(R)
        strat.save(meter.dir_ema)
        rets.append(float(np.mean([R[j] for j in range(1, env.nteams + 1)])))
        accs.append(hit / max(1, dec))
        if log and (ep < 3 or (ep + 1) % 50 == 0):
            print(f"synth: ep {ep+1:4d} return {rets[-1]:+7.3f} acc {accs[-1]:.2f} "
                  f"dprog {dprog:+.4f} {strat.table()}", flush=True)
    stats = dict(follow=float(np.mean(follow_meas)), pred=float(np.mean(follow_pred)),
                 recip=float(np.mean(recip)) if recip else 0.0)
    return np.array(rets), np.array(accs), strat, stats


def _ticks(rows, count, seed=3):
    g = np.random.default_rng(seed)
    out = []
    for t in range(count):
        X = (g.standard_normal((rows, REQ_WIDTH)) * 2.0).astype(np.float32)
        X[:, 0] = np.arange(rows)
        X[:, 1] = np.arange(rows) % TEAMS
        out.append(X)
    return out


def _fake_slot(rid, rows, usable=4090):
    X = np.zeros((rows, REQ_WIDTH), np.float32)
    X[:, 0] = np.arange(rows)
    buf = np.zeros(HDRSZ + rows * REQ_WIDTH * 4, np.uint8)
    buf[:HDRSZ] = pack_hdr(REQ, rid, rid, REQ_WIDTH, rows, 0, 1, rows)
    buf[HDRSZ:] = np.frombuffer(X.tobytes(), np.uint8)
    return buf


def check(rows):
    ok = True

    def report(name, good, detail):
        nonlocal ok
        ok = ok and good
        print(("PASS" if good else "FAIL") + f" {name}: {detail}", flush=True)

    rx = Reassembler(REQ, REQ_WIDTH, 128, 4090)
    got = [rx.feed(_fake_slot(rid, 16)) is not None for rid in (5, 6, 1, 2, 3, 4)]
    report("reassembler adopts a restarted session", got == [True, True, False, False, True, True],
           f"ids 5,6 then restart 1..4 -> completions {got}, resyncs {rx.resync}")

    st = Strategy("", 1)
    Xh = np.zeros((2, EXT_WIDTH), np.float32)
    Xh[:, 0] = [7, 8]
    Xh[:, 1] = 1
    Xh[:, 2] = [100, 100]
    Xh[0, 16] = 4.0
    Gh = np.tile(np.arange(TEAMS, dtype=np.float32), (2, 1))
    mixes = {1: dict(mix=np.full(TEAMS, 0.2, np.float32))}
    st.decide(Xh, Gh, mixes, True, False)
    first = st.leaders[1]
    Xh[1, 16] = 4.5
    st.decide(Xh, Gh, mixes, True, False)
    stay = st.leaders[1]
    Xh[1, 16] = 6.0
    st.decide(Xh, Gh, mixes, True, False)
    moved = st.leaders[1]
    report("leader hysteresis", first == 7 and stay == 7 and moved == 8,
           f"leader {first} held at +0.075 dominance ({stay}), moved at +0.30 ({moved}), "
           f"hysteresis {HYSTERESIS}")

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

        r1, a1, t1, _ = run_synth(15, 256, 42, 45, 15, 0.1, "", log=False)
        r2, a2, t2, _ = run_synth(15, 256, 42, 45, 15, 0.1, "", log=False)
        report("reinforce deterministic per seed",
               all(t1.theta[k] == t2.theta[k] for k in t1.names) and np.allclose(r1, r2),
               "two fresh 15-episode synth runs, identical scalars and returns")
        moved = max(abs(t1.theta[k] - THETA[k]) for k in t1.names)
        report("reinforce updates applied", moved > 0,
               f"max scalar movement {moved:.3f} after 15 episodes")
    else:
        print("skip ctx/trained checks: mlx not importable here", flush=True)
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
    ap.add_argument("--synth", type=int, default=0)
    ap.add_argument("--episode", type=int, default=300)
    ap.add_argument("--hold", type=int, default=25)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--train-seed", type=int, default=SEED)
    ap.add_argument("--theta", default=THETA_PATH)
    ap.add_argument("--eval", action="store_true")
    a = ap.parse_args()

    if a.check:
        sys.exit(0 if check(a.rows) else 1)
    if a.bench:
        bench(a.bench, a.rows, a.ctx)
        return
    if a.synth:
        rets, accs, strat, stats = run_synth(
            a.synth, a.ctx, a.train_seed, a.episode, a.hold, a.lr,
            a.theta if a.theta != THETA_PATH else "")
        k = max(1, min(100, a.synth // 4))
        chance = 1.0 / TEAMS
        print(f"synth: return first{k} {rets[:k].mean():+.4f} -> last{k} {rets[-k:].mean():+.4f}, "
              f"accuracy {accs[:k].mean():.2f} -> {accs[-k:].mean():.2f}", flush=True)
        print(f"synth: stackelberg asymmetry: P(follower==leader) {stats['follow']:.3f} "
              f"(analytic prediction {stats['pred']:.3f}, chance {chance:.2f}), "
              f"P(leader==prev follower majority) {stats['recip']:.3f}", flush=True)
        print(f"synth: learned scalars: {strat.table()}", flush=True)
        good = (rets[-k:].mean() > rets[:k].mean() and accs[-k:].mean() > accs[:k].mean()
                and stats["follow"] > chance + 0.05
                and abs(stats["follow"] - stats["pred"]) < 0.06
                and stats["follow"] > stats["recip"] + 0.05)
        sys.exit(0 if good else 1)

    trained = a.policy == "trained"
    pick_of = POLICIES.get(a.policy, np.argmax)
    m = attach(a.peer)
    rx16 = Reassembler(REQ, REQ_WIDTH, a.maxrows, m.usable)
    rx20 = Reassembler(REQ, EXT_WIDTH, a.maxrows, m.usable)
    tx = TxWindow(m)
    bk = "mlx" if mx is not None else "np"
    solver = CtxSolver(a.ctx, bk)
    backend = f"{bk}-ctx(T={a.ctx},RES={RES},FF={FF})"
    print(f"worker: policy={a.policy} backend={backend} peer={a.peer} "
          f"usable={m.usable} req_rows/slot={rows_per_slot(m.usable, REQ_WIDTH)} "
          f"resp_rows/slot={rows_per_slot(m.usable, RESP_WIDTH)} widths 16|{EXT_WIDTH}", flush=True)

    strat = meter = None
    if trained:
        strat = Strategy(a.theta, a.train_seed, a.lr)
        meter = RewardMeter()
        mode = "greedy eval, no updates" if a.eval else "sampling + online REINFORCE"
        print(f"worker: dominance = {DOM_FORMULA} (width-16 stream reads "
              f"weapons/powerup/since_spawn as 0)", flush=True)
        print(f"worker: strategy = per-team Stackelberg: most dominant bot leads "
              f"(hysteresis {HYSTERESIS}), commits by quantal response over the solve's "
              f"scores + the context window's fictitious-play mix; followers best-respond "
              f"to the committed pick", flush=True)
        print(f"worker: reward = signed cart progress per {a.episode}-tick episode "
              f"(goal direction measured from progress-vs-occupancy) + 0.1 * push-radius time",
              flush=True)
        print(f"worker: scalars {strat.table()} ({mode}, hold {a.hold}, lr {a.lr}, "
              f"file {strat.path}, episodes so far {strat.eps})", flush=True)
        print(f"worker: timescale: this process emits strategy at the request rate; "
              f"havocbot owns all millisecond motor control", flush=True)

    tr = open(a.trace, "w", buffering=1) if a.trace else None
    if tr:
        tr.write("req,tick,live,prog,pick0,pick1,pick2,pick3,pick4,"
                 "held0,held1,held2,held3,held4,d1,d2,x1,x2,y1,y2\n")

    t0, served, blocks, short = time.time(), 0, 0, 0
    tick_in_ep = 0
    while a.secs <= 0.0 or time.time() - t0 < a.secs:
        done, src, rx = None, a.peer, rx16
        for buf, s in m.read(dtype=np.uint8):
            d16 = rx16.feed(buf)
            d20 = rx20.feed(buf)
            if d16:
                done, src, rx = d16, s, rx16
            if d20:
                done, src, rx = d20, s, rx20
        if done is None:
            time.sleep(0.0002)
            continue
        n = done["rows"]
        X = rx.stage[:n]
        G = solver.solve(X)
        if trained:
            meter.tick(X)
            boundary = tick_in_ep % a.hold == 0
            pick, Wt, lead = strat.decide(X, G, solver.mixes(), boundary, not a.eval)
            tick_in_ep += 1
            if tick_in_ep >= a.episode:
                R, dirs, dprog, push = meter.finish()
                if not a.eval:
                    strat.update(R)
                    strat.save(meter.dir_ema)
                print(f"learn: ep {strat.eps:4d} dprog {dprog:+.4f} "
                      f"dirs {[int(d) for d in dirs[1:]]} "
                      f"returns {[round(float(R[j]), 3) for j in range(1, TEAMS + 1)]} "
                      f"{strat.table()}", flush=True)
                tick_in_ep = 0
        else:
            pick = pick_of(G, axis=1).astype(np.float32)
            Wt = np.zeros((n, TEAMS), np.float32)
            Wt[np.arange(n), pick.astype(np.int32)] = 1.0
            lead = np.zeros(n, np.float32)
        out = np.zeros((n, RESP_WIDTH), np.float32)
        out[:, 0] = X[:, 0]
        out[:, 1] = pick
        out[:, 2:2 + TEAMS] = Wt
        out[:, 7] = lead
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
    print(f"worker: {blocks} blocks, {served} rows, short {short}, dropped {rx16.dropped + rx20.dropped}", flush=True)


if __name__ == "__main__":
    main()
