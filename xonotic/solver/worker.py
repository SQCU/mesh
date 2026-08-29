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
K_CARTS = 2
ROLES = ("push", "suppress", "escort")
TEAM_HYST = 0.2
PUSH_RADIUS = 512.0
HYSTERESIS = 0.15
THETA_PATH = "~/.mesh-plc-strategy.npz"
POLICIES = {"nearest": np.argmax, "inverted": np.argmin}
THETA = dict(log_tau_leader=0.0, log_tau_follower=0.0, commit=1.0,
             fp_leader=0.5, fp_follower=0.5, score_leader=1.0, score_follower=1.0,
             log_tau_team=0.0, suppress_appetite=0.8, lead_bias=0.8,
             cart_inertia=0.3, gram_weight=0.5)
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
                W2=f(EXPERTS, FF, RES), Og=f(RES, TEAMS), Og2=f(RES, K_CARTS * TEAMS))


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
        self.m_cart = np.zeros(T, np.int32)
        self.m_dom = np.zeros(T, np.float32)
        self.m_pos = np.zeros((T, 3), np.float32)
        self.G2 = None
        self.kappa = np.zeros((TEAMS * K_CARTS, TEAMS * K_CARTS), np.float32)

    def _insert(self, X, Xn, bot_cart=None):
        n = Xn.shape[0]
        if n >= self.T:
            idx = np.arange(self.T)
            X, Xn = X[-self.T:], Xn[-self.T:]
            if bot_cart is not None:
                bot_cart = bot_cart[-self.T:]
            self.pos = 0
        else:
            idx = (self.pos + np.arange(n)) % self.T
            self.pos = int((self.pos + n) % self.T)
        self.C[idx] = Xn
        self.m_team[idx] = X[:, 1].astype(np.int32)
        self.m_obj[idx] = np.clip(X[:, 15], 0, TEAMS - 1).astype(np.int32)
        self.m_cart[idx] = 0 if bot_cart is None else np.clip(bot_cart, 0, K_CARTS - 1)
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
            carts = np.bincount(self.m_cart[sel], minlength=K_CARTS).astype(np.float64)
            nodes = np.full((K_CARTS, TEAMS), 1.0 / TEAMS, np.float32)
            for c in range(K_CARTS):
                sc = sel & (self.m_cart == c)
                if sc.any():
                    mc = np.bincount(self.m_obj[sc], minlength=TEAMS).astype(np.float64)
                    nodes[c] = (mc / mc.sum()).astype(np.float32)
            out[j] = dict(mix=(m / m.sum()).astype(np.float32),
                          carts=(carts / carts.sum()).astype(np.float32),
                          nodes=nodes,
                          centroid=p.mean(axis=0),
                          spread=float(p.std()),
                          dom=float(self.m_dom[sel].mean()), rows=k)
        return out

    def _group_mask(self):
        M = np.zeros((TEAMS * K_CARTS, self.T), np.float32)
        for j in range(1, TEAMS + 1):
            for c in range(K_CARTS):
                sel = (self.m_team == j) & (self.m_cart == c)
                k = float(sel.sum())
                if k:
                    i = (j - 1) * K_CARTS + c
                    M[i, sel] = 1.0 / k
                    M[i] -= 1.0 / self.T
        return M

    @staticmethod
    def _kappa_norm(K):
        d = np.sqrt(np.maximum(np.diag(K), 0.0))
        den = np.outer(d, d) + 1e-9
        Kn = K / den
        np.fill_diagonal(Kn, 0.0)
        return np.nan_to_num(Kn).astype(np.float32)

    def solve(self, X, bot_cart=None):
        Xb = np.ascontiguousarray(X[:, :REQ_WIDTH], np.float32)
        Xn = (Xb / (1.0 + np.abs(Xb))).astype(np.float32)
        idx = self._insert(X, Xn, bot_cart)
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
        G2 = Zcur @ (Gm @ w["Og2"])
        V = mx.array(self._group_mask()) @ Z
        Kp = V @ Gm @ V.T
        mx.eval(G, Zc, G2, Kp)
        self.Zc = np.nan_to_num(np.asarray(memoryview(Zc))).astype(np.float32)
        self.G2 = np.nan_to_num(np.asarray(memoryview(G2))).astype(np.float32)
        self.kappa = self._kappa_norm(np.asarray(memoryview(Kp)).astype(np.float64))
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
            self.G2 = np.nan_to_num(Zcur @ (Gm @ w["Og2"])).astype(np.float32)
            V = self._group_mask() @ Z
            self.kappa = self._kappa_norm((V @ Gm @ V.T).astype(np.float64))
            return Zcur @ (Gm @ w["Og"])


def softmax(z):
    z = z - np.max(z)
    e = np.exp(z)
    return e / e.sum()


def bind_match(X):
    """Provisional single-cart binding for the live wire; the k-cart layout
    lands in PORT.md from the game-rules side and replaces only this
    function and the meter's inputs."""
    n = X.shape[0]
    prog = np.zeros(K_CARTS, np.float64)
    prog[0] = float(X[0, 12])
    return dict(carts=1, prog=prog,
                controller=np.zeros(K_CARTS, np.int32),
                score=np.zeros(TEAMS + 1, np.float64),
                bot_cart=np.zeros(n, np.int32))


class MatchMeter:
    """Environmental rewards measured from the match state on the wire.

    Per tick: each cart's progress delta and which team's bots hold its push
    radius; per-(team, cart) goal directions are EMAs of delta times
    occupancy advantage. Per episode: banked control points relative to the
    other teams, plus the signed-progress and push-radius terms."""

    def __init__(self, alpha=0.02):
        self.alpha = alpha
        self.dir_ema = np.zeros((TEAMS + 1, K_CARTS))
        self.last_prog = None
        self.last_score = None
        self._reset_episode()

    def _reset_episode(self):
        self.dprog = np.zeros(K_CARTS)
        self.dbank = np.zeros(TEAMS + 1)
        self.push = np.zeros(TEAMS + 1)
        self.rows = np.zeros(TEAMS + 1)

    def tick(self, X, ms):
        prog = np.asarray(ms["prog"], np.float64)
        d = np.zeros(K_CARTS) if self.last_prog is None else prog - self.last_prog
        self.last_prog = prog.copy()
        d[np.abs(d) > 0.05] = 0.0
        score = np.asarray(ms["score"], np.float64)
        if self.last_score is not None:
            ds = score - self.last_score
            ds[np.abs(ds) > 10] = 0.0
            self.dbank += ds
        self.last_score = score.copy()
        team = X[:, 1].astype(np.int32)
        near = X[:, 11] < PUSH_RADIUS
        bc = np.asarray(ms["bot_cart"], np.int32)
        for c in range(K_CARTS):
            occ = np.zeros(TEAMS + 1)
            selc = near & (bc == c)
            for j in range(1, TEAMS + 1):
                occ[j] = float(np.sum(selc & (team == j)))
            tot = occ.sum()
            for j in range(1, TEAMS + 1):
                share = occ[j] - (tot - occ[j]) / max(1, TEAMS - 1)
                self.dir_ema[j, c] = (1 - self.alpha) * self.dir_ema[j, c] \
                    + self.alpha * d[c] * share
        for j in range(1, TEAMS + 1):
            tj = team == j
            self.rows[j] += float(np.sum(tj))
            self.push[j] += float(np.sum(tj & near))
        self.dprog += d

    def dirs(self):
        return np.where(self.dir_ema >= 0, 1.0, -1.0)

    def finish(self):
        dirs = self.dirs()
        push_frac = self.push / np.maximum(1.0, self.rows)
        R = np.zeros(TEAMS + 1)
        for j in range(1, TEAMS + 1):
            rivals = [self.dbank[o] for o in range(1, TEAMS + 1) if o != j]
            R[j] = (self.dbank[j] - max(rivals)) + 0.5 * self.dbank[j] \
                + 2.0 * float(dirs[j] @ self.dprog) + 0.05 * push_frac[j]
        out = (R, dirs, self.dprog.copy(), self.dbank.copy(), push_frac)
        self._reset_episode()
        return out


class Strategy:
    """Two nested closed-form Stackelberg games; only named scalars learn.

    Between teams: the score/tempo-dominant team (dominance measured from
    banked score gap, carts controlled and controlled progress, hysteresis
    TEAM_HYST) commits its allocation first - a quantal response over
    (cart, role) with roles push/suppress/escort; trailing teams best-respond
    with a lead_bias term on the leader's committed push mass, a
    suppress_appetite on enemy-controlled carts, cart_inertia on their own
    previous allocation, and gram_weight on the context window's Gram
    coupling between their and the controller's activity on that cart.
    Within teams: the dominant bot leads node choice per the committed
    (cart, role), exactly the earlier machinery; a suppress role flips the
    score slice, which is the regression rule in utility form. REINFORCE
    with per-team EMA baselines trains the twelve scalars."""

    def __init__(self, path, seed=SEED, lr=0.1):
        self.path = os.path.expanduser(path) if path else ""
        self.lr = lr
        self.rng = np.random.default_rng(seed)
        self.names = sorted(THETA)
        self.theta = dict(THETA)
        self.base = np.zeros(TEAMS + 1, np.float64)
        self.eps = 0
        if self.path and os.path.exists(self.path):
            z = np.load(self.path, allow_pickle=False)
            saved = {str(k): float(v) for k, v in zip(z["names"], z["theta"])} \
                if "names" in z else {}
            for k in self.names:
                if k in saved:
                    self.theta[k] = saved[k]
            self.base = z["base"].astype(np.float64)
            self.eps = int(z["eps"])
        self.leaders = {}
        self.team_lead = None
        self.alloc = {}
        self.acomps = {}
        self.prev_alloc = {}
        self.held = {}
        self.held_leader = {}
        self._reset_episode()

    def _reset_episode(self):
        self.g = {j: {k: 0.0 for k in self.names} for j in range(1, TEAMS + 1)}
        self.ndec = np.zeros(TEAMS + 1, np.int64)

    def save(self):
        if not self.path:
            return
        tmp = self.path + ".tmp.npz"
        np.savez(tmp, names=np.array(self.names),
                 theta=np.array([self.theta[k] for k in self.names]),
                 base=self.base, eps=self.eps)
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

    def _team_leader(self, ms, present):
        D = {}
        score = np.asarray(ms["score"], np.float64)
        for j in present:
            others = [score[o] for o in present if o != j]
            gap = score[j] - (np.mean(others) if others else 0.0)
            ctl = sum(1 for c in range(ms["carts"]) if ms["controller"][c] == j)
            cprog = sum(ms["prog"][c] for c in range(ms["carts"])
                        if ms["controller"][c] == j)
            D[j] = 1.0 * gap + 0.5 * ctl + 0.3 * cprog
        best = max(sorted(D), key=lambda j: D[j])
        if self.team_lead in D and D[best] < D[self.team_lead] + TEAM_HYST:
            return self.team_lead
        self.team_lead = best
        return best

    def _quantal(self, u0, comps, keys, tau_key, gacc, action=None, sample=False):
        tau = float(np.exp(self.theta[tau_key]))
        u = np.array(u0, np.float64)
        for k in keys:
            u = u + self.theta[k] * comps[k]
        z = u / tau
        pi = softmax(z)
        if action is None:
            p = (pi / pi.sum()).astype(np.float64)
            action = int(self.rng.choice(len(p), p=p / p.sum())) if sample \
                else int(np.argmax(pi))
        if gacc is not None:
            for k in keys:
                gacc[k] += (comps[k][action] - float(pi @ comps[k])) / tau
            gacc[tau_key] += -(z[action] - float(pi @ z))
        return action, pi.astype(np.float32)

    def _allocate(self, ms, kappa, present):
        lead_team = self._team_leader(ms, present)
        order = [lead_team] + [j for j in present if j != lead_team]
        self.alloc, self.acomps = {}, {}
        nA = K_CARTS * len(ROLES)
        for j in order:
            u0 = np.zeros(nA)
            supp = np.zeros(nA)
            leadp = np.zeros(nA)
            prev = self.prev_alloc.get(j, np.full(nA, 1.0 / nA))
            gram = np.zeros(nA)
            for c in range(K_CARTS):
                i = c * len(ROLES)
                if c >= ms["carts"]:
                    u0[i:i + len(ROLES)] = -30.0
                    continue
                ctl = int(ms["controller"][c])
                u0[i + 0] = 1.0 if ctl in (0, j) else 0.25
                u0[i + 2] = 0.4 if ctl == j else 0.0
                if ctl not in (0, j):
                    supp[i + 1] = 1.0
                    gram[i + 1] = float(kappa[(j - 1) * K_CARTS + c,
                                              (ctl - 1) * K_CARTS + c])
                if j != lead_team and lead_team in self.alloc:
                    leadp[i + 1] = float(self.alloc[lead_team][i + 0])
            comps = dict(suppress_appetite=supp, lead_bias=leadp,
                         cart_inertia=prev, gram_weight=gram)
            _, pi = self._quantal(u0, comps,
                                  ["suppress_appetite", "lead_bias",
                                   "cart_inertia", "gram_weight"],
                                  "log_tau_team", None)
            self.alloc[j] = pi.astype(np.float64)
            self.acomps[j] = (u0, comps)
            self.prev_alloc[j] = pi.astype(np.float64)
        return lead_team

    def tick(self, X, G2, mixes, kappa, ms, boundary, sample):
        n = X.shape[0]
        team = X[:, 1].astype(np.int32)
        ids = X[:, 0].astype(np.int32)
        dom = dominance(X)
        Gs = G2.reshape(n, K_CARTS, TEAMS).astype(np.float64)
        Gs = Gs - Gs.mean(axis=2, keepdims=True)
        Gs = Gs / (Gs.std(axis=2, keepdims=True) + 1e-9)
        pick = np.zeros(n, np.float32)
        Wt = np.full((n, TEAMS), 1.0 / TEAMS, np.float32)
        lead = np.zeros(n, np.float32)
        cart = np.zeros(n, np.int32)
        present = sorted(set(int(t) for t in team) & set(range(1, TEAMS + 1)))
        if not present:
            return pick, Wt, lead, cart
        if boundary:
            self._allocate(ms, kappa, present)
        nodemix = {}
        for j in present:
            others = [mixes[o]["nodes"] for o in mixes if o != j]
            nodemix[j] = (np.mean(others, axis=0) if others
                          else np.full((K_CARTS, TEAMS), 1.0 / TEAMS)).astype(np.float64)
        for j in present:
            rows = np.nonzero(team == j)[0]
            A = self.alloc.get(j)
            if A is None:
                continue
            li = self._pick_leader(j, ids[rows], dom[rows])
            lrow = rows[li]
            lead[lrow] = 1.0
            gacc = self.g[j] if (boundary and sample) else None
            u0, comps = self.acomps[j]
            akeys = ["suppress_appetite", "lead_bias", "cart_inertia", "gram_weight"]
            ordered = [lrow] + [r for r in rows if r != lrow]
            for r in ordered:
                bid = int(ids[r])
                if boundary:
                    ai, _ = self._quantal(u0, comps, akeys, "log_tau_team",
                                          gacc, None, sample)
                    c, role = divmod(ai, len(ROLES))
                else:
                    hd = self.held.get(bid)
                    c, role = (hd[0], hd[1]) if hd else (0, 0)
                sgn = -1.0 if role == 1 else 1.0
                sl = sgn * Gs[r, c]
                mnode = nodemix[j][c]
                if r == lrow:
                    cn = dict(score_leader=sl, fp_leader=mnode)
                    a, pi = self._quantal(np.zeros(TEAMS), cn,
                                          ["score_leader", "fp_leader"],
                                          "log_tau_leader", gacc,
                                          None if boundary else
                                          (self.held.get(bid) or (0, 0, None))[2],
                                          sample and boundary)
                    if boundary:
                        self.held_leader[j] = (c, a)
                else:
                    lc, la = self.held_leader.get(j, (c, None))
                    onehot = np.zeros(TEAMS)
                    if la is not None and lc == c:
                        onehot[la] = 1.0
                    cn = dict(score_follower=sl, fp_follower=mnode, commit=onehot)
                    a, pi = self._quantal(np.zeros(TEAMS), cn,
                                          ["score_follower", "fp_follower", "commit"],
                                          "log_tau_follower", gacc,
                                          None if boundary else
                                          (self.held.get(bid) or (0, 0, None))[2],
                                          sample and boundary)
                if boundary:
                    self.held[bid] = (c, role, a)
                    if gacc is not None:
                        self.ndec[j] += 1
                pick[r] = float(a)
                Wt[r] = pi
                cart[r] = c
        return pick, Wt, lead, cart

    def held_view(self, X):
        team = X[:, 1].astype(np.int32)
        ids = X[:, 0].astype(np.int32)
        out = {}
        for r in range(X.shape[0]):
            j = int(team[r])
            if j < 1:
                continue
            hd = self.held.get(int(ids[r]))
            if hd:
                out[int(ids[r])] = (j, hd[0], hd[1])
        return out

    def update(self, R):
        for j in range(1, TEAMS + 1):
            if self.ndec[j]:
                adv = (float(R[j]) - float(self.base[j])) / self.ndec[j]
                for k in self.names:
                    self.theta[k] += self.lr * adv * self.g[j][k]
                self.base[j] = 0.9 * self.base[j] + 0.1 * float(R[j])
        for k in ("log_tau_leader", "log_tau_follower", "log_tau_team"):
            self.theta[k] = float(np.clip(self.theta[k], -3.0, 3.0))
        for k in ("commit", "fp_leader", "fp_follower", "score_leader",
                  "score_follower", "suppress_appetite", "lead_bias",
                  "cart_inertia", "gram_weight"):
            self.theta[k] = float(np.clip(self.theta[k], -4.0, 4.0))
        self.eps += 1
        self._reset_episode()

    def table(self):
        t = self.theta
        return (f"tau_T {np.exp(t['log_tau_team']):.2f} "
                f"tau_L {np.exp(t['log_tau_leader']):.2f} "
                f"tau_F {np.exp(t['log_tau_follower']):.2f} "
                f"commit {t['commit']:.2f} fp_L {t['fp_leader']:.2f} "
                f"fp_F {t['fp_follower']:.2f} score_L {t['score_leader']:.2f} "
                f"score_F {t['score_follower']:.2f} "
                f"supp {t['suppress_appetite']:.2f} leadbias {t['lead_bias']:.2f} "
                f"inertia {t['cart_inertia']:.2f} gram {t['gram_weight']:.2f}")

    def alloc_line(self, present):
        parts = []
        for j in present:
            A = self.alloc.get(j)
            if A is None:
                continue
            cs = " ".join(
                f"c{c}[p{A[c*3]:.2f} s{A[c*3+1]:.2f} e{A[c*3+2]:.2f}]"
                for c in range(K_CARTS))
            mark = "*" if j == self.team_lead else " "
            parts.append(f"j{j}{mark}{cs}")
        return "  ".join(parts)


def gram_line(kappa, ms):
    pairs = []
    for c in range(ms["carts"]):
        for a in range(1, TEAMS + 1):
            for b in range(a + 1, TEAMS + 1):
                v = float(kappa[(a - 1) * K_CARTS + c, (b - 1) * K_CARTS + c])
                if v != 0.0:
                    pairs.append((abs(v), f"c{c} j{a}-j{b} {v:+.2f}"))
    pairs.sort(reverse=True)
    return "gram contest: " + (" ".join(p[1] for p in pairs[:4]) if pairs else "none")


class SynthEnv:
    """A stub k-cart match with the regression rule.

    Each cart's controller is the team with the most push-role bots on it;
    control-point crossings bank score for the controller (watermarked, so a
    regressed cart re-banks nothing); suppression by non-controllers pushes
    progress back toward the origin, damped by the controller's escorts. One
    bot per team carries a dominant resource tuple."""

    def __init__(self, seed, teams=3, bots=6):
        self.g = np.random.default_rng(seed)
        self.nteams, self.bots = teams, bots
        self.prog = np.full(K_CARTS, 0.1)
        self.controller = np.zeros(K_CARTS, np.int32)
        self.score = np.zeros(TEAMS + 1)
        self.mark = np.zeros(K_CARTS, np.int32)
        self.cps = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
        self.held = {}

    def ms(self):
        n = self.nteams * self.bots
        bc = np.zeros(n, np.int32)
        for b, (j, c, r) in self.held.items():
            if 0 <= b < n:
                bc[b] = c
        return dict(carts=K_CARTS, prog=self.prog.copy(),
                    controller=self.controller.copy(),
                    score=self.score.copy(), bot_cart=bc)

    def rows(self):
        n = self.nteams * self.bots
        X = np.zeros((n, EXT_WIDTH), np.float32)
        X[:, 0] = np.arange(n)
        X[:, 1] = np.repeat(np.arange(1, self.nteams + 1), self.bots)
        X[:, 2] = 100.0
        X[:, 4] = 0.5
        X[:, 5:8] = (self.g.standard_normal((n, 3)) * 0.5).astype(np.float32)
        X[:, 11] = 100.0 + self.g.uniform(0, 50, n).astype(np.float32)
        X[:, 12] = self.prog[0]
        for i in range(n):
            hd = self.held.get(i)
            X[i, 15] = float(hd[2]) if hd and len(hd) > 2 else 0.0
        for j in range(1, self.nteams + 1):
            sel = X[:, 1] == j
            X[sel, 16] = 2.0
            X[sel, 18] = 20.0
            first = np.nonzero(sel)[0][0]
            X[first, 16] = 7.0
            X[first, 17] = 1.0
        return X

    def step(self, held):
        self.held = {b: (j, c, r) for b, (j, c, r) in held.items()}
        for c in range(K_CARTS):
            pushc = np.zeros(TEAMS + 1, np.int32)
            supp = esc = 0
            for b, (j, cc, r) in self.held.items():
                if cc != c:
                    continue
                if r == 0:
                    pushc[j] += 1
                elif r == 1 and j != self.controller[c]:
                    supp += 1
                elif r == 2 and j == self.controller[c]:
                    esc += 1
            if pushc.max() > 0 and pushc[self.controller[c]] < pushc.max():
                self.controller[c] = int(np.argmax(pushc))
            ctl = int(self.controller[c])
            d = 0.012 * max(0, pushc[ctl] - 1) - 0.008 * np.sqrt(max(0.0, supp - 0.5 * esc))
            d += float(self.g.standard_normal()) * 0.001
            self.prog[c] = float(np.clip(self.prog[c] + d, 0.0, 1.0))
            while (ctl and self.mark[c] < len(self.cps)
                   and self.prog[c] >= self.cps[self.mark[c]]):
                self.score[ctl] += 1
                self.mark[c] += 1
            if self.prog[c] >= 1.0:
                self.prog[c] = 0.1
                self.controller[c] = 0
                self.mark[c] = 0


def run_synth(episodes, T, seed, ep_ticks, hold, lr, theta_path, log=True,
              strat=None, train=True, lead_bias=None):
    solver = CtxSolver(T, "mlx" if mx is not None else "np")
    if strat is None:
        strat = Strategy(theta_path, seed, lr)
    else:
        strat.rng = np.random.default_rng(seed)
    if lead_bias is not None:
        strat.theta["lead_bias"] = lead_bias
    meter = MatchMeter()
    env = SynthEnv(seed + 1)
    banked, rets = [], []
    gang_m, gang_p, conc = [], [], []
    for ep in range(episodes):
        for t in range(ep_ticks):
            ms = env.ms()
            X = env.rows()
            if t % hold == 0:
                G = solver.solve(X, ms["bot_cart"])
                mixes = solver.mixes()
                strat.tick(X, solver.G2, mixes, solver.kappa, ms, True, train)
                lt = strat.team_lead
                lead_carts = [c for c in range(K_CARTS)
                              if ms["controller"][c] == lt]
                for j in range(1, env.nteams + 1):
                    A = strat.alloc.get(j)
                    if A is None:
                        continue
                    conc.append(max(float(A[c*3:(c+1)*3].sum())
                                    for c in range(K_CARTS)))
                    if j != lt and lead_carts:
                        gang_p.append(sum(float(A[c*3+1]) for c in lead_carts))
                for b, (j, c, r) in strat.held_view(X).items():
                    if j != lt and lead_carts:
                        gang_m.append(float(r == 1 and c in lead_carts))
            else:
                strat.tick(X, solver.G2, solver.mixes(), solver.kappa, ms,
                           False, train)
            meter.tick(X, ms)
            env.step(strat.held_view(X))
        R, dirs, dprog, dbank, push = meter.finish()
        if train:
            strat.update(R)
            strat.save()
        banked.append(float(dbank[1:env.nteams + 1].mean()))
        rets.append(float(np.mean([R[j] for j in range(1, env.nteams + 1)])))
        if log and (ep < 3 or (ep + 1) % 50 == 0):
            print(f"synth: ep {ep+1:4d} banked {banked[-1]:5.2f} "
                  f"return {rets[-1]:+7.3f} lead j{strat.team_lead} "
                  f"{strat.table()}", flush=True)
            print(f"synth: alloc {strat.alloc_line(range(1, env.nteams + 1))}",
                  flush=True)
            print("synth: " + gram_line(solver.kappa, ms), flush=True)
    stats = dict(gang=float(np.mean(gang_m)) if gang_m else 0.0,
                 pred=float(np.mean(gang_p)) if gang_p else 0.0,
                 conc=float(np.mean(conc[-max(1, len(conc)//4):])) if conc else 0.0)
    return np.array(rets), np.array(banked), strat, stats


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
    ids = np.array([7, 8], np.int32)
    st._pick_leader(1, ids, np.array([0.675, 0.6]))
    first = st.leaders[1]
    st._pick_leader(1, ids, np.array([0.6, 0.675]))
    stay = st.leaders[1]
    st._pick_leader(1, ids, np.array([0.6, 0.9]))
    moved = st.leaders[1]
    report("leader hysteresis", first == 7 and stay == 7 and moved == 8,
           f"leader {first} held at +0.075 dominance ({stay}), moved at +0.30 ({moved}), "
           f"hysteresis {HYSTERESIS}")
    st2 = Strategy("", 1)
    ms0 = dict(carts=K_CARTS, prog=np.array([0.5, 0.5]),
               controller=np.array([1, 0], np.int32),
               score=np.array([0.0, 3.0, 2.8, 0.0, 0.0, 0.0]),
               bot_cart=np.zeros(2, np.int32))
    l1 = st2._team_leader(ms0, [1, 2, 3])
    ms0["score"][2] = 3.5
    l2 = st2._team_leader(ms0, [1, 2, 3])
    ms0["score"][2] = 4.5
    l3 = st2._team_leader(ms0, [1, 2, 3])
    report("team-leader hysteresis", l1 == 1 and l2 == 1 and l3 == 2,
           f"lead j{l1}, held under +{TEAM_HYST} dominance (j{l2}), moved past it (j{l3})")

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

        r1, b1, t1, _ = run_synth(12, 256, 42, 45, 15, 0.1, "", log=False)
        r2, b2, t2, _ = run_synth(12, 256, 42, 45, 15, 0.1, "", log=False)
        report("reinforce deterministic per seed",
               all(t1.theta[k] == t2.theta[k] for k in t1.names) and np.allclose(r1, r2),
               "two fresh 12-episode k-cart synth runs, identical scalars and returns")
        moved = max(abs(t1.theta[k] - THETA[k]) for k in t1.names)
        report("reinforce updates applied", moved > 0,
               f"max scalar movement {moved:.3f} after 12 episodes")
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
        rets, banked, strat, stats = run_synth(
            a.synth, a.ctx, a.train_seed, a.episode, a.hold, a.lr,
            a.theta if a.theta != THETA_PATH else "")
        k = max(1, min(100, a.synth // 4))
        print(f"synth: return first{k} {rets[:k].mean():+.3f} -> last{k} "
              f"{rets[-k:].mean():+.3f}; team-mean banked {banked[:k].mean():.2f} -> "
              f"{banked[-k:].mean():.2f} (reported, not asserted: at the adversarial "
              f"equilibrium ganging up lowers the leader's banking by design)", flush=True)
        print(f"synth: gang-up: P(trailing suppression on leader cart) "
              f"{stats['gang']:.3f} (closed-form prediction {stats['pred']:.3f})",
              flush=True)
        print(f"synth: pigeonhole: mean max-cart allocation mass "
              f"{stats['conc']:.2f} (full-cover impossible; 0.50 = even split, "
              f"higher = commit or concede)", flush=True)
        ev = max(10, a.synth // 10)
        _, _, _, s_on = run_synth(ev, a.ctx, a.train_seed + 5, a.episode, a.hold,
                                  a.lr, "", log=False,
                                  strat=Strategy("", a.train_seed + 5), train=False)
        _, _, _, s_off = run_synth(ev, a.ctx, a.train_seed + 5, a.episode, a.hold,
                                   a.lr, "", log=False,
                                   strat=Strategy("", a.train_seed + 5), train=False,
                                   lead_bias=0.0)
        print(f"synth: counterfactual at init scalars: gang-up {s_on['gang']:.3f} "
              f"with lead_bias {THETA['lead_bias']} vs {s_off['gang']:.3f} with "
              f"lead_bias 0 (the commitment channel isolated)", flush=True)
        print(f"synth: learned scalars: {strat.table()}", flush=True)
        good = (rets[-k:].mean() > rets[:k].mean()
                and abs(stats["gang"] - stats["pred"]) < 0.06
                and stats["conc"] > 0.55
                and s_on["gang"] > s_off["gang"] + 0.03)
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
        meter = MatchMeter()
        mode = "greedy eval, no updates" if a.eval else "sampling + online REINFORCE"
        print(f"worker: dominance = {DOM_FORMULA} (width-16 stream reads "
              f"weapons/powerup/since_spawn as 0)", flush=True)
        print(f"worker: allocation = inter-team Stackelberg: the score/tempo-dominant "
              f"team (hysteresis {TEAM_HYST}) commits its (cart, role) allocation over "
              f"push/suppress/escort first; trailing teams best-respond with lead_bias "
              f"on its committed push mass, suppress_appetite on enemy carts, "
              f"cart_inertia, and the window Gram's team-pair contest coupling",
              flush=True)
        print(f"worker: strategy = per-team Stackelberg nested inside the allocation: "
              f"most dominant bot leads node choice (hysteresis {HYSTERESIS}) by quantal "
              f"response over the solve's per-(cart,node) scores + the per-(team,cart) "
              f"fictitious-play mix; a suppress role flips the score slice (regression "
              f"toward the cart origin)", flush=True)
        print(f"worker: reward = banked control points (absolute + margin over the "
              f"best rival) per "
              f"{a.episode}-tick episode + 2 * signed per-cart progress (directions "
              f"measured from progress-vs-occupancy) + 0.05 * push-radius time",
              flush=True)
        print(f"worker: scalars {strat.table()}", flush=True)
        print(f"worker: ({mode}, hold {a.hold}, lr {a.lr}, file {strat.path}, "
              f"episodes so far {strat.eps})", flush=True)
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
        if trained:
            ms = bind_match(X)
            G = solver.solve(X, ms["bot_cart"])
            meter.tick(X, ms)
            boundary = tick_in_ep % a.hold == 0
            pick, Wt, lead, cart = strat.tick(X, solver.G2, solver.mixes(),
                                              solver.kappa, ms, boundary, not a.eval)
            tick_in_ep += 1
            if tick_in_ep >= a.episode:
                R, dirs, dprog, dbank, push = meter.finish()
                if not a.eval:
                    strat.update(R)
                    strat.save()
                present = sorted(set(X[:, 1].astype(np.int32)) & set(range(1, TEAMS + 1)))
                print(f"learn: ep {strat.eps:4d} "
                      f"dprog {[round(float(v), 4) for v in dprog]} "
                      f"banked {[round(float(dbank[j]), 1) for j in range(1, TEAMS + 1)]} "
                      f"returns {[round(float(R[j]), 3) for j in range(1, TEAMS + 1)]} "
                      f"lead j{strat.team_lead}", flush=True)
                print(f"learn: {strat.table()}", flush=True)
                print(f"learn: alloc {strat.alloc_line(present)}", flush=True)
                print("learn: " + gram_line(solver.kappa, ms), flush=True)
                tick_in_ep = 0
        else:
            G = solver.solve(X)
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
