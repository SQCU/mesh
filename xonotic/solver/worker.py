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

The trained policy is analytic, never a neural controller. Above the
per-team game, the score/tempo-dominant team commits an allocation over
instruments - cart cells (push/suppress/escort) plus timed item-post
control cells read from the wire's item block - and each team's
bots fill that allocation's tempered quotas by closed-form transport
(dominance-sorted for posts, proximity-sorted for carts), so a distribution
over instruments is expressible and everyone-on-the-leader is just its
low-temperature corner. Item respawn phase is estimated from the
availability bit's toggle history in the context window; holding a post
raises a bot's resource tuple, which raises measured dominance, which can
shift leadership - the feedback loop is the point. Per team the most
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
from xonwire import (Mesh, Reassembler, TxWindow, parse_hdr, REQ, RESP, REQ_WIDTH,
                     RESP_WIDTH, TEAMS, rows_per_slot, pack_hdr, HDRSZ)

SEED, EXPERTS, FF, HID = 20260828, 8, 2048, 64
CTX, RES = 4096, 2048
EXT_WIDTH = 20
K_CARTS = 4
ROLES = ("push", "suppress", "escort")
TEAM_HYST = 0.2
P_MAX = 8
NINST = K_CARTS + P_MAX
NALLOC = K_CARTS * len(("p", "s", "e")) + P_MAX
NBUCKET = 2
ITEM_BASE = K_CARTS * TEAMS
ITEM_COLS = 5
ITEM_COL = 20 + 3 * K_CARTS
ITEM_HORIZON = 20.0
PUSH_RADIUS = 512.0
HYSTERESIS = 0.15
THETA_PATH = "~/.mesh-plc-strategy.npz"
POLICIES = {"nearest": np.argmax, "inverted": np.argmin}
THETA = dict(log_tau_leader=0.0, log_tau_follower=0.0, commit=1.0,
             fp_leader=0.5, fp_follower=0.5, score_leader=1.0, score_follower=1.0,
             log_tau_team=0.0, suppress_appetite=0.8, lead_bias=0.8,
             cart_inertia=0.3, gram_weight=0.5, item_appetite=0.8,
             log_tau_portfolio=0.0)
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
        self.kappa = np.zeros((TEAMS * NINST, TEAMS * NINST), np.float32)
        self.white = np.ones(RES, np.float32)
        self.Vg = np.zeros((TEAMS * NINST, RES), np.float32)

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
        comb = np.clip(X[:, 15], 0, ITEM_BASE + P_MAX - 1).astype(np.int32)
        self.m_obj[idx] = np.where(comb < ITEM_BASE, comb % TEAMS, 0)
        self.m_cart[idx] = np.where(comb < ITEM_BASE, comb // TEAMS,
                                    K_CARTS + (comb - ITEM_BASE))
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
        M = np.zeros((TEAMS * NINST, self.T), np.float32)
        for j in range(1, TEAMS + 1):
            for c in range(NINST):
                sel = (self.m_team == j) & (self.m_cart == c)
                k = float(sel.sum())
                if k:
                    i = (j - 1) * NINST + c
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
        G2 = Zcur @ (Gm @ w["Og2"])
        V = mx.array(self._group_mask()) @ Z
        Kp = V @ Gm @ V.T
        mss = mx.mean(Z * Z, axis=0)
        mx.eval(G, Zc, G2, Kp, V, mss)
        self.Zc = np.nan_to_num(np.asarray(memoryview(Zc))).astype(np.float32)
        self.G2 = np.nan_to_num(np.asarray(memoryview(G2))).astype(np.float32)
        self.kappa = self._kappa_norm(np.asarray(memoryview(Kp)).astype(np.float64))
        self.Vg = np.nan_to_num(np.asarray(memoryview(V))).astype(np.float32)
        self.white = (1.0 / np.sqrt(np.maximum(
            np.asarray(memoryview(mss)), 0.0) + 1e-6)).astype(np.float32)
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
            self.Vg = np.nan_to_num(V).astype(np.float32)
            self.white = (1.0 / np.sqrt(np.maximum(
                np.mean(Z * Z, axis=0), 0.0) + 1e-6)).astype(np.float32)
            return Zcur @ (Gm @ w["Og"])


def softmax(z):
    z = z - np.max(z)
    e = np.exp(z)
    return e / e.sum()


def bind_match(X):
    """The PORT.md section 2 binding. The header width is authoritative:
    width >= 23 carries per-cart state blocks at columns 20+3c (progress,
    controlling team, regression flag) up to PLC_MAX_CARTS = 4 (columns
    20..31); narrower streams are the single-cart game read from column 12.
    The item-post block (8 posts, 5 columns each: class rank with 0 = no
    post, origin/1024, availability bit) follows the cart blocks at column
    32. Width 66 is the superseded two-cart item layout - carts at 20..25,
    posts at 26 - and still binds as it did then. Column 15 is the combined
    objective index everywhere (cart*5+node, posts from ITEM_BASE = 20).
    Banked score is not on the wire - the meter measures it from
    control-point crossings."""
    n, wid = X.shape
    if wid == 66:
        carts, item_col = 2, 26
    else:
        carts = min(K_CARTS, max(1, (wid - 20) // 3)) if wid >= 23 else 1
        item_col = ITEM_COL
    prog = np.zeros(K_CARTS, np.float64)
    controller = np.zeros(K_CARTS, np.int32)
    regress = np.zeros(K_CARTS, np.float64)
    if wid >= 23:
        for c in range(carts):
            prog[c] = float(X[0, 20 + 3 * c])
            controller[c] = int(np.clip(X[0, 21 + 3 * c], 0, TEAMS))
            regress[c] = float(X[0, 22 + 3 * c])
    else:
        prog[0] = float(X[0, 12])
    posts = []
    if wid >= item_col + ITEM_COLS:
        for pp in range(min(P_MAX, (wid - item_col) // ITEM_COLS)):
            b = item_col + ITEM_COLS * pp
            posts.append(dict(rank=float(X[0, b]),
                              pos=X[0, b + 1:b + 4].astype(np.float64),
                              avail=float(X[0, b + 4]) > 0.5))
    comb = np.clip(X[:, 15], 0, ITEM_BASE + P_MAX - 1).astype(np.int32)
    bot_inst = np.where(comb < ITEM_BASE, comb // TEAMS,
                        K_CARTS + (comb - ITEM_BASE)).astype(np.int32)
    return dict(carts=carts, prog=prog, controller=controller, regress=regress,
                score=np.zeros(TEAMS + 1, np.float64), posts=posts,
                bot_cart=np.clip(bot_inst, 0, K_CARTS - 1),
                bot_inst=np.clip(bot_inst, 0, NINST - 1))


class ItemTracker:
    """Fictitious play over item timers: nothing on the wire says when a
    post respawns, so the period is the median of observed down-gaps in the
    availability bit's toggle history, and the phase is ticks since it was
    taken. timing() is 1 while a post is up, ramps toward 1 over the last
    ITEM_HORIZON ticks before the estimated respawn, and sits at a small
    prior while the period is still unobserved."""

    def __init__(self):
        self.t = 0
        self.prev = {}
        self.down_t = {}
        self.gaps = {}
        self.period = {}

    def tick(self, posts):
        self.t += 1
        for i, post in enumerate(posts):
            a = bool(post["avail"])
            was = self.prev.get(i)
            if was is True and not a:
                self.down_t[i] = self.t
            if was is False and a and i in self.down_t:
                g = self.gaps.setdefault(i, [])
                g.append(self.t - self.down_t[i])
                del g[:-8]
                self.period[i] = float(np.median(g))
            self.prev[i] = a

    def timing(self, i, avail):
        if avail:
            return 1.0
        if i in self.period and i in self.down_t:
            tt = max(0.0, self.period[i] - (self.t - self.down_t[i]))
            return float(np.clip(1.0 - tt / ITEM_HORIZON, 0.0, 1.0))
        return 0.3


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
        self.mark = np.zeros(K_CARTS, np.int32)
        self.cps = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
        self.cum = np.zeros(TEAMS + 1, np.float64)
        self._reset_episode()

    def _reset_episode(self):
        self.dprog = np.zeros(K_CARTS)
        self.dbank = np.zeros(TEAMS + 1)
        self.push = np.zeros(TEAMS + 1)
        self.rows = np.zeros(TEAMS + 1)

    def tick(self, X, ms):
        prog = np.asarray(ms["prog"], np.float64)
        first = self.last_prog is None
        d = np.zeros(K_CARTS) if first else prog - self.last_prog
        for c in range(K_CARTS):
            if first or d[c] < -0.5:
                self.mark[c] = int(np.sum(self.cps <= prog[c] + 1e-9))
                continue
            ctl = int(ms["controller"][c])
            while (ctl and self.mark[c] < len(self.cps)
                   and prog[c] >= self.cps[self.mark[c]] - 1e-9):
                self.dbank[ctl] += 1
                self.cum[ctl] += 1
                self.mark[c] += 1
        self.last_prog = prog.copy()
        d[np.abs(d) > 0.05] = 0.0
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

    def __init__(self, path, seed=SEED, lr=0.1, lr_w=0.5, reg=0.01, qkv=False):
        self.path = os.path.expanduser(path) if path else ""
        self.lr, self.lr_w, self.reg, self.qkv = lr, lr_w, reg, qkv
        self.rng = np.random.default_rng(seed)
        self.names = sorted(THETA)
        self.theta = dict(THETA)
        self.W = np.zeros((NBUCKET, RES, NALLOC), np.float32)
        self.M = np.ones(RES, np.float32)
        self.qkvw = 0.0
        self.base = np.zeros(TEAMS + 1, np.float64)
        self.eps = 0
        if self.path and os.path.exists(self.path):
            z = np.load(self.path, allow_pickle=False)
            saved = {str(k): float(v) for k, v in zip(z["names"], z["theta"])} \
                if "names" in z else {}
            for k in self.names:
                if k in saved:
                    self.theta[k] = saved[k]
            if "W" in z.files and z["W"].shape == self.W.shape:
                self.W = z["W"].astype(np.float32)
            if "M" in z.files and z["M"].shape == self.M.shape:
                self.M = z["M"].astype(np.float32)
            if "qkvw" in z.files:
                self.qkvw = float(z["qkvw"])
            self.base = z["base"].astype(np.float64)
            self.eps = int(z["eps"])
        self.leaders = {}
        self.team_lead = None
        self.alloc = {}
        self.acomps = {}
        self.prev_alloc = {}
        self.last_quota = {}
        self.held = {}
        self.held_leader = {}
        self._reset_episode()

    def _reset_episode(self):
        self.g = {j: {k: 0.0 for k in self.names} for j in range(1, TEAMS + 1)}
        self.gW = {}
        self.gM = {}
        self.gqkvw = {}
        self.qkv_state = {}
        self.ent = {}
        self.ndec = np.zeros(TEAMS + 1, np.int64)

    def save(self):
        if not self.path:
            return
        tmp = self.path + ".tmp.npz"
        np.savez(tmp, names=np.array(self.names),
                 theta=np.array([self.theta[k] for k in self.names]),
                 W=self.W, M=self.M, qkvw=self.qkvw, base=self.base, eps=self.eps)
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
        tau_keys = [tau_key] if isinstance(tau_key, str) else list(tau_key)
        tau = float(np.exp(sum(self.theta[k] for k in tau_keys)))
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
            for k in tau_keys:
                gacc[k] += -(z[action] - float(pi @ z))
        return action, pi.astype(np.float32)

    def _allocate(self, ms, kappa, present, timing, dist, feats, white=None,
                  Vg=None):
        lead_team = self._team_leader(ms, present)
        order = [lead_team] + [j for j in present if j != lead_team]
        self.alloc, self.acomps = {}, {}
        P = len(ms.get("posts", []))
        nc = K_CARTS * len(ROLES)
        nA = nc + P
        for j in order:
            u0 = np.zeros(nA)
            supp = np.zeros(nA)
            leadp = np.zeros(nA)
            gram = np.zeros(nA)
            tcomp = np.zeros(nA)
            prev = self.prev_alloc.get(j)
            if prev is None or len(prev) != nA:
                prev = np.full(nA, 1.0 / nA)
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
                    gram[i + 1] = float(kappa[(j - 1) * NINST + c,
                                              (ctl - 1) * NINST + c])
                if j != lead_team and lead_team in self.alloc:
                    leadp[i + 1] = float(self.alloc[lead_team][i + 0])
            for pp in range(P):
                q = nc + pp
                post = ms["posts"][pp]
                if post["rank"] <= 0:
                    u0[q] = -30.0
                    continue
                u0[q] = 0.3 * float(post["rank"]) + 0.5 * float(dist[j][pp])
                tcomp[q] = float(timing[pp])
                gi = (j - 1) * NINST + K_CARTS + pp
                gram[q] = max((float(kappa[gi, (o - 1) * NINST + K_CARTS + pp])
                               for o in present if o != j), default=0.0)
                if j != lead_team and lead_team in self.alloc:
                    leadp[q] = float(self.alloc[lead_team][q])
            bucket = 1 if j == lead_team else 0
            phi = feats.get(j)
            if phi is not None:
                u0 = u0 + (phi @ self.W[bucket]).astype(np.float64)[:nA]
            if self.qkv and white is not None and Vg is not None and phi is not None:
                w2q = (white * white * phi).astype(np.float64)
                Kc = np.zeros((nA, RES), np.float64)
                for c in range(K_CARTS):
                    ks = [Vg[(o - 1) * NINST + c] for o in present if o != j]
                    if ks:
                        Kc[c * len(ROLES):c * len(ROLES) + len(ROLES)] = np.mean(ks, axis=0)
                for pp in range(P):
                    ks = [Vg[(o - 1) * NINST + K_CARTS + pp] for o in present if o != j]
                    if ks:
                        Kc[nc + pp] = np.mean(ks, axis=0)
                att = Kc @ (self.M * w2q)
                u0 = u0 + self.qkvw * att
                self.qkv_state[j] = (att, Kc, w2q)
            self.abucket = getattr(self, "abucket", {})
            self.acomps_phi = getattr(self, "acomps_phi", {})
            self.abucket[j] = bucket
            self.acomps_phi[j] = phi
            comps = dict(suppress_appetite=supp, lead_bias=leadp,
                         cart_inertia=prev, gram_weight=gram,
                         item_appetite=tcomp)
            _, pi = self._quantal(u0, comps,
                                  ["suppress_appetite", "lead_bias",
                                   "cart_inertia", "gram_weight",
                                   "item_appetite"],
                                  "log_tau_team", None)
            self.alloc[j] = pi.astype(np.float64)
            self.acomps[j] = (u0, comps)
            self.prev_alloc[j] = pi.astype(np.float64)
        return lead_team

    @staticmethod
    def _cell_inst(cell):
        nc = K_CARTS * len(ROLES)
        if cell < nc:
            return cell // len(ROLES), cell % len(ROLES)
        return K_CARTS + (cell - nc), 3

    @staticmethod
    def _quota(pi, n):
        q = np.floor(pi * n).astype(np.int64)
        rem = n - int(q.sum())
        if rem > 0:
            order = np.argsort(-(pi * n - q), kind="stable")
            q[order[:rem]] += 1
        return q

    def tick(self, X, G2, mixes, kappa, ms, boundary, sample, itimer=None,
             Zc=None, white=None, Vg=None):
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
        inst = np.zeros(n, np.int32)
        present = sorted(set(int(t) for t in team) & set(range(1, TEAMS + 1)))
        if not present:
            return pick, Wt, lead, inst
        posts = ms.get("posts", [])
        P = len(posts)
        nc = K_CARTS * len(ROLES)
        if boundary:
            timing = [itimer.timing(pp, posts[pp]["avail"]) if itimer
                      else (1.0 if posts[pp]["avail"] else 0.3)
                      for pp in range(P)]
            dist = {}
            feats = {}
            for j in present:
                rows = np.nonzero(team == j)[0]
                dist[j] = [float(np.clip(
                    1.0 - np.linalg.norm(X[rows, 5:8] - posts[pp]["pos"],
                                         axis=1).mean() / 2.0, 0.0, 1.0))
                    for pp in range(P)]
                if Zc is not None:
                    v = Zc[rows].mean(axis=0).astype(np.float64)
                    feats[j] = v / (np.linalg.norm(v) + 1e-9)
            self._allocate(ms, kappa, present, timing, dist, feats, white, Vg)
        nodemix = {}
        for j in present:
            others = [mixes[o]["nodes"] for o in mixes if o != j]
            nodemix[j] = (np.mean(others, axis=0) if others
                          else np.full((K_CARTS, TEAMS), 1.0 / TEAMS)).astype(np.float64)
        akeys = ["suppress_appetite", "lead_bias", "cart_inertia",
                 "gram_weight", "item_appetite"]
        for j in present:
            rows = np.nonzero(team == j)[0]
            if self.alloc.get(j) is None:
                continue
            li = self._pick_leader(j, ids[rows], dom[rows])
            lrow = rows[li]
            lead[lrow] = 1.0
            gacc = self.g[j] if (boundary and sample) else None
            if boundary:
                u0, comps = self.acomps[j]
                ai, pish = self._quantal(u0, comps, akeys,
                                         ["log_tau_team", "log_tau_portfolio"],
                                         gacc, None, sample)
                pv = np.maximum(np.asarray(pish, np.float64), 1e-12)
                self.ent.setdefault(j, []).append(float(-(pv * np.log(pv)).sum()))
                phi = self.acomps_phi.get(j) if hasattr(self, "acomps_phi") else None
                if gacc is not None:
                    self.ndec[j] += 1
                    if phi is not None:
                        tau_c = float(np.exp(self.theta["log_tau_team"]
                                             + self.theta["log_tau_portfolio"]))
                        gv = -np.asarray(pish, np.float64) / tau_c
                        gv[ai] += 1.0 / tau_c
                        b = self.abucket.get(j, 0)
                        gw = self.gW.setdefault(j, {}).setdefault(
                            b, np.zeros((RES, NALLOC), np.float64))
                        gw[:, :len(gv)] += np.outer(phi, gv)
                        if self.qkv and j in self.qkv_state:
                            att, Kc, w2q = self.qkv_state[j]
                            self.gqkvw[j] = self.gqkvw.get(j, 0.0) + float(gv @ att)
                            gmv = self.gM.setdefault(j, np.zeros(RES, np.float64))
                            gmv += self.qkvw * w2q * (gv @ Kc)
                quota = self._quota(np.asarray(pish, np.float64)
                                    / max(1e-9, float(np.sum(pish))), len(rows))
                if quota[ai] == 0:
                    quota[np.argmax(quota)] -= 1
                    quota[ai] += 1
                self.last_quota[j] = (quota.copy(), len(rows))
                assign = {int(ids[lrow]): ai}
                quota[ai] -= 1
                free = [r for r in rows if r != lrow]
                for cell in np.argsort(-pish, kind="stable"):
                    need = int(quota[cell])
                    if need <= 0 or not free:
                        continue
                    ci, role = self._cell_inst(int(cell))
                    if role == 3:
                        suit = [-float(dom[r]) for r in free]
                    else:
                        bc = np.clip(X[np.array(free), 15], 0,
                                     ITEM_BASE + P_MAX - 1).astype(np.int32) // TEAMS
                        suit = [float(X[r, 11]) + 800.0 * (int(bc[i2]) != ci)
                                for i2, r in enumerate(free)]
                    took = [free[i2] for i2 in np.argsort(suit, kind="stable")[:need]]
                    for r in took:
                        assign[int(ids[r])] = int(cell)
                        free.remove(r)
                for r in free:
                    assign[int(ids[r])] = ai
                for r in rows:
                    cell = assign[int(ids[r])]
                    ci, role = self._cell_inst(cell)
                    self.held[int(ids[r])] = (ci, role, 0)
            leader_hd = self.held.get(int(ids[lrow]))
            ordered = [lrow] + [r for r in rows if r != lrow]
            for r in ordered:
                bid = int(ids[r])
                hd = self.held.get(bid) or (0, 0, 0)
                ci, role = hd[0], hd[1]
                inst[r] = ci
                if role == 3:
                    pp = ci - K_CARTS
                    pick[r] = float(ITEM_BASE + pp)
                    W = np.zeros(TEAMS, np.float32)
                    W[0] = 1.0
                    Wt[r] = W
                    if boundary:
                        self.held[bid] = (ci, role, 0)
                    continue
                sgn = -1.0 if role == 1 else 1.0
                sl = sgn * Gs[r, ci]
                mnode = nodemix[j][ci]
                if r == lrow:
                    cn = dict(score_leader=sl, fp_leader=mnode)
                    a, pi = self._quantal(np.zeros(TEAMS), cn,
                                          ["score_leader", "fp_leader"],
                                          "log_tau_leader", gacc,
                                          None if boundary else hd[2],
                                          sample and boundary)
                    if boundary:
                        self.held_leader[j] = (ci, a)
                else:
                    lc, la = self.held_leader.get(j, (ci, None))
                    onehot = np.zeros(TEAMS)
                    if (la is not None and lc == ci and leader_hd
                            and leader_hd[1] != 3):
                        onehot[la] = 1.0
                    cn = dict(score_follower=sl, fp_follower=mnode, commit=onehot)
                    a, pi = self._quantal(np.zeros(TEAMS), cn,
                                          ["score_follower", "fp_follower", "commit"],
                                          "log_tau_follower", gacc,
                                          None if boundary else hd[2],
                                          sample and boundary)
                if boundary:
                    self.held[bid] = (ci, role, a)
                    if gacc is not None:
                        self.ndec[j] += 1
                pick[r] = float(a + TEAMS * ci)
                Wt[r] = pi
        return pick, Wt, lead, inst

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
                out[int(ids[r])] = (j, hd[0], hd[1], hd[2] if hd[2] is not None else 0)
        return out

    def update(self, R):
        for j in range(1, TEAMS + 1):
            if self.ndec[j]:
                adv = (float(R[j]) - float(self.base[j])) / self.ndec[j]
                for k in self.names:
                    self.theta[k] += self.lr * adv * self.g[j][k]
                for b, gw in self.gW.get(j, {}).items():
                    self.W[b] += (self.lr_w * adv * gw).astype(np.float32)
                if self.qkv:
                    self.qkvw += self.lr * adv * self.gqkvw.get(j, 0.0)
                    if j in self.gM:
                        self.M += (self.lr_w * adv * self.gM[j]).astype(np.float32)
                self.base[j] = 0.9 * self.base[j] + 0.1 * float(R[j])
        if self.reg > 0:
            self.W *= 1.0 - self.reg
            if self.qkv:
                self.M += self.reg * (1.0 - self.M)
                self.qkvw *= 1.0 - self.reg
            for k in self.names:
                self.theta[k] += self.reg * (THETA[k] - self.theta[k])
        for k in ("log_tau_leader", "log_tau_follower", "log_tau_team"):
            self.theta[k] = float(np.clip(self.theta[k], -3.0, 3.0))
        self.theta["log_tau_portfolio"] = float(
            np.clip(self.theta["log_tau_portfolio"], -3.0, 3.0))
        self.qkvw = float(np.clip(self.qkvw, -4.0, 4.0))
        for k in ("commit", "fp_leader", "fp_follower", "score_leader",
                  "score_follower", "suppress_appetite", "lead_bias",
                  "cart_inertia", "gram_weight", "item_appetite"):
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
                f"inertia {t['cart_inertia']:.2f} gram {t['gram_weight']:.2f} "
                f"item {t['item_appetite']:.2f} tau_P {np.exp(t['log_tau_portfolio']):.2f}")

    def alloc_line(self, present):
        parts = []
        for j in present:
            A = self.alloc.get(j)
            if A is None:
                continue
            cs = " ".join(
                f"c{c}[p{A[c*3]:.2f} s{A[c*3+1]:.2f} e{A[c*3+2]:.2f}]"
                for c in range(K_CARTS))
            nc = K_CARTS * len(ROLES)
            if len(A) > nc:
                cs += " " + " ".join(f"i{pp}[{A[nc+pp]:.2f}]"
                                     for pp in range(len(A) - nc))
            mark = "*" if j == self.team_lead else " "
            parts.append(f"j{j}{mark}{cs}")
        return "  ".join(parts)


def gram_line(kappa, ms):
    pairs = []
    insts = list(range(ms["carts"])) + [K_CARTS + pp
                                         for pp in range(len(ms.get("posts", [])))]
    for c in insts:
        nm = f"c{c}" if c < K_CARTS else f"i{c - K_CARTS}"
        for a in range(1, TEAMS + 1):
            for b in range(a + 1, TEAMS + 1):
                v = float(kappa[(a - 1) * NINST + c, (b - 1) * NINST + c])
                if v != 0.0:
                    pairs.append((abs(v), f"{nm} j{a}-j{b} {v:+.2f}"))
    pairs.sort(reverse=True)
    return "gram contest: " + (" ".join(p[1] for p in pairs[:4]) if pairs else "none")


class SynthEnv:
    """A stub k-cart match with the regression rule.

    Each cart's controller is the team with the most push-role bots on it;
    control-point crossings bank score for the controller (watermarked, so a
    regressed cart re-banks nothing); suppression by non-controllers pushes
    progress back toward the origin, damped by the controller's escorts. One
    bot per team carries a dominant resource tuple."""

    def __init__(self, seed, teams=3, bots=8, items=True):
        self.g = np.random.default_rng(seed)
        self.nteams, self.bots = teams, bots
        self.prog = np.full(K_CARTS, 0.1)
        self.controller = np.zeros(K_CARTS, np.int32)
        self.score = np.zeros(TEAMS + 1)
        self.mark = np.zeros(K_CARTS, np.int32)
        self.cps = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
        self.regress = np.zeros(K_CARTS, bool)
        self.held = {}
        self.tick_n = 0
        self.posts = [
            dict(pos=np.array([0.8, 0.2, 0.1]), rank=2.0, period=37,
                 avail=True, respawn=0, up_t=None),
            dict(pos=np.array([-0.6, 0.7, 0.0]), rank=1.0, period=53,
                 avail=True, respawn=0, up_t=None),
            dict(pos=np.array([0.1, -0.8, 0.2]), rank=1.0, period=44,
                 avail=True, respawn=0, up_t=None)]
        if not items:
            self.posts = []
        self.boost = {}
        self.delays = []

    def rows(self):
        n = self.nteams * self.bots
        X = np.zeros((n, ITEM_COL + ITEM_COLS * len(self.posts)), np.float32)
        X[:, 0] = np.arange(n)
        X[:, 1] = np.repeat(np.arange(1, self.nteams + 1), self.bots)
        X[:, 2] = 100.0
        X[:, 4] = 0.5
        X[:, 5:8] = (self.g.standard_normal((n, 3)) * 0.5).astype(np.float32)
        X[:, 11] = 100.0 + self.g.uniform(0, 50, n).astype(np.float32)
        X[:, 12] = self.prog[0]
        for i in range(n):
            hd = self.held.get(i)
            if hd:
                X[i, 15] = float(hd[1] * TEAMS + hd[3]) if hd[1] < K_CARTS \
                    else float(ITEM_BASE + hd[1] - K_CARTS)
                if hd[1] >= K_CARTS:
                    X[i, 5:8] = (self.posts[hd[1] - K_CARTS]["pos"]
                                 + self.g.standard_normal(3) * 0.05)
        for j in range(1, self.nteams + 1):
            sel = X[:, 1] == j
            X[sel, 16] = 2.0
            X[sel, 18] = 20.0
            first = np.nonzero(sel)[0][0]
            X[first, 16] = 7.0
            X[first, 17] = 1.0
        for i in range(n):
            if self.boost.get(i, -1) >= self.tick_n:
                X[i, 16] = 7.0
                X[i, 17] = 1.0
        for c in range(K_CARTS):
            X[:, 20 + 3 * c] = self.prog[c]
            X[:, 21 + 3 * c] = float(self.controller[c])
            X[:, 22 + 3 * c] = 1.0 if self.regress[c] else 0.0
        for pp, post in enumerate(self.posts):
            b = ITEM_COL + ITEM_COLS * pp
            X[:, b] = post["rank"]
            X[:, b + 1:b + 4] = post["pos"].astype(np.float32)
            X[:, b + 4] = 1.0 if post["avail"] else 0.0
        return X

    def step(self, held):
        self.held = dict(held)
        self.tick_n += 1
        for pp, post in enumerate(self.posts):
            inst = K_CARTS + pp
            if not post["avail"]:
                if self.tick_n >= post["respawn"]:
                    post["avail"] = True
                    post["up_t"] = self.tick_n
                continue
            if post["avail"]:
                ctrl = np.zeros(TEAMS + 1, np.int32)
                for b, (j, cc, r, nd) in self.held.items():
                    if cc == inst and r == 3:
                        ctrl[j] += 1
                if ctrl.max() > 0:
                    captor = int(np.argmax(ctrl))
                    post["avail"] = False
                    post["respawn"] = self.tick_n + post["period"]
                    if post["up_t"] is not None:
                        self.delays.append(self.tick_n - post["up_t"])
                    for b, (j, cc, r, nd) in self.held.items():
                        if cc == inst and r == 3 and j == captor:
                            self.boost[b] = self.tick_n + 25
        for c in range(K_CARTS):
            if self.prog[c] >= 1.0:
                self.prog[c] = 0.1
                self.controller[c] = 0
                self.mark[c] = 0
                continue
            pushc = np.zeros(TEAMS + 1, np.int32)
            supp = esc = 0
            for b, (j, cc, r, nd) in self.held.items():
                if cc != c:
                    continue
                if r == 0:
                    pushc[j] += 3 if self.boost.get(b, -1) >= self.tick_n else 1
                elif r == 1 and j != self.controller[c]:
                    supp += 1
                elif r == 2 and j == self.controller[c]:
                    esc += 1
            if pushc.max() > 0 and pushc[self.controller[c]] < pushc.max():
                self.controller[c] = int(np.argmax(pushc))
            ctl = int(self.controller[c])
            d = 0.02 * pushc[ctl] - 0.012 * np.sqrt(max(0.0, supp - 0.5 * esc))
            d += float(self.g.standard_normal()) * 0.001
            self.regress[c] = d < 0
            self.prog[c] = float(np.clip(self.prog[c] + d, 0.0, 1.0))
            while (ctl and self.mark[c] < len(self.cps)
                   and self.prog[c] >= self.cps[self.mark[c]] - 1e-9):
                self.score[ctl] += 1
                self.mark[c] += 1


def run_synth(episodes, T, seed, ep_ticks, hold, lr, theta_path, log=True,
              strat=None, train=True, overrides=None, explore=None,
              env_items=True, lr_w=0.5, reg=0.01, qkv=False):
    solver = CtxSolver(T, "mlx" if mx is not None else "np")
    if strat is None:
        strat = Strategy(theta_path, seed, lr, lr_w, reg, qkv)
    else:
        strat.rng = np.random.default_rng(seed)
    for k, v in (overrides or {}).items():
        strat.theta[k] = v
    sample = train if explore is None else explore
    meter = MatchMeter()
    itimer = ItemTracker()
    env = SynthEnv(seed + 1, items=env_items)
    banked, rets = [], []
    gang_m, gang_p, conc, spread = [], [], [], []
    anti, waste, ents, support = [], [], [], []
    for ep in range(episodes):
        ep_insts = {}
        for t in range(ep_ticks):
            X = env.rows()
            ms = bind_match(X)
            itimer.tick(ms["posts"])
            meter.tick(X, ms)
            ms["score"] = meter.cum.copy()
            if t % hold == 0:
                G = solver.solve(X)
                mixes = solver.mixes()
                strat.tick(X, solver.G2, mixes, solver.kappa, ms, True, sample,
                           itimer, solver.Zc, solver.white, solver.Vg)
                lt = strat.team_lead
                lead_carts = [c for c in range(K_CARTS)
                              if ms["controller"][c] == lt]
                for j in range(1, env.nteams + 1):
                    A = strat.alloc.get(j)
                    if A is None:
                        continue
                    nc2 = K_CARTS * len(ROLES)
                    imass = [float(A[c*3:(c+1)*3].sum()) for c in range(K_CARTS)] \
                        + [float(v) for v in A[nc2:]]
                    conc.append(max(imass))
                    if j != lt and lead_carts and j in strat.last_quota:
                        q, nb = strat.last_quota[j]
                        gang_p.append(sum(float(q[c*3+1]) for c in lead_carts) / nb)
                hv = strat.held_view(X)
                per_team = {}
                for b, (j, c, r, nd) in hv.items():
                    if j != lt and lead_carts:
                        gang_m.append(float(r == 1 and c in lead_carts))
                    per_team.setdefault(j, set()).add((c, r))
                for j, cells in per_team.items():
                    spread.append(len(set(c for c, r in cells)))
                    ep_insts.setdefault(j, set()).update(c for c, r in cells)
                for pp, post in enumerate(env.posts):
                    if not post["avail"]:
                        tt = post["respawn"] - env.tick_n
                        m = sum(float(strat.alloc[j][K_CARTS * len(ROLES) + pp])
                                for j in strat.alloc)
                        (anti if tt <= ITEM_HORIZON else waste).append(m)
            else:
                strat.tick(X, solver.G2, solver.mixes(), solver.kappa, ms,
                           False, sample, itimer, solver.Zc, solver.white, solver.Vg)
            env.step(strat.held_view(X))
        R, dirs, dprog, dbank, push = meter.finish()
        ee = [float(np.mean(v)) for v in strat.ent.values() if v]
        if ee:
            ents.append(float(np.mean(ee)))
        support.extend(len(v) for v in ep_insts.values())
        if train:
            strat.update(R)
            strat.save()
        else:
            strat._reset_episode()
        banked.append(float(dbank[1:env.nteams + 1].mean()))
        rets.append(float(np.mean([R[j] for j in range(1, env.nteams + 1)])))
        if log and (ep < 3 or (ep + 1) % 50 == 0):
            print(f"synth: ep {ep+1:4d} banked {banked[-1]:5.2f} "
                  f"return {rets[-1]:+7.3f} H {ents[-1] if ents else 0:.2f} "
                  f"|W| {np.abs(strat.W).max():.2f} lead j{strat.team_lead} "
                  f"{strat.table()}", flush=True)
            print(f"synth: alloc {strat.alloc_line(range(1, env.nteams + 1))}",
                  flush=True)
            print("synth: " + gram_line(solver.kappa, ms), flush=True)
    stats = dict(gang=float(np.mean(gang_m)) if gang_m else 0.0,
                 pred=float(np.mean(gang_p)) if gang_p else 0.0,
                 conc=float(np.mean(conc[-max(1, len(conc)//4):])) if conc else 0.0,
                 bank_meter=float(meter.cum[1:env.nteams + 1].sum()),
                 bank_truth=float(env.score[1:env.nteams + 1].sum()),
                 spread=float(np.mean(spread)) if spread else 0.0,
                 delay=float(np.mean(env.delays)) if env.delays else -1.0,
                 captures=len(env.delays),
                 anti=float(np.mean(anti)) if anti else 0.0,
                 waste=float(np.mean(waste)) if waste else 0.0,
                 entropy=float(np.mean(ents)) if ents else 0.0,
                 entropy_last=float(np.mean(ents[-max(1, len(ents)//4):]))
                 if ents else 0.0,
                 support=float(np.mean(support)) if support else 0.0,
                 wnorm=float(np.abs(strat.W).max()))
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
    X26 = np.zeros((4, 26), np.float32)
    X26[:, 15] = [7, 2, 9, 0]
    X26[0, 20], X26[0, 21], X26[0, 22] = 0.35, 2, 0
    X26[0, 23], X26[0, 24], X26[0, 25] = 0.8, 3, 1
    X26[:, 20:26] = X26[0, 20:26]
    msb = bind_match(X26)
    report("bind_match reads the width-26 contract",
           msb["carts"] == 2 and abs(msb["prog"][0] - 0.35) < 1e-6
           and abs(msb["prog"][1] - 0.8) < 1e-6
           and list(msb["controller"][:2]) == [2, 3]
           and list(msb["regress"][:2]) == [0, 1]
           and list(msb["bot_cart"]) == [1, 0, 1, 0],
           f"carts {msb['carts']} prog {[round(float(v),2) for v in msb['prog']]} "
           f"controllers {list(msb['controller'])} bot_cart {list(msb['bot_cart'])}")
    X66 = np.zeros((2, 66), np.float32)
    X66[:, 20:26] = X26[0, 20:26]
    X66[:, 26] = 3.0
    X66[:, 27:30] = [0.1, 0.2, 0.3]
    X66[:, 30] = 1.0
    X66[:, 31] = 4.0
    X66[:, 35] = 0.0
    ms66 = bind_match(X66)
    X72 = np.zeros((2, 72), np.float32)
    X72[:, 20:32] = np.tile([0.5, 1, 0], 4)
    X72[:, 26], X72[:, 29] = 0.7, 0.9
    X72[:, 32] = 6.0
    X72[:, 33:36] = [0.4, 0.5, 0.6]
    X72[:, 36] = 1.0
    ms72 = bind_match(X72)
    report("bind_match serves legacy 66 and current 72 side by side",
           ms66["carts"] == 2 and len(ms66["posts"]) == 8
           and ms66["posts"][0]["rank"] == 3.0 and ms66["posts"][0]["avail"]
           and ms66["posts"][1]["rank"] == 4.0 and not ms66["posts"][1]["avail"]
           and ms72["carts"] == 4 and abs(ms72["prog"][2] - 0.7) < 1e-6
           and abs(ms72["prog"][3] - 0.9) < 1e-6
           and len(ms72["posts"]) == 8 and ms72["posts"][0]["rank"] == 6.0
           and abs(ms72["posts"][0]["pos"][1] - 0.5) < 1e-6,
           f"66: carts {ms66['carts']} post0 rank {ms66['posts'][0]['rank']}; "
           f"72: carts {ms72['carts']} prog {[round(float(v),2) for v in ms72['prog']]} "
           f"post0 rank {ms72['posts'][0]['rank']}")
    mt = MatchMeter()
    Xd = np.zeros((2, 26), np.float32)
    Xd[:, 1] = 1
    mk = lambda p, ctl: dict(carts=K_CARTS,
                             prog=np.array([p] + [0.0] * (K_CARTS - 1)),
                             controller=np.array([ctl] + [0] * (K_CARTS - 1),
                                                 np.int32),
                             regress=np.zeros(K_CARTS),
                             score=np.zeros(TEAMS + 1),
                             bot_cart=np.zeros(2, np.int32))
    for pv in (0.15, 0.25, 0.18, 0.26, 0.99, 1.0, 0.1, 0.25):
        mt.tick(Xd, mk(pv, 1))
    report("meter banks watermarked crossings", mt.cum[1] == 6.0,
           f"0.15->0.25 (+1), regress-recross (+0), ->0.99 (+3), ->1.0 (+1), "
           f"reset, ->0.25 (+1): cum {int(mt.cum[1])} == 6")
    it = ItemTracker()
    seq = [1, 1, 1] + [0] * 10 + [1, 1] + [0] * 10 + [1] + [0] * 7
    for av in seq:
        it.tick([dict(avail=bool(av))])
    tim = it.timing(0, False)
    report("item tracker learns the respawn period",
           it.period.get(0) == 10.0 and abs(tim - 0.8) < 1e-9
           and it.timing(0, True) == 1.0,
           f"toggle history gives period {it.period.get(0)}, timing ramps to "
           f"{tim:.2f} four ticks out, 1.0 while up")
    st2 = Strategy("", 1)
    ms0 = dict(carts=K_CARTS,
               prog=np.array([0.5, 0.5] + [0.0] * (K_CARTS - 2)),
               controller=np.array([1, 0] + [0] * (K_CARTS - 2), np.int32),
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

        ro1, bo1, _, _ = run_synth(2, 256, 42, 45, 15, 0.1, "", log=False,
                                   train=False, explore=True)
        rq0, bq0, sq0, _ = run_synth(2, 256, 42, 45, 15, 0.1, "", log=False,
                                     train=False, explore=True, qkv=True)
        report("qkv inert at init (qkvw=0)",
               np.array_equal(ro1, rq0) and np.array_equal(bo1, bq0)
               and sq0.qkvw == 0.0 and np.all(sq0.M == 1.0),
               "qkv on with no updates reproduces the qkv-off returns bit-for-bit")
        rq1, _, tq1, _ = run_synth(12, 256, 42, 45, 15, 0.1, "", log=False, qkv=True)
        rq2, _, tq2, _ = run_synth(12, 256, 42, 45, 15, 0.1, "", log=False, qkv=True)
        dev = float(np.max(np.abs(tq1.M - 1.0)))
        report("qkv trains, deterministic and bounded",
               np.array_equal(rq1, rq2) and tq1.qkvw == tq2.qkvw
               and np.array_equal(tq1.M, tq2.M)
               and (abs(tq1.qkvw) > 0 or dev > 0)
               and abs(tq1.qkvw) <= 4.0 and dev < 5.0 and np.all(np.isfinite(tq1.M)),
               f"qkvw {tq1.qkvw:+.3f} max|M-1| {dev:.3f} after 12 episodes, "
               f"identical across two runs")
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
    ap.add_argument("--lr-w", type=float, default=0.5)
    ap.add_argument("--reg", type=float, default=0.01)
    ap.add_argument("--qkv", action="store_true")
    a = ap.parse_args()

    if a.check:
        sys.exit(0 if check(a.rows) else 1)
    if a.bench:
        bench(a.bench, a.rows, a.ctx)
        return
    if a.synth:
        rets, banked, strat, stats = run_synth(
            a.synth, a.ctx, a.train_seed, a.episode, a.hold, a.lr,
            a.theta if a.theta != THETA_PATH else "",
            lr_w=a.lr_w, reg=a.reg, qkv=a.qkv)
        k = max(1, min(100, a.synth // 4))
        print(f"synth: mean shared-theta return (the REINFORCE objective) first{k} "
              f"{rets[:k].mean():+.3f} -> last{k} {rets[-k:].mean():+.3f}; team-mean "
              f"banked {banked[:k].mean():.2f} -> {banked[-k:].mean():.2f} (banked "
              f"reported, not asserted: the untrained broad portfolio is already "
              f"near-maximally productive, and competitive play trades collective "
              f"banking for margin)", flush=True)
        print(f"synth: gang-up: P(trailing suppression on leader cart) "
              f"{stats['gang']:.3f} (closed-form prediction {stats['pred']:.3f})",
              flush=True)
        print(f"synth: portfolio concentration (reported): mean max instrument "
              f"mass {stats['conc']:.2f} over the last quarter; the expressible "
              f"range is asserted by the tau_P knob below", flush=True)
        print(f"synth: banking measured from wire crossings {stats['bank_meter']:.0f} "
              f"vs environment truth {stats['bank_truth']:.0f}", flush=True)
        ev = max(10, a.synth // 10)
        _, _, _, s_on = run_synth(ev, a.ctx, a.train_seed + 5, a.episode, a.hold,
                                  a.lr, "", log=False,
                                  strat=Strategy("", a.train_seed + 5), train=False,
                                  overrides={"suppress_appetite": 0.0,
                                             "lead_bias": 2.0},
                                  explore=True, env_items=False)
        _, _, _, s_off = run_synth(ev, a.ctx, a.train_seed + 5, a.episode, a.hold,
                                   a.lr, "", log=False,
                                   strat=Strategy("", a.train_seed + 5), train=False,
                                   overrides={"suppress_appetite": 0.0,
                                              "lead_bias": 0.0},
                                   explore=True, env_items=False)
        print(f"synth: counterfactual at init scalars: gang-up {s_on['gang']:.3f} "
              f"with lead_bias 2 vs {s_off['gang']:.3f} with lead_bias 0, both at "
              f"suppress_appetite 0, cart-only game - all suppression must flow "
              f"through the committed-leader term", flush=True)
        _, _, _, p_hi = run_synth(ev, a.ctx, a.train_seed + 9, a.episode, a.hold,
                                  a.lr, "", log=False,
                                  strat=Strategy("", a.train_seed + 9), train=False,
                                  overrides={"log_tau_portfolio": 1.5}, explore=True)
        _, _, _, p_lo = run_synth(ev, a.ctx, a.train_seed + 9, a.episode, a.hold,
                                  a.lr, "", log=False,
                                  strat=Strategy("", a.train_seed + 9), train=False,
                                  overrides={"log_tau_portfolio": -2.0}, explore=True)
        print(f"synth: portfolio concentration knob: mean instruments per team "
              f"{p_hi['spread']:.2f} at tau_P e^1.5 vs {p_lo['spread']:.2f} at "
              f"tau_P e^-2 (low tau_P = everyone on the leader's instrument)",
              flush=True)
        _, _, _, i_on = run_synth(ev, a.ctx, a.train_seed + 13, a.episode, a.hold,
                                  a.lr, "", log=False,
                                  strat=Strategy("", a.train_seed + 13), train=False,
                                  overrides={"item_appetite": 1.5}, explore=True)
        _, _, _, i_off = run_synth(ev, a.ctx, a.train_seed + 13, a.episode, a.hold,
                                   a.lr, "", log=False,
                                   strat=Strategy("", a.train_seed + 13), train=False,
                                   overrides={"item_appetite": 0.0}, explore=True)
        r_on = i_on["anti"] / (i_on["waste"] + 1e-9)
        r_off = i_off["anti"] / (i_off["waste"] + 1e-9)
        print(f"synth: item timing channel: allocation mass on a down post inside "
              f"the last {ITEM_HORIZON:.0f} ticks before its estimated respawn vs "
              f"earlier: {i_on['anti']:.2f}/{i_on['waste']:.2f} (ratio {r_on:.2f}) "
              f"at item_appetite 1.5, {i_off['anti']:.2f}/{i_off['waste']:.2f} "
              f"(ratio {r_off:.2f}) at 0; {i_on['captures']}/{i_off['captures']} "
              f"captures; the learned appetite is "
              f"{strat.theta['item_appetite']:+.2f} - whether an equilibrium buys "
              f"items at all is its own business", flush=True)
        _, _, _, z0 = run_synth(max(6, a.synth // 20), a.ctx, a.train_seed + 17,
                                a.episode, a.hold, a.lr, "", log=False,
                                strat=Strategy("", a.train_seed + 17), train=False,
                                explore=True)
        print(f"synth: zero-update policy: entropy {z0['entropy']:.2f} nats, "
              f"{z0['support']:.1f} distinct instruments actually sampled per team "
              f"per episode of {NALLOC} cells - broad weighted sampling from the "
              f"closed-form priors alone (W_logit = 0)", flush=True)
        _, _, nstrat, nr = run_synth(a.synth, a.ctx, a.train_seed, a.episode,
                                     a.hold, a.lr, "", log=False,
                                     lr_w=a.lr_w, reg=0.0, qkv=a.qkv)
        print(f"synth: regularizer counterfactual: last-quarter policy entropy "
              f"{stats['entropy_last']:.2f} nats with L2 decay {a.reg}/update "
              f"(max|W| {stats['wnorm']:.2f}) vs {nr['entropy_last']:.2f} with "
              f"decay 0 (max|W| {nr['wnorm']:.2f}) - the floor is the "
              f"regularizer's work", flush=True)
        print(f"synth: learned scalars: {strat.table()}", flush=True)
        good = (rets[-k:].mean() > rets[:k].mean()
                and banked[-k:].mean() > 0
                and abs(stats["gang"] - stats["pred"]) < 0.06
                and s_on["gang"] > s_off["gang"] + 0.03
                and abs(stats["bank_meter"] - stats["bank_truth"]) < 2.5
                and p_hi["spread"] > p_lo["spread"] + 1.5
                and p_lo["spread"] < 2.5
                and i_on["captures"] > 0 and i_off["captures"] > 0
                and r_on > r_off + 0.15
                and z0["entropy"] > 1.2 and z0["support"] >= 6.0
                and stats["entropy_last"] > nr["entropy_last"] + 0.1
                and stats["entropy_last"] > 0.8)
        sys.exit(0 if good else 1)

    trained = a.policy == "trained"
    pick_of = POLICIES.get(a.policy, np.argmax)
    m = attach(a.peer)
    rxs = {}
    tx = TxWindow(m)
    bk = "mlx" if mx is not None else "np"
    solver = CtxSolver(a.ctx, bk)
    backend = f"{bk}-ctx(T={a.ctx},RES={RES},FF={FF})"
    print(f"worker: policy={a.policy} backend={backend} peer={a.peer} "
          f"usable={m.usable} resp_rows/slot={rows_per_slot(m.usable, RESP_WIDTH)} "
          f"request width from each block's header (16/20/26 all served, "
          f"combined cart*5+node objectives)", flush=True)

    strat = meter = itimer = None
    if trained:
        strat = Strategy(a.theta, a.train_seed, a.lr, a.lr_w, a.reg, a.qkv)
        meter = MatchMeter()
        itimer = ItemTracker()
        mode = "greedy eval, no updates" if a.eval else "sampling + online REINFORCE"
        print(f"worker: dominance = {DOM_FORMULA} (width-16 stream reads "
              f"weapons/powerup/since_spawn as 0)", flush=True)
        print(f"worker: allocation = inter-team Stackelberg: the score/tempo-dominant "
              f"team (hysteresis {TEAM_HYST}) commits its allocation over cart cells "
              f"(push/suppress/escort) plus item-post control cells first; trailing "
              f"teams best-respond with lead_bias on its committed mass, "
              f"suppress_appetite on enemy carts, cart_inertia, item_appetite on "
              f"post timing, and the window Gram's contest coupling", flush=True)
        print(f"worker: policy = a sampled, regularized logit field: W_logit "
              f"({NBUCKET} buckets x {RES} Gram-conditioned features x {NALLOC} "
              f"instrument cells = {NBUCKET * RES * NALLOC} parameters) adds to the "
              f"closed-form priors; allocations and picks are SAMPLED from its "
              f"softmax, REINFORCE moves it (lr_w {a.lr_w}), and L2 decay "
              f"{a.reg}/update pulls it back toward the priors so untrained play is "
              f"a broad weighted sampling of effective strategies and trained play "
              f"peaks without collapsing; the {len(strat.names)} named scalars stay "
              f"the temperature/appetite knobs", flush=True)
        print(f"worker: portfolio = followers fill the tempered allocation's quotas by "
              f"closed-form transport (suitability: dominance for posts, proximity "
              f"for carts); low tau_P reproduces everyone-on-the-leader as one "
              f"expressible shape among many", flush=True)
        print(f"worker: item posts per PORT.md: 8 posts x 5 columns (class rank, "
              f"origin, availability) at column {ITEM_COL} after the four cart "
              f"state blocks (legacy width 66 still binds with posts at 26); "
              f"respawn phase is estimated from the availability bit's toggle "
              f"history in the window (fictitious play over timers)", flush=True)
        print(f"worker: feedback loop by design: holding a post raises that bot's "
              f"resource tuple on later rows, which raises measured dominance, which "
              f"shifts intra-team leadership and team strength", flush=True)
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
        done, src, rx = None, a.peer, None
        for buf, s in m.read(dtype=np.uint8):
            h = parse_hdr(buf)
            if h is None or h["kind"] != REQ:
                continue
            r = rxs.get(h["width"])
            if r is None:
                r = rxs[h["width"]] = Reassembler(REQ, h["width"], a.maxrows, m.usable)
            d = r.feed(buf)
            if d:
                done, src, rx = d, s, r
        if done is None:
            time.sleep(0.0002)
            continue
        n = done["rows"]
        X = rx.stage[:n]
        if trained:
            ms = bind_match(X)
            G = solver.solve(X)
            itimer.tick(ms.get("posts", []))
            meter.tick(X, ms)
            ms["score"] = meter.cum.copy()
            boundary = tick_in_ep % a.hold == 0
            pick, Wt, lead, inst = strat.tick(X, solver.G2, solver.mixes(),
                                              solver.kappa, ms, boundary,
                                              not a.eval, itimer, solver.Zc,
                                              solver.white, solver.Vg)
            tick_in_ep += 1
            if tick_in_ep >= a.episode:
                R, dirs, dprog, dbank, push = meter.finish()
                ent = {j: round(float(np.mean(v)), 2)
                       for j, v in strat.ent.items() if v}
                if not a.eval:
                    strat.update(R)
                    strat.save()
                else:
                    strat._reset_episode()
                present = sorted(set(X[:, 1].astype(np.int32)) & set(range(1, TEAMS + 1)))
                print(f"learn: ep {strat.eps:4d} "
                      f"dprog {[round(float(v), 4) for v in dprog]} "
                      f"banked {[round(float(dbank[j]), 1) for j in range(1, TEAMS + 1)]} "
                      f"returns {[round(float(R[j]), 3) for j in range(1, TEAMS + 1)]} "
                      f"lead j{strat.team_lead} policy-entropy {ent} "
                      f"|W| {np.abs(strat.W).max():.2f}", flush=True)
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
            hist = np.bincount(pick.astype(np.int32) % TEAMS, minlength=TEAMS).tolist()
            back = np.bincount(np.clip(X[:, 15], 0, K_CARTS * TEAMS - 1).astype(np.int32) % TEAMS,
                               minlength=TEAMS).tolist()
            col = lambda t, c: float(X[X[:, 1] == t, c].mean()) if (X[:, 1] == t).any() else 0.0
            tr.write(",".join(str(v) for v in
                     [done["req_id"], done["tick"], int((X[:, 1] > 0).sum()),
                      float(X[0, 12])] + hist + back +
                     [col(1, 11), col(2, 11), col(1, 5), col(2, 5), col(1, 6), col(2, 6)]) + "\n")
        if not a.quiet:
            hist = np.bincount(pick.astype(np.int32) % TEAMS, minlength=TEAMS).tolist()
            live = int((X[:, 1] > 0).sum())
            back = np.bincount(np.clip(X[:, 15], 0, K_CARTS * TEAMS - 1).astype(np.int32) % TEAMS,
                               minlength=TEAMS).tolist()
            print(f"worker: req {done['req_id']} tick {done['tick']} rows {n} live {live} "
                  f"-> node {src} chunks {took}/{chunks} picks {hist} held {back}", flush=True)
    print(f"worker: {blocks} blocks, {served} rows, short {short}, "
          f"dropped {sum(r.dropped for r in rxs.values())}", flush=True)


if __name__ == "__main__":
    main()
