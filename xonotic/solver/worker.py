"""The solver worker: the mesh answers the game server's bot-plan requests (ef6adb10^:xonotic/solver/worker.py,
restored onto HEAD's wire and library).

The numerical solve is one prepared Function: request broadcast, persistent
context update, indexed matrix products, reductions and column placement.
The game policy reads its result pages after completion. There is no eager
NumPy/MPI/Torch numerical executor.

    PYTHONPATH=xonotic:rdma bin/mesh-python -B -m solver.worker serve CONFIG.json
    PYTHONPATH=xonotic:rdma bin/mesh-python -B -m solver.worker --bench 20
    PYTHONPATH=xonotic:rdma bin/mesh-python -B -m solver.worker --synth 400

`serve` is one rank per node. Rank zero receives the engine's local wire frames,
declares each request shape, and supplies the request pages. The compiled
broadcast distributes their values; administrative messages contain only the
request kind and shape. Device/compiler capabilities and column shares are
preparation operands from the service configuration.

Request rows (bridge/PORT.md section 2 as the solver read it on 08-29, width 72): id, team, health, armor, ammo,
origin/1024 (3), velocity/1024 (3), distance to the nearest cart, its progress, friends and enemies within 700 units,
the team's held objective, weapon count, powerup, time since spawn, pad; four cart blocks (progress, controlling team,
regression flag) at 20+3c; eight item posts (class rank, origin/1024, availability) at 32+5p.  At HEAD the game
publishes OBSERVATION/CART/TEAM/EVENT rows (payload/STRATEGY-IO.md) and request_rows derives these columns from them;
the held objective is the per-team vote over this worker's own last answer (sv_payload_mesh.qc's vote, which the game
held then); item posts are the items the bots perceived (EVENT ITEM_HERE / ITEM_GONE), each at class rank 1, since the
wire carries no item class.  Response rows (width 8): id, the pick (cart*5+node, or ITEM_BASE+post), the five node
weights, leader flag.

The heavy solve is the design's workload (design/mesh-coprocessor-demo.md): per-tick rows join a persistent context
window of CTX recent rows, the window is lifted into a RES-wide residual basis, routed through EXPERTS experts grouped
per expert, and a Gram matrix over the residual conditions the current rows' objective scores.  CTX=4096 holds the
measured T/E >= 512 rows-per-expert floor and RES=2048 the orthogonalisation floor: ~104 GFLOP against ~270 MB of
resident weights per solve.  On a world it is Megatron-LM's tensor parallelism [Shoeybi et al. 2019] as
planner/plan.py divides its experts: rank r holds the columns `columns.parts[r]` of every expert's up projection and the
same rows of its down projection, so its expert output is a partial sum one all-reduce completes; the window's Gram
Z^T Z is then column-parallel over the same share of its RES columns, its products with the objective bases and the
group means row-parallel (one all-reduce of their partial sums), the conditioned current rows gathered by columns
(the compiled ragged gather). The routing is argmax(C (Ein Rt)) in float64, lowered to the declared CPU scalar compiler, so every
rank routes every row alike.  The shares are rdma/allocate.py's Balancer's, moved every tick on the tick's own times
(the expert all-reduce's last row carries each rank's own work from the rows' arrival to the all-reduce, as
plan.py's carries its segment).

Policies over those scores:
  nearest / inverted   frozen A/B pair, argmax / argmin, disjoint on every row
  trained              closed-form Stackelberg strategy, learnable only in a
                       small vector of named scalars and the logit field

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
mixture and score weights) and the logit field W_logit (ac3b543b: 2 context
buckets x 2048 Gram-conditioned features x 20 instrument cells, L2 decay
toward logit 0) against rewards measured from the request stream:
signed cart progress per episode, in a goal direction itself measured from
progress-versus-occupancy correlation, plus a small push-radius term. The
solver emits strategy at the request rate; havocbot keeps all
millisecond-scale motor control.
"""
import argparse, json, os, select, signal, socket, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rdma"))
sys.path.insert(0, str(ROOT / "xonotic"))

from solver.xonwire import (Engine, RuntimeFrames, frames, kind_of, report,
                            OBSERVATION_KIND, CART_KIND, TEAM_KIND)
from payload.tools.strategy_io_schema import CS, EVT, EVT_KIND, OBS, TS

SEED, EXPERTS, FF, HID = 20260828, 8, 2048, 64
CTX, RES = 4096, 2048
REQ_WIDTH, RESP_WIDTH, TEAMS = 16, 8, 5
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
REQUEST_WIDTH = ITEM_COL + ITEM_COLS * P_MAX
OBJECTIVES = TEAMS
COMBINED = OBJECTIVES * K_CARTS
TOTAL = COMBINED + P_MAX
MESH_RADIUS = 700.0
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
NOUT = TEAMS + K_CARTS * TEAMS
NV = TEAMS * NINST

import torch
import torch.distributed._functional_collectives as collective
from torch_mesh import _stream
from torch_mesh.declarations import Function
from torch_mesh.engines import host
from torch_mesh.partition import allgatherv
from mesh_statistics import now, measure
from prepared import World, pages


def _init(seed):
    g = np.random.default_rng(seed)
    return lambda *s: (g.standard_normal(s) * (1.0 / np.sqrt(s[-2]))).astype(np.float32)


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
    (seed, request history). The source is captured by Function. `world`
    and `columns` declare the collective and each rank's share; compiler
    configuration chooses the hardware implementation."""

    def __init__(self, T=CTX, device="mps", world=None, columns=None, assignment_data=None, configuration=None):
        self.T, self.device, self.configuration = T, device, configuration
        self.world, self.assignment_data = world, assignment_data
        self.previous_function = None
        self.pending_feedback = None
        self.columns = list(columns) if columns else [FF]
        if world is not None and len(self.columns) != world.size:
            raise ValueError(f"columns {self.columns}: one part a rank of {world.size}")
        if sum(self.columns) != FF:
            raise ValueError(f"columns {self.columns} must sum to {FF}")
        w = ctx_weights(REQ_WIDTH)
        self.route = pages(w["Ein"].astype(np.float64) @ w["Rt"].astype(np.float64), device="cpu")
        w["W1T"] = np.ascontiguousarray(w.pop("W1").transpose(0, 2, 1))
        w["O"] = np.ascontiguousarray(np.concatenate((w.pop("Og"), w.pop("Og2")), axis=1))
        self.w = {k: pages(v, device=device) for k, v in w.items()}
        g = np.random.default_rng(SEED + 1)
        C0 = (g.standard_normal((T, REQ_WIDTH)) * 0.5).astype(np.float32)
        self.context = pages((C0 / (1.0 + np.abs(C0))).astype(np.float32), device=device)
        self.C = host(self.context).numpy()
        self.pos = 0
        self.expert_rows = np.zeros(EXPERTS, np.int64)
        self.Zc = None
        self.teams = pages(np.full(T, -1, np.int64), device=device)
        self.m_team = host(self.teams).numpy()
        self.m_obj = np.full(T, -1, np.int32)
        self.carts = pages(np.zeros(T, np.int64), device=device)
        self.m_cart = host(self.carts).numpy()
        self.m_dom = np.zeros(T, np.float32)
        self.m_pos = np.zeros((T, 3), np.float32)
        self.G2 = None
        self.kappa = np.zeros((TEAMS * NINST, TEAMS * NINST), np.float32)
        self.white = np.ones(RES, np.float32)
        self.Vg = np.zeros((TEAMS * NINST, RES), np.float32)
        self.programs = {}
        self.active = None
        self.solves = 0

    def rank(self):
        return self.world.rank if self.world is not None else 0

    def span(self, extent, rank=None):
        rank = self.rank() if rank is None else rank
        lo = sum(self.columns[:rank])
        return lo * extent // FF, (lo + self.columns[rank]) * extent // FF

    def _insert(self, X, idx):
        # Application policy state, after the graph's input broadcast completed.
        # The numerical graph itself updates context/team/cart page operands.
        comb = np.clip(X[:, 15], 0, ITEM_BASE + P_MAX - 1).astype(np.int32)
        self.m_obj[idx] = np.where(comb < ITEM_BASE, comb % TEAMS, 0)
        self.m_dom[idx] = dominance(X)
        self.m_pos[idx] = X[:, 5:8]

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

    def configure(self, rows):
        # design/whole-function.md
        if self.assignment_data is None or self.world is None:
            return None
        import mesh
        import assignment
        from mesh_source import revision
        data = self.assignment_data
        workload = {'rows': rows, 'context': self.T, 'residual': RES, 'columns': FF,
                    'experts': EXPERTS, 'objectives': NOUT, 'group_values': NV}
        family = {'source': 'xonotic-context-solver', 'members': self.world.members, 'workload': workload}
        graphs = data.get('graphs', {})
        decisions = assignment.domains(data['decisions'], graphs)
        observations = [row for row in data['observations'] if row['workload'] == workload]
        proposal = assignment.assign(decisions, observations, step=self.solves, family=family) if self.rank() == 0 else None
        proposal = mesh.broadcast_configuration(self.world.native, proposal)
        self.columns = list(proposal['parts']['columns'])
        return proposal

    def _declare(self, rows):
        key = rows, tuple(self.columns)
        if key in self.programs:
            return self.programs[key]
        group = self.world.group if self.world else None
        T = self.T
        size = len(self.columns)
        lo, hi = self.span(FF)
        rlo, rhi = self.span(RES)
        widths = [b - a for a, b in (self.span(RES, r) for r in range(len(self.columns)))]
        request = pages(np.zeros((rows, REQUEST_WIDTH), np.float32), device=self.device)
        indices = pages(np.zeros(rows, np.int64), device=self.device)
        previous = pages(np.zeros(size, np.float32), device=self.device)
        w = self.w

        def source(X, idx, prior, C, teams, carts, route, Ein, W1T, W2, O):
            if group is not None:
                X = collective.wait_tensor(collective.broadcast(X, 0, group.group_name))
            features = X[:, :REQ_WIDTH]
            C.copy_(C.index_copy(0, idx, features / (1.0 + torch.abs(features))))
            teams.copy_(teams.index_copy(0, idx, X[:, 1].to(torch.int64)))
            combined = X[:, 15].clamp(0, ITEM_BASE + P_MAX - 1).to(torch.int64)
            cart = torch.where(combined < ITEM_BASE, combined // TEAMS, K_CARTS + combined - ITEM_BASE)
            carts.copy_(carts.index_copy(0, idx, cart))
            # This small replicated router keeps its declared double arithmetic.
            # Its scalar contraction is compiled for a capable CPU implementation.
            routing = (C.to(device='cpu', dtype=torch.float64)[:, :, None] * route[None]).sum(dim=1)
            selected = routing.argmax(dim=1)
            H = C @ Ein
            first = W1T[:, lo:hi].transpose(1, 2).index_select(0, selected)
            second = W2[:, lo:hi].index_select(0, selected)
            Y = torch.bmm(torch.relu(torch.bmm(H[:, None, :], first)), second)[:, 0, :]
            complete = torch.cat((Y.reshape(-1), prior))
            if group is not None:
                complete = collective.wait_tensor(collective.all_reduce(complete, 'sum', group.group_name))
            Z = H + complete[:T * RES].reshape(T, RES)
            squared = Z * Z
            tr = squared.sum() / T
            mss = squared.mean(dim=0)
            Zcur = Z.index_select(0, idx)
            labels = torch.arange(NV, device=teams.device)
            members = (teams[None, :] == (labels // NINST + 1)[:, None]) & (carts[None, :] == (labels % NINST)[:, None])
            counts = members.sum(dim=1, keepdim=True)
            mask = torch.where(counts > 0, members.to(Z.dtype) / counts.clamp(min=1) - 1.0 / T, 0.0)
            V = mask @ Z
            Gm = Z.T @ Z[:, rlo:rhi] / T
            P = torch.cat(((Gm @ O[rlo:rhi]).reshape(-1), ((V @ Gm) @ V[:, rlo:rhi].T).reshape(-1)))
            if group is not None:
                P = collective.wait_tensor(collective.all_reduce(P, 'sum', group.group_name))
            GmO = P[:RES * NOUT].reshape(RES, NOUT)
            G = torch.nan_to_num(Zcur @ GmO[:, :TEAMS])
            G2 = torch.nan_to_num(Zcur @ GmO[:, TEAMS:])
            block = (Zcur @ Gm) * (RES / tr)
            Zc = allgatherv(block, 1, widths, group.group_name) if group is not None else block
            K = P[RES * NOUT:].reshape(NV, NV).to(device='cpu', dtype=torch.float64)
            d = torch.sqrt(torch.diagonal(K).clamp(min=0))
            norm = torch.nan_to_num(K / (d[:, None] * d[None, :] + 1e-9))
            kappa = (norm * (1 - torch.eye(NV, device=K.device, dtype=K.dtype))).to(device=Zcur.device, dtype=torch.float32)
            white = torch.rsqrt(mss.clamp(min=0) + 1e-6)
            expert_rows = (selected[:, None] == torch.arange(EXPERTS, device=selected.device)[None]).sum(dim=0)
            return dict(G=G, G2=G2, Zc=torch.nan_to_num(Zc), kappa=kappa, Vg=torch.nan_to_num(V),
                        white=white, expert_rows=expert_rows, request=X, times=complete[T * RES:])

        function = Function(source, request, indices, previous, self.context, self.teams, self.carts,
                            self.route, w['Ein'], w['W1T'], w['W2'], w['O'], configuration=self.configuration,
                            source_scope={'function': 'xonotic-context-solver'})
        result_shapes = dict(G=(rows, TEAMS), G2=(rows, NOUT - TEAMS), Zc=(rows, RES), kappa=(NV, NV),
                             Vg=(NV, RES), white=(RES,), expert_rows=(EXPERTS,), request=(rows, REQUEST_WIDTH), times=(size,))
        destinations = {name: pages(np.zeros(shape, np.int64 if name == 'expert_rows' else np.float32),
                                    device='cpu' if name == 'expert_rows' else self.device)
                        for name, shape in result_shapes.items()}
        calls = function.declare(group, destinations=destinations.values())
        state = dict(function=function, calls=calls, request=host(request).numpy(), indices=host(indices).numpy(),
                     previous=host(previous).numpy(), results={name: host(value).numpy() for name, value in destinations.items()})
        self.programs.clear()
        self.programs[key] = state
        return state

    def prepare(self, rows):
        """Resolve this invocation's graph and routing before any numerical work."""
        import assignment
        from mesh_source import revision
        proposal = self.configure(rows)
        state = self._declare(rows)
        if self.world is not None:
            data = self.assignment_data
            graphs = data.get('graphs', {}) if data else {}
            routing = assignment.route_operands(proposal['parts'], 'world', graphs['world']) if graphs and proposal else None
            self.world.group.plan(state['calls'], routing=routing)
            if data is not None:
                graphs = {'world': assignment.graph(self.world.native, self.world.group._plan)}
                data['graphs'] = graphs
                data['decisions'] = {key: {**value, 'parts': proposal['parts'][key]}
                                     for key, value in data['decisions'].items()}
                realized = assignment.domains(data['decisions'], graphs)
                proposal.update(parts={key: value['parts'] for key, value in realized.items()},
                                domains=assignment.choice_domains(realized), graphs=graphs,
                                workload={'rows': rows, 'context': self.T, 'residual': RES, 'columns': FF,
                                          'experts': EXPERTS, 'objectives': NOUT, 'group_values': NV},
                                implementation={'source': revision(ROOT), 'compiler': self.configuration},
                                episode=f'context-solver:{self.solves}')
        state['function'].prepare()
        state['proposal'] = proposal
        self.active = state
        return state

    def submit(self, X):
        """Populate application inputs, then submit the already prepared function."""
        state = self.active
        n = len(state['indices'])
        state['request'][...] = 0
        state['request'][:, :X.shape[1]] = X[-n:]
        state['indices'][...] = (self.pos + np.arange(n)) % self.T if n < self.T else np.arange(self.T)
        self.pos = (self.pos + n) % self.T if n < self.T else 0
        state['previous'][...] = 0
        state['previous'][self.rank()] = self.previous_function['seconds'] if self.previous_function else 0.0
        self.started = now()
        state['function'].run()
        self.completion = state['function'].completion()
        return self.completion

    def result(self, ended):
        """Read pages after the application's completion observation."""
        state, values = self.active, self.active['results']
        self._insert(values['request'], state['indices'])
        self.observe(values['times'])
        self.Zc, self.G2, self.kappa = values['Zc'], values['G2'], values['kappa']
        self.Vg, self.white, self.expert_rows = values['Vg'], values['white'], values['expert_rows']
        self.solves += 1
        useful = sum(value.nbytes for key, value in values.items() if key not in ('times', 'request'))
        self.previous_function = {'proposal': state['proposal'], **measure(self.started, ended, useful, len(values['G']))}
        if self.pending_feedback is not None:
            previous, seconds = self.pending_feedback
            self.pending_feedback = None
            self.assignment_data['observations'].append({**{key: previous[key] for key in
                ('parts', 'domains', 'graphs', 'family', 'workload', 'implementation', 'proposal', 'episode')},
                'scope': 'function', 'complete': True, 'seconds': seconds,
                'timing_scope': 'maximum rank whole-solve latency carried by the existing reduction'})
            import assignment
            self.assignment_data['observations'][:] = assignment.window(self.assignment_data['observations'])
        return values['G']


    def observe(self, times):
        # Previous complete solve is carried by an existing numerical reduction.
        # Never change the columns of the solve that is currently executing.
        previous = self.previous_function
        if self.assignment_data is None or self.world is None or previous is None:
            return
        self.pending_feedback = previous['proposal'], max(float(t) for t in times[:self.world.size])


def softmax(z):
    z = z - np.max(z)
    e = np.exp(z)
    return e / e.sum()


def bind_match(X):
    """The PORT.md section 2 binding. The header width is authoritative:
    width >= 23 carries per-cart state blocks at columns 20+3c (progress,
    controlling team, regression flag) up to K_CARTS = 4 (columns
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


class ItemPosts:
    """The item posts the bots perceived: each distinct item an EVENT row names (ITEM_HERE / ITEM_GONE, the game's
    perception in payload_perceive) in the order first seen, up to P_MAX, its availability its latest event's."""

    def __init__(self):
        self.posts = {}

    def observe(self, events):
        rows = np.concatenate([np.asarray(part, np.float32).reshape(-1, EVT["EPISODE"] + 1) for part in events]) \
            if events else np.zeros((0, EVT["EPISODE"] + 1), np.float32)
        for row in rows[np.argsort(rows[:, EVT["TIME"]], kind="stable")]:
            kind = int(row[EVT["KIND"]])
            if kind not in (EVT_KIND["ITEM_HERE"], EVT_KIND["ITEM_GONE"]):
                continue
            subject = int(row[EVT["SUBJECT"]])
            if subject in self.posts or len(self.posts) < P_MAX:
                self.posts[subject] = (row[EVT["POS_X"]:EVT["POS_Z"] + 1] / 1024.0, kind == EVT_KIND["ITEM_HERE"])

    def ranks(self):
        out = np.zeros(P_MAX, np.float32)
        out[:len(self.posts)] = 1.0
        return out


def request_rows(observations, carts, objective, posts):
    """HEAD's OBSERVATION and CART rows (payload/tools/strategy_io_schema.py) bound to the request layout above, as
    sv_payload_mesh.qc's payload_mesh_tick filled it (08-29): a row a client slot, zero where the slot holds no
    player; the cart and item blocks the same on every row."""
    obs = np.asarray(observations, np.float32)
    n = len(obs)
    X = np.zeros((n, REQUEST_WIDTH), np.float32)
    if not n:
        return X
    player = (obs[:, OBS["PRESENT"]] > 0) & (obs[:, OBS["TEAM"]] >= 1)
    team = np.where(player, obs[:, OBS["TEAM"]], 0).astype(np.int64)
    pos = obs[:, OBS["POS_X"]:OBS["POS_Z"] + 1]
    vel = obs[:, OBS["VEL_X"]:OBS["VEL_Z"] + 1]
    X[:, 0] = np.where(player, obs[:, OBS["ID"]], 0)
    X[:, 1] = team
    X[:, 2] = np.where(player, obs[:, OBS["HEALTH"]], 0)
    X[:, 3] = np.where(player, obs[:, OBS["ARMOR"]], 0)
    ammo = sum(obs[:, OBS[name]] for name in ("AMMO_SHELLS", "AMMO_BULLETS", "AMMO_ROCKETS", "AMMO_CELLS"))
    X[:, 4] = np.where(player, np.clip(ammo / 400.0, 0, 1), 0)
    X[:, 5:8] = np.where(player[:, None], pos / 1024.0, 0)
    X[:, 8:11] = np.where(player[:, None], vel / 1024.0, 0)
    cart = np.asarray(carts, np.float32).reshape(-1, CS["CHECKPOINT_COUNT"] + 1)
    cart = cart[np.argsort(cart[:, CS["ID"]], kind="stable")]
    length = cart[:, CS["PATH_LENGTH"]]
    progress = np.divide(cart[:, CS["PATH_POSITION"]], length, out=np.zeros_like(length), where=length > 0)
    if len(cart):
        distance = np.linalg.norm(pos[:, None, :] - cart[None, :, CS["POS_X"]:CS["POS_Z"] + 1], axis=2)
        nearest = np.argmin(distance, axis=1)
        X[:, 11] = np.where(player, distance[np.arange(n), nearest], 0)
        X[:, 12] = np.where(player, progress[nearest], 0)
    apart = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
    near = (apart <= MESH_RADIUS) & player[:, None] & player[None, :] & ~np.eye(n, dtype=bool)
    same = team[:, None] == team[None, :]
    X[:, 13] = (near & same).sum(axis=1)
    X[:, 14] = (near & ~same).sum(axis=1)
    X[:, 15] = np.where((team >= 1) & (team <= TEAMS), objective[np.clip(team, 0, TEAMS)], 0)
    words = obs[:, OBS["WEAPONS_X"]:OBS["WEAPONS_Z"] + 1].astype(np.int64)
    X[:, 16] = np.where(player, sum(((words >> bit) & 1).sum(axis=1) for bit in range(24)), 0)
    X[:, 17] = np.where(player & (obs[:, OBS["STRENGTH_FINISHED"]] > obs[:, OBS["ENGINE_TIME"]]), 1, 0)
    X[:, 18] = np.where(player, np.clip(obs[:, OBS["ENGINE_TIME"]] - obs[:, OBS["SPAWN_TIME"]], 0, 999), 0)
    for c in range(min(K_CARTS, len(cart))):
        X[:, 20 + 3 * c] = progress[c]
        X[:, 21 + 3 * c] = cart[c, CS["CONTROL_TEAM"]]
        X[:, 22 + 3 * c] = 1.0 if cart[c, CS["SPEED"]] < 0 else 0.0
    for p, (origin, available) in enumerate(list(posts.posts.values())[:P_MAX]):
        b = ITEM_COL + ITEM_COLS * p
        X[:, b] = 1.0
        X[:, b + 1:b + 4] = origin
        X[:, b + 4] = 1.0 if available else 0.0
    return X


def vote(X, pick, carts, ranks, objective):
    """sv_payload_mesh.qc's per-team objective (08-29): each team's most-voted pick among its players' picks, a pick
    past the carts' objectives bound to the last, an item pick only on a live post; ties keep the earlier objective."""
    nobj = int(np.clip(carts * OBJECTIVES, 1, COMBINED))
    votes = np.zeros((TEAMS + 1, TOTAL))
    for j, o in zip(X[:, 1].astype(np.int64), np.floor(pick).astype(np.int64)):
        if not (ITEM_BASE <= o < TOTAL and ranks[o - ITEM_BASE] > 0):
            o = int(np.clip(o, 0, nobj - 1))
        if 1 <= j <= TEAMS:
            votes[j, o] += 1
    out = objective.copy()
    for j in range(1, TEAMS + 1):
        best = 0
        for o in range(TOTAL):
            if nobj <= o < ITEM_BASE:
                continue
            if votes[j, o] > best:
                best, out[j] = votes[j, o], o
    return out


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

    def tick(self, X, ms, reset):
        prog = np.asarray(ms["prog"], np.float64)
        first = self.last_prog is None
        d = np.zeros(K_CARTS) if first else prog - self.last_prog
        for c in range(K_CARTS):
            if first or reset[c]:
                self.mark[c] = int(np.sum(self.cps <= prog[c] + 1e-9))
                d[c] = 0.0
                continue
            ctl = int(ms["controller"][c])
            while (ctl and self.mark[c] < len(self.cps)
                   and prog[c] >= self.cps[self.mark[c]] - 1e-9):
                self.dbank[ctl] += 1
                self.cum[ctl] += 1
                self.mark[c] += 1
        self.last_prog = prog.copy()
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

    def __init__(self, state=None, seed=SEED, lr=0.1, lr_w=0.5, reg=0.01, qkv=False):
        self.lr, self.lr_w, self.reg, self.qkv = lr, lr_w, reg, qkv
        self.rng = np.random.default_rng(seed)
        self.names = sorted(THETA)
        self.theta = dict(THETA)
        self.W = np.zeros((NBUCKET, RES, NALLOC), np.float32)
        self.M = np.ones(RES, np.float32)
        self.qkvw = 0.0
        self.base = np.zeros(TEAMS + 1, np.float64)
        self.eps = 0
        if state:
            z = state
            saved = {str(k): float(v) for k, v in zip(z["names"], z["theta"])} \
                if "names" in z else {}
            for k in self.names:
                if k in saved:
                    self.theta[k] = saved[k]
            if "W" in z and z["W"].shape == self.W.shape:
                self.W = z["W"].astype(np.float32)
            if "M" in z and z["M"].shape == self.M.shape:
                self.M = z["M"].astype(np.float32)
            if "qkvw" in z:
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

    def state(self):
        return dict(names=np.array(self.names), theta=np.array([self.theta[k] for k in self.names]),
                    W=self.W.copy(), M=self.M.copy(), qkvw=self.qkvw, base=self.base.copy(), eps=self.eps)

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
                qh = np.nan_to_num(white * phi).astype(np.float64)
                qh = qh / (np.linalg.norm(qh) + 1e-9)
                Kc = np.zeros((nA, RES), np.float64)
                for c in range(K_CARTS):
                    ks = [Vg[(o - 1) * NINST + c] for o in present if o != j]
                    if ks:
                        Kc[c * len(ROLES):c * len(ROLES) + len(ROLES)] = np.mean(ks, axis=0)
                for pp in range(P):
                    ks = [Vg[(o - 1) * NINST + K_CARTS + pp] for o in present if o != j]
                    if ks:
                        Kc[nc + pp] = np.mean(ks, axis=0)
                Kc = np.nan_to_num(white * Kc)
                Kc = Kc / (np.linalg.norm(Kc, axis=1, keepdims=True) + 1e-9)
                att = np.nan_to_num(Kc @ (self.M * qh))
                u0 = u0 + self.qkvw * att
                self.qkv_state[j] = (att, Kc, qh)
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
                            att, Kc, qh = self.qkv_state[j]
                            self.gqkvw[j] = self.gqkvw.get(j, 0.0) + float(gv @ att)
                            gmv = self.gM.setdefault(j, np.zeros(RES, np.float64))
                            gmv += np.nan_to_num(self.qkvw * qh * (gv @ Kc))
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
        self.reset = np.zeros(K_CARTS, bool)
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
        self.reset = self.prog >= 1.0
        for c in range(K_CARTS):
            if self.reset[c]:
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


def run_synth(episodes, T, seed, ep_ticks, hold, lr, theta_values=None, log=True,
              strat=None, train=True, overrides=None, explore=None,
              env_items=True, lr_w=0.5, reg=0.01, qkv=False):
    solver = CtxSolver(T)
    if strat is None:
        strat = Strategy(theta_values, seed, lr, lr_w, reg, qkv)
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
            meter.tick(X, ms, env.reset)
            ms["score"] = meter.cum.copy()
            if t % hold == 0:
                solver.prepare(min(len(X), solver.T))
                completion = solver.submit(X)
                while (ended := _stream.completion_observe(completion)) is None:
                    time.sleep(0)
                G = solver.result(ended)
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


def bench(reps, rows, T, device, configuration):
    solver = CtxSolver(T, device=device, configuration=configuration)
    observations = []
    for X in _ticks(rows, reps, seed=9):
        # Preparation is outside the numerical interval. Both the local demo
        # and service submit this same complete Function.
        solver.prepare(min(len(X), T))
        completion = solver.submit(X)
        while (ended := _stream.completion_observe(completion)) is None:
            time.sleep(0)
        solver.result(ended)
        observations.append(solver.previous_function)
    middle = sorted(observations, key=lambda row: row['seconds'])[len(observations) // 2]
    seconds = middle['seconds']
    print(json.dumps(dict(program='xonotic-context-solver', context=T, rows=rows,
        seconds=seconds, rows_per_second=rows / seconds,
        useful_gigabytes_per_second=middle['useful_gigabytes_per_second'],
        gigaflops_per_second=ctx_flops(T, rows) / seconds / 1e9, repetitions=reps)), flush=True)


REQUEST, STOP = 1, 2


def serve(config):
    world = World(config["device"])
    shares = config["columns"]
    assignment_data = {'decisions': {"columns": {"parts": list(shares["parts"]), "grain": shares["grain"],
                                                 "low": shares["low"]}}, 'observations': []}
    solver = CtxSolver(config["ctx"], config["device"], world, shares["parts"], assignment_data, config.get("compiler"))
    header = np.zeros(4, np.int64)
    stopping = {"signal": None}
    wake, woken = os.pipe()
    os.set_blocking(woken, False)
    signal.set_wakeup_fd(woken)

    def stop(signum, frame):
        stopping["signal"] = signum
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, stop)
    report("solver_started", rank=world.rank, size=world.size, device=solver.device,
           ctx=config["ctx"], columns=solver.columns,
           flops_per_solve=ctx_flops(config["ctx"], config["maxrows"]))
    result = None
    try:
        if world.rank == 0:
            result = relay(config, world, solver, header, stopping, wake)
        else:
            while True:
                message = world.configuration(None)
                kind, n = int(message[0]), int(message[1])
                if kind == STOP:
                    break
                if kind == REQUEST:
                    solver.prepare(min(n, solver.T))
                    completion = solver.submit(np.zeros((n, REQUEST_WIDTH), np.float32))
                    while (ended := _stream.completion_observe(completion)) is None:
                        time.sleep(0)
                    solver.result(ended)
    finally:
        report("solver_finished", rank=world.rank, solves=solver.solves, columns=solver.columns,
               whole_function=solver.previous_function, signal=stopping["signal"])
        world.free()
    return result


def relay(config, world, solver, header, stopping, wake):
    measurements = []
    def record(event, **fields):
        measurements.append({'event': event, **fields})
    trained = config["policy"] == "trained"
    pick_of = POLICIES.get(config["policy"], np.argmax)
    response = kind_of(config["response"])
    engine = Engine(config["socket"], config["framebytes"])
    joined = RuntimeFrames(config["framebytes"])
    posts = ItemPosts()
    objective = np.zeros(TEAMS + 1)
    route = {}
    # TBD: the game's plan consumer. rdma/xonwire.def has no PLAN kind and sv_payload_mesh.qc's consumer
    # (mesh_open(node, NONE, PLAN, 8, maxclients), mesh_scatter_rows into pick, weights and leader, the per-team vote;
    # ef6adb10^) is not restored, so the solver's rows reach no bot. Deferred by the operator 2026-10-17.
    if response is None:
        record("response_kind_undefined", response=config["response"], wire=str(engine.path),
               detail="rdma/xonwire.def defines no such kind; the solve runs and its rows are not sent")
    strat = meter = itimer = None
    if trained:
        strat = Strategy(config.get("theta_values"), config["train_seed"], config["lr"], config["lr_w"], config["reg"],
                         config["qkv"])
        meter = MatchMeter()
        itimer = ItemTracker()
        record("strategy", dominance=DOM_FORMULA, scalars=strat.table(), episodes=strat.eps,
               mode="greedy eval, no updates" if config["eval"] else "sampling + online REINFORCE",
               logit_field=NBUCKET * RES * NALLOC)
    served, blocks, solve_s = 0, 0, 0.0
    tick_in_ep = 0
    episode = None
    try:
        while stopping["signal"] is None:
            ready = joined.take()
            if ready is None:
                readable, _, _ = select.select([engine, wake], [], [])
                if wake in readable:
                    os.read(wake, 64)
                    continue
                for node, address, frame in engine.receive():
                    joined.feed(frame)
                    route[OBSERVATION_KIND] = (node, address)
                ready = joined.take()
                if ready is None:
                    continue
            (session, tick), records, events = ready
            done = records[OBSERVATION_KIND][0]
            posts.observe(list(events.values()))
            X = request_rows(records[OBSERVATION_KIND][1], records[CART_KIND][1], objective, posts)[:config["maxrows"]]
            n = len(X)
            if not n:
                continue
            header[0], header[1] = REQUEST, n
            world.configuration(header.tolist())
            solver.prepare(min(n, solver.T))
            completion = solver.submit(X)
            # This host application can receive more frames while numerical
            # work runs. Completion changes no compiled dispatch ordering.
            while (ended := _stream.completion_observe(completion)) is None:
                readable, _, _ = select.select([engine, wake], [], [], config['completion_poll_seconds'])
                if wake in readable:
                    os.read(wake, 64)
                if engine in readable:
                    for node, address, frame in engine.receive():
                        joined.feed(frame)
                        route[OBSERVATION_KIND] = (node, address)
            G = solver.result(ended)
            solve_s += solver.previous_function['seconds']
            if trained:
                ms = bind_match(X)
                itimer.tick(ms.get("posts", []))
                current = (session, int(records[TEAM_KIND][1][0, TS["EPISODE"]]))
                meter.tick(X, ms, np.full(K_CARTS, episode is not None and current != episode))
                episode = current
                ms["score"] = meter.cum.copy()
                boundary = tick_in_ep % config["hold"] == 0
                pick, Wt, lead, inst = strat.tick(X, solver.G2, solver.mixes(), solver.kappa, ms, boundary,
                                                  not config["eval"], itimer, solver.Zc, solver.white, solver.Vg)
                tick_in_ep += 1
                if tick_in_ep >= config["episode"]:
                    R, dirs, dprog, dbank, push = meter.finish()
                    ent = {j: round(float(np.mean(v)), 2) for j, v in strat.ent.items() if v}
                    if not config["eval"]:
                        strat.update(R)
                    else:
                        strat._reset_episode()
                    present = sorted(set(X[:, 1].astype(np.int32)) & set(range(1, TEAMS + 1)))
                    record("learn", episode=strat.eps, dprog=[round(float(v), 4) for v in dprog],
                           banked=[round(float(dbank[j]), 1) for j in range(1, TEAMS + 1)],
                           returns=[round(float(R[j]), 3) for j in range(1, TEAMS + 1)], lead=strat.team_lead,
                           policy_entropy=ent, w_max=float(np.abs(strat.W).max()), scalars=strat.table(),
                           alloc=strat.alloc_line(present), gram=gram_line(solver.kappa, ms))
                    tick_in_ep = 0
            else:
                pick = pick_of(G, axis=1).astype(np.float32)
                Wt = np.zeros((n, TEAMS), np.float32)
                Wt[np.arange(n), pick.astype(np.int32)] = 1.0
                lead = np.zeros(n, np.float32)
            objective = vote(X, pick, int(min(K_CARTS, len(records[CART_KIND][1]))), posts.ranks(), objective)
            out = np.zeros((n, RESP_WIDTH), np.float32)
            out[:, 0] = X[:, 0]
            out[:, 1] = pick
            out[:, 2:2 + TEAMS] = Wt
            out[:, 7] = lead
            if response is not None and OBSERVATION_KIND in route:
                node, address = route[OBSERVATION_KIND]
                engine.send(address, node, frames(response, done["req_id"], tick, out, config["framebytes"], session))
            served += n
            blocks += 1
            if not config.get("quiet"):
                record("plan", request=done["req_id"], tick=tick, rows=n, live=int((X[:, 1] > 0).sum()),
                       picks=np.bincount(pick.astype(np.int32) % TEAMS, minlength=TEAMS).tolist(),
                       objective=objective[1:].astype(int).tolist(), columns=solver.columns,
                       seconds=solver.previous_function['seconds'],
                       rows_per_second=solver.previous_function['units_per_second'],
                       useful_gigabytes_per_second=solver.previous_function['useful_gigabytes_per_second'])
    finally:
        header[...] = 0
        header[0] = STOP
        world.configuration(header.tolist())
        record("relay_finished", blocks=blocks, rows=served, seconds=solve_s,
               rows_per_second=served / solve_s if solve_s else 0,
               transport=engine.report(), frames=joined.report())
        engine.close()
    return {'event': 'solver_result', 'strategy': strat.state() if strat is not None else None, 'measurements': measurements}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", choices=("serve",))
    ap.add_argument("config", nargs="?")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--compiler", type=Path)
    ap.add_argument("--bench", type=int, default=0)
    ap.add_argument("--rows", type=int, default=64)
    ap.add_argument("--ctx", type=int, default=CTX)
    ap.add_argument("--synth", type=int, default=0)
    ap.add_argument("--episode", type=int)
    ap.add_argument("--hold", type=int)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--train-seed", type=int, default=SEED)
    ap.add_argument("--theta", default=THETA_PATH)
    ap.add_argument("--lr-w", type=float, default=0.5)
    ap.add_argument("--reg", type=float, default=0.01)
    ap.add_argument("--qkv", action="store_true")
    a = ap.parse_args()

    if a.mode == "serve":
        if not a.config:
            ap.error("serve takes its configuration document (xonotic/README.md \"The solver\")")
        config = json.loads(Path(a.config).read_text())
        if config.get('theta') and os.path.isfile(config['theta']):
            with np.load(config['theta'], allow_pickle=False) as source:
                config['theta_values'] = {name: source[name] for name in source.files}
        result = serve(config)
        print(json.dumps(result, default=lambda value: np.asarray(value).tolist()), flush=True)
        return
    if a.bench:
        bench(a.bench, a.rows, a.ctx, a.device, json.loads(a.compiler.read_text()) if a.compiler else None)
        return
    if a.synth:
        if a.episode is None or a.hold is None:
            ap.error("--synth takes its episode length and hold in ticks (--episode, --hold)")
        theta_values = None
        if a.theta != THETA_PATH:
            with np.load(a.theta, allow_pickle=False) as source:
                theta_values = {name: source[name] for name in source.files}
        rets, banked, strat, stats = run_synth(
            a.synth, a.ctx, a.train_seed, a.episode, a.hold, a.lr,
            theta_values,
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
    ap.error("a mode: serve CONFIG, --check, --bench N or --synth N")


if __name__ == "__main__":
    main()
