"""The strategy estimator: one forward pass composing qkv -> DPP diag(K) -> head -> value.

This is the assembled learned strategy operator of ``design/payload-spec.md`` §2 and
``design/rl-training-spec.md`` §0-§4 — the single shared-weight policy ``W_all`` plus the
value stack. It threads the committed pipeline end to end for one strategy step:

    q_b,k_m,v_m = QKV( x_b, beta_b, z_m )              (qkv.py; payload-spec §2.3)
    scores      = Q @ K^T                              player x instrument affinity (§2.4)
    b           = softplus(scores)                     per-player appetite (DPP quality)
    diag(K)     = DPP marginal inclusion of (q_i, K)   (dpp.py; §2.4, repulsion-only)
    dw/dt       = SwiGLU(RMSNorm([diag(K); b]))        (head.py; §2.5 velocity)
    w'          = w + dw/dt * Delta                    forward-Euler replicator (§3.2)
    a, logpi    = categorical-sample(w')               (head.py; §5 sampling NOT argmax)
    V_phi       = ValueHead([Q; b; dw/dt; SUCC])       (value.py; §2.1 per-player R^l)

Everything fed IN is a STOPGRAD feature (``rl-training-spec`` §2.1 / §4): the featurization
``x_b`` / ``beta_b`` (``featurize.py`` belief, numpy) and the closed-form Game-1
``PW`` / ``SUCC`` (``game.py``, numpy) enter detached — the REINFORCE gradient flows only
into ``W_all`` (the qkv projections + the mixing head) and the value/aux heads. The
``SUCC`` anticipatory feature is folded into the value head's final intermediate exactly
as ``value.py`` documents (``rl-training-spec`` §2.2: value "inputs superset ``SUCC`` =>
anticipatory").

Learned (mlx) vs computed (numpy). The learned surface — ``QKVProjector``, ``MixingHead``,
``StrategyValue`` — is mlx (Apple; the mini). The feature glue (``StrategyState`` and
``state_from_cartsim``) is pure numpy so it is importable and exercisable without mlx on
any host; only :meth:`StrategyEstimator.forward` needs mlx. The module therefore imports
mlx **lazily**: constructing / running the estimator requires mlx, but importing the
module and building states does not.

``forward(state) -> (action, logpi, value, dw_dt)`` is the required public entry. Extra
diagnostics from the same pass (the integrated logits ``w'``, ``diag(K)``, the auxiliary
value ``Vtilde``, the raw appetite ``b``) are stashed on ``estimator.last`` for the
training loop / logging without widening the 4-tuple.

Spec: ``payload-spec.md`` §2.3-§2.5, §3.2, §5 ; ``dpp-mixing-and-overlay.md`` §2, §4 ;
      ``rl-training-spec.md`` §0-§4 ; companion sibling modules ``qkv`` / ``dpp`` /
      ``head`` / ``value`` / ``game`` / ``featurize`` / ``cartsim``.

Public surface
--------------
- ``StrategyState``          : dataclass — the per-step stopgrad features (numpy).
- ``ForwardResult``          : namedtuple bundling every output of one pass.
- ``StrategyEstimator``      : the composed operator; ``forward`` + ``learned_params``.
- ``StrategyEstimator.for_cartsim`` : build one sized to a :class:`cartsim.CartSim`.
- ``state_from_cartsim``     : featurize a ``CartSim`` cartstate -> ``StrategyState`` (numpy).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Optional, Sequence

import numpy as np

from .game import succ_feature
from .qkv import QKVProjector, QKVShapes, team_pool

# mlx is imported lazily inside the methods that need it (constructing the learned
# modules / running forward), so this module stays importable on an mlx-less host for
# the numpy feature glue (StrategyState, state_from_cartsim).
_mx = None


def _mlx():
    """Import mlx on demand; raise a clear error on a host without it (e.g. node 0)."""
    global _mx
    if _mx is None:
        try:
            import mlx.core as mx  # type: ignore
        except ImportError as e:  # pragma: no cover - host-dependent
            raise RuntimeError(
                "estimator.forward / StrategyEstimator need mlx (the learned surface). "
                "Run it on the Apple host that has mlx (the Mac mini, ~/.venv-mesh). The "
                "numpy feature glue (StrategyState, state_from_cartsim) works without mlx."
            ) from e
        _mx = mx
    return _mx


__all__ = [
    "StrategyState",
    "ForwardResult",
    "StrategyEstimator",
    "state_from_cartsim",
]


@dataclass
class StrategyState:
    """One strategy step's STOPGRAD features (numpy) — the input to :meth:`forward`.

    All fields are computed / detached w.r.t. the policy gradient (``rl-training-spec``
    §2.1 / §4): ``x`` / ``beta`` come from the deterministic featurization (``featurize``),
    ``z`` are the instrument descriptors, ``carts`` / ``teams`` feed the closed-form Game-1
    ``PW`` / ``SUCC`` (``game``), and ``w`` is the current integrated strategy weight state
    (the logits ``dw/dt`` is added to). Shapes follow ``qkv.QKVShapes``.

    Fields
    ------
    x        : (l, d_x)     per-player engine feature rows ``x_b`` (payload-spec §2.1).
    beta     : (l, d_beta)  per-player egocentric belief rows ``beta_b`` (§2.2).
    z        : (M, d_z)     per-instrument descriptor rows ``z_m`` (§2.1).
    w        : (l, M)       current per-player integrated strategy weights = logits.
    carts    : list         cartstate carts for ``PW`` / ``SUCC`` (game.Cart or pairs).
    teams    : sequence     explicit team roster (aligns succ_feature across calls).
    team_of  : (l,) int     player -> team map (for the A_team pooling activation).
    """

    x: np.ndarray
    beta: np.ndarray
    z: np.ndarray
    w: np.ndarray
    carts: list
    teams: Sequence
    team_of: Sequence[int]


class ForwardResult(NamedTuple):
    """Everything one :meth:`StrategyEstimator.forward` pass produces.

    The first four fields are the required ``forward`` return; the rest are diagnostics
    from the same pass (also stashed on ``estimator.last``).
    """

    action: object      # (l,) int   sampled instrument per player (payload-spec §5)
    logpi: object       # (l,)       log pi(a|.) of the sampled action (REINFORCE, §2.1)
    value: object       # (l, l)     per-player value VECTOR V_phi in R^l (rl-training §2.1)
    dw_dt: object       # (l, M)     per-player strategy velocity (§2.5)
    w_next: object      # (l, M)     integrated logits w + dw/dt * Delta (§3.2)
    diag_k: object      # (M,)       DPP marginal inclusion vector (§2.4)
    appetite: object    # (l, M)     per-player appetite b = softplus(Q@K^T)
    vtilde: object      # (l, l)     auxiliary value from the query (rl-training §2.1)
    q_team: object      # (k, d)     per-team pooled query activation A_team (§0)


class StrategyEstimator:
    """The composed strategy operator: qkv -> DPP diag(K) -> RMSNorm/SwiGLU head -> value.

    Bundles the four learned sub-modules into one shared-weight policy ``W_all`` plus the
    value stack (``rl-training-spec`` §0 / §4). One :meth:`forward` is one strategy step:
    it projects the query/keys/values, forms the player x instrument appetite, folds it
    through the honest repulsion-only DPP marginal ``diag(K)``, gates that into a per-player
    velocity ``dw/dt``, integrates the strategy weights one forward-Euler step, samples the
    instrument (NOT argmax) with its log-prob, and reads the per-player value vector off the
    final intermediate (with ``SUCC`` folded in for anticipation).

    Parameters
    ----------
    shapes : QKVShapes
        The step dimensions (``k`` teams, ``M`` instruments, ``l`` players, feature/proj
        widths). ``j_instruments`` is the instrument count ``M``.
    delta : float
        Forward-Euler step ``Delta`` for ``w += dw/dt * Delta`` (dpp-mixing §4; a STABILITY
        parameter, not a scheduling knob).
    temperature : float
        Sampling temperature for the categorical selection (head.sample_strategy; §5).
    hidden : int, optional
        Mixing-head SwiGLU hidden width (defaults to ``4*M``).
    seed : int, optional
        Seeds the qkv/head/value parameter init and the sampling RNG (reproducible).
    """

    def __init__(
        self,
        shapes: QKVShapes,
        *,
        delta: float = 0.5,
        temperature: float = 1.0,
        hidden: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        mx = _mlx()
        from .head import MixingHead        # local import: mlx-dependent
        from .value import StrategyValue

        self.shapes = shapes
        self.delta = float(delta)
        self.temperature = float(temperature)
        self.k = shapes.k_teams
        self.M = shapes.j_instruments       # instrument count
        self.l = shapes.l_players
        self._seed = seed
        self._rng_counter = 0

        if seed is not None:
            mx.random.seed(seed)

        # --- learned surface ------------------------------------------------
        self.qkv = QKVProjector(shapes, seed=seed)
        self.head = MixingHead(self.M, hidden=hidden)

        # SUCC feature width = 2*k + 1 (game.succ_feature layout), folded into the value
        # head's final intermediate to make V anticipatory (rl-training §2.2).
        self.succ_dim = 2 * self.k + 1
        # final intermediate per player = [ Q (d) ; appetite b (M) ; dw/dt (M) ; SUCC (2k+1) ]
        self.d_intermediate = shapes.d + self.M + self.M + self.succ_dim
        self.value = StrategyValue(
            d_intermediate=self.d_intermediate, d_query=shapes.d, l=self.l
        )

        self.last: Optional[ForwardResult] = None

    # ------------------------------------------------------------- constructor
    @classmethod
    def for_cartsim(
        cls,
        sim,
        *,
        d: int = 16,
        d_v: int = 16,
        delta: float = 0.5,
        temperature: float = 1.0,
        hidden: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> "StrategyEstimator":
        """Build an estimator sized to a :class:`cartsim.CartSim` (matching feature widths).

        Uses the same feature layout as :func:`state_from_cartsim`: ``d_x = k + 3``,
        ``d_beta = 2*j``, ``d_z = 4 + k``, ``M = 2*j + 1``. ``d`` / ``d_v`` are the
        query/key and value projection widths (free choices; dpp-mixing marks them
        non-fixed).
        """
        k, j = sim.k, sim.j
        shapes = QKVShapes(
            k_teams=k,
            j_instruments=sim.M,   # instrument count 2j+1
            l_players=sim.l,
            d_x=k + 3,
            d_beta=2 * j,
            d_z=4 + k,
            d=d,
            d_v=d_v,
        )
        return cls(shapes, delta=delta, temperature=temperature, hidden=hidden, seed=seed)

    # ------------------------------------------------------------------ forward
    def forward(self, state: StrategyState) -> ForwardResult:
        """One strategy step: features -> (action, logpi, value, dw_dt) (+ diagnostics).

        Returns a :class:`ForwardResult`; the first four fields ``(action, logpi, value,
        dw_dt)`` are the required ``forward`` outputs (unpack them directly, or read the
        named fields). All inputs are wrapped in ``stop_gradient`` so the policy gradient
        touches only the learned weights (``rl-training-spec`` §2.1 / §4).
        """
        mx = _mlx()
        from .dpp import dpp_marginals
        from .head import integrate_weights, sample_strategy
        from . import head as head_mod

        # --- stopgrad features (rl-training §2.1/§4) ---
        x = mx.stop_gradient(mx.array(np.asarray(state.x, dtype=np.float32)))
        beta = mx.stop_gradient(mx.array(np.asarray(state.beta, dtype=np.float32)))
        z = mx.stop_gradient(mx.array(np.asarray(state.z, dtype=np.float32)))
        w = mx.stop_gradient(mx.array(np.asarray(state.w, dtype=np.float32)))

        # --- qkv projections (payload-spec §2.3) ---
        Q = self.qkv.query(x, beta)          # (l, d)  = per-player activation A_player
        K = self.qkv.key(z)                  # (M, d)
        V = self.qkv.value(z)                # (M, d_v)  (behavioural value; carried for parity)
        Q_team = team_pool(Q, list(state.team_of), self.k)   # (k, d) = A_team activation

        # --- player x instrument appetite (the coupling; §2.4) ---
        scores = Q @ K.T                     # (l, M)
        appetite = mx.logaddexp(scores, mx.zeros_like(scores))   # softplus -> b >= 0

        # --- DPP marginal inclusion diag(K), repulsion-only (dpp.py; §2.4) ---
        # per-instrument quality q_i = mean appetite over players; diversity feats = keys K.
        quality = mx.mean(appetite, axis=0)  # (M,)
        diag_k = dpp_marginals(quality, K)   # (M,)  in [0,1]
        diag_k = mx.stop_gradient(diag_k)    # enters the head as a detached feature (§2.1)

        # --- mixing head: velocity dw/dt (head.py; §2.5) ---
        dw_dt = self.head(diag_k, appetite)  # (l, M)

        # --- forward-Euler replicator step -> logits (§3.2) ---
        w_next = integrate_weights(w, dw_dt, self.delta)   # (l, M)

        # --- weighted sampling (NOT argmax; §5) + log-prob (REINFORCE; §2.1) ---
        key = None
        if self._seed is not None:
            key = mx.random.key(self._seed + self._rng_counter)
            self._rng_counter += 1
        action, logpi = sample_strategy(w_next, temperature=self.temperature, key=key)

        # --- value: final intermediate with SUCC folded in (value.py; §2.1/§2.2) ---
        succ_np = succ_feature(state.carts, teams=state.teams).astype(np.float32)  # (2k+1,)
        succ = mx.stop_gradient(mx.array(succ_np))
        succ_rows = mx.broadcast_to(succ[None, :], (self.l, self.succ_dim))
        final_intermediate = mx.concatenate([Q, appetite, dw_dt, succ_rows], axis=1)  # (l, d_int)
        V_phi, Vtilde = self.value(final_intermediate, Q)   # (l, l), (l, l)

        result = ForwardResult(
            action=action, logpi=logpi, value=V_phi, dw_dt=dw_dt,
            w_next=w_next, diag_k=diag_k, appetite=appetite, vtilde=Vtilde,
            q_team=Q_team,
        )
        self.last = result
        return result

    # --------------------------------------------------------------- params
    def learned_params(self) -> dict:
        """The trainable leaves: qkv (``W_all`` projections) + head + value/aux heads.

        The REINFORCE policy gradient (``rl-training-spec`` §3 ``L_pg``) updates the qkv +
        head weights; the value regression / aux imitation (``L_v`` / ``L_aux``) update the
        value heads. Returned as a nested dict for the optimizer and for save/load; the
        stopgrad features are not here (they never learn).
        """
        return {
            "qkv": self.qkv.learned_params(),
            "head": self.head.parameters(),
            "value": self.value.parameters(),
        }


# --------------------------------------------------------------------------- #
# Feature glue (numpy): CartSim cartstate -> StrategyState. Pure / mlx-free so it
# is exercisable on any host; this is the featurize/game read the estimator sits on.
# --------------------------------------------------------------------------- #

def state_from_cartsim(sim, cstate, *, w: Optional[np.ndarray] = None) -> StrategyState:
    """Featurize a :class:`cartsim.CartSim` cartstate into a :class:`StrategyState` (numpy).

    Builds the stopgrad feature rows the estimator consumes, in the layout
    :meth:`StrategyEstimator.for_cartsim` sizes to:

      * ``x`` (l, k+3): per player [ team one-hot(k), own-cart depth/L, own banked/L,
        is-projected-winner ] — the bot's OWN known state (payload-spec §2.1).
      * ``beta`` (l, 2j): observed per-cart [ depth/L, color/k ] — a right-sized egocentric
        belief stand-in (the full V-cell pipeline is ``featurize.belief``; §2.2).
      * ``z`` (M, 4+k): per instrument [ kind one-hot(3), target depth/L, target color
        one-hot(k) ] (idle -> zeros) — the per-instrument descriptor ``z_m`` (§2.1).
      * ``carts`` / ``teams``: the cartstate carts + roster for the closed-form ``PW``/``SUCC``.
      * ``w`` (l, M): the current integrated strategy weights (defaults to zeros = the
        untrained broad-sampling start, payload-spec §5).

    All numpy / detached; nothing here learns (``rl-training-spec`` §4).
    """
    from .cartsim import to_carts, decode_instrument

    k, j, l, M, L = sim.k, sim.j, sim.l, sim.M, sim.L
    Lf = float(max(L, 1))
    pos = cstate.pos
    control = cstate.control
    banked = cstate.banked
    carts = to_carts(cstate)
    pw = sim.projected_winner(cstate)

    # x_b (l, k+3)
    x = np.zeros((l, k + 3), dtype=np.float32)
    for p in range(l):
        team = int(sim.team_of[p])
        x[p, team] = 1.0
        own = [c for c in range(j) if int(control[c]) == team]
        own_depth = max((float(pos[c]) for c in own), default=0.0)
        x[p, k + 0] = own_depth / Lf
        x[p, k + 1] = float(banked[team]) / Lf
        x[p, k + 2] = 1.0 if (pw is not None and team == pw) else 0.0

    # beta_b (l, 2j): observed cart depths + colors (same for all players here — the
    # right-sized belief stand-in; per-player occlusion gating is featurize.belief).
    beta_row = np.zeros(2 * j, dtype=np.float32)
    for c in range(j):
        beta_row[2 * c] = float(pos[c]) / Lf
        beta_row[2 * c + 1] = float(control[c]) / float(max(k, 1))
    beta = np.tile(beta_row, (l, 1)).astype(np.float32)

    # z_m (M, 4+k)
    z = np.zeros((M, 4 + k), dtype=np.float32)
    kind_idx = {"push_cart": 0, "suppress_cart": 1, "idle": 2}
    for m in range(M):
        kind, cart = decode_instrument(m, j)
        z[m, kind_idx[kind]] = 1.0
        if cart >= 0:
            z[m, 3] = float(pos[cart]) / Lf
            col = int(control[cart])
            if col >= 0:
                z[m, 4 + col] = 1.0

    if w is None:
        w = np.zeros((l, M), dtype=np.float32)
    else:
        w = np.asarray(w, dtype=np.float32).reshape(l, M)

    return StrategyState(
        x=x, beta=beta, z=z, w=w, carts=carts,
        teams=list(range(k)), team_of=list(np.asarray(sim.team_of).tolist()),
    )
