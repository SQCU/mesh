"""Strategy-estimator featurization projections and Q/K/V.

This module implements the *learned featurization projections* of the strategy
estimator: the per-bot query and the per-instrument key/value, exactly as fixed
by ``design/payload-spec.md`` §2.3::

    q_b = W_q * [ x_b ; beta_b ]      (per-bot query: known self-state AND observed world)
    k_m = W_k * z_m                    (per-instrument learned key)
    v_m = W_v * z_m                    (per-instrument learned behavioral value)

    Learned: W_q, W_k, W_v.  Given: x_b, z_m, beta_b.

It is the projection stage that sits between featurization (``payload-spec`` §2.1-2.2:
the per-bot engine vector ``x_b``, the egocentric belief ``beta_b``, and the
per-instrument descriptor ``z_m``) and the coupling / DPP / mixing-head stages
(``payload-spec`` §2.4-2.5, ``dpp-mixing-and-overlay.md``). It produces the query,
key and value tensors those downstream stages consume; it does NOT itself form the
coupling kernel, the marginal-inclusion vector ``diag(K)``, the velocity head, or
the intercentrality overlay.

Position in the training stack (``rl-training-spec.md`` §0-§2, §4):

  * These three matrices ``W_q, W_k, W_v`` are part of the single shared-weight
    policy ``W_all`` — the ONLY trainable surface here. There is one shared set of
    projection weights for the whole match; there is no per-team or per-player copy.
  * "Per-team" and "per-player" are *activations*, not weights (``rl-training-spec``
    §0): the per-player query ``q_b`` IS the per-player activation ``A_player`` at
    this layer, and a per-team pooling of the player queries (see
    :func:`team_pool`) is the per-team activation ``A_team``. The weights are shared;
    only the activations are indexed by player / team.
  * Everything fed IN is ``stopgrad`` w.r.t. the policy update: ``x_b`` and ``z_m``
    are engine/Game-2 features, and ``beta_b`` is the computed belief (Game-1 side,
    ``rl-training-spec`` §1, §4). The gradient of the REINFORCE loss (``rl-training``
    §2.1, §3) flows into ``W_q, W_k, W_v`` and nothing upstream of them. This module
    therefore keeps the projection params as the learnable leaves and treats its
    inputs as constants — callers pass ``stop_gradient``-ed feature arrays.

Learned vs computed vs frozen (``rl-training-spec`` §4): this whole module is on the
*learned* side (mlx params, differentiable). The deterministic Game-1 features it
consumes (``PW``, ``SUCC``, the V-cell featurization that builds ``beta_b``) live in
sibling numpy/plain-python modules, not here.

Backend: the learned/differentiable projections use **mlx** (Apple; matches the
mini the solver runs on), per the project rule that differentiable parts are mlx and
deterministic parts are numpy. The module stays importable without mlx installed —
:class:`QKVShapes` and the shape/labelling helpers are pure python — and only the
:class:`QKVProjector` constructor requires mlx.

--------------------------------------------------------------------------------
Shape glossary (used consistently below)
--------------------------------------------------------------------------------
  k   = number of teams.
  l   = number of players (bots) across all teams. Each player belongs to exactly
        one team; the mapping is a ``team_of`` vector of length ``l`` with entries in
        ``[0, k)``. (The estimator is one shared-weight policy over all ``l`` players;
        teams are an activation grouping, not separate policies.)
  j   = number of objectives / instruments. Per ``payload-spec`` §2.1 the instrument
        set is {push cart, suppress cart, contest post, hunt rival, explore cell};
        ``j`` is however many concrete instruments are live this strategy step.
  D_x    = width of the per-bot engine feature vector x_b   (``payload-spec`` §2.1).
  D_beta = width of the egocentric belief vector    beta_b  (``payload-spec`` §2.2).
  D_z    = width of the per-instrument descriptor   z_m     (``payload-spec`` §2.1).
  d      = query/key projection width (the attention "head" width). q_b and k_m must
           share this width so the query.key inner product the coupling stage forms
           is defined.
  d_v    = value projection width (the behavioural-value width; may differ from d).

Batched tensors this module produces per strategy step:
  Q : (l, d)      one query row per player       — per-player activation A_player.
  K : (j, d)      one key   row per instrument.
  V : (j, d_v)    one value row per instrument.
And, via :func:`team_pool`:
  Q_team : (k, d) one pooled query per team       — per-team activation A_team.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

try:  # differentiable backend (Apple mlx); deterministic siblings use numpy
    import mlx.core as mx
    import mlx.nn as nn

    _HAVE_MLX = True
    _ModuleBase = nn.Module
except ImportError:  # keep the module importable / shape-testable without mlx
    mx = None  # type: ignore
    nn = None  # type: ignore
    _HAVE_MLX = False

    class _ModuleBase:  # minimal stand-in so the class body imports without mlx
        pass


# The canonical instrument families of payload-spec §2.1, in order. `j` concrete
# instruments are drawn from these families each strategy step (e.g. one
# "push cart" instrument per live cart). Exposed for labelling / assertions only;
# the projections are agnostic to which family a column came from.
INSTRUMENT_FAMILIES: tuple[str, ...] = (
    "push_cart",     # push cart k
    "suppress_cart", # suppress cart k
    "contest_post",  # contest post p
    "hunt_rival",    # hunt rival r   (defined only for OBSERVED enemies; §2.2)
    "explore_cell",  # explore cell c
)


@dataclass(frozen=True)
class QKVShapes:
    """The dimensions of one strategy step's Q/K/V, per the shape glossary above.

    Pure python; importable and testable without mlx. Groups the counts (``k`` teams,
    ``j`` instruments, ``l`` players) with the feature/projection widths so callers and
    tests can reason about shapes without constructing mlx params.

    Cites ``payload-spec`` §2.1-2.3 (feature widths and the q/k/v projections) and
    ``rl-training-spec`` §0 (l players, k teams as activation groupings).
    """

    k_teams: int
    j_instruments: int
    l_players: int
    d_x: int       # width of x_b
    d_beta: int    # width of beta_b
    d_z: int       # width of z_m
    d: int         # query/key width
    d_v: int       # value width

    def __post_init__(self) -> None:
        for name in (
            "k_teams", "j_instruments", "l_players",
            "d_x", "d_beta", "d_z", "d", "d_v",
        ):
            v = getattr(self, name)
            if not isinstance(v, int) or v <= 0:
                raise ValueError(f"QKVShapes.{name} must be a positive int, got {v!r}")

    @property
    def d_query_in(self) -> int:
        """Input width of W_q: the concatenation [x_b ; beta_b] (payload-spec §2.3)."""
        return self.d_x + self.d_beta

    # --- expected tensor shapes (for assertions / documentation) ---
    @property
    def W_q_shape(self) -> tuple[int, int]:
        """(d, d_x + d_beta)."""
        return (self.d, self.d_query_in)

    @property
    def W_k_shape(self) -> tuple[int, int]:
        """(d, d_z)."""
        return (self.d, self.d_z)

    @property
    def W_v_shape(self) -> tuple[int, int]:
        """(d_v, d_z)."""
        return (self.d_v, self.d_z)

    @property
    def Q_shape(self) -> tuple[int, int]:
        """(l, d) — one query row per player (A_player)."""
        return (self.l_players, self.d)

    @property
    def K_shape(self) -> tuple[int, int]:
        """(j, d) — one key row per instrument."""
        return (self.j_instruments, self.d)

    @property
    def V_shape(self) -> tuple[int, int]:
        """(j, d_v) — one value row per instrument."""
        return (self.j_instruments, self.d_v)

    @property
    def Q_team_shape(self) -> tuple[int, int]:
        """(k, d) — one pooled query per team (A_team)."""
        return (self.k_teams, self.d)


def _require_mlx() -> None:
    if not _HAVE_MLX:
        raise RuntimeError(
            "qkv.py needs mlx for the learned projections. "
            "Install it (e.g. `uv pip install mlx`) on the Apple host that runs the "
            "solver. The pure-shape helpers (QKVShapes, team_pool label logic) do not "
            "need mlx."
        )


class QKVProjector(_ModuleBase):  # type: ignore[misc]
    """Learned q/k/v projections of the strategy estimator (payload-spec §2.3).

    Holds the three learnable matrices ``W_q, W_k, W_v`` (part of the shared policy
    ``W_all``, ``rl-training-spec`` §0) as mlx parameters, and maps featurization
    inputs to the per-player query and per-instrument key/value::

        q_b = W_q @ [x_b ; beta_b]      -> Q : (l, d)      (A_player activation)
        k_m = W_k @ z_m                 -> K : (j, d)
        v_m = W_v @ z_m                 -> V : (j, d_v)

    No bias terms: the spec writes the projections as bare linear maps
    ``W_q * [x_b ; beta_b]`` etc., and the downstream RMSNorm in the mixing head
    (``dpp-mixing-and-overlay.md`` §2) absorbs any offset. Kept bias-free to match the
    spec literally.

    Only these parameters learn. Inputs (``x_b``, ``beta_b``, ``z_m``) are stopgrad
    features (``rl-training-spec`` §2.1, §4); callers pass them as plain arrays and the
    REINFORCE gradient flows solely into ``W_q, W_k, W_v``.
    """

    def __init__(self, shapes: QKVShapes, *, seed: Optional[int] = None,
                 dtype=None) -> None:
        _require_mlx()
        super().__init__()
        self.shapes = shapes
        if dtype is None:
            dtype = mx.float32

        if seed is not None:
            mx.random.seed(seed)

        # Fan-in scaled init (1/sqrt(in)) so the pre-norm logits start O(1); the L2
        # regulariser (rl-training-spec §3, L_reg) pulls toward small weights during
        # training. Shapes exactly as QKVShapes documents.
        def _init(rows: int, cols: int):
            scale = 1.0 / (cols ** 0.5)
            return mx.random.uniform(low=-scale, high=scale, shape=(rows, cols)).astype(dtype)

        self.W_q = _init(*shapes.W_q_shape)   # (d, d_x + d_beta)
        self.W_k = _init(*shapes.W_k_shape)   # (d, d_z)
        self.W_v = _init(*shapes.W_v_shape)   # (d_v, d_z)

    # ------------------------------------------------------------------ queries
    def query(self, x, beta):
        """Per-player query q_b = W_q @ [x_b ; beta_b]  (payload-spec §2.3).

        Args:
            x:    (l, d_x)    per-bot engine feature rows x_b  (payload-spec §2.1).
            beta: (l, d_beta) egocentric belief rows beta_b    (payload-spec §2.2).
        Returns:
            Q: (l, d) — one query row per player; this IS the per-player activation
            A_player (rl-training-spec §0). Rows are ordered by player index; use the
            caller's ``team_of`` to recover team membership.

        The concatenation [x_b ; beta_b] is load-bearing (payload-spec §2.2-2.3): the
        query is a function of BOTH the bot's own known state and the occlusion-gated,
        spatialised world it has observed. beta_b is the sole spatial-mixing product in
        the system (§2.2), so this concat is where geometry enters the query.
        """
        x = mx.array(x)
        beta = mx.array(beta)
        if x.shape[0] != beta.shape[0]:
            raise ValueError(
                f"x and beta must share the player axis: {x.shape[0]} vs {beta.shape[0]}"
            )
        h = mx.concatenate([x, beta], axis=1)          # (l, d_x + d_beta)
        return h @ self.W_q.T                           # (l, d)

    # -------------------------------------------------------------- keys / values
    def key(self, z):
        """Per-instrument key k_m = W_k @ z_m  (payload-spec §2.3).

        Args:  z: (j, d_z) per-instrument descriptor rows (payload-spec §2.1).
        Returns: K: (j, d) — one key row per instrument, sharing width ``d`` with Q so
                 the coupling stage's query.key inner product is defined.
        """
        z = mx.array(z)
        return z @ self.W_k.T                            # (j, d)

    def value(self, z):
        """Per-instrument behavioural value v_m = W_v @ z_m  (payload-spec §2.3).

        Args:  z: (j, d_z) per-instrument descriptor rows.
        Returns: V: (j, d_v) — the behavioural payload of committing to each instrument.
        """
        z = mx.array(z)
        return z @ self.W_v.T                            # (j, d_v)

    # ------------------------------------------------------------------ combined
    def __call__(self, x, beta, z):
        """Compute (Q, K, V) for one strategy step.

        Args:
            x:    (l, d_x)     per-bot engine features.
            beta: (l, d_beta)  egocentric beliefs.
            z:    (j, d_z)     per-instrument descriptors.
        Returns:
            (Q, K, V) with shapes (l, d), (j, d), (j, d_v).

        This is the featurize->project step of the strategy step (payload-spec §3.2,
        "featurize x_b, z_m ; project q_b, k_m, v_m"). Downstream, the coupling stage
        forms the L-ensemble kernel and its marginal-inclusion vector diag(K) from
        these keys/quality (§2.4), and the RMSNorm->SwiGLU head turns that into the
        weight velocity dw/dt (§2.5) — none of which is done here.
        """
        return self.query(x, beta), self.key(z), self.value(z)

    # -------------------------------------------------------------- learned params
    def learned_params(self) -> dict:
        """The learnable leaves W_q, W_k, W_v (the qkv slice of shared W_all).

        These are the only tensors on the learned side of this module
        (rl-training-spec §4); the REINFORCE policy gradient (§2.1, §3) updates exactly
        these. Returned as a dict for the trainer/optimiser and for save/load.
        """
        return {"W_q": self.W_q, "W_k": self.W_k, "W_v": self.W_v}


def team_pool(Q, team_of: Sequence[int], k_teams: int):
    """Pool per-player queries into per-team activations A_team (rl-training-spec §0).

    "Per-team" is an ACTIVATION grouping of the shared-weight policy, never a separate
    weight set: A_team is a pooling of the per-player queries A_player over each team's
    players. This computes the mean player-query per team.

    Args:
        Q:       (l, d) per-player queries from :meth:`QKVProjector.query`.
        team_of: length-l sequence with entries in [0, k_teams); player -> team.
        k_teams: number of teams k.
    Returns:
        Q_team: (k_teams, d). A team with no live players yields a zero row.

    mlx path when available; falls back to a pure-python/numpy-free reduction so the
    grouping logic is testable without mlx (the grouping is not itself a learned op).
    """
    if len(team_of) != _rows(Q):
        raise ValueError(
            f"team_of length {len(team_of)} != number of player rows {_rows(Q)}"
        )
    for t in team_of:
        if not (0 <= int(t) < k_teams):
            raise ValueError(f"team index {t} out of range [0, {k_teams})")

    if _HAVE_MLX and mx is not None and isinstance(Q, mx.array):
        d = Q.shape[1]
        sums = mx.zeros((k_teams, d), dtype=Q.dtype)
        counts = mx.zeros((k_teams,), dtype=Q.dtype)
        idx = mx.array([int(t) for t in team_of])
        # scatter-add player rows into their team bucket
        sums = sums.at[idx].add(Q)
        counts = counts.at[idx].add(mx.ones((idx.shape[0],), dtype=Q.dtype))
        denom = mx.maximum(counts, mx.ones_like(counts))[:, None]
        return sums / denom

    # pure-python fallback (no mlx): mean-pool nested lists
    rows = [list(r) for r in Q]
    d = len(rows[0]) if rows else 0
    sums = [[0.0] * d for _ in range(k_teams)]
    counts = [0] * k_teams
    for r, t in zip(rows, team_of):
        t = int(t)
        counts[t] += 1
        for c in range(d):
            sums[t][c] += float(r[c])
    return [
        [s / counts[t] if counts[t] else 0.0 for s in sums[t]]
        for t in range(k_teams)
    ]


def _rows(Q) -> int:
    """Row count of an mlx array or a python sequence-of-sequences."""
    if _HAVE_MLX and mx is not None and isinstance(Q, mx.array):
        return Q.shape[0]
    return len(Q)


__all__ = [
    "QKVShapes",
    "QKVProjector",
    "team_pool",
    "INSTRUMENT_FAMILIES",
]
