"""The strategy operator's coupling: a GRAM matrix + a SwiGLU, at a wide IR.

SPEC §8:

    where idd a softmax come from? why are you talking about attention? im
    pretty sure a gram matrix and a swiglu were described earlier.

    how wide did you think the hidden states were supposed to be for this?
    under 128d? maybe you were slippin.

So there is no softmax, no attention, no query/key/value heads and no row-wise
normalization of the coupling.  The all-to-all term is the literal second
moment of the player-row set under a learned metric, plus an additive bilinear
term in the per-pair relation features:

    Z    = rmsnorm(R) * gamma                       (n, d)
    A    = w_metric                                 (d, d)     learned
    G    = (Z A)(Z A)^T / d  +  E . w_rel           (n, n)     = Z M Z^T + psi(E)
    Mix  = (G Z) / n                                (n, d)
    IR   = R + SwiGLU([Z ; Mix])                    (n, d)

``M = A A^T`` is PSD, so the content half of ``G`` is a genuine Gram (a positive
semidefinite second moment) and not a similarity score.  ``psi(E) = E . w_rel``
is bilinear in the pair: linear in the per-edge feature vector, one scalar per
ordered pair, shared weights over every edge.

Properties this file is required to have, and does:

* **The Gram lands in the IR.**  SPEC §7: "if we're spending flops on a gram
  matrix that gram matrix better fucking end up in the IR consumed by
  subsequent probes."  ``Mix = G Z / n`` is half of the SwiGLU input, so the
  IR the value probes and the appetite read is a function of ``G``.  ``G`` is
  also returned so the live path can log it.
* **Differentiable end to end.**  Nothing is stop_gradient'd here; the reward
  gradient reaches ``w_metric``, ``w_rel``, ``norm`` and the SwiGLU weights.
* **Count-invariant.**  Every parameter shape is a function of ``width`` and
  ``EDGE_WIDTH`` only.  Adding a player adds a row and a column to ``Z``/``G``
  and changes no weight's shape.
* **Irreducibly O(n^2).**  ``Z A (Z A)^T`` and ``G Z`` are both dense over the
  full interacting row set; there is no factorization that skips a pair.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx
import mlx.nn as nn

__all__ = ["EDGE_WIDTH", "edge_features", "GramSwiGLU"]

# Width of the per-pair relation feature vector E[i, j, :].  Fixed: the additive
# bilinear term's weight is (EDGE_WIDTH, 1) regardless of how many players or
# teams there are.
EDGE_WIDTH = 12


def edge_features(team_of, hierarchy, winner_mask) -> np.ndarray:
    """Per-PAIR relation features ``E`` in ``(n, n, EDGE_WIDTH)``, deterministic.

    Built only from data already on the ``StrategyState``: the player->team map,
    the per-player relative-hierarchy rows (``runtime.hierarchy_rows``:
    ``[own/scale, rival_max/scale, rival_mean/scale, (own-rival_mean)/scale,
    rank, is_winner, denial_budget, 1/#teams]``) and the projected-winner mask.

    Columns, for the ordered pair (i, j):
      0 self (i == j)                  6 own[i]
      1 same team                      7 own[j] - own[i]  (relative standing)
      2 rival (different team)         8 rival_max[j]
      3 winner_mask[j]                 9 denial budget[j]
      4 winner_mask[i]                10 rank[j]
      5 own[j]                        11 bias
    """
    team_of = np.asarray(team_of).reshape(-1)
    hierarchy = np.asarray(hierarchy, dtype=np.float32)
    winner = np.asarray(winner_mask, dtype=np.float32).reshape(-1)
    n = team_of.shape[0]
    same = (team_of[:, None] == team_of[None, :]).astype(np.float32)
    own, rival_max, rank, denial = (hierarchy[:, 0], hierarchy[:, 1],
                                    hierarchy[:, 4], hierarchy[:, 6])
    E = np.zeros((n, n, EDGE_WIDTH), dtype=np.float32)
    E[:, :, 0] = np.eye(n, dtype=np.float32)
    E[:, :, 1] = same
    E[:, :, 2] = 1.0 - same
    E[:, :, 3] = np.broadcast_to(winner[None, :], (n, n))
    E[:, :, 4] = np.broadcast_to(winner[:, None], (n, n))
    E[:, :, 5] = np.broadcast_to(own[None, :], (n, n))
    E[:, :, 6] = np.broadcast_to(own[:, None], (n, n))
    E[:, :, 7] = own[None, :] - own[:, None]
    E[:, :, 8] = np.broadcast_to(rival_max[None, :], (n, n))
    E[:, :, 9] = np.broadcast_to(denial[None, :], (n, n))
    E[:, :, 10] = np.broadcast_to(rank[None, :], (n, n))
    E[:, :, 11] = 1.0
    return E


class GramSwiGLU(nn.Module):
    """``rows -> (IR, Gram)``.  See the module docstring for the algebra."""

    def __init__(self, width: int, hidden: int | None = None, d_edge: int = EDGE_WIDTH):
        super().__init__()
        if int(width) < 128:
            raise ValueError(
                f"IR width {width} < 128; SPEC §8 requires >=128d hidden states"
            )
        self.width = int(width)
        self.d_edge = int(d_edge)
        hidden = int(hidden or 4 * width)
        self.hidden = hidden
        scale = (2 * width) ** -0.5
        self.norm = mx.ones((width,))
        # A: the learned metric factor.  M = A Aᵀ is PSD by construction.
        # A starts at the identity so G starts as the plain second moment; the
        # 1/width perturbation and the 1/width normalization below keep the
        # Gram's diagonal at O(1) for unit-RMS rows AT ANY WIDTH -- without
        # that the operator overflows downstream once the IR is widened to 128.
        self.w_metric = mx.eye(self.width) + mx.random.normal((self.width, self.width)) / self.width
        # psi: the additive bilinear per-pair term, shared over every edge.
        self.w_rel = mx.random.normal((self.d_edge, 1)) * self.d_edge ** -0.5
        self.w_gate = mx.random.normal((2 * width, hidden)) * scale
        self.w_up = mx.random.normal((2 * width, hidden)) * scale
        self.w_down = mx.random.normal((hidden, width)) * hidden ** -0.5

    def gram(self, normalized: mx.array, edges: mx.array | None = None) -> mx.array:
        projected = normalized @ self.w_metric              # Z A            (n, d)
        gram = projected @ projected.T / self.width         # Z A Aᵀ Zᵀ      (n, n)
        if edges is not None:
            gram = gram + (edges @ self.w_rel)[..., 0]      # + psi(E)       (n, n)
        return gram

    def __call__(self, rows: mx.array, edges: mx.array | None = None):
        normalized = rows * mx.rsqrt(mx.mean(mx.square(rows), axis=-1, keepdims=True) + 1e-6)
        normalized = normalized * self.norm
        gram = self.gram(normalized, edges)
        mixed = gram @ normalized / mx.maximum(mx.array(rows.shape[0]), mx.array(1))
        joined = mx.concatenate((normalized, mixed), axis=-1)
        ir = rows + (nn.silu(joined @ self.w_gate) * (joined @ self.w_up)) @ self.w_down
        return ir, gram
