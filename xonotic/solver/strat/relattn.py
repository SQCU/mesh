"""Learned relational all-to-all encoder — the O(n^2) cross-team coupling.

This module REPLACES the deterministic cross-team surrogate that `estimator.forward`
used to run (a `team_pool` mean over per-team player queries + a stop_gradient'd
per-team instrument DPP + hand-computed rival scalars in `_hierarchy_rows`). None of
those carried a *learned* pairwise term over agents; the cross-team channel was fixed,
hand-designed features (`design/dominance-demo.md` §Notes(b): "the cross-team coupling
here is deterministic, not a learned Gram").

What this is instead (`design/rl-training-spec.md` §4, §5; `design/dominance-demo.md`
§3 hypothesis): a stacked RELATIONAL-ATTENTION all-to-all encoder over the player rows,
differentiable end to end, with a genuine per-PAIR learned bias:

    per layer:  Q = R Wq,  K = R Wk,  V = R Wv        (d_row -> d_head, INDEPENDENT of n)
                A = softmax( Q Kᵀ / sqrt(d_head) + φ_rel(E) )
                R' = R + (A V) Wo                        (Wo: d_head -> d_row)

`E` are per-PAIR relation features (self, same-team, rival-team, relative hierarchy /
nimber, is-projected-winner, denial budget) — the count-invariant per-edge cross-team
term. `φ_rel` is a SHARED MLP (`d_rel -> hidden -> n_heads`); the same weights score
every edge, so adding a player adds a row/column to `E`, never a coordinate to a weight.
The whole thing is O(n^2·d), irreducibly all-to-all, and NOT stop_gradient'd — the
REINFORCE reward gradient flows through Wq/Wk/Wv/Wo and φ_rel.

The deterministic hierarchy/SUCC features are kept as OPTIONAL priors folded into `E`
(the raw per-edge feature vector), not as the sole cross-team channel: the learned
attention decides how to use them.

No parameter shape depends on k (teams), l (players), or M (instruments). Backend is
mlx (Apple; matches the mini), same as the rest of the learned surface.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx
import mlx.nn as nn

__all__ = [
    "EDGE_WIDTH",
    "edge_features",
    "RelAttnLayer",
    "RelationalEncoder",
]

# Width of the per-edge relation feature vector E[i, j, :]. Fixed (count-invariant):
# the encoder's φ_rel input width does not depend on how many players/teams there are.
EDGE_WIDTH = 12


def edge_features(team_of, hierarchy, winner_mask) -> np.ndarray:
    """Per-PAIR relation features E in (l, l, EDGE_WIDTH), deterministic (a prior).

    Built only from data already on the StrategyState — the player->team map, the
    per-player relative-hierarchy rows (`estimator._hierarchy_rows`, HIERARCHY_WIDTH=8:
    [own/scale, rival_max/scale, rival_mean/scale, (own-rival_mean)/scale, rank, is_winner,
    denial_budget, 1/#teams]) and the winner mask — so it works on every code path
    (cartsim, live server) without threading a new field. It is the OPTIONAL prior that
    φ_rel consumes; the attention is free to ignore or reshape it.

    Columns (per edge i attends to j):
      0 self (i==j)
      1 same_team
      2 rival (different team)
      3 winner_mask[j]          (is j on the projected-winner team)
      4 winner_mask[i]
      5 hier_own[j]             (j's team nimber / scale)
      6 hier_own[i]
      7 hier_own[j] - hier_own[i]   (relative standing — the de-aliasing signal)
      8 hier_rival_max[j]
      9 hier_denial[j]          (j's team denial budget / total depth)
     10 hier_rank[j]            (how many rivals j outranks, normalized)
     11 1.0                     (bias)
    """
    team_of = np.asarray(team_of).reshape(-1)
    hierarchy = np.asarray(hierarchy, dtype=np.float32)
    winner = np.asarray(winner_mask, dtype=np.float32).reshape(-1)
    l = team_of.shape[0]
    same = (team_of[:, None] == team_of[None, :]).astype(np.float32)   # (l, l)
    eye = np.eye(l, dtype=np.float32)
    rival = 1.0 - same
    own = hierarchy[:, 0]                # own nimber / scale
    rival_max = hierarchy[:, 1]
    rank = hierarchy[:, 4]
    denial = hierarchy[:, 6]
    E = np.zeros((l, l, EDGE_WIDTH), dtype=np.float32)
    E[:, :, 0] = eye
    E[:, :, 1] = same
    E[:, :, 2] = rival
    E[:, :, 3] = np.broadcast_to(winner[None, :], (l, l))
    E[:, :, 4] = np.broadcast_to(winner[:, None], (l, l))
    E[:, :, 5] = np.broadcast_to(own[None, :], (l, l))
    E[:, :, 6] = np.broadcast_to(own[:, None], (l, l))
    E[:, :, 7] = own[None, :] - own[:, None]
    E[:, :, 8] = np.broadcast_to(rival_max[None, :], (l, l))
    E[:, :, 9] = np.broadcast_to(denial[None, :], (l, l))
    E[:, :, 10] = np.broadcast_to(rank[None, :], (l, l))
    E[:, :, 11] = 1.0
    return E


class RelAttnLayer(nn.Module):
    """One relational-attention layer with a shared per-edge bias MLP φ_rel.

    Parameter shapes are all independent of the number of rows n: Wq/Wk/Wv are
    (d_row, d_head*n_heads), Wo is (d_head*n_heads, d_row), and φ_rel is
    (EDGE_WIDTH -> hidden -> n_heads). The per-edge bias φ_rel(E) is added inside the
    softmax, once per head, so it is a genuine learned pairwise coupling.
    """

    def __init__(self, d_row: int, d_edge: int, n_heads: int = 2, hidden_rel: int = 16):
        super().__init__()
        self.d_row = int(d_row)
        self.n_heads = int(n_heads)
        self.d_head = self.d_row  # per-head width == row width (proj is count-invariant)
        s = self.d_row ** -0.5
        # per-head projections stacked on the last axis: (d_row, n_heads*d_head)
        self.w_q = mx.random.normal((self.d_row, self.n_heads * self.d_head)) * s
        self.w_k = mx.random.normal((self.d_row, self.n_heads * self.d_head)) * s
        self.w_v = mx.random.normal((self.d_row, self.n_heads * self.d_head)) * s
        self.w_o = mx.random.normal((self.n_heads * self.d_head, self.d_row)) * (
            (self.n_heads * self.d_head) ** -0.5
        )
        # φ_rel: shared per-edge MLP -> one bias per head
        sr = d_edge ** -0.5
        self.rel_up = mx.random.normal((d_edge, hidden_rel)) * sr
        self.rel_down = mx.random.normal((hidden_rel, self.n_heads)) * (hidden_rel ** -0.5)

    def __call__(self, R: mx.array, E: mx.array) -> mx.array:
        n = R.shape[0]
        h, dh = self.n_heads, self.d_head
        q = (R @ self.w_q).reshape(n, h, dh)
        k = (R @ self.w_k).reshape(n, h, dh)
        v = (R @ self.w_v).reshape(n, h, dh)
        # per-head content scores: (h, n, n)
        q = mx.transpose(q, (1, 0, 2))
        k = mx.transpose(k, (1, 0, 2))
        v = mx.transpose(v, (1, 0, 2))
        scores = (q @ mx.transpose(k, (0, 2, 1))) / (dh ** 0.5)   # (h, n, n)
        # φ_rel(E): (n, n, h) -> (h, n, n)
        rel = nn.silu(E @ self.rel_up) @ self.rel_down             # (n, n, h)
        rel = mx.transpose(rel, (2, 0, 1))                         # (h, n, n)
        attn = mx.softmax(scores + rel, axis=-1)                   # (h, n, n)
        ctx = attn @ v                                             # (h, n, dh)
        ctx = mx.transpose(ctx, (1, 0, 2)).reshape(n, h * dh)      # (n, h*dh)
        return R + ctx @ self.w_o                                  # (n, d_row) residual


class RelationalEncoder(nn.Module):
    """Stacked relational all-to-all encoder: R0 (l, d) -> R' (l, d).

    Two to three `RelAttnLayer`s. Consumes the per-player row set R0 (the audited-real
    `qkv.query([x;β])`) and the per-edge relation prior E, and returns the count-invariant
    contextualized rows R'. Every downstream reader (policy appetite, W/L value pool) uses
    R' instead of the old `team_pool` mean.
    """

    def __init__(self, d_row: int, d_edge: int = EDGE_WIDTH, n_layers: int = 2,
                 n_heads: int = 2, hidden_rel: int = 16):
        super().__init__()
        self.n_layers = int(n_layers)
        self.layers = [
            RelAttnLayer(d_row, d_edge, n_heads=n_heads, hidden_rel=hidden_rel)
            for _ in range(self.n_layers)
        ]

    def __call__(self, R: mx.array, E: mx.array) -> mx.array:
        for layer in self.layers:
            R = layer(R, E)
        return R
