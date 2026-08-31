from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .game import Cart, projected_winner, succession, team_nimbers


@dataclass(frozen=True)
class GameContext:
    teams: tuple[int, ...]
    team_of: tuple[int, ...]
    levels: int = 8


@dataclass
class CartSnapshot:
    pos: np.ndarray
    control: np.ndarray
    banked: np.ndarray
    levels: int = 8


def carts(snapshot: CartSnapshot) -> list[Cart]:
    return [
        Cart(None if int(control) < 0 else int(control), int(np.floor(position)))
        for position, control in zip(snapshot.pos, snapshot.control)
    ]


def winner(context: GameContext, snapshot: CartSnapshot):
    return projected_winner(carts(snapshot), context.teams)


def hierarchy(context: GameContext, snapshot: CartSnapshot) -> np.ndarray:
    values = team_nimbers(carts(snapshot), context.teams)
    scale = float(max(1, snapshot.levels, max(values.values(), default=0)))
    out = np.zeros(len(context.teams), dtype=np.float32)
    for team in context.teams:
        rivals = [values[other] for other in context.teams if other != team]
        center = float(np.mean(rivals)) if rivals else 0.0
        out[team] = np.tanh((values.get(team, 0) - center) / scale)
    return out


def loser_ranks(context: GameContext, snapshot: CartSnapshot) -> np.ndarray:
    cart_rows = carts(snapshot)
    values = team_nimbers(cart_rows, context.teams)
    current = projected_winner(cart_rows, context.teams)
    teams = np.asarray(context.teams, dtype=np.int64)
    scores = np.asarray([values[team] for team in context.teams], dtype=np.int64)
    current_mask = np.asarray([team == current for team in context.teams], dtype=bool)
    loser_mask = ~current_mask
    ordered_ranks = np.sum(
        (scores[:, None] > scores[None, :]) & loser_mask[None, :], axis=1
    ).astype(np.int64)
    ordered_ranks = np.where(current_mask, int(np.sum(loser_mask)), ordered_ranks)
    ranks = np.zeros(len(context.teams), dtype=np.int64)
    ranks[teams] = ordered_ranks
    return ranks


def hierarchy_rows(context: GameContext, snapshot: CartSnapshot):
    cart_rows = carts(snapshot)
    nimbers = team_nimbers(cart_rows, context.teams)
    order = succession(cart_rows, context.teams)
    denial = {team: amount for team, amount in order}
    current = projected_winner(cart_rows, context.teams)
    scale = float(max(1, snapshot.levels, max(nimbers.values(), default=0)))
    total = float(max(1, sum(cart.depth for cart in cart_rows)))
    rows = np.zeros((len(context.team_of), 8), dtype=np.float32)
    mask = np.zeros(len(context.team_of), dtype=bool)
    for player, team in enumerate(context.team_of):
        own = float(nimbers.get(team, 0))
        rivals = [float(nimbers.get(other, 0)) for other in context.teams if other != team]
        center = float(np.mean(rivals)) if rivals else 0.0
        rows[player] = (
            own / scale,
            max(rivals, default=0.0) / scale,
            center / scale,
            (own - center) / scale,
            (sum(own > rival for rival in rivals) - sum(own < rival for rival in rivals)) / max(1, len(rivals)),
            float(team == current),
            float(denial.get(team, 0)) / total,
            1.0 / max(1, len(context.teams)),
        )
        mask[player] = team == current
    return rows, mask


def role_rewards(context: GameContext, before: CartSnapshot, after: CartSnapshot) -> np.ndarray:
    before_winner = winner(context, before)
    after_winner = winner(context, after)
    before_ranks = loser_ranks(context, before)
    after_ranks = loser_ranks(context, after)
    teams = np.asarray(context.teams, dtype=np.int64)
    winner_mask = teams == before_winner
    rank_flip = after_ranks[teams] > before_ranks[teams]
    team_rewards = np.where(
        winner_mask,
        -float(after_winner != before_winner),
        rank_flip.astype(np.float32),
    ).astype(np.float32)
    return team_rewards[np.asarray(context.team_of, dtype=np.int64)]
