import hashlib
import json
from dataclasses import dataclass
import math
import numpy as np

GAME_CONTRACT = "checkpoint-control-integral-v1"

def checkpoint_ownership(distances, control, position):
    return np.where(np.asarray(distances, dtype=np.float32) <= np.float32(position), control, -1)

@dataclass(frozen=True)
class GameValue:
    projected_role: int | None
    checkpoints_held: tuple[int, ...]
    scores: tuple[float, ...]
    score_limit: float
    checkpoint_rate: float
    time_to_win: tuple[float, ...]
    role_ranks: dict[int, int]
    succession: tuple[tuple[int, int], ...]
    denial_steps: int
    tied_roles: tuple[int, ...]
    rank_tiers: tuple[tuple[int, ...], ...]

def projected_winner(scores, held, limit, rate):
    scores = np.asarray(scores, dtype=np.float32)
    rates = np.asarray(held, dtype=np.float32) * np.float32(rate)
    remaining = np.maximum(np.float32(limit) - scores, np.float32(0))
    times = tuple(
        float(value / speed) if speed > 0 else (0.0 if score >= np.float32(limit) else math.inf)
        for score, value, speed in zip(scores, remaining, rates)
    )
    first = min(times, default=math.inf)
    leaders = tuple(i for i, value in enumerate(times) if value == first and math.isfinite(value))
    return leaders[0] if len(leaders) == 1 else None, times, leaders

@dataclass(frozen=True)
class ScoreStep:
    scores: tuple[float, ...]
    elapsed: float
    finished: bool
    winner: int | None
    tied_roles: tuple[int, ...]

def integrate_scores(scores, held, limit, rate, elapsed):
    winner, times, tied = projected_winner(scores, held, limit, rate)
    duration = np.maximum(np.float32(elapsed), np.float32(0))
    first = min(times, default=math.inf)
    finished = bool(first <= duration)
    duration = np.float32(first) if finished else duration
    updated = np.asarray(scores, dtype=np.float32) + duration * np.asarray(held, dtype=np.float32) * np.float32(rate)
    return ScoreStep(tuple(float(value) for value in updated), float(duration), finished,
                     winner if finished else None, tied if finished and len(tied) > 1 else ())

def evaluate_cartstate(depths, controls, scores, score_limit, checkpoint_rate):
    scores = tuple(float(value) for value in scores)
    held = [0] * len(scores)
    for depth, control in zip(depths, controls):
        if 0 <= int(control) < len(held):
            held[int(control)] += int(depth)
    projected, times, tied = projected_winner(scores, held, score_limit, checkpoint_rate)
    tiers = tuple(tuple(i for i, own in enumerate(times) if own == value) for value in sorted(set(times)))
    ranks = {team: sum(len(tier) for tier in tiers[position + 1:])
             for position, tier in enumerate(tiers) for team in tier}
    remaining = held.copy()
    leaders = tied
    order = [(team, 0) for team in leaders]
    seen = set(leaders)
    steps = previous = 0
    while leaders and all(remaining[team] > 0 and scores[team] < score_limit for team in leaders):
        for team in leaders:
            remaining[team] -= 1
        steps += len(leaders)
        _, _, leaders = projected_winner(scores, remaining, score_limit, checkpoint_rate)
        entering = tuple(team for team in leaders if team not in seen)
        if entering:
            order.extend((team, steps - previous) for team in entering)
            previous = steps
            seen.update(entering)
    return GameValue(projected, tuple(held), scores, float(score_limit), float(checkpoint_rate),
                     times, ranks, tuple(order), steps, tied if len(tied) > 1 else (), tiers)

SPARSE_REWARD_CONTRACT = {
    "game": GAME_CONTRACT,
    "source": "cart_state_transition",
    "winner_rows": {"event": "projected_winner_loses_role", "value": -1.0},
    "loser_rows": {"event": "upward_loser_rank_flip", "value": 1.0},
    "otherwise": 0.0,
    "terminal_reward": None,
}

SPARSE_REWARD_FINGERPRINT = hashlib.sha256(
    json.dumps(
        SPARSE_REWARD_CONTRACT, sort_keys=True, separators=(",", ":")
    ).encode()
).hexdigest()[:16]

TERMINAL_REWARD_CONTRACT = {
    "game": GAME_CONTRACT,
    "source": "server_round_outcome",
    "terminal_reward": "one_for_winning_team_zero_otherwise",
    "otherwise": 0.0,
    "role_changes_terminate": False,
    "draw_reward": 0.0,
    "truncation": "unlabelled",
}

def reward_contract(arm):
    return TERMINAL_REWARD_CONTRACT if arm == "terminal_win" else SPARSE_REWARD_CONTRACT

def reward_fingerprint(arm):
    return hashlib.sha256(json.dumps(reward_contract(arm), sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]

@dataclass(frozen=True)
class GameContext:
    teams: tuple[int, ...]
    team_of: tuple[int, ...]

@dataclass
class CartSnapshot:
    pos: np.ndarray
    control: np.ndarray
    depths: np.ndarray
    checkpoints: np.ndarray
    scores: np.ndarray
    held: np.ndarray
    score_limit: float
    checkpoint_rate: float
    episode: int
    finished: bool

def formal_game_value(context: GameContext, snapshot: CartSnapshot):
    return evaluate_cartstate(snapshot.depths, snapshot.control, snapshot.scores,
                              snapshot.score_limit, snapshot.checkpoint_rate)

def formal_value_record(value, snapshot):
    return {
        "contract": GAME_CONTRACT,
        "projected_role": None if value.projected_role is None else int(value.projected_role),
        "checkpoints_held": list(value.checkpoints_held),
        "scores": list(value.scores),
        "score_limit": value.score_limit,
        "checkpoint_rate": value.checkpoint_rate,
        "time_to_win": [float(t) if np.isfinite(t) else None for t in value.time_to_win],
        "tied_roles": list(value.tied_roles),
        "rank_tiers": [list(tier) for tier in value.rank_tiers],
        "role_ranks": {
            str(role): int(rank) for role, rank in value.role_ranks.items()
        },
        "succession": [
            [int(role), int(amount)] for role, amount in value.succession
        ],
        "denial_steps": value.denial_steps,
        "state": {"pos": snapshot.pos.tolist(), "depths": snapshot.depths.tolist(), "control": snapshot.control.tolist(),
                  "checkpoints": snapshot.checkpoints.tolist(), "scores": snapshot.scores.tolist(),
                  "held": snapshot.held.tolist(), "score_limit": snapshot.score_limit,
                  "checkpoint_rate": snapshot.checkpoint_rate, "episode": snapshot.episode,
                  "finished": snapshot.finished},
        "held_checkpoint_residual": (snapshot.held - np.asarray(value.checkpoints_held)).tolist(),
    }

def cart_snapshot_record(state):
    arrays = {name: np.asarray(state[name]) for name in ("pos", "depths", "control", "checkpoints", "scores", "held")}
    return CartSnapshot(**{**state, **arrays})

def formal_projection_record(value, teams):
    return {
        "PW": 0 if value.projected_role is None else int(value.projected_role) + 1,
        "SUCC": [
            [int(role) + 1, float(amount)] for role, amount in value.succession
        ],
        "loser_ranks": [int(value.role_ranks.get(role, 0)) for role in teams],
    }

def winner(context: GameContext, snapshot: CartSnapshot):
    return formal_game_value(context, snapshot).projected_role

def loser_ranks(context: GameContext, snapshot: CartSnapshot, value=None) -> np.ndarray:
    value = value or formal_game_value(context, snapshot)
    teams = np.asarray(context.teams, dtype=np.int64)
    ranks = np.zeros(len(context.teams), dtype=np.int64)
    ranks[teams] = np.asarray([value.role_ranks.get(team, 0) for team in context.teams], dtype=np.int64)
    return ranks

def hierarchy_rows(context: GameContext, snapshot: CartSnapshot, value=None):
    value = value or formal_game_value(context, snapshot)
    order = value.succession
    denial = {team: amount for team, amount in order}
    current = value.projected_role
    total = float(max(1, sum(snapshot.checkpoints)))
    rows = np.zeros((len(context.team_of), 8), dtype=np.float32)
    mask = np.zeros(len(context.team_of), dtype=bool)
    for player, team in enumerate(context.team_of):
        rows[player] = (
            snapshot.scores[team] / snapshot.score_limit if snapshot.score_limit else 1.0,
            value.checkpoints_held[team] / total,
            snapshot.checkpoint_rate,
            snapshot.score_limit,
            value.role_ranks[team] / max(1, len(context.teams) - 1),
            float(team == current),
            float(denial.get(team, 0)) / total,
            1.0 / max(1, len(context.teams)),
        )
        mask[player] = team == current
    return rows, mask

def role_rewards(context: GameContext, before: CartSnapshot, after: CartSnapshot) -> np.ndarray:
    before_value = formal_game_value(context, before)
    after_value = formal_game_value(context, after)
    before_winner = before_value.projected_role
    after_winner = after_value.projected_role
    before_ranks = loser_ranks(context, before, before_value)
    after_ranks = loser_ranks(context, after, after_value)
    teams = np.asarray(context.teams, dtype=np.int64)
    winner_mask = teams == before_winner
    rank_flip = after_ranks[teams] > before_ranks[teams]
    team_rewards = np.where(
        winner_mask,
        -float(after_winner != before_winner),
        rank_flip.astype(np.float32),
    ).astype(np.float32)
    return team_rewards[np.asarray(context.team_of, dtype=np.int64)]
