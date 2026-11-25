import hashlib
import json
from dataclasses import dataclass
import math
import numpy as np

GAME_CONTRACT = "checkpoint-control-integral-v1"
WINNER_BARRIER_MARGIN = 0.25
WINNER_BARRIER_WEIGHT = 0.5
NIM_BOUNDED_STEP = 2


def finish_rates(scores, held, rate, limit, eps=1e-12):
    scores = np.asarray(scores, dtype=np.float64)
    held = np.asarray(held, dtype=np.float64)
    remaining = np.maximum(float(limit) - scores, eps)
    return np.where(held > 0, (float(rate) * held) / remaining, 0.0)


def winning_margin(rho, eps=1e-12):
    rho = np.asarray(rho, dtype=np.float64)
    k = len(rho)
    out = np.zeros(k, dtype=np.float64)
    for i in range(k):
        best_other = float(np.max(np.delete(rho, i))) if k > 1 else 0.0
        out[i] = (rho[i] - best_other) / (rho[i] + best_other + eps)
    return out


def bounded_nim_sum(heaps, max_take=NIM_BOUNDED_STEP):
    mod = int(max_take) + 1
    nimbers = [max(0, int(h)) % mod for h in heaps]
    total = 0
    for val in nimbers:
        total ^= val
    winning = [
        1.0 if (total != 0 and (v ^ total) < v and (v - (v ^ total)) <= max(0, int(h))) else 0.0
        for h, v in zip(heaps, nimbers)
    ]
    return int(total), np.asarray(winning, dtype=np.float32)


def lane_thermographs(snapshot, eps_dist=0.05):
    scores = np.asarray(snapshot.scores, dtype=np.float64)
    held = np.asarray(snapshot.held, dtype=np.float64)
    rate = float(snapshot.checkpoint_rate)
    limit = float(snapshot.score_limit)
    k = len(scores)
    j = len(snapshot.depths)
    base_rho = finish_rates(scores, held, rate, limit)
    base_m = winning_margin(base_rho)
    pos = np.clip(np.asarray(snapshot.pos, dtype=np.float64), 0.0, 1.0)
    cp_max = np.maximum(np.asarray(snapshot.checkpoints, dtype=np.float64), 1.0)
    seg = pos * cp_max
    dist_push = np.maximum(np.ceil(seg + 1e-9) - seg, eps_dist)
    dist_deny = np.maximum(seg - np.floor(seg - 1e-9), eps_dist)
    deny = np.zeros(j, dtype=np.float32)
    push = np.zeros(j, dtype=np.float32)
    for idx in range(j):
        owner = int(snapshot.control[idx])
        depth = int(snapshot.depths[idx])
        if 0 <= owner < k and depth > 0 and held[owner] > 0:
            h_mod = held.copy()
            h_mod[owner] -= 1.0
            m_mod = winning_margin(finish_rates(scores, h_mod, rate, limit))
            deny[idx] = float(np.max(np.abs(m_mod - base_m)) / dist_deny[idx])
        if depth < int(cp_max[idx]):
            best_gain = 0.0
            for team in range(k):
                h_mod = held.copy()
                h_mod[team] += 1.0
                m_mod = winning_margin(finish_rates(scores, h_mod, rate, limit))
                best_gain = max(best_gain, float(m_mod[team] - base_m[team]))
            push[idx] = float(best_gain / dist_push[idx])
    rem_heaps = [max(0, int(c) - int(d)) for c, d in zip(snapshot.checkpoints, snapshot.depths)]
    nim_val, nim_win = bounded_nim_sum(rem_heaps)
    return {
        "finish_rates": base_rho.astype(np.float32),
        "winning_margins": base_m.astype(np.float32),
        "deny": deny,
        "push": push,
        "nim_sum": nim_val,
        "nim_winning": nim_win,
    }


def cgt_tensor_features(carts, teams, cart_present, team_present, eps_dist=0.05):
    import torch
    from payload.tools.strategy_io_schema import CS, TS

    with torch.no_grad():
        device, dtype = carts.device, carts.dtype
        k = teams.shape[0]
        j = carts.shape[0]
        t_valid = team_present.to(torch.bool)
        scores = teams[:, TS["SCORE"]].to(torch.float32)
        held = teams[:, TS["CHECKPOINTS_HELD"]].to(torch.float32)
        rate = teams[:, TS["CHECKPOINT_RATE"]].to(torch.float32)
        limit = torch.clamp(teams[:, TS["SCORE_LIMIT"]].to(torch.float32), min=1e-6)
        rem = torch.clamp(limit - scores, min=1e-6)
        rho = torch.where(t_valid & (held > 0), (rate * held) / rem, torch.zeros_like(scores))
        if k > 1:
            eye_mask = torch.eye(k, device=device, dtype=torch.bool)
            other_rho = rho[None, :].expand(k, k).masked_fill(eye_mask, -1.0)
            best_other = other_rho.amax(dim=1).clamp(min=0.0)
        else:
            best_other = torch.zeros_like(rho)
        margins = torch.where(t_valid, (rho - best_other) / (rho + best_other + 1e-6), torch.zeros_like(rho))
        rho_norm = torch.where(t_valid, torch.tanh(rho), torch.zeros_like(rho))
        team_extra = torch.stack((rho_norm, margins), dim=-1).to(dtype)

        c_valid = cart_present.to(torch.bool)
        path_len = torch.clamp(carts[:, CS["PATH_LENGTH"]].to(torch.float32), min=1e-6)
        pos_frac = torch.clamp(carts[:, CS["PATH_POSITION"]].to(torch.float32) / path_len, 0.0, 1.0)
        cp_count = torch.clamp(carts[:, CS["CHECKPOINT_COUNT"]].to(torch.float32), min=1.0)
        cp_depth = torch.clamp(carts[:, CS["CHECKPOINT_DEPTH"]].to(torch.float32), min=0.0)
        seg = pos_frac * cp_count
        dist_push = torch.clamp(torch.ceil(seg + 1e-6) - seg, min=eps_dist)
        dist_deny = torch.clamp(seg - torch.floor(seg - 1e-6), min=eps_dist)

        owner_ids = carts[:, CS["CONTROL_TEAM"]].to(torch.float32)
        team_ids = teams[:, TS["ID"]].to(torch.float32)
        owner_match = (owner_ids[:, None] == team_ids[None, :]) & t_valid[None, :]
        has_owner = owner_match.any(dim=1)
        owner_idx = owner_match.to(torch.int64).argmax(dim=1)
        owner_held = held[owner_idx]
        owner_rho = rho[owner_idx]
        # Marginal finish-rate shift on +/-1 checkpoint flip (exact closed form)
        delta_rho_deny = torch.where(
            c_valid & has_owner & (cp_depth > 0) & (owner_held > 0),
            owner_rho / torch.clamp(owner_held, min=1.0),
            torch.zeros_like(owner_rho),
        )
        t_deny = torch.tanh(delta_rho_deny / (owner_rho + best_other[owner_idx] + 1e-6) / dist_deny)
        max_rho = rho.amax().clamp(min=1e-6)
        delta_rho_push = torch.where(
            c_valid & (cp_depth < cp_count),
            (rate.amax().clamp(min=0.0) / rem.amin().clamp(min=1e-6)),
            torch.zeros_like(owner_rho),
        )
        t_push = torch.tanh((delta_rho_push / (max_rho + delta_rho_push + 1e-6)) / dist_push)

        rem_heaps = torch.where(c_valid, torch.clamp((cp_count - cp_depth).round().to(torch.int64), min=0), torch.zeros(j, device=device, dtype=torch.int64))
        mod = NIM_BOUNDED_STEP + 1
        nimbers = rem_heaps % mod
        nim_total = torch.zeros((), device=device, dtype=torch.int64)
        for idx in range(j):
            nim_total = torch.bitwise_xor(nim_total, nimbers[idx])
        target_nim = torch.bitwise_xor(nimbers, nim_total)
        nim_move = ((nim_total != 0) & (target_nim < nimbers) & ((nimbers - target_nim) <= rem_heaps) & c_valid).to(torch.float32)
        nim_mod = torch.where(c_valid, nimbers.to(torch.float32) / float(mod), torch.zeros_like(t_deny))
        cart_extra = torch.stack((t_deny, t_push, nim_move, nim_mod), dim=-1).to(dtype)
        return cart_extra, team_extra

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
    "winner_boundary_barrier": {"margin": WINNER_BARRIER_MARGIN, "weight": WINNER_BARRIER_WEIGHT},
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
    lanes = lane_thermographs(snapshot)
    return {
        "contract": GAME_CONTRACT,
        "projected_role": None if value.projected_role is None else int(value.projected_role),
        "checkpoints_held": list(value.checkpoints_held),
        "scores": list(value.scores),
        "score_limit": value.score_limit,
        "checkpoint_rate": value.checkpoint_rate,
        "time_to_win": [float(t) if np.isfinite(t) else None for t in value.time_to_win],
        "finish_rates": lanes["finish_rates"].tolist(),
        "winning_margins": lanes["winning_margins"].tolist(),
        "lane_thermograph_deny": lanes["deny"].tolist(),
        "lane_thermograph_push": lanes["push"].tolist(),
        "lane_nim_sum": lanes["nim_sum"],
        "lane_nim_winning": lanes["nim_winning"].tolist(),
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
    after_margins = winning_margin(
        finish_rates(after_value.scores, after_value.checkpoints_held, after.checkpoint_rate, after.score_limit)
    )
    teams = np.asarray(context.teams, dtype=np.int64)
    winner_mask = teams == before_winner
    rank_flip = after_ranks[teams] > before_ranks[teams]
    if before_winner is not None and after_winner == before_winner:
        deficit = max(0.0, WINNER_BARRIER_MARGIN - float(after_margins[before_winner]))
        winner_reward = -WINNER_BARRIER_WEIGHT * (deficit * deficit)
    else:
        winner_reward = -float(after_winner != before_winner)
    team_rewards = np.where(
        winner_mask,
        winner_reward,
        rank_flip.astype(np.float32),
    ).astype(np.float32)
    return team_rewards[np.asarray(context.team_of, dtype=np.int64)]
