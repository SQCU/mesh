"""PFSP + alpha-Rank + Permanent Scripted Anchors + FCP Mixed-Teammate League Curriculum.

Implements non-brittle 3-team multi-agent league matchmaking:
  1. Permanent non-evictable anchors: stock HavocBot (`default`) + 4 deterministic scripted
     PLAN archetypes (`script_cart_rush`, `script_leader_gank`, `script_powerup_roam`,
     `script_vanguard_ambush`) to prevent mutual-passivity fixed points.
  2. Prioritized Fictitious Self-Play (PFSP): weights opponents by `(1 - p_win)^2` so any
     strategy or exploiter currently beating `main` is replayed immediately.
  3. 3-team alpha-Rank (Omidshafiei et al. 2019): exact Markov-Conley chain stationary
     distribution over the 3-way empirical payoff tensor M(i, j, k).
  4. Fictitious Co-Play (FCP): mixes 1 `default` teammate inside `train`'s own 3-bot squad on
     a fraction of servers so policies adapt to non-mirror teammates & human players.
  5. Physics & lane-scaled horizon perturbations from `curriculum.py`.
"""
from __future__ import annotations

import math
import numpy as np

from payload.tools.strategy_io_schema import CS, OBS, TS
from solver.strat.curriculum import PERTURBATIONS
from solver.strat.game_value import effective_leader
from solver.strat.plan_policy import PLAN_WIDTH


SCRIPTED_ANCHORS = (
    "script_cart_rush",
    "script_leader_gank",
    "script_powerup_roam",
    "script_vanguard_ambush",
)

PHYSICS_PERTURBATION_NAMES = tuple(PERTURBATIONS.keys())  # baseline, fast, slow, volatile


def scripted_plan_rows(
    arm_name: str,
    obs_rows: np.ndarray,
    cart_rows: np.ndarray,
    team_rows: np.ndarray,
    bot_arms: np.ndarray,
    maxclients: int = 16,
    cell_centroids: np.ndarray | None = None,
    projected_leader_idx: int | None = None,
) -> np.ndarray:
    """Emit deterministic 16-float parametric PLAN rows for permanent scripted anchor archetypes."""
    out = np.zeros((maxclients, PLAN_WIDTH), dtype=np.float32)
    if arm_name not in SCRIPTED_ANCHORS or len(obs_rows) == 0 or len(cart_rows) == 0:
        return out

    team_ids = team_rows[:, TS["ID"]].astype(np.int64)
    n_carts = len(cart_rows)
    cart_ctrl = cart_rows[:, CS["CONTROL_TEAM"]].astype(np.int64)
    cart_pos = cart_rows[:, CS["POS_X"] : CS["POS_Z"] + 1]
    bot_pos = obs_rows[:, OBS["POS_X"] : OBS["POS_Z"] + 1]

    leader_team_id = 0.0
    if projected_leader_idx is not None and 0 <= projected_leader_idx < len(team_ids):
        leader_team_id = float(team_ids[projected_leader_idx])
    elif len(team_rows) > 0:
        leader_team_id = float(team_ids[int(np.argmax(team_rows[:, TS["SCORE"]]))])

    for i in range(len(obs_rows)):
        if str(bot_arms[i]) != arm_name:
            continue
        eid = int(round(float(obs_rows[i, OBS["ID"]])))
        if not (1 <= eid <= maxclients):
            continue
        my_team = int(round(float(obs_rows[i, OBS["TEAM"]])))
        dists = np.linalg.norm(cart_pos - bot_pos[i : i + 1], axis=1)
        nearest_cart = int(np.argmin(dists))

        row = np.zeros((PLAN_WIDTH,), dtype=np.float32)
        row[0] = float(eid)
        if arm_name == "script_cart_rush":
            # All 3 bots push own/nearest cart hard with low combat distraction
            own_carts = [c for c in range(n_carts) if int(cart_ctrl[c]) == my_team]
            cpick = own_carts[0] if own_carts else nearest_cart
            row[1] = float(cpick)
            row[2 + min(cpick, 2)] = 1.6
            row[5] = 1.4    # w3: high cart_scale
            row[6] = -0.6   # w4: lower enemy chase
            row[14] = 0.4   # atk_cyl
            row[15] = -0.3  # defensive hold on cart
        elif arm_name == "script_leader_gank":
            # Focus-fire the leading team + contest their cart aggressively
            lead_carts = [c for c in range(n_carts) if int(cart_ctrl[c]) == int(leader_team_id) and int(leader_team_id) != my_team]
            cpick = lead_carts[0] if lead_carts else nearest_cart
            row[1] = float(cpick)
            row[2 + min(cpick, 2)] = 1.2
            row[5] = 0.6
            row[6] = 1.4    # high enemy aggression
            row[7] = leader_team_id if int(leader_team_id) != my_team else 0.0
            row[14] = 1.2   # high atk_cyl
            row[15] = 0.7   # aggressive posture
        elif arm_name == "script_powerup_roam":
            # Prioritize powerups & items while contesting mid-map Voronoi cells
            row[1] = float(nearest_cart)
            row[5] = 0.3
            row[6] = 0.5
            row[8] = 1.4    # item_w
            row[9] = 1.8    # pwr_w
            if cell_centroids is not None and len(cell_centroids) > 0:
                c_idx = (eid + my_team) % len(cell_centroids)
                row[11] = float(cell_centroids[c_idx, 0])
                row[12] = float(cell_centroids[c_idx, 1])
                row[13] = 0.35
            row[15] = 0.4
        elif arm_name == "script_vanguard_ambush":
            # Push ahead of cart (+240u vanguard lead) to clear chokes before cart arrives
            own_carts = [c for c in range(n_carts) if int(cart_ctrl[c]) == my_team]
            cpick = own_carts[0] if own_carts else nearest_cart
            row[1] = float(cpick)
            row[2 + min(cpick, 2)] = 1.2
            row[5] = 1.1
            row[6] = 0.8
            row[10] = 240.0 if (i % 2 == 0) else 0.0  # vanguard lead + pusher split!
            row[14] = 0.8
            row[15] = 0.5
        out[eid - 1] = row
    return out


def alpha_rank_3team(
    arms: list[str],
    matches: list[dict[str, float]],
    *,
    alpha: float = 2.0,
    m_pop: int = 20,
) -> dict[str, float]:
    """Compute exact 3-player symmetric alpha-Rank (Omidshafiei et al. 2019) over a 3-way
    empirical payoff tensor estimated from live 3-team matches.

    For each strategy i in {0..K-1}, computes pairwise & triplet win-rate payoffs, constructs
    the single-mutant fixation probability matrix over monomorphic populations (K states),
    and returns the unique stationary distribution pi_alpha."""
    k = len(arms)
    if k == 0:
        return {}
    if k == 1:
        return {arms[0]: 1.0}
    idx = {a: i for i, a in enumerate(arms)}
    # Estimate pairwise win-rate matrix P[i, j] (regularized with 0.5 prior)
    wins = np.full((k, k), 0.5, dtype=np.float64)
    games = np.full((k, k), 1.0, dtype=np.float64)
    np.fill_diagonal(wins, 0.0)
    np.fill_diagonal(games, 0.0)
    # Also estimate 3-way empirical payoff M[i, j, l] where available, falling back to pairwise mean
    tri_sum = np.zeros((k, k, k), dtype=np.float64)
    tri_cnt = np.zeros((k, k, k), dtype=np.float64)
    for m in matches:
        present = [a for a in m if a in idx]
        if len(present) < 2:
            continue
        vals = {a: float(m[a]) for a in present}
        for i_a in range(len(present)):
            for i_b in range(i_a + 1, len(present)):
                a, b = present[i_a], present[i_b]
                ia, ib = idx[a], idx[b]
                s = 1.0 if vals[a] > vals[b] + 1e-9 else (0.0 if vals[b] > vals[a] + 1e-9 else 0.5)
                wins[ia, ib] += s
                wins[ib, ia] += 1.0 - s
                games[ia, ib] += 1.0
                games[ib, ia] += 1.0
        if len(present) == 3:
            a, b, c = present
            ia, ib, ic = idx[a], idx[b], idx[c]
            arr = np.asarray([vals[a], vals[b], vals[c]], dtype=np.float64)
            # Rank-normalized score in [0, 1] for each seat in the 3-team match
            for pos, (me, o1, o2) in enumerate(((ia, ib, ic), (ib, ia, ic), (ic, ia, ib))):
                v_me = arr[pos]
                others = [arr[j] for j in range(3) if j != pos]
                sc = 0.5 * sum(1.0 if v_me > vo + 1e-9 else (0.0 if vo > v_me + 1e-9 else 0.5) for vo in others)
                tri_sum[me, o1, o2] += sc
                tri_sum[me, o2, o1] += sc
                tri_cnt[me, o1, o2] += 1.0
                tri_cnt[me, o2, o1] += 1.0

    pw = np.where(games > 0, wins / np.maximum(games, 1e-12), 0.5)
    np.fill_diagonal(pw, 0.5)

    def _payoff(mutant: int, opp1: int, opp2: int) -> float:
        if tri_cnt[mutant, opp1, opp2] > 0:
            return float(tri_sum[mutant, opp1, opp2] / tri_cnt[mutant, opp1, opp2])
        return 0.5 * float(pw[mutant, opp1] + pw[mutant, opp2])

    # Monomorphic Markov chain transition matrix C[r, s] (resident r -> mutant s)
    eta = 1.0 / max(1, k - 1)
    C = np.zeros((k, k), dtype=np.float64)
    for r in range(k):
        for s in range(k):
            if r == s:
                continue
            # Single-mutant invasion payoff against (r, r) vs resident payoff in (r, r, r)
            u_sr = _payoff(s, r, r)
            u_rr = _payoff(r, r, r)
            diff = alpha * (u_sr - u_rr)
            if abs(diff) < 1e-9:
                rho = 1.0 / float(m_pop)
            else:
                # Exact Moran fixation probability (1 - exp(-diff)) / (1 - exp(-m * diff))
                e1 = math.exp(-np.clip(diff, -30.0, 30.0))
                em = math.exp(-np.clip(m_pop * diff, -30.0, 30.0))
                rho = (1.0 - e1) / max(1.0 - em, 1e-12) if diff > 0 else (e1 - 1.0) / max(em - 1.0, 1e-12)
            C[r, s] = eta * max(0.0, min(1.0, rho))
        C[r, r] = max(0.0, 1.0 - C[r].sum())

    # Power iteration to unique stationary distribution pi C = pi
    pi = np.full(k, 1.0 / k, dtype=np.float64)
    for _ in range(500):
        pi = pi @ C
        pi = np.maximum(pi, 0.0)
        pi /= max(pi.sum(), 1e-12)
    return {arms[i]: round(float(pi[i]), 4) for i in range(k)}


def sample_pfsp_server_assignments(
    rng: np.random.Generator,
    n_servers: int,
    *,
    train_arm: str,
    ema_arm: str | None,
    league_arms: list[str],
    win_rates_vs_opp: dict[str, float],
    pfsp_power: float = 2.0,
) -> list[dict]:
    """Build per-server matchmaking configs for N concurrent servers on a Studio, mixing:
      - ~40% PFSP hard opponents weighted by (1 - p_win)^p
      - ~25% current / near-current mirror + league opponent
      - ~20% permanent anchors (`default` + scripted archetypes)
      - ~15% FCP mixed-teammate (`fcp_mixed=True`: 1 default bot on `train`'s own team)
    Plus randomized physics perturbation (`baseline/fast/slow/volatile`)."""
    anchors = ["default", *SCRIPTED_ANCHORS]
    candidates = sorted(set(anchors + list(league_arms)))
    weights = np.asarray(
        [max(0.05, 1.0 - float(win_rates_vs_opp.get(a, 0.45))) ** pfsp_power for a in candidates],
        dtype=np.float64,
    )
    weights /= weights.sum()

    configs = []
    for idx in range(n_servers):
        pert = str(PHYSICS_PERTURBATION_NAMES[int(rng.integers(0, len(PHYSICS_PERTURBATION_NAMES)))])
        fcp_mixed = False
        if idx == 0:
            # Always keep at least 1 permanent anchor / scripted pressure server per Studio
            opp1 = "default"
            opp2 = str(rng.choice(SCRIPTED_ANCHORS))
        elif idx == 1 and ema_arm is not None:
            # Mirror / coalitional self-play pressure
            opp1 = ema_arm
            opp2 = str(rng.choice(candidates, p=weights))
        elif idx == n_servers - 1:
            # FCP mixed-teammate server (1 default bot inside train's own 3-bot team!)
            fcp_mixed = True
            opp1 = str(rng.choice(candidates, p=weights))
            opp2 = "default"
        else:
            # Pure PFSP hard-opponent triplet
            opp1 = str(rng.choice(candidates, p=weights))
            opp2 = str(rng.choice(candidates, p=weights))
        configs.append({
            "team_arms": (train_arm, opp1, opp2),
            "fcp_mixed": fcp_mixed,
            "perturbation": pert,
        })
    return configs
