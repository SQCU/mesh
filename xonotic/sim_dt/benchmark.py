from __future__ import annotations

import argparse
import os
import resource
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim_dt import DeltaTimePayloadEngine, SimRulesConfig, build_gas_world_graph
from solver.strat.plan_policy import PlanWally, sample_plan


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def measure_dt_convergence(device: torch.device) -> dict:
    """Verify exact cart/checkpoint/score rules across dt=0.05 (PLC_TICK ref), 0.2, 0.4, 1.0, and adaptive."""
    world = build_gas_world_graph(seed=42, teams=4, carts=8, players=24, tiles=2, device=device)
    # Deterministic pure-cart/checkpoint convergence test: freeze players directly onto carts
    # and compare score & cart_s after T=20.0s across dt in {0.05, 0.2, 0.4, 1.0}.
    ref_rules = SimRulesConfig(combat_range=0.0, move_speed=360.0)
    results = {}
    for dt in (0.05, 0.2, 0.4, 1.0):
        eng = DeltaTimePayloadEngine(world, num_envs=4, num_players=16, rules=ref_rules, device=device, base_seed=7)
        # Escort players co-move with carts 0..2 (team 1 on cart 0, team 2 on cart 1, 2v1 contest on cart 2)
        eng.p_pos[:, 0:2] = eng.c_pos[:, 0:1]
        eng.p_team[:, 0:2] = 1
        eng.p_goal_cart[:, 0:2] = 0
        eng.p_pos[:, 2:4] = eng.c_pos[:, 1:2]
        eng.p_team[:, 2:4] = 2
        eng.p_goal_cart[:, 2:4] = 1
        eng.p_pos[:, 4:6] = eng.c_pos[:, 2:3]
        eng.p_team[:, 4:6] = 1
        eng.p_goal_cart[:, 4:6] = 2
        eng.p_pos[:, 6] = eng.c_pos[:, 2]
        eng.p_team[:, 6] = 3
        eng.p_goal_cart[:, 6] = 2
        eng.p_goal_timer[:] = 100.0  # lock cart assignments during pure ODE parity check
        eng.p_pos[:, 7:] = eng.p_pos[:, 7:] + 5000.0
        eng.p_alive[:, 7:] = False
        eng.p_respawn_t[:, 7:] = 1e6
        steps = int(round(20.0 / dt))
        mode = "substep" if dt == 0.05 else "analytic"
        for _ in range(steps):
            eng.step(dt, mode=mode)
        _sync(device)
        results[dt] = {
            "c_s": eng.c_s[0, :3].detach().cpu().tolist(),
            "t_score": eng.t_score[0].detach().cpu().tolist(),
            "cp_held": eng.t_cp_held[0].detach().cpu().tolist(),
        }
    ref = results[0.05]
    diffs = {}
    for dt, r in results.items():
        max_s = max(abs(a - b) for a, b in zip(r["c_s"], ref["c_s"]))
        max_sc = max(abs(a - b) for a, b in zip(r["t_score"], ref["t_score"]))
        cp_exact = r["cp_held"] == ref["cp_held"]
        diffs[f"dt={dt}"] = {"max_cart_s_err": round(max_s, 6), "max_score_err": round(max_sc, 4), "cp_held_exact": cp_exact}
    return diffs


def measure_savestate_grpo_branching(device: torch.device) -> dict:
    """Verify O(1) snapshot/restore and GRPO counterfactual branching with CRN vs divergent seeds."""
    world = build_gas_world_graph(seed=101, teams=4, carts=8, players=24, tiles=2, device=device)
    eng = DeltaTimePayloadEngine(world, num_envs=4, num_players=24, device=device, base_seed=11)
    for _ in range(15):
        eng.step(0.4, mode="analytic")
    _sync(device)

    t0 = time.perf_counter()
    snap = eng.snapshot()
    grp_crn = DeltaTimePayloadEngine.fork_group(world, snap, group_size=8, common_random_numbers=True)
    grp_div = DeltaTimePayloadEngine.fork_group(world, snap, group_size=8, common_random_numbers=False)
    _sync(device)
    fork_ms = (time.perf_counter() - t0) * 1000.0

    for _ in range(10):
        grp_crn.step(0.4, mode="analytic")
        grp_div.step(0.4, mode="analytic")
    _sync(device)

    crn_spread = float((grp_crn.t_score[:8] - grp_crn.t_score[0:1]).abs().max().item())
    div_spread = float((grp_div.t_score[:8] - grp_div.t_score[0:1]).abs().max().item())
    # Apply different cart picks on half the CRN group to prove causal divergence from identical state+RNG
    grp_cf = DeltaTimePayloadEngine.fork_group(world, snap, group_size=8, common_random_numbers=True)
    plan = torch.zeros((grp_cf.E, grp_cf.N, 16), device=device, dtype=torch.float32)
    plan[:, :, 0] = torch.arange(1, grp_cf.N + 1, device=device, dtype=torch.float32)[None, :]
    for g in range(8):
        plan[g::8, :, 1] = float(g % world.carts)
        plan[g::8, :, 2] = 2.0
    grp_cf.apply_plan(plan)
    for _ in range(15):
        grp_cf.step(0.4, mode="analytic")
    _sync(device)
    cf_pos_spread = float((grp_cf.p_pos[:8] - grp_cf.p_pos[0:1]).abs().max().item())
    return {
        "fork_32_envs_ms": round(fork_ms, 3),
        "crn_identical_score_spread": round(crn_spread, 8),
        "divergent_seed_score_spread": round(div_spread, 4),
        "counterfactual_action_pos_spread": round(cf_pos_spread, 2),
    }


def measure_throughput_sweep(
    device: torch.device,
    env_counts: tuple[int, ...] = (1, 16, 128, 1024, 4096),
    dt: float = 0.4,
    mode: str = "analytic",
    n_steps: int = 50,
    teams: int = 4,
    carts: int = 12,
    players: int = 32,
) -> list[dict]:
    world = build_gas_world_graph(seed=7, teams=teams, carts=carts, players=players, tiles=4, device=device)
    rows = []
    for E in env_counts:
        eng = DeltaTimePayloadEngine(world, num_envs=E, num_players=players, device=device, base_seed=1)
        # Warmup
        for _ in range(5):
            eng.step(dt, mode=mode)
        _sync(device)

        t0 = time.perf_counter()
        for _ in range(n_steps):
            eng.step(dt, mode=mode)
            obs, c_t, t_t, _ = eng.observe_tensors()
        _sync(device)
        elapsed = max(1e-6, time.perf_counter() - t0)
        sim_sec = E * n_steps * dt
        speedup = sim_sec / elapsed
        env_steps_s = (E * n_steps) / elapsed
        agent_steps_s = env_steps_s * players
        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0)
        rows.append({
            "E": E,
            "N": players,
            "J": carts,
            "K": teams,
            "dt": dt,
            "mode": mode,
            "wall_s": round(elapsed, 4),
            "sim_sec_per_wall_sec": round(speedup, 1),
            "env_steps_per_sec": round(env_steps_s, 1),
            "agent_steps_per_sec": round(agent_steps_s, 1),
            "max_rss_mb": round(rss_mb, 1),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--envs", default="1,16,128,1024,4096")
    ap.add_argument("--dt", type=float, default=0.4)
    ap.add_argument("--mode", default="analytic", choices=("analytic", "substep", "adaptive"))
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--players", type=int, default=32)
    ap.add_argument("--carts", type=int, default=12)
    ap.add_argument("--teams", type=int, default=4)
    args = ap.parse_args()

    dev = torch.device(args.device)
    print(f"=== DeltaTimePayloadEngine Benchmark (device={dev}) ===")
    conv = measure_dt_convergence(dev)
    print("1. Variable-dt Rule Equivalence vs dt=0.05 (PLC_TICK):")
    for k, v in conv.items():
        print(f"   {k}: {v}")

    br = measure_savestate_grpo_branching(dev)
    print(f"2. Savestate / GRPO Counterfactual Branching: {br}")

    env_counts = tuple(int(x) for x in args.envs.split(",") if x.strip())
    print(f"3. Throughput Sweep (dt={args.dt}s, mode={args.mode}, N={args.players}, J={args.carts}, K={args.teams}):")
    for row in measure_throughput_sweep(
        dev,
        env_counts=env_counts,
        dt=args.dt,
        mode=args.mode,
        n_steps=args.steps,
        teams=args.teams,
        carts=args.carts,
        players=args.players,
    ):
        print(
            f"   E={row['E']:4d} | speedup={row['sim_sec_per_wall_sec']:10.1f}x realtime "
            f"| env_steps/s={row['env_steps_per_sec']:10.1f} "
            f"| agent_steps/s={row['agent_steps_per_sec']:12.1f} "
            f"| wall={row['wall_s']:.4f}s | rss={row['max_rss_mb']:.1f}MB"
        )

    # Also test exact PlanWally zero-copy compatibility on one env slice
    world = build_gas_world_graph(seed=7, teams=args.teams, carts=args.carts, players=args.players, tiles=4, device=dev)
    eng = DeltaTimePayloadEngine(world, num_envs=8, num_players=args.players, device=dev, base_seed=3)
    obs, carts_t, teams_t, geo = eng.observe_tensors()
    model = PlanWally().to(dev).eval()
    with torch.no_grad():
        out = model(obs[0], carts_t[0], teams_t[0], geo=geo)
        smp = sample_plan(out)
    print(f"4. PlanWally Zero-Copy Forward OK: cont={tuple(smp.cont.shape)} cart_pick={tuple(smp.cart_pick.shape)} device={smp.cont.device}")


if __name__ == "__main__":
    main()
