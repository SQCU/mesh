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

from sim_dt import (
    CompiledPlanWally,
    CompiledRolloutGraph,
    MetalDeltaTimePayloadEngine,
    SimRulesConfig,
    build_gas_world_graph,
)
from solver.strat.plan_policy import PlanWally, sample_plan


def measure_dt_convergence() -> dict:
    """Verify exact cart/checkpoint/score rules across dt=0.05 (PLC_TICK ref), 0.2, 0.4, 1.0 on Metal 4."""
    world = build_gas_world_graph(seed=42, teams=4, carts=8, players=24, tiles=2, device=torch.device("cpu"))
    ref_rules = SimRulesConfig(combat_range=0.0, move_speed=360.0)
    results = {}
    for dt in (0.05, 0.2, 0.4, 1.0):
        eng = MetalDeltaTimePayloadEngine(world, num_envs=4, num_players=16, rules=ref_rules, base_seed=7)
        try:
            sv = eng.state_views()
            sv["p_pos"][:, 0:2] = sv["c_pos"][:, 0:1]
            sv["p_team"][:, 0:2] = 1
            sv["p_goal_cart"][:, 0:2] = 0
            sv["p_pos"][:, 2:4] = sv["c_pos"][:, 1:2]
            sv["p_team"][:, 2:4] = 2
            sv["p_goal_cart"][:, 2:4] = 1
            sv["p_pos"][:, 4:6] = sv["c_pos"][:, 2:3]
            sv["p_team"][:, 4:6] = 1
            sv["p_goal_cart"][:, 4:6] = 2
            sv["p_pos"][:, 6] = sv["c_pos"][:, 2]
            sv["p_team"][:, 6] = 3
            sv["p_goal_cart"][:, 6] = 2
            sv["p_goal_timer"][:] = 100.0
            sv["p_pos"][:, 7:] = sv["p_pos"][:, 7:] + 5000.0
            sv["p_alive"][:, 7:] = 0
            sv["p_respawn_t"][:, 7:] = 1e6
            steps = int(round(20.0 / dt))
            mode = "substep" if dt == 0.05 else "analytic"
            eng.step(dt, mode=mode, n_steps=steps)
            results[dt] = {
                "c_s": eng.c_s[0, :3].tolist(),
                "t_score": eng.t_score[0].tolist(),
                "cp_held": eng.t_cp_held[0].tolist(),
            }
        finally:
            eng.close()
    ref = results[0.05]
    diffs = {}
    for dt, r in results.items():
        max_s = max(abs(a - b) for a, b in zip(r["c_s"], ref["c_s"]))
        max_sc = max(abs(a - b) for a, b in zip(r["t_score"], ref["t_score"]))
        cp_exact = r["cp_held"] == ref["cp_held"]
        diffs[f"dt={dt}"] = {"max_cart_s_err": round(max_s, 6), "max_score_err": round(max_sc, 4), "cp_held_exact": cp_exact}
    return diffs


def measure_savestate_grpo_branching() -> dict:
    """Verify O(1) snapshot/restore and GRPO counterfactual branching with CRN vs divergent seeds on Metal 4."""
    world = build_gas_world_graph(seed=101, teams=4, carts=8, players=24, tiles=2, device=torch.device("cpu"))
    eng = MetalDeltaTimePayloadEngine(world, num_envs=4, num_players=24, base_seed=11)
    eng.step(0.4, mode="analytic", n_steps=15)

    t0 = time.perf_counter()
    snap = eng.snapshot()
    grp_crn = MetalDeltaTimePayloadEngine.fork_group(world, snap, group_size=8, common_random_numbers=True)
    grp_div = MetalDeltaTimePayloadEngine.fork_group(world, snap, group_size=8, common_random_numbers=False)
    fork_ms = (time.perf_counter() - t0) * 1000.0

    grp_crn.step(0.4, mode="analytic", n_steps=10)
    grp_div.step(0.4, mode="analytic", n_steps=10)

    crn_spread = float((grp_crn.t_score[:8] - grp_crn.t_score[0:1]).abs().max().item())
    div_spread = float((grp_div.t_score[:8] - grp_div.t_score[0:1]).abs().max().item())
    grp_cf = MetalDeltaTimePayloadEngine.fork_group(world, snap, group_size=8, common_random_numbers=True)
    plan = torch.zeros((grp_cf.E, grp_cf.N, 16), dtype=torch.float32)
    plan[:, :, 0] = torch.arange(1, grp_cf.N + 1, dtype=torch.float32)[None, :]
    for g in range(8):
        plan[g::8, :, 1] = float(g % world.carts)
        plan[g::8, :, 2] = 2.0
    grp_cf.apply_plan(plan)
    grp_cf.step(0.4, mode="analytic", n_steps=15)
    cf_pos_spread = float((grp_cf.p_pos[:8] - grp_cf.p_pos[0:1]).abs().max().item())

    eng.close(); grp_crn.close(); grp_div.close(); grp_cf.close()
    return {
        "fork_32_envs_ms": round(fork_ms, 3),
        "crn_identical_score_spread": round(crn_spread, 8),
        "divergent_seed_score_spread": round(div_spread, 4),
        "counterfactual_action_pos_spread": round(cf_pos_spread, 2),
    }


def measure_throughput_sweep(
    env_counts: tuple[int, ...] = (1, 16, 128, 1024, 4096),
    dt: float = 0.4,
    mode: str = "analytic",
    n_steps: int = 50,
    teams: int = 4,
    carts: int = 12,
    players: int = 32,
) -> list[dict]:
    world = build_gas_world_graph(seed=7, teams=teams, carts=carts, players=players, tiles=4, device=torch.device("cpu"))
    model = PlanWally().to(dtype=torch.bfloat16).eval()
    rows = []
    for E in env_counts:
        eng = MetalDeltaTimePayloadEngine(world, num_envs=E, num_players=players, base_seed=1)
        roll_g = CompiledRolloutGraph(eng, model, max_steps=max(64, n_steps + 16), stochastic=True)
        try:
            eng.step(dt, mode=mode, n_steps=5)
            t0 = time.perf_counter()
            eng.step(dt, mode=mode, n_steps=n_steps)
            elapsed_sim = max(1e-6, time.perf_counter() - t0)

            res_cl = roll_g.run_rollout(n_steps, dt=dt, mode=mode, fused_cb=True, warm_steps=3)
            elapsed_cl = max(1e-6, res_cl["wall_ms"] / 1000.0)
            sim_sec = E * n_steps * dt
            rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0)
            rows.append({
                "E": E,
                "N": players,
                "J": carts,
                "K": teams,
                "dt": dt,
                "mode": mode,
                "sim_wall_s": round(elapsed_sim, 4),
                "sim_speedup_x": round(sim_sec / elapsed_sim, 1),
                "agent_steps_per_sec": round((E * n_steps * players) / elapsed_sim, 1),
                "closed_loop_ms_step": round(res_cl["ms_per_step"], 3),
                "closed_loop_speedup_x": round(sim_sec / elapsed_cl, 1),
                "max_rss_mb": round(rss_mb, 1),
            })
        finally:
            roll_g.close()
            eng.close()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", default="1,16,128,1024,4096")
    ap.add_argument("--dt", type=float, default=0.4)
    ap.add_argument("--mode", default="analytic", choices=("analytic", "substep", "adaptive"))
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--players", type=int, default=32)
    ap.add_argument("--carts", type=int, default=12)
    ap.add_argument("--teams", type=int, default=4)
    args = ap.parse_args()

    print("=== Metal 4 Compiled DeltaTimePayloadEngine + PlanWally Benchmark (libmesh-graph.dylib) ===")
    conv = measure_dt_convergence()
    print("1. Variable-dt Rule Equivalence vs dt=0.05 (PLC_TICK):")
    for k, v in conv.items():
        print(f"   {k}: {v}")

    br = measure_savestate_grpo_branching()
    print(f"2. Savestate / GRPO Counterfactual Branching: {br}")

    env_counts = tuple(int(x) for x in args.envs.split(",") if x.strip())
    print(f"3. Throughput Sweep (dt={args.dt}s, mode={args.mode}, N={args.players}, J={args.carts}, K={args.teams}):")
    for row in measure_throughput_sweep(
        env_counts=env_counts,
        dt=args.dt,
        mode=args.mode,
        n_steps=args.steps,
        teams=args.teams,
        carts=args.carts,
        players=args.players,
    ):
        print(
            f"   E={row['E']:4d} | sim={row['sim_speedup_x']:10.1f}x RT ({row['agent_steps_per_sec']/1e6:6.2f}M agent-steps/s) "
            f"| closed-loop={row['closed_loop_speedup_x']:8.1f}x RT ({row['closed_loop_ms_step']:.3f} ms/step) "
            f"| rss={row['max_rss_mb']:.1f}MB"
        )

    world = build_gas_world_graph(seed=7, teams=args.teams, carts=args.carts, players=args.players, tiles=4, device=torch.device("cpu"))
    eng = MetalDeltaTimePayloadEngine(world, num_envs=8, num_players=args.players, base_seed=3)
    obs, carts_t, teams_t, geo = eng.observe_tensors()
    model = PlanWally().to(dtype=torch.bfloat16).eval()
    cpol = CompiledPlanWally(model, E=8, N=args.players, J=args.carts, K=args.teams)
    with torch.no_grad():
        out = cpol.forward(obs, carts_t, teams_t, geo=geo)
        smp = sample_plan(out)
    print(f"4. CompiledPlanWally Zero-Copy Forward OK: cont={tuple(smp.cont.shape)} cart_pick={tuple(smp.cart_pick.shape)} dtype={out.cont_mean.dtype}")
    cpol.close()
    eng.close()


if __name__ == "__main__":
    main()
