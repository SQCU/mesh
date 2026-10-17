"""The RDMA cross test (measurements/policy-feature-integration-20260906 rdma-cross.json, 09-06) on the library: every
rank of the placement's world runs this program; the row-Gram cross Rᵀ V is divided by the placement's `cross` parts
(scale_rpc.RemoteCross; [0, 1] is the 09-06 placement, the whole contraction on rank 1) and compared, on every rank,
with the same contraction computed whole and locally.

    RANK=r WORLD_SIZE=2 MASTER_ADDR=host0 MASTER_PORT=29500 \\
        PYTHONPATH=xonotic:rdma bin/mesh-python -B -m solver.strat.cross PLACEMENT.json [--capture FRAME.npz]

1. Rectangular: a 257 x 13 left and 257 x 19 right operand; the forward and both operand VJPs of the divided
   contraction against the local one (09-06: errors 0, 0, 0).
2. The policy: a native frame (the capture's ChorusArrays fields, else one built by inputs.native_inputs from seeded
   STATE, OBSERVATION, CART, TEAM and EVENT rows in HEAD's wire layout) through cast_header.Wally with the divided
   cross and with the local one: every parameter's gradient of one fixed functional of the outputs compared (09-06:
   the largest discrepancy 3.73e-9), a changed input's effect on the outputs reported (the inputs live through the
   call), and where the placement also divides the scale experts, their shards' gradients against the same columns of
   the whole model's.
The report is one JSON line on rank 0 (`cross_report`); a failure is its value, `passed` false."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "rdma"))
sys.path.insert(0, str(ROOT / "xonotic"))

from payload.tools.strategy_io_schema import CART_WIDTH, CS, EVT, EVT_WIDTH, OBS, OBS_WIDTH, TEAM_WIDTH, TS
from solver.strat.cast_header import Wally, parameter_seed
from solver.strat.inputs import ChorusArrays, native_inputs
from solver.strat.products import World
from solver.strat.scale_config import strategy_widths
from solver.strat.scale_rpc import RemoteCross, RemoteScale
from solver.strat.state_steering import PAGE_STATE_HEADER, PAGE_STATE_WIDTH, PAGE_WIDTH
from solver.strat.strategy import strategy


def native_rows(seed, bots=8, pages=3, events=20, carts=4, teams=2):
    """Seeded rows of the game's wire (payload/STRATEGY-IO.md): each bot's typed STATE pages, its OBSERVATION row,
    CART and TEAM rows and perceived EVENT rows."""
    g = np.random.default_rng(seed)
    obs = np.zeros((bots, OBS_WIDTH), np.float32)
    obs[:, OBS["ID"]] = np.arange(1, bots + 1)
    obs[:, OBS["TEAM"]] = 1 + np.arange(bots) % teams
    obs[:, OBS["PRESENT"]] = 1
    obs[:, OBS["ALIVE"]] = 1
    obs[:, OBS["HEALTH"]] = g.uniform(1, 200, bots)
    obs[:, OBS["POS_X"]:OBS["POS_Z"] + 1] = g.standard_normal((bots, 3)) * 512
    obs[:, OBS["VEL_X"]:OBS["VEL_Z"] + 1] = g.standard_normal((bots, 3)) * 64
    obs[:, OBS["ENGINE_TIME"]] = 30
    state = np.zeros((bots * pages, PAGE_STATE_WIDTH), np.float32)
    for bot in range(bots):
        for page in range(pages):
            row = state[bot * pages + page]
            row[:PAGE_STATE_HEADER] = [bot + 1, 1, 30, 0, -1 if page == 0 else 100 + bot, page * PAGE_WIDTH, 1, 1,
                                       0, 7, 29.5, 0.4, 1.0, 0, 0]
            present = g.random(PAGE_WIDTH) < 0.6
            words = slice(PAGE_STATE_HEADER, PAGE_STATE_HEADER + PAGE_WIDTH)
            row[words] = np.where(present, g.standard_normal(PAGE_WIDTH), 0)
            row[PAGE_STATE_HEADER + PAGE_WIDTH:PAGE_STATE_HEADER + 2 * PAGE_WIDTH] = np.where(present, 0.1 * g.standard_normal(PAGE_WIDTH), 0)
            row[PAGE_STATE_HEADER + 2 * PAGE_WIDTH:PAGE_STATE_HEADER + 3 * PAGE_WIDTH] = np.where(present, 2, 0)
            row[PAGE_STATE_HEADER + 3 * PAGE_WIDTH:PAGE_STATE_HEADER + 4 * PAGE_WIDTH] = np.where(present, 29.0, 0)
    cart = np.zeros((carts, CART_WIDTH), np.float32)
    cart[:, CS["ID"]] = np.arange(carts)
    cart[:, CS["PATH_LENGTH"]] = 1000
    cart[:, CS["PATH_POSITION"]] = g.uniform(0, 1000, carts)
    cart[:, CS["CONTROL_TEAM"]] = g.integers(0, teams + 1, carts)
    cart[:, CS["POS_X"]:CS["POS_Z"] + 1] = g.standard_normal((carts, 3)) * 512
    team = np.zeros((teams, TEAM_WIDTH), np.float32)
    team[:, TS["ID"]] = np.arange(1, teams + 1)
    context = np.zeros((events, EVT_WIDTH), np.float32)
    context[:, EVT["KIND"]] = g.integers(0, 4, events)
    context[:, EVT["TIME"]] = g.uniform(20, 30, events)
    context[:, EVT["TEAM"]] = g.integers(0, teams + 1, events)
    context[:, EVT["POS_X"]:EVT["POS_Z"] + 1] = g.standard_normal((events, 3)) * 512
    return state, obs, cart, team, context


def native_frame(seed, device, capture=None):
    if capture:
        with np.load(capture, allow_pickle=False) as source:
            frame = ChorusArrays(*(source[name] for name in ChorusArrays._fields))
    else:
        _, frame = native_inputs(*native_rows(seed), 0.4, 1.0, None)
    return ChorusArrays(*(torch.from_numpy(np.ascontiguousarray(value)).to(device) for value in frame))


def functional(output, seed):
    """One fixed scalar of the policy's outputs, its weights drawn from `seed` on the host (every rank the same)."""
    g = torch.Generator().manual_seed(seed)
    draw = lambda value: torch.randn(tuple(value.shape), generator=g).to(value.device)
    return ((output.rate.mean * draw(output.rate.mean)).sum() + (output.rate.log_scale * draw(output.rate.log_scale)).sum()
            + (output.value_winnie * draw(output.value_winnie)).sum() + (output.value_lou * draw(output.value_lou)).sum()
            + output.scale_balance)


def gradients(model, frame, seed):
    model.zero_grad(set_to_none=True)
    output = strategy(model, *frame)
    functional(output, seed).backward()
    return output, {name: (value.grad.detach().cpu().clone() if value.grad is not None else torch.zeros_like(value).cpu())
                    for name, value in model.named_parameters()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("placement")
    ap.add_argument("--capture")
    ap.add_argument("--seed", type=int, default=20260906)
    ap.add_argument("--scale-rank", type=int, default=128)
    ap.add_argument("--scale-hidden", type=int, default=341)
    ap.add_argument("--scale-experts", type=int, default=8)
    ap.add_argument("--scale-topk", type=int, default=2)
    args = ap.parse_args()
    placement = json.loads(Path(args.placement).read_text())
    device = placement["device"]
    world = World(device, placement.get("products"))
    report = {"world": world.report(), "device": device, "capture": args.capture, "checks": {}}
    try:
        g = np.random.default_rng(args.seed)
        left = torch.from_numpy(g.standard_normal((257, 13)).astype(np.float32)).to(device)
        right = torch.from_numpy(g.standard_normal((257, 19)).astype(np.float32)).to(device)
        cotangent = torch.from_numpy(g.standard_normal((13, 19)).astype(np.float32)).to(device)
        errors = []
        for divided in (False, True):
            a, b = left.clone().requires_grad_(), right.clone().requires_grad_()
            started = time.perf_counter()
            product = RemoteCross(world)(a, b) if divided else a.T @ b
            product.backward(cotangent)
            elapsed = time.perf_counter() - started
            errors.append((product.detach().cpu(), a.grad.cpu(), b.grad.cpu(), elapsed))
        rectangular = [float((divided - local).abs().max()) for divided, local in zip(errors[1][:3], errors[0][:3])]
        report["rectangular"] = {"left": [257, 13], "right": [257, 19], "errors_forward_left_vjp_right_vjp": rectangular,
                                 "local_s": errors[0][3], "divided_s": errors[1][3]}
        report["checks"]["rectangular_forward_and_both_vjps"] = max(rectangular) <= 1e-5

        frame = native_frame(args.seed, device, args.capture)
        widths = strategy_widths(PAGE_WIDTH, args.scale_rank, args.scale_hidden, args.scale_experts, args.scale_topk)
        seed = parameter_seed(args.seed, "matrix_fusion")
        torch.manual_seed(seed)
        local = Wally(widths).to(device)
        torch.manual_seed(seed)
        shard = world.span('scale', widths.scale_h) if world.parts('scale') else None
        divided = Wally(widths, scale_shard=shard).to(device)
        if world.parts('cross') is not None:
            divided.cross_executor = RemoteCross(world)
        if world.parts('scale') is not None:
            divided.scale_executor = RemoteScale(world)
        started = time.perf_counter()
        local_output, local_gradients = gradients(local, frame, args.seed)
        local_s = time.perf_counter() - started
        started = time.perf_counter()
        divided_output, divided_gradients = gradients(divided, frame, args.seed)
        divided_s = time.perf_counter() - started
        lo, hi = divided.scale_shard
        parameter_errors = {}
        for name, value in divided_gradients.items():
            reference = local_gradients[name]
            if name in divided.sharded:
                reference = reference[:, :, lo:hi] if name in ("scale_w1", "scale_w3") else reference[:, lo:hi]
            parameter_errors[name] = float((value - reference).abs().max()) if value.numel() else 0.0
        report["parameter_gradient_max_errors"] = parameter_errors
        report["largest_parameter_gradient_error"] = max(parameter_errors.values())
        report["forward_error"] = float((divided_output.rate.mean - local_output.rate.mean).abs().max())
        report["local_s"], report["divided_s"] = local_s, divided_s
        changed = ChorusArrays(*frame)._replace(state=frame.state * 1.01 + 0.01)
        with torch.no_grad():
            again = strategy(divided, *changed)
        report["changed_input_effect"] = float((again.rate.mean - divided_output.rate.mean).abs().max())
        report["checks"]["real_policy_parameter_vjps_match_local"] = report["largest_parameter_gradient_error"] <= 1e-6
        report["checks"]["prepared_real_frame_keeps_remote_inputs_live"] = report["changed_input_effect"] > 0
        report["calls"] = {name: remote.last for name, remote in
                           (("cross", getattr(divided, "cross_executor", None)), ("scale", getattr(divided, "scale_executor", None)))
                           if remote is not None}
        report["passed"] = all(report["checks"].values())
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        report["passed"] = False
    if world.rank == 0 or not report["passed"]:
        print(json.dumps({"event": "cross_report", **report}, default=float), flush=True)
    world.close()
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
