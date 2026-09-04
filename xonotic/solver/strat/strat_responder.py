import argparse, hashlib, json, os, signal, socket, sys, time
from collections import deque

import numpy as np
import mlx.core as mx

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "..", "rdma"))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))

from solver.strat.cast_header import Wally, parameter_seed
from solver.strat.matmul import gram_context
from solver.strat.baselines import BaselinePolicy, baseline_strategy, default_strategy
from solver.strat.featurize import SLOT_DIM
from solver.strat.inputs import assemble, player_features, scatter_gather_participants
from solver.strat.strategy import strategy, act, logp_of, sample_controls
from solver.xonwire import (
    GRAM_REQ, GRAM_RESP, GRAM_GRAD_REQ, GRAM_GRAD_RESP,
    GRAM_META, GRAM_META_KIND, GRAM_META_WIDTH,
    Reassembler, FrameStream, frame_count, parse_hdr,
    CART_KIND, EVENT_KIND, OBSERVATION_KIND, STRATEGY_KIND,
)
from solver.strat.instruments import (
    assignment_commitment,
    response_rows,
    update_weight_table,
)
from solver.strat.joracle.probe import (
    LiteralJReporter, literal_source_coordinates,
)
from solver.strat.live_belief import LiveBelief
from solver.strat.row_window import RowWindow
from solver.strat.policy_contract import MATRIX_FUSION_ARMS, MATRIX_FUSION_INTERVENTION_ARMS, architecture_arm, is_matrix_fusion_arm, checkpoint_path
from solver.strat.runtime import (
    build_runtime_frame,
    formal_projection_record,
    formal_value_record,
    role_rewards,
    reward_contract,
    reward_fingerprint,
)
from solver.strat.work_estimate import strategy_work
from solver.strat.scale_config import (
    SCALE_EXPERTS, SCALE_HIDDEN, SCALE_RANK, SCALE_TOPK,
    strategy_widths,
)
from workload import WorkloadMeter
from mesh import Mesh

from payload.tools.strategy_io_schema import (
    CART_WIDTH,
    EVT_WIDTH,
    OBS_WIDTH,
    RESP_WIDTH,
    STRATEGY_DEADLINE_S,
    OBS,
    CS,
    EVT,
    SC,
    OBS_AMMO_COLUMNS,
    OBS_WEAPON_COLUMNS,
    CONTROL_WIDTH,
    OBS_OUTCOME_COUNTER_COLUMNS,
    OBS_ROUTED_OUTCOME_GROUPS,
    XAN_WIDTH,
    TARGET_KIND,
    decode_target,
)

OBS_W, CART_W, EVT_W, RESP_W = OBS_WIDTH, CART_WIDTH, EVT_WIDTH, RESP_WIDTH
MODEL_OBS_W = XAN_WIDTH
CKPT = os.path.join(_HERE, "runs", "policy_ckpt_v6.npz")
ONLINE_CKPT = os.path.join(_HERE, "runs", "policy_online_v6.npz")
TELEM = os.path.join(_HERE, "runs", "cartserver_telemetry.jsonl")
def team_resources(rows, teams):
    out = []
    for team in range(1, teams + 1):
        selected = rows[np.asarray(rows[:, OBS["TEAM"]], dtype=np.int64) == team]
        alive = selected[:, OBS["ALIVE"]] >= 0.5 if len(selected) else np.zeros(0, dtype=bool)
        speeds = np.linalg.norm(selected[:, OBS["VEL_X"]:OBS["VEL_Z"] + 1], axis=1) if len(selected) else np.zeros(0)
        out.append({
            "team": team,
            "players": len(selected),
            "alive": int(alive.sum()),
            "health": float(selected[:, OBS["HEALTH"]].sum()),
            "armor": float(selected[:, OBS["ARMOR"]].sum()),
            "ammo": {
                name.removeprefix("AMMO_").lower(): float(selected[:, OBS[name]].sum())
                for name in OBS_AMMO_COLUMNS
            },
            "weapon_words": {
                name.removeprefix("WEAPONS_").lower(): int(np.bitwise_or.reduce(
                    selected[:, OBS[name]].astype(np.int64), initial=0,
                ))
                for name in OBS_WEAPON_COLUMNS
            },
            "mean_speed": float(speeds.mean()) if len(speeds) else 0.0,
        })
    return out

def load_policy(module, checkpoint, policy_arm):
    from solver.strat.checkpoint_state import load_module_checkpoint

    measurement = load_module_checkpoint(
        module, checkpoint, policy_arm, reward_fingerprint(policy_arm),
    )
    print(json.dumps({"event": "checkpoint_measurement", **measurement}), flush=True)
    return module

def policy_source(arm, model, checkpoint, mode):
    from solver.strat.online import POLICY_VERSIONS

    version_arm = architecture_arm(arm)
    return {
        "arm": arm,
        "mode": mode,
        "checkpoint": checkpoint,
        "checkpoint_bytes": os.path.getsize(checkpoint) if checkpoint and os.path.exists(checkpoint) else 0,
        "checkpoint_sha256": checkpoint_sha256(checkpoint),
        "source_weight_mass": 0 if model is None else int(getattr(model, "checkpoint_source_weight_mass", 0)),
        "live_weight_mass": 0 if model is None else int(getattr(model, "checkpoint_live_weight_mass", 0)),
        "loaded_weight_mass": 0 if model is None else int(getattr(model, "checkpoint_loaded_weight_mass", 0)),
        "composable_weight_mass": 0 if model is None else int(getattr(model, "checkpoint_composable_weight_mass", 0)),
        "source_only_weight_mass": 0 if model is None else int(getattr(model, "checkpoint_source_only_weight_mass", 0)),
        "live_only_weight_mass": 0 if model is None else int(getattr(model, "checkpoint_live_only_weight_mass", 0)),
        "shape_difference_mass": 0 if model is None else int(getattr(model, "checkpoint_shape_difference_mass", 0)),
        "nonfinite_weight_mass": 0 if model is None else int(getattr(model, "checkpoint_nonfinite_weight_mass", 0)),
        "load_exception": None if model is None else getattr(model, "checkpoint_load_exception", None),
        "updates": None if arm == "default" else getattr(model, "checkpoint_updates", None),
        "source_arm": None if model is None else getattr(model, "checkpoint_source_arm", None),
        "live_arm": arm,
        "source_version": None if model is None else getattr(model, "checkpoint_source_version", None),
        "live_version": None if arm == "default" else POLICY_VERSIONS[version_arm],
        "source_architecture": None if model is None else getattr(model, "checkpoint_source_architecture", None),
        "live_architecture": None if model is None else getattr(model, "checkpoint_live_architecture", None),
        "source_reward_contract": None if model is None else getattr(model, "checkpoint_source_reward_contract", None),
        "lineage_initial_sha256": None if model is None else getattr(model, "checkpoint_lineage_initial_sha256", None),
        "live_reward_contract": "fixed_default" if arm == "default" else reward_fingerprint(arm),
        "parameter_seed": None if model is None else getattr(model, "parameter_seed", None),
    }

def checkpoint_sha256(path):
    if not path or not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def array(value, dtype=None):
    out = np.asarray(value)
    return out.astype(dtype) if dtype is not None else out

def finite_range(value):
    values = np.asarray(value)
    finite = values[np.isfinite(values)]
    return None if finite.size == 0 else [
        round(float(finite.min()), 6), round(float(finite.max()), 6),
    ]

def gather_policy_rows(outputs, row_arms, actions, instrument_mass):
    fields = {
        "coupling": "coupling",
        "score": "logits",
        "dw_dt": "dw_dt",
        "dynamics_guidance": "guidance",
        "model_uncertainty": "uncertainty",
        "actuator": "controls",
        "actuator_log_scale": "control_log_scale",
        "actuator_density": "control_density",
        "winner_value": "value_winnie",
        "loser_value": "value_lou",
        "aux_winner_value": "aux_winnie",
        "aux_loser_value": "aux_lou",
    }
    first = next(iter(outputs.values()))
    gathered = {
        name: np.empty_like(array(getattr(first, source), np.float32))
        for name, source in fields.items()
    }
    gathered["diag_k"] = np.empty(
        (len(row_arms), instrument_mass), dtype=np.float32,
    )
    for arm in dict.fromkeys(row_arms.tolist()):
        selected = row_arms == arm
        source = outputs[str(arm)]
        for name, attribute in fields.items():
            gathered[name][selected] = array(
                getattr(source, attribute), np.float32,
            )[selected]
        gathered["diag_k"][selected] = np.broadcast_to(
            array(source.dee, np.float32), (len(row_arms), instrument_mass),
        )[selected]
    return gathered

def gather_policy_row_atoms(outputs, row_arms, actions):
    atoms = [None] * len(row_arms)
    for arm in dict.fromkeys(row_arms.tolist()):
        source = outputs[str(arm)]
        relation = array(source.ir, np.float32)
        belief = array(source.belief, np.float32)
        pooled = array(source.pooled, np.float32)
        for index in np.flatnonzero(row_arms == arm):
            j_row = relation[index, actions[index]].copy()
            atoms[index] = {
                "arm": str(arm),
                "j": j_row,
                "j_labels": tuple(
                    f"{arm}.j.{coordinate}" for coordinate in range(len(j_row))
                ),
                "beta": belief[index].copy(),
                "pooled": pooled[index].copy(),
            }
    return atoms

def tensor_measures(values):
    summaries = {
        name: mx.stack((
            mx.sum(flat), mx.sum(mx.square(flat)), mx.min(flat), mx.max(flat),
        ))
        for name, value in values.items()
        for flat in (value.reshape(-1),)
    }
    mx.eval(*summaries.values())
    out = {}
    for name, summary in summaries.items():
        integral, square_integral, minimum, maximum = np.asarray(summary, dtype=np.float64)
        mass = int(np.prod(values[name].shape))
        mean = float(integral / mass)
        out[name] = {
            "mass": mass,
            "integral": float(integral),
            "square_integral": float(square_integral),
            "mean": mean,
            "variance": float(max(0.0, square_integral / mass - mean * mean)),
            "minimum": float(minimum),
            "maximum": float(maximum),
        }
    return out

def matrix_fusion_intervention_measure(outputs):
    matrix_fusion = outputs.get("matrix_fusion")
    participant_ablated = outputs.get("participant_fusion_ablated")
    residual_ablated = outputs.get("residual_fusion_ablated")
    if matrix_fusion is None or participant_ablated is None or residual_ablated is None:
        return None
    values = {}
    for name, ablated in (
        ("participant", participant_ablated),
        ("residual", residual_ablated),
    ):
        probability = mx.softmax(matrix_fusion.logits, axis=-1)
        ablated_probability = mx.softmax(ablated.logits, axis=-1)
        delta = matrix_fusion.ir - ablated.ir
        values.update({
            f"{name}_ir_coordinate_delta": delta,
            f"{name}_ir_relation_distance": mx.sqrt(mx.sum(mx.square(delta), axis=-1)),
            f"{name}_action_probability_total_variation": 0.5 * mx.sum(
                mx.abs(probability - ablated_probability), axis=-1,
            ),
            f"{name}_actuator_mean_coordinate_delta": matrix_fusion.controls - ablated.controls,
            f"{name}_actuator_log_scale_coordinate_delta": matrix_fusion.control_log_scale - ablated.control_log_scale,
            f"{name}_winner_value_delta": matrix_fusion.value_winnie - ablated.value_winnie,
            f"{name}_loser_value_delta": matrix_fusion.value_lou - ablated.value_lou,
            f"{name}_instrument_weight_velocity_delta": matrix_fusion.dw_dt - ablated.dw_dt,
        })
    values["participant_gram_coordinate_delta"] = matrix_fusion.coupling - participant_ablated.coupling
    return {
        "left": "matrix_fusion",
        "participant_right": "participant_fusion_ablated",
        "residual_right": "residual_fusion_ablated",
        "participant_mass": int(matrix_fusion.ir.shape[0]),
        "instrument_mass": int(matrix_fusion.ir.shape[1]),
        "j_width": int(matrix_fusion.ir.shape[2]),
        "measures": tensor_measures(values),
    }

def distributed_scale_summary(calls, counterfactual_interval=0):
    if not calls:
        return {
            "call_mass": 0, "request_row_mass": 0, "output_row_mass": 0,
            "request_frame_mass": 0, "response_frame_mass": 0,
            "counterfactual_interval_s": float(counterfactual_interval),
            "counterfactual_sample_mass": 0, "calls": [],
        }
    out = dict(calls[-1])
    out["calls"] = calls
    out["call_mass"] = len(calls)
    out["request_row_mass"] = sum(int(call.get("request_row_mass") or 0) for call in calls)
    out["output_row_mass"] = sum(int(call.get("output_row_mass") or 0) for call in calls)
    out["request_frame_mass"] = sum(int(call.get("request_frame_mass") or 0) for call in calls)
    out["response_frame_mass"] = sum(int(call.get("response_frame_mass") or 0) for call in calls)
    out["counterfactual_interval_s"] = float(counterfactual_interval)
    out["counterfactual_sample_mass"] = sum(
        int((call.get("local_counterfactual") or {}).get("observed") is True)
        for call in calls
    )
    return out

def make_policy(arm, widths, hidden):
    if arm in ("matrix_fusion", "terminal_win", "initial_policy"):
        return Wally(widths), strategy
    if arm in ("participant_fusion_ablated", "residual_fusion_ablated"):
        model = Wally(widths)
        return model, participant_fusion_ablated_strategy if arm == "participant_fusion_ablated" else residual_fusion_ablated_strategy
    if arm == "default":
        return None, default_strategy
    return BaselinePolicy(arm, MODEL_OBS_W + widths.d_sem + SLOT_DIM, hidden), baseline_strategy

def participant_fusion_ablated_strategy(model, *args):
    return strategy(
        model, *args,
        participant_fusion_scale=0.0,
        residual_fusion_scale=1.0,
        execute_remote_scale=True,
    )

def residual_fusion_ablated_strategy(model, *args):
    return strategy(
        model, *args,
        participant_fusion_scale=1.0,
        residual_fusion_scale=0.0,
        execute_remote_scale=True,
    )

class RemoteGram:
    def __init__(self, mesh, tx, node, widths, backlog, counterfactual_interval, stopping=None):
        self.mesh, self.tx, self.node = mesh, tx, node
        self.backlog = backlog
        self.stopping = stopping if stopping is not None else {"signal": None}
        self.counterfactual_interval = float(counterfactual_interval)
        self.next_counterfactual = 0.0
        self.last_counterfactual = self.pending = None
        self.sample = False
        self.sequence = 0
        self.receivers = {
            kind: Reassembler(kind, widths.d_scale, mesh.usable)
            for kind in (GRAM_RESP, GRAM_GRAD_RESP)
        }
        self.receivers[GRAM_META_KIND] = Reassembler(GRAM_META_KIND, GRAM_META_WIDTH, mesh.usable)
        self.last = {"request_row_mass": 0, "output_row_mass": 0}

    def request(self, kind, packed, response_kind, source_rows, response_rows):
        self.sequence += 1
        started = time.perf_counter()
        response, measures = self.tx.exchange(
            kind, self.sequence, int(self.sample), packed, self.node,
            {key: self.receivers[key] for key in (response_kind, GRAM_META_KIND)},
            cancel=lambda: self.stopping["signal"] is not None,
            backlog=self.backlog, retry_s=STRATEGY_DEADLINE_S,
        )
        elapsed = time.perf_counter() - started
        self.last = {
            **measures, "operation": "gram_context" if kind == GRAM_REQ else "gram_vjp",
            "sequence": self.sequence,
            "request_row_mass": source_rows, "request_tensor_rows": len(packed),
            "output_row_mass": 0, "processed_row_mass": 0, "completed": False,
            "request_frame_mass": frame_count(packed, self.mesh.usable),
            "attempt_elapsed_s": elapsed, "roundtrip_s": elapsed,
            "transport_queued_frames": int(self.mesh.queued()),
            "transport_inflight_frames": int(self.mesh.inflight()),
            "deadline_s": STRATEGY_DEADLINE_S if self.sample else None,
            "deadline_slack_s": STRATEGY_DEADLINE_S - elapsed if self.sample else None,
            "cancelled": self.stopping["signal"] is not None,
        }
        if response is not None:
            record, output = response[response_kind]
            meta_record, metadata = response[GRAM_META_KIND]
            valid = len(output) == response_rows and int(metadata[0, GRAM_META["ROWS"]]) == source_rows
            self.last.update(
                output_row_mass=len(output), completed=valid,
                processed_row_mass=source_rows if valid else 0,
                response_frame_mass=record.get("frame_mass"),
                metadata_frame_mass=meta_record.get("frame_mass"),
                worker_rows=int(metadata[0, GRAM_META["ROWS"]]),
                worker_compute_s=float(metadata[0, GRAM_META["ELAPSED"]]),
            )
            if valid:
                return output, metadata[0, :3]
        self.last["local_fallback"] = True
        return None

    def __call__(self, rows, probe):
        @mx.custom_function
        def operation(source, vector):
            packed = np.concatenate((np.asarray(vector)[None, :], np.asarray(source)), axis=0)
            response = self.request(GRAM_REQ, packed, GRAM_RESP, len(source), 1)
            if response is None:
                return gram_context(source, vector)
            output, statistics = response
            if self.sample and self.pending is None and time.monotonic() >= self.next_counterfactual:
                self.pending = (packed.copy(), output[0].copy(), dict(self.last))
                self.next_counterfactual = time.monotonic() + self.counterfactual_interval
            return mx.array(output[0]), mx.array(statistics)

        @operation.vjp
        def operation_vjp(primals, cotangents, output):
            source, vector = primals
            cotangent = cotangents[0]
            packed = np.concatenate((
                np.asarray(vector)[None, :], np.asarray(cotangent)[None, :],
                np.asarray(source),
            ), axis=0)
            response = self.request(GRAM_GRAD_REQ, packed, GRAM_GRAD_RESP, len(source), len(source) + 1)
            if response is None:
                return mx.vjp(gram_context, primals, cotangents)[1]
            gradients, _ = response
            return mx.array(gradients[1:]), mx.array(gradients[0])

        return operation(rows, probe)

    def measure_local_counterfactual(self, deadline, distributed_elapsed, calls):
        pending, self.pending = self.pending, None
        if pending is None:
            return
        packed, remote, remote_call = pending
        source, probe = mx.array(packed[1:]), mx.array(packed[0])
        mx.eval(source, probe)
        started = time.perf_counter()
        context, statistics = gram_context(source, probe)
        mx.eval(context, statistics)
        elapsed = time.perf_counter() - started
        local = np.asarray(context, dtype=np.float32)
        difference = np.abs(local - remote)
        maximum = float(np.max(difference)) if difference.size else 0.0
        scale = float(np.max(np.abs(remote))) if remote.size else 0.0
        remote_elapsed = sum(float(call.get("attempt_elapsed_s") or 0.0) for call in calls)
        successful_calls = sum(
            int(bool(call.get("completed")))
            for call in calls
        )
        fallback_calls = len(calls) - successful_calls
        residual_elapsed = max(0.0, float(distributed_elapsed) - remote_elapsed)
        non_scale_estimate = max(0.0, residual_elapsed - elapsed * fallback_calls)
        local_plan_elapsed = elapsed * len(calls) + non_scale_estimate
        local_plan_lower = (
            local_plan_elapsed if fallback_calls == 0 else elapsed * len(calls)
        )
        local_plan_upper = (
            local_plan_elapsed if fallback_calls == 0
            else elapsed * len(calls) + residual_elapsed
        )
        counterfactual = {
            "observed": True,
            "operation": "gram_context", "same_input_snapshot": True,
            "sampled_at": time.time(),
            "sequence": remote_call.get("sequence"),
            "elapsed_s": elapsed,
            "deadline_s": float(deadline),
            "deadline_slack_s": float(deadline) - elapsed,
            "distributed_plan_elapsed_s": float(distributed_elapsed),
            "distributed_plan_slack_s": float(deadline) - float(distributed_elapsed),
            "local_plan_counterfactual_s": local_plan_elapsed,
            "local_plan_counterfactual_slack_s": float(deadline) - local_plan_elapsed,
            "local_plan_counterfactual_lower_s": local_plan_lower,
            "local_plan_counterfactual_upper_s": local_plan_upper,
            "local_plan_counterfactual_lower_slack_s": float(deadline) - local_plan_upper,
            "local_plan_counterfactual_upper_slack_s": float(deadline) - local_plan_lower,
            "remote_roundtrip_s": remote_call.get("roundtrip_s"),
            "remote_call_count": len(calls),
            "remote_success_call_mass": successful_calls,
            "local_fallback_call_mass": fallback_calls,
            "remote_attempt_total_s": remote_elapsed,
            "output_max_abs_error": maximum,
            "output_relative_error": maximum / scale if scale else None,
            "output_finite_fraction": float(np.isfinite(local).sum() / local.size),
        }
        self.last_counterfactual = dict(counterfactual)
        for index, call in enumerate(calls):
            if call.get("sequence") == remote_call.get("sequence"):
                calls[index] = {**call, "local_counterfactual": counterfactual}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peer-node", type=int, default=0)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--train-arms")
    ap.add_argument("--match-metadata", default="{}")
    ap.add_argument("--policy-arm", default="matrix_fusion")
    ap.add_argument("--baseline-hidden", type=int, default=256)
    ap.add_argument("--scale-rank", type=int, default=SCALE_RANK)
    ap.add_argument("--scale-hidden", type=int, default=SCALE_HIDDEN)
    ap.add_argument("--scale-experts", type=int, default=SCALE_EXPERTS)
    ap.add_argument("--scale-topk", type=int, default=SCALE_TOPK)
    ap.add_argument("--distributed-scale", action="store_true")
    ap.add_argument("--team-policy-arms")
    ap.add_argument("--arm-checkpoint", action="append", default=[])
    ap.add_argument("--off-policy-players", type=int, default=0)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--gradient-clip", type=float, default=1.0)
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--save-secs", type=float, default=30.0)
    ap.add_argument("--checkpoint")
    ap.add_argument("--online-checkpoint", default=ONLINE_CKPT)
    ap.add_argument("--initial-checkpoint")
    ap.add_argument("--resume-checkpoint")
    ap.add_argument("--telemetry", default=TELEM)
    ap.add_argument("--environment", default="game2_server")
    ap.add_argument("--navigation-realization")
    ap.add_argument("--append-telemetry", action="store_true")
    ap.add_argument("--model-sample-every", type=int, default=50)
    ap.add_argument("--measure-rows", type=int, default=4000)
    ap.add_argument("--measure-interval", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=20260829)
    ap.add_argument("--control-weight", type=float, default=0.5)
    ap.add_argument("--exploration-weight", type=float, default=0.05)
    ap.add_argument("--replay-capacity", type=int, default=1024,
                    help="optional transition-count bound; zero leaves retention to the memory budget")
    ap.add_argument("--replay-precision", default="float32",
                    help="storage precision label recorded with the replay run")
    ap.add_argument("--replay-memory-mb", type=float, default=256.0,
                    help="byte budget for each retained-history and current-episode pool")
    ap.add_argument("--replay-batch", type=int, default=8,
                    help="transitions per replayed gradient step")
    ap.add_argument("--replay-weight", type=float, default=0.5)
    ap.add_argument("--replay-steps", type=int, default=4,
                    help="replayed gradient steps taken after each fresh segment")
    args = ap.parse_args()
    train_arms = tuple(filter(None, (args.train_arms or args.policy_arm).split(","))) if args.train else ()
    team_policy_arms = tuple(
        arm for arm in (value.strip() for value in (args.team_policy_arms or "").split(","))
        if arm
    )
    arm_checkpoints = {}
    for value in args.arm_checkpoint:
        arm, separator, path = value.partition("=")
        arm_checkpoints[arm if separator else args.policy_arm] = path if separator else value
    active_policy = "mixed" if team_policy_arms else args.policy_arm

    if args.policy_arm != "matrix_fusion":
        if args.online_checkpoint == ONLINE_CKPT:
            args.online_checkpoint = ONLINE_CKPT.replace(".npz", f"_{args.policy_arm}.npz")

    runstate_path = (args.online_checkpoint or CKPT) + ".runstate.json"

    def save_runstate(rng, model_key, resp_id, nt_written, updates):
        try:
            tmp = runstate_path + ".new"
            with open(tmp, "w") as fh:
                json.dump({
                    "rng": rng.bit_generator.state,
                    "model_key": np.asarray(model_key, dtype=np.uint32).tolist(),
                    "resp_id": int(resp_id),
                    "nt_written": int(nt_written),
                    "updates": int(updates),
                    "environment": args.environment,
                    "wrote_at": time.time(),
                }, fh)
            os.replace(tmp, runstate_path)
        except Exception as exc:
            print(f"[responder] runstate save failed: {exc}", flush=True)

    widths = strategy_widths(args.scale_rank, args.scale_hidden,
                             args.scale_experts, args.scale_topk)
    work_meter = WorkloadMeter(
        "xonotic.strategy",
        {"environment": args.environment, "policy_arm": active_policy,
         "host_role": "responder", "host": socket.gethostname()},
    )
    j_reporter = LiteralJReporter(
        args.measure_rows, args.measure_interval,
        os.path.join(os.path.dirname(args.telemetry), f"j-measures.{os.getpid()}"),
    ).start()
    learning_measures = {}
    published_measure_revision = -1
    latest_policy_updates = {}
    outcome_measure = {"scope": "responder_session", "attribution": "assigned_team_arm_not_causal_policy_credit", "rounds": 0, "draws": 0, "wins": {}, "team_rounds": {}}
    mx.random.seed(parameter_seed(args.seed, architecture_arm(args.policy_arm)))
    wally, policy_forward = (None, None) if team_policy_arms else make_policy(args.policy_arm, widths, args.baseline_hidden)
    if wally is not None:
        wally.parameter_seed = parameter_seed(
            args.seed, architecture_arm(args.policy_arm),
        )
    policy_models = {}
    policy_forwards = {}
    policy_provenance = {}
    if team_policy_arms:
        assigned_arms = tuple(dict.fromkeys(team_policy_arms))
        matrix_fusion_checkpoint = arm_checkpoints.get("matrix_fusion")
        intervention_arms = MATRIX_FUSION_INTERVENTION_ARMS
        intervention_checkpoints = [
            arm_checkpoints.get(arm) for arm in intervention_arms
            if arm_checkpoints.get(arm)
        ]
        measures_matrix_fusion = any(
            arm in intervention_arms and arm != "matrix_fusion" for arm in assigned_arms
        )
        shared_matrix_fusion_checkpoint = matrix_fusion_checkpoint or next(
            iter(intervention_checkpoints), None,
        )
        evaluation_arms = list(assigned_arms)
        if measures_matrix_fusion:
            evaluation_arms.extend(
                arm for arm in intervention_arms if arm not in evaluation_arms
            )
        shared_matrix_fusion = None
        for arm in evaluation_arms:
            checkpoint = (
                shared_matrix_fusion_checkpoint if measures_matrix_fusion and arm in intervention_arms
                else arm_checkpoints.get(arm) or (matrix_fusion_checkpoint if arm in intervention_arms else None)
            )
            seed_namespace = architecture_arm(arm)
            seed_value = parameter_seed(args.seed, seed_namespace)
            if measures_matrix_fusion and arm in intervention_arms:
                if shared_matrix_fusion is None:
                    mx.random.seed(seed_value)
                    shared_matrix_fusion = Wally(widths)
                    shared_matrix_fusion.parameter_seed = seed_value
                    if checkpoint:
                        shared_matrix_fusion = load_policy(
                            shared_matrix_fusion, checkpoint, "matrix_fusion",
                        )
                model = shared_matrix_fusion
                forward = (
                    strategy if arm == "matrix_fusion"
                    else participant_fusion_ablated_strategy
                    if arm == "participant_fusion_ablated"
                    else residual_fusion_ablated_strategy
                )
            else:
                mx.random.seed(seed_value)
                model, forward = make_policy(arm, widths, args.baseline_hidden)
                if model is not None:
                    model.parameter_seed = seed_value
                if model is not None and checkpoint and arm not in train_arms:
                    model = load_policy(
                        model, checkpoint, arm,
                    )
            policy_models[arm] = model
            policy_forwards[arm] = forward
            policy_provenance[arm] = policy_source(arm, model, checkpoint, "fixed_default" if arm == "default" else "heldout_model")
        if args.train:
            wally = policy_models.get(args.policy_arm)
            policy_forward = policy_forwards.get(args.policy_arm)
    learners = {}
    for arm in train_arms:
        from solver.strat.online import OnlineLearner

        target = checkpoint_path(args.online_checkpoint, arm) if len(train_arms) > 1 else args.online_checkpoint
        learner_source = arm_checkpoints.get(arm) or (args.resume_checkpoint if len(train_arms) == 1 else None) or target
        learner = OnlineLearner(
            policy_models[arm] if team_policy_arms else wally,
            learning_rate=args.learning_rate,
            gradient_clip=args.gradient_clip,
            checkpoint=target,
            load_checkpoint=learner_source,
            replay_capacity=args.replay_capacity,
            replay_memory_mb=args.replay_memory_mb,
            replay_precision=args.replay_precision,
            replay_batch=args.replay_batch,
            replay_steps=args.replay_steps,
            seed=args.seed,
            policy_forward=policy_forwards[arm] if team_policy_arms else policy_forward,
            policy_arm=arm,
            match_metadata=json.loads(args.match_metadata),
            replay_weight=args.replay_weight,
        )
        learners[arm] = learner
        if args.initial_checkpoint:
            initial_path = checkpoint_path(args.initial_checkpoint, arm) if len(train_arms) > 1 else args.initial_checkpoint
            learner.save(initial_path)
            if not learner.initial_checkpoint_sha256:
                learner.initial_checkpoint_sha256 = checkpoint_sha256(initial_path)
        if team_policy_arms:
            policy_models[arm] = learner.wally
            policy_provenance[arm] = policy_source(
                arm, learner.wally, learner_source, "online_train",
            )
    learner = next(iter(learners.values()), None)
    if not learners and wally is not None and not team_policy_arms:
        wally = load_policy(wally, args.resume_checkpoint or args.checkpoint, args.policy_arm)
    if not team_policy_arms:
        checkpoint = args.resume_checkpoint or (args.online_checkpoint if learner is not None else args.checkpoint)
        policy_provenance[args.policy_arm] = policy_source(
            args.policy_arm,
            learner.wally if learner is not None else wally,
            checkpoint,
            "online_train" if learner is not None else "inference",
        )
    model_key = mx.random.key(args.seed)

    stopping = {"signal": None}
    m = Mesh()
    print(f"[responder] mesh attached region={m.region} slots={m.slots} usable={m.usable}", flush=True)
    ra_obs = Reassembler(OBSERVATION_KIND, OBS_W, m.usable)
    ra_cart = Reassembler(CART_KIND, CART_W, m.usable)
    ra_evt = Reassembler(EVENT_KIND, EVT_W, m.usable)
    tx = FrameStream(m)
    backlog = deque()
    remote_scale = None
    if args.distributed_scale:
        remote_scale = RemoteGram(m, tx, args.peer_node, widths, backlog,
                                  args.measure_interval, stopping=stopping)
        if is_matrix_fusion_arm(args.policy_arm) and wally is not None:
            wally.gram_executor = remote_scale
        if team_policy_arms:
            for arm in MATRIX_FUSION_ARMS:
                if policy_models.get(arm) is not None:
                    policy_models[arm].gram_executor = remote_scale

    def incoming():
        while backlog:
            yield backlog.popleft()
        yield from m.read(np.uint8)

    live_belief = LiveBelief(navigation=args.navigation_realization)
    rng = np.random.default_rng(args.seed)
    last_saved_update = 0
    previous = None
    weight_table = {}
    weight_context = None
    resp_id = 0
    pending_obs = {}
    pending_cart = {}
    pending_evt = {}
    belief_depths = None
    belief_episode = 0
    observed_outcomes = set()
    decision_history = RowWindow(args.measure_rows, len)
    nt_written = 0
    if args.append_telemetry and os.path.exists(runstate_path):
        try:
            with open(runstate_path) as fh:
                rs = json.load(fh)
            rng.bit_generator.state = rs["rng"]
            model_key = mx.array(rs["model_key"], dtype=mx.uint32)
            resp_id = int(rs.get("resp_id", 0))
            nt_written = int(rs.get("nt_written", 0))
            print(f"[responder] resumed runstate: resp_id={resp_id} nt_written={nt_written} "
                  f"updates={rs.get('updates')}", flush=True)
        except Exception as exc:
            print(f"[responder] runstate restore failed: {exc}; fresh RNG/cursor", flush=True)
    os.makedirs(os.path.dirname(args.telemetry) or ".", exist_ok=True)
    telem = open(args.telemetry, "a" if args.append_telemetry else "w")
    stats = dict(slots=0, obs=0, cart=0, evt=0, resp=0, updates=0)
    cgt_measure_integrals = {}
    cgt_nimbers = {}
    t0 = time.time()
    last_report = t0
    last_save_time = t0

    def _stop(signum, _frame):
        stopping["signal"] = int(signum)
        print(f"[responder] signal {signum}: draining and checkpointing", flush=True)

    signal_registration_errors = []
    for _sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(_sig, _stop)
        except (ValueError, OSError) as exc:
            signal_registration_errors.append({
                "signal": int(_sig), "error": f"{type(exc).__name__}: {exc}",
            })

    print("[responder] lifetime: service — runs until SIGINT/SIGTERM/SIGHUP", flush=True)
    if signal_registration_errors:
        print(f"[responder] signal registration errors={signal_registration_errors}", flush=True)

    try:
        while stopping["signal"] is None:
            got_any = False
            for buf, src in incoming():
                got_any = True
                stats["slots"] += 1
                header = parse_hdr(buf)
                if header is None:
                    continue
                kind = header["kind"]
                if kind == OBSERVATION_KIND:
                    record = ra_obs.feed(buf)
                    if record:
                        pending_obs[int(record["tick"])] = record, ra_obs.stage[:record["rows"]].copy()
                        stats["obs"] += 1
                elif kind == CART_KIND:
                    record = ra_cart.feed(buf)
                    if record:
                        pending_cart[int(record["tick"])] = record, ra_cart.stage[:record["rows"]].copy()
                        stats["cart"] += 1
                elif kind == EVENT_KIND:
                    record = ra_evt.feed(buf)
                    if record:
                        pending_evt[int(record["tick"])] = record, ra_evt.stage[:record["rows"]].copy()
                        stats["evt"] += 1

            ready_ticks = pending_obs.keys() & pending_cart.keys() & pending_evt.keys()
            if ready_ticks:
                ready_tick = max(ready_ticks)
                coalesced_observations = sum(tick < ready_tick for tick in pending_obs)
                last_obs = pending_obs.pop(ready_tick)
                last_cart = pending_cart.pop(ready_tick)
                event_ticks = sorted(tick for tick in pending_evt if tick <= ready_tick)
                last_evt = pending_evt[ready_tick][0], np.concatenate([pending_evt.pop(tick)[1] for tick in event_ticks])
                pending_obs = {tick: value for tick, value in pending_obs.items() if tick > ready_tick}
                pending_cart = {tick: value for tick, value in pending_cart.items() if tick > ready_tick}
                ch, cart_rows = last_cart
                oh, obs_rows = last_obs
                all_teams = array(obs_rows[:, OBS["TEAM"]], int)
                active = np.flatnonzero(all_teams >= 1)
                if len(active):
                    work_started = time.perf_counter()
                    rows = obs_rows[active]
                    participant_ids = array(rows[:, OBS["ID"]], int)
                    teams_present = all_teams[active]
                    team_by_id = dict(zip(participant_ids.tolist(), teams_present.tolist()))
                    cart_ctrl = array(cart_rows[:, CS["CONTROL_TEAM"]], int)
                    cart_home = array(cart_rows[:, CS["HOME_TEAM"]], int)
                    configured_teams = array(cart_rows[:, CS["TEAM_COUNT"]], int)
                    k = max(
                        teams_present.tolist()
                        + cart_ctrl[cart_ctrl >= 1].tolist()
                        + cart_home[cart_home >= 1].tolist()
                        + configured_teams[configured_teams >= 1].tolist()
                    )
                    j = cart_rows.shape[0]
                    l = len(active)
                    response_span = work_meter.span(
                        "policy_response", rows=l,
                        operations={
                            "players": l, "teams": k, "carts": j,
                            "host_role": "responder",
                        },
                    ).start()
                    map_key = tuple(
                        (int(cart_rows[c, CS["ID"]]),
                         float(cart_rows[c, CS["PATH_LENGTH"]]))
                        for c in range(j)
                    )
                    positions = np.asarray(cart_rows[:, CS["PATH_POSITION"]], dtype=np.float32)
                    lengths = np.asarray(cart_rows[:, CS["PATH_LENGTH"]], dtype=np.float32)
                    depths = np.divide(positions, lengths, out=np.zeros_like(positions), where=lengths != 0)
                    if (belief_depths is not None and np.max(belief_depths, initial=0) > 0.5
                            and np.max(depths, initial=0) < 0.02):
                        belief_episode += 1
                    belief_depths = depths.copy()
                    episode_context = (map_key, belief_episode)
                    belief_reset = live_belief.sync(episode_context, max(int(ch["tick"]), int(oh["tick"])))
                    if belief_reset:
                        decision_history.clear()
                        previous = None
                        j_reporter.reset()
                    if weight_context != episode_context:
                        weight_table = {}
                        weight_context = episode_context
                    deposited = 0
                    event_tick = None
                    realized_events = []
                    eh, event_rows = last_evt
                    event_tick = int(eh["tick"])
                    deposited = live_belief.ingest(event_rows, EVT)
                    for event in event_rows:
                        kind = int(event[EVT["KIND"]])
                        if kind < 5:
                            continue
                        actor = int(event[EVT["OBSERVER"]])
                        subject = int(event[EVT["SUBJECT"]])
                        realized_events.append({
                            "kind": {5: "damage", 6: "kill", 7: "pickup", 8: "capture" if int(event[EVT["TEAM"]]) > 0 else "tie"}.get(kind, str(kind)),
                            "actor": actor,
                            "actor_team": int(event[EVT["TEAM"]]),
                            "subject": subject,
                            "subject_team": team_by_id.get(subject, 0),
                            "value": float(event[EVT["AMOUNT"]]),
                            "time": float(event[EVT["TIME"]]),
                            "response_seq": int(event[EVT["RESPONSE_SEQ"]]),
                        })

                    if belief_reset and not any(event["kind"] in ("capture", "tie") for event in realized_events):
                        for arm, learning in learners.items():
                            if len(learning.episode):
                                print(json.dumps({"event": "learning_truncation", "arm": arm, **learning.end_episode()}), flush=True)

                    cell_slots, gigi = live_belief.chorus(rows, OBS)
                    belief_diag = live_belief.diagnostics()
                    targets = live_belief.instrument_targets(rows, OBS)
                    tick = build_runtime_frame(
                        rows, cart_rows, targets, k, weight_table=weight_table,
                        navigation=live_belief.navigation_vcmap,
                    )
                    context, cartstate, batch = tick.context, tick.cartstate, tick.batch
                    w_in = np.asarray(tick.weights, dtype=np.float32)
                    semantics = np.asarray(tick.semantics, dtype=np.float32)
                    chorus = assemble(
                        player_features(rows), batch, cell_slots, gigi, semantics, w_in, STRATEGY_DEADLINE_S,
                        args.control_weight, args.exploration_weight, teams_present,
                    )
                    row_policy_arms = np.asarray([
                        team_policy_arms[(int(team) - 1) % len(team_policy_arms)]
                        for team in teams_present
                    ]) if team_policy_arms else np.full(l, args.policy_arm)
                    online_metrics = None
                    policy_updates = {}
                    frame = None
                    if learner is not None:
                        frame = learner.replay.intern(chorus)

                    chorus_mx = tuple(mx.array(a) for a in chorus)
                    model_key, action_key = mx.random.split(model_key)
                    model_key, control_key = mx.random.split(model_key)
                    arm_outputs = {}
                    counterfactual_predictions = {}
                    behavior_versions = {arm: learning.updates for arm, learning in learners.items()}
                    scale_calls = []
                    remote_arms = []
                    if remote_scale is not None:
                        remote_scale.sample = True
                        remote_scale.pending = None
                    if team_policy_arms:
                        actions = np.zeros(l, dtype=np.int64)
                        w_next = np.zeros_like(w_in)
                        for arm in evaluation_arms:
                            remote_sequence = 0 if remote_scale is None else remote_scale.sequence
                            arm_out = policy_forwards[arm](policy_models[arm], *chorus_mx)
                            if remote_scale is not None and remote_scale.sequence > remote_sequence:
                                scale_calls.append(dict(remote_scale.last))
                                if remote_scale.last.get("completed"):
                                    remote_arms.append(arm)
                            arm_outputs[arm] = arm_out
                            arm_actions_mx, _, _ = act(arm_out, action_key)
                            arm_actions = array(arm_actions_mx, np.int64).reshape(l)
                            counterfactual_predictions[arm] = {"actions": arm_actions.tolist(), "winner_value": array(arm_out.value_winnie).tolist(), "loser_value": array(arm_out.value_lou).tolist(), "behavior_updates": behavior_versions.get(arm)}
                            selected = row_policy_arms == arm
                            actions[selected] = arm_actions[selected]
                            w_next[selected] = array(arm_out.weights, np.float32)[selected]
                        out = arm_outputs.get("matrix_fusion", next(iter(arm_outputs.values())))
                    else:
                        remote_sequence = 0 if remote_scale is None else remote_scale.sequence
                        out = policy_forward(wally, *chorus_mx)
                        if remote_scale is not None and remote_scale.sequence > remote_sequence:
                            scale_calls.append(dict(remote_scale.last))
                            if remote_scale.last.get("completed"):
                                remote_arms.append(args.policy_arm)
                        actions_mx, _, _ = act(out, action_key)
                        actions = array(actions_mx, np.int64).reshape(l)
                        w_next = array(out.weights, np.float32)

                    if remote_scale is not None:
                        remote_scale.sample = False
                    realized_outputs = arm_outputs or {args.policy_arm: out}
                    behavior_discrete = np.zeros(l, dtype=np.float32)
                    candidates = np.flatnonzero(row_policy_arms == args.policy_arm)
                    n_off_policy = 0 if args.policy_arm == "default" else min(
                        max(0, args.off_policy_players), len(candidates)
                    )
                    off_policy = np.zeros(l, dtype=bool)
                    if n_off_policy:
                        chosen = rng.choice(candidates, size=n_off_policy, replace=False)
                        for player in chosen:
                            supported_actions = np.flatnonzero(batch.action_mass[player] > 0)
                            by_kind = {}
                            for action in supported_actions:
                                by_kind.setdefault(batch.instruments[action].kind, []).append(action)
                            groups = tuple(by_kind.values())
                            group = groups[int(rng.integers(len(groups)))]
                            actions[player] = rng.choice(group)
                            behavior_discrete[player] = -np.log(len(groups) * len(group))
                        off_policy[chosen] = True
                    n_off_policy = int(off_policy.sum())
                    policy_controlled = rows[:, OBS["CONTROL"]] >= 0.5
                    behavior_arms = np.where(policy_controlled, np.where(off_policy, "uniform", row_policy_arms), "human")
                    controls = np.zeros((l, CONTROL_WIDTH), dtype=np.float32)
                    target_logp = np.zeros(l, dtype=np.float32)
                    control_logp = np.zeros(l, dtype=np.float32)
                    final_actions_mx = mx.array(actions)
                    for arm in dict.fromkeys(row_policy_arms.tolist()):
                        source = realized_outputs[str(arm)]
                        sampled_controls, sampled_logp = sample_controls(
                            source, final_actions_mx, control_key,
                        )
                        discrete_logp = array(logp_of(source, final_actions_mx), np.float32)
                        selected = row_policy_arms == arm
                        controls[selected] = array(sampled_controls, np.float32)[selected]
                        control_logp[selected] = array(sampled_logp, np.float32)[selected]
                        target_logp[selected] = discrete_logp[selected] + control_logp[selected]
                        behavior_discrete[selected & ~off_policy] = discrete_logp[selected & ~off_policy]
                    behavior_logp = behavior_discrete + control_logp
                    weight_table = update_weight_table(batch, w_next, weight_table)
                    cart_depths = tuple(
                        int(np.floor(np.clip(float(value), 0, 1) * cartstate.levels))
                        for value in cartstate.pos
                    )
                    cart_controls = tuple(int(value) for value in cartstate.control)
                    cart_value = tick.game_value
                    game_value = formal_value_record(
                        cart_value,
                        [list(map_key), belief_episode, int(k), list(cart_depths),
                         list(cart_controls)],
                        cartstate.levels,
                    )
                    game_projection = formal_projection_record(
                        cart_value, context.teams,
                    )
                    for name in (
                        "reachable_state_mass", "reachable_role_state_mass",
                        "enumerated_role_state_mass",
                        "role_option_symmetric_difference_mass", "cycle_state_mass",
                    ):
                        cgt_measure_integrals[name] = (
                            cgt_measure_integrals.get(name, 0) + game_value[name]
                        )
                    if cart_value.nimber is not None:
                        nimber = int(cart_value.nimber)
                        cgt_nimbers[nimber] = cgt_nimbers.get(nimber, 0) + 1
                    response = np.zeros((len(obs_rows), RESP_W), dtype=np.float32)
                    local_response = response_rows(
                        batch, actions, controls,
                    )
                    response[active] = local_response
                    resp_id += 1
                    tx.send(STRATEGY_KIND, ch["req_id"], ch["tick"], response, args.peer_node)
                    response_span.finish()
                    work_elapsed = max(time.perf_counter() - work_started, 1e-9)
                    post_response_started = time.perf_counter()
                    realized = gather_policy_rows(
                        realized_outputs, row_policy_arms, actions, len(batch.instruments),
                    )
                    row_atoms = gather_policy_row_atoms(
                        realized_outputs, row_policy_arms, actions,
                    )
                    training_started = time.perf_counter()
                    if learner is not None:
                        if previous is not None:
                            aligned_chorus, successor_present = scatter_gather_participants(
                                chorus, participant_ids, previous["participant_ids"],
                            )
                            successor_frame = learner.replay.intern(aligned_chorus)
                            train_mask = np.ones(previous["players"], dtype=bool)
                            applied_by_id = dict(zip(participant_ids.tolist(), rows[:, OBS["ROUTE_SEQ"]].astype(np.int64).tolist()))
                            applied_mask = np.asarray([applied_by_id.get(int(participant), -1) == previous["request_seq"] for participant in previous["participant_ids"]])
                            with work_meter.span(
                                "policy_optimization", rows=int(train_mask.sum()),
                                operations={
                                    "players": l, "teams": k,
                                    "carts": len(cartstate.pos),
                                    "instruments": len(batch.instruments),
                                    "host_role": "responder",
                                },
                            ):
                                for arm, learning in learners.items():
                                    policy_updates[arm] = learning.observe_attributed([{
                                    "context": previous["context"],
                                    "frame": previous["frame"],
                                    "next_frame": successor_frame,
                                    "snapshot": previous["snapshot"],
                                    "next_snapshot": cartstate,
                                    "actions": previous["actions"],
                                    "controls": previous["controls"],
                                    "behavior_logp": previous["behavior_logp"],
                                    "train_mask": train_mask,
                                    "actor_mask": (previous["behavior_arms"] == arm) & applied_mask,
                                    "behavior_updates": previous["policy_updates"][arm],
                                    "behavior_arms": previous["behavior_arms"],
                                    "behavior_versions": previous["policy_updates"],
                                    "dynamics_mask": applied_mask & successor_present & previous["policy_controlled"],
                                    "sparse_return": role_rewards(
                                        previous["context"], previous["snapshot"], cartstate,
                                    ),
                                    "bootstrap_discount": learning.gamma * successor_present.astype(np.float32),
                                }])
                            online_metrics = policy_updates.get(next(iter(learners)))
                            if online_metrics is None:
                                online_metrics = {}
                            online_metrics["causal_attribution"] = {
                                "groups": 1,
                                "rows": int(train_mask.sum()),
                                "joined_row_mass": int(np.setdiff1d(
                                    participant_ids, previous["participant_ids"],
                                ).size),
                                "mean_age": 1.0,
                                "successor_source_row_mass": len(successor_present),
                                "successor_present_row_mass": int(successor_present.sum()),
                                "departed_row_mass": int((~successor_present).sum()),
                                "applied_actor_source_rows": int(applied_mask.sum()),
                            }
                        for event in realized_events:
                            outcome_key = (event["kind"], event["time"], event["actor_team"], map_key)
                            if event["kind"] in ("capture", "tie") and outcome_key not in observed_outcomes:
                                observed_outcomes.add(outcome_key)
                                winner = int(event["actor_team"])
                                outcome_measure["rounds"] += 1
                                outcome_measure["draws"] += int(winner == 0)
                                for arm in team_policy_arms:
                                    outcome_measure["team_rounds"][arm] = outcome_measure["team_rounds"].get(arm, 0) + 1
                                if winner > 0 and team_policy_arms:
                                    arm = team_policy_arms[winner - 1]
                                    outcome_measure["wins"][arm] = outcome_measure["wins"].get(arm, 0) + 1
                                outcome_measure["last_winning_team"] = winner
                                for arm, learning in learners.items():
                                    result = learning.end_episode(event["actor_team"])
                                    update = policy_updates.get(arm) or {}
                                    policy_updates[arm] = {**update, **result, "gradient_steps": update.get("gradient_steps", 0) + result.get("gradient_steps", 0)}
                                outcome = {"event": "learning_outcome", "at": time.time(), "server_event": event, "team_policy_arms": team_policy_arms, "policies": policy_updates}
                                outcome_path = os.path.join(os.path.dirname(args.telemetry), "outcome.json")
                                with open(outcome_path + ".new", "w") as handle:
                                    json.dump(outcome, handle)
                                os.replace(outcome_path + ".new", outcome_path)
                                print(json.dumps(outcome), flush=True)
                        stats["updates"] = sum(learning.updates for learning in learners.values())
                        due_updates = (
                            args.save_every > 0
                            and stats["updates"] > last_saved_update
                            and stats["updates"] % args.save_every == 0
                        )
                        due_time = (
                            args.save_secs > 0
                            and stats["updates"] > last_saved_update
                            and time.time() - last_save_time >= args.save_secs
                        )
                        if due_updates or due_time:
                            for learning in learners.values():
                                learning.save()
                            save_runstate(rng, model_key, resp_id, nt_written, stats["updates"])
                            last_saved_update = stats["updates"]
                            last_save_time = time.time()
                    training_elapsed = max(0.0, time.perf_counter() - training_started)
                    cart_state_width = cart_rows.size
                    server_state = np.concatenate((
                        array(rows, np.float32),
                        np.broadcast_to(
                            array(cart_rows, np.float32).reshape(1, cart_state_width),
                            (l, cart_state_width),
                        ),
                    ), axis=1)
                    server_state_labels = (
                        [f"observation.{name.lower()}" for name, _ in sorted(OBS.items(), key=lambda item: item[1])]
                        + [
                            f"cart.{int(cart_rows[cart, CS['ID']])}.{name.lower()}"
                            for cart in range(len(cart_rows))
                            for name, _ in sorted(CS.items(), key=lambda item: item[1])
                        ]
                    )
                    model_arrays = {
                        "x": array(chorus.xan, np.float32),
                        "z": array(chorus.zed, np.float32),
                        "cell_slots": array(chorus.cell_slots, np.float32),
                        "gigi": array(chorus.gigi, np.float32),
                        "hierarchy": semantics,
                        "team_ids": array(chorus.team_ids, np.int64),
                        "w": w_in,
                        "action_mass": array(chorus.action_mass, np.float32),
                        "delta": array(chorus.delta, np.float32),
                        "control_weight": array(chorus.control_weight, np.float32),
                        "exploration_weight": array(chorus.exploration_weight, np.float32),
                        "selected_z": array(chorus.zed, np.float32)[actions],
                        "selected_w": w_in[np.arange(l), actions],
                        "selected_action_mass": array(chorus.action_mass, np.float32)[np.arange(l), actions],
                        "server_state": server_state,
                        "coupling": realized["coupling"],
                        "score": realized["score"],
                        "diag_k": realized["diag_k"],
                        "dw_dt": realized["dw_dt"],
                        "dynamics_guidance": realized["dynamics_guidance"],
                        "model_uncertainty": realized["model_uncertainty"],
                        "actuator": realized["actuator"],
                        "actuator_log_scale": realized["actuator_log_scale"],
                        "actuator_density": realized["actuator_density"],
                        "winner_value": realized["winner_value"],
                        "loser_value": realized["loser_value"],
                        "aux_winner_value": realized["aux_winner_value"],
                        "aux_loser_value": realized["aux_loser_value"],
                    }
                    source_features = [None] * l
                    source_feature_labels = [None] * l
                    for arm in dict.fromkeys(row_policy_arms.tolist()):
                        selected = np.flatnonzero(row_policy_arms == arm)
                        arm_model_arrays = dict(model_arrays)
                        arm_model_arrays["beta"] = array(
                            realized_outputs[str(arm)].belief, np.float32,
                        )
                        arm_features, arm_labels, _, _ = literal_source_coordinates(
                            arm_model_arrays, server_state_labels,
                        )
                        for index in selected:
                            source_features[index] = arm_features[index].copy()
                            source_feature_labels[index] = tuple(arm_labels)
                    if len(set(row_policy_arms.tolist())) == 1:
                        for name in ("j", "beta", "pooled"):
                            model_arrays[name] = np.stack([
                                atom[name] for atom in row_atoms
                            ])
                    row_output_shapes = {
                        str(arm): {
                            "participant_mass": int(np.sum(row_policy_arms == arm)),
                            "j": list(array(realized_outputs[str(arm)].ir).shape[2:]),
                            "beta": list(array(realized_outputs[str(arm)].belief).shape[1:]),
                            "pooled": list(array(realized_outputs[str(arm)].pooled).shape[1:]),
                        }
                        for arm in dict.fromkeys(row_policy_arms.tolist())
                    }
                    scale_outputs = {
                        arm: value for arm, value in (
                            arm_outputs or {args.policy_arm: out}
                        ).items() if is_matrix_fusion_arm(arm)
                    }
                    scale_stats_by_arm = {
                        arm: array(value.scale_matrix_stats, np.float32)
                        for arm, value in scale_outputs.items()
                    }
                    scale_load_by_arm = {
                        arm: array(value.scale_expert_load, np.float32)
                        for arm, value in scale_outputs.items()
                    }
                    scale_load = sum(scale_load_by_arm.values(), np.zeros(widths.scale_experts, dtype=np.float32))
                    scale_active = bool(scale_outputs)
                    scale_matrix_range = [
                        min(float(value[0]) for value in scale_stats_by_arm.values()),
                        max(float(value[1]) for value in scale_stats_by_arm.values()),
                    ] if scale_active else None
                    model = {
                        "composer_inputs": [
                            "x", "z", "cell_slots", "gigi", "hierarchy", "team_ids",
                            "w", "action_mass", "delta", "control_weight", "exploration_weight",
                        ],
                        "shapes": {name: list(value.shape) for name, value in model_arrays.items()},
                        "finite_coordinate_mass": {
                            name: int(np.isfinite(value).sum()) for name, value in model_arrays.items()
                        },
                        "nonfinite_coordinate_mass": {
                            name: int((~np.isfinite(value)).sum()) for name, value in model_arrays.items()
                        },
                        "range": {
                            name: finite_range(value)
                            for name, value in model_arrays.items()
                        },
                        "winner_value": model_arrays["winner_value"].tolist(),
                        "loser_value": model_arrays["loser_value"].tolist(),
                        "row_output_shapes": row_output_shapes,
                        "scale_operator": {
                            "output_arm_mass": len(scale_outputs),
                            "causal_position": "pre_head_residual",
                            "residual_rows": l * len(batch.instruments) * len(scale_outputs),
                            "residual_rank": widths.d_scale,
                            "hidden_width": widths.scale_h,
                            "matrix_shape": [widths.d_scale, widths.d_scale] if scale_active else [0, 0],
                            "matrix_finite_coordinate_mass": int(sum(value[2] for value in scale_stats_by_arm.values())) if scale_active else 0,
                            "matrix_coordinate_mass": len(scale_outputs) * widths.d_scale * widths.d_scale,
                            "matrix_range": np.round(scale_matrix_range, 6).tolist() if scale_active else None,
                            "experts": widths.scale_experts,
                            "topk": widths.scale_topk,
                            "expert_load": np.round(scale_load, 3).tolist(),
                            "active_experts": int(np.count_nonzero(scale_load)) if scale_active else 0,
                            "arms": {
                                arm: {
                                    "participant_fusion_scale": 0.0 if arm == "participant_fusion_ablated" else 1.0,
                                    "residual_fusion_scale": 0.0 if arm == "residual_fusion_ablated" else 1.0,
                                    "host_role": "matrix" if arm in remote_arms else "responder",
                                    "operation": "gram_context",
                                    "rows": l * len(batch.instruments),
                                    "matrix_finite_coordinate_mass": int(scale_stats_by_arm[arm][2]),
                                    "matrix_coordinate_mass": widths.d_scale * widths.d_scale,
                                    "matrix_range": np.round(scale_stats_by_arm[arm][:2], 6).tolist(),
                                    "expert_load": np.round(scale_load_by_arm[arm], 3).tolist(),
                                }
                                for arm in scale_outputs
                            },
                            "distributed": distributed_scale_summary(
                                scale_calls,
                                0 if remote_scale is None else remote_scale.counterfactual_interval,
                            ),
                        },
                    }
                    if "j" in model_arrays:
                        model["j_labels"] = list(row_atoms[0]["j_labels"])
                    model["matrix_fusion_intervention"] = matrix_fusion_intervention_measure(arm_outputs)
                    sampled = args.model_sample_every > 0 and (resp_id + 1) % args.model_sample_every == 0
                    model["sampled"] = sampled
                    if arm_outputs:
                        model["arms"] = {
                            arm: {
                                "logit_coordinate_mass": int(np.asarray(chorus.action_mass).sum()),
                                "finite_logit_coordinate_mass": int(np.isfinite(array(value.logits)[np.asarray(chorus.action_mass) > 0]).sum()),
                                "coupling_range": [round(float(array(value.coupling).min()), 6), round(float(array(value.coupling).max()), 6)],
                            }
                            for arm, value in arm_outputs.items()
                        }
                    if sampled:
                        for name in (
                            "x", "beta", "z", "cell_slots", "gigi", "hierarchy", "team_ids",
                            "w", "action_mass", "delta", "control_weight", "exploration_weight",
                            "j", "selected_z", "selected_w", "selected_action_mass", "server_state", "pooled", "coupling", "diag_k", "dw_dt",
                        ):
                            if name in model_arrays:
                                model[name] = model_arrays[name].tolist()
                        model["row_outputs"] = [
                            {
                                "row": index,
                                "arm": atom["arm"],
                                "j": atom["j"].tolist(),
                                "j_labels": list(atom["j_labels"]),
                                "beta": atom["beta"].tolist(),
                                "pooled": atom["pooled"].tolist(),
                            }
                            for index, atom in enumerate(row_atoms)
                        ]
                    assignments = []
                    strategy_focus = np.zeros((k, k), dtype=np.int64)
                    counter_columns = dict(OBS_OUTCOME_COUNTER_COLUMNS)
                    for local, row_index in enumerate(active):
                        team = int(teams_present[local])
                        action = int(actions[local])
                        instrument = batch.instruments[action]
                        kind = instrument.kind.value
                        target_kind_id = int(local_response[local, SC["TARGET_KIND"]])
                        target_id = int(local_response[local, SC["TARGET_ID"]])
                        target_cell = (
                            int(local_response[local, SC["TARGET_CELL_X"]]),
                            int(local_response[local, SC["TARGET_CELL_Y"]]),
                        )
                        gain = float(local_response[local, SC["GAIN"]])
                        commitment = assignment_commitment(
                            batch.participants[local], instrument,
                            controls[local, 1], batch.navigation,
                        )
                        applied_target_kind = int(rows[local, OBS["APPLIED_TARGET_KIND"]])
                        applied_target_id = int(rows[local, OBS["APPLIED_TARGET_ID"]])
                        applied_target_cell = (
                            int(rows[local, OBS["APPLIED_TARGET_CELL_X"]]),
                            int(rows[local, OBS["APPLIED_TARGET_CELL_Y"]]),
                        )
                        goal_target_kind = int(rows[local, OBS["GOAL_TARGET_KIND"]])
                        goal_target_id = int(rows[local, OBS["GOAL_TARGET_ID"]])
                        goal_target_cell = (
                            int(rows[local, OBS["GOAL_TARGET_CELL_X"]]),
                            int(rows[local, OBS["GOAL_TARGET_CELL_Y"]]),
                        )
                        applied_kind, applied_subject = decode_target(
                            applied_target_kind, applied_target_id, *applied_target_cell,
                        )
                        goal_kind, goal_subject = decode_target(
                            goal_target_kind, goal_target_id, *goal_target_cell,
                        )
                        edict = int(participant_ids[local])
                        totals = {
                            name: float(rows[local, OBS[column]])
                            for name, column in counter_columns.items()
                        }
                        observed_response_seq = int(rows[local, OBS["RESPONSE_SEQ"]])
                        route_seq = int(rows[local, OBS["ROUTE_SEQ"]])
                        goal_seq = int(rows[local, OBS["GOAL_SEQ"]])
                        touch_seq = int(rows[local, OBS["TOUCH_SEQ"]])
                        delivered_response_seq = observed_response_seq if observed_response_seq > 0 else None
                        applied_response_seq = route_seq if route_seq > 0 else None
                        delivered_decision = decision_history.get(
                            delivered_response_seq, {}
                        ).get(edict)
                        applied_decision = decision_history.get(
                            applied_response_seq, {}
                        ).get(edict)
                        applied_action = (
                            None if applied_decision is None
                            else applied_decision["action"]
                        )
                        applied_action_target_kind = None if applied_decision is None else applied_decision["target_kind"]
                        applied_action_target_id = None if applied_decision is None else applied_decision["target_id"]
                        applied_action_target_cell = None if applied_decision is None else applied_decision["target_cell"]
                        applied_action_current = bool(
                            applied_decision is not None
                            and rows[local, OBS["TARGET_RESOLVED"]] >= 0.5
                            and applied_action_target_kind == applied_target_kind
                            and (
                                applied_action_target_cell == applied_target_cell
                                if applied_target_kind == TARGET_KIND["CELL"]
                                else applied_action_target_id == applied_target_id
                            )
                        )
                        routed_outcomes = []
                        for sequence_column, routed_columns in OBS_ROUTED_OUTCOME_GROUPS:
                            outcome_seq = int(rows[local, OBS[sequence_column]])
                            if outcome_seq <= 0:
                                continue
                            outcome_decision = decision_history.get(outcome_seq, {}).get(edict)
                            routed_outcomes.append({
                                "response_seq": outcome_seq,
                                "policy_arm": None if outcome_decision is None else outcome_decision["policy_arm"],
                                "behavior": None if outcome_decision is None else outcome_decision["behavior"],
                                "action": None if outcome_decision is None else outcome_decision["action"],
                                "subject": None if outcome_decision is None else outcome_decision["subject"],
                                "target_kind": None if outcome_decision is None else outcome_decision["target_kind"],
                                "target_id": None if outcome_decision is None else outcome_decision["target_id"],
                                "target_cell": None if outcome_decision is None else outcome_decision["target_cell"],
                                "outcomes": {
                                    name: float(rows[local, OBS[column]])
                                    for name, column in routed_columns
                                },
                            })
                        engine_time = float(rows[local, OBS["ENGINE_TIME"]])
                        spawn_time = float(rows[local, OBS["SPAWN_TIME"]])
                        response_time = float(rows[local, OBS["RESPONSE_TIME"]])
                        goal_present = rows[local, OBS["GOAL_PRESENT"]] >= 0.5
                        goal_distance = float(np.linalg.norm(
                            rows[local, OBS["GOAL_POS_X"]:OBS["GOAL_POS_Z"] + 1]
                            - rows[local, OBS["POS_X"]:OBS["POS_Z"] + 1]
                        )) if goal_present else None
                        if instrument.team > 0 and instrument.team != team:
                            strategy_focus[team - 1, instrument.team - 1] += 1
                        assignments.append(
                            dict(
                                row=int(row_index), edict=edict, team=team,
                                controller="bot" if rows[local, OBS["CONTROL"]] >= 0.5 else "human",
                                behavior=str(behavior_arms[local]),
                                policy_arm=str(row_policy_arms[local]),
                                action=action, kind=kind, subject=instrument.subject,
                                target_kind=target_kind_id, target_id=target_id,
                                target_cell=target_cell,
                                gain=gain,
                                commit=float(local_response[local, SC["COMMIT"]]),
                                walking_distance=commitment.walking_distance,
                                walking_time=commitment.walking_time,
                                commit_extension=commitment.extension,
                                spawn=float(local_response[local, SC["SPAWN"]]),
                                delivered_response_seq=delivered_response_seq,
                                delivered_action=None if delivered_decision is None else delivered_decision["action"],
                                delivered_target_kind=None if delivered_decision is None else delivered_decision["target_kind"],
                                delivered_target_id=None if delivered_decision is None else delivered_decision["target_id"],
                                delivered_target_cell=None if delivered_decision is None else delivered_decision["target_cell"],
                                successor_state=server_state[local].tolist(),
                                successor_state_labels=server_state_labels,
                                applied_target_kind=applied_target_kind,
                                applied_target_id=applied_target_id,
                                applied_target_cell=applied_target_cell,
                                applied_kind=applied_kind, applied_subject=applied_subject,
                                applied_action=applied_action,
                                applied_policy_arm=None if applied_decision is None else applied_decision["policy_arm"],
                                applied_behavior=None if applied_decision is None else applied_decision["behavior"],
                                applied_action_subject=None if applied_decision is None else applied_decision["subject"],
                                applied_action_target_kind=applied_action_target_kind,
                                applied_action_target_id=applied_action_target_id,
                                applied_action_target_cell=applied_action_target_cell,
                                applied_response_seq=applied_response_seq,
                                applied_action_current=applied_action_current,
                                target_resolved=bool(rows[local, OBS["TARGET_RESOLVED"]] >= 0.5),
                                goal_target_kind=goal_target_kind,
                                goal_target_id=goal_target_id,
                                goal_target_cell=goal_target_cell,
                                goal_kind=goal_kind, goal_subject=goal_subject,
                                goal_distance=goal_distance,
                                engine_time=engine_time,
                                time_since_spawn=None if spawn_time <= 0 else engine_time - spawn_time,
                                alive=bool(rows[local, OBS["ALIVE"]] >= 0.5),
                                spawn_swizzle_active=bool(rows[local, OBS["SWIZZLE_ACTIVE"]] >= 0.5),
                                spawn_swizzle_epoch=int(rows[local, OBS["SWIZZLE_EPOCH"]]),
                                spawn_swizzle_player_count=int(rows[local, OBS["SWIZZLE_PLAYER_COUNT"]]),
                                spawn_swizzle_spot_count=int(rows[local, OBS["SWIZZLE_SPOT_COUNT"]]),
                                spawn_swizzle_slot_count=int(rows[local, OBS["SWIZZLE_SLOT_COUNT"]]),
                                spawn_swizzle_ticket=int(rows[local, OBS["SWIZZLE_TICKET"]]),
                                spawn_swizzle_cohort=int(rows[local, OBS["SWIZZLE_COHORT"]]),
                                spawn_swizzle_cohort_count=int(rows[local, OBS["SWIZZLE_COHORT_COUNT"]]),
                                spawn_swizzle_generation=int(rows[local, OBS["SWIZZLE_GENERATION"]]),
                                spawn_swizzle_scheduled_time=float(rows[local, OBS["SWIZZLE_SCHEDULED_TIME"]]),
                                spawn_swizzle_actual_time=float(rows[local, OBS["SWIZZLE_ACTUAL_TIME"]]),
                                spawn_swizzle_lane=int(rows[local, OBS["SWIZZLE_LANE"]]),
                                spawn_swizzle_spot=int(rows[local, OBS["SWIZZLE_SPOT"]]),
                                goal_match=bool(
                                    applied_target_kind > 0
                                    and goal_target_kind == applied_target_kind
                                    and (
                                        goal_target_cell == applied_target_cell
                                        if applied_target_kind == TARGET_KIND["CELL"]
                                        else goal_target_id == applied_target_id
                                    )
                                ),
                                target_touch=bool(goal_seq > 0 and touch_seq == goal_seq),
                                request_seq=int(ch["req_id"]),
                                observed_response_seq=observed_response_seq,
                                response_age=None if response_time <= 0 else engine_time - response_time,
                                route_seq=route_seq,
                                route_latency=float(rows[local, OBS["ROUTE_LATENCY"]]),
                                goal_seq=goal_seq,
                                goal_latency=float(rows[local, OBS["GOAL_LATENCY"]]),
                                touch_seq=touch_seq,
                                touch_latency=float(rows[local, OBS["TOUCH_LATENCY"]]),
                                routed_current=bool(delivered_response_seq is not None and route_seq == delivered_response_seq),
                                goal_current=bool(delivered_response_seq is not None and goal_seq == delivered_response_seq),
                                touch_current=bool(delivered_response_seq is not None and touch_seq == delivered_response_seq),
                                outcome_totals=totals,
                                routed_outcomes=routed_outcomes,
                                target_logp=float(target_logp[local]),
                                behavior_logp=float(behavior_logp[local]) if policy_controlled[local] else None,
                            )
                        )
                    for event in realized_events:
                        source = decision_history.get(event["response_seq"], {}).get(event["actor"])
                        event["policy_arm"] = None if source is None else source["policy_arm"]
                        event["aligned_target"] = bool(
                            event["kind"] in ("damage", "kill")
                            and source is not None
                            and source["target_kind"] == TARGET_KIND["RIVAL"]
                            and source["target_id"] == event["subject"]
                        )
                    current_decisions = {
                        item["edict"]: {
                            "action": item["kind"],
                            "policy_arm": item["policy_arm"],
                            "behavior": item["behavior"],
                            "subject": item["subject"],
                            "target_kind": int(item["target_kind"]),
                            "target_id": int(item["target_id"]),
                            "target_cell": tuple(item["target_cell"]),
                            "controls": [
                                float(item["gain"]),
                                float(item["commit"]),
                                float(item["spawn"]),
                            ],
                            "j": row_atoms[local]["j"].copy(),
                            "j_labels": row_atoms[local]["j_labels"],
                            "features": source_features[local].copy(),
                            "feature_labels": source_feature_labels[local],
                            "server_state": server_state[local].copy(),
                            "state_labels": tuple(server_state_labels),
                        }
                        for local, item in enumerate(assignments)
                    }
                    if remote_scale is not None:
                        remote_scale.measure_local_counterfactual(STRATEGY_DEADLINE_S, work_elapsed, scale_calls)
                        model["scale_operator"]["distributed"] = distributed_scale_summary(
                            scale_calls, remote_scale.counterfactual_interval,
                        )
                    stats["resp"] += 1
                    measured_arms = tuple(arm_outputs) if arm_outputs else (args.policy_arm,)
                    gradient_steps = sum(int((update or {}).get("gradient_steps", 0)) for update in policy_updates.values())
                    gradient_batch = sum(int((update or {}).get("batch", 0)) + int((update or {}).get("replay_batch", 0)) for update in policy_updates.values())
                    full_work_parts = [
                        strategy_work(
                            arm, widths, l, len(batch.instruments),
                            len(cell_slots), args.baseline_hidden,
                            int((policy_updates.get(arm) or {}).get("gradient_steps", 0)),
                            int((policy_updates.get(arm) or {}).get("batch", 0)) + int((policy_updates.get(arm) or {}).get("replay_batch", 0)),
                        )
                        for arm in measured_arms
                    ]
                    response_work_parts = [
                        strategy_work(
                            arm, widths, l, len(batch.instruments),
                            len(cell_slots), args.baseline_hidden,
                        )
                        for arm in measured_arms
                    ]
                    work = {
                        key: sum(part[key] for part in response_work_parts)
                        for key in ("lower_flops", "upper_flops", "lower_bytes", "upper_bytes")
                    }
                    meter_parts = [
                        part["local"] if arm in remote_arms else part
                        for arm, part in zip(measured_arms, response_work_parts)
                    ]
                    full_meter_parts = [
                        part["local"] if arm in remote_arms else part
                        for arm, part in zip(measured_arms, full_work_parts)
                    ]
                    meter_work = {
                        key: sum(part[key] for part in meter_parts)
                        for key in ("lower_flops", "upper_flops", "lower_bytes", "upper_bytes")
                    }
                    training_work = {
                        key: sum(full[key] - response[key] for full, response in zip(
                            full_work_parts, response_work_parts,
                        ))
                        for key in ("lower_flops", "upper_flops", "lower_bytes", "upper_bytes")
                    }
                    training_meter_work = {
                        key: sum(full[key] - response[key] for full, response in zip(
                            full_meter_parts, meter_parts,
                        ))
                        for key in ("lower_flops", "upper_flops", "lower_bytes", "upper_bytes")
                    }
                    local_scale_parts = [
                        part for arm, part in zip(measured_arms, response_work_parts)
                        if is_matrix_fusion_arm(arm) and arm not in remote_arms
                    ]
                    local_residual_rows = sum(part["residual_rows"] for part in local_scale_parts)
                    local_parameter_bytes = sum(part["parameter_bytes"] for part in meter_parts)
                    local_counterfactual = None if remote_scale is None else remote_scale.last_counterfactual
                    work.update({
                        "elapsed_s": work_elapsed,
                        "deadline_s": STRATEGY_DEADLINE_S,
                        "deadline_slack_s": STRATEGY_DEADLINE_S - work_elapsed,
                        "lower_gflops_s": work["lower_flops"] / work_elapsed / 1e9,
                        "upper_gflops_s": work["upper_flops"] / work_elapsed / 1e9,
                        "lower_gbs": work["lower_bytes"] / work_elapsed / 1e9,
                        "upper_gbs": work["upper_bytes"] / work_elapsed / 1e9,
                        "arms": list(measured_arms),
                        "distributed_arms": list(remote_arms),
                        "parameter_bytes": sum(part["parameter_bytes"] for part in response_work_parts),
                        "residual_rows": sum(part["residual_rows"] for part in response_work_parts),
                        "residual_rank": max(part["residual_rank"] for part in response_work_parts),
                        "experts": max(part["experts"] for part in response_work_parts),
                        "topk": max(part["topk"] for part in response_work_parts),
                        "optimization": {
                            **training_work,
                            "elapsed_s": training_elapsed,
                            "lower_gflops_s": training_work["lower_flops"] / max(training_elapsed, 1e-9) / 1e9,
                            "upper_gflops_s": training_work["upper_flops"] / max(training_elapsed, 1e-9) / 1e9,
                            "lower_gbs": training_work["lower_bytes"] / max(training_elapsed, 1e-9) / 1e9,
                            "upper_gbs": training_work["upper_bytes"] / max(training_elapsed, 1e-9) / 1e9,
                            "gradient_steps": gradient_steps,
                            "gradient_batch": gradient_batch,
                            "after_response": True,
                        },
                    })
                    measure_revision, current_measures = j_reporter.snapshot()
                    learning_state = {arm: {"updates": learning.updates, "completed_episodes": learning.completed_episodes, "truncated_episodes": learning.truncated_episodes, "pending_states": len(learning.episode), "replay_size": len(learning.replay), **learning.replay.report(), "reward_contract": reward_contract(arm)} for arm, learning in learners.items()}
                    latest_policy_updates.update({arm: {**update, "sampled_at": time.time(), "response": resp_id, "row_scope": "last_optimizer_step", "gradient_steps_scope": "response"} for arm, update in policy_updates.items() if (update or {}).get("gradient_steps", 0)})
                    learning_measures = {
                        "match": {"environment": args.environment, "teams": k, "carts": j, "players": l, "human_rows": int((~policy_controlled).sum()), "bot_rows": int(policy_controlled.sum()), "responses": resp_id, "sampled_at": time.time(), "team_policy_arms": dict(enumerate(team_policy_arms, 1))},
                        "outcomes": dict(outcome_measure),
                        **{f"learning.{arm}": {**state, "last_update": latest_policy_updates.get(arm, {})} for arm, state in learning_state.items()},
                    }
                    published_measures = {**(current_measures if measure_revision != published_measure_revision else {}), **learning_measures}
                    published_measure_revision = measure_revision
                    post_response_measure_elapsed = time.perf_counter() - post_response_started
                    if gradient_steps:
                        work_meter.record(
                            training_elapsed,
                            training_meter_work["lower_flops"], training_meter_work["upper_flops"],
                            training_meter_work["lower_bytes"], training_meter_work["upper_bytes"],
                            rows=max(l, local_residual_rows),
                            operations={
                                "stage": "policy_optimization",
                                "players": l,
                                "teams": k,
                                "carts": len(cartstate.pos),
                                "instruments": len(batch.instruments),
                                "gradient_steps": gradient_steps,
                                "gradient_batch": gradient_batch,
                                "host_role": "responder",
                            },
                        )
                    work_meter.record(
                        work_elapsed, meter_work["lower_flops"], meter_work["upper_flops"],
                        meter_work["lower_bytes"], meter_work["upper_bytes"],
                        deadline_s=STRATEGY_DEADLINE_S,
                        rows=max(l, local_residual_rows),
                        operations={
                            "stage": "policy_response",
                            "players": l,
                            "teams": k,
                            "carts": len(cartstate.pos),
                            "instruments": len(batch.instruments),
                            "cells": len(cell_slots),
                            "arms": list(measured_arms),
                            "distributed_arms": list(remote_arms),
                            "gradient_steps": 0,
                            "gradient_batch": 0,
                            "residual_rows": local_residual_rows,
                            "residual_rank": work["residual_rank"],
                            "experts": work["experts"],
                            "topk": work["topk"],
                            "parameter_bytes": local_parameter_bytes,
                            "host_role": "responder",
                            "remote_call_mass": len(scale_calls),
                            "remote_request_row_mass": sum(int(call.get("request_row_mass") or 0) for call in scale_calls),
                            "remote_output_row_mass": sum(int(call.get("output_row_mass") or 0) for call in scale_calls),
                            "remote_processed_row_mass": sum(int(call.get("processed_row_mass") or 0) for call in scale_calls),
                            "remote_operation": "gram_context",
                            "local_only_plan_elapsed_s": None if local_counterfactual is None else local_counterfactual["local_plan_counterfactual_s"],
                            "local_only_plan_lower_s": None if local_counterfactual is None else local_counterfactual["local_plan_counterfactual_lower_s"],
                            "local_only_plan_upper_s": None if local_counterfactual is None else local_counterfactual["local_plan_counterfactual_upper_s"],
                            "local_only_deadline_slack_s": None if local_counterfactual is None else local_counterfactual["local_plan_counterfactual_slack_s"],
                            "local_only_measure_age_s": None if local_counterfactual is None else time.time() - local_counterfactual["sampled_at"],
                            "post_response_measure_elapsed_s": post_response_measure_elapsed,
                        },
                        measures=published_measures,
                    )
                    previous = dict(
                        context=context, snapshot=cartstate, cartstate=cartstate,
                        request_seq=int(ch["req_id"]),
                        frame=frame, players=l, actions=actions.copy(),
                        controls=controls.copy(),
                        behavior_logp=behavior_logp.copy(),
                        row_policy_arms=row_policy_arms.copy(),
                        behavior_arms=behavior_arms.copy(),
                        policy_controlled=policy_controlled.copy(),
                        off_policy=off_policy.copy(),
                        policy_updates=behavior_versions,
                        train_mask=(row_policy_arms == args.policy_arm).copy(),
                        teams_present=teams_present.copy(),
                        participant_ids=participant_ids.copy(),
                    )
                    live_provenance = {arm: dict(source) for arm, source in policy_provenance.items()}
                    for arm, learning in learners.items():
                        live_provenance[arm]["updates"] = int(learning.updates)
                    line = dict(
                        environment=args.environment,
                        policy_arm=active_policy,
                        team_policy_arms=list(team_policy_arms),
                        policy_provenance=live_provenance,
                        t=round(time.time() - t0, 3),
                        mode="online_train" if learner is not None else "inference",
                        updates=learner.updates if learner is not None else 0,
                        resp_id=resp_id, request_seq=int(ch["req_id"]),
                        coalesced_observations=coalesced_observations,
                        req_tick=int(ch["tick"]), obs_tick=int(oh["tick"]),
                        k=k, j=j, l=l,
                        off_policy_players=n_off_policy,
                        resources=team_resources(rows, k),
                        strategy_focus=strategy_focus.tolist(),
                        carts=[
                            dict(
                                id=int(cart_rows[c, CS["ID"]]),
                                path_position=float(cart_rows[c, CS["PATH_POSITION"]]),
                                path_length=float(cart_rows[c, CS["PATH_LENGTH"]]),
                                depth=float(cartstate.pos[c]),
                                control_team=int(cart_rows[c, CS["CONTROL_TEAM"]]),
                                speed=float(cart_rows[c, CS["SPEED"]]),
                                idle_time=float(cart_rows[c, CS["IDLE_TIME"]]),
                                lead_team=int(cart_rows[c, CS["LEAD_TEAM"]]),
                                second_team=int(cart_rows[c, CS["SECOND_TEAM"]]),
                                home_team=int(cart_rows[c, CS["HOME_TEAM"]]),
                                position_x=float(cart_rows[c, CS["POS_X"]]),
                                position_y=float(cart_rows[c, CS["POS_Y"]]),
                                position_z=float(cart_rows[c, CS["POS_Z"]]),
                                supports_player=bool(cart_rows[c, CS["SUPPORTS_PLAYER"]] >= 0.5),
                                team_count=int(cart_rows[c, CS["TEAM_COUNT"]]),
                                rollback_active=bool(cart_rows[c, CS["ROLLBACK_ACTIVE"]] >= 0.5),
                                rollback_target=float(cart_rows[c, CS["ROLLBACK_TARGET"]]),
                            )
                            for c in range(j)
                        ],
                        **game_projection,
                        game_value=game_value,
                        belief=dict(
                            reset=belief_reset, event_tick=event_tick,
                            deposited=deposited, **belief_diag,
                        ),
                        instrument_count=len(batch.instruments),
                        instrument_counts={
                            kind: sum(instrument.kind.value == kind for instrument in batch.instruments)
                            for kind in sorted({instrument.kind.value for instrument in batch.instruments})
                        },
                        server_state_labels=server_state_labels,
                        assignments=assignments, update=online_metrics,
                        policy_updates=policy_updates,
                        counterfactual_predictions=counterfactual_predictions,
                        learning=learning_state,
                        reward_contract=reward_contract(args.policy_arm),
                        reward_contracts={arm: reward_contract(arm) for arm in realized_outputs},
                        work=work,
                        realized_events=realized_events,
                        dynamics_guidance=None if online_metrics is None else {
                            name: online_metrics.get(name)
                            for name in ("loss_dynamics", "model_one_step_error",
                                         "model_uncertainty", "local_control_sigma_min")
                        },
                        model=model,
                    )
                    measurement_source_keys = {
                        (int(ch["req_id"]), int(item["edict"]))
                        for item in assignments
                    } | {
                        (int(sequence), int(item["edict"]))
                        for item in assignments
                        for sequence in (
                            item.get("delivered_response_seq"),
                            item.get("route_seq"),
                            item.get("goal_seq"),
                            item.get("touch_seq"),
                            *(routed.get("response_seq") for routed in item["routed_outcomes"]),
                        )
                        if sequence is not None and int(sequence) > 0
                    } | {
                        (int(event["response_seq"]), int(event["actor"]))
                        for event in realized_events
                        if int(event.get("response_seq") or 0) > 0
                    }
                    measurement_source_sets = dict(decision_history.items())
                    measurement_source_sets[int(ch["req_id"])] = current_decisions
                    measurement_sources = [
                        {
                            "response_seq": sequence,
                            "edict": edict,
                            "j": source["j"],
                            "j_labels": source["j_labels"],
                            "features": source["features"],
                            "server_state": source["server_state"],
                            "state_labels": source["state_labels"],
                            "feature_labels": source["feature_labels"],
                            "policy_arm": source["policy_arm"],
                            "behavior": source["behavior"],
                            "action": {
                                "kind": source["action"],
                                "target_kind": source["target_kind"],
                                "target_id": source["target_id"],
                                "target_cell": source["target_cell"],
                                "controls": source["controls"],
                            },
                        }
                        for sequence, edict in sorted(measurement_source_keys)
                        if (source := measurement_source_sets.get(sequence, {}).get(edict)) is not None
                    ]
                    decision_history.put(
                        int(ch["req_id"]), current_decisions,
                        {sequence for sequence, _ in measurement_source_keys},
                    )
                    line["source_window"] = decision_history.measure()
                    measurement_model = dict(model)
                    measurement_model.update(model_arrays)
                    j_reporter.ingest(dict(
                        line, model=measurement_model, measure_sources=measurement_sources,
                    ))
                    telem.write(json.dumps(line) + "\n")
                    telem.flush()
                    learning_path = os.path.join(os.path.dirname(args.telemetry), "learning.json")
                    with open(learning_path + ".new", "w") as handle:
                        json.dump({"at": time.time(), "environment": args.environment, "teams": k, "carts": j, "players": l, "responses": resp_id, "team_policy_arms": team_policy_arms, "learning": line["learning"], "policy_updates": policy_updates}, handle)
                    os.replace(learning_path + ".new", learning_path)
                    nt_written += 1

            if not got_any:
                time.sleep(0.003)
            if time.time() - last_report >= 5:
                print(
                    f"[responder] {int(time.time() - t0)}s stats={stats} "
                    f"telem_lines={nt_written} resp_id={resp_id} "
                    f"cgt={cgt_measure_integrals} nimbers={cgt_nimbers}",
                    flush=True,
                )
                last_report = time.time()
    finally:
        j_reporter.stop()
        _, final_measures = j_reporter.snapshot()
        work_meter.close({**final_measures, **learning_measures})
        for arm, learning in learners.items():
            print(json.dumps({"event": "learning_truncation", "arm": arm, **learning.end_episode()}), flush=True)
            learning.save()
        if learners:
            save_runstate(rng, model_key, resp_id, nt_written, sum(learning.updates for learning in learners.values()))
        telem.close()
    print(f"[responder] STOPPED (signal {stopping['signal']}) stats={stats} "
          f"telem_lines={nt_written}", flush=True)

if __name__ == "__main__":
    main()
