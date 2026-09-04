import argparse, errno, fcntl, json, os, signal, socket, sys, time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "..", "rdma"))

from solver.strat.cast_header import Wally, parameter_seed, scale_fuse
from solver.strat.checkpoint_state import (
    ARCH_KEY, ARCH_SPEC_KEY, POLICY_KEY, POLICY_VERSION_KEY, REWARD_CONTRACT_KEY,
    POLICY_VERSIONS, architecture_fingerprint, architecture_spec,
    tensor_tree_measurement, whole_tensor_tree, load_module_checkpoint,
)
from solver.strat.runtime import SPARSE_REWARD_FINGERPRINT
from solver.strat.scale_config import (
    SCALE_EXPERTS, SCALE_HIDDEN, SCALE_RANK, SCALE_TOPK,
    scale_model_digest, strategy_widths,
)
from solver.strat.work_estimate import scale_work
from solver.xonwire import (
    EXPERT_BATCH_BEGIN, EXPERT_BATCH_COMMIT, EXPERT_BATCH_RESP,
    EXPERT_GRAD_META, EXPERT_GRAD_META_KIND, EXPERT_GRAD_META_WIDTH,
    EXPERT_GRAD_REQ, EXPERT_GRAD_RESP,
    EXPERT_META, EXPERT_META_VALUE_WIDTH, EXPERT_META_KIND, EXPERT_REQ,
    EXPERT_RESP, EXPERT_TRAIN_REQ, Reassembler, expert_meta_width,
    parse_hdr, recv_datagram_frames, send_datagram_rows,
)
from payload.tools.strategy_io_schema import STRATEGY_DEADLINE_S
from workload import WorkloadMeter

def bind_service(path, stopping):
    lock = open(path + ".lock", "a+b")
    while stopping["signal"] is None:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            print(json.dumps({"event": "worker_lock_occupied", "path": path}), flush=True)
            time.sleep(1)
    if stopping["signal"] is not None:
        lock.close()
        return None
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    while stopping["signal"] is None:
        service = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        service.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 * 1024 * 1024)
        service.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 16 * 1024 * 1024)
        try:
            service.bind(path)
            return service, lock
        except OSError as exc:
            service.close()
            print(json.dumps({"event": "bind_retry", "path": path, "error": str(exc)}), flush=True)
            time.sleep(1)
            if exc.errno == errno.EADDRINUSE:
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass
    lock.close()
    return None

def configure_scale_training(model):
    model.freeze()
    for name in ("scale_in", "scale_router", "scale_out"):
        getattr(model, name).unfreeze()
    for name in ("scale_w1", "scale_w2", "scale_probe"):
        model.unfreeze(keys=name, strict=True)

def load_optimizer(optimizer, model, path):
    optimizer.init(model.trainable_parameters())
    updates = 0
    live = dict(tree_flatten(optimizer.state))
    measurement = {
        "source_mass": 0, "live_mass": len(live), "source_only_mass": 0,
        "live_only_mass": len(live), "shape_difference_mass": 0,
        "nonfinite_mass": 0, "composable_mass": 0, "load_exception": None,
    }
    if not path or not os.path.isfile(path):
        return updates, measurement
    with np.load(path, allow_pickle=False) as data:
        source = [
            (name[len("__scale_opt__"):], np.asarray(data[name]))
            for name in data.files if name.startswith("__scale_opt__")
        ]
        measurement = tensor_tree_measurement(live.items(), source)
        measurement["load_exception"] = None
        try:
            source_state, _ = whole_tensor_tree(live.items(), source)
            optimizer.state = source_state
        except Exception as error:
            optimizer.state = tree_unflatten(list(live.items()))
            measurement["load_exception"] = f"{type(error).__name__}: {error}"
        if "__scale_updates__" in data.files:
            updates = int(np.asarray(data["__scale_updates__"]))
    return updates, measurement

def save_checkpoint(model, optimizer, path, updates):
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temporary = path + ".new.npz"
    payload = {name: np.asarray(value) for name, value in tree_flatten(model.parameters())}
    for name, value in tree_flatten(optimizer.state):
        payload["__scale_opt__" + name] = np.asarray(value)
    payload["__scale_updates__"] = np.asarray(updates)
    payload[ARCH_KEY] = np.asarray(architecture_fingerprint(model))
    payload[ARCH_SPEC_KEY] = np.asarray(json.dumps(architecture_spec(model), separators=(",", ":")))
    payload[POLICY_KEY] = np.asarray("matrix_fusion")
    payload[POLICY_VERSION_KEY] = np.asarray(POLICY_VERSIONS["matrix_fusion"])
    payload[REWARD_CONTRACT_KEY] = np.asarray(SPARSE_REWARD_FINGERPRINT)
    np.savez(temporary, **payload)
    os.replace(temporary, path)

def transmit_response(service, outbound):
    return [
        send_datagram_rows(
            service, outbound["address"], outbound["node"], outbound["usable"],
            kind, outbound["req_id"], outbound["tick"], rows,
        )
        for kind, rows in outbound["parts"]
    ]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", default=os.environ.get("MESH_EXPERT_SOCKET", "/tmp/mesh-expert-worker.sock"))
    parser.add_argument("--checkpoint")
    parser.add_argument("--output-checkpoint")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--scale-rank", type=int, default=SCALE_RANK)
    parser.add_argument("--scale-hidden", type=int, default=SCALE_HIDDEN)
    parser.add_argument("--scale-experts", type=int, default=SCALE_EXPERTS)
    parser.add_argument("--scale-topk", type=int, default=SCALE_TOPK)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--environment", default="game2_server")
    parser.add_argument("--deadline", type=float, default=STRATEGY_DEADLINE_S)
    args = parser.parse_args()
    stopping = {"signal": None}

    def stop(signum, _frame):
        stopping["signal"] = int(signum)

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, stop)

    bound = bind_service(args.socket, stopping)
    if bound is None:
        return
    service, service_lock = bound
    widths = strategy_widths(args.scale_rank, args.scale_hidden,
                             args.scale_experts, args.scale_topk)
    mx.random.seed(parameter_seed(args.seed, "matrix_fusion"))
    model = Wally(widths)
    checkpoint_measurement = load_module_checkpoint(
        model, args.checkpoint, "matrix_fusion", SPARSE_REWARD_FINGERPRINT,
    )
    configure_scale_training(model)
    optimizer = optim.AdamW(learning_rate=args.learning_rate, weight_decay=1e-4)
    updates, optimizer_measurement = load_optimizer(optimizer, model, args.checkpoint)
    initial_model_digest = scale_model_digest(model)
    work_meter = WorkloadMeter(
        "xonotic.strategy.scale",
        {"environment": args.environment, "host_role": "expert", "host": socket.gethostname()},
    )
    service.settimeout(args.deadline)
    reassemblers = None
    usable = 0
    completed = 0
    gradient_batch = 0
    gradient_atoms = 0
    gradient_accumulator = None
    outbound = None
    response_replays = 0
    print(json.dumps({"event": "expert_worker_live", "host": socket.gethostname(), "socket": args.socket,
                      "checkpoint": args.checkpoint, "initial_scale_model_digest": initial_model_digest,
                      "output_checkpoint": args.output_checkpoint,
                      "updates": updates,
                      "checkpoint_measurement": checkpoint_measurement,
                      "optimizer_moment_measurement": optimizer_measurement,
                      "rank": widths.d_scale, "hidden": widths.scale_h,
                      "experts": widths.scale_experts, "topk": widths.scale_topk}), flush=True)
    try:
        while stopping["signal"] is None:
            try:
                local = recv_datagram_frames(service)
            except socket.timeout:
                if outbound is not None:
                    try:
                        masses = transmit_response(service, outbound)
                        response_replays += 1
                        print(json.dumps({
                            "event": "scale_response_replay",
                            "req_id": outbound["req_id"],
                            "response_kinds": [kind for kind, _ in outbound["parts"]],
                            "response_frame_masses": masses,
                            "response_replays": response_replays,
                        }), flush=True)
                    except OSError as exc:
                        print(json.dumps({
                            "event": "scale_response_replay_error",
                            "req_id": outbound["req_id"],
                            "error": f"{type(exc).__name__}: {exc}",
                        }), flush=True)
                continue
            if local is None:
                continue
            node, address, frames = local
            if reassemblers is None or usable != frames.shape[1]:
                usable = frames.shape[1]
                reassemblers = {
                    EXPERT_REQ: Reassembler(EXPERT_REQ, widths.d_ir, usable),
                    EXPERT_TRAIN_REQ: Reassembler(EXPERT_TRAIN_REQ, widths.d_ir, usable),
                    EXPERT_GRAD_REQ: Reassembler(EXPERT_GRAD_REQ, 2 * widths.d_ir, usable),
                    EXPERT_BATCH_BEGIN: Reassembler(EXPERT_BATCH_BEGIN, 1, usable),
                    EXPERT_BATCH_COMMIT: Reassembler(EXPERT_BATCH_COMMIT, 1, usable),
                }
            header = next((parse_hdr(raw) for raw in frames if parse_hdr(raw) is not None), None)
            if header is None or header["kind"] not in reassemblers:
                continue
            reassembler = reassemblers[header["kind"]]
            record = None
            for raw in frames:
                record = reassembler.feed(raw) or record
            if record is None:
                continue
            outbound = None
            try:
                started = time.perf_counter()
                rows = record["rows"]
                work = scale_work(widths, rows)
                if header["kind"] in (EXPERT_REQ, EXPERT_TRAIN_REQ):
                    source = mx.array(reassembler.stage[:rows])
                    delta, stats, load = scale_fuse(
                        model, source.reshape(rows, 1, widths.d_ir), execute_remote=False,
                    )
                    mx.eval(delta, stats, load)
                    elapsed = time.perf_counter() - started
                    output = np.asarray(delta, dtype=np.float32).reshape(rows, widths.d_ir)
                    metadata = np.zeros((1, expert_meta_width(widths.scale_experts)), dtype=np.float32)
                    metadata[0, EXPERT_META["MATRIX_MIN"]:EXPERT_META["MATRIX_FINITE_MASS"] + 1] = np.asarray(stats, dtype=np.float32)
                    metadata[0, EXPERT_META["ROWS"]] = rows
                    metadata[0, EXPERT_META["ELAPSED"]] = elapsed
                    metadata[0, EXPERT_META_VALUE_WIDTH:] = np.asarray(load, dtype=np.float32)
                    outbound = {
                        "address": address, "node": node, "usable": usable,
                        "req_id": record["req_id"], "tick": record["tick"],
                        "parts": ((EXPERT_RESP, output), (EXPERT_META_KIND, metadata)),
                    }
                    response_frame_mass, metadata_frame_mass = transmit_response(service, outbound)
                    completed += 1
                    training_forward = header["kind"] == EXPERT_TRAIN_REQ
                    event = "scale_train_forward_complete" if training_forward else "scale_complete"
                    print(json.dumps({"event": event, "req_id": record["req_id"],
                                      "rows": rows, "elapsed_s": elapsed,
                                      "response_frame_mass": response_frame_mass,
                                      "metadata_frame_mass": metadata_frame_mass,
                                      "initial_scale_model_digest": initial_model_digest,
                                      "completed": completed, "updates": updates}), flush=True)
                    operation = "training_forward" if training_forward else "response_forward"
                    multiplier = 1
                elif header["kind"] == EXPERT_GRAD_REQ:
                    packed = mx.array(reassembler.stage[:rows])
                    source = packed[:, :widths.d_ir]
                    cotangent = packed[:, widths.d_ir:]

                    def forward(value):
                        return scale_fuse(
                            model, value.reshape(rows, 1, widths.d_ir), execute_remote=False,
                        )[0].reshape(rows, widths.d_ir)

                    def loss_fn():
                        return mx.sum(forward(source) * cotangent)

                    _, gradients = nn.value_and_grad(model, loss_fn)()
                    _, input_gradients = mx.vjp(forward, (source,), (cotangent,))
                    mx.eval(input_gradients[0], gradients)
                    if gradient_batch != record["tick"]:
                        gradient_batch = record["tick"]
                        gradient_atoms = 0
                        gradient_accumulator = None
                    flattened = tree_flatten(gradients)
                    if gradient_accumulator is None:
                        gradient_accumulator = [(name, value) for name, value in flattened]
                    else:
                        previous = dict(gradient_accumulator)
                        gradient_accumulator = [(name, previous[name] + value) for name, value in flattened]
                    gradient_atoms += 1
                    gradient_norm = mx.sqrt(sum(mx.sum(value * value) for _, value in flattened))
                    mx.eval(gradient_norm)
                    elapsed = time.perf_counter() - started
                    output = np.asarray(input_gradients[0], dtype=np.float32).reshape(rows, widths.d_ir)
                    metadata = np.zeros((1, EXPERT_GRAD_META_WIDTH), dtype=np.float32)
                    metadata[0, EXPERT_GRAD_META["ROWS"]] = rows
                    metadata[0, EXPERT_GRAD_META["ELAPSED"]] = elapsed
                    metadata[0, EXPERT_GRAD_META["GRADIENT_NORM"]] = float(np.asarray(gradient_norm))
                    metadata[0, EXPERT_GRAD_META["UPDATES"]] = updates
                    outbound = {
                        "address": address, "node": node, "usable": usable,
                        "req_id": record["req_id"], "tick": record["tick"],
                        "parts": ((EXPERT_GRAD_RESP, output), (EXPERT_GRAD_META_KIND, metadata)),
                    }
                    response_frame_mass, metadata_frame_mass = transmit_response(service, outbound)
                    print(json.dumps({"event": "scale_backward_complete", "req_id": record["req_id"],
                                      "rows": rows, "elapsed_s": elapsed,
                                      "gradient_norm": float(np.asarray(gradient_norm)),
                                      "response_frame_mass": response_frame_mass,
                                      "metadata_frame_mass": metadata_frame_mass,
                                      "updates": updates, "gradient_atoms": gradient_atoms,
                                      "gradient_batch": gradient_batch}), flush=True)
                    operation = "backward_accumulate"
                    multiplier = 2
                else:
                    batch = int(reassembler.stage[0, 0])
                    gradient_norm = mx.array(0.0)
                    committed_atoms = 0
                    if header["kind"] == EXPERT_BATCH_BEGIN:
                        gradient_batch = batch
                        gradient_atoms = 0
                        gradient_accumulator = None
                        operation = "gradient_batch_begin"
                    else:
                        committed_atoms = gradient_atoms if gradient_batch == batch else 0
                        if committed_atoms and gradient_accumulator is not None:
                            accumulated = tree_unflatten([
                                (name, value)
                                for name, value in gradient_accumulator
                            ])
                            accumulated, gradient_norm = optim.clip_grad_norm(
                                accumulated, abs(args.gradient_clip),
                            )
                            optimizer.update(model, accumulated)
                            mx.eval(model.parameters(), optimizer.state, gradient_norm)
                            updates += 1
                            if args.output_checkpoint and updates % max(1, args.save_every) == 0:
                                save_checkpoint(model, optimizer, args.output_checkpoint, updates)
                        gradient_atoms = 0
                        gradient_accumulator = None
                        operation = "gradient_batch_commit"
                    elapsed = time.perf_counter() - started
                    metadata = np.asarray([[
                        batch, committed_atoms, float(np.asarray(gradient_norm)), updates,
                    ]], dtype=np.float32)
                    outbound = {
                        "address": address, "node": node, "usable": usable,
                        "req_id": record["req_id"], "tick": record["tick"],
                        "parts": ((EXPERT_BATCH_RESP, metadata),),
                    }
                    response_frame_mass = transmit_response(service, outbound)[0]
                    print(json.dumps({
                        "event": operation, "gradient_batch": batch,
                        "gradient_atoms": committed_atoms, "updates": updates,
                        "gradient_norm": float(np.asarray(gradient_norm)),
                        "elapsed_s": elapsed, "response_frame_mass": response_frame_mass,
                    }), flush=True)
                    multiplier = 0
                work_meter.record(
                    elapsed, multiplier * work["lower_flops"], multiplier * work["upper_flops"],
                    multiplier * work["lower_bytes"], multiplier * work["upper_bytes"],
                    deadline_s=args.deadline if operation == "response_forward" else None,
                    rows=rows,
                    operations={
                        "host_role": "expert", "operation": operation,
                        "residual_rows": rows, "residual_rank": widths.d_scale,
                        "experts": widths.scale_experts, "topk": widths.scale_topk,
                        "parameter_bytes": work["parameter_bytes"], "scale_updates": updates,
                        "checkpoint_source_weight_mass": checkpoint_measurement["source_weight_mass"],
                        "checkpoint_live_weight_mass": checkpoint_measurement["live_weight_mass"],
                        "checkpoint_loaded_weight_mass": checkpoint_measurement["loaded_weight_mass"],
                        "checkpoint_composable_weight_mass": checkpoint_measurement["composable_weight_mass"],
                        "checkpoint_shape_difference_mass": checkpoint_measurement["shape_difference_mass"],
                        "checkpoint_nonfinite_weight_mass": checkpoint_measurement["nonfinite_weight_mass"],
                        "initial_scale_model_digest": initial_model_digest,
                    },
                )
            except Exception as exc:
                print(json.dumps({"event": "scale_error", "req_id": record["req_id"],
                                  "rows": record["rows"], "error": f"{type(exc).__name__}: {exc}"}), flush=True)
    finally:
        save_checkpoint(model, optimizer, args.output_checkpoint, updates)
        service.close()
        service_lock.close()
        try:
            os.unlink(args.socket)
        except FileNotFoundError as error:
            print(json.dumps({"event":"socket_already_absent","path":args.socket,"error":str(error)}), flush=True)
        print(json.dumps({"event": "expert_worker_stopped", "signal": stopping["signal"],
                          "completed": completed, "updates": updates,
                          "response_replays": response_replays,
                          "output_checkpoint": args.output_checkpoint}), flush=True)

if __name__ == "__main__":
    main()
