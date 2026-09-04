import argparse, errno, fcntl, json, os, signal, socket, sys, time

import mlx.core as mx
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "..", "rdma"))

from solver.strat.matmul import gram_context
from solver.strat.work_estimate import gram_work
from solver.xonwire import (
    GRAM_REQ, GRAM_RESP, GRAM_GRAD_REQ, GRAM_GRAD_RESP,
    GRAM_META_KIND, Reassembler, parse_hdr,
    recv_datagram_frames, send_datagram_rows,
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", default=os.environ.get("MESH_EXPERT_SOCKET", "/tmp/mesh-expert-worker.sock"))
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
    service.settimeout(args.deadline)
    meter = WorkloadMeter(
        "xonotic.strategy.gram",
        {"environment": args.environment, "host_role": "matrix", "host": socket.gethostname()},
    )
    receiver = None
    outbound = None
    completed = response_replays = 0

    def transmit():
        return [
            send_datagram_rows(
                service, outbound["address"], outbound["node"], outbound["usable"],
                kind, outbound["header"]["req_id"], outbound["header"]["tick"], rows,
                session=outbound["header"]["session"],
                cancel=lambda: stopping["signal"] is not None,
            )
            for kind, rows in outbound["parts"]
        ]

    print(json.dumps({"event": "matrix_worker_live", "socket": args.socket,
                      "operation": "gram_context", "parameter_mass": 0}), flush=True)
    try:
        while stopping["signal"] is None:
            try:
                local = recv_datagram_frames(service)
                if local is None:
                    continue
                node, address, frames = local
                header = parse_hdr(frames[0])
                if header is None or header["kind"] not in (GRAM_REQ, GRAM_GRAD_REQ):
                    continue
                identity = (node, address, header["session"], header["req_id"], header["tick"], header["kind"])
                if outbound is not None and outbound["identity"] == identity:
                    if header["offset"] == 0:
                        transmit()
                        response_replays += 1
                    continue
                layout = (header["kind"], header["width"], frames.shape[1])
                if receiver is None or layout != receiver[0] or receiver[1] != (node, address):
                    receiver = (layout, (node, address), Reassembler(*layout))
                assembler = receiver[2]
                record = None
                for frame in frames:
                    record = assembler.feed(frame) or record
                if record is None:
                    continue
                started = time.perf_counter()
                packed = mx.array(assembler.stage[:record["rows"]])
                backward = header["kind"] == GRAM_GRAD_REQ
                source, probe = packed[2 if backward else 1:], packed[0]
                if backward:
                    (_, statistics), (row_gradient, probe_gradient) = mx.vjp(
                        gram_context, (source, probe), (packed[1], mx.zeros((3,))),
                    )
                    output = mx.concatenate((probe_gradient[None, :], row_gradient), axis=0)
                    response_kind = GRAM_GRAD_RESP
                else:
                    context, statistics = gram_context(source, probe)
                    output = context[None, :]
                    response_kind = GRAM_RESP
                mx.eval(output, statistics)
                elapsed = time.perf_counter() - started
                metadata = np.asarray([[*np.asarray(statistics), len(source), elapsed]], dtype=np.float32)
                outbound = {
                    "identity": identity, "address": address, "node": node,
                    "usable": frames.shape[1], "header": header,
                    "parts": ((response_kind, np.asarray(output)), (GRAM_META_KIND, metadata)),
                }
                frame_masses = transmit()
                completed += 1
                work = gram_work(source.shape[1], len(source))
                multiplier = 3 if backward else 1
                meter.record(
                    elapsed, multiplier * work["lower_flops"], multiplier * work["upper_flops"],
                    multiplier * work["lower_bytes"], multiplier * work["upper_bytes"],
                    deadline_s=args.deadline if not backward and header["tick"] else None,
                    rows=len(source),
                    operations={"host_role": "matrix", "operation": "gram_vjp" if backward else "gram_context",
                                "residual_rows": len(source), "residual_rank": source.shape[1],
                                "parameter_bytes": 0},
                )
                print(json.dumps({"event": "gram_backward_complete" if backward else "gram_complete",
                                  "req_id": header["req_id"], "rows": len(source), "elapsed_s": elapsed,
                                  "response_frame_masses": frame_masses, "completed": completed}), flush=True)
            except socket.timeout:
                if outbound is not None:
                    try:
                        transmit()
                        response_replays += 1
                    except OSError as error:
                        print(json.dumps({"event": "gram_replay_error", "error": str(error)}), flush=True)
            except Exception as error:
                print(json.dumps({"event": "gram_error", "error": f"{type(error).__name__}: {error}"}), flush=True)
    finally:
        service.close()
        service_lock.close()
        try:
            os.unlink(args.socket)
        except FileNotFoundError as error:
            print(json.dumps({"event": "socket_already_absent", "path": args.socket, "error": str(error)}), flush=True)
        print(json.dumps({"event": "matrix_worker_stopped", "signal": stopping["signal"],
                          "completed": completed, "response_replays": response_replays}), flush=True)

if __name__ == "__main__":
    main()
