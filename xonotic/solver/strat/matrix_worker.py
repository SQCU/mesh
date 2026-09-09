import argparse, errno, fcntl, json, os, signal, socket, sys, time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', '..', '..', 'rdma'))

from solver.strat.runtime_transport import RuntimeTransport, report
from solver.strat.tensor_mesh import Worker
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
    parser.add_argument('--socket', default=os.environ.get('MESH_EXPERT_SOCKET', '/tmp/mesh-expert-worker.sock'))
    parser.add_argument('--environment', default='game2_server')
    parser.add_argument('--deadline', type=float, default=STRATEGY_DEADLINE_S)
    args = parser.parse_args()
    stopping = {'signal': None}
    def stop(signum, frame):
        stopping['signal'] = signum
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, stop)
    bound = bind_service(args.socket, stopping)
    if bound is None: return
    service, lock = bound
    transport = RuntimeTransport(service, lambda *values: worker.receive(*values))
    worker = Worker(transport, WorkloadMeter('xonotic.strategy.matrix',
        {'environment': args.environment, 'host_role': 'matrix', 'host': socket.gethostname()}))
    report('matrix_worker_live', socket=args.socket, transport_owner=True,
           application=json.loads(os.environ.get('MESH_APPLICATION_IDENTITY', '{"state":"unbundled"}')),
           backend='persistent_mesh_metal')
    status_at = 0
    numerical_error = None
    try:
        while stopping['signal'] is None:
            activity = transport.progress()
            try:
                worker.progress()
                numerical_error = None
            except Exception as error:
                detail = f'{type(error).__name__}: {error}'
                if detail != numerical_error:
                    report('numerical_progress_error', error=detail)
                    numerical_error = detail
            now = time.monotonic()
            if now >= status_at:
                report('runtime_progress', received_frames=transport.received, sent_frames=transport.sent,
                       game_clients=len(transport.clients), queued_network_frames=transport.queued(), **worker.report())
                status_at = now + 5
            if not activity: time.sleep(.0005)
    finally:
        worker.close()
        transport.close(time.monotonic() + 30)
        service.close()
        try:
            os.unlink(args.socket)
        except FileNotFoundError as error:
            report('socket_already_absent', path=args.socket, error=str(error))
        lock.close()
        report('matrix_worker_stopped', signal=stopping['signal'], **worker.report())


if __name__ == '__main__':
    main()
