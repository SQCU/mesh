import http.client, json, os, socket, threading, time

class WorkloadMeter:
    def __init__(self, name, labels=None, endpoint=None):
        self.name = str(name)
        self.labels = {} if labels is None else dict(labels)
        self.pid = os.getpid()
        self.started_at = time.time()
        self.sequence = 0
        address = endpoint or os.environ.get("MESH_TELEMETRY_PUBLISH", "127.0.0.1:8788")
        host, port = address.rsplit(":", 1)
        self.address = (host, int(port))
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setblocking(False)
        self.measure_condition = threading.Condition()
        self.pending_measures = None
        self.measure_thread = None
        self.measure_stopping = False

    def publish_measures(self, measures):
        payload = {
            "schema": 1,
            "pid": self.pid,
            "name": self.name,
            "labels": self.labels,
            "started_at": self.started_at,
            "sampled_at": time.time(),
            "measures": dict(measures),
        }
        with self.measure_condition:
            self.pending_measures = payload
            if self.measure_thread is None:
                self.measure_thread = threading.Thread(
                    target=self.measure_loop, name="mesh-workload-measures", daemon=True,
                )
                self.measure_thread.start()
            self.measure_condition.notify()

    def measure_loop(self):
        payload = None
        while True:
            with self.measure_condition:
                while self.pending_measures is None and payload is None:
                    if self.measure_stopping:
                        return
                    self.measure_condition.wait()
                if self.pending_measures is not None:
                    payload = self.pending_measures
                    self.pending_measures = None
            body = json.dumps(payload, separators=(",", ":")).encode()
            try:
                connection = http.client.HTTPConnection(self.address[0], self.address[1], timeout=0.5)
                connection.request("POST", "/v1/workload-measures", body, {"Content-Type": "application/json"})
                response = connection.getresponse()
                response.read()
                connection.close()
                if response.status < 300:
                    payload = None
            except Exception as error:
                payload["delivery_error"] = f"{type(error).__name__}: {error}"
                payload["delivery_attempts"] = int(payload.get("delivery_attempts", 0)) + 1
            with self.measure_condition:
                if self.measure_stopping:
                    if self.pending_measures is None:
                        return
                    payload = self.pending_measures
                    self.pending_measures = None
                    continue
            if payload is not None:
                with self.measure_condition:
                    self.measure_condition.wait(1.0)

    def close(self, measures=None):
        if measures is not None:
            self.publish_measures(measures)
        with self.measure_condition:
            self.measure_stopping = True
            self.measure_condition.notify_all()
        if self.measure_thread is not None:
            self.measure_thread.join(timeout=1.5)
        self.socket.close()

    def record(self, elapsed_s, flops_lower, flops_upper, bytes_lower, bytes_upper,
               deadline_s=None, rows=None, operations=None, measures=None):
        elapsed = float(elapsed_s)
        lower_flops = None if flops_lower is None else float(flops_lower)
        upper_flops = None if flops_upper is None else float(flops_upper)
        lower_bytes = None if bytes_lower is None else float(bytes_lower)
        upper_bytes = None if bytes_upper is None else float(bytes_upper)
        self.sequence += 1
        payload = {
            "schema": 1,
            "pid": self.pid,
            "name": self.name,
            "labels": self.labels,
            "started_at": self.started_at,
            "sampled_at": time.time(),
            "sequence": self.sequence,
            "elapsed_s": elapsed,
            "deadline_s": None if deadline_s is None else float(deadline_s),
            "rows": None if rows is None else int(rows),
            "operations": {} if operations is None else dict(operations),
            "fp32": {
                "lower_flops": lower_flops,
                "upper_flops": upper_flops,
                "lower_gflops_s": lower_flops / elapsed / 1e9 if elapsed and lower_flops is not None else None,
                "upper_gflops_s": upper_flops / elapsed / 1e9 if elapsed and upper_flops is not None else None,
            },
            "memory": {
                "lower_bytes": lower_bytes,
                "upper_bytes": upper_bytes,
                "lower_gbs": lower_bytes / elapsed / 1e9 if elapsed and lower_bytes is not None else None,
                "upper_gbs": upper_bytes / elapsed / 1e9 if elapsed and upper_bytes is not None else None,
            },
        }
        if measures is not None:
            self.publish_measures(measures)
        body = json.dumps(payload, separators=(",", ":")).encode()
        try:
            self.socket.sendto(body, self.address)
            return True
        except OSError:
            try:
                connection = http.client.HTTPConnection(self.address[0], self.address[1], timeout=0.2)
                connection.request("POST", "/v1/workload", body, {"Content-Type": "application/json"})
                response = connection.getresponse()
                response.read()
                connection.close()
                return response.status < 300
            except Exception:
                return False

    def span(self, stage, rows=None, operations=None, interval=2.0):
        return WorkloadSpan(self, stage, rows, operations, interval)

class WorkloadSpan:
    def __init__(self, meter, stage, rows, operations, interval):
        self.meter = meter
        self.stage = str(stage)
        self.rows = rows
        self.operations = {} if operations is None else dict(operations)
        self.interval = max(0.25, float(interval))
        self.started = None
        self.stop = threading.Event()
        self.thread = None

    def publish(self, terminal=False):
        elapsed = time.monotonic() - self.started
        operations = dict(self.operations, stage=self.stage, terminal=bool(terminal))
        return self.meter.record(elapsed, 0, None, 0, None, rows=self.rows,
                                 operations=operations)

    def run(self):
        while not self.stop.wait(self.interval):
            self.publish()

    def __enter__(self):
        return self.start()

    def start(self):
        self.started = time.monotonic()
        self.publish()
        self.thread = threading.Thread(target=self.run, name='mesh-workload-meter', daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, trace):
        self.finish()

    def finish(self):
        self.stop.set()
        self.thread.join(timeout=self.interval)
        self.publish(True)

__all__ = ["WorkloadMeter", "WorkloadSpan"]
