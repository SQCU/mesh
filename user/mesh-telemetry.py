#!/usr/bin/env mesh-python
import collections, ctypes, json, math, os, platform, plistlib, re, signal, socket, subprocess, sys, threading, time, urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RATE = max(100, int(os.environ.get("MESH_TELEMETRY_RATE", "1000")))
BANDWIDTH_RATE = max(100, int(os.environ.get("MESH_BANDWIDTH_RATE", str(RATE))))
PORT = int(os.environ.get("MESH_TELEMETRY_PORT", "8788"))
RING = max(2, int(os.environ.get("MESH_TELEMETRY_RING", "4096")))
LEASE = max(RATE / 1000 * 2, float(os.environ.get("MESH_WORKLOAD_LEASE", "10")))
STOP = threading.Event()
POWER = None
BANDWIDTH = None
CPU_TICKS = None

def number(line, pattern):
    match = re.search(pattern, line, re.I)
    return float(match.group(1)) if match else None

def numeric_sum(values):
    values = [
        float(value) for value in values
        if isinstance(value, (int, float)) and math.isfinite(value)
    ]
    return sum(values) if values else None

def parse(lines):
    metrics = {}
    clusters = {}
    for line in lines:
        value = number(line, r"^CPU Power:\s*([0-9.]+) mW")
        if value is not None: metrics["cpu_power_w"] = value / 1000
        value = number(line, r"^GPU Power:\s*([0-9.]+) mW")
        if value is not None: metrics["gpu_power_w"] = value / 1000
        value = number(line, r"^ANE Power:\s*([0-9.]+) mW")
        if value is not None: metrics["ane_power_w"] = value / 1000
        value = number(line, r"^Combined Power.*:\s*([0-9.]+) mW")
        if value is not None: metrics["combined_power_w"] = value / 1000
        value = number(line, r"^GPU HW active residency:\s*([0-9.]+)%")
        if value is not None: metrics["gpu_active_pct"] = value
        value = number(line, r"^GPU HW active frequency:\s*([0-9.]+) MHz")
        if value is not None: metrics["gpu_frequency_mhz"] = value
        match = re.search(r"^([^:]*Cluster) (?:HW )?active residency:\s*([0-9.]+)%", line, re.I)
        if match: clusters.setdefault(match.group(1).strip(), {})["active_pct"] = float(match.group(2))
        match = re.search(r"^([^:]*Cluster) (?:HW )?active frequency:\s*([0-9.]+) MHz", line, re.I)
        if match: clusters.setdefault(match.group(1).strip(), {})["frequency_mhz"] = float(match.group(2))
        match = re.search(r"^(?:Thermal pressure|Current pressure level):\s*(.+?)\s*$", line, re.I)
        if match: metrics["thermal_pressure"] = match.group(1)
    if clusters: metrics["cpu_clusters"] = clusters
    return metrics

def output(command, timeout=3):
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except Exception:
        return ""

def executable(paths):
    return next((path for path in paths if path and os.access(path, os.X_OK)), None)

def document(paths):
    for path in paths:
        if not path: continue
        try:
            with open(path) as stream: return json.load(stream), path
        except Exception as error:
            print(json.dumps({"event":"document_read_error","path":path,"error":f"{type(error).__name__}: {error}"}), file=sys.stderr, flush=True)
    return None, None

def gpu():
    try:
        data = subprocess.run(["/usr/sbin/ioreg", "-r", "-c", "AGXAccelerator", "-d", "1", "-a"], capture_output=True, timeout=3).stdout
        stats = (plistlib.loads(data)[0].get("PerformanceStatistics") or {})
        metrics = {
            "gpu_active_pct": stats.get("Device Utilization %"),
            "gpu_renderer_pct": stats.get("Renderer Utilization %"),
            "gpu_tiler_pct": stats.get("Tiler Utilization %"),
            "gpu_memory_bytes": stats.get("In use system memory"),
            "gpu_allocated_bytes": stats.get("Alloc system memory"),
        }
        return {key: value for key, value in metrics.items() if value is not None}
    except Exception:
        return {}

def cpu():
    global CPU_TICKS
    try:
        library = ctypes.CDLL(None)
        library.mach_host_self.restype = ctypes.c_uint32
        ticks = (ctypes.c_uint32 * 4)()
        count = ctypes.c_uint32(4)
        status = library.host_statistics(
            library.mach_host_self(), 3, ctypes.byref(ticks), ctypes.byref(count),
        )
        if status != 0 or count.value < 4:
            return {}, f"host_statistics status {status} count {count.value}"
        current = tuple(int(value) for value in ticks)
        previous, CPU_TICKS = CPU_TICKS, current
        if previous is None:
            return {}, None
        delta = [((current[index] - previous[index]) & 0xffffffff) for index in range(4)]
        total = sum(delta)
        active = delta[0] + delta[1] + delta[3]
        return ({"cpu_active_pct": 100.0 * active / total} if total else {}), None
    except Exception as error:
        return {}, f"{type(error).__name__}: {error}"

def bandwidth():
    path = executable([
        os.environ.get("MESH_BANDWIDTH", ""),
        os.path.join(ROOT, "user", "mesh-bandwidth"),
        os.path.join(HERE, "mesh-bandwidth"),
        "/usr/local/mesh/bin/mesh-bandwidth",
    ])
    if path:
        try: return json.loads(output([path, "-i", "200"]))
        except Exception as error: return {"up": False, "error": f"memory bandwidth sampler: {type(error).__name__}: {error}"}
    return {"up": False, "error": "memory bandwidth sampler unavailable"}

def bridge():
    path = executable([
        os.environ.get("MESH_STAT", ""),
        os.path.join(ROOT, "rdma", "mesh-stat"),
        os.path.join(HERE, "mesh-stat"),
        "/usr/local/mesh/bin/mesh-stat",
    ])
    if path:
        try: return json.loads(output([path]).splitlines()[-1])
        except Exception as error: return {"up": False, "error": f"mesh-stat: {type(error).__name__}: {error}"}
    return {"up": False, "error": "mesh-stat unavailable"}

class TelemetryRing:
    def __init__(self, size=RING, lease=LEASE):
        self.records = collections.deque(maxlen=size)
        self.lease = lease
        self.sequence = 0
        self.lock = threading.RLock()
        self.power = {"up": False, "sampled_at": None, "source": "powermetrics", "error": "sampler starting", "metrics": {}}
        self.bandwidth = {"up": False, "source": "IOReport", "error": "sampler starting"}
        self.producers = {}
        self.unkeyed_producers = collections.deque(maxlen=size)
        self.protocol_events = collections.deque(maxlen=size)
        self.ingest_sequence = 0
        self.expired_producers = 0

    def set_power(self, payload):
        with self.lock: self.power = dict(payload)

    def power_sample(self):
        with self.lock: return dict(self.power)

    def set_bandwidth(self, payload):
        with self.lock: self.bandwidth = dict(payload)

    def bandwidth_sample(self):
        with self.lock: return dict(self.bandwidth)

    def publish_protocol(self, payload):
        with self.lock: self.protocol_events.append(dict(payload))

    def protocol_sample(self):
        with self.lock: return list(self.protocol_events)[-16:]

    def publish_workload(self, payload):
        with self.lock:
            self.ingest_sequence += 1
            sequence = self.ingest_sequence
        try:
            key = (int(payload["pid"]), float(payload["started_at"]), str(payload["name"]))
            sampled = float(payload["sampled_at"])
            with self.lock:
                current = self.producers.get(key)
                if current is None or sampled >= float(current.get("sampled_at", 0)):
                    row = dict(payload)
                    if "measures" not in row and current is not None:
                        row["measures"] = current.get("measures", {})
                    self.producers[key] = row
        except Exception:
            with self.lock:
                self.unkeyed_producers.append({
                    "sequence": sequence, "received_at": time.time(), "payload": payload,
                })
        return sequence

    def publish_measures(self, payload):
        with self.lock:
            self.ingest_sequence += 1
            sequence = self.ingest_sequence
            key = (int(payload["pid"]), float(payload["started_at"]), str(payload["name"]))
            current = dict(self.producers.get(key) or payload)
            current["measures"] = dict(payload.get("measures") or {})
            self.producers[key] = current
            return sequence

    def workload_sample(self, now=None):
        now = time.time() if now is None else float(now)
        with self.lock:
            expired = [key for key, row in self.producers.items() if now - float(row.get("sampled_at", 0)) > self.lease]
            for key in expired: self.producers.pop(key, None)
            self.expired_producers += len(expired)
            active = [dict(row) for row in self.producers.values()]
            unkeyed = [dict(row) for row in self.unkeyed_producers]
            expired_total = self.expired_producers
        deadline_slack = [
            float(row["deadline_s"]) - float(row["elapsed_s"])
            for row in active if row.get("deadline_s") is not None
        ]
        return {
            "up": bool(active),
            "sampled_at": max([row.get("sampled_at", 0) for row in active], default=None),
            "active_producers": len(active),
            "expired_producers": expired_total,
            "unkeyed_producers": unkeyed,
            "rows": numeric_sum(row.get("rows") for row in active),
            "fp32": {
                "lower_gflops_s": numeric_sum((row.get("fp32") or {}).get("lower_gflops_s") for row in active),
                "upper_gflops_s": numeric_sum((row.get("fp32") or {}).get("upper_gflops_s") for row in active),
            },
            "memory": {
                "lower_gbs": numeric_sum((row.get("memory") or {}).get("lower_gbs") for row in active),
                "upper_gbs": numeric_sum((row.get("memory") or {}).get("upper_gbs") for row in active),
            },
            "deadlines": {
                "observations": len(deadline_slack),
                "slack_s": deadline_slack,
                "minimum_slack_s": min(deadline_slack, default=None),
                "mean_slack_s": sum(deadline_slack) / len(deadline_slack) if deadline_slack else None,
            },
            "producers": active,
        }

    def append(self, sample):
        with self.lock:
            self.sequence += 1
            sample = dict(sample)
            sample["stream"] = {"schema": 1, "sequence": self.sequence, "sampled_at": time.time(), "monotonic_ns": time.monotonic_ns()}
            record = dict(sample["stream"])
            record["sample"] = sample
            self.records.append(record)
            return record

    def latest(self):
        with self.lock:
            record = dict(self.records[-1]) if self.records else None
            oldest = self.records[0]["sequence"] if self.records else None
            latest = self.records[-1]["sequence"] if self.records else None
            return {"schema": 1, "capacity": self.records.maxlen, "oldest_sequence": oldest, "latest_sequence": latest, "record": record}

    def history(self, since=0):
        since = int(since)
        with self.lock:
            oldest = self.records[0]["sequence"] if self.records else None
            latest = self.records[-1]["sequence"] if self.records else None
            records = [dict(record) for record in self.records if record["sequence"] > since]
            gap = oldest is not None and since > 0 and since < oldest - 1
            reset = latest is not None and since > latest
            return {"schema": 1, "capacity": self.records.maxlen, "since": since, "oldest_sequence": oldest, "latest_sequence": latest, "gap": gap, "reset": reset, "records": records}

def host_facts():
    model = output(["/usr/sbin/sysctl", "-n", "hw.model"])
    memory = output(["/usr/sbin/sysctl", "-n", "hw.memsize"])
    name = output(["/usr/sbin/scutil", "--get", "LocalHostName"]) or socket.gethostname().split(".")[0]
    capacity_doc, capacity_path = document([
        os.environ.get("MESH_CAPACITY", ""),
        os.path.join(ROOT, "etc", "mesh-capacity.json"),
        "/usr/local/mesh/etc/mesh-capacity.json",
    ])
    return {
        "name": name,
        "host": {"model": model, "memory_bytes": int(memory) if memory.isdigit() else None, "system": platform.platform()},
        "capacity": {"kind": "characterized", "source": (capacity_doc or {}).get("source"), "path": capacity_path, "metrics": ((capacity_doc or {}).get("models") or {}).get(model)},
    }

def snapshot(store, facts=None):
    facts = host_facts() if facts is None else facts
    stat = bridge()
    power = store.power_sample()
    ios = gpu()
    cpu_metrics, cpu_error = cpu()
    bw = store.bandwidth_sample()
    workload = store.workload_sample()
    metrics = dict(power.get("metrics") or {})
    metrics.update(ios)
    metrics.update(cpu_metrics)
    if bw.get("up"):
        for key, source in {
            "memory_read": "AMCC RD", "memory_write": "AMCC WR", "memory_total": "AMCC RD+WR",
            "gpu_memory_read": "AGX RD", "gpu_memory_write": "AGX WR", "gpu_memory_total": "AGX RD+WR",
        }.items():
            for field, value in (bw.get(source) or {}).items(): metrics[key + "_" + field] = value
    sampled_at = power.get("sampled_at")
    machine = {
        "up": bool(metrics),
        "source": [source for source, present in (("powermetrics", bool(power.get("up"))), ("mach", bool(cpu_metrics)), ("ioreg", bool(ios)), ("IOReport", bool(bw.get("up")))) if present],
        "metrics": metrics,
        "bandwidth": bw,
        "age_s": round(max(0.0, time.time() - float(sampled_at)), 3) if sampled_at else None,
    }
    if not power.get("up"): machine["sampler_error"] = power.get("error") or "powermetrics unavailable"
    if cpu_error is not None: machine["cpu_sampler_error"] = cpu_error
    result = dict(stat)
    result.update({
        "schema": 2,
        "observed_at": time.time(),
        "reachable": True,
        "name": facts["name"],
        "host": facts["host"],
        "capacity": facts["capacity"],
        "machine": machine,
        "workload": workload,
        "service": {"protocol_events": store.protocol_sample()},
        "availability": {
            "fabric": "measured" if stat.get("up") else "unavailable",
            "machine": "measured" if machine["up"] else "unavailable",
            "gpu_utilization": "measured_ioreg" if ios else "unavailable",
            "achieved_flops": "bounded_published_workload" if workload.get("up") else "unavailable_without_workload_semantics",
            "memory_bandwidth": "bounded_published_workload_and_hardware_histogram" if workload.get("up") and bw.get("up") else "bounded_published_workload" if workload.get("up") else "bounded_hardware_histogram" if bw.get("up") else "unavailable_without_hardware_byte_counter",
        },
    })
    return result

def power_loop(store):
    global POWER
    command = ["/usr/bin/powermetrics", "-n", "-1", "-i", str(RATE), "-b", "1", "-s", "cpu_power,gpu_power,ane_power,thermal", "--handle-invalid-values"]
    if os.geteuid() != 0:
        command = ["/usr/bin/sudo", "-n"] + command
    while not STOP.is_set():
        try:
            POWER = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            block = []
            produced = False
            for line in POWER.stdout:
                if STOP.is_set(): break
                if line.startswith("*** Sampled system activity") and block:
                    metrics = parse(block)
                    store.set_power({"schema": 1, "up": bool(metrics), "sampled_at": time.time(), "sample_ms": RATE, "source": "powermetrics", "metrics": metrics})
                    produced = produced or bool(metrics)
                    block = []
                block.append(line.rstrip())
            if block and not STOP.is_set():
                metrics = parse(block)
                store.set_power({"schema": 1, "up": bool(metrics), "sampled_at": time.time(), "sample_ms": RATE, "source": "powermetrics", "metrics": metrics})
                produced = produced or bool(metrics)
            if not STOP.is_set() and not produced:
                error = next((line.strip() for line in reversed(block) if line.strip()), "sampler exited")
                store.set_power({"schema": 1, "up": False, "sampled_at": time.time(), "source": "powermetrics", "error": error, "metrics": {}})
        except Exception as error:
            store.set_power({"schema": 1, "up": False, "sampled_at": time.time(), "source": "powermetrics", "error": type(error).__name__, "metrics": {}})
        STOP.wait(2)

def bandwidth_loop(store):
    global BANDWIDTH
    while not STOP.is_set():
        path = executable([
            os.environ.get("MESH_BANDWIDTH", ""),
            os.path.join(ROOT, "user", "mesh-bandwidth"),
            os.path.join(HERE, "mesh-bandwidth"),
            "/usr/local/mesh/bin/mesh-bandwidth",
        ])
        if not path:
            store.set_bandwidth({"up": False, "source": "IOReport", "error": "memory bandwidth sampler unavailable"})
            STOP.wait(2)
            continue
        try:
            BANDWIDTH = subprocess.Popen([path, "-s", str(BANDWIDTH_RATE)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in BANDWIDTH.stdout:
                if STOP.is_set(): break
                try: store.set_bandwidth(json.loads(line))
                except Exception: store.set_bandwidth({"up": False, "source": "IOReport", "error": "invalid sampler record"})
            if not STOP.is_set(): store.set_bandwidth({"up": False, "source": "IOReport", "error": "sampler exited"})
        except Exception as error:
            store.set_bandwidth({"up": False, "source": "IOReport", "error": type(error).__name__})
        STOP.wait(2)

def sample_loop(store):
    facts = host_facts()
    period = RATE / 1000
    while not STOP.is_set():
        started = time.monotonic()
        store.append(snapshot(store, facts))
        STOP.wait(max(0.01, period - (time.monotonic() - started)))

def ingest_loop(store):
    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", PORT))
    server.settimeout(1)
    while not STOP.is_set():
        payload = b""
        try:
            payload, _ = server.recvfrom(65535)
            store.publish_workload(json.loads(payload))
        except socket.timeout:
            continue
        except Exception:
            store.publish_workload({"source": "udp", "received_at": time.time(), "payload_bytes": len(payload)})
    server.close()

class TelemetryHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def send_json(self, payload, status=200):
        body = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        target = urllib.parse.urlsplit(self.path)
        if target.path in ("/", "/v1/latest"):
            self.send_json(self.server.store.latest())
            return
        if target.path == "/v1/history":
            query = urllib.parse.parse_qs(target.query)
            try: since = int((query.get("since") or [0])[0])
            except Exception: since = 0
            self.send_json(self.server.store.history(since))
            return
        self.send_json({"schema": 1, "error": "unknown endpoint", "latest": "/v1/latest", "history": "/v1/history?since=SEQUENCE"}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        try:
            payload = json.loads(body)
        except Exception as error:
            payload = {"source": "http", "body": body.decode(errors="replace"),
                       "parse_error": type(error).__name__}
        sequence = (
            self.server.store.publish_measures(payload)
            if urllib.parse.urlsplit(self.path).path == "/v1/workload-measures"
            else self.server.store.publish_workload(payload)
        )
        self.send_json({"schema": 1, "ingest_sequence": sequence}, 202)

    def log_message(self, *args):
        self.server.store.publish_protocol({"at":time.time(),"client":self.client_address[0],"message":args[0] % args[1:]})

class TelemetryHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, handler, store):
        self.store = store
        super().__init__(address, handler)

class TelemetryHTTPServer6(TelemetryHTTPServer):
    address_family = socket.AF_INET6

    def server_bind(self):
        try: self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        except OSError as error: self.store.publish_protocol({"at":time.time(),"stage":"dual_stack","error":f"{type(error).__name__}: {error}"})
        super().server_bind()

def stop(*_):
    STOP.set()
    if POWER and POWER.poll() is None: POWER.terminate()
    if BANDWIDTH and BANDWIDTH.poll() is None: BANDWIDTH.terminate()

def serve():
    store = TelemetryRing()
    for target in (power_loop, bandwidth_loop, sample_loop, ingest_loop): threading.Thread(target=target, args=(store,), daemon=True).start()
    host = os.environ.get("MESH_TELEMETRY_HOST", "::")
    server_class = TelemetryHTTPServer6 if ":" in host else TelemetryHTTPServer
    server = server_class((host, PORT), TelemetryHandler, store)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    while not STOP.is_set():
        STOP.wait(1)
    server.shutdown()
    server.server_close()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--parse": print(json.dumps(parse(sys.stdin.read().splitlines()), separators=(",", ":")))
    elif len(sys.argv) > 1 and sys.argv[1] == "--once":
        store = TelemetryRing()
        store.set_bandwidth(bandwidth())
        print(json.dumps(snapshot(store), separators=(",", ":")))
    else:
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        serve()
