#!/usr/bin/env mesh-python
import collections, copy, ctypes, json, math, os, platform, plistlib, re, shutil, signal, socket, subprocess, sys, threading, time, urllib.parse
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

CENSUS = re.compile(r"link (\d+) census queue=(\d+) requests=(\d+) frames=(\d+) retired=(\d+) send_completions=(\d+) records=(\d+) posted=(\d+) landed=(\d+) bytes=(\d+)")
RECEIVE_RING = re.compile(r"receive ring=\d+ records=(\d+) posted=(\d+)")
TRACE = re.compile(r'\{"native_trace":(\d+),.*"ns":\[(\d+),\d+,(\d+)\]')
INFO = ("wire window=", "bridge node ", "pair capacity ", "pair up: ", '{"trace_')
COUNTS = ("requests", "frames", "retired", "send_completions", "records", "posted", "landed", "bytes")
FABRIC_KEEP = float(os.environ.get("MESH_FABRIC_KEEP", "600"))
FABRIC_READ = int(os.environ.get("MESH_FABRIC_READ", str(16 << 20)))
BRIDGES = {}
SLOW = {}

def bridge_log(entry):
    try:
        with open(entry["log"], "rb") as stream:
            size = os.fstat(stream.fileno()).st_size
            if size < entry["log_offset"]: entry.update(bridge_counters())
            stream.seek(entry["log_offset"])
            chunk = stream.read(FABRIC_READ)
    except Exception as error:
        entry["log_error"] = f"{type(error).__name__}: {error}"
        return
    end = chunk.rfind(b"\n") + 1
    entry["log_offset"] += end
    entry["log_size"] = size
    for line in chunk[:end].decode(errors="replace").splitlines():
        match = CENSUS.match(line)
        if match:
            link, queue, values = match.group(1), int(match.group(2)), [int(value) for value in match.groups()[2:]]
            total = entry["census"].setdefault(link, dict.fromkeys(("calls",) + COUNTS, 0))
            if queue == 0:
                total["calls"] += 1
                span = entry["spans"].pop(link, None)
                entry["last_call"][link] = {**dict.fromkeys(COUNTS, 0), "span_ns": span[1] - span[0] if span else None}
            for key, value in zip(COUNTS, values):
                total[key] += value
                entry["last_call"].setdefault(link, dict.fromkeys(COUNTS, 0))[key] += value
            continue
        match = TRACE.match(line)
        if match:
            span = entry["spans"].get(match.group(1))
            begin, end_ns = int(match.group(2)), int(match.group(3))
            entry["spans"][match.group(1)] = (min(begin, span[0]), max(end_ns, span[1])) if span else (begin, end_ns)
            entry["traces"] += 1
        elif line.startswith("receive ring="):
            entry["rings"] += 1
            match = RECEIVE_RING.match(line)
            if match and int(match.group(1)) > int(match.group(2)): entry["credit_limited"] += 1
        elif line.startswith("pair up: "): entry["pairs"] += 1
        elif line.startswith("exchange failed"): entry["retries"] += 1
        elif line.strip() and not line.startswith(INFO) and not (line.startswith("register ") and ": " not in line):
            entry["errors"] += 1
            entry["last_error"] = line[:240]

def bridge_counters():
    return {"log_offset": 0, "census": {}, "last_call": {}, "spans": {}, "traces": 0, "rings": 0, "credit_limited": 0, "pairs": 0, "retries": 0, "errors": 0, "last_error": None}

def processes(pids):
    table = {}
    for line in output(["/bin/ps", "-ww", "-o", "pid=,stat=,etime=,command=", "-p", ",".join(map(str, pids))]).splitlines() if pids else ():
        fields = line.split(None, 3)
        if len(fields) == 4 and fields[0].isdigit(): table[int(fields[0])] = {"stat": fields[1], "elapsed": fields[2], "argv": fields[3].split()}
    return table

def fabric_bridges(now):
    table = processes([int(pid) for pid in output(["/usr/bin/pgrep", "-x", "mesh-flow"]).split()])
    stat = executable([os.environ.get("MESH_STAT", ""), os.path.join(ROOT, "rdma", "mesh-stat"), os.path.join(HERE, "mesh-stat"), "/usr/local/mesh/bin/mesh-stat"])
    for pid, process in table.items():
        argv = process["argv"]
        if os.path.basename(argv[0]) != "mesh-flow": continue
        entry = BRIDGES.get(pid)
        if entry is None:
            option = lambda flag, default=None: next((argv[i + 1] for i in range(len(argv) - 1) if argv[i] == flag), default)
            links = [dict(zip(("device", "peer", "local", "remote", "service"), argv[i + 1].split(","))) for i in range(len(argv) - 1) if argv[i] == "--link"]
            log = next((line[1:] for line in output(["/usr/sbin/lsof", "-a", "-p", str(pid), "-d", "2", "-Fn"]).splitlines() if line.startswith("n/")), None)
            entry = BRIDGES[pid] = {"pid": pid, "rank": option("-I"), "region": option("-s", "/mesh0"), "links": links, "log": log, "log_size": None, **bridge_counters()}
        entry.update({"state": process["stat"], "elapsed": process["elapsed"], "exited_at": None})
        try: entry["stat"] = json.loads(output([stat, entry["region"]]).splitlines()[-1]) if stat else {"up": False, "error": "mesh-stat unavailable"}
        except Exception as error: entry["stat"] = {"up": False, "error": f"mesh-stat: {type(error).__name__}: {error}"}
        if entry["log"]: bridge_log(entry)
    for pid, entry in list(BRIDGES.items()):
        if pid in table: continue
        if entry["exited_at"] is None:
            entry.update({"exited_at": now, "state": "exited"})
            if entry["log"]: bridge_log(entry)
        elif now - entry["exited_at"] > FABRIC_KEEP: BRIDGES.pop(pid)
    return [copy.deepcopy({key: value for key, value in entry.items() if key != "spans"}) for entry in BRIDGES.values()]

def fabric_ports():
    try: tree = plistlib.loads(subprocess.run(["/usr/sbin/ioreg", "-r", "-c", "AppleThunderboltIPPort", "-l", "-a"], capture_output=True, timeout=3).stdout)
    except Exception as error: return [{"error": f"ioreg: {type(error).__name__}: {error}"}]
    blocks = {block.split(":", 1)[0]: block for block in re.split(r"\n(?=\S)", output(["/sbin/ifconfig"]))}
    neighbors = {}
    for line in output(["/usr/sbin/ndp", "-an"]).splitlines()[1:]:
        fields = line.split()
        if len(fields) >= 5 and fields[3] != "permanent": neighbors.setdefault(fields[2], []).append({"address": fields[0], "expire": fields[3], "state": fields[4]})
    ports, holders = [], []
    for port in tree:
        for interface in port.get("IORegistryEntryChildren", []):
            name = interface.get("BSD Name")
            if not name: continue
            children = interface.get("IORegistryEntryChildren", [])
            rdma = next((child for child in children if child.get("IOObjectClass") == "AppleThunderboltRDMAInterface"), {})
            connection = next((child for child in port.get("IORegistryEntryChildren", []) if child.get("IOObjectClass") == "AppleThunderboltIPConnection"), None)
            holders += [(name, int(match.group(1))) for child in rdma.get("IORegistryEntryChildren", []) for match in [re.match(r"pid (\d+)", child.get("IOUserClientCreator", ""))] if match]
            block = blocks.get(name, "")
            status = re.search(r"status: (\w+)", block)
            address = re.search(r"inet6 (fe80::[0-9a-f:]+)", block)
            ports.append({
                "iface": name, "device": "rdma_" + name if rdma else None,
                "tb_link": "up" if int(port.get("IOLinkStatus", 0)) & 2 else "down", "speed": port.get("IOLinkSpeed"),
                "link_ups": interface.get("IOLinkActiveCount"), "tb_connection": (connection or {}).get("Thunderbolt IP Connection State"),
                "ip": status.group(1) if status else "unknown", "address": address.group(1) if address else None,
                "neighbors": neighbors.get(name, []),
                "rdma_power": (rdma.get("IOPowerManagement") or {}).get("CurrentPowerState"),
                "holders": [],
                "queue_pairs": sum(child.get("IOObjectClass") == "AppleThunderboltRDMAQueuePair" for child in rdma.get("IORegistryEntryChildren", [])),
            })
    table = processes(sorted({pid for _, pid in holders}))
    for name, pid in holders:
        next(port for port in ports if port["iface"] == name)["holders"].append({"pid": pid, "stat": (table.get(pid) or {}).get("stat"), "command": os.path.basename(((table.get(pid) or {}).get("argv") or ["?"])[0])})
    return ports

def fabric_services():
    found = {}
    owner = os.getuid() or int(output(["/usr/bin/stat", "-f%u", "/dev/console"]) or 0)
    for domain in ("system", f"gui/{owner}"):
        text = output(["/bin/launchctl", "print", domain])
        for match in re.finditer(r"^\s+(\d+|-)\s+(\S+)\s+(io\.mesh\.\S+)$", text, re.M):
            found.setdefault(match.group(3), []).append({"domain": domain, "pid": int(match.group(1)) if match.group(1).isdigit() else 0, "status": match.group(2)})
        for match in re.finditer(r'"(io\.mesh\.[^"]+)" => disabled', text):
            found.setdefault(match.group(1), []).append({"domain": domain, "disabled": True})
    keeper = {"path": "/usr/local/mesh/log/keeper.log"}
    try:
        with open(keeper["path"], "rb") as stream:
            stream.seek(max(0, os.fstat(stream.fileno()).st_size - 4096))
            lines = [line for line in stream.read().decode(errors="replace").splitlines() if line.strip()]
        keeper.update({"last": lines[-1] if lines else None, "alarms": [line for line in lines if re.search(r"DEGRADED|ALARM", line)][-3:], "modified": os.path.getmtime(keeper["path"])})
    except Exception as error: keeper["error"] = f"{type(error).__name__}: {error}"
    return {"services": found, "keeper": keeper, "booted_at": number(output(["/usr/sbin/sysctl", "-n", "kern.boottime"]), r"sec = (\d+)")}

def fabric(now):
    if now - SLOW.get("ports_at", 0) >= 5: SLOW.update(ports_at=now, ports=fabric_ports())
    if now - SLOW.get("services_at", 0) >= 10: SLOW.update(services_at=now, services=fabric_services())
    return {"bridges": fabric_bridges(now), "ports": SLOW["ports"], "ports_at": SLOW["ports_at"], **SLOW["services"], "services_at": SLOW["services_at"]}

class TelemetryRecord(dict):
    def __init__(self, value):
        super().__init__(value)
        self.wire = {}
        self.wire_lock = threading.Lock()

    def encode(self, projected=False):
        def scalars(value):
            return {key: scalars(child) for key, child in value.items()} if isinstance(value, dict) else {"array_length": len(value)} if isinstance(value, list) else value
        with self.wire_lock:
            if projected not in self.wire:
                sample = self["sample"]
                workload = sample.get("workload", {})
                value = {**self, "sample": {**sample, "workload": {**workload, "measure_projection": "scalars_and_array_lengths", "producers": [{**row, "measures": scalars(row.get("measures", {}))} for row in workload.get("producers", [])]}}} if projected else self
                self.wire[projected] = json.dumps(value, separators=(",", ":")).encode()
            return self.wire[projected]


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
                        row["measures_sampled_at"] = current.get("measures_sampled_at")
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
            current["measures"] = {**current.get("measures", {}), **(payload.get("measures") or {})}
            current["measures_sampled_at"] = payload.get("sampled_at")
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
        for row in active:
            measures = row.get("measures", {})
            artifacts = measures.get("artifacts", {}) if isinstance(measures, dict) else {}
            if isinstance(artifacts, dict) and artifacts:
                identity = {key: row[key] for key in ("pid", "started_at", "name")}
                row["measures"] = {**measures, "artifacts": {name: {**artifact, "href": "/v1/artifact?" + urllib.parse.urlencode({**identity, "artifact": name})}
                    if isinstance(artifact, dict) else {"error": "artifact metadata is not an object", "received": artifact}
                    for name, artifact in artifacts.items()}}
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
            if self.records:
                previous = self.records[-1]
                workload = previous["sample"].get("workload", {})
                compact = {**workload, "measures_retention": "latest_sample", "producers": [{**row, "measures": {}} for row in workload.get("producers", [])]}
                self.records[-1] = TelemetryRecord({**previous, "sample": {**previous["sample"], "workload": compact}})
            self.sequence += 1
            sample = dict(sample)
            sample["stream"] = {"schema": 1, "sequence": self.sequence, "sampled_at": time.time(), "monotonic_ns": time.monotonic_ns()}
            record = TelemetryRecord(sample["stream"])
            record["sample"] = sample
            self.records.append(record)
            return record

    def latest(self):
        with self.lock:
            record = self.records[-1] if self.records else None
            oldest = self.records[0]["sequence"] if self.records else None
            latest = self.records[-1]["sequence"] if self.records else None
            return {"schema": 1, "capacity": self.records.maxlen, "oldest_sequence": oldest, "latest_sequence": latest, "record": record}

    def history(self, since=0):
        since = int(since)
        with self.lock:
            oldest = self.records[0]["sequence"] if self.records else None
            latest = self.records[-1]["sequence"] if self.records else None
            records = [record for record in self.records if record["sequence"] > since]
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
    try: links = fabric(time.time())
    except Exception as error: links = {"error": f"{type(error).__name__}: {error}"}
    result = dict(stat)
    result["fabric"] = links
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
        try:
            store.append(snapshot(store, facts))
        except Exception as error:
            store.publish_protocol({"event": "sample_error", "error": f"{type(error).__name__}: {error}", "at": time.time()})
            print(f"telemetry sample: {type(error).__name__}: {error}", flush=True)
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
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
        field = "records" if "records" in payload else "record" if "record" in payload else None
        if field:
            records = payload[field] if field == "records" else [payload[field]]
            encoded = [b"null" if record is None else record.encode(query.get("measures") == ["scalars"]) for record in records]
            value = b"[" + b",".join(encoded) + b"]" if field == "records" else encoded[0]
            body = json.dumps({key: value for key, value in payload.items() if key != field}, separators=(",", ":")).encode()[:-1] + b',"' + field.encode() + b'":' + value + b"}"
        else:
            body = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        target = urllib.parse.urlsplit(self.path)
        if target.path == "/v1/artifact":
            query = urllib.parse.parse_qs(target.query)
            try:
                key = (int(query["pid"][0]), float(query["started_at"][0]), query["name"][0])
                with self.server.store.lock:
                    artifact = self.server.store.producers[key]["measures"]["artifacts"][query["artifact"][0]]
                handle = open(artifact["path"], "rb")
            except Exception as error:
                self.send_json({"error": f"{type(error).__name__}: {error}"}, 404)
                return
            with handle:
                self.send_response(200)
                self.send_header("Content-Type", artifact.get("content_type", "application/octet-stream"))
                self.send_header("Content-Disposition", "attachment")
                self.send_header("Content-Length", str(os.fstat(handle.fileno()).st_size))
                self.end_headers()
                shutil.copyfileobj(handle, self.wfile)
            return
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
