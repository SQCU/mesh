#!/usr/bin/env mesh-python
import concurrent.futures, http.server, json, math, os, socketserver, subprocess, sys, threading, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(ROOT, "bin")]
import mesh_restart
STAT = os.path.join(ROOT, "bin", "mesh-stat")
CONFIGURATION = sys.argv[1] if len(sys.argv) > 1 else None
try:
    with open(CONFIGURATION) as f: PERIOD = float(json.load(f)["period_s"])
    CONFIGURED = None
except (TypeError, OSError, ValueError, KeyError) as error:
    PERIOD, CONFIGURED = None, f"no cadence: the observer's configuration document (serve.py DOCUMENT, etc/observer.json) gives no period_s ({type(error).__name__}: {error})"
NODE_PORT = int(os.environ.get("MESH_TELEMETRY_PORT", "8788"))
KEEP = int(os.environ.get("MESH_KEEP", "600"))
ESTIMATE = int(os.environ.get("MESH_ESTIMATE_WINDOW", "60"))

with open(os.path.join(ROOT, "etc", "mesh-capacity.json")) as f: CAPACITY = json.load(f)
with open(os.environ.get("MESH_NODES", os.path.join(ROOT, "etc", "mesh-nodes.json"))) as f: ROSTER = json.load(f)
nodes = {name:dict(node) for name, node in ROSTER.get("nodes", {}).items()}
hist = {name:[] for name in nodes}
previous = {}
stream_sequences = {}
http_events = []
lock = threading.Lock()

def output(command):
    return subprocess.run(command, capture_output=True, text=True).stdout.strip()

_self = output(["/usr/sbin/scutil", "--get", "LocalHostName"]) or "self"
nodes.setdefault(_self, {}).update({"self":True})
hist.setdefault(_self, [])

def members():
    """The nodes this node's bridge holds a record of (mesh-stat's `nodes`, design/transport.md): each one's
    bridge number and its record, this node's own marked; the roster's other nodes stay its inventory."""
    try: stat = json.loads(output([STAT]).splitlines()[-1])
    except (OSError, ValueError, IndexError) as error: stat = {"up": False, "error": f"{type(error).__name__}: {error}"}
    held = {record["name"]: record for record in stat.get("nodes") or [] if record.get("name")}
    with lock:
        for name, node in nodes.items():
            record = held.get(name)
            node.update({"member": record is not None, "number": (record or {}).get("node"), "record": record})
        for name, record in held.items():
            if name not in nodes:
                nodes[name] = {"member": True, "number": record.get("node"), "record": record}
                hist.setdefault(name, [])
            if record.get("node") == stat.get("node") and name != _self:
                nodes[name]["self"] = True
        nodes[_self]["bridge"] = {key: stat.get(key) for key in ("up", "errno", "error", "node") if key in stat}

def membership(node):
    state = "local" if node.get("self") else "member" if node.get("member") else "roster"
    return {"state":state, "present":state != "roster", "source":"bridge records" if node.get("member") else "local" if node.get("self") else "roster"}

def unavailable(name, node, error):
    model = node.get("model") or (node.get("record") or {}).get("model")
    return {"schema":1,"up":False,"reachable":False,"name":name,"host":{"model":model},"capacity":{"kind":"characterized","source":CAPACITY.get("source"),"metrics":CAPACITY.get("models", {}).get(model)},"error":error}

def telemetry(node, path):
    """This node's telemetry at `path`, over loopback; another node's stays on that node, its record its facts here."""
    if not node.get("self"): raise RuntimeError("its telemetry is read on its own node; this observer serves this node's")
    with urllib.request.urlopen(f"http://127.0.0.1:{NODE_PORT}{path}") as response: return json.load(response), "127.0.0.1"

def node_sample(name, node):
    sequence = stream_sequences.get(name)
    path = "/v1/latest?measures=scalars" if sequence is None else f"/v1/history?since={sequence}&measures=scalars"
    payload, connected = telemetry(node, path)
    if payload.get("reset"):
        stream_sequences.pop(name, None)
        return node_sample(name, node)
    records = payload.get("records") if sequence is not None else [payload.get("record")]
    records = [record for record in records or [] if record and record.get("sample")]
    if records: stream_sequences[name] = int(records[-1]["sequence"])
    return [record["sample"] for record in records], connected

def decorate_sample(sample, node, connected):
    sample["reachable"] = True
    sample["telemetry_transport"] = connected
    model = ((sample.get("host") or {}).get("model")) or node.get("model")
    sample.setdefault("host", {"model":model})
    sample.setdefault("capacity", {"kind":"characterized","source":CAPACITY.get("source"),"metrics":CAPACITY.get("models", {}).get(model)})
    sample.setdefault("machine", {"up":False,"error":"node telemetry not installed"})
    sample.setdefault("availability", {"fabric":"measured" if sample.get("up") else "unavailable","machine":"unavailable","achieved_flops":"unavailable_without_workload_semantics","memory_bandwidth":"unavailable_without_hardware_byte_counter"})
    if sample["machine"].get("up") and sample["availability"].get("machine") == "unavailable": sample["availability"]["machine"] = "measured_gpu_only"
    return sample

def sample(name, node):
    try:
        samples, connected = node_sample(name, node)
        with lock: prior = hist.get(name, [])[-1] if hist.get(name) else None
        s = dict(samples[-1] if samples else prior or {})
        if not s: raise RuntimeError("telemetry ring has no sample")
        decorate_sample(s, node, connected)
        for streamed in samples: decorate_sample(streamed, node, connected)
        s["_stream_samples"] = samples
        return s
    except Exception as e:
        return unavailable(name, node, str(e) or type(e).__name__)

def rates(s, old):
    r = {"kind":"derived_from_bridge_counters","tx_bytes_s":None,"rx_bytes_s":None,"tx_gbps":None,"rx_gbps":None,"tx_fabric_pct":None,"rx_fabric_pct":None}
    if s.get("up") and old and old.get("up"):
        dt = s.get("up_ms", 0) - old.get("up_ms", 0)
        if dt > 0:
            tx = max(0, s.get("sent", 0) - old.get("sent", 0)) * s.get("pgsz", 0) * 1000 / dt
            rx = max(0, s.get("recvd", 0) - old.get("recvd", 0)) * s.get("pgsz", 0) * 1000 / dt
            r.update({"tx_bytes_s":tx,"rx_bytes_s":rx,"tx_gbps":tx*8/1e9,"rx_gbps":rx*8/1e9})
            ceiling = (((s.get("capacity") or {}).get("metrics") or {}).get("fabric_gbs"))
            if ceiling:
                r["tx_fabric_pct"] = tx / (ceiling * 1e9) * 100
                r["rx_fabric_pct"] = rx / (ceiling * 1e9) * 100
    s["rates"] = r
    return s

def moments(values):
    if not values: return {"mean":None,"variance":None,"sd":None,"samples":0}
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return {"mean":mean,"variance":variance,"sd":variance ** .5,"samples":len(values)}

def numeric_sum(values):
    values = [float(value) for value in values if isinstance(value, (int, float)) and math.isfinite(value)]
    return sum(values) if values else None

def capacity_aggregate(rows):
    complete = [row for row in rows
                if isinstance(row.get("gpu_fp32_gflops"), (int, float))
                and isinstance(row.get("cpu_gflops"), (int, float))]
    return {
        "gpu_fp32_gflops":numeric_sum(row.get("gpu_fp32_gflops") for row in rows),
        "cpu_gflops":numeric_sum(row.get("cpu_gflops") for row in rows),
        "fp32_gflops":numeric_sum(
            float(row["gpu_fp32_gflops"]) + float(row["cpu_gflops"])
            for row in complete
        ),
        "memory_gbs":numeric_sum(row.get("memory_gbs") for row in rows),
        "gpu_node_mass":sum(isinstance(row.get("gpu_fp32_gflops"), (int, float)) for row in rows),
        "cpu_node_mass":sum(isinstance(row.get("cpu_gflops"), (int, float)) for row in rows),
        "fp32_node_mass":len(complete),
        "memory_node_mass":sum(isinstance(row.get("memory_gbs"), (int, float)) for row in rows),
    }

def estimate(s, history):
    cap = ((s.get("capacity") or {}).get("metrics") or {})
    live = ((s.get("machine") or {}).get("metrics") or {})
    gpu_peak = cap.get("gpu_fp32_gflops")
    cpu_peak = cap.get("cpu_gflops")
    gpu_active = live.get("gpu_active_pct")
    cluster_activity = [
        row.get("active_pct") for row in (live.get("cpu_clusters") or {}).values()
        if isinstance(row.get("active_pct"), (int, float))
    ]
    cpu_active = live.get("cpu_active_pct")
    if not isinstance(cpu_active, (int, float)):
        cpu_active = max(cluster_activity, default=None)
    workload = s.get("workload") or {}
    workload_fp32 = workload.get("fp32") or {}
    workload_up = bool(workload.get("up"))
    lower = workload_fp32.get("lower_gflops_s") if workload_up else None
    published_upper = workload_fp32.get("upper_gflops_s") if workload_up else None
    gpu_activity_upper = gpu_peak * max(0, min(100, gpu_active)) / 100 if gpu_peak is not None and gpu_active is not None else None
    cpu_activity_upper = cpu_peak * max(0, min(100, cpu_active)) / 100 if cpu_peak is not None and cpu_active is not None else None
    activity_upper = gpu_activity_upper + cpu_activity_upper if gpu_activity_upper is not None and cpu_activity_upper is not None else None
    upper_candidates = [value for value in (published_upper, activity_upper) if value is not None]
    upper = max(float(lower), min(upper_candidates)) if lower is not None and upper_candidates else min(upper_candidates) if upper_candidates else None
    old_upper = [((x.get("estimates") or {}).get("fp32") or {}).get("upper_gflops") for x in history[-ESTIMATE+1:]] if ESTIMATE > 1 else []
    upper_values = [x for x in old_upper + [upper] if x is not None]
    old_lower = [((x.get("estimates") or {}).get("fp32") or {}).get("lower_gflops") for x in history[-ESTIMATE+1:]] if ESTIMATE > 1 else []
    lower_values = [x for x in old_lower + [lower] if x is not None]
    memory_cap = cap.get("memory_gbs")
    memory_lower = live.get("memory_total_lower_gbs")
    memory_upper = live.get("memory_total_upper_gbs")
    direct = memory_lower is not None and memory_upper is not None
    if not direct: memory_lower, memory_upper = None, memory_cap
    old_memory_lower = [((x.get("estimates") or {}).get("memory") or {}).get("lower_gbs") for x in history[-ESTIMATE+1:]] if ESTIMATE > 1 else []
    old_memory_upper = [((x.get("estimates") or {}).get("memory") or {}).get("upper_gbs") for x in history[-ESTIMATE+1:]] if ESTIMATE > 1 else []
    memory_lower_values = [x for x in old_memory_lower + [memory_lower] if x is not None]
    memory_upper_values = [x for x in old_memory_upper + [memory_upper] if x is not None]
    workload_memory = workload.get("memory") or {}
    workload_memory_lower = workload_memory.get("lower_gbs") if workload_up else None
    workload_memory_upper = workload_memory.get("upper_gbs") if workload_up else None
    old_workload_memory_lower = [((x.get("estimates") or {}).get("workload_memory") or {}).get("lower_gbs") for x in history[-ESTIMATE+1:]] if ESTIMATE > 1 else []
    old_workload_memory_upper = [((x.get("estimates") or {}).get("workload_memory") or {}).get("upper_gbs") for x in history[-ESTIMATE+1:]] if ESTIMATE > 1 else []
    workload_memory_lower_values = [x for x in old_workload_memory_lower + [workload_memory_lower] if x is not None]
    workload_memory_upper_values = [x for x in old_workload_memory_upper + [workload_memory_upper] if x is not None]
    s["estimates"] = {
        "kind":"analytic_lower_hardware_activity_upper" if workload_up and activity_upper is not None else "published_workload_envelope" if workload_up else "workload_blind_envelope",
        "window":ESTIMATE,
        "fp32":{"coverage":"published_workloads_and_machine_activity" if workload_up and activity_upper is not None else "published_workloads" if workload_up else "machine_activity","lower_gflops":lower,"upper_gflops":upper,"analytic_upper_gflops":published_upper,"activity_upper_gflops":activity_upper,"gpu_activity_upper_gflops":gpu_activity_upper,"cpu_activity_upper_gflops":cpu_activity_upper,"machine_activity_component_mass":int(gpu_activity_upper is not None)+int(cpu_activity_upper is not None),"cpu_active_pct":cpu_active,"lower_moments":moments(lower_values),"upper_moments":moments(upper_values)},
        "memory":{"coverage":"whole_machine","source":"hardware_histogram" if direct else "capacity_envelope","lower_gbs":memory_lower,"upper_gbs":memory_upper,"lower_moments":moments(memory_lower_values),"upper_moments":moments(memory_upper_values),"lower_bucket_variance":live.get("memory_total_lower_variance"),"upper_bucket_variance":live.get("memory_total_upper_variance")},
        "workload_memory":{"coverage":"published_workloads","source":"analytic_operation_envelope" if workload_up else "unavailable","lower_gbs":workload_memory_lower,"upper_gbs":workload_memory_upper,"lower_moments":moments(workload_memory_lower_values),"upper_moments":moments(workload_memory_upper_values)},
        "deadlines":workload.get("deadlines") or {"observations":0,"slack_s":[],"minimum_slack_s":None,"mean_slack_s":None},
        "rows":workload.get("rows"),
        "producers":workload.get("active_producers"),
    }
    if upper is not None: s.setdefault("availability", {})["achieved_flops"] = "bounded_analytic_lower_hardware_activity_upper" if workload_up and activity_upper is not None else "bounded_published_workload" if workload_up else "bounded_gpu_estimate"
    if workload_memory_upper is not None: s.setdefault("availability", {})["memory_bandwidth"] = "bounded_published_workload_and_hardware_histogram" if direct else "bounded_published_workload"
    elif memory_upper is not None: s.setdefault("availability", {})["memory_bandwidth"] = "bounded_hardware_histogram" if direct else "bounded_capacity_envelope"
    return s

def fabric_snapshot():
    latest = {name: rows[-1] for name, rows in hist.items() if rows}
    connected = [name for name, node in nodes.items() if membership(node).get("present")]
    reachable = [name for name in connected if (latest.get(name) or {}).get("reachable")]
    active = [name for name in reachable if latest[name].get("up")]
    capacity = []
    inventory_capacity = []
    for name in nodes:
        node = nodes[name]
        sample = latest.get(name) or {}
        model = ((sample.get("host") or {}).get("model")) or node.get("model")
        metrics = ((sample.get("capacity") or {}).get("metrics")) or CAPACITY.get("models", {}).get(model) or {}
        inventory_capacity.append(metrics)
        if name in reachable: capacity.append(metrics)
    live_samples = [latest[name] for name in reachable]
    fp32 = [((sample.get("estimates") or {}).get("fp32") or {}) for sample in live_samples]
    fp32 = [row for row in fp32 if row.get("upper_gflops") is not None]
    memory = [((sample.get("estimates") or {}).get("memory") or {}) for sample in live_samples]
    memory = [row for row in memory if row.get("upper_gbs") is not None]
    workload_memory = [((sample.get("estimates") or {}).get("workload_memory") or {}) for sample in live_samples]
    workload_memory = [row for row in workload_memory if row.get("upper_gbs") is not None]
    workloads = [(sample.get("workload") or {}) for sample in live_samples if (sample.get("workload") or {}).get("up")]
    deadlines = [(row.get("deadlines") or {}) for row in workloads]
    deadline_slack = [float(value) for row in deadlines for value in row.get("slack_s") or []]
    phase = {key:numeric_sum(latest[name].get(key) for name in active) for key in ("free", "recv", "send", "app")}
    return {
        "observed_at":time.time(), "nodes":len(nodes), "sampled":len(latest),
        "connected":connected, "reachable":reachable, "active":active,
        "unmapped":sorted(set(nodes) - set(connected)),
        "telemetry_unavailable":sorted(set(connected) - set(reachable)),
        "phase":phase,
        "capacity":capacity_aggregate(capacity),
        "inventory_capacity":capacity_aggregate(inventory_capacity),
        "rates":{
            "tx_gbps":numeric_sum((sample.get("rates") or {}).get("tx_gbps") for sample in live_samples),
            "rx_gbps":numeric_sum((sample.get("rates") or {}).get("rx_gbps") for sample in live_samples),
        },
        "estimates":{
            "fp32":{"lower_gflops":numeric_sum(row.get("lower_gflops") for row in fp32),"upper_gflops":numeric_sum(row.get("upper_gflops") for row in fp32),"nodes":len(fp32)},
            "memory":{"lower_gbs":numeric_sum(row.get("lower_gbs") for row in memory),"upper_gbs":numeric_sum(row.get("upper_gbs") for row in memory),"nodes":len(memory)},
            "workload_memory":{"lower_gbs":numeric_sum(row.get("lower_gbs") for row in workload_memory),"upper_gbs":numeric_sum(row.get("upper_gbs") for row in workload_memory),"nodes":len(workload_memory)},
        },
        "workload":{
            "producers":numeric_sum(row.get("active_producers") for row in workloads),
            "rows":numeric_sum(row.get("rows") for row in workloads),
            "deadlines":{
                "observations":sum(int(row.get("observations") or 0) for row in deadlines),
                "slack_s":deadline_slack,
                "minimum_slack_s":min(deadline_slack, default=None),
                "mean_slack_s":sum(deadline_slack) / len(deadline_slack) if deadline_slack else None,
            },
        },
    }

def visualization_snapshot():
    out = {}
    for name, rows in hist.items():
        if not rows:
            out[name] = []
            continue
        compact = [{key:row.get(key) for key in ("t", "up", "free", "recv", "send", "app", "sent")} for row in rows[:-1]]
        compact.append(rows[-1])
        out[name] = compact
    return out

def record(name, node, started):
    result = sample(name, node)
    batch = result.pop("_stream_samples", None)
    batch = [result] if batch is None else batch
    for s in batch:
        s["t"] = s.get("observed_at", started)
        s["name"] = name
        with lock:
            s["membership"] = membership(nodes.setdefault(name, {}))
            rates(s, previous.get(name))
            if s.get("up"): previous[name] = s
            h = hist.setdefault(name, [])
            estimate(s, h)
            if h:
                workload = h[-1].get("workload", {})
                h[-1] = {**h[-1], "workload": {**workload, "measures_retention": "latest_sample", "producers": [{**row, "measures": {}} for row in workload.get("producers", [])]}}
            h.append(s)
            del h[:-KEEP]

def poll():
    """Every PERIOD: the bridge's records read, and each node recorded where no recording of it is under way: this node's
    telemetry, another node's record alone (its telemetry unavailable here), so a slow telemetry read delays only its own
    samples."""
    asked = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=64) as pool:
        while True:
            started = time.time()
            members()
            with lock: current = [(name, dict(node)) for name, node in nodes.items()]
            for name, node in current:
                if name not in asked or asked[name].done(): asked[name] = pool.submit(record, name, node, started)
            time.sleep(max(0.0, PERIOD - (time.time() - started)))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs): super().__init__(*args, directory=HERE, **kwargs)
    def do_GET(self):
        if self.path.startswith("/data.json") or self.path.startswith("/latest.json") or self.path.startswith("/v1/visualization"):
            with lock:
                payload = {"period":PERIOD,"configuration":CONFIGURATION,"error":CONFIGURED,"inventory":nodes,"fabric":fabric_snapshot(),"service":{"http_events":list(http_events)}}
                if self.path.startswith("/latest.json"):
                    payload["latest"] = {name: rows[-1] for name, rows in hist.items() if rows}
                elif self.path.startswith("/v1/visualization"):
                    payload["nodes"] = visualization_snapshot()
                else:
                    payload["nodes"] = hist
                body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else: super().do_GET()
    def log_message(self, format, *args):
        with lock:
            http_events.append({"at":time.time(),"message":format % args})
            del http_events[:-16]

if __name__ == "__main__":
    mesh_restart.on_change(__file__, CONFIGURATION, os.path.join(ROOT, "etc", "mesh-capacity.json"), os.environ.get("MESH_NODES", os.path.join(ROOT, "etc", "mesh-nodes.json")))
    if PERIOD is not None: threading.Thread(target=poll, daemon=True).start()
    else: print(CONFIGURED, file=sys.stderr, flush=True)
    port = int(os.environ.get("MESH_PORT", "8787"))
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer(("127.0.0.1", port), Handler) as server:
        print(f"mesh observer http://localhost:{port}, a sample every {PERIOD} s ({CONFIGURATION})", flush=True)
        server.serve_forever()
