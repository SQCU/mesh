#!/usr/bin/env mesh-python
import json, os, subprocess, sys, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "rdma")]
import peers
BASE = os.environ.get("MESH_TELEMETRY_URL", "http://127.0.0.1:8788")
# ../design/data-model.md#6-why-code-here-only-loses-waits-and-guards
DEADLINE = 5.0

def expired(wait):
    print(f"mesh-observe: {wait}: past its deadline of {DEADLINE:.3f} s; the program ends (design/data-model.md section 6)", file=sys.stderr, flush=True)
    os._exit(1)

def fetch(path):
    try:
        with urllib.request.urlopen(BASE.rstrip("/") + path, timeout=DEADLINE) as response:
            return json.load(response)
    except OSError as error:
        if isinstance(error, TimeoutError) or isinstance(getattr(error, "reason", None), TimeoutError): expired("the telemetry's read")
        raise

def direct():
    attempts = []
    candidates = [
        os.path.join(HERE, "..", "user", "mesh-telemetry.py"),
        os.path.join(HERE, "mesh-telemetry.py"),
        "/usr/local/mesh/bin/mesh-telemetry.py",
    ]
    for path in candidates:
        if not os.path.isfile(path): continue
        try:
            try: result = subprocess.run([sys.executable, path, "--once"], capture_output=True, text=True, timeout=DEADLINE)
            except subprocess.TimeoutExpired: expired(f"{path} --once")
            if result.returncode == 0: return json.loads(result.stdout.strip().splitlines()[-1])
            attempts.append({"path": path, "returncode": result.returncode, "error": result.stderr.strip()})
        except Exception as error:
            attempts.append({"path": path, "error": f"{type(error).__name__}: {error}"})
    return {"schema": 2, "up": False, "reachable": True, "name": peers.label(peers.host()["node"], peers.table(os.path.join(HERE, "..", "etc", "mesh-nodes.json"), "/usr/local/mesh/etc/mesh-nodes.json")), "error": "resident telemetry stream unavailable", "attempts": attempts}

try:
    if len(sys.argv) > 2 and sys.argv[1] == "--since":
        print(json.dumps(fetch("/v1/history?since=" + str(int(sys.argv[2])) + "&measures=scalars"), separators=(",", ":")))
    else:
        envelope = fetch("/v1/latest?measures=scalars")
        record = envelope.get("record") or {}
        print(json.dumps(record.get("sample") or direct(), separators=(",", ":")))
except Exception:
    print(json.dumps(direct(), separators=(",", ":")))
