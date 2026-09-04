#!/usr/bin/env mesh-python
import json, os, socket, subprocess, sys, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get("MESH_TELEMETRY_URL", "http://127.0.0.1:8788")

def fetch(path):
    with urllib.request.urlopen(BASE.rstrip("/") + path, timeout=3) as response:
        return json.load(response)

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
            result = subprocess.run([sys.executable, path, "--once"], capture_output=True, text=True, timeout=7)
            if result.returncode == 0: return json.loads(result.stdout.strip().splitlines()[-1])
            attempts.append({"path": path, "returncode": result.returncode, "error": result.stderr.strip()})
        except Exception as error:
            attempts.append({"path": path, "error": f"{type(error).__name__}: {error}"})
    return {"schema": 2, "up": False, "reachable": True, "name": socket.gethostname().split(".")[0], "error": "resident telemetry stream unavailable", "attempts": attempts}

try:
    if len(sys.argv) > 2 and sys.argv[1] == "--since":
        print(json.dumps(fetch("/v1/history?since=" + str(int(sys.argv[2]))), separators=(",", ":")))
    else:
        envelope = fetch("/v1/latest")
        record = envelope.get("record") or {}
        print(json.dumps(record.get("sample") or direct(), separators=(",", ":")))
except Exception:
    print(json.dumps(direct(), separators=(",", ":")))
