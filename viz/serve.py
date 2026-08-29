#!/usr/bin/env python3
"""Serve the mesh's live page-table phase space.

Polls every node's shared-memory region through mesh-stat and serves the result
as JSON alongside the viewer. Everything published here is a counter the bridge
actually maintains; nothing is modelled, smoothed or invented. A node that is
not running says so.

    python3 viz/serve.py            # then open http://localhost:8787
"""
import json, os, subprocess, threading, time, http.server, socketserver

HERE   = os.path.dirname(os.path.abspath(__file__))
STAT   = os.path.join(HERE, "..", "rdma", "mesh-stat")
PERIOD = float(os.environ.get("MESH_POLL", "2"))
KEEP   = 600                      # samples of history per node

# name -> how to run mesh-stat there. Local first, then anything reachable.
NODES = [("mbp",  [STAT]),
         ("mini", ["ssh", "-o", "ConnectTimeout=3", "mesh-mini", "~/mesh/rdma/mesh-stat"])]

hist = {n: [] for n, _ in NODES}
lock = threading.Lock()

def sample(cmd):
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=5).stdout.decode()
        return json.loads(out.strip().splitlines()[-1])
    except Exception as e:
        return {"up": False, "err": type(e).__name__}

def poll():
    while True:
        t = time.time()
        for name, cmd in NODES:
            s = sample(cmd); s["t"] = t; s["name"] = name
            with lock:
                h = hist[name]; h.append(s)
                del h[:-KEEP]
        time.sleep(PERIOD)

class H(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **k): super().__init__(*a, directory=HERE, **k)
    def do_GET(self):
        if self.path.startswith("/data.json"):
            with lock: body = json.dumps({"period": PERIOD, "nodes": hist}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers(); self.wfile.write(body)
        else:
            super().do_GET()
    def log_message(self, *a): pass

if __name__ == "__main__":
    threading.Thread(target=poll, daemon=True).start()
    port = int(os.environ.get("MESH_PORT", "8787"))
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", port), H) as s:
        print(f"mesh viewer on http://localhost:{port}  (polling every {PERIOD:g}s)")
        s.serve_forever()
