import argparse, json, os, subprocess, time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse


ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "viewer")


def run(argv):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=8)
    return result.stdout + result.stderr, result.returncode


def json_line(output):
    for line in reversed(output.splitlines()):
        try:
            value = json.loads(line)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
    return {"up": False, "error": output.strip() or "no status returned"}


def telemetry_bytes(source, limit):
    if ":" in source and not source.startswith("/"):
        host, path = source.split(":", 1)
        output, code = run(["ssh", host, "tail", "-c", str(limit), path])
        if code:
            raise RuntimeError(output.strip())
        return output
    with open(source, "rb") as handle:
        handle.seek(0, os.SEEK_END)
        handle.seek(max(0, handle.tell() - limit))
        return handle.read().decode("utf-8", "replace")


def telemetry(source, limit=12_000_000):
    text = telemetry_bytes(source, limit)
    rows = []
    for line in text.splitlines():
        try:
            value = json.loads(line)
            if isinstance(value, dict) and "resp_id" in value:
                rows.append(value)
        except json.JSONDecodeError:
            pass
    return rows[-600:]


def source_age(source):
    if ":" in source and not source.startswith("/"):
        host, path = source.split(":", 1)
        output, code = run(["ssh", host, "stat", "-f", "%m", path])
        return max(0.0, time.time() - float(output.strip())) if code == 0 else None
    return max(0.0, time.time() - os.path.getmtime(source))


def bridge(repo, host=None):
    command = [os.path.join(repo, "bin", "mesh-bridge.sh"), "status"]
    if host:
        command = ["ssh", host, "cd", repo, "&&", "bin/mesh-bridge.sh", "status"]
    output, code = run(command)
    value = json_line(output)
    value["command_ok"] = code == 0
    value["host"] = host or os.uname().nodename
    return value


def games():
    output, _ = run(["ps", "-axo", "pid=,etime=,command="])
    return [line.strip() for line in output.splitlines() if "darkplaces-dedicated" in line and "-xonotic" in line]


def responders(host):
    command = ["ps", "-axo", "pid=,etime=,command="]
    if host:
        command = ["ssh", host, *command]
    output, _ = run(command)
    return [line.strip() for line in output.splitlines() if "solver.strat.strat_responder" in line and "grep" not in line]


def latest_remote(host, repo):
    candidates = []
    for directory, name in [
        ("/private/tmp", "live.jsonl"),
        (os.path.join(repo, "xonotic", "solver", "strat", "runs"), "telemetry.jsonl"),
    ]:
        output, _ = run(["ssh", host, "find", directory, "-name", name, "-type", "f", "-print"])
        candidates.extend(line for line in output.splitlines() if line.startswith("/"))
    dated = []
    for path in candidates:
        output, code = run(["ssh", host, "stat", "-f", "%m", path])
        if code == 0:
            dated.append((float(output.strip()), path))
    if dated:
        return host + ":" + max(dated)[1]
    return host + ":" + os.path.join(repo, "xonotic", "solver", "strat", "runs", "cartserver_telemetry.jsonl")


class Handler(SimpleHTTPRequestHandler):
    server_version = "MeshStrategyViewer/1"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT, **kwargs)

    def do_GET(self):
        if urlparse(self.path).path != "/api/live":
            return super().do_GET()
        try:
            rows = telemetry(self.server.telemetry_source)
            error = None
        except Exception as exc:
            rows, error = [], str(exc)
        body = json.dumps({
            "now": time.time(),
            "source": self.server.telemetry_source,
            "source_age": source_age(self.server.telemetry_source) if not error else None,
            "error": error,
            "rows": rows,
            "nodes": [
                bridge(self.server.local_repo),
                bridge(self.server.remote_repo, self.server.remote_host),
            ],
            "games": games(),
            "responders": responders(self.server.remote_host),
        }, allow_nan=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        print("[viewer] " + format % args, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--telemetry")
    parser.add_argument("--remote-host", default="mesh-mini")
    parser.add_argument("--local-repo", default=os.path.abspath(os.path.join(ROOT, "..", "..", "..", "..")))
    parser.add_argument("--remote-repo", default="/Users/mdot/mesh")
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8791)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.bind, args.port), Handler)
    server.telemetry_source = args.telemetry or latest_remote(args.remote_host, args.remote_repo)
    server.remote_host = args.remote_host
    server.local_repo = args.local_repo
    server.remote_repo = args.remote_repo
    print(f"[viewer] http://{args.bind}:{args.port} source={server.telemetry_source}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
