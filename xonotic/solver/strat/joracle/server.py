from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np

try:
    from .follow import TelemetryFollower, split_source
    from .liveness import RuntimeMeasure
    from .metrics import summarize
    from .probe import RollingProbe
    from .field_measures import field_measures
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from solver.strat.joracle.follow import TelemetryFollower, split_source
    from solver.strat.joracle.liveness import RuntimeMeasure
    from solver.strat.joracle.metrics import summarize
    from solver.strat.joracle.probe import RollingProbe
    from solver.strat.joracle.field_measures import field_measures

WEB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))

def clean(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.floating,)):
        return clean(float(value))
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    return value

def series_row(frame):
    carts = frame.get("carts") or []
    return {
        "t": frame.get("t"),
        "seen_at": frame.get("_seen_at"),
        "epoch": frame.get("_epoch"),
        "resp_id": frame.get("resp_id"),
        "req_tick": frame.get("req_tick"),
        "k": frame.get("k"), "j": frame.get("j"), "l": frame.get("l"),
        "mode": frame.get("mode"),
        "updates": frame.get("updates"),
        "PW": frame.get("PW"),
        "SUCC": frame.get("SUCC"),
        "loser_ranks": frame.get("loser_ranks"),
        "depth": [c.get("depth") for c in carts],
        "control_team": [c.get("control_team") for c in carts],
        "speed": [c.get("speed") for c in carts],
        "focus": frame.get("strategy_focus"),
        "resources": frame.get("resources"),
        "instrument_counts": frame.get("instrument_counts"),
        "instrument_count": frame.get("instrument_count"),
        "belief": frame.get("belief"),
        "loss": (frame.get("update") or {}).get("loss"),
        "loss_pg": (frame.get("update") or {}).get("loss_pg"),
        "advantage": (frame.get("update") or {}).get("advantage"),
        "advantage_w": (frame.get("update") or {}).get("advantage_w"),
        "advantage_l": (frame.get("update") or {}).get("advantage_l"),
        "reward_w": (frame.get("update") or {}).get("reward_w"),
        "reward_l": (frame.get("update") or {}).get("reward_l"),
        "winner_value_mean": _mean((frame.get("model") or {}).get("winner_value")),
        "loser_value_mean": _mean((frame.get("model") or {}).get("loser_value")),
    }

def _mean(value):
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.isfinite(array).any():
        return None
    return round(float(np.nanmean(array)), 5)

def internals(frame):
    if not frame:
        return {"available": False}
    model = frame.get("model") or {}

    def matrix(name):
        value = model.get(name)
        if value is None:
            return None
        array = np.asarray(value, dtype=np.float64)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        if array.ndim != 2:
            return None
        return array.tolist()

    j = model.get("j")
    j_array = np.asarray(j, dtype=np.float64) if j is not None else None
    j_stats = None
    if j_array is not None and j_array.ndim == 2 and j_array.size:
        finite = np.isfinite(j_array)
        values = j_array[finite]
        spectrum = None
        finite_rows = j_array[np.isfinite(j_array).all(axis=1)]
        if len(finite_rows):
            centered = finite_rows - finite_rows.mean(axis=0, keepdims=True)
            spectrum_error = None
            try:
                singular = np.linalg.svd(centered, compute_uv=False)
                spectrum = (singular / singular[0]).tolist() if singular[0] else np.zeros_like(singular).tolist()
            except np.linalg.LinAlgError as error:
                spectrum_error = f"{type(error).__name__}: {error}"
        else:
            spectrum_error = None
        j_stats = {
            "shape": list(j_array.shape),
            "finite": int(finite.sum()), "size": int(finite.size),
            "min": None if not values.size else float(values.min()),
            "max": None if not values.size else float(values.max()),
            "std": None if not values.size else float(values.std()),
            "finite_row_mass": len(finite_rows),
            "spectrum": spectrum,
            "spectrum_error": spectrum_error,
        }

    assignments = sorted(frame.get("assignments") or [], key=lambda a: a.get("row", 0))
    return {
        "available": True,
        "resp_id": frame.get("resp_id"),
        "j": matrix("j"),
        "j_stats": j_stats,
        "coupling": matrix("coupling"),
        "hierarchy": matrix("hierarchy"),
        "x": matrix("x"),
        "beta": matrix("beta"),
        "score_stats": _model_range(model, "score"),
        "w_stats": _model_range(model, "w"),
        "winner_value": clean(model.get("winner_value")),
        "loser_value": clean(model.get("loser_value")),
        "diag_k": clean(model.get("diag_k")),
        "dw_dt": matrix("dw_dt"),
        "advantage": (frame.get("update") or {}).get("advantage"),
        "update": clean(frame.get("update")),
        "assignments": assignments,
        "shapes": model.get("shapes"),
        "finite_coordinate_mass": model.get("finite_coordinate_mass"),
        "ranges": model.get("range"),
        "game_value": frame.get("game_value"),
    }

def _range(value):
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return None
    return {
        "min": round(float(finite.min()), 5), "max": round(float(finite.max()), 5),
        "mean": round(float(finite.mean()), 5), "shape": list(array.shape),
    }

def _model_range(model, name):
    direct = _range(model.get(name))
    if direct is not None:
        return direct
    limits = (model.get("range") or {}).get(name)
    if not isinstance(limits, list) or len(limits) != 2:
        return None
    return {"min": limits[0], "max": limits[1], "mean": None,
            "shape": (model.get("shapes") or {}).get(name)}

class Handler(SimpleHTTPRequestHandler):
    server_version = "MeshJOracle/1"
    quiet = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB, **kwargs)

    def _json(self, payload, code=200):
        body = json.dumps(clean(payload), allow_nan=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError) as error:
            print(json.dumps({"event":"response_disconnect","path":self.path,"error":f"{type(error).__name__}: {error}"}), file=sys.stderr, flush=True)

    def do_GET(self):
        parsed = urlparse(self.path)
        route = parsed.path
        state = self.server.joracle
        if route == "/api/health":
            return self._json({
                "now": time.time(), "started": state["started"],
                "uptime": round(time.time() - state["started"], 1),
                "follower": state["follower"].status(),
                "probe_errors": state["probe"].errors,
                "runtime": state["runtime"].snapshot(),
            })
        if route == "/api/runtime":
            return self._json(state["runtime"].snapshot())
        if route == "/api/metrics":
            return self._json(summarize(state["follower"].snapshot()))
        if route == "/api/live":
            limit = int(parse_qs(parsed.query).get("limit", ["600"])[0])
            frames = state["follower"].snapshot()
            latest = frames[-1] if frames else None

            carrier, behind = None, None
            for index in range(len(frames) - 1, -1, -1):
                if (frames[index].get("model") or {}).get("j") is not None:
                    carrier, behind = frames[index], len(frames) - 1 - index
                    break
            detail = internals(carrier or latest)
            detail["ticks_behind"] = behind
            detail["sampled_stream"] = None if latest is None else (latest.get("model") or {}).get("sampled")
            return self._json({
                "now": time.time(),
                "follower": state["follower"].status(),
                "demo": state["demo"],
                "series": [series_row(f) for f in frames[-limit:]],
                "internals": detail,
                "field_measures": field_measures(latest, carrier),
            })
        if route == "/api/joracle":
            return self._json({"now": time.time(), **state["probe"].status()})
        if route == "/api/frame":
            latest = state["follower"].latest()
            return self._json(latest or {"available": False})
        if route == "/":
            self.path = "/index.html"
        return super().do_GET()

    def log_message(self, fmt, *args):
        if not self.quiet:
            print("[joracle] " + fmt % args, flush=True)

def main(argv=None):
    ap = argparse.ArgumentParser(description="j-oracle live viewer side channel")
    ap.add_argument("--telemetry", required=True,
                    help="responder telemetry JSONL: /abs/path or host:/abs/path")
    ap.add_argument("--bind", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8795)
    ap.add_argument("--frames", type=int, default=900, help="telemetry frames held in memory")
    ap.add_argument("--probe-rows", type=int, default=4000, help="player-rows in the rolling probe window")
    ap.add_argument("--probe-interval", type=float, default=4.0)
    ap.add_argument("--server-address", default="", help="what to tell a human to `connect` to")
    ap.add_argument("--map", dest="map_name", default="")
    ap.add_argument("--note", default="")
    ap.add_argument("--remote-host", default="")
    ap.add_argument("--remote-alias", default="mesh-mini")
    ap.add_argument("--remote-repo", default="/Users/mdot/mesh")
    ap.add_argument("--local-repo", default=REPO)
    ap.add_argument("--runtime-interval", type=float, default=2.0)
    args = ap.parse_args(argv)

    probe = RollingProbe(max_rows=args.probe_rows, interval=args.probe_interval).start()
    follower = TelemetryFollower(args.telemetry, capacity=args.frames, on_frame=probe.ingest,
                                 host_key_alias=args.remote_alias).start()
    source_host, _ = split_source(args.telemetry)
    runtime = RuntimeMeasure(args.local_repo, args.remote_host or source_host,
                             args.runtime_interval, args.remote_alias,
                             args.remote_repo).start()

    server = ThreadingHTTPServer((args.bind, args.port), Handler)
    server.daemon_threads = True
    server.joracle = {
        "follower": follower, "probe": probe, "runtime": runtime, "started": time.time(),
        "demo": {
            "telemetry": args.telemetry,
            "connect": args.server_address,
            "map": args.map_name,
            "note": args.note,
        },
    }
    print(f"[joracle] http://{args.bind}:{args.port}  telemetry={args.telemetry}", flush=True)
    if args.server_address:
        print(f"[joracle] xonotic client:  connect {args.server_address}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[joracle] interrupted", flush=True)
    finally:
        follower.stop()
        probe.stop()
        runtime.stop()
        server.server_close()

if __name__ == "__main__":
    main()
