"""The j-oracle side channel: a local HTTP server over the responder's telemetry.

    python3 -m solver.strat.joracle.server \
        --telemetry mesh-mini:/tmp/mesh-joracle/output/live.jsonl \
        --port 8795

It reads.  It never writes to the game, the responder, the mesh or a checkpoint.
It survives the responder and the game server being killed and restarted: the
follower reattaches, and the page keeps rendering the last frames it has with an
explicit staleness readout rather than a blank screen.

Endpoints
    /                 the viewer page (xonotic/solver/strat/web/)
    /api/live         behavior series + latest internals + field audit
    /api/joracle      the rolling probe report
    /api/frame        the single most recent raw telemetry frame, verbatim
    /api/health       liveness of the tap itself
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np

try:
    from .follow import TelemetryFollower
    from .probe import RollingProbe
    from .expect import audit
except ImportError:                                   # run as a plain script
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from solver.strat.joracle.follow import TelemetryFollower
    from solver.strat.joracle.probe import RollingProbe
    from solver.strat.joracle.expect import audit

WEB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")


def clean(value):
    """JSON-safe: NaN/Inf become null rather than invalid JSON."""
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
    """The compact per-tick record the behavior charts consume."""
    carts = frame.get("carts") or []
    return {
        "t": frame.get("t"),
        "seen_at": frame.get("_seen_at"),
        "epoch": frame.get("_epoch"),
        "resp_id": frame.get("resp_id"),
        "req_tick": frame.get("req_tick"),
        "k": frame.get("k"), "j": frame.get("j"), "l": frame.get("l"),
        "mode": frame.get("mode"), "trained": frame.get("trained"),
        "updates": frame.get("updates"),
        "PW": frame.get("PW"),
        "SUCC": frame.get("SUCC"),
        "loser_ranks": frame.get("loser_ranks"),
        "depth": [c.get("depth") for c in carts],
        "ctrl": [c.get("ctrl") for c in carts],
        "speed": [c.get("speed") for c in carts],
        "progress": [c.get("progress") for c in carts],
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
    """The policy internals panel for the newest frame, downsampled for the wire."""
    if not frame:
        return {"available": False}
    model = frame.get("model") or {}

    def matrix(name, cap_cols=None):
        value = model.get(name)
        if value is None:
            return None
        array = np.asarray(value, dtype=np.float64)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        if array.ndim != 2:
            return None
        if cap_cols and array.shape[1] > cap_cols:
            array = array[:, :cap_cols]
        return np.round(array, 5).tolist()

    ir = model.get("ir")
    ir_array = np.asarray(ir, dtype=np.float64) if ir is not None else None
    ir_stats = None
    if ir_array is not None and ir_array.ndim == 2 and ir_array.size:
        centered = ir_array - ir_array.mean(axis=0, keepdims=True)
        try:
            singular = np.linalg.svd(centered, compute_uv=False)
            spectrum = np.round(singular[:40] / max(singular[0], 1e-12), 5).tolist()
        except np.linalg.LinAlgError:
            spectrum = None
        ir_stats = {
            "shape": list(ir_array.shape),
            "min": round(float(ir_array.min()), 5),
            "max": round(float(ir_array.max()), 5),
            "std": round(float(ir_array.std()), 5),
            "frame_rank": int(np.linalg.matrix_rank(centered, tol=1e-6)),
            "spectrum": spectrum,
        }

    assignments = sorted(frame.get("assignments") or [], key=lambda a: a.get("row", 0))
    return {
        "available": True,
        "resp_id": frame.get("resp_id"),
        "ir": matrix("ir"),
        "ir_stats": ir_stats,
        "gram": matrix("gram"),
        "hierarchy": matrix("hierarchy"),
        "x": matrix("x"),
        "beta": matrix("beta"),
        "score_stats": _range(model.get("score")),
        "w_stats": _range(model.get("w")),
        "winner_value": clean(model.get("winner_value")),
        "loser_value": clean(model.get("loser_value")),
        "diag_k": matrix("diag_k"),
        "appetite": matrix("appetite", cap_cols=64),
        "dw_dt": matrix("dw_dt", cap_cols=64),
        "advantage": (frame.get("update") or {}).get("advantage"),
        "update": clean(frame.get("update")),
        "assignments": assignments,
        "shapes": model.get("shapes"),
        "finite": model.get("finite"),
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
        except (BrokenPipeError, ConnectionResetError):
            pass

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
            })
        if route == "/api/live":
            limit = int(parse_qs(parsed.query).get("limit", ["600"])[0])
            frames = state["follower"].snapshot()
            latest = frames[-1] if frames else None
            # The responder samples the big arrays every --model-sample-every
            # ticks, so the internals panel reads the newest frame that actually
            # carries them and says how far behind that is.  It never fills the
            # gap with the previous tick's numbers pretending to be this tick's.
            carrier, behind = None, None
            for index in range(len(frames) - 1, -1, -1):
                if (frames[index].get("model") or {}).get("ir") is not None:
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
                "audit": audit(latest, carrier),
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
    args = ap.parse_args(argv)

    if args.port == 26012:
        raise SystemExit("refusing port 26012")

    probe = RollingProbe(max_rows=args.probe_rows, interval=args.probe_interval).start()
    follower = TelemetryFollower(args.telemetry, capacity=args.frames, on_frame=probe.ingest).start()

    server = ThreadingHTTPServer((args.bind, args.port), Handler)
    server.daemon_threads = True
    server.joracle = {
        "follower": follower, "probe": probe, "started": time.time(),
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
        server.server_close()


if __name__ == "__main__":
    main()
