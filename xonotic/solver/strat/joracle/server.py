import argparse
import collections
import json
import os
from pathlib import Path
import shutil
import struct
import threading
import time
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit, parse_qs

from .display import compact_j_report, page_j_report, page_model
from .artifact import read_report
from .follow import RunReplica
from ..game_value import GAME_CONTRACT
from ..journal import Journal
from ..report_store import ReportStore, identity as record_identity
from ..policy_reports import reporting_coverage
from ..ratings import Ratings

WEB = Path(__file__).resolve().parents[1] / "web"


def finite_json(value):
    if isinstance(value, dict):
        return {key: finite_json(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(child) for child in value]
    return None if isinstance(value, float) and not (-float("inf") < value < float("inf")) else value


class Viewer:
    def __init__(self, root, capacity, state_directory=None):
        self.source = Path(root).absolute()
        self.root = self.source.resolve()
        state_directory = Path(state_directory) if state_directory else self.source.parent / '.joracle' / record_identity(str(self.source))[:20]
        self.store = ReportStore(state_directory / 'reports.sqlite3')
        self.journal = Journal(self.store)
        self.ratings = Ratings(self.store)
        self.history = []
        self.series = collections.deque(maxlen=capacity)
        self.latest, self.updates, self.j, self.model = {}, {}, {}, {}
        self.j_identity = self.directory = self.error = self.last_frame_at = None
        self.telemetry_path = None
        self.comparison_vectors = {}
        self.study, self.study_identity = {}, None
        self.report_errors = {}
        self.pending_source = None
        self.errors = self.sequence = 0
        self.cache, self.bodies = {}, {}
        self.revision = self.published_revision = 0
        self.viewer_id = uuid.uuid4().hex
        self.stopped = threading.Event()
        self.rounds = {}
        self.replica = None
        self.episode_contexts = {}
        snapshots = [record for record in self.store.records('viewer') if record[0] == str(self.source)]
        if snapshots:
            saved = snapshots[-1][2]
            for name in ('latest', 'updates', 'j', 'model', 'comparison_vectors', 'study', 'history'):
                setattr(self, name, saved.get(name, getattr(self, name)))
            self.series.extend(saved.get('series', ()))
            self.rounds = {tuple(key): value for key, value in saved.get('rounds', ())}
            self.episode_contexts = {tuple(key): value for key, value in saved.get('episode_contexts', ())}
            self.directory = Path(saved['directory']) if saved.get('directory') else None
            self.telemetry_path = Path(saved['telemetry_path']) if saved.get('telemetry_path') else None
            self.root = Path(saved.get('root', self.root))
            self.last_frame_at = saved.get('last_frame_at')
            self.revision += 1
        self.thread = threading.Thread(target=self.run, daemon=True)

    def remember_round(self, record):
        event = dict(record['event'])
        event['episode_id'] = event.get('episode_id', event.get('episode'))
        event['kind'] = event.get('kind', 'score_win' if event['actor_team'] > 0 else 'tie')
        record = {**record, 'source_run': str(self.root), 'event': event}
        key = (str(self.root), record["directory"], event.get('episode_id'), struct.pack("<f", event["time"]).hex(), event["actor_team"])
        previous = self.rounds.get(key, {})
        self.revision += 1
        self.rounds[key] = {**previous, **record, "event": {**previous.get("event", {}), **event}}
        self.ratings.outcome(str(self.root / record['directory']), event)

    def ingest(self, frame):
        key = frame.get('record_id') or record_identity([frame.get('producer_id'), frame.get('producer_pid'), frame.get('recorded_at', frame.get('at')), frame.get('resp_id'), frame.get('request_seq')])
        if not self.store.put('frames', str(self.directory), key, {'response': frame['resp_id']}):
            return
        self.revision += 1
        self.ratings.ingest(str(self.directory), frame)
        if self.latest and (frame["resp_id"] < self.latest["resp_id"] or frame.get("t", 0) < self.latest.get("t", 0)):
            self.series.clear()
        self.last_frame_at = time.time()
        self.latest = {key: value for key, value in frame.items() if key not in ('model', 'measure_sources', 'event_arrivals', 'server_state_labels', 'participant_state_labels')}
        self.latest['assignments'] = [{key: value for key, value in row.items() if key not in ('successor_state', 'successor_state_labels')}
            for row in frame.get('assignments', ())]
        self.latest['policy_comparison'] = {key: value for key, value in (frame.get('policy_comparison') or {}).items() if key != 'vectors'}
        for arm, update in frame.get("policy_updates", {}).items():
            if update and update.get("gradient_steps", 0):
                self.updates[arm] = {**update, "response": frame["resp_id"], "sampled_at": self.last_frame_at, "directory": self.directory.name}
        self.series.append({"response": frame["resp_id"], "t": frame.get("t"),
                            "learning": frame.get("learning", {}), "policy_updates": frame.get("policy_updates", {}),
                            "moe": frame.get("moe", {}),
                            "work": frame.get("work", {}),
                            "carts": [{key: cart.get(key) for key in ("id", "depth", "control_team")} for cart in frame.get("carts", [])]})
        model = frame.get("model", {})
        self.episode_contexts[(str(self.root), self.directory.name, frame.get('episode_id'))] = {
            'team_policy_arms': frame.get('team_policy_arms', []),
            'updates': {arm: state.get('updates') for arm, state in frame.get('learning', {}).items()},
            'observed_request_seq': frame.get('request_seq')}
        comparison = frame.get("policy_comparison") or {}
        if comparison.get("vectors"):
            self.comparison_vectors = {**comparison, 'source_directory': str(self.directory)}
        if model.get("sampled"):
            self.model = {key: model.get(key) for key in ("row_outputs", "j", "j_labels", "coupling", "row_output_shapes", "local_neighborhood", "moe")}
            self.model["response"] = frame["resp_id"]
            self.model["t"] = frame.get("t")
            self.model['source_directory'] = str(self.directory)
        for event in frame.get("realized_events", []):
            if event.get("kind") in ("score_win", "tie") and (frame.get("game_value") or {}).get("contract") == GAME_CONTRACT:
                context = self.episode_contexts.get((str(self.root), self.directory.name, event.get('episode_id')), {})
                self.remember_round({"directory": self.directory.name, "event": event,
                                    'episode_context_recorded': bool(context), **context})

    def refresh(self):
        self.store.put('sources', str(self.root), 'observed', {'path': str(self.root)})
        root = self.source.resolve()
        if root != self.root:
            self.pending_source = str(root)
            ready = any(next(Journal().read(path), None) is not None for path in sorted(root.glob("*/telemetry.jsonl"), reverse=True))
            if ready:
                self.root = root
                self.store.put('sources', str(root), 'observed', {'path': str(root)})
                self.revision += 1
                self.study_identity = None
                self.directory = self.telemetry_path = self.pending_source = None
        outcomes = self.root / "server-outcomes.jsonl"
        if outcomes.exists():
            for record in self.journal.read(outcomes):
                self.remember_round(record)
        index = self.root / "matches.jsonl"
        if index.exists():
            for record in self.journal.read(index):
                cfg, realized = record.get("configuration", {}), record.get("realized", {})
                if not self.store.put('matches', str(self.root), str(record.get('id', record_identity(record))), record):
                    continue
                self.revision += 1
                self.history.append({"source": str(self.root), "ordinal": record.get("ordinal"), "id": record.get("id"), "ended": record.get("ended"),
                                     "map": cfg.get("map"), "teams": cfg.get("teams"), "carts": cfg.get("carts"),
                                     "learning": realized.get("learning", {}), "controllers": realized.get("controllers", {}),
                                     "team_policy_arms": cfg.get("team_policy_arms", []),
                                     "policy_checkpoints": record.get("artifacts", {}).get("policy_checkpoints", {})})
        paths = [self.root] if self.root.is_file() else sorted(self.root.glob("*/telemetry.jsonl"))
        for path in reversed(paths):
            if path == self.telemetry_path:
                break
            frames = self.journal.read(path)
            first = next(frames, None)
            if first is not None:
                if self.directory:
                    for frame in self.journal.read(self.telemetry_path):
                        self.ingest(frame)
                self.directory = path.parent
                self.telemetry_path = path
                self.series.clear()
                self.latest = {}
                self.revision += 1
                self.j_identity = None
                self.ingest(first)
                for frame in frames:
                    self.ingest(frame)
                break
        if self.telemetry_path:
            path = self.telemetry_path
            for frame in self.journal.read(path):
                self.ingest(frame)
            views = list(self.directory.glob("j-measures.*.view.json")) + list(self.directory.glob("j-measures.*.view.npz"))
            reports = views or list(self.directory.glob("j-measures.*.json"))
            if reports:
                report = max(reports, key=lambda p: p.stat().st_mtime_ns)
                identity = (str(report), report.stat().st_mtime_ns)
                if identity != self.j_identity:
                    try:
                        if report.suffix == '.npz':
                            data = read_report(report)
                        else:
                            with report.open() as handle:
                                data = json.load(handle)
                        self.j = data if views else {"sampled_at": data.get("sampled_at"), "generation": data.get("generation"), "full_artifact": str(report), **compact_j_report(data)}
                        self.j['source_directory'] = str(self.directory)
                        if self.j.get('full_artifact'):
                            self.j['full_artifact'] = str(report.parent / Path(self.j['full_artifact']).name)
                        self.j_identity = identity
                        self.revision += 1
                        self.report_errors.pop("j", None)
                    except (OSError, ValueError) as error:
                        self.report_errors["j"] = f"{report}: {type(error).__name__}: {error}"
        study_path = self.root / "study.json"
        if study_path.exists():
            identity = study_path.stat().st_mtime_ns
            if identity != self.study_identity:
                try:
                    with study_path.open() as handle:
                        study = json.load(handle)
                    self.study = {key: study.get(key) for key in ("record_count", "paired_block_count", "paired_round_count", "ratings", "rating_measure")}
                    self.study_identity = identity
                    self.revision += 1
                    self.report_errors.pop("ratings", None)
                except (OSError, ValueError) as error:
                    self.report_errors["ratings"] = f"{study_path}: {type(error).__name__}: {error}"
        composition_ratings = self.ratings.refresh()
        frame = self.latest
        self.sequence += 1
        status = {"sequence": self.sequence, 'content_revision': self.revision, 'viewer_id': self.viewer_id, "sampled_at": time.time(), "frame_seen_at": self.last_frame_at,
                  "producer_state": "awaiting_first_frame" if not frame else ("stale" if self.telemetry_path is None or not self.telemetry_path.exists() or time.time() - self.telemetry_path.stat().st_mtime > 15 else "advancing"),
                  "frame_written_at": self.telemetry_path.stat().st_mtime if self.telemetry_path and self.telemetry_path.exists() else None,
                  "host": os.uname().nodename, "run_directory": str(self.root),
                  "pending_source": self.pending_source, "journal_errors": self.journal.errors,
                  "retained_sources": [source for source, key, value in self.store.records('sources')],
                  "pending_directory": str(paths[-1].parent) if paths and paths[-1] != self.telemetry_path else None,
                  "directory": str(self.directory) if self.directory else None, "environment": frame.get("environment"),
                  "teams": frame.get("k"), "carts": frame.get("j"), "players": frame.get("l"), "response": frame.get("resp_id"),
                  "game_contract": (frame.get("game_value") or {}).get("contract"),
                  "reports": reporting_coverage(frame, self.j, self.study, composition_ratings), "report_errors": self.report_errors,
                  'report_sources': {'j': self.j.get('source_directory'), 'model': self.model.get('source_directory'),
                      'counterfactual_vectors': self.comparison_vectors.get('source_directory')},
                  "replication": self.replica.status if self.replica else None,
                  "application": frame.get("application"),
                  "viewer_application": json.loads(os.environ.get("MESH_APPLICATION_IDENTITY", '{"state":"unbundled"}')),
                  "errors": self.errors, "last_error": self.error}
        if self.revision != self.published_revision or not self.bodies:
            assignments = [{key: row.get(key) for key in ("edict", "team", "controller", "behavior", "policy_arm", "state_width", "velocity_l2", "residual_l2", "applied_response_seq", "request_seq")} for row in frame.get("assignments", [])]
            payloads = {
                "status": status,
                "j": {"status": status, "measure": page_j_report(self.j), "model": page_model(self.j.get("model") or self.model),
                      "learning": frame.get("learning", {}),
                      "series": [{"t": row["t"], "carts": row["carts"]} for row in self.series],
                      "assignments": assignments, "focus": page_model(self.model).get('coupling')},
                "policy": {"status": status, "learning": frame.get("learning", {}), "last_updates": self.updates,
                           "comparison": {key: value for key, value in (frame.get("policy_comparison") or {}).items() if key not in ("vectors", "instruments")},
                           "comparison_vector_sample": {key: self.comparison_vectors.get(key) for key in ("sampled_at", "response", "producer_pid")},
                           "study": self.study, "composition_ratings": composition_ratings,
                           "work": frame.get("work", {}),
                           "series": list(self.series), "history": self.history,
                           "rounds": list(self.rounds.values()), "round_coverage": "engine_journal_plus_viewer_observed_sessions; telemetry_gaps_not_imputed",
                           "provenance": frame.get("policy_provenance", {}), "team_policy_arms": frame.get("team_policy_arms", []),
                           "human_rows": sum(row["controller"] == "human" for row in assignments)},
            }
            self.bodies = {key: json.dumps(finite_json({name: item for name, item in value.items() if name != 'status'}), separators=(',', ':'), allow_nan=False).encode()
                for key, value in payloads.items() if key != 'status'}
            self.cache['counterfactual/full'] = json.dumps(finite_json(self.comparison_vectors), separators=(',', ':'), allow_nan=False).encode()
            saved = {name: getattr(self, name) for name in ('latest', 'updates', 'j', 'model', 'comparison_vectors', 'study', 'history', 'last_frame_at')}
            saved['latest'] = {key: value for key, value in self.latest.items() if key not in ('measure_sources', 'model', 'server_state_labels', 'participant_state_labels')}
            saved.update(series=list(self.series), rounds=list(self.rounds.items()),
                episode_contexts=list(self.episode_contexts.items()), root=str(self.root),
                directory=str(self.directory) if self.directory else None,
                telemetry_path=str(self.telemetry_path) if self.telemetry_path else None)
            self.store.put('viewer', str(self.source), 'current', finite_json(saved))
            self.published_revision = self.revision
        encoded_status = json.dumps(finite_json(status), separators=(',', ':'), allow_nan=False).encode()
        self.cache = {**self.cache, 'status': encoded_status,
            **{key: b'{"status":' + encoded_status + b',' + value[1:] for key, value in self.bodies.items()}}

    def run(self):
        while not self.stopped.is_set():
            try:
                with self.store.db:
                    self.refresh()
                self.error = None
            except Exception as error:
                self.errors += 1
                self.error = f"{type(error).__name__}: {error}"
                status = {**json.loads(self.cache.get("status", b'{}')), "producer_state": "reader_error",
                          "errors": self.errors, "last_error": self.error}
                for key in ("status", "j", "policy"):
                    payload = json.loads(self.cache.get(key, b'{}'))
                    self.cache[key] = json.dumps(status if key == "status" else {**payload, "status": status}).encode()
                print(self.error, flush=True)
            self.stopped.wait(2)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB), **kwargs)

    def do_GET(self):
        route = urlsplit(self.path).path
        state = self.server.viewer
        if route == "/api/j/full":
            try:
                path = Path(state.j["full_artifact"])
                handle = path.open("rb")
            except (KeyError, OSError, TypeError) as error:
                self.send_error(503, f"Full artifact unavailable: {error}")
                return
            with handle:
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream" if path.suffix == ".npz" else "application/json")
                self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
                self.send_header("Content-Length", str(os.fstat(handle.fileno()).st_size))
                self.end_headers()
                shutil.copyfileobj(handle, self.wfile)
            return
        if route.startswith("/api/"):
            body = state.cache.get(route[5:])
            if body is None:
                known = route[5:] in ('status', 'j', 'policy', 'counterfactual/full')
                self.send_error(503 if known else 404, "Report not yet available" if known else "Unknown API route")
                return
            if route == '/api/j' and urlsplit(self.path).query:
                query = parse_qs(urlsplit(self.path).query)
                offset = max(0, int(query.get('offset', ['0'])[0]))
                width = max(1, int(query.get('width', ['200'])[0]))
                payload = json.loads(body)
                payload.update(measure=page_j_report(state.j, query.get('stratum', [''])[0],
                    query.get('filter', [''])[0], offset, width), model=page_model(state.j.get('model') or state.model, offset, width),
                    focus=page_model(state.model, offset, width).get('coupling'))
                body = json.dumps(finite_json(payload), separators=(',', ':'), allow_nan=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.path = {"/": "/index.html", "/j": "/index.html", "/policy": "/policy.html"}.get(route, route)
        super().do_GET()

    def log_message(self, *args):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", "--telemetry", dest="root", required=True)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8795)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--state-directory")
    parser.add_argument("--replicate-from", help="host:path, a local path, or mesh:xonotic to follow the discovered producer")
    parser.add_argument("--replica-source-file")
    args = parser.parse_args()
    replica = RunReplica(args.replicate_from, args.root, args.replica_source_file) if args.replicate_from or args.replica_source_file else None
    state = Viewer(str(Path(args.root) / 'active') if replica else args.root, args.frames, args.state_directory)
    state.replica = replica
    server = ThreadingHTTPServer((args.bind, args.port), Handler)
    server.viewer = state
    state.thread.start()
    if replica:
        replica.thread.start()
    print(f"J viewer http://{args.bind}:{args.port}/j; policy viewer http://{args.bind}:{args.port}/policy", flush=True)
    try:
        server.serve_forever()
    finally:
        state.stopped.set()
        if replica:
            replica.stopped.set()
        state.thread.join(timeout=5)
        if replica:
            replica.thread.join(timeout=5)
        if not state.thread.is_alive():
            state.store.db.close()
        server.server_close()


if __name__ == "__main__":
    main()
