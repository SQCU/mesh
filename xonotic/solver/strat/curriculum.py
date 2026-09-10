import argparse, copy, datetime, glob, itertools, json, math, os, random, shlex, shutil, signal, subprocess, sys, time

from solver.strat.scale_config import SCALE_EXPERTS, SCALE_HIDDEN, SCALE_RANK, SCALE_TOPK
from solver.strat.policy_contract import MATRIX_FUSION_INTERVENTION_ARMS, OPTIMIZATION_ARMS, JOINT_TRAINING_ARMS, is_matrix_fusion_arm, checkpoint_path
from solver.strat.capacity import cart_capacity, engine_player_capacity, team_capacity
from solver.strat.journal import Journal
from solver.strat.checkpoint_state import checkpoint_source
from solver.strat.action_history import ExecutionEvaluation
from solver.strat.map_assets import MapAssets, artifact, discover_maps, resolve_maps
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def payload_defaults():
    out = {}
    path = os.path.join(ROOT, "payload", "cfg", "gamemodes-payload.cfg")
    try:
        with open(path) as stream:
            for raw in stream:
                fields = raw.split('"', 1)[0].split()
                if len(fields) >= 3 and fields[0] in ("set", "seta"):
                    try:
                        out[fields[1]] = float(fields[2])
                    except ValueError:
                        out[fields[1]] = fields[2]
    except OSError as error:
        print(json.dumps({"event":"payload_defaults_read_error","path":path,"error":f"{type(error).__name__}: {error}"}), file=sys.stderr)
    return out

PAYLOAD_DEFAULTS = payload_defaults()
PERTURBATIONS = {
    "baseline": {},
    "fast": {"g_payload_speed": 45, "g_payload_max_speed": 260, "g_payload_push_falloff": 1.0},
    "slow": {"g_payload_speed": 20, "g_payload_max_speed": 120, "g_payload_push_falloff": 0.5},
    "volatile": {
        "g_payload_contest_speed": 36,
        "g_payload_reverse_speed": 24,
        "g_payload_push_falloff": 0.75,
    },
}
RESPAWN_PERTURBATIONS = {
    "fast": (0.5, 0.5, 1.0, 1.5),
    "slow": (2.0, 1.0, 2.0, 3.0),
    "volatile": (1.0, 0.5, 2.0, 4.0),
}

def utcnow():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def csv(text, cast=str):
    return [cast(value.strip()) for value in text.split(",") if value.strip()]

def command(value):
    if value is None:
        return []
    if isinstance(value, str):
        return shlex.split(value)
    return [str(value) for value in value]

def runtime_identity(python):
    helper = shutil.which("mesh-runtime-id.py") or os.path.join(os.path.dirname(ROOT), "bin", "mesh-runtime-id.py")
    try:
        result = subprocess.run([python, helper], capture_output=True, text=True, timeout=10, check=True)
        return json.loads(result.stdout)
    except Exception as exc:
        return {"schema": 1, "launcher": "mesh-python", "error": f"{type(exc).__name__}: {exc}"}

def merge(left, right):
    out = copy.deepcopy(left)
    for key, value in right.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out

def load_manifest(path):
    with open(path) as handle:
        text = handle.read()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(payload, list):
        return payload
    defaults = payload.get("defaults", {})
    train = [merge(defaults, item) for item in payload.get("matches", [])]
    heldout = [merge(defaults, merge({"split": "heldout"}, item)) for item in payload.get("heldout", [])]
    if not train and not heldout:
        return [merge(defaults, payload)]
    return train + heldout

def generated_schedule(count, seed, maps, teams, players, carts, skills, perturbations, off_policy, policy_arms, heldout_fraction, human_counts=None, human_client_command=None, include_comparisons=True):
    space = list(itertools.product(maps, teams, players, carts, skills, perturbations, off_policy, human_counts or [0]))
    rng = random.Random(seed)
    heldout_count = min(len(space), max(0, round(len(space) * heldout_fraction)))
    heldout = set(rng.sample(space, heldout_count))
    rng.shuffle(space)
    out = []
    ordinal = 0
    for index in range(count):
        if index and index % len(space) == 0:
            rng.shuffle(space)
        mapname, team_count, ppt, cart_count, skill, perturbation, off, requested_humans = space[index % len(space)]
        humans = min(requested_humans, team_count * ppt)
        split = "heldout" if space[index % len(space)] in heldout else "train"
        match_seed = rng.randrange(1, 2 ** 31)
        for arm in policy_arms:
            out.append({
                "id": f"generated-{ordinal:05d}-{arm}",
                "map": mapname,
                "teams": team_count,
                "players_per_team": ppt,
                "carts": cart_count,
                "controllers": {"bot": team_count * ppt - humans, "human": humans},
                "skill": skill,
                "perturbation": perturbation,
                "off_policy_players": min(off, team_count * ppt),
                "client_commands": [human_client_command] * humans if human_client_command else [],
                "split": split,
                "seed": match_seed,
                "policy_arm": arm,
                "pair": index,
            })
            ordinal += 1
        if include_comparisons:
            for left, right in itertools.combinations(policy_arms, 2):
                for leg in range(2):
                    pair_arms = (left, right) if leg == 0 else (right, left)
                    out.append({
                        "id": f"generated-{ordinal:05d}-{left}-vs-{right}-leg{leg + 1}",
                        "map": mapname,
                        "teams": team_count,
                        "players_per_team": ppt,
                        "carts": cart_count,
                        "controllers": {"bot": team_count * ppt - humans, "human": humans},
                        "skill": skill,
                        "perturbation": perturbation,
                        "off_policy_players": 0,
                        "client_commands": [human_client_command] * humans if human_client_command else [],
                        "split": "heldout",
                        "seed": match_seed,
                        "policy_arm": "mixed",
                        "team_policy_arms": [pair_arms[t % 2] for t in range(team_count)],
                        "pair": index,
                        "leg": leg + 1,
                    })
                    ordinal += 1
    return out

def study_schedule(repetitions, seed, maps, teams, players, carts, skills, perturbations, policy_arms, arm_checkpoints=None, map_offset=0):
    rng = random.Random(seed)
    checkpoints = {} if arm_checkpoints is None else dict(arm_checkpoints)
    out = []
    ordinal = 0
    comparisons = list(itertools.combinations(policy_arms, 2))
    if "matrix_fusion" in policy_arms:
        comparisons.append(("matrix_fusion", "initial_policy"))
        comparisons.append(("matrix_fusion", "participant_fusion_ablated"))
        comparisons.append(("matrix_fusion", "residual_fusion_ablated"))
    for pindex, perturbation in enumerate(perturbations):
        for repetition in range(repetitions):
            index = pindex * repetitions + repetition
            mapname = maps[(map_offset + index) % len(maps)]
            team_count = teams[index % len(teams)]
            ppt = players[(index // max(1, len(teams))) % len(players)]
            cart_count = carts[(index // max(1, len(teams) * len(players))) % len(carts)]
            skill = skills[(index // max(1, len(teams) * len(players) * len(carts))) % len(skills)]
            match_seed = rng.randrange(1, 2 ** 31)
            for first, second in comparisons:
                for leg in range(2):
                    pair = (first, second) if leg == 0 else (second, first)
                    out.append({
                        "id": f"study-{ordinal:05d}-{first}-vs-{second}-{perturbation}-r{repetition + 1}-leg{leg + 1}",
                        "map": mapname,
                        "teams": team_count,
                        "players_per_team": ppt,
                        "carts": cart_count,
                        "controllers": {"bot": team_count * ppt},
                        "skill": skill,
                        "perturbation": perturbation,
                        "off_policy_players": 0,
                        "split": "heldout",
                        "seed": match_seed,
                        "policy_arm": "mixed",
                        "team_policy_arms": [pair[team % 2] for team in range(team_count)],
                        "arm_checkpoints": checkpoints,
                        "distributed_scale": True,
                        "pair": index,
                        "study_repetition": repetition + 1,
                        "leg": leg + 1,
                    })
                    ordinal += 1
    return out

def normalize(item, index, defaults):
    cfg = merge(defaults, item)
    cfg["id"] = str(cfg.get("id", f"match-{index:05d}"))
    cfg["map"] = str(cfg.get("map", defaults.get("map", "runningmanctf")))
    cfg["teams"] = int(cfg.get("teams", 2))
    cfg["carts"] = int(cfg.get("carts", 2))
    ppt = cfg.get("players_per_team", 2)
    if isinstance(ppt, list):
        values = [int(value) for value in ppt] or [2]
        cfg["players_per_team"] = [values[i % len(values)] for i in range(cfg["teams"])]
    else:
        cfg["players_per_team"] = [int(ppt)] * cfg["teams"]
    total = sum(cfg["players_per_team"])
    controllers = cfg.get("controllers", {})
    if isinstance(controllers, str):
        controllers = {controllers: total}
    cfg["controllers"] = {
        "bot": int(controllers.get("bot", total)),
        "human": int(controllers.get("human", 0)),
        "external": int(controllers.get("external", 0)),
    }
    cfg["skill"] = float(cfg.get("skill", 5))
    cfg["duration"] = float(cfg.get("duration", defaults.get("duration", 600)))
    cfg["off_policy_players"] = int(cfg.get("off_policy_players", 0))
    cfg["seed"] = int(cfg.get("seed", defaults.get("seed", 20260830) + index))
    cfg["split"] = str(cfg.get("split", "train"))
    cfg["policy_arm"] = str(cfg.get("policy_arm", defaults.get("policy_arm", "matrix_fusion")))
    team_policy_arms = cfg.get("team_policy_arms", [])
    if isinstance(team_policy_arms, str):
        team_policy_arms = csv(team_policy_arms)
    cfg["team_policy_arms"] = [str(value) for value in team_policy_arms]
    return cfg

def remote_scale_arm_mass(cfg):
    arms = cfg.get("team_policy_arms") or [cfg.get("policy_arm")]
    return sum(is_matrix_fusion_arm(str(arm)) for arm in arms)

def cvar_args(values):
    out = []
    for key in sorted(values):
        out.extend([f"+{key}", str(values[key])])
    return out

def client_command(value, context, index):
    out = command(value)
    replacements = merge(context, {"client": index})
    for key, replacement in replacements.items():
        out = [token.replace("{" + key + "}", str(replacement)) for token in out]
    return out

def telemetry_summary(path):
    evaluation = ExecutionEvaluation()
    configurations, controllers, arms, first, last, last_config, lines = {}, {}, {}, None, None, None, 0
    if not os.path.exists(path):
        return {"lines": 0, "configurations": [], "controllers": {}}
    with open(path) as handle:
        for raw in handle:
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            evaluation.ingest(row)
            lines += 1
            arm = str(row.get("policy_arm", "unknown"))
            arms[arm] = arms.get(arm, 0) + 1
            first = row if first is None else first
            last = row
            assignments = row.get("assignments", [])
            counts, team_counts = {}, {}
            for assignment in assignments:
                kind = str(assignment.get("controller", "unknown"))
                counts[kind] = counts.get(kind, 0) + 1
                controllers[kind] = controllers.get(kind, 0) + 1
                team = str(assignment.get("team", "unknown"))
                team_counts[team] = team_counts.get(team, 0) + 1
            last_config = {
                "teams": row.get("k"),
                "carts": row.get("j"),
                "players": row.get("l"),
                "players_per_team": team_counts,
                "controllers": counts,
            }
            key = json.dumps(last_config, sort_keys=True)
            configurations[key] = configurations.get(key, 0) + 1
    configuration_rows = [
        merge(json.loads(key), {"observations": value})
        for key, value in sorted(configurations.items())
    ]
    peak_config = max(
        configuration_rows,
        key=lambda row: (int(row.get("players") or 0), int(row.get("observations") or 0)),
        default=None,
    )
    return {
        "lines": lines,
        "execution_history": evaluation.report(),
        "first_tick": first.get("req_tick") if first else None,
        "last_tick": last.get("req_tick") if last else None,
        "responses": last.get("resp_id") if last else 0,
        "last_configuration": last_config,
        "peak_configuration": peak_config,
        "configurations": configuration_rows,
        "controllers": controllers,
        "policy_arms": arms,
        "policy_provenance": (last or {}).get("policy_provenance", {}),
        "learning": (last or {}).get("learning", {}),
        "updates": int((last or {}).get("updates", 0)),
    }

def runtime_log_measure(path):
    lines = 0
    size = None
    error = None
    try:
        size = os.path.getsize(path)
        with open(path, errors="replace") as handle:
            for number, _ in enumerate(handle, 1):
                lines = number
    except OSError as exc:
        error = f"{type(exc).__name__}: {exc}"
    return {"path": path, "bytes": size, "lines": lines, "read_error": error}

# ../../../design/algorithm-sources.md#configuration-storage-layout
def application_command(root, python, module):
    paths = [os.path.join(root, suffix) for suffix in ('xonotic', 'rdma', 'xonotic/payload/tools')]
    return ['env', 'PYTHONPATH=' + os.pathsep.join(paths), python, '-m', module]


# ../../../design/algorithm-sources.md#configuration-storage-layout
def application_revision(root, ssh=(), host=None):
    # ../../../design/algorithm-sources.md#configuration-storage-layout
    def git(*arguments):
        values = ['git', '-C', root, *arguments]
        values = [*ssh, host, shlex.join(values)] if host else values
        return subprocess.check_output(values, text=True).strip()
    if git('status', '--porcelain'):
        raise RuntimeError(f'{host or "local"}:{root}: source must be committed before evaluation')
    git('checkout', 'main')
    return git('rev-parse', 'HEAD')


class Curriculum:
    def __init__(self, args):
        self.args = args
        self.run_dir = os.path.abspath(os.path.expanduser(args.run_dir))
        self.server_prefix = command(args.server_command) or [os.path.abspath(os.path.expanduser(args.engine))]
        application_root = os.path.dirname(ROOT)
        self.responder_prefix = command(args.responder_command) or application_command(application_root, args.python, 'solver.strat.strat_responder')
        self.expert_prefix = command(args.expert_command) or application_command(application_root, args.python, 'solver.strat.matrix_worker')
        self.ssh_prefix = command(args.ssh_command) or ["ssh"]
        self.basedir = os.path.abspath(os.path.expanduser(args.basedir))
        self.entity_tool = os.path.abspath(os.path.expanduser(args.entity_tool))
        self.server_host = args.server_host
        self.remote_run_root = os.path.expanduser(args.remote_run_root)
        self.remote_mesh_root = args.remote_mesh_root
        self.remote_engine = os.path.expanduser(args.remote_engine) if args.remote_engine else os.path.join(self.remote_mesh_root, "xonotic", "darkplaces-work", "darkplaces-dedicated")
        self.remote_basedir = os.path.expanduser(args.remote_basedir) if args.remote_basedir else "/Users/mdot/mesh-workloads/cartlane/Xonotic"
        self.progs = os.path.abspath(os.path.expanduser(args.progs))
        self.csprogs = os.path.abspath(os.path.expanduser(args.csprogs))
        self.build_command = command(args.build_command)
        self.runtime = runtime_identity(args.python)
        if not args.dry_run:
            revision = application_revision(application_root)
            self.runtime['application_revision'] = revision
            if self.server_host:
                remote = application_revision(self.remote_mesh_root, self.ssh_prefix, self.server_host)
                if remote != revision:
                    raise RuntimeError(f'application revisions differ: local={revision}, {self.server_host}={remote}; synchronize committed main before evaluation')
                self.runtime['remote_application_revision'] = remote
        self.generated_checkpoints = set()
        self.generated_bundles = set()
        self.previous_checkpoints = {}
        self.initial_checkpoints = {}
        self.capacity_observations = []
        self.artifact_cache = {}
        self.map_assets = MapAssets(self.basedir, self.entity_tool, args.python, self.progs,
                                    self.csprogs, args.checkpoints_per_lane, args.dry_run, self.artifact_cache)
        self.next_ordinal = 0
        self.next_cycle = 0
        self.stopping = 0
        self.server = None
        self.expert = None
        self.expert_stop = []
        self.clients = []
        if args.checkpoint:
            self.previous_checkpoints["matrix_fusion"] = os.path.abspath(os.path.expanduser(args.checkpoint))
            self.initial_checkpoints["matrix_fusion"] = self.previous_checkpoints["matrix_fusion"]
        for value in args.arm_checkpoint:
            arm, separator, path = value.partition("=")
            if separator:
                self.previous_checkpoints[arm] = os.path.abspath(os.path.expanduser(path))
                self.initial_checkpoints[arm] = self.previous_checkpoints[arm]
        os.makedirs(self.run_dir, exist_ok=True)
        self.index_path = os.path.join(self.run_dir, "matches.jsonl")
        self.event_path = os.path.join(self.run_dir, "supervisor.jsonl")
        open(self.index_path, "a").close()
        self.restore_history()
        self.maps = resolve_maps(args.maps, self.basedir)
        self.build = self.build_gamecode()

    def request_stop(self, signum, frame):
        self.stopping = signum

    def restore_history(self):
        rows = []
        try:
            with open(self.index_path) as stream:
                for line in stream:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(row, dict):
                        rows.append(row)
        except OSError as error:
            print(json.dumps({"event":"history_read_error","path":self.index_path,"error":f"{type(error).__name__}: {error}"}), file=sys.stderr)
        active = os.path.join(self.run_dir, "active-match.json")
        try:
            with open(active) as stream:
                pending = json.load(stream)
            if pending["ordinal"] > max((row.get("ordinal", -1) for row in rows), default=-1):
                rows.append(pending)
        except FileNotFoundError:
            pass
        except (OSError, ValueError, KeyError) as error:
            print(json.dumps({"event":"history_read_error","path":active,"error":f"{type(error).__name__}: {error}"}), file=sys.stderr)
        self.next_ordinal = max((int(row.get("ordinal", -1)) for row in rows), default=-1) + 1
        self.next_cycle = max((int(row.get("cycle", row.get("configuration", {}).get("cycle", -1))) for row in rows), default=-1) + 1
        recovered = {}
        for record in reversed(rows):
            cfg = record.get("configuration") or {}
            checkpoint = ((record.get("artifacts") or {}).get("checkpoint_out") or {}).get("path")
            if cfg.get("split") != "heldout" and checkpoint and os.path.isfile(checkpoint):
                recovered.setdefault(str(cfg.get("policy_arm", "matrix_fusion")), checkpoint)
            for arm, saved in (record.get("artifacts", {}).get("policy_checkpoints") or {}).items():
                path = saved.get("path")
                if cfg.get("split") != "heldout" and path and os.path.isfile(path):
                    recovered.setdefault(arm, path)
            profile = (record.get("execution") or {}).get("operating_profile") or {}
            point = profile.get("target_center_observation")
            if point:
                self.capacity_observations.append({
                    "teams": point.get("teams", cfg.get("teams")),
                    "carts": point.get("carts", cfg.get("carts")),
                    "players": point.get("players"), "point": point,
                    "environment": profile.get("environment"),
                })
        self.previous_checkpoints.update(recovered)
        for record in rows:
            cfg = record.get("configuration") or {}
            initial = ((record.get("artifacts") or {}).get("checkpoint_initial") or {}).get("path")
            if cfg.get("split") != "heldout" and initial and os.path.isfile(initial):
                self.initial_checkpoints.setdefault(str(cfg.get("policy_arm", "matrix_fusion")), initial)

    def observe_capacity(self, cfg, execution):
        profile = execution.get("operating_profile") or {}
        point = profile.get("target_center_observation")
        if not point:
            return
        row = {
            "teams": point.get("teams", cfg["teams"]),
            "carts": point.get("carts", cfg["carts"]),
            "players": point.get("players"), "point": point,
            "environment": profile.get("environment"),
        }
        self.capacity_observations.append(row)
        self.event("capacity_observation", **row)

    def build_gamecode(self):
        record = {"command": self.build_command}
        if self.args.dry_run:
            return record | {"dry_run": True}
        started = utcnow()
        try:
            result = subprocess.run(self.build_command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            return record | {"returncode": result.returncode,
                             "output": result.stdout, "started": started, "ended": utcnow()}
        except Exception as exc:
            return record | {"returncode": None, "error": f"{type(exc).__name__}: {exc}",
                             "started": started, "ended": utcnow()}

    def match_dir(self, cfg):
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in cfg["id"])
        return os.path.join(self.run_dir, f"{cfg['ordinal']:05d}-{safe}")

    def perturbation(self, cfg, entity):
        value = cfg.get("perturbation", "baseline")
        if isinstance(value, dict):
            name = str(value.get("name", "custom"))
            values = value.get("cvars", {key: item for key, item in value.items() if key != "name"})
        else:
            name = str(value)
            values = PERTURBATIONS.get(name, {})
        values = merge(values, cfg.get("server_cvars", {}))
        if name in RESPAWN_PERTURBATIONS:
            total = max(1, sum(cfg["players_per_team"]))
            measured_spawns = int((entity.get("measurements") or {}).get("generic_spawns") or total)
            configured_slots = int(values.get("g_spawn_swizzle_slots", PAYLOAD_DEFAULTS.get("g_spawn_swizzle_slots", 0)))
            slots = max(1, min(
                total, measured_spawns,
                configured_slots if configured_slots > 0 else max(total, measured_spawns),
            ))
            step_factor, small_cycles, large_cycles, max_cycles = RESPAWN_PERTURBATIONS[name]
            step = float(values.get("g_spawn_swizzle_step", PAYLOAD_DEFAULTS.get("g_spawn_swizzle_step", 0))) * step_factor
            cycle = math.ceil(total / slots) * step
            values.update({
                "g_spawn_swizzle": 1,
                "g_spawn_swizzle_slots": slots,
                "g_spawn_swizzle_step": step,
                "g_respawn_delay_small": cycle * small_cycles,
                "g_respawn_delay_small_count": slots,
                "g_respawn_delay_large": cycle * large_cycles,
                "g_respawn_delay_large_count": max(slots + 1, total - slots),
                "g_respawn_delay_max": cycle * max_cycles,
                "g_respawn_delay_forced": 1,
                "g_respawn_waves": 1,
            })
        return name, values

    def commands(self, cfg, directory, entity):
        port = int(cfg.get("port", self.args.port_base))
        players = max(sum(cfg["players_per_team"]), sum(cfg["controllers"].values()))
        maxplayers = int(cfg.get("maxplayers") or engine_player_capacity(players))
        initial_bots = cfg["controllers"]["bot"]
        server_region = str(cfg.get("server_mesh_region", self.args.server_mesh_region))
        responder_region = str(cfg.get("responder_mesh_region", self.args.responder_mesh_region))
        perturbation, cvars = self.perturbation(cfg, entity)
        strategy_node = cfg.get("strategy_node", self.args.strategy_node)
        peer_node = cfg.get("peer_node", self.args.peer_node)
        strategy_node = (0 if self.server_host else 1) if strategy_node is None else int(strategy_node)
        peer_node = (1 if self.server_host else 0) if peer_node is None else int(peer_node)
        distributed_scale = self.args.distributed_scale and bool(cfg.get("distributed_scale", True))
        cvars["g_payload_teams_override"] = cfg["teams"]
        cvars["g_payload_mesh_node"] = strategy_node
        try:
            requested_autoscreenshot_mass = int(float(cvars.get("g_max_info_autoscreenshot", 0)))
        except (TypeError, ValueError):
            requested_autoscreenshot_mass = 0
        cvars["g_max_info_autoscreenshot"] = max(
            requested_autoscreenshot_mass,
            int(entity.get("entity_class_mass", {}).get("info_autoscreenshot", 0)),
        )
        stage = []
        remote_directory = None
        server_prefix = [
            "env", f"MESH_REGION={server_region}",
            f"MESH_EXPERT_SOCKET={self.args.expert_socket}",
        ] + self.server_prefix
        basedir = self.basedir
        userdir = os.path.join(self.run_dir, "userdir")
        stage.append(["rsync", "-a", "--delete", entity["userdir"] + "/", userdir + "/"])
        if self.server_host:
            remote_directory = os.path.join(self.remote_run_root, os.path.basename(self.run_dir))
            userdir = os.path.join(remote_directory, "userdir")
            basedir = self.remote_basedir
            server_prefix = self.ssh_prefix + [
                self.server_host, "--", "env", f"MESH_REGION={server_region}",
                f"MESH_EXPERT_SOCKET={self.args.expert_socket}", self.remote_engine,
            ]
            stage = [
                self.ssh_prefix + [self.server_host, "--", "mkdir", "-p", userdir],
                ["rsync", "-a", "--delete", "-e", shlex.join(self.ssh_prefix), entity["userdir"] + "/", f"{self.server_host}:{userdir}/"],
            ]
        server_values = {
            "developer": 0, "sv_public": 0, "port": port,
            "sv_random_seed": cfg["seed"], "sv_autopause": 0,
            "timelimit": 0, "maxplayers": maxplayers, "bot_join_empty": 1,
            "bot_number": initial_bots, "skill": cfg["skill"], "g_warmup": 0,
            "g_maplist": cfg["map"], "g_maplist_shuffle": 0, "g_maplist_selectrandom": 0,
            **PAYLOAD_DEFAULTS, **cvars, "g_payload": 1,
            "g_payload_point_limit": cfg.get("score_limit", self.args.score_limit),
            "g_payload_checkpoint_rate": cfg.get("checkpoint_score_rate", self.args.checkpoint_score_rate),
        }
        server = server_prefix + [
            "-norunaway", "-xonotic", "-basedir", basedir, "-userdir", userdir,
            "+exec", "gamemodes-payload.cfg",
        ] + cvar_args(server_values) + command(cfg.get("server_args")) + ["+map", cfg["map"]]
        map_transition = "".join(f"{name} {json.dumps(str(value))}\n" for name, value in server_values.items()
                             if name not in ("port", "maxplayers")) + f"changelevel {cfg['map']}\n"
        telemetry = os.path.join(directory, "telemetry.jsonl")
        checkpoint_out = os.path.join(directory, "checkpoint.npz")
        checkpoint_initial = os.path.join(directory, "checkpoint.initial.npz")
        learning_rate = 0.0 if cfg["split"] == "heldout" else float(cfg.get("learning_rate", self.args.learning_rate))
        responder = ["env", f"MESH_REGION={responder_region}"] + self.responder_prefix + [
            "--peer-node", str(peer_node),
            "--off-policy-players", str(cfg["off_policy_players"]),
            "--learning-rate", str(learning_rate), "--save-every", str(self.args.save_every),
            "--gradient-clip", str(self.args.gradient_clip),
            "--baseline-hidden", str(self.args.baseline_hidden),
            "--scale-rank", str(self.args.scale_rank),
            "--scale-hidden", str(self.args.scale_hidden),
            "--scale-experts", str(self.args.scale_experts),
            "--scale-topk", str(self.args.scale_topk),
            "--telemetry", telemetry,
            "--append-telemetry",
            "--seed", str(cfg["seed"]),
            "--environment", str(cfg.get("environment", cfg["id"])),
            "--navigation-realization", entity["measurements_path"],
            "--match-metadata", json.dumps({"match_id": cfg["id"], "configuration": {key: cfg.get(key) for key in ("map", "teams", "carts", "players_per_team", "checkpoints_per_lane", "score_limit", "checkpoint_score_rate", "perturbation")}, "seed": cfg["seed"], "team_policy_arms": cfg["team_policy_arms"]}),
            "--replay-batch", str(self.args.replay_batch),
            "--replay-weight", str(self.args.replay_weight),
        ]
        training_arms = cfg.get("train_arms", [])
        if training_arms:
            responder += ["--train-arms", ",".join(training_arms)]
        if cfg["team_policy_arms"]:
            responder += ["--team-policy-arms", ",".join(cfg["team_policy_arms"])]
            checkpoint_in = {}
            requested = cfg.get("arm_checkpoints", {})
            intervention_arms = MATRIX_FUSION_INTERVENTION_ARMS
            realizes_intervention = any(
                arm in intervention_arms and arm != "matrix_fusion" for arm in cfg["team_policy_arms"]
            )
            canonical_matrix_source = (
                requested.get("matrix_fusion")
                or self.previous_checkpoints.get("matrix_fusion")
                or next((
                    requested.get(arm) or self.previous_checkpoints.get(arm)
                    for arm in intervention_arms
                    if requested.get(arm) or self.previous_checkpoints.get(arm)
                ), None)
            )
            checkpoint_arms = list(dict.fromkeys(cfg["team_policy_arms"]))
            if realizes_intervention:
                checkpoint_arms.extend(
                    arm for arm in intervention_arms if arm not in checkpoint_arms
                )
            for arm in checkpoint_arms:
                source = requested.get(arm)
                if not source and arm == "initial_policy":
                    source = self.initial_checkpoints.get("matrix_fusion")
                if arm in intervention_arms and canonical_matrix_source:
                    source = canonical_matrix_source
                elif not source:
                    source = self.previous_checkpoints.get(arm)
                if source:
                    source = os.path.abspath(os.path.expanduser(source))
                    checkpoint_in[arm] = source
                    responder += ["--arm-checkpoint", f"{arm}={source}"]
            if cfg["split"] != "heldout":
                responder += ["--train", "--policy-arm", cfg["policy_arm"],
                              "--online-checkpoint", checkpoint_out,
                              "--initial-checkpoint", checkpoint_initial]
                active_checkpoint = checkpoint_in.get(cfg["policy_arm"])
                if active_checkpoint:
                    responder += ["--checkpoint", active_checkpoint,
                                  "--resume-checkpoint", active_checkpoint]
        else:
            checkpoint_in = cfg.get("checkpoint", self.previous_checkpoints.get(cfg["policy_arm"]))
            if cfg["split"] != "heldout":
                responder += ["--train", "--policy-arm", cfg["policy_arm"],
                              "--online-checkpoint", checkpoint_out,
                              "--initial-checkpoint", checkpoint_initial]
            else:
                responder += ["--policy-arm", cfg["policy_arm"]]
            if checkpoint_in:
                checkpoint_in = os.path.abspath(os.path.expanduser(checkpoint_in))
                responder += ["--checkpoint", checkpoint_in]
                if cfg["split"] != "heldout":
                    responder += ["--resume-checkpoint", checkpoint_in]
        if distributed_scale and remote_scale_arm_mass(cfg):
            responder += ["--distributed-scale", "--distributed-scale-operation", self.args.distributed_scale_operation]
        expert_prefix = ["env", f"MESH_REGION={server_region}"] + self.expert_prefix
        expert_pid = os.path.join(self.run_dir, "expert.pid")
        if self.server_host:
            native = command(self.args.expert_command) or application_command(self.remote_mesh_root, self.args.remote_python, 'solver.strat.matrix_worker')
            expert_prefix = ['env', f'MESH_REGION={server_region}'] + native
            expert_pid = os.path.join(remote_directory, 'expert.pid')
        expert = expert_prefix + [
            "--socket", self.args.expert_socket,
            "--environment", str(cfg.get("environment", cfg["id"])),
        ]
        if expert_pid:
            transition = f"p={shlex.quote(expert_pid)}; if [ -f \"$p\" ]; then n=$(sed -n '1p' \"$p\"); case $(ps -p \"$n\" -o command= 2>/dev/null) in *solver.strat.expert_worker*|*solver.strat.matrix_worker*) kill -TERM \"$n\"; i=0; while kill -0 \"$n\" 2>/dev/null && [ \"$i\" -lt 30 ]; do sleep 1; i=$((i + 1)); done; if kill -0 \"$n\" 2>/dev/null; then exit 1; fi;; esac; fi"
            wrapped = f"echo $$ > {shlex.quote(expert_pid)}; exec {shlex.join(expert)}"
            if self.server_host:
                expert = self.ssh_prefix + [self.server_host, "--", "sh", "-c", shlex.quote(wrapped)]
                expert_stop = self.ssh_prefix + [self.server_host, "--", "sh", "-c", shlex.quote(transition)]
            else:
                expert = ["sh", "-c", wrapped]
                expert_stop = ["sh", "-c", transition]
        else:
            expert_stop = []
        responder += command(cfg.get("responder_args"))
        context = {"port": port, "map": cfg["map"], "seed": cfg["seed"], "match": cfg["id"], "directory": directory}
        clients = [client_command(item, context, index) for index, item in enumerate(cfg.get("client_commands", []))]
        return {
            "server": server,
            "transition": map_transition,
            "stage": stage,
            "responder": responder,
            "expert": expert,
            "expert_stop": expert_stop,
            "clients": clients,
            "telemetry": telemetry,
            "checkpoint_in": checkpoint_in,
            "checkpoint_out": checkpoint_out,
            "checkpoint_initial": checkpoint_initial,
            "policy_checkpoints": {arm: checkpoint_path(checkpoint_out, arm) for arm in training_arms},
            "port": port,
            "maxplayers": maxplayers,
            "initial_bots": initial_bots,
            "remote_scale_arm_mass": remote_scale_arm_mass(cfg),
            "distributed_scale": distributed_scale,
            "perturbation": {"name": perturbation, "cvars": cvars},
        }

    def launch(self, name, cmd, log_path, cwd):
        handle = open(log_path, "a")
        try:
            proc = subprocess.Popen(cmd, cwd=cwd, stdin=subprocess.PIPE, stdout=handle, stderr=subprocess.STDOUT, text=True)
            return {"name": name, "command": cmd, "process": proc, "log": log_path, "handle": handle, "launched": True, "started": utcnow()}
        except Exception as exc:
            handle.write(f"{type(exc).__name__}: {exc}\n")
            handle.close()
            return {"name": name, "command": cmd, "process": None, "log": log_path, "launched": False, "error": f"{type(exc).__name__}: {exc}"}

    def stop(self, launched):
        proc = launched.get("process")
        if proc is not None and proc.poll() is None and proc.stdin:
            try:
                proc.stdin.write("quit\n")
                proc.stdin.flush()
                launched["quit_sent"] = True
            except Exception as exc:
                launched["quit_error"] = f"{type(exc).__name__}: {exc}"

    def reconcile_clients(self, commands):
        now = time.monotonic()
        for index, cmd in enumerate(commands):
            previous = self.clients[index] if index < len(self.clients) else {}
            proc = previous.get("process")
            if proc is not None and proc.poll() is None:
                old = previous["command"]
                target = cmd[cmd.index("+connect") + 1] if "+connect" in cmd else None
                old_target = old[old.index("+connect") + 1] if "+connect" in old else None
                if target and target != old_target:
                    try:
                        proc.stdin.write("connect " + json.dumps(target) + "\n")
                        proc.stdin.flush()
                        previous["command"] = cmd
                        self.event("client_connect", client=index, pid=proc.pid, target=target)
                    except (OSError, ValueError) as exc:
                        self.event("client_connect_pending", client=index, error=str(exc))
                continue
            if now < previous.get("next_start", 0):
                continue
            if previous:
                self.finish(previous, now)
            current = self.launch(f"client-{index}", cmd, os.path.join(self.run_dir, f"client-{index}.log"), ROOT)
            current["next_start"] = now + 3
            if index < len(self.clients):
                self.clients[index] = current
            else:
                self.clients.append(current)
            self.event("client_start", client=index, launched=current["launched"],
                       pid=current["process"].pid if current["process"] is not None else None,
                       reason="process replacement" if previous else "session start")

    def terminate(self, launched):
        proc = launched.get("process")
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                launched["term_sent"] = True
            except Exception as exc:
                launched["term_error"] = f"{type(exc).__name__}: {exc}"

    def finish(self, launched, deadline):
        proc = launched.get("process")
        if proc is not None:
            timeout = max(0, deadline - time.monotonic())
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                launched["still_running"] = True
            launched["returncode"] = proc.poll()
            if proc.stdin:
                try:
                    proc.stdin.close()
                except BrokenPipeError as error:
                    launched["stdin_close_error"] = f"{type(error).__name__}: {error}"
        handle = launched.get("handle")
        if handle:
            handle.close()
        launched["ended"] = utcnow()
        return {key: value for key, value in launched.items() if key not in ("process", "handle")}

    def execute(self, cfg, commands, directory):
        if self.args.dry_run:
            return {
                "stage": [{"command": cmd, "launched": False} for cmd in commands["stage"]],
                "server": {"command": commands["server"], "launched": False},
                "responder": {"command": commands["responder"], "launched": False},
                "expert": {"command": commands["expert"], "launched": False},
                "expert_stop": {"command": commands["expert_stop"], "launched": False},
                "clients": [{"command": cmd, "launched": False} for cmd in commands["clients"]],
                "dry_run": True,
            }
        stage = []
        for cmd in commands["stage"]:
            started = utcnow()
            try:
                result = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                stage.append({"command": cmd, "returncode": result.returncode, "output": result.stdout,
                              "started": started, "ended": utcnow()})
            except Exception as exc:
                stage.append({"command": cmd, "returncode": None, "error": f"{type(exc).__name__}: {exc}",
                              "started": started, "ended": utcnow()})
        if self.expert is None or self.expert.get('process') is None or self.expert['process'].poll() is not None:
            self.expert = self.launch('expert', commands['expert'], os.path.join(self.run_dir, 'expert.log'), self.args.expert_cwd or ROOT)
            self.expert_stop = commands['expert_stop']
        expert = self.expert
        server_log = os.path.join(self.run_dir, "server.log")
        server_log_start = os.path.getsize(server_log) if os.path.isfile(server_log) else 0
        outcome_journal = Journal()
        if self.server is None or self.server["process"] is None or self.server["process"].poll() is not None:
            self.server = self.launch("server", commands["server"], server_log, self.args.server_cwd or ROOT)
        else:
            self.server["process"].stdin.write(commands["transition"])
            self.server["process"].stdin.flush()
        server = self.server
        outcome_journal.seek(Path(server_log), server_log_start)
        if self.args.startup_secs > 0:
            time.sleep(self.args.startup_secs)
        responder = self.launch("responder", commands["responder"], os.path.join(directory, "responder.log"), self.args.responder_cwd or ROOT)
        self.event("learner_start", match=cfg["id"], ordinal=cfg["ordinal"],
                   reason="match start", launched=responder.get("launched"),
                   log=responder.get("log"), checkpoint_in=commands["checkpoint_in"],
                   checkpoint_out=commands["checkpoint_out"])
        self.reconcile_clients(commands["clients"])
        clients = self.clients
        profile = None
        if cfg.get("operating_profile"):
            from solver.strat.roofline import LiveOperatingProfile
            profile = LiveOperatingProfile(
                {**cfg["operating_profile"], "fixed_population": True}, cfg["teams"], cfg["carts"],
                commands["maxplayers"],
                commands["initial_bots"],
                cfg.get("environment", cfg["id"]),
                os.path.join(directory, "roofline.json"),
            )

        restarts = []
        retired = []
        expert_restarts = []
        expert_retired = []
        outcome_path = os.path.join(directory, "outcome.json")
        server_outcome_at = None
        while not self.stopping:
            time.sleep(0.5)
            for event in outcome_journal.read(Path(server_log), "MESH_OUTCOME "):
                record = {"directory": os.path.basename(directory), "event": event,
                          "team_policy_arms": cfg.get("team_policy_arms", []), "observed_at": utcnow(),
                          "source": "engine_console_independent_of_optimizer"}
                with open(os.path.join(self.run_dir, "server-outcomes.jsonl"), "a") as handle:
                    handle.write(json.dumps(record) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                self.event("server_outcome", directory=record["directory"], server_event=event)
                server_outcome_at = server_outcome_at or time.monotonic()
            if os.path.isfile(outcome_path):
                break
            if server_outcome_at and time.monotonic() - server_outcome_at > self.args.quit_grace:
                self.event("learner_outcome_missing", match=cfg["id"], grace_s=self.args.quit_grace,
                           reason="engine outcome retained; learner did not acknowledge it")
                break
            if profile is not None:
                profile.poll(server)
            if self.stopping:
                break
            server_proc = server.get("process")
            if server_proc is None or server_proc.poll() is not None:
                self.event("server_exit", match=cfg["id"], returncode=None if server_proc is None else server_proc.poll())
                break
            self.reconcile_clients(commands["clients"])
            expert_proc = expert.get("process")
            if commands["expert"] and (expert_proc is None or expert_proc.poll() is not None):
                reason = f"expert exited with returncode {None if expert_proc is None else expert_proc.poll()}"
                self.event("expert_restart", match=cfg["id"], ordinal=cfg["ordinal"],
                           reason=reason, log=expert.get("log"))
                expert_retired.append(self.finish(expert, time.monotonic()))
                index = len(expert_restarts)
                expert = self.launch(
                    "expert", commands["expert"],
                    os.path.join(directory, f"expert.restart{index}.log"),
                    self.args.expert_cwd or ROOT,
                )
                self.expert = expert
                expert_restarts.append({"reason": reason, "at": utcnow(),
                                        "launched": expert.get("launched"),
                                        "log": expert.get("log")})
            proc = responder.get("process")
            if proc is None or proc.poll() is None:
                continue
            reason = f"responder exited with returncode {proc.poll()}"
            self.event("learner_restart", match=cfg["id"], ordinal=cfg["ordinal"],
                       reason=reason, log=responder.get("log"))
            retired.append(self.finish(responder, time.monotonic()))
            index = len(restarts)
            responder = self.launch(
                "responder", commands["responder"],
                os.path.join(directory, f"responder.restart{index}.log"),
                self.args.responder_cwd or ROOT,
            )
            restarts.append({"reason": reason, "at": utcnow(),
                             "launched": responder.get("launched"),
                             "log": responder.get("log")})
        self.event("learner_stop", match=cfg["id"], ordinal=cfg["ordinal"],
                   reason=("supervisor stopping" if self.stopping else "learner acknowledged score outcome" if os.path.isfile(outcome_path)
                           else "engine outcome without learner acknowledgement" if server_outcome_at else "server exited"),
                   signal="SIGTERM",
                   checkpoint_out=commands["checkpoint_out"])
        self.terminate(responder)
        responder_result = self.finish(
            responder, time.monotonic() + self.args.quit_grace,
        )
        with open(server_log, "rb") as source, open(os.path.join(directory, "server.log"), "wb") as target:
            source.seek(server_log_start)
            shutil.copyfileobj(source, target)
        def session_record(item):
            proc = item.get("process")
            return {**{key: value for key, value in item.items() if key not in ("process", "handle")},
                    "pid": None if proc is None else proc.pid,
                    "returncode": None if proc is None else proc.poll(), "persistent": True}
        results = {
            "stage": stage,
            "server": session_record(server),
            "responder": responder_result,
            "expert": session_record(expert),
            "responder_restarts": restarts,
            "responder_retired": retired,
            "expert_restarts": expert_restarts,
            "expert_retired": expert_retired,
            "expert_stop": None,
            "clients": [session_record(client) for client in clients],
        }
        if profile is not None:
            results["operating_profile"] = profile.finish()
        return results

    def run_match(self, cfg):
        cfg["strategy_widths"] = {
            "residual_rank": self.args.scale_rank,
            "hidden_width": self.args.scale_hidden,
            "experts": self.args.scale_experts,
            "topk": self.args.scale_topk,
        }
        directory = self.match_dir(cfg)
        os.makedirs(directory, exist_ok=True)
        started = utcnow()
        entity = self.map_assets.prepare(cfg, directory)
        commands = self.commands(cfg, directory, entity)
        active = os.path.join(self.run_dir, "active-match.json")
        with open(active + ".tmp", "w") as handle:
            json.dump({"ordinal": cfg["ordinal"], "configuration": cfg,
                       "artifacts": {"checkpoint_out": {"path": commands["checkpoint_out"]},
                                     "policy_checkpoints": {arm: {"path": path} for arm, path in commands["policy_checkpoints"].items()}}}, handle)
        os.replace(active + ".tmp", active)
        execution = self.execute(cfg, commands, directory)
        self.observe_capacity(cfg, execution)
        realized = telemetry_summary(commands["telemetry"])
        server_log = runtime_log_measure(os.path.join(directory, "server.log"))
        responder_logs = {
            path: runtime_log_measure(path)
            for path in sorted(glob.glob(os.path.join(directory, "responder*.log")))
        }
        expert_logs = {
            path: runtime_log_measure(path)
            for path in sorted(glob.glob(os.path.join(directory, "expert*.log")))
        }
        runtime_logs = {
            "server": server_log,
            "responders": responder_logs,
            "experts": expert_logs,
        }
        checkpoint_observations = {}
        if not self.args.dry_run and cfg["team_policy_arms"] and cfg["split"] == "heldout":
            checkpoint_in = commands["checkpoint_in"] if isinstance(commands["checkpoint_in"], dict) else {}
            provenance = realized.get("policy_provenance") or {}
            for arm in sorted(set(cfg["team_policy_arms"]) - {"default"}):
                source = provenance.get(arm) or {}
                checkpoint_observations[arm] = {
                    "path": checkpoint_in.get(arm),
                    "path_exists": bool(checkpoint_in.get(arm) and os.path.isfile(checkpoint_in[arm])),
                    "source_weight_mass": int(source.get("source_weight_mass") or 0),
                    "live_weight_mass": int(source.get("live_weight_mass") or 0),
                    "loaded_weight_mass": int(source.get("loaded_weight_mass") or 0),
                    "composable_weight_mass": int(source.get("composable_weight_mass") or 0),
                    "source_only_weight_mass": int(source.get("source_only_weight_mass") or 0),
                    "live_only_weight_mass": int(source.get("live_only_weight_mass") or 0),
                    "shape_difference_mass": int(source.get("shape_difference_mass") or 0),
                    "nonfinite_weight_mass": int(source.get("nonfinite_weight_mass") or 0),
                    "load_exception": source.get("load_exception"),
                    "updates": source.get("updates"),
                    "source_arm": source.get("source_arm"),
                    "live_arm": source.get("live_arm"),
                    "source_version": source.get("source_version"),
                    "live_version": source.get("live_version"),
                    "source_architecture": source.get("source_architecture"),
                    "live_architecture": source.get("live_architecture"),
                    "source_reward_contract": source.get("source_reward_contract"),
                    "live_reward_contract": source.get("live_reward_contract"),
                    "checkpoint_sha256": source.get("checkpoint_sha256"),
                    "lineage_initial_sha256": source.get("lineage_initial_sha256"),
                }
        actual = realized.get("peak_configuration") or realized.get("last_configuration")
        configured = {"teams": cfg["teams"], "carts": cfg["carts"]}
        if not cfg.get("operating_profile"):
            configured.update({
                "players": sum(cfg["controllers"].values()),
                "players_per_team": {str(i + 1): value for i, value in enumerate(cfg["players_per_team"])},
                "controllers": {key: value for key, value in cfg["controllers"].items() if value},
            })
        realization_measures = {
            "dry_run": bool(self.args.dry_run),
            "entity_returncode": entity.get("returncode"),
            "entity_realization_id": entity.get("realization_id"),
            "entity_realization_reuse_mass": entity.get("realization_reuse_mass"),
            "build_returncode": self.build.get("returncode"),
            "telemetry_lines": realized["lines"],
            "expert_command_fields": len(commands["expert"]),
            "expert_launches": int(bool(execution.get("expert", {}).get("launched"))),
            "expert_stop_returncode": (execution.get("expert_stop") or {}).get("returncode"),
            "runtime_log_line_mass": server_log["lines"]
                + sum(row["lines"] for row in responder_logs.values())
                + sum(row["lines"] for row in expert_logs.values()),
            "configured": configured,
            "realized": actual,
            "checkpoint_observations": checkpoint_observations,
        }
        record = {
            "id": cfg["id"], "ordinal": cfg["ordinal"], "split": cfg["split"],
            "started": started, "ended": utcnow(),
            "configuration": cfg, "build": self.build, "entity": entity,
            "runtime": self.runtime,
            "commands": {key: commands[key] for key in ("stage", "server", "transition", "responder", "expert", "expert_stop", "clients")},
            "execution": execution,
            "realized": realized, "runtime_logs": runtime_logs,
            "realization_measures": realization_measures,
            "artifacts": {
                "entity": artifact(entity["entity"], self.artifact_cache),
                "measurements": artifact(entity["measurements_path"], self.artifact_cache),
                "mapinfo": artifact(entity["mapinfo"], self.artifact_cache),
                "telemetry": artifact(commands["telemetry"], self.artifact_cache),
                "checkpoint_in": ({arm: artifact(path, self.artifact_cache) for arm, path in commands["checkpoint_in"].items()}
                                  if isinstance(commands["checkpoint_in"], dict)
                                  else artifact(commands["checkpoint_in"], self.artifact_cache) if commands["checkpoint_in"] else None),
                "checkpoint_out": artifact(commands["checkpoint_out"], self.artifact_cache),
                "checkpoint_initial": artifact(commands["checkpoint_initial"], self.artifact_cache),
                "policy_checkpoints": {arm: artifact(path, self.artifact_cache) for arm, path in commands["policy_checkpoints"].items()},
                "server_log": artifact(os.path.join(directory, "server.log"), self.artifact_cache),
                "responder_log": artifact(os.path.join(directory, "responder.log"), self.artifact_cache),
                "expert_log": artifact(os.path.join(directory, "expert.log"), self.artifact_cache),
                "roofline": artifact(os.path.join(directory, "roofline.json"), self.artifact_cache),
            },
            "port": commands["port"], "perturbation": commands["perturbation"],
        }
        record_path = os.path.join(directory, "match.json")
        with open(record_path, "w") as handle:
            handle.write(json.dumps(record, indent=2, sort_keys=True) + "\n")
        with open(self.index_path, "a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        if cfg["split"] != "heldout" and (self.args.dry_run or os.path.exists(commands["checkpoint_out"])):
            self.previous_checkpoints[cfg["policy_arm"]] = commands["checkpoint_out"]
            self.initial_checkpoints.setdefault(cfg["policy_arm"], commands["checkpoint_initial"])
        for arm, path in commands["policy_checkpoints"].items():
            if os.path.exists(path):
                self.previous_checkpoints[arm] = path
                self.initial_checkpoints.setdefault(arm, checkpoint_path(commands["checkpoint_initial"], arm))
        if not self.args.dry_run:
            self.generated_checkpoints.update([
                commands["checkpoint_out"], commands["checkpoint_initial"],
                *commands["policy_checkpoints"].values(),
                *(checkpoint_path(commands["checkpoint_initial"], arm) for arm in commands["policy_checkpoints"]),
            ])
            retained = set(self.previous_checkpoints.values()) | set(self.initial_checkpoints.values())
            self.generated_bundles.add(commands["checkpoint_out"] + ".runstate.npz")
            retained_bundles = {checkpoint_source(path) for path in retained if os.path.isfile(path)}
            for bundle in self.generated_bundles - retained_bundles:
                Path(bundle).unlink(missing_ok=True)
                for path in Path(bundle).parent.glob(Path(bundle).name + ".steps.*"):
                    path.unlink(missing_ok=True)
                actions = Path(bundle.removesuffix(".runstate.npz") + ".actions")
                if actions.is_dir():
                    shutil.rmtree(actions)
            self.generated_bundles.intersection_update(retained_bundles)
            for path in self.generated_checkpoints - retained:
                Path(path).unlink(missing_ok=True)
            self.generated_checkpoints.intersection_update(retained)
            data = Path(entity["userdir"]) / "data"
            for path in [*data.glob("maps/*.bsp"), data / "progs.dat", data / "csprogs.dat"]:
                path.unlink(missing_ok=True)
        print(json.dumps({"id": cfg["id"], "record": record_path}), flush=True)
        return record

    def event(self, kind, **fields):
        row = {"event": kind, "at": utcnow(), **fields}
        try:
            with open(self.event_path, "a") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        except OSError as error:
            row["event_store_error"] = f"{type(error).__name__}: {error}"
        print(json.dumps(row, sort_keys=True), flush=True)

    def serve(self, make_schedule):
        cycle = self.next_cycle
        ordinal = self.next_ordinal
        defaults = {"duration": self.args.duration, "seed": self.args.seed,
                    "policy_arm": csv(self.args.policy_arms)[0],
                    "map": self.maps[0] if self.maps else "runningmanctf"}
        self.event("supervisor_start", run_dir=self.run_dir, maps=self.maps,
                   build_returncode=self.build.get("returncode"))
        while not self.stopping and (self.args.cycles <= 0 or cycle < self.args.cycles):
            schedule = make_schedule(cycle)
            self.event("cycle_start", cycle=cycle, matches=len(schedule),
                       first_ordinal=ordinal)
            for item in schedule:
                try:
                    cfg = normalize(item, ordinal, defaults)
                    cfg["ordinal"] = ordinal
                    cfg["cycle"] = cycle
                    record = self.run_match(cfg)
                    self.event("match_end", cycle=cycle, ordinal=ordinal,
                               id=record["id"],
                               split=record["split"],
                               map=cfg["map"],
                               telemetry_lines=record.get("realized", {}).get("lines"),
                               restarts=len(record.get("execution", {}).get("responder_restarts", [])))
                except Exception as exc:
                    fallback = item if isinstance(item, dict) else {"input": item}
                    record = {"id": str(fallback.get("id", f"match-{ordinal:05d}")),
                              "ordinal": ordinal, "cycle": cycle, "started": utcnow(),
                              "ended": utcnow(),
                              "configuration": fallback,
                              "runtime": self.runtime,
                              "error": f"{type(exc).__name__}: {exc}"}
                    with open(self.index_path, "a") as handle:
                        handle.write(json.dumps(record, sort_keys=True) + "\n")
                    self.event("match_exception", cycle=cycle, ordinal=ordinal,
                               id=record["id"], exception=record["error"])
                ordinal += 1
                if self.stopping:
                    break
            self.event("cycle_end", cycle=cycle, next_ordinal=ordinal)
            studied = any((item.get("team_policy_arms") if isinstance(item, dict) else None)
                          for item in schedule)
            if studied:
                try:
                    from solver.strat.study import summarize, write_report
                    report = summarize(self.run_dir)
                    report_path = write_report(
                        os.path.join(self.run_dir, "study.json"), report,
                    )
                    self.event("study_measurement", cycle=cycle,
                               records=report.get("record_count"),
                               paired_blocks=report.get("paired_block_count"),
                               paired_rounds=report.get("paired_round_count"),
                               artifact=artifact(report_path, self.artifact_cache))
                except Exception as exc:
                    self.event("study_measurement_error", cycle=cycle,
                               error=f"{type(exc).__name__}: {exc}")
            cycle += 1
        for process in self.clients + ([self.server] if self.server else []):
            self.stop(process)
            self.finish(process, time.monotonic() + self.args.quit_grace)
        server_proc = None if self.server is None else self.server.get('process')
        if self.expert is not None and (server_proc is None or server_proc.poll() is not None):
            if self.expert_stop:
                try:
                    result = subprocess.run(self.expert_stop, cwd=ROOT, capture_output=True, text=True, timeout=35)
                    self.event('runtime_stop', returncode=result.returncode, output=result.stdout + result.stderr)
                except (OSError, subprocess.TimeoutExpired) as error:
                    self.event('runtime_stop_pending', error=str(error))
            self.terminate(self.expert)
            self.finish(self.expert, time.monotonic() + self.args.quit_grace)
        elif self.expert is not None:
            self.event('runtime_retained', reason='game server is still running', log=self.expert.get('log'))
        if not self.args.dry_run and (server_proc is None or server_proc.poll() is not None):
            userdir = os.path.join(self.run_dir, "userdir")
            if os.path.isdir(userdir):
                shutil.rmtree(userdir)
            if self.server_host:
                remote_userdir = os.path.join(self.remote_run_root, os.path.basename(self.run_dir), "userdir")
                result = subprocess.run(self.ssh_prefix + [self.server_host, shlex.join(["rm", "-rf", "--", remote_userdir])],
                                        capture_output=True, text=True)
                self.event("artifact_release", path=remote_userdir, returncode=result.returncode,
                           output=result.stdout + result.stderr)
        self.event("supervisor_stop", signal=self.stopping, cycles=cycle, next_ordinal=ordinal)

    def plan(self, schedule):
        records = []
        defaults = {"duration": self.args.duration, "seed": self.args.seed,
                    "policy_arm": csv(self.args.policy_arms)[0],
                    "map": self.maps[0] if self.maps else "runningmanctf"}
        for index, item in enumerate(schedule):
            cfg = normalize(item, index, defaults)
            cfg["ordinal"] = index
            records.append(self.run_match(cfg))
        return records

def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest")
    ap.add_argument("--entity-file")
    ap.add_argument("--generate", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260830)
    ap.add_argument("--maps", default="auto")
    ap.add_argument("--team-counts", default="4,8,16")
    ap.add_argument("--players-per-team", default="8,16,32")
    ap.add_argument("--cart-counts", default="2,4,8")
    ap.add_argument("--checkpoints-per-lane", type=int, default=4)
    ap.add_argument("--score-limit", type=float, default=1200)
    ap.add_argument("--checkpoint-score-rate", type=float, default=1)
    ap.add_argument("--skills", default="2,5,8")
    ap.add_argument("--perturbations", default="baseline,fast,slow,volatile")
    ap.add_argument("--off-policy-counts", default="0,1,2")
    ap.add_argument("--policy-arms", default="matrix_fusion,ffn,linear,default")
    ap.add_argument("--study-repetitions", type=int, default=0)
    ap.add_argument("--joint-training", action="store_true")
    ap.add_argument("--replay-batch", type=int, default=8)
    ap.add_argument("--replay-weight", type=float, default=0.5)
    ap.add_argument("--cycles", type=int, default=0)
    ap.add_argument("--human-counts", default="0")
    ap.add_argument("--human-client-command")
    ap.add_argument("--heldout-fraction", type=float, default=0.2)
    ap.add_argument("--duration", type=float, default=600, help="Legacy manifest metadata; matches end only on an observed score outcome")
    ap.add_argument("--observer", default="http://127.0.0.1:8787/latest.json")
    ap.add_argument("--memory-target-fraction", type=float, default=0.5)
    ap.add_argument("--bandwidth-node")
    ap.add_argument("--bandwidth-role", default="responder")
    ap.add_argument("--run-dir", default=os.path.join(ROOT, "solver", "strat", "runs", "curriculum"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--engine", default=os.path.expanduser("~/dox/mesh/xonotic/darkplaces-work/darkplaces-dedicated"))
    ap.add_argument("--basedir", default=os.path.expanduser("~/dox/xonotic/Xonotic"))
    ap.add_argument("--server-command")
    ap.add_argument("--server-host")
    ap.add_argument("--ssh-command", default=os.environ.get("MESH_SSH_COMMAND", "ssh"))
    ap.add_argument("--remote-engine")
    ap.add_argument("--remote-basedir")
    ap.add_argument("--remote-run-root", default="/tmp/mesh-xonotic-curriculum")
    ap.add_argument("--remote-mesh-root", default="/Users/mdot/mesh")
    ap.add_argument("--responder-command")
    ap.add_argument("--expert-command")
    ap.add_argument("--server-cwd")
    ap.add_argument("--responder-cwd")
    ap.add_argument("--expert-cwd")
    ap.add_argument("--expert-socket", default="/tmp/mesh-expert-worker.sock")
    ap.add_argument("--remote-python", default="/usr/local/mesh/bin/mesh-python")
    ap.add_argument("--server-mesh-region", default=os.environ.get("MESH_SERVER_REGION", os.environ.get("MESH_REGION", "/mesh0")))
    ap.add_argument("--responder-mesh-region", default=os.environ.get("MESH_RESPONDER_REGION", os.environ.get("MESH_REGION", "/mesh0")))
    ap.add_argument("--python", default=os.environ.get("MESH_PYTHON_LAUNCHER", os.path.join(os.path.dirname(ROOT), "bin", "mesh-python")))
    ap.add_argument("--entity-tool", default=os.path.join(ROOT, "payload", "tools", "mkentfile.py"))
    ap.add_argument("--build-command", default=os.path.join(ROOT, "payload", "build.sh"))
    ap.add_argument("--progs", default=os.path.join(ROOT, "payload-build", "progs.dat"))
    ap.add_argument("--csprogs", default=os.path.join(ROOT, "payload-build", "csprogs.dat"))
    ap.add_argument("--checkpoint")
    ap.add_argument("--arm-checkpoint", action="append", default=[])
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--gradient-clip", type=float, default=1.0)
    ap.add_argument("--baseline-hidden", type=int, default=256)
    ap.add_argument("--scale-rank", type=int, default=SCALE_RANK)
    ap.add_argument("--scale-hidden", type=int, default=SCALE_HIDDEN)
    ap.add_argument("--scale-experts", type=int, default=SCALE_EXPERTS)
    ap.add_argument("--scale-topk", type=int, default=SCALE_TOPK)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--peer-node", type=int)
    ap.add_argument("--strategy-node", type=int)
    ap.add_argument("--distributed-scale", action="store_true")
    ap.add_argument("--distributed-scale-operation", choices=("block", "gram"), default="gram")
    ap.add_argument("--port-base", type=int, default=26100)
    ap.add_argument("--startup-secs", type=float, default=2)
    ap.add_argument("--quit-grace", type=float, default=10)
    ap.add_argument("--round-grace", type=float, default=8)
    return ap

def main(argv=None):
    args = parser().parse_args(argv)
    supervisor = Curriculum(args)
    signal.signal(signal.SIGINT, supervisor.request_stop)
    signal.signal(signal.SIGTERM, supervisor.request_stop)
    maps = supervisor.maps

    def schedule_defaults(rows):
        profile = {
            "observer": args.observer,
            "memory_target_fraction": args.memory_target_fraction,
            "bandwidth_role": args.bandwidth_role,
        }
        if args.bandwidth_node:
            profile["bandwidth_node"] = args.bandwidth_node
        if args.entity_file:
            source = os.path.abspath(os.path.expanduser(args.entity_file))
            for row in rows:
                row.setdefault("entity_file", source)
        for row in rows:
            row.setdefault("checkpoints_per_lane", args.checkpoints_per_lane)
            row.setdefault("score_limit", args.score_limit)
            row.setdefault("checkpoint_score_rate", args.checkpoint_score_rate)
            required_roles = (
                ["matrix", "responder"]
                if args.distributed_scale and row.get("distributed_scale", True)
                and remote_scale_arm_mass(row)
                else ["responder"]
            )
            row["operating_profile"] = merge(profile, merge({
                "required_roles": required_roles,
                "minimum_producer_nodes": 2 if len(required_roles) > 1 else 1,
            }, row.get("operating_profile") or {}))
        return rows

    def make_schedule(cycle):
        if args.manifest:
            return schedule_defaults(load_manifest(args.manifest))
        teams, players, carts = (
            csv(args.team_counts, int), csv(args.players_per_team, int),
            csv(args.cart_counts, int),
        )
        if args.joint_training:
            rows = generated_schedule(
                args.generate or 1, args.seed + cycle, maps, teams, players, carts,
                csv(args.skills, float), csv(args.perturbations), [0],
                [JOINT_TRAINING_ARMS[0]], 0.0, csv(args.human_counts, int),
                args.human_client_command, include_comparisons=False,
            )
            rng = random.Random(args.seed + cycle)
            for index, row in enumerate(rows):
                count = rng.randrange(1, row["teams"])
                arms = [JOINT_TRAINING_ARMS[0]] * count + [JOINT_TRAINING_ARMS[1]] * (row["teams"] - count)
                rng.shuffle(arms)
                row.update(id=f"joint-{cycle:05d}-{index:05d}", team_policy_arms=arms, train_arms=list(JOINT_TRAINING_ARMS))
            return schedule_defaults(rows)
        if args.study_repetitions > 0:
            policy_arms = csv(args.policy_arms)
            perturbations = csv(args.perturbations)
            training = generated_schedule(
                args.generate or 1, args.seed + cycle, maps,
                teams, players, carts, csv(args.skills, float),
                perturbations, csv(args.off_policy_counts, int),
                [arm for arm in policy_arms if arm in OPTIMIZATION_ARMS],
                0.0, csv(args.human_counts, int), args.human_client_command,
                include_comparisons=False,
            )
            studies = study_schedule(
                args.study_repetitions, args.seed + cycle, maps,
                teams, players, carts, csv(args.skills, float),
                perturbations, policy_arms, {},
                map_offset=cycle * args.study_repetitions * len(perturbations),
            )
            return schedule_defaults(training + studies)

        return schedule_defaults(generated_schedule(
            args.generate or 1, args.seed + cycle, maps,
            teams, players, carts, csv(args.skills, float),
            csv(args.perturbations), csv(args.off_policy_counts, int),
            csv(args.policy_arms),
            args.heldout_fraction, csv(args.human_counts, int),
            args.human_client_command,
        ))

    if args.dry_run:
        return supervisor.plan(make_schedule(0))
    supervisor.serve(make_schedule)

if __name__ == "__main__":
    main()
