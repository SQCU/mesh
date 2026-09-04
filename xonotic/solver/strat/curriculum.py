import argparse, copy, datetime, glob, hashlib, itertools, json, math, os, random, re, shlex, shutil, signal, subprocess, sys, time, zipfile

from solver.strat.scale_config import SCALE_EXPERTS, SCALE_HIDDEN, SCALE_RANK, SCALE_TOPK
from solver.strat.policy_contract import MATRIX_FUSION_INTERVENTION_ARMS, OPTIMIZATION_ARMS, is_matrix_fusion_arm
from solver.strat.capacity import cart_capacity, engine_player_capacity, team_capacity

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
        "g_payload_idle_time": 4,
        "g_payload_rollback_speed": 35,
        "g_payload_push_falloff": 0.75,
    },
}
RESPAWN_PERTURBATIONS = {
    "fast": (0.5, 0.5, 1.0, 1.5),
    "slow": (2.0, 1.0, 2.0, 3.0),
    "volatile": (1.0, 0.5, 2.0, 4.0),
}

MEGAMAP_MARKERS = ("mapfuse", "procedurally fused")

def _mapinfo_is_megamap(text):
    lowered = text.lower()
    return any(marker in lowered for marker in MEGAMAP_MARKERS)

def discover_maps(basedir):
    maps, megamaps, joins = {}, set(), set()
    loose = os.path.join(basedir, "data", "maps")
    sources = []
    if os.path.isdir(loose):
        for name in sorted(os.listdir(loose)):
            sources.append(("file", os.path.join(loose, name), name))
    for archive in sorted(glob.glob(os.path.join(basedir, "data", "*.pk3")), reverse=True):
        try:
            with zipfile.ZipFile(archive) as bundle:
                for member in bundle.namelist():
                    if member.startswith("maps/") and member.count("/") == 1:
                        sources.append(("zip", (archive, member), os.path.basename(member)))
        except (zipfile.BadZipFile, OSError):
            continue
    texts = {}
    for kind, where, name in sources:
        stem, ext = os.path.splitext(name)
        if ext == ".bsp":
            maps.setdefault(stem, (kind, where))
        elif ext == ".json" and stem.endswith(".joins"):
            joins.add(stem[: -len(".joins")])
        elif ext == ".mapinfo" and stem not in texts:
            try:
                if kind == "file":
                    with open(where, errors="replace") as handle:
                        texts[stem] = handle.read()
                else:
                    with zipfile.ZipFile(where[0]) as bundle:
                        texts[stem] = bundle.read(where[1]).decode("utf-8", "replace")
            except (zipfile.BadZipFile, OSError, KeyError):
                continue
    for stem in maps:
        if stem in joins or _mapinfo_is_megamap(texts.get(stem, "")):
            megamaps.add(stem)
    names = set(maps)
    megamaps &= names
    return {
        "maps": sorted(names),
        "megamaps": sorted(megamaps),
        "stock": sorted(names - megamaps),
        "non_game": [],
    }

def resolve_maps(spec, basedir):
    requested = csv(spec)
    if not any(token in ("auto", "megamaps") for token in requested):
        return requested
    found = discover_maps(basedir)
    out = []
    for token in requested:
        if token == "megamaps":
            out.extend(found["megamaps"])
        elif token == "auto":
            out.extend(found["megamaps"])
            out.extend(name for name in found["maps"] if name not in found["megamaps"])
        elif token not in out:
            out.append(token)
    deduped = []
    for name in out:
        if name not in deduped:
            deduped.append(name)
    return deduped

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
                        "distributed_scale": set(pair) != {"matrix_fusion", "initial_policy"},
                        "pair": index,
                        "study_repetition": repetition + 1,
                        "leg": leg + 1,
                    })
                    ordinal += 1
    return out

def allocate_population(schedule, total_players):
    for row in schedule:
        team_count = int(row["teams"])
        total = max(team_count, int(total_players))
        base, remainder = divmod(total, team_count)
        row["players_per_team"] = [base + int(team < remainder) for team in range(team_count)]
        row["controllers"] = {"bot": total}
    return schedule

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

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def artifact(path, cache=None):
    if not os.path.exists(path):
        return {"path": path, "exists": False}
    stat = os.stat(path)
    key = (os.path.realpath(path), stat.st_size, stat.st_mtime_ns)
    if cache is not None and key in cache:
        return dict(cache[key])
    row = {"path": path, "exists": True, "bytes": stat.st_size, "sha256": sha256(path)}
    if cache is not None:
        cache[key] = dict(row)
    return row

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
    configurations, controllers, arms, first, last, last_config, lines = {}, {}, {}, None, None, None, 0
    if not os.path.exists(path):
        return {"lines": 0, "configurations": [], "controllers": {}}
    with open(path) as handle:
        for raw in handle:
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
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
        "first_tick": first.get("req_tick") if first else None,
        "last_tick": last.get("req_tick") if last else None,
        "responses": last.get("resp_id") if last else 0,
        "last_configuration": last_config,
        "peak_configuration": peak_config,
        "configurations": configuration_rows,
        "controllers": controllers,
        "policy_arms": arms,
        "policy_provenance": (last or {}).get("policy_provenance", {}),
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

def entity_class_measure(path):
    masses = {}
    error = None
    try:
        with open(path, errors="replace") as stream:
            text = stream.read()
        for match in re.finditer(r'"classname"\s+"([^"]+)"', text):
            name = match.group(1)
            masses[name] = masses.get(name, 0) + 1
    except OSError as exc:
        error = f"{type(exc).__name__}: {exc}"
    return masses, error

class Curriculum:
    def __init__(self, args):
        self.args = args
        self.run_dir = os.path.abspath(os.path.expanduser(args.run_dir))
        self.server_prefix = command(args.server_command) or [os.path.abspath(os.path.expanduser(args.engine))]
        self.responder_prefix = command(args.responder_command) or [args.python, "-m", "solver.strat.strat_responder"]
        self.expert_prefix = command(args.expert_command) or [args.python, "-m", "solver.strat.expert_worker"]
        self.ssh_prefix = command(args.ssh_command) or ["ssh"]
        self.basedir = os.path.abspath(os.path.expanduser(args.basedir))
        self.entity_tool = os.path.abspath(os.path.expanduser(args.entity_tool))
        self.server_host = args.server_host
        self.remote_run_root = os.path.expanduser(args.remote_run_root)
        self.remote_engine = os.path.expanduser(args.remote_engine) if args.remote_engine else os.path.join(self.remote_run_root, "runtime", "darkplaces-dedicated")
        self.remote_basedir = os.path.expanduser(args.remote_basedir) if args.remote_basedir else os.path.join(self.remote_run_root, "runtime", "Xonotic")
        self.progs = os.path.abspath(os.path.expanduser(args.progs))
        self.csprogs = os.path.abspath(os.path.expanduser(args.csprogs))
        self.build_command = command(args.build_command)
        self.runtime = runtime_identity(args.python)
        self.previous_checkpoints = {}
        self.previous_scale_checkpoints = {}
        self.initial_checkpoints = {}
        self.capacity_observations = []
        self.artifact_cache = {}
        program_files = sorted(glob.glob(os.path.join(os.path.dirname(self.entity_tool), "*.py")))
        program_files.append(os.path.join(ROOT, "payload", "cfg", "gamemodes-payload.cfg"))
        self.entity_program_id = hashlib.sha256(json.dumps({
            os.path.relpath(path, ROOT): artifact(path, self.artifact_cache)
            for path in program_files
        }, sort_keys=True).encode()).hexdigest()
        self.entity_realizations = {}
        self.next_ordinal = 0
        self.next_cycle = 0
        self.stopping = 0
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
        self.next_ordinal = max((int(row.get("ordinal", -1)) for row in rows), default=-1) + 1
        self.next_cycle = max((int(row.get("cycle", -1)) for row in rows), default=-1) + 1
        for record in reversed(rows):
            cfg = record.get("configuration") or {}
            checkpoint = ((record.get("artifacts") or {}).get("checkpoint_out") or {}).get("path")
            if cfg.get("split") != "heldout" and checkpoint and os.path.isfile(checkpoint):
                self.previous_checkpoints.setdefault(str(cfg.get("policy_arm", "matrix_fusion")), checkpoint)
            scale_checkpoint = ((record.get("artifacts") or {}).get("scale_checkpoint_out") or {}).get("path")
            if cfg.get("split") != "heldout" and scale_checkpoint and os.path.isfile(scale_checkpoint):
                self.previous_scale_checkpoints.setdefault(str(cfg.get("policy_arm", "matrix_fusion")), scale_checkpoint)
            profile = (record.get("execution") or {}).get("operating_profile") or {}
            point = profile.get("target_center_observation")
            if point:
                self.capacity_observations.append({
                    "teams": point.get("teams", cfg.get("teams")),
                    "carts": point.get("carts", cfg.get("carts")),
                    "players": point.get("players"), "point": point,
                    "environment": profile.get("environment"),
                })
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

    def center_capacity_observation(self):
        return min(
            self.capacity_observations,
            key=lambda row: float(row["point"].get("target_squared_distance"))
            if row["point"].get("target_squared_distance") is not None else math.inf,
        )

    def adaptive_axes(self, teams, players, carts):
        if not self.capacity_observations:
            return teams, players, carts
        center = self.center_capacity_observation()
        center_teams = int(center["teams"])
        center_carts = int(center["carts"])
        center_players = max(1, math.ceil(int(center.get("players") or center_teams) / center_teams))
        teams_limit = team_capacity(max([center_teams, *teams]))
        carts_limit = cart_capacity(max([center_carts, *carts]))
        team_axis = list(dict.fromkeys((
            center_teams, max(2, center_teams // 2), min(teams_limit, center_teams * 2),
        )))
        cart_axis = list(dict.fromkeys((
            center_carts, max(1, center_carts // 2), min(carts_limit, center_carts * 2),
        )))
        return team_axis, [center_players], cart_axis

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

    def archives(self):
        return sorted(glob.glob(os.path.join(self.basedir, "data", "*.pk3")), reverse=True)

    def asset_identity(self, found):
        if found[0] == "file":
            return {"kind": "file", "source": artifact(found[1], self.artifact_cache)}
        stat = os.stat(found[1])
        source = {"path": os.path.realpath(found[1]), "bytes": stat.st_size,
                  "mtime_ns": stat.st_mtime_ns}
        with zipfile.ZipFile(found[1]) as bundle:
            member = bundle.getinfo(found[2])
        return {"kind": "zip", "source": source, "member": found[2],
                "member_bytes": member.file_size, "member_crc32": member.CRC}

    def locate_asset(self, mapname, suffix):
        loose = os.path.join(self.basedir, "data", "maps", mapname + suffix)
        if os.path.exists(loose):
            return ("file", loose)
        member = "maps/" + mapname + suffix
        for archive in self.archives():
            try:
                with zipfile.ZipFile(archive) as bundle:
                    if member in bundle.namelist():
                        return ("zip", archive, member)
            except (zipfile.BadZipFile, OSError):
                continue
        return None

    def extract_asset(self, found, destination):
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        if found[0] == "file":
            shutil.copyfile(found[1], destination)
        else:
            with zipfile.ZipFile(found[1]) as bundle, open(destination, "wb") as target:
                target.write(bundle.read(found[2]))

    def prepare_entity(self, cfg, directory):
        userdir = os.path.join(directory, "userdir")
        maps_dir = os.path.join(userdir, "data", "maps")
        os.makedirs(maps_dir, exist_ok=True)
        data_dir = os.path.join(userdir, "data")
        ent = os.path.join(maps_dir, cfg["map"] + ".ent")
        measurements_path = ent + ".measurements.json"
        mapinfo = os.path.join(maps_dir, cfg["map"] + ".mapinfo")
        record = {"userdir": userdir, "entity": ent, "measurements_path": measurements_path, "mapinfo": mapinfo}
        if self.args.dry_run:
            record["dry_run"] = True
            record["command"] = [self.args.python, self.entity_tool, "<resolved-bsp>", ent, str(cfg["teams"]), str(cfg["carts"]), "<resolved-archive>"]
            record["gamecode"] = {"progs": self.progs, "csprogs": self.csprogs,
                                  "effectinfo": os.path.join(os.path.dirname(self.progs), "effectinfo.txt")}
            return record
        try:
            shutil.copyfile(self.progs, os.path.join(data_dir, "progs.dat"))
            shutil.copyfile(self.csprogs, os.path.join(data_dir, "csprogs.dat"))
            shutil.copyfile(os.path.join(os.path.dirname(self.progs), "effectinfo.txt"),
                            os.path.join(data_dir, "effectinfo.txt"))
            record["gamecode"] = {
                "progs": artifact(os.path.join(data_dir, "progs.dat"), self.artifact_cache),
                "csprogs": artifact(os.path.join(data_dir, "csprogs.dat"), self.artifact_cache),
                "effectinfo": artifact(os.path.join(data_dir, "effectinfo.txt"), self.artifact_cache),
            }
            source_ent = cfg.get("entity_file")
            if source_ent:
                source_ent = os.path.abspath(os.path.expanduser(source_ent))
                shutil.copyfile(source_ent, ent)
                if os.path.exists(source_ent + ".measurements.json"):
                    shutil.copyfile(source_ent + ".measurements.json", measurements_path)
                record["source"] = source_ent
                record["returncode"] = 0
            else:
                source_dir = os.path.join(directory, "source")
                bsp = os.path.join(source_dir, cfg["map"] + ".bsp")
                found = ("file", os.path.abspath(os.path.expanduser(cfg["bsp"]))) if cfg.get("bsp") else self.locate_asset(cfg["map"], ".bsp")
                if found:
                    realization_id = hashlib.sha256(json.dumps({
                        "map": cfg["map"], "teams": cfg["teams"], "carts": cfg["carts"],
                        "program": self.entity_program_id, "source": self.asset_identity(found),
                    }, sort_keys=True).encode()).hexdigest()
                    cached = self.entity_realizations.get(realization_id)
                    reusable = cached and os.path.isfile(cached["entity"]) and os.path.isfile(cached["measurements"])
                    record.update(source=found, realization_id=realization_id,
                                  realization_reuse_mass=int(bool(reusable)))
                    if reusable:
                        shutil.copyfile(cached["entity"], ent)
                        shutil.copyfile(cached["measurements"], measurements_path)
                        record.update(returncode=0, realization_source=cached["entity"])
                    else:
                        self.extract_asset(found, bsp)
                        source_bsp = found[1] if found[0] == "file" else bsp
                        source_archive = found[1] if found[0] == "zip" else ""
                        cmd = [self.args.python, self.entity_tool, source_bsp, ent, str(cfg["teams"]), str(cfg["carts"]), source_archive]
                        record["started"] = utcnow()
                        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                        record.update(command=cmd, output=result.stdout, returncode=result.returncode, ended=utcnow())
                        if result.returncode == 0 and os.path.isfile(ent) and os.path.isfile(measurements_path):
                            self.entity_realizations[realization_id] = {
                                "entity": ent, "measurements": measurements_path,
                            }
                else:
                    record.update(returncode=None, error="map BSP was not found")
            source_mapinfo = cfg.get("mapinfo_file")
            found_mapinfo = ("file", os.path.abspath(os.path.expanduser(source_mapinfo))) if source_mapinfo else self.locate_asset(cfg["map"], ".mapinfo")
            text = ""
            if found_mapinfo:
                temp = os.path.join(directory, "source", cfg["map"] + ".mapinfo")
                self.extract_asset(found_mapinfo, temp)
                with open(temp) as handle:
                    text = handle.read()
            if "gametype plc" not in text:
                text = text.rstrip() + "\ngametype plc\n"
            with open(mapinfo, "w") as handle:
                handle.write(text)
            if os.path.exists(measurements_path):
                with open(measurements_path) as handle:
                    record["measurements"] = json.load(handle)
            else:
                record["measurements"] = None
        except Exception as exc:
            record.update(returncode=None, error=f"{type(exc).__name__}: {exc}")
        record.update(
            entity_exists=int(os.path.exists(ent)),
            measurements_exist=int(os.path.exists(measurements_path)),
            mapinfo_exists=int(os.path.exists(mapinfo)),
        )
        record["entity_class_mass"], record["entity_class_measure_error"] = entity_class_measure(ent)
        return record

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
        port = int(cfg.get("port", self.args.port_base + int(cfg.get("ordinal", 0))))
        players = max(sum(cfg["players_per_team"]), sum(cfg["controllers"].values()))
        maxplayers = int(cfg.get("maxplayers") or engine_player_capacity(players))
        initial_bots = cfg["teams"] if cfg.get("operating_profile") else cfg["controllers"]["bot"]
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
        userdir = entity["userdir"]
        if self.server_host:
            remote_directory = os.path.join(self.remote_run_root, os.path.basename(self.run_dir), os.path.basename(directory))
            userdir = os.path.join(remote_directory, "userdir")
            basedir = self.remote_basedir
            server_prefix = self.ssh_prefix + [
                self.server_host, "--", "env", f"MESH_REGION={server_region}",
                f"MESH_EXPERT_SOCKET={self.args.expert_socket}", self.remote_engine,
            ]
            remote_data = os.path.join(self.remote_basedir, "data")
            engine_source = os.path.abspath(os.path.expanduser(self.args.engine))
            engine_library = os.path.join(os.path.dirname(engine_source), "libjpeg.8.dylib")
            stage = [
                self.ssh_prefix + [self.server_host, "--", "mkdir", "-p", remote_directory, remote_data, os.path.dirname(self.remote_engine)],
                ["rsync", "-a", "-e", shlex.join(self.ssh_prefix), engine_source, f"{self.server_host}:{self.remote_engine}"],
                ["rsync", "-a", "-e", shlex.join(self.ssh_prefix), engine_library, f"{self.server_host}:{os.path.dirname(self.remote_engine)}/"],
                ["rsync", "-aL", "-e", shlex.join(self.ssh_prefix), os.path.join(self.basedir, "data") + "/", f"{self.server_host}:{remote_data}/"],
                ["rsync", "-a", "-e", shlex.join(self.ssh_prefix), entity["userdir"], f"{self.server_host}:{remote_directory}/"],
            ]
        server = server_prefix + [
            "-norunaway", "-xonotic", "-basedir", basedir, "-userdir", userdir,
            "+developer", "0", "+sv_public", "0", "+port", str(port),
            "+sv_random_seed", str(cfg["seed"]),
            "+sv_autopause", "0", "+g_payload", "1",
            "+g_payload_round_timelimit", str(cfg["duration"]),
            "+timelimit", str(max(1, math.ceil((cfg["duration"] + self.args.round_grace) / 60))),
            "+maxplayers", str(maxplayers), "+bot_join_empty", "1",
            "+bot_number", str(initial_bots), "+skill", str(cfg["skill"]),
            "+g_warmup", "0", "+g_maplist", cfg["map"],
            "+g_maplist_shuffle", "0", "+g_maplist_selectrandom", "0",
        ] + cvar_args(cvars) + command(cfg.get("server_args")) + ["+map", cfg["map"]]
        telemetry = os.path.join(directory, "telemetry.jsonl")
        checkpoint_out = os.path.join(directory, "checkpoint.npz")
        scale_checkpoint_out = os.path.join(directory, "checkpoint.scale.npz")
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
        ]
        if cfg["team_policy_arms"]:
            responder += ["--team-policy-arms", ",".join(cfg["team_policy_arms"])]
            checkpoint_in = {}
            requested = cfg.get("arm_checkpoints", {})
            intervention_arms = MATRIX_FUSION_INTERVENTION_ARMS
            realizes_intervention = any(
                arm in intervention_arms for arm in cfg["team_policy_arms"]
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
        expert_collect = []
        if distributed_scale and remote_scale_arm_mass(cfg):
            worker_checkpoint = (
                self.args.scale_worker_checkpoint
                or self.previous_scale_checkpoints.get("matrix_fusion")
                or self.previous_scale_checkpoints.get(cfg["policy_arm"])
            )
            if not worker_checkpoint:
                worker_checkpoint = checkpoint_in.get("matrix_fusion") if isinstance(checkpoint_in, dict) else checkpoint_in
            responder += ["--distributed-scale"]
            expert_checkpoint = worker_checkpoint
            expert_prefix = ["env", f"MESH_REGION={server_region}"] + self.expert_prefix
            expert_pid = os.path.join(self.run_dir, "expert.pid")
            if self.server_host:
                remote_runtime = os.path.join(self.remote_run_root, "runtime")
                remote_xonotic = os.path.join(remote_runtime, "xonotic")
                stage.extend([
                    self.ssh_prefix + [self.server_host, "--", "mkdir", "-p", remote_xonotic, os.path.join(remote_xonotic, "payload"), os.path.join(remote_runtime, "rdma")],
                    ["rsync", "-a", "-e", shlex.join(self.ssh_prefix), "--exclude", "__pycache__", "--exclude", "strat/runs", os.path.join(ROOT, "solver"), f"{self.server_host}:{remote_xonotic}/"],
                    ["rsync", "-a", "-e", shlex.join(self.ssh_prefix), "--exclude", "__pycache__", os.path.join(ROOT, "payload", "tools"), f"{self.server_host}:{os.path.join(remote_xonotic, 'payload')}/"],
                    ["rsync", "-a", "-e", shlex.join(self.ssh_prefix), "--exclude", "__pycache__", os.path.abspath(os.path.join(ROOT, "..", "rdma")) + "/", f"{self.server_host}:{os.path.join(remote_runtime, 'rdma')}/"],
                ])
                if expert_checkpoint:
                    checkpoint_key = hashlib.sha256(os.path.realpath(expert_checkpoint).encode()).hexdigest()[:16]
                    remote_checkpoint = os.path.join(self.remote_run_root, "checkpoints", checkpoint_key + ".npz")
                    stage.extend([
                        self.ssh_prefix + [self.server_host, "--", "mkdir", "-p", os.path.dirname(remote_checkpoint)],
                        ["rsync", "-a", "-e", shlex.join(self.ssh_prefix), expert_checkpoint, f"{self.server_host}:{remote_checkpoint}"],
                    ])
                    expert_checkpoint = remote_checkpoint
                expert_prefix = [
                    "env",
                    f"MESH_REGION={server_region}",
                    f"PYTHONPATH={remote_xonotic}:{os.path.join(remote_xonotic, 'payload', 'tools')}",
                    self.args.remote_python, "-m", "solver.strat.expert_worker",
                ]
                expert_pid = os.path.join(self.remote_run_root, "runtime", "expert.pid")
                remote_scale_checkpoint_out = os.path.join(remote_directory, "checkpoint.scale.npz")
                expert_collect = [
                    "rsync", "-a", "-e", shlex.join(self.ssh_prefix),
                    f"{self.server_host}:{remote_scale_checkpoint_out}", scale_checkpoint_out,
                ] if cfg["split"] != "heldout" else []
            else:
                remote_scale_checkpoint_out = scale_checkpoint_out
            expert = expert_prefix + [
                "--socket", self.args.expert_socket,
                "--scale-rank", str(self.args.scale_rank),
                "--scale-hidden", str(self.args.scale_hidden),
                "--scale-experts", str(self.args.scale_experts),
                "--scale-topk", str(self.args.scale_topk),
                "--seed", str(cfg["seed"]),
                "--environment", str(cfg.get("environment", cfg["id"])),
                "--learning-rate", str(learning_rate),
                "--gradient-clip", str(self.args.gradient_clip),
                "--save-every", str(self.args.save_every),
            ]
            if expert_checkpoint:
                expert += ["--checkpoint", expert_checkpoint]
            if cfg["split"] != "heldout":
                expert += ["--output-checkpoint", remote_scale_checkpoint_out]
            if expert_pid:
                transition = f"p={shlex.quote(expert_pid)}; if [ -f \"$p\" ]; then n=$(sed -n '1p' \"$p\"); case $(ps -p \"$n\" -o command= 2>/dev/null) in *solver.strat.expert_worker*) kill -TERM \"$n\"; i=0; while kill -0 \"$n\" 2>/dev/null && [ \"$i\" -lt 30 ]; do sleep 1; i=$((i + 1)); done; if kill -0 \"$n\" 2>/dev/null; then exit 1; fi;; esac; fi"
                wrapped = f"echo $$ > {shlex.quote(expert_pid)}; exec {shlex.join(expert)}"
                if self.server_host:
                    expert = self.ssh_prefix + [self.server_host, "--", "sh", "-c", shlex.quote(wrapped)]
                    expert_stop = self.ssh_prefix + [self.server_host, "--", "sh", "-c", shlex.quote(transition)]
                else:
                    expert = ["sh", "-c", wrapped]
                    expert_stop = ["sh", "-c", transition]
                stage.insert(0, expert_stop)
            else:
                expert_stop = []
        else:
            expert = []
            expert_stop = []
            expert_collect = []
        responder += command(cfg.get("responder_args"))
        context = {"port": port, "map": cfg["map"], "seed": cfg["seed"], "match": cfg["id"]}
        clients = [client_command(item, context, index) for index, item in enumerate(cfg.get("client_commands", []))]
        return {
            "server": server,
            "stage": stage,
            "responder": responder,
            "expert": expert,
            "expert_stop": expert_stop,
            "expert_collect": expert_collect,
            "clients": clients,
            "telemetry": telemetry,
            "checkpoint_in": checkpoint_in,
            "checkpoint_out": checkpoint_out,
            "scale_checkpoint_out": scale_checkpoint_out,
            "checkpoint_initial": checkpoint_initial,
            "port": port,
            "maxplayers": maxplayers,
            "initial_bots": initial_bots,
            "remote_scale_arm_mass": remote_scale_arm_mass(cfg),
            "distributed_scale": distributed_scale,
            "perturbation": {"name": perturbation, "cvars": cvars},
        }

    def launch(self, name, cmd, log_path, cwd):
        handle = open(log_path, "w")
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
                "expert_collect": {"command": commands["expert_collect"], "launched": False},
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
        expert = self.launch("expert", commands["expert"], os.path.join(directory, "expert.log"), self.args.expert_cwd or ROOT) if commands["expert"] else {"name": "expert", "command": [], "launched": False}
        server = self.launch("server", commands["server"], os.path.join(directory, "server.log"), self.args.server_cwd or ROOT)
        if self.args.startup_secs > 0:
            time.sleep(self.args.startup_secs)
        responder = self.launch("responder", commands["responder"], os.path.join(directory, "responder.log"), self.args.responder_cwd or ROOT)
        self.event("learner_start", match=cfg["id"], ordinal=cfg["ordinal"],
                   reason="match start", launched=responder.get("launched"),
                   log=responder.get("log"), checkpoint_in=commands["checkpoint_in"],
                   checkpoint_out=commands["checkpoint_out"])
        clients = [self.launch(f"client-{i}", cmd, os.path.join(directory, f"client-{i}.log"), ROOT) for i, cmd in enumerate(commands["clients"])]
        profile = None
        if cfg.get("operating_profile"):
            from solver.strat.roofline import LiveOperatingProfile
            profile = LiveOperatingProfile(
                cfg["operating_profile"], cfg["teams"], cfg["carts"],
                commands["maxplayers"],
                commands["initial_bots"],
                cfg.get("environment", cfg["id"]),
                os.path.join(directory, "roofline.json"),
            )

        restarts = []
        retired = []
        expert_restarts = []
        expert_retired = []
        deadline = time.monotonic() + cfg["duration"] + self.args.round_grace
        while time.monotonic() < deadline and not self.stopping:
            time.sleep(min(0.5, max(0.0, deadline - time.monotonic())))
            if profile is not None:
                profile.poll(server)
            if self.stopping:
                break
            expert_proc = expert.get("process")
            if commands["expert"] and (expert_proc is None or expert_proc.poll() is not None):
                reason = f"expert exited with returncode {None if expert_proc is None else expert_proc.poll()}"
                self.event("expert_restart", match=cfg["id"], ordinal=cfg["ordinal"],
                           reason=reason, log=expert.get("log"))
                expert_retired.append(self.finish(expert, time.monotonic()))
                transition = None
                if commands["expert_stop"]:
                    started = utcnow()
                    try:
                        result = subprocess.run(commands["expert_stop"], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                        transition = {"command": commands["expert_stop"], "returncode": result.returncode,
                                      "output": result.stdout, "started": started, "ended": utcnow()}
                    except Exception as exc:
                        transition = {"command": commands["expert_stop"], "returncode": None,
                                      "error": f"{type(exc).__name__}: {exc}", "started": started, "ended": utcnow()}
                index = len(expert_restarts)
                expert = self.launch(
                    "expert", commands["expert"],
                    os.path.join(directory, f"expert.restart{index}.log"),
                    self.args.expert_cwd or ROOT,
                )
                expert_restarts.append({"reason": reason, "at": utcnow(),
                                        "launched": expert.get("launched"),
                                        "log": expert.get("log"), "transition": transition})
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
                   reason="supervisor stopping" if self.stopping else "match duration reached",
                   signal="SIGTERM",
                   checkpoint_out=commands["checkpoint_out"])
        self.terminate(responder)
        responder_result = self.finish(
            responder, time.monotonic() + self.args.quit_grace,
        )
        expert_stop = None
        if commands["expert_stop"]:
            started = utcnow()
            try:
                result = subprocess.run(commands["expert_stop"], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                expert_stop = {"command": commands["expert_stop"], "returncode": result.returncode,
                               "output": result.stdout, "started": started, "ended": utcnow()}
            except Exception as exc:
                expert_stop = {"command": commands["expert_stop"], "returncode": None,
                               "error": f"{type(exc).__name__}: {exc}", "started": started, "ended": utcnow()}
        self.terminate(expert)
        expert_result = self.finish(
            expert, time.monotonic() + self.args.quit_grace,
        )
        expert_collect = None
        if commands["expert_collect"]:
            started = utcnow()
            try:
                result = subprocess.run(commands["expert_collect"], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                expert_collect = {"command": commands["expert_collect"], "returncode": result.returncode,
                                  "output": result.stdout, "started": started, "ended": utcnow()}
            except Exception as exc:
                expert_collect = {"command": commands["expert_collect"], "returncode": None,
                                  "error": f"{type(exc).__name__}: {exc}", "started": started, "ended": utcnow()}
        for client in clients:
            self.stop(client)
        self.stop(server)
        server_result = self.finish(
            server, time.monotonic() + self.args.quit_grace,
        )
        grace = time.monotonic() + self.args.quit_grace
        results = {
            "stage": stage,
            "server": server_result,
            "responder": responder_result,
            "expert": expert_result,
            "responder_restarts": restarts,
            "responder_retired": retired,
            "expert_restarts": expert_restarts,
            "expert_retired": expert_retired,
            "expert_stop": expert_stop,
            "expert_collect": expert_collect,
            "clients": [self.finish(client, grace) for client in clients],
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
        entity = self.prepare_entity(cfg, directory)
        commands = self.commands(cfg, directory, entity)
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
            "commands": {key: commands[key] for key in ("stage", "server", "responder", "expert", "expert_stop", "expert_collect", "clients")},
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
                "scale_checkpoint_out": artifact(commands["scale_checkpoint_out"], self.artifact_cache),
                "checkpoint_initial": artifact(commands["checkpoint_initial"], self.artifact_cache),
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
        if cfg["split"] != "heldout" and (self.args.dry_run or os.path.exists(commands["scale_checkpoint_out"])):
            self.previous_scale_checkpoints[cfg["policy_arm"]] = commands["scale_checkpoint_out"]
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
    ap.add_argument("--skills", default="2,5,8")
    ap.add_argument("--perturbations", default="baseline,fast,slow,volatile")
    ap.add_argument("--off-policy-counts", default="0,1,2")
    ap.add_argument("--policy-arms", default="matrix_fusion,ffn,linear,default")
    ap.add_argument("--study-repetitions", type=int, default=0)
    ap.add_argument("--cycles", type=int, default=0)
    ap.add_argument("--human-counts", default="0")
    ap.add_argument("--human-client-command")
    ap.add_argument("--heldout-fraction", type=float, default=0.2)
    ap.add_argument("--duration", type=float, default=600)
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
    ap.add_argument("--scale-worker-checkpoint")
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
            required_roles = (
                ["expert", "responder"]
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
        teams, players, carts = supervisor.adaptive_axes(
            csv(args.team_counts, int), csv(args.players_per_team, int),
            csv(args.cart_counts, int),
        )
        if args.study_repetitions > 0:
            policy_arms = csv(args.policy_arms)
            perturbations = csv(args.perturbations)
            if not supervisor.capacity_observations:
                team_count = teams[cycle % len(teams)]
                cart_count = carts[(cycle // len(teams)) % len(carts)]
                players_per_team = players[(cycle // max(1, len(teams) * len(carts))) % len(players)]
                return schedule_defaults([{
                    "id": f"capacity-calibration-{cycle:05d}",
                    "map": (maps or ["runningmanctf"])[cycle % len(maps or ["runningmanctf"])],
                    "teams": team_count,
                    "players_per_team": players_per_team,
                    "controllers": {"bot": team_count * players_per_team},
                    "carts": cart_count,
                    "skill": max(csv(args.skills, float)),
                    "perturbation": "baseline",
                    "off_policy_players": 0,
                    "split": "heldout",
                    "seed": args.seed + cycle,
                    "policy_arm": "matrix_fusion",
                }])
            training = generated_schedule(
                args.generate or 1, args.seed + cycle, maps,
                teams, players, carts, csv(args.skills, float),
                perturbations, csv(args.off_policy_counts, int),
                [arm for arm in policy_arms if arm in OPTIMIZATION_ARMS],
                0.0, csv(args.human_counts, int), args.human_client_command,
                include_comparisons=False,
            )
            measured_players = int(supervisor.center_capacity_observation()["point"].get("players") or 0)
            allocate_population(training, measured_players)
            studies = study_schedule(
                args.study_repetitions, args.seed + cycle, maps,
                teams, players, carts, csv(args.skills, float),
                perturbations, policy_arms, {},
                map_offset=cycle * args.study_repetitions * len(perturbations),
            )
            allocate_population(studies, measured_players)
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
