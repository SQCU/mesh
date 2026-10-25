import argparse, copy, datetime, glob, itertools, json, math, os, random, select, shlex, shutil, signal, subprocess, sys, threading
from collections import deque

from solver.strat.scale_config import SCALE_EXPERTS, SCALE_HIDDEN, SCALE_RANK, SCALE_TOPK
from solver.strat.policy_contract import MATRIX_FUSION_INTERVENTION_ARMS, OPTIMIZATION_ARMS, JOINT_TRAINING_ARMS, is_matrix_fusion_arm
from solver.strat.capacity import engine_player_capacity
from solver.strat.checkpoint_state import encode_payload, open_checkpoint
from solver.strat.action_history import ExecutionEvaluation
from solver.strat.map_assets import MapAssets, resolve_maps
from solver.strat.study import summarize

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
        result = subprocess.run([python, helper], capture_output=True, text=True, check=True)
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

def telemetry_summary(rows):
    evaluation = ExecutionEvaluation()
    configurations, controllers, arms, first, last, last_config, lines = {}, {}, {}, None, None, None, 0
    for row in rows:
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
        self.basedir = os.path.abspath(os.path.expanduser(args.basedir))
        self.entity_tool = os.path.abspath(os.path.expanduser(args.entity_tool))
        if args.server_host:
            raise SystemExit('the game host runs rank 0 of the responder world; remote work uses its bridge')
        with open(os.path.expanduser(args.placement)) as source:
            self.placement = json.load(source)
        self.progs = os.path.abspath(os.path.expanduser(args.progs))
        self.csprogs = os.path.abspath(os.path.expanduser(args.csprogs))
        self.build_command = command(args.build_command)
        self.runtime = runtime_identity(args.python)
        if not args.dry_run:
            self.runtime['application_revision'] = application_revision(application_root)
        self.previous_states, self.initial_states, self.shard_states, self.initial_shards = {}, {}, {}, {}
        self.input_states, self.input_shards, self.entities = {}, {}, {}
        self.map_assets = MapAssets(self.basedir, self.entity_tool, args.python, self.progs,
                                    self.csprogs, args.checkpoints_per_lane, args.dry_run)
        self.stopping = 0
        self.server = None
        self.clients = []
        self.events = deque()
        self.supervisor_events = []
        self.server_outcomes = []
        self.records = []
        self.report = None
        self.wake_r, self.wake_w = os.pipe()
        os.set_blocking(self.wake_r, False)
        os.set_blocking(self.wake_w, False)
        sources = {'matrix_fusion': args.checkpoint} if args.checkpoint else {}
        for value in args.arm_checkpoint:
            arm, _, path = value.partition('=')
            sources[arm] = path
        for arm, path in sources.items():
            self.previous_states[arm] = self.load_state(path)
            self.initial_states[arm] = self.previous_states[arm]
            for rank, state in self.input_shards[os.path.abspath(os.path.expanduser(path))].items():
                self.shard_states.setdefault(rank, {})[arm] = state
                self.initial_shards.setdefault(rank, {})[arm] = state
        self.maps = resolve_maps(args.maps, self.basedir)
        self.build = self.build_gamecode()

    def load_state(self, path):
        source = os.path.abspath(os.path.expanduser(path))
        if source not in self.input_states:
            with open_checkpoint(source) as payload:
                self.input_states[source] = encode_payload({key: payload[key] for key in payload.files})
            self.input_shards[source] = {}
            for member in glob.glob(source + '.rank*.npz'):
                rank = member[len(source + '.rank'):-len('.npz')]
                with open_checkpoint(member) as payload:
                    self.input_shards[source][rank] = encode_payload({key: payload[key] for key in payload.files})
        return self.input_states[source]

    def request_stop(self, signum, frame):
        self.stopping = signum

    def build_gamecode(self):
        record = {'command': self.build_command}
        if self.args.dry_run:
            return record | {'dry_run': True}
        started = utcnow()
        result = subprocess.run(self.build_command, cwd=ROOT, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        return record | {'returncode': result.returncode, 'output': result.stdout,
                         'started': started, 'ended': utcnow()}

    @staticmethod
    def asset_key(cfg):
        return tuple(cfg.get(key) for key in ('map', 'teams', 'carts', 'checkpoints_per_lane',
                                             'bsp', 'entity_file', 'mapinfo_file'))

    def prepare(self, configurations):
        userdir = os.path.join(self.run_dir, 'configuration', 'userdir')
        maps_dir = os.path.join(userdir, 'data', 'maps')
        if not self.args.dry_run:
            os.makedirs(maps_dir, exist_ok=True)
        for cfg in configurations:
            for source in [cfg.get('checkpoint'), *(cfg.get('arm_checkpoints') or {}).values()]:
                if source:
                    self.load_state(source)
            key = self.asset_key(cfg)
            if key in self.entities:
                continue
            name = f'mesh_{len(self.entities):05d}'
            directory = os.path.join(self.run_dir, 'configuration', name)
            entity = self.map_assets.prepare(cfg, directory)
            if not self.args.dry_run:
                for suffix, source in (('.bsp', entity['bsp_path']), ('.ent', entity['entity']),
                                       ('.mapinfo', entity['mapinfo'])):
                    destination = os.path.join(maps_dir, name + suffix)
                    if os.path.lexists(destination):
                        os.unlink(destination)
                    os.symlink(source, destination)
                for filename in ('progs.dat', 'csprogs.dat', 'effectinfo.txt', 'gamemodes-payload.cfg'):
                    destination = os.path.join(userdir, 'data', filename)
                    if os.path.lexists(destination):
                        os.unlink(destination)
                    os.symlink(os.path.join(entity['userdir'], 'data', filename), destination)
            entity.update(userdir=userdir, game_map=name)
            self.entities[key] = entity

    def state_packet(self, cfg, entity):
        requested = cfg.get('arm_checkpoints') or {}
        arms = list(dict.fromkeys([*(cfg.get('team_policy_arms') or [cfg['policy_arm']]),
                                  *cfg.get('train_arms', [])]))
        states, shards = {}, {}
        canonical_path = requested.get('matrix_fusion')
        canonical = canonical_path
        canonical = self.input_states[os.path.abspath(os.path.expanduser(canonical))] if canonical else self.previous_states.get('matrix_fusion')
        for arm in arms:
            source = requested.get(arm) or cfg.get('checkpoint')
            state = self.input_states[os.path.abspath(os.path.expanduser(source))] if source else self.previous_states.get(arm)
            if arm == 'initial_policy':
                state = self.initial_states.get('matrix_fusion', state)
            elif arm in MATRIX_FUSION_INTERVENTION_ARMS and canonical is not None:
                state = canonical
                source = canonical_path
            if state is not None:
                states[arm] = state
            source_arm = 'matrix_fusion' if arm == 'initial_policy' or arm in MATRIX_FUSION_INTERVENTION_ARMS else arm
            source_shards = self.initial_shards if arm == 'initial_policy' else self.shard_states
            if source and arm != 'initial_policy':
                for rank, shard in self.input_shards[os.path.abspath(os.path.expanduser(source))].items():
                    shards.setdefault(rank, {})[arm] = shard
            else:
                for rank, values in source_shards.items():
                    if source_arm in values:
                        shards.setdefault(rank, {})[arm] = values[source_arm]
        return {'states': states, 'initial_states': self.initial_states, 'shards': shards,
                'placement': self.placement,
                'navigation_realization': entity.get('measurements')}

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
        strategy_node = 1 if strategy_node is None else int(strategy_node)
        distributed_scale = bool(self.placement.get("products")) and bool(cfg.get("distributed_scale", True))
        cvars["g_payload_teams_override"] = cfg["teams"]
        cvars["g_payload_mesh_node"] = strategy_node
        cvars["g_payload_strategy_interval"] = self.placement["strategy_interval"]
        try:
            requested_autoscreenshot_mass = int(float(cvars.get("g_max_info_autoscreenshot", 0)))
        except (TypeError, ValueError):
            requested_autoscreenshot_mass = 0
        cvars["g_max_info_autoscreenshot"] = max(
            requested_autoscreenshot_mass,
            int(entity.get("entity_class_mass", {}).get("info_autoscreenshot", 0)),
        )
        server_prefix = [
            "env", f"MESH_REGION={server_region}",
            f"MESH_EXPERT_SOCKET={self.placement['socket']}",
            f"MESH_EXPERT_FRAMEBYTES={self.placement['framebytes']}",
        ] + self.server_prefix
        basedir = self.basedir
        userdir = entity["userdir"]
        game_map = entity["game_map"]
        server_values = {
            "developer": 0, "sv_public": 0, "port": port,
            "sv_random_seed": cfg["seed"], "sv_autopause": 0,
            "timelimit": 0, "maxplayers": maxplayers, "bot_join_empty": 1,
            "bot_number": initial_bots, "skill": cfg["skill"], "g_warmup": 0,
            "g_maplist": game_map, "g_maplist_shuffle": 0, "g_maplist_selectrandom": 0,
            **PAYLOAD_DEFAULTS, **cvars, "g_payload": 1,
            "g_payload_point_limit": cfg.get("score_limit", self.args.score_limit),
            "g_payload_checkpoint_rate": cfg.get("checkpoint_score_rate", self.args.checkpoint_score_rate),
        }
        server = server_prefix + [
            "-norunaway", "-xonotic", "-basedir", basedir, "-userdir", userdir,
            "+exec", "gamemodes-payload.cfg",
        ] + cvar_args(server_values) + command(cfg.get("server_args")) + ["+map", game_map]
        map_transition = "".join(f"{name} {json.dumps(str(value))}\n" for name, value in server_values.items()
                             if name not in ("port", "maxplayers")) + f"changelevel {game_map}\n"
        learning_rate = 0.0 if cfg['split'] == 'heldout' else float(cfg.get('learning_rate', self.args.learning_rate))
        responder = ['env', f'MESH_REGION={responder_region}'] + self.responder_prefix + [
            '--state-stdin', '--placement-json', json.dumps(self.placement),
            '--off-policy-players', str(cfg['off_policy_players']),
            '--learning-rate', str(learning_rate), '--gradient-clip', str(self.args.gradient_clip),
            '--baseline-hidden', str(self.args.baseline_hidden),
            '--scale-rank', str(self.args.scale_rank), '--scale-hidden', str(self.args.scale_hidden),
            '--scale-experts', str(self.args.scale_experts), '--scale-topk', str(self.args.scale_topk),
            '--seed', str(cfg['seed']), '--environment', str(cfg.get('environment', cfg['id'])),
            '--match-metadata', json.dumps({'match_id': cfg['id'], 'configuration': {
                key: cfg.get(key) for key in ('map', 'teams', 'carts', 'players_per_team',
                    'checkpoints_per_lane', 'score_limit', 'checkpoint_score_rate', 'perturbation')},
                'seed': cfg['seed'], 'team_policy_arms': cfg['team_policy_arms']}),
            '--replay-batch', str(self.args.replay_batch), '--replay-weight', str(self.args.replay_weight),
            '--policy-arm', cfg['policy_arm'],
        ]
        if cfg.get('train_arms'):
            responder += ['--train-arms', ','.join(cfg['train_arms'])]
        if cfg['team_policy_arms']:
            responder += ['--team-policy-arms', ','.join(cfg['team_policy_arms'])]
        if cfg['split'] != 'heldout':
            responder.append('--train')
        responder += command(cfg.get('responder_args'))
        context = {'port': port, 'map': game_map, 'seed': cfg['seed'], 'match': cfg['id'], 'directory': directory}
        clients = [client_command(item, context, index) for index, item in enumerate(cfg.get('client_commands', []))]
        return {'server': server, 'transition': map_transition, 'responder': responder,
                'clients': clients, 'state': self.state_packet(cfg, entity),
                'port': port, 'maxplayers': maxplayers, 'initial_bots': initial_bots,
                'remote_scale_arm_mass': remote_scale_arm_mass(cfg), 'distributed_scale': distributed_scale,
                'perturbation': {'name': perturbation, 'cvars': cvars}}

    def launch(self, name, cmd, cwd, state=None):
        try:
            proc = subprocess.Popen(cmd, cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True)
        except Exception as exc:
            return {'name': name, 'command': cmd, 'process': None, 'launched': False,
                    'error': f'{type(exc).__name__}: {exc}'}
        launched = {'name': name, 'command': cmd, 'process': proc, 'launched': True,
                    'started': utcnow(), 'bytes': 0, 'lines': 0, 'result': {},
                    'diagnostics': deque(maxlen=32)}
        launched['thread'] = threading.Thread(target=self.copy, args=(launched,), name=f'{name}-output', daemon=True)
        launched['thread'].start()
        if state is not None:
            proc.stdin.write(json.dumps(state) + '\n')
            proc.stdin.flush()
        return launched

    def copy(self, launched):
        proc, pending = launched['process'], b''
        while chunk := os.read(proc.stdout.fileno(), 1 << 16):
            launched['bytes'] += len(chunk)
            *lines, pending = (pending + chunk).split(b'\n')
            launched['lines'] += len(lines)
            for line in lines:
                if b'MESH_OUTCOME ' in line:
                    self.post('line', launched, line.decode(errors='replace'))
                elif line.startswith(b'{'):
                    try:
                        row = json.loads(line)
                    except ValueError:
                        launched['diagnostics'].append(line.decode(errors='replace'))
                        continue
                    kind = row.get('event')
                    if kind == 'responder_finished':
                        launched['result'] = row
                    elif kind == 'outcome_acknowledged':
                        self.post('line', launched, row)
                else:
                    launched['diagnostics'].append(line.decode(errors='replace'))
        proc.stdout.close()
        launched['returncode'] = proc.wait()
        self.post('exit', launched)

    def post(self, kind, launched, line=None):
        self.events.append((kind, launched, line))
        try:
            os.write(self.wake_w, b"\0")
        except BlockingIOError:
            pass

    def next_event(self):
        while not self.events and not self.stopping:
            select.select([self.wake_r], [], [])
            try:
                os.read(self.wake_r, 1 << 12)
            except BlockingIOError:
                pass
        return None if self.stopping else self.events.popleft()

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
            self.start_client(index, cmd, previous)

    def start_client(self, index, cmd, previous):
        if previous:
            self.finish(previous)
        current = self.launch(f"client-{index}", cmd, ROOT)
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

    def finish(self, launched):
        thread = launched.get("thread")
        if thread is not None:
            thread.join()
        proc = launched.get("process")
        if proc is not None and proc.stdin:
            try:
                proc.stdin.close()
            except BrokenPipeError as error:
                launched["stdin_close_error"] = f"{type(error).__name__}: {error}"
        launched['ended'] = utcnow()
        return self.session_record(launched)

    @staticmethod
    def session_record(item):
        proc = item.get('process')
        return {**{key: value for key, value in item.items()
                   if key not in ('process', 'thread', 'result', 'diagnostics')},
                'diagnostics': list(item.get('diagnostics', ())),
                'pid': None if proc is None else proc.pid,
                'returncode': None if proc is None else proc.poll()}

    def server_outcome(self, cfg, line):
        try:
            event = json.loads(line.partition('MESH_OUTCOME ')[2])
        except ValueError as error:
            self.event('server_outcome_unparsed', match=cfg['id'], error=str(error))
            return
        self.server_outcomes.append({'match': cfg['id'], 'event': event,
            'team_policy_arms': cfg.get('team_policy_arms', []), 'observed_at': utcnow(),
            'source': 'engine_console_independent_of_optimizer'})
        self.event('server_outcome', match=cfg['id'], server_event=event)

    def execute(self, cfg, commands):
        if self.args.dry_run:
            return {'server': {'command': commands['server'], 'launched': False},
                    'responder': {'command': commands['responder'], 'launched': False},
                    'clients': [{'command': cmd, 'launched': False} for cmd in commands['clients']],
                    'measurements': [], 'dry_run': True}
        self.events.clear()
        if self.server is None or self.server['process'] is None or self.server['process'].poll() is not None:
            if self.server is not None:
                self.finish(self.server)
            self.server = self.launch('server', commands['server'], self.args.server_cwd or ROOT)
        else:
            self.server['process'].stdin.write(commands['transition'])
            self.server['process'].stdin.flush()
        server = self.server
        responder = self.launch('responder', commands['responder'], self.args.responder_cwd or ROOT, commands['state'])
        self.event('learner_start', match=cfg['id'], ordinal=cfg['ordinal'], launched=responder['launched'])
        self.reconcile_clients(commands['clients'])
        acknowledged = None
        while acknowledged is None and server.get('process') is not None and responder.get('process') is not None:
            event = self.next_event()
            if event is None:
                break
            kind, launched, line = event
            if kind == 'line':
                if launched is server:
                    self.server_outcome(cfg, line)
                elif launched is responder and line.get('event') == 'outcome_acknowledged':
                    acknowledged = line
            elif launched is server or launched is responder:
                self.event(launched['name'] + '_exit', match=cfg['id'], returncode=launched.get('returncode'))
                break
            else:
                for index, client in enumerate(self.clients[:len(commands['clients'])]):
                    if launched is client:
                        self.start_client(index, commands['clients'][index], client)
                        break
        self.terminate(responder)
        responder_result = self.finish(responder)
        result = responder.get('result', {})
        if cfg['split'] != 'heldout':
            self.previous_states.update(result.get('states', {}))
            for rank, values in result.get('shards', {}).items():
                self.shard_states.setdefault(rank, {}).update(values)
            for rank, values in result.get('initial_shards', {}).items():
                for arm, state in values.items():
                    self.initial_shards.setdefault(rank, {}).setdefault(arm, state)
            for arm, state in result.get('initial_states', {}).items():
                self.initial_states.setdefault(arm, state)
        self.event('learner_stop', match=cfg['id'], acknowledgment=acknowledged,
                   state_arms=sorted(result.get('states', {})))
        return {'server': self.session_record(server), 'responder': responder_result,
                'clients': [self.session_record(client) for client in self.clients],
                'measurements': result.get('measurements', []), 'j_measures': result.get('j_measures'),
                'state_arms': sorted(result.get('states', {})),
                'initial_state_arms': sorted(result.get('initial_states', {}))}

    def run_match(self, cfg):
        cfg['strategy_widths'] = {'residual_rank': self.args.scale_rank,
            'hidden_width': self.args.scale_hidden, 'experts': self.args.scale_experts,
            'topk': self.args.scale_topk}
        started = utcnow()
        entity = self.entities[self.asset_key(cfg)]
        commands = self.commands(cfg, cfg['id'], entity)
        execution = self.execute(cfg, commands)
        measurements = execution.pop('measurements')
        realized = telemetry_summary(measurements)
        configured = {'teams': cfg['teams'], 'carts': cfg['carts'],
            'players': sum(cfg['controllers'].values()),
            'players_per_team': {str(i + 1): value for i, value in enumerate(cfg['players_per_team'])},
            'controllers': {key: value for key, value in cfg['controllers'].items() if value}}
        record = {'id': cfg['id'], 'ordinal': cfg['ordinal'], 'split': cfg['split'],
            'started': started, 'ended': utcnow(), 'configuration': cfg, 'entity': entity,
            'runtime': self.runtime, 'execution': execution, 'realized': realized,
            'measurements': measurements,
            'realization_measures': {'dry_run': self.args.dry_run,
                'entity_returncode': entity.get('returncode'), 'build_returncode': self.build.get('returncode'),
                'telemetry_lines': realized['lines'], 'configured': configured,
                'realized': realized.get('peak_configuration') or realized.get('last_configuration'),
                'state_observations': realized.get('policy_provenance', {})},
            'state_arms': execution.get('state_arms', []),
            'initial_state_arms': execution.get('initial_state_arms', []),
            'port': commands['port'], 'perturbation': commands['perturbation']}
        self.records.append(record)
        return record

    def event(self, kind, **fields):
        self.supervisor_events.append({'event': kind, 'at': utcnow(), **fields})

    def serve(self, make_schedule):
        cycle = ordinal = 0
        defaults = {'seed': self.args.seed, 'policy_arm': csv(self.args.policy_arms)[0],
                    'map': self.maps[0] if self.maps else 'runningmanctf'}
        self.event('supervisor_start', maps=self.maps, build_returncode=self.build.get('returncode'))
        try:
            while not self.stopping and (self.args.cycles <= 0 or cycle < self.args.cycles):
                schedule = make_schedule(cycle)
                self.records.clear()
                self.server_outcomes.clear()
                self.supervisor_events[:] = self.supervisor_events[:1]
                self.event('cycle_start', cycle=cycle, matches=len(schedule), first_ordinal=ordinal)
                for item in schedule:
                    cfg = normalize(item, ordinal, defaults)
                    cfg.update(ordinal=ordinal, cycle=cycle)
                    record = self.run_match(cfg)
                    self.event('match_end', cycle=cycle, ordinal=ordinal, id=record['id'],
                               split=record['split'], map=cfg['map'], telemetry_lines=record['realized']['lines'])
                    ordinal += 1
                    if self.stopping:
                        break
                self.event('cycle_end', cycle=cycle, next_ordinal=ordinal)
                self.report = summarize(self.records, self.supervisor_events)
                cycle += 1
        finally:
            for process in self.clients + ([self.server] if self.server else []):
                self.stop(process)
                self.finish(process)
            self.event('supervisor_stop', signal=self.stopping, cycles=cycle, next_ordinal=ordinal)
        return {'cycles': cycle, 'matches': ordinal, 'study': self.report,
                'server_outcomes': self.server_outcomes, 'supervisor': self.supervisor_events}

    def plan(self, schedule):
        defaults = {'seed': self.args.seed, 'policy_arm': csv(self.args.policy_arms)[0],
                    'map': self.maps[0] if self.maps else 'runningmanctf'}
        for index, item in enumerate(schedule):
            cfg = normalize(item, index, defaults)
            cfg['ordinal'] = index
            self.run_match(cfg)
        return self.records

def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest")
    ap.add_argument("--output", help="one final metric report after the supervisor has stopped")
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
    ap.add_argument("--run-dir", default=os.path.join(ROOT, "solver", "strat", "runs", "curriculum"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--engine", default=os.path.expanduser("~/dox/mesh/xonotic/darkplaces-work/darkplaces-dedicated"))
    ap.add_argument("--basedir", default=os.path.expanduser("~/dox/xonotic/Xonotic"))
    ap.add_argument("--server-command")
    ap.add_argument("--server-host")
    ap.add_argument("--responder-command")
    ap.add_argument("--server-cwd")
    ap.add_argument("--responder-cwd")
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
    ap.add_argument("--placement", required=True,
                    help="the responder's placement document (xonotic/README.md \"The policy learner\"): its world's device and products, the engine's socket and frame bytes, the strategy interval")
    ap.add_argument("--strategy-node", type=int)
    ap.add_argument("--port-base", type=int, default=26100)
    return ap

def main(argv=None):
    args = parser().parse_args(argv)
    supervisor = Curriculum(args)
    signal.signal(signal.SIGINT, supervisor.request_stop)
    signal.signal(signal.SIGTERM, supervisor.request_stop)
    signal.set_wakeup_fd(supervisor.wake_w)
    maps = supervisor.maps
    manifest = load_manifest(args.manifest) if args.manifest else None

    def schedule_defaults(rows):
        if args.entity_file:
            source = os.path.abspath(os.path.expanduser(args.entity_file))
            for row in rows:
                row.setdefault("entity_file", source)
        for row in rows:
            row.setdefault("checkpoints_per_lane", args.checkpoints_per_lane)
            row.setdefault("score_limit", args.score_limit)
            row.setdefault("checkpoint_score_rate", args.checkpoint_score_rate)
        return rows

    def make_schedule(cycle):
        if manifest is not None:
            return schedule_defaults(copy.deepcopy(manifest))
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

    if manifest is not None:
        configurations = schedule_defaults(copy.deepcopy(manifest))
    else:
        configurations = schedule_defaults([
            {'map': mapname, 'teams': teams, 'carts': carts}
            for mapname, teams, carts in itertools.product(
                maps, csv(args.team_counts, int), csv(args.cart_counts, int))
        ])
    defaults = {'seed': args.seed, 'policy_arm': csv(args.policy_arms)[0]}
    supervisor.prepare([normalize(row, index, defaults) for index, row in enumerate(configurations)])
    result = supervisor.plan(make_schedule(0)) if args.dry_run else supervisor.serve(make_schedule)
    if args.output:
        from solver.strat.study import write_report
        write_report(args.output, result)
    print(json.dumps(result, sort_keys=True))
    return result

if __name__ == '__main__':
    main()
