import argparse, copy, datetime, glob, hashlib, itertools, json, math, os, random, shlex, shutil, subprocess, sys, time, zipfile


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PERTURBATIONS = {
    "baseline": {},
    "fast": {"g_payload_speed": 45, "g_payload_max_speed": 260},
    "slow": {"g_payload_speed": 20, "g_payload_max_speed": 120},
    "volatile": {
        "g_payload_contest_speed": 36,
        "g_payload_reverse_speed": 24,
        "g_payload_idle_time": 4,
        "g_payload_rollback_speed": 35,
    },
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


def generated_schedule(count, seed, maps, teams, players, carts, skills, perturbations, off_policy, heldout_fraction, human_counts=None, human_client_command=None):
    space = list(itertools.product(maps, teams, players, carts, skills, perturbations, off_policy, human_counts or [0]))
    rng = random.Random(seed)
    heldout_count = min(len(space), max(0, round(len(space) * heldout_fraction)))
    heldout = set(rng.sample(space, heldout_count))
    rng.shuffle(space)
    out = []
    for index in range(count):
        if index and index % len(space) == 0:
            rng.shuffle(space)
        mapname, team_count, ppt, cart_count, skill, perturbation, off, requested_humans = space[index % len(space)]
        humans = min(requested_humans, team_count * ppt)
        split = "heldout" if space[index % len(space)] in heldout else "train"
        out.append({
            "id": f"generated-{index:05d}",
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
            "seed": rng.randrange(1, 2 ** 31),
        })
    return out


def normalize(item, index, defaults):
    cfg = merge(defaults, item)
    cfg["id"] = str(cfg.get("id", f"match-{index:05d}"))
    cfg["map"] = str(cfg.get("map", "runningmanctf"))
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
    return cfg


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path):
    if not os.path.exists(path):
        return {"path": path, "exists": False}
    return {"path": path, "exists": True, "bytes": os.path.getsize(path), "sha256": sha256(path)}


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
    configurations, controllers, first, last, last_config, lines = {}, {}, None, None, None, 0
    if not os.path.exists(path):
        return {"lines": 0, "configurations": [], "controllers": {}}
    with open(path) as handle:
        for raw in handle:
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            lines += 1
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
    return {
        "lines": lines,
        "first_tick": first.get("req_tick") if first else None,
        "last_tick": last.get("req_tick") if last else None,
        "responses": last.get("resp_id") if last else 0,
        "last_configuration": last_config,
        "configurations": [merge(json.loads(key), {"observations": value}) for key, value in sorted(configurations.items())],
        "controllers": controllers,
    }


class Curriculum:
    def __init__(self, args):
        self.args = args
        self.run_dir = os.path.abspath(os.path.expanduser(args.run_dir))
        self.server_prefix = command(args.server_command) or [os.path.abspath(os.path.expanduser(args.engine))]
        self.responder_prefix = command(args.responder_command) or [args.python, "-m", "solver.strat.strat_responder"]
        self.basedir = os.path.abspath(os.path.expanduser(args.basedir))
        self.entity_tool = os.path.abspath(os.path.expanduser(args.entity_tool))
        self.previous_checkpoint = os.path.abspath(os.path.expanduser(args.checkpoint)) if args.checkpoint else None
        os.makedirs(self.run_dir, exist_ok=True)
        self.index_path = os.path.join(self.run_dir, "matches.jsonl")

    def match_dir(self, cfg):
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in cfg["id"])
        return os.path.join(self.run_dir, f"{cfg['ordinal']:05d}-{safe}")

    def locate_asset(self, mapname, suffix):
        loose = os.path.join(self.basedir, "data", "maps", mapname + suffix)
        if os.path.exists(loose):
            return ("file", loose)
        archives = sorted(glob.glob(os.path.join(self.basedir, "data", "*maps*.pk3")), reverse=True)
        member = "maps/" + mapname + suffix
        for archive in archives:
            with zipfile.ZipFile(archive) as bundle:
                if member in bundle.namelist():
                    return ("zip", archive, member)
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
        ent = os.path.join(maps_dir, cfg["map"] + ".ent")
        mapinfo = os.path.join(maps_dir, cfg["map"] + ".mapinfo")
        record = {"userdir": userdir, "entity": ent, "mapinfo": mapinfo}
        if self.args.dry_run:
            record["status"] = "planned"
            record["command"] = [self.args.python, self.entity_tool, "<resolved-bsp>", ent, str(cfg["teams"]), str(cfg["carts"])]
            return record
        try:
            source_ent = cfg.get("entity_file")
            if source_ent:
                shutil.copyfile(os.path.abspath(os.path.expanduser(source_ent)), ent)
                record["source"] = os.path.abspath(os.path.expanduser(source_ent))
                record["returncode"] = 0
            else:
                source_dir = os.path.join(directory, "source")
                bsp = os.path.join(source_dir, cfg["map"] + ".bsp")
                found = ("file", os.path.abspath(os.path.expanduser(cfg["bsp"]))) if cfg.get("bsp") else self.locate_asset(cfg["map"], ".bsp")
                if found:
                    self.extract_asset(found, bsp)
                    cmd = [self.args.python, self.entity_tool, bsp, ent, str(cfg["teams"]), str(cfg["carts"])]
                    record["started"] = utcnow()
                    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                    record.update(command=cmd, output=result.stdout, returncode=result.returncode, source=found, ended=utcnow())
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
            record["status"] = "ready" if os.path.exists(ent) and record.get("returncode") == 0 else "missing"
        except Exception as exc:
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        return record

    def perturbation(self, cfg):
        value = cfg.get("perturbation", "baseline")
        if isinstance(value, dict):
            name = str(value.get("name", "custom"))
            values = value.get("cvars", {key: item for key, item in value.items() if key != "name"})
        else:
            name = str(value)
            values = PERTURBATIONS.get(name, {})
        return name, merge(values, cfg.get("server_cvars", {}))

    def commands(self, cfg, directory, entity):
        port = int(cfg.get("port", self.args.port_base + int(cfg.get("ordinal", 0))))
        players = max(sum(cfg["players_per_team"]), sum(cfg["controllers"].values()))
        maxplayers = int(cfg.get("maxplayers", max(16, players)))
        perturbation, cvars = self.perturbation(cfg)
        server = self.server_prefix + [
            "-xonotic", "-basedir", self.basedir, "-userdir", entity["userdir"],
            "+developer", "0", "+sv_public", "0", "+port", str(port),
            "+sv_autopause", "0", "+g_payload", "1",
            "+g_payload_round_timelimit", str(cfg["duration"]),
            "+timelimit", str(max(1, math.ceil(cfg["duration"] / 60))),
            "+maxplayers", str(maxplayers), "+bot_join_empty", "1",
            "+bot_number", str(cfg["controllers"]["bot"]), "+skill", str(cfg["skill"]),
            "+g_warmup", "0",
        ] + cvar_args(cvars) + command(cfg.get("server_args")) + ["+map", cfg["map"]]
        telemetry = os.path.join(directory, "telemetry.jsonl")
        checkpoint_out = os.path.join(directory, "checkpoint.npz")
        learning_rate = 0.0 if cfg["split"] == "heldout" else float(cfg.get("learning_rate", self.args.learning_rate))
        responder = self.responder_prefix + [
            "--train", "--secs", str(cfg["duration"]),
            "--peer-node", str(cfg.get("peer_node", self.args.peer_node)),
            "--off-policy-players", str(cfg["off_policy_players"]),
            "--learning-rate", str(learning_rate), "--save-every", str(self.args.save_every),
            "--online-checkpoint", checkpoint_out, "--telemetry", telemetry,
            "--seed", str(cfg["seed"]),
        ]
        checkpoint_in = cfg.get("checkpoint", self.previous_checkpoint)
        if checkpoint_in:
            responder += ["--checkpoint", os.path.abspath(os.path.expanduser(checkpoint_in))]
        responder += command(cfg.get("responder_args"))
        context = {"port": port, "map": cfg["map"], "seed": cfg["seed"], "match": cfg["id"]}
        clients = [client_command(item, context, index) for index, item in enumerate(cfg.get("client_commands", []))]
        return {
            "server": server,
            "responder": responder,
            "clients": clients,
            "telemetry": telemetry,
            "checkpoint_in": checkpoint_in,
            "checkpoint_out": checkpoint_out,
            "port": port,
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
                except BrokenPipeError:
                    pass
        handle = launched.get("handle")
        if handle:
            handle.close()
        launched["ended"] = utcnow()
        return {key: value for key, value in launched.items() if key not in ("process", "handle")}

    def execute(self, cfg, commands, directory):
        if self.args.dry_run:
            return {
                "server": {"command": commands["server"], "launched": False},
                "responder": {"command": commands["responder"], "launched": False},
                "clients": [{"command": cmd, "launched": False} for cmd in commands["clients"]],
                "status": "dry_run",
            }
        server = self.launch("server", commands["server"], os.path.join(directory, "server.log"), self.args.server_cwd or ROOT)
        if self.args.startup_secs > 0:
            time.sleep(self.args.startup_secs)
        responder = self.launch("responder", commands["responder"], os.path.join(directory, "responder.log"), self.args.responder_cwd or ROOT)
        clients = [self.launch(f"client-{i}", cmd, os.path.join(directory, f"client-{i}.log"), ROOT) for i, cmd in enumerate(commands["clients"])]
        deadline = time.monotonic() + cfg["duration"]
        while time.monotonic() < deadline:
            time.sleep(min(0.1, deadline - time.monotonic()))
        self.stop(server)
        for client in clients:
            self.stop(client)
        grace = time.monotonic() + self.args.quit_grace
        results = {
            "server": self.finish(server, grace),
            "responder": self.finish(responder, grace),
            "clients": [self.finish(client, grace) for client in clients],
        }
        codes = [results["server"].get("returncode"), results["responder"].get("returncode")] + [item.get("returncode") for item in results["clients"]]
        results["status"] = "complete" if all(code == 0 for code in codes) else "failed"
        return results

    def run_match(self, cfg):
        directory = self.match_dir(cfg)
        os.makedirs(directory, exist_ok=True)
        started = utcnow()
        entity = self.prepare_entity(cfg, directory)
        commands = self.commands(cfg, directory, entity)
        execution = self.execute(cfg, commands, directory)
        realized = telemetry_summary(commands["telemetry"])
        status = execution["status"]
        mismatches = []
        if not self.args.dry_run and entity.get("status") != "ready":
            mismatches.append("entity overlay was not realized")
        if not self.args.dry_run and realized["lines"] == 0:
            mismatches.append("no live responder telemetry was observed")
        actual = realized.get("last_configuration")
        if not self.args.dry_run and actual:
            expected = {
                "teams": cfg["teams"], "carts": cfg["carts"],
                "players": sum(cfg["controllers"].values()),
                "players_per_team": {str(i + 1): value for i, value in enumerate(cfg["players_per_team"])},
                "controllers": {key: value for key, value in cfg["controllers"].items() if value},
            }
            for key in expected:
                if actual.get(key) != expected[key]:
                    mismatches.append(f"{key}: expected {expected[key]!r}, observed {actual.get(key)!r}")
        if mismatches:
            status = "failed"
        record = {
            "id": cfg["id"], "ordinal": cfg["ordinal"], "split": cfg["split"],
            "started": started, "ended": utcnow(), "status": status,
            "configuration": cfg, "entity": entity,
            "commands": {key: commands[key] for key in ("server", "responder", "clients")},
            "execution": execution,
            "realized": realized, "mismatches": mismatches,
            "artifacts": {
                "entity": artifact(entity["entity"]),
                "mapinfo": artifact(entity["mapinfo"]),
                "telemetry": artifact(commands["telemetry"]),
                "checkpoint_in": artifact(commands["checkpoint_in"]) if commands["checkpoint_in"] else None,
                "checkpoint_out": artifact(commands["checkpoint_out"]),
                "server_log": artifact(os.path.join(directory, "server.log")),
                "responder_log": artifact(os.path.join(directory, "responder.log")),
            },
            "port": commands["port"], "perturbation": commands["perturbation"],
        }
        record_path = os.path.join(directory, "match.json")
        with open(record_path, "w") as handle:
            handle.write(json.dumps(record, indent=2, sort_keys=True) + "\n")
        with open(self.index_path, "a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        if cfg["split"] != "heldout" and (self.args.dry_run or record["status"] == "complete" and os.path.exists(commands["checkpoint_out"])):
            self.previous_checkpoint = commands["checkpoint_out"]
        print(json.dumps({"id": cfg["id"], "status": record["status"], "record": record_path}), flush=True)
        return record

    def run(self, schedule):
        records = []
        for index, item in enumerate(schedule):
            try:
                cfg = normalize(item, index, {"duration": self.args.duration, "seed": self.args.seed})
                cfg["ordinal"] = index
                records.append(self.run_match(cfg))
            except Exception as exc:
                fallback = item if isinstance(item, dict) else {"input": item}
                record = {"id": str(fallback.get("id", f"match-{index:05d}")), "ordinal": index, "started": utcnow(), "ended": utcnow(), "status": "failed", "configuration": fallback, "error": f"{type(exc).__name__}: {exc}"}
                with open(self.index_path, "a") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                records.append(record)
                print(json.dumps({"id": record["id"], "status": "failed", "error": record["error"]}), flush=True)
        return records


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest")
    ap.add_argument("--generate", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260830)
    ap.add_argument("--maps", default="runningmanctf")
    ap.add_argument("--team-counts", default="2,3,4,5")
    ap.add_argument("--players-per-team", default="2,4,8")
    ap.add_argument("--cart-counts", default="1,2,3,4")
    ap.add_argument("--skills", default="2,5,8")
    ap.add_argument("--perturbations", default="baseline,fast,slow,volatile")
    ap.add_argument("--off-policy-counts", default="0,1,2")
    ap.add_argument("--human-counts", default="0")
    ap.add_argument("--human-client-command")
    ap.add_argument("--heldout-fraction", type=float, default=0.2)
    ap.add_argument("--duration", type=float, default=600)
    ap.add_argument("--run-dir", default=os.path.join(ROOT, "solver", "strat", "runs", "curriculum"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--engine", default=os.path.expanduser("~/dox/xonotic/build-engine/darkplaces-dedicated"))
    ap.add_argument("--basedir", default=os.path.expanduser("~/dox/xonotic/Xonotic"))
    ap.add_argument("--server-command")
    ap.add_argument("--responder-command")
    ap.add_argument("--server-cwd")
    ap.add_argument("--responder-cwd")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--entity-tool", default=os.path.join(ROOT, "payload", "tools", "mkentfile.py"))
    ap.add_argument("--checkpoint")
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--peer-node", type=int, default=0)
    ap.add_argument("--port-base", type=int, default=26100)
    ap.add_argument("--startup-secs", type=float, default=2)
    ap.add_argument("--quit-grace", type=float, default=10)
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    if args.manifest:
        schedule = load_manifest(args.manifest)
    else:
        count = args.generate or 1
        schedule = generated_schedule(
            count, args.seed, csv(args.maps), csv(args.team_counts, int),
            csv(args.players_per_team, int), csv(args.cart_counts, int),
            csv(args.skills, float), csv(args.perturbations),
            csv(args.off_policy_counts, int), args.heldout_fraction,
            csv(args.human_counts, int), args.human_client_command,
        )
    return Curriculum(args).run(schedule)


if __name__ == "__main__":
    main()
