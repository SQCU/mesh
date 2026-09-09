import datetime
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    requested = [value.strip() for value in spec.split(",") if value.strip()]
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


class MapAssets:
    def __init__(self, basedir, entity_tool, python, progs, csprogs, checkpoints_per_lane, dry_run, artifact_cache):
        self.basedir, self.entity_tool, self.python = basedir, entity_tool, python
        self.progs, self.csprogs = progs, csprogs
        self.checkpoints_per_lane, self.dry_run = checkpoints_per_lane, dry_run
        self.artifact_cache = artifact_cache
        program_files = sorted(glob.glob(os.path.join(os.path.dirname(self.entity_tool), "*.py")))
        program_files.append(os.path.join(ROOT, "payload", "cfg", "gamemodes-payload.cfg"))
        self.entity_program_id = hashlib.sha256(json.dumps({
            os.path.relpath(path, ROOT): artifact(path, self.artifact_cache)
            for path in program_files
        }, sort_keys=True).encode()).hexdigest()
        self.entity_realizations = {}

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

    def prepare(self, cfg, directory):
        userdir = os.path.join(directory, "userdir")
        maps_dir = os.path.join(userdir, "data", "maps")
        os.makedirs(maps_dir, exist_ok=True)
        data_dir = os.path.join(userdir, "data")
        ent = os.path.join(maps_dir, cfg["map"] + ".ent")
        measurements_path = ent + ".measurements.json"
        mapinfo = os.path.join(maps_dir, cfg["map"] + ".mapinfo")
        record = {"userdir": userdir, "entity": ent, "measurements_path": measurements_path, "mapinfo": mapinfo}
        if self.dry_run:
            record["dry_run"] = True
            record["command"] = [self.python, self.entity_tool, "<resolved-bsp>", ent, str(cfg["teams"]), str(cfg["carts"]), "<resolved-archive>"]
            record["gamecode"] = {"progs": self.progs, "csprogs": self.csprogs,
                                  "effectinfo": os.path.join(os.path.dirname(self.progs), "effectinfo.txt")}
            return record
        record["gamecode"] = {}
        for name, source, filename in (("progs", self.progs, "progs.dat"), ("csprogs", self.csprogs, "csprogs.dat"),
                                       ("effectinfo", os.path.join(os.path.dirname(self.progs), "effectinfo.txt"), "effectinfo.txt")):
            destination = os.path.join(data_dir, filename)
            try:
                shutil.copyfile(source, destination)
                record["gamecode"][name] = artifact(destination, self.artifact_cache)
            except OSError as exc:
                record["gamecode"][name] = {"path": destination, "source": source, "error": f"{type(exc).__name__}: {exc}"}
        try:
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
                        "checkpoints_per_lane": cfg.get("checkpoints_per_lane", self.checkpoints_per_lane),
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
                        cmd = [self.python, self.entity_tool, source_bsp, ent, str(cfg["teams"]), str(cfg["carts"]), source_archive, str(cfg.get("checkpoints_per_lane", self.checkpoints_per_lane))]
                        record["started"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
                        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                        record.update(command=cmd, output=result.stdout, returncode=result.returncode, ended=datetime.datetime.now(datetime.timezone.utc).isoformat())
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
