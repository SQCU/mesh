#!/usr/bin/env mesh-python
import argparse
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib


SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8"]
ROOT = Path(__file__).resolve().parents[1]


def report(event, **values):
    print(json.dumps({"event": event, **values}), file=sys.stderr, flush=True)


def command(values, host=None, **kwargs):
    return subprocess.run(SSH + [host, shlex.join(map(str, values))] if host else list(map(str, values)),
                          check=True, **kwargs)


def publish(path, target):
    temporary = path.with_name(f".{path.name}-{time.time_ns()}")
    temporary.symlink_to(target)
    os.replace(temporary, path)


def inventory(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*")) if path.is_file()
            and "__pycache__" not in path.parts and path != root / "application.json"}


def snapshot(source, stage):
    configuration = tomllib.loads((source / "pyproject.toml").read_text())["tool"]["mesh"]["application"]
    def ignored(directory, names):
        return [name for name in names if any(
            fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(str((Path(directory) / name).relative_to(source)), pattern)
            for pattern in configuration["exclude"])]
    for name in configuration["sources"]:
        origin, destination = source / name, stage / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if origin.is_dir():
            shutil.copytree(origin, destination, ignore=ignored)
        else:
            shutil.copy2(origin, destination)
    command(["make", "-B", "-C", stage / "rdma", "libmesh.dylib", ".build/mesh_abi.py"], stdout=sys.stderr)
    files = inventory(stage)
    identity = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    manifest = {"schema": 1, "id": identity, "created_at": time.time(),
                "source": str(source), "files": files, "modules": configuration["modules"]}
    (stage / "application.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")
    return manifest


def activate(target, stage):
    manifest = json.loads((stage / "application.json").read_text())
    if inventory(stage) != manifest["files"]:
        raise RuntimeError(f"incomplete application transfer: {stage}; current generation retained")
    environment = target / "python"
    command(["bash", stage / "bin/mesh-runtime-install.sh", environment, stage], stdout=sys.stderr)
    runtime = (environment / "runtime-current").resolve()
    if not (runtime / ".venv/bin/python").exists():
        runtime = environment.resolve()
    roots = [str(stage / name) for name in ("xonotic", "rdma", "xonotic/payload/tools")]
    imports = "import sys,importlib; sys.path[:0]=" + repr(roots) + "; "
    imports += "; ".join(f"importlib.import_module({name!r})" for name in manifest["modules"])
    command([runtime / ".venv/bin/python", "-I", "-B", "-c", imports], stdout=sys.stderr)
    manifest["python_root"] = str(runtime)
    (stage / "application.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")
    current = (target / "current").resolve()
    if current != stage and (current / "application.json").is_file():
        try:
            existing = json.loads((current / "application.json").read_text())
        except (OSError, ValueError) as error:
            report("application_manifest_repair", root=str(current), error=str(error))
            existing = {}
        if existing.get("id") == manifest["id"] and existing.get("python_root") == str(runtime) and inventory(current) == manifest["files"]:
            shutil.rmtree(stage)
            stage = current
    publish(target / "current", stage)
    report("application_activated", id=manifest["id"], root=str(stage), python_root=str(runtime))


def deploy(args):
    source = Path(args.source).resolve()
    with tempfile.TemporaryDirectory(prefix="mesh-application-") as temporary:
        stage = Path(temporary) / "source"
        stage.mkdir()
        manifest = snapshot(source, stage)
        remote = Path(args.target) / "generations" / f"{time.time_ns()}-{manifest['id'][:12]}"
        command(["mkdir", "-p", remote], args.host)
        destination = f"{args.host}:{shlex.quote(str(remote))}/" if args.host else str(remote) + "/"
        command(["rsync", "-a", "-e", shlex.join(SSH), str(stage) + "/", destination])
        command([args.python, remote / "bin/mesh-application.py", "activate",
                 "--target", args.target, "--stage", remote], args.host)


def run(args):
    root = (Path(args.target) / "current").resolve()
    manifest = json.loads((root / "application.json").read_text())
    roots = [str(root / name) for name in ("xonotic", "rdma", "xonotic/payload/tools")]
    code = "import sys,runpy; sys.path[:0]=" + repr(roots)
    code += "; sys.argv=sys.argv[1:]; runpy.run_module(sys.argv[0],run_name='__main__',alter_sys=True)"
    values = [str(Path(manifest["python_root"]) / ".venv/bin/python"), "-I", "-B", "-c", code, *args.arguments]
    provenance = {"id": manifest["id"], "root": str(root), "python_root": manifest["python_root"],
                  "native_client": manifest["files"]["rdma/libmesh.dylib"],
                  "wire": manifest["files"]["rdma/xonwire.def"]}
    environment = {**os.environ, "MESH_APPLICATION_ROOT": str(root), "MESH_RUNTIME_ROOT": manifest["python_root"],
                   "MESH_APPLICATION_IDENTITY": json.dumps(provenance)}
    report("application_launch", id=manifest["id"], root=str(root), module=args.arguments[0])
    if args.log:
        log = Path(args.log)
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("ab") as output:
            process = subprocess.Popen(values, env=environment, stdin=subprocess.DEVNULL,
                                       stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        report("application_started", pid=process.pid, log=str(log))
    else:
        os.execve(values[0], values, environment)


def main():
    global SSH
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("deploy", "activate", "launch", "run"))
    parser.add_argument("--source", default=str(ROOT))
    parser.add_argument("--target", required=True)
    parser.add_argument("--host")
    parser.add_argument("--ssh-command", default=shlex.join(SSH))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--stage")
    parser.add_argument("--log")
    values = sys.argv[1:]
    boundary = values.index("--") if "--" in values else len(values)
    args = parser.parse_args(values[:boundary])
    SSH = shlex.split(args.ssh_command)
    args.arguments = values[boundary + 1:]
    if args.operation == "activate":
        activate(Path(args.target).resolve(), Path(args.stage).resolve())
    elif args.operation == "run":
        run(args)
    else:
        try:
            deploy(args)
        except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
            report("application_update_pending", error=str(error), target=args.target,
                   recovery="previous complete generation retained; next launch/install retries source")
            if args.operation == "deploy":
                raise
        if args.operation == "launch":
            launcher = Path(args.target) / "current/bin/mesh-application.py"
            values = [args.python, launcher, "run", "--target", args.target]
            values += ["--log", args.log] if args.log else []
            values = list(map(str, values + ["--", *args.arguments]))
            values = SSH + [args.host, shlex.join(values)] if args.host else values
            os.execvp(values[0], values)


if __name__ == "__main__":
    main()
