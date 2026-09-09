#!/usr/bin/env mesh-python
import hashlib, importlib.metadata, json, os, platform, sys

root = os.environ.get("MESH_RUNTIME_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
lock = os.path.join(root, "uv.lock")
config = os.path.join(sys.prefix, "pyvenv.cfg")
values = {}
configuration_error = None
try:
    with open(config) as handle:
        values = dict(line.strip().split(" = ", 1) for line in handle if " = " in line)
except OSError as error:
    configuration_error = f"{type(error).__name__}: {error}"
lock_error = None
try:
    with open(lock, "rb") as handle:
        lock_sha256 = hashlib.sha256(handle.read()).hexdigest()
except OSError as error:
    lock_sha256 = None
    lock_error = f"{type(error).__name__}: {error}"
application_root = os.environ.get("MESH_APPLICATION_ROOT", os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
try:
    with open(os.path.join(application_root, "application.json")) as handle:
        application = json.load(handle)
    application = {key: value for key, value in application.items() if key != "files"}
    application["root"] = application_root
except OSError as error:
    application = {"root": application_root, "state": "unbundled", "error": str(error)}
print(json.dumps({
    "schema": 2,
    "launcher": "mesh-python",
    "python": platform.python_version(),
    "executable": sys.executable,
    "prefix": sys.prefix,
    "implementation": platform.python_implementation(),
    "uv": values.get("uv"),
    "lock_sha256": lock_sha256,
    "configuration_error": configuration_error,
    "lock_error": lock_error,
    "application": application,
    "packages": {name: importlib.metadata.version(name) for name in ("mlx", "mlx-metal", "numpy")},
    "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
}, sort_keys=True))
