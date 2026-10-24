#!/usr/bin/python3
# The substrate as its user's (design/transport.md): the user's mesh checkout built and installed (make all install:
# the bridge and the libraries, each image that changed renamed into rdma/) with its Python bindings, ~/.venv-mesh-uv
# with torch-mesh and the checkout's library on its path, and the bridge as the user's launchd agent io.mesh.bridge, run from
# the checkout with the node's bridge configuration, the operand CONFIGURATION (a JSON, TOML or YAML document,
# etc/bridge/README.md): rdma/bridge_configuration.py writes its canonical form, which the launchd agent's
# definition names, so the keeper's load at boot keeps it.  There is no default configuration.
# A definition this script writes never changes with what it runs: a loaded launchd agent is left running (meshlib.launchd).
# The document's path is recorded beside its canonical form.  The bridge's definition names its
# program and its configuration's path alone: a new binary, libmesh or configuration is installed by rename, and the
# bridge's heartbeat, sent SIGHUP, restarts the bridge on them, its region kept (design/transport.md section 2).  The install
# records what it installed (~/.cache/mesh/facts/io.mesh.bridge, which node-info publishes).
# Run as the user, again after every pull:  mesh-user.sh CONFIGURATION [CHECKOUT]  (~/mesh)
import hashlib, os, shutil, subprocess, sys, tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import meshlib  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)
home = os.path.expanduser("~")
if not 2 <= len(sys.argv) <= 3 or not os.path.isfile(sys.argv[1]):
    print("mesh-user.sh CONFIGURATION [CHECKOUT]: the node's bridge configuration document (JSON, TOML or YAML; "
          "etc/bridge/README.md) is required, the bridge has no default", file=sys.stderr)
    sys.exit(64)
document = os.path.join(os.path.realpath(os.path.dirname(sys.argv[1]) or "."), os.path.basename(sys.argv[1]))
source = os.path.realpath(sys.argv[2] if len(sys.argv) > 2 else os.path.join(home, "mesh"))
if not os.path.isdir(source):
    sys.exit(1)
venv, rdma = os.path.join(home, ".venv-mesh-uv"), os.path.join(source, "rdma")
support, state = os.path.join(home, "Library", "Application Support"), os.path.join(home, ".cache", "mesh")
uv = shutil.which("uv") or "/usr/local/mesh/bin/uv"
python = os.path.join(venv, "bin", "python")


def call(*argv, **kw):
    return subprocess.call([str(a) for a in argv], **kw)


def read(path):
    try:
        with open(path, "rb") as f:
            return f.read()
    except OSError:
        return b""


def text(path):
    return read(path).decode(errors="replace").rstrip("\n")


if call("make", "-C", rdma, "all", "install", stdout=subprocess.DEVNULL):
    sys.exit(1)
if not os.access(python, os.X_OK) and call(uv, "venv", "-q", "--python", text(os.path.join(source, ".python-version")), venv):
    sys.exit(1)
if call(uv, "pip", "install", "-q", "--python", python, "-r", os.path.join(rdma, "requirements.txt")):
    sys.exit(1)
bound = subprocess.run([python, "-c", 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))'], capture_output=True, text=True)
if bound.returncode:
    sys.exit(1)
bound = bound.stdout.rstrip("\n")
csum = meshlib.sha1((bound + "\n").encode() + b"".join(read(os.path.join(rdma, f)) for f in ("mesh-plan.h", "nccl.h", "mesh_c_build.py")))
if not (os.path.isfile(os.path.join(rdma, "_mesh_c" + bound)) and text(os.path.join(venv, "mesh-c.sum")) == csum):
    if call(python, os.path.join(rdma, "mesh_c_build.py"), stdout=subprocess.DEVNULL):
        sys.exit(1)
    with open(os.path.join(venv, "mesh-c.sum"), "w") as f:
        f.write(csum + "\n")
if call(python, "-c", 'import sys, sysconfig; open(sysconfig.get_paths()["purelib"] + "/mesh.pth", "w").write(sys.argv[1] + "\\n")', rdma):
    sys.exit(1)
package = os.path.join(rdma, "torch-mesh")
tracked = subprocess.run(["git", "ls-files", "-z"], cwd=package, capture_output=True).stdout.decode().split("\0")
tracked = [f for f in tracked if f]
total = meshlib.sha1(read(os.path.join(rdma, "libnccl-mesh.dylib")) + b"".join(read(os.path.join(package, f)) for f in tracked))
if text(os.path.join(venv, "torch-mesh.sum")) != total:
    build = tempfile.mkdtemp()
    try:
        for f in tracked:
            if os.path.isfile(os.path.join(package, f)):
                os.makedirs(os.path.dirname(os.path.join(build, f)), exist_ok=True)
                shutil.copy2(os.path.join(package, f), os.path.join(build, f))
        status = call(uv, "pip", "install", "-q", "--python", python, "--no-build-isolation", "--reinstall-package", "torch-mesh", build,
                      env=dict(os.environ, MESH_RDMA=rdma))
    finally:
        shutil.rmtree(build, ignore_errors=True)
    if status:
        sys.exit(status)
    with open(os.path.join(venv, "torch-mesh.sum"), "w") as f:
        f.write(total + "\n")
configuration = os.path.join(support, "io.mesh.bridge", "configuration.plist")
renew = False
if call(python, os.path.join(rdma, "bridge_configuration.py"), document, configuration + ".new"):
    sys.exit(1)
if read(configuration + ".new") == read(configuration) and os.path.exists(configuration):
    os.unlink(configuration + ".new")
else:
    renew = os.path.isfile(configuration)
    os.replace(configuration + ".new", configuration)
with open(os.path.join(support, "io.mesh.bridge", "document"), "w") as f:
    f.write(document + "\n")
agents = os.path.join(home, "Library", "LaunchAgents")
for d in (agents, os.path.join(home, "Library", "Logs"), os.path.join(state, "facts")):
    os.makedirs(d, exist_ok=True)
images = hashlib.sha256(b"".join(read(os.path.join(rdma, f)) for f in ("mesh-heartbeat", "mesh-flow", "libmesh.dylib"))).hexdigest()
before = next((line[len("images="):] for line in text(os.path.join(state, "facts", "io.mesh.bridge")).split("\n")
               if line.startswith("images=")), "")
if renew:
    try:
        shutil.copy(os.path.join(rdma, "mesh-flow"), os.path.join(rdma, "mesh-flow.renew"))
        os.replace(os.path.join(rdma, "mesh-flow.renew"), os.path.join(rdma, "mesh-flow"))
    except OSError:
        sys.exit(1)
with open(os.path.join(agents, "io.mesh.bridge.plist.new"), "w") as f:
    f.write("""<?xml version="1.0" encoding="UTF-8"?>
<!-- the node's bridge (bin/mesh-user.sh): its heartbeat (rdma/mesh-heartbeat), which opens and closes the bridge; a new binary or configuration is installed by rename and the heartbeat's SIGHUP restarts both on them, the region kept; SIGUSR1 releases the bridge and its region, SIGTERM releases them and ends the heartbeat (design/transport.md section 2) -->
<plist version="1.0"><dict><key>Label</key><string>io.mesh.bridge</string>
<key>ProgramArguments</key><array><string>%s/rdma/mesh-heartbeat</string><string>%s/rdma/mesh-flow</string><string>%s</string></array>
<key>LimitLoadToSessionType</key><string>Background</string><key>AbandonProcessGroup</key><true/>
<key>RunAtLoad</key><true/><key>KeepAlive</key><dict><key>SuccessfulExit</key><false/></dict><key>ThrottleInterval</key><integer>60</integer><key>ExitTimeOut</key><integer>60</integer>
<key>StandardErrorPath</key><string>%s/Library/Logs/io.mesh.bridge.log</string></dict></plist>
""" % (source, source, configuration, home))
status, said = meshlib.launchd(os.path.join(agents, "io.mesh.bridge.plist.new"))
if said:
    print(said)
if status:
    sys.exit(1)
if renew or before != images:
    meshlib.run("launchctl", "kill", "SIGHUP", "user/%d/io.mesh.bridge" % os.getuid())
    print("io.mesh.bridge: its new binaries or configuration are installed; its heartbeat restarts the bridge on them (SIGHUP), the region kept")
revision = subprocess.run(["git", "-C", source, "rev-parse", "HEAD"], capture_output=True, text=True)
with open(os.path.join(state, "facts", "io.mesh.bridge"), "w") as f:
    f.write("revision=%s\nimages=%s\n" % (revision.stdout.strip() if revision.returncode == 0 else "unknown", images))
