#!/usr/bin/python3
# The node's identity, facts and topology as key=value lines, `end` last (rdma/peers.py reads them), run by the node's
# resident rank for a {"facts"} request (metal-microbench docs/elastic.md#call-delivery; README.md, mesh-nodeinfo).
import glob, json, os, re, stat, subprocess, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import meshlib  # noqa: E402

meshlib.system_path()
HERE = os.path.dirname(os.path.abspath(__file__))
P = meshlib.PREFIX
out = meshlib.out


def squeeze(value, chars):
    return re.sub("[%s]+" % chars, "_", value)


def fact(key, value):
    return "%s=%s" % (key, squeeze(value.rstrip("\n"), " \t\n\v\f\r_"))


def plutil(key, path):
    p = meshlib.run("plutil", "-extract", key, "raw", "-o", "-", path)
    return p.returncode == 0, p.stdout.rstrip("\n")


def text(path):
    try:
        return open(path).read()
    except OSError:
        return None


def column(path, k):
    t = text(path)
    return "unknown" if t is None else "\n".join((line.split() + ["", ""])[k] for line in t.splitlines())


try:
    user = open(os.path.join(meshlib.MESH, "admin-user")).read().rstrip("\n")
except OSError:
    user = out("id", "-un").rstrip("\n")
home = os.path.expanduser("~" + user)
state = os.path.join(home, ".cache", "mesh")
uuid = meshlib.platform("IOPlatformUUID")


def booted():
    memory = out("sysctl", "-n", "hw.memsize").strip()
    gpu = next((line.split("= ", 1)[1] for line in out("ioreg", "-rc", "AGXAccelerator", "-d", "1").splitlines()
                if '"gpu-core-count"' in line and "= " in line), "")
    return [fact("node_id", uuid), fact("serial", meshlib.platform("IOPlatformSerialNumber")), fact("model", out("sysctl", "-n", "hw.model")),
            fact("cores", out("sysctl", "-n", "hw.ncpu")), fact("memgb", str(int(memory or 0) // 1073741824)),
            fact("macos", out("sw_vers", "-productVersion")), fact("chip", out("sysctl", "-n", "machdep.cpu.brand_string")),
            fact("gpu_cores", gpu)]


def facts(prefix):
    found = []
    for f in meshlib.files(os.path.join(state, "facts", "*")):
        for line in (text(f) or "").splitlines() if os.path.isfile(f) else []:
            m = re.match(r"^%s=(.{1,12})" % prefix, line)
            if m:
                found.append("%s:%s" % (os.path.basename(f), m.group(1)))
    return found


def own():
    lines, boot = [], out("sysctl", "-n", "kern.bootsessionuuid").rstrip("\n")
    cache = os.path.join(os.environ.get("HOME") or "/nonexistent", ".cache", "mesh", "nodeinfo.boot")
    try:
        name = json.load(open(os.path.join(HERE, "..", "etc", "mesh-nodes.json")))["nodes"][uuid]["name"]
    except (OSError, ValueError, KeyError, TypeError):
        name = ""
    lines += [fact("name", str(name))] if name else []
    lines.append(fact("boot_id", boot))
    if boot and (text(cache) or "").split("\n", 1)[0] == "boot_id=" + boot:
        lines += text(cache).splitlines()[1:]
    else:
        fresh = booted()
        try:
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            with open("%s.%d" % (cache, os.getpid()), "w") as f:
                f.write("".join(line + "\n" for line in [fact("boot_id", boot)] + fresh))
            os.replace("%s.%d" % (cache, os.getpid()), cache)
        except OSError:
            pass
        lines += fresh
    ula = next((line.split()[1].split("%")[0] for line in out("ifconfig").splitlines()
                if line.split()[:1] == ["inet6"] and len(line.split()) > 1 and line.split()[1].startswith(P + ":")), "")
    lines.append(fact("ula", ula))
    for line in meshlib.run(os.path.join(HERE, "mesh-stat")).stdout.splitlines():
        m = re.search(r'"region":([1-9][0-9]*)', line.split("[", 1)[0])
        if m:
            lines.append("region=" + m.group(1))
    lines.append(fact("last_reset", "\n".join((line.split() or [""])[0] for line in out("sysctl", "-n", "kern.shutdownreason").splitlines())))
    rdma = meshlib.run("rdma_ctl", "status")
    lines.append(fact("rdma", rdma.stdout + rdma.stderr))
    sdk = out("xcrun", "--show-sdk-path").rstrip("\n")
    lines.append(fact("sdk", "yes" if os.path.isfile(sdk + "/usr/include/infiniband/verbs.h") else "no"))
    lines += [fact("branch", column("/usr/local/mesh/revision", 0)), fact("converged", column("/usr/local/mesh/revision", 1)), fact("account", user)]
    instances = os.path.join(state, "instances.json")
    value = ""
    if os.path.isfile(instances):
        attention = plutil("attention", instances)[1]
        whole = (bool(plutil("coefficients", instances)[1]) and re.search(r"[1-9]", plutil("gbps", instances)[1]) is not None
                 and plutil("readings", instances)[0] and plutil("attention_readings", instances)[0]
                 and re.search(r"(?m)^vector ", attention) is not None and re.search(r"(?m)^tile ", attention) is not None)
        value = ("" if whole else "incomplete:") + "%s/%s" % (plutil("node", instances)[1], plutil("cores", instances)[1])
    lines.append(fact("instances", value))
    revisions = facts("revision")
    for p in meshlib.files(os.path.join(home, "Library", "LaunchAgents", "*.plist")):
        label = os.path.basename(p)[:-len(".plist")]
        if os.path.isfile(os.path.join(state, "facts", label)) or plutil("LimitLoadToSessionType", p)[1] != "Background":
            continue
        r = meshlib.run("git", "-C", os.path.dirname(plutil("ProgramArguments.0", p)[1]), "rev-parse", "--short=12", "HEAD")
        if r.returncode == 0:
            revisions.append("%s:%s" % (label, r.stdout.rstrip("\n")))
    lines += [fact("revisions", ",".join(revisions)), fact("images", ",".join(facts("images")))]
    models, found = os.path.join(home, "models"), []
    for relative in meshlib.collated(os.path.relpath(p, models) for p in glob.glob(os.path.join(models, "*", "*"))):
        path = os.path.join(models, relative)
        try:
            st = os.stat(path)
        except OSError:
            continue
        if stat.S_ISREG(st.st_mode) and len(relative.split()) == 1 and not re.search(r"[,:]", relative):
            found.append("%s:%d" % (relative, st.st_size))
    lines.append(fact("models", ",".join(found)))
    health = text("/var/run/mesh-health")
    lines += [line for line in (health or "").splitlines() if re.match(r"^(disk_free_gib|idle_blockers)=", line)]
    lines.append(fact("health_warnings", "unmeasured" if health is None else str(sum(line.startswith("warn=") for line in health.splitlines()))))
    uptime = re.sub(r",[^,]*users.*", "", re.sub(r".*up ", "", out("uptime").rstrip("\n"), count=1), count=1).replace(" ", "")
    lines.append(fact("uptime", uptime))
    lines += [line for line in (text("/var/run/mesh-fabric") or "").splitlines() if line.startswith("port=")]
    routes = set()
    for line in out("netstat", "-rn", "-f", "inet6").splitlines():
        f = line.split()
        if f and f[0].startswith(P + ":") and f[-1] != "lo0":
            routes.add("route=%s via=%s" % (f[0].split("%")[0], f[-1]))
    return lines + meshlib.collated(routes)


mine = "\n".join(own()).rstrip("\n")
print(mine)
keys = {word.split("=", 1)[0] for line in mine.split("\n") for word in line.split(" ")}
for f in meshlib.files(os.path.join(state, "facts", "*")):
    for line in (text(f) or "").splitlines() if os.path.isfile(f) else []:
        if re.match(r"^[a-z][a-z0-9_]*=", line) and not re.match(r"^(revision|images)=", line):
            line = squeeze(line, " \t_")
            if line.split("=", 1)[0] not in keys:
                print(line)
print("end")
