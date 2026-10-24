#!/usr/bin/python3
# The node at a glance, for a person: its install, power, health, remote access, network planes, fabric, daemons and the
# bridge's record (README.md, mesh-status).
import os, re, socket, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import meshlib  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RED, BOLD, OFF = "\033[31m", "\033[1m", "\033[0m"


def b(title):
    print("\n%s%s%s" % (BOLD, title, OFF))


def field(path, k):
    try:
        return "\n".join((line.split() + ["", ""])[k] for line in open(path).read().splitlines())
    except OSError:
        return "unknown"


def status(interface):
    return "\n".join(line.split()[1] for line in meshlib.out("ifconfig", interface).splitlines() if "status:" in line and len(line.split()) > 1)


def ipaddr(interface):
    return meshlib.out("ipconfig", "getifaddr", interface).rstrip("\n")


b("NODE")
print("  branch %s converged %s" % (field("/usr/local/mesh/revision", 0), field("/usr/local/mesh/revision", 1)))
uuid = meshlib.platform("IOPlatformUUID")
try:
    import json
    name = json.load(open(os.path.join(HERE, "..", "etc", "mesh-nodes.json")))["nodes"][uuid]["name"]
except (OSError, ValueError, KeyError, TypeError):
    name = "unnamed"
up = re.sub(r",.*users.*", "", re.sub(r".*up ", "", meshlib.out("uptime").rstrip("\n"), count=1), count=1)
print("  %s / host UUID %s  up %s" % (name, uuid, up))
b("POWER")
for line in meshlib.out("pmset", "-g", "custom").splitlines():
    if re.match(r"^ (sleep|displaysleep|disksleep|standby|autorestart|womp|powermode)", line):
        f = line.split() + [""]
        print("  %-22s %s" % (f[0], f[1]))
b("ASSERTIONS")
for line in meshlib.out("pmset", "-g", "assertions").splitlines():
    f = line.split()
    if re.search(r"UserIsActive|PreventUserIdleDisplaySleep|PreventUserIdleSystemSleep|PreventSystemSleep", line) and len(f) == 2:
        print("  %-30s %s" % (f[0], f[1]))
b("HEALTH")
if os.path.isfile("/var/run/mesh-health"):
    for line in open("/var/run/mesh-health").read().splitlines():
        print("  " + re.sub(r"^warn=", "WARN ", line))
else:
    print("  no report: /var/run/mesh-health is written by mesh-health.sh, which install.sh runs")
b("REMOTE")
for port, what in ((22, "ssh"), (5900, "screensharing")):
    try:
        socket.create_connection(("127.0.0.1", port), timeout=5).close()
        print("  %s (%d): OPEN" % (what, port))
    except OSError:
        print("  %s (%d): CLOSED" % (what, port))
app = os.path.isdir("/Applications/Tailscale.app")
tailscale = ["/Applications/Tailscale.app/Contents/MacOS/tailscale"] if app else ["/usr/local/mesh/bin/tailscale", "--socket=/var/run/tailscaled.socket"]
state = meshlib.run("plutil", "-extract", "BackendState", "raw", "-o", "-", "-", input=meshlib.out(*tailscale, "status", "--json"), stdin=None)
print("  tailnet: %s %s%s" % (state.stdout.rstrip("\n") if state.returncode == 0 else "not answering",
                              meshlib.out(*tailscale, "ip", "-4").rstrip("\n"), " (Tailscale.app)" if app else ""))
b("NETWORK")
for interface in ("en0", "en1", "bridge0"):
    print("  %s: %s" % (interface, ipaddr(interface) or "<none>"))
b("LINKS")
lan = "\n".join(line.split()[1] for line in meshlib.out("route", "-n", "get", "default").splitlines() if "interface:" in line)
lanok = int(bool(lan) and bool(ipaddr(lan)))
devices = [d for _, d, _, _ in meshlib.rdma()]
fabric = int(any(status(d[5:] if d.startswith("rdma_") else d) == "active" for d in devices))
print("  wifi/lan     : %s" % ("up via " + lan if lanok else "DOWN"))
print("  fabric link  : %s (ifconfig; RDMA peers from the IORegistry in the RDMA section)" % ("up" if fabric else "DOWN"))
if lanok + fabric < 2:
    print("  %sDEGRADED: %d of 2 planes. One more failure strands this node.%s" % (RED, lanok + fabric, OFF))
    print("  %sRepair before running anything that can disrupt the remaining plane.%s" % (RED, OFF))
else:
    print("  redundant: 2 of 2")
b("RDMA")
rdma = meshlib.run("/usr/bin/rdma_ctl", "status")
nvram = (meshlib.out("nvram", "rdma-enable").split() + ["", ""])[1]
print("  rdma_ctl: %s   nvram: %s" % ((rdma.stdout + rdma.stderr).rstrip("\n"), nvram))
for d in devices:
    port = d[5:] if d.startswith("rdma_") else d
    try:
        peer = "\n".join(line.split()[2][len("rdma_peer="):] for line in open("/var/run/mesh-fabric").read().splitlines()
                         if line.split()[:1] == ["port=" + port] and len(line.split()) > 2)
    except OSError:
        peer = ""
    held = ""
    if os.path.isfile("/var/run/mesh-link." + port):
        try:
            held = " (no peer for %d s)" % (int(time.time()) - int(open("/var/run/mesh-link." + port).readline().split()[0]))
        except (IndexError, ValueError):
            held = ""
    print("  %-10s link=%s rdma_peer=%s%s" % (d, status(port), peer, held))
b("THUNDERBOLT")
shown = [line.lstrip(" ") for line in meshlib.out("system_profiler", "SPThunderboltDataType").splitlines()
         if re.search(r"Device Name:|Speed:|Status:", line)]
for line in shown[:12]:
    print("  " + line)
b("DAEMONS")
for label, kind in (("io.mesh.beacon", "resident"), ("io.mesh.fabric", "oneshot"), ("io.mesh.router", "resident"), ("io.mesh.rdma-init", "oneshot")):
    info = meshlib.out("launchctl", "print", "system/" + label)
    pid = next((line.split("= ", 1)[1].replace(" ", "") for line in info.split("\n") if line.startswith("\tpid ") and "= " in line), "")
    if not info:
        print("  %-22s %-9s %sNOT LOADED%s" % (label, kind, RED, OFF))
    elif kind == "resident" and not pid:
        print("  %-22s %-9s %sDOWN%s" % (label, kind, RED, OFF))
    else:
        print("  %-22s %-9s %s" % (label, kind, "pid " + pid if pid else ""))
b("FABRIC PEERS")
record, named = meshlib.stat(HERE), meshlib.names(HERE)
own, count = meshlib.raw(record.get("node")), 0
for node in record.get("nodes") or []:
    if not isinstance(node, dict) or "node" not in node:
        break
    n = meshlib.raw(node["node"])
    print("  %-24s node %-10s %s%s" % (named.get(int(n)) if n.isdigit() and int(n) in named else "node " + n, n,
                                       meshlib.raw(node.get("model")), "  (this node)" if n == own else ""))
    count += 1
if not count:
    print("  this node's bridge holds no record (mesh-stat errno %s)" % (meshlib.raw(record["errno"]) if "errno" in record else "unread"))
print()
