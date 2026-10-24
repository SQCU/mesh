#!/usr/bin/python3
# The node's health report (/var/run/mesh-health): disk against the floor (rdma/mesh-disk.h), idle blockers, ports
# without their RDMA peer (README.md, bin/mesh-health.sh).
import glob, os, re, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import meshlib  # noqa: E402

meshlib.system_path()
OUT, DU = "/var/run/mesh-health", "/var/run/mesh-health.du"
NOW = int(time.time())


def floor(path):
    try:
        return next((line.split()[2] for line in open(path) if line.split()[:2] == ["#define", "MESH_DISK_FLOOR_GIB"]), "")
    except (OSError, IndexError):
        return ""


def gib(column):
    lines = meshlib.out("df", "-k", "/System/Volumes/Data").splitlines()
    try:
        return str(int(int(lines[1].split()[column]) / 1048576))
    except (IndexError, ValueError):
        return ""


def holders():
    found = []
    for line in meshlib.out("pmset", "-g", "assertions").splitlines():
        if re.match(r"^ +pid [0-9]+\(", line) and re.search(r"(UserIsActive|PreventUserIdleDisplaySleep) named", line):
            who = re.sub(r"\s", "_", re.search(r"pid [0-9]+\([^)]*\)", line).group(0)[4:])
            seconds, kind, f = 0, "", line.split()
            for i, w in enumerate(f):
                if re.match(r"^[0-9]+:[0-9][0-9]:[0-9][0-9]$", w):
                    h, m, s = w.split(":")
                    seconds, kind = int(h) * 3600 + int(m) * 60 + int(s), f[i + 1] if i + 1 < len(f) else ""
            found.append("%s:%s:%.1fh" % (who, kind, seconds / 3600))
    return ",".join(found)


def largest():
    rows = []
    for line in meshlib.out("du", "-xk", "-d", "2", "/System/Volumes/Data/Users").splitlines():
        size, _, path = line.partition("\t")
        path = path[len("/System/Volumes/Data"):] if path.startswith("/System/Volumes/Data") else path
        if len(path.split("/")) == 4:
            rows.append((int(size), path))
    return ",".join("%s:%dG" % (path, size / 1048576) for size, path in sorted(rows, key=lambda r: -r[0])[:5])


limit = floor(os.path.join(meshlib.MESH, "bin", "mesh-disk.h")) or "20"
free, used = gib(3), gib(2)
state = "under_floor" if int(free or 0) < int(limit) else "low" if int(free or 0) < 2 * int(limit) else "ok"
blockers, readable, top = holders(), "", ""
if state != "ok":
    try:
        was, readable, top = (open(DU).readline().rstrip("\n").split(" ", 2) + ["", "", ""])[:3]
    except OSError:
        was = ""
    if was != state:
        readable = meshlib.out("du", "-sxk", "/System/Volumes/Data").split("\t")[0].strip()
        readable = str(int(int(readable) / 1048576)) if readable.isdigit() else ""
        top = largest()
        with open(DU, "w") as f:
            f.write("%s %s %s\n" % (state, readable, top))
elif os.path.exists(DU):
    os.unlink(DU)
lines = ["health_updated=" + time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(NOW)),
         "disk_free_gib=%s disk_used_gib=%s disk_floor_gib=%s disk_state=%s" % (free, used, limit, state),
         "idle_blockers=" + (blockers or "none")]
if state != "ok":
    lines.append("disk_readable_gib=%s disk_os_only_gib=%s disk_top=%s" % (
        readable or "unmeasured", int(used or 0) - int(readable) if readable else "unmeasured", top or "unmeasured"))
if state == "low":
    lines.append("warn=disk: %s GiB free, under twice the %s GiB floor (mesh-disk.h); %s GiB used, %s GiB readable by du, the rest "
                 "only the OS can account for (aned's model cache, staged updates); largest: %s" % (free, limit, used, readable or "?", top or "unmeasured"))
if state == "under_floor":
    lines.append("warn=disk: %s GiB free, under the %s GiB floor (mesh-disk.h): logs, records and compiled shares are being skipped; "
                 "%s GiB used, %s GiB readable by du; largest: %s" % (free, limit, used, readable or "?", top or "unmeasured"))
if state != "ok" and blockers:
    lines.append("warn=idle: %s; while a process declares the user active or keeps the display on, dasd runs no idle maintenance "
                 "and aned's SIP-protected model cache is never collected (README.md, power)" % blockers)
for f in meshlib.files("/var/run/mesh-link.*"):
    try:
        t0 = int(open(f).readline().split()[0])
    except (OSError, IndexError, ValueError):
        t0 = NOW
    lines.append("warn=link: %s cable link up but no RDMA peer in the IORegistry, for %d s; the fabric init holds it and never "
                 "brings it down (fabric.log)" % (f[len("/var/run/mesh-link."):], NOW - t0))
with open(OUT + ".tmp", "w") as f:
    f.write("".join(line + "\n" for line in lines))
os.replace(OUT + ".tmp", OUT)
os.chmod(OUT, 0o644)
print("".join(line + "\n" for line in lines), end="")
