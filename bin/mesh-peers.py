#!/usr/bin/python3
# The nodes this node knows, as lines: "node N self|peer [name=...]", "info N via=bridge ..." from its bridge's record,
# "info N via=mdns ..." and "path N kind=K addr=A" from each beacon (_meshnode._tcp) and its addresses, "gone N" when a
# beacon leaves. Runs until stopped; every dns-sd it starts ends with it (README.md, mesh-peers).
import hashlib, os, re, select, signal, stat, subprocess, sys, threading, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import meshlib  # noqa: E402

meshlib.system_path()
here = os.path.dirname(os.path.abspath(__file__))
lock, procs, by_key, counts, ids, seen = threading.Lock(), [], {}, {}, {}, set()
LAN = re.compile(r"^(10\.|192\.168\.|172\.1[6-9]\.|172\.2[0-9]\.|172\.3[01]\.)")


def say(line):
    with lock:
        try:
            sys.stdout.write(line + "\n")
            sys.stdout.flush()
        except BrokenPipeError:
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
            os.kill(os.getpid(), signal.SIGTERM)


def spawn(key, *argv):
    p = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, text=True, errors="replace")
    with lock:
        procs.append(p)
        if key:
            by_key.setdefault(key, []).append(p)
    return p


def end(ps):
    for p in ps:
        if p.poll() is None:
            try:
                p.kill()
            except OSError:
                pass


def start(target, *args):
    threading.Thread(target=target, args=args, daemon=True).start()


def named(n):
    with lock:
        if n in seen:
            return
        seen.add(n)
    nm = names.get(int(n)) if n.isdigit() else None
    say("node %s %s%s" % (n, "self" if n == me else "peer", " name=" + nm if nm else ""))


def norm(a):
    a = a.lower()
    a = a[:-4] if a.endswith("%<0>") else a
    parts = a.split(":")
    return ":".join(p.lstrip("0") or "0" for p in parts) if len(parts) > 2 else a


def kind(a):
    if a in ("0.0.0.0", "127.0.0.1", "::1", "fe80::1%lo0", ""):
        return None
    if LAN.match(a):
        return "lan"
    if a.startswith("169.254."):
        return "fabric-v4ll"
    if a.startswith(meshlib.PREFIX + ":"):
        return "fabric-routed"
    if a.startswith("2") and ":" in a:
        return "lan-v6"
    if a.startswith("fe80:") and "%" in a:
        return "fabric-adjacent" if a.rsplit("%", 1)[1] in ports else None
    return None


def addresses(host, key, ident):
    for line in spawn(key, "dns-sd", "-G", "v4v6", host).stdout:
        f = line.split()
        if len(f) > 1 and f[1] == "Add":
            a = norm(f[5] if len(f) > 5 else "")
            k = kind(a)
            if k:
                say("path %s kind=%s addr=%s" % (ident, k, a))


def follow(instance, key):
    target = ident = ""
    for line in spawn(key, "dns-sd", "-L", instance, "_meshnode._tcp", "local").stdout:
        line = line.rstrip("\n")
        if " can be reached at " in line:
            t = line.split(" can be reached at ", 1)[1].split(":", 1)[0]
            t = t[:-1] if t.endswith(".") else t
            if not t or t == target:
                continue
            target = t
            if ident:
                start(addresses, target, key, ident)
        elif line.startswith(" ") and "=" in line:
            n = next((w[len("node="):] for w in line.split() if w.startswith("node=")), "") or "unidentified-" + key
            if n != ident:
                ident = n
                with lock:
                    ids[key] = ident
                named(ident)
                if target:
                    start(addresses, target, key, ident)
            say("info %s via=mdns%s" % (ident, line))


def reader():
    fd = sys.stdout.fileno()
    if not (stat.S_ISFIFO(os.fstat(fd).st_mode) or stat.S_ISSOCK(os.fstat(fd).st_mode)):
        return
    queue, change = select.kqueue(), select.kevent(fd, select.KQ_FILTER_WRITE, select.KQ_EV_ADD)
    while not any(e.flags & select.KQ_EV_EOF for e in queue.control([change], 1, 1)):
        time.sleep(1)
    os.kill(os.getpid(), signal.SIGTERM)


def stop(signum, _):
    sys.exit(1 if signum in (signal.SIGINT, signal.SIGTERM) else 128 + signum)


for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGALRM):
    signal.signal(s, stop)
start(reader)
try:
    names, me = meshlib.names(here), str(meshlib.fnv(meshlib.platform("IOPlatformUUID")))
    ports = [p[0] for p in meshlib.rdma()]
    for node in meshlib.stat(here).get("nodes") or []:
        if not isinstance(node, dict) or "node" not in node:
            break
        n = meshlib.raw(node["node"])
        named(n)
        say("info %s via=bridge model=%s memory=%s region=%s" % (n, meshlib.raw(node.get("model")), meshlib.raw(node.get("memory")),
                                                                 meshlib.raw(node.get("region"))))
    for line in spawn(None, "dns-sd", "-B", "_meshnode._tcp", "local").stdout:
        f = line.strip().split(None, 6)
        if len(f) < 2 or f[1] not in ("Add", "Rmv"):
            continue
        instance = f[6] if len(f) > 6 else ""
        k = hashlib.sha1(instance.encode()).hexdigest()[:16]
        c = counts.get(k, 0)
        if f[1] == "Add":
            counts[k] = c + 1
            if c == 0:
                start(follow, instance, k)
        elif c > 1:
            counts[k] = c - 1
        else:
            counts.pop(k, None)
            with lock:
                gone, ident = by_key.pop(k, []), ids.pop(k, "unidentified-" + k)
            end(gone)
            say("gone %s" % ident)
finally:
    with lock:
        everything = list(procs)
    end(everything)
