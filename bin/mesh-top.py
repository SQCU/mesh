#!/usr/bin/env mesh-python
import argparse, collections, concurrent.futures, curses, html, json, os, signal, socket, subprocess, sys, time

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "rdma")]
import mesh, mesh_disk, peers

TELEMETRY_PORT = int(os.environ.get("MESH_TELEMETRY_PORT", "8788"))
RESIDENT = ("beacon", "bridge", "router", "nodeinfo", "telemetry", "observer")
SPARK = " ▁▂▃▄▅▆▇█"
PHASES = {getattr(mesh.lib, "MESH_" + name): name for name in ("UNKNOWN", "PAIRING", "PAIRED", "STOPPED")}
NODEINFO_OVER_SSH = "python3 - nodeinfo ::1 <<'PEERS'\n" + open(peers.__file__).read() + "\nPEERS"
USAGE = "mesh-top [-n SECONDS] [--window SECONDS] [--nodes A,B] [--once | --html FILE]   keys: q quit, j/k scroll"

def run(command, timeout):
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True) as process:
        try: stdout, stderr = process.communicate(timeout=timeout)
        except BaseException:
            try: os.killpg(process.pid, signal.SIGTERM)
            except OSError: pass
            process.wait()
            raise
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)

class Node:
    def __init__(self, name, me):
        self.name, self.me = name, me
        self.sample, self.info, self.info_at, self.error, self.transport, self.rtt = None, {}, 0.0, None, None, None
        self.sequence, self.previous, self.retry_at, self.address = None, None, 0.0, None

    def hosts(self):
        return ["127.0.0.1"] if self.me else [self.name + ".local"]

def bracket(host):
    return f"[{host}]" if ":" in host else host

def ssh(host, command, timeout):
    return run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=3", host, command], timeout)

def connect(address, port, timeout):
    family, sockaddr, _ = address
    connection = socket.socket(family, socket.SOCK_STREAM)
    connection.settimeout(timeout)
    try: connection.connect((sockaddr[0], port) + tuple(sockaddr[2:]))
    except Exception:
        connection.close()
        raise
    return connection

def exchange(address, port, request, timeout=2):
    with connect(address, port, min(timeout, 0.7)) as connection:
        connection.settimeout(timeout)
        connection.sendall(request)
        data = b""
        while chunk := connection.recv(1 << 20): data += chunk
    return data

def addresses(host):
    try: return [(family, sockaddr, host) for family, _, _, _, sockaddr in socket.getaddrinfo(host, TELEMETRY_PORT, type=socket.SOCK_STREAM)]
    except Exception: return []

def fetch(node):
    if node.error and time.monotonic() < node.retry_at: return
    started = time.monotonic()
    path = f"/v1/latest?measures=scalars"
    attempts = []
    for address in ([node.address] if node.address else []) + [a for host in node.hosts() for a in addresses(host)]:
        try:
            data = exchange(address, TELEMETRY_PORT, f"GET {path} HTTP/1.1\r\nHost: {bracket(address[2])}\r\nConnection: close\r\n\r\n".encode())
            envelope = json.loads(data.partition(b"\r\n\r\n")[2])
            node.transport, node.address = ("http", address[2]), address
            break
        except Exception as error: attempts.append(f"http {address[2]} {address[1][0]}: {type(error).__name__}")
    else:
        envelope, host = None, node.name + ".local"
        try: envelope, node.transport = json.loads(ssh(host, f"curl -s -m 2 'http://127.0.0.1:{TELEMETRY_PORT}{path}'", 8).stdout), ("ssh", host)
        except Exception as error: attempts.append(f"ssh {host}: {type(error).__name__}")
    node.rtt = time.monotonic() - started
    record = (envelope or {}).get("record") or {}
    if not record.get("sample"):
        node.error, node.transport, node.retry_at = "; ".join(attempts) or f"no address or ssh host answered for {node.name}", None, time.monotonic() + 10
        return
    node.error = None
    if record.get("sequence") != node.sequence:
        node.previous, node.sample, node.sequence = node.sample, record["sample"], record.get("sequence")
    if time.time() - node.info_at > 10:
        try: text = peers.nodeinfo(node.address[2])[0] if node.transport[0] == "http" else ssh(node.transport[1], NODEINFO_OVER_SSH, 8).stdout
        except (OSError, ValueError): text = ""
        if text: node.info, node.info_at = peers.observation(text), time.time()

def fabric(node):
    return (node.sample or {}).get("fabric") or {}

def rate_text(value):
    if value is None: return "-"
    for unit, scale in (("GB/s", 1e9), ("MB/s", 1e6), ("KB/s", 1e3)):
        if value >= scale: return f"{value / scale:.2f} {unit}"
    return f"{value:.0f} B/s"

def count_text(value):
    if value is None: return "-"
    for unit, scale in (("M", 1e6), ("k", 1e3)):
        if value >= scale: return f"{value / scale:.1f}{unit}"
    return f"{value:.0f}"

def duration(seconds):
    seconds = int(seconds)
    days, seconds = divmod(seconds, 86400)
    return (f"{days}d " if days else "") + f"{seconds // 3600}:{seconds % 3600 // 60:02d}"

def shape(nodes, edges):
    degree = collections.Counter(x for edge in edges for x in edge)
    n, e = len(nodes), len(edges)
    if n == 2 and e == 1: return "pair"
    if n > 2 and e == n and all(degree[x] == 2 for x in nodes): return "ring"
    if n > 2 and e == n * (n - 1) // 2: return "mesh"
    if n > 2 and e == n - 1 and max(degree.values(), default=0) == n - 1: return "star"
    if e == n - 1: return "tree"
    return "graph"

class Top:
    def __init__(self, args):
        self.args, self.me, self.nodes, self.seen = args, peers.local_name(), {}, set()
        self.series = collections.defaultdict(lambda: collections.deque(maxlen=240))
        self.pool = concurrent.futures.ThreadPoolExecutor()
        self.browsing = self.pool.submit(peers.browse)
        self.names, self.paired, self.unmapped = {}, set(), None

    def refresh(self):
        """The map of this node's bridge (mesh.current: mesh_map), naming each bridge node; then every node the beacon
        announces (peers.browse) or the map names, this one's own included, and each one's telemetry."""
        if self.browsing.done(): self.seen, self.browsing = self.seen | set(self.browsing.result()), self.pool.submit(peers.browse)
        try:
            now = mesh.current()
            self.names, self.unmapped = dict(zip(now.nodes, now.names)), None
            self.paired = {frozenset((now.names[a], now.names[b])) for a, b in now.map.pairs}
        except (OSError, ValueError) as error: self.paired, self.unmapped = set(), str(error)
        for name in self.seen | set(self.names.values()) | {self.me}:
            if name not in self.nodes and (not self.args.nodes or name in self.args.nodes): self.nodes[name] = Node(name, name == self.me)
        list(self.pool.map(fetch, self.nodes.values()))
        for node in self.nodes.values(): self.account(node)

    def owner(self, address):
        for node in self.nodes.values():
            for port in fabric(node).get("ports") or []:
                if port.get("address") and port["address"] == (address or "").split("%")[0]: return node.name, port["iface"]
        return None, None

    def live(self, node):
        """The node's running bridge (its telemetry's), else None."""
        return next((b for b in fabric(node).get("bridges") or [] if not b.get("exited_at")), None)

    def toward(self, node, peer):
        """The ports of `node`'s running bridge (mesh-stat's peers, by port index) whose peer the map names `peer`."""
        return [p for p in ((self.live(node) or {}).get("stat") or {}).get("peers") or [] if self.names.get(p.get("node")) == peer]

    def account(self, node):
        sample, previous = node.sample, node.previous
        if not sample or sample is getattr(node, "accounted", None): return
        node.accounted = sample
        if not previous: return
        dt = float(sample.get("observed_at", 0)) - float(previous.get("observed_at", 0))
        if dt <= 0: return
        before = {bridge["pid"]: bridge for bridge in previous["fabric"].get("bridges") or []}
        for bridge in sample["fabric"].get("bridges") or []:
            old = before.get(bridge["pid"])
            if any((x.get("log_size") or 0) - x.get("log_offset", 0) > 65536 for x in (old or {}, bridge)): continue
            ports = (bridge.get("stat") or {}).get("peers") or []
            for index, total in (bridge.get("census") or {}).items():
                base = ((old or {}).get("census") or {}).get(index) or {}
                delta = {key: max(0, value - base.get(key, 0)) for key, value in total.items()}
                port = next((p for p in ports if str(p.get("port")) == index), {})
                peer = self.names.get(port.get("node")) or f"node {port.get('node', '?')}"
                end = float(sample["observed_at"])
                self.series[(peer, node.name, "recv")].append((end, dt, delta.get("bytes", 0), delta.get("landed", 0), delta.get("calls", 0)))
                self.series[(node.name, peer, "send")].append((end, dt, delta.get("frames", 0), delta.get("send_completions", 0), delta.get("calls", 0)))

    def rates(self, key, window):
        rows = self.series.get(key) or ()
        if not rows: return None
        end = rows[-1][0]
        recent = [row for row in rows if row[0] > end - window]
        span = sum(row[1] for row in recent) or 1
        last = rows[-1]
        return {"now": last[2] / last[1], "now_ops": last[3] / last[1], "avg": sum(r[2] for r in recent) / span, "avg_ops": sum(r[3] for r in recent) / span,
                "calls": sum(r[4] for r in recent), "peak": max(r[2] / r[1] for r in rows), "spark": [r[2] / r[1] for r in list(rows)[-32:]], "age": time.time() - end}

    def cables(self):
        found = {}
        for node in self.nodes.values():
            for port in fabric(node).get("ports") or []:
                for neighbor in port.get("neighbors") or []:
                    peer, peer_iface = self.owner(neighbor["address"])
                    if peer and peer != node.name:
                        key = frozenset(((node.name, port["iface"]), (peer, peer_iface)))
                        found.setdefault(key, {})[node.name] = neighbor.get("state")
        return found

    def port(self, name, iface):
        node = self.nodes.get(name)
        return node, next((p for p in fabric(node).get("ports") or [] if p.get("iface") == iface), None) if node else None

    def frame(self):
        W = self.args.window
        lines, hints = [], []
        now = time.time()
        add = lambda *segments: lines.append([s if isinstance(s, tuple) else (s, "") for s in segments])
        add((f"mesh top  {time.strftime('%Y-%m-%d %H:%M:%S')}", "head"), (f"   every {self.args.interval:g}s, rates over {W:g}s", "dim"))
        add()
        add(("NODES", "head"))
        add(("  node              chip           cores  mem    up        bridge                               telemetry", "dim"))
        for node in sorted(self.nodes.values(), key=lambda n: n.name):
            s, f, info = node.sample or {}, fabric(node), node.info
            chip = ((s.get("capacity") or {}).get("metrics") or {}).get("chip") or (s.get("host") or {}).get("model") or "?"
            mem = (s.get("host") or {}).get("memory_bytes")
            up = duration(now - f["booted_at"]) if f.get("booted_at") else "?"
            b = self.live(node)
            if b:
                stat = b.get("stat") or {}
                phases = "/".join(PHASES.get(p.get("phase"), "?") for p in stat.get("peers") or []) or ("up" if stat.get("up") else "no region")
                bridge = (f"pid {b['pid']} {phases}", "ok" if "PAIRED" in phases else "warn")
            elif f.get("bridges"):
                b = max(f["bridges"], key=lambda x: x["exited_at"])
                bridge = (f"none (last exited {duration(now - b['exited_at'])} ago)", "bad")
            else: bridge = ("none", "bad")
            transport = f"{node.transport[0]} {node.transport[1]} {node.rtt * 1000:.0f}ms" if node.transport else "unreachable"
            add("  ", (f"{node.name:<17} ", "bold"), f"{chip:<14} {info.get('cores', '?'):>5}  {(str(round(mem / 2**30)) + 'G') if mem else '?':<6} {up:<9} ",
                (f"{bridge[0][:36]:<37}", bridge[1]), (transport, "dim" if node.transport else "bad"))
            services, groups = f.get("services") or {}, collections.defaultdict(list)
            for label in sorted(services):
                rows, short = services[label], label.replace("io.mesh.", "")
                status = next((r.get("status") for r in rows if not r.get("disabled")), None)
                if any(r.get("pid") for r in rows): groups["up"].append(short)
                elif all(r.get("disabled") for r in rows): groups["off"].append(short)
                elif status == "0": groups["ran"].append(short)
                else: groups["failed"].append(f"{short}({status})")
                if short in RESIDENT and not any(r.get("pid") for r in rows) and not all(r.get("disabled") for r in rows):
                    hints.append(("bad", f"{node.name}: resident service {label} is not running (last exit {status})"))
            keeper = f.get("keeper") or {}
            if services:
                add(("       up ", "dim"), (" ".join(groups["up"]), "ok"), ("   ran ok ", "dim"), " ".join(groups["ran"]) or "-",
                    ("   exited " if groups["failed"] else "", "dim"), (" ".join(groups["failed"]), "warn"), (f"   off {len(groups['off'])}", "dim"),
                    (f"   keeper: {keeper['last'][:48]}" if keeper.get("last") else "", "dim"))
            if node.error: hints.append(("bad", f"{node.name}: no telemetry ({node.error})"))
            if f.get("error"): hints.append(("warn", f"{node.name}: telemetry fabric section failed: {f['error']}"))
            for alarm in keeper.get("alarms") or []:
                if now - (keeper.get("modified") or 0) < 3600: hints.append(("warn", f"{node.name}: keeper {alarm}"))
            for b in f.get("bridges") or []:
                if str(b.get("state", "")).startswith("U"): hints.append(("bad", f"{node.name}: bridge pid {b['pid']} is in U (uninterruptible, blocked in a verbs call); never SIGKILL it and do not launch ibv_devinfo there (RDMA-RULES.md)"))
                if b.get("log_error") and not b.get("exited_at"): hints.append(("warn", f"{node.name}: bridge log unreadable by telemetry ({b['log_error']}); no census"))
                if b.get("errors") and not b.get("exited_at"): hints.append(("warn", f"{node.name}: bridge logged {b['errors']} error line(s), last: {b.get('last_error')}"))
            for port in f.get("ports") or []:
                for holder in port.get("holders") or []:
                    if str(holder.get("stat") or "").startswith("U") and holder.get("command") != "mesh-flow":
                        hints.append(("bad", f"{node.name}: {holder['command']} pid {holder['pid']} in U holds {port['device']} (leaked verbs context)"))
        add()
        add(("LINKS", "head"), ("   a<->b by Thunderbolt IP adjacency (ndp); bytes/s are RECV bytes landed, from the bridge's census of each program", "dim"))
        cables = self.cables()
        if not cables: add(("  no cabled fabric neighbour seen", "warn"))
        for key, states in sorted(cables.items(), key=lambda kv: sorted(kv[0], key=str)):
            (a, ai), (b, bi) = sorted(key, key=str)
            na, pa = self.port(a, ai)
            nb, pb = self.port(b, bi)
            def both(getter, good=None):
                values = [getter(p) if p else "?" for p in (pa, pb)]
                style = "ok" if good and all(v == good for v in values) else "warn" if good else ""
                return ("/".join(str(v) for v in values), style)
            add(("  ", ""), (f"{a}:{ai}", "bold"), " <-> ", (f"{b}:{bi}", "bold"), "   TB ", both(lambda p: p.get("tb_link"), "up"),
                ("  link-ups " + "/".join(str((p or {}).get("link_ups", "?")) for p in (pa, pb)), "dim"),
                "   IP ", both(lambda p: p.get("ip"), "active"), "  nbr ", both(lambda p: ",".join(n["state"] for n in p.get("neighbors") or []) or "none"),
                "   RDMA dev ", both(lambda p: {2: "on", 0: "off"}.get(p.get("rdma_power"), p.get("rdma_power")), "on"))
            segments = [("      RDMA port ", "dim")]
            for node, iface, other in ((na, ai, b), (nb, bi, a)):
                ports = self.toward(node, other)
                if ports:
                    phases = "/".join(PHASES.get(q.get("phase"), "?") for q in ports)
                    segments.append((f"{node.name} toward {other}: port {'/'.join(str(q.get('port')) for q in ports)} {phases}   ", "ok" if "PAIRED" in phases else "warn"))
                elif self.live(node): segments.append((f"{node.name}:rdma_{iface} no port of its bridge reaches {other}   ", "warn"))
                else: segments.append((f"{node.name}:rdma_{iface} no bridge   ", "bad"))
            add(*segments)
            for src, dst in ((a, b), (b, a)):
                recv, send = self.rates((src, dst, "recv"), W), self.rates((src, dst, "send"), W)
                spark = "".join(SPARK[min(8, int(8 * v / recv["peak"]))] if recv and recv["peak"] else " " for v in (recv or {}).get("spark", []))
                row = [("      ", ""), (f"{src[:14]}->{dst[:14]}", "accent"), "  "]
                if recv:
                    row += [(f"{rate_text(recv['now']):>11} last", "ok" if recv["now"] else ""), f"  {rate_text(recv['avg']):>11} {W:g}s", f"  peak {rate_text(recv['peak']):>10}  ", (f"{spark:<32}", "accent"),
                            f"  RECV {count_text(recv['avg_ops'])}/s", f"  SEND {count_text((send or {}).get('avg_ops'))}/s", f"  calls {recv['calls']}"]
                else: row.append(("no census yet (rates start after two samples of a node running a bridge)", "dim"))
                add(*row)
            reference = []
            speed = (pa or pb or {}).get("speed")
            if speed: reference.append(f"nominal {speed / 1e9:g} Gb/s = {rate_text(speed / 8)} (IOLinkSpeed)")
            for node, other in ((na, b), (nb, a)):
                bridge = self.live(node) if node else None
                for port in self.toward(node, other):
                    total = (bridge.get("census") or {}).get(str(port.get("port")))
                    if total:
                        reference.append(f"{node.name} port {port.get('port')}: {total['calls']} programs, {rate_text(total['bytes']).replace('/s', '')} landed, {bridge.get('errors', 0)} errors, unretired SEND frames {total['frames'] - total['retired']}")
            for text in reference: add(("      " + text, "dim"))
        add()
        add(("TOPOLOGY", "head"))
        names = sorted({x for key in cables for x, _ in key})
        edges = {frozenset(x for x, _ in key) for key in cables}
        order = lambda edge: sorted(edge)
        if names: add(f"  cabled    {shape(names, edges)}: " + "  ".join(" <-> ".join(order(e)) for e in edges), ("   (Thunderbolt IP adjacency, ndp)", "dim"))
        if self.paired: add(f"  paired    {shape(sorted({x for e in self.paired for x in e}), self.paired)}: " + "  ".join(" <-> ".join(sorted(e)) for e in self.paired), ("   (this node's bridge's map, mesh_map)", "dim"))
        else: add(("  paired    " + (self.unmapped or "the map links no two nodes"), "bad" if self.unmapped else "dim"))
        hints = list(dict.fromkeys(hints)) or [("ok", "none")]
        return lines[:2] + [[("HINTS", "head")]] + [[("  " + ("! " if style != "ok" else "") + text, style)] for style, text in hints] + [[]] + lines[2:]

def ansi(lines, color):
    codes = {"head": "\033[1;36m", "bold": "\033[1m", "dim": "\033[2m", "ok": "\033[32m", "warn": "\033[33m", "bad": "\033[1;31m", "accent": "\033[35m", "": ""}
    return "\n".join("".join((codes[style] + text + "\033[0m") if color and style else text for text, style in line) for line in lines)

def page(lines):
    body = "\n".join("".join(f'<span class="{style}">{html.escape(text)}</span>' if style else html.escape(text) for text, style in line) for line in lines)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Mesh Top</title>
<style>:root{{--bg:#fbfbf9;--fg:#1f2328;--dim:#6e7781;--ok:#1a7f37;--warn:#9a6700;--bad:#cf222e;--head:#0969da;--accent:#8250df}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#0f1115;--fg:#e6edf3;--dim:#8b949e;--ok:#3fb950;--warn:#d29922;--bad:#f85149;--head:#58a6ff;--accent:#bc8cff}}}}
:root[data-theme="dark"]{{--bg:#0f1115;--fg:#e6edf3;--dim:#8b949e;--ok:#3fb950;--warn:#d29922;--bad:#f85149;--head:#58a6ff;--accent:#bc8cff}}
body{{margin:0;padding:16px;background:var(--bg);color:var(--fg)}}pre{{margin:0;overflow-x:auto;font:12.5px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace}}
.head{{color:var(--head);font-weight:700}}.bold{{font-weight:700}}.dim{{color:var(--dim)}}.ok{{color:var(--ok)}}.warn{{color:var(--warn)}}.bad{{color:var(--bad);font-weight:700}}.accent{{color:var(--accent)}}</style></head>
<body><pre>{body}</pre></body></html>
"""

def interactive(screen, top):
    curses.curs_set(0)
    curses.use_default_colors()
    styles = {"": 0, "bold": curses.A_BOLD, "dim": curses.A_DIM}
    for index, (name, color) in enumerate((("ok", curses.COLOR_GREEN), ("warn", curses.COLOR_YELLOW), ("bad", curses.COLOR_RED), ("head", curses.COLOR_CYAN), ("accent", curses.COLOR_MAGENTA)), 1):
        curses.init_pair(index, color, -1)
        styles[name] = curses.color_pair(index) | (curses.A_BOLD if name in ("bad", "head") else 0)
    screen.timeout(100)
    offset, due, lines = 0, 0.0, []
    while True:
        if time.monotonic() >= due:
            top.refresh()
            lines, due = top.frame(), time.monotonic() + top.args.interval
            height, width = screen.getmaxyx()
            screen.erase()
            for row, line in enumerate(lines[offset:offset + height - 1]):
                column = 0
                for text, style in line:
                    if column >= width - 1: break
                    screen.addnstr(row, column, text, width - 1 - column, styles.get(style, 0))
                    column += len(text)
            screen.addnstr(height - 1, 0, USAGE[USAGE.index("keys:"):], width - 1, curses.A_DIM)
            screen.refresh()
        key = screen.getch()
        if key in (ord("q"), 27): return
        if key == ord("j"): offset, due = min(offset + 1, max(0, len(lines) - 2)), 0
        if key == ord("k"): offset, due = max(0, offset - 1), 0

def main():
    parser = argparse.ArgumentParser(prog="mesh-top", usage=USAGE, description="Live connectivity and observed RDMA bandwidth between mesh nodes: the nodes the beacon announces and this node's bridge's map names, each read from its io.mesh.telemetry (:8788) and io.mesh.nodeinfo; sends no RDMA traffic of its own.")
    parser.add_argument("-n", "--interval", type=float, default=1.0)
    parser.add_argument("--window", type=float, default=5.0)
    parser.add_argument("--nodes", type=lambda text: [x for x in text.split(",") if x], default=None)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--html")
    args = parser.parse_args()
    top = Top(args)
    if args.once or args.html:
        top.refresh()
        for _ in range(max(2, int(args.window / args.interval) + 1)):
            time.sleep(args.interval)
            top.refresh()
        lines = top.frame()
        if args.html: mesh_disk.write(args.html, page(lines), f"mesh-top's page {args.html}")
        if args.once or not args.html: print(ansi(lines, sys.stdout.isatty()))
        return
    curses.wrapper(interactive, top)

if __name__ == "__main__":
    main()
