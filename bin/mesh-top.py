#!/usr/bin/env mesh-python
import argparse, collections, concurrent.futures, curses, html, json, os, subprocess, sys, time, urllib.request

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "rdma")]
import mesh, mesh_disk

TELEMETRY_PORT = int(os.environ.get("MESH_TELEMETRY_PORT", "8788"))
LATEST = "/v1/latest?measures=scalars"
RESIDENT = ("beacon", "bridge", "router", "nodeinfo", "telemetry", "observer")
SPARK = " ▁▂▃▄▅▆▇█"
PHASES = {getattr(mesh.lib, "MESH_" + name): name for name in ("UNKNOWN", "PAIRING", "PAIRED", "STOPPED")}
USAGE = "mesh-top [-n SECONDS] [--window SECONDS] [--nodes A,B] [--once | --html FILE]   keys: q quit, j/k scroll"

class Node:
    def __init__(self, name, number, me):
        self.name, self.number, self.me = name, number, me
        self.sample, self.record, self.error, self.rtt = None, {}, None, None
        self.sequence, self.previous, self.asked = None, None, None

def fetch(node):
    """This node's telemetry over loopback, read once; another node's stays on that node, the map giving its facts."""
    started = time.monotonic()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{TELEMETRY_PORT}{LATEST}") as response: envelope = json.load(response)
        node.rtt = time.monotonic() - started
    except (OSError, ValueError) as error:
        node.error, node.rtt = f"{type(error).__name__}: {error}", None
        return
    record = (envelope or {}).get("record") or {}
    if not record.get("sample"):
        node.error = "its telemetry ring has no sample"
        return
    node.error = None
    if record.get("sequence") != node.sequence:
        node.previous, node.sample, node.sequence = node.sample, record["sample"], record.get("sequence")

def fabric(node):
    return ((node.sample if node else None) or {}).get("fabric") or {}

def gib(size):
    return f"{size / 2**30:.0f}G" if size else "?"

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
        self.args, self.nodes = args, {}
        self.me = subprocess.run(["/usr/sbin/scutil", "--get", "LocalHostName"], capture_output=True, text=True).stdout.strip() or "self"
        self.series = collections.defaultdict(lambda: collections.deque(maxlen=240))
        self.pool = concurrent.futures.ThreadPoolExecutor()
        self.names, self.paired, self.unmapped = {}, set(), None

    def refresh(self, wait=False):
        """The map of this node's bridge (mesh.current: mesh_map), naming each bridge node, and its records (mesh.nodes:
        each node's model, memory, window, boot and available memory); then this node's own telemetry asked for where no
        ask of it is under way (`wait`: until it answers)."""
        own, records = None, {}
        try:
            now = mesh.current()
            self.names, self.unmapped, own = dict(zip(now.nodes, now.names)), None, now.node
            self.paired = {frozenset((now.names[a], now.names[b])) for a, b in now.map.pairs}
            records = {record["name"]: record for record in mesh.nodes()}
        except (OSError, ValueError) as error: self.paired, self.unmapped = set(), str(error)
        self.me = self.names.get(own) or self.me
        for number, name in [(own, self.me)] + list(self.names.items()):
            if name not in self.nodes and (not self.args.nodes or name in self.args.nodes or number == own):
                self.nodes[name] = Node(name, number, number == own)
        for node in self.nodes.values():
            node.record = records.get(node.name) or {}
            if node.me and (node.asked is None or node.asked.done()): node.asked = self.pool.submit(fetch, node)
        if wait: concurrent.futures.wait([node.asked for node in self.nodes.values() if node.asked])
        for node in self.nodes.values(): self.account(node)

    def live(self, node):
        """The node's running bridge (its telemetry's), else None."""
        return next((b for b in fabric(node).get("bridges") or [] if not b.get("exited_at")), None)

    def toward(self, node):
        """The ports of `node`'s running bridge (mesh-stat's peers, by port index) by the name the map gives each peer."""
        found = collections.defaultdict(list)
        for port in ((self.live(node) or {}).get("stat") or {}).get("peers") or []:
            found[self.names.get(port.get("node")) or f"node {port.get('node', '?')}"].append(port)
        return found

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

    def frame(self):
        W = self.args.window
        lines, hints = [], []
        now = time.time()
        add = lambda *segments: lines.append([s if isinstance(s, tuple) else (s, "") for s in segments])
        add((f"mesh top  {time.strftime('%Y-%m-%d %H:%M:%S')}", "head"), (f"   every {self.args.interval:g}s, rates over {W:g}s", "dim"))
        add()
        add(("NODES", "head"))
        add(("  node              chip           mem    avail  up        bridge                               telemetry", "dim"))
        for node in sorted(self.nodes.values(), key=lambda n: n.name):
            s, f, r = node.sample or {}, fabric(node), node.record
            chip = ((s.get("capacity") or {}).get("metrics") or {}).get("chip") or (s.get("host") or {}).get("model") or r.get("model") or "?"
            booted = f.get("booted_at") or (r.get("booted") or 0) / 1e9
            up = duration(now - booted) if booted else "?"
            b = self.live(node)
            if b:
                stat = b.get("stat") or {}
                phases = "/".join(PHASES.get(p.get("phase"), "?") for p in stat.get("peers") or []) or ("up" if stat.get("up") else "no region")
                bridge = (f"pid {b['pid']} {phases}", "ok" if "PAIRED" in phases else "warn")
            elif f.get("bridges"):
                b = max(f["bridges"], key=lambda x: x["exited_at"])
                bridge = (f"none (last exited {duration(now - b['exited_at'])} ago)", "bad")
            elif not node.me and r: bridge = (f"in the map, window {gib(r.get('window'))}", "ok")
            else: bridge = ("none", "bad")
            transport = (f"loopback {node.rtt * 1000:.0f}ms", "dim") if node.rtt is not None else ("no telemetry", "bad") if node.error else ("asking", "dim") if node.me else ("on its node", "dim")
            add("  ", (f"{node.name:<17} ", "bold"), f"{chip:<14} {gib((s.get('host') or {}).get('memory_bytes') or r.get('memory')):<6} {gib(r.get('available')):<6} {up:<9} ",
                (f"{bridge[0][:36]:<37}", bridge[1]), transport)
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
            for alarm in keeper.get("alarms") or []: hints.append(("warn", f"{node.name}: keeper {alarm}"))
            for b in f.get("bridges") or []:
                if str(b.get("state", "")).startswith("U"): hints.append(("bad", f"{node.name}: bridge pid {b['pid']} is in U (uninterruptible, blocked in a verbs call); never SIGKILL it and do not launch ibv_devinfo there (RDMA-RULES.md)"))
                if b.get("log_error") and not b.get("exited_at"): hints.append(("warn", f"{node.name}: bridge log unreadable by telemetry ({b['log_error']}); no census"))
                if b.get("errors") and not b.get("exited_at"): hints.append(("warn", f"{node.name}: bridge logged {b['errors']} error line(s), last: {b.get('last_error')}"))
            for port in f.get("ports") or []:
                for holder in port.get("holders") or []:
                    if str(holder.get("stat") or "").startswith("U") and holder.get("command") != "mesh-flow":
                        hints.append(("bad", f"{node.name}: {holder['command']} pid {holder['pid']} in U holds {port['device']} (leaked verbs context)"))
        add()
        own = next((node for node in self.nodes.values() if node.me), None)
        bridge = self.live(own)
        add(("LINKS", "head"), ("   this node's bridge's ports, each peer as the map names it; bytes/s are RECV bytes landed here, from the bridge's census of each program", "dim"))
        toward = self.toward(own)
        if not toward: add(("  " + ("no port of this node's bridge" if bridge else "no bridge runs on this node"), "warn"))
        for peer, ports in sorted(toward.items()):
            phases = "/".join(PHASES.get(q.get("phase"), "?") for q in ports)
            add(("  ", ""), (f"{own.name} port {'/'.join(str(q.get('port')) for q in ports)}", "bold"), " <-> ", (peer, "bold"), "   ", (phases, "ok" if "PAIRED" in phases else "warn"))
            recv, send = self.rates((peer, own.name, "recv"), W), self.rates((own.name, peer, "send"), W)
            if recv:
                spark = "".join(SPARK[min(8, int(8 * v / recv["peak"]))] if recv["peak"] else " " for v in recv["spark"])
                add(("      ", ""), (f"{peer[:14]}->{own.name[:14]}", "accent"), "  ", (f"{rate_text(recv['now']):>11} last", "ok" if recv["now"] else ""), f"  {rate_text(recv['avg']):>11} {W:g}s",
                    f"  peak {rate_text(recv['peak']):>10}  ", (f"{spark:<32}", "accent"), f"  RECV {count_text(recv['avg_ops'])}/s", f"  calls {recv['calls']}")
            if send: add(("      ", ""), (f"{own.name[:14]}->{peer[:14]}", "accent"), f"  SEND {count_text(send['avg_ops'])}/s  frames {count_text(send['avg'])}/s  calls {send['calls']}")
            if not recv and not send: add(("      no census yet (rates start after two samples of this node's bridge)", "dim"))
            for port in ports:
                total = (bridge.get("census") or {}).get(str(port.get("port")))
                if total: add((f"      port {port.get('port')}: {total['calls']} programs, {rate_text(total['bytes']).replace('/s', '')} landed, {bridge.get('errors', 0)} errors, unretired SEND frames {total['frames'] - total['retired']}", "dim"))
        add()
        add(("CABLES", "head"), ("   this node's Thunderbolt IP ports (ioreg, ifconfig, ndp)", "dim"))
        cabled = [p for p in fabric(own).get("ports") or [] if p.get("iface")]
        if not any(p.get("neighbors") for p in cabled): add(("  no cabled fabric neighbour seen", "warn"))
        for p in cabled:
            power = {2: "on", 0: "off"}.get(p.get("rdma_power"), p.get("rdma_power"))
            add(("  ", ""), (f"{own.name}:{p['iface']}", "bold"), "   TB ", (str(p.get("tb_link")), "ok" if p.get("tb_link") == "up" else "warn"), (f"  link-ups {p.get('link_ups', '?')}", "dim"),
                "   IP ", (str(p.get("ip")), "ok" if p.get("ip") == "active" else "warn"), "  nbr ", ",".join(n["state"] for n in p.get("neighbors") or []) or "none",
                "   RDMA dev ", (str(power), "ok" if power == "on" else "warn"), (f"   nominal {p['speed'] / 1e9:g} Gb/s (IOLinkSpeed)" if p.get("speed") else "", "dim"))
        add()
        add(("TOPOLOGY", "head"))
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
        screen.timeout(max(0, int((due - time.monotonic()) * 1000)))
        key = screen.getch()
        if key in (ord("q"), 27): return
        if key == ord("j"): offset, due = min(offset + 1, max(0, len(lines) - 2)), 0
        if key == ord("k"): offset, due = max(0, offset - 1), 0

def main():
    parser = argparse.ArgumentParser(prog="mesh-top", usage=USAGE, description="Live connectivity and observed RDMA bandwidth from this node: the nodes this node's bridge's map names, with the facts their records carry (model, memory, window, available memory, boot), and this node's io.mesh.telemetry (:8788): its services, its cables, its bridge's ports and the bytes landed through each; sends no RDMA traffic of its own.")
    parser.add_argument("-n", "--interval", type=float, default=1.0)
    parser.add_argument("--window", type=float, default=5.0)
    parser.add_argument("--nodes", type=lambda text: [x for x in text.split(",") if x], default=None)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--html")
    args = parser.parse_args()
    top = Top(args)
    if args.once or args.html:
        top.refresh(True)
        for _ in range(max(2, int(args.window / args.interval) + 1)):
            time.sleep(args.interval)
            top.refresh(True)
        lines = top.frame()
        if args.html: mesh_disk.write(args.html, page(lines), f"mesh-top's page {args.html}")
        if args.once or not args.html: print(ansi(lines, sys.stdout.isatty()))
        return
    curses.wrapper(interactive, top)

if __name__ == "__main__":
    main()
