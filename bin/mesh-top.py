#!/usr/bin/env mesh-python
import argparse, collections, concurrent.futures, curses, html, json, os, re, socket, subprocess, sys, threading, time

HERE = os.path.dirname(os.path.realpath(__file__))
TELEMETRY_PORT = int(os.environ.get("MESH_TELEMETRY_PORT", "8788"))
RESIDENT = ("beacon", "router", "nodeinfo", "telemetry", "observer")
SPARK = " ▁▂▃▄▅▆▇█"
PHASES = {0: "UNKNOWN", 1: "PAIRING", 2: "PAIRED", 3: "STOPPED"}
USAGE = "mesh-top [-n SECONDS] [--window SECONDS] [--links MAP] [--nodes A,B] [--probe] [--once | --html FILE]   keys: q quit, p probe idle RDMA ports once (ibv_devinfo), j/k scroll"

def first_file(paths):
    return next((path for path in paths if path and os.path.isfile(os.path.expanduser(path))), None)

def local_name():
    try: return subprocess.run(["/usr/sbin/scutil", "--get", "LocalHostName"], capture_output=True, text=True, timeout=2).stdout.strip() or socket.gethostname().split(".")[0]
    except Exception: return socket.gethostname().split(".")[0]

def link_map(path):
    result = {"path": path, "kind": None, "count": 0, "links": [], "costs": {}, "nodes": {}}
    if not path: return result
    for line in open(os.path.expanduser(path)).read().splitlines():
        field = line.split()
        if not field or field[0].startswith("#"): continue
        if field[0] == "node" and len(field) >= 5: result["nodes"][int(field[1])] = {"host": field[2], "beacon": field[4]}
        elif field[0] in ("mesh", "ring", "tree", "star") and result["kind"] is None: result["kind"], result["count"] = field[0], int(field[1])
        elif len(field) in (2, 4) and field[0].isdigit() and field[1].isdigit():
            pair = tuple(sorted((int(field[0]), int(field[1]))))
            if pair not in result["links"]: result["links"].append(pair)
            if len(field) == 4: result["costs"][pair] = (float(field[2]), float(field[3]))
    return result

def roster(path, links, only):
    nodes = {}
    if path:
        for name, node in json.load(open(path)).get("nodes", {}).items(): nodes[name] = {"name": name, "via": node.get("via"), "aliases": list(node.get("aliases") or ())}
    for rank, node in links["nodes"].items():
        entry = nodes.setdefault(node["beacon"], {"name": node["beacon"], "via": None, "aliases": []})
        entry["rank"] = rank
        if node["host"] != "local" and node["host"] not in entry["aliases"]: entry["aliases"].append(node["host"])
    me = local_name()
    nodes.setdefault(me, {"name": me, "via": None, "aliases": []})["self"] = True
    if only: nodes = {name: node for name, node in nodes.items() if name in only or set(node["aliases"]) & set(only)}
    return nodes

class Node:
    def __init__(self, spec):
        self.spec, self.name, self.rank = spec, spec["name"], spec.get("rank")
        self.sample, self.info, self.info_at, self.error, self.transport, self.rtt = None, {}, 0.0, None, None, None
        self.sequence, self.previous, self.probe, self.probing, self.retry_at, self.address = None, None, None, False, 0.0, None

    def hosts(self):
        if self.spec.get("self"): return ["127.0.0.1"]
        return list(dict.fromkeys([h for h in (self.spec.get("via"), self.name + ".local") if h]))

    def ssh_hosts(self):
        return list(dict.fromkeys(self.spec["aliases"] + [self.name + ".local"]))

def bracket(host):
    return f"[{host}]" if ":" in host else host

def ssh(host, command, timeout):
    return subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=3", host, command], capture_output=True, text=True, timeout=timeout)

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

def nodeinfo_text(address):
    for port in (8099, 8100):
        try: data = exchange(address, port, b"x")
        except Exception: continue
        if data.startswith(b"mesh1 "): return data.split(b"\n", 1)[1].decode(errors="replace")
    return ""

def parse_info(text):
    info = {"ports": [], "routes": []}
    for line in text.splitlines():
        fields = dict(part.split("=", 1) for part in line.split() if "=" in part)
        if line.startswith("port="): info["ports"].append(fields)
        elif line.startswith("route="): info["routes"].append(fields)
        else: info.update(fields)
    return info

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
        envelope = None
        for host in node.ssh_hosts():
            try:
                answer = ssh(host, f"curl -s -m 2 'http://127.0.0.1:{TELEMETRY_PORT}{path}'", 8)
                envelope = json.loads(answer.stdout)
                node.transport = ("ssh", host)
                break
            except Exception as error: attempts.append(f"ssh {host}: {type(error).__name__}")
    node.rtt = time.monotonic() - started
    record = (envelope or {}).get("record") or {}
    if not record.get("sample"):
        node.error, node.transport, node.retry_at = "; ".join(attempts) or f"no address or ssh host answered for {', '.join(node.hosts() + node.ssh_hosts()) or node.name}", None, time.monotonic() + 10
        return
    node.error = None
    if record.get("sequence") != node.sequence:
        node.previous, node.sample, node.sequence = node.sample, record["sample"], record.get("sequence")
    if time.time() - node.info_at > 10:
        text = nodeinfo_text(node.address) if node.transport[0] == "http" else ssh(node.transport[1], "printf x | nc -G 2 -w 2 ::1 8099 || printf x | nc -G 2 -w 2 ::1 8100", 8).stdout.split("\n", 1)[-1]
        if text: node.info, node.info_at = parse_info(text), time.time()

def probe(node):
    command = "pgrep -x ibv_devinfo >/dev/null && echo MESH_TOP_BUSY || /usr/bin/ibv_devinfo 2>&1"
    try:
        text = subprocess.run(["/bin/sh", "-c", command], capture_output=True, text=True, timeout=10).stdout if node.spec.get("self") else ssh(node.ssh_hosts()[0], command, 15).stdout
        states = {m.group(1): (re.search(r"state:\s*(PORT_\w+)", block) or [None, "?"])[1] for block in text.split("hca_id:")[1:] for m in [re.match(r"\s*(\S+)", block)] if m}
        node.probe = {"at": time.time(), "states": states, "note": "an earlier ibv_devinfo is still running there" if "MESH_TOP_BUSY" in text else None}
    except Exception as error:
        node.probe = {"at": time.time(), "states": {}, "note": f"probe did not return: {type(error).__name__}; not retried while it runs"}
    node.probing = False

def fabric(node):
    return ((node.sample or {}).get("fabric") or {}) if isinstance((node.sample or {}).get("fabric"), dict) else {}

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
        self.args = args
        self.links = link_map(args.links)
        self.nodes = [Node(spec) for spec in sorted(roster(args.roster, self.links, args.nodes).values(), key=lambda s: (s.get("rank") is None, s.get("rank") or 0, s["name"]))]
        self.series = collections.defaultdict(lambda: collections.deque(maxlen=240))
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=max(2, 2 * len(self.nodes)))

    def refresh(self):
        list(self.pool.map(fetch, self.nodes))
        for node in self.nodes: self.account(node)

    def owner(self, address):
        address = (address or "").split("%")[0]
        for node in self.nodes:
            for port in fabric(node).get("ports") or []:
                if port.get("address") and port["address"] == address: return node.name, port["iface"]
            for port in node.info.get("ports", []):
                if port.get("address", "").split("%")[0] == address: return node.name, port.get("port")
        return None, None

    def rank_of(self, name):
        return next((n.rank for n in self.nodes if n.name == name and n.rank is not None), 1 << 30)

    def peer_by_rank(self, rank):
        return (self.links["nodes"].get(int(rank)) or {}).get("beacon") if str(rank).isdigit() else None

    def account(self, node):
        sample, previous = node.sample, node.previous
        if not sample or sample is getattr(node, "accounted", None): return
        node.accounted = sample
        if not previous or not isinstance(previous.get("fabric"), dict) or not isinstance(sample.get("fabric"), dict): return
        dt = float(sample.get("observed_at", 0)) - float(previous.get("observed_at", 0))
        if dt <= 0: return
        before = {bridge["pid"]: bridge for bridge in previous["fabric"].get("bridges") or []}
        for bridge in sample["fabric"].get("bridges") or []:
            old = before.get(bridge["pid"])
            specs = bridge.get("links") or []
            for index, (flow, was) in enumerate(zip(bridge.get("flow") or [], (old or {}).get("flow") or [])):
                spec = specs[index] if index < len(specs) else {}
                peer = self.owner(spec.get("remote"))[0] or self.peer_by_rank(spec.get("peer")) or f"rank{spec.get('peer', '?')}"
                delta = lambda *keys: max(0, sum(flow.get(k, 0) - was.get(k, 0) for k in keys))
                self.series[(peer, node.name, "live")].append((float(sample["observed_at"]), dt, delta("receive_bytes", "net_receive_bytes"), delta("receives", "net_receives"), 0))
            if any((x.get("log_size") or 0) - x.get("log_offset", 0) > 65536 for x in (old or {}, bridge)): continue
            for index, total in (bridge.get("census") or {}).items():
                base = ((old or {}).get("census") or {}).get(index) or {}
                delta = {key: max(0, value - base.get(key, 0)) for key, value in total.items()}
                specs = bridge.get("links") or []
                spec = specs[int(index)] if int(index) < len(specs) else {}
                peer = self.owner(spec.get("remote"))[0] or self.peer_by_rank(spec.get("peer")) or f"rank{spec.get('peer', '?')}"
                end = float(sample["observed_at"])
                self.series[(peer, node.name, "recv")].append((end, dt, delta.get("bytes", 0), delta.get("landed", 0), delta.get("calls", 0)))
                self.series[(node.name, peer, "send")].append((end, dt, delta.get("frames", 0) * 4096, delta.get("send_completions", 0), delta.get("calls", 0)))

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
        for node in self.nodes:
            for port in fabric(node).get("ports") or []:
                for neighbor in port.get("neighbors") or []:
                    peer, peer_iface = self.owner(neighbor["address"])
                    if peer and peer != node.name:
                        key = frozenset(((node.name, port["iface"]), (peer, peer_iface)))
                        found.setdefault(key, {})[node.name] = neighbor.get("state")
            if not fabric(node).get("ports"):
                for port in node.info.get("ports", []):
                    peer, peer_iface = self.owner(port.get("peer"))
                    if peer and peer != node.name: found.setdefault(frozenset(((node.name, port.get("port")), (peer, peer_iface))), {})[node.name] = "ndp"
        for node in self.nodes:
            for bridge in fabric(node).get("bridges") or []:
                for spec in bridge.get("links") or []:
                    peer, peer_iface = self.owner(spec.get("remote"))
                    if peer: found.setdefault(frozenset(((node.name, spec["device"].replace("rdma_", "")), (peer, peer_iface))), {})
        return found

    def port(self, name, iface):
        node = next((n for n in self.nodes if n.name == name), None)
        return node, next((p for p in fabric(node).get("ports") or [] if p.get("iface") == iface), None) if node else None

    def bridge_link(self, node, iface):
        for bridge in sorted(fabric(node).get("bridges") or [], key=lambda b: (b.get("exited_at") is not None, -(b.get("exited_at") or 0))):
            for index, spec in enumerate(bridge.get("links") or []):
                if spec.get("device") == "rdma_" + iface:
                    peers = (bridge.get("stat") or {}).get("peers") or []
                    return bridge, (peers[index] if index < len(peers) else None)
        return None, None

    def frame(self):
        W = self.args.window
        lines, hints = [], []
        now = time.time()
        add = lambda *segments: lines.append([s if isinstance(s, tuple) else (s, "") for s in segments])
        add((f"mesh top  {time.strftime('%Y-%m-%d %H:%M:%S')}", "head"), (f"   every {self.args.interval:g}s, rates over {W:g}s   ", "dim"),
            (f"link map {self.links['path'] or 'none'}" + (f" ({self.links['kind']} {self.links['count']})" if self.links["kind"] else ""), "dim"))
        add()
        add(("NODES", "head"))
        add(("  rank node              chip           cores  mem    up        bridge                               telemetry", "dim"))
        for node in self.nodes:
            s, f, info = node.sample or {}, fabric(node), node.info
            chip = ((s.get("capacity") or {}).get("metrics") or {}).get("chip") or (s.get("host") or {}).get("model") or info.get("model") or "?"
            mem = (s.get("host") or {}).get("memory_bytes")
            up = duration(now - f["booted_at"]) if f.get("booted_at") else info.get("uptime", "?")
            live = [b for b in f.get("bridges") or [] if not b.get("exited_at")]
            if live:
                b = live[0]
                stat = b.get("stat") or {}
                phases = "/".join(PHASES.get(p.get("phase"), "?") for p in stat.get("peers") or []) or ("up" if stat.get("up") else "no region")
                bridge = (f"{b['region']} pid {b['pid']} {phases}" + (f" call {stat['client']}" if stat.get("client") else " idle") + (f" +{len(live) - 1}" if len(live) > 1 else ""), "ok" if "PAIRED" in phases else "warn")
            elif f.get("bridges"):
                b = max(f["bridges"], key=lambda x: x["exited_at"])
                bridge = (f"none (last {b['region']} exited {duration(now - b['exited_at'])} ago)", "dim")
            else: bridge = ("none", "dim")
            transport = f"{node.transport[0]} {node.transport[1]} {node.rtt * 1000:.0f}ms" if node.transport else "unreachable"
            add(f"  {node.rank if node.rank is not None else '-':<4} ", (f"{node.name:<17} ", "bold"), f"{chip:<14} {info.get('cores', '?'):>5}  {(str(round(mem / 2**30)) + 'G') if mem else info.get('memgb', '?') + 'G':<6} {up:<9} ",
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
            if (node.probe or {}).get("note"): hints.append(("warn", f"{node.name}: RDMA port probe {node.probe['note']}"))
            if node.sample and not isinstance(node.sample.get("fabric"), dict): hints.append(("warn", f"{node.name}: its io.mesh.telemetry predates the fabric section; links show nodeinfo only until it converges"))
            if info.get("planes") and info.get("planes") != "2": hints.append(("warn", f"{node.name}: DEGRADED, {info['planes']} of 2 planes (lan={info.get('lan')} fabric={info.get('fabric')})"))
            for alarm in keeper.get("alarms") or []:
                if now - (keeper.get("modified") or 0) < 3600: hints.append(("warn", f"{node.name}: keeper {alarm}"))
            for b in f.get("bridges") or []:
                if str(b.get("state", "")).startswith("U"): hints.append(("bad", f"{node.name}: bridge pid {b['pid']} is in U (uninterruptible, blocked in a verbs call); never SIGKILL it and do not launch ibv_devinfo there (RDMA-RULES.md)"))
                if b.get("log_error") and not b.get("exited_at"): hints.append(("warn", f"{node.name}: bridge {b['region']} log unreadable by telemetry ({b['log_error']}); no census"))
                if b.get("errors") and not b.get("exited_at"): hints.append(("warn", f"{node.name}: bridge {b['region']} logged {b['errors']} error line(s), last: {b.get('last_error')}"))
            for port in f.get("ports") or []:
                for holder in port.get("holders") or []:
                    if str(holder.get("stat") or "").startswith("U") and holder.get("command") != "mesh-flow":
                        hints.append(("bad", f"{node.name}: {holder['command']} pid {holder['pid']} in U holds {port['device']} (leaked verbs context)"))
        add()
        add(("LINKS", "head"), ("   a<->b by Thunderbolt IP adjacency (ndp) and live bridge links; bytes/s are RECV bytes landed, from the bridges' per-call census", "dim"))
        cables = self.cables()
        if not cables: add(("  no cabled fabric neighbour seen", "warn"))
        for key, states in sorted(cables.items(), key=lambda kv: sorted((self.rank_of(x), x) for x, _ in kv[0])):
            (a, ai), (b, bi) = sorted(key, key=lambda end: (self.rank_of(end[0]), end[0]))
            na, pa = self.port(a, ai)
            nb, pb = self.port(b, bi)
            ranks = tuple(sorted(n.rank for n in (na, nb) if n and n.rank is not None))
            def both(getter, good=None):
                values = [getter(p) if p else "?" for p in (pa, pb)]
                style = "ok" if good and all(v == good for v in values) else "warn" if good else ""
                return ("/".join(str(v) for v in values), style)
            add(("  ", ""), (f"{a}:{ai}", "bold"), " <-> ", (f"{b}:{bi}", "bold"), "   TB ", both(lambda p: p.get("tb_link"), "up"),
                ("  link-ups " + "/".join(str((p or {}).get("link_ups", "?")) for p in (pa, pb)), "dim"),
                "   IP ", both(lambda p: p.get("ip"), "active"), "  nbr ", both(lambda p: ",".join(n["state"] for n in p.get("neighbors") or []) or "none"),
                "   RDMA dev ", both(lambda p: {2: "on", 0: "off"}.get(p.get("rdma_power"), p.get("rdma_power")), "on"))
            segments = [("      RDMA port ", "dim")]
            for node, iface, p in ((na, ai, pa), (nb, bi, pb)):
                bridge, peer = self.bridge_link(node, iface) if node else (None, None)
                probed = ((node.probe or {}).get("states") or {}).get("rdma_" + iface) if node else None
                if bridge and not bridge.get("exited_at") and peer:
                    phase, client = PHASES.get(peer.get("phase"), "?"), (bridge.get("stat") or {}).get("client")
                    text = f"{phase}, port active" if phase == "PAIRED" else f"{phase} code {peer.get('code')} during call {client}" if client else "bridge idle between calls"
                    segments.append((f"{node.name}:rdma_{iface} {text} (bridge {bridge['region']})   ", "ok" if phase == "PAIRED" else "warn" if client else "dim"))
                elif probed:
                    segments.append((f"{node.name}:rdma_{iface} {probed} (probe {time.strftime('%H:%M:%S', time.localtime(node.probe['at']))})   ", "ok" if probed == "PORT_ACTIVE" else "bad"))
                else: segments.append((f"{node.name}:rdma_{iface} unmeasured (no bridge; p probes)   ", "dim"))
                ip_up = p and p.get("ip") == "active" and p.get("neighbors")
                if ip_up and probed and probed != "PORT_ACTIVE":
                    hints.append(("bad", f"{node.name} rdma_{iface} is {probed} while IP over {iface} is up: reseat the Thunderbolt cable (post-reboot / link-flap quirk; no sudo needed)"))
                elif ip_up and p.get("rdma_power") not in (2, None) and not probed:
                    hints.append(("warn", f"{node.name} rdma_{iface}: IP over {iface} is up but the RDMA device reports power state {p.get('rdma_power')}; likely PORT_DOWN (post-reboot quirk): press p to confirm, then reseat the cable"))
            add(*segments)
            declared = 1e9 / self.links["costs"][ranks][1] if ranks in self.links["costs"] else None
            for src, dst in ((a, b), (b, a)):
                recv, send = self.rates((src, dst, "recv"), W), self.rates((src, dst, "send"), W)
                spark = "".join(SPARK[min(8, int(8 * v / recv["peak"]))] if recv and recv["peak"] else " " for v in (recv or {}).get("spark", []))
                row = [("      ", ""), (f"{src[:14]}->{dst[:14]}", "accent"), "  "]
                if recv:
                    row += [(f"{rate_text(recv['now']):>11} last", "ok" if recv["now"] else ""), f"  {rate_text(recv['avg']):>11} {W:g}s" + (f" {100 * recv['avg'] / declared:3.0f}%" if declared else ""), f"  peak {rate_text(recv['peak']):>10}  ", (f"{spark:<32}", "accent"),
                            f"  RECV {count_text(recv['avg_ops'])}/s", f"  SEND {count_text((send or {}).get('avg_ops'))}/s", f"  calls {recv['calls']}"]
                else: row.append(("no census yet (rates start after two samples of a node running a bridge)", "dim"))
                live = self.rates((src, dst, "live"), W)
                if live: row += [("   live ", "dim"), (f"{rate_text(live['now'])}", "ok" if live["now"] else ""), f" ({count_text(live['now_ops'])} RECV/s, bridge counters)"]
                add(*row)
            reference = []
            if ranks in self.links["costs"]:
                alpha, beta = self.links["costs"][ranks]
                reference.append(f"declared a={alpha:g}us b={beta:g}ns/B = {rate_text(1e9 / beta)} ({os.path.basename(self.links['path'])})")
            speed = (pa or pb or {}).get("speed")
            if speed: reference.append(f"nominal {speed / 1e9:g} Gb/s = {rate_text(speed / 8)} (IOLinkSpeed)")
            for node, iface in ((na, ai), (nb, bi)):
                for bridge in fabric(node).get("bridges") or [] if node else []:
                    index = next((str(i) for i, spec in enumerate(bridge.get("links") or []) if spec.get("device") == "rdma_" + iface), None)
                    total, last = (bridge.get("census") or {}).get(index), (bridge.get("last_call") or {}).get(index) or {}
                    if last.get("span_ns"): reference.append(f"{node.name} last call (MESH_LEDGER): {last['bytes']} bytes landed over {last['span_ns'] / 1e3:.0f} us of crossings = {rate_text(last['bytes'] * 1e9 / last['span_ns'])}")
                    if total and not bridge.get("exited_at"):
                        reference.append(f"{node.name} bridge {bridge['region']}: {total['calls']} calls, {rate_text(total['bytes']).replace('/s', '')} landed, {bridge.get('pairs', 0)} pairings, {bridge.get('retries', 0)} pairing retries, {bridge.get('resets', 0)} pairing resets reconnected, {bridge.get('credit_limited', 0)} credit-limited calls, {bridge.get('errors', 0)} errors, unretired SEND frames {total['frames'] - total['retired']}")
                    flows, peers = bridge.get("flow") or [], (bridge.get("stat") or {}).get("peers") or []
                    if index is not None and int(index) < len(flows) and not bridge.get("exited_at"):
                        f, p = flows[int(index)], peers[int(index)] if int(index) < len(peers) else {}
                        session, regions = p.get("session") or {}, p.get("regions") or {}
                        reference.append(f"{node.name} bridge {bridge['region']} live: flow-control stalls send {f.get('send_stalls', 0)} receive {f.get('receive_stalls', 0)}, credit waits {f.get('credit_waits', 0)}; "
                                         f"program {f.get('sends', 0)} SENDs {rate_text(f.get('send_bytes', 0)).replace('/s', '')}, {f.get('receives', 0)} RECVs {rate_text(f.get('receive_bytes', 0)).replace('/s', '')}; "
                                         f"communicators {f.get('net_sends', 0)} sent {rate_text(f.get('net_send_bytes', 0)).replace('/s', '')}, {f.get('net_receives', 0)} received {rate_text(f.get('net_receive_bytes', 0)).replace('/s', '')}; "
                                         f"session {PHASES.get(session.get('phase'), '?')} ({session.get('sessions', 0)} pairings, {session.get('chunk_frames', 0)} frames a chunk); regions {regions.get('wire', 0)} window + {regions.get('client', 0)} client")
            for text in reference: add(("      " + text, "dim"))
        add()
        add(("COMMUNICATORS", "head"), ("   the bridges' ncclNet-shaped connections (mesh-net.h): one direction of one connection each, keyed by its listen", "dim"))
        shown = 0
        for node in self.nodes:
            for bridge in fabric(node).get("bridges") or []:
                comms, stat = bridge.get("communicators") or [], bridge.get("stat") or {}
                if bridge.get("exited_at") or not (comms or stat.get("clients")): continue
                shown += 1
                add(("  ", ""), (f"{node.name} {bridge['region']}", "bold"), (f"   {len(comms)} comms, {len(stat.get('clients') or [])} clients (pids {', '.join(map(str, stat.get('clients') or [])) or '-'})", "dim"))
                for c in comms:
                    style = "bad" if c.get("state") == "failed" else "ok" if c.get("state") in ("send", "recv") else "dim"
                    add(("    ", ""), (f"#{c['comm']:<3} {c.get('state', '?'):<10}", style),
                        f" {'->' if c.get('state') == 'send' else '<-' if c.get('state') == 'recv' else '  '} node {c.get('peer')}  pid {c.get('pid')}  key {c.get('key', '')[:8]}"
                        f"  posted {c.get('posted', 0)}  completed {c.get('completions', 0)}  {rate_text(c.get('bytes', 0)).replace('/s', '')}  credit waits {c.get('credit_waits', 0)}" + (f"  error {c['error']}" if c.get("error") else ""))
        if not shown: add(("  none open", "dim"))
        add()
        add(("TOPOLOGY", "head"))
        if self.links["kind"]: add(f"  declared  {self.links['kind']} {self.links['count']}: " + "  ".join(f"{x}-{y}" for x, y in self.links["links"]), (f"   ({self.links['path']})", "dim"))
        names = sorted({x for key in cables for x, _ in key})
        edges = {frozenset(x for x, _ in key) for key in cables}
        order = lambda edge: sorted(edge, key=lambda x: (self.rank_of(x), x))
        if names: add(f"  cabled    {shape(names, edges)}: " + "  ".join(" <-> ".join(order(e)) for e in edges), ("   (Thunderbolt IP adjacency, ndp)", "dim"))
        calls = collections.defaultdict(set)
        for node in self.nodes:
            for bridge in fabric(node).get("bridges") or []:
                if bridge.get("exited_at"): continue
                for spec in bridge.get("links") or []:
                    calls[bridge["region"]].add(frozenset((str(bridge.get("rank")), str(spec.get("peer")))))
        for region, call_edges in sorted(calls.items()):
            members = sorted({x for e in call_edges for x in e})
            label = lambda rank: f"{rank}:{self.peer_by_rank(rank) or '?'}"
            add(f"  live      {region}: {shape(members, call_edges)} over " + "  ".join(" <-> ".join(label(x) for x in sorted(e)) for e in call_edges), ("   (running bridges' --link arguments)", "dim"))
        if not calls: add(("  live      no bridge running", "dim"))
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
            if any(node.probing for node in top.nodes): lines.insert(1, [("  probing RDMA ports ...", "warn")])
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
        if key == ord("p"): start_probe(top); due = 0

def start_probe(top):
    for node in top.nodes:
        f = fabric(node)
        busy = [b for b in f.get("bridges") or [] if not b.get("exited_at")]
        wedged = [h for p in f.get("ports") or [] for h in p.get("holders") or [] if str(h.get("stat") or "").startswith("U")]
        if node.probing or busy or wedged or node.error:
            node.probe = {"at": time.time(), "states": (node.probe or {}).get("states", {}), "note": "not probed: " + ("a bridge runs there (its link phase is the port state)" if busy else "a verbs process is in U" if wedged else "unreachable" if node.error else "a probe is outstanding")}
            continue
        node.probing = True
        threading.Thread(target=probe, args=(node,), daemon=True).start()

def main():
    parser = argparse.ArgumentParser(prog="mesh-top", usage=USAGE, description="Live connectivity and observed RDMA bandwidth between mesh nodes, read from each node's io.mesh.telemetry (:8788) and io.mesh.nodeinfo; sends no traffic of its own.")
    parser.add_argument("-n", "--interval", type=float, default=1.0)
    parser.add_argument("--window", type=float, default=5.0)
    parser.add_argument("--links", default=first_file([os.environ.get("MESH_LINK_MAP"), "~/metal-microbench/configs/links/pair-ring.txt"]))
    parser.add_argument("--roster", default=first_file([os.environ.get("MESH_NODES"), os.path.join(HERE, "..", "etc", "mesh-nodes.json"), "~/.local/mesh/etc/mesh-nodes.json", "/usr/local/mesh/etc/mesh-nodes.json"]))
    parser.add_argument("--nodes", type=lambda text: [x for x in text.split(",") if x], default=None)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--html")
    args = parser.parse_args()
    if args.links: args.links = os.path.expanduser(args.links)
    if args.roster: args.roster = os.path.expanduser(args.roster)
    top = Top(args)
    if args.once or args.html:
        top.refresh()
        if args.probe:
            start_probe(top)
            deadline = time.monotonic() + 20
            while any(node.probing for node in top.nodes) and time.monotonic() < deadline: time.sleep(0.2)
        for _ in range(max(2, int(args.window / args.interval) + 1)):
            time.sleep(args.interval)
            top.refresh()
        lines = top.frame()
        if args.html:
            with open(args.html, "w") as stream: stream.write(page(lines))
        if args.once or not args.html: print(ansi(lines, sys.stdout.isatty()))
        return
    if args.probe: start_probe(top)
    curses.wrapper(interactive, top)

if __name__ == "__main__":
    main()
