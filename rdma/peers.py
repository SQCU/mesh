from __future__ import annotations

import ipaddress
import hashlib
import json
import socket
import subprocess
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

BROWSE_SERVICE = "_meshnode._tcp"
NODEINFO_PORTS = (8100, 8099)

KIND_RANK = {"fabric-v4ll": 0, "fabric-routed": 1, "fabric-adjacent": 2,
             "lan": 3, "lan-v6": 4}

def classify(addr: str) -> str | None:
    scope = None
    if "%" in addr:
        addr, scope = addr.split("%", 1)

        if scope == "<0>":
            scope = None
        elif scope == "lo0":
            return None
    try:
        ip = ipaddress.ip_address(addr)
    except ValueError:
        return None
    if ip.is_loopback or ip.is_unspecified:
        return None
    if ip.version == 4:
        if ip.is_link_local:
            return "fabric-v4ll"
        if ip.is_private:
            return "lan"
        return None
    if ip.is_link_local:
        return "fabric-adjacent" if scope else None
    if addr.lower().startswith("fd6d:6573:68:"):
        return "fabric-routed"
    if ip.is_global:
        return "lan-v6"
    return None

@dataclass
class Node:
    name: str
    is_self: bool = False
    paths: list[tuple[str, str]] = field(default_factory=list)

    def ranked(self) -> list[tuple[str, str]]:
        return sorted(self.paths, key=lambda p: KIND_RANK.get(p[0], 99))

def _dns_sd(args: list[str], timeout: float) -> list[str]:
    try:
        p = subprocess.run(["dns-sd", *args], capture_output=True, text=True,
                           timeout=timeout)
        return p.stdout.splitlines()
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        output = getattr(e, "stdout", None) or ""
        return (output.decode(errors="replace") if isinstance(output, bytes) else output).splitlines()

def browse(timeout: float = 3.0) -> list[str]:
    names = []
    for line in _dns_sd(["-t", str(int(timeout)), "-B", BROWSE_SERVICE, "local"], timeout + 1):
        f = line.split()
        if len(f) >= 7 and f[1] == "Add":
            names.append(f[-1])
    return sorted(set(names))

def local_name() -> str:
    try:
        return subprocess.run(["scutil", "--get", "LocalHostName"],
                              capture_output=True, text=True, timeout=3).stdout.strip()
    except Exception:
        return socket.gethostname().split(".")[0]

def addresses(name: str, timeout: float = 3.0) -> list[tuple[str, str]]:
    out = []
    for line in _dns_sd(["-t", str(int(timeout)), "-G", "v4v6", f"{name}.local"], timeout + 1):
        f = line.split()
        if len(f) >= 6 and f[1] == "Add":
            kind = classify(f[5])
            if kind:
                out.append((kind, f[5]))
    return sorted(set(out), key=lambda p: KIND_RANK.get(p[0], 99))

def discover(timeout: float = 3.0) -> list[Node]:
    me = local_name()
    return [Node(n, n == me, addresses(n, timeout)) for n in browse(timeout)]

def nodeinfo(address: str, timeout: float = 2.0, limit: int = 1 << 20) -> str:
    errors = []
    for port in NODEINFO_PORTS:
        try:
            with socket.create_connection((address, port), timeout=timeout) as client:
                client.shutdown(socket.SHUT_WR)
                deadline, data = time.monotonic() + timeout, bytearray()
                while len(data) <= limit:
                    left = deadline - time.monotonic()
                    if left <= 0:
                        raise TimeoutError(address)
                    client.settimeout(left)
                    chunk = client.recv(min(65536, limit + 1 - len(data)))
                    if not chunk:
                        break
                    data.extend(chunk)
                if len(data) > limit:
                    raise ValueError("nodeinfo exceeds local byte limit")
                header, separator, body = bytes(data).partition(b"\n")
                if header.startswith(b"mesh1 "):
                    if not separator or len(body) != int(header.split()[1]):
                        raise ValueError("incomplete nodeinfo frame")
                else:
                    body = bytes(data)
                if not body.rstrip().endswith(b"end"):
                    raise ValueError("incomplete nodeinfo sample")
                return body.decode()
        except (OSError, ValueError) as error:
            errors.append(f"{address}:{port}: {error}")
    raise OSError("; ".join(errors))

def observation(body: str) -> dict:
    facts, ports, routes = {}, [], []
    for line in body.splitlines():
        fields = dict(word.split("=", 1) for word in line.split() if "=" in word)
        if "port" in fields:
            fields["address"] = fields.get("address", fields.get("gid", "none")).split("%", 1)[0].lower()
            fields["peer"] = fields.get("peer", "none").split("%", 1)[0].lower()
            fields["active"] = fields.get("link", fields.get("state")) in ("active", "PORT_ACTIVE")
            for key in ("address", "peer"):
                if fields[key] != "none":
                    fields[key] = str(ipaddress.IPv6Address(fields[key]))
            ports.append(fields)
        elif "route" in fields:
            address = fields["route"].split("/", 1)[0].split("%", 1)[0]
            if classify(address) == "fabric-routed":
                routes.append(address)
        else:
            facts.update(fields)
    return dict(facts=facts, ports=ports, routes=sorted(set(routes)))

def observe_mesh(seeds: list[str] | None = None, timeout: float = 2.0, max_nodes: int = 256) -> dict:
    pending = set(seeds or ["::1"])
    if seeds is None:
        pending.update(address for node in discover(timeout) for _, address in node.ranked())
    visited, observations, errors = set(), {}, {}
    def fetch(address):
        try:
            return address, observation(nodeinfo(address, timeout)), None
        except (OSError, ValueError) as error:
            return address, None, str(error)
    with ThreadPoolExecutor(max_workers=8) as pool:
        while pending and len(visited) < max_nodes:
            wave = sorted(pending - visited)[:max_nodes - len(visited)]
            if not wave:
                break
            visited.update(wave)
            for address, result, error in pool.map(fetch, wave):
                if error:
                    errors[address] = error
                else:
                    observations[address] = result
                    pending.update(result["routes"])
                    if address == "::1":
                        pending.update(f"{p['peer']}%{p['port']}" for p in result["ports"] if p["active"] and p["peer"] != "none")
            pending.difference_update(visited)
    return dict(observations=observations, errors=errors, unvisited=sorted(pending))

def topology(observations: list[dict], max_age: float = 10.0) -> dict:
    nodes, endpoints, edges, errors = {}, {}, set(), []
    for item in sorted(observations, key=lambda item: float(item['facts'].get('sample_age', 0)), reverse=True):
        f = item["facts"]
        identity = f.get("node_id") or f.get("ula") or f.get("name") or f.get("host")
        if not identity:
            errors.append({"error": "observation has no node identity", "observation": item})
            continue
        if not f.get("boot_id"):
            errors.append({"node": identity, "error": "boot identity unavailable"})
        if f.get("sample_status", "complete") != "complete":
            errors.append({"node": identity, "error": "incomplete observation"})
        if float(f.get("sample_age", 0)) > max_age:
            errors.append({"node": identity, "error": "stale observation", "sample_age": f["sample_age"]})
        node = dict(identity=identity, boot=f.get("boot_id"), ports=sorted(
            (dict(port=p["port"], address=p["address"], peer=p["peer"], active=p["active"])
             for p in item["ports"]), key=lambda p: p["port"]))
        node["capabilities"] = {key: f[key] for key in ("model", "cores", "memgb", "macos", "sdk", "rdma") if key in f}
        if identity in nodes and nodes[identity] != node:
            errors.append({"node": identity, "error": "conflicting observations; using the freshest"})
        nodes[identity] = node
    for identity, node in nodes.items():
        for p in node["ports"]:
            if p["active"] and p["address"] != "none":
                if p["address"] in endpoints:
                    errors.append({"address": p["address"], "error": "ambiguous physical endpoint identity"})
                    endpoints[p["address"]] = None
                else:
                    endpoints[p["address"]] = (identity, p)
    unresolved = []
    for address, endpoint in endpoints.items():
        if endpoint is None:
            continue
        identity, port = endpoint
        other = endpoints.get(port["peer"])
        if not other or other[1]["peer"] != address or other[0] == identity:
            unresolved.append([identity, port["port"], port["peer"]])
            continue
        edge = tuple(sorted(((identity, port["port"]), (other[0], other[1]["port"]))))
        edges.add(edge)
    return dict(nodes=sorted(nodes.values(), key=lambda n: n["identity"]), edges=sorted(edges), unresolved=sorted(unresolved), errors=errors)

def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def realize(graph: dict, transfers: list[dict], numerical_identity: str) -> dict:
    identities = [n["identity"] for n in graph["nodes"]]
    adjacency = {identity: [] for identity in identities}
    for a, b in graph["edges"]:
        adjacency[a[0]].append((b[0], a[1]))
        adjacency[b[0]].append((a[0], b[1]))
    routes = []
    for source in identities:
        reached, queue = {source}, deque([(source, None, 0)])
        while queue:
            at, first, depth = queue.popleft()
            for target, port in sorted(adjacency[at]):
                if target in reached:
                    continue
                reached.add(target)
                hop = first or port
                routes.append(dict(source=source, target=target, port=hop, hops=depth + 1))
                queue.append((target, hop, depth + 1))
    paths = {(r["source"], r["target"]) for r in routes}
    seen = set()
    for transfer in transfers:
        source, target = transfer["source"], transfer["target"]
        if source not in adjacency or target not in adjacency or (source != target and (source, target) not in paths):
            raise ValueError("transfer has no observed physical route")
        key = (source, target, transfer["channel"])
        if key in seen or transfer["bytes"] <= 0 or transfer["offset"] < 0:
            raise ValueError("ambiguous transfer identity or extent")
        seen.add(key)
    topology_id = fingerprint(graph)
    program = dict(topology=topology_id, route_policy="local-detour-v1", numerical_identity=numerical_identity,
                   transfers=sorted(transfers, key=lambda t: (t["source"], t["target"], t["channel"])))
    return dict(topology=graph, routes=routes, program=program, topology_id=topology_id, program_id=fingerprint(program))

def bridge_arguments(realized: dict) -> dict:
    graph = realized["topology"]
    identities = {node["identity"]: i for i, node in enumerate(graph["nodes"])}
    endpoints = {(n["identity"], p["port"]): p for n in graph["nodes"] for p in n["ports"]}
    commands = {}
    for identity, number in identities.items():
        links = sorted((tuple(a), tuple(b)) if a[0] == identity else (tuple(b), tuple(a))
                       for a, b in graph["edges"] if identity in (a[0], b[0]))
        ports = {a[1]: i for i, (a, b) in enumerate(links)}
        arguments = ["-I", str(number), "--hop-limit", str(min(65535, len(identities) + 1))]
        for a, b in links:
            local, remote = endpoints[a], endpoints[b]
            arguments += ["--link", f"rdma_{a[1]},{identities[b[0]]},{local['address']}%{a[1]},{remote['address']}%{a[1]}"]
        for route in realized["routes"]:
            if route["source"] == identity:
                arguments += ["--route", f"{identities[route['target']]}:{ports[route['port']]}"]
                adjacency = {n: [] for n in identities if n != identity}
                for a, b in graph["edges"]:
                    if identity not in (a[0], b[0]):
                        adjacency[a[0]].append(b[0]); adjacency[b[0]].append(a[0])
                reachable, queue = {route["target"]}, deque([route["target"]])
                while queue:
                    for neighbor in adjacency[queue.popleft()]:
                        if neighbor not in reachable:
                            reachable.add(neighbor); queue.append(neighbor)
                for a, b in links:
                    if a[1] != route["port"] and b[0] in reachable:
                        arguments += ["--route", f"{identities[route['target']]}:{ports[a[1]]}"]
        commands[identity] = dict(node=number, arguments=arguments)
    return commands

def reachable(addr: str, port: int = 22, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((addr, port), timeout=timeout):
            return True
    except OSError:
        return False

def resolve(pattern: str, port: int = 22, timeout: float = 3.0) -> str | None:
    pat = pattern.lower().replace("mesh-", "")
    for node in discover(timeout):
        if node.is_self or pat not in node.name.lower().replace("-", ""):
            continue
        for _kind, addr in node.ranked():
            if reachable(addr, port, timeout=2.0):
                return addr
    return None

def ssh_argv(pattern: str, *command: str, identity: str | None = None,
             timeout: float = 3.0) -> list[str] | None:
    node = None
    pat = pattern.lower().replace("mesh-", "")
    for n in discover(timeout):
        if not n.is_self and pat in n.name.lower().replace("-", ""):
            node = n
            break
    if node is None:
        return None
    for _kind, addr in node.ranked():
        if reachable(addr, 22, timeout=2.0):

            argv = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=6",
                    "-o", f"HostKeyAlias={node.name}",
                    "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null"]
            if identity:
                argv += ["-i", identity, "-o", "IdentitiesOnly=yes"]
            return argv + [addr, *command]
    return None

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] in ("observe", "compile"):
        if sys.argv[1] == "observe":
            print(json.dumps(observe_mesh(sys.argv[2:] or None), indent=2))
        else:
            from pathlib import Path
            specification = json.loads(Path(sys.argv[2]).read_text())
            observed = specification["observations"]
            graph = topology(list(observed.values()) if isinstance(observed, dict) else observed)
            compiled = realize(graph, specification["transfers"], specification["numerical_identity"])
            compiled["bridges"] = bridge_arguments(compiled)
            print(json.dumps(compiled, indent=2))
        sys.exit(0)
    if len(sys.argv) > 1:
        a = resolve(sys.argv[1])
        print(a if a else "", end="\n" if a else "")
        sys.exit(0 if a else 1)
    for n in discover():
        print(f"node {n.name} {'self' if n.is_self else 'peer'}")
        for kind, addr in n.ranked():
            print(f"  {kind:16} {addr}")
