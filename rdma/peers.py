"""A node's node-info sample (bin/mesh-nodeinfo.sh's lines, `end` last), read into its facts."""
from __future__ import annotations

import re
import uuid

def sample(data: bytes, limit: int) -> str:
    if len(data) > limit:
        raise ValueError("nodeinfo exceeds local byte limit")
    if not data.rstrip().endswith(b"end"):
        raise ValueError("incomplete nodeinfo sample")
    return data.decode()

def node_number(node_id: str) -> int:
    """The bridge node number of the host whose IOPlatformUUID is `node_id` (node-info's node_id): mesh-link.c main's
    FNV-1 of the host UUID's 16 bytes (gethostuuid), its low 31 bits, 1 for 0."""
    number = 0
    for byte in uuid.UUID(node_id).bytes:
        number = (number * 0x01000193 & 0xffffffff) ^ byte
    return number & 0x7fffffff or 1

def host() -> dict:
    """This machine's identity: its host UUID (IOPlatformUUID, the one gethostuuid gives the bridge), its bridge node
    number, and its serial number, read from the IORegistry's archive form (AGENTS.md: a node is its hardware)."""
    import plistlib, subprocess
    entry = plistlib.loads(subprocess.run(["/usr/sbin/ioreg", "-a", "-rd1", "-c", "IOPlatformExpertDevice"],
                                          capture_output=True, check=True, timeout=5).stdout)[0]
    return {"node_id": entry["IOPlatformUUID"], "node": node_number(entry["IOPlatformUUID"]),
            "serial": entry["IOPlatformSerialNumber"]}

def table(*paths: str) -> dict[int, dict]:
    """The name table (etc/mesh-nodes.json: entries keyed by host UUID), from the first of `paths` that reads, keyed by
    bridge node number, each entry with its node_id; {} where none reads. A name is a label for people, resolved from
    identity, never matched to find a node (AGENTS.md: names are virtual)."""
    import json
    for path in paths:
        try:
            with open(path) as f:
                entries = json.load(f).get("nodes") or {}
        except (OSError, ValueError, AttributeError):
            continue
        found = {}
        for node_id, entry in entries.items():
            try:
                found[node_number(node_id)] = dict(entry, node_id=node_id)
            except ValueError:
                continue
        return found
    return {}

def label(number, names: dict[int, dict]) -> str:
    """Node `number`'s name for people: the name table's, else `node <number>`."""
    return (names.get(number) or {}).get("name") or f"node {number}"

def identity(body: str) -> int | None:
    """The bridge node number a node-info sample's node_id names, None where it names none."""
    try:
        return node_number(observation(body)["node_id"])
    except (KeyError, ValueError):
        return None

FIELD = re.compile(r"(?:^|\s)([A-Za-z_][A-Za-z0-9_]*)=(.*?)(?=\s+[A-Za-z_][A-Za-z0-9_]*=|\s*$)")

def fields(line: str) -> dict:
    """A node-info line's fields: each key=value, a value running to the next key= or the line's end, so a value holds
    its spaces (chip=Apple M5 Max, instances=Apple M5 Max/40) and a line of several facts gives each (disk_free_gib=..
    disk_used_gib=.., route=.. via=..)."""
    return {key: value for key, value in FIELD.findall(line)}

def observation(body: str) -> dict:
    """A node-info sample's facts (fields), its port= and route= lines apart."""
    facts = {}
    for line in body.splitlines():
        found = fields(line)
        if not found.keys() & {"port", "route"}:
            facts.update(found)
    return facts
