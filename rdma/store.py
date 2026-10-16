"""The mesh's store of what it measures about its nodes and calls (design/heterogeneity.md), kept across calls in one
JSON file on the node that drives them (~/.local/share/mesh/store.json: a driver uses its own node's store): 'nodes',
each node's rates (RATES) with the model (node-info's) and GPU cores they were measured on, keyed by its name
(node-info's `name`); 'configurations', each call configuration's state (allocate.py's Balancer and the evidence it
moves on).  Every change is one read, change and write of the whole file under an exclusive lock on the file beside it
(`changing`), so drivers and tools on one node never lose each other's records.  Its first records are the mesh's own
(etc/mesh-store.json, installed beside the library).  A node with no record starts from the records of its model's
nodes, else of every node.  A link's cost is the bridges' map's (mesh-plan.h mesh_map), never a record here."""
import contextlib
import fcntl
import json
import os
import statistics
import tempfile
from pathlib import Path

PATH = Path.home() / '.local/share/mesh/store.json'
FIRST = Path(__file__).resolve().parent.parent / 'etc/mesh-store.json'
RATES = ('proj', 'attn', 'prim', 'edge')


def load(path=None):
    """The JSON at `path`, the store's (PATH, read when called) without one; {} where it is missing or unreadable."""
    try:
        return json.loads(Path(path or PATH).read_text())
    except (OSError, ValueError):
        return {}


@contextlib.contextmanager
def changing():
    """The store's state, read under an exclusive lock and written whole when the block ends without an exception."""
    PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PATH.with_suffix('.lock'), 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = load()
        yield state
        with tempfile.NamedTemporaryFile('w', dir=PATH.parent, delete=False) as file:
            json.dump(state, file)
        os.replace(file.name, PATH)


def records(section):
    """A section's records: the first records, under this node's."""
    return {**load(FIRST).get(section, {}), **load().get(section, {})}


def record(section, key, value):
    """A record of `section` ('nodes' or 'configurations') written."""
    with changing() as state:
        state.setdefault(section, {})[key] = value


def node(name, model=None):
    """Node `name`'s record, else its start: its model (`model`, node-info's) and each rate the mean of the records of
    that model's nodes, else of every node's."""
    known = records('nodes')
    if name in known:
        return known[name]
    alike = [r for r in known.values() if r.get('model') == model] or list(known.values())
    start = {'model': model}
    for key in RATES:
        if alike and all(key in r for r in alike):
            start[key] = [statistics.mean(values) for values in zip(*(r[key] for r in alike))]
    return start

