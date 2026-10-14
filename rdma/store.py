"""The mesh's store of what it learns about its nodes, links and calls (design/heterogeneity.md): each node's rates and
each link's alpha-beta cost, keyed by node name (its beacon's, node-info's `name`), and the state of each call
configuration (allocate.py's Balancer and its evidence), kept across calls in one JSON file on the node that drives
them (~/.local/share/mesh/store.json), written whole.  Its first records are the mesh's own (etc/mesh-store.json,
installed beside the library).  A node with no record starts from the records of its model, else of every node; a link
with none, from its port's bandwidth (the bridges' record, peers.py) at the known links' mean alpha."""
import json
import os
import statistics
import tempfile
from pathlib import Path

PATH = Path.home() / '.local/share/mesh/store.json'
FIRST = Path(__file__).resolve().parent.parent / 'etc' / 'mesh-store.json'
RATES = ('proj', 'attn', 'prim', 'edge')


def load(path=PATH):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def save(state):
    PATH.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=PATH.parent, delete=False) as file:
        json.dump(state, file)
    os.replace(file.name, PATH)


def records(section):
    return {**load(FIRST).get(section, {}), **load().get(section, {})}


def node(name, model=None):
    """Node `name`'s record ({'model', 'cores', 'proj', 'attn', 'prim', 'edge'}), else its start: the mean of the
    records of its model, else of every node's."""
    known = records('nodes')
    if name in known:
        return known[name]
    alike = [r for r in known.values() if model and r.get('model') == model] or list(known.values())
    start = {'model': model, 'cores': round(statistics.mean(r['cores'] for r in alike))} if alike else {'model': model}
    for key in RATES:
        if alike and all(key in r for r in alike):
            start[key] = [statistics.mean(values) for values in zip(*(r[key] for r in alike))]
    return start


def link(a, b, bandwidth=0):
    """The link between nodes `a` and `b`: its (alpha us, beta ns a byte), else alpha the known links' mean and beta
    its port's bandwidth's (bits a second)."""
    known = records('links')
    key = ' '.join(sorted((a, b)))
    if key in known:
        return tuple(known[key])
    alpha = statistics.mean(c[0] for c in known.values()) if known else 5.0
    return (alpha, 8e9 / bandwidth if bandwidth else statistics.mean(c[1] for c in known.values()) if known else 0.1)


def record(section, key, value):
    """A record of `section` ('nodes', 'links' keyed 'a b', 'configurations') written."""
    state = load()
    state.setdefault(section, {})[key] = value
    save(state)
