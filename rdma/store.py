"""The mesh's store of what it measures about its nodes, links and calls (design/heterogeneity.md): each node's rates and
each link's alpha-beta cost, keyed by node name (node-info's `name`), and the state of each call configuration
(allocate.py's Balancer and its evidence), kept across calls in one JSON file on the node that drives them
(~/.local/share/mesh/store.json: a driver uses its own node's store).  Every change is one read, change and write of the
whole file under an exclusive lock on the file beside it (`changing`), so drivers and tools on one node never lose each
other's records.  Its first records are the mesh's own (etc/mesh-store.json, installed beside the library).  A node's
facts are not its records: its model is the roster's (etc/mesh-nodes.json) or node-info's, its GPU cores its model's
(etc/mesh-capacity.json).  A node with no record starts from the records of its model's nodes, else of every node."""
import contextlib
import fcntl
import json
import os
import statistics
import tempfile
from pathlib import Path

PATH = Path.home() / '.local/share/mesh/store.json'
ETC = Path(__file__).resolve().parent.parent / 'etc'
RATES = ('proj', 'attn', 'prim', 'edge')


def load(path=PATH):
    try:
        return json.loads(Path(path).read_text())
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
    return {**load(ETC / 'mesh-store.json').get(section, {}), **load().get(section, {})}


def facts(name, model=None):
    """Node `name`'s model (the roster's, else `model`) and its GPU cores (its model's)."""
    model = load(ETC / 'mesh-nodes.json').get('nodes', {}).get(name, {}).get('model') or model
    return {'model': model, 'cores': load(ETC / 'mesh-capacity.json').get('models', {}).get(model, {}).get('gpu_cores')}


def node(name, model=None):
    """Node `name`'s rates ({'proj', 'attn', 'prim', 'edge'}) and facts, else its start: the mean of the records of its
    model's nodes, else of every node's."""
    known, mine = records('nodes'), facts(name, model)
    if name in known:
        return {**known[name], **mine}
    alike = [r for n, r in known.items() if facts(n)['model'] == mine['model']] or list(known.values())
    start = dict(mine)
    for key in RATES:
        if alike and all(key in r for r in alike):
            start[key] = [statistics.mean(values) for values in zip(*(r[key] for r in alike))]
    return start


def link(a, b):
    """The link between nodes `a` and `b` as measured: its (alpha us, beta ns a byte), None where it is not (its cost is
    then the map's, mesh.join)."""
    known = records('links').get(' '.join(sorted((a, b))))
    return tuple(known) if known else None


def record(section, key, value):
    """A record of `section` ('nodes', 'links' keyed 'a b', 'configurations') written."""
    with changing() as state:
        state.setdefault(section, {})[key] = value
