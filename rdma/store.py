"""The mesh's store of what it measures about its calls (design/heterogeneity.md), kept across calls in one JSON file
on the node that drives them (~/.local/share/mesh/store.json: a driver uses its own node's store): 'configurations',
each call configuration's state (allocate.py's Balancer and the evidence it moves on), named by its members' node
numbers (metal-microbench tools/mesh/rates.py configuration).  Every change is one read, change and write of the whole
file under an exclusive lock on the file beside it (`changing`), so drivers and tools on one node never lose each
other's records.  A link's cost is the bridges' map's (mesh-plan.h mesh_map), never a record here."""
import contextlib
import fcntl
import json
import os
import tempfile
from pathlib import Path

PATH = Path.home() / '.local/share/mesh/store.json'


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
    """A section's records."""
    return load().get(section, {})
