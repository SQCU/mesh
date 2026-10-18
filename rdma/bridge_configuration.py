"""The bridge's configuration, one operand its consumer passes (design/prepared-machine.md#M09, etc/bridge/README.md): a
document in JSON, TOML or YAML, by its extension, read here into the one canonical form the bridge reads, a property
list of its name (the document's stem), its purpose, its window in bytes, whether it serves `exec` and, where it keeps
them, `ledger`.  A document states its `purpose` and its window, either `window_gib` (GiB) or `window_percent` (of the
node's physical memory, read where this runs: the node's own bin/mesh-user.sh), and may state `exec`, true where the
bridge runs its peers' commands as its user (design/bridge-and-ipc.md#exec; a document that does not state it true
serves none), and `ledger`, true where the bridge keeps each flow's ledger and writes it at the flow's end
(design/bridge-and-ipc.md#ledger; a document that does not state it true keeps none, and its canonical form names no
`ledger`), and its paced emissions (design/bridge-and-ipc.md#keepalive), each in microseconds and each none where it is
not stated: `keepalive_us`, the most an exclusive flow goes without a frame on its control queue pair while it runs;
`hello_us`, the interval at which a port with no peer says HELLO again (a HELLO lost on its link has no other event);
and `health_us`, the interval at which the node's record is flooded again (design/recovery.md#diagnosis) (canonical
`keepalive`, `hello` and `health`, nanoseconds), and nothing else; the window is at most the node's memory.  There is no
default.

    python bridge_configuration.py DOCUMENT CANONICAL"""
import json
import os
import plistlib
import sys
import tomllib
from pathlib import Path

import yaml

READERS = {'.json': json.loads, '.toml': lambda data: tomllib.loads(data.decode()), '.yaml': yaml.safe_load,
           '.yml': yaml.safe_load}
SIZES = {'window_gib': lambda value, memory: int(value * (1 << 30)),
         'window_percent': lambda value, memory: int(memory * value / 100)}
PACED = {'keepalive_us': 'keepalive', 'hello_us': 'hello', 'health_us': 'health'}


def memory():
    """The node's physical memory, in bytes."""
    return os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')


def load(path):
    """The document at `path` as the bridge reads it, {'name', 'purpose', 'window', 'exec'}, 'ledger' where it keeps
    ledgers and 'keepalive', 'hello' and 'health' (ns) where it states them; ValueError naming what it lacks."""
    path = Path(path)
    reader = READERS.get(path.suffix.lower())
    if reader is None:
        raise ValueError(f'{path}: a bridge configuration is a .json, .toml or .yaml document')
    document = reader(path.read_bytes())
    if not isinstance(document, dict):
        raise ValueError(f'{path}: a bridge configuration is a table of its purpose and its window')
    unknown = sorted(set(document) - {'purpose', 'exec', 'ledger', *PACED, *SIZES})
    if unknown:
        raise ValueError(f'{path}: {", ".join(unknown)}: no key of a bridge configuration')
    purpose = document.get('purpose')
    if not isinstance(purpose, str) or not purpose.strip():
        raise ValueError(f'{path}: purpose, a statement of what the configuration is for, is required')
    sizes = [key for key in SIZES if key in document]
    if len(sizes) != 1:
        raise ValueError(f'{path}: one window is required, window_gib (GiB) or window_percent (of the node\'s memory)')
    value, total = document[sizes[0]], memory()
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f'{path}: {sizes[0]} is a positive number')
    window = SIZES[sizes[0]](value, total)
    if window > total:
        raise ValueError(f'{path}: a window of {window} bytes is more than the node\'s {total} bytes of memory')
    for key in ('exec', 'ledger'):
        if not isinstance(document.get(key, False), bool):
            raise ValueError(f'{path}: {key} is true or false')
    canonical = {'name': path.stem, 'purpose': ' '.join(purpose.split()), 'window': window, 'exec': document.get('exec', False)}
    if document.get('ledger'):
        canonical['ledger'] = True
    for key, name in PACED.items():
        if key not in document:
            continue
        interval = document[key]
        if isinstance(interval, bool) or not isinstance(interval, (int, float)) or interval <= 0:
            raise ValueError(f'{path}: {key} is a positive number of microseconds')
        canonical[name] = max(1, int(round(interval * 1000)))
    return canonical


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    try:
        configuration = load(sys.argv[1])
    except (OSError, ValueError, UnicodeDecodeError, tomllib.TOMLDecodeError, json.JSONDecodeError, yaml.YAMLError) as error:
        sys.exit(f'bridge configuration: {error}')
    canonical = Path(sys.argv[2])
    canonical.parent.mkdir(parents=True, exist_ok=True)
    canonical.write_bytes(plistlib.dumps(configuration))
