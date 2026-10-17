from __future__ import annotations

import argparse
import json
import os
import shutil

import numpy as np

from .checkpoint_state import checkpoint_reference

def export_checkpoint(source, target):
    source = os.path.abspath(os.path.expanduser(source))
    target = os.path.abspath(os.path.expanduser(target))
    os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
    with np.load(source, allow_pickle=False) as archive:
        if '__bundle__' in archive.files:
            bundle = os.path.join(os.path.dirname(source), str(archive['__bundle__']))
            checkpoint_reference(target, bundle, str(archive['__prefix__']))
        else:
            temporary = target + ".new"
            try:
                shutil.copyfile(source, temporary)
                os.replace(temporary, target)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
    size = os.path.getsize(target)
    return {
        "source": source,
        "target": target,
        "bytes": size,
        "payload_bytes": size,
        "dropped_bytes": 0,
    }

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("target")
    args = parser.parse_args(argv)
    print(json.dumps(export_checkpoint(args.source, args.target), sort_keys=True))

if __name__ == "__main__":
    main()
