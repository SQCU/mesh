from __future__ import annotations

import argparse
import json
import os
import shutil

def export_checkpoint(source, target):
    source = os.path.abspath(os.path.expanduser(source))
    target = os.path.abspath(os.path.expanduser(target))
    os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
    temporary = target + ".new"
    shutil.copyfile(source, temporary)
    os.replace(temporary, target)
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
