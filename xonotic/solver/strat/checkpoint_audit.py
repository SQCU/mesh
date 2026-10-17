import argparse
import json

import numpy as np

from .checkpoint_state import open_checkpoint


def audit_checkpoint(data):
    groups = {name: {"arrays": 0, "nonfinite_arrays": 0, "nonfinite_coordinates": 0}
              for name in ("parameters", "optimizer", "replay", "metadata")}
    invalid = {}
    for key in data.files:
        value = np.asarray(data[key])
        if np.issubdtype(value.dtype, np.number):
            group = ("parameters" if not key.startswith("__") else "optimizer"
                     if key.startswith("__opt__") else "replay"
                     if key.startswith("__replay__") else "metadata")
            count = int(np.count_nonzero(~np.isfinite(value)))
            groups[group]["arrays"] += 1
            groups[group]["nonfinite_arrays"] += bool(count)
            groups[group]["nonfinite_coordinates"] += count
            if count:
                invalid[key] = count
    metadata = json.loads(str(data["__replay__meta"])) if "__replay__meta" in data.files else {"frames": [], "items": []}
    frames = {index for index, frame in enumerate(metadata["frames"])
              if any(key in invalid for key in frame["keys"])}
    affected = sum(any(encoded.get("array") in invalid or encoded.get("frame") in frames
                       for encoded in item.values()) for item in metadata["items"])
    return {"groups": groups, "nonfinite_arrays": len(invalid),
            "replay_frames": len(metadata["frames"]), "nonfinite_replay_frames": len(frames),
            "replay_transitions": len(metadata["items"]), "affected_replay_transitions": affected,
            "source_updates": int(data["__updates__"]) if "__updates__" in data.files else None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", nargs="+")
    for path in parser.parse_args().checkpoint:
        with open_checkpoint(path) as data:
            print(json.dumps({"path": path, **audit_checkpoint(data)}, sort_keys=True))
