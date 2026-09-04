from __future__ import annotations

import hashlib
import json
import os

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from .policy_contract import PARAMETERIZED_ARMS

ARCH_KEY = "__arch__"
ARCH_SPEC_KEY = "__arch_spec__"
RNG_KEY = "__rng__"
POLICY_KEY = "__policy_arm__"
POLICY_VERSION_KEY = "__policy_version__"
REWARD_CONTRACT_KEY = "__reward_contract__"
LINEAGE_INITIAL_KEY = "__initial_checkpoint_sha256__"
POLICY_VERSIONS = {arm: 10 for arm in PARAMETERIZED_ARMS}
ARCHITECTURE_VERSION = 5

def architecture_spec(module):
    return sorted(
        [name, [int(d) for d in value.shape]]
        for name, value in tree_flatten(module.parameters())
    )

def architecture_fingerprint(module):
    payload = [ARCHITECTURE_VERSION, architecture_spec(module)]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()[:16]

def _scalar(saved, key):
    return None if key not in saved.files else np.asarray(saved[key]).item()

def _attach(module, measurement):
    for name, value in measurement.items():
        setattr(module, "checkpoint_" + name, value)
    return measurement

def tensor_tree_measurement(live_items, source_items):
    live = dict(live_items)
    source = {name: np.asarray(value).copy() for name, value in source_items}
    source_names = set(source)
    live_names = set(live)
    shared = source_names & live_names
    same_shape = {
        name: value for name, value in source.items()
        if name in live
        and tuple(value.shape) == tuple(live[name].shape)
    }
    return {
        "source_mass": len(source_names),
        "live_mass": len(live_names),
        "source_only_mass": len(source_names - live_names),
        "live_only_mass": len(live_names - source_names),
        "shape_difference_mass": sum(
            tuple(source[name].shape) != tuple(live[name].shape) for name in shared
        ),
        "nonfinite_mass": sum(not np.isfinite(value).all() for value in source.values()),
        "composable_mass": len(same_shape),
        "whole_tree_name_mass": int(source_names == live_names),
        "whole_tree_shape_mass": int(
            source_names == live_names and len(same_shape) == len(live_names)
        ),
    }

def whole_tensor_tree(live_items, source_items):
    live = dict(live_items)
    source = {name: np.asarray(value).copy() for name, value in source_items}
    measurement = tensor_tree_measurement(live.items(), source.items())
    if not measurement["whole_tree_name_mass"]:
        raise ValueError("checkpoint tensor names differ from the live tensor tree")
    if not measurement["whole_tree_shape_mass"]:
        raise ValueError("checkpoint tensor shapes differ from the live tensor tree")
    if measurement["nonfinite_mass"]:
        raise ValueError("checkpoint tensor tree contains non-finite coordinates")
    return tree_unflatten([(name, mx.array(source[name])) for name in live]), measurement

def load_module_checkpoint(module, path, live_arm, live_reward_contract):
    live = dict(tree_flatten(module.parameters()))
    measurement = {
        "path": path,
        "path_exists": bool(path and os.path.isfile(path)),
        "source_weight_mass": 0,
        "live_weight_mass": len(live),
        "loaded_weight_mass": 0,
        "source_only_weight_mass": 0,
        "live_only_weight_mass": len(live),
        "shape_difference_mass": 0,
        "nonfinite_weight_mass": 0,
        "composable_weight_mass": 0,
        "updates": None,
        "source_arm": None,
        "live_arm": live_arm,
        "source_version": None,
        "live_version": POLICY_VERSIONS.get(live_arm),
        "source_architecture": None,
        "live_architecture": architecture_fingerprint(module),
        "source_reward_contract": None,
        "lineage_initial_sha256": None,
        "live_reward_contract": live_reward_contract,
        "load_exception": None,
    }
    if not measurement["path_exists"]:
        return _attach(module, measurement)
    with np.load(path, allow_pickle=False) as saved:
        source = {
            name: np.asarray(saved[name]).copy()
            for name in saved.files if not name.startswith("__")
        }
        tree_measurement = tensor_tree_measurement(live.items(), source.items())
        measurement.update({
            "source_weight_mass": tree_measurement["source_mass"],
            "source_only_weight_mass": tree_measurement["source_only_mass"],
            "live_only_weight_mass": tree_measurement["live_only_mass"],
            "shape_difference_mass": tree_measurement["shape_difference_mass"],
            "nonfinite_weight_mass": tree_measurement["nonfinite_mass"],
            "updates": _scalar(saved, "__updates__"),
            "source_arm": _scalar(saved, POLICY_KEY),
            "source_version": _scalar(saved, POLICY_VERSION_KEY),
            "source_architecture": _scalar(saved, ARCH_KEY),
            "source_reward_contract": _scalar(saved, REWARD_CONTRACT_KEY),
            "lineage_initial_sha256": _scalar(saved, LINEAGE_INITIAL_KEY),
        })
    tree_measurement = tensor_tree_measurement(live.items(), source.items())
    measurement["composable_weight_mass"] = tree_measurement["composable_mass"]
    before = [(name, value) for name, value in live.items()]
    try:
        whole_tensor_tree(live.items(), source.items())
        module.load_weights(
            [(name, mx.array(value)) for name, value in source.items()], strict=True,
        )
        measurement["loaded_weight_mass"] = tree_measurement["source_mass"]
    except Exception as error:
        module.load_weights(before, strict=True)
        measurement["load_exception"] = f"{type(error).__name__}: {error}"
    return _attach(module, measurement)

__all__ = [
    "ARCH_KEY", "ARCH_SPEC_KEY", "RNG_KEY", "POLICY_KEY",
    "POLICY_VERSION_KEY", "REWARD_CONTRACT_KEY", "LINEAGE_INITIAL_KEY", "POLICY_VERSIONS",
    "architecture_spec", "architecture_fingerprint", "tensor_tree_measurement",
    "whole_tensor_tree", "load_module_checkpoint",
]
