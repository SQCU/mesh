from __future__ import annotations

import hashlib
import json
import os

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from .array_tree import pack_state, unpack_state as unpack_tree, write_payload, atomic_save
from .policy_contract import PARAMETERIZED_ARMS, is_matrix_fusion_arm, architecture_arm
from .game_value import reward_fingerprint

ARCH_KEY = "__arch__"
ARCH_SPEC_KEY = "__arch_spec__"
RNG_KEY = "__rng__"
POLICY_KEY = "__policy_arm__"
POLICY_VERSION_KEY = "__policy_version__"
REWARD_CONTRACT_KEY = "__reward_contract__"
LINEAGE_INITIAL_KEY = "__initial_checkpoint_sha256__"
POLICY_VERSIONS = {arm: 21 if is_matrix_fusion_arm(arm) else 19 for arm in PARAMETERIZED_ARMS}
ARCHITECTURE_VERSION = 12

class Payload(dict):
    @property
    def files(self):
        return list(self)

def unpack_state(payload, prefix="__runtime__"):
    from .replay import Frame
    from .inputs import ChorusArrays
    from .game_value import GameContext, CartSnapshot
    from .buffers import ContextualEvent, EventKind
    types = {value.__name__: value for value in (Frame, ChorusArrays, GameContext, CartSnapshot, ContextualEvent, EventKind)}
    return unpack_tree(payload, prefix, types)

def architecture_spec(module):
    return sorted(
        [name, [int(d) for d in value.shape]]
        for name, value in tree_flatten(module.parameters())
    )

def architecture_fingerprint(module):
    payload = [ARCHITECTURE_VERSION, getattr(module, "state_schema", None), architecture_spec(module)]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()[:16]

def _scalar(saved, key):
    return None if key not in saved.files else np.asarray(saved[key]).item()

def _attach(module, measurement):
    for name, value in measurement.items():
        setattr(module, "checkpoint_" + name, value)
    return measurement

def checkpoint_metadata(saved):
    return {name: _scalar(saved, key) for name, key in (
        ('source_arm', POLICY_KEY), ('source_version', POLICY_VERSION_KEY),
        ('source_architecture', ARCH_KEY), ('source_reward_contract', REWARD_CONTRACT_KEY),
        ('source_updates', '__updates__'), ('source_lineage_initial_sha256', LINEAGE_INITIAL_KEY))}

def checkpoint_parameters(module, metadata, source, arm, reward, *, training=False):
    source_arm = metadata['source_arm']
    expected_arm = arm if training else architecture_arm(arm)
    actual_arm = source_arm if training else architecture_arm(source_arm)
    for name, actual, expected in (
        ('policy arm', actual_arm, expected_arm),
        ('policy version', metadata['source_version'], POLICY_VERSIONS[arm]),
        ('native schema and architecture', metadata['source_architecture'], architecture_fingerprint(module)),
        ('reward contract', metadata['source_reward_contract'], reward),
    ):
        if actual != expected:
            raise ValueError(f'checkpoint {name}: {actual} differs from {expected}')
    return whole_tensor_tree(tree_flatten(module.parameters()), source)

def tensor_tree_measurement(live_items, source_items):
    live = dict(live_items)
    source = {name: np.asarray(value) for name, value in source_items}
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
    source = {name: np.asarray(value) for name, value in source_items}
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
    before = list(live.items())
    try:
        with np.load(path, allow_pickle=False) as saved:
            source = [(name, saved[name]) for name in saved.files if not name.startswith('__')]
            tree_measurement = tensor_tree_measurement(live.items(), source)
            metadata = checkpoint_metadata(saved)
            measurement.update(metadata)
            measurement.update({
                'source_weight_mass': tree_measurement['source_mass'],
                'live_only_weight_mass': tree_measurement['live_only_mass'],
                'source_only_weight_mass': tree_measurement['source_only_mass'],
                'shape_difference_mass': tree_measurement['shape_difference_mass'],
                'nonfinite_weight_mass': tree_measurement['nonfinite_mass'],
                'composable_weight_mass': tree_measurement['composable_mass'],
            })
            parameters, _ = checkpoint_parameters(module, metadata, source, live_arm, live_reward_contract)
        module.load_weights(tree_flatten(parameters), strict=True)
        measurement['loaded_weight_mass'] = tree_measurement['source_mass']
        measurement['updates'] = measurement['source_updates']
        measurement['lineage_initial_sha256'] = measurement['source_lineage_initial_sha256']
    except Exception as error:
        module.load_weights(before, strict=True)
        measurement["load_exception"] = f"{type(error).__name__}: {error}"
        measurement["updates"] = 0
        measurement["lineage_initial_sha256"] = None
    return _attach(module, measurement)

def load_policy(module, checkpoint, policy_arm):
    measurement = load_module_checkpoint(
        module, checkpoint, policy_arm, reward_fingerprint(policy_arm),
    )
    print(json.dumps({"event": "checkpoint_measurement", **measurement}), flush=True)
    return module

def policy_source(arm, model, checkpoint, mode):
    version_arm = architecture_arm(arm)
    return {
        "arm": arm,
        "mode": mode,
        "checkpoint": checkpoint,
        "checkpoint_bytes": os.path.getsize(checkpoint) if checkpoint and os.path.exists(checkpoint) else 0,
        "checkpoint_sha256": checkpoint_sha256(checkpoint),
        "source_weight_mass": 0 if model is None else int(getattr(model, "checkpoint_source_weight_mass", 0)),
        "live_weight_mass": 0 if model is None else int(getattr(model, "checkpoint_live_weight_mass", 0)),
        "loaded_weight_mass": 0 if model is None else int(getattr(model, "checkpoint_loaded_weight_mass", 0)),
        "composable_weight_mass": 0 if model is None else int(getattr(model, "checkpoint_composable_weight_mass", 0)),
        "source_only_weight_mass": 0 if model is None else int(getattr(model, "checkpoint_source_only_weight_mass", 0)),
        "live_only_weight_mass": 0 if model is None else int(getattr(model, "checkpoint_live_only_weight_mass", 0)),
        "shape_difference_mass": 0 if model is None else int(getattr(model, "checkpoint_shape_difference_mass", 0)),
        "nonfinite_weight_mass": 0 if model is None else int(getattr(model, "checkpoint_nonfinite_weight_mass", 0)),
        "load_exception": None if model is None else getattr(model, "checkpoint_load_exception", None),
        "continuation": None if model is None else getattr(model, "checkpoint_continuation", None),
        "updates": None if arm == "default" else getattr(model, "checkpoint_updates", None),
        "source_arm": None if model is None else getattr(model, "checkpoint_source_arm", None),
        "live_arm": arm,
        "source_version": None if model is None else getattr(model, "checkpoint_source_version", None),
        "live_version": None if arm == "default" else POLICY_VERSIONS[version_arm],
        "source_architecture": None if model is None else getattr(model, "checkpoint_source_architecture", None),
        "live_architecture": None if model is None else getattr(model, "checkpoint_live_architecture", None),
        "source_reward_contract": None if model is None else getattr(model, "checkpoint_source_reward_contract", None),
        "lineage_initial_sha256": None if model is None else getattr(model, "checkpoint_lineage_initial_sha256", None),
        "live_reward_contract": "fixed_default" if arm == "default" else reward_fingerprint(arm),
        "parameter_seed": None if model is None else getattr(model, "parameter_seed", None),
    }

def checkpoint_sha256(path):
    if not path or not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

__all__ = [
    "ARCH_KEY", "ARCH_SPEC_KEY", "RNG_KEY", "POLICY_KEY",
    "POLICY_VERSION_KEY", "REWARD_CONTRACT_KEY", "LINEAGE_INITIAL_KEY", "POLICY_VERSIONS",
    "architecture_spec", "architecture_fingerprint", "tensor_tree_measurement",
    "whole_tensor_tree", "load_module_checkpoint",
]
