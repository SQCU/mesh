from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .losses import DEFAULT_LOSS_CONFIG, LossConfig


CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"
DEFAULT_TRAINING_CONFIG_PATH = CONFIGS_DIR / "training_default.json"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _as_tuple(val: Any, cast=float) -> tuple:
    return tuple(cast(x) for x in val)


@dataclass(frozen=True)
class NormConfig:
    mp_norm_eps: float = 1e-5
    mp_unit_vec_eps: float = 1e-6
    mp_silu_scale: float = 0.596
    mp_lerp_eps: float = 1e-8
    unit_norm_eps: float = 1e-6
    map_radius_min: float = 128.0
    pos_fallback_scale: float = 1000.0
    vel_scale: float = 400.0
    health_offset: float = 100.0
    health_scale: float = 100.0
    armor_offset: float = 50.0
    armor_scale: float = 100.0
    ammo_scale: float = 50.0
    strength_max_s: float = 30.0
    cart_len_scale: float = 500.0
    cart_speed_scale: float = 100.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "NormConfig":
        if not data:
            return cls()
        return cls(**{f: data[f] for f in cls.__dataclass_fields__ if f in data})


@dataclass(frozen=True)
class ActionPackConfig:
    nav_w_shift: float = 0.8
    nav_w_clip: tuple[float, float] = (-10.0, 10.0)
    cart_ds_scale: float = 120.0
    cart_ds_power: float = 1.6
    cart_ds_clip: float = 480.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ActionPackConfig":
        if not data:
            return cls()
        kw = {f: data[f] for f in cls.__dataclass_fields__ if f in data}
        if "nav_w_clip" in kw:
            kw["nav_w_clip"] = _as_tuple(kw["nav_w_clip"], float)
        return cls(**kw)


@dataclass(frozen=True)
class ModelConfig:
    d: int = 256
    depth: int = 4
    h: int = 1024
    r_gram: int = 128
    r_dpp: int = 64
    recurrent: bool = True
    use_fsq: bool = True
    fsq_levels: tuple[int, ...] = (5, 5, 5, 5)
    use_mealy: bool = True
    type_blend_init_logit: float = -1.8
    evt_blend: float = 0.2
    fovea_blend: float = 0.25
    mealy_blend: float = 0.25
    step_blend_init: float = 0.35
    fsq_in_gain_init: float = 1.2
    fsq_in_gain_bounds: tuple[float, float] = (0.25, 4.0)
    fsq_blend_init_logit: float = -1.5
    fsq_std_min: float = 0.35
    cont_gain_init: float = 0.04
    cont_gain_bound: float = 3.0
    cont_mean_scale: float = 2.0
    cont_bias_init: dict[int, float] = field(default_factory=lambda: {6: 0.10, 8: -1.1, 9: 0.25})
    cont_std_init: float = 0.38
    cont_logstd_bounds: tuple[float, float] = (-2.5, 0.25)
    temp_init: float = 2.0
    temp_bounds: tuple[float, float] = (0.5, 32.0)
    cart_dist_bias_init: float = 1.0
    cart_dist_bias_bounds: tuple[float, float] = (0.0, 6.0)
    cart_dist_scale: float = 400.0
    rival_own_mask_fill: float = -1e4
    value_gain_init: float = 0.5
    value_gain_bounds: tuple[float, float] = (0.05, 5.0)
    value_priv_gain_init: float = 0.15
    value_priv_gain_bounds: tuple[float, float] = (0.0, 2.0)
    skill_id_gain_init: float = 0.5
    skill_id_gain_bounds: tuple[float, float] = (0.05, 3.0)
    dyn_drift_gain_init: float = 0.25
    dyn_action_gain_init: float = 0.25
    dyn_gain_bounds: tuple[float, float] = (0.05, 2.0)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ModelConfig":
        if not data:
            return cls()
        kw: dict[str, Any] = {}
        tuple_float_fields = {
            "fsq_in_gain_bounds", "cont_logstd_bounds", "temp_bounds",
            "cart_dist_bias_bounds", "value_gain_bounds", "value_priv_gain_bounds",
            "skill_id_gain_bounds", "dyn_gain_bounds",
        }
        for f in cls.__dataclass_fields__:
            if f not in data:
                continue
            v = data[f]
            if f == "fsq_levels":
                kw[f] = _as_tuple(v, int)
            elif f in tuple_float_fields:
                kw[f] = _as_tuple(v, float)
            elif f == "cont_bias_init":
                kw[f] = {int(k): float(val) for k, val in v.items()}
            else:
                kw[f] = v
        return cls(**kw)


@dataclass(frozen=True)
class OptimizerConfig:
    lr: float = 0.010
    scalar_lr: float = 5e-3
    exploiter_lr_mult: float = 1.2
    mu: float = 0.90
    nesterov: bool = True
    pp_iterations: int = 2
    pp_beta: float = 0.5
    ns_steps: int = 5
    ns_coeffs: tuple[float, float, float] = (3.4445, -4.7750, 2.0315)
    init_pp_iterations: int = 3
    init_ns_steps: int = 7
    d_init_clamp: tuple[float, float] = (0.1, 10.0)
    d_iter_clamp: tuple[float, float] = (0.05, 20.0)
    scalar_beta2: float = 0.99
    scalar_eps: float = 1e-8
    eps: float = 1e-7
    grad_clip: float = 1.0
    ema_decay: float = 0.88
    ppo_epochs: int = 2

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "OptimizerConfig":
        if not data:
            return cls()
        kw: dict[str, Any] = {}
        for f in cls.__dataclass_fields__:
            if f not in data:
                continue
            v = data[f]
            if f in ("ns_coeffs", "d_init_clamp", "d_iter_clamp"):
                kw[f] = _as_tuple(v, float)
            else:
                kw[f] = v
        return cls(**kw)


@dataclass(frozen=True)
class HarnessConfig:
    stage_dir: str = "~/xonotic-stage"
    dp_bin_rel: str = "darkplaces-dedicated"
    basedir_rel: str = "Xonotic"
    userdir_rel: str = "userdir"
    data_pk3_rel: str = "Xonotic/data/xonotic-20230620-data.pk3"
    framebytes: int = 8192
    strategy_interval: float = 0.4
    payload_speed: float = 80.0
    default_skill: int = 7
    maxclients_default: int = 16
    maxclients_supermap: int = 255
    base_port: int = 26000
    port_stride_per_iter: int = 10
    port_iter_modulo: int = 5
    spawn_stagger_s: float = 0.15
    stop_sigterm_timeout_s: float = 3.0
    stop_sigkill_timeout_s: float = 2.0
    max_episode_wall_s: float = 600.0
    point_limit: float = 120.0
    map_slowmo: dict[str, float] = field(
        default_factory=lambda: {
            "boil": 1.0,
            "stormkeep": 1.2,
            "catharsis": 1.8,
            "afterslime": 1.8,
            "vorix": 1.3,
            "dance": 1.3,
            "glowplant": 1.3,
        }
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "HarnessConfig":
        if not data:
            return cls()
        kw: dict[str, Any] = {}
        for f in cls.__dataclass_fields__:
            if f not in data:
                continue
            if f == "map_slowmo":
                kw[f] = {str(k): float(v) for k, v in data[f].items()}
            else:
                kw[f] = data[f]
        return cls(**kw)


@dataclass(frozen=True)
class LeagueConfig:
    pfsp_power: float = 2.0
    pfsp_min_weight: float = 0.05
    pfsp_default_winrate: float = 0.45
    alpha_rank_alpha: float = 2.0
    alpha_rank_m_pop: int = 20
    roster_dist_probs: dict[str, float] = field(
        default_factory=lambda: {
            "gaussian_lobby": 0.35,
            "team_skewed": 0.30,
            "within_team_skewed": 0.25,
            "uniform_native": 0.10,
        }
    )
    roster_gaussian_std: float = 75.0
    roster_team_skew_std: float = 28.0
    roster_within_stds: tuple[float, float, float] = (25.0, 35.0, 25.0)
    exploiter_server_configs: tuple[dict[str, Any], ...] = (
        {"opp1": "{target}", "opp2": "{target}", "fcp_mixed": False, "perturbation": "baseline"},
        {"opp1": "{target}", "opp2": "default", "fcp_mixed": False, "perturbation": "baseline"},
        {"opp1": "{target}", "opp2": "{target}", "fcp_mixed": False, "perturbation": "fast"},
        {"opp1": "{target}", "opp2": "default", "fcp_mixed": False, "perturbation": "slow"},
        {"opp1": "{target}", "opp2": "{target}", "fcp_mixed": False, "perturbation": "volatile"},
        {"opp1": "{target}", "opp2": "script_cart_rush", "fcp_mixed": False, "perturbation": "baseline"},
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "LeagueConfig":
        if not data:
            return cls()
        kw: dict[str, Any] = {}
        for f in cls.__dataclass_fields__:
            if f not in data:
                continue
            v = data[f]
            if f == "roster_dist_probs":
                kw[f] = {str(k): float(val) for k, val in v.items()}
            elif f == "roster_within_stds":
                kw[f] = _as_tuple(v, float)
            elif f == "exploiter_server_configs":
                kw[f] = tuple(dict(item) for item in v)
            else:
                kw[f] = v
        return cls(**kw)


@dataclass(frozen=True)
class ScheduleConfig:
    device: str = "mps"
    out_dir: str = "~/xonotic-runs/studio4-largebatch-native"
    calibration_json: str = ""
    init_ckpt: str = ""
    supermap_manifest: str = ""
    train_maps: tuple[str, ...] = ("boil", "boil", "stormkeep", "stormkeep", "catharsis", "catharsis")
    servers_per_node: int = 0
    match_supermap_scale_across_ranks: bool = True
    main_iters: int = 10
    exploiter_iters: int = 3
    local_batch: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ScheduleConfig":
        if not data:
            return cls()
        kw: dict[str, Any] = {}
        for f in cls.__dataclass_fields__:
            if f not in data:
                continue
            v = data[f]
            if f == "train_maps":
                kw[f] = tuple(m.strip() for m in v.split(",") if m.strip()) if isinstance(v, str) else tuple(str(m) for m in v)
            else:
                kw[f] = v
        return cls(**kw)


@dataclass(frozen=True)
class EvalConfig:
    eval_maps: tuple[str, ...] = ("boil", "stormkeep", "catharsis", "afterslime")
    eval_seeds_standard: int = 3
    eval_seeds_supermap: int = 2
    eval_triplets: tuple[tuple[str, ...], ...] = (
        ("default", "main_init", "main_final"),
        ("default", "main_mid", "main_final"),
        ("default", "exploiter_1", "main_final"),
        ("script_cart_rush", "main_mid", "main_final"),
    )
    eval_port_offset: int = 200
    eval_triplet_port_stride: int = 10
    skill_id_early_cutoff_s: float = 8.0
    fingerprint_cell_size: float = 256.0
    neuron_audit_maps: tuple[str, ...] = ("boil", "stormkeep")
    neuron_audit_duration_s: float = 10.0
    neuron_audit_port_offset: int = 350
    dead_unit_rel_threshold: float = 0.01

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "EvalConfig":
        if not data:
            return cls()
        kw: dict[str, Any] = {}
        for f in cls.__dataclass_fields__:
            if f not in data:
                continue
            v = data[f]
            if f in ("eval_maps", "neuron_audit_maps"):
                kw[f] = tuple(str(x) for x in v)
            elif f == "eval_triplets":
                kw[f] = tuple(tuple(str(a) for a in trip) for trip in v)
            else:
                kw[f] = v
        return cls(**kw)


@dataclass(frozen=True)
class SeedConfig:
    model_init: int = 2026
    schedule_base: int = 91000
    match_base: int = 30000
    rollout_rng_base: int = 50000
    eval_match_base: int = 75000
    eval_rng_base: int = 88000
    neuron_audit_seeds: tuple[int, ...] = (9901, 9902)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "SeedConfig":
        if not data:
            return cls()
        kw: dict[str, Any] = {}
        for f in cls.__dataclass_fields__:
            if f not in data:
                continue
            kw[f] = _as_tuple(data[f], int) if f == "neuron_audit_seeds" else int(data[f])
        return cls(**kw)


@dataclass(frozen=True)
class WorldDistributionConfig:
    mode: str = "static_maps"
    use_zero_copy_vfs: bool = True
    vfs_shm_mb: int = 384
    vfs_capacity: int = 4096
    maps_pk3_rel: str = "Xonotic/data/xonotic-20230620-maps.pk3"
    family_weights: dict[str, float] = field(
        default_factory=lambda: {"b3313_binary_stitch": 0.65, "gas_world_bsp": 0.35}
    )
    teams_min: int = 2
    teams_max: int = 24
    teams_dist: str = "lognormal"
    teams_log_mean: float = 1.887
    teams_log_std: float = 0.62
    carts_min_extra: int = 1
    carts_max: int = 256
    carts_dist: str = "lognormal"
    carts_mult_log_mean: float = 1.917
    carts_mult_log_std: float = 0.58
    carts_per_team_min: float = 2.0
    carts_per_team_max: float = 12.0
    players_min: int = 12
    players_max: int = 240
    players_per_team_min: int = 2
    players_per_team_max: int = 16
    players_per_team_low: float = 6.0
    players_per_team_high: float = 14.0
    tiles_min: int = 1
    tiles_max: int = 16
    players_per_tile: float = 12.0
    max_carts_per_tile: float = 8.0
    expander_min_degree: int = 3
    allow_rotations: bool = True
    tile_size: float = 1792.0
    joint_widths: tuple[float, ...] = (288.0, 320.0, 352.0, 384.0)
    joint_length: float = 96.0
    checkpoints_per_cart: int = 4
    stock_map_pool: tuple[str, ...] = (
        "boil", "runningman", "warfare", "stormkeep", "fuse", "solarium",
        "opium", "trident", "leave_em_behind", "silentsiege", "dance", "atelier",
    )
    eval_stream_specs: tuple[dict[str, Any], ...] = (
        {"seed": 4101, "family": "b3313_binary_stitch", "teams": 5, "carts": 28, "players": 70, "tiles": 6},
        {"seed": 4102, "family": "gas_world_bsp", "teams": 9, "carts": 52, "players": 108, "tiles": 9},
        {"seed": 4103, "family": "b3313_binary_stitch", "teams": 10, "carts": 47, "players": 120, "tiles": 10},
        {"seed": 4104, "family": "b3313_binary_stitch", "teams": 22, "carts": 112, "players": 240, "tiles": 14},
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "WorldDistributionConfig":
        if not data:
            return cls()
        kw: dict[str, Any] = {}
        for f in cls.__dataclass_fields__:
            if f not in data:
                continue
            v = data[f]
            if f == "family_weights":
                kw[f] = {str(k): float(val) for k, val in v.items()}
            elif f == "joint_widths":
                kw[f] = _as_tuple(v, float)
            elif f == "stock_map_pool":
                kw[f] = tuple(str(x) for x in v)
            elif f == "eval_stream_specs":
                kw[f] = tuple(dict(item) for item in v)
            else:
                kw[f] = v
        return cls(**kw)


@dataclass(frozen=True)
class SimDtConfig:
    dt: float = 0.4
    mode: str = "analytic"
    base_envs: int = 32
    group_size: int = 8
    warmup_steps_min: int = 15
    warmup_steps_max: int = 50
    rollout_horizon: int = 40
    iterations: int = 24
    mid_snapshot_iter: int = 12
    ppo_epochs: int = 3
    minibatch_envs: int = 64
    grpo_weight: float = 0.7
    gae_weight: float = 0.3
    terminal_win_bonus: float = 1.5
    point_limit_base: float = 160.0
    common_random_numbers: bool = True
    eval_interval: int = 6
    eval_matches_per_pair: int = 64
    eval_horizon: int = 180
    pfsp_update_ema: float = 0.8
    darkplaces_transfer_eval: bool = True
    darkplaces_eval_episodes: int = 2
    darkplaces_point_limit: float = 160.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "SimDtConfig":
        if not data:
            return cls()
        return cls(**{f: data[f] for f in cls.__dataclass_fields__ if f in data})


@dataclass(frozen=True)
class TrainingConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    norm: NormConfig = field(default_factory=NormConfig)
    action_pack: ActionPackConfig = field(default_factory=ActionPackConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    harness: HarnessConfig = field(default_factory=HarnessConfig)
    league: LeagueConfig = field(default_factory=LeagueConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    seeds: SeedConfig = field(default_factory=SeedConfig)
    world: WorldDistributionConfig = field(default_factory=WorldDistributionConfig)
    sim_dt: SimDtConfig = field(default_factory=SimDtConfig)
    losses: LossConfig = field(default_factory=lambda: DEFAULT_LOSS_CONFIG)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "TrainingConfig":
        if not data:
            return cls()
        return cls(
            model=ModelConfig.from_dict(data.get("model")),
            norm=NormConfig.from_dict(data.get("norm")),
            action_pack=ActionPackConfig.from_dict(data.get("action_pack")),
            optimizer=OptimizerConfig.from_dict(data.get("optimizer")),
            harness=HarnessConfig.from_dict(data.get("harness")),
            league=LeagueConfig.from_dict(data.get("league")),
            schedule=ScheduleConfig.from_dict(data.get("schedule")),
            eval=EvalConfig.from_dict(data.get("eval")),
            seeds=SeedConfig.from_dict(data.get("seeds")),
            world=WorldDistributionConfig.from_dict(data.get("world")),
            sim_dt=SimDtConfig.from_dict(data.get("sim_dt")),
            losses=LossConfig.from_dict(data.get("losses")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_json_chain(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text())
    merged: dict[str, Any] = {}
    if "extends" in raw:
        ext = raw["extends"]
        parents = [ext] if isinstance(ext, str) else list(ext)
        for p_rel in parents:
            parent = (path.parent / p_rel).resolve()
            merged = _deep_merge(merged, _load_json_chain(parent))
    if "losses_file" in raw:
        lpath = (path.parent / raw["losses_file"]).resolve()
        loss_raw = json.loads(lpath.read_text())
        merged = _deep_merge(merged, {"losses": loss_raw})
    body = {k: v for k, v in raw.items() if k not in ("extends", "losses_file")}
    return _deep_merge(merged, body)


def load_training_config(
    config_path: str | Path | None = None,
    losses_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> TrainingConfig:
    base_path = Path(config_path).expanduser().resolve() if config_path else DEFAULT_TRAINING_CONFIG_PATH
    data = _load_json_chain(base_path) if base_path.exists() else {}
    if losses_path:
        lp = Path(losses_path).expanduser().resolve()
        data = _deep_merge(data, {"losses": json.loads(lp.read_text())})
    if overrides:
        data = _deep_merge(data, overrides)
    return TrainingConfig.from_dict(data)


DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_NORM_CONFIG = NormConfig()
DEFAULT_ACTION_PACK_CONFIG = ActionPackConfig()
DEFAULT_OPTIMIZER_CONFIG = OptimizerConfig()
DEFAULT_HARNESS_CONFIG = HarnessConfig()
DEFAULT_LEAGUE_CONFIG = LeagueConfig()
DEFAULT_EVAL_CONFIG = EvalConfig()
DEFAULT_SEED_CONFIG = SeedConfig()
