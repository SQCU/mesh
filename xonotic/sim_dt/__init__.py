from __future__ import annotations

from .batched_grpo_pfsp import (
    collect_grpo_rollout_group,
    compute_step_rewards_batched,
    eval_darkplaces_transfer,
    eval_sim_dt_league,
    scripted_plan_tensor,
    update_batched_grpo_ppo,
)
from .dataset_sampler import AtemporalBatch, PersistentMatchPool, PolicyVariantDispatcher
from .engine import DeltaTimePayloadEngine, SimRulesConfig, SimSnapshot
from .metal_runtime import (
    CompiledAuroraOptimizer,
    CompiledAuroraState,
    CompiledPlanTrainer,
    CompiledPlanWally,
    CompiledRolloutGraph,
    MetalCGTRewardComputer,
    MetalDeltaTimePayloadEngine,
    MetalSimSnapshot,
)
from .ppo_trainer import AtemporalPPOTrainer
from .world_graph import WorldGraphIR, build_gas_world_graph, world_graph_from_vfs_bundle

__all__ = [
    "AtemporalBatch",
    "PersistentMatchPool",
    "PolicyVariantDispatcher",
    "AtemporalPPOTrainer",
    "DeltaTimePayloadEngine",
    "MetalDeltaTimePayloadEngine",
    "MetalCGTRewardComputer",
    "MetalSimSnapshot",
    "CompiledPlanWally",
    "CompiledPlanTrainer",
    "CompiledAuroraOptimizer",
    "CompiledAuroraState",
    "CompiledRolloutGraph",
    "SimRulesConfig",
    "SimSnapshot",
    "WorldGraphIR",
    "build_gas_world_graph",
    "world_graph_from_vfs_bundle",
    "scripted_plan_tensor",
    "compute_step_rewards_batched",
    "collect_grpo_rollout_group",
    "update_batched_grpo_ppo",
    "eval_sim_dt_league",
    "eval_darkplaces_transfer",
]
