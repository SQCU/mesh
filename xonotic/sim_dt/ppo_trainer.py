"""Single locus of responsibility for atemporal policy + Winnie/Lou value + dynamics optimization.

Consumes flat AtemporalBatch transition tensors (synchronous on-policy OR slightly stale
asynchronous off-policy), executes compiled Metal 4 NAX compressed had_int8 forward +
analytical backward + single-CB CompiledAuroraOptimizer, and logs behavior<->current
KL divergence (approx_kl) and policy staleness as boring first-class metrics.

Zero match lifecycle, warmup, world generation, or PFSP league logic lives here.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from payload.tools.strategy_io_schema import CS, OBS, TS
from solver.strat.losses import LossConfig
from solver.strat.plan_policy import PlanWally, clone_plan_model
from solver.strat.train_config import TrainingConfig

from .dataset_sampler import AtemporalBatch
from .metal_runtime import (
    CompiledAuroraOptimizer,
    CompiledAuroraState,
    CompiledPlanTrainer,
)


def _team_eff_rho_flat(
    carts: torch.Tensor,
    teams: torch.Tensor,
    obs: torch.Tensor,
    partial_cp_weight: float = 0.5,
) -> torch.Tensor:
    scores = teams[:, :, TS["SCORE"]].float()
    held = teams[:, :, TS["CHECKPOINTS_HELD"]].float()
    rate = teams[:, :, TS["CHECKPOINT_RATE"]].float()
    limit = torch.clamp(teams[:, :, TS["SCORE_LIMIT"]].float(), min=1e-6)
    rem = torch.clamp(limit - scores, min=1e-6)
    path_len = torch.clamp(carts[:, :, CS["PATH_LENGTH"]].float(), min=1e-6)
    pos_frac = torch.clamp(carts[:, :, CS["PATH_POSITION"]].float() / path_len, 0.0, 1.0)
    cp_count = torch.clamp(carts[:, :, CS["CHECKPOINT_COUNT"]].float(), min=1.0)
    seg_frac = (pos_frac * cp_count) - torch.floor(pos_frac * cp_count)
    owner = (carts[:, :, CS["CONTROL_TEAM"]].float()[:, :, None] == teams[:, :, TS["ID"]].float()[:, None, :]).float()
    eff_held = held + float(partial_cp_weight) * torch.einsum("ejk,ej->ek", owner, seg_frac)
    rho = torch.where(eff_held > 0.0, (rate * eff_held) / rem, torch.zeros_like(scores))
    mem = (obs[:, :, OBS["TEAM"]].float()[:, :, None] == teams[:, :, TS["ID"]].float()[:, None, :]).float()
    return torch.einsum("enk,ek->en", mem, rho)


class AtemporalPPOTrainer:
    """Cached Metal 4 NAX atemporal PPO/GRPO/Winnie-Lou/Dynamics learner with KL & staleness logging."""

    def __init__(
        self,
        model: PlanWally,
        ema_model: PlanWally | None,
        cfg: TrainingConfig,
    ):
        self.model = model.to(device=torch.device("cpu"))
        self.ema_model = (
            ema_model.to(device=torch.device("cpu"))
            if ema_model is not None
            else clone_plan_model(self.model)
        )
        self.cfg = cfg
        self.aurora_state = CompiledAuroraState.empty()
        self.policy_version: int = 0
        self._cache: dict[tuple[int, int, int, int], tuple[CompiledPlanTrainer, CompiledAuroraOptimizer]] = {}
        self._param_order = [n for n, _ in self.model.named_parameters()]

    def close(self) -> None:
        for tr, opt in self._cache.values():
            opt.close()
            tr.close()
        self._cache.clear()

    def _get_compiled(self, mb_envs: int, N: int, J: int, K: int) -> tuple[CompiledPlanTrainer, CompiledAuroraOptimizer]:
        key = (mb_envs, N, J, K)
        pair = self._cache.get(key)
        if pair is None:
            tr = CompiledPlanTrainer(self.model, E_mb=mb_envs, N=N, J=J, K=K)
            opt = CompiledAuroraOptimizer(tr, self.model, self.ema_model, self.cfg.optimizer, state=self.aurora_state)
            self._cache[key] = (tr, opt)
            return tr, opt
        tr, opt = pair
        tr.sync_weights_(self.model)
        opt.load_from_models_(self.model, self.ema_model)
        return tr, opt

    @torch.no_grad()
    def export_weights_bytes(self) -> bytes:
        """Pack all BF16 parameters into one flat contiguous payload (~2.96 MB) for fast async broadcast."""
        chunks = [
            p.detach().to(device="cpu", dtype=torch.bfloat16).contiguous().view(torch.uint8).numpy().tobytes()
            for _, p in self.model.named_parameters()
        ]
        return b"".join(chunks)

    @torch.no_grad()
    def load_weights_bytes_(self, raw: bytes) -> None:
        off = 0
        for _, p in self.model.named_parameters():
            nb = p.numel() * 2
            arr = np.frombuffer(raw, dtype=np.uint8, count=nb, offset=off).copy()
            t = torch.from_numpy(arr).view(torch.bfloat16).reshape(p.shape)
            p.copy_(t.to(device=p.device, dtype=p.dtype))
            off += nb

    @torch.no_grad()
    def update(self, batch: AtemporalBatch, *, entropy_coef: float) -> dict[str, Any]:
        """Run compiled Metal 4 atemporal PPO + Winnie/Lou + Dynamics update over flat batch."""
        t0 = time.perf_counter()
        lcfg: LossConfig = self.cfg.losses
        sc = self.cfg.sim_dt

        M_tot, N, _ = batch.obs.shape
        J = int(batch.carts.shape[1])
        K = int(batch.teams.shape[1])
        entities_per_env = N + J + K
        staleness = max(0, int(self.policy_version) - int(batch.policy_version))

        if batch.obs_drho is not None:
            obs_drho_f = batch.obs_drho.to(torch.float32)
        else:
            assert batch.obs_next is not None and batch.carts_next is not None and batch.teams_next is not None
            rho_c = _team_eff_rho_flat(batch.carts, batch.teams, batch.obs, partial_cp_weight=lcfg.reward.cart_partial_cp_weight)
            rho_n = _team_eff_rho_flat(batch.carts_next, batch.teams_next, batch.obs_next, partial_cp_weight=lcfg.reward.cart_partial_cp_weight)
            obs_drho_f = torch.asinh(lcfg.dyn_drho_scale * (rho_n - rho_c))
        am_f = batch.actor_mask.to(torch.float32)
        adv_f = batch.adv.clamp(-lcfg.adv_clip, lcfg.adv_clip)

        target_rows = max(256, int(sc.target_microbatch_rows))
        mb_envs = max(2, min(M_tot, int(round(target_rows / max(1, entities_per_env)))))
        n_micros = max(1, (M_tot + mb_envs - 1) // mb_envs)
        target_steps = max(1, int(sc.target_opt_steps_per_epoch))
        accum = max(1, int(round(n_micros / float(target_steps))))

        tr, opt = self._get_compiled(mb_envs, N, J, K)
        tr.set_geo(batch.geo)
        tr.zero_grads_()

        diags: list[dict[str, float]] = []
        opt_steps = 0
        for _ in range(max(1, int(sc.ppo_epochs))):
            perm = torch.randperm(M_tot)
            in_accum = 0
            for start in range(0, M_tot, mb_envs):
                idx = perm[start : start + mb_envs]
                if idx.numel() < 2:
                    continue
                sd = tr.microbatch_fwd_bwd(
                    batch.obs[idx], batch.carts[idx], batch.teams[idx],
                    batch.cont[idx], batch.cart_pick[idx], batch.rival_pick[idx],
                    batch.beh_logp[idx], adv_f[idx], batch.gae_ret[idx], obs_drho_f[idx],
                    am_f[idx], batch.wm_in[idx],
                    lcfg=lcfg,
                    entropy_coef=float(entropy_coef),
                    grad_scale=1.0 / float(accum),
                )
                in_accum += 1
                if in_accum >= accum or (start + mb_envs) >= M_tot:
                    gn, cont_std = opt.step()
                    in_accum = 0
                    opt_steps += 1
                    diags.append({
                        "loss_total": float(sd["loss_total"]),
                        "loss_actor": float(sd["loss_actor"]),
                        "loss_critic": float(sd["loss_w"] + sd["loss_l"]),
                        "loss_dyn": float(sd["loss_dyn"]),
                        "entropy": float(sd["entropy"]),
                        "entropy_disc": float(sd["entropy_disc"]),
                        "clip_frac": float(sd["clip_frac"]),
                        "approx_kl": float(sd["approx_kl"]),
                        "grad_norm": gn,
                        "cont_std": cont_std,
                    })

        opt.export_to_models_(self.model, self.ema_model)
        self.policy_version += opt_steps

        def _mean(k: str) -> float:
            return round(float(np.mean([d[k] for d in diags])), 5) if diags else 0.0

        wall_s = time.perf_counter() - t0
        return {
            "update_wall_s": round(wall_s, 3),
            "opt_steps": opt_steps,
            "policy_version": int(self.policy_version),
            "batch_policy_version": int(batch.policy_version),
            "staleness_steps": staleness,
            "mb_envs": mb_envs,
            "mb_rows": mb_envs * entities_per_env,
            "accum_steps": accum,
            "frames_per_epoch": M_tot,
            "loss_total": _mean("loss_total"),
            "loss_actor": _mean("loss_actor"),
            "loss_critic": _mean("loss_critic"),
            "loss_dyn": _mean("loss_dyn"),
            "entropy": _mean("entropy"),
            "entropy_disc": _mean("entropy_disc"),
            "clip_frac": _mean("clip_frac"),
            "approx_kl": _mean("approx_kl"),
            "grad_norm": _mean("grad_norm"),
            "cont_std": _mean("cont_std"),
        }
