import itertools
import json
import os
from pathlib import Path
import time

import numpy as np



def publish_active_run(telemetry):
    run = Path(telemetry).resolve().parent.parent
    active = run.parent / "active"
    temporary = run.parent / f".active-{os.getpid()}-{time.time_ns()}"
    temporary.symlink_to(run)
    os.replace(temporary, active)
    registry = Path.home() / '.local/share/mesh/xonotic-active.json'
    registry.parent.mkdir(parents=True, exist_ok=True)
    temporary = registry.with_name(f'.xonotic-active-{os.getpid()}.json')
    temporary.write_text(json.dumps({'schema': 1, 'application': 'xonotic', 'root': str(run),
        'telemetry': str(Path(telemetry).resolve()), 'producer_pid': os.getpid(), 'published_at': time.time()}))
    os.replace(temporary, registry)


def reporting_coverage(frame, measure, study, composition=None):
    comparison = frame.get("policy_comparison") or {}
    fields = [
        ("optimization", "Optimizer steps", len(frame.get("learning", {})), "learning + policy_updates"),
        ("j", "J geometry and source joins", len(measure.get("strata", [])), "producer J artifact"),
        ("counterfactual", "Same-input policy divergence", len(comparison.get("pairs", [])), "policy_comparison.pairs"),
        ("counterfactual_outputs", "Counterfactual output vectors", len(comparison.get("rows", [])), "policy_comparison.rows"),
        ("paired_ratings", "Paired-study Elo", len(study.get("ratings") or {}), "study.json; mirrored comparison legs"),
    ]
    reports = [{"id": key, "report": title, "state": "measured" if mass else "awaiting_observations",
                "observations": mass, "source": source} for key, title, mass, source in fields]
    composition = composition or {}
    groups = composition.get('groups', ())
    reports.append({'id': 'composition_ratings', 'report': 'Realized bot / team / controller ratings',
        'state': 'measured' if groups else 'awaiting_attributed_outcomes',
        'observations': sum(group['rounds'] for group in groups),
        'source': 'durable native configuration and acknowledged controller exposure; conditional estimates',
        'excluded_rounds': len(composition.get('excluded_rounds', ()))})
    return reports

def compare_policies(outputs, assignments, versions, sampled=False):
    rows, pairs, vectors = [], [], {}
    for arm, output in outputs.items():
        mean = np.asarray(output["rate_mean"], dtype=np.float64)
        log_scale = np.asarray(output["rate_log_scale"], dtype=np.float64)
        present = np.asarray(output.get('present', np.ones_like(mean)), dtype=bool)
        density = np.asarray(output.get('density', np.ones(len(mean))))
        for index, assignment in enumerate(assignments):
            mask = present[index]
            rows.append({
                "player": assignment["edict"], "team": assignment["team"],
                "assigned_policy": assignment["policy_arm"], "evaluated_policy": arm,
                "policy_updates": versions.get(arm),
                "controls_this_player": assignment["behavior"] == arm,
                "state_width": int(np.count_nonzero(mask)),
                "distribution": "gaussian" if density[index] else "deterministic_zero",
                "rate_mean_l2": float(np.linalg.norm(mean[index][mask])),
                "rate_sigma_rms": float(np.sqrt(np.mean(np.exp(2 * log_scale[index][mask])))) if density[index] and np.any(mask) else 0.0,
                "winner_value": float(output["winner_value"][index]),
                "loser_value": float(output["loser_value"][index]),
            })
        if sampled:
            vectors[arm] = {
                "rate_mean": mean.tolist(), "rate_log_scale": log_scale.tolist(),
                "sampled_rate": np.asarray(output["velocity"]).tolist(),
                "integrated_residual": np.asarray(output["residual"]).tolist(),
                "policy_updates": versions.get(arm),
            }
    for left, right in itertools.combinations(outputs, 2):
        p, q = outputs[left], outputs[right]
        pm, qm = np.asarray(p["rate_mean"], dtype=np.float64), np.asarray(q["rate_mean"], dtype=np.float64)
        pl, ql = np.asarray(p["rate_log_scale"], dtype=np.float64), np.asarray(q["rate_log_scale"], dtype=np.float64)
        mask = np.asarray(p.get('present', np.ones_like(pm)), dtype=bool) & np.asarray(q.get('present', np.ones_like(qm)), dtype=bool)
        pd = np.asarray(p.get('density', np.ones(len(pm)))) > 0
        qd = np.asarray(q.get('density', np.ones(len(qm)))) > 0
        difference = np.where(mask, pm - qm, 0)
        pl, ql = np.where(mask, pl, 0), np.where(mask, ql, 0)
        kl_pq = np.sum(ql - pl + 0.5 * (np.exp(2 * (pl - ql)) + difference ** 2 * np.exp(-2 * ql) - 1), axis=-1)
        kl_qp = np.sum(pl - ql + 0.5 * (np.exp(2 * (ql - pl)) + difference ** 2 * np.exp(-2 * pl) - 1), axis=-1)
        for index, assignment in enumerate(assignments):
            pairs.append({
                "player": assignment["edict"], "team": assignment["team"],
                "assigned_policy": assignment["policy_arm"], "left": left, "right": right,
                "rate_kl_left_right_nats": float(kl_pq[index]) if pd[index] and qd[index] else 0.0 if not pd[index] and not qd[index] else None,
                "rate_kl_right_left_nats": float(kl_qp[index]) if pd[index] and qd[index] else 0.0 if not pd[index] and not qd[index] else None,
                "rate_kl_relation": "mutually_singular" if pd[index] != qd[index] else "same_support",
                "rate_mean_l2": float(np.linalg.norm(difference[index])),
                "sampled_rate_l2": float(np.linalg.norm(np.asarray(p["velocity"])[index] - np.asarray(q["velocity"])[index])),
                "residual_l2": float(np.linalg.norm(np.asarray(p["residual"])[index] - np.asarray(q["residual"])[index])),
            })
    return {"schema": 2, "representation": "full_state_rate",
            "scope": "same_observed_input_before_optimizer_update",
            "value_scope": "each_policy_own_reward_contract", "rows": rows,
            "pairs": pairs, "vectors": vectors or None}
