from .instruments import DESCRIPTOR_WIDTH, KINDS
from .baselines import BASELINE_OUTPUT_WIDTH
from .policy_contract import is_matrix_fusion_arm

def mm(a, b, c):
    return 2 * int(a) * int(b) * int(c)

def dpp_work(instruments, rank):
    rows = int(instruments)
    width = int(rank)
    covariance = mm(width, rows, width)
    conjugate_gradient = width * mm(width, width, rows)
    marginal = 2 * rows * width
    return {
        "forward_flops": covariance + conjugate_gradient + marginal,
        "covariance_flops": covariance,
        "conjugate_gradient_flops": conjugate_gradient,
        "conjugate_gradient_steps": width,
        "marginal_flops": marginal,
        "intermediate_words": width * width + 6 * rows * width,
    }

def _envelope(forward, parameter_words, inputs, intermediates, outputs,
              gradient_steps=0, gradient_batch=0):
    training = int(gradient_steps) * int(gradient_batch)
    lower_flops = int(forward)
    upper_flops = lower_flops * (1 + 6 * training)
    lower_bytes = 4 * (int(inputs) + int(parameter_words) + int(outputs))
    upper_bytes = 4 * (int(inputs) + int(parameter_words) + int(intermediates) + int(outputs)) + 8 * upper_flops
    return {
        "lower_flops": lower_flops,
        "upper_flops": upper_flops,
        "lower_bytes": lower_bytes,
        "upper_bytes": upper_bytes,
        "forward_flops": lower_flops,
        "training_forwards": training,
        "parameter_bytes": 4 * int(parameter_words),
    }

def scale_work(widths, rows, gradient_steps=0, gradient_batch=0):
    physical = int(rows)
    topk = widths.scale_topk
    forward = (
        mm(physical, widths.d_ir, widths.d_scale)
        + mm(physical, widths.d_scale, widths.scale_experts)
        + topk * mm(physical, widths.d_scale, widths.scale_h)
        + topk * mm(physical, widths.scale_h, widths.d_scale)
        + mm(widths.d_scale, physical, widths.d_scale)
        + mm(widths.d_scale, widths.d_scale, 1)
        + mm(physical, widths.d_scale, widths.d_ir)
    )
    parameter_words = (
        2 * widths.d_ir * widths.d_scale
        + widths.d_scale * widths.scale_experts
        + 2 * widths.scale_experts * widths.d_scale * widths.scale_h
        + widths.d_scale
    )
    inputs = physical * widths.d_ir
    outputs = physical * widths.d_ir + widths.scale_experts + 5
    intermediates = (
        physical * (2 * widths.d_scale + topk * widths.scale_h)
        + widths.d_scale * widths.d_scale + widths.scale_experts
    )
    out = _envelope(forward, parameter_words, inputs, intermediates, outputs,
                    gradient_steps, gradient_batch)
    out.update({
        "residual_rows": physical,
        "residual_rank": widths.d_scale,
        "experts": widths.scale_experts,
        "topk": topk,
    })
    return out

def strategy_work(arm, widths, players, instruments, cells, baseline_hidden=256,
                  gradient_steps=0, gradient_batch=0):
    l, m, c = int(players), int(instruments), int(cells)
    if is_matrix_fusion_arm(arm):
        w = widths
        n = l * m
        scale = scale_work(w, n, gradient_steps, gradient_batch)
        dpp = dpp_work(m, w.d)
        topk = scale["topk"]
        forward = (
            mm(c, w.d_c, w.d_beta)
            + mm(l, c, w.d_beta)
            + mm(l, w.d_x + w.d_beta + w.d_sem, w.d)
            + mm(m, w.d_z, w.d)
            + mm(m, w.d_z, w.d_v)
            + mm(l, m, w.d)
            + mm(l, w.d, w.r + w.r_e)
            + mm(l, l, w.r + w.r_e)
            + mm(l, w.d, w.d_ir)
            + mm(l, l, w.d_ir)
            + mm(m, w.d_v, w.d_ir)
            + mm(l * m, w.d_ir, 2 * w.h)
            + mm(l * m, w.h, 1)
            + mm(l * m, w.d_ir, 6)
            + mm(l, w.d, w.d_y)
            + mm(l * m, w.d_ir, w.d_u)
            + 2 * mm(l, w.d_y, w.d_y)
            + 2 * mm(l, w.d_y, w.d_y * w.d_u)
            + 2 * mm(l * m, w.d_u, w.d_y)
            + mm(l * m, w.d_y, w.d)
            + dpp["forward_flops"]
        )
        parameter_words = (
            w.d_c * w.d_beta
            + (w.d_x + w.d_beta + w.d_sem) * w.d
            + w.d_z * (w.d + w.d_v)
            + w.d * w.d_ir + w.d_v * w.d_ir
            + w.d * (w.r + w.r_e)
            + 2 * w.d_ir * w.h + w.h
            + 8 * w.d_ir + 2 * w.d
            + w.d * w.d_y + w.d_ir * w.d_u + w.d_y * w.d
            + 2 * w.d_y * w.d_y + 2 * w.d_y * w.d_y * w.d_u
        )
        inputs = l * w.d_x + m * w.d_z + c * w.d_c + l * c + l * w.d_sem + 2 * l * m
        intermediates = (
            c * w.d_beta + l * (w.d_beta + 2 * w.d + w.d_y)
            + m * (w.d + w.d_v) + 3 * l * m + 2 * l * l
            + l * m * (w.d_ir + 2 * w.d_y + 6) + l * w.d_y * w.d_u
            + dpp["intermediate_words"]
        )
        local = _envelope(forward, parameter_words, inputs, intermediates, l * 4,
                          gradient_steps, gradient_batch)
        combined = {
            key: local[key] + scale[key]
            for key in ("lower_flops", "upper_flops", "lower_bytes", "upper_bytes",
                        "forward_flops", "parameter_bytes")
        }
    else:
        input_width = widths.d_x + widths.d_sem + widths.d_c
        kind_width = len(KINDS)
        output_width = BASELINE_OUTPUT_WIDTH
        belief = mm(l, c, widths.d_c)
        network = mm(l, input_width, baseline_hidden) + mm(l, baseline_hidden, output_width) if arm == "ffn" else mm(l, input_width, output_width) if arm != "default" else 0
        scores = mm(l, kind_width, m) + mm(l, DESCRIPTOR_WIDTH, m) if arm != "default" else 0
        forward = belief + network + scores
        parameter_words = (
            input_width * baseline_hidden + baseline_hidden * output_width
            if arm == "ffn" else input_width * output_width
            if arm != "default" else 0
        )
        inputs = l * widths.d_x + m * widths.d_z + c * widths.d_c + l * c + l * widths.d_sem + 2 * l * m
        intermediates = l * (widths.d_c + input_width + output_width) + 3 * l * m
        local = _envelope(forward, parameter_words, inputs, intermediates, l * 4,
                          gradient_steps, gradient_batch)
        scale = {key: 0 for key in local}
        dpp = {
            "forward_flops": 0,
            "covariance_flops": 0,
            "conjugate_gradient_flops": 0,
            "conjugate_gradient_steps": 0,
            "marginal_flops": 0,
            "intermediate_words": 0,
        }
        combined = dict(local)
    return {
        **combined,
        "training_forwards": local["training_forwards"],
        "residual_rows": l * m if is_matrix_fusion_arm(arm) else 0,
        "residual_rank": widths.d_scale if is_matrix_fusion_arm(arm) else 0,
        "experts": widths.scale_experts if is_matrix_fusion_arm(arm) else 0,
        "topk": widths.scale_topk if is_matrix_fusion_arm(arm) else 0,
        "local": local,
        "scale": scale,
        "dpp": dpp,
    }

__all__ = ["dpp_work", "scale_work", "strategy_work"]
