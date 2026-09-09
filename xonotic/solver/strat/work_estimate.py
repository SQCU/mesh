from .policy_contract import is_matrix_fusion_arm

def mm(a, b, c):
    return 2 * int(a) * int(b) * int(c)

def cross_work(left_rank, right_rank, rows, gradient_steps=0, gradient_batch=0):
    left, right, count = int(left_rank), int(right_rank), int(rows)
    return {**_envelope(mm(left, count, right), 0, count * (left + right), 0, left * right,
                        gradient_steps, gradient_batch),
            'left_rank': left, 'right_rank': right, 'rows': count}

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
    n, w = int(rows), widths
    forward = (mm(n, w.d_ir, w.d_scale) + mm(n, w.d_scale, w.scale_experts)
        + 2 * w.scale_topk * mm(n, w.d_scale, w.scale_h)
        + w.scale_topk * mm(n, w.scale_h, w.d_scale) + mm(n, w.d_scale, w.d_ir))
    parameters = 2 * w.d_ir * w.d_scale + w.d_scale * w.scale_experts + 3 * w.scale_experts * w.d_scale * w.scale_h
    return {**_envelope(forward, parameters, n * w.d_ir,
        n * (2 * w.d_scale + 2 * w.scale_topk * w.scale_h + w.scale_experts),
        n * w.d_ir + w.scale_experts + 4, gradient_steps, gradient_batch),
        "residual_rows": n, "residual_rank": w.d_scale,
        "experts": w.scale_experts, "topk": w.scale_topk}


def strategy_work(arm, widths, players, events, baseline_hidden=256,
                  gradient_steps=0, gradient_batch=0, remote_scale_operation="gram", state_pages=1,
                  observations=None, carts=0, teams=0, navigation_nodes=0, navigation_edges=0,
                  navigation_cells=0, neighbors=0):
    n, e, w = int(players), int(events), widths
    o = n if observations is None else int(observations)
    p, j, k = n * int(state_pages), int(carts), int(teams)
    vn, ve, vc, slots = map(int, (navigation_nodes, navigation_edges, navigation_cells, neighbors))
    tokens, sources, groups = o + p + j + k, e + vn + ve + vc, o + n + j + k
    common_rows = n + p
    main = is_matrix_fusion_arm(arm)
    hidden, d = (w.d_ir, w.d) if main else (int(baseline_hidden), int(baseline_hidden))
    page_features = 6 * w.d_x + 20
    terms = {"raw_projections": mm(o, w.d_obs + 2, d) + mm(p, page_features, d)
        + mm(j, 18, d) + mm(k, 7, d) + mm(e, w.d_c, d)
        + mm(vn, 4, d) + mm(ve, 3, d) + mm(vc, 2, d),
        "local_values_output": mm(sources + 1, d, d) + mm(o, d, d),
        "local_value_contraction": mm(o, slots, d),
        "common_heads": mm(common_rows, hidden, 2 * w.d_x + 2),
        "rate_owner_contribution": 2 * p * w.d_x,
        "value_scalar_readout": 2 * p}
    neighborhood_parameters = 9 * d + (3 if main else 2) * d * d + 1
    raw_parameters = (w.d_obs + 2 + page_features + 18 + 7 + w.d_c) * d
    readout_parameters = hidden * (2 * w.d_x + 2)
    parameters = raw_parameters + neighborhood_parameters + readout_parameters
    scale = scale_work(w, tokens, gradient_steps, gradient_batch) if main else _envelope(0, 0, 0, 0, 0)
    cross = cross_work(w.r_e, w.d_ir, tokens, gradient_steps, gradient_batch) if main else _envelope(0, 0, 0, 0, 0)
    if main:
        terms.update({"local_metric": mm(sources + 1, d, d) + mm(o, d, d),
            "local_gram_contraction": mm(o, slots, d),
            "input_swiglu": 3 * mm(tokens, d, w.h),
            "ir_projection": mm(tokens, d, w.d_ir),
            "gram_projections": mm(tokens, d, w.r + w.r_e),
            "global_cross": cross["forward_flops"],
            "global_apply": mm(tokens, w.r_e, w.d_ir),
            "team_page_cross": mm(p, w.r, w.d_ir),
            "team_outer_products": (o + j + k) * w.r * w.d_ir,
            "team_membership": 2 * mm(k, groups, w.r * w.d_ir),
            "team_apply": mm(tokens, w.r, w.d_ir),
            "output_swiglu": 3 * mm(tokens, w.d_ir, w.h)})
        parameters += 3 * d * w.h + d * (w.r + w.r_e + w.d_ir) + 3 * w.d_ir * w.h
    else:
        terms.update({"baseline_global_sum": max(tokens - 1, 0) * hidden})
        parameters += 5 * hidden
    if arm == "default":
        terms, parameters = {}, 0
    forward = sum(terms.values())
    inputs = p * page_features + o * (w.d_obs + 2) + j * 18 + k * 7 + e * w.d_c + vn * 4 + ve * 3 + vc * 2 + o * slots * 3
    base = _envelope(forward, parameters, inputs,
        common_rows * (2 * w.d_x + 2) + tokens * 6 * hidden + (sources + 1) * 3 * d + o * slots * d,
        3 * p * w.d_x + common_rows * hidden + n * d + 3 * n
        + (common_rows * (w.r + w.r_e) if main else 0), gradient_steps, gradient_batch)
    keys = ("lower_flops", "upper_flops", "lower_bytes", "upper_bytes", "forward_flops", "parameter_bytes")
    combined = {key: base[key] + scale[key] for key in keys}
    remote = scale if main and remote_scale_operation == "block" else cross
    local = {key: combined[key] - remote[key] for key in keys}
    return {**combined, "representation": "raw_rows_state_rate", "estimate": "logical_contractions_with_training_envelope",
        "state_width": w.d_x * int(state_pages), "page_width": w.d_x, "state_pages": int(state_pages),
        "residual_rows": tokens, "common_head_rows": common_rows, "common_head_width": 2 * w.d_x + 2,
        "page_feature_width": page_features, "local_source_rows": sources, "observation_rows": o,
        "event_rows": e, "neighbor_slots": slots, "terms": terms, "local": local, "scale": scale, "remote": remote}
