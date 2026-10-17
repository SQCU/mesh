SCALE_RANK = 2048
SCALE_HIDDEN = 4096
SCALE_EXPERTS = 32
SCALE_TOPK = 2

def strategy_widths(
    state_width,
    d_scale=SCALE_RANK,
    scale_h=SCALE_HIDDEN,
    scale_experts=SCALE_EXPERTS,
    scale_topk=SCALE_TOPK,
):
    from solver.strat.cast_header import Widths
    from payload.tools.strategy_io_schema import EVT_WIDTH

    return Widths(
        state_width, EVT_WIDTH,
        d_scale=d_scale, scale_h=scale_h,
        scale_experts=scale_experts, scale_topk=scale_topk,
    )

__all__ = [
    "SCALE_RANK", "SCALE_HIDDEN", "SCALE_EXPERTS", "SCALE_TOPK",
    "strategy_widths",
]
