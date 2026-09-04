import hashlib

SCALE_RANK = 2048
SCALE_HIDDEN = 4096
SCALE_EXPERTS = 32
SCALE_TOPK = 2

def strategy_widths(
    d_scale=SCALE_RANK,
    scale_h=SCALE_HIDDEN,
    scale_experts=SCALE_EXPERTS,
    scale_topk=SCALE_TOPK,
):
    from payload.tools.strategy_io_schema import XAN_WIDTH
    from solver.strat.cast_header import Widths
    from solver.strat.featurize import SLOT_DIM
    from solver.strat.instruments import DESCRIPTOR_WIDTH

    return Widths(
        XAN_WIDTH, DESCRIPTOR_WIDTH, SLOT_DIM,
        d_scale=d_scale, scale_h=scale_h,
        scale_experts=scale_experts, scale_topk=scale_topk,
    )

def scale_model_digest(model):
    import numpy as np
    from mlx.utils import tree_flatten

    digest = hashlib.sha256()
    if model is None:
        return digest.hexdigest()
    for name, value in sorted(tree_flatten(model.parameters())):
        if not name.startswith("scale_"):
            continue
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(name.encode())
        digest.update(str(array.shape).encode())
        digest.update(array.view(np.uint8))
    return digest.hexdigest()

__all__ = [
    "SCALE_RANK", "SCALE_HIDDEN", "SCALE_EXPERTS", "SCALE_TOPK",
    "strategy_widths", "scale_model_digest",
]
