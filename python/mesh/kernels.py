from dataclasses import dataclass


@dataclass(frozen=True)
class _Operation:
    op: int
    arity: int
    alpha: float = 1
    beta: float = 0


matmul = _Operation(6, 2)
add = _Operation(1, 2, beta=1)
multiply = _Operation(2, 2)
tanh = _Operation(3, 1)
exp = _Operation(4, 1)
row_sum = _Operation(5, 1)
rsqrt = _Operation(7, 1)
swish = _Operation(8, 1)


# design/algorithm-sources.md#single-kernel-interface
def affine(alpha=1, beta=0):
    return _Operation(0, 1, alpha, beta)
