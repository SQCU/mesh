"""Preparation-only tensor placement relations, shared by graph binding/lowering."""
import torch
from views import rebase


def output(shape, dtype, device, out=None):
    """Resolve an explicit output operand once, before command construction."""
    if out is None:
        from . import empty
        return empty(tuple(shape), dtype=dtype, device=device)
    if tuple(out.shape) != tuple(shape) or out.dtype != dtype or out.device.type != torch.device(device).type:
        raise ValueError('output operand must match the declared shape, dtype and device')
    return out


def same_view(a, b):
    return (a.dtype == b.dtype and tuple(a.shape) == tuple(b.shape)
            and torch._C._is_alias_of(a, b) and a.storage_offset() == b.storage_offset()
            and all(n <= 1 or x == y for n, x, y in zip(a.shape, a.stride(), b.stride())))


def indices(described, placed_strides, placed_offset, view):
    """Resolve an alias's element indices without allocating its destination."""
    width, target_width = described.element_size(), view.element_size()
    offset, strides = rebase((*described.shape, width), (*[s * width for s in described.stride()], 1),
                             described.storage_offset() * width, (*[s * width for s in placed_strides], 1),
                             placed_offset * width, (*view.shape, target_width),
                             (*[s * target_width for s in view.stride()], 1), view.storage_offset() * target_width)
    if offset % target_width or any(step % target_width for n, step in zip(view.shape, strides[:-1]) if n > 1) or strides[-1] != 1:
        raise ValueError('placed bytes do not form the declared typed elements')
    return offset // target_width, tuple(max(1, step // target_width) if n <= 1 else step // target_width
                                        for n, step in zip(view.shape, strides[:-1]))


def alias(described, placed, view):
    offset, strides = indices(described, placed.stride(), placed.storage_offset(), view)
    # A dtype view shares the same allocation; it does not convert values.
    width = described.element_size()
    storage = placed if placed.dtype == view.dtype else placed.as_strided((placed.untyped_storage().nbytes() // width,), (1,), 0).view(view.dtype)
    return storage.as_strided(tuple(view.shape), strides, offset)


class Placements:
    """Logical tensor storage mapped to fixed physical output operands."""
    def __init__(self):
        self.bindings = []

    def get(self, described):
        if not isinstance(described, torch.Tensor):
            return None
        failure = None
        for logical, physical in self.bindings:
            if torch._C._is_alias_of(logical, described):
                try:
                    return alias(logical, physical, described)
                except ValueError as error:
                    failure = error
        if failure is not None:
            raise failure
        return None

    def bind(self, described, placed):
        if (tuple(described.shape) != tuple(placed.shape) or described.dtype != placed.dtype
                or described.device.type != placed.device.type):
            raise ValueError('a destination must match its declared shape, dtype and device')
        previous = self.get(described)
        if previous is not None:
            if not same_view(previous, placed):
                raise ValueError('one tensor cannot occupy incompatible collective destinations')
        else:
            self.bindings.append((described, placed))
