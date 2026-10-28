"""Preparation-only tensor placement relations, shared by graph binding/lowering."""
import torch
from .operand import TensorView, descriptor, rebase_typed


def output(shape, dtype, device, out=None):
    """Resolve an explicit output operand once, before command construction."""
    if out is None:
        from . import empty
        return empty(tuple(shape), dtype=dtype, device=device)
    if tuple(out.shape) != tuple(shape) or out.dtype != dtype or out.device.type != torch.device(device).type:
        raise ValueError('output operand must match the declared shape, dtype and device')
    return out


def same_view(a, b):
    from . import _stream
    return _stream.same_operand(a, b)


def relation(described, placed, view):
    """Compose typed aliases with assigned pages using the shared algebra."""
    width, target_width = described.element_size(), view.element_size()
    return rebase_typed(descriptor(view), target_width, descriptor(described), width, placed)


def alias(described, placed, view):
    placed = TensorView.of(placed)
    layout = relation(described, placed.layout, view)
    return TensorView(placed.storage_as(view.dtype), layout)


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
