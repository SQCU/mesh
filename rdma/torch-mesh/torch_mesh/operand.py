"""Preparation-only operands over the shared layout/dependency algebra.

TensorView declares a view of bridge pages. Kernels, copies and API bindings
lower it to native arguments and accesses before a function runs. Layout
metadata describes accesses; it never snapshots tensor contents. A tensor of
page indices can be written and read by the compiled function like other data.
"""
from dataclasses import dataclass

from . import _stream


@dataclass(frozen=True, eq=False)
class TensorView:
    storage: object
    layout: _stream.TensorLayout

    @classmethod
    def of(cls, value):
        if isinstance(value, cls):
            return value
        return cls(value, _stream.TensorLayout(tuple(value.shape), tuple(value.stride()), value.storage_offset()))

    @property
    def shape(self):
        return tuple(self.layout.shape)

    @property
    def dtype(self):
        return self.storage.dtype

    @property
    def device(self):
        return self.storage.device

    def numel(self):
        return self.layout.count

    def element_size(self):
        return self.storage.element_size()

    def slice(self, axis, first, count, step=1):
        return TensorView(self.storage, self.layout.slice(axis, first, count, step))

    def select(self, axis, indices):
        return TensorView(self.storage, self.layout.select(axis, indices))

    def gather(self, shape, indices):
        return TensorView(self.storage, self.layout.gather(shape, indices))

    def permute(self, axes):
        return TensorView(self.storage, self.layout.permute(axes))

    def reshape(self, shape):
        return TensorView(self.storage, self.layout.reshape(shape))

def descriptor(value):
    return TensorView.of(value).layout


def requirement(value, declared):
    """Resolve an explicit API requirement without allocating or copying values."""
    return descriptor(value).required(declared if isinstance(declared, str) else 'fixed',
                                      [] if isinstance(declared, str) else tuple(declared))


def requirements(value, declarations):
    return descriptor(value).requirements([
        (item, ()) if isinstance(item, str) else ('fixed', tuple(item)) for item in declarations])
