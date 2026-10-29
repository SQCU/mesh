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

    @property
    def ndim(self):
        return len(self.shape)

    def dim(self):
        return self.ndim

    def stride(self, axis=None):
        # Strides are a requirement of a particular foreign API. Indexed
        # numerical consumers use layout.declaration instead.
        if not self.layout.affine:
            raise ValueError('the consumer requested strides for an indexed operand')
        steps = tuple(self.layout.strides)
        return steps if axis is None else steps[axis]

    def storage_offset(self):
        return self.layout.offset

    def is_contiguous(self):
        from .layout import strides
        return self.layout.matches(strides(self.shape))

    def is_floating_point(self):
        return self.dtype.is_floating_point

    def storage_as(self, dtype):
        """A scalar handle for the same allocation; no values are converted."""
        if dtype == self.dtype:
            return self.storage
        width = self.storage.untyped_storage().nbytes() // self.element_size()
        return self.storage.as_strided((width,), (1,), 0).view(dtype)

    def typed(self, dtype):
        """Reinterpret equal-width scalars without changing their page selection."""
        if dtype.itemsize != self.element_size():
            raise ValueError('a scalar reinterpretation preserves its declared width')
        return TensorView(self.storage_as(dtype), self.layout)

    def tensor(self):
        """Bind an API which explicitly requires a strided tensor handle."""
        if not self.layout.affine:
            raise ValueError('this API binding requires a strided layout')
        return self.storage.as_strided(self.shape, tuple(self.layout.strides), self.layout.offset)

    def narrow(self, axis, first, count):
        return self.slice(axis % self.ndim, first, count)

    def split(self, sizes, axis=0):
        first, result = 0, []
        for size in sizes:
            result.append(self.narrow(axis, first, size))
            first += size
        return tuple(result)

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

    def t(self):
        return self.permute((1, 0))

    def rows(self, axis):
        """Put one logical axis last for row-wise numerical kernels."""
        import math
        axis %= self.ndim
        axes = (*[d for d in range(self.ndim) if d != axis], axis)
        width = self.shape[axis]
        rows = math.prod(self.shape[d] for d in axes[:-1])
        return self.permute(axes).reshape((rows, width))

    def transfer(self):
        """The common WRITE descriptor, retaining order and repeated reads."""
        from mesh import view_layout
        size = self.element_size()
        if self.layout.affine:
            shift, rows, spans = view_layout(self.shape, [s * size for s in self.layout.strides], size)
            first = self.layout.offset + shift // size
        else:
            runs = self.layout.ordered_spans()
            first = min((at for at, _ in runs), default=self.layout.offset)
            rows, spans = (0, 0, 0), [((at - first) * size, count * size) for at, count in runs]
        base = self.storage.as_strided((int(bool(self.numel())),), (1,), first)
        return base, rows, spans

    def declaration(self, *requirements):
        return dict(source=descriptor(self.storage).state, view=self.layout.state, requirements=requirements)


def ordered_runs(segments):
    """Concatenate page selections as logical intervals for collective cuts."""
    runs, cursor = [], 0
    for identity, value in segments:
        for offset, count in descriptor(value).ordered_spans():
            runs.append((cursor, cursor + count, identity, offset))
            cursor += count
    return runs, cursor

def restore(state):
    shape, steps, offset, maps, index_shape = state
    return _stream.TensorLayout(shape, steps, offset, {axis: values for axis, values in enumerate(maps) if values}, None, index_shape)


def bind(value, declaration):
    """Compose the declared use with the producer's assigned pages once."""
    placed = TensorView.of(value)
    view = restore(declaration['view']).rebase(restore(declaration['source']), placed.layout)
    return TensorView(placed.storage, view)

def descriptor(value):
    return TensorView.of(value).layout


def rebase_typed(view, width, declared, source_width, placed):
    """Compose in scalar units when possible; bytes only for a type view."""
    if width == source_width:
        return view.rebase(declared, placed)
    return view.components(width).rebase(declared.components(source_width), placed.components(source_width)).elements(width)


def requirement(value, declared):
    """Resolve an explicit API requirement without allocating or copying values."""
    return descriptor(value).required(declared if isinstance(declared, str) else 'fixed',
                                      [] if isinstance(declared, str) else tuple(declared))


def requirements(value, declarations):
    return descriptor(value).requirements([
        (item, ()) if isinstance(item, str) else ('fixed', tuple(item)) for item in declarations])


def signature(buffers, outputs, requirements):
    """One operand schema for GPU and host declarations; no payload is read."""
    import json
    views = {i: TensorView.of(value) for i, value in enumerate(buffers) if value is not None}
    declarations = {i: view.declaration(*requirements[i]) for i, view in views.items()}
    results, slots = [], []
    for index in outputs:
        value = views[index].storage
        found = next((j for j, held in enumerate(results) if held is value), None)
        if found is None:
            results.append(value); slots.append([]); found = len(results) - 1
        slots[found].append(index)
    inputs = [i for i in views if i not in outputs]
    return dict(inputs=[views[i].storage for i in inputs], input_slots=inputs, output_slots=slots,
                output_strides=[list(value.stride()) for value in results], shapes=[list(value.shape) for value in results],
                dtypes=[str(value.dtype).removeprefix('torch.') for value in results], devices=[str(value.device) for value in results],
                operands=json.dumps(declarations, separators=(',', ':')))


def fake_outputs(shapes, strides, dtypes, devices):
    import torch
    return [torch.empty_strided(shape, steps, dtype=getattr(torch, dtype), device=device)
            for shape, steps, dtype, device in zip(shapes, strides, dtypes, devices, strict=True)]


def graph_uses(node):
    """Declared input/output selections and API needs, before page placement."""
    import json
    from torch.utils._pytree import tree_flatten
    from .compilation import api_layouts
    schema = getattr(node.target, '_schema', None)
    if schema is None:
        return
    args = {a.name: node.args[i] if i < len(node.args) else node.kwargs.get(a.name, a.default_value)
            for i, a in enumerate(schema.arguments)}
    if schema.name in ('torch_mesh::native_kernel', 'torch_mesh::host_kernel'):
        declarations = json.loads(args['operands'])
        bindings = [(source.meta['val'], slot) for source, slot in zip(args['inputs'], args['input_slots'], strict=True)]
        bindings.extend((value, slot) for value, slots in zip(node.meta['val'], args['output_slots'], strict=True) for slot in slots)
        for value, slot in bindings:
            declaration = declarations[str(slot)]
            view = restore(declaration['view']).rebase(restore(declaration['source']), descriptor(value))
            for requirement in declaration['requirements']:
                yield value, view, requirement
        return
    declaration = api_layouts('graph_operands').get(schema.name)
    if declaration is None:
        return
    layouts = api_layouts(declaration['api'])
    for name, role in declaration['inputs'].items():
        for source in tree_flatten(args[name])[0]:
            if source is None:
                continue
            value = source.meta['val']
            view = descriptor(value)
            axes = declaration.get('permutations', {}).get(name)
            if axes is not None:
                view = view.permute(axes)
            yield value, view, layouts[role]
    for value in tree_flatten(node.meta['val'])[0]:
        yield value, descriptor(value), layouts[declaration['output']]
