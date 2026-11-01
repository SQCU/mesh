"""Higher-order reduction source, lowered into the ordinary page command graph.

Combine and identity run only during source capture. A declaration fixes the
value schema and permitted changes of association/order. The collective compiler
supplies actual page cells; the numerical compiler emits their combine graph.
There is no Python callable or reducer dispatch in the running function.
"""
import itertools
import math
import weakref

import torch


_identities = itertools.count()
_declarations = weakref.WeakValueDictionary()


@torch.library.custom_op('torch_mesh::reduce_function', mutates_args=())
def operation(value: torch.Tensor, declaration: int, group_name: str, root: int,
              counts: list[int]) -> torch.Tensor:
    raise RuntimeError('a reduction function is source for whole-function preparation')


@operation.register_fake
def _(value, declaration, group_name, root, counts):
    declared = _declarations[declaration]
    if counts:
        from .backend import _mesh
        return value.new_empty((counts[_mesh(group_name, 'reduction').rank()], *declared.element_shape))
    return torch.empty_like(value)


def declaration(node, arguments):
    """Resolve the captured source operand before collective preparation."""
    return node.meta.get('mesh_reduction') or _declarations[arguments['declaration']]


class Reduction:
    """A captured combine/identity and an explicit independent-value schema.

    ``combine(a,b)`` returns one value with ``element_shape`` and ``dtype``.
    ``identity`` is a scalar value or a function of a typed empty prototype.
    Reassociation and reordering are independent permissions supplied by the
    caller. Their defaults preserve the declared rank order and left fold.
    """
    def __init__(self, combine, identity, *, dtype, element_shape=(),
                 reassociate=False, reorder=False, configuration=None):
        from torch._subclasses.fake_tensor import FakeTensorMode
        from .declarations import trace
        self.dtype, self.element_shape = dtype, tuple(element_shape)
        self.unit = math.prod(self.element_shape)
        self.flags = int(reassociate) | (int(reorder) << 1)
        self.configuration = configuration
        self.identity = next(_identities)
        _declarations[self.identity] = self
        mode = FakeTensorMode(allow_non_fake_inputs=True)
        with mode:
            prototype = torch.empty(self.element_shape, dtype=dtype, device='meta')
            self.combine = trace(combine, prototype, torch.empty_like(prototype))
            initializer = identity if callable(identity) else lambda value: torch.full_like(value, identity)
            self.initializer = trace(initializer, prototype)
        for graph in (self.combine, self.initializer):
            result = next(node.args[0] for node in graph.graph.nodes if node.op == 'output')
            value = result.meta['val']
            if tuple(value.shape) != self.element_shape or value.dtype != dtype:
                raise ValueError('reduction source must return its declared independent-value schema')
        self.templates = {}

    def __call__(self, value, group, *, root=-1, counts=()):
        """Declare this reduction in an ordinary Function source graph."""
        from torch.fx.experimental.proxy_tensor import get_proxy_mode, get_proxy_slot
        if value.dtype != self.dtype or (self.element_shape and tuple(value.shape[-len(self.element_shape):]) != self.element_shape):
            raise ValueError('collective operand does not match the supplied reduction schema')
        if not self.unit:
            return value.reshape((counts[group.rank()], *self.element_shape)) if counts else value
        result = operation(value, self.identity, group.group_name, root, list(counts))
        proxy = get_proxy_mode()
        if proxy is not None:
            get_proxy_slot(result, proxy.tracer).proxy.node.meta['mesh_reduction'] = self
        return result

    def operator(self, group):
        return self.binding(group).operator

    def binding(self, group):
        """Register only a preparation callback, retained by the communicator."""
        from . import _stream
        from .backend import TYPES
        key = ('function', self.identity)
        if key not in group._operators:
            group._operators[key] = _stream.Reduction(self.lower, self.dtype, self.element_shape, self.flags,
                                                      TYPES[self.dtype], group.handle)
        return group._operators[key]

    def lower(self, sources, destination, schedule, effects):
        """Bind one resolved collective cell into the existing recorder."""
        from .compilation import bind
        from .declarations import BoundGraph, trace
        from .ownership import prepare_expression
        cells, count = destination.shape[0], len(sources)
        supplied = (*sources, destination)
        identities, starts = {}, {}
        for value in supplied:
            storage = value.untyped_storage()._cdata
            identities.setdefault(storage, len(identities))
            starts[storage] = min(starts.get(storage, value.storage_offset()), value.storage_offset())
        aliases = tuple((identities[value.untyped_storage()._cdata],
                         value.storage_offset() - starts[value.untyped_storage()._cdata]) for value in supplied)
        key = cells, count, aliases
        with bind(self.configuration):
            if key not in self.templates:
                def fold(*values):
                    *inputs, output = values
                    if not count:
                        result = torch.vmap(self.initializer)(output)
                    else:
                        result = inputs[0]
                        for value in inputs[1:]:
                            result = torch.vmap(self.combine)(result, value)
                    output.copy_(result)
                    return output
                graph = trace(fold, *supplied)
                arguments = tuple(node.meta['val'] for node in graph.graph.nodes if node.op == 'placeholder')
                # This is immutable compiler IR, not retained invocation state.
                self.templates[key] = dict(graph=graph, operands=arguments)
            template = self.templates[key]
            prepared = prepare_expression(template)
            graph = BoundGraph(template['graph'], supplied, None, self.configuration,
                               destinations=(destination,), prepared=prepared,
                               page_effects=dict(effects, sources=sources, destination=destination, unit=self.unit))
            # These inputs are produced within the enclosing graph. Their
            # typed prior/next versions come from the collective's source cells.
            graph.execution.inputs = ()
            graph.execution.append(schedule)


def sum(dtype, *, element_shape=(), reassociate=False, reorder=False):
    return Reduction(torch.add, 0, dtype=dtype, element_shape=element_shape,
                     reassociate=reassociate, reorder=reorder)


def maximum(dtype, *, element_shape=(), reassociate=False, reorder=False):
    identity = -math.inf if dtype.is_floating_point else False if dtype == torch.bool else torch.iinfo(dtype).min
    return Reduction(torch.maximum, identity, dtype=dtype, element_shape=element_shape,
                     reassociate=reassociate, reorder=reorder)


def partials(dtype, width, *, reassociate=True, reorder=True):
    """The stable [maximum, mass, weighted values] merge as supplied source."""
    def combine(left, right):
        present_left, present_right = left[1] > 0, right[1] > 0
        maximum = torch.maximum(torch.where(present_left, left[0], -math.inf),
                                torch.where(present_right, right[0], -math.inf))
        maximum = torch.where(present_left | present_right, maximum, 0)
        a = torch.where(present_left, (left[0] - maximum).exp(), 0)
        b = torch.where(present_right, (right[0] - maximum).exp(), 0)
        return torch.cat((maximum.reshape(1), left[1:] * a + right[1:] * b))
    return Reduction(combine, 0, dtype=dtype, element_shape=(width,),
                     reassociate=reassociate, reorder=reorder)
