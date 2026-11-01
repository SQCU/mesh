"""Higher-order reduction source, lowered into the ordinary page command graph.

Combine and identity run only during source capture. A declaration fixes the
value schema and permitted changes of association/order. The collective compiler
supplies actual page cells; the numerical compiler emits their combine graph.
There is no Python callable or reducer dispatch in the running function.
"""
import json
import math

import torch



def fold_topology(group, cells, element_bytes, laws, basis=None, counts=(), root=-1):
    """Realize source association once, before tracing its derivative.

    The retained columns are optimizer operands. Native reduction placement and
    this source expansion use the same decoder and value-domain allocation.
    Nothing in the running function traverses this preparation graph.
    """
    from mesh import ffi, lib, check
    size = group.size()
    basis = {} if basis is None else basis
    contributors = tuple(basis.get('contributors',
        tuple((1 << min(64, size - first)) - 1 for first in range(0, size, 64))))
    segment_count = len(counts) if counts else int(basis.get('segments', 1))
    parents = tuple(tuple(row) for row in basis.get('parents', ()))
    segments = tuple(basis.get('segment', (0,) * len(parents)))
    shares = tuple(basis.get('share', (1.0,) * len(parents)))
    arrays = [ffi.new('uint64_t[]', contributors), ffi.new('uint32_t[]', segments),
              ffi.new('uint32_t[]', [rank for row in parents for rank in row]),
              ffi.new('double[]', shares)]
    declaration = ffi.new('ncclMeshFoldBasis *', dict(nodes=size, laws=laws,
        segments=segment_count, trees=len(parents), contributors=arrays[0],
        segment=arrays[1], parents=arrays[2], share=arrays[3]))
    page = ffi.new('size_t *')
    check(lib.ncclMeshCommPage(group.comm, page))
    output = ffi.new('ncclMeshFoldTopology **')
    held_counts = ffi.new('uint64_t[]', counts) if counts else ffi.NULL
    check(lib.ncclMeshFoldPrepare(declaration, cells, element_bytes, page[0], held_counts, output))
    held = ffi.gc(output[0], lib.ncclMeshFoldDrop)
    identities = tuple(basis.get('identity_owner', tuple(range(segment_count)) if counts else (max(0, root),) * segment_count))
    topology = dict(nodes=size, laws=laws, contributors=contributors, segments=segment_count,
        identity_owner=identities,
        parents=tuple(tuple(int(held.basis.parents[t * size + r]) for r in range(size))
                      for t in range(held.basis.trees)),
        segment=tuple(int(held.basis.segment[t]) for t in range(held.basis.trees)),
        share=tuple(float(held.basis.share[t]) for t in range(held.basis.trees)))
    trees = []
    for t in range(held.basis.trees):
        if not held.cells[t]:
            continue
        nodes = tuple(tuple(int(getattr(held.node[i], name)) for name in ('left', 'right', 'owner', 'round'))
                      for i in range(held.node_first[t], held.node_first[t + 1]))
        trees.append(dict(first=int(held.first[t]), cells=int(held.cells[t]),
                          result=int(held.result[t]), nodes=nodes, column=t,
                          segment=int(held.basis.segment[t])))
    return dict(basis=topology, trees=tuple(trees), cells=cells, page=int(page[0]), counts=tuple(counts))


class Reduction:
    """A captured combine/identity and an explicit independent-value schema.

    ``combine(a,b)`` returns one value with ``element_shape`` and ``dtype``.
    ``identity`` is a scalar value or a function of a typed empty prototype.
    Reassociation and reordering are independent permissions supplied by the
    caller. Their defaults preserve the declared rank order and left fold.
    """
    def __init__(self, combine, identity, *, dtype, element_shape=(),
                 reassociate=False, reorder=False, topology=None):
        from torch._subclasses.fake_tensor import FakeTensorMode
        from .declarations import trace
        self.dtype, self.element_shape = dtype, tuple(element_shape)
        self.unit = math.prod(self.element_shape)
        self.flags = int(reassociate) | (int(reorder) << 1)
        self.topology = topology
        self.fold_choices = {}
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

    def __call__(self, value, group, *, root=-1, counts=()):
        """Declare this reduction in an ordinary Function source graph."""
        if value.dtype != self.dtype or (self.element_shape and tuple(value.shape[-len(self.element_shape):]) != self.element_shape):
            raise ValueError('collective operand does not match the supplied reduction schema')
        if not self.unit:
            return value.reshape((counts[group.rank()], *self.element_shape)) if counts else value
        cells = value.numel() // self.unit
        key = (group.size(), cells, tuple(counts), root, json.dumps(self.topology, sort_keys=True))
        if key not in self.fold_choices:
            self.fold_choices[key] = fold_topology(group, cells, self.dtype.itemsize * self.unit,
                                                  self.flags, self.topology, counts, root)
        chosen = self.fold_choices[key]
        from .compilation import fold_occurrence, source_choice
        occurrence = fold_occurrence()
        if occurrence is not None:
            basis = chosen['basis']
            family = dict(kind='fold', source=occurrence, dtype=str(self.dtype), element_shape=self.element_shape,
                combine=self.combine.code, identity=self.initializer.code, root=root, counts=tuple(counts),
                basis={name: data for name, data in basis.items() if name != 'share'})
            shares, choices = list(basis['share']), []
            for segment in range(basis['segments']):
                columns = [i for i, part in enumerate(basis['segment']) if part == segment]
                if not columns:
                    continue
                units = counts[segment] if counts else cells
                record = source_choice(dict(family, segment=segment, columns=columns),
                    [shares[i] for i in columns], grain=0, quantum=1 / units if units else 0)
                for column, share in zip(columns, record['weights'], strict=True):
                    shares[column] = share
                choices.append(record)
            if tuple(shares) != basis['share']:
                chosen = fold_topology(group, cells, self.dtype.itemsize * self.unit,
                    self.flags, dict(basis, share=shares), counts, root)
            chosen = dict(chosen, choices=choices)
        return self._source(value, group, chosen, root, tuple(counts))

    def _source(self, value, group, chosen, root, counts):
        """Emit the chosen fold as ordinary source and selected page transfers.

        Autograd sees the actual left and right operands of every combine. Its
        transpose exchanges cotangents; it never substitutes a gradient reduce
        for the derivative of an arbitrary supplied combine function.
        """
        from .declarations import select_exchange
        from torch.fx.experimental.proxy_tensor import get_proxy_mode, get_proxy_slot
        rank, size, cells = group.rank(), group.size(), chosen['cells']
        flat = value.reshape((cells, *self.element_shape))
        empty = flat[:0]

        def selection(runs):
            return dict(shape=[cells * self.unit], steps=[['select', 0,
                [(first * self.unit, count * self.unit, count * self.unit, 1) for first, count in runs]]])

        def exchange(source, sources, destinations, shape):
            result = select_exchange(source, list(shape), json.dumps(dict(
                sources=[selection(runs) for runs in sources],
                destinations=[selection(runs) for runs in destinations], rank=rank), separators=(',', ':')), group.group_name)
            proxy = get_proxy_mode()
            if proxy is not None:
                get_proxy_slot(result, proxy.tracer).proxy.node.meta['mesh_fold'] = chosen
            return result

        values, owners = [], [[] for _ in range(size)]
        for tree in chosen['trees']:
            first, count = tree['first'], tree['cells']
            local = flat[first:first + count]
            operands = [(local if r == rank else empty, r) for r in range(size)]
            for left, right, owner, _ in tree['nodes']:
                arguments = []
                for index in (left, right):
                    argument, producer = operands[index]
                    sources, destinations = [[] for _ in range(size)], [[] for _ in range(size)]
                    sources[producer] = destinations[owner] = [(first, count)]
                    arguments.append(exchange(argument, sources, destinations,
                        (count if rank == owner else 0, *self.element_shape)))
                operands.append((torch.vmap(self.combine)(*arguments), owner))
            if tree['result'] == (1 << 32) - 1:
                owner = chosen['basis']['identity_owner'][tree['segment']]
                result = torch.vmap(self.initializer)(local if rank == owner else empty)
            else:
                result, owner = operands[tree['result']]
            owners[owner].append((first, count))
            # Zero-width peer results retain the remote edge in the source
            # derivative. Dropping them would erase that rank's cotangent
            # exchanges even though its leaf contributed to the remote fold.
            values.append(result)
        source = torch.cat(values, dim=0) if values else empty
        if counts:
            start = 0
            destinations = []
            for count in counts:
                destinations.append([(start, count)] if count else [])
                start += count
            shape = (counts[rank], *self.element_shape)
        else:
            destinations = [[(0, cells)] if root < 0 or r == root else [] for r in range(size)]
            shape = tuple(value.shape) if root < 0 or rank == root else (0, *self.element_shape)
        return exchange(source, owners, destinations, shape)


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
