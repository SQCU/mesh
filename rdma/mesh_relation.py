"""Finite index relations for graph importers, during preparation only.

This binds the same ISL archive as MetalTensorLayout. Its objects and integers
use mesh's page allocator. No relation object is an execution operand.
"""
import ctypes as C
import json
import math
import os
from pathlib import Path


class _Cell(C.Structure):
    _fields_ = [('domain', C.c_void_p), ('members', C.POINTER(C.c_size_t)), ('count', C.c_size_t)]


def coordinates(shape, stem):
    return [f'{stem}{i}' for i in range(len(shape))]


def bounds(shape, names):
    return [f'0 <= {name} < {int(n)}' for name, n in zip(names, shape)]


def flat(shape, names):
    value = '0'
    for n, name in zip(shape, names):
        value = f'({value}) * {int(n)} + {name}'
    return value


class Algebra:
    """One preparation context; the library location is an explicit build operand."""
    def __init__(self, library=None):
        directory = Path(os.environ.get('MESH_LIBDIR', Path(__file__).parent))
        self.lib = C.CDLL(str(library or directory / 'libnccl-mesh.dylib'))
        specs = {
            'ctx_alloc': (C.c_void_p, ()), 'ctx_free': (None, (C.c_void_p,)),
            'ctx_last_error_msg': (C.c_char_p, (C.c_void_p,)),
            'val_get_num_si': (C.c_long, (C.c_void_p,)), 'val_free': (C.c_void_p, (C.c_void_p,)),
        }
        for kind in ('map', 'set'):
            specs.update({f'{kind}_read_from_str': (C.c_void_p, (C.c_void_p, C.c_char_p)),
                          f'{kind}_copy': (C.c_void_p, (C.c_void_p,)),
                          f'{kind}_free': (C.c_void_p, (C.c_void_p,)),
                          f'{kind}_to_str': (C.c_void_p, (C.c_void_p,)),
                          f'{kind}_is_equal': (C.c_int, (C.c_void_p, C.c_void_p)),
                          f'{kind}_is_empty': (C.c_int, (C.c_void_p,)),
                          f'{kind}_union': (C.c_void_p, (C.c_void_p, C.c_void_p))})
        for name in ('map_apply_range', 'map_intersect_range', 'map_intersect_domain', 'map_intersect', 'map_subtract_domain',
                     'set_intersect', 'set_subtract', 'set_flat_product'):
            specs[name] = C.c_void_p, (C.c_void_p, C.c_void_p)
        specs['map_domain'] = C.c_void_p, (C.c_void_p,)
        specs['map_range'] = C.c_void_p, (C.c_void_p,)
        specs['map_reverse'] = C.c_void_p, (C.c_void_p,)
        specs['map_lexmin'] = C.c_void_p, (C.c_void_p,)
        specs['map_is_single_valued'] = C.c_int, (C.c_void_p,)
        specs['map_dim'] = C.c_int, (C.c_void_p, C.c_int)
        specs['set_lexmin'] = C.c_void_p, (C.c_void_p,)
        specs['set_project_out'] = C.c_void_p, (C.c_void_p, C.c_int, C.c_uint, C.c_uint)
        for name in ('set_dim_min_val', 'set_dim_max_val'):
            specs[name] = C.c_void_p, (C.c_void_p, C.c_int)
        for name, (result, arguments) in specs.items():
            function = getattr(self.lib, 'isl_' + name)
            function.restype, function.argtypes = result, arguments
            setattr(self, name, function)
        self.lib.mesh_isl_free.argtypes, self.lib.mesh_isl_free.restype = (C.c_void_p,), None
        self.lib.MeshIndexSetRuns.argtypes = C.c_void_p, C.POINTER(C.c_size_t)
        self.lib.MeshIndexSetRuns.restype = C.POINTER(C.c_int64)
        self.lib.MeshIndexSetCover.argtypes = C.POINTER(C.c_void_p), C.c_size_t, C.POINTER(C.c_size_t)
        self.lib.MeshIndexSetCover.restype = C.POINTER(_Cell)
        self.lib.MeshIndexCompose.argtypes = C.c_void_p, C.c_void_p
        self.lib.MeshIndexCompose.restype = C.c_void_p
        self.lib.MeshIndexNormalize.argtypes = C.c_void_p,
        self.lib.MeshIndexNormalize.restype = C.c_void_p
        self.lib.MeshIndexSelectionDomain.argtypes = C.c_void_p, C.c_char_p, C.POINTER(C.c_void_p)
        self.lib.MeshIndexSelectionDomain.restype = C.c_void_p
        for name in ('MeshIndexSelectionRelation', 'MeshIndexLayoutRelation'):
            function = getattr(self.lib, name)
            function.argtypes = C.c_void_p, C.c_char_p, C.POINTER(C.c_size_t), C.POINTER(C.c_void_p)
            function.restype = C.c_void_p
        self.lib.MeshIndexRelationTerms.argtypes = C.c_void_p, C.POINTER(C.c_void_p)
        self.lib.MeshIndexRelationTerms.restype = C.c_void_p
        self.ctx = self.ctx_alloc()
        self.references = 1

    def __del__(self):
        if getattr(self, 'ctx', None):
            self.release()

    def release(self):
        # Cyclic Python graph declarations may finalize the context owner before
        # its relations. Keep ISL alive until their native objects are freed.
        self.references -= 1
        if not self.references:
            self.ctx_free(self.ctx)
            self.ctx = None

    def checked(self, value):
        if not value:
            message = self.ctx_last_error_msg(self.ctx)
            raise ValueError(message.decode() if message else 'integer relation could not be resolved')
        return value

    def boolean(self, value):
        if value < 0:
            self.checked(None)
        return bool(value)

    def set(self, text, rank):
        return IndexSet(self, self.set_read_from_str(self.ctx, text.encode()), rank)

    def map(self, source_shape, target_shape, equations):
        """A declared coordinate relation between two finite tensor domains."""
        x, y = coordinates(source_shape, 'x'), coordinates(target_shape, 'y')
        conditions = [*bounds(source_shape, x), *bounds(target_shape, y), *equations(x, y)]
        text = '{ [' + ','.join(x) + '] -> [' + ','.join(y) + '] : ' + (' and '.join(conditions) or 'true') + ' }'
        return IndexRelation(self, self.map_read_from_str(self.ctx, text.encode()), source_shape)

    def window(self, iteration_shape, source_shape, *, axes, strides, dilations,
               padding, transposed, equations):
        """design/compiler-followups-2026-10-30.md#window-relations"""
        def accesses(x, y):
            relations = list(equations(x, y))
            for (source, position, tap), stride, dilation, pad in zip(
                    axes, strides, dilations, padding, strict=True):
                source, position, tap = y[source], x[position], x[tap]
                left, right = (position, source) if transposed else (source, position)
                relations.append(f'{left} + {int(pad)} = {right} * {int(stride)} + {tap} * {int(dilation)}')
            return relations
        return self.map(iteration_shape, source_shape, accesses)

    def selection(self, declaration):
        """The final page binder's selection domain, in logical flat positions."""
        message = C.c_void_p()
        result = self.lib.MeshIndexSelectionDomain(self.ctx, json.dumps(declaration).encode(), C.byref(message))
        if message.value:
            try:
                raise ValueError(C.string_at(message.value).decode())
            finally:
                self.lib.mesh_isl_free(message.value)
        return IndexSet(self, result, 1)

    def selection_relation(self, declaration):
        """Ordered local positions mapped to semantic positions by the binder."""
        return self._declared_relation(self.lib.MeshIndexSelectionRelation, declaration)

    def layout(self, configuration):
        """The shared layout's logical ordinals mapped to storage positions."""
        return self._declared_relation(self.lib.MeshIndexLayoutRelation, configuration)

    def _declared_relation(self, function, declaration):
        message, count = C.c_void_p(), C.c_size_t()
        result = function(self.ctx, json.dumps(declaration).encode(), C.byref(count), C.byref(message))
        if message.value:
            try:
                raise ValueError(C.string_at(message.value).decode())
            finally:
                self.lib.mesh_isl_free(message.value)
        return IndexRelation(self, result, (count.value,))

    def exchanges(self, sources, destinations):
        """design/compiler-followups-2026-10-30.md#selected-page-crossings"""
        incoming = [self.selection_relation(value) for value in sources]
        outgoing = [self.selection_relation(value) for value in destinations]
        domains = [value.image(value.domain()) for value in incoming]
        for at, domain in enumerate(domains):
            if any(not domain.binary(previous, 'intersect').empty() for previous in domains[:at]):
                raise ValueError('selection exchange requires declared disjoint source ownership')
        for destination, requested in enumerate(outgoing):
            missing = requested.domain()
            for source, provided in enumerate(incoming):
                matching = requested.then(provided.inverse((math.prod(sources[source]['shape']),))).lexmin()
                selected = matching.domain()
                if selected.empty():
                    continue
                positions = self.selection_relation({'shape': list(requested.shape),
                    'steps': [['select', 0, selected.spans()]]})
                yield source, destination, positions.then(matching), positions, provided.shape[0], requested.shape[0]
                missing = missing.binary(selected, 'subtract')
            if not missing.empty():
                raise ValueError(f'selection exchange destination {destination} has no declared source for {missing}')

    def ownership(self, shape, axis, unit, extent):
        names = coordinates(shape, 'i')
        equation = f'p = floor({names[axis]} / {unit}) mod {extent}' if all(shape) else 'false'
        text = '{ [' + ','.join(names) + '] -> [p] : ' + ' and '.join([*bounds(shape, names), equation]) + ' }'
        return IndexRelation(self, self.map_read_from_str(self.ctx, text.encode()), shape)

    def cover(self, domains):
        """The native cover's disjoint domains and their contributor sets."""
        domains = tuple(domains)
        if not domains:
            return ()
        count = C.c_size_t()
        cells = self.lib.MeshIndexSetCover((C.c_void_p * len(domains))(*(d.pointer for d in domains)),
                                         len(domains), C.byref(count))
        result = []
        try:
            for k in range(count.value):
                cell = cells[k]
                result.append((IndexSet(self, cell.domain, domains[0].rank),
                               tuple(cell.members[i] for i in range(cell.count))))
                self.lib.mesh_isl_free(cell.members)
        finally:
            self.lib.mesh_isl_free(cells)
        return tuple(result)

    def unconstrained(self, shape, extent):
        """Every declared position may be used by every owner."""
        names = coordinates(shape, 'i')
        text = '{ [' + ','.join(names) + '] -> [p] : ' + ' and '.join([
            *bounds(shape, names), f'0 <= p < {int(extent)}']) + ' }'
        return IndexRelation(self, self.map_read_from_str(self.ctx, text.encode()), shape)


class _Object:
    def __init__(self, algebra, pointer):
        self.algebra = algebra
        self.pointer = algebra.checked(pointer)
        algebra.references += 1

    def __del__(self):
        if getattr(self, 'pointer', None):
            getattr(self.algebra, self.kind + '_free')(self.pointer)
            self.algebra.release()

    def copy(self):
        return getattr(self.algebra, self.kind + '_copy')(self.pointer)

    def __deepcopy__(self, memo):
        # Every consuming ISL operation receives copy(); graph copies share the
        # immutable relation and keep its native context alive through this object.
        memo[id(self)] = self
        return self

    def __str__(self):
        text = self.algebra.checked(getattr(self.algebra, self.kind + '_to_str')(self.pointer))
        try:
            return C.string_at(text).decode()
        finally:
            self.algebra.lib.mesh_isl_free(text)

    def equal(self, other):
        return self.algebra.boolean(getattr(self.algebra, self.kind + '_is_equal')(self.pointer, other.pointer))

    def empty(self):
        return self.algebra.boolean(getattr(self.algebra, self.kind + '_is_empty')(self.pointer))


class IndexSet(_Object):
    kind = 'set'

    def __init__(self, algebra, pointer, rank):
        super().__init__(algebra, pointer)
        self.rank = rank

    def binary(self, other, operation, rank=None):
        return IndexSet(self.algebra, getattr(self.algebra, 'set_' + operation)(self.copy(), other.copy()),
                        self.rank if rank is None else rank)

    def project(self, axis, count=1):
        # isl_dim_set is the public enum's output/set dimension (3).
        result = self.copy()
        for first, removed in ((axis + count, self.rank - axis - count), (0, axis)):
            result = self.algebra.set_project_out(result, 3, first, removed)
        return IndexSet(self.algebra, result, count)

    def separates(self, axis):
        """Whether prefix and suffix coordinates vary independently.

        The surviving boundaries give the finest contiguous axis grouping whose
        selected domain is a product. This uses set algebra, not point traversal.
        """
        left, right = self.project(0, axis), self.project(axis, self.rank - axis)
        return left.binary(right, 'flat_product', self.rank).equal(self)

    def edge(self, maximum=False):
        value = self.algebra.checked((self.algebra.set_dim_max_val if maximum else self.algebra.set_dim_min_val)(self.copy(), 0))
        try:
            return self.algebra.val_get_num_si(value)
        finally:
            self.algebra.val_free(value)

    def spans(self):
        """The shared native set-to-run lowering; repetitions stay compact."""
        count = C.c_size_t()
        runs = self.algebra.checked(self.algebra.lib.MeshIndexSetRuns(self.pointer, C.byref(count)))
        try:
            return tuple(tuple(runs[k * 4 + j] for j in range(4)) for k in range(count.value))
        finally:
            self.algebra.lib.mesh_isl_free(runs)

    def cartesian(self):
        """Axis projections if their product is exactly this domain, else None."""
        axes = [self.project(axis) for axis in range(self.rank)]
        product = self.algebra.set('{ [] }', 0)
        for axis in axes:
            product = product.binary(axis, 'flat_product', product.rank + 1)
        return [axis.spans() for axis in axes] if product.equal(self) else None

    def products(self, boundaries=None):
        """Disjoint products across the declared coordinate boundaries.

        Prefixes sharing the same suffix set form one factor, even when those
        prefixes are repeated stripes. ISL compares whole fibers; no tensor
        positions are expanded. With no argument every axis is a factor. An
        explicit empty boundary list keeps the original nonempty domain.

        These are position sets. Repeated occurrences and their ordering remain
        in the caller's page selection, not in this set decomposition.
        """
        boundaries = tuple(range(1, self.rank) if boundaries is None else sorted(set(boundaries)))
        if self.empty():
            return ()
        if not boundaries:
            return (self,)
        axis, *tail = boundaries
        tail = tuple(boundary - axis for boundary in tail)
        remaining, pieces = self, []
        while not remaining.empty():
            left = remaining.project(0, axis)
            right = remaining.project(axis, self.rank - axis)
            rectangle = left.binary(right, 'flat_product', self.rank)
            if not rectangle.equal(remaining):
                seed = IndexSet(self.algebra, self.algebra.set_lexmin(left.copy()), axis)
                fiber = remaining.binary(seed.binary(right, 'flat_product', self.rank), 'intersect')
                right = fiber.project(axis, self.rank - axis)
                rectangle = left.binary(right, 'flat_product', self.rank)
                different = remaining.binary(rectangle, 'subtract').binary(
                    rectangle.binary(remaining, 'subtract'), 'union')
                left = left.binary(different.project(0, axis), 'subtract')
                rectangle = left.binary(right, 'flat_product', self.rank)
            pieces.extend(left.binary(piece, 'flat_product', self.rank) for piece in right.products(tail))
            remaining = remaining.binary(rectangle, 'subtract')
        return tuple(pieces)


class IndexRelation(_Object):
    kind = 'map'

    def __init__(self, algebra, pointer, shape):
        super().__init__(algebra, algebra.lib.MeshIndexNormalize(pointer))
        self.shape = tuple(map(int, shape))

    def pullback(self, shape, equations):
        """Compose target coordinates -> source coordinates -> owner labels."""
        return self.algebra.map(shape, self.shape, equations).then(self)

    def then(self, other):
        """Compose this relation's target coordinates with the next relation."""
        return IndexRelation(self.algebra, self.algebra.lib.MeshIndexCompose(self.copy(), other.copy()), self.shape)

    def inverse(self, shape):
        """Reverse the relation with its new domain shape explicitly supplied."""
        return IndexRelation(self.algebra, self.algebra.map_reverse(self.copy()), shape)

    def single_valued(self):
        return self.algebra.boolean(self.algebra.map_is_single_valued(self.pointer))

    def lexmin(self):
        """Choose one target for each source, preserving every source coordinate."""
        return IndexRelation(self.algebra, self.algebra.map_lexmin(self.copy()), self.shape)

    def image(self, domain):
        """Positions accessed by the selected source coordinates."""
        selected = self.algebra.map_intersect_domain(self.copy(), domain.copy())
        rank = self.algebra.map_dim(self.pointer, 3)  # isl_dim_out
        return IndexSet(self.algebra, self.algebra.map_range(selected), rank)

    def preimage(self, positions):
        """Source coordinates that access the selected target positions."""
        selected = self.algebra.map_intersect_range(self.copy(), positions.copy())
        return IndexSet(self.algebra, self.algebra.map_domain(selected), len(self.shape))

    def domain(self):
        return IndexSet(self.algebra, self.algebra.map_domain(self.copy()), len(self.shape))

    def subtract_domain(self, positions):
        return IndexRelation(self.algebra, self.algebra.map_subtract_domain(self.copy(), positions.copy()), self.shape)

    def terms(self):
        """Emit the common index expression from this functional relation."""
        message = C.c_void_p()
        result = self.algebra.lib.MeshIndexRelationTerms(self.pointer, C.byref(message))
        if message.value:
            try:
                raise ValueError(C.string_at(message.value).decode())
            finally:
                self.algebra.lib.mesh_isl_free(message.value)
        try:
            return json.loads(C.string_at(self.algebra.checked(result)))
        finally:
            self.algebra.lib.mesh_isl_free(result)

    def reshape(self, shape):
        return self.pullback(shape, lambda x, y: [f'{flat(shape, x)} = {flat(self.shape, y)}'])

    def union(self, other):
        return IndexRelation(self.algebra, self.algebra.map_union(self.copy(), other.copy()), self.shape)

    def intersect(self, other):
        return IndexRelation(self.algebra, self.algebra.map_intersect(self.copy(), other.copy()), self.shape)

    def complete(self, extent):
        """An operation imposes no owner constraint outside the values it uses."""
        universe = self.algebra.unconstrained(self.shape, extent)
        outside = self.algebra.map_subtract_domain(universe.copy(), self.algebra.map_domain(self.copy()))
        return IndexRelation(self.algebra, self.algebra.map_union(outside, self.copy()), self.shape)

    def axes(self, shape, pairs):
        return self.pullback(shape, lambda x, y: [f'{x[b]} = {y[a]}' for a, b in pairs])

    def depends(self, axis):
        projected = self.axes(self.shape, [(d, d) for d in range(len(self.shape)) if d != axis])
        return not projected.equal(self)

    def selected(self, first, count):
        owners = self.algebra.set(f'{{ [p] : {int(first)} <= p < {int(first + count)} }}', 1)
        selected = self.algebra.map_intersect_range(self.copy(), owners.copy())
        return IndexSet(self.algebra, self.algebra.map_domain(selected), len(self.shape))


def convolution_domain(algebra, input_shape, weight_shape, *, stride, padding, dilation,
                       transposed, output_padding, groups, output_shape=None):
    """Canonical N,C,spatial convolution as one grouped matrix access domain.

    Frontends supply their declared axis permutations before this interface.
    Normal weights are Cout,Cin/group,spatial; transposed weights are
    Cin,Cout/group,spatial. All returned maps are preparation operands.
    """
    import math
    input_shape, weight_shape = tuple(input_shape), tuple(weight_shape)
    spatial, kernel = input_shape[2:], weight_shape[2:]
    outputs = tuple((n - 1) * s - 2 * p + d * (k - 1) + o + 1 if transposed else
                    (n + 2 * p - d * (k - 1) - 1) // s + 1
                    for n, k, s, p, d, o in zip(spatial, kernel, stride, padding, dilation, output_padding, strict=True))
    if output_shape is not None:
        outputs = tuple(output_shape[2:])
    batches, channels = input_shape[:2]
    inner_channels, ranks = channels // groups, len(spatial)
    columns = weight_shape[1] if transposed else weight_shape[0] // groups
    iteration = (groups, batches, *outputs, inner_channels, *kernel)
    reduction, rows = inner_channels * math.prod(kernel), batches * math.prod(outputs)
    access = algebra.window(iteration, input_shape,
        axes=[(2 + d, 2 + d, 3 + ranks + d) for d in range(ranks)],
        strides=stride, dilations=dilation, padding=padding, transposed=transposed,
        equations=lambda x, y: [f'{y[0]} = {x[1]}', f'{y[1]} = {x[0]} * {inner_channels} + {x[2 + ranks]}'])
    source_count = math.prod(input_shape)
    scalar = algebra.map(input_shape, (source_count + 1,), lambda x, y: [f'{y[0]} = {flat(input_shape, x)}'])
    address = access.then(scalar)
    absent = algebra.map(iteration, (source_count + 1,), lambda x, y: [f'{y[0]} = {source_count}']).subtract_domain(access.domain())
    neutral = not absent.empty()
    if neutral:
        address = address.union(absent)
    domain = algebra.map((math.prod(iteration),), iteration, lambda x, y: [f'{x[0]} = {flat(iteration, y)}'])
    matrix_shapes = ((groups, rows, reduction), (groups, reduction, columns), (groups, rows, columns))
    weight_steps = ([['reshape', [groups, inner_channels, columns, *kernel]],
                     ['permute', [0, 1, *range(3, 3 + ranks), 2]]] if transposed else
                    [['reshape', [groups, columns, inner_channels, *kernel]],
                     ['permute', [0, *range(2, 3 + ranks), 1]]])
    weight_steps.append(['reshape', list(matrix_shapes[1])])
    output_shape = (batches, groups * columns, *outputs)
    output_steps = [['reshape', [batches, groups, columns, *outputs]],
                    ['permute', [1, 0, *range(3, 3 + ranks), 2]], ['reshape', list(matrix_shapes[2])]]
    canonical = dict(input=domain.then(address),
        weight=algebra.selection_relation({'shape': weight_shape, 'steps': weight_steps}),
        output=algebra.selection_relation({'shape': output_shape, 'steps': output_steps}))
    product_shape = (*matrix_shapes[2], reduction)
    axes = dict(input=(0, 1, 3), weight=(0, 3, 2), output=(0, 1, 2))
    shapes = dict(zip(axes, matrix_shapes, strict=True))
    product_accesses = {name: algebra.map(product_shape, relation.shape,
        lambda x, y, name=name: [f'{y[0]} = {flat(shapes[name], [x[i] for i in axes[name]])}']).then(relation)
        for name, relation in canonical.items()}
    # The neutral page is locally supplied, not an input owned by another rank.
    product_accesses['input'] = product_accesses['input'].then(algebra.map(
        (source_count + 1,), (source_count,), lambda x, y: [f'{x[0]} = {y[0]}']))
    return dict(input_shape=input_shape, output_shape=output_shape, iteration_shape=iteration,
                matrix_shapes=matrix_shapes, input_access=access,
                input_indices=None if canonical['input'].empty() else canonical['input'].terms(),
                input_neutral=neutral, weight_steps=weight_steps, output_steps=output_steps,
                canonical_accesses=canonical, product_shape=product_shape, product_accesses=product_accesses)


def owned_domains(algebra, owners, output, inputs):
    """Resolve ordered output owners into exact work and source-page domains.

    Relations map the common iteration coordinates to logical flat positions.
    Owners retain their ordered selections; set images only choose which work
    and source pages are needed. Repeated reads never change output ordering.
    """
    result = []
    for selection in owners:
        order = algebra.selection_relation(selection)
        work = output.preimage(order.image(order.domain()))
        result.append(dict(selection=selection, order=order, work=work,
                           reads={slot: access.image(work) for slot, access in inputs.items()}))
    return tuple(result)
