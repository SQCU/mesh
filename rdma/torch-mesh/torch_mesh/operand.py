"""Preparation-only operands over the shared layout/dependency algebra.

TensorView declares a view of bridge pages. Kernels, copies and API bindings
lower it to native arguments and accesses before a function runs. Layout
metadata describes accesses; it never snapshots tensor contents. A tensor of
page indices can be written and read by the compiled function like other data.
"""
from dataclasses import dataclass, replace

from . import _stream


@dataclass(frozen=True, eq=False)
class TensorView:
    storage: object
    layout: _stream.TensorLayout
    # Preparation-only pointer origin in the final page buffer. None binds
    # the ordinary layout origin. Access/dependency positions stay unchanged.
    binding_origin: int | None = None

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
        from math import lcm
        unit = lcm(self.element_size(), dtype.itemsize)
        width = self.storage.untyped_storage().nbytes() // unit * unit // self.element_size()
        return self.storage.as_strided((width,), (1,), 0).view(dtype)

    def typed(self, dtype):
        """Reinterpret equal-width scalars without changing their page selection."""
        if dtype.itemsize != self.element_size():
            raise ValueError('a scalar reinterpretation preserves its declared width')
        return replace(self, storage=self.storage_as(dtype))

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
        return replace(self, layout=self.layout.slice(axis, first, count, step))

    def select(self, axis, indices):
        return replace(self, layout=self.layout.select(axis, indices))

    def gather(self, shape, indices):
        return replace(self, layout=self.layout.gather(shape, indices))

    def permute(self, axes):
        return replace(self, layout=self.layout.permute(axes))

    def reshape(self, shape):
        return replace(self, layout=self.layout.reshape(shape))

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

    def write_layout(self):
        """This view as a WRITE names it (nccl.h ncclMeshLayout): its storage (a view at the storage's first element, for
        its page table), element bits, shape, strides and offset in elements; a mapped view's logical-order element runs
        in place of strides.  Abstract layout: the library lowers it to pages, no address is read here."""
        start = self.storage.as_strided((int(bool(self.storage.numel())),), (1,), 0)
        runs = [] if self.layout.affine else [value for run in self.layout.ordered_spans() for value in run]
        strides = list(self.layout.strides) if self.layout.affine else []
        return (start, 8 * self.element_size(), list(self.shape), strides, int(self.layout.offset),
                not self.layout.affine, runs)

    def declaration(self, *requirements):
        return dict(source=descriptor(self.storage).state, view=self.layout.state, requirements=requirements,
                    binding_origin=self.binding_origin)


def ordered_runs(segments):
    """Concatenate page selections as logical intervals for collective cuts."""
    runs, cursor = [], 0
    for identity, value in segments:
        for offset, count in descriptor(value).ordered_spans():
            runs.append((cursor, cursor + count, identity, offset))
            cursor += count
    return runs, cursor

def restore(state):
    shape, steps, offset, maps, index_shape, coordinates = state
    return _stream.TensorLayout(shape, steps, offset, {axis: values for axis, values in enumerate(maps) if values}, None, index_shape, coordinates)


def layout_configuration(layout):
    """The shared native layout schema, without expanding selected positions."""
    import json
    shape, strides, offset, maps, index_shape, coordinates = layout.state
    return dict(shape=shape, strides=strides, offset=offset, index_shape=index_shape,
                maps={str(axis): values for axis, values in enumerate(maps) if values},
                index_coordinates=json.loads(coordinates))


class _IndexTerms:
    """Emit the shared integer grammar through Torch's expression interpreter."""
    _operators = dict(add='add', floordiv='div', mod='mod', python_mod='mod',
                      minimum='min', maximum='max', eq='eq', le='le', lt='lt', ge='ge', gt='gt',
                      and_='and', or_='or', where='select')

    def __getattr__(self, name):
        operation = self._operators[name]
        return lambda *values: [operation, *values]

    @staticmethod
    def constant(value, dtype):
        if value in (True, False):
            return int(bool(value))
        if value.is_Integer:
            return int(value)
        raise TypeError(f'index term contains non-integer literal {value}')

    @staticmethod
    def sym_sum(values):
        return ['add', *values] if len(values) > 1 else values[0] if values else 0

    @staticmethod
    def mul(left, right):
        # The shared layout grammar puts a constant scale after its index.
        return ['mul', right, left] if isinstance(left, int) else ['mul', left, right]

    @staticmethod
    def modular_indexing(value, divisor, modulus):
        return ['mod', ['div', value, divisor], modulus]

    @staticmethod
    def not_(value):
        return ['eq', value, 0]

    @staticmethod
    def ne(left, right):
        return ['eq', ['eq', left, right], 0]

    @staticmethod
    def identity(value):
        return value

    @staticmethod
    def abs(value):
        return ['max', value, ['mul', value, -1]]

    @staticmethod
    def pow_by_natural(value, exponent):
        result = 1
        while exponent:
            if exponent % 2:
                result = ['mul', result, value]
            value, exponent = ['mul', value, value], exponent // 2
        return result

    @staticmethod
    def expr_cond_pair(value, condition):
        return value, condition

    @staticmethod
    def piecewise(*branches):
        result, condition = branches[-1]
        if condition != 1:
            raise ValueError('an index expression must declare its complete domain')
        for value, condition in reversed(branches[:-1]):
            result = ['select', condition, value, result]
        return result


def index_terms(expression, variables):
    """Serialize existing scalar index IR; the shared algebra performs solving."""
    from torch.utils._sympy.interp import sympy_interp
    return sympy_interp(_IndexTerms(), variables, expression)


def index_expression(expression, index, operations):
    """Interpret the shared index grammar in a caller's scalar compiler."""
    from functools import reduce
    if expression == 'i':
        return index
    if isinstance(expression, (int, bool)):
        return expression
    operation, *arguments = expression
    values = [index_expression(value, index, operations) for value in arguments]
    return operations[operation](*values) if operation == 'select' else reduce(operations[operation], values)


def symbolic_index(expression, index):
    """Use Torch's symbolic scalar operators for a compiled page-index term."""
    from torch.utils._sympy.functions import Where
    from torch.utils._sympy.reference import ReferenceAnalysis
    operations = {name: getattr(ReferenceAnalysis, target) for name, target in {
        'add': 'add', 'mul': 'mul', 'div': 'floordiv', 'mod': 'mod',
        'eq': 'eq', 'le': 'le', 'lt': 'lt', 'ge': 'ge', 'gt': 'gt',
        'and': 'and_', 'or': 'or_', 'min': 'minimum', 'max': 'maximum'}.items()}
    return index_expression(expression, index, dict(operations, select=Where))


def iteration_order(domain, shape):
    """Compact selected work with the same ordered selection used by pages."""
    import math
    from mesh_relation import flat
    shape = tuple(map(int, shape))
    flatten = domain.algebra.map(shape, (math.prod(shape),), lambda source, target: [f'{target[0]} = {flat(shape, source)}'])
    selected = flatten.image(domain)
    selection = dict(shape=[math.prod(shape)], steps=[['select', 0, selected.spans()]])
    order = domain.algebra.selection_relation(selection)
    return order, domain.equal(flatten.domain())


def scalar_expression(body, value, cache):
    """Fold scalar index syntax with Torch's existing symbolic operators.

    Payload reads and other unresolved scalar producers remain their original
    body nodes. No value is loaded to decide a preparation-time relation.
    """
    import torch
    import sympy
    from torch._inductor.index_propagation import SymPyOps, TypedExpr
    from torch.utils._sympy.reference import ReferenceAnalysis
    from torch.utils import _pytree

    if not isinstance(value, torch.fx.Node):
        return value
    if value in cache:
        return cache[value]
    result = None
    if value.op == 'call_module' and value.target == 'get_index':
        result = body.indexing_exprs[value.args[0]]
    elif value.op == 'call_method':
        unresolved = False
        def argument(item):
            nonlocal unresolved
            result = scalar_expression(body, item, cache)
            unresolved |= isinstance(item, torch.fx.Node) and result is None
            return result
        args, kwargs = _pytree.tree_map(argument, (value.args[1:], value.kwargs))
        if not unresolved:
            if hasattr(SymPyOps, value.target):
                result = getattr(SymPyOps, value.target)(*args, **kwargs)
                if result is NotImplemented:
                    result = None
            elif value.target in ('eq', 'ne', 'lt', 'le', 'gt', 'ge', 'and_', 'or_', 'not_',
                                  'logical_and', 'logical_or', 'logical_not'):
                operation = {'logical_and': 'and_', 'logical_or': 'or_', 'logical_not': 'not_'}.get(value.target, value.target)
                result = TypedExpr(getattr(ReferenceAnalysis, operation)(*(sympy.sympify(item.expr) for item in args)), torch.bool)
            if isinstance(result, TypedExpr) and not result.is_constant() and result.dtype != torch.bool:
                # The symbolic handler describes integer arithmetic. Retain a
                # typed scalar operation when overflow or rounding could alter
                # that mathematical expression.
                if result.dtype.is_floating_point or result.dtype.is_complex:
                    result = None
                else:
                    from torch.utils._sympy.value_ranges import bound_sympy, ValueRanges
                    bounds = bound_sympy(result.expr, {var: ValueRanges(0, size - 1)
                        for var, size in {**body.var_ranges, **body.indirect_var_ranges}.items()})
                    limits = torch.iinfo(result.dtype)
                    if bounds.lower < limits.min or bounds.upper > limits.max:
                        result = None
    cache[value] = result
    return result


def index_relation(algebra, expression, variables, shape):
    """An IR address expression over its declared common iteration domain."""
    import math
    from .layout import strides
    shape = tuple(map(int, shape))
    coordinates = layout_configuration(_stream.TensorLayout(shape, strides(shape)))['index_coordinates']
    terms = index_terms(expression, {var: value for var, value in zip(variables, coordinates, strict=True) if var != 0})
    return algebra.layout(dict(shape=shape, index_shape=[math.prod(shape)], strides=[1], offset=0,
                               maps={}, index_coordinates=[terms])).reshape(shape)


def byte_relation(access):
    """A typed access's scalar relation expressed in its named storage bytes."""
    relation = access['relation']
    if relation is None:
        return None
    width, extent = access['dtype'].itemsize, int(access['storage_bytes'])
    components = relation.algebra.map(((extent + width - 1) // width,), (extent,),
        lambda scalar, byte: [f'{width} * {scalar[0]} <= {byte[0]} < {width} * ({scalar[0]} + 1)'])
    return relation.then(components)


def access_domain(access, work):
    """Project an assigned work operand onto its declared possible byte reaches.

    Index values remain ordinary scalar producers. Their declared value axes
    extend the work domain; no payload is inspected and no work is assigned.
    Original predicates remain attached to the access and execute in its body.
    Unknown predicates cannot remove a possible reach during preparation.
    """
    space = access.get('index_space')
    relation = access['relation'] if space is None else space['relation']
    domain = work
    if space is not None and len(space['shape']) != space['work_dimensions']:
        count = space['work_dimensions']
        projection = work.algebra.map(space['shape'], space['shape'][:count],
            lambda source, target: [f'{y} = {x}' for x, y in zip(source[:count], target, strict=True)])
        domain = projection.preimage(work)
    relation = None if relation is None else byte_relation(dict(access, relation=relation))
    return dict(access=access, relation=relation, domain=domain,
                positions=None if relation is None else relation.image(domain))


def access_owner(access, owner):
    """Compile complete typed-field ownership into the existing index grammar.

    ``owner`` is the caller's declared byte selection. A scalar belongs to it
    only when every byte of that scalar belongs to it. This prepares a predicate
    on the original live index; it neither reads an index page nor assigns work.
    """
    import sympy
    width, extent = access['dtype'].itemsize, int(access['storage_bytes'])
    algebra, count = owner.algebra, extent // width
    components = algebra.map((count,), (extent,),
        lambda scalar, byte: [f'{width} * {scalar[0]} <= {byte[0]} < {width} * ({scalar[0]} + 1)'])
    missing = components.image(components.domain()).binary(owner, 'subtract')
    absent = components.preimage(missing)
    present = components.domain().binary(absent, 'subtract')
    covered = components.image(present)
    if not covered.equal(owner.binary(components.image(components.domain()), 'intersect')):
        raise ValueError('a scalar read port requires complete typed owner fields; partial bytes need a byte-typed read declaration')
    yes = algebra.map((count,), (2,), lambda x, y: [f'{y[0]} = 1']).subtract_domain(absent)
    no = algebra.map((count,), (2,), lambda x, y: [f'{y[0]} = 0']).subtract_domain(present)
    return sympy.Eq(symbolic_index(yes.union(no).terms(), access['index']), 1)


def bound_access(access, work, source, source_dtype, target):
    """Compose a declared byte reach with the argument's assigned byte layout.

    ``source`` retains original storage coordinates in ``source_dtype`` units;
    ``target`` supplies bytes in that same ordered domain. This is a dependency
    footprint, not a request to deliver every possibly addressed value.
    """
    projected = access_domain(access, work)
    if projected['positions'] is None:
        raise ValueError(f"access {access['name']} still requires its scalar index declaration")
    selected = _stream.TensorLayout.runs(projected['positions'].spans())
    original = source.components(source_dtype.itemsize).reshape((source.count * source_dtype.itemsize,))
    return dict(projected, layout=selected.rebase(original, target.reshape((target.count,))))


def bind(value, declaration):
    """Compose the declared use with the producer's assigned pages once."""
    placed = TensorView.of(value)
    return replace(placed, layout=binding_layout(placed.layout, declaration),
                   binding_origin=declaration.get('binding_origin', placed.binding_origin))


def binding_layout(layout, declaration):
    """The same declared-use relation for logical IR and assigned pages."""
    return restore(declaration['view']).rebase(restore(declaration['source']), layout)

def descriptor(value):
    return value if isinstance(value, _stream.TensorLayout) else TensorView.of(value).layout


def relation_layout(relation, shape, extent):
    """An exact logical-to-storage relation in its declared scalar units."""
    import json
    if not all(shape):
        from .layout import strides
        return _stream.TensorLayout(tuple(shape), strides(shape))
    import math
    flat = relation.reshape((math.prod(shape),))
    return _stream.TensorLayout(tuple(shape), (1,), 0, {}, None,
                                (extent,), json.dumps([flat.terms()]))


def relation_view(value, relation, shape):
    """Bind prepared logical indices to the same operand's assigned pages."""
    value = TensorView.of(value)
    logical = _stream.TensorLayout((value.numel(),), (1,))
    mapped = relation_layout(relation, shape, value.numel())
    return replace(value, layout=mapped.rebase(logical, value.layout.reshape((value.numel(),))))


def byte_operand(value):
    """The same selected bytes, with the generated pointer bound at its root.

    Keeping the base separate from indexed positions also permits one argument
    to contain mixed scalar widths. Typed loads apply their width after this
    byte relation; no scalar is rounded or expanded into a different value.
    """
    value = TensorView.of(value)
    layout = value.layout if value.element_size() == 1 else value.layout.components(value.element_size())
    import torch
    return TensorView(value.storage_as(torch.uint8), layout, binding_origin=0)


def rebase_typed(view, width, declared, source_width, placed):
    """Compose in scalar units when possible; bytes only for a type view."""
    if width == source_width:
        return view.rebase(declared, placed)
    return view.components(width).rebase(declared.components(source_width), placed.components(source_width)).elements(width)


def requirement(value, declared):
    """Resolve an explicit API requirement without allocating or copying values."""
    return required_layout(descriptor(value), storage_requirement(declared)[0])


def storage_requirement(declared):
    """One consumer's layout, byte footprint and argument-address alignment."""
    if isinstance(declared, dict):
        return declared['layout'], int(declared['capacity']), int(declared['alignment'])
    return declared, 0, 1


def requirements(value, declarations):
    declarations = [storage_requirement(item)[0] for item in declarations]
    return descriptor(value).requirements([
        restore(item['view']) if isinstance(item, dict) else
        (item, ()) if isinstance(item, str) else ('fixed', tuple(item)) for item in declarations])


def required_layout(view, declared):
    """Resolve an exact page relation or an API's accepted layout family."""
    if isinstance(declared, dict):
        return view.requirements([restore(declared['view'])])
    return view.required(declared if isinstance(declared, str) else 'fixed',
                         [] if isinstance(declared, str) else tuple(declared))


def signature(buffers, outputs, requirements):
    """One operand schema for GPU and host declarations; no payload is read."""
    import json
    import torch
    views = {i: TensorView.of(value) for i, value in enumerate(buffers) if value is not None}
    results, slots = [], []
    for index in outputs:
        value = views[index].storage
        found = next((j for j, held in enumerate(results)
                      if held.dtype == value.dtype and torch._C._is_alias_of(held, value)), None)
        if found is None:
            # A result handle represents its storage root. Returning the first
            # slice would discard the other slots' offsets when fake outputs
            # recreate that handle during capture.
            root = value
            while root._base is not None:
                root = root._base
            if root.dtype == value.dtype and root.storage_offset() == 0:
                value = root
            else:
                value = value.as_strided((value.untyped_storage().nbytes() // value.element_size(),), (1,), 0)
            results.append(value); slots.append([]); found = len(results) - 1
        slots[found].append(index)
        views[index] = replace(views[index], storage=results[found])
    declarations = {i: view.declaration(*requirements[i]) for i, view in views.items()}
    inputs = [i for i in views if i not in outputs]
    return dict(inputs=[views[i].storage for i in inputs], input_slots=inputs, output_slots=slots,
                output_strides=[list(value.stride()) for value in results], shapes=[list(value.shape) for value in results],
                dtypes=[str(value.dtype).removeprefix('torch.') for value in results], devices=[str(value.device) for value in results],
                operands=json.dumps(declarations, separators=(',', ':')))


def fake_outputs(shapes, strides, dtypes, devices):
    import torch
    return [torch.empty_strided(shape, steps, dtype=getattr(torch, dtype), device=device)
            for shape, steps, dtype, device in zip(shapes, strides, dtypes, devices, strict=True)]


def graph_bindings(node):
    """One slot decoder for native/host declarations and their equalities."""
    import json
    schema = getattr(node.target, '_schema', None)
    args = {a.name: node.args[i] if i < len(node.args) else node.kwargs.get(a.name, a.default_value)
            for i, a in enumerate(schema.arguments)}
    declarations = json.loads(args['operands'])
    bindings = [(source.meta['val'], slot) for source, slot in zip(args['inputs'], args['input_slots'], strict=True)]
    bindings.extend((value, slot) for value, slots in zip(node.meta['val'], args['output_slots'], strict=True) for slot in slots)
    uses = {}
    for value, slot in bindings:
        declaration = declarations[str(slot)]
        view = binding_layout(descriptor(value), declaration)
        uses[slot] = value, view, declaration['requirements']
    return args, uses


def graph_uses(node):
    """Declared input/output selections and API needs, before page placement."""
    from torch.utils._pytree import tree_flatten
    from .compilation import api_layouts
    schema = getattr(node.target, '_schema', None)
    if schema is None:
        return
    if schema.name in ('torch_mesh::native_kernel', 'torch_mesh::host_kernel'):
        _, bindings = graph_bindings(node)
        for value, view, requirements in bindings.values():
            for requirement in requirements:
                yield value, view, requirement
        return
    args = {a.name: node.args[i] if i < len(node.args) else node.kwargs.get(a.name, a.default_value)
            for i, a in enumerate(schema.arguments)}
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
