"""Named preparation bindings and the existing passive shader declarations.

Numerical adapters describe their commands here before either frontend binds
pages. Views use TensorLayout in their named buffer's scalar coordinates;
workspace names and initializers are explicit compiler operands.
"""
import math
import torch

from . import _stream


def dense(shape):
    from .layout import strides
    return _stream.TensorLayout(tuple(shape), tuple(strides(shape)))


def configuration():
    from .compilation import operands
    _, threads, memory = _stream.device_facts()
    from mesh_compute import hardware
    cores = hardware()['gpu']['cores']
    return dict(geometry=operands(), capabilities=dict(compute_units=cores, max_threads_per_threadgroup=threads,
                                                       threadgroup_memory_bytes=memory))


def descriptions(operands):
    return {name: dict(shape=tuple(value.shape), dtype=value.dtype)
            for name, value in operands.items() if value is not None}


def rows(view, axis):
    layout = view['layout']
    axis %= len(layout.shape)
    axes = tuple(d for d in range(len(layout.shape)) if d != axis) + (axis,)
    count = math.prod(layout.shape[d] for d in axes[:-1])
    return dict(view, layout=layout.permute(axes).reshape((count, layout.shape[axis])))


class Commands:
    def __init__(self, operands, *, geometry, capabilities):
        from .compilation import relations
        self.algebra = relations()
        self.geometry, self.capabilities = geometry, capabilities
        self.bindings = {name: dict(value, role=True) for name, value in operands.items() if value is not None}
        self.commands = []

    def view(self, name, layout=None, *, dtype=None, binding_origin=None, source_domain=None):
        value = self.bindings[name]
        return dict(name=name, dtype=value['dtype'] if dtype is None else dtype,
                    layout=dense(value['shape']) if layout is None else layout, binding_origin=binding_origin,
                    source_domain=source_domain)

    def temporary(self, name, shape, dtype, *, initializer=None, requirements=()):
        self.bindings[name] = dict(shape=tuple(shape), dtype=dtype, initializer=initializer,
                                   requirements=tuple(requirements), role=False)
        return self.view(name)

    def shader(self, source, function, operands, *, grid, threadgroup, access,
               constants=None, definitions=None, indices=None, instance='', thread_grid=False,
               threadgroup_policy=None, limits=None, access_layouts=None, dispatch=None, coherent_accesses=False):
        from mesh_relation import Algebra
        from .operand import layout_configuration
        algebra = self.algebra
        prefix = ''.join(f'#define {key} {value}\n' for key, value in (definitions or {}).items())
        constants = {} if constants is None else constants
        count = max([len(operands), *(slot + 1 for slot in constants)], default=0)
        arguments, types, origins, accesses = [], [None] * count, [None] * count, []
        for slot, view in enumerate(operands):
            if view is None:
                continue
            name, dtype, source_layout = view['name'], view['dtype'], view['layout']
            binding = self.bindings[name]
            argument = dict(slot=slot, name=name, dtype=dtype, index_dtype=dtype, binding_origin=view.get('binding_origin'),
                source=source_layout, source_dtype=dtype, storage_dtype=binding['dtype'],
                storage_bytes=math.prod(binding['shape']) * binding['dtype'].itemsize,
                requirements=binding.get('requirements', ()), source_domain=view.get('source_domain'))
            arguments.append(argument);types[slot] = dtype
            reached = source_layout if access_layouts is None else access_layouts.get(slot, source_layout)
            source_relation = algebra.layout(layout_configuration(source_layout))
            for side, shift in (('reads', 0), ('writes', self.geometry['ACCESS_WRITE_SHIFT'])):
                if not access & (1 << (slot + shift)):
                    continue
                if argument['source_domain'] is not None:
                    from .operand import iteration_order, relation_layout
                    relations = [use['group_relation'] for use in dispatch if use['side'] == side and use['slot'] == slot]
                    positions = relations[0].image(relations[0].domain())
                    for relation in relations[1:]:
                        positions = positions.binary(relation.image(relation.domain()), 'union')
                    order, _ = iteration_order(positions, (source_layout.count,))
                    local = relation_layout(order, order.shape, source_layout.count)
                    relation = order.then(source_relation)
                else:
                    relation = algebra.layout(layout_configuration(reached))
                    local = (dense(source_layout.shape) if reached is source_layout else
                             reached.rebase(source_layout, dense(source_layout.shape)))
                typed = dict(name=name, dtype=dtype, storage_bytes=argument['storage_bytes'], relation=relation)
                accesses.append(dict(side=side, slot=slot, access=typed, work=relation.domain(), origins=(),
                                     argument_layout=local.components(dtype.itemsize).reshape((local.count * dtype.itemsize,))))
        self.commands.append(dict(kind='shader', source=prefix + source + instance, function=function,
            arguments=tuple(arguments), types=tuple(types), index_dtype=None, coherent_accesses=coherent_accesses,
            index_layouts={} if indices is None else indices,
            binding_origins=tuple(origins), constants=constants, access=access, accesses=tuple(accesses),
            grid=tuple(grid), threadgroup=None if threadgroup is None else tuple(threadgroup),
            threadgroup_policy=threadgroup_policy, limits=limits, thread_grid=thread_grid, dispatch=dispatch))

    def declaration(self):
        return dict(commands=tuple(self.commands), bindings=dict(self.bindings))


def bind(declaration, operands):
    """Resolve named preparation bindings into the ordinary shader operands."""
    from .layout import allocate
    from .operand import TensorView, rebase_typed, requirements, storage_requirement
    values = {name: TensorView.of(value) for name, value in operands.items() if value is not None}
    for name, description in declaration['bindings'].items():
        if description['role']:
            continue
        contracts = [storage_requirement(value) for value in description['requirements']]
        layout = requirements(dense(description['shape']), description['requirements'])
        capacity = max((extent for _, extent, _ in contracts), default=0)
        alignment = math.lcm(*(align for _, _, align in contracts))
        values[name] = allocate(layout, description['dtype'], 'mps', extent=capacity, alignment=alignment)
        initializer = description.get('initializer')
        if initializer is not None:
            from .engines import host
            host(values[name].storage).fill_(initializer)
    result = []
    for command in declaration['commands']:
        views = [None] * len(command['types'])
        for argument in command['arguments']:
            actual = values[argument['name']]
            description = declaration['bindings'][argument['name']]
            domain = argument.get('source_domain')
            if domain is None:
                layout = rebase_typed(argument['source'], argument['source_dtype'].itemsize,
                    dense(description['shape']), description['dtype'].itemsize, actual.layout)
            else:
                from .operand import layout_configuration, relation_layout
                algebra = domain.algebra
                source = algebra.layout(layout_configuration(argument['source']))
                source = source.subtract_domain(source.domain().binary(domain, 'subtract'))
                placed = source.then(algebra.layout(layout_configuration(actual.layout)))
                layout = relation_layout(placed, argument['source'].shape, actual.storage.numel())
            views[argument['slot']] = TensorView(actual.storage_as(argument['dtype']), layout,
                actual.binding_origin if argument['binding_origin'] is None else argument['binding_origin'])
        bound = dict(command, operands=tuple(views))
        result.append(dict(bound, reaches=bind_accesses(bound)))
    return tuple(result)


def bind_accesses(command):
    """Place the retained argument-local byte reaches on their final pages."""
    from .operand import TensorView
    arguments = {argument['slot']: argument for argument in command['arguments']}
    reaches = []
    for use in command['accesses']:
        slot = use['slot'];argument, actual = arguments[slot], command['operands'][slot]
        original = dense((argument['source'].count * argument['source_dtype'].itemsize,))
        placed = use['argument_layout'].rebase(original,
            actual.layout.components(actual.element_size()).reshape((original.count,)))
        reaches.append((slot, 1 if use['side'] == 'reads' else 2,
                        TensorView(actual.storage_as(torch.uint8), placed, 0)))
    return tuple(reaches)
