"""Compile page production/observation into the commands which use those pages.

All sets below are preparation operands. The emitted program receives flat
group-indexed threshold/contribution pages and its ordinary payload pages.
It neither searches page ranges nor constructs dependency edges at execution.
"""
import math

import torch

from . import _stream
from .operand import TensorView, byte_operand, index_relation, layout_configuration


def _physical(value, algebra):
    from mesh_relation import IndexRelation
    count, pointer = _stream.page_region(value, algebra.ctx)
    return IndexRelation(algebra, pointer, (count,))


def _groups(command):
    from .compilation import operands
    grid = tuple(command['grid'])
    group = command['threadgroup']
    policy = command.get('threadgroup_policy')
    if policy is not None:
        preferred = max(operands()['SIMD_WIDTH'] if policy['minimum_simd'] else 1, policy['preferred'])
        command['threadgroup'] = group = (min(_stream.device_facts()[1], preferred), 1, 1)
        command.pop('threadgroup_policy')
    if group is None:
        left = min(_stream.device_facts()[1], operands()['ROW_THREADS'])
        group = []
        for extent in grid:
            width = min(left, max(1, extent))
            group.append(width)
            left //= width
        command['threadgroup'] = group = tuple(group)
    counts = (tuple((n + width - 1) // width for n, width in zip(grid, group, strict=True))
              if command['thread_grid'] else grid)
    return group, counts


def refine(section, projection, storage, cuts, algebra):
    """Refine CPU source domains with the same placed read/write relations.

    A cut first selects actual stored bytes. Forward relation images identify
    affected outputs; ordinary backward projection then includes each complete
    required fiber. Thus a cut through a reduction input never invents a partial
    result or duplicates an output. Remaining work appears exactly once.
    """
    from .compilation import implementation_assignment, implementations
    from mesh_compute import assigned_implementation
    from .ownership import project_numerical, binding_uses
    from .operand import byte_relation
    if not cuts:
        return projection
    choices, catalog = implementation_assignment()['scalar'], implementations()
    cpu = {item['operation'].get_operation_name() for item in section['retained'].consumer_constraints()
           if catalog[assigned_implementation(item['required'], choices)]['compiler'] == 'cpp_scalar'}
    if not cpu:
        return projection
    assigned = {id(record['domain']): record['work'] for record in projection['records']}
    assigned.update({id(record['domain']['finish']): record['finish']['work']
        for record in projection['records'] if record.get('finish') is not None})
    placed = {}
    for name, binding in storage.items():
        physical = _physical(binding['operand'], algebra)
        original = algebra.layout(layout_configuration(binding['source'].components(binding['source_dtype'].itemsize)))
        placed[name] = physical, original
    records = [record for record in projection['records']
               if record['effect']['operation'].get_operation_name() not in cpu]
    seen, finished = {}, {}

    def append(selected):
        for record in selected:
            if record['effect']['operation'].get_operation_name() not in cpu:
                continue
            key = id(record['domain'])
            work = record['work'] if key not in seen else record['work'].binary(seen[key], 'subtract')
            seen[key] = record['work'] if key not in seen else seen[key].binary(record['work'], 'union')
            finish = record['finish']
            if finish is not None:
                chosen = finish['work'] if key not in finished else finish['work'].binary(finished[key], 'subtract')
                finished[key] = finish['work'] if key not in finished else finished[key].binary(finish['work'], 'union')
                finish = dict(finish, work=chosen)
            records.append(dict(record, work=work, finish=finish))

    for cut in cuts:
        relation = _physical(cut, algebra)
        positions = relation.image(relation.domain())
        affected = {name: original.image(physical.preimage(positions))
                    for name, (physical, original) in placed.items()}
        def add(name, positions):
            affected[name] = positions if name not in affected else affected[name].binary(positions, 'union')
        for effect in section['effects']:
            for domain in effect['domains']:
                for part in (domain, domain.get('finish')):
                    if part is None:
                        continue
                    work = algebra.set('{ [' + ','.join(f'i{k}' for k in range(len(part['work']))) + '] : false }',
                                       len(part['work']))
                    for side in ('reads', 'writes'):
                        for access in part[side].values():
                            wanted = affected.get(access['name'])
                            if wanted is None or wanted.empty():
                                continue
                            mapping = byte_relation(access)
                            selected = assigned[id(part)] if mapping is None else mapping.preimage(wanted)
                            work = work.binary(selected, 'union')
                    work = work.binary(assigned[id(part)], 'intersect')
                    for access in part['writes'].values():
                        mapping = byte_relation(access)
                        if mapping is not None:
                            add(access['name'], mapping.image(work))
        demands = {}
        for key, binding in binding_uses(section['output_bindings']):
            if key not in projection['outputs'] or binding['name'] not in affected:
                continue
            wanted = byte_relation(binding).preimage(affected[binding['name']]).binary(projection['outputs'][key], 'intersect')
            demands[key] = wanted if key not in demands else demands[key].binary(wanted, 'union')
        selected = project_numerical(section, demands, algebra, constraints=assigned)
        append(selected['records'])
    append(projection['records'])
    return dict(projection, records=tuple(records))


def logical_dispatch_accesses(command, algebra):
    """Retain source coordinates and argument-local selected byte relations.

    This stage uses neither allocated payload nor launch geometry. An exporter
    retains these same operands until target capabilities fix its launch. An
    already grouped numerical recipe retains its declared group relation.
    """
    bindings = {argument['slot']: argument for argument in command['arguments']}
    result = []
    for access in command['dispatch']:
        argument = bindings[access['slot']]
        if 'group_relation' in access:
            width = command['types'][access['slot']].itemsize
            extent = argument['source'].count * argument['source_dtype'].itemsize
            components = algebra.map((extent // width,), (extent,),
                lambda x, y: [f'{width} * {x[0]} <= {y[0]} < {width} * ({x[0]} + 1)'])
            grouped = access['group_relation'].then(components)
            result.append(dict(side=access['side'], relation=grouped, slot=access['slot'],
                               grouped=True, active=(), reductions=()))
            continue
        shape = access['shape']
        relation = index_relation(algebra, access['index'], access['variables'], shape)
        unit = argument['dtype'].itemsize
        components = algebra.map((argument['storage_bytes'] // unit,), (argument['storage_bytes'],),
            lambda x, y: [f'{unit} * {x[0]} <= {y[0]} < {unit} * ({x[0]} + 1)'])
        selected = algebra.layout(layout_configuration(argument['source'].components(argument['source_dtype'].itemsize)))
        reached = relation.then(components).then(selected.inverse((argument['storage_bytes'],)))
        result.append(dict(side=access['side'], slot=access['slot'], relation=reached,
                           grouped=False, active=tuple(access['active']), reductions=tuple(access['reductions'])))
    return tuple(result)


def logical_group_accesses(command, algebra, *, group, counts):
    """Project retained source coordinates using the target launch operands."""
    from mesh_relation import IndexRelation
    total, result = math.prod(counts), []
    for access in logical_dispatch_accesses(command, algebra):
        if access['grouped']:
            result.append(access)
            continue
        relation = access['relation']
        axes = [-1] * len(relation.shape)
        for dimension, axis in enumerate(access['active']):
            axes[axis] = dimension
        grouped = IndexRelation(algebra, _stream.page_group_relation(
            relation.pointer, group, counts, axes, access['reductions']), (total,))
        result.append(dict(side=access['side'], relation=grouped, slot=access['slot']))
    return total, tuple(result)


def _group_accesses(command, algebra):
    """Compose the shared logical access relations with the assigned pages."""
    if command['kind'] in ('cpu', 'ane'):
        reached = []
        for slot, mode, value in command['reaches']:
            relation = _physical(value, algebra)
            positions = relation.image(relation.domain())
            selected = algebra.map((1,), relation.shape, lambda x, y: []).then(relation)
            for side, flag in (('reads', 1), ('writes', 2)):
                if mode & flag:
                    reached.append(dict(side=side, relation=selected,
                                        positions=positions, slot=slot))
        return 1, tuple(reached)
    group, counts = _groups(command)
    total, logical = logical_group_accesses(command, algebra, group=group, counts=counts)
    physical = {access['slot']: _physical(command['operands'][access['slot']], algebra)
                for access in logical}
    result = []
    for access in logical:
        relation = access['relation'].then(physical[access['slot']])
        result.append(dict(access, relation=relation, positions=relation.image(relation.domain())))
    return total, tuple(result)


def transfer_effects(programs, occurrences, schedule):
    """Adapt retained frontend declarations to the common page-effect compiler."""
    from .compilation import relations, bind
    from .native import declared_commands
    algebra = relations()
    commands, descriptions = [], []
    backends = dict(shader=_stream.PageBackend.Metal, cpu=_stream.PageBackend.CPU, ane=_stream.PageBackend.ANE)
    for program in programs:
        with bind(program['configuration'], algebra=algebra):
            program['commands'] = tuple(declared_commands(program['commands']))
            for command in program['commands']:
                groups, accesses = _group_accesses(command, algebra)
                gpu = command['kind'] == 'shader'
                names = [''] * len(command['operands'])
                if command.get('names'):
                    for argument in command['arguments']:
                        names[argument['slot']] = argument['name']
                descriptions.append(dict(order=program['order'], groups=groups,
                    backend=backends[command['kind']], source=command['source'],
                    function=command['function'] if gpu else '',
                    grid=_groups(command)[1] if gpu else (1, 1, 1),
                    metadata_slot=len(command['operands']), argument_names=names,
                    coherent_accesses=command.get('coherent_accesses', False),
                    accesses=tuple(dict(slot=access['slot'],
                        access=1 if access['side'] == 'reads' else 2, relation=access['relation'])
                        for access in accesses)))
                commands.append(command)
    solved = _stream.compile_page_effects(algebra.ctx, descriptions,
        [dict(order=order, owner=owner) for order, owner in occurrences])
    for source in solved['sources']:
        command = commands[source['index']]
        command['source'] = source['source']
        command['observed_slots'] = set(source['observed_slots'])
        command['published_slots'] = set(source['published_slots'])
        metadata = source['metadata']
        if metadata is not None:
            slot = len(command['operands'])
            # The shared source addresses fixed offsets from this composition,
            # whose binding may begin inside the backend buffer.
            view = TensorView.of(metadata)
            command['operands'] = [*command['operands'], view]
            command['types'] = (*command['types'], torch.uint8)
            command['binding_origins'] = (*command['binding_origins'], view.binding_origin)
            if command['kind'] in ('cpu', 'ane'):
                _, _, layout = _stream.page_backing(view)
                command['argument_specs'] = (*command['argument_specs'],
                    dict(slot=slot, kind='pointer', name='mesh_effects', type='uint8_t*',
                         dtype=torch.uint8, byte_offset=layout.index(0)))
        command['reaches'] = (*command['reaches'],
            *((slot, access, byte_operand(value)) for slot, access, value in source['reaches']))
    schedule.results(solved['results'])
    return programs


def prepare(steps, schedule, output=None):
    """Lower transfers, solve page effects, then bind/record the whole function.

    CommLower fixes one declaration-owned occurrence. CommEncode later consumes
    exactly that occurrence. Numerical code remains source throughout the effect
    solve; no compiled address record is inspected to recover a page relation.
    """
    from .native import Program
    from .compilation import bind, relations
    algebra = relations()

    def source_copy(value):
        # Copy mutable compiler records, retaining their page/source identities.
        if type(value) is dict:
            return {key: source_copy(item) for key, item in value.items()}
        if type(value) is list:
            return [source_copy(item) for item in value]
        if type(value) is tuple:
            return tuple(source_copy(item) for item in value)
        if type(value) is set:
            return set(value)
        return value

    declarations = []
    for step in steps:
        if isinstance(step, dict) and 'transfer' in step:
            if step['before']:
                declarations.append(dict(commands=step['before'], output=None, pages=(), configuration=step['configuration']))
            declarations.append(step)
            if step['after']:
                declarations.append(dict(commands=step['after'], output=None, pages=(), configuration=step['configuration']))
        else:
            declarations.append(step)
    programs, prepared, effects = [], [], []
    for index, step in enumerate(declarations):
        order = index
        if 'numerical' in step:
            programs.append(dict(step, order=order))
        elif 'transfer' not in step:
            program = source_copy(step)
            program['order'] = order
            programs.append(program)
        else:
            bound = (_stream.view_write_operands(step['bindings'], step['group'].handle, 0)
                     if step['transfer'] == 'write' else
                     _stream.Occurrence(step['group'].handle, step['operands']))
            bound.lower()
            prepared.append((order, bound))
            effects.extend(TensorView(storage, layout, 0) for storage, layout in bound.source_views(algebra.ctx))
    for index, declaration in enumerate(programs):
        if 'numerical' not in declaration:
            continue
        with bind(declaration['configuration'], algebra=algebra):
            retained = declaration['numerical']
            projection = refine(declaration['section'], declaration['projection'], declaration['placement']['storage'],
                                effects, algebra)
            retained.select_domains(projection['records'])
            program = retained.emit(declaration['operands'], **declaration['placement'])
            programs[index] = dict(program, order=declaration['order'], configuration=declaration['configuration'])
    if prepared:
        programs = transfer_effects(programs, prepared, schedule)
    for program in programs:
        with bind(program['configuration'], algebra=algebra):
            prepared.append((program['order'], Program.prepare(program)))
    for _, command in sorted(prepared, key=lambda item: item[0]):
        command.append(schedule)
