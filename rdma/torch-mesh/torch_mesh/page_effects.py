"""Compile page production/observation into the commands which use those pages.

All sets below are preparation operands. The emitted program receives flat
group-indexed threshold/contribution pages and its ordinary payload pages.
It neither searches page ranges nor constructs dependency edges at execution.
"""
import math
from dataclasses import dataclass

import torch

from . import _stream
from .operand import TensorView, byte_operand, index_relation, layout_configuration


@dataclass(frozen=True)
class Coverage:
    """Declared UInt32 active/count page operands, shared with transport.

    The fixed cut domain is explicit. Values are produced by the ordinary
    index/count graph; this object neither evaluates nor caches their contents.
    """
    active: object
    expected: object
    cuts: int


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
        started = _compose_record(command, access, physical[access['slot']])
        relation = access['relation'].then(physical[access['slot']])
        _compose_record(command, access, None, started)
        result.append(dict(access, relation=relation, positions=relation.image(relation.domain())))
    return total, tuple(result)


_COMPOSE = None


def _compose_record(command, access, physical, started=None):
    """Each access's composition with its pages, one JSON line before (its operands' isl text) and one after (its
    seconds), in MESH_PREPARE_RECORD's directory: a rank ended while composing names the composition and its operands
    in its last line. A log of preparation, read by people; nothing reads it back. Added for the LSTM demonstration,
    whose realization sat in one MeshIndexCompose's isl_map_compute_divs on every rank (2026-11-01)."""
    import json, os, time
    global _COMPOSE
    if _COMPOSE is None:
        directory = os.environ.get('MESH_PREPARE_RECORD')
        _COMPOSE = open(os.path.join(directory, f'compose-{os.getpid()}.jsonl'), 'w') if directory else False
    if not _COMPOSE:
        return None
    line = dict(command=command.get('name', command.get('kind')), kind=command['kind'], slot=access['slot'],
                side=access['side'])
    if started is None:
        line.update(logical=str(access['relation']), physical=str(physical))
    else:
        line.update(seconds=round(time.monotonic() - started, 6))
    _COMPOSE.write(json.dumps(line) + '\n')
    _COMPOSE.flush()
    return time.monotonic()


def _site_accesses(command, site, algebra):
    """Place a source-local cell's declared accesses on the same page roots."""
    logical = logical_dispatch_accesses(dict(command, dispatch=site['dispatch']), algebra)
    physical = {access['slot']: _physical(command['operands'][access['slot']], algebra) for access in logical}
    return tuple(dict(slot=access['slot'], access=1 if access['side'] == 'reads' else 2,
                      relation=access['relation'].reshape((site['cells'],)).then(physical[access['slot']])) for access in logical)


def describe(programs):
    """Adapt retained frontend declarations to the common page-effect compiler."""
    from .compilation import relations, bind
    from .native import declared_commands
    algebra = relations()
    descriptions = []
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
                    sites=tuple(dict(name=site['name'], cells=site['cells'], scope=site['scope'],
                                     join_scope=site.get('join_scope', 0),
                                     accesses=_site_accesses(command, site, algebra))
                                for site in command.get('sites', ())),
                    contributions=tuple(command.get('contributions', ())),
                    announcements=tuple(command.get('announcements', ())),
                    accesses=tuple(dict(slot=access['slot'],
                        access=1 if access['side'] == 'reads' else 2, relation=access['relation'])
                        for access in accesses)))
    return descriptions


def bind_effects(programs, occurrences, schedule, declaration):
    commands = [command for program in programs for command in program['commands']]
    solved = declaration.bind(
        [dict(order=order, owner=owner, frame_orders=frames) for order, owner, frames in occurrences])
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
    return programs, solved['recordings']



def source_copy(value):
    """Copy mutable compiler records, retaining page and source identities."""
    if type(value) is dict:
        return {key: source_copy(item) for key, item in value.items()}
    if type(value) is list:
        return [source_copy(item) for item in value]
    if type(value) is tuple:
        return tuple(source_copy(item) for item in value)
    if type(value) is set:
        return set(value)
    return value


class Plan:
    """One immutable source declaration and its solved occurrence families.

    Numerical source, launch/access relations and page assignments are retained
    here during preparation. Occurrences bind fresh absolute transport words
    and private prefix pages; execution still submits only recorded functions.
    """
    def __init__(self, steps):
        from .compilation import relations
        self.algebra = relations()
        self.declarations, self.positions = [], []
        for original, step in enumerate(steps):
            first = len(self.declarations)
            if 'transfer' in step:
                if step['before']:
                    self.declarations.append(dict(commands=step['before'], output=None, pages=(), configuration=step['configuration']))
                self.declarations.append(step)
                if step['after']:
                    self.declarations.append(dict(commands=step['after'], output=None, pages=(), configuration=step['configuration']))
            else:
                self.declarations.append(step)
            self.positions.extend([original] * (len(self.declarations) - first))
        self.origins = {original: 2 * index for index, original in enumerate(self.positions)}
        # Each declared binding phase retains both its exact numerical cut
        # refinement and the dependency solution compiled for those cells.
        self.variants = {}

    def append(self, schedule):
        from .native import Program
        from .compilation import bind
        prepared, phase = [], []
        for index, step in enumerate(self.declarations):
            if 'transfer' not in step:
                continue
            bound = (_stream.view_write_operands(step['bindings'], step['group'].handle, 0)
                     if step['transfer'] == 'write' else
                     _stream.Occurrence(step['group'].handle, step['operands']))
            bound.lower(step.get('invocation', 0))
            frames = tuple(self.origins[self.positions[index] + offset] + 1 for offset in step.get('frame_orders', ()))
            prepared.append((2 * index, bound, frames))
            phase.append(tuple(bound.schema(self.algebra.ctx)))
        phase = tuple(phase)
        if phase not in self.variants:
            cuts = [TensorView(storage, layout, 0) for _, bound, _ in prepared
                    for storage, layout in bound.source_views(self.algebra.ctx)]
            programs, emitted = [], {}
            for index, step in enumerate(self.declarations):
                if 'transfer' in step:
                    continue
                if 'numerical' not in step:
                    programs.append(dict(source_copy(step), order=2 * index))
                    continue
                # Repeated uses of this exact retained numerical declaration
                # share its emission. The enclosing plan owns all these objects.
                key = tuple(id(step[name]) for name in
                            ('numerical', 'section', 'projection', 'placement', 'operands', 'configuration'))
                if key not in emitted:
                    with bind(step['configuration'], algebra=self.algebra):
                        retained = step['numerical']
                        projection = refine(step['section'], step['projection'], step['placement']['storage'], cuts, self.algebra)
                        retained.select_domains(projection['records'])
                        emitted[key] = retained.emit(step['operands'], **step['placement'])
                programs.append(dict(source_copy(emitted[key]), order=2 * index, configuration=step['configuration']))
            descriptions = describe(programs)
            effects = _stream.PageEffectsPlan(self.algebra, descriptions)
            self.variants[phase] = tuple(programs), effects
        retained, effects = self.variants[phase]
        programs = source_copy(retained)
        programs, prepared = bind_effects(programs, prepared, schedule, effects)
        for program in programs:
            with bind(program['configuration'], algebra=self.algebra):
                prepared.append((program['order'], Program.prepare(program), None))
        for _, command, selected in sorted(prepared, key=lambda item: item[0]):
            if selected is None:
                command.append(schedule)
            else:
                command.append(schedule, selected)
