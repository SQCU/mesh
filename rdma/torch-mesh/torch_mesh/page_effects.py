"""Compile page production/observation into the commands which use those pages.

All sets below are preparation operands. The emitted program receives flat
group-indexed threshold/contribution pages and its ordinary payload pages.
It neither searches page ranges nor constructs dependency edges at execution.
"""
import math
import re

import torch

from . import _stream
from .operand import TensorView, byte_operand, index_relation, layout_configuration


def _physical(value, algebra):
    root, extent, layout = _stream.page_backing(value)
    relation = algebra.layout(layout_configuration(layout))
    return root, extent, relation


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
        root, _, physical = _physical(binding['operand'], algebra)
        original = algebra.layout(layout_configuration(binding['source'].components(binding['source_dtype'].itemsize)))
        placed[name] = root, physical, original
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
        root, _, relation = _physical(cut, algebra)
        positions = relation.image(relation.domain())
        affected = {name: original.image(physical.preimage(positions))
                    for name, (base, physical, original) in placed.items() if base == root}
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


def _group_accesses(command, algebra):
    """Project generated lane/loop coordinates onto physical page bytes."""
    from .operand import relation_layout
    if command['kind'] in ('cpu', 'ane'):
        reached = []
        for slot, mode, value in command['reaches']:
            root, extent, relation = _physical(value, algebra)
            positions = relation.image(relation.domain())
            selected = algebra.map((1,), (extent,), lambda x, y: [])
            selected = selected.subtract_range(selected.image(selected.domain()).binary(positions, 'subtract'))
            for side, flag in (('reads', 1), ('writes', 2)):
                if mode & flag:
                    reached.append(dict(side=side, root=root, extent=extent, relation=selected,
                                        positions=positions, slot=slot))
        return 1, tuple(reached)
    group, counts = _groups(command)
    total = math.prod(counts)
    bindings = {argument['slot']: argument for argument in command['arguments']}
    result = []
    for access in command['dispatch']:
        if 'group_relation' in access:
            root, extent, actual = _physical(command['operands'][access['slot']], algebra)
            dtype = command['types'][access['slot']]
            width = dtype.itemsize
            components = algebra.map((actual.shape[0] // width,), actual.shape,
                lambda x, y: [f'{width} * {x[0]} <= {y[0]} < {width} * ({x[0]} + 1)'])
            grouped = access['group_relation'].then(components).then(actual)
            result.append(dict(side=access['side'], root=root, extent=extent, relation=grouped,
                               positions=grouped.image(grouped.domain()), slot=access['slot']))
            continue
        argument = bindings[access['slot']]
        shape = access['shape']
        relation = index_relation(algebra, access['index'], access['variables'], shape)
        unit = argument['dtype'].itemsize
        components = algebra.map((argument['storage_bytes'] // unit,), (argument['storage_bytes'],),
            lambda x, y: [f'{unit} * {x[0]} <= {y[0]} < {unit} * ({x[0]} + 1)'])
        selected = algebra.layout(layout_configuration(argument['source'].components(argument['source_dtype'].itemsize)))
        root, extent, actual = _physical(command['operands'][access['slot']], algebra)
        reached = relation.then(components).then(selected.inverse((argument['storage_bytes'],))).then(actual)
        axes = {axis: index for index, axis in enumerate(access['active'])}
        def equations(x, y):
            indices = ['0', '0', '0']
            for axis, dim in axes.items():
                if axis not in access['reductions']:
                    indices[dim] = f'floor({x[axis]} / {group[dim]})'
            return [f'{y[0]} = {indices[0]} + {counts[0]} * ({indices[1]} + {counts[1]} * {indices[2]})']
        grouped = algebra.map(shape, (total,), equations).inverse((total,)).then(reached)
        result.append(dict(side=access['side'], root=root, extent=extent, relation=grouped,
                           positions=grouped.image(grouped.domain()), slot=access['slot']))
    return total, tuple(result)


def _table(rows, dtype):
    from . import empty
    shape = (len(rows), len(rows[0])) if rows else (0, 0)
    result = empty(shape, dtype=dtype, device='cpu')
    if rows:
        result.numpy()[:] = rows
    return result


def _append(command, value, mode):
    slot = len(command['operands'])
    view = byte_operand(value)
    command['operands'].append(view)
    command['types'] = (*command['types'], torch.uint8)
    command['binding_origins'] = (*command['binding_origins'], view.binding_origin)
    command['reaches'] = (*command['reaches'], (slot, mode, view))
    return slot, f'(mesh_effect_{slot} + mesh_index<{slot}>(0))'


def _metadata_pages(command, first, text):
    """Bind flat effect arrays at constant offsets in one composed page operand.

    Fields retain their within-page order; their backing pages may be scattered.
    Distinct fields may repeat a physical page in this virtual page selection.
    """
    views = command['operands'][first:]
    owners = [view.storage for view in views]
    _, page_bytes = _stream.page_domain(owners)
    pages, fields, offsets, cursor = [], {}, {}, 0
    for slot, value in enumerate(views, first):
        physical = _stream.physical_layout(value)
        if physical in fields:
            offsets[slot] = fields[physical]
            continue
        spans = physical.ordered_spans()
        offsets[slot] = fields[physical] = cursor * page_bytes + spans[0][0] % page_bytes
        previous = None
        for begin, count in spans:
            start, end = begin // page_bytes, (begin + count + page_bytes - 1) // page_bytes
            if previous == start:
                start += 1
            if end > start:
                pages.append((start, end - start));cursor += end - start
            previous = end - 1
    order = _stream.TensorLayout.runs(pages)
    storage = _stream.page_storage(owners, order, page_bytes)
    view = TensorView(storage, _stream.TensorLayout((storage.numel(),), (1,)), 0)
    for slot, offset in offsets.items():
        text = text.replace(f'(mesh_effect_{slot} + mesh_index<{slot}>(0))',
                            f'(mesh_effect_{first} + {offset}ul)')
    command['operands'] = [*command['operands'][:first], view]
    command['types'] = (*command['types'][:first], torch.uint8)
    command['binding_origins'] = (*command['binding_origins'][:first], 0)
    command['reaches'] = tuple((first, mode,
        TensorView(storage, _stream.TensorLayout((operand.numel(),), (1,), offsets[slot]), 0))
        if slot >= first else (slot, mode, operand) for slot, mode, operand in command['reaches'])
    return text


def _inject_cpu_many(command, records):
    """Emit the solved host effects once into the existing numerical entry."""
    from mesh import RDMA
    command['operands'] = list(command['operands'])
    fields, before, after, helpers = [], [], [], []

    def operand(value, mode):
        slot, _ = _append(command, value, mode)
        _, _, layout = _stream.page_backing(value)
        command['argument_specs'] = (*command['argument_specs'],
            dict(slot=slot, kind='pointer', name=f'mesh_effect_{slot}', type='uint8_t*', dtype=torch.uint8))
        fields.append(f'uint8_t* a{slot};')
        return f'(a.a{slot} + {layout.index(0)}ul)'

    for domain, record in enumerate(records):
        effects, thresholds, outputs = record['effects'], record['thresholds'][0], record['contributions'][0]
        for index, threshold in enumerate(thresholds):
            if not threshold:
                continue
            observation = effects['observations'][index]
            word = operand(observation['word'], 1 | 32)
            violation = operand(observation['violation'], 1 | 2 | 128)
            before.append(f'mesh_step_observe((volatile uint64_t *){word}, {threshold}ul, '
                f'{violation}, {observation["polls"]}u, {observation["channel"]}u, {observation["entry"]}ul);')
        if not outputs:
            continue
        cuts = operand(effects['cuts'], 1)
        completion = operand(effects['completion'], 1 | 2 | 128)
        totals = operand(record['expected'], 1)
        counters = operand(record['arrivals'], 1 | 2 | 128)
        members, values, publishes = ['const uint64_t *cuts;'], [f'(const uint64_t *){cuts}'], []
        for index, publication in enumerate(effects['publications']):
            word = operand(publication['word'], 2 | 128)
            members.append(f'volatile uint64_t *word{index};')
            values.append(f'(volatile uint64_t *){word}')
            publishes.append(f'mesh_publish(word{index}, cuts[(p - 1) * {2 + len(effects["observations"])}ul '
                             f'+ {2 + publication["observation"]}]);')
        prefix = f'MeshEffectPrefix{domain}'
        helpers.append('struct ' + prefix + ' {' + ''.join(members) +
                       'void at(uint32_t p) {' + ''.join(publishes) + '} };')
        after.append(f'{{ uint32_t *mesh_settled = (uint32_t *){completion};')
        for cut in outputs:
            after.append(f'if (__atomic_add_fetch((uint32_t *){counters} + {cut}, 1u, __ATOMIC_ACQ_REL) == '
                         f'((const uint32_t *){totals})[{cut}]) '
                         f'__atomic_store_n(mesh_settled + {cut}, 2u, __ATOMIC_RELEASE);')
        after.append(prefix + ' mesh_publish_prefix = {' + ','.join(values) + '};')
        count = len(effects['rows'])
        after.append(f'mesh_graph_settle(mesh_settled, 0, {count}u, mesh_settled + {count}, mesh_publish_prefix); }}')
    if not before and not after:
        return
    if after:
        after.insert(0, '__atomic_thread_fence(__ATOMIC_SEQ_CST);')
    if command['kind'] == 'ane':
        from .host import ane_source
        command['source'] = ane_source(fields, helpers, before, after)
        return
    source = command['source']
    structure = source.index('struct MeshCpuArguments {')
    end_structure = source.index('};', structure)
    source = source[:end_structure] + ''.join(fields) + source[end_structure:]
    entry = source.index('extern "C" void mesh_cpu_entry(')
    open_entry = source.index('{', entry)
    binding = source.index(';', open_entry) + 1
    end_entry = source.rindex('}')
    command['source'] = ('#include "' + str(RDMA) + '/mesh-publication.h"\n' + ''.join(helpers) +
        source[:binding] + ''.join(before) + source[binding:end_entry] + ''.join(after) + source[end_entry:])


def _inject_many(command, records):
    """Compose all solved page effects before the single source emission.

    Channels keep independent completed-prefix domains. They share the entry's
    observation fence and publication fence, and one group-indexed metadata row.
    """
    records = [record for record in records if any(any(row) for row in record['thresholds'])
               or any(record['contributions'])]
    if not records:
        return
    if command['kind'] in ('cpu', 'ane'):
        return _inject_cpu_many(command, records)
    command['operands'] = list(command['operands'])
    initial = len(command['operands'])
    rows = [[] for _ in records[0]['thresholds']]
    domains, stride = [], 0
    for record in records:
        observations = [i for i in range(len(record['effects']['observations']))
                        if any(row[i] for row in record['thresholds'])]
        publishing = max(map(len, record['contributions']), default=0)
        domains.append((record, observations, publishing, stride))
        for row, thresholds, cuts in zip(rows, record['thresholds'], record['contributions'], strict=True):
            row.extend([*(thresholds[i] for i in observations), *cuts, *([-1] * (publishing - len(cuts)))])
        stride += len(observations) + publishing
    _, table = _append(command, _table(rows, torch.int64), 1)
    _, groups = _groups(command)
    group_index = f'(mesh_group.x + {groups[0]}u * (mesh_group.y + {groups[1]}u * mesh_group.z))'
    before = [f'device const long *mesh_effect_row = (device const long *){table} + {stride}ul * {group_index};']
    reads, writes, helpers = [], [], []
    for domain, (record, observations, publishing, column_start) in enumerate(domains):
        effects = record['effects']
        for column, index in enumerate(observations, column_start):
            observation = effects['observations'][index]
            _, word = _append(command, observation['word'], 1 | 32)
            _, violation = _append(command, observation['violation'], 1 | 2 | 128)
            reads.append(f'mesh_step_observe({word}, ulong(mesh_effect_row[{column}]), {violation}, '
                         f'{observation["polls"]}u, {observation["channel"]}u, {observation["entry"]}ul);')
        if not publishing:
            continue
        _, cuts = _append(command, effects['cuts'], 1)
        _, completion = _append(command, effects['completion'], 1 | 2 | 128)
        _, totals = _append(command, record['expected'], 1)
        _, counters = _append(command, record['arrivals'], 1 | 2 | 128)
        fields, values, publishes = ['device const ulong *cuts;'], [f'(device const ulong *){cuts}'], []
        for index, publication in enumerate(effects['publications']):
            _, word = _append(command, publication['word'], 2 | 128)
            fields.append(f'volatile coherent(system) device ulong *word{index};')
            values.append(f'(volatile coherent(system) device ulong *){word}')
            column = 2 + publication['observation']
            publishes.append(f'mesh_publish(word{index}, cuts[(p - 1) * {2 + len(effects["observations"])}ul + {column}]);')
        prefix = f'mesh_effect_prefix_{domain}'
        helpers.append('struct ' + prefix + ' {' + ''.join(fields) +
                       'void at(uint p) {' + ''.join(publishes) + '} };\n')
        writes.append(f'{{ device atomic_uint *mesh_settled = (device atomic_uint *){completion};')
        for column in range(column_start + len(observations), column_start + len(observations) + publishing):
            writes.append(f'{{ const long cut = mesh_effect_row[{column}]; if (cut >= 0) {{ '
                f'uint arrived = atomic_fetch_add_explicit((device atomic_uint *){counters} + cut, 1u, memory_order_relaxed) + 1u; '
                f'if (arrived == ((device const uint *){totals})[cut]) '
                'atomic_store_explicit(mesh_settled + cut, 2u, memory_order_relaxed); } }')
        writes.append(prefix + ' mesh_publish_prefix = {' + ','.join(values) + '};')
        count = len(effects['rows'])
        writes.append(f'mesh_graph_settle(mesh_settled, 0, {count}u, mesh_settled + {count}, mesh_publish_prefix); }}')
    fence = ('atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, '
             'static_cast<thread_scope>(MESH_SYSTEM_THREAD_SCOPE));')
    if reads:
        before.extend(('if (mesh_lane == 0) {' + ''.join(reads) + '}',
                       'threadgroup_barrier(mem_flags::mem_device);', fence))
    after = ('threadgroup_barrier(mem_flags::mem_device); if (mesh_lane == 0) { ' +
             fence + ''.join(writes) + ' }') if writes else ''
    combined = _metadata_pages(command, initial, '\n'.join(helpers) + '\n@@BEFORE@@\n' + '\n'.join(before) +
                               '\n@@AFTER@@\n' + after)
    helpers, combined = combined.split('\n@@BEFORE@@\n')
    before, after = combined.split('\n@@AFTER@@\n')
    command['source'] = shader_source(command, initial, helpers, before, after)


def shader_source(command, initial, helpers, before, after):
    """Compose effects into retained source; page binding is a separate operand."""
    source = command['source']
    from .shader import entries, body_scope
    def compose(parameters, body):
        added = (', ' if parameters.strip() else '') + f'device uchar *mesh_effect_{initial} [[buffer({initial})]]'
        aliases = []
        for attribute, target in (('thread_index_in_threadgroup', 'mesh_lane'), ('threadgroup_position_in_grid', 'mesh_group')):
            declared = re.search(r'\b(uint[123]?|ushort[123]?)\s+(\w+)\s*\[\[' + attribute + r'\]\]', parameters)
            if declared is None:
                geometry = re.search(r'\b(uint[123]?|ushort[123]?)\s+\w+\s*\[\[(?:thread_position_in_grid|thread_position_in_threadgroup|threads_per_threadgroup|threads_per_grid)\]\]', parameters)
                kind = 'uint' if target == 'mesh_lane' else geometry.group(1) if geometry is not None else 'uint3'
                name = target if target == 'mesh_lane' or kind == 'uint3' else target + '_argument'
                added += f', {kind} {name} [[{attribute}]]'
                if name != target:
                    expression = f'uint3({name}, 0)' if kind.endswith('2') else f'uint3({name})' if kind.endswith('3') else f'uint3({name}, 0, 0)'
                    aliases.append(f'const auto {target} = {expression};')
            elif declared.group(2) != target:
                kind, name = declared.groups()
                expression = name if target == 'mesh_lane' or kind == 'uint3' else f'uint3({name}, 0)' if kind.endswith('2') else f'uint3({name})' if kind.endswith('3') else f'uint3({name}, 0, 0)'
                aliases.append(f'const auto {target} = {expression};')
        if command.get('names'):
            pieces = parameters.split(',')
            for slot in command.get('observed_slots', set()) | command.get('published_slots', set()):
                pieces[slot] = re.sub(r'\bconstant\s+', 'volatile coherent(system) device const ', pieces[slot])
                if 'coherent(system)' not in pieces[slot]:
                    pieces[slot] = re.sub(r'\bdevice\s+', 'volatile coherent(system) device ', pieces[slot])
            parameters = ','.join(pieces)
        elif 'MESH_MATRIX_' not in source or '#define MESH_ENTRY_QUALIFIERS' in source:
            for slot in command.get('observed_slots', set()) | command.get('published_slots', set()):
                parameters = re.sub(r'\b(device|constant)\s+([^,]*?\[\[buffer\(' + str(slot) + r'\)\]\])',
                    lambda match: 'volatile coherent(system) device ' + ('const ' if match.group(1) == 'constant' else '') + match.group(2), parameters)
        return parameters + added, ''.join(aliases) + before + body_scope(body) + after
    source = entries(source, command['function'], compose)
    observed = sum(1 << slot for slot in command.get('observed_slots', ()))
    published = sum(1 << slot for slot in command.get('published_slots', ()))
    return (f'#undef MESH_OBSERVED_BINDINGS\n#define MESH_OBSERVED_BINDINGS {observed}ul\n'
            f'#undef MESH_PUBLISHED_BINDINGS\n#define MESH_PUBLISHED_BINDINGS {published}ul\n' + helpers + source)


def lower(programs, effects, *, emit=True):
    """One exact group/page projection for every typed numerical command."""
    if not effects['rows']:
        return []
    from mesh_relation import Algebra
    from . import empty
    algebra = Algebra()
    from .native import declared_commands
    for program in programs:
        program['commands'] = tuple(declared_commands(program['commands']))
    commands = [(command, program.get('order', 0)) for program in programs for command in program['commands']]
    projected = []
    for command, order in commands:
        if command['kind'] not in ('shader', 'cpu', 'ane') or command['kind'] == 'shader' and command.get('dispatch') is None:
            raise ValueError('page-effect lowering requires the numerical command\'s dispatch-to-page relation')
        count, accesses = _group_accesses(command, algebra)
        projected.append(dict(command=command, accesses=accesses, order=order,
            thresholds=[[0] * len(effects['observations']) for _ in range(count)],
            contributions=[set() for _ in range(count)]))
    for observation_index, observation in enumerate(effects['observations']):
        if observation.get('publication_only'):
            continue
        destination = observation['source'] == len(effects['sources'])
        source = TensorView.of(effects['destination'] if destination else
                               effects['sources'][observation['source']])
        side = 'writes' if destination else 'reads'
        for cut_index, row in enumerate(effects['rows']):
            selected = TensorView(source.storage, source.layout.reshape((source.numel(),)).select_runs(
                0, [(row[0] * effects['unit'], row[1] * effects['unit'])]))
            root, extent, relation = _physical(selected, algebra)
            remaining = relation.image(relation.domain())
            for program in projected:
                if program['order'] <= observation.get('after', -1):
                    continue
                for access in program['accesses']:
                    if access['root'] != root:
                        continue
                    if access['side'] == side:
                        spans = access['relation'].preimage(remaining).spans()
                        if spans:
                            program['command'].setdefault('observed_slots', set()).add(access['slot'])
                        for first, width, stride, repeats in spans:
                            for group in (first + repetition * stride + lane
                                          for repetition in range(repeats) for lane in range(width)):
                                program['thresholds'][group][observation_index] = max(
                                    program['thresholds'][group][observation_index], row[2 + observation_index])
                for access in program['accesses']:
                    if access['root'] == root and access['side'] == 'writes':
                        remaining = remaining.binary(access['positions'], 'subtract')
    expected = []
    targets = effects.get('targets')
    for cut_index, row in enumerate(effects['rows']) if effects['publications'] else ():
        target = TensorView.of(effects['destination'] if targets is None else targets[cut_index])
        selected = (TensorView(target.storage, target.layout.reshape((target.numel(),)).select_runs(
            0, [(row[0] * effects['unit'], row[1] * effects['unit'])])) if targets is None else target)
        root, extent, relation = _physical(selected, algebra)
        remaining = relation.image(relation.domain())
        if 'initial_positions' in effects:
            remaining = remaining.binary(effects['initial_positions'][cut_index], 'subtract')
        for program in reversed(projected):
            if 'producer_before' in effects and program['order'] >= effects['producer_before'][cut_index]:
                continue
            writes = [item for item in program['accesses'] if item['root'] == root and item['side'] in ('writes', 'available')]
            for access in writes:
                spans = access['relation'].preimage(remaining).spans()
                if spans:
                    program['command'].setdefault('published_slots', set()).add(access['slot'])
                for first, width, stride, repeats in spans:
                    for group in (first + repetition * stride + lane
                                  for repetition in range(repeats) for lane in range(width)):
                        program['contributions'][group].add(cut_index)
            for access in writes:
                remaining = remaining.binary(access['positions'], 'subtract')
        if not remaining.empty():
            raise ValueError('published output pages require their numerical producer relation')
        expected.append(sum(cut_index in cells for program in projected for cells in program['contributions']))
    totals = _table([[value] for value in expected], torch.uint32)
    arrivals = empty((len(expected),), dtype=torch.uint32, device='cpu')
    arrivals.zero_()
    solved = [dict(command=program['command'], thresholds=program['thresholds'],
        contributions=[sorted(values) for values in program['contributions']],
        effects=effects, expected=totals, arrivals=arrivals) for program in projected]
    if emit:
        for record in solved:
            _inject_many(record['command'], (record,))
    return solved


def _effect(rows, *, sources=(), destination=None, observations=(), publications=(), completed=(), **extra):
    from . import empty
    completion = empty((len(rows) + 1,), dtype=torch.uint32, device='cpu')
    completion.zero_()
    if completed:
        completion.numpy()[list(completed)] = 2
    return dict(rows=tuple(rows), cuts=_table(rows, torch.uint64), completion=completion,
                sources=tuple(sources), destination=destination, observations=tuple(observations),
                publications=tuple(publications), unit=1, **extra)


def transfer_effects(programs, records, schedule):
    """Solve exact produced/received page versions before any kernel is bound.

    Native placement supplies typed page words and selected payload ranges.
    The same projection used for reductions inserts those effects into their
    real numerical producers and consumers. External values contribute only
    their publication/observation boundary, never a staging copy.
    """
    from mesh_relation import Algebra
    from .native import declared_commands
    from .operand import relation_view
    algebra = Algebra()
    for program in programs:
        program['commands'] = tuple(declared_commands(program['commands']))
    publishers, observers = {}, {}
    for record in records:
        root, _, layout = _stream.page_backing(record['word'])
        key = root, layout.index(0)
        if record['kind'] == 'publish':
            publishers.setdefault(key, []).append(record)
        else:
            observers.setdefault((record['order'], record['call'], key), []).append(record)

    accesses = [(program['order'], access) for program in programs for command in program['commands']
                for access in _group_accesses(command, algebra)[1]]
    arrivals = sorted((record for group in observers.values() for record in group),
                      key=lambda record: record['order'], reverse=True)
    initial_inputs = {}

    # Entry values are descriptor inputs. Earlier WRITE arrivals become those
    # descriptors' actual prerequisites; no numerical dispatch stands in for
    # observing and republishing them.
    for rows in publishers.values():
        for record in rows:
            selected = record['selected']
            root, _, relation = _physical(selected, algebra)
            missing = relation.image(relation.domain())
            for order, access in accesses:
                if order < record['order'] and access['root'] == root and access['side'] == 'writes':
                    missing = missing.binary(access['positions'], 'subtract')
            record['initial_positions'] = missing
            if not missing.empty():
                domain = relation.preimage(missing)
                selection = algebra.selection_relation(dict(shape=[selected.numel()], steps=[['select', 0, domain.spans()]]))
                record['initial_view'] = relation_view(selected, selection, selection.shape)
                for first, width, stride, repeats in domain.spans():
                    for repetition in range(repeats):
                        begin = first + repetition * stride
                        part = algebra.set(f'{{ [i] : {begin} <= i < {begin + width} }}', 1)
                        dependencies, remaining = [], relation.image(part)
                        for arrival in arrivals:
                            if arrival['order'] >= record['order']:
                                continue
                            received_root, _, received = _physical(arrival['selected'], algebra)
                            if received_root != root:
                                continue
                            covered = received.image(received.domain()).binary(remaining, 'intersect')
                            if not covered.empty():
                                dependencies.append(arrival)
                                remaining = remaining.binary(covered, 'subtract')
                        initial_inputs.setdefault(record['owner'], []).append(dict(call=record['call'],
                            frame=record['frame'], first=record['first'] + begin,
                            bytes=width, observations=dependencies))
    for owner, inputs in initial_inputs.items():
        owner.source_inputs(inputs)
    programs.sort(key=lambda program: program['order'])
    solved = {}
    def collect(effects):
        for record in lower(programs, effects, emit=False):
            solved.setdefault(id(record['command']), []).append(record)
    for records in publishers.values():
        records.sort(key=lambda row: row['threshold'])
        first = records[0]
        rows = [(0, record['bytes'], record['threshold']) for record in records]
        effects = _effect(rows, observations=(dict(publication_only=True),),
            publications=(dict(word=first['word'], observation=0),),
            targets=tuple(record['selected'] for record in records),
            producer_before=tuple(record['order'] for record in records),
            initial_positions=tuple(record['initial_positions'] for record in records),
            initial_views=tuple(record['initial_view'] for record in records if 'initial_view' in record),
            completed=tuple(index for index, record in enumerate(records)
                if record['initial_positions'].equal(_physical(record['selected'], algebra)[2].image(
                    _physical(record['selected'], algebra)[2].domain()))))
        collect(effects)
    for records in observers.values():
        first = records[0]
        observation = {key: first[key] for key in ('word', 'violation', 'polls', 'channel', 'entry')}
        observation.update(source=0, after=first['order'])
        effects = _effect([(record['first'], record['bytes'], record['threshold']) for record in records],
            sources=(first['view'],), observations=(observation,))
        collect(effects)
    consumed = {}
    for records in solved.values():
        command = records[0]['command']
        prerequisites = {}
        for record in records:
            for index, observation in enumerate(record['effects']['observations']):
                if observation.get('publication_only'):
                    continue
                root, _, layout = _stream.page_backing(observation['word'])
                key = root, layout.index(0)
                threshold = max((row[index] for row in record['thresholds']), default=0)
                consumed[key] = max(consumed.get(key, 0), threshold)
            if any(record['contributions']):
                for view in record['effects'].get('initial_views', ()):
                    prerequisites[_stream.physical_layout(view)] = view
        command['reaches'] = (*command['reaches'], *((-1, 1, view) for view in prerequisites.values()))
        _inject_many(command, records)
    # Numerical completion already entails every arrival that its commands
    # observed. Only terminal words beyond those thresholds remain result
    # operands, observed by existing nonblocking status, once per physical word.
    terminal = {}
    for record in arrivals:
        root, _, layout = _stream.page_backing(record['word'])
        key = root, layout.index(0)
        if record['threshold'] > consumed.get(key, 0):
            previous = terminal.get(key)
            if previous is None or previous[1] < record['threshold']:
                terminal[key] = record['word'], record['threshold']
    schedule.results(list(terminal.values()))
    return programs


def prepare(steps, schedule, output=None):
    """Lower transfers, solve page effects, then bind/record the whole function.

    CommLower fixes one declaration-owned occurrence. CommEncode later consumes
    exactly that occurrence. Numerical code remains source throughout the effect
    solve; no compiled address record is inspected to recover a page relation.
    """
    from .native import Program

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
                declarations.append(dict(commands=step['before'], output=None, pages=()))
            declarations.append(step)
            if step['after']:
                declarations.append(dict(commands=step['after'], output=None, pages=()))
        else:
            declarations.append(step)
    programs, prepared, effects = [], [], []
    for index, step in enumerate(declarations):
        order = index * 4
        if 'numerical' in step:
            programs.append(dict(step, order=order))
        elif 'transfer' not in step:
            program = source_copy(step)
            program['order'] = order
            programs.append(program)
        elif step['transfer'] == 'write':
            bound = _stream.view_write_operands(step['bindings'], step['group'].handle, 0)
            bound.lower()
            prepared.append((order, bound, True))
            for record in bound.page_effects():
                view = byte_operand(step['views'][record['call']])
                selected = TensorView(view.storage, view.layout.reshape((view.numel(),)).select_runs(
                    0, [(record['first'], record['bytes'])]), view.binding_origin)
                effects.append(dict(record, order=order, view=view, selected=selected, owner=bound))
        else:
            bound = _stream.Collectives(step['bindings'], step['group'].handle, step['operands'])
            bound.lower()
            prepared.append((order, bound, False))
    for index, declaration in enumerate(programs):
        if 'numerical' not in declaration:
            continue
        retained = declaration['numerical']
        projection = refine(declaration['section'], declaration['projection'], declaration['placement']['storage'],
                            [record['selected'] for record in effects], declaration['algebra'])
        retained.select_domains(projection['records'])
        program = retained.emit(declaration['operands'], **declaration['placement'])
        programs[index] = dict(program, order=declaration['order'])
    if effects:
        programs = transfer_effects(programs, effects, schedule)
    prepared.extend((program['order'], Program.prepare(program), False) for program in programs)
    for _, command, write in sorted(prepared, key=lambda item: item[0]):
        if write:
            schedule.writes(command)
        else:
            command.append(schedule)
