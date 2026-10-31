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


def refine(section, projection, effects, algebra):
    """Partition CPU work by the supplied output cuts using normal demand flow.

    Metal expresses these independent tiles as one grid. CPU expresses them
    as ordinary native tasks. Both select the same scalar producer domains;
    no host task contains a scheduler or invokes another numerical executor.
    """
    from .compilation import implementation_assignment, implementations
    from mesh_compute import assigned_implementation
    from .ownership import project_numerical
    if not effects['rows']:
        return projection
    choices, catalog = implementation_assignment()['scalar'], implementations()
    cpu = {item['operation'].get_operation_name() for item in section['retained'].consumer_constraints()
           if catalog[assigned_implementation(item['required'], choices)]['compiler'] == 'cpp_scalar'}
    if not cpu:
        return projection
    records = [record for record in projection['records']
               if record['effect']['operation'].get_operation_name() not in cpu]
    seen, finished = {}, {}
    for first, count, *_ in effects['rows']:
        domain = algebra.set(f'{{ [i] : {first * effects["unit"]} <= i < {(first + count) * effects["unit"]} }}', 1)
        demands = {key: positions.binary(domain, 'intersect') for key, positions in projection['outputs'].items()}
        selected = project_numerical(section, demands, algebra)
        for record in selected['records']:
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
    return dict(projection, records=tuple(records))


def _group_accesses(command, algebra):
    """Project generated lane/loop coordinates onto physical page bytes."""
    from .operand import relation_layout
    if command['kind'] == 'cpu':
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


def _inject_cpu(command, thresholds, contributions, effects, expected, arrivals):
    """Put a cut's page effects in its existing compiled native task body."""
    observations = [index for index, threshold in enumerate(thresholds[0]) if threshold]
    outputs = contributions[0]
    if not observations and not outputs:
        return
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
    for index in observations:
        observation = effects['observations'][index]
        word = operand(observation['word'], 1 | 32)
        violation = operand(observation['violation'], 1 | 2 | 128)
        before.append(f'mesh_step_observe((volatile uint64_t *){word}, {thresholds[0][index]}ul, '
                      f'{violation}, {observation["polls"]}u, {observation["channel"]}u, {observation["entry"]}ul);')
    if outputs:
        cuts = operand(effects['cuts'], 1)
        completion = operand(effects['completion'], 1 | 2 | 128)
        totals = operand(expected, 1)
        counters = operand(arrivals, 1 | 2 | 128)
        members, values, publishes = ['const uint64_t *cuts;'], [f'(const uint64_t *){cuts}'], []
        for index, publication in enumerate(effects['publications']):
            word = operand(publication['word'], 2 | 128)
            members.append(f'volatile uint64_t *word{index};')
            values.append(f'(volatile uint64_t *){word}')
            publishes.append(f'mesh_publish(word{index}, cuts[(p - 1) * {2 + len(effects["observations"])}ul '
                             f'+ {2 + publication["observation"]}]);')
        helpers.append('struct MeshEffectPrefix {' + ''.join(members) +
                       'void at(uint32_t p) {' + ''.join(publishes) + '} };')
        after.append('__atomic_thread_fence(__ATOMIC_SEQ_CST);')
        after.append(f'uint32_t *mesh_settled = (uint32_t *){completion};')
        for cut in outputs:
            after.append(f'if (__atomic_add_fetch((uint32_t *){counters} + {cut}, 1u, __ATOMIC_ACQ_REL) == '
                         f'((const uint32_t *){totals})[{cut}]) '
                         f'__atomic_store_n(mesh_settled + {cut}, 2u, __ATOMIC_RELEASE);')
        after.append('MeshEffectPrefix mesh_publish_prefix = {' + ','.join(values) + '};')
        count = len(effects['rows'])
        after.append(f'mesh_graph_settle(mesh_settled, 0, {count}u, mesh_settled + {count}, mesh_publish_prefix);')
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


def _inject(command, thresholds, contributions, effects, expected, arrivals):
    """Emit actual dependency effects in a numerical command's entry/exit."""
    if command['kind'] == 'cpu':
        return _inject_cpu(command, thresholds, contributions, effects, expected, arrivals)
    observations = [i for i in range(len(effects['observations'])) if any(row[i] for row in thresholds)]
    publishing = max(map(len, contributions), default=0)
    if not observations and not publishing:
        return
    command['operands'] = list(command['operands'])
    initial = len(command['operands'])
    threshold_rows = [[row[i] for i in observations] for row in thresholds]
    table_rows = [values + [*cuts, *([-1] * (publishing - len(cuts)))]
                  for values, cuts in zip(threshold_rows, contributions, strict=True)]
    table = _table(table_rows, torch.int64)
    _, table_pointer = _append(command, table, 1)
    stride = len(observations) + publishing
    _, groups = _groups(command)
    group_index = f'(mesh_group.x + {groups[0]}u * (mesh_group.y + {groups[1]}u * mesh_group.z))'
    before = [f'device const long *mesh_effect_row = (device const long *){table_pointer} + {stride}ul * {group_index};']
    after, helpers = [], []
    if observations:
        before.append('if (mesh_lane == 0) {')
        for column, index in enumerate(observations):
            observation = effects['observations'][index]
            _, word = _append(command, observation['word'], 1 | 32)
            _, violation = _append(command, observation['violation'], 1 | 2 | 128)
            before.append(f'mesh_step_observe({word}, ulong(mesh_effect_row[{column}]), {violation}, '
                          f'{observation["polls"]}u, {observation["channel"]}u, {observation["entry"]}ul);')
        before.append('}\nthreadgroup_barrier(mem_flags::mem_device);')
        before.append('atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, '
                      'static_cast<thread_scope>(MESH_SYSTEM_THREAD_SCOPE));')
    if publishing:
        _, cuts = _append(command, effects['cuts'], 1)
        _, completion = _append(command, effects['completion'], 1 | 2 | 128)
        _, totals = _append(command, expected, 1)
        _, counters = _append(command, arrivals, 1 | 2 | 128)
        fields, values, publishes = ['device const ulong *cuts;'], [f'(device const ulong *){cuts}'], []
        for index, publication in enumerate(effects['publications']):
            _, word = _append(command, publication['word'], 2 | 128)
            fields.append(f'volatile coherent(system) device ulong *word{index};')
            values.append(f'(volatile coherent(system) device ulong *){word}')
            column = 2 + publication['observation']
            publishes.append(f'mesh_publish(word{index}, cuts[(p - 1) * {2 + len(effects["observations"])}ul + {column}]);')
        helpers.append('struct mesh_effect_prefix {' + ''.join(fields) +
                       'void at(uint p) {' + ''.join(publishes) + '} };\n')
        after.append('threadgroup_barrier(mem_flags::mem_device);\nif (mesh_lane == 0) {')
        after.append('atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, '
                     'static_cast<thread_scope>(MESH_SYSTEM_THREAD_SCOPE));')
        after.append(f'device atomic_uint *mesh_settled = (device atomic_uint *){completion};')
        for column in range(publishing):
            after.append(f'{{ const long cut = mesh_effect_row[{len(observations) + column}]; if (cut >= 0) {{ '
                f'uint arrived = atomic_fetch_add_explicit((device atomic_uint *){counters} + cut, 1u, memory_order_relaxed) + 1u; '
                f'if (arrived == ((device const uint *){totals})[cut]) '
                'atomic_store_explicit(mesh_settled + cut, 2u, memory_order_relaxed); } }')
        after.append('mesh_effect_prefix mesh_publish_prefix = {' + ','.join(values) + '};')
        count = len(effects['rows'])
        after.append(f'mesh_graph_settle(mesh_settled, 0, {count}u, mesh_settled + {count}, mesh_publish_prefix);\n}}')
    source = command['source']
    added = ''.join(f', device uchar *mesh_effect_{slot} [[buffer({slot})]]'
                    for slot in range(initial, len(command['operands'])))
    added += ', uint mesh_lane [[thread_index_in_threadgroup]], uint3 mesh_group [[threadgroup_position_in_grid]]'
    matches = tuple(re.finditer(r'kernel\s+void\s+' + re.escape(command['function']) + r'\s*\(', source))
    for match in reversed(matches):
        start, cursor, depth = match.end(), match.end(), 1
        while depth:
            depth += (source[cursor] == '(') - (source[cursor] == ')')
            cursor += 1
        end = cursor - 1
        opening = source.index('{', cursor)
        closing, depth = opening + 1, 1
        while depth:
            depth += (source[closing] == '{') - (source[closing] == '}')
            closing += 1
        parameters = source[start:end].rstrip()
        if command.get('names'):
            pieces = parameters.split(',')
            for slot in command.get('observed_slots', ()):
                pieces[slot] = re.sub(r'\bconstant\s+', 'volatile coherent(system) device const ', pieces[slot])
                if 'coherent(system)' not in pieces[slot]:
                    pieces[slot] = re.sub(r'\bdevice\s+', 'volatile coherent(system) device ', pieces[slot])
            parameters = ','.join(pieces)
        source = (source[:start] + parameters + added + source[end:opening + 1] +
                  '\n'.join(before) + source[opening + 1:closing - 1] + '\n'.join(after) + source[closing - 1:])
    observed = sum(1 << slot for slot in command.get('observed_slots', ()))
    command['source'] = f'#define MESH_OBSERVED_BINDINGS {observed}ul\n' + ''.join(helpers) + source


def lower(programs, effects):
    """One exact group/page projection for every typed numerical command."""
    if not effects['rows']:
        return
    from mesh_relation import Algebra
    from . import empty
    algebra = Algebra()
    from .native import declared_commands
    for program in programs:
        program['commands'] = tuple(declared_commands(program['commands']))
    commands = [command for program in programs for command in program['commands']]
    projected = []
    for command in commands:
        if command['kind'] not in ('shader', 'cpu') or command['kind'] == 'shader' and command.get('dispatch') is None:
            raise ValueError('page-effect lowering requires the numerical command\'s dispatch-to-page relation')
        count, accesses = _group_accesses(command, algebra)
        projected.append(dict(command=command, accesses=accesses,
            thresholds=[[0] * len(effects['observations']) for _ in range(count)],
            contributions=[set() for _ in range(count)]))
    for observation_index, observation in enumerate(effects['observations']):
        source = TensorView.of(effects['sources'][observation['source']])
        for cut_index, row in enumerate(effects['rows']):
            selected = TensorView(source.storage, source.layout.reshape((source.numel(),)).select_runs(
                0, [(row[0] * effects['unit'], row[1] * effects['unit'])]))
            root, extent, relation = _physical(selected, algebra)
            remaining = relation.image(relation.domain())
            for program in projected:
                for access in program['accesses']:
                    if access['root'] != root:
                        continue
                    if access['side'] == 'reads':
                        spans = access['relation'].preimage(remaining).spans()
                        if spans:
                            program['command'].setdefault('observed_slots', set()).add(access['slot'])
                        for first, width in spans:
                            for group in range(first, first + width):
                                program['thresholds'][group][observation_index] = max(
                                    program['thresholds'][group][observation_index], row[2 + observation_index])
                for access in program['accesses']:
                    if access['root'] == root and access['side'] == 'writes':
                        remaining = remaining.binary(access['positions'], 'subtract')
    target = TensorView.of(effects['destination'])
    expected = []
    for cut_index, row in enumerate(effects['rows']):
        selected = TensorView(target.storage, target.layout.reshape((target.numel(),)).select_runs(
            0, [(row[0] * effects['unit'], row[1] * effects['unit'])]))
        root, extent, relation = _physical(selected, algebra)
        remaining = relation.image(relation.domain())
        for program in reversed(projected):
            writes = [item for item in program['accesses'] if item['root'] == root and item['side'] == 'writes']
            for access in writes:
                for first, width in access['relation'].preimage(remaining).spans():
                    for group in range(first, first + width):
                        program['contributions'][group].add(cut_index)
            for access in writes:
                remaining = remaining.binary(access['positions'], 'subtract')
        if not remaining.empty():
            raise ValueError('published output pages require their numerical producer relation')
        expected.append(sum(cut_index in cells for program in projected for cells in program['contributions']))
    totals = _table([[value] for value in expected], torch.uint32)
    arrivals = empty((len(expected),), dtype=torch.uint32, device='cpu')
    arrivals.zero_()
    for program in projected:
        _inject(program['command'], program['thresholds'],
                [sorted(values) for values in program['contributions']], effects, totals, arrivals)
