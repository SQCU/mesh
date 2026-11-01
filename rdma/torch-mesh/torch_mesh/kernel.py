"""Numerical kernel declarations lowered to the common prepared command records."""
from functools import lru_cache
from pathlib import Path
import struct
import json
import math
import re

import torch


@torch.library.custom_op('torch_mesh::native_kernel', mutates_args=(),
    schema='(Tensor[] inputs, int[] input_slots, int[][] output_slots, int[][] output_strides, int[][] shapes, str[] dtypes, str[] devices, '
           'str operands, str source, str name, str instance, int[] words, int constant_slot, int[] grid, int[] threads) -> Tensor[]')
def operation(inputs: list[torch.Tensor], input_slots: list[int], output_slots: list[list[int]], output_strides: list[list[int]],
              shapes: list[list[int]], dtypes: list[str], devices: list[str], operands: str, source: str, name: str, instance: str,
              words: list[int], constant_slot: int, grid: list[int], threads: list[int]) -> list[torch.Tensor]:
    raise RuntimeError('native kernels require whole-function preparation')


@operation.register_fake
def _shape(inputs, input_slots, output_slots, output_strides, shapes, dtypes, devices, operands, source, name, instance, words, constant_slot, grid, threads):
    from .operand import fake_outputs
    return fake_outputs(shapes, output_strides, dtypes, devices)


def _word(value, k):
    """A word a kernel reads as uint (or, two's complement, int): one past 32 bits is refused, as _stream's word_of
    refuses it, rather than narrowed into another offset, stride or bound."""
    value = int(value)
    if not -2**31 <= value < 2**32:
        raise ValueError(f"a kernel's word {k} is {value}, which no 32-bit word holds")
    return value & 0xffffffff


def declare(source, name, buffers, outputs, words, constant_slot, grid, threads, instance='', *, indexed, requirements, dispatch, indices=None, prefix='', domains=()):
    """Carry each use's layout and requirement across FX's tensor schema.

    The serialized operand descriptions contain only compiler metadata. Values,
    including computed indices, stay in the tensor operands. Preparation composes
    these descriptions with producer placement and emits the common records.
    """
    from .operand import signature
    uses = {i: ('indexed' if i in indexed else requirements[i],) for i, value in enumerate(buffers) if value is not None}
    operands = signature(buffers, outputs, uses)
    declarations = json.loads(operands['operands'])
    if indices:
        declarations['indices'] = indices
    if prefix:
        declarations['prefix'] = prefix
    if domains:
        declarations['domains'] = domains
    declarations['dispatch'] = [dict(side=use['side'], slot=use['slot'], relation=str(use['group_relation']))
                                for use in dispatch]
    operands['operands'] = json.dumps(declarations, separators=(',', ':'))
    return operation(**operands, source=source, name=name, instance=instance,
                     words=list(words), constant_slot=constant_slot, grid=list(grid), threads=list(threads))


@lru_cache(None)
def _text(source):
    return Path(__file__).with_name(source).read_text()


def prepare(inputs, input_slots, output_slots, output_strides, shapes, dtypes, devices, operands, source, name, instance, words, constant_slot, grid, threads, *, outputs=None):
    from . import layout
    from .operand import bind
    from .placement import output
    from .native import empty_strided
    declarations = json.loads(operands)
    results = []
    for shape, steps, dtype, device, out in zip(shapes, output_strides, dtypes, devices,
            outputs if outputs is not None else [None] * len(shapes), strict=True):
        if out is None:
            out = empty_strided(tuple(shape), tuple(steps), getattr(torch, dtype), device)
        results.append(output(shape, getattr(torch, dtype), device, out))
    if any(v == 0 for v in grid):
        return results, []
    commands, addresses, after = [], {}, []
    def prepare_binding(index, value, writing):
        declaration = declarations[str(index)]
        view = bind(value, declaration)
        requirements = declaration['requirements']
        placed, copies = layout.resolve(view, requirements, writing=writing)
        (after if writing else commands).extend(copies)
        addresses[index] = placed
    for index, value in zip(input_slots, inputs, strict=True):
        prepare_binding(index, value, False)
    for slots, value in zip(output_slots, results, strict=True):
        for index in slots:
            prepare_binding(index, value, True)
    parameters = dict(source=source, name=name, instance=instance,
        words=words, constant_slot=constant_slot, grid=tuple(grid), threads=tuple(threads),
        prefix=declarations.get('prefix', ''), indices=declarations.get('indices', {}),
        dispatch=declarations['dispatch'])
    values = {str(slot): value for slot, value in addresses.items()}
    commands.append(dict(kind='kernel', operands=values, parameters=parameters, before=(), after=()))
    return results, commands + after


def dispatch_relations(algebra, records, count):
    """Restore the serialized common group-to-logical-scalar equations."""
    from mesh_relation import IndexRelation
    return tuple(dict(side=use['side'], slot=int(use['slot']), group_relation=IndexRelation(algebra,
        algebra.map_read_from_str(algebra.ctx, use['relation'].encode()), (count,))) for use in records)


def selected_source(source, function, original_grid, run):
    """Emit a selected group run with its original numerical coordinates."""
    from .shader import entries
    first, width, stride, repeats = run
    if tuple(original_grid[1:]) == (1, 1) and first == 0 and width == original_grid[0] and repeats == 1:
        return source
    def compose(parameters, body):
        aliases, needs = [], set()
        ordinal = f'({first}ul + (mesh_selected_group.x/{width}ul)*{stride}ul + mesh_selected_group.x%{width}ul)'
        coordinates = f'uint3({ordinal}%{original_grid[0]}ul, ({ordinal}/{original_grid[0]}ul)%{original_grid[1]}ul, {ordinal}/{original_grid[0]*original_grid[1]}ul)'
        extents = f'uint3({original_grid[0]},{original_grid[1]},{original_grid[2]})'
        def builtin(match):
            kind, name, attribute = match.groups()
            if attribute == 'threadgroup_position_in_grid':
                expression = coordinates
            elif attribute == 'thread_position_in_grid':
                needs.update(('lane', 'threads'))
                expression = f'({coordinates}) * mesh_selected_threads + mesh_selected_lane'
            elif attribute == 'threadgroups_per_grid':
                expression = extents
            elif attribute == 'threads_per_grid':
                needs.add('threads');expression = f'{extents} * mesh_selected_threads'
            elif attribute == 'threads_per_threadgroup':
                needs.add('threads');expression = 'mesh_selected_threads'
            else:
                needs.add('lane');expression = 'mesh_selected_lane'
            component = '' if kind.endswith('3') else '.xy' if kind.endswith('2') else '.x'
            aliases.append(f'const {kind} {name} = {kind}(({expression}){component});')
            return ''
        pattern = r'\b(uint[123]?|ushort[123]?)\s+(\w+)\s*\[\[(threadgroup_position_in_grid|thread_position_in_grid|threadgroups_per_grid|threads_per_grid|threads_per_threadgroup|thread_position_in_threadgroup)\]\]'
        parameters = re.sub(pattern, builtin, parameters)
        parameters = re.sub(r',\s*(?=,|$)', '', parameters).strip().strip(',')
        added = ['uint3 mesh_selected_group [[threadgroup_position_in_grid]]']
        if 'threads' in needs:
            added.append('uint3 mesh_selected_threads [[threads_per_threadgroup]]')
        if 'lane' in needs:
            added.append('uint3 mesh_selected_lane [[thread_position_in_threadgroup]]')
        return ','.join(filter(None, (parameters, *added))), ''.join(aliases) + body
    return entries(source, function, compose)


def declaration(operands, parameters, configuration):
    """Retain the numerical source and exact dispatch/page equations."""
    from . import _stream
    from .command_declarations import Commands, dense
    from mesh_relation import Algebra
    from .operand import iteration_order, relation_layout
    commands = Commands(operands, **configuration)
    count = max([parameters['constant_slot'], *(int(slot) for slot in operands), *parameters.get('vacant', {})]) + 1
    views = [None] * count
    mappings = parameters.get('layouts', {})
    for slot in operands:
        views[int(slot)] = commands.view(slot, mappings.get(slot),
            binding_origin=None if slot in parameters.get('fixed_origins', ()) else 0 if slot in mappings else None,
            source_domain=parameters.get('source_domains', {}).get(slot))
    dispatch = parameters['dispatch']
    if dispatch and 'relation' in dispatch[0]:
        dispatch = dispatch_relations(Algebra(), dispatch, math.prod(parameters['grid']))
    shift = configuration['geometry']['ACCESS_WRITE_SHIFT']
    access = sum(1 << (use['slot'] + (shift if use['side'] == 'writes' else 0))
                 for use in { (item['side'], item['slot']): item for item in dispatch }.values())
    access_layouts = {}
    for use in dispatch:
        slot = use['slot'];positions = use['group_relation'].image(use['group_relation'].domain())
        if slot in access_layouts:
            positions = positions.binary(access_layouts[slot], 'union')
        access_layouts[slot] = positions
    for slot, positions in tuple(access_layouts.items()):
        if str(slot) in parameters.get('source_domains', {}):
            del access_layouts[slot]
            continue
        order, _ = iteration_order(positions, (views[slot]['layout'].count,))
        reached = relation_layout(order, order.shape, views[slot]['layout'].count)
        access_layouts[slot] = reached.rebase(dense((views[slot]['layout'].count,)), views[slot]['layout'].reshape((views[slot]['layout'].count,)))
    words = parameters['words']
    constants = ({parameters['constant_slot']: struct.pack(f'{len(words)}I', *(_word(v, k) for k, v in enumerate(words)))} if words else {})
    source = parameters['prefix'] + _text(parameters['source']) + '\n' + parameters['instance']
    source = ''.join(_stream.index_values(int(slot), values) for slot, values in parameters['indices'].items()) + source
    if 'run' in parameters:
        source = selected_source(source, parameters['name'], parameters['original_grid'], parameters['run'])
    commands.shader(source, parameters['name'], views, grid=parameters['grid'], threadgroup=parameters['threads'],
        constants=constants, indices=parameters.get('vacant'), access=access, access_layouts=access_layouts, dispatch=dispatch)
    return commands.declaration()


def domain(algebra, operands, named, declarations):
    """The retained native declaration's work and physical source relations."""
    from .operand import layout_configuration
    work = (math.prod(named['grid']),)
    reads, writes, uses = {}, {}, {}
    dispatch = dispatch_relations(algebra, declarations['dispatch'], work[0])
    for item in dispatch:
        role, side = str(item['slot']), item['side']
        value = dict(operands[item['slot']])
        # An opaque arithmetic body consumes its declared pages. A scalar
        # getter is substituted only by consumers that declare such a reader.
        value.pop('reader', None)
        logical = item['group_relation']
        storage = algebra.layout(layout_configuration(value['view']))
        access = dict(value, relation=logical.then(storage))
        targets = reads if side == 'reads' else writes
        if role in targets:
            access['relation'] = targets[role]['relation'].union(access['relation'])
        targets[role] = access
        use = uses.setdefault(role, dict(access=value, logical=logical, logical_sides={}, storage=storage,
            view=value['view'], requirements=declarations[role]['requirements'], sides=set()))
        use['logical'] = use['logical'].union(logical)
        use['logical_sides'][side] = logical if side not in use['logical_sides'] else use['logical_sides'][side].union(logical)
        use['sides'].add(side)
        use['side'] = 'readwrite' if len(use['sides']) == 2 else 'read' if side == 'reads' else 'write'
    parameters = {key: named[key] for key in ('source', 'name', 'instance', 'words', 'constant_slot', 'grid', 'threads')}
    parameters.update(prefix=declarations.get('prefix', ''), indices=declarations.get('indices', {}))
    return dict(kind='dispatch', work=work, reads=reads, writes=writes, reduction=None, finish=None,
                uses=uses, parameters=parameters)


def selected(request):
    """Project the declared group fibers and their API constraints together."""
    from .operand import iteration_order, relation_layout, layout_configuration, required_layout, storage_requirement
    domain, work = request['domain'], request['work']
    algebra = work.algebra
    result = []
    for first, width, stride, repeats in work.spans():
        count = width * repeats
        order = algebra.map((count,), domain['work'], lambda i, o:
            [f'{o[0]} = {first} + floor({i[0]}/{width})*{stride} + {i[0]} mod {width}'])
        operands, layouts, dispatch, fixed_origins, vacant, source_domains = {}, {}, [], [], {}, {}
        for role, use in domain['uses'].items():
            relation = order.then(use['logical'])
            positions = relation.image(relation.domain())
            positions_order, _ = iteration_order(positions, (use['view'].count,))
            selected_count = positions_order.shape[0]
            if not selected_count:
                vacant[int(role)] = use['view']
                continue
            inverse = positions_order.inverse((use['view'].count,))
            layouts[role] = relation_layout(inverse, use['view'].shape, selected_count)
            source_domains[role] = positions
            contracts = []
            for requirement in use['requirements']:
                kind, capacity, alignment = storage_requirement(requirement)
                if kind == 'indexed':
                    continue
                target = required_layout(use['view'], kind)
                chosen = positions_order.then(algebra.layout(layout_configuration(target)))
                start, length = target.storage_span()
                contract = relation_layout(chosen, (selected_count,), start + length)
                contracts.append(dict(layout=dict(view=contract.state), capacity=capacity, alignment=alignment,
                                      origin=(target.index(0) - (contract.index(0) if contract.count else 0)) * use['access']['dtype'].itemsize))
                fixed_origins.append(role)
            operands[role] = dict(access=use['access'], relation=positions_order.then(use['storage']),
                shape=(selected_count,), requirements=tuple(contracts), access_mode=use['side'])
            for side in use['sides']:
                dispatch.append(dict(side=side, slot=int(role), group_relation=order.then(use['logical_sides'][side])))
        parameters = dict(domain['parameters'], layouts=layouts, fixed_origins=tuple(fixed_origins), vacant=vacant, source_domains=source_domains, dispatch=tuple(dispatch),
            original_grid=domain['parameters']['grid'], grid=(count, 1, 1), run=(first, width, stride, repeats))
        result.append(dict(adapter='kernel', operands=operands, parameters=parameters))
    return tuple(result)
