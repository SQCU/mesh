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
    """A value's bytes as page bytes, and the inverse where its layout has a written one (None otherwise). type: a
    TensorValue -> its bytes in its own space (design/page-maps-2026-11-02.md); a TensorView -> region bytes."""
    from mesh_relation import IndexRelation
    from .operand import TensorValue
    if isinstance(value, TensorValue):
        width = value.element_size()
        bytes_ = value.layout if width == 1 else value.layout.components(width)
        count, pointer, inverse = _stream.value_region(bytes_, algebra.ctx)
    else:
        count, pointer, inverse = _stream.page_region(value, algebra.ctx)
    return IndexRelation(algebra, pointer, (count,)), (IndexRelation(algebra, inverse, (1,)) if inverse else None)


def _root(operand):
    """an access's root for the native solve: a value's space serial, None (the region's) for a view"""
    from .operand import TensorValue
    return operand.space.serial if isinstance(operand, TensorValue) else None


def _spans(operand, access):
    """a value access's exact byte runs in its space ((lo, hi), ...); () for a view's, which the region root keeps"""
    from .operand import TensorValue
    positions = access.get('positions')
    return tuple(positions.spans()) if isinstance(operand, TensorValue) and positions is not None else ()


# the routed program's step ops (mesh-plan.h): SEND publishes a piece of a buffer, REDUCE and COPY receive one
_SEND, _REDUCE, _COPY = 0, 1, 2


def _value_bytes(value, first, count):
    """a TensorValue's space root and the bytes [first, first + count) of the value, as bytes of its space: a collective
    operand is contiguous-typed (design/page-maps-2026-11-02.md, "The run's solve", 15). None where the value is not a
    placed one (a view of storage, or a lease's constant map, which the solve does not place)."""
    from .operand import TensorValue
    if not isinstance(value, TensorValue) or value.space.lease or count <= 0 or first < 0:
        return None
    origin = value.layout.index(0) * value.dtype.itemsize if value.layout.count else 0
    return value.space.serial, origin + first, count


def _transfer_accesses(t, transfer, post_key):
    """type: a transfer occurrence (order, invocation, records, buffers, views) -> its accesses (semantic key, begin key,
    end key, mode, root, invocation, first byte, bytes) and links (from key, to key) on its operands' values, with the
    routed buffer 0 read as versions (design/page-maps-2026-11-02.md, step 4): a SEND of buffer-0 bytes no earlier
    step of the call wrote reads the send operand, else the result; a COPY writes the result over [post, landed]; a
    REDUCE combines into the result once landed. The collective's own buffers are its own until it declares them."""
    P, invocation = transfer['order'], transfer['invocation']
    accesses, links = [], []
    written = {}
    for r in transfer['records']:
        if r['buffer'] == 0 and r['op'] != _SEND:
            written.setdefault(r['call'], []).append((r['round'], r['offset'], r['offset'] + r['bytes']))
    for k, r in enumerate(transfer['records']):
        if r['buffer'] != 0 or not r['bytes']:
            continue
        source, destination = transfer['views'][r['call']]
        origins = transfer['buffers'][r['call']]
        lo, hi = r['offset'], r['offset'] + r['bytes']
        if r['op'] == _SEND:
            begin, end = (P, 1, t, k), (P, 3, t, k)
            links.append((begin, end))
            # the parts of [lo, hi) an earlier round of this call wrote are the result's; the rest the send operand's
            cuts = sorted({lo, hi} | {x for round_, a, b in written.get(r['call'], ()) if round_ < r['round']
                                      for x in (a, b) if lo < x < hi})
            for a, b in zip(cuts, cuts[1:]):
                done = any(round_ < r['round'] and w0 <= a and b <= w1 for round_, w0, w1 in written.get(r['call'], ()))
                value, origin = ((destination, origins['result_origin']) if done or source is None
                                 else (source, origins['send_origin']))
                placed = _value_bytes(value, a - origin, b - a)
                if placed is not None:
                    accesses.append(((P, 1, t, k), begin, end, 1, placed[0], invocation, placed[1], placed[2]))
        else:
            post, landed = post_key(invocation, t, k), (P, 2, t, k)
            links.append((post, landed))
            placed = _value_bytes(destination, lo - origins['result_origin'], hi - lo)
            if placed is not None:
                if r['op'] == _COPY:
                    accesses.append(((P, 2, t, k), post, landed, 2, placed[0], invocation, placed[1], placed[2]))
                else:
                    accesses.append(((P, 2, t, k), landed, landed, 3, placed[0], invocation, placed[1], placed[2]))
    return accesses, links


def _declared_cuts(transfers):
    """The collective's declared cuts as value layouts (design/page-maps-2026-11-02.md, "Cuts before any lowering"): each
    landing's bytes in its destination value, a flat view of the value's space, read from the declaration without
    lowering. A cut off its element width is a declaration defect, raised."""
    cuts, seen = [], set()
    for transfer in transfers:
        for r in transfer['records']:
            if r['op'] == _SEND or r['buffer'] != 0 or not r['bytes']:
                continue
            destination = transfer['views'][r['call']][1]
            placed = _value_bytes(destination, r['offset'] - transfer['buffers'][r['call']]['result_origin'], r['bytes'])
            if placed is None or placed in seen:
                continue
            seen.add(placed)
            width = destination.dtype.itemsize
            if placed[1] % width or placed[2] % width:
                raise ValueError('a landing cut off its element width (design/page-maps-2026-11-02.md)')
            cuts.append(destination.flat(placed[1] // width, placed[2] // width))
    return cuts


def assign_pages(descriptions, invocation_of, spaces, transfers=()):
    """The run's page assignment by the run's solve (mesh-page-map.h mesh_run_solve; design/page-maps-2026-11-02.md,
    "The run's solve, each choice and its ground").

    type: (descriptions, each command's accesses with their space root and byte runs; invocation_of(order) -> the
    command's run invocation; spaces, serial -> ValueSpace; transfers, each collective occurrence's dict(order,
    invocation, records, buffers, views) from backend.group_transfers) -> dict(edges, bins, places, items, arena):
    - an event is a command, or a transfer's SEND, completion, post or landing; events take the run's order, a
      landing's post first in its invocation; an item is a (space, invocation), or for a state one version, (out, i)
      and (in, i + 1) being one item;
    - accesses are taken in their semantic order (a transfer's at its step), each over [begin, end];
    - a state's first version is pinned at the start of its lease's bin, live from before the run, and its last
      version likewise, live past it; each such lease is one bin, and the arena is the last;
    - an item is contiguous where a command reaches it through a contiguous operand type: a TensorView, one storage and
      an affine layout, is contiguous_input and contiguous_output, as is every transfer's operand; an access marked
      `paged` reaches its pages through the value's page table (design/page-maps-2026-11-02.md, "The run's solve", 15);
    - edges are the solve's (earlier event key, later event key, kind); places, each item key's runs (first position,
      pages); bins, each (first, pages, owner), owner ('state', state) or 'arena'; items, each (space serial,
      invocation)'s key; arena, the arena's pages in use.
    A lease-backed space is its caller's own storage at every invocation and makes no item."""
    order = sorted(range(len(descriptions)), key=lambda i: (descriptions[i]['order'], i))
    raw, links, starts = [], [], {}
    for c in order:
        description = descriptions[c]
        invocation = invocation_of(description['order'])
        starts[invocation] = min(starts.get(invocation, description['order']), description['order'])
        key = (description['order'], 0, c)
        for access in description['accesses']:
            root = access.get('root')
            if root is None or not access.get('spans') or spaces[root].lease:
                continue
            for lo, hi in access['spans']:
                if hi > lo:
                    raw.append((key, key, key, access['access'], root, invocation, lo, hi - lo, access.get('paged')))
    for transfer in transfers:
        starts[transfer['invocation']] = min(starts.get(transfer['invocation'], transfer['order']), transfer['order'])
    for t, transfer in enumerate(transfers):
        made, joined = _transfer_accesses(t, transfer, lambda invocation, t, k: (starts[invocation], -1, t, k))
        raw.extend(access + (False,) for access in made)
        links.extend(joined)
    raw.sort(key=lambda access: access[0])
    keys = sorted({k for access in raw for k in access[1:3]} | {k for link in links for k in link})
    event = {key: at for at, key in enumerate(keys)}
    items, index, pages, contiguous, accesses, versions, leases = {}, {}, [], [], [], {}, {}
    for _, begin, end, mode, root, invocation, lo, count, paged in raw:
        space = spaces[root]
        if space.state is not None:
            state, side = space.state
            key = ('state', state, invocation if side == 'in' else invocation + 1)
            versions.setdefault(state, set()).add(key[2])
            if space.pins:
                leases[state] = space.pins
        else:
            key = ('space', root, invocation)
        items[root, invocation] = key
        if key not in index:
            index[key] = len(pages)
            pages.append(space.pages)
            contiguous.append(0)
        item = index[key]
        pages[item] = max(pages[item], space.pages)
        contiguous[item] |= 0 if paged else 1
        accesses.append((event[begin], event[end], mode, item, lo, count))
    page_bytes = next(iter(spaces.values())).page_bytes if spaces else 1
    bins, first = [], 0
    pin, lives = [_stream.RUN_FREE] * len(pages), [0] * len(pages)
    for state, seen in versions.items():
        if state not in leases:
            continue
        held = sum(count for _, count, _ in leases[state])
        bins.append((first, held, ('state', state)))
        for version, life in ((min(seen), _stream.RUN_BEFORE), (max(seen), _stream.RUN_AFTER)):
            item = index['state', state, version]
            pin[item], lives[item] = first, lives[item] | life
        first += held
    # the arena's bound is every item's pages, so the bound itself never refuses; the region refuses at its lease
    bins.append((first, max(1, sum(pages)), 'arena'))
    solved = _stream.run_solve(page_bytes, len(keys), accesses, [(event[a], event[b]) for a, b in links],
                               list(zip(pages, pin, lives, contiguous)), [(at, held) for at, held, _ in bins])
    return dict(edges=[(keys[f], keys[t], kind) for f, t, kind, *_ in solved['edges']], bins=bins,
                places={key: tuple(solved['runs'][item]) for key, item in index.items()}, items=items,
                arena=solved['arena'])


def _through(relation, physical):
    """relation composed with a value's pages: through their written inverse where there is one, a preimage with no
    intermediate bytes to eliminate. Added by the LSTM demonstration session: then() left the operand's flattened bytes
    existential and isl eliminated them by a parametric integer program, 4.7 s to over 35 s for one shard's transposed
    weight read (2026-11-01); through the inverse the same map takes under a millisecond."""
    function, inverse = physical
    return relation.through(inverse) if inverse is not None else relation.then(function)


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
        physical = _physical(binding['operand'], algebra)[0]
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
        relation = _physical(cut, algebra)[0]
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
            relation = _physical(value, algebra)[0]
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
        relation = _through(access['relation'], physical[access['slot']])
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
                      relation=_through(access['relation'].reshape((site['cells'],)), physical[access['slot']])) for access in logical)


def describe(programs):
    """Adapt retained frontend declarations to the common page-effect compiler."""
    from .compilation import relations, bind
    from .native import declared_commands
    algebra = relations()
    descriptions, described = [], {}
    backends = dict(shader=_stream.PageBackend.Metal, cpu=_stream.PageBackend.CPU, ane=_stream.PageBackend.ANE)
    for program in programs:
        # an occurrence of an emission already described: its commands' accesses are the emission's, composed with
        # their pages once (Plan.append's `emission`, the declaration's key). Changed by the LSTM demonstration
        # session: every occurrence composed them again, the same isl composition five times at 4.5 s each in the
        # 64-step plan (compose record, 2026-11-01). What must stay: an emission's accesses composed once.
        key = program.pop('emission', None)
        if key is not None and key in described:
            program['commands'], made = described[key]
            descriptions.extend(dict(description, order=program['order']) for description in made)
            continue
        made = []
        with bind(program['configuration'], algebra=algebra):
            program['commands'] = tuple(declared_commands(program['commands']))
            for command in program['commands']:
                groups, accesses = _group_accesses(command, algebra)
                gpu = command['kind'] == 'shader'
                names = [''] * len(command['operands'])
                if command.get('names'):
                    for argument in command['arguments']:
                        names[argument['slot']] = argument['name']
                made.append(dict(order=program['order'], groups=groups,
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
                        access=1 if access['side'] == 'reads' else 2, relation=access['relation'],
                        root=_root(command['operands'][access['slot']]),
                        spans=_spans(command['operands'][access['slot']], access))
                        for access in accesses)))
        descriptions.extend(made)
        if key is not None:
            described[key] = program['commands'], made
    return descriptions


_SOURCES = None


def _source_record(command):
    """Each distinct generated kernel source, once, as a file in MESH_PREPARE_RECORD's directory (sources-<pid>/), named
    by its function: what the page-effects compiler put around each numerical kernel, for people to read. Nothing
    reads it back."""
    import os
    global _SOURCES
    if _SOURCES is None:
        directory = os.environ.get('MESH_PREPARE_RECORD')
        _SOURCES = {'dir': os.path.join(directory, f'sources-{os.getpid()}'), 'seen': set()} if directory else False
        if _SOURCES:
            os.makedirs(_SOURCES['dir'], exist_ok=True)
    text = command.get('source')
    if not _SOURCES or not isinstance(text, str) or text in _SOURCES['seen']:
        return
    _SOURCES['seen'].add(text)
    name = f"{len(_SOURCES['seen']):03d}-{command.get('kind', 'command')}-{command.get('function') or 'source'}.metal"
    with open(os.path.join(_SOURCES['dir'], name.replace('/', '_')), 'w') as out:
        out.write(text)


def bind_effects(programs, occurrences, schedule, declaration):
    commands = [command for program in programs for command in program['commands']]
    solved = declaration.bind(
        [dict(order=order, owner=owner, frame_orders=frames) for order, owner, frames in occurrences])
    for source in solved['sources']:
        command = commands[source['index']]
        command['source'] = source['source']
        _source_record(command)
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
        # each run's page assignment, in its append order (assign_pages), for the lowering
        self.assignments = []

    def append(self, schedule):
        from .native import Program
        from .compilation import bind
        prepared = []
        # each collective occurrence's transfers, read from its declaration before any occurrence lowers (the C
        # cursor counts every lowering of a communicator, a WRITE's too): the run's solve's transport accesses
        from .backend import group_transfers
        transfers, ahead = [], {}
        for index, step in enumerate(self.declarations):
            if 'transfer' not in step:
                continue
            handle = step['group'].handle
            behind = ahead.get(handle, 0)
            ahead[handle] = behind + 1
            if step['transfer'] != 'collective':
                continue
            _, records, buffers = group_transfers(step['group'], behind, len(step['calls']))
            transfers.append(dict(order=2 * index, invocation=step.get('run_invocation', 0), records=records,
                                  buffers=buffers, views=step['views']))
        # values carry no binding phase, so the plan has one variant, and its cuts are the collective's declared ones,
        # read without lowering (design/page-maps-2026-11-02.md, integration steps 1 and 2)
        phase = ()
        if phase not in self.variants:
            cuts = _declared_cuts(transfers)
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
                programs.append(dict(source_copy(emitted[key]), order=2 * index, configuration=step['configuration'],
                                     emission=key))
            descriptions = describe(programs)
            effects = _stream.PageEffectsPlan(self.algebra, descriptions)
            self.variants[phase] = tuple(programs), effects, descriptions
        retained, effects, descriptions = self.variants[phase]
        # the run's page assignment (assign_pages): each (space, invocation)'s arena place or pin, kept for the lowering
        from .operand import TensorValue
        spaces = {operand.space.serial: operand.space for program in retained for command in program['commands']
                  for operand in command.get('operands', ()) if isinstance(operand, TensorValue)}
        invocation_of = lambda order: self.declarations[order // 2].get('run_invocation', 0)
        for transfer in transfers:
            for pair in transfer['views']:
                spaces.update((value.space.serial, value.space) for value in pair if isinstance(value, TensorValue))
        self.assignments.append(assign_pages(descriptions, invocation_of, spaces, transfers))
        programs = source_copy(retained)
        # the final lowering: every value operand a storage view at its occurrence (lowering.lower_programs)
        from .lowering import lower_programs
        page_bytes = next(iter(spaces.values())).page_bytes if spaces else 1
        programs, arena = lower_programs(programs, self.assignments[-1], invocation_of, page_bytes, 'mps')
        self.arenas = getattr(self, 'arenas', []) + [arena]
        # each occurrence's spaces bound to its runs, then the occurrence lowered, in declaration order: the C
        # cursor's (lowering.bind_spaces; design/page-maps-2026-11-02.md, "The collective lowering in two phases")
        from .lowering import bind_spaces
        for index, step in enumerate(self.declarations):
            if 'transfer' not in step:
                continue
            if step['transfer'] == 'write':
                bound = _stream.view_write_operands(step['bindings'], step['group'].handle, 0)
            else:
                values = [value for pair in step['views'] for value in pair if value is not None]
                held = bind_spaces(values, step.get('run_invocation', 0), self.assignments[-1], arena)
                operands = [operand for operand in step['operands'] if not isinstance(operand, TensorValue)] + held
                bound = _stream.Occurrence(step['group'].handle, operands)
            bound.lower(step.get('invocation', 0))
            frames = tuple(self.origins[self.positions[index] + offset] + 1 for offset in step.get('frame_orders', ()))
            prepared.append((2 * index, bound, frames))
        programs, prepared = bind_effects(programs, prepared, schedule, effects)
        for program in programs:
            with bind(program['configuration'], algebra=self.algebra):
                prepared.append((program['order'], Program.prepare(program), None))
        for _, command, selected in sorted(prepared, key=lambda item: item[0]):
            if selected is None:
                command.append(schedule)
            else:
                command.append(schedule, selected)
