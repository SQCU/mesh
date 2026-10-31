"""Preparation-only tensor placement relations, shared by graph binding/lowering."""
import torch
from .operand import TensorView, descriptor, rebase_typed


def output(shape, dtype, device, out=None):
    """Resolve an explicit output operand once, before command construction."""
    if out is None:
        from . import empty
        return empty(tuple(shape), dtype=dtype, device=device)
    if tuple(out.shape) != tuple(shape) or out.dtype != dtype or out.device.type != torch.device(device).type:
        raise ValueError('output operand must match the declared shape, dtype and device')
    return out


def same_view(a, b):
    from . import _stream
    return _stream.same_operand(a, b)


def relation(described, placed, view):
    """Compose typed aliases with assigned pages using the shared algebra."""
    width, target_width = described.element_size(), view.element_size()
    return rebase_typed(descriptor(view), target_width, descriptor(described), width, placed)


def alias(described, placed, view):
    placed = TensorView.of(placed)
    layout = relation(described, placed.layout, view)
    return TensorView(placed.storage_as(view.dtype), layout)


class Placements:
    """Complete logical storage families assigned to final bridge pages."""
    def __init__(self):
        self.bindings, self.fixed, self.equalities, self.owners = [], [], [], []

    def get(self, described):
        if not isinstance(described, torch.Tensor):
            return None
        for logical, physical in self.bindings:
            if torch._C._is_alias_of(logical, described):
                return alias(logical, physical, described)
        return None

    def bind(self, described, placed, *, view=None):
        view = descriptor(described) if view is None else view
        if (tuple(view.shape) != tuple(placed.shape) or described.dtype != placed.dtype
                or described.device.type != placed.device.type):
            raise ValueError('a destination must match its declared shape, dtype and device')
        self.fixed.append((described, view, placed))

    def equate(self, left, right, left_view=None, right_view=None):
        """Two declared uses name the same values in the same logical order."""
        left_view = descriptor(left) if left_view is None else left_view
        right_view = descriptor(right) if right_view is None else right_view
        if tuple(left_view.shape) != tuple(right_view.shape) or left.dtype != right.dtype:
            raise ValueError('shared page uses require the same ordered value type')
        self.equalities.append((left, right, left_view, right_view))

    def declare(self, described, requirements, materializations=(), *, fields=(), uses=()):
        """Unite logical storage families before physical pages are assigned.

        ``fields`` supplies (root, byte view, scalar width). ``uses`` supplies
        (root, byte view, scalar width, layout kind, typed target view,
        byte capacity, alignment). These extend the same placement equations
        to compact byte families without inventing whole-storage scalar views.
        """
        from . import _stream
        roots, families = [], []

        def root(value):
            found = next((r for r in roots if torch._C._is_alias_of(r['logical'], value)), None)
            if found is None:
                count = value.untyped_storage().nbytes()
                whole = value.as_strided((count // value.element_size(),), (1,), 0).view(torch.uint8)
                domain = _stream.TensorLayout((count,), (1,))
                family = {'domain': domain, 'members': []}
                found = {'logical': whole, 'mapping': domain, 'family': family}
                family['members'].append(found); roots.append(found); families.append(family)
            return found

        def use(value, view=None):
            entry = root(value)
            layout = descriptor(value) if view is None else view
            return layout.components(value.element_size()).rebase(
                descriptor(entry['logical']), entry['mapping'])

        for left, right, left_view, right_view in self.equalities:
            a, b = root(left), root(right)
            source, target = use(left, left_view), use(right, right_view)
            if a['family'] is b['family']:
                if not source.equivalent(target):
                    family = a['family']
                    domain, mapping = _stream.TensorLayout.identify(family['domain'].count, source, target)
                    for member in family['members']:
                        member['mapping'] = member['mapping'].rebase(family['domain'], mapping)
                    family['domain'] = domain
                continue
            first, second = a['family'], b['family']
            domain, left_map, right_map = _stream.TensorLayout.unite(first['domain'], second['domain'], source, target)
            for family, mapping in ((first, left_map), (second, right_map)):
                for member in family['members']:
                    member['mapping'] = member['mapping'].rebase(family['domain'], mapping)
                    member['family'] = first
            first['members'].extend(second['members']);first['domain'] = domain
            families[:] = [family for family in families if family is not second]
        for logical in (*materializations, *(logical for logical, _, _ in self.fixed)):
            if isinstance(logical, torch.Tensor):
                root(logical)
        for value, *_ in requirements:
            root(value)
        for value, *_ in (*fields, *uses):
            root(value)
        self.declaration = (tuple(described), tuple(requirements), roots, families, use, tuple(fields), tuple(uses))

    def materializes(self, value):
        """Whether this value belongs to a declared materialization family."""
        return isinstance(value, torch.Tensor) and any(
            torch._C._is_alias_of(root['logical'], value) for root in self.declaration[2])

    def solve(self, described=None, requirements=None):
        """Solve the retained logical families, then bind their actual pages."""
        import math
        from . import _stream
        from .layout import allocate, solve_layout
        if described is not None:
            self.declare(described, requirements)
        described, requirements, roots, families, use, fields, uses = self.declaration
        self.bindings, self.owners = [], []
        for family in families:
            members, domain = family['members'], family['domain']
            related = lambda value: any(torch._C._is_alias_of(member['logical'], value) for member in members)
            values = [value for value in described if isinstance(value, torch.Tensor) and related(value)]
            values += [member['logical'] for member in members]
            fixed = [(logical, view, placed) for logical, view, placed in self.fixed if related(logical)]
            demands = [(use(value, view), value.element_size(), kind, view, capacity, alignment)
                       for value, view, kind, capacity, alignment in requirements if related(value)]
            # These uses already describe byte positions. Preserve their
            # scalar-component axis instead of appending the root's uint8 axis.
            demands.extend((use(value, view).reshape(view.shape), width, kind, target, capacity, alignment)
                           for value, view, width, kind, target, capacity, alignment in uses if related(value))
            typed = [(use(value), value.element_size()) for value in values]
            typed.extend((use(value, view).reshape(view.shape), width) for value, view, width in fields if related(value))
            alignment = math.lcm(*(width for _, width in typed), *(entry[5] for entry in demands))
            if not fixed:
                mapping, extent = solve_layout(domain.count, demands, typed)
                if mapping is None:
                    mapping, extent = solve_layout(domain.count, [], typed)
                placed = allocate(mapping, torch.uint8, members[0]['logical'].device.type, extent=extent, alignment=alignment)
                self.owners.append(placed.storage)
            else:
                actuals = [TensorView.of(actual) for _, _, actual in fixed]
                owners = [value.storage for value in actuals]
                physical_pages, page = _stream.page_domain(owners)
                alignment = math.lcm(alignment, page)
                fixed_bytes = physical_pages.components(page).reshape((physical_pages.count * page,))
                fixed_domain = _stream.TensorLayout(tuple(fixed_bytes.shape), (1,))
                def assignments():
                    return _stream.TensorLayout.assign(domain.count,
                        [use(logical, view) for logical, view, _ in fixed],
                        [_stream.physical_layout(actual) for actual in actuals])
                covered, physical, _ = assignments()
                unique = physical.intersection(physical)
                representatives = physical.rebase(unique, unique.rebase(physical, covered))
                if not representatives.equivalent(covered):
                    family_domain = domain
                    changed, mapping = _stream.TensorLayout.identify(domain.count, covered, representatives)
                    for member in members:
                        member['mapping'] = member['mapping'].rebase(domain, mapping)
                    domain = family['domain'] = changed
                    covered, physical, _ = assignments()
                    typed = [(view.rebase(family_domain, mapping), width) for view, width in typed]
                    demands = [(view.rebase(family_domain, mapping), width, kind, target, capacity, aligned)
                               for view, width, kind, target, capacity, aligned in demands]
                # Include complete caller pages as ordinary typed fields. The
                # unknown bytes remain reserved; no new value can overwrite them.
                joint, original, pages = _stream.TensorLayout.unite(domain, fixed_domain,
                    covered, physical.rebase(fixed_bytes, fixed_domain))
                typed = [(view.rebase(domain, original), width) for view, width in typed]
                typed.append((pages, page))
                demanded = [(view.rebase(domain, original), width, kind, target, capacity, aligned)
                            for view, width, kind, target, capacity, aligned in demands]
                mapping, extent = solve_layout(joint.count, demanded, typed)
                if mapping is None:
                    # An incompatible consumer gets the existing explicit AOT
                    # conversion edge. Fixed pages still use the same solver.
                    mapping, extent = solve_layout(joint.count, [], typed)
                placed_pages = pages.rebase(joint, mapping).reshape((physical_pages.count, page)).elements(page)
                first, length = mapping.storage_span()
                count = -(-max(first + length, extent) // page)
                known, locations, missing = _stream.TensorLayout.assign(count, [placed_pages], [physical_pages])
                if missing.count:
                    from . import empty
                    extra = empty((missing.count * page,), dtype=torch.uint8, device=actuals[0].device.type)
                    owners.append(extra)
                    extra_pages = _stream.physical_layout(extra).reshape((missing.count, page)).elements(page)
                    known, locations, missing = _stream.TensorLayout.assign(count, [known, missing], [locations, extra_pages])
                order = _stream.TensorLayout((count,), (1,)).rebase(known, locations)
                storage = _stream.page_storage(owners, order, alignment)
                placed = TensorView(storage, original.rebase(joint, mapping))
                self.owners.append(storage)
            for member in members:
                mapping = member['mapping'].rebase(domain, placed.layout)
                self.bindings.append((member['logical'], TensorView(placed.storage, mapping)))


def _selected_layout(record, algebra):
    """Resolve one ordered typed use in its original storage coordinates."""
    from .operand import relation_layout
    binding = record['binding']
    relation = record.get('relation')
    if relation is None:
        relation = algebra.selection_relation(record['selection']).then(binding['relation'])
    return relation_layout(relation, record.get('shape', relation.shape),
                           int(binding['storage_bytes']) // binding['dtype'].itemsize)


def selected_view(storage, record, *, algebra, layout=None):
    """Compose an explicit typed use with its assigned selected page bytes."""
    binding = record['binding']
    assigned = storage[binding['name']]
    physical = TensorView.of(assigned['operand'])
    layout = _selected_layout(record, algebra) if layout is None else layout
    mapped = rebase_typed(layout, binding['dtype'].itemsize, assigned['source'],
                          assigned['source_dtype'].itemsize, physical.layout)
    return TensorView(physical.storage_as(binding['dtype']), mapped)


def field_storage(bindings, domains, *, algebra):
    """Relate each semantic value to all of its selected local storage uses.

    The field is a virtual coordinate domain, not another physical buffer.
    Existing placement equations give its uses the same pages. Only bytes
    selected by the retained work enter those equations.
    """
    import math
    from torch.utils import _pytree
    from .operand import byte_relation
    fields, equalities = {}, []
    for field, uses in bindings.items():
        value = _pytree.key_get(field[0].meta['val'], field[1])
        relation = algebra.map((value.numel(),), (value.numel(),), lambda x, y: [f'{x[0]} = {y[0]}'])
        canonical = fields[field] = dict(name=('field', field), dtype=value.dtype,
            storage_bytes=value.numel() * value.element_size(), relation=relation)
        width = value.element_size()

        def bytes_(relation, extent):
            count = math.prod(relation.shape) * width
            scalars = algebra.map((count,), relation.shape, lambda x, y: [f'{y[0]} = floor({x[0]}/{width})'])
            components = algebra.map((count,), (extent,), lambda x, y: [f'{x[0]} mod {width} = {y[0]} mod {width}'])
            return scalars.then(byte_relation(dict(dtype=value.dtype, storage_bytes=extent, relation=relation))).intersect(components)

        for binding in uses:
            positions = domains.get(binding['name'])
            if positions is None:
                continue
            for source, storage in binding['coordinates']:
                left, right = bytes_(source, canonical['storage_bytes']), bytes_(storage, binding['storage_bytes'])
                selected = right.preimage(positions)
                order = algebra.selection_relation(dict(shape=list(right.shape), steps=[['select', 0, selected.spans()]]))
                equalities.append((dict(binding=dict(canonical, dtype=torch.uint8), relation=order.then(left), shape=order.shape),
                    dict(binding=dict(binding, dtype=torch.uint8), relation=order.then(right), shape=order.shape)))
    return fields, tuple(equalities)


def selected(projection, *, algebra, device, results=(), fixed=(), requirements=(), equalities=()):
    """Place one projected numerical span through the ordinary family solver.

    Canonical storage demands and returned ``source`` layouts are bytes.
    Results and fixed operands carry ``binding`` plus an ordered ``selection``;
    results additionally declare their ``shape``. A consumer use instead
    supplies its exact scalar ``relation``, ``shape``, ``requirements``, access
    role and stable ``key``. Fixed records supply an existing page ``operand``.
    ``equalities`` pairs ordered typed uses that name the same values, including
    local collective identity edges. No tensor payload is inspected here.

    Returned storage feeds retained native IR directly. Per-use conversion
    records stay attached to that consumer's read/write relation, rather than
    introducing section-wide barriers. Unresolved access projection remains
    the caller's unfulfilled declaration; this helper does not invent reads.
    """
    from torch._subclasses.fake_tensor import FakeTensorMode
    from . import _stream
    from .operand import layout_configuration, storage_requirement
    from .layout import resolve, copies

    results, fixed, requirements = tuple(results), tuple(fixed), tuple(requirements)
    domains, widths = dict(projection['storage_bytes']), []

    def retain(binding, layout, width, *, typed=True):
        name = binding['name']
        relation = algebra.layout(layout_configuration(layout))
        positions = relation.image(relation.domain())
        domains[name] = positions if name not in domains else domains[name].binary(positions, 'union')
        if typed and layout.count:
            widths.append((name, layout, width))

    for record in projection['records']:
        for part in (record, record.get('finish')):
            if part is None:
                continue
            for side in ('reads', 'writes'):
                for use in part[side].values():
                    access = use['access']
                    if use['positions'] is None:
                        continue
                    # The projection already applied the assigned work and
                    # any declared live index-value axes. Its byte positions
                    # retain complete typed fields without inventing a static
                    # work-only relation for a changing index operand.
                    layout = _stream.TensorLayout.runs(use['positions'].spans())
                    retain(access, layout, access['dtype'].itemsize)
    requested = [(record, _selected_layout(record, algebra)) for record in results]
    consumers = [(record, _selected_layout(record, algebra)) for record in requirements]
    supplied = [(record, _selected_layout(record, algebra)) for record in fixed]
    equivalent = [tuple((record, _selected_layout(record, algebra)) for record in pair) for pair in equalities]
    for record, layout in (*requested, *consumers, *(item for pair in equivalent for item in pair)):
        retain(record['binding'], layout.components(record['binding']['dtype'].itemsize), record['binding']['dtype'].itemsize)

    original = {name: _stream.TensorLayout.runs(domain.spans()) for name, domain in domains.items()}
    mode = FakeTensorMode()
    with mode:
        roots = {name: torch.empty((layout.count,), dtype=torch.uint8, device=device) for name, layout in original.items()}
    compact = {name: descriptor(value) for name, value in roots.items()}
    placements, fields, demands = Placements(), [], []

    def use(name, layout):
        return layout.rebase(original[name], compact[name])

    for (left, a), (right, b) in equivalent:
        first, second = left['binding'], right['binding']
        placements.equate(roots[first['name']], roots[second['name']],
            use(first['name'], a.components(first['dtype'].itemsize)),
            use(second['name'], b.components(second['dtype'].itemsize)))
    for name, layout, width in widths:
        fields.append((roots[name], use(name, layout), width))
    for record, layout in supplied:
        binding, actual = record['binding'], TensorView.of(record['operand'])
        name, width = binding['name'], binding['dtype'].itemsize
        if actual.dtype != binding['dtype']:
            raise ValueError('fixed pages must carry their declared scalar type')
        if name not in domains:
            continue
        logical_bytes = layout.components(width).reshape((layout.count * width,))
        # Use the compact selected byte domain already lowered above. Feeding
        # the original nested work equations through another inverse repeats
        # elimination which the retained run layout has already resolved.
        relation = algebra.layout(layout_configuration(logical_bytes))
        compact_domain = algebra.layout(layout_configuration(original[name]))
        selected_bytes = relation.preimage(compact_domain.image(compact_domain.domain()))
        order = _stream.TensorLayout.runs(selected_bytes.spans())
        if not order.count:
            continue
        ordinals = _stream.TensorLayout((logical_bytes.count,), (1,))
        source = order.rebase(ordinals, logical_bytes)
        target = order.rebase(ordinals, actual.layout.components(width).reshape((logical_bytes.count,)))
        placements.bind(roots[name], TensorView(actual.storage_as(torch.uint8), target), view=use(name, source))
    for record, layout in consumers:
        binding, width = record['binding'], record['binding']['dtype'].itemsize
        source = use(binding['name'], layout.components(width))
        for requirement in record['requirements']:
            kind, capacity, alignment = storage_requirement(requirement)
            if kind == 'indexed':
                if not capacity and alignment == 1:
                    continue
                # Indexed APIs constrain the argument origin/footprint, not
                # the placement of every other scalar in this selected use.
                count = int(bool(layout.count))
                target = _stream.TensorLayout((count,), (1,))
                positions = source.reshape((layout.count, width)).slice(0, 0, count, 1)
                demands.append((roots[binding['name']], positions, width, 'contiguous', target, capacity, alignment))
            else:
                demands.append((roots[binding['name']], source, width, kind, layout, capacity, alignment))
    placements.declare(tuple(roots.values()), (), tuple(roots.values()), fields=fields, uses=demands)
    placements.solve()
    storage = {name: dict(source=original[name], source_dtype=torch.uint8, operand=placements.get(root))
               for name, root in roots.items()}

    resolved, owners = {}, list(placements.owners)
    for record, layout in consumers:
        value = selected_view(storage, record, algebra=algebra, layout=layout)
        writing = record['access'] == 'write'
        operand, commands = resolve(value, record['requirements'], writing=writing)
        before = () if writing else tuple(commands)
        after = tuple(commands) if writing else tuple(copies(operand, value)) if record['access'] == 'readwrite' else ()
        owners.append(operand.storage)
        resolved[record['key']] = dict(record, operand=operand, before=before, after=after)
    return dict(storage=storage, results=tuple(selected_view(storage, record, algebra=algebra, layout=layout)
                                              for record, layout in requested),
                pages=tuple(owners), uses=resolved)
