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

    def bind(self, described, placed):
        if (tuple(described.shape) != tuple(placed.shape) or described.dtype != placed.dtype
                or described.device.type != placed.device.type):
            raise ValueError('a destination must match its declared shape, dtype and device')
        self.fixed.append((described, placed))

    def equate(self, left, right, left_view=None, right_view=None):
        """Two declared uses name the same values in the same logical order."""
        left_view = descriptor(left) if left_view is None else left_view
        right_view = descriptor(right) if right_view is None else right_view
        if tuple(left_view.shape) != tuple(right_view.shape) or left.dtype != right.dtype:
            raise ValueError('shared page uses require the same ordered value type')
        self.equalities.append((left, right, left_view, right_view))

    def solve(self, described, requirements):
        """Unite full domains, solve layout constraints, then bind actual pages."""
        import math
        from . import _stream
        from .layout import allocate, solve_layout
        self.bindings, self.owners = [], []
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
        for logical, _ in self.fixed:
            root(logical)
        for value, *_ in requirements:
            root(value)
        for family in families:
            members, domain = family['members'], family['domain']
            related = lambda value: any(torch._C._is_alias_of(member['logical'], value) for member in members)
            values = [value for value in described if isinstance(value, torch.Tensor) and related(value)]
            values += [member['logical'] for member in members]
            fixed = [(logical, placed) for logical, placed in self.fixed if related(logical)]
            demands = [(value, view, kind, capacity, alignment) for value, view, kind, capacity, alignment in requirements if related(value)]
            alignment = math.lcm(*(value.element_size() for value in values), *(entry[4] for entry in demands))
            if not fixed:
                mapping, extent = solve_layout(domain.count, [(use(value, view), value.element_size(), kind, view, capacity, aligned)
                    for value, view, kind, capacity, aligned in demands],
                    [(use(value), value.element_size()) for value in values])
                placed = allocate(mapping, torch.uint8, members[0]['logical'].device.type, extent=extent, alignment=alignment)
                self.owners.append(placed.storage)
            else:
                uses, targets, actuals = [], [], []
                for logical, actual in fixed:
                    uses.append(use(logical))
                    targets.append(_stream.physical_layout(actual))
                    actuals.append(TensorView.of(actual))
                covered, physical, missing = _stream.TensorLayout.assign(domain.count, uses, targets)
                if missing.count:
                    extra = allocate(_stream.TensorLayout((missing.count,), (1,)), torch.uint8,
                                     actuals[0].device.type, alignment=alignment)
                    uses.append(missing);actuals.append(extra)
                storage, mappings = _stream.page_storage([value.storage for value in actuals])
                targets = [descriptor(value).components(value.element_size()).rebase(
                    _stream.TensorLayout(tuple(mapping.shape), (1,)), mapping)
                    for value, mapping in zip(actuals, mappings, strict=True)]
                covered, physical, missing = _stream.TensorLayout.assign(domain.count, uses, targets)
                placed = TensorView(storage, domain.rebase(covered, physical))
                self.owners.append(placed.storage)
            for member in members:
                mapping = member['mapping'].rebase(domain, placed.layout)
                self.bindings.append((member['logical'], TensorView(placed.storage, mapping)))
