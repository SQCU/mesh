#!/usr/bin/env mesh-python
import argparse
import collections
import json
import math
import os
import random
import shutil
import sys
import tempfile
import time
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
import mapsrc as MS
import mapgen as G
import mkentfile as M
import negspace as NS
import placement as P
from rdma.workload import WorkloadMeter

BSP_COORDINATE_EXTENT = 65536.0

def transfer_site(source):
    component = [index for index in M.largest_component(source.navadj)
                 if tuple(round(value, 1) for value in source.navnodes[index]) in source.wpset]
    if not component:
        component = M.largest_component(source.navadj)
    choices = []
    for index in component:
        point = [float(value) for value in source.navnodes[index]]
        lo = np.asarray(point) + np.asarray(NS.PL_MIN)
        hi = np.asarray(point) + np.asarray(NS.PL_MAX)
        if source.ns.covered(NS.box_H(lo, hi), lo, hi):
            choices.append((len(source.navadj[index]), source.ns.clearance(point, cap=256.0),
                            -index, point, lo.tolist(), hi.tolist()))
    degree, clearance, negative_index, point, lo, hi = max(choices)
    return {'p': point, 'node': -negative_index, 'degree': degree,
            'clearance': clearance, 'trigger_bounds': [lo, hi],
            'trigger_free_residual_mass': 0}

def survey_sources(names, pk3, survey_dir):
    surveyed = {}
    observations = []
    for name in names:
        try:
            source = P.load_src(name, survey_dir, pk3, quiet=True)
            sites = P.map_sites(source)
            detected = len(sites)
            relaxed = 0
            if not sites:
                source._sites = None
                sites = P.map_sites(source, maxsites=max(1, len(source.navnodes)), minsep=0.0)
                relaxed = len(sites)
            transfer = transfer_site(source)
            directions = collections.Counter(site_direction(site) for site in sites)
            surveyed[name] = (source, sites, transfer)
            observations.append({'name': source.name, 'site_mass': len(sites),
                                 'detected_site_mass': detected,
                                 'relaxed_site_mass': relaxed,
                                 'site_direction_mass': {
                                     '%d,%d' % direction: mass
                                     for direction, mass in sorted(directions.items())
                                 },
                                 'outbound_free_span_integral': sum(
                                     max(0.0, site['t_exit'] - site['t_out'])
                                     for site in sites
                                 ),
                                 'support_domain_atom_mass': sum(
                                     site['support_domain_atom_mass'] for site in sites),
                                 'support_residual_atom_mass': sum(
                                     site['support_residual_atom_mass'] for site in sites),
                                 'transfer_site_mass': 1,
                                 'transfer_node': transfer['node'],
                                 'transfer_clearance': transfer['clearance'],
                                 'transfer_trigger_free_residual_mass':
                                     transfer['trigger_free_residual_mass']})
            print('survey %-18s site_mass=%d' % (name, len(sites)))
        except Exception as exc:
            observations.append({'name': name, 'site_mass': 0,
                                 'error': '%s: %s' % (type(exc).__name__, exc)})
            print('survey %-18s site_mass=0 error=%s: %s' %
                  (name, type(exc).__name__, exc))
    return surveyed, observations

def realize_authored_source(name, source, pk3, directory):
    data = (open(name + '.map', 'rb').read() if '/' in name
            else P.pk3_read(pk3, 'maps/%s.map' % name))
    path = os.path.join(directory, source.name + '.map')
    with open(path, 'wb') as handle:
        handle.write(data)
    return path, {'name': source.name, 'byte_mass': len(data),
                  'artifact_mass': int(bool(data))}

def site_direction(site):
    return tuple(int(round(value)) for value in site['dir'][:2])

def cell_direction(left, right):
    delta = [right[axis] - left[axis] for axis in range(2)]
    return tuple(0 if value == 0 else int(math.copysign(1, value)) for value in delta)

def bridge_backbone_cells(count):
    cells = [(0, 0, 0)]
    directions = ((1, 0), (0, 1))
    for index in range(1, count):
        direction = directions[(index - 1) % len(directions)]
        cells.append((cells[-1][0] + direction[0] * 3,
                      cells[-1][1] + direction[1] * 3, 0))
    return cells

def bridge_leaf_slots(cells):
    occupied = {tuple(cell[:2]) for cell in cells}
    consumed = [set() for _ in cells]
    for index in range(len(cells) - 1):
        direction = cell_direction(cells[index], cells[index + 1])
        consumed[index].add(direction)
        consumed[index + 1].add((-direction[0], -direction[1]))
    slots = []
    for bridge, cell in enumerate(cells):
        for direction in P.SITE_DIRS:
            direction = tuple(int(value) for value in direction[:2])
            target = (cell[0] + direction[0], cell[1] + direction[1])
            if direction not in consumed[bridge] and target not in occupied:
                slots.append((bridge, direction, (target[0], target[1], 0)))
    return slots

def leaf_assignment(names, surveyed, slots):
    candidates = [
        [index for index, (_, direction, _) in enumerate(slots)
         if (-direction[0], -direction[1]) in {
             site_direction(site) for site in surveyed[name][1]
         }]
        for name in names
    ]
    assigned = {}

    def place(source, seen):
        for slot in candidates[source]:
            if slot in seen:
                continue
            seen.add(slot)
            if slot not in assigned or place(assigned[slot], seen):
                assigned[slot] = source
                return True
        return False

    order = sorted(range(len(names)), key=lambda index: (len(candidates[index]), index))
    if not all(place(source, set()) for source in order):
        return None
    result = [None] * len(names)
    for slot, source in assigned.items():
        result[source] = slot
    return result

def required_bridge_count(names, transfer_mass, surveyed):
    count = 1
    while True:
        cells = bridge_backbone_cells(count)
        if len(names) + count >= transfer_mass:
            slots = bridge_leaf_slots(cells)
            if leaf_assignment(names, surveyed, slots) is not None:
                return count
        count += 1

def transfer_cells(occupied, count):
    if count == 0:
        return []
    width = math.ceil(math.sqrt(count))
    start_x = max(cell[0] for cell in occupied) + 3
    start_y = min(cell[1] for cell in occupied)
    return [(start_x + index % width, start_y + index // width, 0)
            for index in range(count)]

def fabric_layout(sources, names, bridge_mass, stock_mass, transfer_mass, surveyed):
    bridge_cells = bridge_backbone_cells(bridge_mass)
    stock_names = names[bridge_mass:bridge_mass + stock_mass]
    slots = bridge_leaf_slots(bridge_cells)
    assignment = leaf_assignment(stock_names, surveyed, slots)
    stock_cells = [slots[index][2] for index in assignment]
    cells = bridge_cells + stock_cells
    cells.extend(transfer_cells(cells, transfer_mass))
    min_x = min(cell[0] for cell in cells)
    min_y = min(cell[1] for cell in cells)
    cells = [(cell[0] - min_x, cell[1] - min_y, 0) for cell in cells]
    columns = max(cell[0] for cell in cells) + 1
    rows = max(cell[1] for cell in cells) + 1
    offsets, boundaries, slack = P.pack_offsets(sources, cells, columns, rows, 1)
    topology = [(index, index + 1) for index in range(bridge_mass - 1)]
    topology.extend((slots[slot][0], bridge_mass + source)
                    for source, slot in enumerate(assignment))
    roles = [set() for _ in sources]
    degrees = [0] * len(sources)
    for left, right in topology:
        direction = cell_direction(cells[left], cells[right])
        roles[left].add(direction)
        roles[right].add((-direction[0], -direction[1]))
        degrees[left] += 1
        degrees[right] += 1
    return topology, degrees, cells, offsets, boundaries, slack, columns, rows, roles

def transfer_edges(corridor_mass, transfer_mass, cells):
    if not corridor_mass:
        return [(index, index + 1) for index in range(max(0, transfer_mass - 1))]
    candidates = np.arange(corridor_mass, dtype=np.int64)
    coordinates = np.asarray(cells, dtype=np.int64)
    loads = np.zeros(corridor_mass, dtype=np.int64)
    edges = []
    for transfer in range(corridor_mass, corridor_mass + transfer_mass):
        distance = np.abs(
            coordinates[candidates, :2] - coordinates[transfer, :2]
        ).sum(axis=1)
        partner = int(candidates[np.lexsort((candidates, distance, loads))[0]])
        loads[partner] += 1
        edges.append((partner, transfer))
    return edges

def select_edge_sites(edges, names, surveyed, offsets, cells):
    used = [[0] * len(surveyed[name][1]) for name in names]
    selected = []
    reuse_mass = 0

    def mouth(tile, site_index):
        site = surveyed[names[tile]][1][site_index]
        return [site['p'][axis] + offsets[tile][axis]
                + site['dir'][axis] * (site['t_out'] + 48.0) for axis in range(3)]

    for left, right in edges:
        corridor_direction = cell_direction(cells[left], cells[right])
        options = []
        for left_site, a in enumerate(surveyed[names[left]][1]):
            if site_direction(a) != corridor_direction:
                continue
            for right_site, b in enumerate(surveyed[names[right]][1]):
                if site_direction(b) != (-corridor_direction[0], -corridor_direction[1]):
                    continue
                pa, pb = mouth(left, left_site), mouth(right, right_site)
                displacement = np.asarray(pb) - np.asarray(pa)
                distance = float(np.linalg.norm(displacement))
                direction_vector = displacement / max(distance, 1.0)
                alignment = float(np.dot(a['dir'], direction_vector)
                                  - np.dot(b['dir'], direction_vector))
                reuse = used[left][left_site] + used[right][right_site]
                options.append((-reuse, alignment * 1024.0 - distance + a['score'] + b['score'],
                                left_site, right_site))
        _, _, left_site, right_site = max(options)
        reuse_mass += int(used[left][left_site] > 0) + int(used[right][right_site] > 0)
        used[left][left_site] += 1
        used[right][right_site] += 1
        selected.append((left_site, right_site))
    return selected, reuse_mass

def edge_site_points(edges, selected, names, surveyed, offsets):
    points = []
    for (left, right), (left_site, right_site) in zip(edges, selected):
        row = []
        for tile, site_index in ((left, left_site), (right, right_site)):
            site = surveyed[names[tile]][1][site_index]
            row.append([site['p'][axis] + offsets[tile][axis]
                        + site['dir'][axis] * (site['t_out'] + 48.0)
                        for axis in range(3)])
        points.append(row)
    return points

def align_portal_planes(edges, selected, names, surveyed, offsets, slack):
    before = [math.dist(*pair) for pair in edge_site_points(
        edges, selected, names, surveyed, offsets,
    )]
    sweep_mass = 0
    for axis in (0, 1):
        terms = [[] for _ in offsets]
        for (left, right), (left_site, right_site) in zip(edges, selected):
            a = surveyed[names[left]][1][left_site]
            b = surveyed[names[right]][1][right_site]
            if abs(a['dir'][axis]) > 0.5:
                continue
            local_a = a['p'][axis] + a['dir'][axis] * (a['t_out'] + 48.0)
            local_b = b['p'][axis] + b['dir'][axis] * (b['t_out'] + 48.0)
            terms[left].append((right, local_b - local_a))
            terms[right].append((left, local_a - local_b))
        seen = set()
        while True:
            state = tuple(offset[axis] for offset in offsets)
            if state in seen:
                break
            seen.add(state)
            moved = False
            for tile, neighbors in enumerate(terms):
                if not neighbors:
                    continue
                value = sum(offsets[other][axis] + delta
                            for other, delta in neighbors) / len(neighbors)
                value = P.portal_coordinate(value, *slack[tile][axis])
                moved = moved or value != offsets[tile][axis]
                offsets[tile][axis] = value
            sweep_mass += 1
            if not moved:
                break
    after = [math.dist(*pair) for pair in edge_site_points(
        edges, selected, names, surveyed, offsets,
    )]
    return {'coordinate_sweep_mass': sweep_mass,
            'corridor_length_before_integral': sum(before),
            'corridor_length_after_integral': sum(after),
            'corridor_length_before_maximum': max(before, default=None),
            'corridor_length_after_maximum': max(after, default=None)}

def fit_edge_sites(edges, names, surveyed, offsets, slack, cells):
    seen = set()
    best = None
    initial_selected, _ = select_edge_sites(edges, names, surveyed, offsets, cells)
    initial_lengths = [math.dist(*pair) for pair in edge_site_points(
        edges, initial_selected, names, surveyed, offsets,
    )]
    coordinate_sweep_mass = 0
    while True:
        selected, reuse_mass = select_edge_sites(edges, names, surveyed, offsets, cells)
        identity = tuple(selected)
        if identity in seen:
            break
        seen.add(identity)
        measure = align_portal_planes(edges, selected, names, surveyed, offsets, slack)
        coordinate_sweep_mass += measure['coordinate_sweep_mass']
        lengths = [math.dist(*pair) for pair in edge_site_points(
            edges, selected, names, surveyed, offsets,
        )]
        candidate = (sum(lengths), max(lengths, default=0.0), identity,
                     [list(offset) for offset in offsets], reuse_mass)
        if best is None or candidate[:3] < best[:3]:
            best = candidate
    for target, source in zip(offsets, best[3]):
        target[:] = source
    final_lengths = [math.dist(*pair) for pair in edge_site_points(
        edges, list(best[2]), names, surveyed, offsets,
    )]
    measure = {'site_selection_fixed_point_mass': len(seen),
               'coordinate_sweep_mass': coordinate_sweep_mass,
               'corridor_length_before_integral': sum(initial_lengths),
               'corridor_length_after_integral': sum(final_lengths),
               'corridor_length_before_maximum': max(initial_lengths, default=None),
               'corridor_length_after_maximum': max(final_lengths, default=None)}
    return list(best[2]), best[4], measure

def align_portal_heights(edges, selected, names, surveyed, offsets, slack):
    count = len(names)
    if count < 2:
        return {'atom_mass': 0, 'integral': 0.0, 'square_integral': 0.0,
                'mean': None, 'variance': None, 'maximum_absolute': None}
    adjacency = [[] for _ in range(count)]
    for (left, right), (left_site, right_site) in zip(edges, selected):
        delta = (surveyed[names[left]][1][left_site]['p'][2]
                 - surveyed[names[right]][1][right_site]['p'][2])
        adjacency[left].append((right, delta))
        adjacency[right].append((left, -delta))
    reached = {0}
    stack = [0]
    while stack:
        left = stack.pop()
        for right, delta in adjacency[left]:
            if right in reached:
                continue
            offsets[right][2] = P.portal_coordinate(
                offsets[left][2] + delta, *slack[right][2],
            )
            reached.add(right)
            stack.append(right)
    residuals = np.asarray([
        surveyed[names[right]][1][right_site]['p'][2] + offsets[right][2]
        - surveyed[names[left]][1][left_site]['p'][2] - offsets[left][2]
        for (left, right), (left_site, right_site) in zip(edges, selected)
    ], dtype=np.float64)
    centered = residuals - residuals.mean() if len(residuals) else residuals
    return {
        'atom_mass': len(residuals),
        'reachable_tile_mass': len(reached),
        'unreachable_tile_mass': count - len(reached),
        'integral': float(residuals.sum()),
        'square_integral': float(np.square(residuals).sum()),
        'mean': None if len(residuals) == 0 else float(residuals.mean()),
        'variance': None if len(residuals) == 0 else float(np.mean(np.square(centered))),
        'maximum_absolute': None if len(residuals) == 0 else float(np.abs(residuals).max()),
    }

def q4(value):
    return round(value / P.PORTAL_QUANTUM) * P.PORTAL_QUANTUM

def box_faces(lo, hi, texture):
    brush = MS.quad_prism(
        [lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]],
        [hi[0], hi[1], lo[2]], [lo[0], hi[1], lo[2]],
        [0.0, 0.0, hi[2] - lo[2]], texture, texture,
    )
    return brush.to_faces()

def transfer_connect(entities, edges, names, surveyed, offsets):
    joins = []
    links = []
    incidence = np.asarray(edges, dtype=np.int64).reshape((-1, 2))
    counts = np.bincount(incidence.ravel(), minlength=len(names))
    ordinals = np.zeros(len(names), dtype=np.int64)
    for index, (left, right) in enumerate(edges):
        points = [P.vadd(surveyed[names[tile]][2]['p'], offsets[tile])
                  for tile in (left, right)]
        bounds = []
        for tile in (left, right):
            local = surveyed[names[tile]][2]['trigger_bounds']
            lower = np.asarray(local[0], dtype=np.float64) + offsets[tile]
            upper = np.asarray(local[1], dtype=np.float64) + offsets[tile]
            count = int(counts[tile])
            ordinal = int(ordinals[tile])
            ordinals[tile] += 1
            axis = int(np.argmax(upper[:2] - lower[:2]))
            interval = np.linspace(lower[axis], upper[axis], count + 1)
            lower[axis] = interval[ordinal]
            upper[axis] = interval[ordinal + 1]
            bounds.append((lower.tolist(), upper.tolist()))
        targets = ['mesh_transfer_%d_%d' % (index, side) for side in range(2)]
        for side in range(2):
            trigger = MS.Ent()
            trigger.keys = [('classname', 'trigger_teleport'), ('target', targets[1 - side])]
            trigger.brushes = [box_faces(bounds[side][0], bounds[side][1], 'common/trigger')]
            destination = MS.Ent()
            destination.keys = [('classname', 'misc_teleporter_dest'),
                                ('targetname', targets[side]),
                                ('origin', '%.10g %.10g %.10g' % tuple(points[side]))]
            marker = MS.Ent()
            marker.keys = [('classname', 'light'),
                           ('origin', '%.10g %.10g %.10g' %
                            (points[side][0], points[side][1], points[side][2] + 64.0)),
                           ('light', '600'), ('_color', '0.25 0.65 1')]
            entities.extend((trigger, destination, marker))
        links.extend(((points[0], points[1]), (points[1], points[0])))
        joins.append({'a': left, 'b': right, 'kind': 'teleporter',
                      'sa': points[0], 'sb': points[1],
                      'source_a': points[0], 'source_b': points[1],
                      'chain': points, 'width': 0.0, 'height': 0.0,
                      'carve_depth': 0.0, 'embed_depth': 0.0,
                      'carve_clearance': 0.0,
                      'longitudinal_seal_overlap': 0.0,
                      'transverse_seal_overlap': 0.0, 'seal_overlap': 0.0,
                      'horizontal_span': math.dist(points[0][:2], points[1][:2]),
                      'rise': points[1][2] - points[0][2], 'grade': 0.0,
                      'transfer_trigger_volume': sum(
                          math.prod(bounds[side][1][axis] - bounds[side][0][axis]
                                    for axis in range(3)) for side in range(2))})
    return joins, links

def portal(tile, name, site, offset):
    d = site['dir']
    axis = 0 if abs(d[0]) > 0.5 else 1
    other = 1 - axis
    sign = 1.0 if d[axis] > 0 else -1.0
    point = [site['p'][i] + offset[i] for i in range(3)]
    a0 = point[axis] + sign * (site['t_in'] - 16.0)
    a1 = point[axis] + sign * (site['t_out'] + 16.0)
    lo = [0.0, 0.0, 0.0]
    hi = [0.0, 0.0, 0.0]
    lo[axis], hi[axis] = q4(min(a0, a1)), q4(max(a0, a1))
    lo[other], hi[other] = q4(point[other] - P.DOOR_W / 2.0), q4(point[other] + P.DOOR_W / 2.0)
    lo[2], hi[2] = q4(point[2] + P.DOOR_SILL), q4(point[2] + P.DOOR_SILL + P.DOOR_H)
    floor = lo[2]
    mouth = [0.0, 0.0, floor + 24.0]
    mouth[axis] = point[axis] + sign * (site['t_exit'] + P.WALL * 2.0)
    mouth[other] = (lo[other] + hi[other]) / 2.0
    inner = list(mouth)
    sweep = [mouth[0], mouth[1], floor]
    sweep[axis] = lo[axis] if sign > 0 else hi[axis]
    inner[axis] = sweep[axis]
    return {'tile': tile, 'name': name, 'kind': site['kind'], 'axis': axis,
            'sgn': sign, 'node': point, 'mouth': mouth, 'inner': inner,
            'floor': sweep, 'aperture': [lo, hi],
            'outbound_free_span': max(0.0, site['t_exit'] - site['t_out']),
            'support_domain_atom_mass': site['support_domain_atom_mass'],
            'support_residual_atom_mass': site['support_residual_atom_mass'],
            'support_source_solid_candidate_mass': site['support_source_solid_candidate_mass']}

def hall_shell(a, b, axis, width, height, thickness):
    other = 1 - axis
    z = (a[2] + b[2]) / 2.0
    along_lo, along_hi = sorted((a[axis], b[axis]))
    cross_lo = min(a[other], b[other]) - width / 2.0
    cross_hi = max(a[other], b[other]) + width / 2.0

    def bounds(alo, ahi, clo, chi, zlo, zhi):
        lo = [0.0, 0.0, zlo]
        hi = [0.0, 0.0, zhi]
        lo[axis], hi[axis] = alo, ahi
        lo[other], hi[other] = clo, chi
        return box_faces(lo, hi, 'exx/base-crete03')

    rows = [
        bounds(along_lo - thickness, along_hi + thickness,
               cross_lo - thickness, cross_hi + thickness, z - thickness, z),
        bounds(along_lo - thickness, along_hi + thickness,
               cross_lo - thickness, cross_hi + thickness,
               z + height, z + height + thickness),
        bounds(along_lo - thickness, along_hi + thickness,
               cross_lo - thickness, cross_lo, z - thickness, z + height + thickness),
        bounds(along_lo - thickness, along_hi + thickness,
               cross_hi, cross_hi + thickness, z - thickness, z + height + thickness),
    ]
    endpoints = sorted((a, b), key=lambda point: point[axis])
    for side, point in enumerate(endpoints):
        edge = along_lo if side == 0 else along_hi
        door_lo = point[other] - width / 2.0
        door_hi = point[other] + width / 2.0
        rows.append(bounds(edge - thickness, edge + thickness,
                           cross_lo - thickness, door_lo, z, z + height))
        rows.append(bounds(edge - thickness, edge + thickness,
                           door_hi, cross_hi + thickness, z, z + height))
    return rows, cross_hi - cross_lo

def carve_and_connect(world, portals, entities=()):
    joins = []
    connector_nodes = []
    connector_links = []
    carve_depth = P.WALL * 2.0
    carve_clearance = P.WALL / 4.0
    transverse_overlap = P.WALL - carve_clearance
    patch_cut_resolution = transverse_overlap
    approach_overlap = 1.0 + max(
        abs(NS.CART_RIDER_MIN[0]), abs(NS.CART_RIDER_MIN[1]),
        abs(NS.CART_RIDER_MAX[0]), abs(NS.CART_RIDER_MAX[1]),
    ) / P.WALL
    brush_owners = [world] + [entity for entity in entities if entity.brushes]
    brushes = []
    owners = []
    for owner in brush_owners:
        brushes.extend(owner.brushes)
        owners.extend([owner] * len(owner.brushes))
    extents = [MS.brush_bounds(brush) for brush in brushes]
    lower = np.asarray([
        lo if lo is not None else [np.inf, np.inf, np.inf]
        for lo, _ in extents
    ], dtype=np.float64).reshape(-1, 3)
    upper = np.asarray([
        hi if hi is not None else [-np.inf, -np.inf, -np.inf]
        for _, hi in extents
    ], dtype=np.float64).reshape(-1, 3)
    active = np.ones(len(brushes), dtype=bool)
    patch_owners = [world] + [entity for entity in entities if entity.patches]
    patches = []
    patch_destinations = []
    for owner in patch_owners:
        patches.extend(owner.patches)
        patch_destinations.extend([owner] * len(owner.patches))
    patch_extents = [patch.bounds() for patch in patches]
    patch_lower = np.asarray([
        lo if lo is not None else [np.inf, np.inf, np.inf]
        for lo, _ in patch_extents
    ], dtype=np.float64).reshape(-1, 3)
    patch_upper = np.asarray([
        hi if hi is not None else [-np.inf, -np.inf, -np.inf]
        for _, hi in patch_extents
    ], dtype=np.float64).reshape(-1, 3)
    patch_active = np.ones(len(patches), dtype=bool)

    def append_brushes(rows, destinations=None):
        nonlocal lower, upper, active
        rows = list(rows)
        if not rows:
            return
        destinations = [world] * len(rows) if destinations is None else list(destinations)
        row_extents = [MS.brush_bounds(brush) for brush in rows]
        brushes.extend(rows)
        owners.extend(destinations)
        lower = np.concatenate((lower, np.asarray([
            lo if lo is not None else [np.inf, np.inf, np.inf]
            for lo, _ in row_extents
        ], dtype=np.float64)))
        upper = np.concatenate((upper, np.asarray([
            hi if hi is not None else [-np.inf, -np.inf, -np.inf]
            for _, hi in row_extents
        ], dtype=np.float64)))
        active = np.concatenate((active, np.ones(len(rows), dtype=bool)))

    def append_patches(rows, destinations):
        nonlocal patch_lower, patch_upper, patch_active
        rows = list(rows)
        if not rows:
            return
        extents = [patch.bounds() for patch in rows]
        patches.extend(rows)
        patch_destinations.extend(destinations)
        patch_lower = np.concatenate((patch_lower, np.asarray([
            lo if lo is not None else [np.inf, np.inf, np.inf]
            for lo, _ in extents
        ], dtype=np.float64)))
        patch_upper = np.concatenate((patch_upper, np.asarray([
            hi if hi is not None else [-np.inf, -np.inf, -np.inf]
            for _, hi in extents
        ], dtype=np.float64)))
        patch_active = np.concatenate((patch_active, np.ones(len(rows), dtype=bool)))

    def patch_meets_cutter(patch, cutter_halfspaces, carve_lo, carve_hi):
        control = patch.control()[:, :, :3]
        if not control.size:
            return False
        for triangle in NS._patch_triangles(NS._patch_collision_grid(control)):
            block = NS._triangle_prism(*triangle)
            if block is None:
                continue
            triangle = np.asarray(triangle, dtype=np.float64)
            lo = np.maximum(triangle.min(axis=0) - NS.PATCH_COLLISION_SNAP, carve_lo)
            hi = np.minimum(triangle.max(axis=0) + NS.PATCH_COLLISION_SNAP, carve_hi)
            if np.all(lo <= hi) and NS.nonempty(np.vstack((block, cutter_halfspaces)), lo, hi):
                return True
        return False

    def patch_inside_cutter(patch, cutter_halfspaces):
        control = patch.control()[:, :, :3].reshape((-1, 3))
        return bool(len(control) and np.all(
            control @ cutter_halfspaces[:, :3].T
            <= cutter_halfspaces[:, 3][None, :] + NS.COLLISION_EPSILON
        ))

    def carve(cutter):
        cutter_faces = cutter.to_faces()
        carve_lo, carve_hi = MS.brush_bounds(cutter_faces)
        carve_lo = np.asarray(carve_lo, dtype=np.float64)
        carve_hi = np.asarray(carve_hi, dtype=np.float64)
        selected = np.flatnonzero(
            active & np.all(lower < carve_hi, axis=1)
            & np.all(upper > carve_lo, axis=1)
        )
        replacements = []
        destinations = []
        for index in selected:
            rows = MS.subtract_convex(brushes[index], cutter_faces)
            replacements.extend(rows)
            destinations.extend([owners[index]] * len(rows))
        active[selected] = False
        append_brushes(replacements, destinations)
        cutter_halfspaces = MS.brush_halfspaces(cutter_faces)
        selected_patches = np.flatnonzero(
            patch_active & np.all(patch_lower < carve_hi, axis=1)
            & np.all(patch_upper > carve_lo, axis=1)
        )
        replacement_patches = []
        replacement_destinations = []
        for index in selected_patches:
            pending = patches[index].quadratic_blocks()
            while pending:
                patch = pending.pop()
                lo, hi = patch.bounds()
                if (np.any(hi <= carve_lo) or np.any(lo >= carve_hi)
                        or not patch_meets_cutter(
                            patch, cutter_halfspaces, carve_lo, carve_hi,
                        )):
                    replacement_patches.append(patch)
                    replacement_destinations.append(patch_destinations[index])
                elif (not patch_inside_cutter(patch, cutter_halfspaces)
                      and np.max(hi - lo) > patch_cut_resolution):
                    pending.extend(patch.subdivide_quadratic())
        patch_active[selected_patches] = False
        append_patches(replacement_patches, replacement_destinations)

    for i in range(0, len(portals), 2):
        a, b = portals[i], portals[i + 1]
        direction_residual = (int(a['axis'] != b['axis']) + int(a['sgn'] == b['sgn'])
                              + int((b['floor'][a['axis']] - a['floor'][a['axis']])
                                    * a['sgn'] <= 0.0))
        approach_floor_mass = 0
        endpoint_brushes = []
        endpoint_embed_depths = []
        for endpoint in (a, b):
            node_floor = [endpoint['node'][0], endpoint['node'][1],
                          endpoint['node'][2] + P.DOOR_SILL]
            site_to_aperture = max(
                0.0,
                endpoint['sgn'] * (
                    endpoint['floor'][endpoint['axis']]
                    - node_floor[endpoint['axis']]
                ),
            )
            endpoint_embed_depth = (
                carve_depth + site_to_aperture + patch_cut_resolution
                + transverse_overlap
            )
            endpoint_embed_depths.append(endpoint_embed_depth)
            cutter_a = [node_floor[0], node_floor[1],
                        node_floor[2] - carve_clearance]
            cutter_b = [endpoint['mouth'][0], endpoint['mouth'][1],
                        endpoint['mouth'][2] - 24.0 - carve_clearance]
            carve(MS.corridor_volume(
                cutter_a, cutter_b, P.DOOR_W + 2.0 * carve_clearance,
                P.DOOR_H + 2.0 * carve_clearance, carve_depth,
            ))
            endpoint_brushes.extend(
                brush.to_faces() for brush in MS.connector(
                    endpoint['floor'],
                    [endpoint['mouth'][0], endpoint['mouth'][1],
                     endpoint['mouth'][2] - 24.0],
                    P.DOOR_W, P.DOOR_H, P.WALL,
                    endpoint_embed_depth / P.WALL,
                    'exx/floor-tread01', 'exx/base-crete03', 'exx/base-metal04',
                )
            )
            approach = MS.connector(
                node_floor, endpoint['floor'], P.DOOR_W, P.DOOR_H, P.WALL,
                approach_overlap,
                'exx/floor-tread01', 'exx/base-crete03', 'exx/base-metal04',
            )
            if approach:
                endpoint_brushes.append(approach[0].to_faces())
                approach_floor_mass += 1
        hall_a = [a['mouth'][0], a['mouth'][1], a['mouth'][2] - 24.0]
        hall_b = [b['mouth'][0], b['mouth'][1], b['mouth'][2] - 24.0]
        hall, hall_cross_span = hall_shell(
            hall_a, hall_b, a['axis'], P.DOOR_W, P.DOOR_H, P.WALL,
        )
        other = 1 - a['axis']
        hall_carve_a = list(hall_a)
        hall_carve_b = list(hall_b)
        hall_carve_a[other] = hall_carve_b[other] = (
            hall_a[other] + hall_b[other]
        ) / 2.0
        hall_carve_a[2] -= carve_clearance
        hall_carve_b[2] -= carve_clearance
        carve(MS.corridor_volume(
            hall_carve_a, hall_carve_b,
            hall_cross_span + 2.0 * carve_clearance,
            P.DOOR_H + 2.0 * carve_clearance, P.WALL,
        ))
        append_brushes(endpoint_brushes)
        append_brushes(hall)
        hull_radius = max(
            abs(NS.CART_RIDER_MIN[a['axis']]), abs(NS.CART_RIDER_MAX[a['axis']]),
        )
        hall_entry_depth = P.WALL + carve_clearance + hull_radius
        entry_a = list(a['mouth'])
        entry_b = list(b['mouth'])
        entry_a[a['axis']] += a['sgn'] * hall_entry_depth
        entry_b[b['axis']] += b['sgn'] * hall_entry_depth
        middle = (entry_a[a['axis']] + entry_b[a['axis']]) / 2.0
        corner_a = list(entry_a)
        corner_b = list(entry_b)
        corner_a[a['axis']] = middle
        corner_b[b['axis']] = middle
        chain = [a['node']]

        def extend(target):
            start = chain[-1]
            distance = math.dist(start, target)
            parts = max(1, int(math.ceil(distance / 256.0)))
            for step in range(1, parts + 1):
                point = [start[axis] + (target[axis] - start[axis]) * step / parts
                         for axis in range(3)]
                if point != chain[-1]:
                    chain.append(point)

        for target in (a['inner'], a['mouth'], entry_a, corner_a, corner_b,
                       entry_b, b['mouth'], b['inner'], b['node']):
            extend(target)
        for point in chain[1:-1]:
            connector_nodes.append(point)
        for left, right in zip(chain, chain[1:]):
            connector_links.extend(((left, right), (right, left)))
        horizontal = math.dist(a['floor'][:2], b['floor'][:2])
        rise = b['floor'][2] - a['floor'][2]
        longitudinal_overlap = min(
            depth - carve_depth - patch_cut_resolution
            - max(0.0, endpoint['sgn'] * (
                endpoint['floor'][endpoint['axis']]
                - endpoint['node'][endpoint['axis']]
            ))
            for depth, endpoint in zip(endpoint_embed_depths, (a, b))
        )
        joins.append({'a': a['tile'], 'b': b['tile'], 'kind': 'corridor',
                      'sa': a['mouth'], 'sb': b['mouth'],
                      'source_a': a['node'], 'source_b': b['node'], 'chain': chain,
                      'width': P.DOOR_W, 'height': P.DOOR_H,
                      'carve_depth': carve_depth,
                      'embed_depth': max(endpoint_embed_depths),
                      'carve_clearance': carve_clearance,
                      'longitudinal_seal_overlap': longitudinal_overlap,
                      'transverse_seal_overlap': transverse_overlap,
                      'seal_overlap': min(longitudinal_overlap, transverse_overlap),
                      'direction_residual_mass': direction_residual,
                      'hall_cross_span': hall_cross_span,
                      'hall_entry_depth': hall_entry_depth,
                      'approach_floor_mass': approach_floor_mass,
                      'horizontal_span': horizontal, 'rise': rise,
                      'grade': abs(rise) / max(horizontal, 1.0)})
    for owner in brush_owners:
        owner.brushes = []
    for owner in patch_owners:
        owner.patches = []
    for index, brush in enumerate(brushes):
        if active[index]:
            owners[index].brushes.append(brush)
    for index, patch in enumerate(patches):
        if patch_active[index]:
            patch_destinations[index].patches.append(patch)
    return joins, connector_nodes, connector_links

def write_waypoints(path, order, surveyed, offsets, connector_nodes, connector_links):
    stamp = time.strftime('%Y-%m-%d %H:%M:%S')
    rows = ['//WAYPOINT_VERSION 1.04', '//WAYPOINT_SYMMETRY 0', '//WAYPOINT_TIME ' + stamp]
    links = ['//WAYPOINT_VERSION 1.04', '//WAYPOINT_TIME ' + stamp]
    seen = set()
    for name, offset in zip(order, offsets):
        src = surveyed[name][0]
        for left, right, flags in src.wptriples:
            a = P.vadd(left, offset)
            b = P.vadd(right, offset)
            key = (P.vstr(a), P.vstr(b), int(flags))
            if key not in seen:
                rows.extend((key[0], key[1], str(key[2])))
                seen.add(key)
        for left, right in src.cachelinks:
            links.append(P.vstr(P.vadd(left, offset)) + '*' + P.vstr(P.vadd(right, offset)))
    for point in connector_nodes:
        key = (P.vstr(point), P.vstr(point), 0)
        if key not in seen:
            rows.extend((key[0], key[1], '0'))
            seen.add(key)
    for left, right in connector_links:
        links.append(P.vstr(left) + '*' + P.vstr(right))
    with open(path + '.waypoints', 'w') as handle:
        handle.write('\n'.join(rows) + '\n')
    with open(path + '.waypoints.cache', 'w') as handle:
        handle.write('\n'.join(links) + '\n')
    return (len(rows) - 3) // 3, len(links) - 2

def navmesh_measure(order, surveyed, offsets, connector_nodes, connector_links, joins):
    def key(point):
        return tuple(round(float(value), 3) for value in point)

    nodes = []
    index = {}
    adjacency = []
    region = {}

    def node(point, tile=None):
        identity = key(point)
        if identity not in index:
            index[identity] = len(nodes)
            nodes.append([float(value) for value in point])
            adjacency.append(set())
        if tile is not None:
            region[identity] = tile
        return index[identity]

    for tile, (name, offset) in enumerate(zip(order, offsets)):
        source = surveyed[name][0]
        local = [node(P.vadd(point, offset), tile) for point in source.navnodes]
        for left, neighbors in enumerate(source.navadj):
            for right in neighbors:
                adjacency[local[left]].add(local[right])
    for point in connector_nodes:
        node(point)
    for left, right in connector_links:
        adjacency[node(left)].add(node(right))
    reps = {}
    for tile in range(len(order)):
        choices = [index[identity] for identity, owner in region.items() if owner == tile]
        reps[tile] = choices[len(choices) // 2] if choices else None
    result = P.navmesh_solve(nodes, adjacency, region, key, len(order), reps)
    join_navigation = []
    cart_width = NS.CART_RIDER_MAX[0] - NS.CART_RIDER_MIN[0]
    cart_height = NS.CART_RIDER_MAX[2] - NS.CART_RIDER_MIN[2]
    for join in joins:
        chain = [index.get(key(point)) for point in join['chain']]
        graph_link_mass = sum(
            left is not None and right is not None
            and right in adjacency[left] and left in adjacency[right]
            for left, right in zip(chain, chain[1:])
        )
        expected_graph_link_mass = max(0, len(chain) - 1)
        source_a = index.get(key(join['source_a']))
        source_b = index.get(key(join['source_b']))
        source_attachment_mass = int(
            source_a is not None and region.get(key(join['source_a'])) == join['a']
        ) + int(
            source_b is not None and region.get(key(join['source_b'])) == join['b']
        )
        aperture_mass = int(join['width'] >= cart_width) + int(join['height'] >= cart_height)
        cart_navigable = (graph_link_mass == expected_graph_link_mass
                          and source_attachment_mass == 2 and aperture_mass == 2)
        join_navigation.append({
            'source': join['a'], 'target': join['b'],
            'graph_link_mass': graph_link_mass,
            'expected_graph_link_mass': expected_graph_link_mass,
            'source_attachment_mass': source_attachment_mass,
            'aperture_dimension_mass': aperture_mass,
            'cart_navigable': cart_navigable,
        })
    return {
        'node_mass': result['n_nodes'],
        'walking_diameter': result['walk_diameter'],
        'walking_median': result['walk_median'],
        'unreachable_pair_mass': result['unreachable_pairs'],
        'reachable_node_mass_by_tile': result['coverage'],
        'region_walk': [{'source': left, 'successor': right, 'distance': distance}
                        for (left, right), distance in sorted(result['region_walk'].items())],
        'join_navigation': join_navigation,
    }

def join_geometry_measure(ns, joins, navmesh):
    segment_owners = []
    segment_locals = []
    segment_starts = []
    segment_ends = []
    for index, join in enumerate(joins):
        if join['kind'] == 'corridor':
            segment_mass = max(0, len(join['chain']) - 1)
            segment_owners.extend([index] * segment_mass)
            segment_locals.extend(range(segment_mass))
            segment_starts.extend(join['chain'][:-1])
            segment_ends.extend(join['chain'][1:])
    free, supported, relation_measures = ns.segment_relations(
        segment_starts, segment_ends,
        NS.CART_RIDER_MIN, NS.CART_RIDER_MAX,
    )
    segment_owners = np.asarray(segment_owners, dtype=np.int64)
    segment_locals = np.asarray(segment_locals, dtype=np.int64)
    clearance_by_join = np.bincount(
        segment_owners, weights=~free, minlength=len(joins),
    ).astype(np.int64)
    support_by_join = np.bincount(
        segment_owners, weights=~supported, minlength=len(joins),
    ).astype(np.int64)
    clearance_gap_mass = int(clearance_by_join.sum())
    support_gap_mass = int(support_by_join.sum())
    direction_residual_mass = 0
    rows = []
    for index, join in enumerate(joins):
        clearance_gaps = int(clearance_by_join[index])
        support_gaps = int(support_by_join[index])
        direction_residual = int(join.get('direction_residual_mass', 0))
        geometry_residual = clearance_gaps + support_gaps + direction_residual
        owner_rows = segment_owners == index
        navigation = navmesh['join_navigation'][index]
        navigation['clearance_gap_mass'] = clearance_gaps
        navigation['support_gap_mass'] = support_gaps
        navigation['direction_residual_mass'] = direction_residual
        navigation['geometry_residual_mass'] = geometry_residual
        if join['kind'] == 'corridor':
            structural = (
                navigation['graph_link_mass'] == navigation['expected_graph_link_mass']
                and navigation['source_attachment_mass'] == 2
                and navigation['aperture_dimension_mass'] == 2
            )
            navigation['cart_navigable'] = bool(
                structural and geometry_residual == 0
            )
            join['cart_navigable'] = navigation['cart_navigable']
        direction_residual_mass += direction_residual
        rows.append({
            'join': index,
            'kind': join['kind'],
            'clearance_gap_mass': clearance_gaps,
            'clearance_gap_segments': segment_locals[owner_rows & ~free].tolist(),
            'support_gap_mass': support_gaps,
            'support_gap_segments': segment_locals[owner_rows & ~supported].tolist(),
            'direction_residual_mass': direction_residual,
            'geometry_residual_mass': geometry_residual,
        })
    return {
        'join_mass': len(joins),
        'corridor_join_mass': sum(join['kind'] == 'corridor' for join in joins),
        'clearance_gap_mass': clearance_gap_mass,
        'support_gap_mass': support_gap_mass,
        'direction_residual_mass': direction_residual_mass,
        'geometry_residual_mass': (clearance_gap_mass + support_gap_mass
                                   + direction_residual_mass),
        'relation_measures': relation_measures,
        'joins': rows,
    }

BUNDLE_NAMES = ('fused.map', 'fused.bsp', 'fused.ent', 'fused.waypoints',
                'fused.waypoints.cache', 'fused.mapinfo', 'fused.joins.json',
                'fused.metrics.json', 'fused.measurements.json', 'fused.compile.json')

def compiler_overlay_rows(outdir):
    path = os.path.join(outdir, 'fused.compile.json')
    if not os.path.exists(path):
        return []
    record = json.load(open(path))
    measures = record.get('measures') or {}
    return measures.get('realized_missing_image_aliases') or []

def stage_compiler_overlays(workdir, outdir, rows):
    for row in rows:
        name = row['logical_path']
        source = os.path.join(workdir, 'fs', 'data', *name.split('/'))
        destination = os.path.join(outdir, 'asset-overlays', *name.split('/'))
        if os.path.exists(source):
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            shutil.copyfile(source, destination)

def missing_bundle_artifacts(outdir):
    missing = [name for name in BUNDLE_NAMES if not os.path.exists(os.path.join(outdir, name))]
    missing.extend(
        'asset-overlays/' + row['logical_path']
        for row in compiler_overlay_rows(outdir)
        if not os.path.exists(os.path.join(outdir, 'asset-overlays', *row['logical_path'].split('/')))
    )
    return missing

def bundle(outdir):
    target = os.path.join(outdir, 'fused.pk3')
    missing = missing_bundle_artifacts(outdir)
    handle = tempfile.NamedTemporaryFile(dir=outdir, prefix='fused.pk3.', delete=False)
    temporary = handle.name
    handle.close()
    try:
        with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for name in BUNDLE_NAMES:
                path = os.path.join(outdir, name)
                if os.path.exists(path):
                    archive.write(path, 'maps/' + name)
            for row in compiler_overlay_rows(outdir):
                name = row['logical_path']
                path = os.path.join(outdir, 'asset-overlays', *name.split('/'))
                if os.path.exists(path):
                    archive.write(path, name)
        os.chmod(temporary, 0o644)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return target, missing

def release_residual(metrics, missing):
    measurements = metrics.get('map_measurements') or {}
    construction = measurements.get('cart_construction_measures') or {}
    stage_codes = metrics.get('q3map2_stage_returncodes') or {}
    components = (metrics.get('region_graph') or {}).get('component_mass')
    requested_maps = int(metrics.get('requested_maps') or 0)
    realized_stock_maps = int(metrics.get('realized_stock_maps') or metrics.get('realized_maps') or 0)
    compiler = metrics.get('q3map2_compile_measures') or {}
    source_observations = metrics.get('source_observations') or []
    join_geometry = metrics.get('join_geometry') or {}
    residuals = {
        'unfinished_source_mass': int(metrics.get('unfinished_source_mass') or 0),
        'missing_stock_map_mass': max(0, requested_maps - realized_stock_maps),
        'compiler_stage_residual_mass': sum(int(int(code) != 0) for code in stage_codes.values()),
        'compiler_leak_line_mass': int(metrics.get('q3map2_leak_line_mass') or 0),
        'compiler_shaderlist_missing_mass': int(compiler.get('shaderlist_missing_mass') or 0),
        'compiler_missing_image_line_mass': int(compiler.get('missing_image_line_mass') or 0),
        'compiler_missing_file_line_mass': int(compiler.get('missing_file_line_mass') or 0),
        'compiler_error_line_mass': int(compiler.get('error_line_mass') or 0),
        'missing_bsp_mass': 1 - min(1, int(metrics.get('bsp_artifact_mass') or 0)),
        'bsp_problem_mass': int(metrics.get('bsp_problem_mass') or 0),
        'source_translation_error_mass': int(metrics.get('source_translation_error_mass') or 0),
        'portal_support_residual_mass': sum(
            int(row.get('support_residual_atom_mass') or 0) for row in source_observations
        ),
        'corridor_geometry_residual_mass': int(
            join_geometry.get('geometry_residual_mass') or 0),
        'authored_source_missing_mass': int(metrics.get('authored_source_missing_mass') or 0),
        'bsp_coordinate_excess_axis_mass': int(metrics.get('bsp_coordinate_excess_axis_mass') or 0),
        'negative_space_error_mass': int(metrics.get('negative_space_error_mass') or 0),
        'overlay_error_mass': int(metrics.get('overlay_error_mass') or 0),
        'region_component_residual_mass': 0 if components is None else abs(int(components) - 1),
        'cart_construction_residual_mass': int(construction.get('construction_residual_mass') or 0),
        'team_cart_nonadvanceable_pair_mass': int(measurements.get('team_cart_nonadvanceable_pair_mass') or 0),
        'missing_bundle_artifact_mass': len(missing),
    }
    residuals['release_residual_mass'] = sum(residuals.values())
    return residuals

def finish_existing(args):
    started = time.time()
    meter = WorkloadMeter('mapfuse', {'environment': 'mapfuse-%d' % args.seed,
                                      'host_role': 'map-builder'})
    outdir = os.path.abspath(os.path.expanduser(args.out))
    workdir = os.path.abspath(os.path.expanduser(args.work or os.path.join(outdir, '.source')))
    pk3 = os.path.abspath(os.path.expanduser(args.pk3))
    assetroot = os.path.abspath(os.path.expanduser(
        args.basepath or os.path.dirname(os.path.dirname(pk3))
    ))
    MS.XONDIR = assetroot
    MS.Q3MAP2 = os.path.abspath(os.path.expanduser(args.q3map2))
    bsp = os.path.join(outdir, 'fused.bsp')
    compile_path = os.path.join(outdir, 'fused.compile.json')
    compile_record = json.load(open(compile_path))
    prior_codes = compile_record.get('stage_returncodes') or {}
    if (not compile_record.get('vis') or not compile_record.get('light')
            or any(int(code) != 0 for code in prior_codes.values())):
        source_map = os.path.join(workdir, 'fs', 'data', 'maps', 'fused.map')
        meta_ok = int(prior_codes.get('meta', 1)) == 0
        stages = (['meta', 'vis', 'light'] if not meta_ok else
                  [stage for stage in ('vis', 'light')
                   if int(prior_codes.get(stage, 1)) != 0])
        with meter.span('q3map2-resume', operations={'input_bytes': os.path.getsize(source_map)}):
            new_codes, _, logs, new_measures = MS.compile_map(
                source_map, workdir, True, True, q3map2=MS.Q3MAP2,
                basepath=assetroot, stages=stages,
            )
        for stage, output in logs.items():
            with open(os.path.join(outdir, 'q3map2-' + stage + '.log'), 'w') as handle:
                handle.write(output)
        stage_codes = {**prior_codes, **new_codes}
        compile_measures = (new_measures if 'meta' in stages
                            else compile_record.get('measures') or {})
        stage_logs = []
        for stage in ('meta', 'vis', 'light'):
            path = os.path.join(outdir, 'q3map2-' + stage + '.log')
            if os.path.exists(path):
                stage_logs.append(open(path).read())
        leak_line_mass = sum('leaked' in line.lower()
                             for output in stage_logs for line in output.splitlines())
        built_bsp = os.path.join(workdir, 'fs', 'data', 'maps', 'fused.bsp')
        compile_record = {
            'schema': 2, 'stage_returncodes': stage_codes,
            'leak_line_mass': leak_line_mass, 'measures': compile_measures,
            'vis': int(stage_codes.get('vis', 1)) == 0,
            'light': int(stage_codes.get('light', 1)) == 0,
            'stages': sorted(set(compile_record.get('stages') or ()) | set(logs)),
            'bsp_artifact_mass': int(os.path.exists(built_bsp)),
        }
        with open(compile_path, 'w') as handle:
            json.dump(compile_record, handle, indent=2, sort_keys=True)
            handle.write('\n')
        if os.path.exists(built_bsp):
            shutil.copyfile(built_bsp, bsp)
    cache = bsp + '.negspace.npz'
    if os.path.exists(cache):
        ns = NS.load_saved(cache)
        if ns.schema == NS.NEGSPACE_SCHEMA:
            print('resume loaded', cache)
        else:
            ns = None
    else:
        ns = None
    if ns is None:
        with meter.span('negative-space', operations={'input_bytes': os.path.getsize(bsp)}):
            ns = NS.from_bsp(bsp)
        NS.save(ns, cache)
        print('resume wrote', cache)
    waypoint_count, link_count, compiled_waypoint_measures = P.realize_waypoint_files(
        os.path.join(outdir, 'fused'), ns,
    )
    with meter.span('payload-overlay', rows=args.carts,
                    operations={'teams': args.teams, 'carts': args.carts}):
        M.emit(bsp, os.path.join(outdir, 'fused.ent'), args.teams, args.carts, pk3, ns=ns)
    measurements_source = os.path.join(outdir, 'fused.ent.measurements.json')
    measurements_path = os.path.join(outdir, 'fused.measurements.json')
    shutil.copyfile(measurements_source, measurements_path)
    measurements = json.load(open(measurements_path))
    joins = json.load(open(os.path.join(outdir, 'fused.joins.json')))
    for row in joins.get('portals', []):
        row.pop('support_decomposition_overflow_mass', None)
    metrics_path = os.path.join(outdir, 'fused.metrics.json')
    prior = json.load(open(metrics_path)) if os.path.exists(metrics_path) else {}
    for row in prior.get('source_observations') or []:
        row.pop('support_decomposition_overflow_mass', None)
    map_rows = joins.get('maps', [])
    join_rows = joins.get('joins', [])
    navmesh = prior.get('navmesh') or {}
    if len(navmesh.get('join_navigation') or []) == len(join_rows):
        join_geometry = join_geometry_measure(ns, join_rows, navmesh)
        joins['joins'] = join_rows
        with open(os.path.join(outdir, 'fused.joins.json'), 'w') as handle:
            json.dump(joins, handle, indent=2)
            handle.write('\n')
    else:
        join_geometry = prior.get('join_geometry') or {}
    noncart_incidence = [0] * len(map_rows)
    for row in join_rows:
        if not row.get('cart_navigable'):
            noncart_incidence[int(row['a'])] += 1
            noncart_incidence[int(row['b'])] += 1
    graph = P.region_graph_solve(
        len(map_rows), [(int(row['a']), int(row['b'])) for row in join_rows],
    )
    graph_metrics = {'component_mass': len(graph['components']),
                     'components': graph['components'],
                     'articulation_tile_mass': len(graph['articulation']),
                     'articulation_tiles': graph['articulation'],
                     'cut_edge_mass': len(graph['cutedges']),
                     'cut_edges': graph['cutedges'],
                     'degree': graph['degree'],
                     'hop_diameter': graph['hop_diameter']}
    coordinate_extent = [
        max((abs(float(row[key][axis])) for row in map_rows for key in ('mins', 'maxs')),
            default=0.0)
        for axis in range(3)
    ]
    coordinate_excess = [max(0.0, value - BSP_COORDINATE_EXTENT)
                         for value in coordinate_extent]
    bsp_problems = P.check_bsp(open(bsp, 'rb').read())
    realized_stock = sum(not bool(row.get('bridge')) for row in map_rows)
    realized_bridges = len(map_rows) - realized_stock
    stage_compiler_overlays(workdir, outdir,
                            (compile_record.get('measures') or {}).get(
                                'realized_missing_image_aliases') or [])
    metrics = {**prior, 'schema': 3, 'resumed': True,
               'requested_maps': int(prior.get('requested_maps') or realized_stock),
               'requested_map_names': prior.get('requested_map_names') or [
                   item['name'] for item in map_rows if not item.get('bridge')],
               'realized_stock_maps': realized_stock,
               'requested_bridges': int(prior.get('requested_bridges') or realized_bridges),
               'realized_bridges': realized_bridges,
               'realized_maps': len(map_rows),
               'maps': [item['name'] for item in map_rows],
               'joins': len(join_rows),
               'cart_navigable_join_mass': sum(bool(row.get('cart_navigable'))
                                                for row in join_rows),
               'noncart_join_mass': sum(not bool(row.get('cart_navigable'))
                                        for row in join_rows),
               'noncart_join_incidence': noncart_incidence,
               'join_geometry': join_geometry, 'navmesh': navmesh,
               'compiled_waypoint_projection': compiled_waypoint_measures,
               'bsp_coordinate_extent': coordinate_extent,
               'bsp_coordinate_excess': coordinate_excess,
               'bsp_coordinate_excess_axis_mass': sum(value > 0 for value in coordinate_excess),
               'region_graph': graph_metrics,
               'q3map2_stage_returncodes': compile_record.get('stage_returncodes', {}),
               'q3map2_leak_line_mass': int(compile_record.get('leak_line_mass') or 0),
               'q3map2_compile_measures': compile_record.get('measures') or {},
               'bsp_artifact_mass': int(os.path.exists(bsp)),
               'bsp_problem_mass': len(bsp_problems),
               'bsp_problems': bsp_problems,
               'map_measurements': measurements, 'wall_seconds': round(time.time() - started, 3)}
    observed_capacity = {os.path.basename(str(row.get('name', ''))): int(row.get('site_mass') or 0)
                         for row in prior.get('source_observations') or []}
    metrics['socket_capacity'] = [observed_capacity.get(row['name'], 0) for row in map_rows]
    metrics.pop('effective_socket_capacity', None)
    metrics.pop('socket_capacity_extension_mass', None)
    metrics.pop('portal_support_decomposition_overflow_mass', None)
    metrics.pop('noncart_join_budget_residual_mass', None)
    with open(metrics_path, 'w') as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write('\n')
    missing = missing_bundle_artifacts(outdir)
    residuals = release_residual(metrics, missing)
    metrics.update(residuals)
    with open(metrics_path, 'w') as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write('\n')
    archive, missing = bundle(outdir)
    print('finished %s maps=%d joins=%d carts=%d teams=%d' %
          (archive, metrics['realized_maps'], metrics['joins'], args.carts, args.teams))
    print('bundle missing_artifact_mass=%d artifacts=%s release_residual_mass=%d' %
          (len(missing), missing, residuals['release_residual_mass']))
    return 0

def generate(args):
    started = time.time()
    rng = random.Random(args.seed)
    meter = WorkloadMeter('mapfuse', {'environment': 'mapfuse-%d' % args.seed,
                                      'host_role': 'map-builder'})
    pk3 = os.path.abspath(os.path.expanduser(args.pk3))
    q3map2 = os.path.abspath(os.path.expanduser(args.q3map2))
    assetroot = os.path.abspath(os.path.expanduser(
        args.basepath or os.path.dirname(os.path.dirname(pk3))
    ))
    MS.XONDIR = assetroot
    MS.Q3MAP2 = q3map2
    G.BASEPATH = assetroot
    outdir = os.path.abspath(os.path.expanduser(args.out))
    workdir = os.path.abspath(os.path.expanduser(args.work or os.path.join(outdir, '.source')))
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(workdir, 'fs', 'data', 'maps'), exist_ok=True)
    pool = P.navigable_names(pk3)
    explicit = [x for x in args.names.split(',') if x] if args.names else []
    if explicit:
        requested = explicit
    elif args.maps in ('all', 'sample'):
        requested = list(pool)
        rng.shuffle(requested)
    else:
        requested_count = max(0, int(args.maps))
        requested = list(pool)
        rng.shuffle(requested)
        requested = requested[:requested_count]
    print('mapfuse seed=%d requested=%d pool=%d source=q3map2' %
          (args.seed, len(requested), len(pool)))
    with meter.span('site-survey', rows=len(requested), operations={'candidate_maps': len(pool)}):
        surveyed, observations = survey_sources(requested, pk3, outdir)
    stock_keys = [name for name in requested if name in surveyed]
    unfinished_stock = [name for name in requested if name not in surveyed]
    stock_corridor = [name for name in stock_keys if surveyed[name][1]]
    stock_transfer = [name for name in stock_keys if not surveyed[name][1]]
    required_bridges = required_bridge_count(
        stock_corridor, len(stock_transfer), surveyed,
    )
    if args.bridges == 'auto':
        bridge_count = required_bridges
    else:
        bridge_count = max(int(args.bridges), required_bridges)
    generated = []
    bridge_names = []
    bridge_observations = []
    bridge_dir = os.path.join(workdir, 'bridges')
    with meter.span('bridge-source', rows=bridge_count):
        for index in range(bridge_count):
            name = 'bridge%d_%d' % (args.seed, index)
            bridge_names.append(name)
            try:
                base, ports = G.build_bridge_tile(bridge_dir, name, rng.getrandbits(63),
                                                  ['e', 'w'], ['n', 's'],
                                                  q3map2=q3map2, basepath=assetroot)
                generated.append(base)
                bridge_observations.append({'name': name, 'artifact_mass': 1,
                                            'port_mass': len(ports)})
            except (Exception, SystemExit) as exc:
                bridge_observations.append({'name': name, 'artifact_mass': 0,
                                            'error': '%s: %s' % (type(exc).__name__, exc)})
                print('bridge %s artifact_mass=0 error=%s: %s' %
                      (name, type(exc).__name__, exc))
    if generated:
        bridge_surveyed, bridge_surveys = survey_sources(generated, pk3, outdir)
        surveyed.update(bridge_surveyed)
        observations.extend(bridge_surveys)
    bridge_keys = [name for name in generated if name in surveyed]
    corridor_keys = bridge_keys + stock_corridor
    transfer_keys = stock_transfer
    source_keys = corridor_keys + transfer_keys
    unfinished_bridges = [name for name in bridge_names
                          if not any(os.path.basename(value) == name for value in source_keys)]
    if not source_keys:
        target = os.path.join(outdir, 'fused.pk3')
        if os.path.exists(target):
            os.unlink(target)
        metrics = {'schema': 3, 'seed': args.seed, 'requested_maps': len(requested),
                   'requested_map_names': requested,
                   'realized_maps': 0, 'source_observations': observations,
                   'bridge_observations': bridge_observations,
                   'unfinished_stock_names': unfinished_stock,
                   'unfinished_bridge_names': unfinished_bridges,
                   'unfinished_source_mass': len(unfinished_stock) + len(unfinished_bridges),
                   'runnable_bundle_mass': 0,
                   'wall_seconds': round(time.time() - started, 3)}
        with open(os.path.join(outdir, 'fused.metrics.json'), 'w') as handle:
            json.dump(metrics, handle, indent=2, sort_keys=True)
            handle.write('\n')
        print('maps=0 runnable_bundle_mass=0 unfinished_source_mass=%d' %
              metrics['unfinished_source_mass'])
        return 0
    sources = [surveyed[name][0] for name in source_keys]
    corridor_mass = len(corridor_keys)
    (topology, degrees, cells, offsets, boundaries, slack, columns, rows,
     directional_roles) = fabric_layout(
        sources, source_keys, len(bridge_keys), len(stock_corridor),
        len(transfer_keys), surveyed,
    )
    capacities = [len(surveyed[name][1])
                  if index < corridor_mass else 0
                  for index, name in enumerate(source_keys)]
    corridor_edges = topology
    transfers = transfer_edges(corridor_mass, len(transfer_keys), cells)
    for left, right in transfers:
        degrees[left] += 1
        degrees[right] += 1
    edges = corridor_edges + transfers
    selections, socket_reuse_mass, planar_closure = fit_edge_sites(
        corridor_edges, source_keys, surveyed, offsets, slack, cells,
    )
    height_closure = align_portal_heights(
        corridor_edges, selections, source_keys[:corridor_mass], surveyed,
        offsets[:corridor_mass], slack[:corridor_mass],
    )
    world = MS.Ent()
    display_names = [source.name for source in sources]
    world.keys = [('classname', 'worldspawn'), ('message', 'fused ' + '+'.join(display_names))]
    others = []
    tiles = []
    dropped = 0
    translation_error_mass = 0
    authored_source_observations = []
    tile_source_realizations = []
    fs_maps = os.path.join(workdir, 'fs', 'data', 'maps')
    with meter.span('source-assembly', rows=len(source_keys)):
        for index, (name, offset) in enumerate(zip(source_keys, offsets)):
            src = surveyed[name][0]
            authored_map, authored_observation = realize_authored_source(
                name, src, pk3, fs_maps,
            )
            authored_source_observations.append(authored_observation)
            (brushes, patches, entities, count_dropped, count_translation_errors,
             tile_realization) = MS.place_tile(authored_map, offset, workdir, src.name)
            tile_realization['name'] = src.name
            tile_source_realizations.append(tile_realization)
            world.brushes.extend(brushes)
            world.patches.extend(patches)
            others.extend(entities)
            dropped += count_dropped
            translation_error_mass += count_translation_errors
            tiles.append({'name': src.name,
                          'mins': [src.bounds[0][k] + offset[k] for k in range(3)],
                          'maxs': [src.bounds[1][k] + offset[k] for k in range(3)],
                          'bridge': name in generated,
                          'degree': degrees[index]})
            print('place %-18s offset=%s sites=%d' %
                  (src.name, [round(x) for x in offset], len(surveyed[name][1])))
        portals = []
        for (left, right), (left_site, right_site) in zip(corridor_edges, selections):
            portals.append(portal(left, sources[left].name,
                                  surveyed[source_keys[left]][1][left_site], offsets[left]))
            portals.append(portal(right, sources[right].name,
                                  surveyed[source_keys[right]][1][right_site], offsets[right]))
        joins, connector_nodes, connector_links = carve_and_connect(world, portals, others)
        transfer_joins, transfer_links = transfer_connect(
            others, transfers, source_keys, surveyed, offsets,
        )
        joins.extend(transfer_joins)
        connector_links.extend(transfer_links)
        source_map = os.path.join(fs_maps, 'fused.map')
        MS.write_map(source_map, [world] + others)
    shutil.copyfile(source_map, os.path.join(outdir, 'fused.map'))
    waypoint_count, link_count = write_waypoints(os.path.join(outdir, 'fused'), source_keys,
                                                  surveyed, offsets, connector_nodes,
                                                  connector_links)
    with open(os.path.join(outdir, 'fused.mapinfo'), 'w') as handle:
        handle.write('title Fused %s\ndescription source-compiled navigable map fusion\n'
                     'author mapfuse\ngametype dm\ngametype tdm\ngametype plc\n' %
                     '+'.join(display_names))
    region_graph = P.region_graph_solve(len(source_keys), edges)
    cut_edges = set(region_graph['cutedges'])
    for index, join in enumerate(joins):
        join['exclusive'] = (index in cut_edges
                             or min(region_graph['degree'][join['a']],
                                    region_graph['degree'][join['b']]) == 1)
        join['prominent'] = join['exclusive']
    navmesh = navmesh_measure(source_keys, surveyed, offsets, connector_nodes,
                              connector_links, joins)
    for join, navigation in zip(joins, navmesh['join_navigation']):
        join['cart_navigable'] = navigation['cart_navigable']
    vantages = {}
    for index, (source, offset) in enumerate(zip(sources, offsets)):
        vantages[index] = [P.vadd(point, offset) for point in P.walk_nodes(source)[:3]]
    record_portals = [{key: value for key, value in item.items() if key not in ('inner', 'floor')}
                      for item in portals]
    with meter.span('q3map2', rows=len(world.brushes),
                    operations={'input_bytes': os.path.getsize(source_map)}):
        stage_codes, leak_line_mass, logs, compile_measures = MS.compile_map(
            source_map, workdir, not args.no_vis, not args.no_light,
            q3map2=q3map2, basepath=assetroot,
        )
    for stage, text in logs.items():
        with open(os.path.join(outdir, 'q3map2-' + stage + '.log'), 'w') as handle:
            handle.write(text)
    built_bsp = os.path.join(fs_maps, 'fused.bsp')
    bsp_artifact_mass = int(os.path.exists(built_bsp))
    with open(os.path.join(outdir, 'fused.compile.json'), 'w') as handle:
        json.dump({'schema': 2, 'stage_returncodes': stage_codes,
                   'leak_line_mass': leak_line_mass,
                   'measures': compile_measures,
                   'vis': not args.no_vis and int(stage_codes.get('vis', 1)) == 0,
                   'light': not args.no_light and int(stage_codes.get('light', 1)) == 0,
                   'stages': sorted(logs), 'bsp_artifact_mass': bsp_artifact_mass},
                  handle, indent=2, sort_keys=True)
        handle.write('\n')
    stage_compiler_overlays(workdir, outdir,
                            compile_measures.get('realized_missing_image_aliases') or [])
    ns = None
    compiled_waypoint_measures = {}
    negative_space_error = None
    bsp_problems = []
    if bsp_artifact_mass:
        shutil.copyfile(built_bsp, os.path.join(outdir, 'fused.bsp'))
        bsp_problems = P.check_bsp(open(built_bsp, 'rb').read())
        try:
            with meter.span('negative-space', operations={'input_bytes': os.path.getsize(built_bsp)}):
                ns = NS.from_bsp(os.path.join(outdir, 'fused.bsp'))
        except Exception as exc:
            negative_space_error = '%s: %s' % (type(exc).__name__, exc)
    if ns is not None:
        waypoint_count, link_count, compiled_waypoint_measures = P.realize_waypoint_files(
            os.path.join(outdir, 'fused'), ns,
        )
    join_geometry = {
        'join_mass': len(joins),
        'corridor_join_mass': len(corridor_edges),
        'clearance_gap_mass': len(corridor_edges),
        'support_gap_mass': 0,
        'direction_residual_mass': sum(
            int(join.get('direction_residual_mass', 0)) for join in joins
        ),
        'geometry_residual_mass': len(corridor_edges) + sum(
            int(join.get('direction_residual_mass', 0)) for join in joins
        ),
        'joins': [],
    }
    if ns is not None:
        join_geometry = join_geometry_measure(ns, joins, navmesh)
    noncart_incidence = [0] * len(source_keys)
    for join in joins:
        if not join['cart_navigable']:
            noncart_incidence[join['a']] += 1
            noncart_incidence[join['b']] += 1
    record = MS.joins_record(tiles, joins, record_portals, vantages_per_tile=vantages)
    MS.write_joins_json(os.path.join(outdir, 'fused.joins.json'), record)
    overlay_error = None
    if bsp_artifact_mass:
        try:
            with meter.span('payload-overlay', rows=args.carts,
                            operations={'teams': args.teams, 'carts': args.carts}):
                M.emit(os.path.join(outdir, 'fused.bsp'), os.path.join(outdir, 'fused.ent'),
                       args.teams, args.carts, pk3, ns=ns)
        except Exception as exc:
            overlay_error = '%s: %s' % (type(exc).__name__, exc)
            print('payload overlay error:', overlay_error)
    measurements_source = os.path.join(outdir, 'fused.ent.measurements.json')
    measurements = {}
    if os.path.exists(measurements_source):
        shutil.copyfile(measurements_source, os.path.join(outdir, 'fused.measurements.json'))
        measurements = json.load(open(measurements_source))
    graph_metrics = {'component_mass': len(region_graph['components']),
                     'components': region_graph['components'],
                     'articulation_tile_mass': len(region_graph['articulation']),
                     'articulation_tiles': region_graph['articulation'],
                     'cut_edge_mass': len(region_graph['cutedges']),
                     'cut_edges': region_graph['cutedges'],
                     'degree': region_graph['degree'],
                     'hop_diameter': region_graph['hop_diameter']}
    coordinate_extent = [
        max(abs(source.bounds[edge][axis] + offsets[index][axis])
            for index, source in enumerate(sources) for edge in (0, 1))
        for axis in range(3)
    ]
    coordinate_excess = [max(0.0, value - BSP_COORDINATE_EXTENT)
                         for value in coordinate_extent]
    metrics = {'schema': 3, 'seed': args.seed, 'requested_maps': len(requested),
               'requested_map_names': requested,
               'realized_stock_maps': len(stock_keys), 'requested_bridges': bridge_count,
               'realized_bridges': len(source_keys) - len(stock_keys),
               'realized_maps': len(source_keys), 'maps': display_names,
               'unfinished_stock_names': unfinished_stock,
               'unfinished_bridge_names': unfinished_bridges,
               'unfinished_source_mass': len(unfinished_stock) + len(unfinished_bridges),
               'cells': cells, 'placement_shape': [columns, rows, 1],
               'bsp_coordinate_extent': coordinate_extent,
               'bsp_coordinate_excess': coordinate_excess,
               'bsp_coordinate_excess_axis_mass': sum(value > 0 for value in coordinate_excess),
               'placement_coordinate_quantum': P.PORTAL_QUANTUM,
               'placement_off_lattice_coordinate_mass': int(sum(
                   abs(value / P.PORTAL_QUANTUM - round(value / P.PORTAL_QUANTUM))
                   > np.finfo(np.float64).eps
                   for offset in offsets for value in offset
               )),
               'offsets': offsets, 'joins': len(joins),
               'cart_navigable_join_mass': sum(join['cart_navigable'] for join in joins),
               'noncart_join_mass': sum(not join['cart_navigable'] for join in joins),
               'noncart_join_incidence': noncart_incidence,
               'corridor_join_mass': len(corridor_edges),
               'transfer_join_mass': len(transfers),
               'corridor_cell_span_integral': sum(
                   abs(cells[left][0] - cells[right][0])
                   + abs(cells[left][1] - cells[right][1])
                   for left, right in corridor_edges
               ),
               'socket_capacity': capacities,
               'directional_topology_roles': [
                   sorted([list(direction) for direction in role])
                   for role in directional_roles
               ],
               'portal_height_closure': height_closure,
               'portal_planar_closure': planar_closure,
               'join_geometry': join_geometry,
               'corridor_grade_measure': {
                   'atom_mass': len(corridor_edges),
                   'integral': sum(join['grade'] for join in joins
                                   if join['kind'] == 'corridor'),
                   'square_integral': sum(join['grade'] ** 2 for join in joins
                                          if join['kind'] == 'corridor'),
                   'maximum': max((join['grade'] for join in joins
                                   if join['kind'] == 'corridor'), default=None),
               },
               'socket_reuse_mass': socket_reuse_mass, 'waypoints': waypoint_count,
               'waypoint_links': link_count, 'lightgrid_brushes_dropped': dropped,
               'source_cache_endpoint_outside_definition_mass': sum(
                   source.cache_endpoint_outside_definition_mass for source in sources),
               'source_cache_link_outside_definition_mass': sum(
                   source.cache_link_outside_definition_mass for source in sources),
               'source_waypoint_outside_negative_space_mass': sum(
                   source.waypoint_outside_negative_space_mass for source in sources),
               'source_waypoint_projection_unresolved_mass': sum(
                   source.waypoint_projection_unresolved_mass for source in sources),
               'source_waypoint_projection_sweep_mass': sum(
                   source.waypoint_projection_measures['projection_sweep_mass']
                   for source in sources),
               'source_waypoint_projection_candidate_pair_mass': sum(
                   source.waypoint_projection_measures['candidate_pair_mass']
                   for source in sources),
               'source_waypoint_projection_plane_evaluation_mass': sum(
                   source.waypoint_projection_measures['plane_evaluation_mass']
                   for source in sources),
               'source_waypoint_projection_directional_null_pair_mass': sum(
                   source.waypoint_projection_measures['directional_null_pair_mass']
                   for source in sources),
               'source_waypoint_projection_world_boundary_reconciliation_mass': sum(
                   source.waypoint_projection_measures['world_boundary_reconciliation_mass']
                   for source in sources),
               'source_waypoint_displaced_mass': sum(
                   source.waypoint_displaced_mass for source in sources),
               'source_waypoint_displacement_integral': sum(
                   source.waypoint_displacement_integral for source in sources),
               'source_waypoint_displacement_square_integral': sum(
                   source.waypoint_displacement_square_integral for source in sources),
               'source_waypoint_displacement_maximum': max((
                   source.waypoint_displacement_maximum for source in sources
               ), default=0.0),
               'compiled_waypoint_projection': compiled_waypoint_measures,
               'source_translation_error_mass': translation_error_mass,
               'authored_source_observations': authored_source_observations,
               'authored_source_artifact_mass': sum(
                   item['artifact_mass'] for item in authored_source_observations
               ),
               'authored_source_missing_mass': sum(
                   1 - item['artifact_mass'] for item in authored_source_observations
               ),
               'tile_source_realizations': tile_source_realizations,
               'placed_source_entity_mass': sum(
                   item['placed_entity_mass'] for item in tile_source_realizations
               ),
               'q3map2_compile_measures': compile_measures,
               'q3map2_stage_returncodes': stage_codes,
               'q3map2_leak_line_mass': leak_line_mass, 'bsp_artifact_mass': bsp_artifact_mass,
               'bsp_problem_mass': len(bsp_problems), 'bsp_problems': bsp_problems,
               'negative_space_error_mass': int(negative_space_error is not None),
               'negative_space_error': negative_space_error,
               'overlay_error_mass': int(overlay_error is not None),
               'overlay_error': overlay_error, 'map_measurements': measurements,
               'source_observations': observations,
               'bridge_observations': bridge_observations,
               'region_graph': graph_metrics, 'navmesh': navmesh,
               'wall_seconds': round(time.time() - started, 3)}
    with open(os.path.join(outdir, 'fused.metrics.json'), 'w') as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write('\n')
    missing = missing_bundle_artifacts(outdir)
    residuals = release_residual(metrics, missing)
    metrics.update(residuals)
    with open(os.path.join(outdir, 'fused.metrics.json'), 'w') as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write('\n')
    archive, missing = bundle(outdir)
    print('wrote %s maps=%d joins=%d waypoints=%d links=%d q3map2=%s leak_lines=%d measurements=%d missing_artifact_mass=%d' %
          (archive, len(source_keys), len(joins), waypoint_count, link_count, stage_codes,
           leak_line_mass, len(measurements), len(missing)))
    print('release_residual_mass=%d residuals=%s' %
          (residuals['release_residual_mass'], residuals))
    return 0

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('seed', type=int)
    parser.add_argument('--maps', default='all')
    parser.add_argument('--names', default='')
    parser.add_argument('--bridges', default='auto')
    parser.add_argument('--out', default='/private/tmp/mapfuse')
    parser.add_argument('--work', default='')
    parser.add_argument('--pk3', default='~/dox/xonotic/Xonotic/data/xonotic-20230620-maps.pk3')
    parser.add_argument('--q3map2', default=MS.Q3MAP2)
    parser.add_argument('--basepath', default=os.environ.get('XON_BASEPATH', ''))
    parser.add_argument('--teams', type=int, default=8)
    parser.add_argument('--carts', type=int, default=8)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--no-vis', action='store_true')
    parser.add_argument('--no-light', action='store_true')
    args = parser.parse_args(argv)
    return finish_existing(args) if args.resume else generate(args)

if __name__ == '__main__':
    raise SystemExit(main())
