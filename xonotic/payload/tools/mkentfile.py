import struct, sys, re, math, os, json, zlib

from pathlib import Path

import numpy as np
import negspace as _NS
import navmesh as NAV
from strategy_io_schema import MAP_MEASUREMENT_SCHEMA
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from rdma.workload import WorkloadMeter

EPS = 0.25

def push_cvars():
    r, h, s = 160.0, 96.0, 30.0
    error = None
    cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cfg', 'gamemodes-payload.cfg')
    try:
        for line in open(cfg):
            m = re.match(r'\s*set\s+g_payload_(push_radius|push_height|speed)\s+(\S+)', line)
            if m:
                if m.group(1) == 'push_radius':
                    r = float(m.group(2))
                elif m.group(1) == 'push_height':
                    h = float(m.group(2))
                elif m.group(1) == 'speed':
                    s = float(m.group(2))
    except Exception as exc:
        error = '%s: %s' % (type(exc).__name__, exc)
    return r, h, s, error

PUSH_R, PUSH_H, PAYLOAD_SPEED, PUSH_CONFIGURATION_ERROR = push_cvars()
CART_RIDE = -_NS.CART_MIN[2]
SPAWN_APPROACH_SPEED = 320.0
PATH_MIN = PUSH_R + EPS
CART_ORIGIN_SEP = max(
    _NS.CART_MAX[0] - _NS.CART_MIN[0],
    _NS.CART_MAX[1] - _NS.CART_MIN[1],
)
PLAYER_SPAWN_CLASSES = {
    'info_player_deathmatch', 'info_player_start', 'info_player_survivor',
    'info_player_race', 'info_player_attacker', 'info_player_defender',
    'team_CTF_redplayer', 'team_CTF_redspawn', 'team_CTF_blueplayer',
    'team_CTF_bluespawn', 'team_redplayer', 'team_blueplayer',
}
NEUTRAL_SPAWN_CLASSES = {
    'info_player_deathmatch', 'info_player_start', 'info_player_survivor',
    'info_player_race',
}

def track_length(track):
    return sum(math.dist(track[i], track[i + 1]) for i in range(len(track) - 1))

def cart_speed_scales(tracks, approach_distances=None):
    lengths = [track_length(track) for track in tracks]
    approaches = list(approach_distances) if approach_distances is not None else [0.0] * len(lengths)
    approach_times = [None if distance is None or not math.isfinite(distance)
                      else distance / SPAWN_APPROACH_SPEED for distance in approaches]
    raw = [length / PAYLOAD_SPEED + approach for length, approach in zip(lengths, approach_times)
           if approach is not None]
    finite_approaches = [value for value in approach_times if value is not None]
    target = max(sum(raw) / max(len(raw), 1),
                 max(finite_approaches, default=0.0) + min(lengths, default=0.0) / PAYLOAD_SPEED)
    available = [None if approach is None else max(EPS, target - approach)
                 for approach in approach_times]
    scales = [1.0 if seconds is None else length / (PAYLOAD_SPEED * seconds)
              for length, seconds in zip(lengths, available)]
    traversal_times = [length / max(PAYLOAD_SPEED * scale, EPS) for length, scale in zip(lengths, scales)]
    end_to_end = [None if approach is None else approach + traversal
                  for approach, traversal in zip(approach_times, traversal_times)]
    finite_end_to_end = [value for value in end_to_end if value is not None]
    ratio = (max(finite_end_to_end) / max(min(finite_end_to_end), EPS)
             if finite_end_to_end else None)
    return lengths, scales, approach_times, traversal_times, end_to_end, ratio

def entity_value(block, key):
    match = re.search(r'"%s"\s+"([^"]*)"' % re.escape(key), block)
    return match.group(1) if match else ''

def entity_origin(block):
    value = entity_value(block, 'origin')
    return tuple(float(x) for x in value.split()) if value else None

def is_player_spawn(block):
    classname = entity_value(block, 'classname')
    return classname in PLAYER_SPAWN_CLASSES or re.fullmatch(r'info_player_team\d+', classname) is not None

def point_segment_distance(point, a, b):
    delta = tuple(b[i] - a[i] for i in range(3))
    denom = sum(value * value for value in delta)
    t = 0.0 if denom == 0 else max(0.0, min(1.0,
        sum((point[i] - a[i]) * delta[i] for i in range(3)) / denom))
    return math.dist(point, tuple(a[i] + t * delta[i] for i in range(3)))

def cart_clearance(point, tracks):
    return min((point_segment_distance(point, track[i], track[i + 1])
                for track in tracks for i in range(len(track) - 1)), default=math.inf)

def cart_origin_clearance(point, tracks):
    return min((
        math.hypot(point[0] - track[0][0], point[1] - track[0][1])
        if track and abs(point[2] - (track[0][2] + CART_RIDE)) <= PUSH_H else math.inf
        for track in tracks
    ), default=math.inf)

def spawn_occupies_cart_origin(point, track):
    if not track:
        return False
    origin = track[0]
    return (math.hypot(point[0] - origin[0], point[1] - origin[1]) <= PUSH_R
            and abs(point[2] - (origin[2] + CART_RIDE)) <= PUSH_H)

def rider_gap_counts(tracks, ns):
    counts = []
    for track in tracks:
        points = np.asarray(track, dtype=np.float64).reshape((-1, 3))
        free, _, _ = ns.segment_relations(
            points[:-1], points[1:], _NS.CART_RIDER_MIN, _NS.CART_RIDER_MAX,
        )
        counts.append(int((~free).sum()))
    return counts

def track_construction_measures(tracks, count, ns):
    gaps = rider_gap_counts(tracks, ns)
    solver = NAV.PathSolver(ns)
    motion_gaps = []
    for track in tracks:
        free, grounded, _ = solver.path_relations(track)
        motion_gaps.append(int((~(free & grounded)).sum()))
    lengths = [track_length(track) for track in tracks]
    origins = [track[0] for track in tracks if track]
    collisions = sum(
        math.dist(origins[i], origins[j]) < CART_ORIGIN_SEP
        for i in range(len(origins)) for j in range(i + 1, len(origins))
    )
    missing = max(0, count - len(tracks))
    surplus = max(0, len(tracks) - count)
    short_nodes = sum(len(track) < 2 for track in tracks)
    short_paths = sum(length < PATH_MIN for length in lengths)
    distinct = len(set(tuple(round(float(x), 1) for x in point) for point in origins))
    origin_residual = max(0, count - distinct)
    residual = missing + surplus + short_nodes + short_paths + sum(motion_gaps) + collisions + origin_residual
    separation = min((math.dist(origins[i], origins[j])
                      for i in range(len(origins)) for j in range(i + 1, len(origins))),
                     default=math.inf)
    return {
        'cart_mass': len(tracks),
        'requested_cart_mass': count,
        'missing_cart_mass': missing,
        'surplus_cart_mass': surplus,
        'path_segment_mass': sum(max(0, len(track) - 1) for track in tracks),
        'short_node_path_mass': short_nodes,
        'short_path_mass': short_paths,
        'rider_gap_segment_mass': sum(gaps),
        'distinct_origin_mass': distinct,
        'origin_identity_residual_mass': origin_residual,
        'origin_collision_pair_mass': collisions,
        'origin_separation': separation if len(origins) > 1 else None,
        'construction_residual_mass': residual,
        'rider_gap_by_cart': gaps,
        'cart_motion_gap_by_cart': motion_gaps,
        'cart_motion_gap_segment_mass': sum(motion_gaps),
    }

def entity_tracks(entities):
    blocks = [block for block in re.findall(r'\{[^{}]*\}', entities)
              if entity_value(block, 'classname') == 'plc_path']
    points = {entity_value(block, 'targetname'): entity_origin(block) for block in blocks}
    targets = {entity_value(block, 'targetname'): entity_value(block, 'target') for block in blocks}
    incoming = {target for target in targets.values() if target in points}
    starts = sorted(set(points) - incoming)
    starts.extend(name for name in sorted(points) if name not in starts)
    tracks, visited = [], set()
    for start in starts:
        if start in visited or points[start] is None:
            continue
        track, name, local = [], start, set()
        while name in points and name not in local and points[name] is not None:
            track.append(points[name])
            local.add(name)
            visited.add(name)
            name = targets.get(name, '')
        if track:
            tracks.append(track)
    return tracks

def spread_points(points, limit):
    points = sorted(set(tuple(float(x) for x in point) for point in points))
    if len(points) <= limit:
        return points
    chosen = [points[0]]
    distance = [math.dist(point, chosen[0]) for point in points]
    while len(chosen) < limit:
        index = max(range(len(points)), key=lambda i: distance[i])
        chosen.append(points[index])
        distance = [min(distance[i], math.dist(points[i], points[index])) for i in range(len(points))]
    return chosen

def localized_points(points):
    remaining = sorted(set(tuple(point) for point in points))
    if not remaining:
        return []
    ordered = [remaining.pop(0)]
    while remaining:
        index = min(range(len(remaining)), key=lambda i: math.dist(ordered[-1], remaining[i]))
        ordered.append(remaining.pop(index))
    return ordered

class SpawnAccessRelation(object):
    def __init__(self, tracks, graph):
        self.origins = [track[0] for track in tracks if track]
        self.nodes, self.adj = (graph.nodes, graph.adj) if graph is not None else ([], [])
        self.node_array = np.asarray(self.nodes, dtype=np.float64)
        self.nearest_cache = {}
        self.origin_nodes = [self.nearest(origin)[0] for origin in self.origins]
        self.origin_attachment = np.asarray(
            [self.nearest(origin)[1] for origin in self.origins], dtype=np.float64,
        )
        self.distance = {
            node: np.asarray(NAV.dijkstra(self.adj, node)[0], dtype=np.float64)
            for node in set(self.origin_nodes)
        } if self.nodes and self.adj else {}

    def nearest(self, point):
        key = tuple(float(value) for value in point)
        if key not in self.nearest_cache:
            if len(self.node_array):
                delta = self.node_array - np.asarray(key, dtype=np.float64)
                index = int(np.argmin(np.einsum('ij,ij->i', delta, delta)))
                self.nearest_cache[key] = (index, float(np.linalg.norm(delta[index])))
            else:
                self.nearest_cache[key] = (-1, 0.0)
        return self.nearest_cache[key]

    def matrix(self, points):
        points = list(points)
        if not points or not self.origins:
            return np.empty((len(points), len(self.origins)), dtype=np.float64)
        if self.distance:
            attached = [self.nearest(point) for point in points]
            nodes = [row[0] for row in attached]
            offsets = np.asarray([row[1] for row in attached], dtype=np.float64)
            matrix = np.asarray([
                [self.distance[origin][node] for origin in self.origin_nodes]
                for node in nodes
            ], dtype=np.float64)
            return matrix + offsets[:, None] + self.origin_attachment[None, :]
        point_array = np.asarray(points, dtype=np.float64)
        origin_array = np.asarray(self.origins, dtype=np.float64)
        delta = point_array[:, None, :] - origin_array[None, :, :]
        return np.sqrt(np.einsum('ijk,ijk->ij', delta, delta))

def spawn_access_matrix(points, tracks, graph, access=None):
    relation = access if access is not None else SpawnAccessRelation(tracks, graph)
    return relation.matrix(points)

def balanced_spawn_points(points, tracks, graph, limit, access=None):
    matrix = spawn_access_matrix(points, tracks, graph, access)
    finite_mask = np.isfinite(matrix)
    reachable = [i for i in range(len(points)) if finite_mask[i].any()]
    if not reachable or not matrix.shape[1]:
        return []
    size = min(len(reachable), limit)
    finite = [i for i in reachable if finite_mask[i].all()]
    if not finite:
        group = []
        uncovered = set(range(matrix.shape[1]))
        available = set(reachable)
        while available and uncovered and len(group) < size:
            index = min(available, key=lambda i: (
                -sum(finite_mask[i, column] for column in uncovered),
                sum(float(matrix[i, column]) for column in uncovered if finite_mask[i, column]),
                i,
            ))
            group.append(index)
            available.remove(index)
            uncovered -= {column for column in uncovered if finite_mask[index, column]}
        while available and len(group) < size:
            index = max(available, key=lambda i: min(
                (math.dist(points[i], points[chosen]) for chosen in group), default=math.inf,
            ))
            group.append(index)
            available.remove(index)
        return localized_points([points[i] for i in group])
    finite_rows = matrix[finite]
    row_ratio = np.max(finite_rows, axis=1) / np.maximum(np.min(finite_rows, axis=1), 1.0)
    row_sum = np.sum(finite_rows, axis=1)
    coordinates = np.asarray([points[i] for i in finite], dtype=np.float64)
    centroid = np.median(coordinates, axis=0)
    center_distance = np.sqrt(np.sum((coordinates - centroid) ** 2, axis=1))
    order = np.lexsort((np.asarray(finite), center_distance, row_sum, row_ratio))
    chosen = [finite[int(index)] for index in order[:size]]
    return localized_points([points[i] for i in chosen])

def spawn_access_metrics(points, tracks, graph, access=None):
    matrix = spawn_access_matrix(points, tracks, graph, access)
    nodes, adj = (graph.nodes, graph.adj) if graph is not None else ([], [])
    relation = 'stock_playerbot_navigation' if nodes and adj else 'straight_line_geometric'
    if not len(points) or not matrix.shape[1]:
        return {
            'relation': relation,
            'nonfinite_count': int(matrix.size),
            'cart_median_ratio': None,
            'cart_first_claim_ratio': None,
            'per_spawn_ratio_p90': None,
            'per_spawn_ratio_max': None,
            'cart_distance': [],
        }
    safe = np.where(np.isfinite(matrix), matrix, np.nan)
    cart_distance = []
    for column in range(matrix.shape[1]):
        values = matrix[:, column][np.isfinite(matrix[:, column])]
        cart_distance.append({
            'finite_spawns': len(values),
            'spawns': len(points),
            'minimum': round(float(values.min()), 3) if len(values) else None,
            'median': round(float(np.median(values)), 3) if len(values) else None,
            'p90': round(float(np.percentile(values, 90)), 3) if len(values) else None,
            'maximum': round(float(values.max()), 3) if len(values) else None,
        })
    medians = [row['median'] for row in cart_distance if row['median'] is not None]
    minimums = [row['minimum'] for row in cart_distance if row['minimum'] is not None]
    row_min = np.nanmin(safe, axis=1)
    row_max = np.nanmax(safe, axis=1)
    return {
        'relation': relation,
        'nonfinite_count': int(matrix.size - np.isfinite(matrix).sum()),
        'cart_median_ratio': round(max(medians) / max(min(medians), 1.0), 6) if medians else None,
        'cart_first_claim_ratio': round(max(minimums) / max(min(minimums), 1.0), 6) if minimums else None,
        'per_spawn_ratio_p90': round(float(np.nanpercentile(row_max / np.maximum(row_min, 1.0), 90)), 6),
        'per_spawn_ratio_max': round(float(np.nanmax(row_max / np.maximum(row_min, 1.0))), 6),
        'cart_distance': cart_distance,
    }

def spawn_overlay(entities, tracks, graph, limit=None, clearance=None, ns=None, access=None):
    clearance = PUSH_R + EPS if clearance is None else float(clearance)
    blocks = re.findall(r'\{[^{}]*\}', entities)
    originals = [entity_origin(block) for block in blocks
                 if is_player_spawn(block)]
    original_spawn_classes = [entity_value(block, 'classname') for block in blocks
                              if is_player_spawn(block)]
    nodes, adj = (graph.nodes, graph.adj) if graph is not None else ([], [])
    candidates = [point for point in originals if point]
    candidates.extend(point for index, point in enumerate(nodes) if adj[index])
    if ns is not None:
        body_width = _NS.PL_MAX[0] - _NS.PL_MIN[0]
        spawn_mins = (_NS.PL_MIN[0] - body_width, _NS.PL_MIN[1] - body_width,
                      _NS.PL_MIN[2])
        spawn_maxs = (_NS.PL_MAX[0] + body_width, _NS.PL_MAX[1] + body_width,
                      _NS.PL_MAX[2])
        spawn_lift = (abs(_NS.PL_MIN[2]),)
        standing, present, standing_measures = ns.standing_points(
            candidates, mins=spawn_mins, maxs=spawn_maxs, lift=spawn_lift,
        )
        candidates = standing[present].tolist()
        candidates = spread_points(candidates, len(candidates))
    else:
        standing_measures = None
    clear = [point for point in candidates if cart_origin_clearance(point, tracks) >= clearance]
    covered = np.isfinite(spawn_access_matrix(clear, tracks, graph, access)).any(axis=0) if clear else np.zeros(len(tracks), dtype=bool)
    all_access = spawn_access_matrix(candidates, tracks, graph, access)
    selected = list(dict.fromkeys(tuple(point) for point in clear))
    selected_set = set(selected)
    for cart in np.flatnonzero(~covered):
        reachable = [
            index for index in range(len(candidates))
            if all_access.shape[1] > cart and np.isfinite(all_access[index, cart])
            and not any(spawn_occupies_cart_origin(candidates[index], track) for track in tracks)
        ]
        if reachable:
            index = max(reachable, key=lambda value: cart_origin_clearance(candidates[value], tracks))
            point = tuple(candidates[index])
            if point not in selected_set:
                selected.append(point)
                selected_set.add(point)
    clear = [list(point) for point in selected]
    limit = max(1, len(originals), len(tracks)) if limit is None else max(1, int(limit))
    chosen = balanced_spawn_points(clear, tracks, graph, limit, access) if tracks else localized_points(
        spread_points(clear, min(len(clear), limit))
    )
    recovery_mass = 0
    if not chosen:
        recovery = list(candidates) or [point for point in originals if point]
        if not recovery and ns is not None:
            for index in np.argsort(np.prod(np.maximum(ns.hi - ns.lo, 0.0), axis=1))[::-1]:
                point = ns.standing_point((ns.lo[index] + ns.hi[index]) * 0.5)
                if point is not None:
                    recovery.append(list(point))
                    break
        chosen = localized_points(spread_points(recovery, min(len(recovery), limit)))
        recovery_mass = len(chosen)

    def keep(match):
        block = match.group(0)
        return '' if is_player_spawn(block) else block

    stripped = re.sub(r'\{[^{}]*\}', keep, entities).rstrip()
    spawned = ['\n'.join(('{', '"classname" "info_player_deathmatch"',
                            '"origin" "%.9g %.9g %.9g"' % point, '}'))
               for point in chosen]
    minimum = min((cart_origin_clearance(point, tracks) for point in chosen), default=None)
    measures = spawn_access_metrics(chosen, tracks, graph, access)
    measures['standing_point_measures'] = standing_measures
    return (stripped + '\n' + '\n'.join(spawned) + '\n', chosen, minimum, len(clear),
            measures, original_spawn_classes, recovery_mass, clearance)

def checkpoint_chain(track, count):
    points = np.asarray(track, dtype=np.float64)
    distances = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    checkpoints = np.linspace(0, distances[-1], int(count) + 1)[1:]
    samples = np.unique(np.r_[distances, checkpoints])
    positions = np.stack([np.interp(samples, distances, points[:, axis]) for axis in range(3)], axis=1)
    return positions.tolist(), np.isin(samples, checkpoints).tolist()

def emit(bsp, out, kteams, kcarts, pk3arg='', ns=None, checkpoints=4):
    mapname = os.path.splitext(os.path.basename(bsp))[0]
    meter = WorkloadMeter('xonotic-map-entity-builder', {
        'map': mapname, 'environment': 'map-entity-builder',
    })
    d = Path(bsp).read_bytes()
    off, ln = struct.unpack_from('<ii', d, 8)
    ents = d[off:off + ln].split(b'\0')[0].decode('latin-1')
    if ns is None:
        cache = bsp + '.negspace.npz'
        cache_id = cache + '.bsp.json'
        identity = {'schema': _NS.NEGSPACE_SCHEMA, 'bytes': len(d), 'crc32': zlib.crc32(d)}
        try:
            if os.path.exists(cache) and json.load(open(cache_id)) == identity:
                ns = _NS.load_saved(cache)
                print('nav: loaded negative-space cache', cache)
        except Exception as exc:
            print('nav: negative-space cache unavailable:', repr(exc))
        if ns is not None and not hasattr(ns, 'blk_H'):
            print('nav: upgrading negative-space cache with swept-hull brush index')
            ns = None
        if ns is None:
            ns = _NS.from_bsp(d, mask=_NS.MASK_PLAYERSOLID)
            _NS.save(ns, cache)
            with open(cache_id, 'w') as handle:
                json.dump(identity, handle, sort_keys=True)
                handle.write('\n')
            print('nav: wrote negative-space cache', cache)

    with meter.span('cart-construction', rows=kcarts, operations={
        'teams': kteams, 'requested_carts': kcarts,
    }):
        navigation = NAV.Navmesh.load(bsp, mapname, pk3arg, data=d)
        tracks, paths, track_stats = NAV.place_carts(
            navigation, ns, kcarts, PUSH_R, PATH_MIN, CART_ORIGIN_SEP,
        )
        if not tracks:
            raise ValueError(f'{mapname}: cart construction produced no traversable tracks; no payload overlay was written')
        graph = navigation
        track_sources = ['navigation'] * len(tracks)
        construction = track_construction_measures(tracks, kcarts, ns)
    construction_source = '+'.join(sorted(set(track_sources))) if track_sources else 'none'
    construction_source_mass = {
        source: track_sources.count(source) for source in sorted(set(track_sources))
    }

    placed_tracks = [[[p[0], p[1], p[2] - CART_RIDE] for p in track] for track in tracks]
    spawn_access_relation = SpawnAccessRelation(placed_tracks, graph)
    (entities, spawn_points, spawn_minimum, spawn_candidates, spawn_access,
     original_spawn_classes, spawn_recovery_mass, spawn_origin_clearance) = spawn_overlay(
        ents.rstrip('\0'), placed_tracks, graph, ns=ns, access=spawn_access_relation,
    )
    approach_distances = [row['median'] for row in spawn_access['cart_distance']]
    lengths, speed_scales, approach_times, traversal_times, end_to_end_times, end_to_end_ratio = cart_speed_scales(placed_tracks, approach_distances)
    extra = []
    for target in track_stats['navigation_realization']['targets']:
        extra.append('\n'.join(['{', '"classname" "plc_nav_target"',
                                '"plc_nav_id" "%d"' % target['id'],
                                '"origin" "%.6f %.6f %.6f"' % tuple(target['position']), '}']))
    named = []
    for c, source_track in enumerate(placed_tracks):
        track, checkpoint_flags = checkpoint_chain(source_track, checkpoints)
        NC = len(track)
        names = ['plc%dn%d' % (c, i) for i in range(NC)]
        named.append((names, track))
        for i, (name, p) in enumerate(zip(names, track)):
            e = ['{', '"classname" "plc_path"', '"targetname" "%s"' % name,
                 '"origin" "%.6f %.6f %.6f"' % tuple(p)]
            if i + 1 < NC:
                e.append('"target" "%s"' % names[i + 1])
            if checkpoint_flags[i]:
                e.append('"spawnflags" "1"')
            e.append('}')
            extra.append('\n'.join(e))
        extra.append('\n'.join(['{', '"classname" "func_plc_cart"',
                                '"target" "%s"' % names[0],
                                '"radius" "%.9g"' % PUSH_R,
                                '"height" "%.9g"' % PUSH_H,
                                '"plc_speed_scale" "%.9g"' % speed_scales[c], '}']))

    for t in range(kteams):
        extra.append('\n'.join(['{', '"classname" "plc_team"', '"cnt" "%d"' % t, '}']))

    for c, (names, track) in enumerate(named):
        L = sum(math.dist(track[i], track[i + 1]) for i in range(len(track) - 1))
        print('cart %d: %s -> %s length %.0f nodes %d' % (c, track[0], track[-1], L, len(track)))
    sep = min(math.dist(pa, pb) for _, ta in named[:1] for pa in ta
              for _, tb in named[1:] for pb in tb) if len(named) > 1 else None
    print('teams', kteams, 'requested_carts', kcarts, 'realized_carts', len(named),
          'min_inter_track_node_distance', sep)

    rider_gap_by_cart = construction['rider_gap_by_cart']
    rider_gap_segments = sum(rider_gap_by_cart)
    Path(out).write_text(entities + '\n'.join(extra) + '\n')
    length_ratio = max(lengths) / max(min(lengths), 1.0) if lengths else None
    traversal_time_ratio = max(traversal_times) / max(min(traversal_times), EPS) if traversal_times else None
    origin_separation = min((math.dist(placed_tracks[i][0], placed_tracks[j][0])
                             for i in range(len(placed_tracks))
                             for j in range(i + 1, len(placed_tracks))), default=None)
    spawn_cart_matrix = spawn_access_matrix(
        spawn_points, placed_tracks, graph, spawn_access_relation,
    )
    cart_path_measures = []
    for c, track in enumerate(placed_tracks):
        finite = np.isfinite(spawn_cart_matrix[:, c]) if spawn_cart_matrix.shape[1] > c else np.zeros(len(spawn_points), dtype=bool)
        distances = spawn_cart_matrix[:, c][finite] if spawn_cart_matrix.shape[1] > c else np.empty(0)
        origin_distances = [math.dist(point, track[0]) for point in spawn_points]
        path_non_degenerate = len(track) > 1 and lengths[c] > 0
        rider_continuous = rider_gap_by_cart[c] == 0
        cart_motion_supported = construction['cart_motion_gap_by_cart'][c] == 0
        spawn_reachable = bool(finite.sum()) and spawn_access['relation'] == 'stock_playerbot_navigation'
        row = {
            'cart': c,
            'path_nodes': len(track),
            'path_segments': max(0, len(track) - 1),
            'path_length': lengths[c],
            'rider_gap_segments': rider_gap_by_cart[c],
            'finite_spawn_routes': int(finite.sum()),
            'spawn_routes': len(spawn_points),
            'path_non_degenerate_mass': int(path_non_degenerate),
            'rider_continuous_mass': int(rider_continuous),
            'cart_motion_supported_mass': int(cart_motion_supported),
            'spawn_reachable_mass': int(spawn_reachable),
            'advanceable_mass': int(path_non_degenerate and cart_motion_supported and spawn_reachable),
            'spawn_route_min': float(distances.min()) if len(distances) else None,
            'spawn_route_median': float(np.median(distances)) if len(distances) else None,
            'spawn_route_max': float(distances.max()) if len(distances) else None,
            'origin_spawn_distance_min': min(origin_distances) if origin_distances else None,
            'origin_spawn_distance_median': float(np.median(origin_distances)) if origin_distances else None,
            'origin_spawn_distance_max': max(origin_distances) if origin_distances else None,
        }
        cart_path_measures.append(row)
    team_objective_measures = [
        {
            'team': team + 1,
            'controllable_cart_ids': list(range(len(placed_tracks))),
            'controllable_cart_count': len(placed_tracks),
            'advanceable_cart_ids': [
                row['cart'] for row in cart_path_measures if row['advanceable_mass']
            ],
            'advanceable_cart_count': sum(
                row['advanceable_mass'] for row in cart_path_measures
            ),
            'spawn_reachable_cart_ids': [
                row['cart'] for row in cart_path_measures if row['spawn_reachable_mass']
            ],
            'rider_continuous_cart_ids': [
                row['cart'] for row in cart_path_measures if row['rider_continuous_mass']
            ],
            'nominal_lane_cart': team % len(placed_tracks) if placed_tracks else None,
            'nominal_lane_endpoint': list(placed_tracks[team % len(placed_tracks)][-1]) if placed_tracks else None,
        }
        for team in range(kteams)
    ]
    shared_spawn_team_pairs = len(spawn_points) * kteams
    cart_route_mass = sum(row['spawn_reachable_mass'] for row in cart_path_measures)
    cart_advanceable_mass = sum(row['advanceable_mass'] for row in cart_path_measures)
    cart_rider_continuous_mass = sum(row['rider_continuous_mass'] for row in cart_path_measures)
    cart_path_non_degenerate_mass = sum(row['path_non_degenerate_mass'] for row in cart_path_measures)
    team_cart_pair_mass = kteams * len(placed_tracks)
    team_cart_advanceable_pair_mass = kteams * cart_advanceable_mass
    spawn_cart_origin_occupancy_pair_mass = sum(
        spawn_occupies_cart_origin(point, track)
        for point in spawn_points for track in placed_tracks
    )
    measurements = {
        'schema': MAP_MEASUREMENT_SCHEMA,
        'map': mapname,
        'negative_space_schema': _NS.NEGSPACE_SCHEMA,
        'compiled_collision_brush_mass': int(getattr(ns, 'compiled_brush_mass', 0)),
        'compiled_collision_patch_triangle_mass': int(
            getattr(ns, 'patch_triangle_mass', 0)
        ),
        'teams': kteams,
        'carts': kcarts,
        'realized_carts': len(placed_tracks),
        'cart_path_plan': track_stats['cart_path_plan'],
        'cart_push_radius': PUSH_R,
        'cart_push_height': PUSH_H,
        'rider_continuity_relation': 'cart_supported_rider_swept_clearance',
        'push_configuration_error_mass': int(PUSH_CONFIGURATION_ERROR is not None),
        'push_configuration_error': PUSH_CONFIGURATION_ERROR,
        'team_declarations': kteams if placed_tracks else 0,
        'generic_spawns': len(spawn_points),
        'original_player_spawn_mass': len(original_spawn_classes),
        'original_team_labeled_spawn_mass': sum(
            name not in NEUTRAL_SPAWN_CLASSES for name in original_spawn_classes
        ),
        'residual_team_labeled_spawn_mass': 0,
        'spawn_candidates_clear': spawn_candidates,
        'spawn_hull_clear': len(spawn_points),
        'spawn_cart_origin_clearance': spawn_origin_clearance,
        'spawn_cart_origin_clearance_mass': sum(
            cart_origin_clearance(point, placed_tracks) >= spawn_origin_clearance for point in spawn_points
        ),
        'spawn_cart_origin_clearance_residual_mass': sum(
            cart_origin_clearance(point, placed_tracks) < spawn_origin_clearance for point in spawn_points
        ),
        'spawn_recovery_mass': spawn_recovery_mass,
        'spawn_cart_origin_clearance_min': spawn_minimum,
        'spawn_path_clearance_min': min(
            (cart_clearance(point, placed_tracks) for point in spawn_points), default=None,
        ),
        'shared_spawn_team_pairs': shared_spawn_team_pairs,
        'spawn_team_pairs': shared_spawn_team_pairs,
        'spawn_team_access_ratio': shared_spawn_team_pairs / max(1, len(spawn_points) * kteams),
        'spawn_cart_origin_occupancy_pair_mass': spawn_cart_origin_occupancy_pair_mass,
        'spawn_access_relation': spawn_access['relation'],
        'spawn_standing_point_measures': spawn_access['standing_point_measures'],
        'spawn_access_origin_node_mass': len(set(spawn_access_relation.origin_nodes)),
        'spawn_access_distance_row_mass': len(spawn_access_relation.distance),
        'spawn_access_nearest_cache_mass': len(spawn_access_relation.nearest_cache),
        'spawn_access_origin_attachment_distance': spawn_access_relation.origin_attachment.tolist(),
        'spawn_cart_nonfinite_distances': spawn_access['nonfinite_count'],
        'spawn_cart_median_ratio': spawn_access['cart_median_ratio'],
        'spawn_cart_first_claim_ratio': spawn_access['cart_first_claim_ratio'],
        'spawn_cart_per_spawn_ratio_p90': spawn_access['per_spawn_ratio_p90'],
        'spawn_cart_per_spawn_ratio_max': spawn_access['per_spawn_ratio_max'],
        'spawn_cart_distance': spawn_access['cart_distance'],
        'path_lengths': lengths,
        'path_length_ratio': length_ratio,
        'cart_speed_scales': speed_scales,
        'spawn_approach_speed': SPAWN_APPROACH_SPEED,
        'spawn_cart_approach_times': approach_times,
        'spawn_cart_approach_missing_mass': sum(value is None for value in approach_times),
        'nominal_traversal_times': traversal_times,
        'nominal_traversal_time_ratio': traversal_time_ratio,
        'nominal_end_to_end_times': end_to_end_times,
        'nominal_end_to_end_time_ratio': end_to_end_ratio,
        'path_overlap_max': track_stats['overlap_max'],
        'headon_flow': track_stats['headon'],
        'push_zone_counterflow': track_stats.get('push_counterflow'),
        'flow_alignment': track_stats['flow_alignment'],
        'distinct_origins': len(set(tuple(round(float(x), 1) for x in track[0]) for track in placed_tracks)),
        'origin_separation': origin_separation,
        'rider_gap_segments': rider_gap_segments,
        'rider_continuous_cart_mass': sum(value == 0 for value in rider_gap_by_cart),
        'cart_construction_source': construction_source,
        'cart_construction_source_mass': construction_source_mass,
        'cart_construction_measures': construction,
        'team_cart_control_pair_mass': team_cart_pair_mass,
        'team_cart_advanceable_pair_mass': team_cart_advanceable_pair_mass,
        'team_cart_nonadvanceable_pair_mass': team_cart_pair_mass - team_cart_advanceable_pair_mass,
        'team_cart_spawn_unreachable_pair_mass': kteams * (len(placed_tracks) - cart_route_mass),
        'team_cart_rider_discontinuous_pair_mass': kteams * (len(placed_tracks) - cart_rider_continuous_mass),
        'team_cart_path_degenerate_pair_mass': kteams * (len(placed_tracks) - cart_path_non_degenerate_mass),
        'cart_path_measures': cart_path_measures,
        'team_objective_measures': team_objective_measures,
    }
    measurements.update(track_stats)
    measurements['checkpoints_per_lane'] = int(checkpoints)
    measurements['checkpoint_distances'] = [np.linspace(0, length, int(checkpoints) + 1)[1:].tolist() for length in lengths]
    with open(out + '.measurements.json', 'w') as handle:
        json.dump(measurements, handle, indent=2, sort_keys=True)
        handle.write('\n')
    meter.close({
        'teams': kteams,
        'carts': kcarts,
        'realized_carts': len(placed_tracks),
        'cart_construction_measures': construction,
        'rider_gap_segments': rider_gap_segments,
        'spawn_team_pairs': shared_spawn_team_pairs,
        'team_cart_control_pair_mass': team_cart_pair_mass,
        'team_cart_advanceable_pair_mass': team_cart_advanceable_pair_mass,
    })
    print('wrote', out)

if __name__ == '__main__':
    bsp, out = sys.argv[1], sys.argv[2]

    kteams = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    kcarts = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    pk3arg = sys.argv[5] if len(sys.argv) > 5 else ''
    checkpoints = int(sys.argv[6]) if len(sys.argv) > 6 else 4
    emit(bsp, out, kteams, kcarts, pk3arg, checkpoints=checkpoints)
