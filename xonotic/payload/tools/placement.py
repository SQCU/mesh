import struct, sys, os, math, subprocess, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mkentfile as M
import negspace as NS
from negspace import box_H

MARGIN, CORW, CORH, WALL, FLOORTHK = 896.0, 288.0, 224.0, 32.0, 32.0
PORTAL_QUANTUM = 4.0
DOOR_W = math.ceil((max(NS.CART_RIDER_MAX[0] - NS.CART_RIDER_MIN[0],
                        NS.CART_RIDER_MAX[1] - NS.CART_RIDER_MIN[1]) + NS.EPS)
                   / PORTAL_QUANTUM) * PORTAL_QUANTUM
DOOR_H = math.ceil((NS.CART_RIDER_MAX[2] - NS.CART_RIDER_MIN[2] + NS.EPS)
                   / PORTAL_QUANTUM) * PORTAL_QUANTUM
DOOR_SILL = math.floor(NS.CART_RIDER_MIN[2] / PORTAL_QUANTUM) * PORTAL_QUANTUM
SITE_DIRS = ((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0),
             (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))
WAYPOINT_GENERATED = 1 << 23
WAYPOINT_SERIALIZATION_QUANTUM = 0.1

CORR_SOFT, CORR_PEN = 2600.0, 3.0
SPAN_SEG, SPAN_CLEAR = 1600.0, 384.0
CORW_PROM, PROM_LIGHT = 448.0, 700.0
LSZ = (0, 72, 16, 36, 48, 4, 4, 40, 12, 8, 44, 4, 72, 104, 49152, 8, 1)
TRIGTEX, EMPTYTEX = ('textures/common/trigger', 0, 0x40000000), ('textures/common/caulk', 0, 0)

def vadd(a, b):
    return [a[i] + b[i] for i in range(3)]

def vsub(a, b):
    return [a[i] - b[i] for i in range(3)]

def vdot(a, b):
    return sum(a[i] * b[i] for i in range(3))

def vscale(a, s):
    return [a[i] * s for i in range(3)]

def vnorm(a):
    L = math.sqrt(vdot(a, a)) or 1.0
    return [x / L for x in a]

def portal_coordinate(value, lo, hi):
    lower = math.ceil(lo / PORTAL_QUANTUM) * PORTAL_QUANTUM
    upper = math.floor(hi / PORTAL_QUANTUM) * PORTAL_QUANTUM
    return min(max(round(value / PORTAL_QUANTUM) * PORTAL_QUANTUM, lower), upper)

def vcross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]

def fnum(x):
    s = '%.1f' % x
    return s[:-2] if s.endswith('.0') else s

def vstr(v):
    return "'%s %s %s'" % (fnum(v[0]), fnum(v[1]), fnum(v[2]))

def pk3_read(pk3, path):
    r = subprocess.run(['unzip', '-p', pk3, path], capture_output=True)
    return r.stdout if r.returncode == 0 else b''

def navigable_names(pk3):
    out = subprocess.run(['unzip', '-l', pk3], capture_output=True, text=True).stdout
    names = set()
    for l in out.splitlines():
        f = l.split()[-1] if l.split() else ''
        if f.startswith('maps/') and f.endswith('.waypoints') and not f.endswith('.race.waypoints'):
            names.add(os.path.basename(f)[:-len('.waypoints')])
    return sorted(names)

def parse_waypoint_relation(wptext, cachetext):
    wptriples = []
    lines = [line.rstrip('\r') for line in wptext.splitlines()]
    index = 0
    while index < len(lines) and lines[index].startswith('//'):
        index += 1
    while index + 3 <= len(lines):
        try:
            left = [float(value) for value in lines[index].strip().strip("'").split()]
            right = [float(value) for value in lines[index + 1].strip().strip("'").split()]
            flags = int(float(lines[index + 2].strip()))
        except (ValueError, IndexError):
            break
        wptriples.append((left, right, flags))
        index += 3
    cachelinks = []
    for line in cachetext.splitlines():
        line = line.strip()
        if not line or line.startswith('//'):
            continue
        parts = line.split('*')
        if len(parts) == 2 and parts[0].strip():
            left = [float(value) for value in parts[0].strip().strip("'").split()]
            right = [float(value) for value in parts[1].strip().strip("'").split()]
            cachelinks.append((left, right))
    return wptriples, cachelinks

def reconcile_waypoint_relation(ns, source_wptriples, cachelinks):
    key = lambda point: tuple(round(value, 1) for value in point)
    waypoint_mins = (NS.PL_MIN[0] - 1.0, NS.PL_MIN[1] - 1.0, NS.PL_MIN[2])
    waypoint_maxs = (NS.PL_MAX[0] + 1.0, NS.PL_MAX[1] + 1.0, NS.PL_MAX[2])
    source_definitions = {
        key([(m1[axis] + m2[axis]) / 2.0 for axis in range(3)])
        for m1, m2, _ in source_wptriples
    }
    implicit_definitions = {
        key(point) for edge in cachelinks for point in edge
        if key(point) not in source_definitions
    }
    coordinate_map = {}
    displacement = {}
    fixed_definitions = {}
    projection_sources = {}
    for m1, m2, _ in source_wptriples:
        center = [(m1[axis] + m2[axis]) / 2.0 for axis in range(3)]
        identity = key(center)
        if m1 == m2:
            projection_sources.setdefault(identity, center)
        else:
            fixed_definitions[identity] = center
    for identity in implicit_definitions:
        projection_sources.setdefault(identity, identity)
    projection_sources = {
        identity: point for identity, point in projection_sources.items()
        if identity not in fixed_definitions
    }
    coordinate_map.update(fixed_definitions)
    for identity in fixed_definitions:
        displacement[identity] = 0.0
    identities = sorted(projection_sources)
    points = np.asarray([projection_sources[identity] for identity in identities], dtype=np.float64)
    if ns is None:
        projected = points.copy()
        distances = np.zeros(len(points), dtype=np.float64)
        projection_measures = {
            'input_point_mass': len(points),
            'input_penetration_point_mass': 0,
            'input_penetration_pair_mass': 0,
            'projection_sweep_mass': 0,
            'candidate_pair_mass': 0,
            'plane_evaluation_mass': 0,
            'directional_null_pair_mass': 0,
            'world_boundary_reconciliation_mass': 0,
            'residual_penetration_point_mass': 0,
        }
    else:
        projected, distances, projection_measures = ns.project_many(
            points, waypoint_mins, waypoint_maxs,
            tolerance=WAYPOINT_SERIALIZATION_QUANTUM,
        )
    for index, identity in enumerate(identities):
        coordinate_map[identity] = [
            round(value / WAYPOINT_SERIALIZATION_QUANTUM)
            * WAYPOINT_SERIALIZATION_QUANTUM for value in projected[index]
        ]
        displacement[identity] = float(distances[index])

    def reconcile(point):
        return coordinate_map[key(point)]

    wptriples = []
    for m1, m2, flags in source_wptriples:
        center = [(m1[axis] + m2[axis]) / 2.0 for axis in range(3)]
        if m1 == m2:
            realized = reconcile(center)
            wptriples.append((realized, realized, flags))
        else:
            wptriples.append((m1, m2, flags))
    for identity in sorted(implicit_definitions):
        realized = reconcile(identity)
        wptriples.append((realized, realized, WAYPOINT_GENERATED))
    realized_cachelinks = [(reconcile(a), reconcile(b)) for a, b in cachelinks]
    wpdefinitions = {
        key([(m1[axis] + m2[axis]) / 2.0 for axis in range(3)])
        for m1, m2, _ in wptriples
    }
    measures = {
        'waypoint_outside_negative_space_mass': projection_measures[
            'input_penetration_point_mass'
        ],
        'waypoint_projection_unresolved_mass': projection_measures[
            'residual_penetration_point_mass'
        ],
        'waypoint_displaced_mass': sum(value > 0.0 for value in displacement.values()),
        'waypoint_displacement_integral': sum(displacement.values()),
        'waypoint_displacement_square_integral': sum(
            value * value for value in displacement.values()
        ),
        'waypoint_displacement_maximum': max(displacement.values(), default=0.0),
        'cache_endpoint_outside_definition_mass': len(implicit_definitions),
        'cache_link_outside_definition_mass': 0,
        'waypoint_projection_measures': projection_measures,
    }
    return wptriples, realized_cachelinks, wpdefinitions, measures

def realize_waypoint_files(path, ns):
    with open(path + '.waypoints', encoding='latin-1') as handle:
        wptext = handle.read()
    with open(path + '.waypoints.cache', encoding='latin-1') as handle:
        cachetext = handle.read()
    wptriples, cachelinks = parse_waypoint_relation(wptext, cachetext)
    wptriples, cachelinks, _, measures = reconcile_waypoint_relation(
        ns, wptriples, cachelinks,
    )
    stamp = time.strftime('%Y-%m-%d %H:%M:%S')
    rows = ['//WAYPOINT_VERSION 1.04', '//WAYPOINT_SYMMETRY 0', '//WAYPOINT_TIME ' + stamp]
    links = ['//WAYPOINT_VERSION 1.04', '//WAYPOINT_TIME ' + stamp]
    for left, right, flags in wptriples:
        rows.extend((vstr(left), vstr(right), str(int(flags))))
    links.extend(vstr(left) + '*' + vstr(right) for left, right in cachelinks)
    with open(path + '.waypoints', 'w') as handle:
        handle.write('\n'.join(rows) + '\n')
    with open(path + '.waypoints.cache', 'w') as handle:
        handle.write('\n'.join(links) + '\n')
    return len(wptriples), len(cachelinks), measures

class Src:
    def __init__(self, name, data, wptext, cachetext, with_ns=True):
        self.name, self.data = name, data
        L = lambda i: struct.unpack_from('<ii', data, 8 + i * 8)
        raw = {}
        for i in range(17):
            o, n = L(i)
            raw[i] = data[o:o + n]
        self.ents = raw[0].split(b'\0')[0].decode('latin-1')
        self.textures = [(raw[1][i:i + 64].split(b'\0')[0].decode('latin-1'),) +
                         struct.unpack_from('<2i', raw[1], i + 64) for i in range(0, len(raw[1]), 72)]
        self.planes = [list(struct.unpack_from('<4f', raw[2], i)) for i in range(0, len(raw[2]), 16)]
        self.nodes = [list(struct.unpack_from('<9i', raw[3], i)) for i in range(0, len(raw[3]), 36)]
        self.leafs = [list(struct.unpack_from('<12i', raw[4], i)) for i in range(0, len(raw[4]), 48)]
        self.leaffaces = list(struct.unpack('<%di' % (len(raw[5]) // 4), raw[5]))
        self.leafbrushes = list(struct.unpack('<%di' % (len(raw[6]) // 4), raw[6]))
        self.models = [list(struct.unpack_from('<6f4i', raw[7], i)) for i in range(0, len(raw[7]), 40)]
        self.brushes = [list(struct.unpack_from('<3i', raw[8], i)) for i in range(0, len(raw[8]), 12)]
        self.sides = [list(struct.unpack_from('<2i', raw[9], i)) for i in range(0, len(raw[9]), 8)]
        self.verts = [list(struct.unpack_from('<10f4B', raw[10], i)) for i in range(0, len(raw[10]), 44)]
        self.mesh = raw[11]
        self.effects = [(raw[12][i:i + 64], ) + struct.unpack_from('<2i', raw[12], i + 64)
                        for i in range(0, len(raw[12]), 72)]
        self.faces = [(list(struct.unpack_from('<12i', raw[13], i)), raw[13][i + 48:i + 104])
                      for i in range(0, len(raw[13]), 104)]
        self.lightmaps = raw[14]
        self.bounds = (self.models[0][0:3], self.models[0][3:6])
        self.ns = NS.from_bsp(data, mask=NS.MASK_PLAYERSOLID) if with_ns else None
        source_wptriples, cachelinks = parse_waypoint_relation(wptext, cachetext)
        self.wptriples, self.cachelinks, self.wpdefinitions, measures = (
            reconcile_waypoint_relation(self.ns, source_wptriples, cachelinks)
        )
        for name, value in measures.items():
            setattr(self, name, value)
        closed_cache = '\n'.join(vstr(a) + '*' + vstr(b) for a, b in self.cachelinks)
        self.navnodes, self.navadj = M.parse_cache(closed_cache)
        self.wpset = {tuple(round(x, 1) for x in m1) for m1, m2, fl in self.wptriples
                      if m1 == m2 and not fl & M.WPF_BAD}

        self.solidtex = [t[2] & 1 == 1 for t in self.textures]
        self.cliptex = [bool(t[2] & 0x430000) for t in self.textures]
        self._ebrush = None

    def edit_index(self):
        if self._ebrush is not None:
            return self._ebrush
        lo0 = np.array(self.bounds[0], dtype=np.float64) - 4096.0
        hi0 = np.array(self.bounds[1], dtype=np.float64) + 4096.0
        grid = {}
        ntex = len(self.textures)
        for bi, (fs, ns2, tx) in enumerate(self.brushes):
            if tx < 0 or tx >= ntex or not (self.solidtex[tx] or self.cliptex[tx]):
                continue
            if ns2 <= 0 or fs < 0 or fs + ns2 > len(self.sides):
                continue
            H = np.array([self.planes[self.sides[k][0]] for k in range(fs, fs + ns2)],
                         dtype=np.float64)
            blo, bhi = NS.bounds_of(H, lo0, hi0)
            for cx in range(int(blo[0] // 1024), int(bhi[0] // 1024) + 1):
                for cy in range(int(blo[1] // 1024), int(bhi[1] // 1024) + 1):
                    grid.setdefault((cx, cy), []).append((bi, list(blo), list(bhi)))
        self._ebrush = grid
        return grid

def solid_runs(ns, p0, d, maxt=1100.0):
    p1 = [p0[i] + d[i] * maxt for i in range(3)]
    iv = ns.segment_intervals(p0, p1)
    runs = []
    t = 0.0
    for s0, s1 in iv:
        if s0 > t + 1e-9:
            runs.append((t * maxt, s0 * maxt))
        t = max(t, s1)
    if t < 1.0 - 1e-9:
        runs.append((t * maxt, maxt))
    return runs

def portal_support(ns, node, d, t_in, t_out):
    axis = 0 if abs(d[0]) > 0.5 else 1
    other = 1 - axis
    sign = 1.0 if d[axis] > 0 else -1.0
    axial = sorted((node[axis] + sign * (t_in + 1.0),
                    node[axis] + sign * (t_out - 1.0)))
    half_width = DOOR_W / 2.0
    outside = half_width + WALL
    clearance = WALL / 4.0
    floor = node[2] + DOOR_SILL
    ceiling = floor + DOOR_H
    spans = (
        (node[other] - outside, node[other] - half_width - clearance,
         floor - WALL, ceiling + WALL),
        (node[other] + half_width + clearance, node[other] + outside,
         floor - WALL, ceiling + WALL),
        (node[other] - outside, node[other] + outside,
         floor - WALL, floor - clearance),
        (node[other] - outside, node[other] + outside,
         ceiling + clearance, ceiling + WALL),
    )
    residual = 0
    solids = 0
    for lateral_lo, lateral_hi, zlo, zhi in spans:
        lo = [0.0, 0.0, zlo]
        hi = [0.0, 0.0, zhi]
        lo[axis], hi[axis] = axial
        lo[other], hi[other] = lateral_lo, lateral_hi
        measure = ns.solid_incidence(box_H(lo, hi), np.asarray(lo), np.asarray(hi))
        residual += 1 - measure['incidence_mass']
        solids += measure['source_solid_candidate_mass']
    return {'support_domain_atom_mass': len(spans),
            'support_residual_atom_mass': residual,
            'support_source_solid_candidate_mass': solids}

def map_sites(src, maxsites=12, minsep=1024.0):
    got = getattr(src, '_sites', None)
    if got is not None:
        return got
    ns = src.ns
    comp = [i for i in M.largest_component(src.navadj)
            if tuple(round(x, 1) for x in src.navnodes[i]) in src.wpset]
    if not comp:
        comp = M.largest_component(src.navadj)
    cand = []
    for d in SITE_DIRS:
        axis = 0 if abs(d[0]) > 0.5 else 1
        U = 1 - axis
        pr = sorted(comp, key=lambda i: -(src.navnodes[i][0] * d[0] + src.navnodes[i][1] * d[1]))
        for i in pr:
            node = [float(x) for x in src.navnodes[i]]
            eye = [node[0], node[1], node[2] + DOOR_SILL + DOOR_H * 0.5]
            runs = solid_runs(ns, eye, d)
            if not runs:
                continue
            t_in, t_out = runs[0]
            if t_in > 640.0 or t_in < 24.0:
                continue
            thick = t_out - t_in
            if thick > 384.0 or thick < 8.0:
                continue
            if len(runs) > 1 and runs[1][0] - t_out < 224.0:
                continue

            zlo = node[2] + DOOR_SILL
            zhi = zlo + DOOR_H
            lo = [0.0, 0.0, zlo]
            hi = [0.0, 0.0, zhi]
            lo[U] = node[U] - DOOR_W / 2.0
            hi[U] = node[U] + DOOR_W / 2.0
            leading = (NS.CART_RIDER_MAX[axis] if d[axis] > 0
                       else -NS.CART_RIDER_MIN[axis])
            approach = [node[value] + d[value] * max(0.0, t_in - leading - 3.0)
                        for value in range(3)]
            if (not ns.fits(node, NS.CART_RIDER_MIN, NS.CART_RIDER_MAX)
                    or not ns.segment_free(
                        node, approach, NS.CART_RIDER_MIN, NS.CART_RIDER_MAX)):
                continue
            exit_t = ((src.bounds[1][axis] - node[axis]) * d[axis]
                      if d[axis] > 0 else (src.bounds[0][axis] - node[axis]) * d[axis])
            if exit_t < t_out:
                continue
            if exit_t - 1.0 >= t_out + 5.0:
                f0 = node[axis] + d[axis] * (t_out + 3.0)
                f1 = node[axis] + d[axis] * (exit_t - 1.0)
                flo, fhi = list(lo), list(hi)
                flo[axis], fhi[axis] = min(f0, f1), max(f0, f1)
                if not ns.covered(box_H(flo, fhi), flo, fhi):
                    continue
            support = portal_support(ns, node, d, t_in, t_out)
            if support['support_residual_atom_mass']:
                continue
            cell = -1
            for back in (2.0, 8.0, 20.0, 40.0):
                q = [eye[0] + d[0] * (t_in - back), eye[1] + d[1] * (t_in - back), eye[2]]
                cell = ns.cell_at(q)
                if cell >= 0:
                    break
            if cell < 0:
                continue
            narrow = ns.clearance(node, cap=576.0, mins=NS.PL_MIN, maxs=NS.PL_MAX) < 288.0
            deg = len(src.navadj[i])
            cont = narrow or deg <= 2
            score = ((1.5 if narrow else 0.0) + (0.5 if deg <= 3 else 0.0) +
                     max(0.0, (384.0 - thick) / 384.0) +
                     max(0.0, (640.0 - t_in) / 640.0))
            cand.append({'p': node, 'dir': list(d), 't_in': t_in, 't_out': t_out,
                         't_exit': exit_t,
                         'thick': thick, 'deg': deg, 'narrow': narrow,
                         'cell': cell,
                         'node': i, 'kind': 'continue' if cont else 'newcut',
                         'score': round(score, 3), **support})
    cand.sort(key=lambda s2: -s2['score'])
    sites = []
    for s2 in cand:
        if all(math.dist(s2['p'], t['p']) >= minsep or
               vdot(s2['dir'], t['dir']) < 0.3 and math.dist(s2['p'], t['p']) >= minsep / 2
               for t in sites):
            sites.append(s2)
        if len(sites) >= maxsites:
            break
    src._sites = sites
    return sites

def walk_nodes(src):
    got = getattr(src, '_walk', None)
    if got is not None:
        return got
    comp = [i for i in M.largest_component(src.navadj)
            if tuple(round(x, 1) for x in src.navnodes[i]) in src.wpset]
    if not comp:
        comp = M.largest_component(src.navadj)
    if not comp:
        comp = list(range(len(src.navnodes)))
    P = [list(src.navnodes[i]) for i in comp]
    src._walk = np.asarray(P, dtype=float)
    return src._walk

def walk_extent(src):
    got = getattr(src, '_walkext', None)
    if got is not None:
        return got
    W = walk_nodes(src)
    if len(W) == 0:
        got = (list(src.bounds[0]), list(src.bounds[1]))
    else:
        got = (list(W.min(0)), list(W.max(0)))
    src._walkext = got
    return got

def pack_offsets(srcs, cells, cols, rows, levels):
    j = len(srcs)
    hull = [[s.bounds[1][a] - s.bounds[0][a] for a in range(3)] for s in srcs]
    colw = [MARGIN + max([hull[m][0] for m in range(j) if cells[m][0] == c] or [0]) for c in range(cols)]
    rowh = [MARGIN + max([hull[m][1] for m in range(j) if cells[m][1] == r] or [0]) for r in range(rows)]
    levh = [MARGIN + max([hull[m][2] for m in range(j) if cells[m][2] == z] or [0]) for z in range(levels)]
    xed, yed, zed = [-sum(colw) / 2.0], [-sum(rowh) / 2.0], [-sum(levh) / 2.0]
    for w in colw:
        xed.append(xed[-1] + w)
    for h in rowh:
        yed.append(yed[-1] + h)
    for h in levh:
        zed.append(zed[-1] + h)
    offsets, slack = [], []
    for m, s in enumerate(srcs):
        c, r, z = cells[m]
        wlo, whi = walk_extent(s)
        W = walk_nodes(s)
        medz = float(np.median(W[:, 2])) if len(W) else (s.bounds[0][2] + s.bounds[1][2]) / 2
        off = [(xed[c] + xed[c + 1]) / 2.0 - (wlo[0] + whi[0]) / 2.0,
               (yed[r] + yed[r + 1]) / 2.0 - (wlo[1] + whi[1]) / 2.0,
               (zed[z] + zed[z + 1]) / 2.0 - medz]
        sl = []
        for a, ed, ci in ((0, xed, c), (1, yed, r), (2, zed, z)):
            lo = ed[ci] + 32.0 - s.bounds[0][a]
            hi = ed[ci + 1] - 32.0 - s.bounds[1][a]
            if lo > hi:
                lo = hi = (ed[ci] + ed[ci + 1]) / 2.0 - (s.bounds[0][a] + s.bounds[1][a]) / 2.0
            off[a] = portal_coordinate(off[a], lo, hi)
            sl.append((lo, hi))
        offsets.append(off)
        slack.append(sl)
    return offsets, (xed, yed, zed), slack

def split_tree(items, eds):
    if len(items) == 1:
        return ('leaf', items[0][3])
    best = None
    for axis in (2, 1, 0):
        vals = sorted({i[axis] for i in items})
        if len(vals) < 2:
            continue
        mid = vals[len(vals) // 2]
        lo = [i for i in items if i[axis] < mid]
        hi = [i for i in items if i[axis] >= mid]
        if lo and hi:
            best = (axis, mid, lo, hi)
            break
    if best is None:
        return ('leaf', items[0][3])
    axis, mid, lo, hi = best
    return ('node', axis, eds[axis][mid], split_tree(lo, eds), split_tree(hi, eds))

def check_bsp(d):
    probs = []
    L = lambda i: struct.unpack_from('<ii', d, 8 + i * 8)
    cnt = {}
    for i in range(1, 17):
        o, n = L(i)
        if o + n > len(d):
            probs.append('lump %d out of file' % i)
        if LSZ[i] > 1 and n % LSZ[i]:
            probs.append('lump %d funny size %d' % (i, n))
        cnt[i] = n // LSZ[i] if LSZ[i] else 0
    get = lambda i, k, fmt, sz: struct.unpack_from(fmt, d, L(i)[0] + k * sz)
    for k in range(cnt[9]):
        pi, ti = get(9, k, '<2i', 8)
        if not (0 <= pi < cnt[2]) or not (0 <= ti < cnt[1]):
            probs.append('brushside %d bad refs %d %d' % (k, pi, ti))
    for k in range(cnt[8]):
        fs, ns, tx = get(8, k, '<3i', 12)
        if fs < 0 or fs + ns > cnt[9] or not (0 <= tx < cnt[1]):
            probs.append('brush %d bad range' % k)
    nverts, nelems = cnt[10], cnt[11]
    for k in range(cnt[13]):
        h = get(13, k, '<12i', 104)
        if not (0 <= h[0] < cnt[1]):
            probs.append('face %d bad tex' % k)
        if h[1] >= cnt[12]:
            probs.append('face %d bad effect' % k)
        if h[3] < 0 or h[3] + h[4] > nverts:
            probs.append('face %d bad verts' % k)
        if h[5] < 0 or h[5] + h[6] > nelems:
            probs.append('face %d bad elems' % k)
        if h[7] >= cnt[14]:
            probs.append('face %d bad lightmap' % k)
    for k in range(cnt[7]):
        mm = get(7, k, '<6f4i', 40)
        if mm[6] < 0 or mm[6] + mm[7] > cnt[13] or mm[8] < 0 or mm[8] + mm[9] > cnt[8]:
            probs.append('model %d bad range' % k)
    for k in range(cnt[5]):
        v = get(5, k, '<i', 4)[0]
        if not (0 <= v < cnt[13]):
            probs.append('leafface %d bad' % k)
    for k in range(cnt[6]):
        v = get(6, k, '<i', 4)[0]
        if not (0 <= v < cnt[8]):
            probs.append('leafbrush %d bad' % k)
    for k in range(cnt[4]):
        lf = get(4, k, '<12i', 48)
        if lf[8] < 0 or lf[8] + lf[9] > cnt[5] or lf[10] < 0 or lf[10] + lf[11] > cnt[6]:
            probs.append('leaf %d bad range' % k)
    if cnt[3] == 0:
        probs.append('no nodes')
    for k in range(cnt[3]):
        nd = get(3, k, '<9i', 36)
        if not (0 <= nd[0] < cnt[2]):
            probs.append('node %d bad plane' % k)
        for ci in (1, 2):
            c = nd[ci]
            if c >= cnt[3] or (c < 0 and -1 - c >= cnt[4]):
                probs.append('node %d bad child %d' % (k, c))
    return probs

DIRNAME = {(1, 0): 'e', (-1, 0): 'w', (0, 1): 'n', (0, -1): 's'}

def load_src(n, outdir, pk3, with_ns=True, quiet=False):
    if '/' in n:
        base = os.path.expanduser(n)
        n = os.path.basename(base)
        data = open(base + '.bsp', 'rb').read()
        wp = open(base + '.waypoints', encoding='latin-1').read()
        cache = open(base + '.waypoints.cache', encoding='latin-1').read()
    else:
        data = pk3_read(pk3, 'maps/%s.bsp' % n)
        wp = pk3_read(pk3, 'maps/%s.waypoints' % n).decode('latin-1')
        cache = M.load_cache(n, os.path.join(outdir, n + '.bsp'), pk3)[0]
    src = Src(n, data, wp, cache, with_ns=with_ns)
    if not quiet:
        print('src %s: bounds %s %s models=%d faces=%d brushes=%d wp=%d links=%d' %
              (n, [round(x) for x in src.bounds[0]], [round(x) for x in src.bounds[1]],
               len(src.models), len(src.faces), len(src.brushes),
               len(src.wptriples), len(src.cachelinks)))
    return src

def region_graph_solve(j, edges_ab):
    adj = [[] for _ in range(j)]
    for ei, (a, b) in enumerate(edges_ab):
        adj[a].append((b, ei))
        adj[b].append((a, ei))
    disc = [-1] * j
    low = [0] * j
    par = [-1] * j
    arts, cutedges, comps = set(), [], []
    timer = 0
    for s0 in range(j):
        if disc[s0] != -1:
            continue
        comp = []
        stack = [(s0, iter(adj[s0]), -1)]
        disc[s0] = low[s0] = timer
        timer += 1
        comp.append(s0)
        rootkids = 0
        while stack:
            u, it, pe = stack[-1]
            advanced = False
            for v, ei in it:
                if ei == pe:
                    continue
                if disc[v] == -1:
                    disc[v] = low[v] = timer
                    timer += 1
                    comp.append(v)
                    par[v] = u
                    if u == s0:
                        rootkids += 1
                    stack.append((v, iter(adj[v]), ei))
                    advanced = True
                    break
                low[u] = min(low[u], disc[v])
            if advanced:
                continue
            stack.pop()
            if stack:
                pu = stack[-1][0]
                low[pu] = min(low[pu], low[u])
                if low[u] > disc[pu]:
                    cutedges.append(pe)
                if pu != s0 and low[u] >= disc[pu]:
                    arts.add(pu)
        if rootkids > 1:
            arts.add(s0)
        comps.append(sorted(comp))

    big = max(comps, key=len) if comps else []
    bigset = set(big)
    diam, ecc = 0, {}
    from collections import deque
    for s0 in big:
        d = {s0: 0}
        q = deque([s0])
        while q:
            u = q.popleft()
            for v, _ in adj[u]:
                if v not in d and v in bigset:
                    d[v] = d[u] + 1
                    q.append(v)
        ecc[s0] = max(d.values())
        diam = max(diam, ecc[s0])
    return dict(components=comps, articulation=sorted(arts), cutedges=sorted(cutedges),
                degree=[len(adj[i]) for i in range(j)], hop_diameter=diam,
                eccentricity=ecc, adj=adj)

def navmesh_solve(nodes2, dadj, region, key, j, reps):
    import heapq
    N = len(nodes2)
    W = [[(v, math.dist(nodes2[u], nodes2[v])) for v in dadj[u]] for u in range(N)]

    def dij(src):
        dist = [float('inf')] * N
        dist[src] = 0.0
        pq = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u] + 1e-9:
                continue
            for v, w in W[u]:
                nd = d + w
                if nd < dist[v] - 1e-9:
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        return dist

    D = {}
    cover = {}
    for m, r in reps.items():
        if r is None:
            continue
        dist = dij(r)
        reach = [i for i in range(N) if dist[i] < float('inf')]
        cover[m] = len(reach)
        for m2, r2 in reps.items():
            if r2 is None or m2 == m:
                continue
            D[(m, m2)] = dist[r2] if dist[r2] < float('inf') else None
    fin = [v for v in D.values() if v is not None]
    return dict(region_walk=D, coverage=cover, n_nodes=N,
                walk_diameter=max(fin) if fin else 0.0,
                walk_median=sorted(fin)[len(fin) // 2] if fin else 0.0,
                unreachable_pairs=sum(1 for v in D.values() if v is None))
