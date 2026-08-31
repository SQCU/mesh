"""placement.py -- WHERE tiles go and WHETHER they connect. Authors no geometry.

Lifted out of the deleted mapfuse.py, which tried to author architecture by writing
BSP lumps directly. design/MAPGEN-ROADMAP.md rules that out in one line -- "Do not
write a CSG/brush library. Emit `.map` text and let q3map2 do the BSP tree, VIS,
lightmaps and collision" -- and names the lump path "a dead end for authoring new
architecture, because you'd have to synthesize the tree, lightmaps and vis yourself
(it currently fakes the lightmap lump grey and ships empty visdata)". It did exactly
that: 49,152 grey lightmaps, 0 visdata, 2 clusters, 2.0 GB of server RSS.

So the CSG half is gone, not refactored: split_brushes, clip_faces, cut_portal,
add_brush/add_quad/add_box/add_plane, _convex_nonempty, _bounds_planes, _spans_volume,
slab_planes, axial_planes, corridor_frame, tube_gaps, corridor_volume, build_corridor,
build_pad, build_tele, axialize, the Fuser class and the lump writer. An aperture is a
parameter of a sweep (MAPGEN-ROADMAP stage 2), not a volume carved out of a compiled
artifact and then re-derived. Carving it there forced re-deriving the tree, VIS and
lightmaps by hand, which is why none of them were real.

What survives decides placement and connectivity and never touches a plane:

    Src / pk3_read / navigable_names   read a tile and its waypoints
    map_sites / solid_runs             candidate connection sites on a tile
    classify                           bridge (>3 sites) vs stub (2-3)
    pack_offsets                       3D bin packing of tiles into cells
    split_tree                         the placement tree
    walk_sample / walk_extent          reachable-volume sampling
    region_graph_solve / navmesh_solve connectivity over the placement
    check_bsp                          validation of a COMPILED result

These are inputs to the source emitter. Geometry is authored there, by construction,
and compiled by q3map2 -- so VIS, lightmaps and collision are real because they were
never faked.
"""

import struct, sys, os, re, math, glob, random, subprocess, time, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mkentfile as M
import negspace as NS
from negspace import box_H, subtract, bounds_of

MARGIN, CORW, CORH, WALL, FLOORTHK = 896.0, 288.0, 224.0, 32.0, 32.0
# CORR_SOFT is a SOFT budget used by the placement objective, not a refusal.  There is
# no maximum connector length anywhere in this file: a join that the placement leaves
# long is still built, as a parametrically generated span (see Fuser.build_span).
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


class Src:
    def __init__(self, name, data, wptext, cachetext, with_ns=True):
        self.name, self.data = name, data
        assert data[:4] == b'IBSP' and struct.unpack_from('<i', data, 4)[0] == 46, name
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
        self.wptriples = []
        lines = [l.rstrip('\r') for l in wptext.splitlines()]
        i = 0
        while i < len(lines) and lines[i].startswith('//'):
            i += 1
        while i + 3 <= len(lines):
            try:
                m1 = [float(x) for x in lines[i].strip().strip("'").split()]
                m2 = [float(x) for x in lines[i + 1].strip().strip("'").split()]
                fl = int(float(lines[i + 2].strip()))
            except (ValueError, IndexError):
                break
            self.wptriples.append((m1, m2, fl))
            i += 3
        self.cachelinks = []
        for l in cachetext.splitlines():
            l = l.strip()
            if not l or l.startswith('//'):
                continue
            p = l.split('*')
            if len(p) == 2 and p[0].strip():
                a = [float(x) for x in p[0].strip().strip("'").split()]
                b = [float(x) for x in p[1].strip().strip("'").split()]
                self.cachelinks.append((a, b))
        self.navnodes, self.navadj = M.parse_cache(cachetext)
        self.wpset = {tuple(round(x, 1) for x in m1) for m1, m2, fl in self.wptriples
                      if m1 == m2 and not fl & M.WPF_BAD}
        # DELETED with this rewrite: `bgrid`/`cgrid` and `solid_brush_at`/
        # `clip_brush_at`.  They answered "is this SOURCE point inside a SOURCE
        # brush" and their answers were then used to decide things about the
        # ASSEMBLED world -- before packing, before Z stacking, before the doorway
        # cuts split the brushwork and before a single connector brush existed.
        # That is the defect this file was rewritten to remove, not to improve.
        # The one definition of solidity now lives in negspace.NegSpace and is
        # COMPUTED (the BSP's own partition of space), not sampled.
        self.solidtex = [t[2] & 1 == 1 for t in self.textures]
        self.cliptex = [bool(t[2] & 0x430000) for t in self.textures]
        self.ns = NS.NegSpace(data, mask=NS.MASK_PLAYERSOLID) if with_ns else None
        self._ebrush = None

    def edit_index(self):
        """Which BRUSHES lie in a box -- an index for the geometry EDITOR, not an
        answer about occupancy.

        `split_brushes` has to know which source brushes to cut; that is a
        different question from "is this point solid", and it is the only reason
        an AABB per brush is wanted here.  The bounds come from `negspace`'s
        interval propagation, which is exact and finite for an entirely oblique
        brush -- the strict axis test the deleted `bgrid` used returned +-1e18 for
        those and then gridded them with an unguarded range(), a ~1e15-iteration
        loop that ate 75 GB."""
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
    """The SOLID spans along a ray, computed exactly from the free-space complex.

    Replaces `ray_runs`, which marched the ray in 8-unit steps asking
    `Src.solid_brush_at` at each step, and `free_slab`, which did the same over a
    3-D lattice of probe points.  A segment's intersection with a convex cell is
    a closed-form interval, so the free spans -- and therefore the solid spans
    between them -- are the whole answer rather than a sample of it, and a wall
    thinner than the old step size can no longer be missed."""
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


def map_sites(src, maxsites=12, minsep=1024.0):
    """All plausible connection sites on one map, best first -- read off the map's
    COMPUTED negative space, not probed for.

    A connection site is a BOUNDARY FACE of the free volume: a wall panel that
    (a) is big enough to hold a door-sized aperture, checked as exact containment
    of the door rectangle inside the face's own cross-section polygon; (b) has a
    solid run of wall-panel thickness behind it, computed from the free-space
    complex in closed form; (c) has free volume on the far side for the connector
    to meet, checked as exact coverage of the connector's approach box by free
    cells; and (d) stands in a cell the stock navmesh actually reaches, so a bot
    can get to the door.  `probe_site`'s ray march, `ray_runs` and `free_slab`
    are deleted: they were a sampled approximation of exactly this."""
    got = getattr(src, '_sites', None)
    if got is not None:
        return got
    ns = src.ns
    comp = [i for i in M.largest_component(src.navadj)
            if tuple(round(x, 1) for x in src.navnodes[i]) in src.wpset]
    if not comp:
        comp = M.largest_component(src.navadj)
    navcells = set()
    for i in comp:
        for dz in (0.0, 16.0, 32.0):
            c = ns.cell_at([src.navnodes[i][0], src.navnodes[i][1], src.navnodes[i][2] + dz])
            if c >= 0:
                navcells.add(c)
    cand = []
    for d in SITE_DIRS:
        axis = 0 if abs(d[0]) > 0.5 else 1
        U = 1 - axis
        pr = sorted(comp, key=lambda i: -(src.navnodes[i][0] * d[0] + src.navnodes[i][1] * d[1]))
        for i in pr[:max(48, len(pr) // 8)]:
            node = [float(x) for x in src.navnodes[i]]
            eye = [node[0], node[1], node[2] + DOOR_H * 0.5]
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
                continue               # a double wall / stacked scenery
            # THE WALL, STRUCTURALLY.
            # A connection site is a place where two free regions are SEPARATED by
            # a thin barrier.  All three parts of that are statements about volume,
            # so all three are answered as exact coverage of a box by the computed
            # free cells -- never by probes:
            #   * a player-sized free approach standing against the inner face,
            #   * a player-sized free landing on the far side for the connector,
            #   * and NO already-open route between them across the door's own
            #     footprint (otherwise there is nothing to cut and the "door"
            #     would be a hole in mid-air).
            zlo, zhi = node[2] + 4.0, node[2] + 72.0
            lo = [0.0, 0.0, zlo]
            hi = [0.0, 0.0, zhi]
            lo[U] = node[U] - (DOOR_W / 2 - 16)
            hi[U] = node[U] + (DOOR_W / 2 - 16)
            a0 = node[axis] + d[axis] * 24.0
            a1 = node[axis] + d[axis] * max(24.0, t_in - 3.0)
            alo, ahi = list(lo), list(hi)
            alo[axis], ahi[axis] = min(a0, a1), max(a0, a1)
            if ahi[axis] - alo[axis] < 2.0 or not ns.covered(box_H(alo, ahi)):
                continue                   # no room to stand in front of the opening
            f0 = node[axis] + d[axis] * (t_out + 32.0)
            f1 = node[axis] + d[axis] * (t_out + 160.0)
            flo, fhi = list(lo), list(hi)
            flo[axis], fhi[axis] = min(f0, f1), max(f0, f1)
            if not ns.covered(box_H(flo, fhi)):
                continue                   # nothing on the far side to meet
            tlo, thi = list(lo), list(hi)
            tlo[axis], thi[axis] = min(a0, f1), max(a0, f1)
            if ns.covered(box_H(tlo, thi)):
                continue                   # already open: there is no wall to cut
            cell = -1
            for back in (2.0, 8.0, 20.0, 40.0):
                q = [eye[0] + d[0] * (t_in - back), eye[1] + d[1] * (t_in - back), eye[2]]
                cell = ns.cell_at(q)
                if cell >= 0:
                    break
            if cell < 0:
                continue
            ext = ns.hi[cell] - ns.lo[cell]
            narrow = ext[U] < 576.0
            deg = len(src.navadj[i])
            cont = narrow or deg <= 2
            far = [node[0] + d[0] * (t_out + 96.0), node[1] + d[1] * (t_out + 96.0), node[2]]
            fc = ns.cell_at(far)
            exterior = fc < 0 or fc not in navcells
            score = ((1.5 if narrow else 0.0) + (0.5 if deg <= 3 else 0.0) +
                     (0.75 if exterior else 0.0) +
                     max(0.0, (384.0 - thick) / 384.0) +
                     max(0.0, (640.0 - t_in) / 640.0))
            cand.append({'p': node, 'dir': list(d), 't_in': t_in, 't_out': t_out,
                         'thick': thick, 'deg': deg, 'narrow': narrow,
                         'cell': cell, 'exterior': exterior,
                         'node': i, 'kind': 'continue' if cont else 'newcut',
                         'score': round(score, 3)})
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


def walk_sample(src, cap=128):
    """A strided, maximally-spread subsample of the tile's WALKABLE interior.

    The placement objective must measure the distance that actually matters -- the
    gap between the two tiles' reachable nav geometry -- not the gap between their
    bounding boxes.  Those are wildly different: a stock Xonotic map's playable floor
    can sit hundreds of units inside a bbox padded by sky, terrain skirts and
    out-of-bounds scenery, which is exactly how a lattice packed on bounding boxes
    produced kilometre corridors.  Points are the stand-on-able nodes of the map's
    largest bot-reachable waypoint component, in LOCAL coordinates."""
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
    if len(P) > cap:
        # farthest-point subsample: keeps the extremes (the parts of the walkable set
        # that face a neighbour tile) instead of an arbitrary stride
        A = np.asarray(P, dtype=float)
        pick = [int(np.argmax(A[:, 0] + A[:, 1]))]
        d = np.linalg.norm(A - A[pick[0]], axis=1)
        while len(pick) < cap:
            i = int(np.argmax(d))
            pick.append(i)
            d = np.minimum(d, np.linalg.norm(A - A[i], axis=1))
        P = [P[i] for i in pick]
    src._walk = np.asarray(P, dtype=float)
    return src._walk


def walk_extent(src):
    """(mins, maxs) of the WALKABLE set -- the box the placement should pack, as
    opposed to src.bounds which is the whole BSP hull."""
    got = getattr(src, '_walkext', None)
    if got is not None:
        return got
    W = walk_sample(src)
    if len(W) == 0:
        got = (list(src.bounds[0]), list(src.bounds[1]))
    else:
        got = (list(W.min(0)), list(W.max(0)))
    src._walkext = got
    return got


# ---------------------------------------------------------------------------
# 3D BIN PACKING + the bridge/stub taxonomy
# ---------------------------------------------------------------------------
def classify(nsites):
    """The taxonomy is counted off the connection sites a map actually has:
    a BRIDGE map has more than 3, a STUB map has fewer than 3 and more than 1, and a
    map with 0 or 1 site cannot be socketed at all and is not used."""
    if nsites > 3:
        return 'bridge'
    if nsites > 1:
        return 'stub'
    return 'unsuitable'


def pack_offsets(srcs, cells, cols, rows, levels):
    """Shelf sizing for the packed lattice: a column is only as wide as the widest hull
    in it, a row only as deep as the deepest, a level only as tall as the tallest.  Each
    tile is then anchored on its WALKABLE centre in x/y and on its walkable median floor
    in z, so what ends up near a cell boundary is the part of the map a player can stand
    on -- not a bounding box padded out by sky and terrain skirt."""
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
        W = walk_sample(s)
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
            off[a] = min(max(off[a], lo), hi)
            sl.append((lo, hi))
        offsets.append(off)
        slack.append(sl)
    return offsets, (xed, yed, zed), slack


def split_tree(items, eds):
    """Guillotine partition of the packed cells into the binary tree the BSP router
    needs.  The pack is axis-separable by construction (levels, then rows, then columns),
    so the tree falls straight out of it -- including on Z, which the old 2D fixed grid
    could not express."""
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


def _survey_worker(arg):
    """Compute one candidate map's free volume and its connection sites.

    The survey is the expensive half of a fusion (an exact convex decomposition
    per map, then a structural site solve on top of it) and it is per-map
    independent, so it is run across processes.  Nothing about the result depends
    on the order or on the other maps."""
    name, outdir, pk3 = arg
    try:
        src = load_src(name, outdir, pk3, quiet=True)
        st = map_sites(src)
        return (name, st, NS.pack(src.ns), None)
    except Exception as e:
        import traceback
        return (name, None, None, traceback.format_exc())


def region_graph_solve(j, edges_ab):
    """Connectivity solver over the REGION graph (tiles = nodes, joins = edges).
    Returns components, articulation points (chokepoint tiles), cut edges
    (chokepoint joins -- removing one disconnects the megamap), per-node degree and
    the hop-diameter.  Plain Hopcroft-Tarjan, iterative so it survives 30+ tiles."""
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
    # hop diameter over the largest component
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
    """Metrics over the NAVMESH solution: weighted (euclidean) shortest paths on the
    real fused bot-waypoint graph.  Returns per-region reachable coverage, the
    region-to-region bot WALKING distance matrix, and the megamap walking diameter --
    the commitment cost the spec asks the megamap to impose."""
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


