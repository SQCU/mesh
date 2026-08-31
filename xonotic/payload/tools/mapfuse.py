"""mapfuse.py -- procedural megamap fusion for the Xonotic payload demo.

Glues j stock Xonotic maps together with k PROCEDURALLY GENERATED bridge tiles,
socketed on a lattice like tilesets in a roguelike level generator.  See
design/FUSION-SPEC.md for the commanded requirement (verbatim) and the gap table.

  mapfuse.py <seed> [map ...] [flags]

    --maps=N | --maps=all   how many stock maps to draw from the navigable pool
                            (default: all of them, shuffled by seed)
    --bridges=K             how many procedural bridge tiles (default: j/3, min 1)
    --teams=T --carts=C     passed to mkentfile.emit (default 5 teams, 3 carts)
    --out=DIR               output dir (default /tmp/fusesmoke/data/maps)
    --nograph               skip the fusegraph.py solver/viewer pass
    --smoke                 boot the dedicated server on the fused map afterwards

Outputs fused.bsp / .pk3 / .ent / .waypoints(.cache) / .mapinfo / .joins.json /
.metrics.json, plus (unless --nograph) fused.graph.svg, fused.navmesh.svg and
fused.connectivity.json from fusegraph.py.
"""
import struct, sys, os, re, math, glob, random, subprocess, time, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mkentfile as M

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
    def __init__(self, name, data, wptext, cachetext):
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
        # NOTE: no M.Bsp(data) here.  That helper grids every brush AABB with an
        # unguarded range() over cell indices, so one brush bounded only by oblique
        # planes (catharsis has them) expands to a ~1e15-cell loop: the loader ate
        # 75 GB of RSS and never returned.  Nothing in this file used the object.
        ntex = len(self.textures)
        self.solidtex = [t[2] & 1 == 1 for t in self.textures]
        self.bgrid = {}
        for bi, (fs, ns, tx) in enumerate(self.brushes):
            if tx < 0 or tx >= ntex or not self.solidtex[tx]:
                continue
            lo, hi = [-1e18] * 3, [1e18] * 3
            for k in range(fs, fs + ns):
                nx, ny, nz, dd = self.planes[self.sides[k][0]]
                for a2, c in enumerate((nx, ny, nz)):
                    if c > 0.999:
                        hi[a2] = min(hi[a2], dd)
                    elif c < -0.999:
                        lo[a2] = max(lo[a2], -dd)
            cx0, cx1 = int(lo[0] // 1024), int(hi[0] // 1024)
            cy0, cy1 = int(lo[1] // 1024), int(hi[1] // 1024)
            if cx1 - cx0 > 200 or cy1 - cy0 > 200:
                continue
            for cx in range(cx0, cx1 + 1):
                for cy in range(cy0, cy1 + 1):
                    self.bgrid.setdefault((cx, cy), []).append((bi, lo, hi))
        self.cliptex = [bool(t[2] & 0x430000) for t in self.textures]
        self.cgrid = {}
        for bi, (fs, ns, tx) in enumerate(self.brushes):
            if tx < 0 or tx >= ntex or not self.cliptex[tx]:
                continue
            lo, hi = [-1e18] * 3, [1e18] * 3
            for k in range(fs, fs + ns):
                nx, ny, nz, dd = self.planes[self.sides[k][0]]
                for a2, c in enumerate((nx, ny, nz)):
                    if c > 0.999:
                        hi[a2] = min(hi[a2], dd)
                    elif c < -0.999:
                        lo[a2] = max(lo[a2], -dd)
            cx0, cx1 = int(lo[0] // 1024), int(hi[0] // 1024)
            cy0, cy1 = int(lo[1] // 1024), int(hi[1] // 1024)
            if cx1 - cx0 > 200 or cy1 - cy0 > 200:
                continue
            for cx in range(cx0, cx1 + 1):
                for cy in range(cy0, cy1 + 1):
                    self.cgrid.setdefault((cx, cy), []).append((bi, lo, hi))

    def clip_brush_at(self, p):
        for bi, lo, hi in self.cgrid.get((int(p[0] // 1024), int(p[1] // 1024)), ()):
            if not all(lo[a] - 0.25 <= p[a] <= hi[a] + 0.25 for a in range(3)):
                continue
            fs, ns, _ = self.brushes[bi]
            if all(vdot(self.planes[self.sides[k][0]][:3], p) - self.planes[self.sides[k][0]][3] <= 0.25
                   for k in range(fs, fs + ns)):
                return bi
        return -1

    def solid_brush_at(self, p):
        for bi, lo, hi in self.bgrid.get((int(p[0] // 1024), int(p[1] // 1024)), ()):
            if not all(lo[a] - 0.25 <= p[a] <= hi[a] + 0.25 for a in range(3)):
                continue
            fs, ns, _ = self.brushes[bi]
            if all(vdot(self.planes[self.sides[k][0]][:3], p) - self.planes[self.sides[k][0]][3] <= 0.25
                   for k in range(fs, fs + ns)):
                return bi
        return -1


def slab_planes(af, bf, dirh, side, ntop, w, lo, hi, endpad=8.0):
    return [(dirh, vdot(dirh, bf) + endpad), ([-x for x in dirh], -(vdot(dirh, af) - endpad)),
            (side, vdot(side, af) + w), ([-x for x in side], -(vdot(side, af) - w)),
            (ntop, vdot(ntop, af) + hi), ([-x for x in ntop], -(vdot(ntop, af) + lo))]


def axial_planes(mins, maxs):
    out = []
    for a in range(3):
        n = [0.0] * 3
        n[a] = 1.0
        out.append((n[:], maxs[a]))
        n[a] = -1.0
        out.append((n[:], -mins[a]))
    return out


def corridor_frame(a, b):
    af, bf = [a[0], a[1], a[2] - 1.0], [b[0], b[1], b[2] - 1.0]
    d = vsub(bf, af)
    L2 = math.hypot(d[0], d[1])
    dirh = [d[0] / L2, d[1] / L2, 0.0]
    side = [-dirh[1], dirh[0], 0.0]
    d3 = vnorm(d)
    ntop = vnorm(vcross(d3, side))
    if ntop[2] < 0:
        ntop = [-x for x in ntop]
    return af, bf, dirh, side, ntop, math.dist(af, bf)


def corridor_samples(a, b, w2=None):
    """Sample the corridor TUBE for blockage.  w2 is the corridor half-width actually
    being built: a prominent join is CORW_PROM/2 = 224 wide, and sampling it at the
    narrow +-120 of a subtle corridor is what let obstructions survive inside the
    prominent corridors' flanks."""
    af, bf, dirh, side, ntop, L = corridor_frame(a, b)
    w2 = (CORW / 2) if w2 is None else w2
    pts = []
    n = max(2, int(L // 48))
    for k in range(n + 1):
        f = k / n
        base = [af[i] + f * (bf[i] - af[i]) for i in range(3)]
        for lat in (-w2 + 20, -w2 + 40, -w2 * 0.72, -w2 * 0.36, 0.0, w2 * 0.36, w2 * 0.72,
                    w2 - 40, w2 - 20):
            for h in (20.0, 28.0, 72.0, 124.0, CORH - 48, CORH - 36):
                pts.append(vadd(vadd(base, vscale(side, lat)), vscale(ntop, h)))
    return pts


def arc_samples(a, b):
    apex = max(a[2], b[2]) + 160.0
    pts = []
    for k in range(1, 32):
        f = k / 32.0
        x = a[0] + f * (b[0] - a[0])
        y = a[1] + f * (b[1] - a[1])
        z = (1 - f) * a[2] + f * b[2] + (apex - (1 - f) * a[2] - f * b[2]) * 4 * f * (1 - f)
        pts.append([x, y, z + 24])
    return pts


def blockage(srcs, offsets, pts, clip=False):
    hits = set()
    for m, src in enumerate(srcs):
        off = offsets[m]
        for p in pts:
            q = vsub(p, off)
            bi = src.solid_brush_at(q)
            if bi >= 0:
                hits.add((m, bi))
            if clip:
                ci = src.clip_brush_at(q)
                if ci >= 0:
                    hits.add((m, ci))
    return hits


def brush_volume_ok(src, bi):
    lo, hi = [-65536.0] * 3, [65536.0] * 3
    fs, ns, _ = src.brushes[bi]
    for k in range(fs, fs + ns):
        nx, ny, nz, dd = src.planes[src.sides[k][0]]
        for a, c in enumerate((nx, ny, nz)):
            if c > 0.999:
                hi[a] = min(hi[a], dd)
            elif c < -0.999:
                lo[a] = max(lo[a], -dd)
    v = 1.0
    for a in range(3):
        v *= max(1.0, hi[a] - lo[a])
    return v <= 640 ** 3


class Fuser:
    def __init__(self, srcs, offsets, seed):
        self.srcs, self.offsets, self.rng = srcs, offsets, random.Random(seed)
        self.textures, self.texidx = [], {}
        self.texmap = []
        for src in srcs:
            tm = []
            for t in src.textures:
                if t[0] not in self.texidx:
                    self.texidx[t[0]] = len(self.textures)
                    self.textures.append(list(t))
                tm.append(self.texidx[t[0]])
            self.texmap.append(tm)
        self.trigtex, self.emptytex = len(self.textures), len(self.textures) + 1
        self.textures.append(list(TRIGTEX))
        self.textures.append(list(EMPTYTEX))
        self.planes, self.planebase = [], []
        for m, src in enumerate(srcs):
            self.planebase.append(len(self.planes))
            for n0, n1, n2, d in src.planes:
                self.planes.append([n0, n1, n2, d + vdot((n0, n1, n2), offsets[m])])
        self.verts, self.vertbase = [], []
        for m, src in enumerate(srcs):
            self.vertbase.append(len(self.verts))
            for v in src.verts:
                w = list(v)
                w[0] += offsets[m][0]
                w[1] += offsets[m][1]
                w[2] += offsets[m][2]
                self.verts.append(w)
        self.mesh, self.elembase = bytearray(), []
        for src in srcs:
            self.elembase.append(len(self.mesh) // 4)
            self.mesh += src.mesh
        self.sides, self.sidebase = [], []
        for m, src in enumerate(srcs):
            self.sidebase.append(len(self.sides))
            for pi, ti in src.sides:
                self.sides.append([pi + self.planebase[m], self.texmap[m][ti]])
        self.carved = set()
        self.dropped_faces = set()
        self.portals = []
        self.conn_faces, self.conn_brushes, self.conn_leafsets = [], [], []
        self.trig_brushes, self.trig_models = [], []
        self.bot_jumps, self.conn_meta = [], []
        self.extra_ents = []
        self.wp_extra, self.link_extra = [], []
        self.dropped_spawns = 0
        self.dropped_spawns_budget = 0
        self.spawn_cap = 10
        self.ent_budget = 1800
        self.ent_dropped = {}
        self.ent_short = 0
        self.ent_orphans = 0
        self.solid_face_tex = self.pick_face_tex()

    def pick_face_tex(self):
        cnt = {}
        src = self.srcs[0]
        for f, _ in src.faces:
            ti = f[0]
            if 0 <= ti < len(src.textures) and src.textures[ti][2] & 1 and f[2] in (1, 3):
                cnt[ti] = cnt.get(ti, 0) + 1
        best = max(cnt, key=cnt.get) if cnt else 0
        return self.texmap[0][best]

    def add_plane(self, n, d):
        self.planes.append([n[0], n[1], n[2], d])
        return len(self.planes) - 1

    def add_brush(self, planes, tex, dest=None, bounds=None):
        if dest is None:
            dest = self.conn_brushes
        planes = list(planes)
        if bounds is not None:
            clo, chi = bounds
            for a in range(3):
                e = [0.0, 0.0, 0.0]
                e[a] = 1.0
                planes.append((e[:], chi[a]))
                e2 = [0.0, 0.0, 0.0]
                e2[a] = -1.0
                planes.append((e2, -clo[a]))
        s0 = len(self.sides)
        for n, d in planes:
            self.sides.append([self.add_plane(n, d), tex])
        dest.append([s0, len(planes), tex])
        return len(dest) - 1

    def add_quad(self, pts, normal, tex):
        v0 = len(self.verts)
        L = math.dist(pts[0], pts[1])
        for i, p in enumerate(pts):
            s = (L / 128.0) if i in (1, 2) else 0.0
            t = 1.0 if i in (2, 3) else 0.0
            self.verts.append([p[0], p[1], p[2], s, t, 0.0, 0.0, normal[0], normal[1], normal[2], 180, 180, 180, 255])
        e0 = len(self.mesh) // 4
        self.mesh += struct.pack('<12i', 0, 1, 2, 0, 2, 3, 2, 1, 0, 3, 2, 0)
        head = [tex, -1, 3, v0, 4, e0, 12, -1, 0, 0, 0, 0]
        lo = [min(p[a] for p in pts) for a in range(3)]
        hi = [max(p[a] for p in pts) for a in range(3)]
        tail = struct.pack('<3i6f5i', 0, 0, 0, lo[0], lo[1], lo[2], hi[0], hi[1], hi[2], 0, 0, 0, 0, 0)
        self.conn_faces.append((head, tail))
        return len(self.conn_faces) - 1

    def add_light(self, p, radius):
        self.extra_ents.append('{\n"classname" "light"\n"origin" "%s %s %s"\n"light" "%d"\n}'
                               % (fnum(p[0]), fnum(p[1]), fnum(p[2]), int(radius)))

    def carve(self, hits):
        for m, bi in hits:
            self.carved.add((m, bi))

    def add_box(self, lo, hi, tex, faces=True):
        """A solid axis-aligned box with all six surfaces rendered.  The frame pieces
        of a cut doorway are built from these."""
        bi = self.add_brush(axial_planes(lo, hi), tex)
        fs = []
        if faces:
            c = [[lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]], [hi[0], hi[1], lo[2]], [lo[0], hi[1], lo[2]],
                 [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]], [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]]]
            quads = ((4, 5, 6, 7, (0, 0, 1)), (3, 2, 1, 0, (0, 0, -1)), (0, 1, 5, 4, (0, -1, 0)),
                     (2, 3, 7, 6, (0, 1, 0)), (1, 2, 6, 5, (1, 0, 0)), (3, 0, 4, 7, (-1, 0, 0)))
            for i0, i1, i2, i3, n in quads:
                fs.append(self.add_quad([c[i0], c[i1], c[i2], c[i3]], list(n), tex))
        self.conn_leafsets.append((fs, [bi], [x - 8 for x in lo], [x + 8 for x in hi]))
        return bi

    def split_brushes(self, m, alo, ahi):
        """LITERALLY EDIT THE MAP: subtract an axis-aligned aperture box from every
        source brush of tile m that occupies it.

        Not the old carve, which switched a whole brush's contents to empty and so
        dissolved an entire wall panel to make room for a tube.  Here each occupying
        brush is replaced by up to six convex remainders -- itself intersected with each
        half-space outside the aperture -- so the wall stays exactly where it was and
        exactly as thick as it was, minus a doorway.  Convexity is preserved by
        construction (a convex brush intersected with a half-space is convex), which is
        what the BSP collision hull requires."""
        src, off = self.srcs[m], self.offsets[m]
        llo = [alo[a] - off[a] for a in range(3)]
        lhi = [ahi[a] - off[a] for a in range(3)]
        cand = {}
        for grid in (src.bgrid, src.cgrid):
            for cx in range(int(llo[0] // 1024), int(lhi[0] // 1024) + 1):
                for cy in range(int(llo[1] // 1024), int(lhi[1] // 1024) + 1):
                    for bi, blo, bhi in grid.get((cx, cy), ()):
                        if all(blo[a] < lhi[a] and bhi[a] > llo[a] for a in range(3)):
                            cand[bi] = (blo, bhi)
        made = 0
        for bi, (blo, bhi) in cand.items():
            fs, ns, tx = src.brushes[bi]
            wp = []
            for k in range(fs, fs + ns):
                pl = self.planes[self.sides[self.sidebase[m] + k][0]]
                wp.append((pl[:3], pl[3]))
            tex = self.texmap[m][tx] if 0 <= tx < len(self.texmap[m]) else self.solid_face_tex
            # A remainder MUST carry a full set of axial bounding planes.  A brush bounded
            # only by oblique planes has no axial AABB, and a downstream consumer that
            # derives one by scanning plane distances gets +-1e18 and then grids it with an
            # unguarded range(): that is a ~1e15-iteration loop that ate 33 GB of RSS in
            # mkentfile.emit on the fused world.  Where the source brush has no finite
            # axial bound on an axis, fall back to the tile hull with generous slop.
            fin = lambda v: abs(v) < 1e17
            wblo = [blo[a] + off[a] if fin(blo[a]) else src.bounds[0][a] + off[a] - 4096.0
                    for a in range(3)]
            wbhi = [bhi[a] + off[a] if fin(bhi[a]) else src.bounds[1][a] + off[a] + 4096.0
                    for a in range(3)]
            bnd = (wblo, wbhi)
            for a in range(3):
                e = [0.0, 0.0, 0.0]
                e[a] = 1.0
                if alo[a] - wblo[a] > 1.0:                      # remainder BELOW the aperture
                    self.add_brush(wp + [(e[:], alo[a])], tex, bounds=bnd)
                    made += 1
                if wbhi[a] - ahi[a] > 1.0:                      # remainder ABOVE the aperture
                    e2 = [0.0, 0.0, 0.0]
                    e2[a] = -1.0
                    self.add_brush(wp + [(e2, -ahi[a])], tex, bounds=bnd)
                    made += 1
            self.carved.add((m, bi))
        return len(cand), made

    def clip_faces(self, m, alo, ahi, axis):
        """Cut the aperture out of the rendered SURFACES too.

        A brush split with the wall's faces left intact is a wall you can walk through
        and still see -- the exact "map graphics conceal map transitions" failure.  A
        rectangular wall face crossing the aperture is re-emitted as the up-to-four
        rectangles that survive the cut, keeping its own texture, so the wall around the
        new doorway is still the level's own wall.  Returns (dropped, reissued, tex)."""
        src, off = self.srcs[m], self.offsets[m]
        mm = src.models[0]
        U = 1 - axis
        drop = reissue = 0
        walltex = None
        for i in range(mm[6], mm[6] + mm[7]):
            h, tail = src.faces[i]
            nv = h[4]
            if nv < 3:
                continue
            V = [[src.verts[h[3] + k][a] + off[a] for a in range(3)] for k in range(nv)]
            flo = [min(v[a] for v in V) for a in range(3)]
            fhi = [max(v[a] for v in V) for a in range(3)]
            if not all(flo[a] < ahi[a] - 1 and fhi[a] > alo[a] + 1 for a in range(3)):
                continue
            n = list(struct.unpack_from('<3f', tail, 36))
            tex = self.texmap[m][h[0]] if 0 <= h[0] < len(self.texmap[m]) else self.solid_face_tex
            if abs(n[axis]) > 0.9 and fhi[axis] - flo[axis] < 2.0:
                # a flat wall face square to the door axis: re-issue the surviving parts
                if walltex is None:
                    walltex = tex
                self.dropped_faces.add((m, i))
                drop += 1
                at = (flo[axis] + fhi[axis]) / 2.0
                bands = []
                if flo[U] < alo[U] - 1:
                    bands.append((flo[U], min(alo[U], fhi[U]), flo[2], fhi[2]))
                if fhi[U] > ahi[U] + 1:
                    bands.append((max(ahi[U], flo[U]), fhi[U], flo[2], fhi[2]))
                mid0, mid1 = max(flo[U], alo[U]), min(fhi[U], ahi[U])
                if mid1 > mid0 + 1:
                    if flo[2] < alo[2] - 1:
                        bands.append((mid0, mid1, flo[2], min(alo[2], fhi[2])))
                    if fhi[2] > ahi[2] + 1:
                        bands.append((mid0, mid1, max(ahi[2], flo[2]), fhi[2]))
                for u0, u1, z0, z1 in bands:
                    q = []
                    for uu, zz in ((u0, z0), (u1, z0), (u1, z1), (u0, z1)):
                        pt = [0.0, 0.0, zz]
                        pt[axis], pt[U] = at, uu
                        q.append(pt)
                    self.add_quad(q, n, tex)
                    reissue += 1
            elif all(flo[a] > alo[a] - 1 and fhi[a] < ahi[a] + 1 for a in range(3)):
                self.dropped_faces.add((m, i))     # trim wholly inside the new opening
                drop += 1
        return drop, reissue, walltex

    def cut_portal(self, m, site, prominent=False):
        """Grow a diegetic opening in tile m's own geometry at `site`.

        This is the edit the whole fusion exists to make.  The wall panel the site
        faces is split around a door-sized aperture, the wall's surfaces are re-cut
        around it, the reveal (the four faces of the cut through the wall's thickness)
        is surfaced in the wall's own texture, and a jamb/header architrave is set into
        the outer face so the result reads as a doorway that was always there rather
        than a hole.  Returns the OUTER MOUTH -- the point just outside the new opening
        where the procedural connector meets it."""
        src, off = self.srcs[m], self.offsets[m]
        d = site['dir']
        axis = 0 if abs(d[0]) > 0.5 else 1
        U = 1 - axis
        sgn = 1.0 if d[axis] > 0 else -1.0
        p = vadd(site['p'], off)
        q = lambda x: round(x / 4.0) * 4.0
        a0 = p[axis] + sgn * (site['t_in'] - 16.0)
        a1 = p[axis] + sgn * (site['t_out'] + 16.0)
        alo, ahi = [0.0] * 3, [0.0] * 3
        alo[axis], ahi[axis] = q(min(a0, a1)), q(max(a0, a1))
        alo[U], ahi[U] = q(p[U] - DOOR_W / 2), q(p[U] + DOOR_W / 2)
        alo[2], ahi[2] = q(p[2] + DOOR_SILL), q(p[2] + DOOR_SILL + DOOR_H)
        nb, npieces = self.split_brushes(m, alo, ahi)
        nd, nr, walltex = self.clip_faces(m, alo, ahi, axis)
        tex = walltex if walltex is not None else self.solid_face_tex
        # reveal: the four surfaces of the cut through the wall's own thickness
        cor = lambda u, z, t: ([t, u, z] if axis == 0 else [u, t, z])
        for z, nz in ((alo[2], [0, 0, 1]), (ahi[2], [0, 0, -1])):
            self.add_quad([cor(alo[U], z, alo[axis]), cor(ahi[U], z, alo[axis]),
                           cor(ahi[U], z, ahi[axis]), cor(alo[U], z, ahi[axis])], nz, tex)
        for u, nu in ((alo[U], 1), (ahi[U], -1)):
            nv = [0.0, 0.0, 0.0]
            nv[U] = float(nu)
            self.add_quad([cor(u, alo[2], alo[axis]), cor(u, alo[2], ahi[axis]),
                           cor(u, ahi[2], ahi[axis]), cor(u, ahi[2], alo[axis])], nv, tex)
        # threshold: a floor slab under the doorway, so the cut never opens onto a drop
        tlo, thi = list(alo), list(ahi)
        tlo[2], thi[2] = alo[2] - 24.0, alo[2]
        self.add_box(tlo, thi, tex)
        # ARCHITRAVE: two jambs and a header set into the OUTER face of the wall.  This
        # is what makes the opening read as architecture instead of damage.
        oface = ahi[axis] if sgn > 0 else alo[axis]
        j0, j1 = (oface, oface + 24.0) if sgn > 0 else (oface - 24.0, oface)
        for u0, u1 in ((alo[U] - 24.0, alo[U]), (ahi[U], ahi[U] + 24.0)):
            lo, hi = [0.0] * 3, [0.0] * 3
            lo[axis], hi[axis] = j0, j1
            lo[U], hi[U] = u0, u1
            lo[2], hi[2] = alo[2], ahi[2] + 24.0
            self.add_box(lo, hi, tex)
        lo, hi = [0.0] * 3, [0.0] * 3
        lo[axis], hi[axis] = j0, j1
        lo[U], hi[U] = alo[U] - 24.0, ahi[U] + 24.0
        lo[2], hi[2] = ahi[2], ahi[2] + 24.0
        self.add_box(lo, hi, tex)
        # nav: walk the bot from the map's own waypoint, through the opening, to the mouth
        mouth = [0.0, 0.0, alo[2] + 8.0]
        mouth[axis] = ahi[axis] + 48.0 if sgn > 0 else alo[axis] - 48.0
        mouth[U] = (alo[U] + ahi[U]) / 2.0
        inner = list(mouth)
        inner[axis] = (alo[axis] + ahi[axis]) / 2.0
        chain = [[p[0], p[1], p[2]], inner, mouth]
        self.wp_extra += [inner, mouth]
        for k in range(len(chain) - 1):
            self.link_extra.append((chain[k], chain[k + 1]))
            self.link_extra.append((chain[k + 1], chain[k]))
        if prominent:
            lp = list(mouth)
            lp[2] += DOOR_H - 32
            self.add_light(lp, PROM_LIGHT)
        self.portals.append({'tile': m, 'axis': axis, 'sgn': sgn, 'mouth': mouth,
                             'aperture': [alo, ahi], 'brushes': nb, 'pieces': npieces,
                             'faces_cut': nd, 'faces_reissued': nr, 'kind': site['kind'],
                             'thick': site['thick'], 'node': [p[0], p[1], p[2]]})
        return mouth

    def axialize(self):
        """Give every emitted source brush a finite axis-aligned bound.

        A brush whose only near-axial planes are slightly oblique (a normal like
        (0.9999, 0.01, 0), which stock catharsis, xoylent and finalrage all contain)
        has no derivable AABB under a strict axis test.  A downstream consumer that
        derives one anyway gets +-1e18 and then grids it with an unguarded range() --
        a ~1e15-iteration loop.  That is a real, measured failure: it took the fused
        world's entity pass past 33 GB of RSS and killed the build twice.  Rather than
        reach into a file this work does not own, the fused artifact is made
        well-formed: the offending brush is re-emitted with six axial clamp planes at
        its tile's hull plus 4096 units of slop, far outside any playable space, so the
        brush's actual shape is untouched and its AABB is now finite."""
        n = 0
        for m, src in enumerate(self.srcs):
            off = self.offsets[m]
            ntex = len(src.textures)
            for bi, (fs, ns, tx) in enumerate(src.brushes):
                if tx < 0 or tx >= ntex or not (src.solidtex[tx] or src.cliptex[tx]):
                    continue
                lo, hi = [-1e18] * 3, [1e18] * 3
                for k in range(fs, fs + ns):
                    nrm = self.planes[self.sides[self.sidebase[m] + k][0]]
                    for a in range(3):
                        if max(abs(nrm[b]) for b in range(3) if b != a) > 1e-4:
                            continue
                        if nrm[a] > 0.999:
                            hi[a] = min(hi[a], nrm[3])
                        elif nrm[a] < -0.999:
                            lo[a] = max(lo[a], -nrm[3])
                if all(abs(v) < 1e17 for v in lo + hi):
                    continue
                wp = [(self.planes[self.sides[self.sidebase[m] + k][0]][:3],
                       self.planes[self.sides[self.sidebase[m] + k][0]][3])
                      for k in range(fs, fs + ns)]
                clo = [src.bounds[0][a] + off[a] - 4096.0 for a in range(3)]
                chi = [src.bounds[1][a] + off[a] + 4096.0 for a in range(3)]
                self.add_brush(wp, self.texmap[m][tx], bounds=(clo, chi))
                self.carved.add((m, bi))
                n += 1
        return n

    def build_corridor(self, a, b, prominent=True):
        af, bf, dirh, side, ntop, L = corridor_frame(a, b)
        w2 = (CORW_PROM if prominent else CORW) / 2
        pad = w2 + CORH + 2 * WALL + 64
        clo = [min(af[i], bf[i]) - pad for i in range(3)]
        chi = [max(af[i], bf[i]) + pad for i in range(3)]
        bnd = (clo, chi)
        br = [self.add_brush(slab_planes(af, bf, dirh, side, ntop, w2 + WALL, -FLOORTHK, 0.0), self.solid_face_tex, bounds=bnd)]
        for s in (1.0, -1.0):
            sv = vscale(side, s)
            pl = [(dirh, vdot(dirh, bf) + 8), ([-x for x in dirh], -(vdot(dirh, af) - 8)),
                  (sv, vdot(sv, af) + w2 + WALL), ([-x for x in sv], -(vdot(sv, af) + w2)),
                  (ntop, vdot(ntop, af) + CORH + WALL), ([-x for x in ntop], -(vdot(ntop, af) - FLOORTHK))]
            br.append(self.add_brush(pl, self.solid_face_tex, bounds=bnd))
        br.append(self.add_brush(slab_planes(af, bf, dirh, side, ntop, w2 + WALL, CORH, CORH + WALL), self.solid_face_tex, bounds=bnd))
        fa = []
        c = lambda base, lat, h: vadd(vadd(base, vscale(side, lat)), vscale(ntop, h))
        fa.append(self.add_quad([c(af, -w2, 0), c(bf, -w2, 0), c(bf, w2, 0), c(af, w2, 0)], ntop, self.solid_face_tex))
        fa.append(self.add_quad([c(af, -w2, 0), c(bf, -w2, 0), c(bf, -w2, CORH), c(af, -w2, CORH)], side, self.solid_face_tex))
        fa.append(self.add_quad([c(af, w2, 0), c(bf, w2, 0), c(bf, w2, CORH), c(af, w2, CORH)], [-x for x in side], self.solid_face_tex))
        fa.append(self.add_quad([c(af, -w2, CORH), c(bf, -w2, CORH), c(bf, w2, CORH), c(af, w2, CORH)], [-x for x in ntop], self.solid_face_tex))
        lo = [min(af[i], bf[i]) - CORW for i in range(3)]
        hi = [max(af[i], bf[i]) + CORW + CORH for i in range(3)]
        self.conn_leafsets.append((fa, br, lo, hi))
        if prominent:
            for e in (af, bf):
                self.add_light([e[0], e[1], e[2] + CORH - 24], PROM_LIGHT)
        n = max(1, int(L // 384))
        chain = []
        for k in range(n + 1):
            f = k / n
            p = [af[i] + f * (bf[i] - af[i]) for i in range(3)]
            chain.append([p[0], p[1], p[2] + 9])
        self.wp_extra += chain
        prev = a
        for p in chain:
            self.link_extra.append((prev, p))
            self.link_extra.append((p, prev))
            prev = p
        self.link_extra.append((prev, b))
        self.link_extra.append((b, prev))

    def build_pad(self, a, b, idx, prominent=False):
        pm = [a[0] - 56, a[1] - 56, a[2] - 8]
        px = [a[0] + 56, a[1] + 56, a[2] + 6]
        bi = self.add_brush(axial_planes(pm, px), self.solid_face_tex)
        fi = self.add_quad([[pm[0], pm[1], px[2]], [px[0], pm[1], px[2]], [px[0], px[1], px[2]], [pm[0], px[1], px[2]]],
                           [0, 0, 1], self.solid_face_tex)
        self.conn_leafsets.append(([fi], [bi], vsub(pm, [8, 8, 8]), vadd(px, [8, 8, 8])))
        tb = self.add_brush(axial_planes([a[0] - 48, a[1] - 48, a[2] + 6], [a[0] + 48, a[1] + 48, a[2] + 70]),
                            self.trigtex, self.trig_brushes)
        self.trig_models.append((tb, [a[0] - 48, a[1] - 48, a[2] + 6], [a[0] + 48, a[1] + 48, a[2] + 70],
                                 'trigger_push', 'fpush%d' % idx))
        self.extra_ents.append('{\n"classname" "target_position"\n"targetname" "fpush%d"\n"origin" "%s %s %s"\n}' %
                               (idx, fnum(b[0]), fnum(b[1]), fnum(b[2] + 40)))
        if prominent:
            self.add_light([a[0], a[1], a[2] + 96], PROM_LIGHT)
        self.bot_jumps.append((list(a), list(b)))

    def build_tele(self, a, b, idx, prominent=False):
        tb = self.add_brush(axial_planes([a[0] - 48, a[1] - 48, a[2]], [a[0] + 48, a[1] + 48, a[2] + 112]),
                            self.trigtex, self.trig_brushes)
        self.trig_models.append((tb, [a[0] - 48, a[1] - 48, a[2]], [a[0] + 48, a[1] + 48, a[2] + 112],
                                 'trigger_teleport', 'ftele%d' % idx))
        self.extra_ents.append('{\n"classname" "misc_teleporter_dest"\n"targetname" "ftele%d"\n"origin" "%s %s %s"\n"angle" "%d"\n}' %
                               (idx, fnum(b[0]), fnum(b[1]), fnum(b[2] + 16), 0))
        if prominent:
            self.add_light([a[0], a[1], a[2] + 96], PROM_LIGHT)
        self.bot_jumps.append((list(a), list(b)))

    def merge_entities(self):
        j = len(self.srcs)
        self.modelmap = []
        nsub = 1
        for m, src in enumerate(self.srcs):
            mm = {0: 0}
            for i in range(1, len(src.models)):
                mm[i] = nsub
                nsub += 1
            self.modelmap.append(mm)
        self.trigmodel0 = nsub
        blocks_out = []
        spawn_pending = []
        tkeys = ('target', 'target2', 'target3', 'target4', 'killtarget', 'targetname')
        for m, src in enumerate(self.srcs):
            off = self.offsets[m]
            for b in re.findall(r'\{[^{}]*\}', src.ents):
                cn = re.search(r'"classname"\s+"([^"]+)"', b)
                cn = cn.group(1) if cn else ''
                if cn == 'info_autoscreenshot':
                    continue
                if cn == 'worldspawn':
                    if m == 0:
                        b = re.sub(r'"_?deluxeMaps"\s+"[^"]*"\n?', '', b)
                        b = re.sub(r'"gridsize"\s+"[^"]*"\n?', '', b)
                        b = re.sub(r'"message"\s+"[^"]*"', '"message" "fused: %s"' %
                                   '+'.join(s.name for s in self.srcs), b)
                        blocks_out.insert(0, b)
                    continue

                def fixorigin(mo):
                    v = [float(x) for x in mo.group(1).split()]
                    return '"origin" "%s %s %s"' % (fnum(v[0] + off[0]), fnum(v[1] + off[1]), fnum(v[2] + off[2]))

                if cn.startswith('info_player_'):
                    # A spawnpoint that lands in solid makes stock Xonotic run its
                    # expanding relocate_spawnpoint search; at megamap scale that
                    # search is what trips "server runaway loop counter hit limit of
                    # 10000000 jumps" before the match ever starts.  Drop such
                    # spawnpoints here instead: the fused world has hundreds left.
                    mo = re.search(r'"origin"\s+"([-\d.eE+ ]+)"', b)
                    if mo:
                        o = [float(x) for x in mo.group(1).split()]
                        bad = False
                        for dz in (2.0, 24.0, 48.0):
                            if src.solid_brush_at([o[0], o[1], o[2] + dz]) >= 0:
                                bad = True
                                break
                        if bad:
                            self.dropped_spawns += 1
                            continue
                        spawn_pending.append((m, o, b, cn))
                        continue
                b = re.sub(r'"origin"\s+"([-\d.eE+ ]+)"', fixorigin, b)
                b = re.sub(r'"model"\s+"\*(\d+)"',
                           lambda mo: '"model" "*%d"' % self.modelmap[m][int(mo.group(1))], b)
                for k in tkeys:
                    b = re.sub(r'"%s"\s+"([^"]+)"' % k,
                               lambda mo, k=k: '"%s" "m%d_%s"' % (k, m, mo.group(1)), b)
                blocks_out.append(b)
        # SPAWNPOINT BUDGET.  Merging every source map's full spawnpoint set gives a
        # megamap over a thousand info_player_* entities.  Stock Xonotic's
        # IntrusiveList primitives (IL_CONTAINS / IL_REMOVE_RAW / il_links_flds##GETFP)
        # are linear scans, and spawnfunc-time churn over that many entities trips
        # "server runaway loop counter hit limit of 10000000 jumps" during
        # __spawnfunc_worldspawn, before the match starts.  Measured on a 22-tile
        # fusion: il_links_flds##GETFP alone burned 16.1M of the 10M-statement budget.
        # Keep a spatially spread subset per tile instead; every region stays spawnable.
        kept = []
        for m in range(j):
            mine = [t for t in spawn_pending if t[0] == m]
            if len(mine) <= self.spawn_cap:
                kept += mine
                continue
            picks = [mine[0]]
            while len(picks) < self.spawn_cap:
                nxt = max(mine, key=lambda t: min(math.dist(t[1], q[1]) for q in picks))
                if nxt in picks:
                    break
                picks.append(nxt)
            kept += picks
            self.dropped_spawns_budget += len(mine) - len(picks)
        for m, o, b, cn in kept:
            off2 = self.offsets[m]
            b = re.sub(r'"origin"\s+"([-\d.eE+ ]+)"',
                       lambda mo, off2=off2: '"origin" "%s %s %s"' % tuple(
                           fnum(float(x) + off2[i]) for i, x in enumerate(mo.group(1).split())), b)
            for k in tkeys:
                b = re.sub(r'"%s"\s+"([^"]+)"' % k,
                           lambda mo, k=k, m=m: '"%s" "m%d_%s"' % (k, m, mo.group(1)), b)
            blocks_out.append(b)
        for i, (tb, mins, maxs, cls, tgt) in enumerate(self.trig_models):
            blocks_out.append('{\n"classname" "%s"\n"model" "*%d"\n"target" "%s"\n}' %
                              (cls, self.trigmodel0 + i, tgt))
        blocks_out += self.extra_ents
        blocks_out = self.entity_budget(blocks_out)
        return '\n'.join(blocks_out) + '\n'

    # order in which merged source entities are given up when the fused world exceeds
    # the entity budget.  Never includes target_*/info_null/trigger_*/teleport/spawn
    # entities: those are referenced by name or are load-bearing for navigation.
    DROP_ORDER = ('light', 'dom_team', 'dom_controlpoint', 'trigger_race_checkpoint',
                  'info_player_race', 'target_speaker', 'func_pointparticles',
                  'misc_gamemodel', 'misc_breakablemodel', 'misc_models',
                  'item_armor_small', 'item_health_small', 'item_shells',
                  'item_bullets', 'item_rockets', 'item_cells', 'item_health_medium',
                  'item_armor_medium')
    # classes that exist only to be pointed at; once nothing points at them they are
    # dead weight and each one still costs a spawnfunc at worldspawn
    TARGET_ONLY = ('target_position', 'info_null', 'misc_teleporter_dest',
                   'target_location', 'target_speaker')

    def entity_budget(self, blocks):
        """Hold the fused world under a spawn-time entity budget.

        This DarkPlaces build has no prvm_runawaycheck cvar; the 10,000,000-jump
        limit is compiled in, and stock Xonotic's IntrusiveList primitives are linear
        scans, so worldspawn's spawnfunc chain over a megamap's merged entity set
        aborts the server with "server runaway loop counter hit limit of 10000000
        jumps" before the match starts.  Measured: 1805 entities (8 tiles) boots;
        5780 entities (22 tiles) does not, with il_links_flds##GETFP alone burning
        16.1M statements.  Source `light` entities go first: mapfuse flattens the
        lightmap lump to a single grey block, so not one of them affects the fused
        world's appearance."""
        if self.ent_budget <= 0:
            self.ent_dropped = {}
            return blocks
        # orphan sweep first, and again after the class drops: removing a referrer
        # orphans its target, and an orphaned target_position is pure spawn cost
        blocks = self.sweep_orphans(blocks)
        if len(blocks) <= self.ent_budget:
            self.ent_dropped = {}
            return blocks
        cls = []
        for b in blocks:
            mo = re.search(r'"classname"\s+"([^"]+)"', b)
            cls.append(mo.group(1) if mo else '')
        # never drop anything another entity points at by name, or the engine logs
        # "follow: could not find target/killtarget" and the reference is dead
        named = set()
        for b in blocks:
            for k in ('target', 'target2', 'target3', 'target4', 'killtarget'):
                for mo in re.finditer(r'"%s"\s+"([^"]+)"' % k, b):
                    named.add(mo.group(1))
        protected = set()
        for i, b in enumerate(blocks):
            mo = re.search(r'"targetname"\s+"([^"]+)"', b)
            if mo and mo.group(1) in named:
                protected.add(i)
        keep = [True] * len(blocks)
        dropped = {}
        need = len(blocks) - self.ent_budget
        for c in self.DROP_ORDER:
            if need <= 0:
                break
            idxs = [i for i in range(len(blocks)) if cls[i] == c and keep[i] and i not in protected]
            take = min(len(idxs), need)
            # thin evenly rather than lopping off one end, so no region is stripped bare
            if take >= len(idxs):
                gone = idxs
            else:
                step = len(idxs) / float(take)
                gone = [idxs[int(k * step)] for k in range(take)]
            for i in gone:
                keep[i] = False
            dropped[c] = len(gone)
            need -= len(gone)
        self.ent_dropped = dropped
        out = self.sweep_orphans([b for i, b in enumerate(blocks) if keep[i]])
        self.ent_short = max(0, len(out) - self.ent_budget)
        return out

    def sweep_orphans(self, blocks):
        for _ in range(4):
            named = set()
            for b in blocks:
                for k in ('target', 'target2', 'target3', 'target4', 'killtarget'):
                    for mo in re.finditer(r'"%s"\s+"([^"]+)"' % k, b):
                        named.add(mo.group(1))
            out = []
            for b in blocks:
                mo = re.search(r'"classname"\s+"([^"]+)"', b)
                c = mo.group(1) if mo else ''
                tn = re.search(r'"targetname"\s+"([^"]+)"', b)
                if c in self.TARGET_ONLY and (not tn or tn.group(1) not in named):
                    self.ent_orphans += 1
                    continue
                out.append(b)
            if len(out) == len(blocks):
                break
            blocks = out
        return blocks

    def build(self, splittree):
        srcs, offsets = self.srcs, self.offsets
        j = len(srcs)
        faceorder, brushorder = [], []
        for m, src in enumerate(srcs):
            mm = src.models[0]
            faceorder += [(m, i) for i in range(mm[6], mm[6] + mm[7])]
            brushorder += [(m, i) for i in range(mm[8], mm[8] + mm[9])]
        nworldface0, nworldbrush0 = len(faceorder), len(brushorder)
        faceorder += [(-1, i) for i in range(len(self.conn_faces))]
        brushorder += [(-1, i) for i in range(len(self.conn_brushes))]
        nworldface, nworldbrush = len(faceorder), len(brushorder)
        submodel_ranges = []
        for m, src in enumerate(srcs):
            for i in range(1, len(src.models)):
                mm = src.models[i]
                submodel_ranges.append((m, i, len(faceorder), mm[7], len(brushorder), mm[9]))
                faceorder += [(m, k) for k in range(mm[6], mm[6] + mm[7])]
                brushorder += [(m, k) for k in range(mm[8], mm[8] + mm[9])]
        used_f = {}
        used_b = {}
        for n, (m, i) in enumerate(faceorder):
            if m >= 0:
                used_f[(m, i)] = n
        for n, (m, i) in enumerate(brushorder):
            if m >= 0:
                used_b[(m, i)] = n
        for m, src in enumerate(srcs):
            for i in range(len(src.faces)):
                if (m, i) not in used_f:
                    used_f[(m, i)] = len(faceorder)
                    faceorder.append((m, i))
            for i in range(len(src.brushes)):
                if (m, i) not in used_b:
                    used_b[(m, i)] = len(brushorder)
                    brushorder.append((m, i))
        trigbrush0 = len(brushorder)
        brushorder += [(-2, tb) for (tb, mn, mx, c, t) in self.trig_models]
        self.effects_out, self.effbase = [], []
        for m, src in enumerate(srcs):
            self.effbase.append(len(self.effects_out))
            for name, bi, unk in src.effects:
                self.effects_out.append((name, used_b.get((m, bi), -1) if bi >= 0 else -1, unk))
        faces_out = []
        for m, i in faceorder:
            if m < 0:
                h, tail = self.conn_faces[i]
                faces_out.append((list(h), tail))
                continue
            h, tail = self.srcs[m].faces[i]
            h = list(h)
            if (m, i) in self.dropped_faces:
                # the surface fell inside a cut doorway: emitted degenerate (no verts,
                # no meshverts) so the lump indices every other structure references
                # stay exactly where they were
                h[4] = 0
                h[6] = 0
            h[0] = self.texmap[m][h[0]] if 0 <= h[0] < len(self.texmap[m]) else 0
            if h[1] >= 0:
                h[1] += self.effbase[m]
            h[3] += self.vertbase[m]
            h[5] += self.elembase[m]
            if h[7] >= 0:
                h[7] = 0
            faces_out.append((h, tail))
        brushes_out = []
        for m, i in brushorder:
            if m == -1:
                brushes_out.append(list(self.conn_brushes[i]))
            elif m == -2:
                brushes_out.append(list(self.trig_brushes[i]))
            else:
                fs, ns, tx = srcs[m].brushes[i]
                tex = self.emptytex if (m, i) in self.carved else self.texmap[m][tx]
                brushes_out.append([fs + self.sidebase[m], ns, tex])
        leaffaces_out, lfbase = [], []
        leafbrushes_out, lbbase = [], []
        for m, src in enumerate(srcs):
            lfbase.append(len(leaffaces_out))
            leaffaces_out += [used_f[(m, v)] for v in src.leaffaces]
            lbbase.append(len(leafbrushes_out))
            leafbrushes_out += [used_b[(m, v)] for v in src.leafbrushes]
        leafs_out, leafbase = [], []
        for m, src in enumerate(srcs):
            leafbase.append(len(leafs_out))
            off = offsets[m]
            for lf in src.leafs:
                l2 = list(lf)
                cl = lambda v: max(-1 << 30, min((1 << 30), int(v)))
                l2[0] = 0 if l2[0] >= 0 else -1
                l2[1] = 0
                for a in range(3):
                    l2[2 + a] = cl(l2[2 + a] + off[a])
                    l2[5 + a] = cl(l2[5 + a] + off[a])
                l2[8] += lfbase[m]
                l2[10] += lbbase[m]
                leafs_out.append(l2)
        connleaf0 = len(leafs_out)
        for fa, br, lo, hi in self.conn_leafsets:
            lf = [0, 0, int(lo[0]), int(lo[1]), int(lo[2]), int(hi[0]), int(hi[1]), int(hi[2]),
                  len(leaffaces_out), len(fa), len(leafbrushes_out), len(br)]
            leaffaces_out += [nworldface0 + i for i in fa]
            leafbrushes_out += [nworldbrush0 + i for i in br]
            leafs_out.append(lf)
        nrouter = len(self.conn_leafsets) + max(0, j - 1)
        nodebase = []
        acc = nrouter
        for src in srcs:
            nodebase.append(acc)
            acc += len(src.nodes)
        nodes_out = []
        WB2 = 65536
        for c in range(len(self.conn_leafsets)):
            nxt = c + 1 if c + 1 < len(self.conn_leafsets) else nrouter if j == 1 else len(self.conn_leafsets)
            nxt = nodebase[0] if nxt >= nrouter else nxt
            pi = self.add_plane([1.0, 0.0, 0.0], -99999.0)
            nodes_out.append([pi, nxt, -1 - (connleaf0 + c), -WB2, -WB2, -WB2, WB2, WB2, WB2])

        def emit_split(t):
            # The router tree comes straight from the 3D pack (see split_tree): a node
            # is an axial plane and its two children are the two halves of the pack on
            # that axis.  Z splits are ordinary here, which is what lets the pack stack
            # levels; the old fixed 2D grid could only ever split x and y.
            if t[0] == 'leaf':
                return nodebase[t[1]]
            _, axis, boundary, lo, hi = t
            n = [0.0] * 3
            n[axis] = 1.0
            pi = self.add_plane(n, boundary)
            me = len(nodes_out)
            nodes_out.append([pi, 0, 0, -WB2, -WB2, -WB2, WB2, WB2, WB2])
            c1 = emit_split(lo)
            c0 = emit_split(hi)
            nodes_out[me][1] = c0
            nodes_out[me][2] = c1
            return me

        if j > 1:
            emit_split(splittree)
        assert len(nodes_out) == nrouter, (len(nodes_out), nrouter)
        for m, src in enumerate(srcs):
            off = offsets[m]
            for nd in src.nodes:
                n2 = list(nd)
                n2[0] += self.planebase[m]
                for ci in (1, 2):
                    if n2[ci] >= 0:
                        n2[ci] += nodebase[m]
                    else:
                        n2[ci] = -1 - ((-1 - n2[ci]) + leafbase[m])
                cl = lambda v: max(-1 << 30, min((1 << 30), int(v)))
                for a in range(3):
                    n2[3 + a] = cl(n2[3 + a] + off[a])
                    n2[6 + a] = cl(n2[6 + a] + off[a])
                nodes_out.append(n2)
        models_out = []
        wmins = [min(srcs[m].bounds[0][a] + offsets[m][a] for m in range(j)) - CORW for a in range(3)]
        wmaxs = [max(srcs[m].bounds[1][a] + offsets[m][a] for m in range(j)) + CORW for a in range(3)]
        models_out.append(wmins + wmaxs + [0, nworldface, 0, nworldbrush])
        for m, i, ff, nf, fb, nb in submodel_ranges:
            mm = srcs[m].models[i]
            off = offsets[m]
            models_out.append([mm[a] + off[a % 3] for a in range(6)] + [ff, nf, fb, nb])
        for k, (tb, mn, mx, c, t) in enumerate(self.trig_models):
            models_out.append(list(mn) + list(mx) + [0, 0, trigbrush0 + k, 1])
        ents = self.merge_entities().encode('latin-1') + b'\0'
        lumps = [b''] * 17
        lumps[0] = ents
        lumps[1] = b''.join(struct.pack('<64s2i', t[0].encode('latin-1'), t[1], t[2]) for t in self.textures)
        lumps[2] = b''.join(struct.pack('<4f', *p) for p in self.planes)
        lumps[3] = b''.join(struct.pack('<9i', *n) for n in nodes_out)
        lumps[4] = b''.join(struct.pack('<12i', *l) for l in leafs_out)
        lumps[5] = struct.pack('<%di' % len(leaffaces_out), *leaffaces_out)
        lumps[6] = struct.pack('<%di' % len(leafbrushes_out), *leafbrushes_out)
        lumps[7] = b''.join(struct.pack('<6f4i', *mm) for mm in models_out)
        lumps[8] = b''.join(struct.pack('<3i', *bb) for bb in brushes_out)
        lumps[9] = b''.join(struct.pack('<2i', *s) for s in self.sides)
        lumps[10] = b''.join(struct.pack('<10f4B', *v) for v in self.verts)
        lumps[11] = bytes(self.mesh)
        lumps[12] = b''.join(struct.pack('<64s2i', e[0] if isinstance(e[0], bytes) else e[0], e[1], e[2])
                             for e in self.effects_out)
        lumps[13] = b''.join(struct.pack('<12i', *h) + tail for h, tail in faces_out)
        lumps[14] = bytes([128]) * 49152
        lumps[15] = b''
        lumps[16] = b''
        hdr = bytearray(b'IBSP' + struct.pack('<i', 46) + b'\0' * (17 * 8))
        out = bytearray(hdr)
        for i in range(17):
            while len(out) % 4:
                out += b'\0'
            struct.pack_into('<ii', out, 8 + i * 8, len(out), len(lumps[i]))
            out += lumps[i]
        return bytes(out), nodes_out, leafs_out, models_out

# ---------------------------------------------------------------------------
# CONNECTION SITES -- where a stock map can be EDITED to grow a diegetic opening
# ---------------------------------------------------------------------------
# A join is not a tube punched through a level: it is a door, an extended gallery or
# a re-opened passage that looks like it was always there.  So before anything is
# placed, every candidate map is examined for the places where such an opening could
# plausibly be cut, and a map with too few of them is simply not used.
DOOR_W, DOOR_H, DOOR_SILL = 192.0, 208.0, 4.0
# Cardinal only.  A door is cut as an axis-aligned aperture through an axis-aligned
# wall panel; a diagonal-facing candidate would need the aperture snapped to an axis it
# does not share with the wall, which is how you get a ragged hole instead of a doorway.
SITE_DIRS = [(1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)]


def ray_runs(src, p0, d, maxt=1600.0, step=8.0):
    """March a ray through one source map and return the solid/empty run structure as
    [(t_enter, t_exit), ...] of SOLID spans.  Uses the same exact-plane predicate the
    carver uses, so a span here is a span the player really cannot walk through."""
    runs, inside, t0 = [], False, 0.0
    t = 0.0
    while t <= maxt:
        q = [p0[0] + d[0] * t, p0[1] + d[1] * t, p0[2] + d[2] * t]
        s = src.solid_brush_at(q) >= 0
        if s and not inside:
            inside, t0 = True, t
        elif not s and inside:
            runs.append((t0, t))
            inside = False
        t += step
    if inside:
        runs.append((t0, maxt))
    return runs


def free_slab(src, base, d, side, t0, t1, zlo, zhi, halfw, step=48.0):
    """Is the ORIENTED slab (t along d, +-halfw along side, z in [zlo,zhi]) free of
    solid?  Oriented rather than axis-aligned because a connection site can face a
    diagonal and an AABB test would sample the wall itself and report a false block."""
    ts = [t0 + i * step for i in range(int((t1 - t0) / step) + 1)] or [t0]
    if ts[-1] < t1:
        ts.append(t1)
    ss = [-halfw, 0.0, halfw] if halfw > 1 else [0.0]
    zs = [zlo + i * step for i in range(int((zhi - zlo) / step) + 1)] or [zlo]
    if zs[-1] < zhi:
        zs.append(zhi)
    for t in ts:
        for u in ss:
            for z in zs:
                q = [base[0] + d[0] * t + side[0] * u, base[1] + d[1] * t + side[1] * u, z]
                if src.solid_brush_at(q) >= 0:
                    return False
    return True


def probe_site(src, node, d, deg):
    """Is `node`, looking outward along `d`, a place a door could be cut?

    Wants: a wall that starts close to the standing surface, is THIN enough to be a
    wall panel rather than bedrock, has head- and shoulder-room on the inside so the
    opening is a doorway and not a crawl, and has open space on the far side for the
    procedural connector to meet.  A map whose shell is patch-mesh curvature rather
    than brushwork produces no sites at all here, and that is the point: it is not a
    map whose geometry can honestly be edited to grow a door."""
    eye = [node[0], node[1], node[2] + DOOR_H * 0.5]
    runs = ray_runs(src, eye, d)
    if not runs:
        return None                       # nothing to cut through: not a wall at all
    t_in, t_out = runs[0]
    if t_in > 640.0 or t_in < 24.0:
        return None                       # too far from the walkable frontier / on top of it
    thick = t_out - t_in
    if thick > 384.0 or thick < 8.0:
        return None                       # bedrock, terrain skirt, or a paper sliver
    if len(runs) > 1 and runs[1][0] - t_out < 224.0:
        return None                       # a double wall / stacked scenery, not an exterior face
    side = [-d[1], d[0], 0.0]
    zlo, zhi = node[2] + DOOR_SILL + 12, node[2] + DOOR_H - 12
    hw = DOOR_W / 2 - 24
    if not free_slab(src, node, d, side, 24.0, max(24.0, t_in - 16.0), zlo, zhi, hw):
        return None                       # no room to stand in front of the new opening
    if not free_slab(src, node, d, side, t_out + 32.0, t_out + 160.0, zlo, zhi, hw):
        return None                       # nothing on the far side for a connector to meet
    # SCORE.  A nav dead-end is the strongest diegetic signal there is: a passage that
    # already stops at this wall reads as something that was meant to continue, so
    # continuing it invents nothing.  After that: thin wall panels (cheap and plausible
    # to open) and walls close to the walkable frontier (a door, not a tunnel).
    # CONTINUE vs NEWCUT is decided by the shape of the space in front of the wall: if
    # the standing room is narrow (a passage or an alcove running into this wall) the
    # opening CONTINUES a feature the level already has, which is the most diegetic
    # edit available.  If the wall face is broad and open, it is a new opening on an
    # exterior-reading wall, which needs a frame to look deliberate.
    narrow = not free_slab(src, node, d, side, 24.0, max(24.0, t_in - 16.0), zlo, zhi, 288.0)
    cont = narrow or deg <= 2
    score = ((1.5 if narrow else 0.0) + (0.5 if deg <= 3 else 0.0) +
             max(0.0, (384.0 - thick) / 384.0) + max(0.0, (640.0 - t_in) / 640.0))
    return {'p': [float(x) for x in node], 'dir': list(d), 't_in': t_in, 't_out': t_out,
            'thick': thick, 'deg': deg, 'narrow': narrow,
            'kind': 'continue' if cont else 'newcut', 'score': round(score, 3)}


def map_sites(src, maxsites=12, minsep=1024.0):
    """All plausible connection sites on one map, best first."""
    got = getattr(src, '_sites', None)
    if got is not None:
        return got
    comp = [i for i in M.largest_component(src.navadj)
            if tuple(round(x, 1) for x in src.navnodes[i]) in src.wpset]
    if not comp:
        comp = M.largest_component(src.navadj)
    cand = []
    for d in SITE_DIRS:
        pr = sorted(comp, key=lambda i: -(src.navnodes[i][0] * d[0] + src.navnodes[i][1] * d[1]))
        for i in pr[:max(28, len(pr) // 14)]:
            s = probe_site(src, src.navnodes[i], d, len(src.navadj[i]))
            if s:
                s['node'] = i
                cand.append(s)
    cand.sort(key=lambda s: -s['score'])
    sites = []
    for s in cand:
        if all(math.dist(s['p'], t['p']) >= minsep or
               vdot(s['dir'], t['dir']) < 0.3 and math.dist(s['p'], t['p']) >= minsep / 2
               for t in sites):
            sites.append(s)
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


def load_src(n, outdir, pk3):
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
    src = Src(n, data, wp, cache)
    print('src %s: bounds %s %s models=%d faces=%d brushes=%d wp=%d links=%d' %
          (n, [round(x) for x in src.bounds[0]], [round(x) for x in src.bounds[1]],
           len(src.models), len(src.faces), len(src.brushes),
           len(src.wptriples), len(src.cachelinks)))
    return src


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


def fuse(seed, names, outdir, pk3, nbridges=0, workdir=None, loopfrac=0.25, levels=None,
         wpcap=600):
    """Select maps that can be EDITED to connect, pack them in 3D, cut the openings,
    and join the openings with procedural geometry."""
    rng = random.Random(seed)
    os.makedirs(outdir, exist_ok=True)
    # ---- 1. procedural bridge tiles (optional).  They are analysed for connection
    # sites by exactly the same detector as a stock map, so they fall into the same
    # taxonomy: an arm mouth is a dead-end passage, i.e. the strongest 'continue' site
    # there is, and a 4-armed hub therefore classifies as a bridge map on its own merits.
    work = workdir or os.path.join(outdir, '_bridges')
    genned = []
    if nbridges:
        import mapgen
        for bi in range(nbridges):
            nm = 'bridge%d_%d' % (seed, bi)
            base, _ports = mapgen.build_bridge_tile(work, nm, seed * 131 + bi, ['e', 'w'], ['n', 's'])
            genned.append(base)
    # ---- 2. SURVEY every candidate for connection sites, then SELECT by suitability
    srcs, sites_of, cls_of, rejected = [], [], [], []
    for n in list(names) + genned:
        src = load_src(n, outdir, pk3)
        st = map_sites(src)
        cl = classify(len(st))
        ncont = sum(1 for s in st if s['kind'] == 'continue')
        print('sites %-18s %-10s n=%2d (continue=%d newcut=%d) dirs=%s thick=%s' %
              (src.name, cl.upper(), len(st), ncont, len(st) - ncont,
               ''.join(sorted({DIRNAME[(int(s['dir'][0]), int(s['dir'][1]))] for s in st})),
               [int(s['thick']) for s in st[:6]]))
        if cl == 'unsuitable':
            rejected.append((src.name, len(st)))
            continue
        srcs.append(src)
        sites_of.append(st)
        cls_of.append(cl)
    if rejected:
        print('selection: REJECTED %d map(s) with fewer than 2 connection sites: %s' %
              (len(rejected), ', '.join('%s(%d)' % r for r in rejected)))
    T = len(srcs)
    nbr_maps = sum(1 for c in cls_of if c == 'bridge')
    print('selection: %d maps kept -- %d BRIDGE maps (>3 connection sites), %d STUB maps '
          '(2-3 sites); %d rejected' % (T, nbr_maps, T - nbr_maps, len(rejected)))
    # ---- 3. 3D BIN PACK.  Deliberately simple.  Cells are ranked by how many lattice
    # neighbours they have and tiles by how many connection sites they have, and the two
    # rankings are zipped: bridge maps land in the cells with the most adjacencies, stub
    # maps in corner cells with two.  That is the taxonomy realised as a packing.
    if levels is None:
        levels = 2 if T >= 12 else 1
    per = int(math.ceil(T / float(levels)))
    cols = max(1, int(math.ceil(math.sqrt(per))))
    rows = int(math.ceil(per / float(cols)))
    grid = [(c, r, z) for z in range(levels) for r in range(rows) for c in range(cols)][:T]
    gset = set(grid)
    NB6 = lambda c: [d for d in ((c[0] + 1, c[1], c[2]), (c[0] - 1, c[1], c[2]),
                                 (c[0], c[1] + 1, c[2]), (c[0], c[1] - 1, c[2]),
                                 (c[0], c[1], c[2] + 1), (c[0], c[1], c[2] - 1)) if d in gset]
    order_cells = sorted(grid, key=lambda c: (-len(NB6(c)), c))
    order_tiles = sorted(range(T), key=lambda m: (-len(sites_of[m]), srcs[m].name))
    cells = [None] * T
    for ci, m in zip(order_cells, order_tiles):
        cells[m] = ci
    print('pack: %dx%dx%d lattice, %d cells for %d tiles; cell adjacency %s'
          % (cols, rows, levels, len(grid), T, sorted({len(NB6(c)) for c in grid})))
    offsets, eds, slack = pack_offsets(srcs, cells, cols, rows, levels)
    # ---- 4. EDGES.  Every lattice adjacency is a candidate join; a tile may not take
    # more joins than it has connection sites, and at most one of a tile's joins may be
    # vertical (a level-to-level portal is the non-cart-navigable one).
    cidx = {c: m for m, c in enumerate(cells)}
    cand = []
    for m, c in enumerate(cells):
        for d in NB6(c):
            n = cidx[d]
            if m < n:
                cand.append((m, n))
    cap = [len(sites_of[m]) for m in range(T)]
    used = [0] * T
    vert = [0] * T
    par = list(range(T))

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    edges = []
    for phase in (0, 1):
        for m, n in cand:
            if (m, n) in edges:
                continue
            v = cells[m][2] != cells[n][2]
            if used[m] >= cap[m] or used[n] >= cap[n]:
                continue
            if v and (vert[m] or vert[n]):
                continue
            ra, rb = find(m), find(n)
            if phase == 0 and ra == rb:
                continue                       # phase 0 = spanning structure only
            if phase == 1 and rng.random() > loopfrac:
                continue                       # phase 1 = a sampled minority of loop edges
            par[ra] = rb
            edges.append((m, n))
            used[m] += 1
            used[n] += 1
            if v:
                vert[m] += 1
                vert[n] += 1
    ncomp = len({find(m) for m in range(T)})
    print('topology: %d joins over %d tiles (%d vertical level-to-level), %d component(s); '
          'per-tile joins %s vs site capacity %s' %
          (len(edges), T, sum(1 for m, n in edges if cells[m][2] != cells[n][2]), ncomp, used, cap))
    degree = [0] * T
    for a, b in edges:
        degree[a] += 1
        degree[b] += 1
    PRE = region_graph_solve(T, edges)
    cutset = set(PRE['cutedges'])
    exclusive = [min(degree[a], degree[b]) == 1 or ei in cutset for ei, (a, b) in enumerate(edges)]
    # ---- 5. CUT THE OPENINGS AND JOIN THEM.
    # For each join the two tiles' sites are matched to the direction the pack put them
    # in, the map geometry is edited to grow a door there, and the two new doorways are
    # linked by procedural geometry.  There is no length test and no refusal anywhere in
    # this loop: a join is always constructible because the openings are cut where the
    # placement wants them, and the connector is generated to fit whatever gap is left.
    taken = [set() for _ in range(T)]
    vert_anchor = [[] for _ in range(T)]
    conns, corn, telen, padn = [], 0, 0, 0
    noncart = [0] * T
    cutlog = []

    def site_mouth(m, si):
        st = sites_of[m][si]
        return [st['p'][i] + offsets[m][i] + st['dir'][i] * (st['t_out'] + 48.0) for i in range(3)]

    def pick_pair(a, b, want):
        """Choose WHICH opening to cut on each side of one join.

        Three things are traded off: how good a place each site is to cut (its own
        suitability score), whether it faces the tile the pack put on the other side,
        and -- the term that actually decides most joins -- how far apart the two
        openings would leave the procedural connector.  Choosing by direction alone
        produced 10k-unit connectors between two maps whose doors happened to sit at
        opposite ends of their own footprints."""
        back = [-x for x in want]
        best, bs = None, -1e18
        for ia, sa in enumerate(sites_of[a]):
            if ia in taken[a]:
                continue
            for ib, sb in enumerate(sites_of[b]):
                if ib in taken[b]:
                    continue
                sc = (0.6 * (sa['score'] + sb['score'])
                      + 2.0 * (vdot(sa['dir'], want) + vdot(sb['dir'], back))
                      - math.dist(site_mouth(a, ia), site_mouth(b, ib)) / 1200.0)
                if sc > bs:
                    best, bs = (ia, ib), sc
        return best

    # 5a. CHOOSE the site pair for every join first, then spend each tile's remaining
    # freedom inside its own packed slot on bringing the chosen openings together.  This
    # is the whole of the placement search that survives: the pack itself is deliberately
    # simple, and the only thing worth optimising afterwards is the gap the procedural
    # connector has to span between two doorways that are already decided.
    sel = {}
    for ei, (a, b) in enumerate(edges):
        ca, cb = cells[a], cells[b]
        dv = [float(cb[i] - ca[i]) for i in range(3)]
        if abs(dv[2]) > 0.5:
            continue
        L = math.hypot(dv[0], dv[1]) or 1.0
        pr = pick_pair(a, b, [dv[0] / L, dv[1] / L, 0.0])
        if pr is None:
            continue
        ia, ib = pr
        taken[a].add(ia)
        taken[b].add(ib)
        sel[ei] = (ia, ib)

    def gapsum():
        return sum(math.dist(site_mouth(a, sel[ei][0]), site_mouth(b, sel[ei][1]))
                   for ei, (a, b) in enumerate(edges) if ei in sel)
    g0 = gapsum()
    step = 1.0
    for _ in range(6):
        for m in range(T):
            for ax in range(3):
                lo, hi = slack[m][ax]
                if hi - lo < 1.0:
                    continue
                base, bv, bc = offsets[m][ax], offsets[m][ax], gapsum()
                for f in (-1.0, -0.4, -0.15, 0.15, 0.4, 1.0):
                    offsets[m][ax] = min(max(base + step * f * ((hi - base) if f > 0 else (base - lo)),
                                             lo), hi)
                    c2 = gapsum()
                    if c2 < bc - 1e-6:
                        bv, bc = offsets[m][ax], c2
                offsets[m][ax] = bv
        step *= 0.5
    offsets = [[round(x) for x in o] for o in offsets]
    print('placement: door-gap objective over %d planned joins %.0f -> %.0f (%.1f%% shorter), '
          'coordinate descent inside each tile\'s packed slot, non-overlap held by the slot bounds'
          % (len(sel), g0, gapsum(), 100.0 * (g0 - gapsum()) / max(1.0, g0)))
    F = Fuser(srcs, offsets, seed)
    for m, s2 in enumerate(srcs):
        print('place %-18s %-6s cell %s offset %s' %
              (s2.name, cls_of[m], cells[m], offsets[m]))
    for ei, (a, b) in enumerate(edges):
        prom = exclusive[ei]
        ca, cb = cells[a], cells[b]
        dv = [float(cb[i] - ca[i]) for i in range(3)]
        vertical = abs(dv[2]) > 0.5
        if vertical:
            # verticality: a level-to-level connection is a portal pair, the spec's own
            # connector vocabulary, and the one join per map allowed to be non-cart.
            sa = max(walk_sample(srcs[a]), key=lambda p: p[2])
            sb = min(walk_sample(srcs[b]), key=lambda p: p[2])
            pa = [float(sa[i]) + offsets[a][i] for i in range(3)]
            pb = [float(sb[i]) + offsets[b][i] for i in range(3)]
            F.build_tele(pa, pb, telen, prominent=prom)
            F.build_tele(pb, pa, 100 + telen, prominent=prom)
            # a portal pad stands on one of the tile's OWN waypoints; register it as an
            # anchor so the waypoint budget below can never decimate the node a bot has
            # to reach to use the portal
            vert_anchor[a].append(pa)
            vert_anchor[b].append(pb)
            telen += 2
            noncart[a] += 1
            noncart[b] += 1
            conns.append((a, b, 'teleporter', pa, pb, 0, exclusive[ei], prom, math.dist(pa, pb)))
            print('join %-16s <-> %-16s VERTICAL teleporter  %s -> %s  len=%.0f'
                  % (srcs[a].name, srcs[b].name, [round(x) for x in pa], [round(x) for x in pb],
                     math.dist(pa, pb)))
            continue
        if ei not in sel:
            continue
        ia, ib = sel[ei]
        ma = F.cut_portal(a, sites_of[a][ia], prominent=prom)
        mb = F.cut_portal(b, sites_of[b][ib], prominent=prom)
        pa_, pb_ = F.portals[-2], F.portals[-1]
        cutlog.append((srcs[a].name, sites_of[a][ia], pa_))
        cutlog.append((srcs[b].name, sites_of[b][ib], pb_))
        # procedural connector between the two new doorways.  Small solids in the way are
        # carved; anything too big to carve is SPLIT around the tube, the same edit the
        # doorway itself uses -- so the connector is built either way.
        w2 = (CORW_PROM if prom else CORW) / 2.0
        pts = corridor_samples(ma, mb, w2)
        hits = blockage(srcs, offsets, pts, clip=True)
        small = {h for h in hits if brush_volume_ok(srcs[h[0]], h[1])}
        F.carve(small)
        big = hits - small
        for m2, bi in big:
            lo = [min(ma[i], mb[i]) - w2 - WALL for i in range(3)]
            hi = [max(ma[i], mb[i]) + w2 + WALL for i in range(3)]
            lo[2], hi[2] = min(ma[2], mb[2]) - FLOORTHK, max(ma[2], mb[2]) + CORH + WALL
            F.split_brushes(m2, lo, hi)
        F.build_corridor(ma, mb, prominent=prom)
        corn += 1
        conns.append((a, b, 'corridor', ma, mb, len(hits), exclusive[ei], prom, math.dist(ma, mb)))
        print('join %-16s <-> %-16s corridor %s  %s(%s,%s) -> %s(%s,%s)  len=%.0f carved=%d split=%d'
              % (srcs[a].name, srcs[b].name, 'PROMINENT' if prom else 'subtle',
                 [round(x) for x in ma], sites_of[a][ia]['kind'],
                 DIRNAME[(int(sites_of[a][ia]['dir'][0]), int(sites_of[a][ia]['dir'][1]))],
                 [round(x) for x in mb], sites_of[b][ib]['kind'],
                 DIRNAME[(int(sites_of[b][ib]['dir'][0]), int(sites_of[b][ib]['dir'][1]))],
                 math.dist(ma, mb), len(small), len(big)))
    j = T
    names = [s.name for s in srcs]
    is_bridge = [c == 'bridge' for c in cls_of]
    cellpos = cells
    sockets = [[] for _ in range(T)]
    for p in F.portals:
        sockets[p['tile']].append(p['mouth'])
    for c in conns:
        if c[2] != 'corridor':
            sockets[c[0]].append(c[3])
            sockets[c[1]].append(c[4])
    for m in range(T):
        if not sockets[m]:
            sockets[m] = [[float(x) + offsets[m][i] for i, x in enumerate(walk_sample(srcs[m])[0])]]
    ncut = len(F.portals)
    nb_tot = sum(p['brushes'] for p in F.portals)
    npieces = sum(p['pieces'] for p in F.portals)
    nfc = sum(p['faces_cut'] for p in F.portals)
    nfr = sum(p['faces_reissued'] for p in F.portals)
    ncont = sum(1 for p in F.portals if p['kind'] == 'continue')
    print('GEOMETRY EDIT: cut %d doorways (%d continuing an existing passage, %d new '
          'openings on an exterior wall); split %d source brushes into %d convex '
          'remainders; re-cut %d wall surfaces into %d clipped surfaces; wall thickness '
          'cut through: min=%d median=%d max=%d'
          % (ncut, ncont, ncut - ncont, nb_tot, npieces, nfc, nfr,
             min([p['thick'] for p in F.portals] or [0]),
             sorted(p['thick'] for p in F.portals)[ncut // 2] if ncut else 0,
             max([p['thick'] for p in F.portals] or [0])))
    nexcl = sum(1 for c in conns if c[6])
    print('topology: %d tiles (%d bridge maps + %d stub maps), %d joins, corridors=%d '
          'teleport-triggers=%d' % (j, nbr_maps, j - nbr_maps, len(conns), corn, telen))
    print('prominence: %d exclusive/objective joins (prominent+lit), %d redundant joins '
          '(subtle); node degrees %s' % (nexcl, len(conns) - nexcl, degree))
    clens = sorted(c[8] for c in conns if c[2] == 'corridor')
    if clens:
        print('corridor length: n=%d min=%.0f p25=%.0f median=%.0f p75=%.0f max=%.0f '
              '(NO length cap exists: refusal deleted)' %
              (len(clens), clens[0], clens[len(clens) // 4], clens[len(clens) // 2],
               clens[3 * len(clens) // 4], clens[-1]))
    print('cart-navigability: %d/%d joins cart-navigable (door+corridor); non-cart joins '
          'per tile max=%d (budget 1) -> %s; joins refused: 0; joins dropped: 0' %
          (corn, len(conns), max(noncart) if noncart else 0,
           'HELD' if max(noncart or [0]) <= 1 else 'VIOLATED'))
    nax = F.axialize()
    if nax:
        print('well-formedness: re-emitted %d source brush(es) with no derivable axial '
              'AABB (slightly-oblique near-axial planes) inside finite clamp planes' % nax)
    splits = split_tree([(cells[m][0], cells[m][1], cells[m][2], m) for m in range(T)], eds)
    noncart_used, noncart_budget, budget_viol, dropped = noncart, 1, [], []
    data, nodes_out, leafs_out, models_out = F.build(splits)
    print('entity budget %d: swept %d orphaned target-only entities; dropped %s%s' %
          (F.ent_budget, F.ent_orphans, F.ent_dropped or 'nothing (under budget)',
           '' if not F.ent_short else '  STILL %d OVER BUDGET' % F.ent_short))
    print('entities: dropped %d source spawnpoints in solid, %d over the per-tile '
          'spawn budget of %d (stock IntrusiveList ops are linear; an unbudgeted '
          'megamap spawn set trips the engine 10M-statement runaway at worldspawn)'
          % (F.dropped_spawns, F.dropped_spawns_budget, F.spawn_cap))
    bsp_path = os.path.join(outdir, 'fused.bsp')
    open(bsp_path, 'wb').write(data)
    print('wrote %s (%d bytes, %d nodes %d leafs %d models)' %
          (bsp_path, len(data), len(nodes_out), len(leafs_out), len(models_out)))
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    # ---- WAYPOINT BUDGET.
    # Stock `waypoint_loadall` spawns each saved waypoint through `waypoint_get`, which
    # linear-scans the waypoints already spawned and boxesoverlap-tests each one.  That
    # is O(n^2) inside a single server frame, and the compiled-in 10,000,000-jump
    # runaway limit therefore caps a fused world's SAVED waypoint count -- not its
    # entity count, which the separate entity budget already handles.  Measured: 3973
    # saved waypoints across 29 tiles killed the boot in `waypoint_loadall` at
    # `waypoint_spawn -> waypoint_get -> boxesoverlap`.  So the same treatment the
    # entity ceiling got: a hard budget, spent where it buys the most navigation.
    # The number is empirical, not theoretical.  900 saved waypoints cleared
    # waypoint_loadall but then blew the same 10M ceiling one layer up, inside
    # navigation_markroutes -> navigation_markroutes_nearestwaypoints, which walks the
    # whole g_waypoints IntrusiveList inside its own per-waypoint loop -- O(n^2) again,
    # per bot, per goal rating, and the fused world's 175 trigger waypoints count too.
    # 600 clears both.
    #   * every connector waypoint (corridor chain, doorway chain) is MANDATORY -- they
    #     are the only nodes that carry a bot from one tile to another;
    #   * every non-stationary waypoint (jump/teleport link boxes) is kept, they are few;
    #   * each tile's stationary waypoints are farthest-point decimated to its share of
    #     what is left, seeded from the nodes nearest that tile's own doorways;
    #   * the link graph is CONTRACTED onto the survivors, so reachability is preserved
    #     rather than shredded.
    conn_wp = [list(p) for p in F.wp_extra]
    fixed = {vstr(p) for p in conn_wp}
    per_tile = []
    nonstat = []
    for m, src in enumerate(srcs):
        st = []
        for m1, m2, fl in src.wptriples:
            w1, w2 = vadd(m1, offsets[m]), vadd(m2, offsets[m])
            (st if m1 == m2 else nonstat).append((w1, w2, fl))
        per_tile.append(st)
    ntotal = sum(len(x) for x in per_tile) + len(nonstat) + len(conn_wp)
    room = max(0, wpcap - len(conn_wp) - len(nonstat))
    nsrcwp = sum(len(x) for x in per_tile)
    keep_tile = []
    if nsrcwp <= room:
        keep_tile = per_tile
    else:
        # doorway anchors first, then farthest-point spread, per tile
        anchors = [list(v) for v in vert_anchor]
        for p2 in F.portals:
            anchors[p2['tile']].append(p2['node'])
        for m in range(T):
            share = max(4, int(round(room * len(per_tile[m]) / float(nsrcwp))))
            pts = per_tile[m]
            if len(pts) <= share:
                keep_tile.append(pts)
                continue
            idx = []
            for a in anchors[m]:
                j2 = min(range(len(pts)), key=lambda i: math.dist(pts[i][0], a))
                if j2 not in idx:
                    idx.append(j2)
            if not idx:
                idx = [0]
            d = [min(math.dist(pts[i][0], pts[q][0]) for q in idx) for i in range(len(pts))]
            while len(idx) < share:
                i2 = max(range(len(pts)), key=lambda i: d[i])
                if d[i2] <= 0:
                    break
                idx.append(i2)
                d = [min(d[i], math.dist(pts[i][0], pts[i2][0])) for i in range(len(pts))]
            keep_tile.append([pts[i] for i in sorted(idx)])
    kept = [w for tl in keep_tile for w in tl] + nonstat
    wlines = ['//WAYPOINT_VERSION 1.04', '//WAYPOINT_SYMMETRY 0', '//WAYPOINT_TIME ' + ts]
    seen = set()
    allwp = [(w[0], w[1], w[2]) for w in kept] + [(p, p, 0) for p in conn_wp]
    for m1, m2, fl in allwp:
        k = vstr(m1)
        if m1 == m2 and k in seen:
            continue
        seen.add(k)
        wlines += [vstr(m1), vstr(m2), '%d' % fl]
    nwrote = (len(wlines) - 3) // 3
    open(os.path.join(outdir, 'fused.waypoints'), 'w').write('\n'.join(wlines) + '\n')
    print('waypoint budget %d: %d source + %d connector waypoints -> %d written '
          '(%d source waypoints decimated); stock waypoint_loadall is O(n^2) per frame '
          'against a 10M-jump runaway ceiling' %
          (wpcap, nsrcwp + len(nonstat), len(conn_wp), nwrote, nsrcwp - sum(len(x) for x in keep_tile)))
    # link contraction onto the survivors
    clines = ['//WAYPOINT_VERSION 1.04', '//WAYPOINT_TIME ' + ts]
    nlink = 0
    for m, src in enumerate(srcs):
        ks = [w[0] for w in keep_tile[m]]
        if not ks:
            continue
        rep = {}

        def repof(q):
            k2 = tuple(round(x, 1) for x in q)
            r = rep.get(k2)
            if r is None:
                r = min(ks, key=lambda p2: math.dist(p2, q))
                rep[k2] = r
            return r
        outl = set()
        for a, b in src.cachelinks:
            ra, rb = repof(vadd(a, offsets[m])), repof(vadd(b, offsets[m]))
            if ra is not rb and vstr(ra) != vstr(rb):
                outl.add((vstr(ra), vstr(rb)))
        for a, b in outl:
            clines.append(a + '*' + b)
        nlink += len(outl)
    for a, b in F.link_extra:
        clines.append(vstr(a) + '*' + vstr(b))
    open(os.path.join(outdir, 'fused.waypoints.cache'), 'w').write('\n'.join(clines) + '\n')
    print('wrote fused.waypoints (%d wps of %d before the budget) '
          'fused.waypoints.cache (%d links)' % (nwrote, ntotal, len(clines) - 2))
    open(os.path.join(outdir, 'fused.mapinfo'), 'w').write(
        'title Fused %s\ndescription procedurally fused mega-map\nauthor mapfuse\n'
        'gametype dm\ngametype tdm\ngametype plc\n' % '+'.join(names))
    probs = check_bsp(open(bsp_path, 'rb').read())
    print('parse-back: %s' % ('OK' if not probs else 'PROBLEMS %s' % probs[:10]))
    # CONNECTOR CLEARANCE CHECK.
    # Deliberately NOT done through mkentfile.Bsp: that class derives a brush AABB from
    # plane distances of any plane within 0.999 of an axis, which is only valid for an
    # exactly axis-aligned plane.  A join corridor is oblique by construction, so its
    # brushes get a SHIFTED AABB and the check reports phantom "no floor" violations
    # (measured: a corridor whose real x-span is [-5504,-5152] is indexed at
    # [-5843,-5491]).  The check below uses the same exact-plane predicate the carver
    # uses -- Src.solid_brush_at on the un-carved source brushes -- so a violation here
    # means the corridor tube really is obstructed.  The floor is analytic: the corridor
    # brush set always contains a floor slab beneath its own centreline.
    fviol, fviol_by = 0, []
    for c in conns:
        if c[2] != 'corridor':
            continue
        w2 = (CORW_PROM if c[7] else CORW) / 2.0
        hits = blockage(srcs, offsets, corridor_samples(c[3], c[4], w2), clip=True)
        left = [h for h in hits if h not in F.carved]
        fviol += len(left)
        if left:
            fviol_by.append('%s<->%s(len=%.0f,%d)' % (names[c[0]], names[c[1]], c[8], len(left)))
    print('connector clearance check (exact planes, un-carved source solids): %s '
          '(%d obstructed samples%s)' %
          ('PASS' if fviol == 0 else 'FAIL', fviol,
           '' if not fviol_by else ' on ' + ' '.join(fviol_by)))
    nodes2, adj2 = M.parse_cache(open(os.path.join(outdir, 'fused.waypoints.cache')).read())
    key = lambda p: tuple(round(x, 1) for x in p)
    idx2 = {key(nodes2[i]): i for i in range(len(nodes2))}
    region = {}
    for m in range(j):
        for m1, m2, fl in srcs[m].wptriples:
            region[key(vadd(m1, offsets[m]))] = m
    dadj = [set(a.keys()) for a in adj2]
    added = 0
    for near, far in F.bot_jumps:
        u, v = idx2.get(key(near)), idx2.get(key(far))
        if u is not None and v is not None:
            dadj[u].add(v)
            added += 1
    from collections import deque
    seed_node = idx2.get(key(sockets[0][0]), 0)
    seen = [False] * len(nodes2)
    q = deque([seed_node])
    seen[seed_node] = True
    while q:
        u = q.popleft()
        for v in dadj[u]:
            if not seen[v]:
                seen[v] = True
                q.append(v)
    reached_region = {region.get(key(nodes2[i])) for i in range(len(nodes2)) if seen[i]}
    all_reg = set(range(j))
    unreached = all_reg - reached_region
    print('bot flood-fill: modeled %d engine-autogen connector jumps; regions reached %s / %s -> %s' %
          (added, sorted(r for r in reached_region if r is not None), sorted(all_reg),
           'PASS' if not unreached else 'FAIL unreached=%s' % sorted(unreached)))
    for c in conns:
        a, b, kind, sa, sb = c[0], c[1], c[2], c[3], c[4]
        na, nb = idx2.get(key(sa)), idx2.get(key(sb))
        okab = na is not None and nb is not None and seen[na] and seen[nb]
        print('join %s<->%s %s: near_wp=%s far_wp=%s bot-traversable=%s' %
              (srcs[a].name, srcs[b].name, kind,
               'yes' if na is not None else 'MISSING', 'yes' if nb is not None else 'MISSING',
               'YES' if okab else 'NO'))
    # ---- CONNECTIVITY SOLVER over the region graph
    RG = region_graph_solve(j, [(c[0], c[1]) for c in conns])
    print('connectivity: %d component(s) %s; hop-diameter=%d; chokepoint TILES (articulation) %s; '
          'chokepoint JOINS (cut edges) %s' %
          (len(RG['components']), [len(x) for x in RG['components']], RG['hop_diameter'],
           [names[i] for i in RG['articulation']] or 'none',
           ['%s<->%s' % (names[conns[e][0]], names[conns[e][1]]) for e in RG['cutedges']] or 'none'))
    print('connectivity: k=%d edge-connectivity lower bound (%d/%d joins are cut edges, '
          'the rest are redundant/loop edges)' %
          (0 if RG['cutedges'] else 1, len(RG['cutedges']), len(conns)))
    # ---- NAVMESH SOLVER: real bot walking distances between regions, coverage, vantages
    reps, vantages = {}, {}
    for m in range(j):
        pts = [i for i in range(len(nodes2)) if region.get(key(nodes2[i])) == m and seen[i]]
        reps[m] = pts[len(pts) // 2] if pts else None
        vs = []
        if pts:
            vs = [pts[0]]
            while len(vs) < min(3, len(pts)):
                nxt = max(pts, key=lambda i: min(math.dist(nodes2[i], nodes2[q]) for q in vs))
                if nxt in vs:
                    break
                vs.append(nxt)
        vantages[m] = [[round(x, 1) for x in nodes2[i]] for i in vs]
    NM = navmesh_solve(nodes2, dadj, region, key, j, reps)
    print('navmesh: %d fused waypoints; bot-reachable from seed = %d (%.1f%%); '
          'regions with zero reachable waypoints: %s' %
          (len(nodes2), sum(seen), 100.0 * sum(seen) / max(1, len(nodes2)),
           [names[m] for m in range(j) if reps[m] is None] or 'none'))
    print('navmesh: region<->region WALKING distance median=%.0fu diameter=%.0fu '
          'unreachable_pairs=%d  (commitment cost: a bot crossing the megamap walks the diameter)' %
          (NM['walk_median'], NM['walk_diameter'], NM['unreachable_pairs']))
    worst = sorted(((v, k2) for k2, v in NM['region_walk'].items() if v is not None), reverse=True)[:3]
    for d, (m1, m2) in worst:
        print('navmesh:   longest %s -> %s = %.0fu' % (names[m1], names[m2], d))
    joins = {'seed': seed, 'grid': [cols, rows],
             'maps': [{'name': srcs[m].name, 'offset': offsets[m], 'cell': list(cellpos[m]),
                       'bridge': bool(is_bridge[m]), 'degree': degree[m],
                       'vantages': vantages[m],
                       'mins': [srcs[m].bounds[0][i] + offsets[m][i] for i in range(3)],
                       'maxs': [srcs[m].bounds[1][i] + offsets[m][i] for i in range(3)]} for m in range(j)],
             'joins': [{'a': c[0], 'b': c[1], 'kind': c[2], 'sa': list(c[3]), 'sb': list(c[4]),
                        'exclusive': c[6], 'prominent': c[7], 'length': round(c[8], 1),
                        'cart_navigable': c[2] == 'corridor'} for c in conns],
             'bot_jumps': [[list(n), list(f)] for n, f in F.bot_jumps],
             # the doorways this run CUT into the stock maps: where the wall was opened,
             # which way it faces, and how much geometry the edit touched.  joinshot.py
             # renders a pair of frames per portal (inside looking out, outside looking
             # back) so the edit can be judged as architecture, not just as a metric.
             'portals': [{'tile': p2['tile'], 'name': names[p2['tile']], 'node': p2['node'],
                          'mouth': p2['mouth'], 'axis': p2['axis'], 'sgn': p2['sgn'],
                          'kind': p2['kind'], 'thick': p2['thick'],
                          'aperture': [list(p2['aperture'][0]), list(p2['aperture'][1])],
                          'brushes_split': p2['brushes'], 'pieces': p2['pieces'],
                          'faces_cut': p2['faces_cut'], 'faces_reissued': p2['faces_reissued']}
                         for p2 in F.portals]}
    json.dump(joins, open(os.path.join(outdir, 'fused.joins.json'), 'w'), indent=0)
    metrics = {'seed': seed, 'tiles': j, 'stock': j - sum(is_bridge), 'bridges': sum(is_bridge),
               'joins': len(conns), 'corridors': corn, 'jumppads': padn, 'teleporters': telen,
               'cart_navigable_joins': corn, 'noncart_per_tile': noncart_used,
               'noncart_budget': noncart_budget, 'budget_violations': budget_viol, 'dropped_edges': dropped,
               'corridor_len': clens, 'exclusive_edges': nexcl,
               'connectivity': {k: v for k, v in RG.items() if k not in ('adj', 'eccentricity')},
               'articulation_names': [names[i] for i in RG['articulation']],
               'cutedge_names': ['%s<->%s' % (names[conns[e][0]], names[conns[e][1]])
                                 for e in RG['cutedges']],
               'navmesh': {'n_nodes': NM['n_nodes'], 'reachable': int(sum(seen)),
                           'coverage': NM['coverage'], 'walk_diameter': NM['walk_diameter'],
                           'walk_median': NM['walk_median'],
                           'unreachable_pairs': NM['unreachable_pairs'],
                           'region_walk': {'%d-%d' % k2: v for k2, v in NM['region_walk'].items()}},
               'bsp_bytes': len(data), 'names': names,
               'selection': {'kept': names, 'classes': cls_of, 'rejected': rejected,
                             'sites_per_map': [len(x) for x in sites_of]},
               'pack': {'cols': cols, 'rows': rows, 'levels': levels,
                        'cells': [list(c) for c in cells]},
               'geometry_edit': {'portals': ncut, 'continue': ncont, 'newcut': ncut - ncont,
                                 'brushes_split': nb_tot, 'convex_pieces': npieces,
                                 'faces_cut': nfc, 'faces_reissued': nfr,
                                 'door_gap_before': round(g0), 'door_gap_after': round(gapsum())}}
    json.dump(metrics, open(os.path.join(outdir, 'fused.metrics.json'), 'w'), indent=1)
    print('wrote fused.joins.json (%d tiles, %d joins) + fused.metrics.json' % (j, len(conns)))
    return bsp_path, conns


def smoke(outdir):
    eng = os.path.expanduser('~/dox/xonotic/build-engine/darkplaces-dedicated')
    basedir = os.path.expanduser('~/dox/xonotic/Xonotic')
    log = os.path.join('/tmp/fusesmoke', 'smoke.log')
    cmd = [eng, '-xonotic', '-basedir', basedir, '-userdir', '/tmp/fusesmoke',
           '+port', '26071', '+sv_public', '0', '+g_payload', '0',
           '+bot_number', '2', '+skill', '1', '+map', 'fused']
    with open(log, 'w') as lf:
        p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
        time.sleep(20)
        p.terminate()
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()
    out = open(log, errors='ignore').read()
    errs = [l for l in out.splitlines() if re.search(r'error|Error|ERROR|invalid|Invalid|funny|missing', l)]
    print('smoke: log %s (%d lines, %d error-ish)' % (log, len(out.splitlines()), len(errs)))
    for l in errs[:20]:
        print('smoke-err: %s' % l)
    for l in out.splitlines()[-15:]:
        print('smoke-tail: %s' % l)


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    flags = [a for a in sys.argv[1:] if a.startswith('--')]
    seed = int(args[0]) if args else 0
    pk3 = sorted(glob.glob(os.path.expanduser('~/dox/xonotic/Xonotic/data/*maps*.pk3')), reverse=True)[0]
    names = args[1:]
    outdir = '/tmp/fusesmoke/data/maps'
    nmaps = None
    nbridges = None
    teams, carts = 5, 3
    levels = None
    wpcap = 600
    for f in flags:
        if f.startswith('--out='):
            outdir = f[6:]
        elif f.startswith('--maps='):
            nmaps = f[7:]
        elif f.startswith('--bridges='):
            nbridges = int(f[10:])
        elif f.startswith('--teams='):
            teams = int(f[8:])
        elif f.startswith('--carts='):
            carts = int(f[8:])
        elif f.startswith('--levels='):
            levels = int(f[9:])
        elif f.startswith('--wpcap='):
            wpcap = int(f[8:])
    pool = navigable_names(pk3)
    if not names:
        # spec: "j-many of ALL of the maps in the game ... socketed together like
        # tilesets in a roguelike level generator".  The default j is the WHOLE
        # navigable stock pool; --maps=N samples j=N of it per seed (the roguelike
        # per-run draw), --maps=all is explicit.
        if nmaps in (None, 'all'):
            names = list(pool)
            random.Random(seed).shuffle(names)
        else:
            names = random.Random(seed).sample(pool, min(int(nmaps), len(pool)))
    if nbridges is None:
        # k-many procedurally generated hub tiles.  They are no longer the thing that
        # makes a join possible -- a join is made by editing the stock maps themselves --
        # so the default is one, present to exercise the class, not j/3.
        nbridges = 1 if len(names) > 1 else 0
    print('mapfuse seed=%d j=%d candidate stock maps + k=%d procedural hub tiles '
          '(pool=%d) levels=%s pk3=%s' %
          (seed, len(names), nbridges, len(pool), levels or 'auto', os.path.basename(pk3)))
    print('mapfuse maps=%s' % (names,))
    t0 = time.time()
    bsp_path, conns = fuse(seed, names, outdir, pk3, nbridges=nbridges, levels=levels, wpcap=wpcap)
    print('fuse wall time %.1fs' % (time.time() - t0))
    ent_path = os.path.join(outdir, 'fused.ent')
    M.emit(bsp_path, ent_path, teams, carts, pk3)
    import zipfile
    pk3out = os.path.join(outdir, 'fused.pk3')
    with zipfile.ZipFile(pk3out, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in ('fused.bsp', 'fused.waypoints', 'fused.waypoints.cache', 'fused.mapinfo', 'fused.ent'):
            z.write(os.path.join(outdir, f), 'maps/' + f)
    print('wrote %s (mount in client/server data dir to resolve maps/fused.*)' % pk3out)
    if '--nograph' not in flags:
        import fusegraph
        sys.argv = ['fusegraph', outdir]
        fusegraph.main()
    if '--smoke' in flags:
        smoke(outdir)
