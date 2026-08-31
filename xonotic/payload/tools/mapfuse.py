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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mkentfile as M

MARGIN, CORW, CORH, WALL, FLOORTHK, MAXCORLEN = 896.0, 288.0, 224.0, 32.0, 32.0, 6000.0
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
        self.geom = M.Bsp(data)
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


def pick_sockets(src, k=3):
    comp = [i for i in M.largest_component(src.navadj)
            if tuple(round(x, 1) for x in src.navnodes[i]) in src.wpset]
    if not comp:
        comp = M.largest_component(src.navadj)
    if not comp:
        return []
    cx = sum(src.navnodes[i][0] for i in comp) / len(comp)
    cy = sum(src.navnodes[i][1] for i in comp) / len(comp)
    cand = sorted(comp, key=lambda i: -math.hypot(src.navnodes[i][0] - cx, src.navnodes[i][1] - cy))
    cand = cand[:max(k * 8, len(cand) // 4)]
    picks = [cand[0]]
    while len(picks) < k and len(picks) < len(cand):
        best = max(cand, key=lambda i: min(math.dist(src.navnodes[i], src.navnodes[p]) for p in picks))
        if best in picks:
            break
        picks.append(best)
    return [list(src.navnodes[i]) for i in picks]


def pick_sockets_toward(src, offset, targets, kmin=3, minsep=384.0):
    """Sockets chosen FACING their partner tile.

    The earlier picker took the k nav nodes furthest from the map centroid, with no
    reference to which neighbour a socket was going to be joined to -- which is how a
    join corridor ended up 4685 units long ("the first corridor i found connecting two
    levels was really long").  Here each target is the world centre of the partner
    tile, and its socket is the closest stand-on-able node of the map's largest
    bot-reachable waypoint component to that target, with a minimum separation so two
    joins do not land on the same doorway.  Extra spread sockets are appended so the
    all-pairs fallback in fuse() still has choices."""
    comp = [i for i in M.largest_component(src.navadj)
            if tuple(round(x, 1) for x in src.navnodes[i]) in src.wpset]
    if not comp:
        comp = M.largest_component(src.navadj)
    if not comp:
        return []
    world = {i: vadd(src.navnodes[i], offset) for i in comp}
    picks = []
    for t in targets:
        cand = sorted(comp, key=lambda i: math.dist(world[i], t))
        got = None
        for i in cand:
            if all(math.dist(world[i], world[q]) >= minsep for q in picks):
                got = i
                break
        picks.append(got if got is not None else cand[0])
    # top up with maximally-spread extras for the fallback search
    pool = list(comp)
    while len(picks) < kmin and len(picks) < len(pool):
        nxt = max(pool, key=lambda i: min(math.dist(world[i], world[q]) for q in picks) if picks else 0)
        if nxt in picks:
            break
        picks.append(nxt)
    return [list(world[i]) for i in picks]


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
        n = max(1, int(L // 224))
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

    def build(self, routersplits):
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

        def emit_split(items):
            if len(items) == 1:
                return nodebase[items[0][2]]
            axis = 0 if max(i[0] for i in items) > max(i[1] for i in items) else 1
            items = sorted(items, key=lambda t: t[axis])
            mid = len(items) // 2
            boundary = routersplits[(tuple(sorted(i[2] for i in items)), axis)]
            n = [0.0] * 3
            n[axis] = 1.0
            pi = self.add_plane(n, boundary)
            me = len(nodes_out)
            nodes_out.append([pi, 0, 0, -WB2, -WB2, -WB2, WB2, WB2, WB2])
            c1 = emit_split(items[:mid])
            c0 = emit_split(items[mid:])
            nodes_out[me][1] = c0
            nodes_out[me][2] = c1
            return me

        if j > 1:
            emit_split([(self.cellpos[m][0], self.cellpos[m][1], m) for m in range(j)])
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

def plan_tiles(nsrc, k):
    """Lay T = nsrc + k tiles on a rectangular lattice and choose which cells are
    procedural BRIDGE tiles.  Returns (cols, rows, cells, bridge_cell_idx, nbr).

    The lattice is the roguelike tileset: every tile is a room, every lattice
    adjacency is a candidate socket.  Bridge cells are chosen by greedy
    max-coverage over lattice neighbours so each bridge ends up wired to several
    other map nodes (the spec's 'connector' map), and cells that touch exactly one
    bridge end up degree-1 (the spec's exclusive, prominent objective entrance)."""
    T = nsrc + k
    cols = max(1, int(math.ceil(math.sqrt(T))))
    rows = int(math.ceil(T / cols))
    cells = [(i % cols, i // cols) for i in range(T)]
    cs = set(cells)
    nbr = {c: [d for d in ((c[0] + 1, c[1]), (c[0] - 1, c[1]), (c[0], c[1] + 1), (c[0], c[1] - 1))
               if d in cs] for c in cells}
    bridges, covered = [], set()
    while len(bridges) < k:
        rest = [c for c in cells if c not in bridges]
        if not rest:
            break
        best = max(rest, key=lambda c: (len(set(nbr[c]) - covered - set(bridges)), len(nbr[c]),
                                        -abs(c[0] - (cols - 1) / 2) - abs(c[1] - (rows - 1) / 2)))
        bridges.append(best)
        covered |= set(nbr[best])
    return cols, rows, cells, bridges, nbr


def plan_edges(cells, bridges, nbr, rng=None, loopfrac=0.25):
    """Lattice-adjacency topology.  Every (bridge, neighbour) lattice adjacency is
    an edge; then union-find completion over the remaining adjacencies so the
    region graph is connected.  Only lattice-ADJACENT cells are ever joined, which
    is what keeps join corridors short while the megamap as a whole stays long."""
    idx = {c: i for i, c in enumerate(cells)}
    E = []
    seen = set()
    for b in bridges:
        for n in nbr[b]:
            k = (min(idx[b], idx[n]), max(idx[b], idx[n]))
            if k not in seen:
                seen.add(k)
                E.append(k)
    par = list(range(len(cells)))

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    def uni(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        par[ra] = rb
        return True

    for a, b in E:
        uni(a, b)
    for c in cells:
        for n in nbr[c]:
            k = (min(idx[c], idx[n]), max(idx[c], idx[n]))
            if k in seen:
                continue
            if uni(k[0], k[1]):
                seen.add(k)
                E.append(k)
    # LOOP edges: extra lattice adjacencies beyond the spanning structure.  Without
    # them every join is a cut edge and the prominence rule is vacuous -- the spec
    # only lets a join be subtle/weakly-signposted when it is REDUNDANT.
    if rng is not None and loopfrac > 0:
        spare = []
        for c in cells:
            for n in nbr[c]:
                k = (min(idx[c], idx[n]), max(idx[c], idx[n]))
                if k not in seen and k not in spare:
                    spare.append(k)
        rng.shuffle(spare)
        for k in spare[:int(round(loopfrac * len(cells)))]:
            seen.add(k)
            E.append(k)
    return E


def choose_layout(srcs, seed, cellpos=None, grid=None):
    j = len(srcs)
    exts = [[s.bounds[1][a] - s.bounds[0][a] for a in range(3)] for s in srcs]
    if grid:
        cols, rows = grid
    else:
        cols = max(1, int(math.ceil(math.sqrt(j))))
        rows = int(math.ceil(j / cols))
    cp = list(cellpos) if cellpos else [(m % cols, m // cols) for m in range(j)]
    # NON-UNIFORM lattice bands: a column is only as wide as its widest map, a row
    # only as tall as its tallest.  A single uniform cell sized to the biggest map in
    # the pool is what made join corridors kilometres long -- a small map sat marooned
    # in the middle of a cell sized for catharsis.
    colw = [MARGIN + max([exts[m][0] for m in range(j) if cp[m][0] == c] or [0]) for c in range(cols)]
    rowh = [MARGIN + max([exts[m][1] for m in range(j) if cp[m][1] == r] or [0]) for r in range(rows)]
    xed, yed = [-sum(colw) / 2.0], [-sum(rowh) / 2.0]
    for w in colw:
        xed.append(xed[-1] + w)
    for h in rowh:
        yed.append(yed[-1] + h)
    offsets, cellpos_out = [], []
    for m, s in enumerate(srcs):
        col, row = cp[m]
        ccx = (xed[col] + xed[col + 1]) / 2.0
        ccy = (yed[row] + yed[row + 1]) / 2.0
        mc = [(s.bounds[0][a] + s.bounds[1][a]) / 2 for a in range(3)]
        offsets.append([round(ccx - mc[0]), round(ccy - mc[1]), round(-mc[2])])
        cellpos_out.append((col, row))
    for a in range(j):
        for b in range(a + 1, j):
            la = [srcs[a].bounds[0][i] + offsets[a][i] for i in range(3)]
            ha = [srcs[a].bounds[1][i] + offsets[a][i] for i in range(3)]
            lb = [srcs[b].bounds[0][i] + offsets[b][i] for i in range(3)]
            hb = [srcs[b].bounds[1][i] + offsets[b][i] for i in range(3)]
            assert ha[0] <= lb[0] or hb[0] <= la[0] or ha[1] <= lb[1] or hb[1] <= la[1], (a, b)
    cellpos = cellpos_out
    splits = {}

    def register(items, axis0):
        if len(items) <= 1:
            return
        axis = 0 if max(i[0] for i in items) > max(i[1] for i in items) else 1
        its = sorted(items, key=lambda t: t[axis])
        mid = len(its) // 2
        lohalf, hihalf = its[:mid], its[mid:]
        clo = max(i[axis] for i in lohalf)
        boundary = xed[clo + 1] if axis == 0 else yed[clo + 1]
        splits[(tuple(sorted(i[2] for i in its)), axis)] = boundary
        register(lohalf, axis)
        register(hihalf, axis)

    register([(cellpos_out[m][0], cellpos_out[m][1], m) for m in range(j)], 0)
    return offsets, cellpos_out, splits, (xed, yed, cp)


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


def fuse(seed, names, outdir, pk3, nbridges=0, workdir=None, loopfrac=0.25):
    rng = random.Random(seed)
    os.makedirs(outdir, exist_ok=True)
    nsrc = len(names)
    cols, rows, cells, bridge_cells, nbr = plan_tiles(nsrc, nbridges)
    T = len(cells)
    bset = set(bridge_cells)
    cellidx = {c: i for i, c in enumerate(cells)}
    edges = plan_edges(cells, bridge_cells, nbr, rng=rng, loopfrac=loopfrac)
    print('lattice: %dx%d, %d tiles (%d stock maps + %d procedural bridge tiles), '
          '%d lattice edges' % (cols, rows, T, nsrc, len(bridge_cells), len(edges)))
    # ---- generate the procedural bridge tiles, arms aimed at their lattice neighbours
    work = workdir or os.path.join(outdir, '_bridges')
    hub_ports = {}
    hub_base = {}
    if bridge_cells:
        import mapgen
        for bi, c in enumerate(bridge_cells):
            dirs = []
            for n in nbr[c]:
                if cellidx[c] < cellidx[n] or True:
                    dirs.append(DIRNAME[(n[0] - c[0], n[1] - c[1])])
            dirs = [d for d in ('e', 'w', 'n', 's') if d in dirs]
            lo = [d for k, d in enumerate(dirs) if k % 2 == 0]
            hi = [d for k, d in enumerate(dirs) if k % 2 == 1]
            nm = 'bridge%d_%d' % (seed, bi)
            base, ports = mapgen.build_bridge_tile(work, nm, seed * 131 + bi, lo, hi)
            hub_base[c] = base
            hub_ports[c] = ports
    # ---- order the tile list so tile index == lattice cell index
    srcs = [None] * T
    stock = list(names)
    si = 0
    for i, c in enumerate(cells):
        if c in bset:
            srcs[i] = load_src(hub_base[c], outdir, pk3)
        else:
            srcs[i] = load_src(stock[si], outdir, pk3)
            si += 1
    names = [x.name for x in srcs]
    j = T
    is_bridge = [cells[i] in bset for i in range(T)]
    cellpos = list(cells)
    offsets, cellpos, splits, bands = choose_layout(srcs, seed, cellpos=cellpos, grid=(cols, rows))
    # NUDGE: slide each tile inside its own lattice band toward the mean direction of
    # its joined neighbours.  A band is as wide as its widest map, so a smaller map has
    # slack; spending that slack on the neighbours it is actually socketed to shortens
    # every join corridor and never moves a tile out of its band, so the axial band
    # splits (and therefore the BSP router) are untouched.
    xed, yed, cpp = bands
    nb_tmp = [[] for _ in range(j)]
    for a, b in edges:
        nb_tmp[a].append(b)
        nb_tmp[b].append(a)
    for m in range(j):
        if not nb_tmp[m]:
            continue
        cx, cy = cpp[m]
        tgt = [0.0, 0.0]
        for n in nb_tmp[m]:
            nx, ny = cpp[n]
            tgt[0] += (nx - cx)
            tgt[1] += (ny - cy)
        L = math.hypot(*tgt) or 1.0
        for a2, ed, ci in ((0, xed, cx), (1, yed, cy)):
            ext = srcs[m].bounds[1][a2] - srcs[m].bounds[0][a2]
            slack = max(0.0, (ed[ci + 1] - ed[ci] - ext) / 2.0 - 64.0)
            offsets[m][a2] = round(offsets[m][a2] + slack * (tgt[a2] / L))
    for m, s in enumerate(srcs):
        print('place %s%s at cell %s offset %s' %
              (s.name, ' [BRIDGE]' if is_bridge[m] else '', cellpos[m], offsets[m]))
    F = Fuser(srcs, offsets, seed)
    F.cellpos = cellpos
    degree = [0] * j
    for a, b in edges:
        degree[a] += 1
        degree[b] += 1
    # world-space centre of every tile, so a socket can be chosen FACING its partner
    tilec = [[(srcs[m].bounds[0][a] + srcs[m].bounds[1][a]) / 2 + offsets[m][a] for a in range(3)]
             for m in range(j)]
    nbrs_of = [[] for _ in range(j)]
    for a, b in edges:
        nbrs_of[a].append(b)
        nbrs_of[b].append(a)
    sockets, sockmap = [], {}
    for m, s in enumerate(srcs):
        c = cells[m]
        if is_bridge[m]:
            # a bridge tile's sockets ARE its arm mouths, one arm per lattice neighbour
            sk = [vadd(pt['p'], offsets[m]) for pt in hub_ports[c]]
        else:
            # aim at the MIDPOINT between the two tile centres (i.e. at the shared
            # lattice boundary), not at the partner's centre: that is the point the
            # join corridor has to reach, and it is what keeps the corridor short.
            tg = [[(tilec[m][a] + tilec[n][a]) / 2 for a in range(3)] for n in nbrs_of[m]]
            sk = pick_sockets_toward(s, offsets[m], tg, kmin=max(3, degree[m] + 1))
        for n, pt in zip(nbrs_of[m], sk):
            sockmap[(m, n)] = pt
        sockets.append(sk)
        print('sockets %s (deg %d)%s: %s' %
              (s.name, degree[m], ' [BRIDGE]' if is_bridge[m] else '',
               [[round(x) for x in p] for p in sk]))
    # A CUT edge is the exclusive mode of entry to everything behind it, exactly like a
    # degree-1 tile's single edge; both get the prominent (wide, lit) template.  A
    # non-cut (loop) edge is redundant and may be subtle.
    PRE = region_graph_solve(j, edges)
    cutset = set(PRE['cutedges'])
    exclusive = [min(degree[a], degree[b]) == 1 or ei in cutset for ei, (a, b) in enumerate(edges)]
    def _edge_minsep(i):
        a, b = edges[i]
        return min(math.dist(sa, sb) for sa in sockets[a] for sb in sockets[b])
    order = sorted(range(len(edges)), key=lambda i: (not exclusive[i], _edge_minsep(i)))
    # spec: "not all level-level connections (at least one maximum per map) need to
    # even be cart-path-navigable" -> AT MOST ONE non-cart-navigable join per tile.
    # A corridor is cart-navigable (the cart can be pushed along it); a teleporter or
    # a jumppad is not.  So corridors are tried FIRST and always, and the
    # teleporter/jumppad fallback is rationed against a per-tile budget of 1.
    noncart_used = [0] * j
    noncart_budget = 1
    used_sock = set()
    conns = []
    telen = padn = corn = 0
    budget_viol = []
    dropped = []
    alive = set(range(len(edges)))

    def _still_connected(sub):
        par = list(range(j))

        def find(x):
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x
        n = j
        for e in sub:
            ra, rb = find(edges[e][0]), find(edges[e][1])
            if ra != rb:
                par[ra] = rb
                n -= 1
        return n == 1
    for ei in order:
        a, b = edges[ei]
        excl = exclusive[ei]
        prom = excl
        pairs = sorted(((sa, sb) for sa in sockets[a] for sb in sockets[b]
                        if tuple(sa) not in used_sock and tuple(sb) not in used_sock),
                       key=lambda p: math.dist(p[0], p[1]))
        des = (sockmap.get((a, b)), sockmap.get((b, a)))
        if des[0] is not None and des[1] is not None:
            pairs = ([(des[0], des[1])] +
                     [pq for pq in pairs if pq[0] is not des[0] or pq[1] is not des[1]])
        if not pairs:
            pairs = sorted(((sa, sb) for sa in sockets[a] for sb in sockets[b]),
                           key=lambda p: math.dist(p[0], p[1]))
        kind, pick, carved = 'teleporter', pairs[0], set()
        may_noncart = noncart_used[a] < noncart_budget and noncart_used[b] < noncart_budget
        cw2 = (CORW_PROM if prom else CORW) / 2.0
        for maxhits in (16, 48, 96):
            if kind == 'corridor':
                break
            if maxhits != 16 and may_noncart:
                break            # only strain the carver when the budget forbids a portal
            for sa, sb in pairs:
                if math.dist(sa, sb) > MAXCORLEN:
                    continue
                hits = blockage(srcs, offsets, corridor_samples(sa, sb, cw2), clip=True)
                if all(brush_volume_ok(srcs[m], bi) for m, bi in hits) and len(hits) <= maxhits:
                    kind, pick, carved = 'corridor', (sa, sb), hits
                    break
        if kind != 'corridor' and not may_noncart:
            if _still_connected(alive - {ei}):
                # a redundant loop edge that cannot be a corridor and cannot spend a
                # non-cart-navigable slot is simply not built: dropping it costs
                # nothing (the region graph stays connected without it) and it is the
                # only way to keep "at most one non-cart join per map" a hard rule.
                alive.discard(ei)
                dropped.append((names[a], names[b]))
                print('edge %s <-> %s: DROPPED (loop edge, no corridor possible, '
                      'non-cart budget spent on both sides)' % (names[a], names[b]))
                continue
            budget_viol.append((names[a], names[b]))
        if kind == 'teleporter':
            for sa, sb in pairs:
                lo, hi = (sa, sb) if sa[2] <= sb[2] else (sb, sa)
                if hi[2] - lo[2] < 64:
                    continue
                hits = blockage(srcs, offsets, arc_samples(lo, hi))
                if not hits:
                    kind, pick = 'jumppad', (lo, hi)
                    break
        sa, sb = pick
        if kind == 'corridor':
            F.carve(carved)
            F.build_corridor(sa, sb, prominent=prom)
            corn += 1
        elif kind == 'jumppad':
            noncart_used[a] += 1
            noncart_used[b] += 1
            F.build_pad(sa, sb, padn, prominent=prom)
            F.build_tele(sb, sa, 1000 + padn, prominent=prom)
            padn += 1
            telen += 1
        else:
            noncart_used[a] += 1
            noncart_used[b] += 1
            F.build_tele(sa, sb, telen, prominent=prom)
            F.build_tele(sb, sa, 100 + telen, prominent=prom)
            telen += 2
        used_sock.add(tuple(sa))
        used_sock.add(tuple(sb))
        conns.append((a, b, kind, sa, sb, len(carved), excl, prom, math.dist(sa, sb)))
        print('edge %s(deg%d) <-> %s(deg%d): %s %s  %s -> %s  len=%.0f carved=%d' %
              (srcs[a].name, degree[a], srcs[b].name, degree[b], kind,
               'PROMINENT/exclusive' if excl else 'subtle/redundant',
               [round(x) for x in sa], [round(x) for x in sb], math.dist(sa, sb), len(carved)))
    # light the procedural bridge tiles: they are the connector class, and an unlit
    # connector is exactly the "map graphics conceal map transitions" failure.
    for m in range(j):
        if not is_bridge[m]:
            continue
        c0 = tilec[m]
        F.add_light([c0[0], c0[1], srcs[m].bounds[0][2] + offsets[m][2] + 420.0], 900)
        for pt in sockets[m]:
            F.add_light([pt[0], pt[1], pt[2] + 160.0], PROM_LIGHT)
    nexcl = sum(1 for c in conns if c[6])
    nbr_tiles = sum(1 for x in is_bridge if x)
    print('topology: %d tiles (%d stock + %d procedural bridge), %d edges (%d tree + %d loops), '
          'corridors=%d jumppads=%d teleport-triggers=%d' %
          (j, j - nbr_tiles, nbr_tiles, len(edges), j - 1, len(edges) - (j - 1), corn, padn, telen))
    print('prominence: %d exclusive/objective edges (prominent+lit), %d redundant edges (subtle); '
          'node degrees %s' % (nexcl, len(conns) - nexcl, degree))
    clens = sorted(c[8] for c in conns if c[2] == 'corridor')
    if clens:
        print('corridor length: n=%d min=%.0f median=%.0f max=%.0f (cap %.0f)' %
              (len(clens), clens[0], clens[len(clens) // 2], clens[-1], MAXCORLEN))
    print('cart-navigability: %d/%d joins cart-navigable (corridor); non-cart joins per tile '
          'max=%d (budget %d); %d loop edges dropped -> %s' %
          (corn, len(conns), max(noncart_used) if noncart_used else 0, noncart_budget,
           len(dropped), 'HELD' if not budget_viol else 'VIOLATED on %s' % budget_viol))
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
    wlines = ['//WAYPOINT_VERSION 1.04', '//WAYPOINT_SYMMETRY 0', '//WAYPOINT_TIME ' + ts]
    seen = set()
    allwp = []
    for m, src in enumerate(srcs):
        for m1, m2, fl in src.wptriples:
            allwp.append((vadd(m1, offsets[m]), vadd(m2, offsets[m]), fl))
    for p in F.wp_extra:
        allwp.append((p, p, 0))
    for m1, m2, fl in allwp:
        k = vstr(m1)
        if m1 == m2 and k in seen:
            continue
        seen.add(k)
        wlines += [vstr(m1), vstr(m2), '%d' % fl]
    open(os.path.join(outdir, 'fused.waypoints'), 'w').write('\n'.join(wlines) + '\n')
    clines = ['//WAYPOINT_VERSION 1.04', '//WAYPOINT_TIME ' + ts]
    for m, src in enumerate(srcs):
        for a, b in src.cachelinks:
            clines.append(vstr(vadd(a, offsets[m])) + '*' + vstr(vadd(b, offsets[m])))
    for a, b in F.link_extra:
        clines.append(vstr(a) + '*' + vstr(b))
    open(os.path.join(outdir, 'fused.waypoints.cache'), 'w').write('\n'.join(clines) + '\n')
    print('wrote fused.waypoints (%d wps) fused.waypoints.cache (%d links)' % (len(allwp), len(clines) - 2))
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
             'bot_jumps': [[list(n), list(f)] for n, f in F.bot_jumps]}
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
               'bsp_bytes': len(data), 'names': names}
    json.dump(metrics, open(os.path.join(outdir, 'fused.metrics.json'), 'w'), indent=1)
    print('wrote fused.joins.json (%d tiles, %d joins) + fused.metrics.json' % (j, len(conns)))
    return bsp_path, conns


def smoke(outdir):
    eng = os.path.expanduser('~/dox/xonotic/build-engine/darkplaces-dedicated')
    basedir = os.path.expanduser('~/dox/xonotic/Xonotic')
    log = os.path.join('/tmp/fusesmoke', 'smoke.log')
    cmd = [eng, '-xonotic', '-basedir', basedir, '-userdir', '/tmp/fusesmoke',
           '+port', '26013', '+sv_public', '0', '+g_payload', '0',
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
        # k-many bridge tiles: default one connector per ~3 stock maps, >=1 whenever
        # there is more than one map to socket together.
        nbridges = max(1, round(len(names) / 3.0)) if len(names) > 1 else 0
    print('mapfuse seed=%d j=%d stock maps + k=%d procedural bridge tiles (pool=%d) pk3=%s' %
          (seed, len(names), nbridges, len(pool), os.path.basename(pk3)))
    print('mapfuse maps=%s' % (names,))
    t0 = time.time()
    bsp_path, conns = fuse(seed, names, outdir, pk3, nbridges=nbridges)
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
