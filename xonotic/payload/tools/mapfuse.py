import struct, sys, os, re, math, glob, random, subprocess, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mkentfile as M

MARGIN, CORW, CORH, WALL, FLOORTHK, MAXCORLEN = 2048.0, 288.0, 224.0, 32.0, 32.0, 14000.0
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


def corridor_samples(a, b):
    af, bf, dirh, side, ntop, L = corridor_frame(a, b)
    pts = []
    n = max(2, int(L // 64))
    for k in range(n + 1):
        f = k / n
        base = [af[i] + f * (bf[i] - af[i]) for i in range(3)]
        for lat in (-CORW / 2 + 24, 0.0, CORW / 2 - 24):
            for h in (24.0, 100.0, CORH - 40):
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


def blockage(srcs, offsets, pts):
    hits = set()
    for m, src in enumerate(srcs):
        off = offsets[m]
        for p in pts:
            q = vsub(p, off)
            bi = src.solid_brush_at(q)
            if bi >= 0:
                hits.add((m, bi))
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
        self.extra_ents = []
        self.wp_extra, self.link_extra = [], []
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

    def add_brush(self, planes, tex, dest=None):
        if dest is None:
            dest = self.conn_brushes
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

    def carve(self, hits):
        for m, bi in hits:
            self.carved.add((m, bi))

    def build_corridor(self, a, b):
        af, bf, dirh, side, ntop, L = corridor_frame(a, b)
        w2 = CORW / 2
        br = [self.add_brush(slab_planes(af, bf, dirh, side, ntop, w2 + WALL, -FLOORTHK, 0.0), self.solid_face_tex)]
        for s in (1.0, -1.0):
            sv = vscale(side, s)
            pl = [(dirh, vdot(dirh, bf) + 8), ([-x for x in dirh], -(vdot(dirh, af) - 8)),
                  (sv, vdot(sv, af) + w2 + WALL), ([-x for x in sv], -(vdot(sv, af) + w2)),
                  (ntop, vdot(ntop, af) + CORH + WALL), ([-x for x in ntop], -(vdot(ntop, af) - FLOORTHK))]
            br.append(self.add_brush(pl, self.solid_face_tex))
        br.append(self.add_brush(slab_planes(af, bf, dirh, side, ntop, w2 + WALL, CORH, CORH + WALL), self.solid_face_tex))
        fa = []
        c = lambda base, lat, h: vadd(vadd(base, vscale(side, lat)), vscale(ntop, h))
        fa.append(self.add_quad([c(af, -w2, 0), c(bf, -w2, 0), c(bf, w2, 0), c(af, w2, 0)], ntop, self.solid_face_tex))
        fa.append(self.add_quad([c(af, -w2, 0), c(bf, -w2, 0), c(bf, -w2, CORH), c(af, -w2, CORH)], side, self.solid_face_tex))
        fa.append(self.add_quad([c(af, w2, 0), c(bf, w2, 0), c(bf, w2, CORH), c(af, w2, CORH)], [-x for x in side], self.solid_face_tex))
        fa.append(self.add_quad([c(af, -w2, CORH), c(bf, -w2, CORH), c(bf, w2, CORH), c(af, w2, CORH)], [-x for x in ntop], self.solid_face_tex))
        lo = [min(af[i], bf[i]) - CORW for i in range(3)]
        hi = [max(af[i], bf[i]) + CORW + CORH for i in range(3)]
        self.conn_leafsets.append((fa, br, lo, hi))
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

    def build_pad(self, a, b, idx):
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
        self.extra_ents.append('{\n"classname" "info_notnull"\n"targetname" "fpush%d"\n"origin" "%s %s %s"\n}' %
                               (idx, fnum(b[0]), fnum(b[1]), fnum(b[2] + 40)))
        self.wp_extra.append([a[0], a[1], a[2] + 7])
        self.link_extra.append((a, [a[0], a[1], a[2] + 7]))
        self.link_extra.append(([a[0], a[1], a[2] + 7], a))
        self.link_extra.append(([a[0], a[1], a[2] + 7], b))

    def build_tele(self, a, b, idx):
        tb = self.add_brush(axial_planes([a[0] - 48, a[1] - 48, a[2]], [a[0] + 48, a[1] + 48, a[2] + 112]),
                            self.trigtex, self.trig_brushes)
        self.trig_models.append((tb, [a[0] - 48, a[1] - 48, a[2]], [a[0] + 48, a[1] + 48, a[2] + 112],
                                 'trigger_teleport', 'ftele%d' % idx))
        self.extra_ents.append('{\n"classname" "misc_teleporter_dest"\n"targetname" "ftele%d"\n"origin" "%s %s %s"\n"angle" "%d"\n}' %
                               (idx, fnum(b[0]), fnum(b[1]), fnum(b[2] + 16), 0))
        self.link_extra.append((a, b))

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

                b = re.sub(r'"origin"\s+"([-\d.eE+ ]+)"', fixorigin, b)
                b = re.sub(r'"model"\s+"\*(\d+)"',
                           lambda mo: '"model" "*%d"' % self.modelmap[m][int(mo.group(1))], b)
                for k in tkeys:
                    b = re.sub(r'"%s"\s+"([^"]+)"' % k,
                               lambda mo, k=k: '"%s" "m%d_%s"' % (k, m, mo.group(1)), b)
                blocks_out.append(b)
        for i, (tb, mins, maxs, cls, tgt) in enumerate(self.trig_models):
            blocks_out.append('{\n"classname" "%s"\n"model" "*%d"\n"target" "%s"\n}' %
                              (cls, self.trigmodel0 + i, tgt))
        blocks_out += self.extra_ents
        return '\n'.join(blocks_out) + '\n'

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

def choose_layout(srcs, seed):
    j = len(srcs)
    exts = [[s.bounds[1][a] - s.bounds[0][a] for a in range(3)] for s in srcs]
    cw = max(e[0] for e in exts) + MARGIN
    ch = max(e[1] for e in exts) + MARGIN
    cols = max(1, int(math.ceil(math.sqrt(j))))
    rows = int(math.ceil(j / cols))
    x0 = -(cols * cw) / 2
    y0 = -(rows * ch) / 2
    offsets, cellpos = [], []
    for m, s in enumerate(srcs):
        col, row = m % cols, m // cols
        ccx = x0 + (col + 0.5) * cw
        ccy = y0 + (row + 0.5) * ch
        mc = [(s.bounds[0][a] + s.bounds[1][a]) / 2 for a in range(3)]
        offsets.append([round(ccx - mc[0]), round(ccy - mc[1]), round(-mc[2])])
        cellpos.append((col, row))
    for a in range(j):
        for b in range(a + 1, j):
            la = [srcs[a].bounds[0][i] + offsets[a][i] for i in range(3)]
            ha = [srcs[a].bounds[1][i] + offsets[a][i] for i in range(3)]
            lb = [srcs[b].bounds[0][i] + offsets[b][i] for i in range(3)]
            hb = [srcs[b].bounds[1][i] + offsets[b][i] for i in range(3)]
            assert ha[0] <= lb[0] or hb[0] <= la[0] or ha[1] <= lb[1] or hb[1] <= la[1], (a, b)
    splits = {}

    def register(items, axis0):
        if len(items) <= 1:
            return
        axis = 0 if max(i[0] for i in items) > max(i[1] for i in items) else 1
        its = sorted(items, key=lambda t: t[axis])
        mid = len(its) // 2
        lohalf, hihalf = its[:mid], its[mid:]
        clo = max(i[axis] for i in lohalf)
        boundary = (x0 + (clo + 1) * cw) if axis == 0 else (y0 + (clo + 1) * ch)
        splits[(tuple(sorted(i[2] for i in its)), axis)] = boundary
        register(lohalf, axis)
        register(hihalf, axis)

    register([(cellpos[m][0], cellpos[m][1], m) for m in range(j)], 0)
    return offsets, cellpos, splits


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


def fuse(seed, names, outdir, pk3):
    rng = random.Random(seed)
    os.makedirs(outdir, exist_ok=True)
    srcs = []
    for n in names:
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
        srcs.append(Src(n, data, wp, cache))
        names = [x.name for x in srcs] + names[len(srcs):]
        print('src %s: bounds %s %s models=%d faces=%d brushes=%d wp=%d links=%d' %
              (n, [round(x) for x in srcs[-1].bounds[0]], [round(x) for x in srcs[-1].bounds[1]],
               len(srcs[-1].models), len(srcs[-1].faces), len(srcs[-1].brushes),
               len(srcs[-1].wptriples), len(srcs[-1].cachelinks)))
    j = len(srcs)
    offsets, cellpos, splits = choose_layout(srcs, seed)
    for m, s in enumerate(srcs):
        print('place %s at cell %s offset %s' % (s.name, cellpos[m], offsets[m]))
    F = Fuser(srcs, offsets, seed)
    F.cellpos = cellpos
    sockets = []
    for m, s in enumerate(srcs):
        sk = [vadd(p, offsets[m]) for p in pick_sockets(s)]
        sockets.append(sk)
        print('sockets %s: %s' % (s.name, [[round(x) for x in p] for p in sk]))
    edges = [(rng.randrange(i), i) for i in range(1, j)]
    nloops = rng.randrange(0, max(1, j - 1))
    for _ in range(nloops):
        a, b = rng.sample(range(j), 2) if j > 1 else (0, 0)
        if j > 1 and (min(a, b), max(a, b)) not in [(min(x), max(x)) for x in edges]:
            edges.append((min(a, b), max(a, b)))
    corridor_used = [0] * j
    used_sock = set()
    conns = []
    telen = padn = corn = 0
    for a, b in edges:
        pairs = sorted(((sa, sb) for sa in sockets[a] for sb in sockets[b]
                        if tuple(sa) not in used_sock and tuple(sb) not in used_sock),
                       key=lambda p: math.dist(p[0], p[1]))
        if not pairs:
            pairs = sorted(((sa, sb) for sa in sockets[a] for sb in sockets[b]),
                           key=lambda p: math.dist(p[0], p[1]))
        kind, pick, carved = 'teleporter', pairs[0], set()
        if corridor_used[a] == 0 and corridor_used[b] == 0:
            for sa, sb in pairs:
                if math.dist(sa, sb) > MAXCORLEN:
                    continue
                hits = blockage(srcs, offsets, corridor_samples(sa, sb))
                if all(brush_volume_ok(srcs[m], bi) for m, bi in hits) and len(hits) <= 8:
                    kind, pick, carved = 'corridor', (sa, sb), hits
                    break
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
            F.build_corridor(sa, sb)
            corridor_used[a] += 1
            corridor_used[b] += 1
            corn += 1
        elif kind == 'jumppad':
            F.build_pad(sa, sb, padn)
            F.build_tele(sb, sa, 1000 + padn)
            padn += 1
            telen += 1
        else:
            F.build_tele(sa, sb, telen)
            F.build_tele(sb, sa, 100 + telen)
            telen += 2
        used_sock.add(tuple(sa))
        used_sock.add(tuple(sb))
        conns.append((a, b, kind, sa, sb, len(carved)))
        print('edge %s <-> %s: %s  %s -> %s  carved=%d' %
              (srcs[a].name, srcs[b].name, kind, [round(x) for x in sa], [round(x) for x in sb], len(carved)))
    print('topology: %d maps, %d edges (%d tree + %d loops), corridors=%d jumppads=%d teleport-triggers=%d' %
          (j, len(edges), j - 1, len(edges) - (j - 1), corn, padn, telen))
    data, nodes_out, leafs_out, models_out = F.build(splits)
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
    G = M.Bsp(open(bsp_path, 'rb').read())
    fviol = 0
    for c in conns:
        if c[2] != 'corridor':
            continue
        af, bf = c[3], c[4]
        n = max(2, int(math.dist(af, bf) // 128))
        for k in range(n + 1):
            f = k / n
            q = [af[i] + f * (bf[i] - af[i]) for i in range(3)]
            q[2] += 24
            if G.inside(q):
                fviol += 1
            if G.floor(q[0], q[1], q[2]) is None:
                fviol += 1
    print('connector floor check: %s (%d violations)' % ('PASS' if fviol == 0 else 'FAIL', fviol))
    nodes2, adj2 = M.parse_cache(open(os.path.join(outdir, 'fused.waypoints.cache')).read())
    comp = set(M.largest_component(adj2))
    idx2 = {tuple(round(x, 1) for x in nodes2[i]): i for i in range(len(nodes2))}
    missing = 0
    for m in range(j):
        for p in sockets[m]:
            k = tuple(round(x, 1) for x in p)
            if idx2.get(k, -1) not in comp:
                missing += 1
    print('nav connectivity: %s (%d/%d sockets in main component, %d nodes total)' %
          ('PASS' if missing == 0 else 'FAIL', sum(len(s) for s in sockets) - missing,
           sum(len(s) for s in sockets), len(nodes2)))
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
    if not names:
        pool = navigable_names(pk3)
        names = random.Random(seed).sample(pool, 3)
    outdir = '/tmp/fusesmoke/data/maps'
    for f in flags:
        if f.startswith('--out='):
            outdir = f[6:]
    print('mapfuse seed=%d maps=%s pk3=%s' % (seed, names, os.path.basename(pk3)))
    bsp_path, conns = fuse(seed, names, outdir, pk3)
    ent_path = os.path.join(outdir, 'fused.ent')
    M.emit(bsp_path, ent_path, 5, 3, pk3)
    import zipfile
    pk3out = os.path.join(outdir, 'fused.pk3')
    with zipfile.ZipFile(pk3out, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in ('fused.bsp', 'fused.waypoints', 'fused.waypoints.cache', 'fused.mapinfo', 'fused.ent'):
            z.write(os.path.join(outdir, f), 'maps/' + f)
    print('wrote %s (mount in client/server data dir to resolve maps/fused.*)' % pk3out)
    if '--smoke' in flags:
        smoke(outdir)
