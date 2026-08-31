"""spiralgen.py -- procedural helical-tunnel .map generator for Xonotic / idTech3.

Geometry kernel is deliberately tiny: every solid is a *convex brush built by
construction*, so no CSG library and no convex decomposition are ever needed.

    swept surface -> quad strip -> 2 triangles -> extrude each triangle
                  along a fixed offset vector -> triangular prism (5 planes)

A triangular prism extruded along a constant vector is convex for ANY input
triangle, so an arbitrarily curved sweep (helix, cone, torus knot, ...) becomes
a pile of legal brushes with zero geometric failure modes.  Everything else in
this file is bookkeeping.

Output is a plain .map for q3map2 (-game xonotic).  q3map2 then supplies BSP,
VIS, lightmaps and collision -- none of which we have to synthesize ourselves.
"""
import argparse, json, math, os, random, sys

# ---------------------------------------------------------------- vector math
def vadd(a, b):  return (a[0] + b[0], a[1] + b[1], a[2] + b[2])
def vsub(a, b):  return (a[0] - b[0], a[1] - b[1], a[2] - b[2])
def vmul(a, s):  return (a[0] * s, a[1] * s, a[2] * s)
def vdot(a, b):  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
def vcross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])
def vlen(a):     return math.sqrt(vdot(a, a))
def vnorm(a):
    L = vlen(a)
    return (a[0] / L, a[1] / L, a[2] / L) if L > 1e-9 else (0.0, 0.0, 0.0)


# ---------------------------------------------------------------- .map writing
def fmt(x):
    """idTech .map wants integers where possible; q3map2 snaps planes anyway."""
    r = round(x, 4)
    return str(int(r)) if abs(r - round(r)) < 1e-6 else ('%.4f' % r)


class Brush:
    """A convex solid, stored as faces.  Faces are (points, texture).

    Point winding is never trusted: normals are derived and then flipped to
    point away from the brush centroid, which is correct for any convex solid.
    q3map2 computes  n = (p2 - p0) x (p1 - p0)  and treats the interior as the
    negative halfspace (see netradiant-custom tools/quake3/common/qmath.h:196),
    so an outward-facing plane is emitted as (v0, v2, v1) for CCW-from-outside
    input -- we just resolve the sign numerically instead of reasoning about it.
    """

    def __init__(self, detail=False):
        self.faces = []
        self.detail = detail

    def add_face(self, pts, tex):
        self.faces.append((list(pts), tex))
        return self

    def centroid(self):
        allp = [p for pts, _ in self.faces for p in pts]
        n = len(allp)
        return (sum(p[0] for p in allp) / n, sum(p[1] for p in allp) / n, sum(p[2] for p in allp) / n)

    def to_map(self, scale=0.5):
        cflags = 134217728 if self.detail else 0   # CONTENTS_DETAIL
        c = self.centroid()
        out = ['{']
        for pts, tex in self.faces:
            p0, p1, p2 = pts[0], pts[1], pts[2]
            n = vnorm(vcross(vsub(p2, p0), vsub(p1, p0)))
            if vlen(n) < 1e-9:
                continue                                  # degenerate, drop it
            if vdot(n, vsub(c, p0)) > 0:                   # normal points inward
                p1, p2 = p2, p1
            out.append('( %s %s %s ) ( %s %s %s ) ( %s %s %s ) %s 0 0 0 %g %g %d 0 0' % (
                fmt(p0[0]), fmt(p0[1]), fmt(p0[2]),
                fmt(p1[0]), fmt(p1[1]), fmt(p1[2]),
                fmt(p2[0]), fmt(p2[1]), fmt(p2[2]), tex, scale, scale, cflags))
        out.append('}')
        return '\n'.join(out)


def tri_prism(a, b, c, v, tex_cap, tex_side, detail=False):
    """Triangle (a,b,c) extruded by vector v -> a convex 5-sided brush.

    Convex for any non-degenerate triangle and any non-parallel v, which is the
    whole reason this is the only primitive the generator needs.
    """
    if vlen(vcross(vsub(b, a), vsub(c, a))) < 1e-6:
        return None                                        # slivers help nobody
    a2, b2, c2 = vadd(a, v), vadd(b, v), vadd(c, v)
    br = Brush(detail)
    br.add_face([a, b, c], tex_cap)
    br.add_face([a2, b2, c2], tex_cap)
    br.add_face([a, b, b2, a2], tex_side)
    br.add_face([b, c, c2, b2], tex_side)
    br.add_face([c, a, a2, c2], tex_side)
    return br


def quad_prism(a, b, c, d, v, tex_cap, tex_side, detail=False):
    """Planar quad (a,b,c,d) extruded by v -> a convex 6-sided brush."""
    a2, b2, c2, d2 = vadd(a, v), vadd(b, v), vadd(c, v), vadd(d, v)
    br = Brush(detail)
    br.add_face([a, b, c, d], tex_cap)
    br.add_face([a2, b2, c2, d2], tex_cap)
    for (p, q, p2, q2) in ((a, b, a2, b2), (b, c, b2, c2), (c, d, c2, d2), (d, a, d2, a2)):
        br.add_face([p, q, q2, p2], tex_side)
    return br


def coplanar(a, b, c, d, eps=0.01):
    n = vcross(vsub(b, a), vsub(c, a))
    L = vlen(n)
    return L > 1e-9 and abs(vdot(n, vsub(d, a))) / L < eps


def strip(ring_lo, ring_hi, offset_fn, tex_cap, tex_side, detail=False, skip=()):
    """Quad strip between two vertex rings.

    A planar quad becomes one 6-sided brush; a non-planar one (which is what a
    helical floor always produces -- two radial edges at different heights are
    skew, so their quad cannot be flat) is split into two triangles first.  The
    extrusion keeps both cases convex by construction.
    """
    out = []
    for i in range(len(ring_lo) - 1):
        if i in skip:            # an APERTURE: this segment of shell is simply absent
            continue
        p00, p01 = ring_lo[i], ring_lo[i + 1]
        p10, p11 = ring_hi[i], ring_hi[i + 1]
        v = offset_fn(i)
        if coplanar(p00, p01, p11, p10):
            out.append(quad_prism(p00, p01, p11, p10, v, tex_cap, tex_side, detail))
        else:
            for tri in ((p00, p01, p11), (p00, p11, p10)):
                br = tri_prism(tri[0], tri[1], tri[2], v, tex_cap, tex_side, detail)
                if br:
                    out.append(br)
    return out


# ---------------------------------------------------------------- the spiral
class Spiral:
    def __init__(self, args):
        self.a = args
        self.rng = random.Random(args.seed)

    def frame(self, t):
        """Centerline point + local radial/tangent basis at parameter t in [0,1]."""
        a = self.a
        th = 2.0 * math.pi * a.turns * t * (1 if a.handed > 0 else -1)
        # seeded wobble keeps a randomizer honest without breaking convexity
        wob = 1.0 + a.wobble * math.sin(th * a.wobble_freq + self.phase)
        r = (a.radius + a.radius_growth * a.turns * t) * wob
        z = a.rise * a.turns * t
        u = (math.cos(th), math.sin(th), 0.0)               # outward radial
        c = (r * u[0], r * u[1], z)
        return c, u

    def aperture_spans(self, n):
        """Segment ranges where the OUTER shell is absent -- the connection sites.

        An aperture is a parameter of the sweep, not a volume carved out of a
        compiled artifact afterwards (design/MAPGEN-ROADMAP.md stage 2). Because
        it is chosen here, its facing, its free volume and the vantage points
        looking through it are all known by construction -- there is nothing to
        recover later by ray marching, and nothing that can disagree with the
        geometry, because it IS the geometry.
        """
        k = int(self.a.apertures)
        if k <= 0:
            return []
        w = max(1, int(self.a.aperture_segments))
        lo, hi = 2, n - 2 - w                      # never on an end cap
        if hi <= lo:
            return []
        return [(lo + int((hi - lo) * (j + 0.5) / k), w) for j in range(k)]

    def build(self):
        a = self.a
        self.phase = self.rng.uniform(0, 2 * math.pi)
        n = int(a.turns * a.segments) + 1
        hw, T = a.width * 0.5, a.thickness

        # four vertex rings: the tube's interior corners, plus outer shells
        ib, ob, it_, ot = [], [], [], []                    # in/out x bottom/top
        centers, radials = [], []
        for i in range(n):
            t = i / float(n - 1)
            c, u = self.frame(t)
            centers.append(c); radials.append(u)
            inner = vadd(c, vmul(u, -hw))
            outer = vadd(c, vmul(u, +hw))
            ib.append(inner)
            ob.append(outer)
            it_.append(vadd(inner, (0, 0, a.height)))
            ot.append(vadd(outer, (0, 0, a.height)))

        brushes = []
        TEXF, TEXW, TEXC = a.tex_floor, a.tex_wall, a.tex_ceil
        CAULK = 'common/caulk'

        # floor slab: spans the full width incl. wall thickness, extruded down.
        # The deliberate corner overlap with the walls is what makes the shell
        # leak-proof without any mitre logic.
        OV = T * a.overlap          # generous overrun, see note below
        fb = [vadd(p, vmul(radials[i], -OV)) for i, p in enumerate(ib)]
        fo = [vadd(p, vmul(radials[i], +OV)) for i, p in enumerate(ob)]
        brushes += strip(fb, fo, lambda i: (0, 0, -T), TEXF, CAULK)

        cb = [vadd(p, vmul(radials[i], -OV)) for i, p in enumerate(it_)]
        co = [vadd(p, vmul(radials[i], +OV)) for i, p in enumerate(ot)]
        brushes += strip(cb, co, lambda i: (0, 0, T), TEXC, CAULK)

        # walls: full height (z-T .. z+H+T) so they cover the floor/ceiling corners
        wb = [vadd(p, (0, 0, -OV)) for p in ib]
        wt = [vadd(p, (0, 0, OV)) for p in it_]
        brushes += strip(wb, wt, lambda i: vmul(radials[i], -T), TEXW, CAULK)
        wb = [vadd(p, (0, 0, -OV)) for p in ob]
        wt = [vadd(p, (0, 0, OV)) for p in ot]
        spans = self.aperture_spans(n)
        holes = set(i for s0, w in spans for i in range(s0, s0 + w))
        brushes += strip(wb, wt, lambda i: vmul(radials[i], +T), TEXW, CAULK, skip=holes)

        # Each aperture gets a PLUG built from the identical strip, kept in a
        # separate list. Standalone the plug ships and the shell is sealed; a join
        # drops the plug and mates a connector to the mouth. Same geometry either
        # way, so a joined map cannot differ from the one that was validated.
        self.plugs = [strip(wb, wt, lambda i: vmul(radials[i], +T), TEXW, CAULK,
                            skip=set(range(n - 1)) - set(range(s0, s0 + w)))
                      for s0, w in spans]
        self.apertures = []
        for j, (s0, w) in enumerate(spans):
            mid = s0 + w // 2
            c, u = centers[mid], radials[mid]
            mouth = vadd(c, vmul(u, hw + T))            # centre of the opening
            self.apertures.append({
                'id': j,
                'origin': [round(v, 2) for v in mouth],
                'normal': [round(v, 4) for v in u],     # outward, into free space
                'width': round(vlen(vsub(centers[s0 + w], centers[s0])), 2),
                'height': round(a.height, 2),
                'segments': [s0, s0 + w],
                # vantages: eye inside looking out, and outside looking in. These
                # are what joinshot turns into info_autoscreenshot entities.
                'vantages': [
                    {'origin': [round(v, 2) for v in vadd(vadd(c, vmul(u, -hw * 0.5)), (0, 0, 40.0))],
                     'angles': [0.0, round(math.degrees(math.atan2(u[1], u[0])) % 360.0, 2), 0.0],
                     'side': 'in'},
                    {'origin': [round(v, 2) for v in vadd(vadd(mouth, vmul(u, 160.0)), (0, 0, 40.0))],
                     'angles': [0.0, round((math.degrees(math.atan2(-u[1], -u[0]))) % 360.0, 2), 0.0],
                     'side': 'out'},
                ],
            })

        # end caps seal the tube; without these the map leaks at both mouths
        for idx, sgn in ((0, -1), (n - 1, +1)):
            c, u = centers[idx], radials[idx]
            tan = vnorm(vsub(centers[min(idx + 1, n - 1)], centers[max(idx - 1, 0)]))
            quad = [vadd(vadd(c, vmul(u, -hw - OV)), (0, 0, -OV)),
                    vadd(vadd(c, vmul(u, +hw + OV)), (0, 0, -OV)),
                    vadd(vadd(c, vmul(u, +hw + OV)), (0, 0, a.height + OV)),
                    vadd(vadd(c, vmul(u, -hw - OV)), (0, 0, a.height + OV))]
            off = vmul(tan, sgn * T * 2.0)
            for tri in ((quad[0], quad[1], quad[2]), (quad[0], quad[2], quad[3])):
                br = tri_prism(tri[0], tri[1], tri[2], off, TEXW, TEXW)
                if br:
                    brushes.append(br)

        return brushes, centers, radials

    def entities(self, centers, radials):
        a = self.a
        n = len(centers)
        ents = []
        step = max(1, n // max(1, a.spawns))
        for i in range(2, n - 2, step):
            c = centers[i]
            nxt = centers[min(i + 1, n - 1)]
            d = vsub(nxt, c)
            yaw = math.degrees(math.atan2(d[1], d[0])) % 360.0
            ents.append(('info_player_deathmatch',
                         {'origin': '%s %s %s' % (fmt(c[0]), fmt(c[1]), fmt(c[2] + 26)),
                          'angle': fmt(yaw)}))
        lstep = max(1, n // max(1, a.lights))
        for i in range(1, n - 1, lstep):
            c = centers[i]
            ents.append(('light',
                         {'origin': '%s %s %s' % (fmt(c[0]), fmt(c[1]), fmt(c[2] + a.height - 24)),
                          'light': str(a.light_intensity)}))
        # weapons/items sprinkled along the climb so the tunnel plays as a route
        pool = ['weapon_machinegun', 'weapon_grenadelauncher', 'weapon_electro',
                'weapon_vortex', 'item_armor_small', 'item_health_medium',
                'weapon_rocketlauncher', 'item_bullets', 'item_rockets']
        istep = max(1, n // max(1, a.items))
        for k, i in enumerate(range(3, n - 3, istep)):
            c = centers[i]
            ents.append((self.rng.choice(pool),
                         {'origin': '%s %s %s' % (fmt(c[0]), fmt(c[1]), fmt(c[2] + 24))}))
        return ents


# ---------------------------------------------------------------- emit
def write_waypoints(path, centers, spacing=320.0):
    """Xonotic bot waypoints: 3 lines each (min origin, max origin, flags).

    A corridor's waypoint graph is just its centerline resampled, so a swept
    tunnel gets bot navigation for free -- the engine links consecutive
    waypoints itself when no .waypoints.cache is present.
    """
    pts, acc = [centers[0]], 0.0
    for i in range(1, len(centers)):
        acc += vlen(vsub(centers[i], centers[i - 1]))
        if acc >= spacing:
            pts.append(centers[i]); acc = 0.0
    if pts[-1] != centers[-1]:
        pts.append(centers[-1])
    L = ['//WAYPOINT_VERSION 1.04', '//WAYPOINT_SYMMETRY 0']
    for c in pts:
        o = "'%s %s %s'" % (fmt(c[0]), fmt(c[1]), fmt(c[2] + 24))
        L += [o, o, '0']
    with open(path, 'w') as f:
        f.write('\n'.join(L) + '\n')
    return len(pts)


def write_svg(path, centers, args):
    """Top view + elevation of the centerline, so a randomizer sweep can be
    eyeballed without launching the game."""
    xs = [c[0] for c in centers]; ys = [c[1] for c in centers]; zs = [c[2] for c in centers]
    R = max(max(map(abs, xs)), max(map(abs, ys))) * 1.1 or 1.0
    H = (max(zs) - min(zs)) or 1.0
    W, PH = 320, 320
    def top(c):  return (W / 2 + c[0] / R * (W / 2), PH / 2 - c[1] / R * (PH / 2))
    def ele(c):
        rr = math.hypot(c[0], c[1])
        return (W + 40 + W / 2 + (rr / R) * (W / 2) * math.cos(math.atan2(c[1], c[0])),
                PH - (c[2] - min(zs)) / H * PH)
    def poly(f):
        return ' '.join('%.1f,%.1f' % f(c) for c in centers)
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" '
           'viewBox="0 0 %d %d">' % (2 * W + 40, PH + 30, 2 * W + 40, PH + 30),
           '<rect x="0" y="0" width="%d" height="%d" fill="#111"/>' % (2 * W + 40, PH + 30),
           '<polyline points="%s" fill="none" stroke="#4ec9b0" stroke-width="2"/>' % poly(top),
           '<polyline points="%s" fill="none" stroke="#dcdcaa" stroke-width="2"/>' % poly(ele),
           '<text x="6" y="%d" fill="#888" font-family="monospace" font-size="11">top  '
           'seed=%d turns=%g r=%g</text>' % (PH + 20, args.seed, args.turns, args.radius),
           '<text x="%d" y="%d" fill="#888" font-family="monospace" font-size="11">'
           'elevation  rise=%g/turn  total=%.0f</text>' % (W + 46, PH + 20, args.rise, max(zs) - min(zs)),
           '</svg>']
    with open(path, 'w') as f:
        f.write('\n'.join(svg))


def write_map(path, brushes, ents, args):
    L = ['// generated by spiralgen.py  seed=%d turns=%g' % (args.seed, args.turns),
         '{', '"classname" "worldspawn"',
         '"_minlight" "%d"' % args.minlight,
         '"message" "%s"' % args.name]
    for b in brushes:
        L.append(b.to_map(args.texscale))
    L.append('}')
    for cls, kv in ents:
        L.append('{')
        L.append('"classname" "%s"' % cls)
        for k, v in kv.items():
            L.append('"%s" "%s"' % (k, v))
        L.append('}')
    L.append('')
    with open(path, 'w') as f:
        f.write('\n'.join(L))


def randomize(a):
    """Draw shape parameters from the seed. ONE definition: main() and e2e.py both
    call this, so a sweep actually varies the geometry. It previously lived inside
    main(), which meant any other caller silently got eight identical maps."""
    if not a.randomize:
        return a
    r = random.Random(a.seed)
    a.turns = round(r.uniform(3, 12), 1)
    a.radius = r.choice([640, 768, 1024, 1280, 1536])
    a.rise = r.choice([512, 640, 768, 896, 1024])
    a.width = r.choice([256, 320, 384, 448])
    a.height = r.choice([224, 256, 320])
    a.radius_growth = r.choice([0, 0, 0, 64, -48])
    a.handed = r.choice([1, -1])
    a.wobble = r.choice([0.0, 0.0, 0.05, 0.1])
    a.segments = r.choice([24, 32, 48])
    return a


def build_parser():
    p = argparse.ArgumentParser(description='Procedural helical-tunnel map generator for Xonotic.')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--name', default='spiral')
    p.add_argument('--out', default='.')
    p.add_argument('--turns', type=float, default=6.0, help='number of full revolutions')
    p.add_argument('--radius', type=float, default=1024.0, help='centerline radius (qu)')
    p.add_argument('--radius-growth', type=float, default=0.0, help='radius added per turn (conical)')
    p.add_argument('--rise', type=float, default=768.0, help='vertical rise per turn (qu)')
    p.add_argument('--width', type=float, default=320.0, help='corridor width (qu)')
    p.add_argument('--height', type=float, default=256.0, help='corridor height (qu)')
    p.add_argument('--thickness', type=float, default=32.0, help='shell thickness (qu)')
    p.add_argument('--segments', type=int, default=32, help='segments per turn')
    p.add_argument('--handed', type=int, default=1, choices=[1, -1], help='1=ccw, -1=cw')
    p.add_argument('--wobble', type=float, default=0.0, help='radial wobble amplitude (0..0.3)')
    p.add_argument('--wobble-freq', type=float, default=3.0)
    p.add_argument('--apertures', type=int, default=0,
                   help='connection sites: gaps in the outer shell that let this map '
                        'socket into others. Each ships PLUGGED so the map still seals '
                        'standalone; a join drops the plug (MAPGEN-ROADMAP stage 2)')
    p.add_argument('--aperture-segments', type=int, default=2,
                   help='width of each aperture, in sweep segments')
    p.add_argument('--spawns', type=int, default=8)
    p.add_argument('--lights', type=int, default=24)
    p.add_argument('--items', type=int, default=10)
    p.add_argument('--light-intensity', type=int, default=300)
    p.add_argument('--minlight', type=int, default=24)
    p.add_argument('--texscale', type=float, default=0.5)
    p.add_argument('--waypoint-spacing', type=float, default=320.0)
    p.add_argument('--overlap', type=float, default=3.0,
                   help='shell overrun in units of --thickness. Adjacent shell pieces '
                        'are offset along their OWN ring radial, so at >1 they overlap '
                        'instead of abutting; exact abutment leaves sub-unit slivers '
                        'that q3map2 reports as leaks on coarse segmentation.')
    p.add_argument('--tex-floor', default='exx/floor/floor_clang01')
    p.add_argument('--tex-wall', default='exx/base/base_crete01')
    p.add_argument('--tex-ceil', default='exx/base/base_metal03')
    p.add_argument('--randomize', action='store_true',
                   help='draw the shape parameters from the seed instead of the flags')
    return p


def main():
    p = build_parser()
    a = p.parse_args()

    a = randomize(a)

    s = Spiral(a)
    brushes, centers, radials = s.build()
    ents = s.entities(centers, radials)
    os.makedirs(a.out, exist_ok=True)
    # plugs ship with the standalone map, so what compiles here is what a join
    # mates to, minus the plug it removes.
    plugged = brushes + [b for plug in getattr(s, 'plugs', []) for b in plug]
    mp = os.path.join(a.out, a.name + '.map')
    write_map(mp, plugged, ents, a)
    aps = getattr(s, 'apertures', [])
    if aps:
        # The contract tools/joinshot.py, joinview.py, fusecheck.py and
        # fusegraph.py read. Written by the generator because every field is a
        # property of the sweep -- not recovered from a compiled artifact.
        joins = {'tiles': [{'name': a.name, 'offset': [0, 0, 0], 'apertures': aps}],
                 'joins': [], 'generator': 'spiralgen', 'seed': a.seed}
        with open(os.path.join(a.out, a.name + '.joins.json'), 'w') as fh:
            json.dump(joins, fh, indent=1)
        print('apertures: %d (plugged), joins.json written' % len(aps))
    write_svg(os.path.join(a.out, a.name + '.svg'), centers, a)
    nwp = write_waypoints(os.path.join(a.out, a.name + '.waypoints'), centers, a.waypoint_spacing)

    zs = [c[2] for c in centers]
    xs = [c[0] for c in centers]; ys = [c[1] for c in centers]
    meta = {'seed': a.seed, 'brushes': len(brushes), 'entities': len(ents),
            'waypoints': nwp,
            'turns': a.turns, 'radius': a.radius, 'rise_per_turn': a.rise,
            'total_rise': round(max(zs) - min(zs), 1),
            'bbox': [round(min(xs), 1), round(min(ys), 1), round(min(zs), 1),
                     round(max(xs), 1), round(max(ys), 1), round(max(zs) + a.height, 1)],
            'path_len': round(sum(vlen(vsub(centers[i + 1], centers[i]))
                                  for i in range(len(centers) - 1)), 1)}
    with open(os.path.join(a.out, a.name + '.meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))
    print('wrote', mp)


if __name__ == '__main__':
    main()
