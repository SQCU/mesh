#!/usr/bin/env python3
"""mapsrc -- assemble a fused world as q3map2 .map SOURCE, not as BSP lumps.

WHY THIS REPLACES THE LUMP WRITER
---------------------------------
`mapfuse` writes BSP lumps directly, which means it must synthesise the BSP tree,
the PVS and the lightmaps itself.  It does not: the shipped 29-tile world has
`visdata len = 0` (no PVS at all), `lightvols len = 0` (no light grid), ONE
128x128 grey lightmap for 200 946 faces, and 67 416 leafs collapsed onto 2
distinct cluster indices.  With empty visdata the visible set is the WHOLE world,
so all 634 live shaders are resident at once and panning the camera changes the
visible set and thrashes the texture cache.  Bad occlusion, draw-call latency and
"panning retextures the scene" are one bug with three faces, and it cannot be
repaired in a lump writer -- only replaced by a real compile.

The route: get .map source for every tile, place it, add the procedural
connectors, and hand the whole thing to q3map2 for tree + VIS + lighting +
collision, exactly as `mapgen/spiralgen.py` + `build.sh` already do.

STOCK MAPS COME BACK AS SOURCE, LOSSLESSLY ENOUGH
-------------------------------------------------
An earlier draft of FUSION-SPEC 8.7 claimed this route was blocked because
13 091 patch faces have no brush representation and brush-face texture alignment
is not recoverable from a BSP.  That was wrong: q3map2 decompiles its own format.
`-convert -format map_220 -readbsp` returns Valve-220 source with explicit
texture AXES (so alignment is exact, not defaulted) and real `patchDef2` blocks.
Verified on trident: 599 brushes -> 599 brushes, 177 patch faces -> 177
patchDefs, 97 entities.  Recompiled, that source yields 484 clusters, 30 920
bytes of visdata and a 56 749-entry light grid -- against the stock map's 489 /
31 240 / 56 749.

TRANSLATION IS EXACT, INCLUDING TEXTURE ALIGNMENT
-------------------------------------------------
Placing a tile means translating its geometry by the pack offset.  In Valve 220 a
face's texture coordinate is `u = (P . axis_u)/scale_u + shift_u`, so moving the
geometry by `t` and keeping the same texture on the same surface requires
`shift_u' = shift_u - (axis_u . t)/scale_u`.  Brush face points and patch control
points translate directly.  Getting this wrong is invisible in geometry and
obvious in game, so it is done here once rather than per caller.
"""

import math
import os
import re
import subprocess
import numpy as np
import sys

Q3MAP2 = os.path.expanduser('~/dox/xonotic/netradiant-custom/install/q3map2.arm64')
XONDIR = os.path.expanduser('~/dox/xonotic/Xonotic')

FACE_RE = re.compile(
    r'\(\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*\)\s*'
    r'\(\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*\)\s*'
    r'\(\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*\)\s*'
    r'(\S+)\s*'
    r'\[\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*\]\s*'
    r'\[\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*\]\s*'
    r'([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)(.*)$')

CTRL_RE = re.compile(r'\(\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*\)')


def decompile(bsp, workdir, name=None):
    """Stock BSP -> Valve-220 .map source, via q3map2's own converter (cached)."""
    name = name or os.path.splitext(os.path.basename(bsp))[0]
    maps = os.path.join(workdir, 'fs', 'data', 'maps')
    os.makedirs(maps, exist_ok=True)
    dst = os.path.join(maps, name + '.bsp')
    if os.path.abspath(bsp) != os.path.abspath(dst):
        open(dst, 'wb').write(open(bsp, 'rb').read())
    out = os.path.join(maps, name + '_converted.map')
    if not os.path.exists(out) or os.path.getmtime(out) < os.path.getmtime(dst):
        subprocess.run([Q3MAP2, '-game', 'xonotic', '-fs_basepath', XONDIR,
                        '-fs_homepath', os.path.join(workdir, 'fs'),
                        '-convert', '-format', 'map_220', '-readbsp', dst],
                       capture_output=True)
    return out


class Face(object):
    __slots__ = ('p', 'tex', 'ua', 'us', 'va', 'vs', 'rot', 'sx', 'sy', 'tail')

    def translate(self, t):
        self.p = [[c[i] + t[i] for i in range(3)] for c in self.p]
        # Valve 220: u = (P.axis)/scale + shift.  Moving P by t must leave u the
        # same, so the shift absorbs the projection of t onto the axis.
        self.us -= (self.ua[0] * t[0] + self.ua[1] * t[1] + self.ua[2] * t[2]) / (self.sx or 1.0)
        self.vs -= (self.va[0] * t[0] + self.va[1] * t[1] + self.va[2] * t[2]) / (self.sy or 1.0)

    def text(self):
        pts = ' '.join('( %.6g %.6g %.6g )' % tuple(c) for c in self.p)
        return ('%s %s [ %.8f %.8f %.8f %.8f ] [ %.8f %.8f %.8f %.8f ] %g %.8f %.8f%s'
                % (pts, self.tex, self.ua[0], self.ua[1], self.ua[2], self.us,
                   self.va[0], self.va[1], self.va[2], self.vs,
                   self.rot, self.sx, self.sy, self.tail))


class Patch(object):
    """A patchDef block, preserved VERBATIM.

    The block has structure -- an inner brace, a shader line, a dimension line
    `( 9 3 0 0 0 )`, then the control grid inside its own parens -- and the
    dimension line is a 5-tuple exactly like a control point, so a naive
    "rewrite every 5-tuple" pass corrupts it.  Keeping the raw lines and
    translating only the lines that hold NESTED tuples (`( ( x y z s t ) ... )`,
    which the dimension line never is) means the emitted block is byte-identical
    to q3map2's own output apart from the coordinates that had to move.
    """
    __slots__ = ('lines',)

    def __init__(self):
        self.lines = []

    def translate(self, t):
        out = []
        for ln in self.lines:
            if '( (' in ln:
                def fix(m):
                    v = [float(x) for x in m.groups()]
                    return ('( %.6f %.6f %.6f %.6f %.6f )'
                            % (v[0] + t[0], v[1] + t[1], v[2] + t[2], v[3], v[4]))
                ln = CTRL_RE.sub(fix, ln)
            out.append(ln)
        self.lines = out

    def text(self):
        return '\n'.join(self.lines)


class Ent(object):
    def __init__(self):
        self.keys = []
        self.brushes = []   # list of list-of-Face
        self.patches = []

    def translate(self, t):
        for b in self.brushes:
            for f in b:
                f.translate(t)
        for p in self.patches:
            p.translate(t)
        for i, (k, v) in enumerate(self.keys):
            if k == 'origin':
                try:
                    o = [float(x) for x in v.split()]
                    self.keys[i] = (k, '%.6g %.6g %.6g' % (o[0] + t[0], o[1] + t[1], o[2] + t[2]))
                except ValueError:
                    pass

    def get(self, k, default=''):
        for kk, vv in self.keys:
            if kk == k:
                return vv
        return default


def parse_map(path):
    """Parse Valve-220 .map source into entities with brushes and patches."""
    ents = []
    cur = None
    brush = None
    patch = None
    depth = 0
    for raw in open(path, encoding='latin-1'):
        line = raw.strip()
        if not line or line.startswith('//'):
            continue
        # A patch owns its own braces: patchDef2 sits at depth 2 and opens a
        # depth-3 block, so capture verbatim until the depth falls back below 2.
        if patch is not None:
            if line == '{':
                depth += 1
                patch.lines.append(line)
                continue
            if line == '}':
                depth -= 1
                if depth >= 2:
                    patch.lines.append(line)
                    continue
                cur.patches.append(patch)
                patch = None
                brush = None
                continue
            patch.lines.append(line)
            continue
        if line == '{':
            depth += 1
            if depth == 1:
                cur = Ent()
            elif depth == 2:
                brush = []
            continue
        if line == '}':
            if depth == 2:
                if brush:
                    cur.brushes.append(brush)
                brush = None
            elif depth == 1 and cur is not None:
                ents.append(cur)
                cur = None
            depth -= 1
            continue
        if depth == 1 and line.startswith('"'):
            m = re.match(r'"([^"]*)"\s+"(.*)"$', line)
            if m:
                cur.keys.append((m.group(1), m.group(2)))
            continue
        if depth >= 2:
            if line.startswith('patchDef'):
                patch = Patch()
                patch.lines = [line]
                continue
            m = FACE_RE.match(line)
            if m:
                g = m.groups()
                f = Face()
                f.p = [[float(g[0]), float(g[1]), float(g[2])],
                       [float(g[3]), float(g[4]), float(g[5])],
                       [float(g[6]), float(g[7]), float(g[8])]]
                f.tex = g[9]
                f.ua = [float(g[10]), float(g[11]), float(g[12])]
                f.us = float(g[13])
                f.va = [float(g[14]), float(g[15]), float(g[16])]
                f.vs = float(g[17])
                f.rot = float(g[18])
                f.sx = float(g[19])
                f.sy = float(g[20])
                f.tail = g[21] or ''
                if brush is not None:
                    brush.append(f)
    return ents



# ---------------------------------------------------------------------------
# CSG IN SOURCE.  The aperture cut belongs here, not on compiled lumps.
# ---------------------------------------------------------------------------
# q3map2 is a functor from .map source to BSP, and `-convert -readbsp` is a
# section of it, so the source category is always reachable.  Cutting a doorway
# as an endomorphism on BSP means re-deriving by hand everything the functor
# produces -- tree, PVS, lightmaps, light grid -- which is exactly why the lump
# writer shipped `Visdata len = 0` and one grey lightmap.  Expressed here, the
# compiler owns those steps, and two whole classes of defect become
# INEXPRESSIBLE: a degenerate collision brush and a duplicate carve are things
# you can only produce by hand-authoring compiled geometry.
#
# PLANE CONVENTION, measured not assumed: in Quake .map the three face points
# are ordered so (p1-p0) x (p2-p0) points INTO the brush.  Verified on
# decompiled trident -- for all 10 faces of brush 0 the centroid satisfies
# n.c > d.  So a face's OUTWARD half-space is (-n, -d).


def face_plane(f):
    """Outward plane (N, D) of a face: brush interior satisfies N.p <= D."""
    p0, p1, p2 = [np.asarray(x, dtype=float) for x in f.p]
    n = np.cross(p1 - p0, p2 - p0)
    ln = np.linalg.norm(n)
    if ln < 1e-9:
        return None, None
    n = n / ln
    return -n, float(-(n @ p0))


def make_face(N, D, proto):
    """A face with outward plane (N, D), borrowing `proto`'s texture and axes.

    Valve-220 texture axes are world-space, so a cut surface inheriting the
    wall's axes lines up with the wall it was cut from."""
    N = np.asarray(N, dtype=float)
    N = N / (np.linalg.norm(N) or 1.0)
    o = N * D
    a = np.array([1.0, 0.0, 0.0]) if abs(N[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(N, a)
    u /= (np.linalg.norm(u) or 1.0)
    v = np.cross(N, u)
    f = Face()
    # cross(v, u) = -N, i.e. inward, which is the .map convention
    f.p = [[float(x) for x in o], [float(x) for x in (o + v * 64.0)],
           [float(x) for x in (o + u * 64.0)]]
    f.tex = proto.tex
    f.ua, f.us = list(proto.ua), proto.us
    f.va, f.vs = list(proto.va), proto.vs
    f.rot, f.sx, f.sy = proto.rot, proto.sx, proto.sy
    f.tail = proto.tail
    return f


def brush_bounds(faces, lo0=-131072.0, hi0=131072.0):
    """AABB of a brush from its own outward planes (interval propagation)."""
    from negspace import bounds_of
    rows = []
    for f in faces:
        N, D = face_plane(f)
        if N is not None:
            rows.append([N[0], N[1], N[2], D])
    if not rows:
        return None, None
    return bounds_of(np.array(rows, dtype=float),
                     np.array([lo0] * 3), np.array([hi0] * 3))


def subtract_box(faces, lo, hi):
    r"""brush \ axis-aligned box, as convex remainder brushes.

    For each box half-space h_q emit (brush AND outside-h_q AND inside
    h_0..h_{q-1}); every piece is convex and their union is exactly the brush
    minus the box.  Empty pieces are dropped on AABB alone, which is SOUND
    because the box contains the piece; the converse is never used.  A sliver
    that slips through is q3map2's problem now, not a degenerate collision
    brush, because q3map2 recomputes the hull."""
    blo, bhi = brush_bounds(faces)
    if blo is None:
        return [faces]
    if any(blo[i] >= hi[i] or bhi[i] <= lo[i] for i in range(3)):
        return [faces]
    proto = faces[0]
    planes = []
    for a in range(3):
        e = [0.0, 0.0, 0.0]
        e[a] = 1.0
        planes.append((e[:], hi[a]))
        e2 = [0.0, 0.0, 0.0]
        e2[a] = -1.0
        planes.append((e2, -lo[a]))
    out, acc = [], []
    for N, D in planes:
        piece = list(faces) + [make_face([-N[0], -N[1], -N[2]], -D, proto)]
        piece += [make_face(An, Ad, proto) for An, Ad in acc]
        plo, phi = brush_bounds(piece)
        if plo is not None and all(phi[i] - plo[i] > 0.5 for i in range(3)):
            out.append(piece)
        acc.append((N, D))
    return out


# ---------------------------------------------------------------------------
# THE AUTHORED-SOLID PRIMITIVE.  One definition.
# ---------------------------------------------------------------------------
# This replaces both `box_brush`/`tube` (here) and `Brush`/`tri_prism`/
# `quad_prism` (spiralgen).  They were two implementations of one insight, and
# their docstrings say the same sentence twice --
#   spiralgen: "Point winding is never trusted: normals are derived and then
#              flipped to point away from the brush centroid"
#   mapsrc:    "the winding is verified directly ... rather than trusted"
# -- the second written only after a hand-ordered winding shipped four of six
# faces inside-out, so a connector tube was not solid and the leak was reported
# against an unrelated entity 1400 units away.  Deriving the sign numerically
# is correct for ANY convex solid, so the general primitive absorbs the
# axis-aligned special case rather than sitting beside it.
#
# TEXTURE AXES: spiralgen emitted the standard Quake face format
# (`tex 0 0 0 sx sy flags`), which cannot appear in a Valve-220 file, and a .map
# is one format throughout.  Since the decompiled tiles are Valve 220, the
# primitive emits Valve 220, and the per-face axis choice lives here once
# instead of at each call site.


def _valve_axes(n):
    """Texture axes for a face normal: the two world axes most perpendicular to
    it.  Choosing by the normal's dominant axis is what stops a face getting a
    degenerate (zero-area) texture projection."""
    ax = max(range(3), key=lambda i: abs(n[i]))
    if ax == 0:
        return [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]
    if ax == 1:
        return [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]
    return [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]


class Brush(object):
    """A convex solid held as faces; winding resolved numerically on emit."""

    def __init__(self, detail=False):
        self.faces = []
        self.detail = detail

    def add_face(self, pts, tex):
        self.faces.append(([list(p) for p in pts], tex))
        return self

    def centroid(self):
        allp = [p for pts, _ in self.faces for p in pts]
        n = float(len(allp))
        return [sum(p[i] for p in allp) / n for i in range(3)]

    def to_faces(self, scale=0.5):
        c = np.asarray(self.centroid(), dtype=float)
        out = []
        for pts, tex in self.faces:
            p0, p1, p2 = [np.asarray(pts[i], dtype=float) for i in (0, 1, 2)]
            n = np.cross(p2 - p0, p1 - p0)
            ln = np.linalg.norm(n)
            if ln < 1e-9:
                continue                                  # degenerate, drop it
            n = n / ln
            if float(n @ (c - p0)) > 0:                   # points inward: flip
                p1, p2 = p2, p1
                n = -n
            f = Face()
            f.p = [[float(x) for x in p0], [float(x) for x in p1], [float(x) for x in p2]]
            f.tex = tex
            f.ua, f.va = _valve_axes(n)
            f.us = f.vs = 0.0
            f.rot = 0.0
            f.sx = f.sy = scale
            f.tail = ' %d 0 0' % (134217728 if self.detail else 0)
            out.append(f)
        return out


def _v(a, b=None):
    return np.asarray(a, dtype=float) if b is None else np.asarray(b, dtype=float) - np.asarray(a, dtype=float)


def tri_prism(a, b, c, v, tex_cap, tex_side, detail=False):
    """Triangle (a,b,c) extruded by v -> convex 5-sided brush.  Convex for any
    non-degenerate triangle and any non-parallel v, which is why this is the
    only primitive an authored connector needs."""
    if np.linalg.norm(np.cross(_v(a, b), _v(a, c))) < 1e-6:
        return None
    a2, b2, c2 = [list(np.asarray(p, float) + np.asarray(v, float)) for p in (a, b, c)]
    br = Brush(detail)
    br.add_face([a, b, c], tex_cap)
    br.add_face([a2, b2, c2], tex_cap)
    br.add_face([a, b, b2, a2], tex_side)
    br.add_face([b, c, c2, b2], tex_side)
    br.add_face([c, a, a2, c2], tex_side)
    return br


def quad_prism(a, b, c, d, v, tex_cap, tex_side, detail=False):
    """Planar quad extruded by v -> convex 6-sided brush."""
    a2, b2, c2, d2 = [list(np.asarray(p, float) + np.asarray(v, float)) for p in (a, b, c, d)]
    br = Brush(detail)
    br.add_face([a, b, c, d], tex_cap)
    br.add_face([a2, b2, c2, d2], tex_cap)
    for (p, q, p2, q2) in ((a, b, a2, b2), (b, c, b2, c2), (c, d, c2, d2), (d, a, d2, a2)):
        br.add_face([p, q, q2, p2], tex_side)
    return br


def connector(a, b, width, height, thickness=32.0, overlap=2.0,
              tex_floor='common/caulk', tex_wall='common/caulk',
              tex_ceil='common/caulk'):
    """A swept prism corridor from mouth `a` to mouth `b`, authored SEALED.

    `a` and `b` are the floor-centre points of the two mouths.  The swept
    cross-section IS the aperture: width x height, which is the parameter the
    caller also cuts through the tile shell, so the opening is a parameter of
    the sweep rather than something discovered afterwards.

    Shell pieces are offset along their own normal by `overlap * thickness` so
    they OVERLAP rather than abut.  Exact abutment leaves sub-unit slivers that
    q3map2 reports as leaks, which is why spiralgen defaults this to >= 2.

    The ends are deliberately open: they are the doorways."""
    A = np.asarray(a, dtype=float)
    B = np.asarray(b, dtype=float)
    d = B - A
    L = float(np.linalg.norm(d))
    if L < 1.0:
        return []
    dirh = d / L
    up = np.array([0.0, 0.0, 1.0])
    side = np.cross(dirh, up)
    if np.linalg.norm(side) < 1e-6:
        side = np.array([1.0, 0.0, 0.0])
    side = side / np.linalg.norm(side)
    hw = width / 2.0
    T = thickness
    OV = T * overlap
    out = []
    # extend the sweep INTO both mouths so the shell overlaps the tile hull
    A2 = A - dirh * OV
    B2 = B + dirh * OV

    def quad(off_lo, off_hi):
        return [list(A2 + off_lo), list(B2 + off_lo), list(B2 + off_hi), list(A2 + off_hi)]

    # floor and ceiling: full width plus the wall footprint, extruded outward
    f_lo, f_hi = side * -(hw + T), side * (hw + T)
    out.append(quad_prism(*quad(f_lo, f_hi), v=list(up * -T),
                          tex_cap=tex_floor, tex_side=tex_floor))
    out.append(quad_prism(*[list(np.asarray(p) + up * height) for p in quad(f_lo, f_hi)],
                          v=list(up * T), tex_cap=tex_ceil, tex_side=tex_ceil))
    # side walls: full height including the floor/ceiling corners
    for sgn in (-1.0, 1.0):
        base = side * (sgn * hw)
        q = [list(np.asarray(p) - up * OV) for p in quad(base, base)]
        q = [q[0], q[1],
             list(np.asarray(q[1]) + up * (height + 2 * OV)),
             list(np.asarray(q[0]) + up * (height + 2 * OV))]
        out.append(quad_prism(*q, v=list(side * (sgn * T)),
                              tex_cap=tex_wall, tex_side=tex_wall))
    return [b for b in out if b is not None]


def write_map(path, ents):
    out = []
    for i, e in enumerate(ents):
        out.append('// entity %d\n{' % i)
        for k, v in e.keys:
            out.append('"%s" "%s"' % (k, v))
        n = 0
        for b in e.brushes:
            out.append('// brush %d\n{' % n)
            n += 1
            for f in b:
                out.append(f.text())
            out.append('}')
        for p in e.patches:
            out.append('{')
            out.append(p.text())
            out.append('}')
        out.append('}')
    open(path, 'w', encoding='latin-1').write('\n'.join(out) + '\n')
    return path


def compile_map(mappath, workdir, vis=True, light=True, extra=()):
    """Run the proven q3map2 pipeline.  Returns (ok, leaked, logs)."""
    base = [Q3MAP2, '-game', 'xonotic', '-fs_basepath', XONDIR,
            '-fs_homepath', os.path.join(workdir, 'fs')]
    logs = {}
    r = subprocess.run(base + ['-meta'] + list(extra) + [mappath],
                       capture_output=True, text=True)
    logs['meta'] = r.stdout + r.stderr
    leaked = 'leaked' in logs['meta'].lower()
    if r.returncode != 0:
        return False, leaked, logs
    if vis and not leaked:
        r = subprocess.run(base + ['-vis', mappath], capture_output=True, text=True)
        logs['vis'] = r.stdout + r.stderr
    if light:
        r = subprocess.run(base + ['-light', '-fast', '-bounce', '1', mappath],
                           capture_output=True, text=True)
        logs['light'] = r.stdout + r.stderr
    return True, leaked, logs



# ---------------------------------------------------------------------------
# THE GENERATOR'S OWN RECORD OF WHAT IT BUILT
# ---------------------------------------------------------------------------
# `fused.joins.json` is read by joinshot, joinview, fusecheck and fusegraph.  It
# used to be written by mapfuse's `fuse()`, which recovered these fields by
# archaeology on geometry it had already cut.  Authored, every one of them is an
# INPUT to the sweep rather than a measurement of its aftermath: an aperture's
# two sides, its facing, and the points that look through it are parameters, so
# this record is exact by construction instead of approximate by recovery.



# shaders that must NEVER survive a merge, dropped unconditionally at placement
MERGE_DROP_SHADERS = frozenset(('common/lightgrid',))


def place_tile(bsp, off, workdir, name=None, keep_classes=None):
    """Decompile one tile and PLACE it: the single entry point for putting a map
    into a fused world.

    The `common/lightgrid` drop lives here, unconditionally, because it is a
    MERGE INVARIANT and its failure is silent: a lightgrid brush clips the
    compiled world to its own volume, so the first tile's box culls every other
    tile -- brushes still in the lump, no leak reported, map boots, one tile
    simply gone.  Anything that places a tile by another route reintroduces that,
    so there should not be another route.

    Returns (world_brushes, patches, entities, n_dropped)."""
    name = name or os.path.splitext(os.path.basename(bsp))[0]
    ents = parse_map(decompile(bsp, workdir, name))
    keep_classes = keep_classes or (lambda cn: cn == 'light' or cn.startswith(
        ('info_player', 'item_', 'weapon_')))
    brushes, patches, out, ndrop = [], [], [], 0
    for i, e in enumerate(ents):
        e.translate(off)
        if i == 0:
            for b in e.brushes:
                if any(f.tex in MERGE_DROP_SHADERS for f in b):
                    ndrop += 1
                    continue
                brushes.append(b)
            patches += e.patches
        elif keep_classes(e.get('classname')):
            out.append(e)
    return brushes, patches, out, ndrop


def joins_record(tiles, joins, portals, bot_jumps=(), vantages_per_tile=None):
    """Build the `fused.joins.json` structure.

    tiles    -- [{'name', 'mins', 'maxs', 'bridge', 'degree'}]
    joins    -- [{'a','b','kind','sa','sb','exclusive','prominent'}] (length derived)
    portals  -- [{'tile','name','kind','axis','sgn','node','mouth','aperture'}]
    """
    out_tiles = []
    for i, t in enumerate(tiles):
        v = (vantages_per_tile or {}).get(i, [])
        out_tiles.append({'name': t['name'],
                          'mins': [float(x) for x in t['mins']],
                          'maxs': [float(x) for x in t['maxs']],
                          'bridge': bool(t.get('bridge', False)),
                          'degree': int(t.get('degree', 0)),
                          'vantages': [[float(c) for c in p] for p in v]})
    out_joins = []
    for j in joins:
        sa = [float(x) for x in j['sa']]
        sb = [float(x) for x in j['sb']]
        out_joins.append({'a': int(j['a']), 'b': int(j['b']), 'kind': j['kind'],
                          'sa': sa, 'sb': sb,
                          'length': round(float(math.dist(sa, sb)), 1),
                          'exclusive': bool(j.get('exclusive', True)),
                          'prominent': bool(j.get('prominent', True)),
                          'cart_navigable': j['kind'] == 'corridor'})
    out_portals = []
    for p in portals:
        out_portals.append({'tile': int(p['tile']), 'name': p['name'],
                            'kind': p.get('kind', 'continue'),
                            'axis': int(p['axis']), 'sgn': float(p['sgn']),
                            'node': [float(x) for x in p['node']],
                            'mouth': [float(x) for x in p['mouth']],
                            'aperture': [[float(x) for x in p['aperture'][0]],
                                         [float(x) for x in p['aperture'][1]]]})
    return {'maps': out_tiles, 'joins': out_joins, 'portals': out_portals,
            'bot_jumps': [[[float(x) for x in n], [float(x) for x in f]]
                          for n, f in bot_jumps]}


def write_joins_json(path, rec):
    import json as _json
    _json.dump(rec, open(path, 'w'), indent=0)
    return path


if __name__ == '__main__':
    m = parse_map(sys.argv[1])
    print('entities=%d worldspawn brushes=%d patches=%d'
          % (len(m), len(m[0].brushes) if m else 0, len(m[0].patches) if m else 0))
