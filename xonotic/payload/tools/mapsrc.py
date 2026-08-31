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

import os
import re
import subprocess
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


def box_brush(lo, hi, tex, sx=0.5, sy=0.5):
    """An axis-aligned solid box as six Valve-220 faces."""
    fs = []
    axes = [([0.0, 1.0, 0.0], [0.0, 0.0, -1.0]),
            ([1.0, 0.0, 0.0], [0.0, 0.0, -1.0]),
            ([1.0, 0.0, 0.0], [0.0, -1.0, 0.0])]
    defs = [(0, hi[0], 1), (0, lo[0], -1), (1, hi[1], 1), (1, lo[1], -1),
            (2, hi[2], 1), (2, lo[2], -1)]
    for ax, val, sgn in defs:
        u, v = (1, 2) if ax == 0 else ((0, 2) if ax == 1 else (0, 1))
        p = []
        for du, dv in ((0, 0), (1, 0), (1, 1)) if sgn > 0 else ((0, 0), (0, 1), (1, 1)):
            q = [0.0, 0.0, 0.0]
            q[ax] = val
            q[u] = hi[u] if du else lo[u]
            q[v] = hi[v] if dv else lo[v]
            p.append(q)
        f = Face()
        f.p = p
        f.tex = tex
        f.ua, f.va = list(axes[ax][0]), list(axes[ax][1])
        f.us = f.vs = 0.0
        f.rot, f.sx, f.sy = 0.0, sx, sy
        f.tail = ''
        fs.append(f)
    return fs


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


if __name__ == '__main__':
    m = parse_map(sys.argv[1])
    print('entities=%d worldspawn brushes=%d patches=%d'
          % (len(m), len(m[0].brushes) if m else 0, len(m[0].patches) if m else 0))
