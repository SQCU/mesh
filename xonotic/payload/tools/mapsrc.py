#!/usr/bin/env mesh-python
import json
import glob
import math
import os
import posixpath
import re
import stat
import subprocess
import zipfile
import numpy as np
import sys
from functools import lru_cache

Q3MAP2 = os.environ.get(
    'Q3MAP2', os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'bin', 'mesh-q3map2')),
)
XONDIR = os.environ.get('XON_BASEPATH', os.path.expanduser('~/dox/xonotic/Xonotic'))

FACE_RE = re.compile(
    r'\(\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*\)\s*'
    r'\(\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*\)\s*'
    r'\(\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*\)\s*'
    r'(\S+)\s*'
    r'\[\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*\]\s*'
    r'\[\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*\]\s*'
    r'([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)(.*)$')

BP_FACE_RE = re.compile(
    r'\(\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*\)\s*'
    r'\(\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*\)\s*'
    r'\(\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*\)\s*'
    r'\(\s*\(\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*\)\s*'
    r'\(\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*\)\s*\)\s*'
    r'(\S+)(.*)$')

QUAKE_FACE_RE = re.compile(
    r'\(\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*\)\s*'
    r'\(\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*\)\s*'
    r'\(\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*\)\s*'
    r'(\S+)\s*'
    r'([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+'
    r'([-\d.eE+]+)\s+([-\d.eE+]+)(.*)$')

CTRL_RE = re.compile(r'\(\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*\)')
COMPILER_ALIAS_EXTENSIONS = frozenset(('.jpg', '.jpeg', '.png', '.tga', '.pcx',
                                       '.bmp', '.md3', '.ase', '.obj', '.skin',
                                       '.iqm', '.mdl'))

class Face(object):
    __slots__ = ('p', 'tex', 'bp', 'tail')

    def translate(self, t):
        n, _ = face_plane(self)
        if n is not None:
            tx, ty = brush_primitive_axes(n)
            u = sum(t[i] * tx[i] for i in range(3))
            v = sum(t[i] * ty[i] for i in range(3))
            self.bp[0][2] -= self.bp[0][0] * u + self.bp[0][1] * v
            self.bp[1][2] -= self.bp[1][0] * u + self.bp[1][1] * v
        self.p = [[c[i] + t[i] for i in range(3)] for c in self.p]

    def text(self):
        pts = ' '.join('( %.10g %.10g %.10g )' % tuple(c) for c in self.p)
        return ('%s ( ( %.10g %.10g %.10g ) ( %.10g %.10g %.10g ) ) %s%s'
                % (pts, self.bp[0][0], self.bp[0][1], self.bp[0][2],
                   self.bp[1][0], self.bp[1][1], self.bp[1][2], self.tex, self.tail))

class Patch(object):
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

    def control(self):
        rows = []
        for line in self.lines:
            values = CTRL_RE.findall(line)
            if len(values) >= 3:
                rows.append([[float(value) for value in point] for point in values])
        if not rows or len({len(row) for row in rows}) != 1:
            return np.zeros((0, 0, 5), dtype=np.float64)
        return np.asarray(rows, dtype=np.float64)

    def bounds(self):
        control = self.control()
        if not control.size:
            return None, None
        points = control[:, :, :3].reshape((-1, 3))
        return points.min(axis=0), points.max(axis=0)

    def quadratic_blocks(self):
        control = self.control()
        height, width = control.shape[:2]
        if height < 3 or width < 3 or not height % 2 or not width % 2:
            return [self]
        return [
            self._with_control(control[y:y + 3, x:x + 3])
            for y in range(0, height - 1, 2)
            for x in range(0, width - 1, 2)
        ]

    def _with_control(self, control):
        rows = [index for index, line in enumerate(self.lines)
                if len(CTRL_RE.findall(line)) >= 3]
        prefix = list(self.lines[:rows[0]])
        suffix = list(self.lines[rows[-1] + 1:])
        for index, line in enumerate(prefix):
            values = re.findall(r'[-\d.eE+]+', line)
            if line.startswith('(') and len(values) >= 2:
                values[0] = str(control.shape[1])
                values[1] = str(control.shape[0])
                prefix[index] = '( %s )' % ' '.join(values)
        patch = Patch()
        patch.lines = prefix + [
            '( ' + ' '.join(
                '( %.10g %.10g %.10g %.10g %.10g )' % tuple(point)
                for point in row
            ) + ' )'
            for row in control
        ] + suffix
        return patch

    def subdivide_quadratic(self):
        control = self.control()
        if control.shape[:2] != (3, 3):
            return self.quadratic_blocks()
        horizontal = np.empty((3, 5, 5), dtype=np.float64)
        horizontal[:, 0] = control[:, 0]
        horizontal[:, 1] = 0.5 * (control[:, 0] + control[:, 1])
        horizontal[:, 2] = 0.25 * (control[:, 0] + 2.0 * control[:, 1] + control[:, 2])
        horizontal[:, 3] = 0.5 * (control[:, 1] + control[:, 2])
        horizontal[:, 4] = control[:, 2]
        realized = np.empty((5, 5, 5), dtype=np.float64)
        realized[0] = horizontal[0]
        realized[1] = 0.5 * (horizontal[0] + horizontal[1])
        realized[2] = 0.25 * (horizontal[0] + 2.0 * horizontal[1] + horizontal[2])
        realized[3] = 0.5 * (horizontal[1] + horizontal[2])
        realized[4] = horizontal[2]
        return [
            self._with_control(realized[y:y + 3, x:x + 3])
            for y in (0, 2) for x in (0, 2)
        ]

class Ent(object):
    def __init__(self):
        self.keys = []
        self.brushes = []
        self.patches = []
        self.translation_error_mass = 0
        self.implicit_origin_mass = 0

    def translate(self, t):
        for b in self.brushes:
            for f in b:
                f.translate(t)
        for p in self.patches:
            p.translate(t)
        for i, (k, v) in enumerate(self.keys):
            if k == 'origin':
                try:
                    parts = v.strip().strip("'").split()
                    if len(parts) != 3:
                        raise ValueError(f"origin coordinate mass {len(parts)}")
                    o = [float(x) for x in parts]
                    self.keys[i] = (k, '%.6g %.6g %.6g' % (o[0] + t[0], o[1] + t[1], o[2] + t[2]))
                except ValueError as error:
                    self.translation_error_mass += 1
                    print(json.dumps({"event":"origin_translation_error","origin":v,"error":f"{type(error).__name__}: {error}"}), file=sys.stderr)
        if (not self.brushes and not self.patches
                and self.get('classname') != 'worldspawn'
                and not any(key == 'origin' for key, _ in self.keys)):
            self.keys.append(('origin', '%.6g %.6g %.6g' % tuple(t)))
            self.implicit_origin_mass += 1
        return self.translation_error_mass

    def get(self, k, default=''):
        for kk, vv in self.keys:
            if kk == k:
                return vv
        return default

def brush_primitive_axes(normal):
    n = [0.0 if abs(value) < 1e-6 else value for value in normal]
    ry = -math.atan2(n[2], math.hypot(n[1], n[0]))
    rz = math.atan2(n[1], n[0])
    return ([-math.sin(rz), math.cos(rz), 0.0],
            [-math.sin(ry) * math.cos(rz),
             -math.sin(ry) * math.sin(rz), -math.cos(ry)])

def texture_matrix(face, u_vector, u_shift, v_vector, v_shift):
    normal, distance = face_plane(face)
    x_axis, y_axis = brush_primitive_axes(normal)
    rows = []
    for vector, shift in ((u_vector, u_shift), (v_vector, v_shift)):
        rows.append([
            sum(vector[i] * x_axis[i] for i in range(3)) / 64.0,
            sum(vector[i] * y_axis[i] for i in range(3)) / 64.0,
            (shift + distance * sum(vector[i] * normal[i] for i in range(3))) / 64.0,
        ])
    return rows

def quake_texture_vectors(normal, rotate, scale_u, scale_v):
    axes = (
        ((0, 0, 1), (1, 0, 0), (0, -1, 0)),
        ((0, 0, -1), (1, 0, 0), (0, -1, 0)),
        ((1, 0, 0), (0, 1, 0), (0, 0, -1)),
        ((-1, 0, 0), (0, 1, 0), (0, 0, -1)),
        ((0, 1, 0), (1, 0, 0), (0, 0, -1)),
        ((0, -1, 0), (1, 0, 0), (0, 0, -1)),
    )
    _, u, v = max(axes, key=lambda row: sum(row[0][i] * normal[i] for i in range(3)))
    values = [list(u), list(v)]
    su = next((i for i, value in enumerate(values[0]) if value), 2)
    sv = next((i for i, value in enumerate(values[1]) if value), 2)
    angle = math.radians(rotate)
    sine, cosine = math.sin(angle), math.cos(angle)
    for row in values:
        a, b = row[su], row[sv]
        row[su] = cosine * a - sine * b
        row[sv] = sine * a + cosine * b
    return ([value / (scale_u or 1.0) for value in values[0]],
            [value / (scale_v or 1.0) for value in values[1]])

def parse_map(path):
    ents = []
    cur = None
    brush = None
    patch = None
    depth = 0
    for raw in open(path, encoding='latin-1'):
        line = raw.strip()
        if not line or line.startswith('//'):
            continue

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
            m = BP_FACE_RE.match(line)
            if m:
                g = m.groups()
                f = Face()
                f.p = [[float(g[0]), float(g[1]), float(g[2])],
                       [float(g[3]), float(g[4]), float(g[5])],
                       [float(g[6]), float(g[7]), float(g[8])]]
                f.bp = [[float(value) for value in g[9:12]],
                        [float(value) for value in g[12:15]]]
                f.tex = g[15]
                f.tail = g[16] or ''
                if brush is not None:
                    brush.append(f)
                continue
            m = FACE_RE.match(line)
            if m:
                g = m.groups()
                f = Face()
                f.p = [[float(g[0]), float(g[1]), float(g[2])],
                       [float(g[3]), float(g[4]), float(g[5])],
                       [float(g[6]), float(g[7]), float(g[8])]]
                f.tex = g[9]
                u = [float(g[10]), float(g[11]), float(g[12])]
                v = [float(g[14]), float(g[15]), float(g[16])]
                f.bp = texture_matrix(
                    f, [value / (float(g[19]) or 1.0) for value in u], float(g[13]),
                    [value / (float(g[20]) or 1.0) for value in v], float(g[17]),
                )
                f.tail = g[21] or ''
                if brush is not None:
                    brush.append(f)
                continue
            m = QUAKE_FACE_RE.match(line)
            if m:
                g = m.groups()
                f = Face()
                f.p = [[float(g[0]), float(g[1]), float(g[2])],
                       [float(g[3]), float(g[4]), float(g[5])],
                       [float(g[6]), float(g[7]), float(g[8])]]
                f.tex = g[9]
                u, v = quake_texture_vectors(face_plane(f)[0], float(g[12]),
                                             float(g[13]), float(g[14]))
                f.bp = texture_matrix(f, u, float(g[10]), v, float(g[11]))
                f.tail = g[15] or ''
                if brush is not None:
                    brush.append(f)
    return ents

def face_plane(f):
    p0, p1, p2 = [np.asarray(x, dtype=float) for x in f.p]
    n = np.cross(p1 - p0, p2 - p0)
    ln = np.linalg.norm(n)
    if ln < 1e-9:
        return None, None
    n = n / ln
    return -n, float(-(n @ p0))

def make_face(N, D, proto):
    N = np.asarray(N, dtype=float)
    N = N / (np.linalg.norm(N) or 1.0)
    o = N * D
    a = np.array([1.0, 0.0, 0.0]) if abs(N[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(N, a)
    u /= (np.linalg.norm(u) or 1.0)
    v = np.cross(N, u)
    f = Face()

    f.p = [[float(x) for x in o], [float(x) for x in (o + v * 64.0)],
           [float(x) for x in (o + u * 64.0)]]
    f.tex = proto.tex
    f.bp = [list(row) for row in proto.bp]
    f.tail = proto.tail
    return f

def brush_bounds(faces, lo0=-131072.0, hi0=131072.0):
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

def brush_halfspaces(faces):
    halfspaces = []
    for face in faces:
        normal, distance = face_plane(face)
        if normal is not None:
            halfspaces.append([normal[0], normal[1], normal[2], distance])
    return np.asarray(halfspaces, dtype=np.float64)

def subtract_convex(faces, cutter_faces):
    from negspace import vertices
    source = brush_halfspaces(faces)
    cutter = brush_halfspaces(cutter_faces)
    if len(source) < 4 or len(cutter) < 4:
        return [faces]
    intersection = np.vstack((source, cutter))
    if len(vertices(intersection)) < 4:
        return [faces]
    proto = faces[0]
    out, acc = [], []
    for row in cutter:
        N, D = row[:3], row[3]
        piece = list(faces) + [make_face([-N[0], -N[1], -N[2]], -D, proto)]
        piece += [make_face(An, Ad, proto) for An, Ad in acc]
        piece_halfspaces = brush_halfspaces(piece)
        if len(vertices(piece_halfspaces)) >= 4:
            out.append(piece)
        acc.append((list(N), D))
    return out

class Brush(object):

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
                continue
            n = n / ln
            if float(n @ (c - p0)) > 0:
                p1, p2 = p2, p1
                n = -n
            f = Face()
            f.p = [[float(x) for x in p0], [float(x) for x in p1], [float(x) for x in p2]]
            f.tex = tex
            f.bp = [[1.0 / (64.0 * scale), 0.0, 0.0],
                    [0.0, 1.0 / (64.0 * scale), 0.0]]
            f.tail = ' %d 0 0' % (134217728 if self.detail else 0)
            out.append(f)
        return out

def _v(a, b=None):
    return np.asarray(a, dtype=float) if b is None else np.asarray(b, dtype=float) - np.asarray(a, dtype=float)

def tri_prism(a, b, c, v, tex_cap, tex_side, detail=False):
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
    a2, b2, c2, d2 = [list(np.asarray(p, float) + np.asarray(v, float)) for p in (a, b, c, d)]
    br = Brush(detail)
    br.add_face([a, b, c, d], tex_cap)
    br.add_face([a2, b2, c2, d2], tex_cap)
    for (p, q, p2, q2) in ((a, b, a2, b2), (b, c, b2, c2), (c, d, c2, d2), (d, a, d2, a2)):
        br.add_face([p, q, q2, p2], tex_side)
    return br

def corridor_volume(a, b, width, height, overlap=0.0, tex='common/caulk'):
    A = np.asarray(a, dtype=float)
    B = np.asarray(b, dtype=float)
    d = B - A
    length = float(np.linalg.norm(d))
    if length < 1.0:
        return None
    direction = d / length
    up = np.array([0.0, 0.0, 1.0])
    side = np.cross(direction, up)
    if np.linalg.norm(side) < 1e-6:
        side = np.array([1.0, 0.0, 0.0])
    side /= np.linalg.norm(side)
    start = A - direction * overlap
    end = B + direction * overlap
    half_width = width / 2.0
    lower_left = start - side * half_width
    lower_right = start + side * half_width
    upper_right = lower_right + up * height
    upper_left = lower_left + up * height
    return quad_prism(lower_left, lower_right, upper_right, upper_left,
                      end - start, tex, tex)

def connector(a, b, width, height, thickness=32.0, overlap=2.0,
              tex_floor='common/caulk', tex_wall='common/caulk',
              tex_ceil='common/caulk'):
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

    A2 = A - dirh * OV
    B2 = B + dirh * OV

    def quad(off_lo, off_hi):
        return [list(A2 + off_lo), list(B2 + off_lo), list(B2 + off_hi), list(A2 + off_hi)]

    f_lo, f_hi = side * -(hw + T), side * (hw + T)
    out.append(quad_prism(*quad(f_lo, f_hi), v=list(up * -T),
                          tex_cap=tex_floor, tex_side=tex_floor))
    out.append(quad_prism(*[list(np.asarray(p) + up * height) for p in quad(f_lo, f_hi)],
                          v=list(up * T), tex_cap=tex_ceil, tex_side=tex_ceil))

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
    for e in ents:
        out.append('{')
        for k, v in e.keys:
            out.append('"%s" "%s"' % (k, v))
        for b in e.brushes:
            out.append('{')
            out.append('brushDef')
            out.append('{')
            for f in b:
                out.append(f.text())
            out.append('}')
            out.append('}')
        for p in e.patches:
            out.append('{')
            out.append(p.text())
            out.append('}')
        out.append('}')
    open(path, 'w', encoding='latin-1').write('\n'.join(out) + '\n')
    return path

@lru_cache(maxsize=None)
def asset_paths(root):
    names = set()
    for archive in sorted(glob.glob(os.path.join(root, 'data', '*.pk3'))):
        with zipfile.ZipFile(archive) as source:
            names.update(source.namelist())
    data = os.path.join(root, 'data')
    names.update(
        os.path.relpath(path, data).replace(os.sep, '/')
        for path in glob.glob(os.path.join(data, '**', '*'), recursive=True)
        if os.path.isfile(path)
    )
    return frozenset(names)

def ase_compiler_payload(payload):
    output = []
    normal_depth = 0
    brace_depth = 0
    geom_depth = None
    geom_name = None
    transform_depth = None
    normal_block_mass = 0
    transform_name_rewrite_mass = 0
    for line in payload.decode('latin-1').splitlines(keepends=True):
        opens = line.count('{')
        closes = line.count('}')
        if normal_depth:
            normal_depth += opens - closes
            brace_depth += opens - closes
            continue
        if re.match(r'\s*\*MESH_NORMALS\s*\{', line):
            normal_depth = opens - closes
            normal_block_mass += 1
            brace_depth += opens - closes
            continue
        if geom_depth is None and re.match(r'\s*\*GEOMOBJECT\s*\{', line):
            geom_depth = brace_depth + opens - closes
            geom_name = None
            transform_depth = None
        match = re.match(r'(\s*\*NODE_NAME\s+)"([^"]*)"(.*)', line)
        if geom_depth is not None and match:
            if geom_name is None:
                geom_name = match.group(2)
            elif transform_depth is not None and match.group(2) != geom_name:
                ending = '\n' if line.endswith('\n') else ''
                line = '%s"%s"%s%s' % (
                    match.group(1), geom_name, match.group(3).rstrip('\r\n'), ending,
                )
                transform_name_rewrite_mass += 1
        if geom_depth is not None and re.match(r'\s*\*NODE_TM\s*\{', line):
            transform_depth = brace_depth + opens - closes
        output.append(line)
        brace_depth += opens - closes
        if transform_depth is not None and brace_depth < transform_depth:
            transform_depth = None
        if geom_depth is not None and brace_depth < geom_depth:
            geom_depth = None
            geom_name = None
    return (''.join(output).encode('latin-1'), normal_block_mass,
            transform_name_rewrite_mass)

def obj_compiler_payload(payload, logical_name):
    lines = payload.decode('latin-1').splitlines(keepends=True)
    materials = sorted(set(
        line.strip().split(None, 1)[1]
        for line in lines
        if line.strip().lower().startswith('usemtl ') and len(line.strip().split(None, 1)) == 2
    ))
    material_name = posixpath.splitext(posixpath.basename(logical_name))[0] + '.mtl'
    body = [line for line in lines if not line.strip().lower().startswith('mtllib ')]
    model_name = re.sub(r'[^A-Za-z0-9_.-]', '_', posixpath.basename(logical_name))
    model = ('mtllib %s\no %s\n' % (material_name, model_name)
             + ''.join(body)).encode('latin-1')
    material = ''.join('newmtl %s\n' % name for name in materials).encode('latin-1')
    return model, material, materials

def realize_compiler_assets(workdir, mappath, basepath=XONDIR):
    concrete = set()
    declared = set()
    members = {}
    links = []
    asset_archives = sorted(glob.glob(os.path.join(basepath, 'data', '*.pk3')))
    for archive in asset_archives:
        with zipfile.ZipFile(archive) as source:
            for info in source.infolist():
                members[info.filename] = (archive, info.filename)
                if info.filename.startswith('scripts/') and info.filename.endswith('.shader'):
                    concrete.add(info.filename[len('scripts/'):-len('.shader')])
                if info.filename.endswith('shaderlist.txt'):
                    declared.update(source.read(info).decode('utf-8', 'replace').split())
                if (stat.S_ISLNK(info.external_attr >> 16)
                        and posixpath.splitext(info.filename)[1].lower() in COMPILER_ALIAS_EXTENSIONS):
                    links.append((info.filename, source.read(info).decode('utf-8', 'replace').strip()))
    data = os.path.join(basepath, 'data')
    scripts = os.path.join(data, 'scripts')
    concrete.update(
        os.path.relpath(path, scripts).replace(os.sep, '/')[:-len('.shader')]
        for path in glob.glob(os.path.join(scripts, '**', '*.shader'), recursive=True)
    )
    for path in glob.glob(os.path.join(data, '**', 'shaderlist.txt'), recursive=True):
        declared.update(open(path, encoding='utf-8', errors='replace').read().split())
    directory = os.path.join(workdir, 'fs', 'data')
    realized_links = []
    for name, destination in links:
        target = posixpath.normpath(posixpath.join(posixpath.dirname(name), destination))
        source = members.get(target)
        if source is None:
            loose = os.path.join(data, *target.split('/'))
            payload = open(loose, 'rb').read() if os.path.isfile(loose) else None
        else:
            with zipfile.ZipFile(source[0]) as archive:
                payload = archive.read(source[1])
        if payload is not None:
            path = os.path.join(directory, *name.split('/'))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as handle:
                handle.write(payload)
            realized_links.append(name)
    ase_names = sorted(set(re.findall(
        r'"model"\s+"([^\"]+\.ase)"',
        open(mappath, encoding='latin-1').read(), flags=re.IGNORECASE,
    )))
    normalized_models = []
    normalized_blocks = 0
    transform_rewrites = 0
    for name in ase_names:
        source = members.get(name)
        if source is None:
            loose = os.path.join(data, *name.split('/'))
            payload = open(loose, 'rb').read() if os.path.isfile(loose) else None
        else:
            with zipfile.ZipFile(source[0]) as archive:
                payload = archive.read(source[1])
        if payload is None:
            continue
        normalized, block_mass, rewrite_mass = ase_compiler_payload(payload)
        if not block_mass and not rewrite_mass:
            continue
        path = os.path.join(directory, *name.split('/'))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as handle:
            handle.write(normalized)
        normalized_blocks += block_mass
        transform_rewrites += rewrite_mass
        normalized_models.append({'logical_path': name, 'source_byte_mass': len(payload),
                                  'compiler_byte_mass': len(normalized),
                                  'normal_block_mass': block_mass,
                                  'transform_name_rewrite_mass': rewrite_mass})
    obj_names = sorted(set(re.findall(
        r'"model"\s+"([^\"]+\.obj)"',
        open(mappath, encoding='latin-1').read(), flags=re.IGNORECASE,
    )))
    normalized_obj_models = []
    obj_material_mass = 0
    for name in obj_names:
        source = members.get(name)
        if source is None:
            loose = os.path.join(data, *name.split('/'))
            payload = open(loose, 'rb').read() if os.path.isfile(loose) else None
        else:
            with zipfile.ZipFile(source[0]) as archive:
                payload = archive.read(source[1])
        if payload is None:
            continue
        model, material, materials = obj_compiler_payload(payload, name)
        path = os.path.join(directory, *name.split('/'))
        material_path = os.path.splitext(path)[0] + '.mtl'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as handle:
            handle.write(model)
        with open(material_path, 'wb') as handle:
            handle.write(material)
        obj_material_mass += len(materials)
        normalized_obj_models.append({
            'logical_path': name,
            'material_logical_path': posixpath.splitext(name)[0] + '.mtl',
            'source_byte_mass': len(payload),
            'compiler_byte_mass': len(model),
            'material_mass': len(materials),
        })
    empty_modules = sorted(declared - concrete)
    for name in empty_modules:
        path = os.path.join(directory, 'scripts', *name.split('/')) + '.shader'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, 'wb').close()
    names = concrete | declared
    directory = os.path.join(workdir, 'fs', 'data', 'scripts')
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, 'shaderlist.txt')
    with open(path, 'w') as handle:
        handle.write('\n'.join(sorted(names)) + '\n')
    return {
        'realized_shaderlist': path,
        'realized_shader_mass': len(names),
        'concrete_shader_mass': len(concrete),
        'declared_empty_shader_module_mass': len(empty_modules),
        'declared_empty_shader_modules': empty_modules,
        'realized_asset_alias_mass': len(realized_links),
        'realized_asset_aliases': realized_links,
        'normalized_ase_model_mass': len(normalized_models),
        'normalized_ase_normal_block_mass': normalized_blocks,
        'normalized_ase_transform_name_rewrite_mass': transform_rewrites,
        'normalized_ase_models': normalized_models,
        'normalized_obj_model_mass': len(normalized_obj_models),
        'normalized_obj_material_mass': obj_material_mass,
        'normalized_obj_models': normalized_obj_models,
    }

def compiler_image_sources(basepath=XONDIR):
    extensions = frozenset(('.jpg', '.jpeg', '.png', '.tga', '.pcx', '.bmp'))
    sources = {}
    for archive in sorted(glob.glob(os.path.join(basepath, 'data', '*.pk3'))):
        with zipfile.ZipFile(archive) as source:
            for name in source.namelist():
                if posixpath.splitext(name)[1].lower() in extensions:
                    sources.setdefault(name.lower(), []).append(('archive', archive, name))
    data = os.path.join(basepath, 'data')
    for path in sorted(glob.glob(os.path.join(data, '**', '*'), recursive=True)):
        if not os.path.isfile(path) or os.path.splitext(path)[1].lower() not in extensions:
            continue
        name = os.path.relpath(path, data).replace(os.sep, '/')
        sources.setdefault(name.lower(), []).append(('loose', path, name))
    return sources

def compiler_image_alias_source(logical_name, sources):
    target = posixpath.splitext(logical_name.lower())[0]
    target_base = posixpath.basename(target)
    target_parts = target.split('/')[:-1]
    candidates = []
    for name, locations in sources.items():
        stem = posixpath.splitext(name)[0]
        base = posixpath.basename(stem)
        exact_path = int(stem == target)
        exact_base = int(base == target_base)
        prefix_base = int(target_base.startswith(base + '_'))
        if not (exact_path or exact_base or prefix_base):
            continue
        source_parts = stem.split('/')[:-1]
        shared = len(set(target_parts) & set(source_parts))
        suffix = 0
        for left, right in zip(reversed(target_parts), reversed(source_parts)):
            if left != right:
                break
            suffix += 1
        rank = (-exact_path, -exact_base, -prefix_base, -suffix, -shared,
                abs(len(target_base) - len(base)), name)
        for location in locations:
            candidates.append((rank + (location[1],), name, location))
    return min(candidates, default=None)

def realize_missing_compiler_images(workdir, logical_names, sources, realized):
    rows = []
    directory = os.path.join(workdir, 'fs', 'data')
    for logical_name in sorted(set(logical_names)):
        logical_name = posixpath.splitext(logical_name)[0]
        if logical_name in realized:
            continue
        match = compiler_image_alias_source(logical_name, sources)
        if match is None:
            continue
        _, source_name, location = match
        if location[0] == 'archive':
            with zipfile.ZipFile(location[1]) as archive:
                payload = archive.read(location[2])
        else:
            with open(location[1], 'rb') as handle:
                payload = handle.read()
        extension = posixpath.splitext(source_name)[1]
        destination_name = logical_name + extension
        destination = os.path.join(directory, *destination_name.split('/'))
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        with open(destination, 'wb') as handle:
            handle.write(payload)
        row = {'logical_path': destination_name, 'source_path': source_name,
               'source_archive': os.path.basename(location[1]) if location[0] == 'archive' else None,
               'byte_mass': len(payload)}
        rows.append(row)
        realized[logical_name] = row
    return rows

def compile_map(mappath, workdir, vis=True, light=True, extra=(),
                q3map2=Q3MAP2, basepath=XONDIR, stages=None):
    stages = tuple(stages if stages is not None else
                   (('meta',) + (('vis',) if vis else ()) + (('light',) if light else ())))
    asset_measures = realize_compiler_assets(workdir, mappath, basepath) if 'meta' in stages else {}
    image_sources = compiler_image_sources(basepath) if 'meta' in stages else {}
    reactive_aliases = {}
    base = [q3map2, '-game', 'xonotic', '-fs_basepath', basepath,
            '-fs_homepath', os.path.join(workdir, 'fs')]
    logs = {}
    codes = {}
    capacity = None
    meta_attempts = []
    final_meta = ''
    if 'meta' in stages:
        while True:
            capacity_args = [] if capacity is None else ['-maxmapdrawsurfs', str(capacity)]
            r = subprocess.run(base + ['-meta'] + list(extra) + capacity_args + [mappath],
                               capture_output=True, text=True)
            final_meta = r.stdout + r.stderr
            exhausted = re.search(r'max_map_draw_surfs \((\d+)\) exceeded', final_meta)
            missing_images = re.findall(
                r"Couldn't find image for shader\s+['\"]?([^'\"\s]+)", final_meta,
            )
            new_aliases = realize_missing_compiler_images(
                workdir, missing_images, image_sources, reactive_aliases,
            )
            meta_attempts.append({
                'capacity': capacity,
                'returncode': r.returncode,
                'capacity_exhausted': exhausted is not None,
                'missing_image_mass': len(missing_images),
                'realized_missing_image_alias_mass': len(new_aliases),
            })
            if exhausted is not None:
                capacity = max(int(exhausted.group(1)) * 2,
                               (capacity or int(exhausted.group(1))) * 2)
            if exhausted is None and not new_aliases:
                break
        logs['meta'] = '\n'.join(
            'meta_attempt capacity=%s returncode=%d capacity_exhausted=%d' %
            (str(attempt['capacity']), attempt['returncode'], attempt['capacity_exhausted'])
            for attempt in meta_attempts
        ) + '\n' + final_meta
        codes['meta'] = r.returncode
    final_logs = {'meta': final_meta} if 'meta' in stages else {}
    if 'vis' in stages:
        r = subprocess.run(base + ['-vis', '-fast', mappath], capture_output=True, text=True)
        logs['vis'] = r.stdout + r.stderr
        final_logs['vis'] = logs['vis']
        codes['vis'] = r.returncode
    if 'light' in stages:
        r = subprocess.run(base + ['-light', '-fast', '-bounce', '1', mappath],
                           capture_output=True, text=True)
        logs['light'] = r.stdout + r.stderr
        final_logs['light'] = logs['light']
        codes['light'] = r.returncode
    leak_lines = sum(
        'leaked' in line.lower()
        for text in final_logs.values() for line in text.splitlines()
    )
    meta_lines = final_logs.get('meta', '').splitlines()
    vis_text = final_logs.get('vis', '')
    visibility_clusters = re.search(r'Total clusters:\s*(\d+)', vis_text)
    visibility_pairs = re.search(r'Total visible clusters:\s*(\d+)', vis_text)
    visibility_density = re.search(
        r'Average clusters visible:\s*[0-9.]+\s*\(([0-9.]+)%/total\)', vis_text,
    )
    unresolved_images = sorted(set(re.findall(
        r"Couldn't find image for shader\s+['\"]?([^'\"\s]+)", final_logs.get('meta', ''),
    )))
    warning_lines = [line.strip() for line in meta_lines if 'warning:' in line.lower()]
    error_lines = [line.strip() for line in meta_lines if 'error:' in line.lower()]
    asset_archives = sorted(glob.glob(os.path.join(basepath, 'data', '*.pk3')))
    measures = {
        'meta_attempts': meta_attempts,
        'visibility_cluster_mass': int(visibility_clusters.group(1)) if visibility_clusters else 0,
        'visibility_pair_mass': int(visibility_pairs.group(1)) if visibility_pairs else 0,
        'visibility_pair_density': (float(visibility_density.group(1)) / 100.0
                                    if visibility_density else 0.0),
        'asset_archive_mass': len(asset_archives),
        'asset_archives': [os.path.basename(path) for path in asset_archives],
        **asset_measures,
        'compiler_image_source_mass': sum(len(rows) for rows in image_sources.values()),
        'realized_missing_image_alias_mass': len(reactive_aliases),
        'realized_missing_image_aliases': [reactive_aliases[name]
                                           for name in sorted(reactive_aliases)],
        'shaderlist_missing_mass': sum('No shaderlist.txt found' in line for line in meta_lines),
        'missing_image_line_mass': sum("Couldn't find image" in line for line in meta_lines),
        'unresolved_missing_image_names': unresolved_images,
        'missing_file_line_mass': sum(
            'unable to open file' in line.lower() or 'script file' in line.lower() and 'was not found' in line.lower()
            for line in meta_lines
        ),
        'warning_line_mass': len(warning_lines),
        'warning_lines': sorted(set(warning_lines)),
        'error_line_mass': len(error_lines),
        'error_lines': sorted(set(error_lines)),
        'ase_error_line_mass': sum('error: ase:' in line.lower() for line in meta_lines),
        'node_without_volume_line_mass': sum('node without a volume' in line.lower()
                                             for line in meta_lines),
        'entity_in_solid_line_mass': sum('entity in solid' in line.lower()
                                         for line in meta_lines),
        'duplicate_triangle_line_mass': sum('duplicate or flipped triangle' in line.lower()
                                            for line in meta_lines),
    }
    return codes, leak_lines, logs, measures

MERGE_DROP_SHADERS = frozenset(('common/lightgrid',))
ENTITY_LINK_KEYS = frozenset(('target', 'target2', 'target3', 'target4',
                              'targetname', 'killtarget'))

def place_tile(mappath, off, workdir, name=None):
    name = name or os.path.splitext(os.path.basename(mappath))[0]
    ents = parse_map(mappath)
    brushes, patches, out, ndrop, translation_error_mass = [], [], [], 0, 0
    implicit_origin_mass = 0
    default_skin_mass = 0
    empty_decal_mass = 0
    available_assets = asset_paths(os.path.abspath(XONDIR))
    namespace = 'mesh_%s_' % re.sub(r'[^A-Za-z0-9_]', '_', name)
    for i, e in enumerate(ents):
        translation_error_mass += e.translate(off)
        implicit_origin_mass += e.implicit_origin_mass
        if i == 0:
            for b in e.brushes:
                if any(f.tex in MERGE_DROP_SHADERS for f in b):
                    ndrop += 1
                    continue
                brushes.append(b)
            patches += e.patches
        else:
            if e.get('classname') == '_decal' and not e.brushes and not e.patches:
                empty_decal_mass += 1
                continue
            model = e.get('model')
            if (e.get('classname') == 'misc_model' and model.lower().endswith('.md3')
                    and not any(key == '_skin' for key, _ in e.keys)
                    and model + '_0.skin' in available_assets):
                e.keys.append(('_skin', '0'))
                default_skin_mass += 1
            e.keys = [(key, namespace + value if key in ENTITY_LINK_KEYS and value else value)
                      for key, value in e.keys]
            out.append(e)
    realization = {
        'source_entity_mass': len(ents),
        'placed_entity_mass': len(out),
        'world_brush_mass': len(brushes),
        'world_patch_mass': len(patches),
        'lightgrid_brush_mass': ndrop,
        'implicit_point_origin_mass': implicit_origin_mass,
        'realized_default_model_skin_mass': default_skin_mass,
        'compiler_inert_empty_decal_mass': empty_decal_mass,
        'worldspawn_properties': [(key, value) for key, value in ents[0].keys
                                  if key != 'classname'] if ents else [],
    }
    return brushes, patches, out, ndrop, translation_error_mass, realization

def joins_record(tiles, joins, portals, bot_jumps=(), vantages_per_tile=None):
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
                          'chain': [[float(value) for value in point]
                                    for point in j.get('chain', (sa, sb))],
                          'length': round(float(math.dist(sa, sb)), 1),
                          'exclusive': bool(j['exclusive']),
                          'prominent': bool(j['prominent']),
                          'cart_navigable': bool(j['cart_navigable']),
                          'width': float(j.get('width', 0.0)),
                          'height': float(j.get('height', 0.0)),
                          'carve_depth': float(j.get('carve_depth', 0.0)),
                          'embed_depth': float(j.get('embed_depth', 0.0)),
                          'carve_clearance': float(j.get('carve_clearance', 0.0)),
                          'longitudinal_seal_overlap': float(j.get('longitudinal_seal_overlap', 0.0)),
                          'transverse_seal_overlap': float(j.get('transverse_seal_overlap', 0.0)),
                          'seal_overlap': float(j.get('seal_overlap', 0.0)),
                          'direction_residual_mass': int(j.get('direction_residual_mass', 0)),
                          'hall_cross_span': float(j.get('hall_cross_span', 0.0)),
                          'hall_entry_depth': float(j.get('hall_entry_depth', 0.0)),
                          'approach_floor_mass': int(j.get('approach_floor_mass', 0)),
                          'transfer_trigger_volume': float(j.get('transfer_trigger_volume', 0.0)),
                          'horizontal_span': float(j.get('horizontal_span', 0.0)),
                          'rise': float(j.get('rise', 0.0)),
                          'grade': float(j.get('grade', 0.0))})
    out_portals = []
    for p in portals:
        out_portals.append({'tile': int(p['tile']), 'name': p['name'],
                            'kind': p.get('kind', 'continue'),
                            'axis': int(p['axis']), 'sgn': float(p['sgn']),
                            'node': [float(x) for x in p['node']],
                            'mouth': [float(x) for x in p['mouth']],
                            'aperture': [[float(x) for x in p['aperture'][0]],
                                         [float(x) for x in p['aperture'][1]]],
                            'outbound_free_span': float(p.get('outbound_free_span', 0.0)),
                            'support_domain_atom_mass': int(p['support_domain_atom_mass']),
                            'support_residual_atom_mass': int(p['support_residual_atom_mass']),
                            'support_source_solid_candidate_mass': int(
                                p['support_source_solid_candidate_mass'])})
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
