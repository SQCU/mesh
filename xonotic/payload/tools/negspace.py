#!/usr/bin/env mesh-python
import math
import os
import re
import struct

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np

CONTENTS_SOLID = 0x1
CONTENTS_PLAYERCLIP = 0x10000
CONTENTS_MONSTERCLIP = 0x20000
CONTENTS_BOTCLIP = 0x400000
MASK_PLAYERSOLID = CONTENTS_SOLID | CONTENTS_PLAYERCLIP
MASK_BOTSOLID = MASK_PLAYERSOLID | CONTENTS_MONSTERCLIP | CONTENTS_BOTCLIP

NEGSPACE_SCHEMA = 3
SERVER_SOLID_BSP_CLASSES = frozenset({
    'func_bobbing', 'func_breakable', 'func_button', 'func_clientwall',
    'func_conveyor', 'func_door', 'func_door_rotating', 'func_door_secret',
    'func_fourier', 'func_pendulum', 'func_plat', 'func_rotating', 'func_static',
    'func_train', 'func_vectormamamam', 'func_wall',
})
DOOR_NONSOLID_SPAWNFLAG = 1 << 10
PATCH_COLLISION_TOLERANCE = 15.0
PATCH_COLLISION_SNAP = 1.0

PL_MIN = (-16.0, -16.0, -24.0)
PL_MAX = (16.0, 16.0, 45.0)

CART_MIN = (-32.0, -32.0, -24.0)
CART_MAX = (32.0, 32.0, 56.0)
CART_RIDER_MIN = CART_MIN
CART_RIDER_MAX = (32.0, 32.0, CART_MAX[2] + PL_MAX[2] - PL_MIN[2])

FLOOR_NZ = 0.7
EPS = 1e-4
COLLISION_EPSILON = 1.0 / 32.0

class _PlaneBlocks(object):
    def __init__(self, flat, offsets):
        self.flat = flat
        self.offsets = offsets

    def __len__(self):
        return len(self.offsets) - 1

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[value] for value in range(*index.indices(len(self)))]
        index = int(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return self.flat[self.offsets[index]:self.offsets[index + 1]]

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

def _lump(d, i):
    return struct.unpack_from('<ii', d, 8 + i * 8)

def _solid_model_indices(d, model):
    if model is not None:
        return (int(model),)
    offset, length = _lump(d, 0)
    text = d[offset:offset + length].split(b'\0')[0].decode('latin-1')
    indices = {0}
    for block in re.findall(r'\{[^{}]*\}', text):
        values = dict(re.findall(r'"([^"]+)"\s+"([^"]*)"', block))
        classname = values.get('classname', '')
        if classname not in SERVER_SOLID_BSP_CLASSES:
            continue
        try:
            spawnflags = int(float(values.get('spawnflags', '0')))
        except ValueError:
            spawnflags = 0
        if classname == 'func_door' and spawnflags & DOOR_NONSOLID_SPAWNFLAG:
            continue
        value = values.get('model', '')
        if value.startswith('*'):
            try:
                indices.add(int(value[1:]))
            except ValueError:
                pass
    return tuple(sorted(indices))

def bounds_of(H, seed_lo, seed_hi, iters=4):
    n = H[:, :3]
    d = H[:, 3]
    lo = np.array(seed_lo, dtype=np.float64)
    hi = np.array(seed_hi, dtype=np.float64)
    for _ in range(iters):
        t1 = n * lo
        t2 = n * hi
        mn = np.minimum(t1, t2)
        other = mn.sum(axis=1, keepdims=True) - mn
        with np.errstate(divide='ignore', invalid='ignore'):
            b = (d[:, None] - other) / n
        up = np.where(n > EPS, b, np.inf)
        dn = np.where(n < -EPS, b, -np.inf)
        nhi = np.minimum(hi, np.nanmin(up, axis=0))
        nlo = np.maximum(lo, np.nanmax(dn, axis=0))
        if np.all(np.abs(nhi - hi) < 0.01) and np.all(np.abs(nlo - lo) < 0.01):
            hi, lo = nhi, nlo
            break
        hi, lo = nhi, nlo
    return lo, hi

PRUNE_ON = True

def prune(H, lo, hi):
    if not PRUNE_ON:
        return H
    n = H[:, :3]
    d = H[:, 3]
    c = 0.5 * (lo + hi)
    e = 0.5 * (hi - lo)
    sup = n @ c + np.abs(n) @ e
    keep = sup > d - 0.05
    nk = int(keep.sum())
    if nk >= len(H) - 6:
        return H
    bx = np.array([[1.0, 0.0, 0.0, float(hi[0])], [-1.0, 0.0, 0.0, float(-lo[0])],
                   [0.0, 1.0, 0.0, float(hi[1])], [0.0, -1.0, 0.0, float(-lo[1])],
                   [0.0, 0.0, 1.0, float(hi[2])], [0.0, 0.0, -1.0, float(-lo[2])]])
    return np.vstack([H[keep], bx])

def _curve_tessellation(control, axis):
    if axis == 1:
        middle = control[:, 1:-1:2]
        left = control[:, 0:-2:2] - middle
        right = control[:, 2::2] - middle
    else:
        middle = control[1:-1:2]
        left = control[0:-2:2] - middle
        right = control[2::2] - middle
    left2 = np.einsum('...i,...i->...', left, left)
    right2 = np.einsum('...i,...i->...', right, right)
    product = np.einsum('...i,...i->...', left, right)
    largest = float(np.max(np.maximum(0.0, left2 * right2 - product * product), initial=0.0))
    factor = (largest / 64.0) ** 0.25 / PATCH_COLLISION_TOLERANCE
    if factor < 0.0001:
        return 0
    if factor < 2.0:
        return 1
    return min(1024, int(math.floor(math.log(factor) / math.log(2.0))) + 1)

def _patch_collision_grid(control):
    height, width = control.shape[:2]
    xtess = _curve_tessellation(control, 1)
    ytess = _curve_tessellation(control, 0)
    xmax = max(1, 2 * xtess)
    ymax = max(1, 2 * ytess)
    out_width = (width - 1) * xtess + 1 if xtess else (width - 1) // 2 + 1
    out_height = (height - 1) * ytess + 1 if ytess else (height - 1) // 2 + 1
    out = np.zeros((out_height, out_width, 3), dtype=np.float64)
    py = np.arange(ymax + 1, dtype=np.float64) / ymax
    px = np.arange(xmax + 1, dtype=np.float64) / xmax
    wy = np.column_stack(((1.0 - py) ** 2, 2.0 * (1.0 - py) * py, py ** 2))
    wx = np.column_stack(((1.0 - px) ** 2, 2.0 * (1.0 - px) * px, px ** 2))
    for row in range(0, height - 1, 2):
        for column in range(0, width - 1, 2):
            patch = control[row:row + 3, column:column + 3]
            middle = np.tensordot(wy, patch, axes=(1, 0))
            block = np.einsum('xb,ybk->yxk', wx, middle)
            y0 = row * ymax // 2
            x0 = column * xmax // 2
            out[y0:y0 + ymax + 1, x0:x0 + xmax + 1] = block
    return np.floor(out / PATCH_COLLISION_SNAP) * PATCH_COLLISION_SNAP

def _patch_triangles(grid):
    height, width = grid.shape[:2]
    for y in range(height - 1):
        if y % 2:
            for x in range(width - 2, -1, -1):
                yield grid[y + 1, x], grid[y + 1, x + 1], grid[y, x + 1]
                yield grid[y, x], grid[y + 1, x], grid[y, x + 1]
        else:
            for x in range(width - 1):
                yield grid[y, x], grid[y + 1, x], grid[y, x + 1]
                yield grid[y + 1, x], grid[y + 1, x + 1], grid[y, x + 1]

def _triangle_prism(a, b, c):
    normal = np.cross(b - a, c - a)
    length = float(np.linalg.norm(normal))
    if length * length < 0.001:
        return None
    normal /= length
    planes = [[*normal, float(np.dot(normal, a) + PATCH_COLLISION_SNAP)],
              [*(-normal), float(np.dot(-normal, a) + PATCH_COLLISION_SNAP)]]
    for left, right, other in ((a, b, c), (b, c, a), (c, a, b)):
        side = np.cross(right - left, normal)
        side_length = float(np.linalg.norm(side))
        if side_length <= EPS:
            return None
        side /= side_length
        distance = float(np.dot(side, left))
        if np.dot(side, other) > distance:
            side = -side
            distance = -distance
        planes.append([*side, distance])
    return np.asarray(planes, dtype=np.float64)

def _compiled_collision_solids(d, mask, model, world_lo, world_hi):
    to, tl = _lump(d, 1)
    po, pl = _lump(d, 2)
    mo, ml = _lump(d, 7)
    bo, bl = _lump(d, 8)
    so, sl = _lump(d, 9)
    vo, vl = _lump(d, 10)
    fo, fl = _lump(d, 13)
    contents = np.frombuffer(d, '<i4', (tl // 72) * 18, to).reshape(-1, 18)[:, 17]
    planes = np.frombuffer(d, '<f4', (pl // 16) * 4, po).reshape(-1, 4).astype(np.float64)
    models = np.frombuffer(d, '<i4', (ml // 40) * 10, mo).reshape(-1, 10)
    brushes = np.frombuffer(d, '<i4', (bl // 12) * 3, bo).reshape(-1, 3)
    sides = np.frombuffer(d, '<i4', (sl // 8) * 2, so).reshape(-1, 2)
    vertices = np.frombuffer(d, dtype=np.dtype([('p', '<f4', 3), ('tail', 'V32')]),
                             count=vl // 44, offset=vo)['p'].astype(np.float64)
    faces = np.frombuffer(d, dtype=np.dtype([('head', '<i4', 12), ('tail', 'V56')]),
                          count=fl // 104, offset=fo)
    scopes = [index for index in _solid_model_indices(d, model)
              if 0 <= index < len(models)]
    brush_indices = []
    face_indices = []
    for index in scopes:
        first_face, face_count = int(models[index, 6]), int(models[index, 7])
        first_brush, brush_count = int(models[index, 8]), int(models[index, 9])
        face_indices.extend(range(first_face, first_face + face_count))
        brush_indices.extend(range(first_brush, first_brush + brush_count))
    blocks = []
    bounds = []
    brush_mass = 0
    patch_triangle_mass = 0
    for index in brush_indices:
        first, count, texture = brushes[index]
        if (count <= 0 or first < 0 or first + count > len(sides)
                or texture < 0 or texture >= len(contents)
                or not contents[texture] & mask):
            continue
        block = planes[sides[first:first + count, 0]]
        lo, hi = bounds_of(block, world_lo - 512.0, world_hi + 512.0)
        if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)) and np.all(lo <= hi)):
            continue
        blocks.append(block)
        bounds.append((lo, hi))
        brush_mass += 1
    for index in face_indices:
        head = faces[index]['head']
        texture, face_type = int(head[0]), int(head[2])
        first, count = int(head[3]), int(head[4])
        if (face_type != 2 or texture < 0 or texture >= len(contents)
                or not contents[texture] & mask or first < 0 or count <= 0
                or first + count > len(vertices)):
            continue
        width, height = struct.unpack_from('<2i', faces[index]['tail'], 48)
        if width < 3 or height < 3 or width * height != count:
            continue
        control = vertices[first:first + count].reshape(height, width, 3)
        for triangle in _patch_triangles(_patch_collision_grid(control)):
            block = _triangle_prism(*triangle)
            if block is None:
                continue
            points = np.asarray(triangle)
            blocks.append(block)
            bounds.append((points.min(axis=0) - PATCH_COLLISION_SNAP,
                           points.max(axis=0) + PATCH_COLLISION_SNAP))
            patch_triangle_mass += 1
    return blocks, bounds, brush_mass, patch_triangle_mass

def nonempty(H, lo=None, hi=None):
    v = vertices(H, lo, hi)
    return len(v) > 0

def vertices(H, lo=None, hi=None):
    H = np.asarray(H, dtype=np.float64)
    k = len(H)
    if k < 4:
        return np.zeros((0, 3))
    chunks = []
    for a in range(k - 2):
        b, c = np.triu_indices(k - a - 1, 1)
        b += a + 1
        c += a + 1
        A = H[np.column_stack((np.full(len(b), a, dtype=np.int64), b, c))]
        n0, n1, n2 = A[:, 0, :3], A[:, 1, :3], A[:, 2, :3]
        x12 = np.cross(n1, n2)
        x20 = np.cross(n2, n0)
        x01 = np.cross(n0, n1)
        det = np.einsum('ij,ij->i', n0, x12)
        live = np.abs(det) > 1e-7
        if not live.any():
            continue
        A = A[live]
        det = det[live]
        P = (A[:, 0, 3, None] * x12[live]
             + A[:, 1, 3, None] * x20[live]
             + A[:, 2, 3, None] * x01[live]) / det[:, None]
        P = P[np.isfinite(P).all(axis=1)]
        if not len(P):
            continue
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            viol = P @ H[:, :3].T - H[:, 3]
        P = P[(viol <= 0.05).all(axis=1)]
        if len(P):
            chunks.append(P)
    if not chunks:
        return np.zeros((0, 3))
    P = np.vstack(chunks)
    if len(P) > 1:
        P = np.unique(np.round(P, 3), axis=0)
    return P

def shrink_H(H, mins, maxs):
    n = H[:, :3]
    mn = np.asarray(mins, dtype=np.float64)
    mx = np.asarray(maxs, dtype=np.float64)
    sup = (np.where(n > 0, mx, mn) * n).sum(axis=1)
    out = H.copy()
    out[:, 3] = H[:, 3] - sup
    return out

def subtract(pieces, B, seed_lo, seed_hi, exact_empty=True, minext=0.05):
    out = []
    for C in pieces:
        clo, chi = bounds_of(C, seed_lo, seed_hi)
        cc = 0.5 * (clo + chi)
        ce = 0.5 * (chi - clo)
        lowv = B[:, :3] @ cc - np.abs(B[:, :3]) @ ce - B[:, 3]
        if np.any(lowv > 0.0):
            out.append(C)
            continue
        highv = B[:, :3] @ cc + np.abs(B[:, :3]) @ ce - B[:, 3]
        acc = []
        for q in range(len(B)):
            if highv[q] <= 0.0:
                acc.append(B[q])
                continue
            R = np.vstack([C, -B[q:q + 1]])
            if acc:
                R = np.vstack([R, np.array(acc, dtype=np.float64)])
            rlo, rhi = bounds_of(R, clo, chi)
            if np.all(rhi - rlo > minext):
                Rp = prune(R, rlo, rhi)

                if (not exact_empty) or len(vertices(Rp)) >= 4:
                    out.append(Rp)
            acc.append(B[q])
    return out

def box_H(lo, hi):
    rows = []
    for a in range(3):
        e = [0.0, 0.0, 0.0]
        e[a] = 1.0
        rows.append(e + [float(hi[a])])
        e2 = [0.0, 0.0, 0.0]
        e2[a] = -1.0
        rows.append(e2 + [float(-lo[a])])
    return np.array(rows, dtype=np.float64)

class Portal(object):
    __slots__ = ('a', 'b', 'n', 'd', 'poly', 'area', 'radius', 'centre')

    def __init__(self, a, b, n, d, poly, area, radius, centre):
        self.a, self.b = a, b
        self.n, self.d = n, d
        self.poly = poly
        self.area = area
        self.radius = radius
        self.centre = centre

def _basis(n):
    n = np.asarray(n, dtype=np.float64)
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, a)
    u /= (np.linalg.norm(u) or 1.0)
    v = np.cross(n, u)
    return u, v

def _clip2(poly, a, b, c):
    out = []
    m = len(poly)
    for i in range(m):
        p, q = poly[i], poly[(i + 1) % m]
        dp = a * p[0] + b * p[1] - c
        dq = a * q[0] + b * q[1] - c
        if dp <= 1e-9:
            out.append(p)
        if (dp > 1e-9) != (dq > 1e-9):
            t = dp / (dp - dq)
            out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    return out

def _area2(poly):
    s = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        s += x1 * y2 - x2 * y1
    return abs(s) * 0.5

def _inradius(poly, edges):
    if len(poly) < 3:
        return 0.0
    ext = max(max(abs(p[0]) for p in poly), max(abs(p[1]) for p in poly)) or 1.0

    def fits(r):
        pg = [(-2 * ext, -2 * ext), (2 * ext, -2 * ext), (2 * ext, 2 * ext), (-2 * ext, 2 * ext)]
        for (a, b, c) in edges:
            pg = _clip2(pg, a, b, c - r)
            if len(pg) < 3:
                return False
        return True

    if not fits(0.5):
        return 0.0
    lo, hi = 0.5, ext
    for _ in range(10):
        mid = 0.5 * (lo + hi)
        if fits(mid):
            lo = mid
        else:
            hi = mid
    return lo

def _interval_pair_count(left_lo, left_hi, right_lo, right_hi):
    events = []
    events.extend((float(value), 0, 0) for value in left_lo)
    events.extend((float(value) + 0.25, 1, 0) for value in left_hi)
    events.extend((float(value), 0, 1) for value in right_lo)
    events.extend((float(value) + 0.25, 1, 1) for value in right_hi)
    active = [0, 0]
    count = 0
    for _, end, side in sorted(events):
        if end:
            active[side] -= 1
        else:
            count += active[1 - side]
            active[side] += 1
    return count

def _indexed_face_pairs(lo, hi, left, right):
    left_lo = lo[[face[0] for face in left]]
    left_hi = hi[[face[0] for face in left]]
    right_lo = lo[[face[0] for face in right]]
    right_hi = hi[[face[0] for face in right]]
    axis = min(range(3), key=lambda value: _interval_pair_count(
        left_lo[:, value], left_hi[:, value], right_lo[:, value], right_hi[:, value],
    ))
    events = []
    events.extend((float(left_lo[index, axis]), 0, 0, index) for index in range(len(left)))
    events.extend((float(left_hi[index, axis]) + 0.25, 1, 0, index) for index in range(len(left)))
    events.extend((float(right_lo[index, axis]), 0, 1, index) for index in range(len(right)))
    events.extend((float(right_hi[index, axis]) + 0.25, 1, 1, index) for index in range(len(right)))
    active = [set(), set()]
    for _, end, side, index in sorted(events):
        if end:
            active[side].discard(index)
            continue
        other = 1 - side
        candidates = np.fromiter(active[other], dtype=np.int64, count=len(active[other]))
        if len(candidates):
            source_lo = left_lo[index] if side == 0 else right_lo[index]
            source_hi = left_hi[index] if side == 0 else right_hi[index]
            candidate_lo = right_lo[candidates] if side == 0 else left_lo[candidates]
            candidate_hi = right_hi[candidates] if side == 0 else left_hi[candidates]
            mask = np.all((source_lo <= candidate_hi + 0.25)
                          & (candidate_lo <= source_hi + 0.25), axis=1)
            for candidate in candidates[mask]:
                yield (left[index], right[int(candidate)]) if side == 0 else (left[int(candidate)], right[index])
        active[side].add(index)

class NegSpace(object):

    def __init__(self, src, mask=MASK_PLAYERSOLID, model=None, cell=512.0,
                 verbose=False):
        if isinstance(src, (bytes, bytearray)):
            d = bytes(src)
        else:
            d = open(src, 'rb').read()
        self.mask = mask
        self.gridcell = float(cell)

        po, pl = _lump(d, 2)
        no, nl = _lump(d, 3)
        lo_, ll = _lump(d, 4)
        mo, ml = _lump(d, 7)
        planes = np.frombuffer(d, '<f4', (pl // 16) * 4, po).reshape(-1, 4).astype(np.float64)
        nodes = np.frombuffer(d, '<i4', (nl // 36) * 9, no).reshape(-1, 9)
        leafs = np.frombuffer(d, '<i4', (ll // 48) * 12, lo_).reshape(-1, 12)
        modf = np.frombuffer(d, '<f4', (ml // 40) * 10, mo).reshape(-1, 10)

        world_model = 0 if model is None else int(model)
        self.world_lo = modf[world_model, 0:3].astype(np.float64) - 64.0
        self.world_hi = modf[world_model, 3:6].astype(np.float64) + 64.0
        head = 0 if world_model == 0 else None
        self.planes = planes

        self.blk_H, block_bounds, self.compiled_brush_mass, self.patch_triangle_mass = (
            _compiled_collision_solids(d, mask, model, self.world_lo, self.world_hi)
        )
        self.blk_lo = np.asarray([value[0] for value in block_bounds])
        self.blk_hi = np.asarray([value[1] for value in block_bounds])
        self._index_blocks(True)

        wb = []
        for a in range(3):
            e = [0.0, 0.0, 0.0]
            e[a] = 1.0
            wb.append(e + [float(self.world_hi[a])])
            e2 = [0.0, 0.0, 0.0]
            e2[a] = -1.0
            wb.append(e2 + [float(-self.world_lo[a])])
        wb = np.array(wb, dtype=np.float64)

        self.cells = []
        self.cell_leaf = []
        self.cell_node_faces = []
        stack = [(head, [])]
        nleaf_open = 0
        nleaf_solid = 0
        nsplit = 0
        while stack:
            ni, path = stack.pop()
            if ni < 0:
                li = -1 - ni
                if li >= len(leafs):
                    continue

                if leafs[li][0] < 0:
                    nleaf_solid += 1
                else:
                    nleaf_open += 1
                H = np.array([p for p, _ in path], dtype=np.float64) if path else np.zeros((0, 4))
                H = np.vstack([H, wb]) if len(H) else wb.copy()
                lo, hi = bounds_of(H, self.world_lo, self.world_hi)
                if np.any(hi - lo < 1.0):
                    continue
                H = prune(H, lo, hi)

                blk = self._blocking_in(lo, hi)
                pieces = [H]
                for bi in blk:
                    pieces = subtract(pieces, self.blk_H[bi], lo, hi,
                                      exact_empty=False, minext=0.25)
                    nsplit += 1
                for C in pieces:
                    clo, chi = bounds_of(C, lo, hi)
                    if np.any(chi - clo < 1.0):
                        continue
                    self.cells.append(prune(C, clo, chi))
                    self.cell_leaf.append(li)
                continue
            if ni >= len(nodes):
                continue
            nd = nodes[ni]
            pi = int(nd[0])
            if not (0 <= pi < len(planes)):
                continue
            pn = planes[pi]

            stack.append((int(nd[1]), path + [(np.array([-pn[0], -pn[1], -pn[2], -pn[3]]), (pi, -1))]))
            stack.append((int(nd[2]), path + [(np.array([pn[0], pn[1], pn[2], pn[3]]), (pi, 1))]))

        self.n_open_leaves = nleaf_open
        self.n_solid_leaves = nleaf_solid
        self.n_detail_splits = nsplit
        self._finish(verbose)
        self.schema = NEGSPACE_SCHEMA

    def _index_blocks(self, build_grid=False):
        bgrid = {}
        BC = self.gridcell
        if build_grid:
            g0 = np.floor(self.blk_lo / BC).astype(np.int64)
            g1 = np.floor(self.blk_hi / BC).astype(np.int64)
            extents = g1 - g0 + 1
            volume = np.prod(extents, axis=1)
            incidence_capacity = max(1, len(self.blk_H))
            indexed = np.flatnonzero(volume <= incidence_capacity)
            if len(indexed):
                counts = volume[indexed]
                offsets = np.zeros(len(indexed) + 1, dtype=np.int64)
                np.cumsum(counts, out=offsets[1:])
                owners = np.repeat(indexed, counts)
                local = np.arange(offsets[-1], dtype=np.int64) - np.repeat(offsets[:-1], counts)
                owner_extents = extents[owners]
                keys = g0[owners].copy()
                keys[:, 2] += local % owner_extents[:, 2]
                local //= owner_extents[:, 2]
                keys[:, 1] += local % owner_extents[:, 1]
                keys[:, 0] += local // owner_extents[:, 1]
                order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
                keys = keys[order]
                values = owners[order]
                starts = np.r_[0, np.flatnonzero(np.any(keys[1:] != keys[:-1], axis=1)) + 1]
                ends = np.r_[starts[1:], len(keys)]
                for start, end in zip(starts, ends):
                    bgrid[tuple(int(value) for value in keys[start])] = values[start:end].tolist()
            bgrid['big'] = np.flatnonzero(volume > incidence_capacity).tolist()
        self.blk_grid = bgrid
        self.blk_cell = BC
        self._index_solid_bvh('blk', self.blk_lo, self.blk_hi)
        self._index_solid_planes(self.blk_H)

    @staticmethod
    def _morton_dilate(values):
        values = np.asarray(values, dtype=np.uint64) & np.uint64(0x1fffff)
        values = (values | values << np.uint64(32)) & np.uint64(0x1f00000000ffff)
        values = (values | values << np.uint64(16)) & np.uint64(0x1f0000ff0000ff)
        values = (values | values << np.uint64(8)) & np.uint64(0x100f00f00f00f00f)
        values = (values | values << np.uint64(4)) & np.uint64(0x10c30c30c30c30c3)
        return (values | values << np.uint64(2)) & np.uint64(0x1249249249249249)

    def _index_solid_bvh(self, prefix, lo, hi):
        lo = np.asarray(lo, dtype=np.float64).reshape((-1, 3))
        hi = np.asarray(hi, dtype=np.float64).reshape((-1, 3))
        if not len(lo):
            setattr(self, prefix + '_bvh_order', np.zeros(0, dtype=np.int64))
            setattr(self, prefix + '_bvh_lo', ())
            setattr(self, prefix + '_bvh_hi', ())
            return
        centroids = 0.5 * (lo + hi)
        lower = centroids.min(axis=0)
        extent = np.maximum(centroids.max(axis=0) - lower, np.finfo(np.float64).eps)
        scale = np.float64((1 << 21) - 1)
        quantized = np.minimum(
            np.floor((centroids - lower) * (scale / extent)), scale,
        ).astype(np.uint64)
        codes = (self._morton_dilate(quantized[:, 0])
                 | self._morton_dilate(quantized[:, 1]) << np.uint64(1)
                 | self._morton_dilate(quantized[:, 2]) << np.uint64(2))
        order = np.argsort(codes, kind='stable')
        levels_lo = [lo[order]]
        levels_hi = [hi[order]]
        while len(levels_lo[-1]) > 1:
            child_lo = levels_lo[-1]
            child_hi = levels_hi[-1]
            parent_lo = child_lo[::2].copy()
            parent_hi = child_hi[::2].copy()
            pairs = len(child_lo) // 2
            parent_lo[:pairs] = np.minimum(parent_lo[:pairs], child_lo[1::2])
            parent_hi[:pairs] = np.maximum(parent_hi[:pairs], child_hi[1::2])
            levels_lo.append(parent_lo)
            levels_hi.append(parent_hi)
        setattr(self, prefix + '_bvh_order', order)
        setattr(self, prefix + '_bvh_lo', tuple(levels_lo))
        setattr(self, prefix + '_bvh_hi', tuple(levels_hi))

    def _solid_bvh_relation_rows(self, prefix, query_mass, intersects):
        levels_lo = getattr(self, prefix + '_bvh_lo')
        levels_hi = getattr(self, prefix + '_bvh_hi')
        if not levels_lo or not query_mass:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        owners = np.arange(query_mass, dtype=np.int64)
        nodes = np.zeros(query_mass, dtype=np.int64)
        present = intersects(owners, levels_lo[-1][nodes], levels_hi[-1][nodes])
        owners = owners[present]
        nodes = nodes[present]
        for level in range(len(levels_lo) - 2, -1, -1):
            left = nodes * 2
            right = left + 1
            child_owners = np.repeat(owners, 2)
            child_nodes = np.column_stack((left, right)).reshape(-1)
            valid = child_nodes < len(levels_lo[level])
            child_owners = child_owners[valid]
            child_nodes = child_nodes[valid]
            present = intersects(
                child_owners,
                levels_lo[level][child_nodes],
                levels_hi[level][child_nodes],
            )
            owners = child_owners[present]
            nodes = child_nodes[present]
            if not len(owners):
                break
        order = getattr(self, prefix + '_bvh_order')
        return owners, order[nodes]

    def _solid_relation_rows(self, lo, hi):
        lo = np.asarray(lo, dtype=np.float64).reshape((-1, 3))
        hi = np.asarray(hi, dtype=np.float64).reshape((-1, 3))
        prefix = 's' if getattr(self, 'solids', None) is not None else 'blk'
        return self._solid_bvh_relation_rows(
            prefix, len(lo),
            lambda owners, box_lo, box_hi: np.all(
                (hi[owners] > box_lo) & (lo[owners] < box_hi), axis=1,
            ),
        )

    def _solid_ray_relation_rows(self, starts, ends, mins, maxs, padding):
        starts = np.asarray(starts, dtype=np.float64).reshape((-1, 3))
        ends = np.asarray(ends, dtype=np.float64).reshape((-1, 3))
        prefix = 's' if getattr(self, 'solids', None) is not None else 'blk'
        minimum = np.asarray(mins, dtype=np.float64)
        maximum = np.asarray(maxs, dtype=np.float64)
        return self._solid_bvh_relation_rows(
            prefix, len(starts),
            lambda owners, box_lo, box_hi: self._ray_box_rows(
                starts[owners], ends[owners],
                box_lo - maximum - padding,
                box_hi - minimum + padding,
            ),
        )

    @staticmethod
    def _ray_box_rows(starts, ends, lo, hi):
        delta = ends - starts
        stationary = delta == 0.0
        contained = (~stationary
                     | ((starts >= lo) & (starts <= hi))).all(axis=1)
        inverse = np.zeros_like(delta)
        np.divide(1.0, delta, out=inverse, where=~stationary)
        first = (lo - starts) * inverse
        last = (hi - starts) * inverse
        lower = np.where(stationary, -np.inf, np.minimum(first, last)).max(axis=1)
        upper = np.where(stationary, np.inf, np.maximum(first, last)).min(axis=1)
        return contained & (np.maximum(lower, 0.0) <= np.minimum(upper, 1.0))

    def _index_solid_planes(self, solids):
        if isinstance(solids, _PlaneBlocks):
            planes = solids.flat
            self.solid_plane_counts = np.diff(solids.offsets)
        else:
            values = [value[0] if isinstance(value, tuple) else value for value in solids]
            self.solid_plane_counts = np.asarray([len(value) for value in values], dtype=np.int64)
            planes = np.vstack(values) if values else np.zeros((0, 4), dtype=np.float64)
        self.solid_plane_offsets = np.zeros(len(self.solid_plane_counts) + 1, dtype=np.int64)
        np.cumsum(self.solid_plane_counts, out=self.solid_plane_offsets[1:])
        lengths = np.linalg.norm(planes[:, :3], axis=1)
        lengths = np.where(lengths > 0.0, lengths, 1.0)
        planes[:, :3] /= lengths[:, None]
        planes[:, 3] /= lengths
        self.solid_plane_n0 = np.ascontiguousarray(planes[:, 0])
        self.solid_plane_n1 = np.ascontiguousarray(planes[:, 1])
        self.solid_plane_n2 = np.ascontiguousarray(planes[:, 2])
        self.solid_plane_distances = np.ascontiguousarray(planes[:, 3])
        if solids is getattr(self, 'blk_H', None) and not isinstance(solids, _PlaneBlocks):
            self.blk_H = _PlaneBlocks(planes, self.solid_plane_offsets)

    def _blocking_in(self, lo, hi):
        BC = self.blk_cell
        out = []
        seen = set()
        for x in range(int(math.floor(lo[0] / BC)), int(math.floor(hi[0] / BC)) + 1):
            for y in range(int(math.floor(lo[1] / BC)), int(math.floor(hi[1] / BC)) + 1):
                for z in range(int(math.floor(lo[2] / BC)), int(math.floor(hi[2] / BC)) + 1):
                    for i in self.blk_grid.get((x, y, z), ()):
                        if i not in seen:
                            seen.add(i)
                            out.append(i)
        for i in self.blk_grid.get('big', ()):
            if i not in seen:
                seen.add(i)
                out.append(i)
        if not out:
            return []
        idx = np.array(out, dtype=np.int64)
        m = ((self.blk_lo[idx, 0] < hi[0]) & (self.blk_hi[idx, 0] > lo[0]) &
             (self.blk_lo[idx, 1] < hi[1]) & (self.blk_hi[idx, 1] > lo[1]) &
             (self.blk_lo[idx, 2] < hi[2]) & (self.blk_hi[idx, 2] > lo[2]))
        return [int(v) for v in idx[m]]

    def _finish(self, verbose=False):
        N = len(self.cells)
        self.counts = np.asarray([len(cell) for cell in self.cells], dtype=np.int32)
        self.lo = np.zeros((N, 3))
        self.hi = np.zeros((N, 3))
        for i, H in enumerate(self.cells):
            l, h = bounds_of(H, self.world_lo, self.world_hi)
            self.lo[i] = l
            self.hi[i] = h
        self._index()
        self.portals = None
        self.adj = None
        if verbose:
            v = self.volume()
            print('negspace: %d convex free cells from %d open leaves (%d solid leaves, '
                  '%d detail-brush subtractions), free-cell AABB volume %.3g u^3, '
                  '%d grid cells'
                  % (N, self.n_open_leaves, self.n_solid_leaves,
                     self.n_detail_splits, v, len(self.grid)))

    def _index(self):
        c = self.gridcell
        g = {}
        g0 = np.floor(self.lo / c).astype(np.int64)
        g1 = np.floor(self.hi / c).astype(np.int64)
        for i in range(len(self.cells)):
            for x in range(g0[i, 0], g1[i, 0] + 1):
                for y in range(g0[i, 1], g1[i, 1] + 1):
                    for z in range(g0[i, 2], g1[i, 2] + 1):
                        g.setdefault((x, y, z), []).append(i)
        self.grid = {k: np.array(v, dtype=np.int64) for k, v in g.items()}

    def volume(self):
        e = np.maximum(self.hi - self.lo, 0.0)
        return float((e[:, 0] * e[:, 1] * e[:, 2]).sum())

    def translated(self, t):
        out = NegSpace.__new__(NegSpace)
        out.mask = self.mask
        out.gridcell = self.gridcell
        t = np.asarray(t, dtype=np.float64)
        out.cells = []
        for H in self.cells:
            G = H.copy()
            G[:, 3] = H[:, 3] + H[:, :3] @ t
            out.cells.append(G)
        out.cell_leaf = list(self.cell_leaf)
        out.world_lo = self.world_lo + t
        out.world_hi = self.world_hi + t
        out.n_open_leaves = self.n_open_leaves
        out.n_solid_leaves = self.n_solid_leaves
        out.n_detail_splits = self.n_detail_splits
        out.portals = None
        out.adj = None
        out._finish()
        out.schema = NEGSPACE_SCHEMA
        return out

    @staticmethod
    def union(parts):
        out = NegSpace.__new__(NegSpace)
        out.mask = parts[0].mask
        out.gridcell = parts[0].gridcell
        out.cells = []
        out.cell_leaf = []
        out.cell_tile = []
        for ti, p in enumerate(parts):
            out.cells += p.cells
            out.cell_leaf += p.cell_leaf
            out.cell_tile += [ti] * len(p.cells)
        out.world_lo = np.min(np.array([p.world_lo for p in parts]), axis=0)
        out.world_hi = np.max(np.array([p.world_hi for p in parts]), axis=0)
        out.n_open_leaves = sum(p.n_open_leaves for p in parts)
        out.n_solid_leaves = sum(p.n_solid_leaves for p in parts)
        out.n_detail_splits = sum(p.n_detail_splits for p in parts)
        out._finish()
        out.schema = NEGSPACE_SCHEMA
        return out

    def add_cells(self, Hs, tile=-1):
        if not hasattr(self, 'cell_tile'):
            self.cell_tile = [-1] * len(self.cells)
        for H in Hs:
            self.cells.append(np.asarray(H, dtype=np.float64))
            self.cell_leaf.append(-1)
            self.cell_tile.append(tile)
        self.world_lo = np.minimum(self.world_lo, np.min(
            [bounds_of(np.asarray(H, dtype=np.float64), self.world_lo - 8192, self.world_hi + 8192)[0]
             for H in Hs], axis=0)) if Hs else self.world_lo
        self._finish()

    def edit(self, add=(), remove=(), verbose=False):
        rem = [np.asarray(B, dtype=np.float64) for B in remove]
        touch = {}
        for bi, B in enumerate(rem):
            blo, bhi = bounds_of(B, self.world_lo - 8192.0, self.world_hi + 8192.0)
            for i in self._cells_in_box(blo - 1.0, bhi + 1.0):
                touch.setdefault(int(i), []).append(bi)
        keep = np.ones(len(self.cells), dtype=bool)
        newc = []
        nrem = 0
        for i, bis in touch.items():
            pieces = [self.cells[i]]
            for bi in bis:
                pieces = subtract(pieces, rem[bi], self.lo[i], self.hi[i],
                                  exact_empty=False)
            keep[i] = False
            newc += pieces
            nrem += 1
        self.cells = [c for i, c in enumerate(self.cells) if keep[i]] + newc
        self.cell_leaf = ([l for i, l in enumerate(self.cell_leaf) if keep[i]]
                          + [-1] * len(newc))
        if hasattr(self, 'cell_tile'):
            self.cell_tile = ([t for i, t in enumerate(self.cell_tile) if keep[i]]
                              + [-1] * len(newc))
        nadd = 0
        for H in add:
            H = np.asarray(H, dtype=np.float64)
            self.cells.append(H)
            self.cell_leaf.append(-1)
            if hasattr(self, 'cell_tile'):
                self.cell_tile.append(-1)
            nadd += 1
            lo2, hi2 = bounds_of(H, self.world_lo - 16384.0, self.world_hi + 16384.0)
            self.world_lo = np.minimum(self.world_lo, lo2)
            self.world_hi = np.maximum(self.world_hi, hi2)
        self.portals = None
        self.adj = None
        self._finish()
        if verbose:
            print('negspace: edit applied -- %d free cells re-cut around %d new '
                  'procedural solids, %d opened regions added; %d cells total'
                  % (nrem, len(rem), nadd, len(self.cells)))
        return nrem, nadd

    def _cands(self, p):
        c = self.gridcell
        k = (int(math.floor(p[0] / c)), int(math.floor(p[1] / c)), int(math.floor(p[2] / c)))
        return self.grid.get(k)

    def _solid_cands(self, lo, hi):
        _, solids = self._solid_relation_rows(
            np.asarray(lo, dtype=np.float64).reshape((1, 3)),
            np.asarray(hi, dtype=np.float64).reshape((1, 3)),
        )
        return solids

    def _solid_geom(self, i):
        if getattr(self, 'solids', None) is not None:
            return self.solids[i]
        return self.blk_H[i], self.blk_lo[i], self.blk_hi[i]

    def cell_at(self, p, tol=0.25):
        if (getattr(self, 'solids', None) is not None
                or getattr(self, 'blk_H', None) is not None and not self.cells):

            px = np.array([float(p[0]), float(p[1]), float(p[2])])
            if np.any(px < self.world_lo - tol) or np.any(px > self.world_hi + tol):
                return -1
            for i in self._solid_cands(px - 0.5, px + 0.5):
                H, blo, bhi = self._solid_geom(i)
                if np.any(px < blo - tol) or np.any(px > bhi + tol):
                    continue

                if (H[:, :3] @ px - H[:, 3]).max() <= tol:
                    return -1
            return 0
        cand = self._cands(p)
        if cand is None:
            return -1
        px, py, pz = float(p[0]), float(p[1]), float(p[2])
        m = ((self.lo[cand, 0] - tol <= px) & (px <= self.hi[cand, 0] + tol) &
             (self.lo[cand, 1] - tol <= py) & (py <= self.hi[cand, 1] + tol) &
             (self.lo[cand, 2] - tol <= pz) & (pz <= self.hi[cand, 2] + tol))
        for i in cand[m]:
            H = self.cells[i]
            if (H[:, 0] * px + H[:, 1] * py + H[:, 2] * pz - H[:, 3]).max() <= tol:
                return int(i)
        return -1

    def free(self, p):
        return self.cell_at(p) >= 0

    def covered(self, H, lo=None, hi=None, tol=1.0):
        if lo is None:
            lo, hi = bounds_of(H, self.world_lo - 4096.0, self.world_hi + 4096.0)
        if (getattr(self, 'solids', None) is not None
                or getattr(self, 'blk_H', None) is not None):
            if np.any(lo < self.world_lo) or np.any(hi > self.world_hi):
                return False
            for index in self._solid_cands(lo, hi):
                solid, slo, shi = self._solid_geom(index)
                if np.any(hi <= slo) or np.any(lo >= shi):
                    continue
                combined = np.vstack([H, solid])
                rlo, rhi = bounds_of(combined, np.maximum(lo, slo), np.minimum(hi, shi))
                if np.all(rhi - rlo > tol) and len(vertices(prune(combined, rlo, rhi))) >= 4:
                    return False
            return True
        cand = self._cells_in_box(lo, hi)
        pieces = [H]
        for i in cand:
            pieces = subtract(pieces, self.cells[i], lo, hi,
                              exact_empty=False, minext=tol)
            if not pieces:
                return True
        return not pieces

    def solid_incidence(self, H, lo=None, hi=None, tol=0.25):
        if lo is None:
            lo, hi = bounds_of(H, self.world_lo - 4096.0, self.world_hi + 4096.0)
        query_lo = np.asarray(lo, dtype=np.float64).reshape((1, 3))
        query_hi = np.asarray(hi, dtype=np.float64).reshape((1, 3))
        _, candidates = self._solid_relation_rows(query_lo, query_hi)
        for index in candidates:
            solid, slo, shi = self._solid_geom(index)
            lower = np.maximum(query_lo[0], slo)
            upper = np.minimum(query_hi[0], shi)
            points = vertices(np.vstack((H, solid)), lower, upper)
            if len(points) >= 4 and np.all(points.max(axis=0) - points.min(axis=0) > tol):
                return {'incidence_mass': 1, 'source_solid_candidate_mass': len(candidates)}
        return {'incidence_mass': 0, 'source_solid_candidate_mass': len(candidates)}

    def _fits_source(self, p, mins, maxs, tol=COLLISION_EPSILON):
        px = np.array([float(p[0]), float(p[1]), float(p[2])])
        mn = np.asarray(mins, dtype=float)
        mx = np.asarray(maxs, dtype=float)
        blo = px + mn
        bhi = px + mx
        if np.any(blo < self.world_lo) or np.any(bhi > self.world_hi):
            return False
        for i in self._solid_cands(blo, bhi):
            H, slo, shi = self._solid_geom(i)
            if np.any(bhi <= slo) or np.any(blo >= shi):
                continue
            n = H[:, :3]

            off = np.where(n > 0, mn, mx)
            dist = H[:, 3] - (off * n).sum(axis=1)
            if (n @ px - dist).max() <= -tol:
                return False
        return True

    def _source_pair_rows(self, points, mins, maxs):
        return self._solid_relation_rows(points + mins, points + maxs)

    def _source_segment_pair_rows(self, starts, ends, mins, maxs, padding):
        return self._solid_ray_relation_rows(
            starts, ends, mins, maxs, padding,
        )

    def _source_plane_rows(self, pair_solids):
        counts = self.solid_plane_counts[pair_solids]
        offsets = np.zeros(len(counts) + 1, dtype=np.int64)
        np.cumsum(counts, out=offsets[1:])
        pair_rows = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
        local_rows = np.arange(offsets[-1], dtype=np.int64) - np.repeat(offsets[:-1], counts)
        plane_rows = self.solid_plane_offsets[pair_solids[pair_rows]] + local_rows
        return counts, offsets, pair_rows, plane_rows

    def _solid_plane_streams(self, rows):
        return (
            self.solid_plane_n0[rows], self.solid_plane_n1[rows],
            self.solid_plane_n2[rows], self.solid_plane_distances[rows],
        )

    @staticmethod
    def _plane_dot_rows(n0, n1, n2, points, owners):
        result = np.empty_like(n0)
        work = np.empty_like(n0)
        np.multiply(n0, points[owners, 0], out=result)
        np.multiply(n1, points[owners, 1], out=work)
        np.add(result, work, out=result)
        np.multiply(n2, points[owners, 2], out=work)
        np.add(result, work, out=result)
        return result

    @staticmethod
    def _plane_support(n0, n1, n2, positive, negative):
        center = 0.5 * (positive + negative)
        extent = 0.5 * (positive - negative)
        result = np.empty_like(n0)
        work = np.empty_like(n0)
        np.multiply(n0, center[0], out=result)
        np.abs(n0, out=work)
        work *= extent[0]
        result += work
        for normal, axis in ((n1, 1), (n2, 2)):
            np.multiply(normal, center[axis], out=work)
            result += work
            np.abs(normal, out=work)
            work *= extent[axis]
            result += work
        return result

    def _source_penetrations(self, points, mins, maxs, tol):
        pair_owners, pair_solids = self._source_pair_rows(points, mins, maxs)
        if not len(pair_owners):
            return {
                'owners': pair_owners,
                'solids': pair_owners,
                'n0': np.zeros(0, dtype=np.float64),
                'n1': np.zeros(0, dtype=np.float64),
                'n2': np.zeros(0, dtype=np.float64),
                'signed': np.zeros(0, dtype=np.float64),
                'pair_mass': 0,
                'plane_mass': 0,
            }
        counts, offsets, pair_rows, plane_rows = self._source_plane_rows(pair_solids)
        n0, n1, n2, plane_distances = self._solid_plane_streams(plane_rows)
        distances = plane_distances - self._plane_support(
            n0, n1, n2, mins, maxs,
        )
        owner_rows = pair_owners[pair_rows]
        signed = self._plane_dot_rows(
            n0, n1, n2, points, owner_rows,
        ) - distances
        pair_maximum = np.maximum.reduceat(signed, offsets[:-1])
        repeated_maximum = np.repeat(pair_maximum, counts)
        row_ids = np.arange(len(signed), dtype=np.int64)
        selected = np.where(signed == repeated_maximum, row_ids, len(signed))
        selected = np.minimum.reduceat(selected, offsets[:-1])
        active = pair_maximum <= -tol
        return {
            'owners': pair_owners[active],
            'solids': pair_solids[active],
            'n0': n0[selected[active]],
            'n1': n1[selected[active]],
            'n2': n2[selected[active]],
            'signed': pair_maximum[active],
            'pair_mass': len(pair_owners),
            'plane_mass': len(signed),
        }

    @staticmethod
    def _solve_spd3(a00, a01, a02, a11, a12, a22, b0, b1, b2):
        l00 = np.sqrt(a00)
        l10 = a01 / l00
        l20 = a02 / l00
        l11 = np.sqrt(a11 - l10 * l10)
        l21 = (a12 - l20 * l10) / l11
        l22 = np.sqrt(a22 - l20 * l20 - l21 * l21)
        y0 = b0 / l00
        y1 = (b1 - l10 * y0) / l11
        y2 = (b2 - l20 * y0 - l21 * y1) / l22
        x2 = y2 / l22
        x1 = (y1 - l21 * x2) / l11
        x0 = (y0 - l10 * x1 - l20 * x2) / l00
        return np.column_stack((x0, x1, x2))

    def _source_directions(self, owners, n0, n1, n2, depths):
        starts = np.r_[0, np.flatnonzero(owners[1:] != owners[:-1]) + 1]
        work = np.empty_like(n0)
        def product_sum(left, right):
            np.multiply(left, right, out=work)
            return np.add.reduceat(work, starts)
        delta = self._solve_spd3(
            1.0 + product_sum(n0, n0), product_sum(n0, n1), product_sum(n0, n2),
            1.0 + product_sum(n1, n1), product_sum(n1, n2),
            1.0 + product_sum(n2, n2), product_sum(n0, depths),
            product_sum(n1, depths), product_sum(n2, depths),
        )
        lengths = np.linalg.norm(delta, axis=1)
        missing = lengths <= np.finfo(np.float64).eps
        if np.any(missing):
            order = np.lexsort((np.arange(len(owners)), depths, owners))
            ordered_owners = owners[order]
            first = np.r_[True, ordered_owners[1:] != ordered_owners[:-1]]
            chosen = order[first]
            use = missing[owners[chosen]]
            targets = owners[chosen[use]]
            scales = depths[chosen[use]]
            delta[targets, 0] = n0[chosen[use]] * scales
            delta[targets, 1] = n1[chosen[use]] * scales
            delta[targets, 2] = n2[chosen[use]] * scales
            lengths = np.linalg.norm(delta, axis=1)
        directions = np.zeros_like(delta)
        active = lengths > 0.0
        directions[active] = delta[active] / lengths[active, None]
        return directions

    def _source_line_intervals(self, points, directions, pair_owners,
                               pair_solids, mins, maxs, tol):
        counts, offsets, pair_rows, plane_rows = self._source_plane_rows(pair_solids)
        n0, n1, n2, plane_distances = self._solid_plane_streams(plane_rows)
        distances = (plane_distances
                     - self._plane_support(n0, n1, n2, mins, maxs) + tol)
        owner_rows = pair_owners[pair_rows]
        signed = self._plane_dot_rows(
            n0, n1, n2, points, owner_rows,
        ) - distances
        slopes = self._plane_dot_rows(
            n0, n1, n2, directions, owner_rows,
        )
        stationary = slopes == 0.0
        outside = np.logical_or.reduceat(stationary & (signed > 0.0), offsets[:-1])
        parameters = np.zeros_like(signed)
        np.divide(-signed, slopes, out=parameters, where=~stationary)
        lower = np.maximum.reduceat(
            np.where(slopes < 0.0, parameters, -np.inf), offsets[:-1],
        )
        upper = np.minimum.reduceat(
            np.where(slopes > 0.0, parameters, np.inf), offsets[:-1],
        )
        return lower, upper, ~outside & (upper >= np.maximum(lower, 0.0)), len(signed)

    @staticmethod
    def _source_component_extend(component, owners, lower, upper, capacity):
        result = component.copy()
        if not len(owners):
            return result
        lower = np.maximum(lower, 0.0)
        upper = np.minimum(upper, capacity[owners])
        order = np.lexsort((upper, lower, owners))
        owners = owners[order]
        lower = lower[order]
        upper = upper[order]
        starts = np.r_[0, np.flatnonzero(owners[1:] != owners[:-1]) + 1]
        ends = np.r_[starts[1:], len(owners)]
        groups = np.cumsum(np.r_[True, owners[1:] != owners[:-1]]) - 1
        stride = np.nextafter(float(np.max(capacity)), np.inf)
        prefix = np.maximum.accumulate(upper + groups * stride) - groups * stride
        previous = np.r_[0.0, prefix[:-1]]
        previous[starts] = component[owners[starts]]
        gaps = lower > previous + EPS
        cumulative = np.cumsum(gaps)
        baseline = np.zeros(len(starts), dtype=np.int64)
        baseline[1:] = cumulative[starts[1:] - 1]
        connected = cumulative - np.repeat(baseline, ends - starts) == 0
        values = component[owners]
        np.copyto(values, upper, where=connected)
        extended = np.maximum.reduceat(values, starts)
        result[owners[starts]] = extended
        return result

    def _source_ray_component_steps(self, points, directions, mins, maxs,
                                    domain_lo, domain_hi, tol,
                                    seed_owners, seed_solids):
        with np.errstate(divide='ignore', invalid='ignore'):
            capacity = np.where(
                directions > 0.0,
                (domain_hi - points) / directions,
                np.where(directions < 0.0, (domain_lo - points) / directions, np.inf),
            ).min(axis=1)
        solid_mass = len(self.solid_plane_counts)
        seed_keys = np.unique(seed_owners * solid_mass + seed_solids)
        seen = seed_keys
        owners = seed_keys // solid_mass
        solids = seed_keys % solid_mass
        lower, upper, present, plane_mass = self._source_line_intervals(
            points, directions, owners, solids, mins, maxs, tol,
        )
        interval_owners = owners[present]
        interval_lower = lower[present]
        interval_upper = upper[present]
        component = self._source_component_extend(
            np.zeros(len(points), dtype=np.float64), interval_owners,
            interval_lower, interval_upper, capacity,
        )
        front = np.zeros(len(points), dtype=np.float64)
        active = np.flatnonzero(component > 0.0)
        sweep_mass = 0
        while len(active):
            target = np.minimum(
                np.maximum(2.0 * component[active], component[active] + tol),
                capacity[active],
            )
            starts = points[active] + directions[active] * front[active, None]
            ends = points[active] + directions[active] * target[:, None]
            local_owners, local_solids = self._solid_ray_relation_rows(
                starts, ends, mins, maxs, tol,
            )
            fresh_owners = active[local_owners]
            keys = np.unique(fresh_owners * solid_mass + local_solids)
            indices = np.searchsorted(seen, keys)
            fresh = (indices == len(seen))
            bounded = ~fresh
            fresh[bounded] = seen[indices[bounded]] != keys[bounded]
            fresh_keys = keys[fresh]
            previous = component.copy()
            if len(fresh_keys):
                new_owners = fresh_keys // solid_mass
                new_solids = fresh_keys % solid_mass
                lower, upper, present, evaluated = self._source_line_intervals(
                    points, directions, new_owners, new_solids, mins, maxs, tol,
                )
                plane_mass += evaluated
                seen = np.union1d(seen, fresh_keys)
                component = self._source_component_extend(
                    component, new_owners[present], lower[present], upper[present], capacity,
                )
            front[active] = target
            grew = component > previous + EPS
            active = np.flatnonzero(
                grew & (component >= front - EPS) & (front < capacity - EPS)
            )
            sweep_mass += 1
        successful = component < capacity - EPS
        component[successful] = np.nextafter(component[successful], np.inf)
        return component, successful, len(seen), plane_mass, sweep_mass

    def project_many(self, points, mins=PL_MIN, maxs=PL_MAX,
                     tolerance=COLLISION_EPSILON):
        origin = np.asarray(points, dtype=np.float64).reshape((-1, 3))
        if not len(origin):
            return origin.copy(), np.zeros(0, dtype=np.float64), {
                'input_point_mass': 0,
                'input_penetration_point_mass': 0,
                'input_penetration_pair_mass': 0,
                'projection_sweep_mass': 0,
                'candidate_pair_mass': 0,
                'plane_evaluation_mass': 0,
                'directional_null_pair_mass': 0,
                'world_boundary_reconciliation_mass': 0,
                'residual_penetration_point_mass': 0,
            }
        mn = np.zeros(3) if mins is None else np.asarray(mins, dtype=np.float64)
        mx = np.zeros(3) if maxs is None else np.asarray(maxs, dtype=np.float64)
        domain_lo = self.world_lo - mn + tolerance
        domain_hi = self.world_hi - mx - tolerance
        current = np.minimum(np.maximum(origin, domain_lo), domain_hi)
        candidate_pair_mass = 0
        plane_evaluation_mass = 0
        directional_null_pair_mass = 0
        boundary_mass = int(np.count_nonzero(np.any(current != origin, axis=1)))
        sweep_mass = 0
        initial_points = 0
        initial_pairs = 0
        center = 0.5 * (domain_lo + domain_hi)
        penetration = self._source_penetrations(current, mn, mx, COLLISION_EPSILON)
        candidate_pair_mass += penetration['pair_mass']
        plane_evaluation_mass += penetration['plane_mass']
        owners = penetration['owners']
        initial_points = len(np.unique(owners))
        initial_pairs = len(owners)
        if len(owners):
            active_points = np.unique(owners)
            depths = tolerance - penetration['signed']
            active_owners = np.searchsorted(active_points, owners)
            directions = self._source_directions(
                active_owners,
                penetration['n0'], penetration['n1'], penetration['n2'], depths,
            )
            steps, successful, pair_mass, plane_mass, ray_sweeps = self._source_ray_component_steps(
                current[active_points], directions, mn, mx,
                domain_lo, domain_hi, tolerance,
                active_owners, penetration['solids'],
            )
            candidate_pair_mass += pair_mass
            plane_evaluation_mass += plane_mass
            current[active_points[successful]] += (
                directions[successful] * steps[successful, None]
            )
            bounded_points = active_points[~successful]
            directional_null_pair_mass += len(bounded_points)
            boundary_mass += len(bounded_points)
            sweep_mass = ray_sweeps
            if len(bounded_points):
                inward = center[None, :] - current[bounded_points]
                inward_length = np.linalg.norm(inward, axis=1)
                centered = inward_length <= np.finfo(np.float64).eps
                if np.any(centered):
                    spans = domain_hi - domain_lo
                    axis = int(np.argmax(spans))
                    inward[centered] = 0.0
                    inward[centered, axis] = 1.0
                    inward_length[centered] = 1.0
                inward /= inward_length[:, None]
                bounded_local = np.flatnonzero(~successful)
                seed_mask = ~successful[active_owners]
                fallback_owners = np.searchsorted(
                    bounded_local, active_owners[seed_mask],
                )
                steps, successful, pair_mass, plane_mass, ray_sweeps = (
                    self._source_ray_component_steps(
                        current[bounded_points], inward, mn, mx,
                        domain_lo, domain_hi, tolerance,
                        fallback_owners, penetration['solids'][seed_mask],
                    )
                )
                candidate_pair_mass += pair_mass
                plane_evaluation_mass += plane_mass
                current[bounded_points[successful]] += (
                    inward[successful] * steps[successful, None]
                )
                directional_null_pair_mass += int((~successful).sum())
                sweep_mass += ray_sweeps
            residual = self._source_penetrations(current, mn, mx, COLLISION_EPSILON)
            candidate_pair_mass += residual['pair_mass']
            plane_evaluation_mass += residual['plane_mass']
        else:
            residual = penetration
        residual_point_mass = len(np.unique(residual['owners']))
        delta = current - origin
        distances = np.linalg.norm(delta, axis=1)
        return current, distances, {
            'input_point_mass': len(origin),
            'input_penetration_point_mass': initial_points,
            'input_penetration_pair_mass': initial_pairs,
            'projection_sweep_mass': sweep_mass,
            'candidate_pair_mass': candidate_pair_mass,
            'plane_evaluation_mass': plane_evaluation_mass,
            'directional_null_pair_mass': directional_null_pair_mass,
            'world_boundary_reconciliation_mass': boundary_mass,
            'residual_penetration_point_mass': residual_point_mass,
        }

    @staticmethod
    def _linear_interval_rows(left, right, offsets):
        outside = np.logical_or.reduceat((left > 0.0) & (right > 0.0), offsets[:-1])
        entry = (left > 0.0) & (right <= 0.0)
        leave = (left <= 0.0) & (right > 0.0)
        crossing = entry | leave
        denominator = np.ones_like(left)
        np.subtract(left, right, out=denominator, where=crossing)
        parameter = np.divide(left, denominator, out=np.zeros_like(left), where=crossing)
        lower = np.maximum.reduceat(np.where(entry, parameter, 0.0), offsets[:-1])
        upper = np.minimum.reduceat(np.where(leave, parameter, 1.0), offsets[:-1])
        return lower, upper, outside

    def _segment_relations_batch(self, starts, ends, mins, maxs,
                                 activation, minimum_normal):
        starts = np.asarray(starts, dtype=np.float64).reshape((-1, 3))
        ends = np.asarray(ends, dtype=np.float64).reshape((-1, 3))
        mins = np.asarray(mins, dtype=np.float64)
        maxs = np.asarray(maxs, dtype=np.float64)
        free = np.ones(len(starts), dtype=bool)
        supported = np.zeros(len(starts), dtype=bool)
        pair_owners, pair_solids = self._source_segment_pair_rows(
            starts, ends, mins, maxs, activation + EPS,
        )
        measures = {
            'segment_mass': len(starts),
            'candidate_pair_mass': len(pair_owners),
            'plane_evaluation_mass': 0,
            'support_face_mass': 0,
            'support_constraint_mass': 0,
        }
        if not len(pair_owners):
            return free, supported, measures
        counts, offsets, pair_rows, plane_rows = self._source_plane_rows(pair_solids)
        n0, n1, n2, plane_distances = self._solid_plane_streams(plane_rows)
        distances = plane_distances - self._plane_support(
            n0, n1, n2, mins, maxs,
        )
        segment_rows = pair_owners[pair_rows]
        left = (self._plane_dot_rows(n0, n1, n2, starts, segment_rows) - distances
                + COLLISION_EPSILON)
        right = (self._plane_dot_rows(n0, n1, n2, ends, segment_rows) - distances
                 + COLLISION_EPSILON)
        lower, upper, outside = self._linear_interval_rows(left, right, offsets)
        collision = ~outside & (upper > lower)
        free[np.unique(pair_owners[collision])] = False
        measures['plane_evaluation_mass'] += len(left)

        upward = n2 >= minimum_normal
        if not np.any(upward):
            return free, supported, measures
        face_plane_rows = plane_rows[upward]
        face_pairs = pair_rows[upward]
        face_owners = pair_owners[face_pairs]
        face_rows = np.flatnonzero(upward)
        fn0, fn1, fn2, face_distances = self._solid_plane_streams(face_plane_rows)
        face_support = self._plane_support(fn0, fn1, fn2, mins, maxs)
        face_left = self._plane_dot_rows(
            fn0, fn1, fn2, starts, face_owners,
        ) + face_support - face_distances
        face_right = self._plane_dot_rows(
            fn0, fn1, fn2, ends, face_owners,
        ) + face_support - face_distances
        constraint_left = np.column_stack((
            -face_left - COLLISION_EPSILON,
            face_left - activation - COLLISION_EPSILON,
        ))
        constraint_right = np.column_stack((
            -face_right - COLLISION_EPSILON,
            face_right - activation - COLLISION_EPSILON,
        ))
        face_lower = np.zeros(len(face_pairs), dtype=np.float64)
        face_upper = np.ones(len(face_pairs), dtype=np.float64)
        face_outside = np.any((constraint_left > 0.0) & (constraint_right > 0.0), axis=1)
        face_entry = (constraint_left > 0.0) & (constraint_right <= 0.0)
        face_leave = (constraint_left <= 0.0) & (constraint_right > 0.0)
        face_crossing = face_entry | face_leave
        face_denominator = np.ones_like(constraint_left)
        np.subtract(constraint_left, constraint_right, out=face_denominator, where=face_crossing)
        face_parameter = np.divide(
            constraint_left, face_denominator,
            out=np.zeros_like(constraint_left), where=face_crossing,
        )
        face_lower = np.maximum(face_lower, np.max(np.where(face_entry, face_parameter, 0.0), axis=1))
        face_upper = np.minimum(face_upper, np.min(np.where(face_leave, face_parameter, 1.0), axis=1))

        general_support = (
            np.where(n0 > 0.0, maxs[0], mins[0]) * n0
            + np.where(n1 > 0.0, maxs[1], mins[1]) * n1
            + n2 * mins[2]
        )
        general_left = (self._plane_dot_rows(
            n0, n1, n2, starts, segment_rows,
        ) + general_support - plane_distances)
        general_right = (self._plane_dot_rows(
            n0, n1, n2, ends, segment_rows,
        ) + general_support - plane_distances)
        general_outside = (general_left > 0.0) & (general_right > 0.0)
        general_entry = (general_left > 0.0) & (general_right <= 0.0)
        general_leave = (general_left <= 0.0) & (general_right > 0.0)
        general_crossing = general_entry | general_leave
        general_denominator = np.ones_like(general_left)
        np.subtract(
            general_left, general_right,
            out=general_denominator, where=general_crossing,
        )
        general_parameter = np.divide(
            general_left, general_denominator,
            out=np.zeros_like(general_left), where=general_crossing,
        )
        lower_rows = np.where(general_entry, general_parameter, 0.0)
        upper_rows = np.where(general_leave, general_parameter, 1.0)
        lower_first = np.maximum.reduceat(lower_rows, offsets[:-1])
        upper_first = np.minimum.reduceat(upper_rows, offsets[:-1])
        repeated_lower = np.repeat(lower_first, counts)
        repeated_upper = np.repeat(upper_first, counts)
        lower_first_mass = np.add.reduceat(lower_rows == repeated_lower, offsets[:-1])
        upper_first_mass = np.add.reduceat(upper_rows == repeated_upper, offsets[:-1])
        lower_second = np.maximum.reduceat(
            np.where(lower_rows < repeated_lower, lower_rows, 0.0), offsets[:-1],
        )
        upper_second = np.minimum.reduceat(
            np.where(upper_rows > repeated_upper, upper_rows, 1.0), offsets[:-1],
        )
        outside_mass = np.add.reduceat(general_outside, offsets[:-1])
        other_lower = lower_first[face_pairs].copy()
        other_upper = upper_first[face_pairs].copy()
        unique_lower = ((lower_rows[face_rows] == lower_first[face_pairs])
                        & (lower_first_mass[face_pairs] == 1))
        unique_upper = ((upper_rows[face_rows] == upper_first[face_pairs])
                        & (upper_first_mass[face_pairs] == 1))
        other_lower[unique_lower] = lower_second[face_pairs[unique_lower]]
        other_upper[unique_upper] = upper_second[face_pairs[unique_upper]]
        other_outside = outside_mass[face_pairs] - general_outside[face_rows] > 0
        face_lower = np.maximum(face_lower, other_lower)
        face_upper = np.minimum(face_upper, other_upper)
        face_outside |= other_outside
        face_active = ~face_outside & (face_upper >= face_lower)
        measures['plane_evaluation_mass'] += 2 * len(face_left) + len(general_left)
        measures['support_face_mass'] = int(face_active.sum())
        measures['support_constraint_mass'] = int(2 * len(face_left) + len(general_left))
        if not np.any(face_active):
            return free, supported, measures
        interval_owner = face_owners[face_active]
        interval_lower = face_lower[face_active]
        interval_upper = face_upper[face_active]
        order = np.lexsort((interval_upper, interval_lower, interval_owner))
        interval_owner = interval_owner[order]
        interval_lower = interval_lower[order]
        interval_upper = interval_upper[order]
        group_starts = np.r_[0, np.flatnonzero(interval_owner[1:] != interval_owner[:-1]) + 1]
        group_ends = np.r_[group_starts[1:], len(interval_owner)]
        groups = np.cumsum(np.r_[True, interval_owner[1:] != interval_owner[:-1]]) - 1
        stride = np.nextafter(float(np.max(interval_upper)), np.inf)
        shifted_upper = interval_upper + groups * stride
        prefix_upper = np.maximum.accumulate(shifted_upper) - groups * stride
        previous_upper = np.r_[0.0, prefix_upper[:-1]]
        previous_upper[group_starts] = 0.0
        discontinuity = interval_lower > previous_upper + EPS
        discontinuity_mass = np.add.reduceat(discontinuity, group_starts)
        reaches_end = prefix_upper[group_ends - 1] >= 1.0 - EPS
        supported[interval_owner[group_starts]] = (discontinuity_mass == 0) & reaches_end
        return free, supported, measures

    def segment_relations(self, starts, ends, mins=PL_MIN, maxs=PL_MAX,
                          activation=1.0, minimum_normal=FLOOR_NZ):
        starts = np.asarray(starts, dtype=np.float64).reshape((-1, 3))
        ends = np.asarray(ends, dtype=np.float64).reshape((-1, 3))
        free = np.ones(len(starts), dtype=bool)
        supported = np.zeros(len(starts), dtype=bool)
        measures = {
            'segment_mass': len(starts),
            'working_set_mass': 0,
            'candidate_pair_mass': 0,
            'plane_evaluation_mass': 0,
            'support_face_mass': 0,
            'support_constraint_mass': 0,
        }
        if not len(starts):
            return free, supported, measures
        working_set_mass = min(
            len(starts), max(os.cpu_count() or 1, math.isqrt(len(starts))),
        )
        bounds = np.linspace(0, len(starts), working_set_mass + 1, dtype=np.int64)
        measures['working_set_mass'] = working_set_mass
        for begin, end in zip(bounds[:-1], bounds[1:]):
            batch_free, batch_supported, batch_measures = self._segment_relations_batch(
                starts[begin:end], ends[begin:end], mins, maxs,
                activation, minimum_normal,
            )
            free[begin:end] = batch_free
            supported[begin:end] = batch_supported
            for name in ('candidate_pair_mass', 'plane_evaluation_mass',
                         'support_face_mass', 'support_constraint_mass'):
                measures[name] += batch_measures[name]
        return free, supported, measures

    def fits(self, p, mins=PL_MIN, maxs=PL_MAX):
        if (getattr(self, 'solids', None) is not None
                or getattr(self, 'blk_H', None) is not None):
            return self._fits_source(p, mins, maxs)
        lo = (p[0] + mins[0] + 0.03, p[1] + mins[1] + 0.03, p[2] + mins[2] + 0.03)
        hi = (p[0] + maxs[0] - 0.03, p[1] + maxs[1] - 0.03, p[2] + maxs[2] - 0.03)
        if np.any(np.asarray(hi) <= np.asarray(lo)):
            return False
        return self.covered(box_H(lo, hi), np.asarray(lo), np.asarray(hi))

    def fits_many(self, points, mins=PL_MIN, maxs=PL_MAX):
        points = np.asarray(points, dtype=np.float64).reshape((-1, 3))
        if not len(points):
            return np.ones(0, dtype=bool)
        if (getattr(self, 'solids', None) is None
                and getattr(self, 'blk_H', None) is None):
            return np.asarray([self.fits(point, mins, maxs) for point in points], dtype=bool)
        mn = np.asarray(mins, dtype=np.float64)
        mx = np.asarray(maxs, dtype=np.float64)
        present = np.all((points + mn >= self.world_lo)
                         & (points + mx <= self.world_hi), axis=1)
        rows = np.flatnonzero(present)
        if len(rows):
            penetration = self._source_penetrations(
                points[rows], mn, mx, COLLISION_EPSILON,
            )
            if len(penetration['owners']):
                present[rows[np.unique(penetration['owners'])]] = False
        return present

    def intersects_free(self, H, lo=None, hi=None):
        if lo is None:
            lo, hi = bounds_of(H, self.world_lo - 4096.0, self.world_hi + 4096.0)
        for i in self._cells_in_box(lo, hi):
            R = np.vstack([H, self.cells[i]])
            rlo, rhi = bounds_of(R, lo, hi)
            if np.any(rhi - rlo < 0.5):
                continue
            if len(vertices(prune(R, rlo, rhi))) >= 4:
                return True
        return False

    def _cells_in_box(self, lo, hi):
        c = self.gridcell
        out = []
        for x in range(int(math.floor(lo[0] / c)), int(math.floor(hi[0] / c)) + 1):
            for y in range(int(math.floor(lo[1] / c)), int(math.floor(hi[1] / c)) + 1):
                for z in range(int(math.floor(lo[2] / c)), int(math.floor(hi[2] / c)) + 1):
                    g = self.grid.get((x, y, z))
                    if g is not None:
                        out.append(g)
        if not out:
            return np.zeros(0, dtype=np.int64)
        cand = np.unique(np.concatenate(out))
        m = ((self.lo[cand, 0] < hi[0]) & (self.hi[cand, 0] > lo[0]) &
             (self.lo[cand, 1] < hi[1]) & (self.hi[cand, 1] > lo[1]) &
             (self.lo[cand, 2] < hi[2]) & (self.hi[cand, 2] > lo[2]))
        return cand[m]

    def _segment_intervals_source(self, a, b, mins=None, maxs=None):
        A = np.asarray(a, dtype=float)
        B = np.asarray(b, dtype=float)
        blo = np.minimum(A, B)
        bhi = np.maximum(A, B)
        if mins is not None:
            blo = blo + np.asarray(mins, dtype=float)
            bhi = bhi + np.asarray(maxs, dtype=float)
            domain_lo = self.world_lo - np.asarray(mins, dtype=float)
            domain_hi = self.world_hi - np.asarray(maxs, dtype=float)
        else:
            domain_lo = self.world_lo
            domain_hi = self.world_hi
        domain_start, domain_end = 0.0, 1.0
        delta = B - A
        for axis in range(3):
            if abs(delta[axis]) < 1e-12:
                if A[axis] < domain_lo[axis] or A[axis] > domain_hi[axis]:
                    return []
                continue
            first = (domain_lo[axis] - A[axis]) / delta[axis]
            second = (domain_hi[axis] - A[axis]) / delta[axis]
            domain_start = max(domain_start, min(first, second))
            domain_end = min(domain_end, max(first, second))
            if domain_end <= domain_start:
                return []
        solid = []
        for i in self._solid_cands(blo, bhi):
            H, slo, shi = self._solid_geom(i)
            if np.any(bhi <= slo) or np.any(blo >= shi):
                continue
            n = H[:, :3]
            if mins is None:
                dist = H[:, 3]
            else:
                off = np.where(n > 0, np.asarray(mins, dtype=float),
                               np.asarray(maxs, dtype=float))
                dist = H[:, 3] - (off * n).sum(axis=1)
            da = n @ A - dist
            db = n @ B - dist
            da += COLLISION_EPSILON
            db += COLLISION_EPSILON
            t0, t1, ok = 0.0, 1.0, True
            for k in range(len(H)):
                x, y = da[k], db[k]
                if x > 0 and y > 0:
                    ok = False
                    break
                if x <= 0 and y <= 0:
                    continue
                t = x / (x - y)

                if x > y:
                    t0 = max(t0, t)
                else:
                    t1 = min(t1, t)
                if t0 > t1:
                    ok = False
                    break
            if ok and t1 > t0:
                solid.append((t0, t1))
        if not solid:
            return [(domain_start, domain_end)]
        solid.sort()
        merged = [list(solid[0])]
        for s0, e0 in solid[1:]:
            if s0 <= merged[-1][1] + 1e-9:
                merged[-1][1] = max(merged[-1][1], e0)
            else:
                merged.append([s0, e0])
        free = []
        t = domain_start
        for s0, e0 in merged:
            if e0 <= domain_start or s0 >= domain_end:
                continue
            s0 = max(s0, domain_start)
            e0 = min(e0, domain_end)
            if s0 > t + 1e-9:
                free.append((t, s0))
            t = max(t, e0)
        if t < domain_end - 1e-9:
            free.append((t, domain_end))
        return free

    def segment_intervals(self, a, b, mins=None, maxs=None):
        if (getattr(self, 'solids', None) is not None
                or getattr(self, 'blk_H', None) is not None):
            return self._segment_intervals_source(a, b, mins, maxs)
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        cand = self._cells_along(a, b)
        segs = []
        mn = None if mins is None else np.asarray(mins, dtype=np.float64)
        mx = None if maxs is None else np.asarray(maxs, dtype=np.float64)

        def consume(H, starts):
            n = H[:, :3]
            d = H[:, 3]
            if mn is not None:
                d = d - (np.where(n > 0, mx, mn) * n).sum(axis=1)
            da = n @ a - d
            db = n @ b - d
            entry = (da > 0) & (db <= 0)
            leave = (da <= 0) & (db > 0)
            crossing = entry | leave
            denominator = np.ones_like(da)
            np.subtract(da, db, out=denominator, where=crossing)
            t = np.divide(da, denominator, out=np.zeros_like(da), where=crossing)
            t0 = np.maximum.reduceat(np.where(entry, t, 0.0), starts)
            t1 = np.minimum.reduceat(np.where(leave, t, 1.0), starts)
            outside = np.logical_or.reduceat((da > 0) & (db > 0), starts)
            ok = ~outside & (t1 > t0)
            segs.extend(zip(t0[ok].tolist(), t1[ok].tolist()))

        flat = getattr(self, '_flat', None)
        offsets = getattr(self, '_offsets', None)
        for base in range(0, len(cand), 16384):
            batch = cand[base:base + 16384]
            counts = self.counts[batch].astype(np.int64)
            starts = np.empty(len(batch), dtype=np.int64)
            starts[0] = 0
            np.cumsum(counts[:-1], out=starts[1:])
            if flat is not None and offsets is not None:
                rows = np.arange(int(counts.sum()), dtype=np.int64)
                rows += np.repeat(offsets[batch] - starts, counts)
                consume(flat[rows], starts)
            else:
                width = int(counts.max())
                H = np.zeros((len(batch), width, 4), dtype=np.float64)
                H[:, :, 3] = np.inf
                for slot, index in enumerate(batch):
                    cell = self.cells[int(index)]
                    H[slot, :len(cell)] = cell
                consume(H.reshape(-1, 4), np.arange(len(batch), dtype=np.int64) * width)
        if not segs:
            return []
        segs.sort()
        out = [list(segs[0])]
        for s, e in segs[1:]:
            if s <= out[-1][1] + 1e-6:
                out[-1][1] = max(out[-1][1], e)
            else:
                out.append([s, e])
        return [(s, e) for s, e in out]

    def segment_gaps(self, a, b, mins=None, maxs=None):
        iv = self.segment_intervals(a, b, mins, maxs)
        gaps = []
        t = 0.0
        for s, e in iv:
            if s > t + 1e-6:
                gaps.append((t, s))
            t = max(t, e)
        if t < 1.0 - 1e-6:
            gaps.append((t, 1.0))
        return gaps

    def segment_free(self, a, b, mins=None, maxs=None):
        return not self.segment_gaps(a, b, mins, maxs)

    def support_intervals(self, a, b, mins=PL_MIN, maxs=PL_MAX,
                          activation=1.0, minimum_normal=FLOOR_NZ):
        if (getattr(self, 'solids', None) is not None
                or getattr(self, 'blk_H', None) is not None):
            return self._support_intervals_source(
                a, b, mins, maxs, activation, minimum_normal,
            )
        if not self.cells:
            return []
        if self.portals is None:
            self.build_portals()
        used = set()
        for portal in self.portals:
            key = tuple(round(float(value), 4) for value in portal.n) + (round(float(portal.d), 3),)
            used.add((portal.a, key))
            used.add((portal.b, tuple(-value for value in key)))
        A = np.asarray(a, dtype=np.float64)
        B = np.asarray(b, dtype=np.float64)
        mn = np.asarray(mins, dtype=np.float64)
        mx = np.asarray(maxs, dtype=np.float64)
        lo = np.minimum(A, B) + mn - activation - EPS
        hi = np.maximum(A, B) + mx + activation + EPS
        intervals = []
        for index in self._cells_in_box(lo, hi):
            H = self.cells[int(index)]
            S = shrink_H(H, mn, mx)
            da = S[:, :3] @ A - S[:, 3]
            db = S[:, :3] @ B - S[:, 3]
            t0, t1 = 0.0, 1.0
            outside = False
            for left, right in zip(da, db):
                if left > 0.0 and right > 0.0:
                    outside = True
                    break
                if left > 0.0 or right > 0.0:
                    crossing = left / (left - right)
                    if left > right:
                        t0 = max(t0, float(crossing))
                    else:
                        t1 = min(t1, float(crossing))
            if outside or t1 < t0:
                continue
            for row, plane in enumerate(H):
                if plane[2] > -minimum_normal:
                    continue
                key = tuple(round(float(value), 4) for value in plane[:3]) + (round(float(plane[3]), 3),)
                if (int(index), key) in used:
                    continue
                normal = float(np.linalg.norm(plane[:3]))
                lower = -activation * normal
                left = float(da[row])
                right = float(db[row])
                s0, s1 = t0, t1
                if left < lower and right < lower:
                    continue
                if left < lower or right < lower:
                    crossing = (left - lower) / (left - right)
                    if left < right:
                        s0 = max(s0, crossing)
                    else:
                        s1 = min(s1, crossing)
                if s1 >= s0:
                    intervals.append((s0, s1))
        if not intervals:
            return []
        intervals.sort()
        merged = [list(intervals[0])]
        for start, end in intervals[1:]:
            if start <= merged[-1][1] + 1e-6:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        return [(start, end) for start, end in merged]

    def _support_intervals_source(self, a, b, mins, maxs,
                                  activation, minimum_normal):
        A = np.asarray(a, dtype=np.float64)
        B = np.asarray(b, dtype=np.float64)
        mn = np.asarray(mins, dtype=np.float64)
        mx = np.asarray(maxs, dtype=np.float64)
        lo = np.minimum(A, B) + mn - activation - EPS
        hi = np.maximum(A, B) + mx + activation + EPS
        intervals = []
        for index in self._solid_cands(lo, hi):
            H, slo, shi = self._solid_geom(index)
            if np.any(hi <= slo) or np.any(lo >= shi):
                continue
            for face, plane in enumerate(H):
                n = plane[:3]
                normal = float(np.linalg.norm(n))
                if normal == 0.0 or n[2] / normal < minimum_normal:
                    continue
                low_offset = np.where(n > 0, mn, mx)
                va = float(n @ A + n @ low_offset - plane[3])
                vb = float(n @ B + n @ low_offset - plane[3])
                tolerance = COLLISION_EPSILON * normal
                constraints = [(-va - tolerance, -vb - tolerance),
                               (va - activation * normal - tolerance,
                                vb - activation * normal - tolerance)]
                for other, row in enumerate(H):
                    if other == face:
                        continue
                    horizontal = np.where(row[:2] > 0, mx[:2], mn[:2])
                    offset = float(row[:2] @ horizontal + row[2] * mn[2])
                    constraints.append((float(row[:3] @ A + offset - row[3]),
                                        float(row[:3] @ B + offset - row[3])))
                t0, t1 = 0.0, 1.0
                for left, right in constraints:
                    if left > 0.0 and right > 0.0:
                        t0, t1 = 1.0, 0.0
                        break
                    if left > 0.0 or right > 0.0:
                        crossing = left / (left - right)
                        if left > right:
                            t0 = max(t0, crossing)
                        else:
                            t1 = min(t1, crossing)
                    if t1 < t0:
                        break
                if t1 >= t0:
                    intervals.append((t0, t1))
        if not intervals:
            return []
        intervals.sort()
        merged = [list(intervals[0])]
        for start, end in intervals[1:]:
            if start <= merged[-1][1] + 1e-6:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        return [(start, end) for start, end in merged]

    def support_gaps(self, a, b, mins=PL_MIN, maxs=PL_MAX,
                     activation=1.0, minimum_normal=FLOOR_NZ):
        intervals = self.support_intervals(a, b, mins, maxs, activation, minimum_normal)
        gaps = []
        cursor = 0.0
        for start, end in intervals:
            if start > cursor + 1e-6:
                gaps.append((cursor, start))
            cursor = max(cursor, end)
        if cursor < 1.0 - 1e-6:
            gaps.append((cursor, 1.0))
        return gaps

    def segment_supported(self, a, b, mins=PL_MIN, maxs=PL_MAX,
                          activation=1.0, minimum_normal=FLOOR_NZ):
        return (self.segment_free(a, b, mins, maxs)
                and not self.support_gaps(a, b, mins, maxs, activation, minimum_normal))

    def _cells_along(self, a, b):
        c = self.gridcell
        L = float(np.linalg.norm(np.asarray(b) - np.asarray(a)))
        nstep = max(1, int(L / (c * 0.5)) + 1)
        seen = set()
        out = []
        for i in range(nstep + 1):
            f = i / float(nstep)
            px = a[0] + f * (b[0] - a[0])
            py = a[1] + f * (b[1] - a[1])
            pz = a[2] + f * (b[2] - a[2])
            cx, cy, cz = int(math.floor(px / c)), int(math.floor(py / c)), int(math.floor(pz / c))
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        k = (cx + dx, cy + dy, cz + dz)
                        if k in seen:
                            continue
                        seen.add(k)
                        g = self.grid.get(k)
                        if g is not None:
                            out.append(g)
        if not out:
            return np.zeros(0, dtype=np.int64)
        return np.unique(np.concatenate(out))

    def project(self, p, mins=PL_MIN, maxs=PL_MAX,
                tolerance=COLLISION_EPSILON):
        projected, distances, _ = self.project_many(
            np.asarray(p, dtype=np.float64)[None, :], mins, maxs, tolerance,
        )
        return projected[0].tolist(), float(distances[0])

    def floor_under_many(self, points, maxdrop=512.0, footprint=None):
        points = np.asarray(points, dtype=np.float64).reshape((-1, 3))
        floors = np.full(len(points), np.nan, dtype=np.float64)
        measures = {
            'point_mass': len(points),
            'ray_mass': 0,
            'working_set_mass': 0,
            'candidate_pair_mass': 0,
            'plane_evaluation_mass': 0,
            'support_point_mass': 0,
        }
        if not len(points):
            return floors, measures
        if (getattr(self, 'solids', None) is None
                and getattr(self, 'blk_H', None) is None):
            for index, point in enumerate(points):
                value = self.floor_under(point, maxdrop, footprint)
                if value is not None:
                    floors[index] = value
            measures['support_point_mass'] = int(np.isfinite(floors).sum())
            return floors, measures
        offsets = [(0.0, 0.0, 0.0)]
        if footprint:
            hx, hy = footprint
            offsets += [(sx * hx, sy * hy, 0.0) for sx in (-1, 1) for sy in (-1, 1)]
            offsets += [(sx * hx, 0.0, 0.0) for sx in (-1, 1)]
            offsets += [(0.0, sy * hy, 0.0) for sy in (-1, 1)]
        offsets = np.asarray(offsets, dtype=np.float64)
        measures['ray_mass'] = len(points) * len(offsets)
        working_set_mass = min(len(points), os.cpu_count() or 1)
        measures['working_set_mass'] = working_set_mass
        bounds = np.linspace(0, len(points), working_set_mass + 1, dtype=np.int64)
        zero = np.zeros(3, dtype=np.float64)
        drop = float(maxdrop)
        for begin, end in zip(bounds[:-1], bounds[1:]):
            starts = (points[begin:end, None, :] + offsets[None, :, :]).reshape((-1, 3))
            ends = starts.copy()
            ends[:, 2] -= drop
            pair_owners, pair_solids = self._source_segment_pair_rows(
                starts, ends, zero, zero, 0.0,
            )
            hit = np.full(len(starts), np.inf, dtype=np.float64)
            if len(pair_owners):
                counts, plane_offsets, pair_rows, plane_rows = self._source_plane_rows(pair_solids)
                n0, n1, n2, distances = self._solid_plane_streams(plane_rows)
                ray_rows = pair_owners[pair_rows]
                left = self._plane_dot_rows(
                    n0, n1, n2, starts, ray_rows,
                ) - distances
                right = self._plane_dot_rows(
                    n0, n1, n2, ends, ray_rows,
                ) - distances
                lower, upper, outside = self._linear_interval_rows(
                    left, right, plane_offsets,
                )
                collision = ~outside & (upper > lower) & (lower > COLLISION_EPSILON)
                np.minimum.at(hit, pair_owners[collision], lower[collision])
                measures['candidate_pair_mass'] += len(pair_owners)
                measures['plane_evaluation_mass'] += len(left)
            inside = np.all((starts >= self.world_lo) & (starts <= self.world_hi), axis=1)
            boundary = np.divide(
                starts[:, 2] - self.world_lo[2], drop,
                out=np.full(len(starts), np.inf, dtype=np.float64),
                where=drop > 0.0,
            )
            boundary = np.where(inside & (boundary > 0.0) & (boundary < 1.0),
                                boundary, np.inf)
            hit = np.minimum(hit, boundary)
            height = np.where(np.isfinite(hit), starts[:, 2] - hit * drop, -np.inf)
            height = height.reshape((end - begin, len(offsets)))
            maximum = np.max(height, axis=1)
            present = np.isfinite(maximum)
            floors[np.arange(begin, end)[present]] = maximum[present]
        measures['support_point_mass'] = int(np.isfinite(floors).sum())
        return floors, measures

    def floor_under(self, p, maxdrop=512.0, footprint=None):
        if (getattr(self, 'solids', None) is not None
                or getattr(self, 'blk_H', None) is not None):
            floors, _ = self.floor_under_many(
                np.asarray(p, dtype=np.float64)[None, :], maxdrop, footprint,
            )
            return float(floors[0]) if np.isfinite(floors[0]) else None
        offs = [(0.0, 0.0)]
        if footprint:
            hx, hy = footprint
            offs += [(sx * hx, sy * hy) for sx in (-1, 1) for sy in (-1, 1)]
            offs += [(sx * hx, 0.0) for sx in (-1, 1)] + [(0.0, sy * hy) for sy in (-1, 1)]
        best = None
        for dx, dy in offs:
            a = (float(p[0]) + dx, float(p[1]) + dy, float(p[2]))
            b = (a[0], a[1], a[2] - maxdrop)
            run = None
            for s0, e0 in self.segment_intervals(a, b):
                if s0 <= 1e-6:
                    run = (s0, e0)
                    break
            if run is None or run[1] >= 1.0 - 1e-9:
                continue
            z = float(a[2] - run[1] * maxdrop)
            if best is None or z > best:
                best = z
        return best

    def solid_at(self, p):
        return self.cell_at(p) < 0

    def standable(self, p, mins=PL_MIN, maxs=PL_MAX, hover=64.0, lift=2.0):
        fz = self.floor_under(p, hover, footprint=(abs(mins[0]), abs(mins[1])))
        if fz is None:
            return False
        return self.fits([p[0], p[1], fz - mins[2] + lift], mins, maxs)

    def clearance(self, p, cap=512.0, mins=PL_MIN, maxs=PL_MAX, tol=1.0):
        h0, h1 = abs(mins[0]), abs(maxs[0])
        if not self.fits(p, mins, maxs):
            return 0.0
        lo, hi = max(h0, h1), float(cap)
        if self.fits(p, (-hi, -hi, mins[2]), (hi, hi, maxs[2])):
            return float(cap)
        for _ in range(12):
            if hi - lo <= tol:
                break
            mid = 0.5 * (lo + hi)
            if self.fits(p, (-mid, -mid, mins[2]), (mid, mid, maxs[2])):
                lo = mid
            else:
                hi = mid
        return float(lo)

    def trace_fraction(self, a, b, mins=None, maxs=None):
        gaps = self.segment_gaps(a, b, mins, maxs)
        return 1.0 if not gaps else float(gaps[0][0])

    def trace(self, a, b, mins=None, maxs=None):
        f = self.trace_fraction(a, b, mins, maxs)
        if f >= 1.0:
            return None
        return (a[0] + f * (b[0] - a[0]),
                a[1] + f * (b[1] - a[1]),
                a[2] + f * (b[2] - a[2]))

    def standing_points(self, points, mins=PL_MIN, maxs=PL_MAX, maxdrop=512.0,
                        lift=(1.0, 4.0, 12.0, 26.0)):
        points = np.asarray(points, dtype=np.float64).reshape((-1, 3))
        projected, _, projection = self.project_many(points, mins, maxs)
        floors, floor_measures = self.floor_under_many(
            projected, maxdrop, footprint=(abs(mins[0]), abs(mins[1])),
        )
        lifts = np.asarray(lift, dtype=np.float64).reshape(-1)
        realized = np.full_like(projected, np.nan)
        present = np.isfinite(floors)
        if len(lifts) and np.any(present):
            rows = np.flatnonzero(present)
            candidates = np.repeat(projected[rows, None, :], len(lifts), axis=1)
            candidates[:, :, 2] = (floors[rows, None] - float(mins[2])
                                    + lifts[None, :])
            relation = self.fits_many(candidates.reshape((-1, 3)), mins, maxs).reshape(
                (len(rows), len(lifts))
            )
            available = np.any(relation, axis=1)
            first = np.argmax(relation, axis=1)
            realized[rows[available]] = candidates[
                np.flatnonzero(available), first[available], :
            ]
            present[rows] = available
        measures = {
            'input_point_mass': len(points),
            'realized_point_mass': int(present.sum()),
            'unrealized_point_mass': int((~present).sum()),
            'lift_coordinate_mass': len(lifts),
            'projection': projection,
            'floor': floor_measures,
        }
        return realized, present, measures

    def standing_point(self, p, mins=PL_MIN, maxs=PL_MAX, maxdrop=512.0,
                       lift=(1.0, 4.0, 12.0, 26.0)):
        realized, present, _ = self.standing_points(
            np.asarray(p, dtype=np.float64)[None, :], mins, maxs, maxdrop, lift,
        )
        return realized[0].tolist() if present[0] else None

    def build_portals(self, min_radius=0.0, verbose=False):
        key = {}
        for i, H in enumerate(self.cells):
            for k in range(len(H)):
                n = H[k, :3]
                d = H[k, 3]
                kk = (round(float(n[0]), 4), round(float(n[1]), 4), round(float(n[2]), 4),
                      round(float(d), 3))
                key.setdefault(kk, []).append((i, k))
        portals = []
        adj = [[] for _ in range(len(self.cells))]
        done = set()
        for kk, lst in key.items():
            nk = (round(-kk[0], 4), round(-kk[1], 4), round(-kk[2], 4), round(-kk[3], 3))
            other = key.get(nk)
            if not other:
                continue
            if (nk, kk) in done:
                continue
            done.add((kk, nk))
            sections = {}
            for (i, ki), (j, kj) in _indexed_face_pairs(self.lo, self.hi, lst, other):
                    if i == j:
                        continue
                    if (i, ki) not in sections:
                        sections[(i, ki)] = self._cross(i, ki)
                    if (j, kj) not in sections:
                        sections[(j, kj)] = self._cross(j, kj)
                    pg, edges = sections[(i, ki)]
                    if len(pg) < 3:
                        continue
                    pg2, e2 = sections[(j, kj)]
                    n = self.cells[i][ki, :3]
                    d = self.cells[i][ki, 3]
                    if len(pg2) < 3:
                        continue
                    u, v, o = self._frame(n, d)
                    pg = list(pg)
                    ed = list(edges)
                    for (a2, b2, c2) in e2:
                        pg = _clip2(pg, a2, b2, c2)
                        if len(pg) < 3:
                            break
                        ed.append((a2, b2, c2))
                    if len(pg) < 3:
                        continue
                    ar = _area2(pg)
                    if ar < 1.0:
                        continue
                    rr = _inradius(pg, ed)
                    if rr < min_radius:
                        continue
                    cu = sum(q[0] for q in pg) / len(pg)
                    cv = sum(q[1] for q in pg) / len(pg)
                    ctr = o + u * cu + v * cv
                    p = Portal(i, j, n.copy(), float(d), pg, ar, rr,
                               [float(x) for x in ctr])
                    portals.append(p)
                    adj[i].append((j, len(portals) - 1))
                    adj[j].append((i, len(portals) - 1))
        self.portals = portals
        self.adj = adj
        if verbose:
            rs = sorted(p.radius for p in portals)
            print('negspace: %d portals between free cells; inscribed radius '
                  'min=%.1f p25=%.1f median=%.1f max=%.1f; %d admit a player '
                  '(r>=16)' % (len(portals), rs[0] if rs else 0,
                               rs[len(rs) // 4] if rs else 0, rs[len(rs) // 2] if rs else 0,
                               rs[-1] if rs else 0, sum(1 for r in rs if r >= 16.0)))
        return portals

    def _origin(self, n, d):
        a = int(np.argmax(np.abs(n)))
        o = np.zeros(3)
        o[a] = d / n[a]
        return o

    def _frame(self, n, d):
        n = np.asarray(n, dtype=np.float64)
        if (n[0], n[1], n[2]) < (0.0, 0.0, 0.0):
            n, d = -n, -d
        u, v = _basis(n)
        o = self._origin(n, d)
        return u, v, o

    def _cross(self, i, k):
        H = self.cells[i]
        u, v, o = self._frame(H[k, :3], H[k, 3])
        ext = float(max(np.max(self.hi[i] - self.lo[i]), 64.0)) * 2.0 + 512.0
        cu = float(np.dot(0.5 * (self.lo[i] + self.hi[i]) - o, u))
        cv = float(np.dot(0.5 * (self.lo[i] + self.hi[i]) - o, v))
        pg = [(cu - ext, cv - ext), (cu + ext, cv - ext), (cu + ext, cv + ext), (cu - ext, cv + ext)]
        edges = []
        for q in range(len(H)):
            if q == k:
                continue
            nq = H[q, :3]
            dq = H[q, 3]
            a = float(np.dot(nq, u))
            b = float(np.dot(nq, v))
            c = float(dq - np.dot(nq, o))
            if abs(a) < 1e-9 and abs(b) < 1e-9:
                if c < -1e-6:
                    return [], []
                continue
            nrm = math.hypot(a, b)
            a, b, c = a / nrm, b / nrm, c / nrm
            edges.append((a, b, c))
            pg = _clip2(pg, a, b, c)
            if len(pg) < 3:
                return [], []
        return pg, edges

    def boundary_faces(self, min_radius=48.0):
        if self.portals is None:
            self.build_portals()
        used = set()
        for p in self.portals:
            kk = (round(float(p.n[0]), 4), round(float(p.n[1]), 4), round(float(p.n[2]), 4),
                  round(p.d, 3))
            used.add((p.a, kk))
            used.add((p.b, (round(-kk[0], 4), round(-kk[1], 4), round(-kk[2], 4), round(-kk[3], 3))))
        out = []
        for i, H in enumerate(self.cells):
            for k in range(len(H)):
                n = H[k, :3]
                d = H[k, 3]
                kk = (round(float(n[0]), 4), round(float(n[1]), 4), round(float(n[2]), 4),
                      round(float(d), 3))
                if (i, kk) in used:
                    continue
                pg, ed = self._cross(i, k)
                if len(pg) < 3:
                    continue
                ar = _area2(pg)
                if ar < 4.0 * min_radius * min_radius:
                    continue
                rr = _inradius(pg, ed)
                if rr < min_radius:
                    continue
                u, v, o = self._frame(n, d)
                cu = sum(q[0] for q in pg) / len(pg)
                cv = sum(q[1] for q in pg) / len(pg)
                ctr = o + u * cu + v * cv
                out.append({'cell': i, 'k': k, 'n': [float(x) for x in n], 'd': float(d),
                            'area': ar, 'radius': rr, 'centre': [float(x) for x in ctr]})
        return out

    def components(self):
        if self.adj is None:
            self.build_portals()
        seen = [-1] * len(self.cells)
        comps = []
        for s in range(len(self.cells)):
            if seen[s] >= 0:
                continue
            ci = len(comps)
            stack = [s]
            seen[s] = ci
            mem = []
            while stack:
                u = stack.pop()
                mem.append(u)
                for (w, _pi) in self.adj[u]:
                    if seen[w] < 0:
                        seen[w] = ci
                        stack.append(w)
            comps.append(mem)
        self.comp_of = seen
        return comps

def pack(ns):
    counts = np.array([len(c) for c in ns.cells], dtype=np.int64)
    flat = np.vstack(ns.cells) if ns.cells else np.zeros((0, 4))
    return (counts, flat, ns.world_lo, ns.world_hi, ns.gridcell, ns.mask,
            ns.n_open_leaves, ns.n_solid_leaves, ns.n_detail_splits)

def unpack(t):
    counts, flat, wlo, whi, gc, mask, nol, nsl, nds = t
    ns = NegSpace.__new__(NegSpace)
    off = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(counts, out=off[1:])
    ns.cells = [flat[off[i]:off[i + 1]] for i in range(len(counts))]
    ns.cell_leaf = [-1] * len(counts)
    ns.world_lo, ns.world_hi = wlo, whi
    ns.gridcell, ns.mask = gc, mask
    ns.n_open_leaves, ns.n_solid_leaves = nol, nsl
    ns.n_detail_splits = nds
    ns.portals = None
    ns.adj = None
    ns._finish()
    ns.schema = NEGSPACE_SCHEMA
    return ns

NONSOLID_SHADERS = frozenset((
    'common/hint', 'common/skip', 'common/areaportal', 'common/nodrawnonsolid',
    'common/origin', 'common/lightgrid', 'common/trigger', 'common/weapclip',
    'common/monsterclip', 'common/botclip', 'common/donotenter',
    'common/clusterportal', 'common/antiportal', 'common/full_clip',
))

def brush_is_solid(br, nonsolid=NONSOLID_SHADERS):
    faces = getattr(br, 'faces', None)
    if faces is not None:
        texs = {f[1] for f in faces}
    elif br and hasattr(br[0], 'p'):
        texs = {f.tex for f in br}
    else:
        return True
    texs = {t.split('textures/')[-1] for t in texs}
    if not texs:
        return True
    return not texs.issubset(nonsolid)

def brush_points(br, off=(0.0, 0.0, 0.0)):
    faces = getattr(br, 'faces', None)
    if faces is not None:
        polys = [list(f[0]) for f in faces]
    elif br and hasattr(br[0], 'p'):
        polys = [list(f.p) for f in br]
    else:
        return []
    return [[p[i] + off[i] for i in range(3)] for poly in polys for p in poly]

def brush_planes(br, off=(0.0, 0.0, 0.0)):
    if isinstance(br, np.ndarray):
        H = br.astype(np.float64)
        return [((float(H[i, 0]), float(H[i, 1]), float(H[i, 2])),
                 float(H[i, 3] + H[i, 0] * off[0] + H[i, 1] * off[1] + H[i, 2] * off[2]))
                for i in range(len(H))]
    faces = getattr(br, 'faces', None)
    if faces is None and br and hasattr(br[0], 'p'):
        out = []
        for f in br:
            p0, p1, p2 = [np.asarray(x, dtype=float) for x in f.p]
            n = np.cross(p1 - p0, p2 - p0)
            L = float(np.linalg.norm(n))
            if L < 1e-9:
                continue
            n = -n / L
            d = float(n @ p0)
            out.append(((float(n[0]), float(n[1]), float(n[2])),
                        d + n[0] * off[0] + n[1] * off[1] + n[2] * off[2]))
        return out
    if faces is not None:
        polys = [list(f[0]) for f in faces]
    else:
        return [((n[0], n[1], n[2]),
                 d + n[0] * off[0] + n[1] * off[1] + n[2] * off[2]) for n, d in br]
    pts = [p for poly in polys for p in poly]
    if not pts:
        return []
    c = [sum(p[i] for p in pts) / float(len(pts)) for i in range(3)]
    out = []
    for poly in polys:
        a, b, cc = poly[0], poly[1], poly[2]
        u = [b[i] - a[i] for i in range(3)]
        v = [cc[i] - a[i] for i in range(3)]
        n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]
        L = math.sqrt(sum(x * x for x in n))
        if L < 1e-9:
            continue
        n = [x / L for x in n]
        d = sum(n[i] * a[i] for i in range(3))
        if sum(n[i] * c[i] for i in range(3)) > d:
            n = [-x for x in n]
            d = -d
        out.append(((n[0], n[1], n[2]),
                    d + n[0] * off[0] + n[1] * off[1] + n[2] * off[2]))
    return out

def from_brushes(tiles, cell=512.0, pad=64.0, mask=MASK_PLAYERSOLID, verbose=False):
    solids = []
    lo = [1e30] * 3
    hi = [-1e30] * 3
    for brushes, off in tiles:
        for br in brushes:
            if not brush_is_solid(br):
                continue
            pl = brush_planes(br, off)
            if len(pl) < 4:
                continue
            H = np.array([[n[0], n[1], n[2], d] for n, d in pl], dtype=np.float64)
            pts = [] if (getattr(br, 'faces', None) is None and br
                         and hasattr(br[0], 'p')) else brush_points(br, off)
            if pts:
                P = np.asarray(pts, dtype=np.float64)
                blo, bhi = P.min(axis=0), P.max(axis=0)
            else:
                blo, bhi = bounds_of(H, np.array([-131072.0] * 3),
                                     np.array([131072.0] * 3))
                if not (np.all(np.isfinite(blo)) and np.all(np.isfinite(bhi))):
                    continue
            solids.append((H, blo, bhi))
            for a2 in range(3):
                lo[a2] = min(lo[a2], float(blo[a2]))
                hi[a2] = max(hi[a2], float(bhi[a2]))
    ns = NegSpace.__new__(NegSpace)
    ns.schema = NEGSPACE_SCHEMA
    ns.mask = mask
    ns.gridcell = float(cell)
    ns.cells = []
    ns.cell_leaf = []
    ns.cell_tile = []
    ns.n_open_leaves = ns.n_solid_leaves = ns.n_detail_splits = 0
    ns.portals = None
    ns.adj = None
    ns.lo = np.zeros((0, 3))
    ns.hi = np.zeros((0, 3))
    ns.grid = {}
    ns.solids = solids
    if not solids:
        ns.world_lo = np.array([-pad] * 3)
        ns.world_hi = np.array([pad] * 3)
        ns.sgrid = {}
        ns._index_solid_bvh('s', np.zeros((0, 3)), np.zeros((0, 3)))
        ns._index_solid_planes(solids)
        return ns
    ns.world_lo = np.array([lo[a2] - pad for a2 in range(3)])
    ns.world_hi = np.array([hi[a2] + pad for a2 in range(3)])
    ns.sgrid = {}
    ns._index_solid_bvh(
        's',
        np.asarray([value[1] for value in solids], dtype=np.float64),
        np.asarray([value[2] for value in solids], dtype=np.float64),
    )
    ns._index_solid_planes(solids)
    if verbose:
        print('negspace(source): %d solid brushes, world %s..%s'
              % (len(solids), [int(x) for x in ns.world_lo], [int(x) for x in ns.world_hi]))
    return ns

def from_bsp(src, mask=MASK_PLAYERSOLID, model=None, cell=512.0, verbose=False):
    d = bytes(src) if isinstance(src, (bytes, bytearray)) else open(src, 'rb').read()
    mo, ml = _lump(d, 7)
    modf = np.frombuffer(d, '<f4', (ml // 40) * 10, mo).reshape(-1, 10)
    world_model = 0 if model is None else int(model)
    world_lo = modf[world_model, 0:3].astype(np.float64) - 64.0
    world_hi = modf[world_model, 3:6].astype(np.float64) + 64.0
    blocks, bounds, brush_mass, patch_triangle_mass = _compiled_collision_solids(
        d, mask, model, world_lo, world_hi,
    )
    ns = NegSpace.__new__(NegSpace)
    ns.mask = mask
    ns.gridcell = float(cell)
    ns.cells = []
    ns.cell_leaf = []
    ns.world_lo = world_lo
    ns.world_hi = world_hi
    ns.n_open_leaves = 0
    ns.n_solid_leaves = 0
    ns.n_detail_splits = 0
    ns.blk_H = blocks
    ns.compiled_brush_mass = brush_mass
    ns.patch_triangle_mass = patch_triangle_mass
    ns.blk_lo = np.asarray([value[0] for value in bounds]) if bounds else np.zeros((0, 3))
    ns.blk_hi = np.asarray([value[1] for value in bounds]) if bounds else np.zeros((0, 3))
    ns._index_blocks()
    ns._finish(verbose)
    ns.schema = NEGSPACE_SCHEMA
    return ns

def save(ns, path):
    cells = ns.cells
    counts = np.array([len(c) for c in cells], dtype=np.int64)
    flat = np.vstack(cells) if cells else np.zeros((0, 4))
    blocks = getattr(ns, 'blk_H', ())
    if isinstance(blocks, _PlaneBlocks):
        block_counts = np.diff(blocks.offsets)
        block_flat = blocks.flat
    else:
        block_counts = np.array([len(block) for block in blocks], dtype=np.int64)
        block_flat = np.vstack(blocks) if blocks else np.zeros((0, 4))
    np.savez_compressed(path, schema=np.array([NEGSPACE_SCHEMA]),
                        compiled_brush_mass=np.array([getattr(ns, 'compiled_brush_mass', 0)]),
                        patch_triangle_mass=np.array([getattr(ns, 'patch_triangle_mass', 0)]),
                        counts=counts, flat=flat,
                        world_lo=ns.world_lo, world_hi=ns.world_hi,
                        gridcell=np.array([ns.gridcell]),
                        mask=np.array([ns.mask]),
                        tile=np.array(getattr(ns, 'cell_tile', [-1] * len(cells)),
                                      dtype=np.int64),
                        block_counts=block_counts, block_flat=block_flat,
                        block_lo=np.asarray(getattr(ns, 'blk_lo', np.zeros((0, 3)))),
                        block_hi=np.asarray(getattr(ns, 'blk_hi', np.zeros((0, 3)))))
    return path if path.endswith('.npz') else path + '.npz'

def load_saved(path):
    z = np.load(path)
    ns = NegSpace.__new__(NegSpace)
    counts = z['counts']
    flat = z['flat']
    off = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(counts, out=off[1:])
    ns.cells = [flat[off[i]:off[i + 1]] for i in range(len(counts))]
    ns._flat = flat
    ns._offsets = off
    ns.cell_leaf = [-1] * len(counts)
    ns.cell_tile = list(z['tile'])
    ns.world_lo = z['world_lo']
    ns.world_hi = z['world_hi']
    ns.gridcell = float(z['gridcell'][0])
    ns.mask = int(z['mask'][0])
    schema = int(z['schema'][0]) if 'schema' in z else 1
    ns.compiled_brush_mass = int(z['compiled_brush_mass'][0]) if 'compiled_brush_mass' in z else 0
    ns.patch_triangle_mass = int(z['patch_triangle_mass'][0]) if 'patch_triangle_mass' in z else 0
    ns.n_open_leaves = 0
    ns.n_solid_leaves = 0
    ns.n_detail_splits = 0
    ns.portals = None
    ns.adj = None
    if 'block_counts' in z:
        block_counts = z['block_counts']
        block_flat = z['block_flat']
        block_offsets = np.r_[0, np.cumsum(block_counts)]
        ns.blk_H = _PlaneBlocks(block_flat, block_offsets)
        ns.blk_lo = z['block_lo']
        ns.blk_hi = z['block_hi']
        ns._index_blocks()
    ns._finish()
    ns.schema = schema
    z.close()
    return ns

if __name__ == '__main__':
    import sys
    import time
    t0 = time.time()
    ns = NegSpace(sys.argv[1], verbose=True)
    print('negspace: computed in %.1fs' % (time.time() - t0))
    t0 = time.time()
    ns.build_portals(verbose=True)
    print('negspace: portals in %.1fs' % (time.time() - t0))
    cs = ns.components()
    print('negspace: %d components, largest %d cells' % (len(cs), max(len(c) for c in cs)))
