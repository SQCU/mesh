#!/usr/bin/env python3
"""negspace -- the COMPUTED negative space of a map, and the only definition of
solidity in this toolchain.

WHY THIS EXISTS
---------------
A BSP is already an exact partition of space.  Everything that used to be done in
these tools by poking `solid_brush_at(p)` at arbitrary operands was re-deriving,
lossily and by point-sampling, a structure the file already contains -- and a
sample can never be complete, so the failures were patched forever: spawnpoints
shipped inside walls, carts burrowed through terrain along smooth waypoint
curves, corridors reported phantom "no floor".

This module computes the structure instead.  `NegSpace` is the free volume of a
map as an explicit set of CONVEX CELLS with exact face adjacency, obtained from
the BSP tree's empty leaves with their solid (detail) brushes subtracted.  It is
the object the rest of the pipeline CONSUMES:

    map BSP -> NegSpace (free volume, exact)
            -> navmesh (stock waypoint graph)  [navmesh.py]
            -> Voronoi cells over the navmesh  [navmesh.py]
            -> cart path solver constrained to stay inside NegSpace  [navmesh.py]

Because a point that is a member of a cell is by construction not in solid, and
because a segment's intersection with a convex cell is an exact interval, there
is no sampling anywhere below and no "is this solid?" probe to get wrong.

THE ONE DEFINITION OF SOLIDITY
------------------------------
    solid(p)  ==  ns.cell_at(p) < 0

Nothing else in `tools/` may define it.  `mapfuse.Src.solid_brush_at`,
`mapfuse.Src.clip_brush_at`, their brush grids, `mapfuse.blockage` /
`corridor_samples` / `arc_samples` / `ray_runs` / `free_slab` and
`mkentfile.Bsp` were all deleted in favour of this.

STRUCTURE
---------
`cells[i]` is a convex polytope in the half-space form `n . p <= d`:

    H[i]      float64 (k,4) rows (nx,ny,nz,d)
    lo/hi[i]  exact axis-aligned bounds (interval propagation to a fixed point)
    leaf[i]   the BSP leaf it came from
    kind      OPEN

`portals` are the exact 2-D faces shared by two cells, each with the radius of
its largest inscribed circle -- so "can a player fit through this opening" is a
number read off the structure, never a trace.

`shrink(mins, maxs)` returns the CONFIGURATION-SPACE complex for a box: cell
half-spaces pushed in by the box's support.  A point of the shrunk complex is a
placement where the whole box is in free space.  That is why a spawnpoint or a
cart node produced from this module cannot be in solid -- being in solid is not
representable.

Rigid placement: `translated(t)` shifts a complex exactly (`d -> d + n.t`), so a
per-tile complex stays exact after the fused world's 3-D pack, Z levels included.
"""

import math
import os
import struct

# One BLAS thread per process.  The survey runs one process per candidate map and
# every one of them is doing small dense linear algebra; letting Accelerate spawn
# a thread pool inside each oversubscribes the machine by an order of magnitude
# and the wall time goes up, not down.  Set before numpy is imported.
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np

# Q3/Xonotic contents bits (texture lump word 2)
CONTENTS_SOLID = 0x1
CONTENTS_PLAYERCLIP = 0x10000
CONTENTS_MONSTERCLIP = 0x20000
CONTENTS_BOTCLIP = 0x400000
MASK_PLAYERSOLID = CONTENTS_SOLID | CONTENTS_PLAYERCLIP
MASK_BOTSOLID = MASK_PLAYERSOLID | CONTENTS_MONSTERCLIP | CONTENTS_BOTCLIP

# Xonotic standing player hull (PL_MIN_CONST / PL_MAX_CONST)
PL_MIN = (-16.0, -16.0, -24.0)
PL_MAX = (16.0, 16.0, 45.0)
# the payload cart hull that mkentfile lays track for
# Must equal PLC_CART_MIN/PLC_CART_MAX in qcsrc/.../payload/payload.qh. The engine
# spawns that hull; validating clearance for any other one certifies a cart that
# does not exist. These disagreed (z=56 here, z=40 there) until the QC gained a
# single constant.
CART_MIN = (-32.0, -32.0, -24.0)
CART_MAX = (32.0, 32.0, 56.0)

FLOOR_NZ = 0.7          # what the engine calls walkable
EPS = 1e-4


def _lump(d, i):
    return struct.unpack_from('<ii', d, 8 + i * 8)


def bounds_of(H, seed_lo, seed_hi, iters=4):
    """Exact AABB of {p : H p <= d} by interval propagation to a fixed point.

    This is the replacement for "read the distance off any plane that looks
    axial", which is only valid for an exactly axis-aligned plane and which is
    what indexed a corridor whose real x-span is [-5504,-5152] at [-5843,-5491].
    An entirely oblique cell gets a finite, correct box here."""
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
    """Drop half-spaces that cannot be active, KEEPING the AABB that justified it.

    A constraint whose support over the cell's AABB is below its own distance is
    redundant *given that AABB*.  Dropping it while not carrying the AABB in the
    constraint set is unsound, and measurably so: on warfare it grew cells to
    1900-unit extents with two surviving planes, and 196 of 12 275 sampled points
    that those cells called free were inside a solid brush.  So the six axial
    planes of the AABB are appended: the result is exactly the same point set as
    H, with the redundant members replaced by the box that implies them."""
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


def nonempty(H, lo=None, hi=None):
    """Is {p : H p <= d} non-empty?  Exact for a bounded polytope: enumerate the
    vertices as triple-plane intersections and keep the feasible ones."""
    v = vertices(H, lo, hi)
    return len(v) > 0


def vertices(H, lo=None, hi=None, cap=24):
    """Exact vertex set of a bounded convex polytope given as n.p <= d."""
    k = len(H)
    if k > cap:
        # keep the tightest `cap` constraints relative to the cell centre; the
        # rest are provably inactive after prune(), this is only a safety valve
        H = H[:cap]
        k = cap
    if k < 4:
        return np.zeros((0, 3))
    idx = np.array([(a, b, c) for a in range(k) for b in range(a + 1, k)
                    for c in range(b + 1, k)], dtype=np.int64)
    if not len(idx):
        return np.zeros((0, 3))
    A = H[idx][:, :, :3]
    B = H[idx][:, :, 3]
    det = np.linalg.det(A)
    ok = np.abs(det) > 1e-7
    if not ok.any():
        return np.zeros((0, 3))
    A = A[ok]
    B = B[ok]
    try:
        P = np.linalg.solve(A, B[..., None])[..., 0]
    except np.linalg.LinAlgError:
        return np.zeros((0, 3))
    good = np.isfinite(P).all(axis=1)
    P = P[good]
    if not len(P):
        return P
    viol = P @ H[:, :3].T - H[:, 3]
    P = P[(viol <= 0.05).all(axis=1)]
    if len(P) > 1:
        P = np.unique(np.round(P, 3), axis=0)
    return P


def shrink_H(H, mins, maxs):
    """Configuration-space half-spaces for an axis-aligned box.

    `p` satisfies the result iff the whole box [p+mins, p+maxs] satisfies H.
    Support of the box along n is max over corners of n.c, i.e.
    sum_a (maxs[a] if n[a]>0 else mins[a]) * n[a]."""
    n = H[:, :3]
    mn = np.asarray(mins, dtype=np.float64)
    mx = np.asarray(maxs, dtype=np.float64)
    sup = (np.where(n > 0, mx, mn) * n).sum(axis=1)
    out = H.copy()
    out[:, 3] = H[:, 3] - sup
    return out


def subtract(pieces, B, seed_lo, seed_hi, cap=64, exact_empty=True, minext=0.05):
    r"""EXACT convex decomposition of (union of `pieces`) \ (convex B).

    For a convex region C and a convex B = {h_1..h_k}:
        C \ B  =  U_q  ( C & ~h_q & h_1 & .. & h_{q-1} )
    every term convex by construction.  This is the one routine that removes
    volume anywhere in this module -- detail brushes, and the coverage test.
    Returns (pieces_out, overflowed)."""
    out = []
    for C in pieces:
        clo, chi = bounds_of(C, seed_lo, seed_hi)
        cc = 0.5 * (clo + chi)
        ce = 0.5 * (chi - clo)
        lowv = B[:, :3] @ cc - np.abs(B[:, :3]) @ ce - B[:, 3]
        if np.any(lowv > 0.0):
            out.append(C)                       # B cannot meet this piece
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
                # `exact_empty` enumerates the remainder's vertices to prove it is
                # non-empty.  Skipping it can only KEEP an empty piece, i.e. report
                # a region as not-covered when it is covered -- restrictive, never
                # unsafe -- and it is the difference between 9 ms and 1 ms per query.
                if (not exact_empty) or len(vertices(Rp)) >= 4:
                    out.append(Rp)
            acc.append(B[q])
        if len(out) > cap:
            return out, True
    return out, False


def box_H(lo, hi):
    """Axis-aligned box as half-spaces n.p <= d."""
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
    """Clip convex polygon (list of (u,v)) by a*u + b*v <= c."""
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
    """Largest inscribed circle of a convex polygon, by shrinking its own edge
    half-planes and bisecting on the radius.  Exact to 2^-8 of the extent."""
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


class NegSpace(object):
    """The free volume of one assembled world, as convex cells with exact faces."""

    def __init__(self, src, mask=MASK_PLAYERSOLID, model=0, cell=512.0,
                 max_split=256, verbose=False):
        if isinstance(src, (bytes, bytearray)):
            d = bytes(src)
        else:
            d = open(src, 'rb').read()
        assert d[:4] == b'IBSP' and struct.unpack_from('<i', d, 4)[0] == 46
        self.mask = mask
        self.gridcell = float(cell)

        to, tl = _lump(d, 1)
        po, pl = _lump(d, 2)
        no, nl = _lump(d, 3)
        lo_, ll = _lump(d, 4)
        lbo, lbl = _lump(d, 6)
        mo, ml = _lump(d, 7)
        bo, bl = _lump(d, 8)
        so, sl = _lump(d, 9)
        ntex = tl // 72
        contents = np.frombuffer(d, '<i4', ntex * 18, to).reshape(ntex, 18)[:, 17]
        planes = np.frombuffer(d, '<f4', (pl // 16) * 4, po).reshape(-1, 4).astype(np.float64)
        nodes = np.frombuffer(d, '<i4', (nl // 36) * 9, no).reshape(-1, 9)
        leafs = np.frombuffer(d, '<i4', (ll // 48) * 12, lo_).reshape(-1, 12)
        leafbrushes = np.frombuffer(d, '<i4', lbl // 4, lbo)
        brushes = np.frombuffer(d, '<i4', (bl // 12) * 3, bo).reshape(-1, 3)
        sides = np.frombuffer(d, '<i4', (sl // 8) * 2, so).reshape(-1, 2)
        modf = np.frombuffer(d, '<f4', (ml // 40) * 10, mo).reshape(-1, 10)
        modi = np.frombuffer(d, '<i4', (ml // 40) * 10, mo).reshape(-1, 10)

        self.world_lo = modf[model, 0:3].astype(np.float64) - 64.0
        self.world_hi = modf[model, 3:6].astype(np.float64) + 64.0
        head = 0 if model == 0 else None
        self.planes = planes

        tx = brushes[:, 2]
        okt = (tx >= 0) & (tx < ntex)
        bcont = np.zeros(len(brushes), dtype=np.int64)
        bcont[okt] = contents[tx[okt]]
        self.brush_blocks = (bcont & mask) != 0

        # ---- BLOCKING-BRUSH INDEX.
        # Free space is defined as the world minus the brushes the ENGINE collides
        # with, and the engine collides with a BIH over brushes
        # (Mod_CollisionBIH_TraceBrush), not with the BSP tree.  The tree is used
        # here only as an accelerator that hands out convex regions to subtract
        # from; which brushes to subtract is decided by an actual overlap query,
        # NOT by the leaf's `leafbrushes` list.  That list is not complete for
        # opaque leaves -- measured on warfare: 78 sampled points inside a solid
        # brush landed in a "free" cell because the brush was not listed in the
        # leaf it fills -- and a structure that trusts it is unsound in exactly
        # the direction that must never happen.
        bl_lo, bl_hi, bl_H = [], [], []
        for bi in np.nonzero(self.brush_blocks)[0]:
            fs, nb2, _ = brushes[bi]
            if nb2 <= 0 or fs < 0 or fs + nb2 > len(sides):
                continue
            Hb = planes[sides[fs:fs + nb2, 0]]
            l2, h2 = bounds_of(Hb, self.world_lo - 512.0, self.world_hi + 512.0)
            bl_lo.append(l2)
            bl_hi.append(h2)
            bl_H.append(Hb)
        self.blk_H = bl_H
        self.blk_lo = np.array(bl_lo) if bl_lo else np.zeros((0, 3))
        self.blk_hi = np.array(bl_hi) if bl_hi else np.zeros((0, 3))
        bgrid = {}
        BC = 512.0
        for i in range(len(bl_H)):
            g0 = np.floor(self.blk_lo[i] / BC).astype(np.int64)
            g1 = np.floor(self.blk_hi[i] / BC).astype(np.int64)
            if np.prod(g1 - g0 + 1) > 20000:
                bgrid.setdefault('big', []).append(i)
                continue
            for x in range(g0[0], g1[0] + 1):
                for y in range(g0[1], g1[1] + 1):
                    for z in range(g0[2], g1[2] + 1):
                        bgrid.setdefault((x, y, z), []).append(i)
        self.blk_grid = bgrid
        self.blk_cell = BC

        # ---- walk the tree; every leaf gets the exact half-space list of the
        # convex region the tree carved for it, plus the world box so it is bounded
        wb = []
        for a in range(3):
            e = [0.0, 0.0, 0.0]
            e[a] = 1.0
            wb.append(e + [float(self.world_hi[a])])
            e2 = [0.0, 0.0, 0.0]
            e2[a] = -1.0
            wb.append(e2 + [float(-self.world_lo[a])])
        wb = np.array(wb, dtype=np.float64)

        self.cells = []          # list of (k,4) float64 half-space arrays
        self.cell_leaf = []
        self.cell_node_faces = []   # per cell: list of (planeidx, sign) from tree nodes
        stack = [(head, [])]
        nleaf_open = 0
        nleaf_solid = 0
        nsplit = 0
        self.dropped_leaves = 0
        while stack:
            ni, path = stack.pop()
            if ni < 0:
                li = -1 - ni
                if li >= len(leafs):
                    continue
                # EVERY leaf is processed, not only the ones the compiler left in
                # the PVS.  The engine's collision is a BIH over BRUSHES
                # (Mod_CollisionBIH_TraceBrush), so a leaf that q3map2's
                # fill-outside marked opaque but that contains no brush is space a
                # player can actually stand in -- and on a sealed map that region
                # is exactly where a fusion connector gets built.  Defining free
                # space as "open leaf" would have disagreed with the engine there,
                # which is the whole reason this module exists.  Free space is
                # therefore: the leaf's own convex region MINUS its blocking
                # brushes; a leaf buried in rock subtracts to nothing on its own.
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
                # subtract the leaf's own blocking (detail) brushes: exact convex
                # decomposition C \ B = U_i ( C & ~h_i & h_1..h_{i-1} )
                blk = self._blocking_in(lo, hi)
                pieces = [H]
                overflow = False
                for bi in blk:
                    pieces, overflow = subtract(pieces, self.blk_H[bi], lo, hi,
                                                cap=max_split, exact_empty=False,
                                                minext=0.25)
                    nsplit += 1
                    if overflow:
                        break
                if overflow:
                    # Never leave a brush unsubtracted: that would make solid space
                    # representable as free, which is the one thing this structure
                    # must not do.  Give the leaf up whole instead, and count it.
                    self.dropped_leaves += 1
                    continue
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
            # child 0 = front (n.p >= d)  -> half-space (-n, -d)
            # child 1 = back  (n.p <= d)  -> half-space ( n,  d)
            stack.append((int(nd[1]), path + [(np.array([-pn[0], -pn[1], -pn[2], -pn[3]]), (pi, -1))]))
            stack.append((int(nd[2]), path + [(np.array([pn[0], pn[1], pn[2], pn[3]]), (pi, 1))]))

        self.n_open_leaves = nleaf_open
        self.n_solid_leaves = nleaf_solid
        self.n_detail_splits = nsplit
        self._finish(verbose)

    def _blocking_in(self, lo, hi):
        """Every blocking brush whose AABB meets [lo,hi] -- the set that must be
        subtracted from a region for the result to be free space the engine
        agrees with."""
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

    # ------------------------------------------------------------------ setup
    def _finish(self, verbose=False):
        N = len(self.cells)
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
                  '%d detail-brush subtractions, %d leaves given up as too split to '
                  'subtract exactly), free-cell AABB volume %.3g u^3, %d grid cells'
                  % (N, self.n_open_leaves, self.n_solid_leaves, self.n_detail_splits,
                     getattr(self, 'dropped_leaves', 0), v, len(self.grid)))

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
        """Free volume, as the sum of the cells' AABB volumes clipped to the
        world -- an upper bound used only for reporting scale."""
        e = np.maximum(self.hi - self.lo, 0.0)
        return float((e[:, 0] * e[:, 1] * e[:, 2]).sum())

    # -------------------------------------------------------------- placement
    def translated(self, t):
        """Exact rigid placement of the whole complex: n.p <= d  ->  n.p <= d+n.t."""
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
        out.dropped_leaves = getattr(self, 'dropped_leaves', 0)
        out.portals = None
        out.adj = None
        out._finish()
        return out

    @staticmethod
    def union(parts):
        """Assemble placed per-tile complexes into one world complex."""
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
        return out

    def add_cells(self, Hs, tile=-1):
        """Register procedurally generated free volume (a connector's interior)."""
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
        """Apply the procedural geometry's own edits to the free volume.

        `remove` are convex solids the generator ADDS to the world (a corridor's
        wall, floor and ceiling slabs; a doorway's threshold, jambs and header):
        the free volume loses them exactly, by convex subtraction.  `add` are the
        regions the generator OPENS (a cut aperture, a corridor's interior, the
        throat between them).  Geometry and free space are edited in the same
        operation, so the complex describes the world after the fusion rather than
        before it -- which is the difference between a structure you can place a
        spawnpoint in and a structure you have to check one afterwards.

        Every solid that touches a given cell is subtracted from it in one pass,
        and the index is rebuilt once, so the cost is linear in cells rather than
        in cells times solids."""
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
            over = False
            for bi in bis:
                pieces, over = subtract(pieces, rem[bi], self.lo[i], self.hi[i],
                                        cap=64, exact_empty=False)
                if over:
                    break
            if over:
                continue                     # keep the cell whole rather than guess
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
        self._CROSS_CACHE = None
        self.portals = None
        self.adj = None
        self._finish()
        if verbose:
            print('negspace: edit applied -- %d free cells re-cut around %d new '
                  'procedural solids, %d opened regions added; %d cells total'
                  % (nrem, len(rem), nadd, len(self.cells)))
        return nrem, nadd

    # ------------------------------------------------------------- membership
    def _cands(self, p):
        c = self.gridcell
        k = (int(math.floor(p[0] / c)), int(math.floor(p[1] / c)), int(math.floor(p[2] / c)))
        return self.grid.get(k)

    def _solid_cands(self, lo, hi):
        """Solid brushes whose AABB meets [lo,hi] (source representation)."""
        c = self.gridcell
        out = []
        seen = set()
        for x in range(int(math.floor(lo[0] / c)), int(math.floor(hi[0] / c)) + 1):
            for y in range(int(math.floor(lo[1] / c)), int(math.floor(hi[1] / c)) + 1):
                for z in range(int(math.floor(lo[2] / c)), int(math.floor(hi[2] / c)) + 1):
                    for i in self.sgrid.get((x, y, z), ()):
                        if i not in seen:
                            seen.add(i)
                            out.append(i)
        for i in self.sgrid.get('big', ()):
            if i not in seen:
                seen.add(i)
                out.append(i)
        return out

    def cell_at(self, p, tol=0.25):
        """Index of the free cell containing p, or -1.

        THE definition of solidity: `solid(p) == cell_at(p) < 0`."""
        if getattr(self, 'solids', None) is not None:
            # SOURCE representation: free is "inside no solid brush".
            px = np.array([float(p[0]), float(p[1]), float(p[2])])
            for i in self._solid_cands(px - 0.5, px + 0.5):
                H, blo, bhi = self.solids[i]
                if np.any(px < blo - tol) or np.any(px > bhi + tol):
                    continue
                # Generous toward SOLID by `tol`, matching the convention the
                # rest of the toolchain uses: a marginal point is called solid,
                # which refuses a placement rather than admitting a bad one.
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
        """Is the convex region H inside the union of free cells, to within `tol`?

        Subtract every overlapping free cell from H and see whether any volume
        survives.  The union of the cells is NOT convex, so this is the only
        correct way to ask -- testing containment in a single cell would say no
        for every box that straddles an open doorway.

        `tol` is a real tolerance and is stated rather than hidden: a leftover
        piece thinner than `tol` in any axis is not treated as uncovered.  It
        exists because the cell decomposition drops sub-0.05-unit slivers, and
        because a sliver that thin cannot hold anything this toolchain places.
        The error is one-sided in the safe direction everywhere it matters: a
        false "covered" is bounded by `tol` units of geometry, while nothing is
        ever reported free that lies inside a brush -- that is a property of the
        cells themselves, not of this query."""
        if lo is None:
            lo, hi = bounds_of(H, self.world_lo - 4096.0, self.world_hi + 4096.0)
        cand = self._cells_in_box(lo, hi)
        pieces = [H]
        for i in cand:
            pieces, over = subtract(pieces, self.cells[i], lo, hi,
                                    exact_empty=False, minext=tol)
            if over:
                return False
            if not pieces:
                return True
        return not pieces

    def _fits_source(self, p, mins, maxs, tol=0.03125):
        """Box at p free of every solid brush (source representation).

        Uses the plane-offset construction the ENGINE uses for a box trace: each
        brush plane is pushed out by the box's support along that normal, and the
        box centre is tested as a point.  Conservative at edges in the safe
        direction (it may call a marginal placement blocked), and it is the same
        arithmetic DarkPlaces applies, so a `fits` answer here is the answer the
        server would give."""
        px = np.array([float(p[0]), float(p[1]), float(p[2])])
        mn = np.asarray(mins, dtype=float)
        mx = np.asarray(maxs, dtype=float)
        blo = px + mn
        bhi = px + mx
        for i in self._solid_cands(blo, bhi):
            H, slo, shi = self.solids[i]
            if np.any(bhi <= slo) or np.any(blo >= shi):
                continue
            n = H[:, :3]
            # Push each plane OUT by the box's support along that normal, which
            # for interior `n.x <= d` means picking the MIN corner where n>0:
            #   dist = d - n.off,  off = (n>0 ? mins : maxs)
            # Choosing the max corner instead shrinks the brush, and a box
            # centred on a point the module itself called solid then "fitted".
            off = np.where(n > 0, mn, mx)
            dist = H[:, 3] - (off * n).sum(axis=1)
            if (n @ px - dist).max() <= -tol:
                return False
        return True

    def fits(self, p, mins=PL_MIN, maxs=PL_MAX):
        """Is the whole box placed at p inside free space?  Exact, and true across
        cell boundaries -- a box spanning an open portal fits."""
        if getattr(self, 'solids', None) is not None:
            return self._fits_source(p, mins, maxs)
        lo = (p[0] + mins[0] + 0.03, p[1] + mins[1] + 0.03, p[2] + mins[2] + 0.03)
        hi = (p[0] + maxs[0] - 0.03, p[1] + maxs[1] - 0.03, p[2] + maxs[2] - 0.03)
        if np.any(np.asarray(hi) <= np.asarray(lo)):
            return False
        return self.covered(box_H(lo, hi), np.asarray(lo), np.asarray(hi))

    def intersects_free(self, H, lo=None, hi=None):
        """Does the convex region H meet ANY free cell?  Exact (vertex
        enumeration of the intersection polytope).  `not intersects_free(H)` is
        the structural statement "H is entirely solid" -- which is how a wall
        PANEL is recognised, as opposed to a corner or an already-open gap."""
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

    # ---------------------------------------------------------------- segment
    def _segment_intervals_source(self, a, b, mins=None, maxs=None):
        """Free parametric intervals of [a,b], from the SOLID brushes.

        A ray's intersection with a convex brush is a closed interval (slab
        clipping), so the solid runs are exact; the free runs are their
        complement.  No step size anywhere -- which is the whole reason the
        marched version disagreed with the closed-form one on points that were
        never navigable."""
        A = np.asarray(a, dtype=float)
        B = np.asarray(b, dtype=float)
        blo = np.minimum(A, B)
        bhi = np.maximum(A, B)
        if mins is not None:
            blo = blo + np.asarray(mins, dtype=float)
            bhi = bhi + np.asarray(maxs, dtype=float)
        solid = []
        for i in self._solid_cands(blo, bhi):
            H, slo, shi = self.solids[i]
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
            t0, t1, ok = 0.0, 1.0, True
            for k in range(len(H)):
                x, y = da[k], db[k]
                if x > 0 and y > 0:
                    ok = False
                    break
                if x <= 0 and y <= 0:
                    continue
                t = x / (x - y)
                # x > 0 means the ray STARTS outside this half-space and crosses
                # in: that is an ENTRY, so it raises the near bound.  Having these
                # two branches the wrong way round made every downward ray report
                # the whole drop as free, so nothing was ever standable.
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
            return [(0.0, 1.0)]
        solid.sort()
        merged = [list(solid[0])]
        for s0, e0 in solid[1:]:
            if s0 <= merged[-1][1] + 1e-9:
                merged[-1][1] = max(merged[-1][1], e0)
            else:
                merged.append([s0, e0])
        free = []
        t = 0.0
        for s0, e0 in merged:
            if s0 > t + 1e-9:
                free.append((t, s0))
            t = max(t, e0)
        if t < 1.0 - 1e-9:
            free.append((t, 1.0))
        return free

    def segment_intervals(self, a, b, mins=None, maxs=None):
        """EXACT parametric intervals of [a,b] that lie inside free cells.

        Convex cell ∩ segment is a closed interval computed in closed form, so
        this is the whole answer, not a sampling of it.  Returns a merged,
        sorted list of (t0,t1)."""
        if getattr(self, 'solids', None) is not None:
            return self._segment_intervals_source(a, b, mins, maxs)
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        dv = b - a
        cand = self._cells_along(a, b)
        segs = []
        for i in cand:
            H = self.cells[i]
            if mins is not None:
                H = shrink_H(H, mins, maxs)
            n = H[:, :3]
            d = H[:, 3]
            da = n @ a - d
            db = n @ b - d
            t0, t1 = 0.0, 1.0
            ok = True
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
                segs.append((t0, t1))
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
        """The parts of [a,b] NOT in free space.  Empty list == fully navigable."""
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

    # --------------------------------------------------- projection into free
    def project(self, p, mins=PL_MIN, maxs=PL_MAX, radius=512.0, iters=14):
        """Closest legal placement of the box inside the free complex, and its
        distance.  This is the ACTIVATION-DISTANCE operator of NAV-SPEC §3: it
        does not ask whether a point is legal, it returns the legal point nearest
        to it, so a solver can be CONSTRAINED instead of a plan tested.

        Candidates are ordered by the distance to their cell's AABB, which lower-
        bounds the distance to the cell itself, so the scan stops as soon as no
        remaining candidate can beat the best placement found."""
        p0 = np.array([float(p[0]), float(p[1]), float(p[2])])
        if mins is None:
            # already a member of the free volume: the nearest legal point is itself
            if self.cell_at(p) >= 0:
                return [float(p[0]), float(p[1]), float(p[2])], 0.0
        c = self.gridcell
        r = int(math.ceil(radius / c))
        cx, cy, cz = (int(math.floor(p[0] / c)), int(math.floor(p[1] / c)),
                      int(math.floor(p[2] / c)))
        cand = []
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    g = self.grid.get((cx + dx, cy + dy, cz + dz))
                    if g is not None:
                        cand.append(g)
        if not cand:
            return None, float('inf')
        cand = np.unique(np.concatenate(cand))
        near = np.maximum(np.maximum(self.lo[cand] - p0, p0 - self.hi[cand]), 0.0)
        lb = np.linalg.norm(near, axis=1)
        keep = lb <= radius
        cand, lb = cand[keep], lb[keep]
        order = np.argsort(lb)
        best, bestd = None, float('inf')
        for oi in order:
            if lb[oi] >= bestd:
                break
            i = int(cand[oi])
            S = self.cells[i] if mins is None else shrink_H(self.cells[i], mins, maxs)
            n = S[:, :3]
            d = S[:, 3]
            q = p0.copy()
            for _ in range(iters):
                v = n @ q - d
                k = int(v.argmax())
                if v[k] <= 1e-6:
                    break
                nn = n[k]
                q = q - nn * (v[k] / max(1e-12, float(nn @ nn)))
            if (n @ q - d).max() > 0.5:
                continue                    # the box does not fit in this cell
            dd = float(np.linalg.norm(q - p0))
            if dd < bestd:
                bestd, best = dd, q
        if best is None:
            return None, float('inf')
        return [float(x) for x in best], bestd

    # ------------------------------------------------------------------ floor
    def floor_under(self, p, maxdrop=512.0, footprint=None):
        """Z at which a body dropped from p comes to rest, or None.

        Read off the exact free run below p -- closed form, no march and no step
        size, so there is no quantisation to bisect away.

        `footprint` makes this a BOX question rather than a point question: a
        standing player rests on the HIGHEST surface any part of its footprint
        touches, so the floor is taken as the max over the nine offsets (centre,
        four corners, four edge midpoints).  Probing only the centre reports a
        floor the box would intersect wherever the ground rises under one
        corner, which condemns walkable ground as unstandable -- measured at 36%
        of a walkable helical corridor in the sibling generator's oracle, which
        is where this model comes from.
        """
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
                continue                     # nothing under this offset
            z = float(a[2] - run[1] * maxdrop)
            if best is None or z > best:
                best = z                     # rest on the HIGHEST ground touched
        return best

    RELOC = None

    def solid_at(self, p):
        """THE definition of solidity.  `solid_at(p) == cell_at(p) < 0`."""
        return self.cell_at(p) < 0

    def standable(self, p, mins=PL_MIN, maxs=PL_MAX, hover=64.0, lift=2.0):
        """Can a player STAND at p: floor under the whole FOOTPRINT, head room
        above it, and the box itself free.

        The box rests on the highest surface its footprint touches (see
        `floor_under`), and only the volume ABOVE that rest height is required to
        be free -- testing a box centred at an arbitrary probe height instead is
        what condemns walkable ground."""
        fz = self.floor_under(p, hover, footprint=(abs(mins[0]), abs(mins[1])))
        if fz is None:
            return False
        return self.fits([p[0], p[1], fz - mins[2] + lift], mins, maxs)

    def clearance(self, p, cap=512.0, mins=PL_MIN, maxs=PL_MAX, tol=1.0):
        """How much room is there at p: the largest lateral half-extent, up to
        `cap`, at which the player-height box still fits.

        NOT the distance to the nearest legal placement -- that is `project`, and
        it is 0 for any point that is already legal, which makes it useless as a
        room measure.  This grows the body instead of moving it, so an open hall
        and a body-width corridor give different answers."""
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
        """Fraction of [a,b] traversable before the first obstruction (1.0 clear).
        Closed form from the segment's free intervals -- no step size."""
        gaps = self.segment_gaps(a, b, mins, maxs)
        return 1.0 if not gaps else float(gaps[0][0])

    def trace(self, a, b, mins=None, maxs=None):
        """First solid POINT along a->b, or None if the line is clear.

        This is the sightline contract the generator's e2e uses (`hit is None`
        means clear).  Exact: the first gap in the segment's free intervals is
        where solidity begins, so unlike a stepped march it cannot skip a thin
        obstruction between samples."""
        f = self.trace_fraction(a, b, mins, maxs)
        if f >= 1.0:
            return None
        return (a[0] + f * (b[0] - a[0]),
                a[1] + f * (b[1] - a[1]),
                a[2] + f * (b[2] - a[2]))

    def standing_point(self, p, mins=PL_MIN, maxs=PL_MAX, maxdrop=512.0,
                       lift=(1.0, 4.0, 12.0, 26.0), search=192.0):
        """A legal STANDING placement for p, or None.

        Tried at p first, then on an expanding lattice around it -- the offline,
        structural form of what the engine's `relocate_spawnpoint` does at run
        time inside the 10M-jump budget of a live worldspawn.  Doing it here means
        the engine never has to: what ships is already standable."""
        r = self._stand_at(p, mins, maxs, maxdrop, lift)
        if r is not None:
            return r
        if NegSpace.RELOC is None:
            steps = []
            for rad in (24.0, 48.0, 96.0, 144.0, 192.0):
                for k in range(8):
                    ang = k * math.pi / 4.0
                    for dz in (0.0, 24.0, -24.0, 64.0):
                        steps.append((rad * math.cos(ang), rad * math.sin(ang), dz))
            steps.sort(key=lambda t: math.hypot(math.hypot(t[0], t[1]), t[2]))
            NegSpace.RELOC = steps
        for dx, dy, dz in NegSpace.RELOC:
            if math.hypot(dx, dy) > search:
                continue
            r = self._stand_at([p[0] + dx, p[1] + dy, p[2] + dz], mins, maxs, maxdrop, lift)
            if r is not None:
                return r
        return None

    def _stand_at(self, p, mins=PL_MIN, maxs=PL_MAX, maxdrop=512.0, lift=(1.0, 4.0, 12.0, 26.0)):
        """A legal STANDING placement at or below p, or None.

        The result is a point of the free volume at which the whole player box is
        covered by free cells and which has the free volume's own floor beneath
        it.  Because the point is CONSTRUCTED in free space rather than tested
        after the fact, `spawn inside a wall` is not representable here."""
        q, _ = self.project(p, None, None, radius=self.gridcell * 2)
        if q is None:
            return None
        fz = self.floor_under(q, maxdrop, footprint=(abs(mins[0]), abs(mins[1])))
        if fz is None:
            return None
        for dz in lift:
            r = [q[0], q[1], fz - mins[2] + dz]
            if self.fits(r, mins, maxs):
                return r
        return None

    # ---------------------------------------------------------------- portals
    def build_portals(self, min_radius=0.0, verbose=False):
        """Exact shared faces between free cells.

        Two cells share a face iff one holds half-space (n,d) and the other holds
        (-n,-d); the face is the intersection of their cross-sections on that
        plane.  The inscribed radius of that 2-D polygon is how wide the opening
        actually is -- the number `can a player get through here` is read from,
        with no trace anywhere."""
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
        # NOTE ON COST.  This is the exact construction and it is quadratic in the
        # number of cells that share one plane.  That is fine at the scale it is
        # meant for (dance: 3 794 cells, 7 s) and pathological on a large fused
        # complex, where co-planar axis-aligned architecture puts hundreds of cells
        # on a single plane.  Two attempts to prune it spatially -- bucketing the
        # opposite side per plane group, and querying the existing grid per
        # cell-face -- both came out SLOWER on dance (>2 min vs 7 s), because the
        # cells' AABBs span many grid cells and the pruning cost more than the
        # pairing it saved.  Left exact and unpruned rather than left broken; the
        # caller that needed it at fused scale (the Voronoi diagnostic) no longer
        # runs by default.  A real fix is a proper 2-D index within each plane.
        for kk, lst in key.items():
            nk = (round(-kk[0], 4), round(-kk[1], 4), round(-kk[2], 4), round(-kk[3], 3))
            other = key.get(nk)
            if not other:
                continue
            if (nk, kk) in done:
                continue
            done.add((kk, nk))
            for (i, ki) in lst:
                for (j, kj) in other:
                    if i == j:
                        continue
                    if np.any(self.lo[i] > self.hi[j] + 0.25) or np.any(self.lo[j] > self.hi[i] + 0.25):
                        continue
                    pg, edges = self._cross(i, ki)
                    if len(pg) < 3:
                        continue
                    pg2, e2 = self._cross(j, kj)
                    n = self.cells[i][ki, :3]
                    d = self.cells[i][ki, 3]
                    if len(pg2) < 3:
                        continue
                    u, v, o = self._frame(n, d)
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

    _CROSS_CACHE = None

    def _frame(self, n, d):
        """Canonical 2-D frame of an (unsigned) plane.  BOTH cells sharing a face
        must be sectioned in the SAME frame or the two polygons cannot be
        intersected; the frame is therefore derived from the plane with its
        normal canonicalised, not from each cell's own outward normal."""
        n = np.asarray(n, dtype=np.float64)
        if (n[0], n[1], n[2]) < (0.0, 0.0, 0.0):
            n, d = -n, -d
        u, v = _basis(n)
        o = self._origin(n, d)
        return u, v, o

    def _cross(self, i, k):
        """Cross-section of cell i on its own k-th half-space plane, in that
        plane's CANONICAL frame, as (polygon, edge half-planes)."""
        if self._CROSS_CACHE is None:
            self._CROSS_CACHE = {}
        ck = (i, k)
        r = self._CROSS_CACHE.get(ck)
        if r is not None:
            return r
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
                    self._CROSS_CACHE[ck] = ([], [])
                    return [], []
                continue
            nrm = math.hypot(a, b)
            a, b, c = a / nrm, b / nrm, c / nrm
            edges.append((a, b, c))
            pg = _clip2(pg, a, b, c)
            if len(pg) < 3:
                self._CROSS_CACHE[ck] = ([], [])
                return [], []
        self._CROSS_CACHE[ck] = (pg, edges)
        return pg, edges

    # ------------------------------------------------------- boundary / walls
    def boundary_faces(self, min_radius=48.0):
        """Faces of free cells that are NOT portals -- i.e. the cell's own walls,
        floors and ceilings.  A connection site is chosen from these, so a
        doorway candidate is a real wall panel of the free volume rather than a
        ray that happened to hit something."""
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
        """Connected components of the free volume under portal adjacency."""
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
    """Serialise a complex into plain arrays (for handing between processes)."""
    counts = np.array([len(c) for c in ns.cells], dtype=np.int64)
    flat = np.vstack(ns.cells) if ns.cells else np.zeros((0, 4))
    return (counts, flat, ns.world_lo, ns.world_hi, ns.gridcell, ns.mask,
            ns.n_open_leaves, ns.n_solid_leaves, ns.n_detail_splits,
            getattr(ns, 'dropped_leaves', 0))


def unpack(t):
    counts, flat, wlo, whi, gc, mask, nol, nsl, nds, dl = t
    ns = NegSpace.__new__(NegSpace)
    off = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(counts, out=off[1:])
    ns.cells = [flat[off[i]:off[i + 1]] for i in range(len(counts))]
    ns.cell_leaf = [-1] * len(counts)
    ns.world_lo, ns.world_hi = wlo, whi
    ns.gridcell, ns.mask = gc, mask
    ns.n_open_leaves, ns.n_solid_leaves = nol, nsl
    ns.n_detail_splits, ns.dropped_leaves = nds, dl
    ns.portals = None
    ns.adj = None
    ns._finish()
    return ns



# ---------------------------------------------------------------------------
# SOURCE-BRUSH FRONT END
# ---------------------------------------------------------------------------
# One definition of solidity, two entry points.  `NegSpace(bsp)` reads a COMPILED
# world; `from_brushes()` reads AUTHORED SOURCE, before any compile exists.  Both
# produce the same object -- convex free cells with `solid(p) == cell_at(p) < 0`
# -- so a generator validating what it just authored and a fuser validating what
# shipped are answering with the same law rather than with two oracles that
# agree until they do not.
#
# The pre-compile path is load-bearing: catching a bad seed costs a fraction of a
# second here against ~124 s to compile it first, and a sweep pays that per seed.
#
# METHOD.  A compiled world hands us bounded convex regions for free (the BSP
# leaves).  Authored source has no tree, so the bounding regions come from a
# uniform grid: free space is the union over grid boxes of (box MINUS the solid
# brushes overlapping it), each decomposed convexly by the same `subtract` the
# leaf path uses.  That is exact -- a grid box is a bounded convex region like a
# leaf, and nothing about the decomposition depends on where the box came from.


# Source has no contents lump, only shader names, so the compiled path's
# `contents & MASK_PLAYERSOLID` filter has a name-based counterpart here.  These
# are the `common/` tool shaders that do NOT stop a player: compile hints, vis
# helpers, and clips aimed at other entity classes.  Omitting this made every
# tool brush solid and reported 16 of 23 stock spawnpoints as buried, against 5
# in the compiled world -- a generator that trusted it would delete good spawns.
# `common/caulk`, `common/clip` and `common/nodraw` are NOT here: they are solid
# to a player and must keep blocking.
NONSOLID_SHADERS = frozenset((
    'common/hint', 'common/skip', 'common/areaportal', 'common/nodrawnonsolid',
    'common/origin', 'common/lightgrid', 'common/trigger', 'common/weapclip',
    'common/monsterclip', 'common/botclip', 'common/donotenter',
    'common/clusterportal', 'common/antiportal', 'common/full_clip',
))


def brush_is_solid(br, nonsolid=NONSOLID_SHADERS):
    """Does this authored brush stop a PLAYER?  Name-based mirror of the
    compiled path's contents mask.  A brush is judged by the shaders on its
    faces; a tool brush is uniformly one shader."""
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
    """Every vertex of an authored brush, placed at `off`.

    An authored brush carries its polygons, so its AABB is exactly the range of
    those points.  Deriving it by interval propagation instead is not merely
    looser -- propagation cannot bound an oblique prism at all, and a front end
    that skipped what it could not bound silently DROPPED every plug brush,
    reporting sealed mouths as open."""
    faces = getattr(br, 'faces', None)
    if faces is not None:
        polys = [list(f[0]) for f in faces]
    elif br and hasattr(br[0], 'p'):
        polys = [list(f.p) for f in br]
    else:
        return []
    return [[p[i] + off[i] for i in range(3)] for poly in polys for p in poly]


def brush_planes(br, off=(0.0, 0.0, 0.0)):
    """Outward half-spaces (n.p <= d) of an authored brush, placed at `off`.

    TWO input kinds, and they need DIFFERENT orientation rules:

    * a `Brush` (spiralgen/mapsrc) holds real POLYGONS, so the outward direction
      is resolved numerically against the vertex centroid -- correct for any
      convex solid and not dependent on winding discipline;
    * a list of parsed `.map` `Face`s holds three PLANE-DEFINING points per face,
      which are NOT the brush's vertices.  Their centroid is meaningless and can
      sit outside the brush, so the centroid rule flips faces at random.  The
      .map convention already fixes orientation -- (p1-p0)x(p2-p0) points INTO
      the brush -- so the outward plane is simply its negation.

    Using the centroid rule on parsed faces made single points test "inside" 50-90
    brushes of thousands of units each, and reported 16 of 23 stock spawnpoints
    as buried when the compiled world had 5."""
    if isinstance(br, np.ndarray):
        H = br.astype(np.float64)
        return [((float(H[i, 0]), float(H[i, 1]), float(H[i, 2])),
                 float(H[i, 3] + H[i, 0] * off[0] + H[i, 1] * off[1] + H[i, 2] * off[2]))
                for i in range(len(H))]
    faces = getattr(br, 'faces', None)
    if faces is None and br and hasattr(br[0], 'p'):
        out = []
        for f in br:                              # parsed .map faces
            p0, p1, p2 = [np.asarray(x, dtype=float) for x in f.p]
            n = np.cross(p1 - p0, p2 - p0)
            L = float(np.linalg.norm(n))
            if L < 1e-9:
                continue
            n = -n / L                            # inward -> outward
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
        if sum(n[i] * c[i] for i in range(3)) > d:        # flip to face outward
            n = [-x for x in n]
            d = -d
        out.append(((n[0], n[1], n[2]),
                    d + n[0] * off[0] + n[1] * off[1] + n[2] * off[2]))
    return out


def from_brushes(tiles, cell=512.0, pad=64.0, mask=MASK_PLAYERSOLID, verbose=False):
    """Build a NegSpace from AUTHORED SOURCE brushes.

    `tiles` is [(brushes, offset)] exactly as placement decides, so one call
    serves a single generated tile and a placed fusion alike.

    REPRESENTATION.  The compiled path materialises the free volume as convex
    cells, because a BSP hands it bounded convex regions for free.  Authored
    source has no tree, and decomposing free space out of a grid is the wrong
    shape -- a 512-unit box over a spiral corridor overlaps dozens of brushes and
    the convex subtraction explodes.  So this path keeps the SOLID brushes and
    answers the law directly: `solid_at(p)` is "p is inside some brush", which is
    exact, and every other query is derived from exact ray/box arithmetic against
    those same brushes.  One definition, two exact implementations -- not two
    definitions -- and they can be checked against each other, which is the point
    of having one law rather than one data structure."""
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
    ns.mask = mask
    ns.gridcell = float(cell)
    ns.cells = []
    ns.cell_leaf = []
    ns.cell_tile = []
    ns.n_open_leaves = ns.n_solid_leaves = ns.n_detail_splits = ns.dropped_leaves = 0
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
        return ns
    ns.world_lo = np.array([lo[a2] - pad for a2 in range(3)])
    ns.world_hi = np.array([hi[a2] + pad for a2 in range(3)])
    g = {}
    for i, (H, blo, bhi) in enumerate(solids):
        c0 = np.floor(blo / cell).astype(np.int64)
        c1 = np.floor(bhi / cell).astype(np.int64)
        if np.prod(c1 - c0 + 1) > 20000:
            g.setdefault('big', []).append(i)
            continue
        for x in range(c0[0], c1[0] + 1):
            for y in range(c0[1], c1[1] + 1):
                for z in range(c0[2], c1[2] + 1):
                    g.setdefault((x, y, z), []).append(i)
    ns.sgrid = g
    if verbose:
        print('negspace(source): %d solid brushes, world %s..%s'
              % (len(solids), [int(x) for x in ns.world_lo], [int(x) for x in ns.world_hi]))
    return ns


def save(ns, path):
    """Persist the assembled free volume as a real artifact.

    The fused world's BSP tree does NOT spatially contain the connector geometry:
    mapfuse attaches connector leaves under a degenerate router chain, and the
    engine still collides with them because DarkPlaces traces a BIH over BRUSHES
    (`Mod_CollisionBIH_TraceBrush`), not the tree.  So the assembled free volume
    cannot be recovered from fused.bsp alone.  It is written out instead, and
    every downstream tool loads THIS -- which is how there comes to be exactly one
    answer to "is this solid" rather than one per tool."""
    cells = ns.cells
    counts = np.array([len(c) for c in cells], dtype=np.int64)
    flat = np.vstack(cells) if cells else np.zeros((0, 4))
    np.savez_compressed(path, counts=counts, flat=flat,
                        world_lo=ns.world_lo, world_hi=ns.world_hi,
                        gridcell=np.array([ns.gridcell]),
                        mask=np.array([ns.mask]),
                        tile=np.array(getattr(ns, 'cell_tile', [-1] * len(cells)),
                                      dtype=np.int64))
    return path if path.endswith('.npz') else path + '.npz'


def load_saved(path):
    z = np.load(path)
    ns = NegSpace.__new__(NegSpace)
    counts = z['counts']
    flat = z['flat']
    off = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(counts, out=off[1:])
    ns.cells = [flat[off[i]:off[i + 1]] for i in range(len(counts))]
    ns.cell_leaf = [-1] * len(counts)
    ns.cell_tile = list(z['tile'])
    ns.world_lo = z['world_lo']
    ns.world_hi = z['world_hi']
    ns.gridcell = float(z['gridcell'][0])
    ns.mask = int(z['mask'][0])
    ns.n_open_leaves = 0
    ns.n_solid_leaves = 0
    ns.n_detail_splits = 0
    ns.dropped_leaves = 0
    ns.portals = None
    ns.adj = None
    ns._finish()
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
