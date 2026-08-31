import struct, sys, os, re, math, glob, random, subprocess, time, zipfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mkentfile as M
from mapfuse import fnum, vstr, check_bsp, vadd, vsub, vdot, vscale, vnorm, vcross

Q3MAP2 = os.environ.get('Q3MAP2', os.path.expanduser('~/dox/xonotic/netradiant-custom/install/q3map2'))
BASEPATH = os.environ.get('XON_BASEPATH', os.path.expanduser('~/dox/xonotic/Xonotic'))
TEX = dict(floor='exx/floor-tread01', wall='exx/wall-bigrib02', ceil='exx/panel-metalbig04',
           trim='exx/trim-01', pad='exx/floor-clang01', light='trak5x/light/light_light3a',
           caulk='common/caulk', trigger='common/trigger')
THK, DOORH, GRID = 32.0, 192.0, 176.0


def basis(n):
    w = [0.0, 0.0, 1.0] if abs(n[2]) < 0.9 else [1.0, 0.0, 0.0]
    u = vnorm(vcross(n, w))
    return u, vcross(n, u)


def face_pts(n, o):
    u, v = basis(n)
    return [o, vadd(o, vscale(v, 64)), vadd(o, vscale(u, 64))]


class Gen:
    def __init__(self, seed, name):
        self.rng, self.name = random.Random(seed), name
        self.world_brushes, self.ents, self.brush_ents = [], [], []
        self.floors, self.wps, self.links = [], [], []

    def box(self, mins, maxs, tex, texmap=None, dest=None):
        fs = []
        for a in range(3):
            for s, coord in ((1, maxs), (-1, mins)):
                n = [0.0] * 3
                n[a] = float(s)
                o = list(coord) if s > 0 else list(mins)
                o2 = [maxs[k] if k != a else coord[a] for k in range(3)] if s > 0 else \
                     [mins[k] if k != a else coord[a] for k in range(3)]
                t = texmap.get((a, s), tex) if texmap else tex
                fs.append((face_pts(n, o2), t))
        (dest if dest is not None else self.world_brushes).append(fs)

    def slab(self, x0, y0, x1, y1, z0, z1, tex, top=None):
        tm = {(2, 1): top} if top else None
        self.box([x0, y0, z0], [x1, y1, z1], tex, tm)

    def wall_with_door(self, axis, at, lo0, lo1, z0, z1, door_c, door_w, tex):
        dz1 = min(z1, z0 + DOORH)
        d0, d1 = door_c - door_w / 2, door_c + door_w / 2
        segs = [(lo0, d0, z0, z1), (d1, lo1, z0, z1), (d0, d1, dz1, z1)]
        for s0, s1, sz0, sz1 in segs:
            if s1 - s0 < 1 or sz1 - sz0 < 1:
                continue
            if axis == 0:
                self.box([at, s0, sz0], [at + THK, s1, sz1], tex)
            else:
                self.box([s0, at, sz0], [s1, at + THK, sz1], tex)

    def wall(self, axis, at, lo0, lo1, z0, z1, tex):
        if axis == 0:
            self.box([at, lo0, z0], [at + THK, lo1, z1], tex)
        else:
            self.box([lo0, at, z0], [lo1, at + THK, z1], tex)

    def room(self, cx, cy, z, w, d, h, doors):
        x0, x1, y0, y1 = cx - w / 2, cx + w / 2, cy - d / 2, cy + d / 2
        self.slab(x0 - THK, y0 - THK, x1 + THK, y1 + THK, z - THK, z, TEX['caulk'], top=TEX['floor'])
        self.slab(x0 - THK, y0 - THK, x1 + THK, y1 + THK, z + h, z + h + THK, TEX['caulk'])
        for side, axis, at, lo0, lo1 in (('w', 0, x0 - THK, y0 - THK, y1 + THK), ('e', 0, x1, y0 - THK, y1 + THK),
                                         ('s', 1, y0 - THK, x0 - THK, x1 + THK), ('n', 1, y1, x0 - THK, x1 + THK)):
            dr = doors.get(side)
            if dr:
                self.wall_with_door(axis, at, lo0, lo1, z, z + h, dr[0], dr[1], TEX['wall'])
            else:
                self.wall(axis, at, lo0, lo1, z, z + h, TEX['wall'])
        self.slab(cx - 32, cy - 32, cx + 32, cy + 32, z + h - 2, z + h, TEX['light'])
        self.floors.append((x0 + 24, y0 + 24, x1 - 24, y1 - 24, z))
        self.ents.append({'classname': 'light', 'origin': '%g %g %g' % (cx, cy, z + h - 48), 'light': '600'})

    def corridor(self, axis, c0, c1, perp, z, w, h):
        w2 = w / 2
        if axis == 0:
            x0, x1, y0, y1 = c0, c1, perp - w2, perp + w2
        else:
            y0, y1, x0, x1 = c0, c1, perp - w2, perp + w2
        self.slab(x0 - THK, y0 - THK, x1 + THK, y1 + THK, z - THK, z, TEX['caulk'], top=TEX['trim'])
        self.slab(x0 - THK, y0 - THK, x1 + THK, y1 + THK, z + h, z + h + THK, TEX['caulk'])
        if axis == 0:
            self.wall(1, y0 - THK, x0, x1, z, z + h, TEX['wall'])
            self.wall(1, y1, x0, x1, z, z + h, TEX['wall'])
        else:
            self.wall(0, x0 - THK, y0, y1, z, z + h, TEX['wall'])
            self.wall(0, x1, y0, y1, z, z + h, TEX['wall'])
        self.floors.append((x0 + 24, y0 + 24, x1 - 24, y1 - 24, z))

    def ledge(self, x0, y0, x1, y1, z):
        self.slab(x0, y0, x1, y1, z - 24, z, TEX['caulk'], top=TEX['floor'])
        self.floors.append((x0 + 16, y0 + 16, x1 - 16, y1 - 16, z))

    def jumppad(self, at, target, idx):
        self.slab(at[0] - 48, at[1] - 48, at[0] + 48, at[1] + 48, at[2], at[2] + 8, TEX['pad'])
        tn = 'jp%d' % idx
        br = []
        self.box([at[0] - 40, at[1] - 40, at[2] + 8], [at[0] + 40, at[1] + 40, at[2] + 72], TEX['trigger'], dest=br)
        self.brush_ents.append(({'classname': 'trigger_push', 'target': tn}, br))
        self.ents.append({'classname': 'target_position', 'targetname': tn,
                          'origin': '%g %g %g' % (target[0], target[1], target[2] + 48)})
        p = [at[0], at[1], at[2] + 8]
        self.wps.append(p)
        self.links.append((p, list(target), 1))

    def teleporter(self, at, dest, ang, idx):
        tn = 'tp%d' % idx
        br = []
        self.box([at[0] - 40, at[1] - 40, at[2]], [at[0] + 40, at[1] + 40, at[2] + 128], TEX['trigger'], dest=br)
        self.brush_ents.append(({'classname': 'trigger_teleport', 'target': tn}, br))
        self.ents.append({'classname': 'misc_teleporter_dest', 'targetname': tn, 'angle': '%d' % ang,
                          'origin': '%g %g %g' % (dest[0], dest[1], dest[2] + 16)})
        self.links.append((list(at), list(dest), 1))

    def spawn(self, cls, p, ang=0):
        self.ents.append({'classname': cls, 'origin': '%g %g %g' % (p[0], p[1], p[2] + 32), 'angle': '%d' % ang})

    def emit_map(self, path):
        def brush_txt(fs):
            out = ['{']
            for pts, t in fs:
                out.append('( %s ) ( %s ) ( %s ) %s 0 0 0 0.5 0.5 0 0 0' %
                           (' '.join(fnum(c) for c in pts[0]), ' '.join(fnum(c) for c in pts[1]),
                            ' '.join(fnum(c) for c in pts[2]), t))
            out.append('}')
            return '\n'.join(out)
        out = ['{', '"classname" "worldspawn"', '"message" "%s"' % self.name]
        out += [brush_txt(fs) for fs in self.world_brushes]
        out.append('}')
        for e in self.ents:
            out.append('{')
            out += ['"%s" "%s"' % kv for kv in e.items()]
            out.append('}')
        for keys, brs in self.brush_ents:
            out.append('{')
            out += ['"%s" "%s"' % kv for kv in keys.items()]
            out += [brush_txt(fs) for fs in brs]
            out.append('}')
        open(path, 'w').write('\n'.join(out) + '\n')
        print('wrote %s (%d world brushes, %d ents, %d brush ents)' %
              (path, len(self.world_brushes), len(self.ents), len(self.brush_ents)))

    def gen_nav(self):
        rect_wps = []
        for (x0, y0, x1, y1, z) in self.floors:
            nx = max(1, int((x1 - x0) // GRID))
            ny = max(1, int((y1 - y0) // GRID))
            pts = {}
            for i in range(nx + 1):
                for k in range(ny + 1):
                    p = [round(x0 + (x1 - x0) * i / nx, 1), round(y0 + (y1 - y0) * k / ny, 1), z]
                    pts[(i, k)] = p
                    self.wps.append(p)
            for (i, k), p in pts.items():
                for di, dk in ((1, 0), (0, 1), (1, 1), (1, -1)):
                    q = pts.get((i + di, k + dk))
                    if q:
                        self.links.append((p, q, 0))
            rect_wps.append(list(pts.values()))
        for a in range(len(rect_wps)):
            for b in range(a + 1, len(rect_wps)):
                best = min(((math.dist(p, q), p, q) for p in rect_wps[a] for q in rect_wps[b]),
                           key=lambda t: t[0])
                if best[0] <= 300 and abs(best[1][2] - best[2][2]) <= 64:
                    self.links.append((best[1], best[2], 0))

    def write_nav(self, outdir, mapname):
        ts = time.strftime('%Y-%m-%d %H:%M:%S')
        seen = set()
        wl = ['//WAYPOINT_VERSION 1.04', '//WAYPOINT_SYMMETRY 0', '//WAYPOINT_TIME ' + ts]
        for p in self.wps:
            k = vstr(p)
            if k in seen:
                continue
            seen.add(k)
            wl += [k, k, '0']
        cl = ['//WAYPOINT_VERSION 1.04', '//WAYPOINT_TIME ' + ts]
        for a, b, oneway in self.links:
            cl.append(vstr(a) + '*' + vstr(b))
            if not oneway:
                cl.append(vstr(b) + '*' + vstr(a))
        open(os.path.join(outdir, mapname + '.waypoints'), 'w').write('\n'.join(wl) + '\n')
        open(os.path.join(outdir, mapname + '.waypoints.cache'), 'w').write('\n'.join(cl) + '\n')
        print('nav: %d waypoints %d links' % (len(seen), len(cl) - 2))



# ---------------------------------------------------------------------------
# Procedural BRIDGE TILE  (the "k-many bridge maps" of the fusion spec)
#
# A bridge tile is a small procedurally generated map whose only purpose is to be
# socketed to several other maps at once -- the 'connector' map of the spec, the
# thing that is allowed to have subtle/weakly-signposted edges *because* it has
# multiple edges to multiple other map nodes.  It is real q3map2-compiled brush
# geometry, not a corridor slab: a two-tier hub chamber with cardinal arms.
#
#   * up to 4 arms on the ground tier   (z = 0)
#   * up to 4 arms on the gallery tier  (z = TIER) reached by jumppads, and a
#     teleporter back down -- the commanded use of jumppad + portal + verticality
#   * every arm ends in a thin caulk PLUG brush so q3map2 sees a sealed hull; the
#     plug is small enough (<< 640^3) that mapfuse's corridor carver removes it
#     when it punches the join corridor through.
#
# Returns (Gen, ports) where ports is a list of dicts:
#     {'p': [x,y,z] floor point just inside the arm mouth, 'dir': [dx,dy,0],
#      'tier': 0|1, 'name': 'e0'|...}
# ---------------------------------------------------------------------------
DIRS = {'e': (1, 0), 'w': (-1, 0), 'n': (0, 1), 's': (0, -1)}
HUBW, ARMW, ARMH, ARML, TIER, PLUGT = 1024.0, 288.0, 224.0, 640.0, 320.0, 32.0


def bridge_tile(seed, arms_lo, arms_hi, name='bridge'):
    """arms_lo / arms_hi: iterables of 'e','w','n','s' -- which cardinal arms to
    cut on the ground tier and on the gallery tier."""
    rng = random.Random(seed)
    g = Gen(seed, name)
    h = HUBW / 2.0
    hub_h = TIER + 320.0
    arms_lo = list(arms_lo) + list(arms_hi)      # all ports on the ground tier:
    arms_hi = []                                 # a port must be cart-navigable
    doors_lo = {d: (0.0, ARMW) for d in arms_lo}
    g.room(0.0, 0.0, 0.0, HUBW, HUBW, hub_h, doors_lo)
    ports = []

    def arm(d, z, tier):
        dx, dy = DIRS[d]
        axis = 0 if dx else 1
        c0 = h if (dx > 0 or dy > 0) else -(h + ARML)
        c1 = c0 + ARML
        g.corridor(axis, c0, c1, 0.0, z, ARMW, ARMH)
        # end plug: sealed for q3map2, small enough for mapfuse to carve
        far = c1 if (dx > 0 or dy > 0) else c0
        sgn = 1 if (dx > 0 or dy > 0) else -1
        lo = [0.0] * 3
        hi = [0.0] * 3
        for a in range(3):
            if a == axis:
                lo[a], hi[a] = (far, far + PLUGT) if sgn > 0 else (far - PLUGT, far)
            elif a == 2:
                lo[a], hi[a] = z - THK, z + ARMH + THK
            else:
                lo[a], hi[a] = -ARMW / 2 - THK, ARMW / 2 + THK
        g.box(lo, hi, TEX['caulk'])
        px = (far - sgn * 96.0) if axis == 0 else 0.0
        py = (far - sgn * 96.0) if axis == 1 else 0.0
        p = [round(px, 1), round(py, 1), z]
        g.wps.append(p)
        ports.append({'p': p, 'dir': [float(dx), float(dy), 0.0], 'tier': tier, 'name': d + str(tier)})

    for d in arms_lo:
        arm(d, 0.0, 0)
    # VERTICALITY: a gallery ring ledge above the hub floor, reached by a jumppad and
    # left by a teleporter -- "ways to use the portal and jump pad and verticality".
    # It is an interior feature, never a port, so every port stays cart-navigable.
    gw = 288.0
    g.ledge(-h, -h, h, -h + gw, TIER)
    g.ledge(-h, h - gw, h, h, TIER)
    g.ledge(-h, -h + gw, -h + gw, h - gw, TIER)
    g.ledge(h - gw, -h + gw, h, h - gw, TIER)
    padat = [0.0, -(h - 200.0), 0.0]
    land = [0.0, h - gw / 2, TIER]
    g.jumppad(padat, land, 0)
    g.teleporter(land, [0.0, -(h - 400.0), 0.0], 90, 0)
    g.spawn('info_player_deathmatch', [0.0, 0.0, 0.0], rng.randrange(360))
    g.gen_nav()
    # gen_nav only grid-links coplanar floors; snap the jumppad/teleporter links onto
    # the nearest REAL generated waypoint so the gallery is genuinely reachable in the
    # fused waypoint cache rather than dangling on a synthetic endpoint.
    wpset = {tuple(w) for w in g.wps}

    def snap(q):
        if tuple(q) in wpset:
            return list(q)
        cand = [w for w in g.wps if abs(w[2] - q[2]) < 1.0] or g.wps
        return list(min(cand, key=lambda w: math.dist(w, q)))
    g.links = [(snap(a), snap(b), o) for a, b, o in g.links]
    # a port must be a REAL node of the tile's waypoint graph, otherwise mapfuse
    # sockets the join corridor onto a point no bot can stand on or route through.
    linked = set()
    for a, b, o in g.links:
        linked.add(tuple(a)); linked.add(tuple(b))
    for pt in ports:
        cand = [w for w in g.wps if tuple(w) in linked and abs(w[2] - pt['p'][2]) < 1.0]
        if cand:
            pt['p'] = list(min(cand, key=lambda w: math.dist(w, pt['p'])))
    return g, ports


def build_bridge_tile(outdir, name, seed, arms_lo, arms_hi):
    """Emit + q3map2-compile a bridge tile and its nav files into outdir.
    Returns (basepath_without_ext, ports)."""
    os.makedirs(outdir, exist_ok=True)
    g, ports = bridge_tile(seed, arms_lo, arms_hi, name)
    mappath = os.path.join(outdir, name + '.map')
    g.emit_map(mappath)
    compile_map(mappath)
    g.write_nav(outdir, name)
    base = os.path.join(outdir, name)
    probs = check_bsp(open(base + '.bsp', 'rb').read())
    print('bridge %s: parse-back %s  ports=%s' %
          (name, 'OK' if not probs else 'PROBLEMS %s' % probs[:5],
           [p['name'] for p in ports]))
    return base, ports


def compile_map(mappath, verbose=False):
    env = dict(os.environ)
    for stage in (['-meta'], ['-vis', '-threads', '1'], ['-light', '-threads', '1', '-fast', '-samples', '2', '-bounce', '2']):
        cmd = [Q3MAP2, '-game', 'xonotic', '-fs_basepath', BASEPATH] + stage + [mappath]
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        tail = [l for l in r.stdout.splitlines() if l.strip()][-3:]
        leaked = [l for l in r.stdout.splitlines() if 'LEAK' in l.upper() or 'WARNING' in l]
        print('q3map2 %s: rc=%d %s' % (stage[0], r.returncode, '; '.join(tail)))
        for l in leaked[:6]:
            print('q3map2 %s: %s' % (stage[0], l.strip()))
        if r.returncode:
            print(r.stdout[-2000:])
            print(r.stderr[-1000:])
            raise SystemExit('q3map2 %s failed' % stage[0])


def ring_arena(seed, nrooms):
    rng = random.Random(seed)
    g = Gen(seed, 'arena seed %d' % seed)
    nrooms = max(4, nrooms + (nrooms & 1))
    cols = nrooms // 2
    pitch = 1536.0
    cells = [(i, 0) for i in range(cols)] + [(i, 1) for i in range(cols - 1, -1, -1)]
    rooms = []
    for gx, gy in cells:
        rooms.append([gx * pitch, gy * pitch, rng.choice([704, 832, 960]),
                      rng.choice([704, 832, 960]), rng.choice([288, 320, 384]), {}])
    tower = rng.randrange(nrooms)
    rooms[tower][4] = 640
    pairs = []
    for i in range(nrooms):
        jn = (i + 1) % nrooms
        axis = 0 if cells[i][1] == cells[jn][1] else 1
        a, b = rooms[i], rooms[jn]
        if axis == 0:
            lo, hi = (a, b) if a[0] < b[0] else (b, a)
            lo[5]['e'] = (a[1], 224)
            hi[5]['w'] = (a[1], 224)
        else:
            lo, hi = (a, b) if a[1] < b[1] else (b, a)
            lo[5]['n'] = (a[0], 224)
            hi[5]['s'] = (a[0], 224)
        pairs.append((i, jn, axis))
    for cx, cy, w, d, h, doors in rooms:
        g.room(cx, cy, 0.0, w, d, h, doors)
    for i, jn, axis in pairs:
        a, b = rooms[i], rooms[jn]
        if axis == 0:
            lo, hi = (a, b) if a[0] < b[0] else (b, a)
            g.corridor(0, lo[0] + lo[2] / 2, hi[0] - hi[2] / 2, a[1], 0.0, 192, 224)
        else:
            lo, hi = (a, b) if a[1] < b[1] else (b, a)
            g.corridor(1, lo[1] + lo[3] / 2, hi[1] - hi[3] / 2, a[0], 0.0, 192, 224)
    tx, ty, tw, td, th = rooms[tower][:5]
    lx0, ly0 = tx + tw / 2 - 264, ty - 128
    g.ledge(lx0, ly0, lx0 + 232, ly0 + 256, 288.0)
    padat = [tx - tw / 2 + 96, ty, 0.0]
    ledge_c = [lx0 + 116, ly0 + 128, 288.0]
    g.jumppad(padat, ledge_c, 0)
    far = rooms[(tower + nrooms // 2) % nrooms]
    g.teleporter(ledge_c, [far[0], far[1], 0.0], 90, 0)
    for i, (cx, cy, w, d, h, doors) in enumerate(rooms):
        g.spawn('info_player_deathmatch', [cx + 128, cy + 128, 0], rng.randrange(360))
        g.spawn('info_player_team%d' % (i % 4 + 1), [cx - 128, cy - 128, 0], rng.randrange(360))
        if i % 2:
            g.spawn('info_player_team%d' % (i % 4 + 1), [cx - 128, cy + 128, 0], rng.randrange(360))
    g.gen_nav()
    return g


def smoke(outdir, mapname):
    log = os.path.join(outdir, 'smoke.log')
    cmd = [os.path.expanduser('~/dox/xonotic/build-engine/darkplaces-dedicated'), '-xonotic',
           '-basedir', BASEPATH, '-userdir', os.path.dirname(os.path.dirname(outdir)),
           '+port', '26013', '+sv_public', '0', '+g_payload', '0',
           '+bot_number', '2', '+skill', '1', '+map', mapname]
    with open(log, 'w') as lf:
        p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
        time.sleep(20)
        p.terminate()
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()
    out = open(log, errors='ignore').read()
    errs = [l for l in out.splitlines() if re.search(r'Host_Error|OBJECT ERROR|not found|invalid|funny', l)]
    print('smoke: %d lines, %d error-ish -> %s' % (len(out.splitlines()), len(errs), log))
    for l in errs[:10]:
        print('smoke-err: %s' % l)
    for l in out.splitlines()[-6:]:
        print('smoke-tail: %s' % l)


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    flags = [a for a in sys.argv[1:] if a.startswith('--')]
    seed = int(args[0]) if args else 0
    nrooms = 6
    outdir = '/tmp/mapgensmoke/data/maps'
    for f in flags:
        if f.startswith('--rooms='):
            nrooms = int(f[8:])
        if f.startswith('--out='):
            outdir = f[6:]
    os.makedirs(outdir, exist_ok=True)
    mapname = 'genarena%d' % seed
    print('mapgen seed=%d rooms=%d out=%s' % (seed, nrooms, outdir))
    g = ring_arena(seed, nrooms)
    mappath = os.path.join(outdir, mapname + '.map')
    g.emit_map(mappath)
    compile_map(mappath)
    bsp = os.path.join(outdir, mapname + '.bsp')
    probs = check_bsp(open(bsp, 'rb').read())
    print('parse-back: %s' % ('OK' if not probs else 'PROBLEMS %s' % probs[:8]))
    g.write_nav(outdir, mapname)
    B = M.Bsp(open(bsp, 'rb').read())
    tb = M.trigger_boxes(open(bsp, 'rb').read())
    intrig = lambda q: any(all(lo[k] - 1 <= q[k] <= hi[k] + 1 for k in range(3)) for lo, hi in tb)
    viol = sum(1 for p in g.wps
               if (B.inside((p[0], p[1], p[2] + 24)) and not intrig((p[0], p[1], p[2] + 24)))
               or B.floor(p[0], p[1], p[2] + 24) is None)
    print('wp standable check: %s (%d/%d violations)' % ('PASS' if viol == 0 else 'FAIL', viol, len(g.wps)))
    open(os.path.join(outdir, mapname + '.mapinfo'), 'w').write(
        'title GenArena %d\ndescription parametric payload arena\nauthor mapgen\n'
        'gametype dm\ngametype tdm\ngametype plc\n' % seed)
    M.emit(bsp, os.path.join(outdir, mapname + '.ent'), 5, 3, '')
    with zipfile.ZipFile(os.path.join(outdir, mapname + '.pk3'), 'w', zipfile.ZIP_DEFLATED) as z:
        for ext in ('.bsp', '.waypoints', '.waypoints.cache', '.mapinfo', '.ent'):
            z.write(os.path.join(outdir, mapname + ext), 'maps/' + mapname + ext)
    print('wrote %s' % os.path.join(outdir, mapname + '.pk3'))
    if '--smoke' in flags:
        smoke(outdir, mapname)
