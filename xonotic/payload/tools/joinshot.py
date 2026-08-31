#!/usr/bin/env python3
"""joinshot.py -- headless egocentric client renders at map-fusion join edges.

Where joinview.py only *probes* joins with a raycast (the dedicated server has no
GL context and cannot render), joinshot drives the real Xonotic/DarkPlaces client
in its built-in software rasterizer (DPSOFTRAST) under a windowless SDL video
driver, so it produces actual rendered PNG frames of what a player sees crossing
each level<->level join.

Mechanism (all stock engine, no engine/qc edits):
  * SDL_VIDEODRIVER=dummy + `vid_soft 1`  -> DarkPlaces Software Rasterizer, no window.
  * For each join we emit an `info_autoscreenshot` entity (origin + view angles) into
    an override `maps/<map>.ent`. Xonotic's stock `impulse 143` cheat teleports a
    noclipping player onto the next such entity (setting its view angles) and deletes
    it, so repeated impulses walk our camera list in order.
  * The engine's own `screenshot foo.png` writes the frame (scr_screenshot_png 1).

Both sides of every edge are captured:
  * corridor  : eye on side A looking through toward B, and eye on B looking toward A
                (the walk-through sightline each way).
  * teleporter/jumppad : an APPROACH frame (backed off the near pad/portal, looking at
                it) and a LANDING frame (at the far endpoint looking outward into the
                destination map) -- transport is instant/ballistic, so there is no
                straight-line sightline to shoot.

Usage:
    joinshot.py <fused_map_dir> [--out DIR] [--width W] [--height H]
                [--step SEC] [--settle SEC] [--xonotic DIR] [--keep]

<fused_map_dir> is a mapfuse output dir containing fused.pk3 + fused.joins.json
(e.g. /tmp/fuse_v7/data/maps).  PNGs land in --out (default <dir>/joinshots).
"""
import os, sys, json, math, struct, zipfile, shutil, subprocess, argparse, tempfile, time

EYE = 40.0          # eye height above the join floor point
BACK = 176.0        # how far to back off a teleporter/jumppad pad for the approach shot
LOOK = 320.0        # forward distance of the landing-shot aim point


def vectoangles_view(dx, dy, dz):
    """Return (pitch, yaw, roll) in the convention Xonotic's info_autoscreenshot
    bakes: vectoangles() then negated pitch, which is what the player view wants."""
    yaw = math.degrees(math.atan2(dy, dx))
    horiz = math.hypot(dx, dy)
    pitch = -math.degrees(math.atan2(dz, horiz))   # negated, per info_autoscreenshot_findtarget
    return pitch, yaw, 0.0


def cameras_for_join(idx, jn):
    """Yield (shot_name, eye_xyz, angles_pyr) for both sides of one join."""
    sa, sb = jn['sa'], jn['sb']
    kind = jn['kind']
    dx, dy = sb[0] - sa[0], sb[1] - sa[1]
    hlen = math.hypot(dx, dy) or 1.0
    ux, uy = dx / hlen, dy / hlen           # horizontal A->B unit
    tag = 'j%02d_%s' % (idx, kind)
    if kind == 'corridor':
        # walk-through sightline each way
        ea = [sa[0], sa[1], sa[2] + EYE]
        aa = [sb[0], sb[1], sb[2] + EYE]
        yield ('%s_a_through' % tag, ea, vectoangles_view(aa[0]-ea[0], aa[1]-ea[1], aa[2]-ea[2]))
        eb = [sb[0], sb[1], sb[2] + EYE]
        ab = [sa[0], sa[1], sa[2] + EYE]
        yield ('%s_b_through' % tag, eb, vectoangles_view(ab[0]-eb[0], ab[1]-eb[1], ab[2]-eb[2]))
    else:
        # teleporter / jumppad: approach the near pad, then the landing view
        ea = [sa[0] - ux * BACK, sa[1] - uy * BACK, sa[2] + EYE]
        aa = [sa[0], sa[1], sa[2] + EYE * 0.5]
        yield ('%s_a_approach' % tag, ea, vectoangles_view(aa[0]-ea[0], aa[1]-ea[1], aa[2]-ea[2]))
        eb = [sb[0], sb[1], sb[2] + EYE]
        ab = [sb[0] + ux * LOOK, sb[1] + uy * LOOK, sb[2] + EYE]
        yield ('%s_b_landing' % tag, eb, vectoangles_view(ab[0]-eb[0], ab[1]-eb[1], ab[2]-eb[2]))





def cameras_for_portal(idx, pt):
    """Two frames of one CUT DOORWAY -- the actual deliverable of a geometry edit.

    A join sightline shot from inside the connector shows the connector, not the edit.
    These stand back inside the host map looking at the new opening in its own wall, and
    then outside the wall looking back at it, which is where a hole reads as a hole and a
    doorway reads as a doorway."""
    node, mouth = pt['node'], pt['mouth']
    ax, sgn = pt['axis'], pt['sgn']
    d = [0.0, 0.0, 0.0]
    d[ax] = sgn
    tag = 'p%02d_%s_%s' % (idx, pt['name'][:12], pt['kind'])
    # inside the map, backed off the wall, looking at the new doorway
    ea = [node[0] - d[0] * 240.0, node[1] - d[1] * 240.0, node[2] + EYE]
    aa = [mouth[0], mouth[1], mouth[2] + 72.0]
    yield ('%s_in' % tag, ea, vectoangles_view(aa[0] - ea[0], aa[1] - ea[1], aa[2] - ea[2]))
    # outside the wall, looking back at the opening in the level's own facade
    eb = [mouth[0] + d[0] * 264.0, mouth[1] + d[1] * 264.0, mouth[2] + EYE]
    ab = [node[0], node[1], node[2] + 72.0]
    yield ('%s_out' % tag, eb, vectoangles_view(ab[0] - eb[0], ab[1] - eb[1], ab[2] - eb[2]))

# ---------------------------------------------------------------------------
# Region vantage cameras + the VOID AUDIT.
#
# Rendering only the joins was not sufficient evidence that a fusion works: a live
# client showed a world that was "almost entirely black void ... a single small
# isolated island of structures", and every offline number (3 maps, 3 joins,
# flood-fill OK) had passed.  A join camera stands INSIDE a corridor, where the
# corridor's own four walls fill the frame, so a region that renders as nothing is
# invisible to it.  The region cameras below stand on real bot-reachable waypoints
# inside each fused region and look outward on four yaws; the audit then measures
# how much of each frame is void.  A region that renders black from its own floor is
# the exact failure in that screenshot, and it now fails offline.
# ---------------------------------------------------------------------------

def cameras_for_region(idx, mp):
    """Yield cameras standing on this region's own vantage waypoints."""
    name = ''.join(ch if ch.isalnum() else '_' for ch in mp.get('name', 'r%d' % idx))
    for vi, v in enumerate(mp.get('vantages', [])[:2]):
        for yi, yaw in enumerate((0.0, 90.0, 180.0, 270.0)):
            yield ('r%02d_%s_v%d_y%d' % (idx, name, vi, int(yaw)),
                   [v[0], v[1], v[2] + EYE], (0.0, yaw, 0.0))


def cameras_overview(joins):
    """One high camera per region looking down, plus one over the whole megamap."""
    mins = [min(m['mins'][a] for m in joins['maps']) for a in range(3)]
    maxs = [max(m['maxs'][a] for m in joins['maps']) for a in range(3)]
    cx, cy = (mins[0] + maxs[0]) / 2, (mins[1] + maxs[1]) / 2
    yield ('ov_world', [cx, cy, maxs[2] + 512.0], (89.0, 0.0, 0.0))
    for i, m in enumerate(joins['maps']):
        mx = (m['mins'][0] + m['maxs'][0]) / 2
        my = (m['mins'][1] + m['maxs'][1]) / 2
        yield ('ov%02d' % i, [mx, my, m['maxs'][2] + 256.0], (89.0, 0.0, 0.0))


def _defilter(raw, pos, pw, ph, nch):
    """De-filter one PNG pixel block; returns (bytes, new_pos)."""
    stride = pw * nch
    out = bytearray(ph * stride)
    prev = bytearray(stride)
    for y in range(ph):
        ft = raw[pos]
        pos += 1
        line = bytearray(raw[pos:pos + stride])
        pos += stride
        if ft == 1:
            for x in range(nch, stride):
                line[x] = (line[x] + line[x - nch]) & 255
        elif ft == 2:
            for x in range(stride):
                line[x] = (line[x] + prev[x]) & 255
        elif ft == 3:
            for x in range(stride):
                aa = line[x - nch] if x >= nch else 0
                line[x] = (line[x] + ((aa + prev[x]) >> 1)) & 255
        elif ft == 4:
            for x in range(stride):
                aa = line[x - nch] if x >= nch else 0
                bb = prev[x]
                cc = prev[x - nch] if x >= nch else 0
                pp = aa + bb - cc
                pa, pb, pc = abs(pp - aa), abs(pp - bb), abs(pp - cc)
                pr = aa if (pa <= pb and pa <= pc) else (bb if pb <= pc else cc)
                line[x] = (line[x] + pr) & 255
        out[y * stride:(y + 1) * stride] = line
        prev = line
    return out, pos


ADAM7 = ((0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4),
         (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2))


def read_png_rgb(path):
    """Minimal PNG reader: 8-bit gray/RGB/RGBA, both non-interlaced and Adam7.
    There is no PIL on this box, and DarkPlaces writes INTERLACED PNGs -- which is
    why the first version of this audit could not read a single engine frame."""
    import zlib as _z
    d = open(path, 'rb').read()
    if d[:8] != b'\x89PNG\r\n\x1a\n':
        raise ValueError('not a png')
    i, idat, w = 8, b'', None
    while i < len(d):
        ln = struct.unpack_from('>I', d, i)[0]
        typ = d[i + 4:i + 8]
        body = d[i + 8:i + 8 + ln]
        if typ == b'IHDR':
            w, h, depth, ctype = struct.unpack_from('>IIBB', body, 0)[:4]
            inter = body[12]
        elif typ == b'IDAT':
            idat += body
        elif typ == b'IEND':
            break
        i += 12 + ln
    if w is None or depth != 8:
        raise ValueError('unsupported png (depth=%s)' % depth)
    nch = {0: 1, 2: 3, 4: 2, 6: 4}[ctype]
    raw = _z.decompress(idat)
    if not inter:
        out, _ = _defilter(raw, 0, w, h, nch)
        return w, h, nch, bytes(out)
    full = bytearray(w * h * nch)
    pos = 0
    for x0, y0, dx, dy in ADAM7:
        pw = (w - x0 + dx - 1) // dx
        ph = (h - y0 + dy - 1) // dy
        if pw <= 0 or ph <= 0:
            continue
        blk, pos = _defilter(raw, pos, pw, ph, nch)
        for py in range(ph):
            ry = y0 + py * dy
            src = py * pw * nch
            for px in range(pw):
                rx = x0 + px * dx
                o = (ry * w + rx) * nch
                full[o:o + nch] = blk[src + px * nch:src + (px + 1) * nch]
    return w, h, nch, bytes(full)


def frame_stats(path, dark=18, hud_rows=0.12):
    """Fraction of VOID (near-black) pixels and the distinct-luma count of a frame.
    The bottom hud_rows of the frame are skipped (residual HUD/console text)."""
    w, h, nch, px = read_png_rgb(path)
    y1 = int(h * (1.0 - hud_rows))
    void = 0
    tot = 0
    lum = set()
    for y in range(y1):
        base = y * w * nch
        for x in range(0, w, 2):
            o = base + x * nch
            if nch >= 3:
                r, g, b = px[o], px[o + 1], px[o + 2]
            else:
                r = g = b = px[o]
            L = (r * 299 + g * 587 + b * 114) // 1000
            tot += 1
            lum.add(L >> 2)
            if L <= dark:
                void += 1
    return dict(void=void / max(1, tot), levels=len(lum), w=w, h=h)


def void_audit(outdir, shots, void_max=0.90, levels_min=6):
    """Grade every captured frame.  A frame that is almost entirely near-black with
    almost no distinct luma levels is a VOID frame: the camera stood on real
    walkable geometry and the engine drew nothing."""
    rows, bad = [], []
    for name in shots:
        fp = os.path.join(outdir, name + '.png')
        if not os.path.exists(fp):
            rows.append((name, None))
            bad.append((name, 'MISSING'))
            continue
        try:
            st = frame_stats(fp)
        except Exception as e:
            rows.append((name, None))
            bad.append((name, 'UNREADABLE %s' % e))
            continue
        rows.append((name, st))
        if st['void'] >= void_max and st['levels'] <= levels_min:
            bad.append((name, 'VOID void=%.2f levels=%d' % (st['void'], st['levels'])))
    print('void audit: %d frames graded, %d void/missing' % (len(rows), len(bad)))
    for name, st in rows:
        if st:
            print('  %-40s void=%.2f levels=%3d %dx%d' % (name, st['void'], st['levels'],
                                                          st['w'], st['h']))
    for name, why in bad:
        print('  VOID-AUDIT FAIL: %s -- %s' % (name, why))
    json.dump({n: st for n, st in rows}, open(os.path.join(outdir, 'voidaudit.json'), 'w'), indent=1)
    print('void audit: %s (wrote voidaudit.json)' % ('PASS' if not bad else 'FAIL'))
    return not bad


def read_base_ent(mapdir):
    """The authoritative entity list for the fused map lives in the pk3's
    maps/fused.ent (mapfuse writes it there).  Fall back to the bsp lump 0."""
    pk3 = os.path.join(mapdir, 'fused.pk3')
    if os.path.exists(pk3):
        with zipfile.ZipFile(pk3) as z:
            for n in z.namelist():
                if n.endswith('fused.ent'):
                    return z.read(n).decode('latin-1')
    bsp = os.path.join(mapdir, 'fused.bsp')
    d = open(bsp, 'rb').read()
    lo, ln = struct.unpack_from('<ii', d, 8)   # lump 0 = entities
    return d[lo:lo + ln].split(b'\0')[0].decode('latin-1')


def build_ent(base, cams):
    out = [base.rstrip()]
    for name, eye, ang in cams:
        out.append('{')
        out.append('"classname" "info_autoscreenshot"')
        out.append('"origin" "%.1f %.1f %.1f"' % (eye[0], eye[1], eye[2]))
        out.append('"angles" "%.2f %.2f %.2f"' % (ang[0], ang[1], ang[2]))
        out.append('}')
    return '\n'.join(out) + '\n'


HUD_OFF = ['weapons', 'ammo', 'powerups', 'healtharmor', 'notify', 'timer', 'radar',
           'score', 'vote', 'modicons', 'pressedkeys', 'chat', 'engineinfo',
           'infomessages', 'physics', 'centerprint', 'buffs', 'itemstime',
           'quickmenu', 'strafehud']


def write_base_cfg(path, mapname, w, h):
    """The boot config: set everything up, load the map, and start a 1 Hz poll that
    re-execs js_step.cfg.  Map-load time varies wildly on a shared box, so rather than
    guess a preroll we leave js_step.cfg empty and let the Python side fill it in the
    instant the player actually spawns (Python watches the log for "is now playing")."""
    L = ['// generated by joinshot.py -- headless join capture',
         'cl_allow_uid2name 0', 'cl_allow_uidtracking 0',
         'sv_cheats 1', 'sv_spectate 0', 'g_warmup 0', 'g_max_info_autoscreenshot 999',
         'sv_clientcommand_antispam_time 0', 'sv_clientcommand_antispam_count 9999',
         'bot_number 0', 'timelimit 0', 'g_maxplayers 0',
         # kill the blocking stats.xonotic.org HTTP calls that stall connect on a
         # box with no route to the internet
         'g_playerstats_gamereport_uri ""', 'g_playerstats_playerbasic_uri ""',
         'g_playerstats_playerdetail_uri ""', 'sv_eventlog 0',
         'scr_screenshot_png 1', 'scr_screenshot_timestamp 0', 'scr_screenshot_gammaboost 1',
         'r_texture_dds_load 0', 'gl_texturecompression 0',
         # DPSOFTRAST can't decompress dds, so textures load from full-res tga/png;
         # downscale hard to keep first-load time and RAM sane on a shared box.
         'gl_picmip 3', 'r_texture_max_size 128', 'r_lerpimages 0',
         'r_drawviewmodel 0', 'crosshair 0', 'con_notify 0', 'scr_centertime 0',
         'cl_deathscoreboard 0', 'r_bloom 0', 'r_motionblur 0', 'r_damageblur 0',
         'vid_width %d' % w, 'vid_height %d' % h]
    for p in HUD_OFF:
        L.append('hud_panel_%s 0' % p)
    L.append('map %s' % mapname)
    # With sv_spectate 0 the server auto-joins the client ~MIN_SPEC_TIME(=1s) after
    # it connects; we do NOT spam `cmd join` (repeated join requests keep the client
    # from ever settling into a spawn).  A single fallback join is issued once, from
    # the Python side, only if auto-join has not fired.  The poll re-execs the step
    # file so Python can hand over the shot sequence the instant the player spawns.
    L.append('alias js_poll "exec js_step.cfg; defer 1 js_poll"')
    L.append('defer 6 js_poll')
    open(path, 'w').write('\n'.join(L) + '\n')


def build_step(shots, step):
    """The shot sequence, fired once when Python detects the player has spawned.
    First line disarms the poll loop so this runs exactly once."""
    L = ['alias js_poll ""', 'togglemenu 0', 'god', 'noclip']
    t = 1.5
    for name in shots:
        L.append('defer %.1f "impulse 143"' % t)
        L.append('defer %.1f "togglemenu 0"' % (t + 0.4))
        L.append('defer %.1f "screenshot %s.png"' % (t + step - 0.4, name))
        t += step
    L.append('defer %.1f "echo JOINSHOT_DONE"' % (t + 0.2))
    L.append('defer %.1f "quit"' % (t + 1.0))
    return '\n'.join(L) + '\n', t + 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mapdir', help='mapfuse output dir with fused.pk3 + fused.joins.json')
    ap.add_argument('--out', default=None)
    ap.add_argument('--width', type=int, default=320)
    ap.add_argument('--height', type=int, default=200)
    ap.add_argument('--settle', type=float, default=300.0,
                    help='max seconds to wait for the player to spawn (map load; raise under load)')
    ap.add_argument('--step', type=float, default=2.6, help='seconds between shots')
    ap.add_argument('--xonotic', default=os.path.expanduser('~/dox/xonotic/Xonotic'))
    ap.add_argument('--keep', action='store_true', help='keep the temp run dir')
    ap.add_argument('--regions', action='store_true',
                    help='also render outward from each fused region\'s own vantage waypoints')
    ap.add_argument('--only-regions', action='store_true', help='skip the join cameras')
    ap.add_argument('--overview', action='store_true', help='add top-down overview cameras')
    ap.add_argument('--no-doors', action='store_true',
                    help='skip the cut-doorway cameras (inside/outside each new opening)')
    ap.add_argument('--limit', type=int, default=0, help='cap the number of frames')
    ap.add_argument('--audit-only', action='store_true',
                    help='do not run the engine; just grade PNGs already in --out')
    ap.add_argument('--no-audit', action='store_true')
    args = ap.parse_args()

    mapdir = os.path.abspath(args.mapdir)
    joins = json.load(open(os.path.join(mapdir, 'fused.joins.json')))
    out = args.out or os.path.join(mapdir, 'joinshots')
    os.makedirs(out, exist_ok=True)

    cams, shots = [], []
    if not args.only_regions:
        for i, jn in enumerate(joins['joins']):
            for name, eye, ang in cameras_for_join(i, jn):
                cams.append((name, eye, ang)); shots.append(name)
    if not args.no_doors:
        for i, pt in enumerate(joins.get('portals', [])):
            for name, eye, ang in cameras_for_portal(i, pt):
                cams.append((name, eye, ang)); shots.append(name)
    if args.regions:
        for i, mp in enumerate(joins['maps']):
            for name, eye, ang in cameras_for_region(i, mp):
                cams.append((name, eye, ang)); shots.append(name)
    if args.overview:
        for name, eye, ang in cameras_overview(joins):
            cams.append((name, eye, ang)); shots.append(name)
    if args.limit:
        cams, shots = cams[:args.limit], shots[:args.limit]
    print('%d joins, %d regions -> %d camera frames' %
          (len(joins['joins']), len(joins['maps']), len(cams)))
    if args.audit_only:
        return 0 if void_audit(out, shots) else 2

    bin_ = os.path.join(args.xonotic, 'Xonotic.app/Contents/MacOS/xonotic-osx-sdl-bin')
    if not os.path.exists(bin_):
        sys.exit('client binary not found: %s' % bin_)

    rundir = tempfile.mkdtemp(prefix='joinshot_')
    os.makedirs(os.path.join(rundir, 'data', 'maps'), exist_ok=True)
    # Repack the fused pk3 under a UNIQUE map name so no stray fused.* in the base
    # data dir can shadow our entity override, and inject the info_autoscreenshot
    # markers into the .ent.  ('joinshotmap' is not present in stock/base data.)
    mapname = 'joinshotmap'
    ent = build_ent(read_base_ent(mapdir), cams)
    src_pk3 = os.path.join(mapdir, 'fused.pk3')
    dst_pk3 = os.path.join(rundir, 'data', 'zzz-%s.pk3' % mapname)
    with zipfile.ZipFile(src_pk3) as zin, zipfile.ZipFile(dst_pk3, 'w', zipfile.ZIP_DEFLATED) as zout:
        for n in zin.namelist():
            base = os.path.basename(n)
            if base == 'fused.ent':
                continue                          # replaced below
            if base.startswith('fused.'):         # maps/fused.* -> maps/<mapname>.*
                nn = 'maps/%s.%s' % (mapname, base.split('.', 1)[1])
                zout.writestr(nn, zin.read(n))
            else:
                zout.writestr(n, zin.read(n))
        zout.writestr('maps/%s.ent' % mapname, ent)
    write_base_cfg(os.path.join(rundir, 'data', 'joinshot.cfg'), mapname, args.width, args.height)
    step_path = os.path.join(rundir, 'data', 'js_step.cfg')
    open(step_path, 'w').write('// waiting for spawn\n')     # empty until spawn detected
    step_seq, shot_budget = build_step(shots, args.step)

    log = os.path.join(rundir, 'run.log')
    hard = int(args.settle + shot_budget + 30)
    cmd = [bin_, '-basedir', args.xonotic, '-userdir', rundir, '-nosound', '-noconfig',
           '+vid_soft', '1', '+vid_fullscreen', '0',
           '+cl_curl_enabled', '0', '+sv_public', '0', '+exec', 'joinshot.cfg']
    env = dict(os.environ, SDL_VIDEODRIVER='dummy')
    print('booting client (software rasterizer, windowless); waiting for spawn...')
    t0 = time.time()
    lf = open(log, 'w')
    proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT,
                            preexec_fn=os.setsid)

    def logtext():
        try:
            return open(log, 'r', errors='replace').read()
        except OSError:
            return ''

    def kill():
        import signal as _sig
        for s in (_sig.SIGTERM, _sig.SIGKILL):
            try:
                os.killpg(os.getpgid(proc.pid), s)
            except Exception:
                pass
            if proc.poll() is not None:
                break

    armed = False
    join_nudged_at = None
    while True:
        if proc.poll() is not None:
            break
        txt = logtext()
        # single fallback join, once, a few seconds after connect if auto-join is slow
        if (not armed and join_nudged_at is None and 'is now playing' not in txt
                and ('changed name to' in txt or ') connected' in txt or ' connected\x1b' in txt
                     or 'connected' in txt)):
            join_nudged_at = time.time()
            open(step_path, 'w').write('togglemenu 0\ncmd join\n')
        if join_nudged_at and not armed and time.time() - join_nudged_at > 3:
            open(step_path, 'w').write('// waiting for spawn\n')   # stop nudging
            join_nudged_at = -1
        if not armed and 'is now playing' in txt:
            armed = True
            open(step_path, 'w').write(step_seq)   # hand the engine the shot sequence
            print('spawned after %.0fs; capturing %d frames...' % (time.time() - t0, len(shots)))
        if armed and 'JOINSHOT_DONE' in txt:
            for _ in range(30):
                if proc.poll() is not None:
                    break
                time.sleep(0.1)
            break
        if not armed and time.time() - t0 > args.settle:
            print('ERROR: player never spawned within %.0fs (map load too slow?)' % args.settle)
            break
        if time.time() - t0 > hard:
            print('ERROR: hard timeout'); break
        time.sleep(0.5)
    kill()
    lf.close()
    print('engine ran %.0fs' % (time.time() - t0))

    # collect PNGs (engine writes under <userdir>/data/ or data/screenshots/)
    found = 0
    for name in shots:
        src = None
        for cand in (os.path.join(rundir, 'data', name + '.png'),
                     os.path.join(rundir, 'data', 'screenshots', name + '.png')):
            if os.path.exists(cand):
                src = cand; break
        if src:
            shutil.copy(src, os.path.join(out, name + '.png')); found += 1
        else:
            print('  MISSING frame: %s' % name)
    print('captured %d/%d frames -> %s' % (found, len(shots), out))
    ok = True
    if not args.no_audit:
        ok = void_audit(out, shots)
    if not args.keep:
        shutil.rmtree(rundir, ignore_errors=True)
    else:
        print('run dir kept: %s (log: %s)' % (rundir, log))
    return 0 if (found and ok) else 1


if __name__ == '__main__':
    sys.exit(main())
