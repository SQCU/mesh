#!/usr/bin/env mesh-python
import os, sys, json, math, struct, zipfile, shutil, subprocess, argparse, tempfile, time

EYE = 40.0
BACK = 176.0
LOOK = 320.0

def vectoangles_view(dx, dy, dz):
    yaw = math.degrees(math.atan2(dy, dx))
    horiz = math.hypot(dx, dy)
    pitch = -math.degrees(math.atan2(dz, horiz))
    return pitch, yaw, 0.0

def cameras_for_join(idx, jn):
    sa, sb = jn['sa'], jn['sb']
    kind = jn['kind']
    dx, dy = sb[0] - sa[0], sb[1] - sa[1]
    hlen = math.hypot(dx, dy) or 1.0
    ux, uy = dx / hlen, dy / hlen
    tag = 'j%02d_%s' % (idx, kind)
    if kind == 'corridor':

        ea = [sa[0], sa[1], sa[2] + EYE]
        aa = [sb[0], sb[1], sb[2] + EYE]
        yield ('%s_a_through' % tag, ea, vectoangles_view(aa[0]-ea[0], aa[1]-ea[1], aa[2]-ea[2]))
        eb = [sb[0], sb[1], sb[2] + EYE]
        ab = [sa[0], sa[1], sa[2] + EYE]
        yield ('%s_b_through' % tag, eb, vectoangles_view(ab[0]-eb[0], ab[1]-eb[1], ab[2]-eb[2]))
    else:

        ea = [sa[0] - ux * BACK, sa[1] - uy * BACK, sa[2] + EYE]
        aa = [sa[0], sa[1], sa[2] + EYE * 0.5]
        yield ('%s_a_approach' % tag, ea, vectoangles_view(aa[0]-ea[0], aa[1]-ea[1], aa[2]-ea[2]))
        eb = [sb[0], sb[1], sb[2] + EYE]
        ab = [sb[0] + ux * LOOK, sb[1] + uy * LOOK, sb[2] + EYE]
        yield ('%s_b_landing' % tag, eb, vectoangles_view(ab[0]-eb[0], ab[1]-eb[1], ab[2]-eb[2]))

def cameras_for_portal(idx, pt):
    node, mouth = pt['node'], pt['mouth']
    ax, sgn = pt['axis'], pt['sgn']
    d = [0.0, 0.0, 0.0]
    d[ax] = sgn
    tag = 'p%02d_%s_%s' % (idx, pt['name'][:12], pt['kind'])

    ea = [node[0] - d[0] * 240.0, node[1] - d[1] * 240.0, node[2] + EYE]
    aa = [mouth[0], mouth[1], mouth[2] + 72.0]
    yield ('%s_in' % tag, ea, vectoangles_view(aa[0] - ea[0], aa[1] - ea[1], aa[2] - ea[2]))

    eb = [mouth[0] + d[0] * 264.0, mouth[1] + d[1] * 264.0, mouth[2] + EYE]
    ab = [node[0], node[1], node[2] + 72.0]
    yield ('%s_out' % tag, eb, vectoangles_view(ab[0] - eb[0], ab[1] - eb[1], ab[2] - eb[2]))

def cameras_for_region(idx, mp):
    name = ''.join(ch if ch.isalnum() else '_' for ch in mp.get('name', 'r%d' % idx))
    for vi, v in enumerate(mp.get('vantages', [])[:2]):
        for yi, yaw in enumerate((0.0, 90.0, 180.0, 270.0)):
            yield ('r%02d_%s_v%d_y%d' % (idx, name, vi, int(yaw)),
                   [v[0], v[1], v[2] + EYE], (0.0, yaw, 0.0))

def cameras_overview(joins):
    mins = [min(m['mins'][a] for m in joins['maps']) for a in range(3)]
    maxs = [max(m['maxs'][a] for m in joins['maps']) for a in range(3)]
    cx, cy = (mins[0] + maxs[0]) / 2, (mins[1] + maxs[1]) / 2
    yield ('ov_world', [cx, cy, maxs[2] + 512.0], (89.0, 0.0, 0.0))
    for i, m in enumerate(joins['maps']):
        mx = (m['mins'][0] + m['maxs'][0]) / 2
        my = (m['mins'][1] + m['maxs'][1]) / 2
        yield ('ov%02d' % i, [mx, my, m['maxs'][2] + 256.0], (89.0, 0.0, 0.0))

def _defilter(raw, pos, pw, ph, nch):
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

def _tga_to_png(path):
    import struct as _s
    import zlib as _z
    d = open(path, 'rb').read()
    if d[:4] == b'\x89PNG':
        return False
    if len(d) < 18 or d[2] != 2:
        return False
    idlen = d[0]
    w, h = _s.unpack_from('<HH', d, 12)
    bpp = d[16]
    desc = d[17]
    B = bpp // 8
    px = d[18 + idlen:]
    rows = []
    for y in range(h):
        sy = y if (desc & 0x20) else (h - 1 - y)
        o = sy * w * B
        row = bytearray(b'\x00')
        for x in range(w):
            b_, g_, r_ = px[o + x * B], px[o + x * B + 1], px[o + x * B + 2]
            row += bytes((r_, g_, b_))
        rows.append(bytes(row))
    raw = b''.join(rows)

    def chunk(tag, data):
        c = _s.pack('>I', len(data)) + tag + data
        return c + _s.pack('>I', _z.crc32(tag + data) & 0xffffffff)
    png = (b'\x89PNG\r\n\x1a\n'
           + chunk(b'IHDR', _s.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0))
           + chunk(b'IDAT', _z.compress(raw, 9))
           + chunk(b'IEND', b''))
    open(path, 'wb').write(png)
    return True

def frame_visual_measures(outdir, shots):
    rows = []
    missing = []
    unreadable = []
    for name in shots:
        fp = os.path.join(outdir, name + '.png')
        if not os.path.exists(fp):
            rows.append((name, None))
            missing.append(name)
            continue
        try:
            st = frame_stats(fp)
        except Exception as e:
            rows.append((name, None))
            unreadable.append({'name': name, 'error': '%s: %s' % (type(e).__name__, e)})
            continue
        rows.append((name, st))
    observed = [st for _, st in rows if st is not None]
    void = [float(st['void']) for st in observed]
    levels = [int(st['levels']) for st in observed]
    record = {
        'schema': 1,
        'requested_frame_mass': len(shots),
        'observed_frame_mass': len(observed),
        'missing_frame_mass': len(missing),
        'missing_frames': missing,
        'unreadable_frame_mass': len(unreadable),
        'unreadable_frames': unreadable,
        'void_fraction_measure': {
            'mass': len(void),
            'minimum': min(void) if void else None,
            'mean': sum(void) / len(void) if void else None,
            'maximum': max(void) if void else None,
            'variance': sum((value - sum(void) / len(void)) ** 2 for value in void) / len(void) if void else None,
        },
        'color_level_measure': {
            'mass': len(levels),
            'minimum': min(levels) if levels else None,
            'mean': sum(levels) / len(levels) if levels else None,
            'maximum': max(levels) if levels else None,
            'variance': sum((value - sum(levels) / len(levels)) ** 2 for value in levels) / len(levels) if levels else None,
        },
        'frames': {name: st for name, st in rows},
    }
    print('frame visual measures: requested=%d observed=%d missing=%d unreadable=%d' %
          (len(shots), len(observed), len(missing), len(unreadable)))
    for name, st in rows:
        if st:
            print('  %-40s void=%.2f levels=%3d %dx%d' % (name, st['void'], st['levels'],
                                                          st['w'], st['h']))
    with open(os.path.join(outdir, 'frame-visual-measures.json'), 'w') as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write('\n')
    return record

def read_base_ent(mapdir):
    pk3 = os.path.join(mapdir, 'fused.pk3')
    if os.path.exists(pk3):
        with zipfile.ZipFile(pk3) as z:
            for n in z.namelist():
                if n.endswith('fused.ent'):
                    return z.read(n).decode('latin-1')
    bsp = os.path.join(mapdir, 'fused.bsp')
    d = open(bsp, 'rb').read()
    lo, ln = struct.unpack_from('<ii', d, 8)
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
    L = ['cl_allow_uid2name 0', 'cl_allow_uidtracking 0',
         'sv_cheats 1', 'sv_spectate 0', 'g_warmup 0', 'g_max_info_autoscreenshot 999',
         'sv_clientcommand_antispam_time 0', 'sv_clientcommand_antispam_count 9999',
         'bot_number 0', 'timelimit 0', 'g_maxplayers 0',

         'g_playerstats_gamereport_uri ""', 'g_playerstats_playerbasic_uri ""',
         'g_playerstats_playerdetail_uri ""', 'sv_eventlog 0',
         'scr_screenshot_png 1', 'scr_screenshot_timestamp 0', 'scr_screenshot_gammaboost 1',
         'r_texture_dds_load 0', 'gl_texturecompression 0',

         'gl_picmip 3', 'r_texture_max_size 128', 'r_lerpimages 0',
         'r_drawviewmodel 0', 'crosshair 0', 'con_notify 0', 'scr_centertime 0',
         'cl_deathscoreboard 0', 'r_bloom 0', 'r_motionblur 0', 'r_damageblur 0',
         'vid_width %d' % w, 'vid_height %d' % h]
    for p in HUD_OFF:
        L.append('hud_panel_%s 0' % p)
    L.append('map %s' % mapname)

    L.append('alias js_poll "exec js_step.cfg; defer 1 js_poll"')
    L.append('defer 6 js_poll')
    open(path, 'w').write('\n'.join(L) + '\n')

def build_step(shots, step):
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
    ap.add_argument('--measure-only', action='store_true',
                    help='do not run the engine; measure PNGs already in --out')
    ap.add_argument('--no-measures', action='store_true')
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
    if args.measure_only:
        for _n in shots:
            _p = os.path.join(out, _n + '.png')
            if os.path.exists(_p):
                _tga_to_png(_p)
        frame_visual_measures(out, shots)
        return 0

    bin_ = os.environ.get('JOINSHOT_CLIENT') or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', '..', 'darkplaces-work', 'darkplaces-sdl')
    bin_ = os.path.normpath(bin_)
    if not os.path.exists(bin_):
        bin_ = os.path.join(args.xonotic, 'Xonotic.app/Contents/MacOS/xonotic-osx-sdl-bin')
    if not os.path.exists(bin_):
        sys.exit('client binary not found: %s' % bin_)
    print('client: %s' % bin_)

    rundir = tempfile.mkdtemp(prefix='joinshot_')
    os.makedirs(os.path.join(rundir, 'data', 'maps'), exist_ok=True)

    mapname = 'joinshotmap'
    ent = build_ent(read_base_ent(mapdir), cams)
    src_pk3 = os.path.join(mapdir, 'fused.pk3')
    dst_pk3 = os.path.join(rundir, 'data', 'zzz-%s.pk3' % mapname)
    with zipfile.ZipFile(src_pk3) as zin, zipfile.ZipFile(dst_pk3, 'w', zipfile.ZIP_DEFLATED) as zout:
        for n in zin.namelist():
            base = os.path.basename(n)
            if base == 'fused.ent':
                continue
            if base.startswith('fused.'):
                nn = 'maps/%s.%s' % (mapname, base.split('.', 1)[1])
                zout.writestr(nn, zin.read(n))
            else:
                zout.writestr(n, zin.read(n))
        zout.writestr('maps/%s.ent' % mapname, ent)
    write_base_cfg(os.path.join(rundir, 'data', 'joinshot.cfg'), mapname, args.width, args.height)
    step_path = os.path.join(rundir, 'data', 'js_step.cfg')
    open(step_path, 'w').write('')
    step_seq, shot_budget = build_step(shots, args.step)

    log = os.path.join(rundir, 'run.log')
    hard = int(args.settle + shot_budget + 30)

    cmd = [bin_, '-xonotic', '-basedir', args.xonotic, '-userdir', rundir, '-nosound', '-noconfig',
           '+vid_width', str(args.width), '+vid_height', str(args.height),
           '+vid_fullscreen', '0',
           '+cl_curl_enabled', '0', '+sv_public', '0', '+exec', 'joinshot.cfg']
    env = dict(os.environ)
    if os.environ.get('JOINSHOT_SOFT') == '1':
        cmd[6:6] = ['+vid_soft', '1']
        env['SDL_VIDEODRIVER'] = 'dummy'
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

    def terminate():
        import signal as _sig
        try:
            os.killpg(os.getpgid(proc.pid), _sig.SIGTERM)
        except Exception as error:
            print(json.dumps({"event":"client_signal_error","pid":proc.pid,"signal":int(_sig.SIGTERM),"error":f"{type(error).__name__}: {error}"}), file=sys.stderr)
        for _ in range(120):
            if proc.poll() is not None:
                break
            time.sleep(0.25)
        if proc.poll() is None:
            print(json.dumps({"event":"client_termination_pending","pid":proc.pid}), file=sys.stderr)

    armed = False
    join_nudged_at = None
    while True:
        if proc.poll() is not None:
            break
        txt = logtext()

        if (not armed and join_nudged_at is None and 'is now playing' not in txt
                and ('changed name to' in txt or ') connected' in txt or ' connected\x1b' in txt
                     or 'connected' in txt)):
            join_nudged_at = time.time()
            open(step_path, 'w').write('togglemenu 0\ncmd join\n')
        if join_nudged_at and not armed and time.time() - join_nudged_at > 3:
            open(step_path, 'w').write('')
            join_nudged_at = -1
        if not armed and 'is now playing' in txt:
            armed = True
            open(step_path, 'w').write(step_seq)
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
    terminate()
    lf.close()
    print('engine ran %.0fs' % (time.time() - t0))

    found = 0
    for name in shots:
        src = None
        for cand in (os.path.join(rundir, 'data', name + '.png'),
                     os.path.join(rundir, 'data', 'screenshots', name + '.png')):
            if os.path.exists(cand):
                src = cand; break
        if src:
            dst = os.path.join(out, name + '.png')
            shutil.copy(src, dst)
            _tga_to_png(dst)
            found += 1
        else:
            print('  MISSING frame: %s' % name)
    print('captured %d/%d frames -> %s' % (found, len(shots), out))
    if not args.no_measures:
        frame_visual_measures(out, shots)
    if not args.keep:
        shutil.rmtree(rundir, ignore_errors=True)
    else:
        print('run dir kept: %s (log: %s)' % (rundir, log))
    return 0

if __name__ == '__main__':
    sys.exit(main())
