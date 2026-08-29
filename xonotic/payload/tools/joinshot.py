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
    args = ap.parse_args()

    mapdir = os.path.abspath(args.mapdir)
    joins = json.load(open(os.path.join(mapdir, 'fused.joins.json')))
    out = args.out or os.path.join(mapdir, 'joinshots')
    os.makedirs(out, exist_ok=True)

    cams, shots = [], []
    for i, jn in enumerate(joins['joins']):
        for name, eye, ang in cameras_for_join(i, jn):
            cams.append((name, eye, ang)); shots.append(name)
    print('%d joins -> %d camera frames' % (len(joins['joins']), len(cams)))

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
    if not args.keep:
        shutil.rmtree(rundir, ignore_errors=True)
    else:
        print('run dir kept: %s (log: %s)' % (rundir, log))
    return 0 if found else 1


if __name__ == '__main__':
    sys.exit(main())
