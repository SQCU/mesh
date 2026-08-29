import sys, os, glob, subprocess, io, re, contextlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mkentfile as M

pk3 = sys.argv[1] if len(sys.argv) > 1 else sorted(
    glob.glob(os.path.expanduser('~/dox/xonotic/Xonotic/data/*maps*.pk3')), reverse=True)[0]
kcarts = int(sys.argv[2]) if len(sys.argv) > 2 else 4
work = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.environ.get('TMPDIR', '/tmp'), 'plcsweep')
os.makedirs(work, exist_ok=True)

names = sorted(set(os.path.basename(l.split()[-1])[:-len('.waypoints')]
                   for l in subprocess.run(['unzip', '-l', pk3], capture_output=True, text=True).stdout.splitlines()
                   if l.strip().endswith('.waypoints')))

print('sweep %d maps carts=%d push_r=%.0f push_h=%.0f' % (len(names), kcarts, M.PUSH_R, M.PUSH_H))
rows, clean, admitted_maps, failed = [], [], [], []
for name in names:
    base = name[:-5] if name.endswith('.race') else name
    bsp = os.path.join(work, name + '.bsp')
    if not os.path.exists(bsp):
        for cand in (name, base):
            r = subprocess.run(['unzip', '-p', pk3, 'maps/%s.bsp' % cand], capture_output=True)
            if r.returncode == 0 and r.stdout[:4] == b'IBSP':
                open(bsp, 'wb').write(r.stdout)
                break
    if not os.path.exists(bsp):
        failed.append((name, 'no bsp'))
        print('%-24s NO BSP' % name)
        continue
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            M.emit(bsp, os.path.join(work, name + '.ent'), 5, kcarts, pk3)
        out = buf.getvalue()
    except Exception as e:
        failed.append((name, str(e)))
        print('%-24s EMIT FAILED: %s' % (name, e))
        continue
    g = lambda pat: re.search(pat, out)
    mb = g(r'bad=(\d+) \(gen=(\d+) flag=(\d+) trig=(\d+) solid=(\d+) fly=(\d+) nofloor=(\d+)\)')
    ma = g(r'admitted_total=(\d+)')
    mv = g(r'validation samples=(\d+) solid_viol=(\d+) float_viol=(\d+) corridor_viol=(\d+) exempt_segs=(\d+) exempt_corridor_viol=(\d+) (\w+)')
    mw = g(r'walk min=(\d+)')
    mf = g(r'FALLBACK')
    if mf or not mb or not mv:
        failed.append((name, 'fallback or missing stats'))
        print('%-24s FALLBACK/PARTIAL' % name)
        continue
    adm = int(ma.group(1)) if ma else 0
    row = (name, mb.group(1), mb.group(2), mb.group(3), mb.group(4),
           mb.group(5), mb.group(6), mb.group(7), adm,
           mv.group(1), mv.group(2), mv.group(3), mv.group(4), mv.group(6), mv.group(7))
    rows.append(row)
    ok = mv.group(7) == 'PASS'
    if adm:
        admitted_maps.append(name)
    if ok and not adm:
        clean.append(name)
    print('%-24s bad=%-4s (gen=%-3s flag=%-2s trig=%-2s sol=%-3s fly=%-3s nf=%-2s) adm=%-2d samp=%-5s viol s/f/c=%s/%s/%s exC=%s %s' % row)

print('zero-violation clean maps, no admitted edges (%d): %s' % (len(clean), ' '.join(clean)))
print('maps needing admitted bad edges (%d): %s' % (len(admitted_maps), ' '.join(admitted_maps) or 'none'))
print('failed (%d): %s' % (len(failed), failed or 'none'))
bad_valid = [r[0] for r in rows if r[10] != '0' or r[11] != '0' or r[12] != '0']
print('maps with validation violations (%d): %s' % (len(bad_valid), ' '.join(bad_valid) or 'none'))
