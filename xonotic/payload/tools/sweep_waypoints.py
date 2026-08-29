import sys, os, glob, subprocess, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mkentfile as M

pk3 = sys.argv[1] if len(sys.argv) > 1 else sorted(
    glob.glob(os.path.expanduser('~/dox/xonotic/Xonotic/data/*maps*.pk3')), reverse=True)[0]
kcarts = int(sys.argv[2]) if len(sys.argv) > 2 else 4

names = sorted(set(os.path.basename(l.split()[-1])[:-len('.waypoints')]
                   for l in subprocess.run(['unzip', '-l', pk3], capture_output=True, text=True).stdout.splitlines()
                   if l.strip().endswith('.waypoints')))

print('sweep %d maps carts=%d k=%d' % (len(names), kcarts, max(3, kcarts)))
print('%-24s %6s %5s %6s %6s %6s %6s %s' % ('map', 'nodes', 'orig', 'min', 'mean', 'max', 'ratio', 'note'))
worst = 0.0
fails = []
for name in names:
    txt = M.load_cache(name, pk3, pk3)
    if not txt:
        fails.append((name, 'no cache'))
        print('%-24s %6s %5s %6s %6s %6s %6s %s' % (name, '-', '-', '-', '-', '-', '-', 'FALLBACK no cache'))
        continue
    nodes, adj = M.parse_cache(txt)
    comp = M.largest_component(adj)
    if len(comp) < 3:
        fails.append((name, 'degenerate graph %d' % len(comp)))
        print('%-24s %6d %5s %6s %6s %6s %6s %s' % (name, len(nodes), '-', '-', '-', '-', '-', 'FALLBACK degenerate'))
        continue
    o, dm = M.kcenter(adj, max(3, kcarts))
    ws = [dm[o[i]][o[j]] for i in range(len(o)) for j in range(i + 1, len(o))]
    r = max(ws) / min(ws) if min(ws) else 0
    worst = max(worst, r)
    print('%-24s %6d %5d %6.0f %6.0f %6.0f %6.2f' %
          (name, len(nodes), len(o), min(ws), sum(ws) / len(ws), max(ws), r))
print('worst-case balance ratio across maps: %.2f' % worst)
if fails:
    print('fell back to spawn-origin method:')
    for n, why in fails:
        print('  %-24s %s' % (n, why))
else:
    print('all maps parsed and yielded >=3 equidistant origins')
