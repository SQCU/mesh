import re, sys

L = 5914.897461
SPEED = 40.0
rows = []
for line in open(sys.argv[1], errors='ignore'):
    m = re.search(r'plcdbg t=([\d.]+) s=([\d.]+) v=(-?[\d.]+) n=(\d+),(\d+),(\d+) g=([\d.]+),([\d.]+),([\d.]+)', line)
    if m:
        g = [float(m.group(7)), float(m.group(8)), float(m.group(9))]
        rows.append((float(m.group(1)), float(m.group(2)), float(m.group(3)),
                     [int(m.group(4)), int(m.group(5)), int(m.group(6))], g))
    m = re.search(r'plcdbg t=([\d.]+) s=([\d.]+) v=(-?[\d.]+) n=(\d+),(\d+) pl=', line)
    if m:
        rows.append((float(m.group(1)), float(m.group(2)), float(m.group(3)),
                     [int(m.group(4)), int(m.group(5)), 0], [0.0, L, -1.0]))

start = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
ok = bad = 0
push_samples = 0
coalition = 0
for i in range(1, len(rows)):
    t, s, v, n, g = rows[i]
    if t < start:
        continue
    if sum(n) == 0:
        continue
    push_samples += 1
    pplus = pminus = 0.0
    sides = set()
    for j in range(3):
        if g[j] < 0:
            continue
        w = min(n[j], 3)
        if w == 0:
            continue
        d = g[j] - s
        if d > 0:
            pplus += w
            sides.add(1)
        elif d < 0:
            pminus += w
            sides.add(-1)
    if len(sides) > 1:
        coalition += 1
    exp = max(-200.0, min(200.0, SPEED * (pplus - pminus)))
    if abs(exp - v) < 1e-3:
        ok += 1
    else:
        bad += 1
        if bad <= 8:
            print('MISMATCH t=%.2f s=%.1f n=%s g=%s expected v=%.1f got v=%.1f' % (t, s, n, g, exp, v))
print('samples=%d occupied=%d law_ok=%d law_mismatch=%d two_sided=%d' % (len(rows), push_samples, ok, bad, coalition))
