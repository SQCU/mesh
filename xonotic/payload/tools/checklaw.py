import re, sys

SPEED = 40.0
CAP = 3
MAXV = 200.0
rows = []
for line in open(sys.argv[1], errors='ignore'):
    m = re.search(r'plcdbg t=([\d.]+) cart=(\d+) s=([\d.]+) v=(-?[\d.]+) ctrl=(\d+)'
                  r' n=(\d+),(\d+),(\d+),(\d+) live=(\d+)', line)
    if m:
        rows.append((float(m.group(1)), int(m.group(2)), float(m.group(3)),
                     float(m.group(4)), int(m.group(5)),
                     [int(m.group(i)) for i in range(6, 10)], int(m.group(10))))

start = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
ok = bad = occupied = contested = regress = 0
for t, cart, s, v, ctrl, n, live in rows:
    if t < start or not live:
        continue
    w = [min(x, CAP) for x in n]
    if sum(w) == 0:
        continue
    occupied += 1
    best = max(w)
    exp_ctrl = w.index(best) + 1 if w.count(best) == 1 else 0
    pplus = w[exp_ctrl - 1] if exp_ctrl else 0
    exp = max(-MAXV, min(MAXV, SPEED * (pplus - (sum(w) - pplus))))
    if sum(1 for x in w if x) > 1:
        contested += 1
    if v < 0:
        regress += 1
    if exp_ctrl == ctrl and abs(exp - v) < 1e-3:
        ok += 1
    else:
        bad += 1
        if bad <= 8:
            print('MISMATCH t=%.2f cart=%d s=%.1f n=%s expected ctrl=%d v=%.1f got ctrl=%d v=%.1f'
                  % (t, cart, s, n, exp_ctrl, exp, ctrl, v))
print('samples=%d occupied=%d law_ok=%d law_mismatch=%d contested=%d regressing=%d'
      % (len(rows), occupied, ok, bad, contested, regress))
