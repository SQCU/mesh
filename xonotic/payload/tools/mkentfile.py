import struct, sys, re

bsp, out = sys.argv[1], sys.argv[2]
d = open(bsp, 'rb').read()
assert d[:4] == b'IBSP', d[:4]
off, ln = struct.unpack_from('<ii', d, 8)
ents = d[off:off + ln].split(b'\0')[0].decode('latin-1')

blocks = re.findall(r'\{[^{}]*\}', ents)
mclass = {}
for b in blocks:
    m = re.search(r'"model"\s+"(\*\d+)"', b)
    if m:
        mclass[m.group(1)] = re.search(r'"classname"\s+"([^"]+)"', b).group(1)
models = sorted(mclass, key=lambda s: int(s[1:]))
visible = [m for m in models if not mclass[m].startswith('trigger_')]
cart_model = (visible or models)[0]
spawns = [b for b in blocks if 'info_player_team1' in b or 'info_player_team2' in b]


def origin(b):
    m = re.search(r'"origin"\s+"([-\d. ]+)"', b)
    return [float(x) for x in m.group(1).split()] if m else None


pts = [origin(b) for b in spawns if origin(b)]
print('inline models:', models[:6], 'cart model:', cart_model, mclass[cart_model], 'team spawns:', len(pts))

pts.sort(key=lambda p: (p[0], p[1]))
idx = [0, len(pts) // 4, len(pts) // 2, (3 * len(pts)) // 4, len(pts) - 1]
track = [[pts[i][0], pts[i][1], pts[i][2] + 16] for i in idx]

k = max(2, min(5, int(sys.argv[3]))) if len(sys.argv) > 3 else 2

chain = [('plcn%d' % i, p) for i, p in enumerate(track)]
start = 'plcn2'
if k == 5:
    mid = [(track[1][a] + track[2][a]) / 2 for a in range(3)]
    chain = chain[:2] + [('plcs', mid)] + chain[2:]
    start = 'plcs'

pos = dict(chain)

extra = []
for i, (name, p) in enumerate(chain):
    e = ['{', '"classname" "plc_path"', '"targetname" "%s"' % name,
         '"origin" "%.0f %.0f %.0f"' % tuple(p)]
    if i + 1 < len(chain):
        e.append('"target" "%s"' % chain[i + 1][0])
    if name == start:
        e.append('"spawnflags" "1"')
    e.append('}')
    extra.append('\n'.join(e))

extra.append('\n'.join(['{', '"classname" "func_plc_cart"',
                        '"model" "%s"' % cart_model,
                        '"target" "plcn0"', '"plc_start" "%s"' % start,
                        '"speed" "40"', '}']))
goals = [(4, 'plcn0'), (13, 'plcn4'), (12, 'plcn1'), (9, 'plcn3'), (3, 'plcn2')][:k]
for cnt, node in goals:
    extra.append('\n'.join(['{', '"classname" "plc_goal"',
                            '"cnt" "%d"' % cnt, '"target" "%s"' % node,
                            '"radius" "64"',
                            '"origin" "%.0f %.0f %.0f"' % tuple(pos[node]),
                            '}']))

import math
arc = {}
s = 0.0
for i, (name, p) in enumerate(chain):
    if i:
        q = chain[i - 1][1]
        s += math.dist(p, q)
    arc[name] = s
print('teams', k, 'start', start, 's=%.0f' % arc[start],
      'goal s:', {n: round(arc[n]) for _, n in goals},
      'min |start-goal|', round(min(abs(arc[n] - arc[start]) for _, n in goals)))

open(out, 'w').write(ents.rstrip('\0') + '\n' + '\n'.join(extra) + '\n')
print('wrote', out, 'track', track[0], '->', track[-1])
