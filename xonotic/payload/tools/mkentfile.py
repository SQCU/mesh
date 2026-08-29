import struct, sys, re, math

bsp, out = sys.argv[1], sys.argv[2]
kteams = max(2, min(5, int(sys.argv[3]))) if len(sys.argv) > 3 else 2
kcarts = max(1, min(4, int(sys.argv[4]))) if len(sys.argv) > 4 else 2
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
if not visible:
    visible = models
spawns = [b for b in blocks if 'info_player_team1' in b or 'info_player_team2' in b]


def origin(b):
    m = re.search(r'"origin"\s+"([-\d. ]+)"', b)
    return [float(x) for x in m.group(1).split()] if m else None


pts = [origin(b) for b in spawns if origin(b)]
print('inline models:', models[:6], 'visible:', visible[:kcarts], 'team spawns:', len(pts))

spread = [max(p[a] for p in pts) - min(p[a] for p in pts) for a in (0, 1)]
split_axis = 0 if spread[0] >= spread[1] else 1
walk_axis = 1 - split_axis
pts.sort(key=lambda p: p[split_axis])

extra = []
tracks = []
for c in range(kcarts):
    lo = (len(pts) * c) // kcarts
    hi = (len(pts) * (c + 1)) // kcarts
    part = sorted(pts[lo:hi], key=lambda p: (p[walk_axis], p[split_axis]))
    idx = sorted({0, len(part) // 4, len(part) // 2, (3 * len(part)) // 4, len(part) - 1})
    while len(idx) < 5:
        idx.append(idx[-1])
    track = [[part[i][0], part[i][1], part[i][2] + 16] for i in idx[:5]]
    names = ['plc%dn%d' % (c, i) for i in range(5)]
    tracks.append((names, track))
    for i, (name, p) in enumerate(zip(names, track)):
        e = ['{', '"classname" "plc_path"', '"targetname" "%s"' % name,
             '"origin" "%.0f %.0f %.0f"' % tuple(p)]
        if i + 1 < 5:
            e.append('"target" "%s"' % names[i + 1])
        if i == 2:
            e.append('"spawnflags" "1"')
        e.append('}')
        extra.append('\n'.join(e))
    extra.append('\n'.join(['{', '"classname" "func_plc_cart"',
                            '"model" "%s"' % visible[c % len(visible)],
                            '"target" "%s"' % names[0], '"speed" "40"', '}']))

goal_cnts = [4, 13, 12, 9, 3][:kteams]
for t, cnt in enumerate(goal_cnts):
    names, track = tracks[t % kcarts]
    extra.append('\n'.join(['{', '"classname" "plc_goal"',
                            '"cnt" "%d"' % cnt, '"target" "%s"' % names[4],
                            '"radius" "64"',
                            '"origin" "%.0f %.0f %.0f"' % tuple(track[4]),
                            '}']))

for c, (names, track) in enumerate(tracks):
    L = sum(math.dist(track[i], track[i + 1]) for i in range(4))
    print('cart %d: %s -> %s length %.0f' % (c, track[0], track[4], L))
sep = min(math.dist(pa, pb) for _, ta in tracks[:1] for pa in ta
          for _, tb in tracks[1:] for pb in tb) if kcarts > 1 else -1
print('teams', kteams, 'carts', kcarts, 'min inter-track node distance %.0f' % sep)

open(out, 'w').write(ents.rstrip('\0') + '\n' + '\n'.join(extra) + '\n')
print('wrote', out)
