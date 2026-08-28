import struct, sys, re

bsp, out = sys.argv[1], sys.argv[2]
d = open(bsp, 'rb').read()
assert d[:4] == b'IBSP', d[:4]
off, ln = struct.unpack_from('<ii', d, 8)
ents = d[off:off + ln].split(b'\0')[0].decode('latin-1')

blocks = re.findall(r'\{[^{}]*\}', ents)
models = sorted({m for b in blocks for m in re.findall(r'"model"\s+"(\*\d+)"', b)},
                key=lambda s: int(s[1:]))
spawns = [b for b in blocks if 'info_player_team1' in b or 'info_player_team2' in b]


def origin(b):
    m = re.search(r'"origin"\s+"([-\d. ]+)"', b)
    return [float(x) for x in m.group(1).split()] if m else None


pts = [origin(b) for b in spawns if origin(b)]
print('inline models:', models[:6], 'team spawns:', len(pts))

pts.sort(key=lambda p: (p[0], p[1]))
idx = [0, len(pts) // 4, len(pts) // 2, (3 * len(pts)) // 4, len(pts) - 1]
track = [[pts[i][0], pts[i][1], pts[i][2] + 16] for i in idx]

extra = []
for i, p in enumerate(track):
    e = ['{', '"classname" "plc_path"', '"targetname" "plcn%d"' % i,
         '"origin" "%.0f %.0f %.0f"' % tuple(p)]
    if i + 1 < len(track):
        e.append('"target" "plcn%d"' % (i + 1))
    if i == 2:
        e.append('"spawnflags" "1"')
    e.append('}')
    extra.append('\n'.join(e))

extra.append('\n'.join(['{', '"classname" "func_plc_cart"',
                        '"model" "%s"' % models[0],
                        '"target" "plcn0"', '"plc_start" "plcn2"',
                        '"speed" "40"', '}']))
goals = [(4, 'plcn0', 0), (13, 'plcn4', 4)]
if len(sys.argv) > 3 and sys.argv[3] == '3':
    goals.append((12, 'plcn1', 1))
for cnt, node, ti in goals:
    extra.append('\n'.join(['{', '"classname" "plc_goal"',
                            '"cnt" "%d"' % cnt, '"target" "%s"' % node,
                            '"radius" "64"',
                            '"origin" "%.0f %.0f %.0f"' % tuple(track[ti]),
                            '}']))

open(out, 'w').write(ents.rstrip('\0') + '\n' + '\n'.join(extra) + '\n')
print('wrote', out, 'track', track[0], '->', track[-1])
