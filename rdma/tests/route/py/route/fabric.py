"""Fabric files (tests/fabric-*.txt): `node NAME compute|relay`, `link FROM TO GBps us cables` (GB/s is bytes a ns;
latency in us, stored in ns), `duplex SHARE`, `#` comments. A fabric is one struct route_fabric record of
native.dtype, so an array of them is the C array the simulator and plugins read in place."""
import numpy as np

from .native import const, dtype

FABRIC = dtype('struct route_fabric')


def load(path):
    f = np.zeros((), FABRIC)
    f['duplex'] = 1
    names = []
    for line in open(path):
        w = line.split()
        if not w or w[0].startswith('#'):
            continue
        if w[0] == 'duplex':
            f['duplex'] = float(w[1])
        elif w[0] == 'node':
            if len(names) == const('ROUTE_N'):
                raise ValueError(f'{path}: more than ROUTE_N nodes')
            if w[2] == 'compute':
                f['compute'][f['ncompute']] = len(names); f['ncompute'] += 1
            names.append(w[1]); f['nodes'] += 1
        elif w[0] == 'link':
            l = int(f['lanes'])
            if l == const('ROUTE_L'):
                raise ValueError(f'{path}: more than ROUTE_L lanes')
            f['from'][l], f['to'][l] = names.index(w[1]), names.index(w[2])
            f['rate'][l], f['latency'][l], f['cables'][l] = float(w[3]), float(w[4]) * 1e3, int(w[5])
            f['lanes'] += 1
    return f


# ---- the training fabrics (formerly sim.c route_network): one of six kinds, each equally likely ----
def _cabled(rng, n, compute, pairs):
    """n nodes, `compute` the ranks, each pair cabled both ways: 1-4 cables of 4-12 GB/s each, 2-13 us, a duplex share
    of 0.85-1 for the fabric"""
    f = np.zeros((), FABRIC)
    f['nodes'] = n; f['ncompute'] = len(compute); f['compute'][:len(compute)] = compute
    f['duplex'] = 0.85 + 0.15 * rng.random()
    for u, v in pairs:
        c = 1 + int(rng.random() * 4); each = 4 + 8 * rng.random()
        for a, b in ((u, v), (v, u)):
            l = int(f['lanes'])
            f['from'][l], f['to'][l], f['rate'][l], f['latency'][l], f['cables'][l] = a, b, c * each, 2e3 + 11e3 * rng.random(), c
            f['lanes'] += 1
    return f


def random_fabric(rng, measured):
    """kind 0: a random connected graph of degree at most 4 with up to two relays; 1: a ring (a path of 3 for n 3);
    2: a star, its hub a rank or a relay; 3: a ring of ranks and a relay hub cabled to three of them; 4: a full mesh;
    5: the measured cluster, every lane's rate and latency jittered by exp(0.3 N(0, 1)). n: 3-6 nodes."""
    kind = int(rng.random() * 6); n = 3 + int(rng.random() * 4)
    N = const('ROUTE_N')
    if kind == 0:
        pairs, degree, joined = [], [0] * n, set()
        for v in range(1, n):
            u = int(rng.random() * v); pairs.append((u, v)); joined.add((u, v)); degree[u] += 1; degree[v] += 1
        for u in range(n):
            for v in range(u + 1, n):
                if (u, v) not in joined and degree[u] < 4 and degree[v] < 4 and rng.random() < 0.35:
                    pairs.append((u, v)); degree[u] += 1; degree[v] += 1
        relays = min(int(rng.random() * 3), n - 2)
        return _cabled(rng, n, list(range(n - relays)), pairs)
    if kind == 1:
        pairs = [(v, (v + 1) % n) for v in range(n)]
        return _cabled(rng, n, list(range(n)), pairs[:-1] if n == 3 else pairs)
    if kind == 2:
        hub = rng.random() < 0.5
        return _cabled(rng, n, list(range(1 if hub else 0, n)), [(0, v) for v in range(1, n)])
    if kind == 3:
        m = min(max(n - 1, 3), N - 1)
        return _cabled(rng, m + 1, list(range(m)), [(v, (v + 1) % m) for v in range(m)] + [(m, v) for v in range(min(3, m))])
    if kind == 4:
        return _cabled(rng, n, list(range(n)), [(u, v) for u in range(n) for v in range(u + 1, n)])
    f = measured.copy()
    for l in range(int(f['lanes'])):   # rate, then latency, a normal each, lane by lane
        f['rate'][l] *= np.exp(0.3 * rng.standard_normal()); f['latency'][l] *= np.exp(0.3 * rng.standard_normal())
    return f
