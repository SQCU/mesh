# The routing policy of design/switched-multipath-dma.md, trained on the GPU against a batched fluid model of the
# fabric (MLX). A first pass: the fluid model is differentiable, so gradients come from backpropagation through the
# whole rollout, not from perturbations.
#
# The fluid model, B scenarios in lockstep, in time slots (each scenario's slot width scaled to its floor):
# - a scenario is a random fabric (3 to 6 nodes, up to 2 relays only, connected, random rates and latencies, random
#   background bytes queued on some lanes at the start) and a transaction (all-gather or all-reduce, 6 KB to 3 MB a
#   rank, 16 KiB pages);
# - each commodity (a source and its destinations) has K candidate trees: the shortest-path tree under the lanes'
#   unloaded delays, and K - 1 shortest-path trees under randomly perturbed delays;
# - each slot, every lane serves the bytes waiting on it, from every tree edge on it and its background, by processor
#   sharing at its rate; bytes sent reach the next node after the lane's latency and one page time (store and forward);
# - a source releases its commodity's bytes into its trees at its out-lanes' rate, split by the policy's weighting at
#   the time, refreshed every REFRESH slots; an all-reduce's owner releases its reduced bytes as the slowest of its
#   incoming partials lands;
# - the objective is a smooth maximum over (commodity, destination) of the area above its delivered fraction (the
#   time it waits, on average, for its bytes), over the floor.
#
# The policy: a 64-wide residual stream through two layers of a GRU block and a feed-forward block expanding to 192,
# recurrent over refreshes. Each candidate tree reaches it as N low-bit node tokens (parent, in-lane queue level, rate
# band), embedded with RoPE positions at the embedding only, pooled, and one feed-forward block; its logit is that
# dotted with the stream.
#
# Reported each LOG steps, without gradients, on held-out scenarios: the true completion (the first slot at which every
# destination holds every byte) over the floor, for the policy, for all traffic on the unloaded shortest-path tree, and
# for a uniform split over the candidates; and steps a second.
#
# usage: python switch_mlx.py [--batch 128] [--steps 2000] [--kind ag|ar|both]
import argparse, math, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

N, L, K, PAGE = 6, 30, 8, 16384.0   # nodes, lanes, candidates a commodity, page bytes
SLOTS, REFRESH, H = 480, 8, 16      # slots a rollout, slots between refreshes, the latency ring's depth
CTX, D, E = 12, 64, 192
VOCAB = (N + 2) * 16 * 4   # a node's token: its parent (a node, the source, or out), its in-lane's queue level, its rate band

def scenario(rng, kind):
    n = int(rng.integers(3, 7)); relays = min(int(rng.integers(0, 3)), n - 2); compute = list(range(n - relays))
    joined = np.zeros((n, n), bool); degree = np.zeros(n, int)
    for v in range(1, n):
        u = int(rng.integers(0, v)); joined[u, v] = joined[v, u] = True; degree[u] += 1; degree[v] += 1
    for u in range(n):
        for v in range(u + 1, n):
            if not joined[u, v] and degree[u] < 4 and degree[v] < 4 and rng.random() < 0.35:
                joined[u, v] = joined[v, u] = True; degree[u] += 1; degree[v] += 1
    src, dst, rate, lat = [], [], [], []
    for u in range(n):
        for v in range(u + 1, n):
            if joined[u, v]:
                c = int(rng.integers(1, 5))
                for a, b in ((u, v), (v, u)):
                    src.append(a); dst.append(b); rate.append(c * rng.uniform(4, 12)); lat.append(rng.uniform(2e3, 13e3))
    nl = len(src); rate = np.array(rate); lat = np.array(lat)
    background = np.where(rng.random(nl) < 0.3, rate * rng.uniform(0, 50e3, nl), 0.0)
    S = float(round(math.exp(math.log(6144) + rng.random() * (math.log(3145728) - math.log(6144)))))
    q = min(S, PAGE); nc = len(compute)
    # commodities: (source, destination set, demand, phase); phase 2 is an all-reduce owner's reduced pages
    comm = []
    if kind == 0:
        for s in compute: comm.append((s, [d for d in compute if d != s], S, 0, -1))
    else:
        for o in compute:
            for s in compute:
                if s != o: comm.append((s, [o], S / nc, 1, o))
        for o in compute: comm.append((o, [d for d in compute if d != o], S / nc, 2, o))
    # the floor: the cut bound and the slowest pair's shortest delay for one piece
    need = (nc - 1) * S if kind == 0 else 2 * (nc - 1) * S / nc; give = S if kind == 0 else need
    cut = 0.0
    for v in compute:
        inr = rate[[i for i in range(nl) if dst[i] == v]].sum(); outr = rate[[i for i in range(nl) if src[i] == v]].sum()
        cut = max(cut, need / inr, give / outr)
    dist = np.full((n, n), np.inf); np.fill_diagonal(dist, 0)
    for i in range(nl): dist[src[i], dst[i]] = min(dist[src[i], dst[i]], lat[i] + q / rate[i])
    for m in range(n): dist = np.minimum(dist, dist[:, m:m + 1] + dist[m:m + 1, :])
    floor = max(cut, max(dist[a, b] for a in compute for b in compute if a != b))
    # candidate trees: shortest-path trees under unloaded delays (k = 0) and randomly perturbed ones
    trees = []
    for (s, ds, dem, ph, o) in comm:
        cand = []
        for k in range(K):
            w = (lat + q / rate) * (1 if k == 0 else np.exp(0.6 * rng.standard_normal(nl)))
            at = np.full(n, np.inf); by = np.full(n, -1); at[s] = 0; done = np.zeros(n, bool)
            for _ in range(n):
                u = int(np.argmin(np.where(done, np.inf, at)))
                if not np.isfinite(at[u]): break
                done[u] = True
                for i in range(nl):
                    if src[i] == u and at[u] + w[i] < at[dst[i]]: at[dst[i]] = at[u] + w[i]; by[dst[i]] = i
            keep = np.zeros(n, bool)
            for d in ds:
                v = d
                while v != s: keep[v] = True; v = src[by[v]]
            cand.append([int(by[v]) if keep[v] else -1 for v in range(n)])
        trees.append(cand)
    dt = max(floor / 80, lat.max() / (H - 2))
    return dict(n=n, compute=compute, src=src, dst=dst, rate=rate, lat=lat, background=background, S=S, kind=kind,
                comm=comm, trees=trees, floor=floor, dt=dt)

def batch(scens, J):
    """the scenarios as padded arrays: lanes L, commodities J, candidates K, nodes N"""
    B = len(scens)
    a = dict(rate=np.zeros((B, L)), lat=np.zeros((B, L)), bg=np.zeros((B, L)), lsrc=np.zeros((B, L), int), lvalid=np.zeros((B, L)),
             lane=np.full((B, J, K, N), L, int), tail=np.zeros((B, J, K, N), int), src=np.zeros((B, J), int),
             dest=np.zeros((B, J, N)), demand=np.zeros((B, J)), phase=np.zeros((B, J), int), owner=np.zeros((B, J), int),
             jvalid=np.zeros((B, J)), floor=np.zeros(B), dt=np.zeros(B), S=np.zeros(B), kind=np.zeros(B), nodes=np.zeros(B))
    for b, s in enumerate(scens):
        nl = len(s['src'])
        a['rate'][b, :nl] = s['rate']; a['lat'][b, :nl] = s['lat']; a['bg'][b, :nl] = s['background']
        a['lsrc'][b, :nl] = s['src']; a['lvalid'][b, :nl] = 1
        a['floor'][b] = s['floor']; a['dt'][b] = s['dt']; a['S'][b] = s['S']; a['kind'][b] = s['kind']; a['nodes'][b] = s['n']
        for j, (src, ds, dem, ph, o) in enumerate(s['comm']):
            a['src'][b, j] = src; a['demand'][b, j] = dem; a['phase'][b, j] = ph; a['owner'][b, j] = max(o, 0); a['jvalid'][b, j] = 1
            for d in ds: a['dest'][b, j, d] = 1
            for k in range(K):
                for v in range(N):
                    l = s['trees'][j][k][v] if v < s['n'] else -1
                    if l >= 0: a['lane'][b, j, k, v] = l; a['tail'][b, j, k, v] = s['src'][l]
                    else: a['tail'][b, j, k, v] = src
    return {k: mx.array(v.astype(np.float32) if v.dtype == np.float64 else v.astype(np.int32)) for k, v in a.items()}

class Block(nn.Module):
    def __init__(self):
        super().__init__(); self.gru = nn.GRU(D, D); self.up = nn.Linear(D, E); self.down = nn.Linear(E, D)
    def __call__(self, x, h):
        h = self.gru(x[:, None, :], hidden=h)[:, -1, :]
        x = x + h
        return x + self.down(nn.relu(self.up(x))), h

class Policy(nn.Module):
    """A candidate tree is its N node tokens, embedded, rotated by their node's position (RoPE, at the embedding only),
    pooled, then one feed-forward block. The stream: the node's context and the embedding of its last weighting's
    expected choice, through two layers of a GRU block (recurrent over refreshes) and a feed-forward block."""
    def __init__(self):
        super().__init__()
        self.inp = nn.Linear(CTX + D, D); self.blocks = [Block(), Block()]
        self.tok = nn.Embedding(VOCAB, D); self.cup = nn.Linear(D, E); self.cdown = nn.Linear(E, D)
    def __call__(self, context, last, tokens, state):
        x = self.inp(mx.concatenate([context, last], -1)); hs = []
        for blk, h in zip(self.blocks, state):
            x, h = blk(x, h); hs.append(h)
        M = tokens.shape[0]
        e = mx.fast.rope(self.tok(tokens).reshape(M * K, N, D), D, traditional=False, base=10000.0, scale=1.0, offset=0)
        e = e.reshape(M, K, N, D).mean(-2)
        e = e + self.cdown(nn.relu(self.cup(e)))                               # [M, K, D]
        return (e @ x[:, :, None])[..., 0] / 8.0, hs, e                        # logits [M, K]

def rollout(policy, a, mode):
    """the fluid model's rollout; mode 'policy', 'spt' (all on candidate 0) or 'uniform'. Returns the objective and
    each scenario's true completion over its floor"""
    B, J = a['src'].shape
    rate, lat, dt, floor = a['rate'], a['lat'], a['dt'], a['floor']
    cap = rate * dt[:, None]                                                    # bytes a lane serves a slot
    lag = mx.minimum(mx.ceil((lat + PAGE / mx.maximum(rate, 1e-6)) / dt[:, None]), H - 1).astype(mx.int32)  # [B, L]
    laneg = mx.minimum(a['lane'], L - 1); edge = (a['lane'] < L).astype(mx.float32)   # [B, J, K, N]
    bidx = mx.arange(B)[:, None, None, None]
    elag = mx.take_along_axis(mx.broadcast_to(lag[:, None, None, :], (B, J, K, L)), laneg, -1)
    nodes = mx.arange(N); dest = a['dest']; demand = a['demand']; jv = a['jvalid']
    srcmask = (nodes[None, None, :] == a['src'][..., None]).astype(mx.float32)   # [B, J, N]
    # the source's release rate: its out-lanes' total a slot
    outcap = (cap[:, None, :] * (a['lsrc'][:, None, :] == a['src'][..., None]) * a['lvalid'][:, None, :]).sum(-1)  # [B, J]
    arrived = mx.zeros((B, J, K, N)); sent = mx.zeros((B, J, K, N)); assigned = mx.zeros((B, J, K))
    released = mx.zeros((B, J)); bg = a['bg']; hist = [mx.zeros((B, J, K, N))] * H
    p = mx.full((B, J, K), 1.0 / K); state = [mx.zeros((B * J, D)), mx.zeros((B * J, D))]; last = mx.zeros((B * J, D))
    area = mx.zeros((B, J, N)); done = mx.full((B,), float(SLOTS)); total = mx.zeros((B, J, N))
    if mode == 'spt': p = mx.concatenate([mx.ones((B, J, 1)), mx.zeros((B, J, K - 1))], -1)
    # unloaded completion of each candidate: depth latency, a feature
    for t in range(SLOTS):
        # what each edge can send: its tail's arrivals not yet sent on it
        tailarr = mx.take_along_axis(arrived, a['tail'], -1)
        avail = mx.maximum(tailarr - sent, 0) * edge
        laneq = mx.zeros((B, L + 1)).at[mx.broadcast_to(bidx, avail.shape).reshape(-1), a['lane'].reshape(-1)].add(avail.reshape(-1))[:, :L]
        want = laneq + bg
        share = mx.minimum(1.0, cap / mx.maximum(want, 1e-3))
        if mode == 'policy' and t % REFRESH == 0:
            # the policy's inputs, from the fluid state now: each candidate's node tokens
            qtime = want / mx.maximum(rate, 1e-6)                                # ns of queue a lane
            lq = mx.take_along_axis(mx.broadcast_to(qtime[:, None, None, :], (B, J, K, L)), laneg, -1) * edge
            lr = mx.take_along_axis(mx.broadcast_to(rate[:, None, None, :], (B, J, K, L)), laneg, -1)
            parent = mx.where(edge > 0, a['tail'], mx.where(nodes[None, None, None, :] == a['src'][:, :, None, None], N, N + 1))
            qlevel = mx.clip(mx.floor(mx.log2(1 + 64 * lq / floor[:, None, None, None])), 0, 15).astype(mx.int32)
            rband = mx.clip(mx.floor(mx.log2(mx.maximum(lr, 1.0))) - 2, 0, 3).astype(mx.int32) * (edge > 0)
            tokens = (parent * 64 + qlevel * 4 + rband).astype(mx.int32)       # [B, J, K, N]
            ownq = ((qtime[:, None, :] * (a['lsrc'][:, None, :] == a['src'][..., None])).sum(-1)) / floor[:, None]
            ctx = mx.stack([mx.broadcast_to(mx.log1p(a['S'] / PAGE)[:, None], (B, J)), mx.full((B, J), 1.0), mx.full((B, J), t / SLOTS * 1.0),
                            ownq, ownq, (a['lsrc'][:, None, :] == a['src'][..., None]).sum(-1) / 8.0, mx.full((B, J), K / 8.0),
                            dest.sum(-1) / 8, mx.broadcast_to((rate.sum(-1) / mx.maximum(a['lvalid'].sum(-1), 1) / 10)[:, None], (B, J)),
                            mx.broadcast_to((lat.sum(-1) / mx.maximum(a['lvalid'].sum(-1), 1) / 1e4)[:, None], (B, J)),
                            mx.broadcast_to(a['kind'][:, None], (B, J)), (a['phase'] == 2).astype(mx.float32)], -1)
            logits, state, emb = policy(ctx.reshape(B * J, CTX), last, tokens.reshape(B * J, K, N), state)
            p = mx.softmax(logits, -1).reshape(B, J, K)
            last = (p.reshape(B * J, K)[..., None] * emb).sum(-2)
        # release at sources: pool = demand (or an owner's reduced bytes) less released, at the out-lanes' rate
        if True:
            got = (arrived * dest[:, :, None, :]).sum(2)                        # [B, J, N] delivered
            pool_cap = demand
            # an all-reduce owner's reduced bytes: the slowest incoming partial to it
            red = mx.where(a['phase'] == 1, got.max(-1) / mx.maximum(demand, 1e-6), 1e9)   # phase-1 delivered fraction
            ownmatch = (a['owner'][:, :, None] == a['owner'][:, None, :]) & (a['phase'][:, :, None] == 2) & (a['phase'][:, None, :] == 1)
            frac = mx.where(ownmatch, red[:, None, :], 1e9).min(-1)
            pool_cap = mx.where(a['phase'] == 2, demand * mx.minimum(frac, 1.0), demand)
            r = mx.minimum(mx.maximum(pool_cap - released, 0), outcap) * jv
            released = released + r; assigned = assigned + r[..., None] * p
        sendinc = avail * mx.take_along_axis(mx.broadcast_to(share[:, None, None, :], (B, J, K, L)), laneg, -1)
        bg = bg - bg * share
        sent = sent + sendinc
        hist = hist[1:] + [sendinc]
        stack = mx.stack(hist[::-1], 0)                                         # stack[h] = increment h slots ago
        landed = mx.take_along_axis(stack, elag[None], 0)[0] * edge
        arrived = arrived + landed + srcmask[:, :, None, :] * (assigned[..., None] - arrived * srcmask[:, :, None, :]) * 1.0
        got = (arrived * dest[:, :, None, :]).sum(2)
        frac = mx.minimum(got / mx.maximum(demand[..., None], 1e-6), 1.0) * dest
        area = area + (dest - frac) * dt[:, None, None]
        complete = ((frac >= 0.999) | (dest == 0)).all(-1).all(-1) | (jv.sum(-1) == 0)
        done = mx.where(complete & (done == SLOTS), float(t + 1), done)
    # a smooth maximum (temperature 0.05) over every (commodity, destination) of its area over the floor
    valid = (dest * jv[..., None]) > 0
    vals = mx.where(valid, area / floor[:, None, None], -1e9).reshape(B, -1)
    objective = (0.05 * mx.logsumexp(vals / 0.05, -1)).mean()
    return objective, done * dt / floor

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--batch', type=int, default=128); ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--kind', default='both'); ap.add_argument('--log', type=int, default=25); ap.add_argument('--pool', type=int, default=1024)
    args = ap.parse_args()
    rng = np.random.default_rng(20261119)
    kinds = {'ag': [0], 'ar': [1], 'both': [0, 1]}[args.kind]
    t0 = time.time()
    pools = {k: [scenario(rng, k) for _ in range(args.pool)] for k in kinds}
    held = {k: batch([scenario(rng, k) for _ in range(args.batch)], 6 if k == 0 else 36) for k in kinds}
    print(f'scenarios made in {time.time() - t0:.1f} s', flush=True)
    policy = Policy(); mx.eval(policy.parameters())
    nparams = sum(v.size for _, v in nn.utils.tree_flatten(policy.parameters()))
    print(f'policy: {nparams} parameters', flush=True)
    opt = optim.Adam(learning_rate=1e-3)
    loss_and_grad = nn.value_and_grad(policy, lambda a: rollout(policy, a, 'policy')[0])
    t0 = time.time(); n = 0
    for step in range(args.steps):
        k = kinds[step % len(kinds)]
        pick = rng.choice(len(pools[k]), args.batch, replace=False)
        a = batch([pools[k][i] for i in pick], 6 if k == 0 else 36)
        loss, grads = loss_and_grad(a)
        opt.update(policy, grads); mx.eval(policy.parameters(), opt.state, loss); n += 1
        if step % args.log == 0:
            rate = n / (time.time() - t0); rows = []
            for kk in kinds:
                r = {m: rollout(policy, held[kk], m)[1] for m in ('policy', 'spt', 'uniform')}
                mx.eval(*r.values())
                rows.append(f"{'ag' if kk == 0 else 'ar'}: policy {np.mean(np.array(r['policy'])):.3f} spt {np.mean(np.array(r['spt'])):.3f} "
                            f"uniform {np.mean(np.array(r['uniform'])):.3f}")
            print(f'step {step}: loss {loss.item():.4f}, {rate:.2f} steps/s | T/floor ' + ' | '.join(rows), flush=True)
            t0 = time.time(); n = 0

if __name__ == '__main__':
    main()
