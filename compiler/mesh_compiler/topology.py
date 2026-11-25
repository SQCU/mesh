# Ticket T3: Graph-aware topology engine for AOT placement (C2 locality), switch-aligned emission order,
# and exact min-cut / Steiner capacity bounds for heterogeneous compute-share derivation (shares.py).
from collections import deque
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class DirectedLink:
    src: object
    dst: object
    cables: int
    rate_gbs: float    # aggregate directional bandwidth (GB/s) across all cables on (src -> dst)
    latency_us: float  # one-way base latency (us)


class TopologyGraph:
    """Explicit physical fabric graph (nodes + heterogeneous multi-cable directed links)."""

    def __init__(self, nodes, links: list[DirectedLink]):
        self.nodes = tuple(nodes)
        self.links = tuple(links)
        self.adj: dict[object, dict[object, DirectedLink]] = {u: {} for u in self.nodes}
        for lk in self.links:
            self.adj.setdefault(lk.src, {})[lk.dst] = lk
            self.adj.setdefault(lk.dst, {})

    @classmethod
    def cluster_5node(cls) -> "TopologyGraph":
        """Measured 5-node Apple Silicon TB5 RDMA cluster (A=m5ult2, B=m5ult1, C=studio2, D=studio3, M=mini)."""
        # Per-cable measured rates (~9.5-10.5 GB/s/cable on TB5 Studios, ~7.3-12 GB/s on M):
        specs = [
            ("A", "B", 3, 29.5, 1.5), ("B", "A", 3, 29.5, 1.5),
            ("C", "D", 4, 38.0, 1.5), ("D", "C", 4, 38.0, 1.5),
            ("B", "D", 2, 19.0, 1.5), ("D", "B", 2, 19.0, 1.5),
            ("A", "C", 1, 9.5, 1.5),  ("C", "A", 1, 9.5, 1.5),
            ("A", "M", 1, 8.8, 1.5),  ("M", "A", 1, 8.8, 1.5),
            ("B", "M", 1, 8.8, 1.5),  ("M", "B", 1, 8.8, 1.5),
            ("C", "M", 1, 9.2, 1.5),  ("M", "C", 1, 9.2, 1.5),
        ]
        return cls(("A", "B", "C", "D", "M"), [DirectedLink(*s) for s in specs])

    @classmethod
    def torus(cls, rows: int, cols: int, *, twisted: bool = True, cable_gbs: float = 10.0, latency_us: float = 1.5):
        """Parametric H x W Studio column topology (2D torus or twisted torus with +1 column shift)."""
        nodes = tuple((r, c) for r in range(rows) for c in range(cols))
        edges = set()
        for r, c in nodes:
            if cols > 1:
                edges.add(((r, c), (r, (c + 1) % cols)))
            if rows > 1:
                nc = (c + 1) % cols if (twisted and r == rows - 1) else c
                edges.add(((r, c), ((r + 1) % rows, nc)))
        links = []
        for u, v in sorted(edges):
            if u == v:
                continue
            links.append(DirectedLink(u, v, 1, cable_gbs, latency_us))
            links.append(DirectedLink(v, u, 1, cable_gbs, latency_us))
        return cls(nodes, links)

    def hop_distances(self) -> dict[tuple, int]:
        dist = {}
        for s in self.nodes:
            d = {s: 0}
            q = deque([s])
            while q:
                u = q.popleft()
                for v in self.adj[u]:
                    if v not in d:
                        d[v] = d[u] + 1
                        q.append(v)
            for t, val in d.items():
                dist[(s, t)] = val
        return dist

    def avg_distance(self, ranks=None) -> float:
        ranks = tuple(self.nodes if ranks is None else ranks)
        dist = self.hop_distances()
        vals = [dist[(u, v)] for u in ranks for v in ranks if u != v]
        return sum(vals) / len(vals) if vals else 0.0

    def egress_capacity(self, u) -> float:
        return sum(lk.rate_gbs for lk in self.adj.get(u, {}).values())

    def ingress_capacity(self, u) -> float:
        return sum(lk.rate_gbs for lk in self.links if lk.dst == u)

    def bottleneck_rate_gbs(self, ranks=None) -> float:
        """Minimum directional per-rank bottleneck bandwidth (GB/s) across `ranks` (exact min-bisection for <=10 ranks)."""
        from itertools import combinations
        nodes = tuple(self.nodes if ranks is None else ranks)
        p = len(nodes)
        if p <= 1:
            return 10.0
        rates = []
        for u in nodes:
            cap = min(self.egress_capacity(u), self.ingress_capacity(u))
            if cap > 0.0:
                rates.append(cap)
        mid = p // 2
        if p <= 10 and mid > 0:
            # Exact min-bisection over all balanced partitions fixing nodes[0] on left side
            rest = nodes[1:]
            for pick in combinations(rest, mid - 1):
                left = (nodes[0],) + pick
                left_set = set(left)
                right = tuple(x for x in nodes if x not in left_set)
                bis = self.max_flow(left, right)
                if math.isfinite(bis) and bis > 0.0:
                    rates.append(bis / mid)
        elif mid > 0:
            bis = self.max_flow(nodes[:mid], nodes[mid:])
            if math.isfinite(bis) and bis > 0.0:
                rates.append(bis / mid)
        return min(rates) if rates else 10.0

    def max_flow(self, sources, sinks) -> float:
        """Exact Edmonds-Karp max-flow (GB/s) from super-source(sources) to super-sink(sinks)."""
        src_set, dst_set = set(sources), set(sinks)
        if not src_set or not dst_set or (src_set & dst_set):
            return math.inf
        S, T = ("__S__",), ("__T__",)
        cap: dict[object, dict[object, float]] = {S: {}, T: {}}
        for u in self.nodes:
            cap[u] = {}
        inf = sum(lk.rate_gbs for lk in self.links) + 1.0
        for u in src_set:
            cap[S][u] = inf
            cap[u].setdefault(S, 0.0)
        for v in dst_set:
            cap[v][T] = inf
            cap[T].setdefault(v, 0.0)
        for lk in self.links:
            cap[lk.src][lk.dst] = cap[lk.src].get(lk.dst, 0.0) + lk.rate_gbs
            cap[lk.dst].setdefault(lk.src, 0.0)
        flow = 0.0
        while True:
            parent = {S: None}
            q = deque([S])
            while q and T not in parent:
                u = q.popleft()
                for v, c in cap[u].items():
                    if v not in parent and c > 1e-9:
                        parent[v] = u
                        q.append(v)
            if T not in parent:
                break
            aug = math.inf
            v = T
            while v != S:
                u = parent[v]
                aug = min(aug, cap[u][v])
                v = u
            v = T
            while v != S:
                u = parent[v]
                cap[u][v] -= aug
                cap[v][u] = cap[v].get(u, 0.0) + aug
                v = u
            flow += aug
        return flow

    def bisection_capacity(self, ranks=None) -> tuple[float, tuple, tuple]:
        """Find minimum balanced bisection cut (GB/s) over `ranks` (exact for |ranks| <= 12, spectral/BFS cut above)."""
        ranks = tuple(self.nodes if ranks is None else ranks)
        n = len(ranks)
        half = n // 2
        if n <= 1:
            return math.inf, ranks, ()
        if n <= 12:
            import itertools
            best_cap, best_s, best_t = math.inf, (), ()
            fixed = ranks[0]
            rest = ranks[1:]
            for comb in itertools.combinations(rest, half - 1):
                s = (fixed, *comb)
                t = tuple(r for r in ranks if r not in s)
                c = self.max_flow(s, t)
                if c < best_cap:
                    best_cap, best_s, best_t = c, s, t
            return best_cap, best_s, best_t
        # For larger meshes (e.g. 6x3 = 18 nodes), take minimum over BFS-ordered prefix cuts from every node.
        best_cap, best_s, best_t = math.inf, (), ()
        dist = self.hop_distances()
        for root in ranks:
            ordered = sorted(ranks, key=lambda u: (dist.get((root, u), 999), str(u)))
            s, t = tuple(ordered[:half]), tuple(ordered[half:])
            c = self.max_flow(s, t)
            if c < best_cap:
                best_cap, best_s, best_t = c, s, t
        return best_cap, best_s, best_t


def graph_locality_order(topo: TopologyGraph, ranks) -> list:
    """Order `ranks` so consecutive logical indices r, r+1 minimize physical hop distance (for C2 interval coalescing).
    Uses multi-start nearest-neighbor + 2-opt refinement over the exact hop metric."""
    ranks = list(ranks)
    n = len(ranks)
    if n <= 2:
        return ranks
    dist = topo.hop_distances()

    def path_cost(seq):
        return sum(dist.get((seq[i], seq[i + 1]), 99) for i in range(len(seq) - 1))

    best = None
    best_cost = math.inf
    for start in ranks:
        seq = [start]
        rem = [r for r in ranks if r != start]
        while rem:
            last = seq[-1]
            nxt = min(rem, key=lambda u: (dist.get((last, u), 99), -topo.egress_capacity(u), str(u)))
            seq.append(nxt)
            rem.remove(nxt)
        # 2-opt path refinement
        improved = True
        while improved:
            improved = False
            for i in range(n - 1):
                for j in range(i + 2, n):
                    cand = seq[:i + 1] + seq[i + 1:j + 1][::-1] + seq[j + 1:]
                    if path_cost(cand) < path_cost(seq):
                        seq = cand
                        improved = True
        c = path_cost(seq)
        if c < best_cost:
            best_cost, best = c, seq
    return best


@dataclass(frozen=True)
class PhaseStrategy:
    """Explicit parametric schedule & backend strategy (no symbolic dispatch trees).
    coalesce_group (g in divisors of P): dst(r, p) = (p + r // g) % P.
      - g = P : destination-aligned temporal coalescing (max in-switch reduction)
      - g = 1 : ingress-rotated (max concurrent per-node ingress, zero incast)
      - 1 < g < P : g topologically adjacent ranks coalesce while P // g destinations receive concurrently."""
    coalesce_group: int
    tiles: int
    op: int


def phase_destination(rank: int, phase: int, world_size: int, coalesce_group: int) -> int:
    return (phase + rank // max(1, coalesce_group)) % max(1, world_size)


def select_phase_strategy(
    topo: TopologyGraph,
    ranks,
    bytes_per_shard: int,
    phase_time_s: float,
    op: int,
    *,
    tiles: int = 1,
) -> PhaseStrategy:
    """Closed-form parametric coalescing group g from fabric cut/ingress vs per-phase injection rate."""
    ranks = tuple(ranks)
    p = max(1, len(ranks))
    t = max(1, int(tiles))
    b_phase = bytes_per_shard / t
    if phase_time_s <= 0.0 or b_phase <= 0.0:
        return PhaseStrategy(p, t, op)
    c_in = min(topo.ingress_capacity(r) for r in ranks) if ranks else math.inf
    c_cut = topo.bisection_capacity(ranks)[0] if len(ranks) > 1 else c_in
    c_eff = min(c_in, c_cut) * 1e9
    raw_g = max(1, min(p, int((c_eff * phase_time_s) // b_phase)))
    g = max((d for d in range(1, p + 1) if p % d == 0 and d <= raw_g), default=1)
    return PhaseStrategy(g, t, op)


def switch_aligned_order(
    send_records: list[dict],
    topo: TopologyGraph,
    *,
    ranks=None,
    strategy: PhaseStrategy | None = None,
) -> list[dict]:
    """Order REDUCE_SEND emissions under explicit parametric PhaseStrategy (with multi-hop source lead)."""
    dist = topo.hop_distances()
    order_ranks = tuple(ranks) if ranks is not None else tuple(dict.fromkeys(r["dst"] for r in send_records))
    p = max(1, len(order_ranks))
    pos = {u: i for i, u in enumerate(order_ranks)}
    g = p if strategy is None else max(1, min(p, strategy.coalesce_group))

    def phase_of(rec):
        si = pos.get(rec["src"], 0)
        di = pos.get(rec["dst"], 0)
        return (di - si // g) % p

    return sorted(
        send_records,
        key=lambda rec: (
            rec.get("block", 0) if strategy is None else phase_of(rec),
            str(rec["dst"]),
            -dist.get((rec["src"], rec["dst"]), 0),
            rec.get("group", 0),
        ),
    )
