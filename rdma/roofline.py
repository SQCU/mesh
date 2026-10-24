"""design/whole-function.md#roofline"""


def union_ns(intervals):
    end, total = None, 0
    for a, b in sorted((a, b) for a, b in intervals if b > a):
        if end is None or a > end:
            total += b - a
        elif b > end:
            total += b - end
        end = b if end is None else max(end, b)
    return total


def causal_offsets(ranks, offsets):
    """Tighten clock bounds using send eligibility preceding first arrival.

    An edge i->j bounds offset[j]-offset[i]. Neither bandwidth nor an assumed
    symmetric network delay enters these constraints. Keep the raw probe bounds
    in the report so the refinement remains independently reproducible.
    """
    nodes = [r['rank'] for r in ranks]
    bound = {(a, b): 0 if a == b else float('inf') for a in nodes for b in nodes}
    for rank in nodes:
        lo, hi = offsets[str(rank)]
        bound[0, rank] = min(bound[0, rank], hi)
        bound[rank, 0] = min(bound[rank, 0], -lo)
    arrivals = {(w['cable'], w['peer'], w['first'], w['last']): (r['rank'], w)
                for r in ranks for w in r['windows'] if w['direction'] == 1}
    for rank in ranks:
        source = rank['rank']
        for w in rank['windows']:
            if w['direction']:
                continue
            key = w['cable'], source, w['first'], w['last']
            if key not in arrivals or not w['start_hi_ns']:
                raise ValueError('clock constraint has no complete source/destination observation')
            destination, landed = arrivals[key]
            if not landed['start_hi_ns']:
                raise ValueError('clock constraint has no observed arrival')
            bound[destination, source] = min(bound[destination, source],
                                             landed['start_hi_ns'] - w['start_lo_ns'])
    for via in nodes:
        for a in nodes:
            for b in nodes:
                bound[a, b] = min(bound[a, b], bound[a, via] + bound[via, b])
    if any(bound[r, r] < 0 for r in nodes):
        raise ValueError('clock probes and page causality have inconsistent offset bounds')
    return {str(r): [-bound[r, 0], bound[0, r]] for r in nodes}


def roofline(cables, ranks, useful_bytes, offsets):
    """Match source eligibility to destination landing, once per hop.

    Offsets bound root-clock minus rank-clock. Transition brackets preserve
    observation uncertainty. Idle cables retain their capacity in the roofline.
    """
    probes = offsets
    offsets = causal_offsets(ranks, probes)
    capacity = sum(2 * c['bits_per_second'] / 8 for c in cables)
    lower, upper, loads = [], [], {}
    per_node = {r['rank']: {'ingress': [], 'egress': []} for r in ranks}
    arrivals = {(w['cable'], w['peer'], w['first'], w['last']): (r['rank'], w)
                for r in ranks for w in r['windows'] if w['direction'] == 1}
    for rank in ranks:
        source = rank['rank']
        source_offset = offsets[str(source)]
        for w in rank['windows']:
            if w['direction']:
                continue
            key = w['cable'], source, w['first'], w['last']
            if key not in arrivals:
                raise ValueError('a page transaction has no matching destination observation')
            destination, landed = arrivals.pop(key)
            if not w['start_hi_ns'] or not landed['end_hi_ns']:
                raise ValueError('an unfinished page transaction has no attained bandwidth')
            if landed['bytes'] != w['bytes'] or w['peer'] != destination:
                raise ValueError('source and destination page observations disagree')
            destination_offset = offsets[str(destination)]
            inner = (w['start_hi_ns'] + source_offset[1], landed['end_lo_ns'] + destination_offset[0])
            outer = (w['start_lo_ns'] + source_offset[0], landed['end_hi_ns'] + destination_offset[1])
            lower.append(inner); upper.append(outer)
            arc = 2 * w['cable'] + (source == cables[w['cable']]['nodes'][1])
            loads[str(arc)] = loads.get(str(arc), 0) + w['bytes']
            per_node[source]['egress'].append((inner, outer, w['bytes']))
            per_node[destination]['ingress'].append((inner, outer, w['bytes']))
    if arrivals:
        raise ValueError('destination observations have unmatched page transactions')
    nodes = []
    for rank, directions in per_node.items():
        node = {'rank': rank}
        node_capacity = sum(c['bits_per_second'] / 8 for c in cables if rank in c['nodes'])
        for direction, entries in directions.items():
            wire = sum(e[2] for e in entries)
            lo = union_ns(e[0] for e in entries) * 1e-9
            hi = union_ns(e[1] for e in entries) * 1e-9
            node[direction] = {'wire_bytes': wire, 'capacity_bytes_per_second': node_capacity,
                               'window_seconds_bounds': [lo, hi], 'all_links_ideal_seconds': wire / node_capacity,
                               'wire_bandwidth_bytes_per_second_bounds': [wire / hi if hi else None, wire / lo if lo else None]}
        nodes.append(node)
    lo, hi = union_ns(lower) * 1e-9, union_ns(upper) * 1e-9
    wire = sum(loads.values())
    ideal = wire / capacity
    binding = max((v / (cables[int(e) // 2]['bits_per_second'] / 8) for e, v in loads.items()), default=0)
    return {'capacity_bytes_per_second': capacity, 'wire_bytes': wire, 'useful_bytes': useful_bytes,
            'hop_and_padding_amplification': wire / useful_bytes, 'directed_wire_bytes': loads,
            'window_seconds_bounds': [lo, hi], 'all_links_ideal_seconds': ideal,
            'current_routing_bottleneck_seconds': binding,
            'wire_bandwidth_bytes_per_second_bounds': [wire / hi if hi else None, wire / lo if lo else None],
            'useful_bandwidth_bytes_per_second_bounds': [useful_bytes / hi if hi else None, useful_bytes / lo if lo else None],
            'capacity_utilization_bounds': [ideal / hi if hi else None, ideal / lo if lo else None],
            'unused_capacity_bytes_bounds': [capacity * lo - wire, capacity * hi - wire],
            'clock_probe_bounds_ns': probes, 'clock_offset_bounds_ns': offsets, 'nodes': nodes}


def finishes(cables, ranks, stages, offsets):
    """Completion residuals for a repeated compiled group, after the full call.

    Every stage has the same entry sequence on each cable. Page-prefix order
    identifies stages without inserting phase barriers or modifying transport.
    """
    arrivals = {(w['cable'], w['peer'], w['first'], w['last']): (r['rank'], w)
                for r in ranks for w in r['windows'] if w['direction'] == 1}
    groups = [{} for _ in range(stages)]
    for rank in ranks:
        source = rank['rank']
        for cable, c in enumerate(cables):
            entries = sorted((w for w in rank['windows'] if not w['direction'] and w['cable'] == cable),
                             key=lambda w: w['first'])
            if not entries:
                continue
            if len(entries) % stages:
                raise ValueError('trace does not match repeated compiled stages')
            count = len(entries) // stages
            arc = str(2 * cable + (source == c['nodes'][1]))
            for layer in range(stages):
                part = entries[layer * count:(layer + 1) * count]
                starts, ends, inner, outer, own_inner, own_outer, own_starts = [], [], [], [], [], [], []
                for w in part:
                    dest, landed = arrivals[w['cable'], source, w['first'], w['last']]
                    start = [w['start_lo_ns'] + offsets[str(source)][0], w['start_hi_ns'] + offsets[str(source)][1]]
                    end = [landed['end_lo_ns'] + offsets[str(dest)][0], landed['end_hi_ns'] + offsets[str(dest)][1]]
                    starts.append(start); ends.append(end)
                    inner.append((start[1], end[0])); outer.append((start[0], end[1]))
                    if not w.get('relay', 0):
                        own_inner.append((start[1], end[0])); own_outer.append((start[0], end[1]))
                        own_starts.append(start)
                groups[layer][arc] = {'source': source, 'bytes': sum(w['bytes'] for w in part),
                                     'relay_bytes': sum(w['bytes'] for w in part if w.get('relay', 0)),
                                     'start_ns': [min(s[k] for s in starts) for k in (0, 1)],
                                     'finish_ns': [max(e[k] for e in ends) for k in (0, 1)],
                                     'own_window_seconds_bounds': [union_ns(own_inner) * 1e-9, union_ns(own_outer) * 1e-9],
                                     'own_start_ns': [min(s[k] for s in own_starts) for k in (0, 1)] if own_starts else None,
                                     'window_seconds_bounds': [union_ns(inner) * 1e-9, union_ns(outer) * 1e-9]}
    report, penalty = [], 0.0
    tails = [0.0] * len(ranks)
    releases = [0.0] * len(ranks)
    for group in groups:
        if len(group) != 2 * len(cables):
            raise ValueError('equal-finish observation requires traffic on every directed cable')
        origin = min(v['start_ns'][0] for v in group.values())
        end = max(v['finish_ns'][1] for v in group.values())
        span = (end - origin) * 1e-9
        mid = [((v['finish_ns'][0] - origin) + (v['finish_ns'][1] - origin)) * 0.5e-9 for v in group.values()]
        mean = sum(mid) / len(mid)
        variance = sum((t - mean) ** 2 for t in mid) / len(mid)
        penalty += variance / span ** 2 / stages
        spread = [max(0, max(v['finish_ns'][0] for v in group.values()) - min(v['finish_ns'][1] for v in group.values())) * 1e-9,
                  (max(v['finish_ns'][1] for v in group.values()) - min(v['finish_ns'][0] for v in group.values())) * 1e-9]
        for rank in range(len(ranks)):
            tails[rank] += max(sum(v['own_window_seconds_bounds']) / 2 for v in group.values() if v['source'] == rank)
            releases[rank] += min((sum(v['own_start_ns']) / 2 - origin) * 1e-9
                                  for v in group.values() if v['source'] == rank and v['own_start_ns']) / stages
        report.append({'span_seconds': span, 'spread_seconds_bounds': spread,
                       'rms_finish_seconds': variance ** 0.5, 'arcs': group})
    return {'stages': report, 'finish_variance': penalty, 'delivery_tail_seconds': tails,
            'producer_release_seconds': releases}
