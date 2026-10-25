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


def unit_lengths(window):
    """Exact DMA payload lengths, with explicit historical full-page decoding.

    New byte-extent traces must carry their immutable length vector. A mean
    bytes/unit is never evidence of the lengths of variable-size units.
    """
    units = window['last'] - window['first']
    if not isinstance(units, int) or units <= 0:
        raise ValueError('invalid transport-unit coverage')
    kind = window.get('unit_kind', 'page')
    if kind not in ('page', 'byte_extent'):
        raise ValueError('unknown transport-unit kind')
    if 'unit_bytes' in window:
        lengths = tuple(window['unit_bytes'])
        if (len(lengths) != units or any(not isinstance(n, int) or isinstance(n, bool) or n <= 0 for n in lengths)
                or sum(lengths) != window['bytes']):
            raise ValueError('transport-unit lengths disagree with payload byte counts')
        return lengths
    if kind == 'byte_extent':
        raise ValueError('byte-extent observations require exact unit lengths')
    if not isinstance(window['bytes'], int) or window['bytes'] <= 0 or window['bytes'] % units:
        raise ValueError('historical full-page byte counts must be integral')
    return (window['bytes'] // units,) * units


def matched_transfers(ranks):
    """One observation per SEND, matching equal directed page *coverage*.

    SEND and receive entry boundaries may differ. First arrival is observed
    only for a receive entry's first page, and completion only for its final
    page. An interior SEND boundary therefore gets the conservative interval
    [containing receive's first-arrival lower, final-landing upper], not a
    fabricated interior arrival/completion timestamp. ``*_exact`` means that
    the boundary has its own observed bracket, not that clock/poll uncertainty
    has disappeared. All returned arrival/completion times use destination time.

    Coverage on each directed physical cable must match exactly and must not
    overlap itself on either side. Exact payload lengths per unit must agree.
    Multiple splits count the SEND bytes once, never the enclosing receive's
    bytes repeatedly. Original window dictionaries remain available to callers.
    """
    channels, lengths = {}, {}
    for rank in ranks:
        for window in rank['windows']:
            direction = window['direction']
            if 'peer' not in window or direction not in (0, 1) or window['last'] <= window['first'] or window['bytes'] <= 0:
                raise ValueError('invalid directed page observation')
            source, destination = ((rank['rank'], window['peer']) if not direction else
                                   (window['peer'], rank['rank']))
            key = window['cable'], source, destination
            lengths[id(window)] = unit_lengths(window)
            channels.setdefault(key, ([], []))[direction].append(window)
    result = []
    for (cable, source, destination), (sends, receives) in sorted(channels.items()):
        for entries in (sends, receives):
            entries.sort(key=lambda w: (w['first'], w['last']))
            if any(a['last'] > b['first'] for a, b in zip(entries, entries[1:])):
                raise ValueError('duplicate or overlapping directed page coverage')
        if not sends or not receives:
            raise ValueError('page coverage has no matching source/destination observation')
        by_send = [[] for _ in sends]
        si = ri = 0
        s_at, r_at = sends[0]['first'], receives[0]['first']
        while si < len(sends) and ri < len(receives):
            send, receive = sends[si], receives[ri]
            if s_at != r_at:
                raise ValueError('source and destination page coverage disagree')
            through = min(send['last'], receive['last'])
            if (lengths[id(send)][s_at - send['first']:through - send['first']] !=
                    lengths[id(receive)][r_at - receive['first']:through - receive['first']]):
                raise ValueError('source and destination page byte counts disagree')
            if 'wire' in send and 'wire' in receive and send['wire'] != receive['wire']:
                raise ValueError('source and destination wire identities disagree')
            by_send[si].append(receive)
            s_at = r_at = through
            if through == send['last']:
                si += 1
                if si < len(sends):
                    s_at = sends[si]['first']
            if through == receive['last']:
                ri += 1
                if ri < len(receives):
                    r_at = receives[ri]['first']
        if si != len(sends) or ri != len(receives):
            raise ValueError('source and destination page coverage has unmatched ranges')
        for send, landed in zip(sends, by_send):
            first, last = landed[0], landed[-1]
            if not send['start_hi_ns'] or any(not r['start_hi_ns'] or not r['end_hi_ns'] for r in landed):
                raise ValueError('unfinished page transaction has no complete arrival observation')
            if send['start_lo_ns'] > send['start_hi_ns'] or any(
                r['start_lo_ns'] > r['start_hi_ns'] or r['end_lo_ns'] > r['end_hi_ns']
                or r['start_lo_ns'] > r['end_hi_ns'] for r in landed
            ):
                raise ValueError('inconsistent page observation timestamp brackets')
            arrival_exact = send['first'] == first['first']
            completion_exact = send['last'] == last['last']
            arrival = [first['start_lo_ns'], first['start_hi_ns'] if arrival_exact else first['end_hi_ns']]
            completion = [last['end_lo_ns'] if completion_exact else last['start_lo_ns'], last['end_hi_ns']]
            if len(landed) > 1:
                completion[0] = max(completion[0], *(r['end_lo_ns'] for r in landed[:-1]))
            if completion[0] > completion[1]:
                raise ValueError('ordered receive completion observations disagree')
            result.append({'source': source, 'destination': destination, 'send': send, 'receives': tuple(landed),
                           'arrival_ns': arrival, 'completion_ns': completion,
                           'arrival_exact': arrival_exact, 'completion_exact': completion_exact,
                           'unit_bytes': lengths[id(send)]})
    return tuple(result)


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
    for match in matched_transfers(ranks):
        source, destination, send = match['source'], match['destination'], match['send']
        bound[destination, source] = min(bound[destination, source],
                                         match['arrival_ns'][1] - send['start_lo_ns'])
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
    explicit_spans, credit_frames = False, 0
    per_node = {r['rank']: {'ingress': [], 'egress': []} for r in ranks}
    for match in matched_transfers(ranks):
        source, destination, w = match['source'], match['destination'], match['send']
        explicit_spans |= 'unit_bytes' in w or w.get('unit_kind') == 'byte_extent'
        credit_frames += sum((n + 4095) // 4096 for n in match['unit_bytes'])
        source_offset = offsets[str(source)]
        destination_offset = offsets[str(destination)]
        inner = (w['start_hi_ns'] + source_offset[1], match['completion_ns'][0] + destination_offset[0])
        outer = (w['start_lo_ns'] + source_offset[0], match['completion_ns'][1] + destination_offset[1])
        lower.append(inner); upper.append(outer)
        arc = 2 * w['cable'] + (source == cables[w['cable']]['nodes'][1])
        loads[str(arc)] = loads.get(str(arc), 0) + w['bytes']
        per_node[source]['egress'].append((inner, outer, w['bytes']))
        per_node[destination]['ingress'].append((inner, outer, w['bytes']))
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
    result = {'capacity_bytes_per_second': capacity, 'wire_bytes': wire, 'useful_bytes': useful_bytes,
            'hop_and_padding_amplification': wire / useful_bytes, 'directed_wire_bytes': loads,
            'window_seconds_bounds': [lo, hi], 'all_links_ideal_seconds': ideal,
            'current_routing_bottleneck_seconds': binding,
            'wire_bandwidth_bytes_per_second_bounds': [wire / hi if hi else None, wire / lo if lo else None],
            'useful_bandwidth_bytes_per_second_bounds': [useful_bytes / hi if hi else None, useful_bytes / lo if lo else None],
            'capacity_utilization_bounds': [ideal / hi if hi else None, ideal / lo if lo else None],
            'unused_capacity_bytes_bounds': [capacity * lo - wire, capacity * hi - wire],
            'clock_probe_bounds_ns': probes, 'clock_offset_bounds_ns': offsets, 'nodes': nodes}
    if explicit_spans:
        result.update(wire_payload_bytes=wire, payload_accounting='exact_dma_span_lengths',
                      credit_frames_4096=credit_frames, physical_framing_overhead_bytes=None)
    return result


def finishes(cables, ranks, stages, offsets):
    """Completion residuals for a repeated compiled group, after the full call.

    The caller declares repeated stages with the same byte/range/relay layout
    on each cable. Verify that translated entry pattern, not merely divisibility
    of the entry count. Other stage layouts need explicit stage metadata; do
    not fabricate stage boundaries from their observed number of entries.
    """
    if not isinstance(stages, int) or stages <= 0:
        raise ValueError('positive integral repeated-stage count required')
    matched = {id(m['send']): m for m in matched_transfers(ranks)}
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
            signature = lambda part: [(w['first'] - part[0]['first'], w['last'] - part[0]['first'],
                                       w['bytes'], w.get('relay', 0), matched[id(w)]['unit_bytes']) for w in part]
            pattern = signature(entries[:count])
            if any(signature(entries[layer * count:(layer + 1) * count]) != pattern for layer in range(stages)):
                raise ValueError('trace entry layout does not repeat across declared stages; explicit stage metadata required')
            arc = str(2 * cable + (source == c['nodes'][1]))
            for layer in range(stages):
                part = entries[layer * count:(layer + 1) * count]
                starts, ends, inner, outer, own_inner, own_outer, own_starts = [], [], [], [], [], [], []
                for w in part:
                    match = matched[id(w)]
                    dest = match['destination']
                    start = [w['start_lo_ns'] + offsets[str(source)][0], w['start_hi_ns'] + offsets[str(source)][1]]
                    end = [match['completion_ns'][k] + offsets[str(dest)][k] for k in (0, 1)]
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
