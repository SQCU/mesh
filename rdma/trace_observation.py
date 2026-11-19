"""Exact physical transport observations; no execution or publication controls."""
from collections import defaultdict
import statistics

from roofline import matched_transfers, unit_lengths


class SpanTrace:
    """Capture immutable span lengths before releasing the native observer."""
    def __init__(self, comm):
        import mesh
        self.mesh, self.comm, self.room = mesh, comm, 4096
        self.handle = mesh.ffi.new('ncclMeshTrace_t *')

    def start(self):
        self.mesh.check(self.mesh.lib.ncclMeshTraceStart(self.comm, self.room, self.handle))

    def finish(self):
        m = self.mesh
        windows = m.ffi.new('struct ncclMeshWindow[]', self.room)
        count, stats, length_count = m.ffi.new('int *'), m.ffi.new('struct ncclMeshTraceStats *'), m.ffi.new('size_t *')
        try:
            m.check(m.lib.ncclMeshTraceSpanLengths(self.handle[0], m.ffi.NULL, 0, length_count))
            lengths = m.ffi.new('uint64_t[]', max(1, length_count[0]))
            m.check(m.lib.ncclMeshTraceSpanLengths(self.handle[0], lengths, length_count[0], length_count))
        finally:
            m.check(m.lib.ncclMeshTraceStop(self.handle[0], windows, self.room, count, stats))
        if stats.incomplete:
            raise ValueError(f'{stats.incomplete} physical trace windows did not complete')
        rows, at = [], 0
        for window in windows[0:count[0]]:
            row = {key: int(getattr(window, key)) for key in (
                'start_lo_ns', 'start_hi_ns', 'end_lo_ns', 'end_hi_ns', 'bytes', 'wire', 'first', 'last',
                'peak_queue_bytes', 'cable_peak_queue_bytes', 'cable', 'peer', 'direction', 'relay')}
            end = at + row['last'] - row['first']
            if end > length_count[0]:
                raise ValueError('trace windows exceed exact span-length storage')
            row.update(unit_kind='byte_extent', unit_bytes=[int(lengths[u]) for u in range(at, end)],
                       queue_byte_seconds=float(window.queue_byte_seconds))
            unit_lengths(row)
            rows.append(row)
            at = end
        if at != length_count[0]:
            raise ValueError('trace windows and exact span-length storage disagree')
        return dict(windows=rows, polls=int(stats.polls), max_gap_ns=int(stats.max_gap_ns), incomplete=0)


def _quantiles(values):
    values = sorted(values)
    return dict(median=round(statistics.median(values), 2), p10=round(values[len(values)//10], 2),
                p90=round(values[9*len(values)//10], 2)) if values else None


def _summary(intervals):
    return dict(midpoint=_quantiles([(lo+hi)/2000 for lo, hi in intervals]),
                lower=_quantiles([lo/1000 for lo, _ in intervals]),
                upper=_quantiles([hi/1000 for _, hi in intervals]))


def _difference(after, before):
    return after[0]-before[1], after[1]-before[0]


def tick_summary(ranks, rank, ticks):
    """Summarize a repeated function only when physical interval shapes prove it.

    SEND/COPY entry partitions may differ. Validate cross-rank unit coverage and
    bytes first; then partition each local physical channel independently by its
    repeated interval signature. Never pair entries by peer or time ordinal.
    Coalesced/non-repeating traces need explicit stage metadata; keep raw windows
    and report that limitation rather than fabricate intermediate timestamps.
    """
    if ticks <= 0:
        raise ValueError('positive tick count required')
    matches = matched_transfers(ranks)
    local = next(value for value in ranks if value['rank'] == rank)
    windows = local['windows']
    result = dict(entries=len(windows), physical_transfers=len(matches), requested_ticks=ticks,
                  sent_payload_bytes=sum(w['bytes'] for w in windows if w['direction'] == 0),
                  landed_payload_bytes=sum(w['bytes'] for w in windows if w['direction'] == 1),
                  windows=windows, tick_layout_supported=False)
    grouped = defaultdict(list)
    for w in windows:
        grouped[w['direction'], w['cable'], w['wire'], w['peer']].append(w)
    stages = [[] for _ in range(ticks)]
    try:
        for key, entries in sorted(grouped.items()):
            entries.sort(key=lambda w: (w['first'], w['last']))
            if len(entries) % ticks:
                raise ValueError('physical entries do not divide into the declared tick count')
            if any(a['last'] != b['first'] for a, b in zip(entries, entries[1:])):
                raise ValueError('physical channel has a gap in its observed unit sequence')
            count = len(entries)//ticks
            signature = lambda part: tuple((w['first']-part[0]['first'], w['last']-part[0]['first'],
                                            tuple(w['unit_bytes']), w['relay']) for w in part)
            expected = signature(entries[:count])
            for tick in range(ticks):
                part = entries[tick*count:(tick+1)*count]
                if signature(part) != expected:
                    raise ValueError('physical interval shape changes across ticks; explicit stage metadata required')
                stages[tick].extend(part)
        starts, ends, physical = [], [], defaultdict(list)
        for stage in stages:
            own = [w for w in stage if w['direction'] == 0 and not w['relay']]
            land = [w for w in stage if w['direction'] == 1]
            if not own or not land:
                raise ValueError('a tick lacks observable local producer publication or incoming completion')
            start = tuple(min(w[f'start_{bound}_ns'] for w in own) for bound in ('lo', 'hi'))
            end = tuple(max(w[f'end_{bound}_ns'] for w in land) for bound in ('lo', 'hi'))
            starts.append(start); ends.append(end)
            for cable, wire, peer in sorted({(w['cable'], w['wire'], w['peer']) for w in land}):
                incoming = [w for w in land if (w['cable'], w['wire'], w['peer']) == (cable, wire, peer)]
                finish = tuple(max(w[f'end_{bound}_ns'] for w in incoming) for bound in ('lo', 'hi'))
                physical[f'{cable}:{wire:016x}:{peer}'].append(_difference(finish, start))
        intervals = dict(tick_us=[_difference(b, a) for a, b in zip(starts, starts[1:])],
                         publish_to_landed_us=[_difference(b, a) for a, b in zip(starts, ends)],
                         landed_to_next_publish_us=[_difference(b, a) for a, b in zip(ends, starts[1:])])
        for name, values in intervals.items():
            summary = _summary(values)
            result[name], result[name+'_bounds'] = summary['midpoint'], {k: summary[k] for k in ('lower', 'upper')}
        for direction, name in ((0, 'send_window_us'), (1, 'landing_window_us')):
            result[name] = _summary([_difference((w['end_lo_ns'], w['end_hi_ns']),
                                                 (w['start_lo_ns'], w['start_hi_ns']))
                                     for w in windows if w['direction'] == direction])
        result.update(ticks=ticks, tick_layout_supported=True,
                      physical_landed_after_publish_us={key: _summary(values) for key, values in physical.items()},
                      timing_semantics='local producer publication to all local physical landings; midpoint plus poll bounds')
    except ValueError as error:
        result['tick_layout_reason'] = str(error)
    return result
