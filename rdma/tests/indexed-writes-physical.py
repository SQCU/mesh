"""Three-rank indexed WRITE correctness through public library calls only.

Run with the normal RANK/WORLD_SIZE/NCCL_COMM_ID mesh environment. Every rank
needs a little over 1 GiB of available registered lease storage for the last
case. That case initializes/verifies sparse sentinels, not the entire payload.
Every case has fresh retained storage; no receipt or storage-reuse protocol is
implemented here. Output is correctness/physical counters, never throughput.
"""
import json
import time

import numpy as np

from mesh import ffi, lib, check
from mesh_mpi import Comm, empty
from routing import Fabric, QueueSample
from views import byte_spans

EXTENT = 1 << 30
POISON = 173


def mapped_case(rank, allocate=empty):
    """Different ordered source/relay/final maps, including a repeated read."""
    maps = [
        [(65573, 32771), (17, 65539), (65573, 32771)],
        [(35, 65541), (90001, 65540)],
        [(120003, 43691), (19, 43695), (60011, 43695)],
    ]
    spans = maps[rank]
    backing = allocate(max(offset + size for offset, size in spans) + 41, np.uint8)
    backing.fill(POISON)
    if rank == 0:
        for offset, size in spans:
            backing[offset:offset + size] = (np.arange(offset, offset + size, dtype=np.uint64) * 13 + 29) % 127
    expected = np.concatenate([
        ((np.arange(offset, offset + size, dtype=np.uint64) * 13 + 29) % 127).astype(np.uint8)
        for offset, size in maps[0]
    ])
    return dict(name='different-maps-repeated-source', backing=backing, own=backing,
                spans=spans, expected=expected, kind='spans')


def reversed_case(rank, allocate=empty):
    """One logical matrix, three different original strides and storage orders."""
    shape = (31, 17)
    origins = [19, 47 + 30 * 43 + 16 * 2, 29]
    strides = [(1, 37), (-43, -2), (23, 1)]
    backing = allocate(1471, np.uint8)
    backing.fill(POISON)
    own = np.ndarray(shape, np.uint8, buffer=backing, offset=origins[rank], strides=strides[rank])
    expected = ((np.arange(np.prod(shape), dtype=np.uint32) * 7 + 11) % 127).astype(np.uint8)
    if rank == 0:
        own[:] = expected.reshape(shape)
    shift, spans = byte_spans(shape, strides[rank], 1)
    return dict(name='transposed-source-reversed-target', backing=backing, own=own,
                spans=[(origins[rank] + shift + offset, size) for offset, size in spans],
                expected=expected, kind='view')


def registration_case(rank, allocate=empty, extent=EXTENT):
    """A span longer than one registration must cross an actual MR boundary.

    Sparse sentinel windows are the only initialized payload. Uninitialized
    payload bytes are transferred but are not claimed as numerically verified.
    The allocation and its untouched payload are never copied into a temporary.
    """
    size, offset = extent + 19, 17 + 12 * rank
    backing = allocate(offset + size + 47, np.uint8)
    backing[:offset] = POISON
    backing[offset + size:] = POISON
    starts = sorted({0, min(4093, size // 4), size // 2 - 7, extent - 64, extent + 3})
    probes = []
    for logical in starts:
        count = min(16, size - logical)
        values = ((np.arange(logical, logical + count, dtype=np.uint64) * 23 + 3) % 127).astype(np.uint8)
        if rank == 0:
            backing[offset + logical:offset + logical + count] = values
        elif rank == 1:
            backing[offset + logical:offset + logical + count] = POISON
        probes.append((logical, values))
    return dict(name='unaligned-registration-crossing', backing=backing, own=backing,
                spans=[(offset, size)], probes=probes, kind='spans', registration_extent=extent)


def verify(case, rank):
    backing, spans = case['backing'], case['spans']
    if 'probes' in case:
        offset, size = spans[0]
        np.testing.assert_array_equal(backing[:offset], POISON)
        np.testing.assert_array_equal(backing[offset + size:], POISON)
        if rank == 1:
            for logical, expected in case['probes']:
                np.testing.assert_array_equal(backing[offset + logical:offset + logical + len(expected)], expected)
        return sum(len(values) for _, values in case['probes']) if rank == 1 else 0
    guard = np.ones(backing.size, dtype=bool)
    for offset, size in spans:
        guard[offset:offset + size] = False
    np.testing.assert_array_equal(backing[guard], POISON)
    if rank == 1:
        # This concatenation is diagnostic output inspection, after Execute;
        # the library transmitted directly into the original indexed storage.
        actual = np.concatenate([backing[offset:offset + size] for offset, size in spans])
        np.testing.assert_array_equal(actual, case['expected'])
        return len(actual)
    return 0


def retired(array):
    """Observe local DMA retirement only, outside the tested execution."""
    ready, limit = ffi.new('int *'), time.monotonic() + 30
    while True:
        check(lib.ncclMeshMemReusable(ffi.cast('void *', array.ctypes.data), ready))
        if ready[0]:
            return
        if time.monotonic() >= limit:
            raise TimeoutError('source/relay local DMA retirement did not complete')
        time.sleep(0.001)


def traffic(before, after):
    old = {(row['cable'], row['wire']): row for row in before}
    if set(old) != {(row['cable'], row['wire']) for row in after}:
        raise AssertionError('physical fabric changed during the regression')
    return [dict(cable=row['cable'], wire=row['wire'], peer=row['peer'],
                 sent_bytes=row['sent'] - old[row['cable'], row['wire']]['sent'],
                 landed_bytes=row['landed'] - old[row['cable'], row['wire']]['landed']) for row in after]


def main():
    comm = Comm()
    retained, plans = [], []
    try:
        if comm.size != 3:
            raise ValueError('this regression requires exactly three physical ranks')
        fabric = Fabric.of(comm.comm)
        paths = fabric.paths[0, 1]
        covered = sorted({cable for path in paths for cable in path})
        if covered != list(range(len(fabric.cables))):
            raise ValueError('the three-rank 0-to-1 path set must include every discovered cable')
        sample = QueueSample(comm.comm, comm.rank)
        initial = sample.read()
        builders = (mapped_case, reversed_case, registration_case)
        for builder in builders:
            case = builder(comm.rank)
            retained.append(case)
            own = case['own']
            source, landing = (own, None) if comm.rank == 0 else (None, own)
            if case['kind'] == 'view':
                plan = comm.Write_view_prepare(source, landing, 0, 1, max_packet=1 << 20)
            else:
                plan = comm.Write_spans_prepare(source, landing, case['spans'], 0, 1, max_packet=1 << 20)
            plans.append(plan)
            before = sample.read()
            plan.Execute()
            # This does not authorize remote destination reuse. Every original
            # source, landing, plan and owner remains retained through Free.
            retired(case['backing'])
            verified = verify(case, comm.rank)
            delta = traffic(before, sample.read())
            payload = sum(size for _, size in case['spans'])
            if comm.rank == 0:
                assert sum(row['sent_bytes'] for row in delta) == payload, ('source payload', delta, payload)
            if comm.rank == 1:
                assert sum(row['landed_bytes'] for row in delta) == payload, ('destination payload', delta, payload)
            report = dict(rank=comm.rank, case=case['name'], correct=True, logical_payload_bytes=payload,
                          verified_destination_bytes=verified, neighboring_guards_preserved=True,
                          spans=len(case['spans']), retained_through_communicator_free=True,
                          traffic=delta, measurement_scope='correctness and physical counters; no throughput claim')
            if 'probes' in case:
                report.update(registration_extent_bytes=EXTENT, unaligned_span_origin=case['spans'][0][0],
                              actual_registration_crossing_proof='one contiguous declared span is longer than the 1 GiB registration extent',
                              sparse_payload_verification=True, whole_payload_verified=False,
                              initialized_payload_bytes=sum(len(values) for _, values in case['probes']))
            print(json.dumps(report), flush=True)
        final = traffic(initial, sample.read())
        active = sorted(row['cable'] for row in final if row['sent_bytes'] or row['landed_bytes'])
        expected = sorted(cable for _, cable in fabric.adjacent[comm.rank])
        assert active == expected, ('an incident physical cable carried no data', active, expected)
        print(json.dumps(dict(rank=comm.rank, correct=True, cases=len(plans), traffic=final,
                              active_incident_cables=active, discovered_cables=len(fabric.cables),
                              all_discovered_paths=len(paths), throughput_claim=False)), flush=True)
    finally:
        comm.Free()
        # Original array owners intentionally remain live during destruction.
        retained.clear()
        plans.clear()


if __name__ == '__main__':
    main()
