"""Live declared collectives over one full physical communicator.

All payloads use ordinary library operations. Role masks, ragged counts and
the count matrix are configuration operands; no transport is implemented here.
Fresh source leases make reuse independent of a remote acknowledgment: each
case's iterations are the steps of one declared program (design/declared-
programs.md), each on leases of its own, declared where the case begins.
"""
import json
import time

import numpy as np

from mesh import ffi, lib, check
from mesh_mpi import Comm, declare, empty
from routing import QueueSample


def pointer(array):
    return ffi.NULL if array is None else ffi.cast('void *', array.ctypes.data)


def bitmap(ranks, n):
    words = [0] * ((n + 63) // 64)
    for rank in ranks:
        words[rank // 64] |= 1 << (rank % 64)
    return ffi.new('uint64_t[]', words)


def main():
    comm = Comm()
    n, rank = comm.size, comm.rank
    if n < 3:
        raise ValueError('this regression requires two producers and an independent relay/recipient')
    retained = []
    cases = []
    observer = QueueSample(comm.comm, rank)
    before = observer.read()

    def run(what, inputs, outputs, counts=None, root=0, whole=0, average=False):
        amount = 131101
        total = sum(counts) if counts is not None else amount
        sources, destinations = bitmap(inputs, n), bitmap(outputs, n)
        sizes = ffi.new('size_t[]', counts) if counts is not None else ffi.NULL
        op = lib.ncclAvg if average else lib.ncclSum
        form = ffi.new('ncclMeshForm[]', [{'collective': lib.MESH_ALLREDUCE, 'root': 0, 'whole': whole}])
        check(lib.ncclMeshCommForms(comm.comm, form, 1))
        declaration = ffi.new('ncclMeshCall[]', [{'what': what, 'peer': root, 'count': amount,
            'counts': sizes, 'type': lib.ncclInt32, 'op': op,
            'contributors': sources, 'recipients': destinations}])
        check(lib.ncclMeshCommPlan(comm.comm, declaration, 1, ffi.NULL))
        demand = ffi.new('uint64_t[]', n*n)
        check(lib.ncclMeshCommPlanDemands(comm.comm, demand, n*n))
        send_count = counts[rank] if what == lib.MESH_ALLGATHER else total
        recv_count = counts[rank] if what == lib.MESH_REDUCE_SCATTER else total
        receives = rank == root if what == lib.MESH_REDUCE else rank in outputs
        leases = [(empty(send_count, np.int32) if rank in inputs else None, empty(recv_count, np.int32) if receives else None)
                  for _ in range(3)]
        retained.extend(x for pair in leases for x in pair if x is not None)

        def issue(args):
            if what == lib.MESH_ALLREDUCE:
                check(lib.ncclAllReduce(*args, total, lib.ncclInt32, op, comm.comm, ffi.NULL))
            elif what == lib.MESH_REDUCE:
                check(lib.ncclReduce(*args, total, lib.ncclInt32, op, root, comm.comm, ffi.NULL))
            elif what == lib.MESH_BROADCAST:
                check(lib.ncclBroadcast(*args, total, lib.ncclInt32, root, comm.comm, ffi.NULL))
            elif what == lib.MESH_ALLGATHER:
                check(lib.ncclMeshAllGatherV(*args, sizes, lib.ncclInt32, comm.comm, ffi.NULL))
            else:
                check(lib.ncclMeshReduceScatterV(*args, sizes, lib.ncclInt32, op, comm.comm, ffi.NULL))

        with declare():   # the case's program: its three iterations, each a step on its own leases
            for source, target in leases:
                issue((pointer(source), pointer(target)))
        for iteration, (source, target) in enumerate(leases):
            if source is not None:
                source[:] = np.arange(send_count, dtype=np.int32) + 100*rank + iteration
            if target is not None:
                target.fill(-99999)
            start = time.perf_counter()
            issue((pointer(source), pointer(target)))
            elapsed = time.perf_counter() - start
            if receives:
                if what == lib.MESH_ALLGATHER:
                    expected = np.concatenate([np.arange(counts[r], dtype=np.int32) + 100*r + iteration for r in range(n)])
                elif what == lib.MESH_BROADCAST:
                    expected = np.arange(total, dtype=np.int32) + 100*root + iteration
                else:
                    expected = len(inputs)*np.arange(total, dtype=np.int32) + sum(100*r + iteration for r in inputs)
                    if average:
                        expected //= len(inputs)
                    if what == lib.MESH_REDUCE_SCATTER:
                        first = sum(counts[:rank])
                        expected = expected[first:first+counts[rank]]
                np.testing.assert_array_equal(target, expected)
            result = {'rank': rank, 'what': int(what), 'whole': whole, 'average': average,
                'inputs': inputs, 'outputs': outputs, 'iteration': iteration, 'seconds': elapsed,
                'logical_directed_bytes': [int(demand[k]) for k in range(n*n)], 'correct': True}
            cases.append(result)
            print(json.dumps(result), flush=True)

    try:
        run(lib.MESH_ALLREDUCE, list(range(n)), list(range(n)))
        run(lib.MESH_ALLREDUCE, [0, 1], [n-1], whole=1)
        run(lib.MESH_ALLREDUCE, [0, 1], [0, n-1], average=True)
        run(lib.MESH_REDUCE, [0, 1], [n-1], root=n-1)
        run(lib.MESH_BROADCAST, [0], list(range(1, n)))
        run(lib.MESH_ALLGATHER, [0, n-1], [1], counts=[65539] + [0]*(n-2) + [32771])
        run(lib.MESH_REDUCE_SCATTER, [0, n-1], [1, n-1], counts=[0, 65539] + [0]*(n-3) + [32771])
        # Sparse exchange between two endpoints; every other rank supplies no
        # operand and still executes its portion of the compiled relay graph.
        amount = 1_048_579
        matrix = [0]*(n*n)
        matrix[1] = amount
        matrix[n] = amount
        local = matrix[rank*n:(rank+1)*n] + [matrix[r*n+rank] for r in range(n)]
        sizes, all_counts = ffi.new('size_t[]', local), ffi.new('size_t[]', matrix)
        declaration = ffi.new('ncclMeshCall[]', [{'what': lib.ncclMeshCallAlltoAllV, 'counts': sizes,
            'type': lib.ncclUint8, 'op': lib.ncclSum, 'matrix': all_counts}])
        check(lib.ncclMeshCommPlan(comm.comm, declaration, 1, ffi.NULL))
        source = empty(amount, np.uint8) if rank < 2 else None
        target = empty(amount, np.uint8) if rank < 2 else None
        if source is not None:
            source[:] = np.arange(amount, dtype=np.uint32) % 251 + rank
            target.fill(255)
        retained.extend(x for x in (source, target) if x is not None)
        with declare():
            check(lib.ncclMeshAlltoAllV(pointer(source), sizes, pointer(target), sizes+n, lib.ncclUint8, comm.comm, ffi.NULL))
        check(lib.ncclMeshAlltoAllV(pointer(source), sizes, pointer(target), sizes+n, lib.ncclUint8, comm.comm, ffi.NULL))
        if target is not None:
            np.testing.assert_array_equal(target, (np.arange(amount, dtype=np.uint32) % 251 + 1-rank).astype(np.uint8))
        # Counted and ordinary operations share one declared group and layout.
        # The logical endpoints still exclude the relay's numerical operand.
        summed_source, summed_target = empty(1003, np.int32), empty(1003, np.int32)
        summed_source[:] = np.arange(1003, dtype=np.int32) + 10*rank
        next_source = empty(amount, np.uint8) if rank < 2 else None
        next_target = empty(amount, np.uint8) if rank < 2 else None
        if next_source is not None:
            next_source[:] = (np.arange(amount, dtype=np.uint32)*7 + rank) % 251
        retained.extend(x for x in (summed_source, summed_target, next_source, next_target) if x is not None)
        declaration = ffi.new('ncclMeshCall[]', [
            {'group': 0, 'what': lib.ncclMeshCallAlltoAllV, 'counts': sizes, 'matrix': all_counts,
             'type': lib.ncclUint8, 'op': lib.ncclSum},
            {'group': 0, 'what': lib.MESH_ALLREDUCE, 'count': 1003, 'type': lib.ncclInt32, 'op': lib.ncclSum}])
        check(lib.ncclMeshCommPlan(comm.comm, declaration, 2, ffi.NULL))

        def grouped():
            check(lib.ncclGroupStart())
            check(lib.ncclMeshAlltoAllV(pointer(next_source), sizes, pointer(next_target), sizes+n, lib.ncclUint8, comm.comm, ffi.NULL))
            check(lib.ncclAllReduce(pointer(summed_source), pointer(summed_target), 1003, lib.ncclInt32, lib.ncclSum, comm.comm, ffi.NULL))
            check(lib.ncclGroupEnd())
        with declare():
            grouped()
        grouped()
        if next_target is not None:
            np.testing.assert_array_equal(next_target, ((np.arange(amount, dtype=np.uint32)*7 + 1-rank) % 251).astype(np.uint8))
        np.testing.assert_array_equal(summed_target, n*np.arange(1003, dtype=np.int32) + 5*n*(n-1))
        after = observer.read()
        traffic = [dict(cable=b['cable'], sent=a['sent']-b['sent'], landed=a['landed']-b['landed'])
                   for b, a in zip(before, after)]
        print(json.dumps({'rank': rank, 'cases': len(cases)+2, 'traffic': traffic, 'correct': True}), flush=True)
    finally:
        comm.Free()


if __name__ == '__main__':
    main()
