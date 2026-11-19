"""design/declared-programs.md "Testing": one rank of a two-rank mesh_mpi communicator over the two-node simulator
(tests/sim-two-nodes.sh): a program of an Allreduce, a Bcast, a Barrier and an allreduce of a number declared with
`with declare():`, run ROUNDS times with every value checked, then a Bcast of another size from rank 1 declared again and run ahead of its receiver.
_mesh_c is found first on PYTHONPATH (the simulator's build of it), then beside rdma/.  usage: sim-mpi.py RANK [ROUNDS]"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

from mesh_mpi import Comm, SUM, empty, declare  # noqa: E402


def main():
    rank, rounds = int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 20
    os.environ['RANK'], os.environ['WORLD_SIZE'] = str(rank), '2'
    comm = Comm()
    x, y, b = empty(1536, np.float32), empty(1536, np.float32), empty(24576, np.float32)
    with declare():
        comm.Allreduce(x, y, op=SUM)
        comm.Bcast(b, root=0)
        comm.Barrier()
        comm.allreduce(1.0)
    for r in range(rounds):
        x[:] = rank + 1 + r
        b[:] = r if rank == 0 else -1
        comm.Allreduce(x, y, op=SUM)
        comm.Bcast(b, root=0)
        comm.Barrier()
        total = comm.allreduce(float(rank + r))
        assert np.all(y == 3 + 2 * r), (r, y[:4])
        assert np.all(b == r), (r, b[:4])
        assert total == 1 + 2 * r, (r, total)
    big = empty(393216, np.float32)
    with declare():
        comm.Bcast(big, root=1)
    big[:] = 7 if rank == 1 else 0
    for r in range(64):
        comm.Bcast(big, root=1)   # the root's call returns at its issue: it runs ahead, its source unchanged
        if rank == 0:
            assert np.all(big == 7), r
            big[:] = 0
    comm.Free()
    print(f'{{"rank":{rank},"bindings":"mesh_mpi","rounds":{rounds},"checked":true}}')


if __name__ == '__main__':
    main()
