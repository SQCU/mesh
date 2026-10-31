"""design/whole-function.md#ahead-of-time"""
import json
from pathlib import Path

# Preparation operands, loaded once when the compiler is imported. A missing
# manifest is an error; numerical execution never reads it.
NUMERICS = json.loads((Path(__file__).resolve().parents[1] / 'etc/compute/solver.json').read_text())
def forms(comm, entries):
    """NCCL.md, Forms as configuration: a configuration's forms installed on comm, in order (none: the library's table),
    each its collective (all-reduce, broadcast, reduce, reduce-scatter, all-gather), trees (own, packed), root, below,
    whole, most and eps, and its shares where any, the communicator's ranks times most (nccl.h ncclMeshForm)."""
    import mesh
    ffi, lib = mesh.ffi, mesh.lib
    collectives = {'all-reduce': lib.ncclMeshCallAllReduce, 'broadcast': lib.ncclMeshCallBroadcast, 'reduce': lib.ncclMeshCallReduce,
                   'reduce-scatter': lib.ncclMeshCallReduceScatter, 'all-gather': lib.ncclMeshCallAllGather}
    trees = {'own': lib.ncclMeshTreesOwn, 'packed': lib.ncclMeshTreesPacked}
    ranks = ffi.new('int *')
    mesh.check(lib.ncclCommCount(comm, ranks))
    made, kept = ffi.new('ncclMeshForm[]', max(len(entries), 1)), []
    for k, f in enumerate(entries):
        shares = f.get('shares')
        if shares is not None and len(shares) != ranks[0] * f['most']:
            raise ValueError(f"form {k}: its shares are the communicator's {ranks[0]} ranks times most ({f['most']})")
        kept.append(ffi.NULL if shares is None else ffi.new('double[]', shares))
        made[k] = {'collective': collectives[f['collective']], 'root': f['root'], 'below': f['below'], 'trees': trees[f['trees']],
                   'whole': f['whole'], 'most': f['most'], 'eps': f['eps'], 'shares': kept[-1]}
    mesh.check(lib.ncclMeshCommForms(comm, made, len(entries)))

class Fabric:
    """A communicator's physical fabric and each pair's write paths (a list of cable lists), as the library holds them
    (nccl.h ncclMeshCommFabric, ncclMeshCommWritePaths): read, never enumerated here (mesh-graph.h, C9)."""
    def __init__(self, nodes, cables, paths):
        self.nodes, self.cables, self.paths = int(nodes), list(cables), dict(paths)

    @classmethod
    def of(cls, comm):
        import mesh
        out = mesh.ffi.new('struct mesh_fabric **')
        mesh.check(mesh.lib.ncclMeshCommFabric(comm, out))
        f = out[0]
        held, paths = mesh.ffi.new('const struct mesh_paths **'), {}
        for a in range(f.nodes):
            for b in range(f.nodes):
                if a != b:
                    mesh.check(mesh.lib.ncclMeshCommWritePaths(comm, a, b, held))
                    p = held[0]
                    paths[a, b] = [[int(p.cable[q]) for q in range(p.first[k], p.first[k + 1])] for k in range(p.count)]
        return cls(f.nodes, [{'wire': int(f.cable[e].wire), 'nodes': list(f.cable[e].node),
                              'bits_per_second': int(f.cable[e].bandwidth)} for e in range(f.cables)], paths)
