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
    def __init__(self, nodes, cables):
        self.nodes, self.cables = int(nodes), list(cables)
        self.adjacent = [[] for _ in range(self.nodes)]
        for e, c in enumerate(self.cables):
            a, b = c['nodes']
            self.adjacent[a].append((b, e))
            self.adjacent[b].append((a, e))
        self.paths = {}
        for source in range(self.nodes):
            for destination in range(self.nodes):
                if source == destination:
                    continue
                paths = []

                def visit(at, seen, path):
                    if at == destination:
                        paths.append(path)
                        return
                    for other, cable in self.adjacent[at]:
                        if other not in seen:
                            visit(other, seen | {other}, path + [cable])

                visit(source, {source}, [])
                if not paths:
                    raise ValueError(f'no path from {source} to {destination}')
                self.paths[source, destination] = paths

    @classmethod
    def of(cls, comm):
        import mesh
        out = mesh.ffi.new('struct mesh_fabric **')
        mesh.check(mesh.lib.ncclMeshCommFabric(comm, out))
        f = out[0]
        return cls(f.nodes, [{'wire': int(f.cable[e].wire), 'nodes': list(f.cable[e].node),
                              'bits_per_second': int(f.cable[e].bandwidth)} for e in range(f.cables)])

    def arcs(self, source, path):
        at, arcs = source, []
        for e in path:
            a, b = self.cables[e]['nodes']
            arcs.append(2 * e + (at == b))
            at = b if at == a else a
        return arcs
