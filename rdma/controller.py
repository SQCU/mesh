"""One complete-function controller for placement, engine shares and physical routes.

Realization/preparation is a configuration lifecycle. It never selects a measured
winner or turns a failed function into a synthetic timing. The immutable proposal
is the evidence identity; a caller must not change its operands inside a function.
"""
import copy
from whole import FunctionFlow


def declare(comm, calls):
    """Realize explicit native call descriptors once at a function boundary."""
    import mesh
    made, arrays = [], []
    for index, call in enumerate(calls):
        row = {'group': index, 'repeat': 1, **call}
        for field, ctype in (('counts', 'size_t'), ('matrix', 'size_t'),
                             ('contributors', 'uint64_t'), ('recipients', 'uint64_t')):
            if row.get(field) is not None:
                arrays.append(mesh.ffi.new(f'{ctype}[]', row[field]))
                row[field] = arrays[-1]
        row.setdefault('counts', mesh.ffi.NULL)
        made.append(row)
    result = mesh.ffi.new('ncclMeshPlan **')
    mesh.check(mesh.lib.ncclMeshCommPlan(comm, mesh.ffi.new('ncclMeshCall[]', made or 1), len(made), result))
    return [] if result[0] == mesh.ffi.NULL else [tuple(result[0].entries[g]) for g in range(result[0].groups)]


def function_demands(comm):
    """Return physical fabric and exact semantic demands of the declared graph."""
    import mesh
    from routing import Fabric
    fabric = Fabric.of(comm)
    values = mesh.ffi.new('uint64_t[]', fabric.nodes * fabric.nodes)
    mesh.check(mesh.lib.ncclMeshCommPlanDemands(comm, values, fabric.nodes * fabric.nodes))
    demands = {(a, b): int(values[a * fabric.nodes + b])
               for a in range(fabric.nodes) for b in range(fabric.nodes)
               if a != b and values[a * fabric.nodes + b]}
    return fabric, demands


class WholeFunctionController:
    VERSION = 'whole-function-controller-v1'

    def __init__(self, decisions, workload, topology=None, schedule=None,
                 calibrate=True, history_limit=64, **options):
        if not decisions:
            raise ValueError('a function controller needs declared decisions')
        self.workload = copy.deepcopy(workload)
        self.topology, self.schedule = copy.deepcopy(topology), copy.deepcopy(schedule)
        self.flow = FunctionFlow(decisions, **options)
        self.history_limit = max(4, int(history_limit))
        self.failures = []
        self._proposal = self.flow.ask(explore=not calibrate)

    @classmethod
    def from_fabric(cls, decisions, fabric, demands, workload, schedule=None, **options):
        demands = {pair: count for pair, count in demands.items() if count > 0}
        routes, bound = fabric.allocate(demands) if demands else ({}, {'seconds': 0})
        obj = cls({**decisions, **routes}, workload,
                  topology={'nodes': fabric.nodes, 'cables': fabric.cables,
                            'demands': [[a, b, n] for (a, b), n in demands.items()],
                            'fluid_bound': bound}, schedule=schedule, **options)
        return obj

    @classmethod
    def from_graphs(cls, decisions, graphs, workload, schedule=None, **options):
        """One vector for several declared communicators in the same function.

        Each graph supplies native nodes/cables/demands and, for a completed
        initial calibration, the exact installed weights. Names preserve the
        program's communicator identity rather than merging unrelated ranks.
        """
        from routing import Fabric
        combined, topology = copy.deepcopy(decisions), {}
        for name, graph in sorted(graphs.items()):
            fabric = Fabric(graph['nodes'], graph['cables'])
            demands = {(int(a), int(b)): int(n) for a, b, n in graph['demands'] if n > 0}
            routes, bound = fabric.allocate(demands) if demands else ({}, {'seconds': 0})
            for key, spec in routes.items():
                if 'weights' in graph:
                    if key not in graph['weights']:
                        raise ValueError(f'installed route weights missing {name}/{key}')
                    values = list(map(float, graph['weights'][key]))
                    import math
                    if len(values) != len(spec['parts']) or not all(v > 0 and math.isfinite(v) for v in values):
                        raise ValueError('installed route weights must name every physical path')
                    spec = {**spec, 'parts': values, 'high': [sum(values)] * len(values)}
                combined[f'flow:{name}/{key}'] = spec
            topology[name] = {'nodes': fabric.nodes, 'cables': copy.deepcopy(fabric.cables),
                              'demands': copy.deepcopy(graph['demands']), 'fluid_bound': bound}
        return cls(combined, workload, topology={'graphs': topology}, schedule=schedule, **options)

    @classmethod
    def from_comm(cls, decisions, comm, workload, schedule=None, **options):
        """Seed from the common native compiler's declared whole-function demand.

        The caller first declares the entire call graph with ncclMeshCommPlan.
        This query counts semantic deliveries with repeats, never relay bytes.
        After installing a new vector the caller re-realizes that declaration.
        """
        fabric, demands = function_demands(comm)
        return cls.from_fabric(decisions, fabric, demands, workload, schedule=schedule, **options)

    def proposal(self, explore=None):
        if explore is False:
            return copy.deepcopy(self.flow.ask(explore=False))
        return copy.deepcopy(self._proposal)

    ask = proposal

    @property
    def parts(self):
        return self.proposal()['parts']

    @property
    def round(self):
        return self.flow.round

    @property
    def history(self):
        return self.flow.history

    def observe(self, observation):
        expected = self._proposal
        if (observation.get('id') != expected['id'] or observation.get('parts') != expected['parts']
                or observation.get('workload') != self.workload):
            raise ValueError('whole-call evidence must name the exact realized proposal and workload')
        record = {**observation, 'phase': expected['phase']}
        self._proposal = self.flow.observe(record)
        self.flow.history[:] = self.flow.history[-self.history_limit:]
        return self.proposal()

    def completed(self, seconds, proposal=None, **evidence):
        """Submit one actually completed whole invocation (or declared repeated graph)."""
        proposal = self.proposal() if proposal is None else proposal
        return self.observe({**evidence, 'id': proposal['id'], 'parts': proposal['parts'],
                             'scope': 'function', 'complete': True,
                             'workload': self.workload, 'seconds': seconds})

    def failed(self, reason, proposal=None):
        """Record a structural/runtime failure. No rollback and no invented cost."""
        proposal = self.proposal() if proposal is None else proposal
        self.failures.append({'proposal': copy.deepcopy(proposal), 'reason': str(reason)})
        self.failures[:] = self.failures[-self.history_limit:]
        return self.proposal()

    def route_operands(self, proposal=None):
        """Serializable route operands, keyed by the declared communicator name."""
        parts = (proposal or self.proposal())['parts']
        if self.topology is None or 'graphs' not in self.topology:
            return {key: list(values) for key, values in parts.items() if key.startswith('route:')}
        return {name: {key[len(prefix):]: list(values) for key, values in parts.items() if key.startswith(prefix)}
                for name in self.topology['graphs'] for prefix in (f'flow:{name}/',)}

    def realize(self, comm, calls, proposal=None):
        """Install vector and recompile its full graph; refresh changed TP demand."""
        entries = declare(comm, calls)
        fabric, demands = function_demands(comm)
        if fabric.nodes != self.topology['nodes'] or fabric.cables != self.topology['cables']:
            raise ValueError('physical topology changed; realize a controller for the new fabric')
        self.topology['demands'] = [[a, b, n] for (a, b), n in demands.items()]
        self.schedule = {'calls': copy.deepcopy(calls)}
        return entries

    def state(self):
        return {'version': self.VERSION, 'workload': copy.deepcopy(self.workload),
                'topology': copy.deepcopy(self.topology), 'schedule': copy.deepcopy(self.schedule),
                'history_limit': self.history_limit, 'flow': self.flow.state(),
                'proposal': self.proposal(), 'failures': copy.deepcopy(self.failures)}

    @classmethod
    def of(cls, state):
        if state.get('version') != cls.VERSION:
            raise ValueError('unsupported whole-function controller state')
        obj = cls.__new__(cls)
        obj.workload, obj.topology = copy.deepcopy(state['workload']), copy.deepcopy(state.get('topology'))
        obj.schedule = copy.deepcopy(state.get('schedule'))
        obj.history_limit = int(state.get('history_limit', 64))
        obj.flow = FunctionFlow.of(state['flow'])
        obj._proposal, obj.failures = copy.deepcopy(state['proposal']), copy.deepcopy(state.get('failures', []))
        return obj
