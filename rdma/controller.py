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


def route_resolution(comm, nodes):
    """Compiler granularity, before repeats or physical relay amplification."""
    import mesh
    values = mesh.ffi.new('uint64_t[]', nodes * nodes)
    mesh.check(mesh.lib.ncclMeshCommPlanRouteUnits(comm, values, nodes * nodes))
    return {(a, b): int(values[a * nodes + b]) for a in range(nodes) for b in range(nodes)
            if a != b and values[a * nodes + b]}


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
    def from_fabric(cls, decisions, fabric, demands, workload, schedule=None, route_units=None, **options):
        demands = {pair: count for pair, count in demands.items() if count > 0}
        routes, bound = fabric.allocate(demands) if demands else ({}, {'seconds': 0})
        for (a, b), units in (route_units or {}).items():
            if units <= 0:
                raise ValueError('route allocation units must be positive')
            spec = routes[f'route:{a}:{b}']
            spec['quantum'] = sum(spec['parts']) / units
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
            units = {(int(a), int(b)): int(n) for a, b, n in graph['route_units']}
            if any(units.get(pair, 0) <= 0 for pair in demands):
                raise ValueError('every declared route demand needs compiler allocation granularity')
            routes, bound = fabric.allocate(demands) if demands else ({}, {'seconds': 0})
            for key, spec in routes.items():
                if 'weights' in graph:
                    if key not in graph['weights']:
                        raise ValueError(f'installed route weights missing {name}/{key}')
                    values = list(map(float, graph['weights'][key]))
                    import math
                    if len(values) != len(spec['parts']) or not all(v >= 0 and math.isfinite(v) for v in values) or not sum(values):
                        raise ValueError('installed route weights must name every physical path and have a positive total')
                    spec = {**spec, 'parts': values, 'high': [sum(values)] * len(values)}
                _, a, b = key.split(':')
                spec['quantum'] = sum(spec['parts']) / units[int(a), int(b)]
                combined[f'flow:{name}/{key}'] = spec
            topology[name] = {'nodes': fabric.nodes, 'cables': copy.deepcopy(fabric.cables),
                              'demands': copy.deepcopy(graph['demands']), 'route_units': copy.deepcopy(graph['route_units']),
                              'fluid_bound': bound}
        return cls(combined, workload, topology={'graphs': topology}, schedule=schedule, **options)

    @classmethod
    def from_comm(cls, decisions, comm, workload, schedule=None, **options):
        """Seed from the common native compiler's declared whole-function demand.

        The caller first declares the entire call graph with ncclMeshCommPlan.
        This query counts semantic deliveries with repeats, never relay bytes.
        After installing a new vector the caller re-realizes that declaration.
        """
        fabric, demands = function_demands(comm)
        return cls.from_fabric(decisions, fabric, demands, workload, schedule=schedule,
                               route_units=route_resolution(comm, fabric.nodes), **options)

    def proposal(self, explore=None):
        proposal = copy.deepcopy(self.flow.ask(explore=False) if explore is False else self._proposal)
        if hasattr(self, 'implementation'):
            proposal['implementation'] = self.implementation
        return proposal

    def bind_implementation(self, identity):
        """Keep placement, but never difference costs across code revisions.

        Called at configuration realization with the committed implementation's
        identity. A source change starts a fresh stencil at the same centre;
        completed history remains available for longitudinal comparisons.
        """
        if getattr(self, 'implementation', None) == identity:
            return self
        self.implementation = copy.deepcopy(identity)
        if self.flow.serial:
            self.flow.serial += 1
        self.flow.samples = []
        self._rollout_points = {}
        self._proposal = self.flow.ask(explore=self._proposal['phase'] != 'centre')
        return self

    def bind_fabric(self, comm):
        """Realize current physical coordinates, retaining the compute allocation.

        A stored route vector names concrete cables and paths. Changed cables
        require new route coordinates and fresh response observations; old
        gradients cannot be applied to different paths. Identical fabrics keep
        their accumulated controller and only refresh the declared demands.
        """
        fabric, demands = function_demands(comm)
        if (self.topology is not None and self.topology.get('nodes') == fabric.nodes
                and self.topology.get('cables') == fabric.cables):
            self.topology['demands'] = [[a, b, n] for (a, b), n in demands.items()]
            return self
        parts = self.proposal()['parts']
        decisions = {name: {**copy.deepcopy(spec), 'parts': parts[name]}
                     for name, spec in self.flow.decisions.items() if not name.startswith('route:')}
        return self.from_fabric(decisions, fabric, demands, self.workload, schedule=self.schedule,
                                route_units=route_resolution(comm, fabric.nodes),
                                history_limit=self.history_limit,
                                learning_rate=self.flow.learning_rate, perturbation=self.flow.perturbation,
                                transaction_weight=self.flow.transaction_weight)

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

    def completed_rollouts(self, records):
        """Accumulate asynchronous evidence for the common finite-difference solver.

        A proposal can execute many times while its statistics arrive in batches.
        Average those completed invocations with their sampling importance. Late
        observations refine any pending stencil point at its original operands;
        they never masquerade as observations of the currently proposed point.
        The ordinary +--+ stencil advances configuration, without blocking an
        executing function or selecting/rejecting measured candidates.
        """
        from whole import SHARE_TICKS, performance
        import math
        from fractions import Fraction
        retained = copy.deepcopy(getattr(self, '_rollouts', []))
        points = copy.deepcopy(getattr(self, '_rollout_points', {}))
        identities = {row['episode'] for row in retained}
        for record in records:
            if record['episode'] in identities:
                raise ValueError('a rollout episode may train the controller only once')
            if record['workload'] != self.workload:
                raise ValueError('a rollout belongs to a different whole function')
            proposal = record['proposal']
            if set(proposal['parts']) != set(self.flow.decisions):
                raise ValueError('rollout operands do not match the controller decisions')
            for name, spec in self.flow.decisions.items():
                parts = proposal['parts'][name]
                if (len(parts) != len(spec['parts']) or sum(parts) != spec['total']
                        or any(not lo <= value <= hi for value, lo, hi in zip(parts, spec['low'], spec['high']))):
                    raise ValueError('rollout must name feasible, actually applied operands')
            behavior = record.get('behavior_probability_ticks', SHARE_TICKS)
            target = record.get('target_probability_ticks', behavior)
            if (type(behavior) is not int or not 0 < behavior <= SHARE_TICKS
                    or type(target) is not int or not 0 <= target <= SHARE_TICKS):
                raise ValueError('sampling probabilities must be Q40 integers with positive behavior support')
            metrics = performance({**record.get('metrics', {}), 'seconds': record['seconds']}, self.flow.transaction_weight)
            contract = [metrics['useful_work_unit'], metrics['useful_work'],
                        metrics.get('useful_transaction_bytes') if self.flow.transaction_weight else None]
            if self.flow.metric_contract is not None and contract != self.flow.metric_contract:
                raise ValueError('rollouts must have invariant useful work and transaction accounting')
            self.flow.metric_contract = contract
            retained.append({**copy.deepcopy(record), 'cost': metrics['cost'],
                             'importance': float(Fraction(target, behavior))})
            identities.add(record['episode'])
            key = str(proposal['id'])
            point = points.setdefault(key, {'record': copy.deepcopy(record), 'weight': 0.0, 'seconds': 0.0})
            if point['record']['proposal'] != proposal:
                raise ValueError('a proposal identity must retain its realized operands')
            weight = float(Fraction(target, behavior))
            total = math.fsum((point['weight'], weight))
            if total:
                point['seconds'] += weight / total * (record['seconds'] - point['seconds'])
            point['weight'] = total
        if not records:
            return self.proposal()
        self._rollouts = retained[-self.history_limit:]
        # Retain summaries by configuration, not by arrival batch. A long run
        # of one proposal must not evict another pending stencil point's mean.
        self._rollout_points = {key: points[key] for key in
                                sorted(points, key=int)[-self.history_limit:]}

        def averaged(proposal_id):
            point = self._rollout_points.get(str(proposal_id))
            return (point['record'], point['seconds']) if point and point['weight'] else None

        # A late peer batch can improve earlier points of the stencil before its
        # fourth point arrives. Their coordinates and signs remain unchanged.
        for sample in self.flow.samples:
            evidence = averaged(sample['id'])
            if evidence is not None:
                row, seconds = evidence
                metrics = performance({**row.get('metrics', {}), 'seconds': seconds}, self.flow.transaction_weight)
                sample.update(seconds=seconds, cost=metrics['cost'], performance=metrics)
        evidence = averaged(self._proposal['id'])
        if evidence is None:
            return self.proposal()
        row, seconds = evidence
        if row['proposal']['parts'] != self._proposal['parts']:
            raise ValueError('a proposal identity must retain its realized operands')
        # Use the same solver as synchronous completed-function observations.
        # In particular, do not clear its stencil or move its centre after a
        # single positive probe: that was the global-regression path's defect.
        return self.observe({**row.get('metrics', {}),
            'id': self._proposal['id'], 'parts': self._proposal['parts'],
            'scope': 'function', 'complete': True, 'workload': self.workload,
            'seconds': seconds})

    def failed(self, reason, proposal=None):
        """Record a structural/runtime failure. No rollback and no invented cost."""
        proposal = self.proposal() if proposal is None else proposal
        self.failures.append({'proposal': copy.deepcopy(proposal), 'reason': str(reason)})
        self.failures[:] = self.failures[-self.history_limit:]
        return self.proposal()

    def route_operands(self, proposal=None):
        """Serializable route operands, keyed by the declared communicator name."""
        from routing import weight_ticks
        parts = (proposal or self.proposal())['parts']
        if self.topology is None or 'graphs' not in self.topology:
            return {key: weight_ticks(values) for key, values in parts.items() if key.startswith('route:')}
        return {name: {key[len(prefix):]: weight_ticks(values) for key, values in parts.items() if key.startswith(prefix)}
                for name in self.topology['graphs'] for prefix in (f'flow:{name}/',)}

    def install_routes(self, comm, proposal=None, graph=None):
        if self.topology is None:
            raise ValueError('route installation requires declared physical topology')
        from routing import Fabric
        if 'graphs' in self.topology:
            if graph not in self.topology['graphs']:
                raise ValueError('route installation must name its declared communicator')
            topology, weights = self.topology['graphs'][graph], self.route_operands(proposal)[graph]
        else:
            topology, weights = self.topology, self.route_operands(proposal)
        fabric = Fabric(topology['nodes'], topology['cables'])
        fabric.install(comm, weights)

    def realize(self, comm, calls, proposal=None):
        """Install vector and recompile its full graph; refresh changed TP demand."""
        self.install_routes(comm, proposal)
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
                'proposal': self.proposal(), 'failures': copy.deepcopy(self.failures),
                'rollouts': copy.deepcopy(getattr(self, '_rollouts', [])),
                'rollout_points': copy.deepcopy(getattr(self, '_rollout_points', {})),
                'implementation': copy.deepcopy(getattr(self, 'implementation', None))}

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
        obj._rollouts = copy.deepcopy(state.get('rollouts', []))
        obj._rollout_points = copy.deepcopy(state.get('rollout_points', {}))
        obj.implementation = copy.deepcopy(state.get('implementation'))
        return obj
