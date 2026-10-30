# design/whole-function.md
import itertools
import time
from pathlib import Path

import assignment
import rollout
import routing
from mesh_source import revision
from mesh_statistics import emit
from . import partition
from .streamed import node

_serial = itertools.count()


def placement(mesh, name, workload, chosen=None, *, parameters, forms=()):
    """The parts and operands of `name` on this collective as it is now (its members' capacities, its fabric), solved
    afresh: no earlier execution's observations are read, since nothing makes a collective the same machines from one
    execution to the next. `chosen`, a placement the caller passes to reuse a fit ({'parts': {name: [...]},
    'operands': {...}}, as a run's 'placed' record names it), is used as given; one whose parts do not fit this
    collective is refused."""
    group = mesh.get_group(name)
    rank, size = group.rank(), group.size()
    key = f'mesh/configuration/{group.group_name}/{name}/{next(_serial)}'
    capacities = group.exchange_configuration(int(node()[1]))
    operand = partition.operand(mesh, name)
    decisions = {name: {'parts': operand.parts.tolist(), 'grain': 1,
                        'low': [1] * size, 'high': [operand.units] * size, 'capacity': capacities}}
    decisions = assignment.domains(decisions, {}, parameters)
    if chosen is not None:
        parts = chosen['parts'][name]
        if len(parts) != size or sum(parts) != sum(decisions[name]['parts']):
            raise ValueError(f'the placement passed for {name} has parts {parts}, which do not fit this collective of '
                             f'{size} members and {sum(decisions[name]["parts"])} units')
        operands = {'parts': chosen['parts'], 'operands': chosen.get('operands', {}), 'models': {}, 'observations': 0,
                    'proposal': 0}
    else:
        operands = assignment.assign(decisions, [], step=0)
    # One proposal on every member (rubric C7): the solve's floating point can round its parts differently on another
    # node's hardware, and a split a row apart makes the members bind different bytes for one WRITE (hourly
    # 2026-10-29T06:21Z, chain-n3 and lstm-n3: "rank 4's bytes 493568 is not rank 0's 495616"). Rank 0's parts and
    # operands are taken, broadcast on the configuration channel.
    proposal = ({'parts': {k: [int(v) for v in values] for k, values in operands['parts'].items()},
                 'operands': {k: int(v) for k, v in operands['operands'].items()}} if rank == 0 else None)
    operands = {**operands, **group.broadcast_configuration(proposal)}
    partition._attach(mesh, name, operands['parts'][name])
    # every member leaves together, each having waited for rank 0's proposal; the fit is named, to be passed again
    emit({'phase': 'placed', 'unix': time.time(), 'placement': {'parts': operands['parts'], 'operands': operands['operands']}})
    return {'mesh': mesh, 'name': name, 'group': group, 'rank': rank, 'size': size,
            'key': key, 'workload': workload, 'records': [],
            'decisions': decisions, 'assignment': operands, 'graphs': {}, 'forms': list(forms),
            'implementation': revision(Path(assignment.__file__).resolve().parents[1]), 'sequence': 0}


def routes(group, name):
    """The declared function's demands (ncclMeshCommPlanDemands) routed over the fabric as it is, the flows that
    finish them soonest (assignment.domains: routing.Fabric.allocate), installed on the communicator, so the next plan
    compiles its transfers along them: what bind does for a partitioned function, for any declared one. The library's
    own routes, every simple path at one weight, are a configuration's absence: on the five nodes they put each byte of
    the all-to-all over about three cables and finish on the Mini's busiest one (design/measurement-regressions-
    2026-10-30.md#link-use). Its graph and decisions."""
    graph = assignment.graph(group.comm)
    decisions = assignment.domains({}, {name: graph})
    assignment.install(group.comm, {key: value['parts'] for key, value in decisions.items()}, name)
    return graph, decisions


def bind(configuration, groups, prepare):
    c = configuration
    mesh, name, group = c['mesh'], c['name'], c['group']
    production_seconds = (assignment.producer_seconds(c['assignment'], c['decisions'],
                                {name: [[rank] for rank in range(c['size'])]}) if c['records'] else [0.0] * c['size'])
    routing.forms(group.comm, c['forms'])   # before the declaration compiles its groups along them
    partition.plan(mesh, name, groups, production_seconds=production_seconds)
    emit({'phase': 'declared', 'unix': time.time()})   # the function traced and compiled (partition.plan's groups)
    graph = {**assignment.graph(group.comm), 'production_seconds': list(production_seconds), 'forms': c['forms']}
    decisions = assignment.domains({k: v for k, v in c['decisions'].items() if not k.startswith('flow:')}, {name: graph})
    previous = c['assignment']['parts']
    operands = {k: previous.get(k, value['parts']) for k, value in decisions.items()}
    group.reset_fused_configuration()
    assignment.install(group.comm, operands, name)
    partition._replan(mesh, name)
    emit({'phase': 'installed', 'unix': time.time()})   # the parts installed and planned again; then the recording
    prepare()
    emit({'phase': 'prepared', 'unix': time.time()})
    c.update(decisions=decisions, graphs={name: graph}, assignment={**c['assignment'], 'parts': operands})
    c['observer'] = rollout.start()
    group._observers = getattr(group, '_observers', []) + [c['observer']]


def record(configuration, completion, started, useful_work_bytes, report):
    c = configuration
    episode = f"{c['key']}:{c['sequence']}"
    c['sequence'] += 1
    row = {'episode': episode, 'parts': c['assignment']['parts'], 'graphs': c['graphs'],
           'workload': c['workload'], 'implementation': c['implementation'],
           'proposal': c['assignment']['proposal'], 'importance': 1.0,
           'rank': c['rank'], 'size': c['size'], 'report': report, 'started': started, 'useful_work_bytes': useful_work_bytes}
    c['observer'][0].put((completion, row))
    return row
