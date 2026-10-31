# design/whole-function.md
import itertools
import time
import copy
import math
from pathlib import Path

import assignment
import routing
from mesh_source import revision
from mesh_statistics import emit, Observations, cohorts, EXECUTED
from . import partition
from .backend import ffi, LIB
from .streamed import node

_serial = itertools.count()


def placement(mesh, name, workload, seed=None, *, parameters, observations, proposal, family, forms=()):
    """Prepare one proposal from declared choices and completed compatible episodes.

    ``seed`` carries prior parts, compiler operands and prepared routing graphs;
    it is the optimizer's starting point, not a second placement path. The caller
    names the source/member family and issued proposal ordinal. Only preparation
    exchanges ready reports; completion observation never plans or communicates.
    """
    group = mesh.get_group(name)
    rank, size = group.rank(), group.size()
    key = f'mesh/configuration/{group.group_name}/{name}/{proposal}/{next(_serial)}'
    seed = seed or {}
    member = ffi.new('uint32_t *')
    LIB.mesh_reached(ffi.NULL, 0, member)
    ready = [*observations, *group.configuration_observations]
    for observer in group.observations:
        ready.extend(observer.poll())
    reports = group.exchange_configuration({'member': int(member[0]), 'capacity': int(node()[1]), 'observations': ready,
        'proposal': proposal, 'family': family, 'workload': workload, 'seed': seed})
    for field in ('proposal', 'family', 'workload', 'seed'):
        if any(report[field] != reports[0][field] for report in reports):
            raise ValueError(f'configuration members disagree on {field}')
    complete, pending = cohorts([row for report in reports for row in report['observations']])
    histories = []
    for row in complete:
        history = next((rows for rows in histories if rows[0]['family'] == row['family']
                        and rows[0]['workload'] == row['workload']), None)
        if history is None:
            histories.append([row])
        else:
            history.append(row)
    group.configuration_observations = [*(row for rows in histories for row in assignment.window(rows)), *pending]
    capacities = [report['capacity'] for report in reports]
    graphs = copy.deepcopy(seed.get('graphs', {}))
    layouts = copy.deepcopy(seed.get('layouts', []))
    declared_family = {'source': copy.deepcopy(family), 'members': [report['member'] for report in reports],
                       'workload': copy.deepcopy(workload)}
    operand = partition.operand(mesh, name)
    decisions = {name: {'parts': operand.parts.tolist(), 'grain': 1, 'model': 'affine',
                        'low': [0] * size, 'high': [operand.units] * size, 'capacity': capacities}}
    parameters = {key: {**value, 'value': seed.get('operands', {}).get(key, value['value'])}
                  for key, value in parameters.items()}
    work = decisions
    decisions = assignment.domains(work, graphs, parameters, layouts=layouts)
    for choice, parts in seed.get('parts', {}).items():
        if choice not in decisions or len(parts) != len(decisions[choice]['parts']) or not math.isclose(sum(parts), sum(decisions[choice]['parts'])):
            raise ValueError(f'{choice}: seed parts do not fit the declared choice domain')
        decisions[choice] = {**decisions[choice], 'parts': list(parts)}
    records = [row for row in complete if row['workload'] == workload]
    # One proposal on every member (rubric C7): the solve's floating point can round its parts differently on another
    # node's hardware, and a split a row apart makes the members bind different bytes for one WRITE (hourly
    # 2026-10-29T06:21Z, chain-n3 and lstm-n3: "rank 4's bytes 493568 is not rank 0's 495616"). Rank 0's parts and
    # operands are taken, broadcast on the configuration channel.
    operands = assignment.assign(decisions, records, step=proposal, family=declared_family) if rank == 0 else None
    agreed = group.broadcast_configuration({'assignment': operands, 'configuration': key} if rank == 0 else None)
    operands, key = agreed['assignment'], agreed['configuration']
    operands['operands']['layout_preferences'] = assignment.layout_operands(operands['parts'], layouts)
    partition._attach(mesh, name, operands['parts'][name])
    # every member leaves together, each having waited for rank 0's proposal; the fit is named, to be passed again
    emit({'phase': 'placed', 'unix': time.time(), 'configuration': key, 'proposal': proposal,
          'placement': {'parts': operands['parts'], 'operands': operands['operands']}})
    return {'mesh': mesh, 'name': name, 'group': group, 'rank': rank, 'size': size,
            'key': key, 'workload': workload, 'records': records, 'family': declared_family,
            'work': work, 'parameters': parameters, 'decisions': decisions, 'assignment': operands,
            'graphs': graphs, 'layouts': layouts, 'forms': list(forms),
            'implementation': revision(Path(assignment.__file__).resolve().parents[1]), 'sequence': 0}


def bind(configuration, groups, prepare):
    """Bind one proposal and retain its executed operands for the next call.

    ``prepare`` returns ``{'layouts': [...]}`` from the Function declarations
    it realized, or an explicit empty list for a declaration with no layout
    choices. The returned seed is accepted by :func:`placement` unchanged.
    """
    c = configuration
    mesh, name, group = c['mesh'], c['name'], c['group']
    production_seconds = (assignment.producer_seconds(c['assignment'], c['decisions'],
                                {name: [[rank] for rank in range(c['size'])]}) if c['assignment']['observations'] else [0.0] * c['size'])
    routing.forms(group.comm, c['forms'])   # before the declaration compiles its groups along them
    graph = c['graphs'].get(name)
    routed = assignment.route_operands(c['assignment']['parts'], name, graph) if graph is not None else None
    partition.plan(mesh, name, groups, production_seconds=production_seconds, routing=routed)
    emit({'phase': 'declared', 'unix': time.time()})   # the function traced and compiled (partition.plan's groups)
    graph = {**assignment.graph(group.comm, group._plan), 'production_seconds': list(production_seconds), 'forms': c['forms']}
    emit({'phase': 'installed', 'unix': time.time()})
    report = prepare()
    layouts = [choice for report in group.exchange_configuration(report) for choice in report['layouts']]
    emit({'phase': 'prepared', 'unix': time.time()})
    work = {key: {**value, 'parts': c['assignment']['parts'][key]} for key, value in c['work'].items()}
    parameters = {key: {**value, 'value': c['assignment']['operands'][key]} for key, value in c['parameters'].items()}
    decisions = assignment.domains(work, {name: graph}, parameters, layouts=layouts)
    parts = {key: value['parts'] for key, value in decisions.items()}
    operands = {**c['assignment']['operands'], 'layout_preferences': assignment.layout_operands(parts, layouts)}
    c.update(decisions=decisions, graphs={name: graph}, layouts=layouts,
             assignment={**c['assignment'], 'parts': parts, 'domains': assignment.choice_domains(decisions), 'operands': operands})
    c['prepared'] = copy.deepcopy({'parts': parts, 'operands': operands, 'graphs': c['graphs'], 'layouts': layouts})
    c['observations'] = Observations(declaration={
        'configuration': c['key'], 'proposal': c['assignment']['proposal'],
        'parts': c['assignment']['parts'], 'domains': c['assignment']['domains'], 'operands': c['assignment']['operands'],
        'graphs': c['graphs'], 'layouts': c['layouts'], 'family': c['family'], 'workload': c['workload'],
        'implementation': c['implementation'], 'rank': c['rank'], 'size': c['size']})
    group.observations.append(c['observations'])
    return c['prepared']


def record(configuration, completion, started, useful_work_bytes, report):
    c = configuration
    episode = f"{c['key']}:{c['sequence']}"
    c['sequence'] += 1
    # Preparation snapshots the immutable executed operands once. Episodes
    # retain that declaration, rather than copying routing arrays at submission.
    declared = c['observations'].declaration
    row = {**{key: declared[key] for key in EXECUTED if key in declared},
           'episode': episode, 'importance': 1.0, 'rank': c['rank'], 'size': c['size'],
           'report': report, 'started': started, 'useful_work_bytes': useful_work_bytes}
    c['observations'].put(completion, row)
    return row
