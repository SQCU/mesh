# design/whole-function.md
import itertools
import json
import os
import sys
from pathlib import Path

import torch.distributed as dist
import assignment
import rollout
from mesh_source import revision
from . import partition
from .streamed import node

_serial = itertools.count()


def placement(mesh, name, workload, observations):
    group = mesh.get_group(name)
    rank, size = group.rank(), group.size()
    store = dist.distributed_c10d._get_default_store()
    key = f'mesh/configuration/{group.group_name}/{name}/{next(_serial)}'
    history = (json.load(sys.stdin) if os.environ.get('MESH_OBSERVATIONS_STDIN') else
               observations if isinstance(observations, list) else rollout.observations(observations))
    store.set(f'{key}/capacity/{rank}', str(node()[1]))
    capacities = [int(store.get(f'{key}/capacity/{p}')) for p in range(size)]
    if rank == 0:
        records = [row for row in history if row['workload'] == workload][-64:]
        store.set(key + '/observations', json.dumps(records))
    records = json.loads(store.get(key + '/observations'))
    operand = partition.operand(mesh, name)
    decisions = {name: {'parts': operand.parts.tolist(), 'grain': 1,
                        'low': [1] * size, 'high': [operand.units] * size, 'capacity': capacities}}
    graphs = records[-1]['graphs'] if records else {}
    current = {'nodes': group.fabric.nodes, 'cables': group.fabric.cables}
    if any(g['nodes'] != current['nodes'] or g['cables'] != current['cables'] for g in graphs.values()):
        records, graphs = [], {}
    decisions = assignment.domains(decisions, graphs)
    operands = assignment.assign(decisions, records)
    partition._attach(mesh, name, operands['parts'][name])
    return {'mesh': mesh, 'name': name, 'group': group, 'rank': rank, 'size': size,
            'store': store, 'key': key, 'history': history, 'workload': workload, 'records': records,
            'decisions': decisions, 'assignment': operands, 'graphs': graphs,
            'implementation': revision(Path(assignment.__file__).resolve().parents[1]), 'sequence': 0}


def bind(configuration, groups, prepare):
    c = configuration
    mesh, name, group = c['mesh'], c['name'], c['group']
    partition.plan(mesh, name, groups)
    graph = assignment.graph(group.comm)
    decisions = assignment.domains({name: c['decisions'][name]}, {name: graph})
    previous = c['assignment']['parts']
    operands = {k: previous.get(k, value['parts']) for k, value in decisions.items()}
    group.reset_fused_configuration()
    assignment.install(group.comm, operands, name)
    partition._replan(mesh, name)
    prepare()
    group.plan_active(True)
    c.update(decisions=decisions, graphs={name: graph}, assignment={**c['assignment'], 'parts': operands})
    c['observer'] = rollout.start(c['store'], c['key'] + '/observations', c['rank'], c['size'], c['history'])
    group._observers = getattr(group, '_observers', []) + [c['observer']]


def record(configuration, completion, started, useful_work_bytes, report):
    c = configuration
    episode = f"{c['key']}:{c['sequence']}"
    c['sequence'] += 1
    row = {'episode': episode, 'parts': c['assignment']['parts'], 'graphs': c['graphs'],
           'workload': c['workload'], 'implementation': c['implementation'], 'costs': {},
           'rank': c['rank'], 'size': c['size'], 'report': report, 'started': started, 'useful_work_bytes': useful_work_bytes}
    c['observer'][0].put((completion, row))
    return row
