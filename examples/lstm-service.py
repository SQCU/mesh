import json
import time
import os
import sys

import numpy as np
import torch
import torch.distributed as dist
import torch_mesh
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Shard
from torch_mesh import partition
from torch_mesh.backend import BROADCAST, Call, bind_source
from torch_mesh.declarations import Function, execution_sequence
from torch_mesh.configuration import placement, bind, record
from torch_mesh import _stream
from torch_mesh.compilation import parameters
from mesh_statistics import now, next_configuration

import chain
import lstm

CHUNK = 512


# docs/writing-a-service.md#the-lstm
def weights(config, values):
    L, H = config['layers'], config['width']
    made = []
    for l in range(L):
        w_ih, w_hh = values[f'w_ih_{l}'], values[f'w_hh_{l}']
        W = torch_mesh.empty(4, H, w_ih.shape[1] + H, dtype=torch.half)
        for c in range(0, 4 * H, CHUNK):
            W.view(4 * H, -1)[c:c + CHUNK].copy_(torch.from_numpy(np.concatenate([w_ih[c:c + CHUNK], w_hh[c:c + CHUNK]], 1)))
        made.append((W, torch_mesh.empty(4, H, dtype=torch.half).copy_(torch.from_numpy(values[f'b_{l}'].reshape(4, H)))))
    return made


# docs/writing-a-service.md#the-lstm
def main(config=None):
    if config is None:
        config = json.load(sys.stdin if sys.argv[1] == '-' else open(sys.argv[1]))
    steps, rows = config['steps'], config['rows']
    inputs, held = None, None
    if os.environ['RANK'] == '0':
        inputs, values = lstm.inputs(config, steps, rows)
        held = weights(config, values)
        del values
    dist.init_process_group(backend='mesh', device_id=torch.device('mps', 0))
    rank, size = dist.get_rank(), dist.get_world_size()
    L, KIN, H = config['layers'], config['input_width'], config['width']
    mesh = DeviceMesh.from_group(dist.group.WORLD, 'mps', mesh_dim_names=('tp',))
    partition.attach(mesh, tp=H)
    workload = {'function': 'lstm_sequence', 'layers': L, 'input_width': KIN,
                'width': H, 'steps': steps, 'rows': rows}
    configuration = placement(mesh, 'tp', workload, config.get('placement'),
        parameters={**parameters(), **config['parameters']}, forms=config.get('forms', ()),
        observations=config.get('observations', []), proposal=config.get('proposal', 0), family=config.get('family', workload))
    chosen = configuration['assignment']['operands']
    compiler = {'LINEAR_ROW_CROSSOVER': chosen['LINEAR_ROW_CROSSOVER'],
                'layout_preferences': chosen['layout_preferences']}
    shape = lambda l: (4, H, (KIN if l == 0 else H) + H)
    split = lambda t: partition.distribute_tensor(t, mesh, [Shard(1)], src_data_rank=0)
    held = held or [(torch_mesh.empty(1, 1, shape(l)[2], dtype=torch.half).expand(shape(l)),
                     torch_mesh.empty(1, 1, dtype=torch.half).expand(4, H)) for l in range(L)]
    tick = lstm.Tick(mesh, [split(w) for w, _ in held], [split(b) for _, b in held], rows, steps)
    del held
    X = torch_mesh.empty(steps, rows, KIN, dtype=torch.half)
    O = torch_mesh.empty(steps, rows, H, dtype=torch.half) if rank == 0 else None
    t, k = torch.empty(1, dtype=torch.int64, device='mps').fill_(L - 1), chosen['steps_per_replay']
    group = mesh.get_group('tp')
    full, rest = divmod(steps, k)
    chunks = [k] * full + [rest] * bool(rest)
    from mesh_relation import Algebra
    algebra = Algebra()
    unit = Function(tick.stepper(X, O, 1), t, configuration=compiler, algebra=algebra,
                    source_scope={'program': 'lstm_sequence', 'phase': 'step'}).capture()
    # One declaration for each distinct chunk length, replayed for every chunk of that length: the chunks differ only
    # in the steps they run, which t (on the GPU) selects. Restored by the LSTM demonstration session from the form
    # before eee4d96b, which declared each of the steps / k chunks with its own occurrence offset: 256 compilations of
    # the same 16-step function at 4096 steps, each about as long as the last (about 1.3 s a step on the workstation,
    # 2026-11-01), so preparation never finished. What must stay: a chunk length is compiled once.
    functions = {}
    for count in dict.fromkeys(chunks):
        function = unit.repeated(count)
        function.declare(group)
        functions[count] = function
    functions = [functions[count] for count in chunks]
    initial = [Function(tick.starting, s, X[s], configuration=compiler, algebra=algebra,
                        source_scope={'program': 'lstm_sequence', 'phase': 'starting', 'step': s}) for s in range(L - 1)]
    for function in initial:
        function.declare(group)
    ordered = [*initial, *functions]
    # Each function's groups in their order; consecutive replays of one single-group declaration as (calls, n), which
    # the plan repeats without expanding their calls (torch_mesh.plan). At one step a replay the flat list was 4096
    # copies of the step's gather, and planning them held every rank for most of a minute (2026-11-01).
    groups = []
    for function in ordered:
        own = function.groups.get(group.group_name, [])
        if len(own) == 1 and groups and isinstance(groups[-1], tuple) and groups[-1][0] is own[0]:
            groups[-1] = (own[0], groups[-1][1] + 1)
        elif len(own) == 1:
            groups.append((own[0], 1))
        else:
            groups.extend(own)
    if rank == 0:
        X.copy_(torch.from_numpy(inputs[:]))
    torch_mesh.check_memory()
    ticks = steps + L - 1
    from torch_mesh.backend import prepare_fragments
    source = {'program': 'lstm_sequence', 'statement': 'inputs'}
    broadcast = bind_source(Call(BROADCAST, X.numel(), X.dtype, root=0), source=source,
        occurrences=_stream.TensorLayout((1,), (1,)),
        work=_stream.TensorLayout((X.numel() * X.element_size(),), (1,)), route_scope=source)
    # the broadcast's lowered fragments, each with the tensors it is issued with (design/eager-execution.md#what-follows)
    inputs, declared = prepare_fragments([(broadcast, X, X)], group)
    program = execution_sequence([inputs, *ordered], O, algebra=algebra)
    def prepare():
        program.realize()
        return {'layouts': [choice for function in ordered for choice in function.layout_choices],
                'choices': [choice for function in ordered for choice in function.choices]}
    bind(configuration, [declared, *groups], prepare)
    started = now()
    program.forward()
    completion = program.completion()
    blocks = partition.split(mesh, 'tp', H)
    summary = dict(blocks=blocks, steps_per_replay=k, layers=L, rows=rows,
                   input_width=KIN, width=H, steps=steps, ticks=ticks, latency_ticks=L)
    record(configuration, completion, started, steps * rows * H * torch.half.itemsize, summary)
    while not configuration['observations'].close():
        time.sleep(0)
    dist.destroy_process_group()
    state = next_configuration(configuration['observations'].declaration,
        [*configuration['group'].configuration_observations, *configuration['observations'].poll()],
        family=configuration['family']['source'])
    return {'output': O, 'configuration': state}


if __name__ == '__main__':
    main()
