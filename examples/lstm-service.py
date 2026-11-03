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
from torch_mesh.declarations import Function, execution_sequence
from torch_mesh.configuration import placement, bind, record
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
    expected = None
    # every rank draws the same inputs and weights from the configuration's seed and keeps its own share: no broadcast
    # of X into X and no scatter of the weights (design/page-maps-2026-11-02.md, Construction 7)
    inputs, values = lstm.inputs(config, steps, rows)
    held = weights(config, values)
    if os.environ['RANK'] == '0':
        # the outputs the run must give, from the same inputs and weights on the CPU (lstm.reference)
        expected = lstm.reference(config, inputs, values, steps)
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
    split = lambda t: partition.distribute_tensor(t, mesh, [Shard(1)])
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
    X.copy_(torch.from_numpy(inputs[:]))
    del inputs
    torch_mesh.check_memory()
    ticks = steps + L - 1
    program = execution_sequence(ordered, O, algebra=algebra)
    def prepare():
        # one realized schedule a run: Executions.run takes the next one (116b67ae ran a second time past the one
        # realized, and every rank faulted in the second forward)
        program.realize(config.get('repeats', 1))
        return {'layouts': [choice for function in ordered for choice in function.layout_choices],
                'choices': [choice for function in ordered for choice in function.choices]}
    bind(configuration, groups, prepare)
    blocks = partition.split(mesh, 'tp', H)
    # the prepared function run `repeats` times, each from the same state (the step counter and every layer's h and c
    # set again on the device, outside the timed function, once the run before has completed): the first run's rate
    # includes its first submission, the later ones are the function's own; each is one record
    for repeat in range(config.get('repeats', 1)):
        if repeat:
            t.fill_(L - 1)
            tick.reset()
            torch.mps.synchronize()
        started = now()
        program.forward()
        completion = program.completion()
        summary = dict(blocks=blocks, steps_per_replay=k, layers=L, rows=rows, repeat=repeat,
                       input_width=KIN, width=H, steps=steps, ticks=ticks, latency_ticks=L)
        record(configuration, completion, started, steps * rows * H * torch.half.itemsize, summary)
        while not configuration['observations'].close():
            time.sleep(0)
    if expected is not None:
        # rank 0's outputs, every step's now complete, against the reference: the largest error over every step, row
        # and unit, and over the last step
        error = np.abs(O.cpu().float().numpy() - expected)
        print(json.dumps(dict(phase='checked', max_abs_error=float(error.max()), last_step_max_abs_error=float(error[-1].max()),
                              output_max_abs=float(np.abs(expected).max()))), flush=True)
    dist.destroy_process_group()
    state = next_configuration(configuration['observations'].declaration,
        [*configuration['group'].configuration_observations, *configuration['observations'].poll()],
        family=configuration['family']['source'])
    return {'output': O, 'configuration': state}


if __name__ == '__main__':
    main()
