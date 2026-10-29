# docs/measurement.md#physical-fabric-dnn
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from mesh_statistics import now
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch_mesh import _stream, partition
from torch_mesh.engines import host
from torch_mesh.backend import Call, WRITE
from torch_mesh.configuration import placement, bind, record
from torch_mesh.dnn import DNN


def main():
    args = SimpleNamespace(**(json.load(sys.stdin) if sys.argv[1] == '-' else json.loads(open(sys.argv[1]).read())))
    # design/replay-templates-2026-10-29.md: {'directory', 'phase'}: 'save' prepares the program as always and saves it,
    # its assignment and its output; 'load' binds that assignment and makes the program from the template, recording
    # nothing, and its output must be the recorded one, bit for bit
    template = getattr(args, 'template', None)
    loading = template is not None and template['phase'] == 'load'
    directory = Path(template['directory']) if template else None
    dist.init_process_group(backend='mesh', device_id=torch.device('mps', 0))
    mesh = DeviceMesh.from_group(dist.group.WORLD, 'mps', mesh_dim_names=('ffn',))
    partition.attach(mesh, ffn=args.expansion // 64)
    workload = {k: getattr(args, k) for k in ('rows', 'width', 'expansion', 'layers')}
    configuration = placement(mesh, 'ffn', workload, args.observations, parameters=args.parameters)
    if loading:
        chosen = json.loads((directory / 'assignment.json').read_text())
        configuration['assignment'] = {**configuration['assignment'], **chosen}
        partition._attach(mesh, 'ffn', chosen['parts']['ffn'])
    args.tile_rows = configuration['assignment']['operands']['tile_rows']
    group = mesh.get_group('ffn')
    dnn = DNN(args, group, group.fabric)
    preparation = {}

    def prepare():
        parts, begun = {'compute': partition.split(mesh, 'ffn', args.expansion)}, now()
        if loading:
            dnn.load(parts, str(directory))
        else:
            dnn.prepare(parts)
        preparation['seconds'] = now() - begun
    bind(configuration, dnn.declare(), prepare)
    rank = dist.get_rank()
    if template is not None and not loading:
        _stream.save(str(directory), [dnn.schedule], dnn.bound())
        chosen = {key: configuration['assignment'][key] for key in ('parts', 'operands')}
        (directory / 'assignment.json').write_text(json.dumps(chosen))
    started = now()
    dnn.run()
    record(configuration, dnn.completion, started, args.rows * args.width * 4,
           {'program': 'fabric-dnn', 'steps': 1, **workload,
            'parts': configuration['assignment']['parts']['ffn'],
            **({'template': template['phase'], 'preparation_seconds': preparation['seconds']} if template else {})})
    if template is not None:
        while not _stream.completed(dnn.completion):
            time.sleep(0.001)
        output, recorded = host(dnn.x).numpy(), directory / f'result.{rank}.npy'
        same = bool(np.array_equal(np.load(recorded), output)) if loading else None
        if not loading:
            np.save(recorded, output)
        print(json.dumps({'template': template['phase'], 'rank': rank, 'preparation_seconds': preparation['seconds'],
                          'output_as_recorded': same}), flush=True)
        if loading and not same:
            raise RuntimeError('template load: the output differs from the recorded run\'s')
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
