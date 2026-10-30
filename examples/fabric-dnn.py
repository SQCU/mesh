# docs/measurement.md#physical-fabric-dnn
import json
import sys
from types import SimpleNamespace
from mesh_statistics import now
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch_mesh import partition
from torch_mesh.configuration import placement, bind, record
from torch_mesh.dnn import DNN


def main():
    args = SimpleNamespace(**(json.load(sys.stdin) if sys.argv[1] == '-' else json.loads(open(sys.argv[1]).read())))
    dist.init_process_group(backend='mesh', device_id=torch.device('mps', 0))
    mesh = DeviceMesh.from_group(dist.group.WORLD, 'mps', mesh_dim_names=('ffn',))
    partition.attach(mesh, ffn=args.expansion // 64)
    workload = {k: getattr(args, k) for k in ('rows', 'width', 'expansion', 'layers')}
    configuration = placement(mesh, 'ffn', workload, args.observations, parameters=args.parameters)
    args.tile_rows = configuration['assignment']['operands']['tile_rows']
    group = mesh.get_group('ffn')
    dnn = DNN(args, group, group.fabric)
    bind(configuration, dnn.declare(), lambda: dnn.prepare({'compute': partition.split(mesh, 'ffn', args.expansion)}))
    started = now()
    dnn.run()
    record(configuration, dnn.completion, started, args.rows * args.width * 4,
           {'program': 'fabric-dnn', 'steps': 1, **workload,
            'parts': configuration['assignment']['parts']['ffn']})
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
