"""A complete streamed read using the ordinary library and asynchronous statistics."""
import argparse
import torch
from torch_mesh import _stream, streamed
from mesh_statistics import now
import rollout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mib', type=int, required=True)
    parser.add_argument('--rounds', type=int, required=True)
    config = parser.parse_args()
    count = config.mib << 16
    data = torch.full((count * 8,), 0x5a5a, dtype=torch.int16, device='mps')
    output = torch.zeros(1024, dtype=torch.int32, device='mps')
    function = streamed.prepare_read(data, output, count, config.rounds)
    observer = rollout.start()
    completion = function.completion()
    started = now()
    function.enqueue()
    observer[0].put((completion, {'episode': '0', 'parts': {}, 'proposal': 0, 'graphs': {},
        'workload': {'mib': config.mib, 'rounds': config.rounds}, 'implementation': 'loaded-program',
        'started': started, 'rank': 0, 'size': 1, 'useful_work_bytes': count * 16 * config.rounds,
        'report': {'program': 'memory-read', 'kind': 'bandwidth', 'node': list(streamed.node()), 'steps': config.rounds}}))
    rollout.close(observer)


if __name__ == '__main__':
    main()
