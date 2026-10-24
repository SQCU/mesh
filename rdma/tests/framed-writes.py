"""Preposted one-way frame train: physical routes, ordered publication and masked consumption."""
import json
import torch
import torch.distributed as dist
import torch_mesh
from torch_mesh import _stream, fused
from routing import Fabric, QueueSample


def main():
    dist.init_process_group(backend='mesh', device_id=torch.device('mps', 0))
    group = dist.group.WORLD
    fabric = Fabric.of(group.comm)
    fabric.install(group.comm, {f'route:{a}:{b}': [1] * len(paths)
                               for (a, b), paths in fabric.paths.items()})
    channels = len(fabric.adjacent[group.rank()])
    frame_bytes, frames = group.page * 32, 7
    elements = frame_bytes // 4
    source = torch_mesh.empty(frames * elements, dtype=torch.int32)
    landing = torch_mesh.empty(frames * elements, dtype=torch.int32)
    result = torch_mesh.empty(frames * elements, dtype=torch.int32)
    poison = -1234567
    result.fill_(poison)
    shader = fused.steps() + f'#define CHANNELS {channels}\n' + r'''
struct Params { uint n, j, valid, pad; mesh_step_binding land[CHANNELS], publish[CHANNELS]; };
kernel void produce(device uchar *out [[buffer(0)]], constant Params &p [[buffer(1)]],
                    uint tid [[thread_position_in_threadgroup]]) {
    for (uint k = 0; k < CHANNELS; k++) if (p.publish[k].p)
        mesh_step_enter(*p.publish[k].p, p.publish[k].v, p.publish[k].progress, tid == 0);
    for (uint i = tid; i < p.n; i += 256)
        mesh_step_write<int>(out, ulong(p.j) * p.n + i, int(i % 997) + int(p.j) * 1009);
    for (uint k = 0; k < CHANNELS; k++) if (p.publish[k].p)
        mesh_step_release(*p.publish[k].p, p.publish[k].a, tid, p.j);
}
kernel void consume(device const uchar *input [[buffer(0)]], device int *out [[buffer(1)]],
                    constant Params &p [[buffer(2)]], uint tid [[thread_position_in_threadgroup]]) {
    for (uint k = 0; k < CHANNELS; k++) if (p.land[k].p) {
        mesh_step_enter(*p.land[k].p, p.land[k].v, p.land[k].progress, tid == 0);
        mesh_step_await(*p.land[k].p, p.land[k].a, p.land[k].v, tid, p.j);
    }
    for (uint i = tid; i < p.valid; i += 256)
        out[ulong(p.j) * p.n + i] = mesh_step_read<int>(input, ulong(p.j) * p.n + i) + 7;
}
'''
    producer, consumer = (_stream.pipeline(shader, name) for name in ('produce', 'consume'))
    for kwargs in ({'repeats': frames + 1, 'stride': frame_bytes},
                   {'repeats': frames, 'stride': frame_bytes + 1}, {'repeats': 0}):
        try:
            fused.writes(source[:elements], [landing[:elements]], [(0, 1)], group, **kwargs)
        except RuntimeError:
            pass
        else:
            raise AssertionError(('invalid frame declaration accepted', kwargs))
    sample = QueueSample(group.comm, group.rank())
    before = sample.read()
    crossing = fused.writes(source[:elements], [landing[:elements]], [(0, 1)], group,
                            repeats=frames, stride=frame_bytes)
    calls = []
    for j in range(frames):
        valid = elements - 17 if j == frames - 1 else elements
        if group.rank() == 0:
            calls.append((producer, [source], [elements, j, valid, 0], 1, 1, 1, 256,
                          channels, None, crossing))
        elif group.rank() == 1:
            calls.append((consumer, [landing, result], [elements, j, valid, 0], 2, 1, 1, 256,
                          channels, crossing, None))
    fused.encode_many(calls)
    torch.mps.synchronize()
    if group.rank() == 1:
        expected = torch.arange(elements, dtype=torch.int32).remainder(997).repeat(frames, 1)
        expected += torch.arange(frames, dtype=torch.int32).view(-1, 1) * 1009 + 7
        expected[-1, -17:] = poison
        torch.testing.assert_close(result.cpu().view(frames, elements), expected, rtol=0, atol=0)
    store = dist.distributed_c10d._get_default_store()
    store.set(f'frames-finished/{group.rank()}', b'1')
    for rank in range(group.size()):
        store.get(f'frames-finished/{rank}')
    after = sample.read()
    traffic = [{'cable': b['cable'], 'sent_bytes': b['sent'] - a['sent'],
                'landed_bytes': b['landed'] - a['landed']} for a, b in zip(before, after)]
    if group.rank() == 1:
        assert sum(t['landed_bytes'] for t in traffic) == frames * frame_bytes
        assert all(t['landed_bytes'] > 0 for t in traffic)
    print(json.dumps({'rank': group.rank(), 'frames': frames, 'frame_bytes': frame_bytes,
                      'one_preposted_group': True, 'masked_tail_elements': 17, 'passed': True,
                      'traffic': traffic}), flush=True)
    store.set(f'frames-drained/{group.rank()}', b'1')
    if group.rank() == 0:
        for rank in range(group.size()):
            store.get(f'frames-drained/{rank}')
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
