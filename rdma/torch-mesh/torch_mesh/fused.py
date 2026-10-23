from . import _stream

HEADER = _stream.metal_prelude() + '\n'


def header():
    return HEADER


pipeline, facts, device = _stream.pipeline, _stream.pipeline_facts, _stream.device_facts


class Exchange:
    def __init__(self, sizes, dtype, group=None):
        import torch
        import torch.distributed as dist
        group = group or dist.distributed_c10d._get_default_group()
        self.handle, self.scale = _stream.exchange_init(list(sizes), torch.empty(0, dtype=dtype), group.handle)
        from . import check_memory
        check_memory()

    def start(self, sizes=None):
        return _stream.exchange_start(self.handle, [n * self.scale for n in sizes] if sizes else [])

    def encode(self, handle, buffers, constants, at, gx, gy, threads, start, lands=False, publishes=False, crossing=14):
        _stream.encode(handle, buffers, constants, at, gx, gy, threads, self.handle, start, lands, publishes, crossing)

    def free(self):
        _stream.exchange_free(self.handle)
        self.handle = 0
