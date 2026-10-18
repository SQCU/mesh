"""The policy's products on the mesh library: the learner's world and the mappings its partitioned products are made of
(design/mesh-coprocessor-demo.md "Policy compute placed on the fabric"; xonotic/README.md "The policy learner").

The learner is one torch program on every rank of a world (torch.distributed's "mesh" backend, rdma/torch-mesh: one
rank a node, RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT as torchrun sets them).  Rank 0 is the game's host and runs the
responder; it announces each model call (World.announce: the call's name and its operands, one broadcast), and every
other rank runs the same call on the same operands (World.receive, execution.serve).  Outside the partitioned products
the ranks compute alike; a product with parts (the configuration's `products`, one part a rank) is divided as
Megatron-LM divides a tensor-parallel block [Shoeybi et al. 2019, mpu/mappings.py]: its operands enter through `copy`
(the identity forward, an all-reduce of their gradient backward), each rank applies its share, and `reduce` sums the
shares (an all-reduce forward, the identity backward).  torch's autograd derives every derivative from that forward,
the collectives' included, so a product's VJPs are the same program's backward on every rank (no remote VJP request,
no executor of its own).  Parts are relative: a product of n rows or columns gives rank r its largest-remainder share
(`split`), [0, 1] the whole product to rank 1 (57628982^'s RemoteCross placing the row-Gram cross on the Mini)."""
import json
import os

import numpy as np
import torch

DTYPES = {name: getattr(torch, name) for name in ('float32', 'float64', 'int64', 'int32', 'uint8', 'bool', 'float16')}


def split(total, parts):
    """Each rank's [lo, hi) of `total` units given relative `parts`: largest remainder, every rank alike."""
    weights = np.asarray(parts, np.float64)
    if weights.ndim != 1 or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError(f'parts {list(parts)}: one nonnegative part a rank, not all zero')
    raw = total * weights / weights.sum()
    counts = np.floor(raw).astype(np.int64)
    for index in np.argsort(-(raw - counts), kind='stable')[:total - int(counts.sum())]:
        counts[index] += 1
    bounds = np.concatenate(([0], np.cumsum(counts)))
    return [(int(lo), int(hi)) for lo, hi in zip(bounds[:-1], bounds[1:])]


class _Copy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, world):
        ctx.world = world
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient):
        return ctx.world.allreduce(gradient.contiguous().clone()), None


class _Reduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, world):
        return world.allreduce(value.contiguous().clone())

    @staticmethod
    def backward(ctx, gradient):
        return gradient, None


class World:
    """This process's place in the learner's world.  One process (WORLD_SIZE unset or 1) is a world of one, whose
    products are whole and whose calls are announced to no one."""

    def __init__(self, device, products=None):
        self.device = device
        self.products = {name: list(parts) for name, parts in (products or {}).items()}
        size = int(os.environ.get('WORLD_SIZE', '1'))
        self.dist = None
        self.rank, self.size = 0, 1
        if size > 1:
            import torch.distributed as dist
            import torch_mesh
            self.dist, self.torch_mesh = dist, torch_mesh
            dist.init_process_group(backend='mesh', **({'device_id': torch.device('mps', 0)} if device == 'mps' else {}))
            self.rank, self.size = dist.get_rank(), dist.get_world_size()
        for name, parts in self.products.items():
            if len(parts) != self.size:
                raise ValueError(f'products.{name} {parts}: one part a rank of {self.size}')
        self.header = self.buffer((3,), torch.int64, 'cpu')
        self.payload = None
        self.calls = 0

    def buffer(self, shape, dtype, device):
        if self.dist is None:
            return torch.empty(shape, dtype=dtype, device=device)
        return self.torch_mesh.empty(tuple(shape), dtype=dtype, device=device)

    def parts(self, name):
        return self.products.get(name) if self.size > 1 else None

    def span(self, name, total):
        parts = self.parts(name)
        return (0, total) if parts is None else split(total, parts)[self.rank]

    def allreduce(self, value, op='sum'):
        if self.dist is not None:
            self.dist.all_reduce(value, op={'sum': self.dist.ReduceOp.SUM, 'max': self.dist.ReduceOp.MAX}[op])
        return value

    def copy(self, value):
        return _Copy.apply(value, self) if self.size > 1 else value

    def reduce(self, value):
        return _Reduce.apply(value, self) if self.size > 1 else value

    def reduce_norm(self, op, value):
        box = torch.tensor([value], dtype=torch.float32)
        return float(self.allreduce(box, op)[0]) if self.size > 1 else value

    def broadcast_(self, tensors):
        """Rank 0's values of `tensors` (same shapes on every rank) written into every rank's, packed in one
        broadcast."""
        tensors = [tensor for tensor in tensors if tensor.numel()]
        if self.size == 1 or not tensors:
            return
        with torch.no_grad():
            packed = self.buffer((sum(tensor.numel() for tensor in tensors),), tensors[0].dtype, tensors[0].device)
            if self.rank == 0:
                torch.cat([tensor.reshape(-1) for tensor in tensors], out=packed)
            self.dist.broadcast(packed, 0)
            if self.rank:
                offset = 0
                for tensor in tensors:
                    tensor.copy_(packed[offset:offset + tensor.numel()].reshape(tensor.shape))
                    offset += tensor.numel()

    def _encode(self, value, chunks, offset):
        if value is None or isinstance(value, (bool, int, float, str)):
            return {'value': value}, offset
        if isinstance(value, np.generic):
            return {'value': value.item()}, offset
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            chunks.append(torch.from_numpy(array.reshape(-1).view(np.uint8).copy()))
            return {'numpy': array.dtype.str, 'shape': list(array.shape), 'offset': offset,
                    'bytes': int(array.nbytes)}, offset + int(array.nbytes)
        if isinstance(value, torch.Tensor):
            array = value.detach().cpu().contiguous()
            data = array.reshape(-1).view(torch.uint8) if array.numel() else torch.zeros(0, dtype=torch.uint8)
            chunks.append(data)
            return {'torch': str(array.dtype).removeprefix('torch.'), 'shape': list(array.shape), 'offset': offset,
                    'bytes': int(data.numel()), 'device': value.device.type}, offset + int(data.numel())
        if isinstance(value, dict):
            items = {}
            for key, part in value.items():
                items[key], offset = self._encode(part, chunks, offset)
            return {'dict': items}, offset
        if isinstance(value, (list, tuple)):
            items = []
            for part in value:
                encoded, offset = self._encode(part, chunks, offset)
                items.append(encoded)
            return {'list': items}, offset
        raise TypeError(f'a call operand of type {type(value).__name__}')

    def _decode(self, entry, payload):
        if 'value' in entry:
            return entry['value']
        if 'dict' in entry:
            return {key: self._decode(part, payload) for key, part in entry['dict'].items()}
        if 'list' in entry:
            return [self._decode(part, payload) for part in entry['list']]
        data = payload[entry['offset']:entry['offset'] + entry['bytes']].clone()
        if 'numpy' in entry:
            return np.frombuffer(data.numpy().tobytes(), entry['numpy']).reshape(entry['shape']).copy()
        dtype = DTYPES[entry['torch']]
        tensor = data.view(dtype).reshape(entry['shape']) if entry['bytes'] else torch.zeros(entry['shape'], dtype=dtype)
        return tensor.to(entry['device'])

    def announce(self, name, *operands):
        """Rank 0: every other rank runs `name` on `operands` (tensors, arrays, scalars and their lists and dicts)."""
        if self.size == 1 or self.rank:
            return
        chunks = []
        encoded, length = self._encode(list(operands), chunks, 0)
        meta = json.dumps({'name': name, 'operands': encoded}).encode()
        self.header[0], self.header[1], self.header[2] = 1, len(meta), length
        self.dist.broadcast(self.header, 0)
        if self.payload is None or self.payload.numel() < len(meta) + length:
            self.payload = self.buffer((max(len(meta) + length, 1 << 16),), torch.uint8, 'cpu')
        self.payload[:len(meta)] = torch.frombuffer(bytearray(meta), dtype=torch.uint8)
        if chunks:
            torch.cat(chunks, out=self.payload[len(meta):len(meta) + length])
        self.dist.broadcast(self.payload[:len(meta) + length], 0)
        self.calls += 1

    def receive(self):
        """Every other rank: the next call rank 0 announced, (name, operands), or None once rank 0 has stopped."""
        self.dist.broadcast(self.header, 0)
        kind, meta, length = (int(value) for value in self.header)
        if kind != 1:
            return None
        if self.payload is None or self.payload.numel() < meta + length:
            self.payload = self.buffer((max(meta + length, 1 << 16),), torch.uint8, 'cpu')
        self.dist.broadcast(self.payload[:meta + length], 0)
        message = json.loads(bytes(self.payload[:meta].numpy()).decode())
        self.calls += 1
        return message['name'], self._decode(message['operands'], self.payload[meta:meta + length])

    def stop(self):
        if self.size > 1 and self.rank == 0:
            self.header[0], self.header[1], self.header[2] = 0, 0, 0
            self.dist.broadcast(self.header, 0)

    def close(self):
        if self.dist is not None:
            self.dist.destroy_process_group()
            self.dist = None

    def report(self):
        return {'rank': self.rank, 'size': self.size, 'products': self.products, 'calls': self.calls}
