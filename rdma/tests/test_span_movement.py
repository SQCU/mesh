"""Production span metadata and local data kernels; network calls are mocked.

The GPU checks execute the actual Torch/native indexed relocation kernel.
They assert byte identity and untouched guards, not transport throughput.
"""
import ast
import contextlib
import itertools
import math
from pathlib import Path
from types import MethodType, SimpleNamespace
import unittest

import torch

from test_generic_plans import functions
from views import byte_spans

ROOT = Path(__file__).resolve().parents[1]


def backend_prepare():
    tree = ast.parse((ROOT / 'torch-mesh/torch_mesh/backend.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ProcessGroupMesh')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'prepare_writes')
    env = {'_stream': SimpleNamespace(SpanWrites=lambda bindings, handle: (bindings, handle))}
    exec(compile(ast.Module(body=[method], type_ignores=[]), 'backend.py', 'exec'), env)
    return MethodType(env['prepare_writes'], SimpleNamespace(handle=23))


class SpanMovementTests(unittest.TestCase):
    def test_initial_scatter_has_one_foreign_import_and_no_destination_packing(self):
        for regional in (False, True):
            original = torch.arange(48, dtype=torch.int32).reshape(8, 6).T
            sizes, source = [3, 3, 2], 1
            groups, outputs, imports = [], [], []
            env = functions('_scatter_spans', '_row', _in_region=lambda _: regional,
                            _empty=lambda shape, like, group: torch.empty(shape, dtype=like.dtype),
                            _copy_view=lambda a, b: (imports.append((a, b)), b.copy_(a)))
            env['torch'] = torch
            for rank in range(3):
                group = SimpleNamespace(deliveries=[], declarations=[])
                def prepare(deliveries, group=group):
                    group.deliveries = deliveries
                    return SimpleNamespace(execute=lambda: None)
                def configuration(calls, group=group):
                    group.declarations = calls
                    return contextlib.nullcontext()
                group.prepare_writes, group.configuration = prepare, configuration
                outputs.append(env['_scatter_spans'](original, 1, sizes, rank, source, group))
                groups.append(group)
            self.assertEqual(len(imports), 0 if regional else 1)
            bound_source = original if regional else imports[0][1]
            self.assertEqual(outputs[source].untyped_storage().data_ptr(), bound_source.untyped_storage().data_ptr())
            for index, (sending, a, b) in enumerate(groups[source].deliveries):
                self.assertEqual(sending.untyped_storage().data_ptr(), bound_source.untyped_storage().data_ptr())
                self.assertEqual(groups[b].deliveries[index][0].data_ptr(), outputs[b].data_ptr())
                groups[b].deliveries[index][0].copy_(sending)
                declared = groups[source].declarations[0][index]
                self.assertEqual(declared.count, sending.numel() * sending.element_size())
                self.assertEqual(declared.dtype, torch.uint8)
            for rank, output in enumerate(outputs):
                self.assertTrue(torch.equal(output, original.narrow(1, sum(sizes[:rank]), sizes[rank])))

    def test_backend_preserves_original_view_storage_and_logical_byte_order(self):
        prepare = backend_prepare()
        storage = torch.arange(720, dtype=torch.int32).reshape(8, 9, 10)
        views = [storage, storage[1:6:2, 2:9:2, 1:9:3], storage.permute(2, 0, 1),
                 storage[2:3, 1:6, 2:8].expand(4, -1, -1), storage[:0]]
        for view in views:
            bindings, handle = prepare([(view, 0, 2)])
            base, spans, source, destination = bindings[0]
            self.assertEqual((handle, source, destination), (23, 0, 2))
            self.assertIs(base, view, 'metadata lowering must retain the original view')
            self.assertEqual(base.untyped_storage().data_ptr(), storage.untyped_storage().data_ptr())
            raw = torch.empty(0, dtype=torch.uint8).set_(base.untyped_storage(), 0,
                                                       (base.untyped_storage().nbytes(),), (1,))
            start = base.storage_offset() * base.element_size()
            actual = torch.cat([raw[start + offset:start + offset + count] for offset, count in spans]) if spans else raw[:0]
            expected = view.contiguous().view(torch.uint8).reshape(-1)
            self.assertTrue(torch.equal(actual, expected))
            self.assertEqual(sum(count for _, count in spans), view.numel() * view.element_size())

    def test_partition_strided_rows_write_originals_into_final_views(self):
        old, new = [4, 3, 1], [3, 3, 2]
        for dtype, shape, dim in itertools.product((torch.uint8, torch.int32, torch.float32),
                                                   ((8, 7), (3, 8, 5)), (0, 1)):
            if shape[dim] != 8:
                continue
            size = torch.empty(0, dtype=dtype).element_size()
            bits = torch.arange(math.prod(shape) * size).remainder(251).to(torch.uint8)
            bits[:min(16, bits.numel())] = 255
            full = bits.view(dtype).reshape(shape)
            sources, stores, groups, outputs = [], [], [], []
            for rank in range(3):
                block = full.narrow(dim, sum(old[:rank]), old[rank])
                # Gaps between every element, plus transposed storage strides.
                typed = torch.empty((block.numel() * 3 + 11,), dtype=dtype)
                typed.view(torch.uint8).fill_(173)
                dense = tuple(torch.empty(tuple(reversed(block.shape))).stride())
                strides = tuple(3 * x for x in reversed(dense))
                view = typed.as_strided(block.shape, strides, 3)
                view.copy_(block)
                sources.append(view)
                stores.append((typed, typed.view(torch.uint8).clone()))
                groups.append(SimpleNamespace(deliveries=[]))
            operands = {'old': SimpleNamespace(capacity=old, units=8),
                        'new': SimpleNamespace(capacity=new, units=8)}
            env = functions('_follow_spans', '_blocks', '_units', _OPERANDS=operands,
                            _empty=lambda shape, like, group: torch.empty(shape, dtype=like.dtype),
                            _copy_view=lambda a, b: b.copy_(a))
            env['torch'] = torch
            for rank, group in enumerate(groups):
                def prepare(deliveries, group=group):
                    group.deliveries = deliveries
                    return SimpleNamespace(execute=lambda: None)
                group.prepare_writes = prepare
                outputs.append(env['_follow_spans'](SimpleNamespace(dim=dim, key='old'), sources[rank],
                                                     shape, 'new', rank, group, lambda: self.fail('CPU publication')))
            relations = [(a, b) for _, a, b in groups[0].deliveries]
            self.assertTrue(relations)
            for group in groups:
                self.assertEqual([(a, b) for _, a, b in group.deliveries], relations)
            for index, (a, b) in enumerate(relations):
                sending = groups[a].deliveries[index][0]
                self.assertEqual(sending.untyped_storage().data_ptr(), sources[a].untyped_storage().data_ptr())
                landing = groups[b].deliveries[index][0]
                self.assertEqual(landing.untyped_storage().data_ptr(), outputs[b].untyped_storage().data_ptr())
                landing.copy_(sending)
                relays = [groups[r].deliveries[index][0] for r in range(3) if r not in (a, b)]
                for relay in relays:
                    self.assertTrue(relay.is_contiguous())
                    self.assertEqual(relay.numel(), sending.numel())
                    self.assertNotIn(relay.untyped_storage().data_ptr(),
                                     [v.untyped_storage().data_ptr() for v in (*sources, *outputs)])
            for rank, output in enumerate(outputs):
                expected = full.narrow(dim, sum(new[:rank]), new[rank]).contiguous()
                self.assertTrue(torch.equal(output.view(torch.uint8), expected.view(torch.uint8)))
                self.assertTrue(torch.equal(stores[rank][0].view(torch.uint8), stores[rank][1]))

    def test_span_binding_rejects_backing_overflow_before_native_communicator_access(self):
        from torch_mesh import _stream
        view = torch.empty(9, dtype=torch.uint8)[3:6]
        with self.assertRaisesRegex(RuntimeError, 'outside its original tensor storage'):
            _stream.SpanWrites([(view, [(5, 2)], 0, 1)], 0)

    @unittest.skipUnless(torch.backends.mps.is_available(), 'requires an Apple GPU')
    def test_actual_gpu_indexed_relocation_preserves_bits_and_neighbors(self):
        from torch_mesh.partition import _copy_view
        for dtype, shape in itertools.product((torch.uint8, torch.int16, torch.int32, torch.int64,
                                               torch.float16, torch.float32), ((3, 7), (2, 3, 5), (0, 5))):
            size = torch.empty(0, dtype=dtype).element_size()
            count = math.prod(shape)
            # Different column/row strides, both with holes and a nonzero offset.
            source_stride = tuple(3 * x for x in torch.empty(shape).stride())
            destination_stride = tuple(2 * x for x in reversed(torch.empty(tuple(reversed(shape))).stride()))
            original = torch.arange((count * 3 + 13) * size).remainder(251).to(torch.uint8)
            original[:min(128, original.numel())] = 255  # NaNs and arbitrary int bits
            poison = torch.full(((count * 2 + 17) * size,), 173, dtype=torch.uint8)
            source = original.view(dtype).as_strided(shape, source_stride, 3)
            expected = poison.clone()
            target = expected.view(dtype).as_strided(shape, destination_stride, 5)
            target.copy_(source)
            x, y = original.to('mps'), poison.to('mps')
            _copy_view(x.view(dtype).as_strided(shape, source_stride, 3),
                       y.view(dtype).as_strided(shape, destination_stride, 5))
            self.assertTrue(torch.equal(y.cpu(), expected), (dtype, shape, 'destination bits/guards'))
            self.assertTrue(torch.equal(x.cpu(), original), (dtype, shape, 'original overwritten'))


if __name__ == '__main__':
    unittest.main()
