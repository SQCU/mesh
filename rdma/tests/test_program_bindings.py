"""Realization, masks and exact buffer aliases in the public Program frontend.

Native numerical effects are mocked; these tests do not claim NIC execution.
The real Program/Ref recording and retained issue functions are exercised.
"""
import importlib.util
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('program_binding_test', ROOT / 'python/mesh/__init__.py',
                                             submodule_search_locations=[str(ROOT / 'python/mesh')])
mesh = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mesh
spec.loader.exec_module(mesh)
collective = importlib.import_module('program_binding_test.collective')


class ProgramBindingsTests(unittest.TestCase):
    def test_logical_queues_and_subsets_share_one_physical_communicator(self):
        world = collective._native.World('cpu', None)
        world.group, world.rank, world.size = object(), 1, 4
        with patch.object(collective._native.dist, 'new_group', side_effect=AssertionError('extra communicator')):
            world.prepare([(0, (0, 2)), (7, (0, 1, 2, 3))])
        self.assertIs(world.of(0, (0, 2)), world.group)
        self.assertIs(world.of(7, (0, 1, 2, 3)), world.group)

    def test_constant_initialization_is_hoisted_and_recurrent_results_remain_correct(self):
        program = mesh.Program()
        initial_count, declarations = [], []
        constant, host, initialized, output = [program.tensor((1, 3), dtype=torch.int32)[0, 0] for _ in range(4)]
        program.constant(constant, 7)
        def once(_program, sources, targets, _coordinate):
            initial_count.append(1)
            targets[0].tensor.copy_(sources[0].tensor * 2)
        def recurrent(_program, sources, targets, _coordinate):
            targets[0].tensor.copy_(sources[0].tensor + sources[1].tensor)
        program._record('kernel', (0,), 0, ((constant,), (initialized,), ()),
                        (once, (constant,), (initialized,), (0, 0)))
        program._record('kernel', (0,), 0, ((initialized, host), (output,), ()),
                        (recurrent, (initialized, host), (output,), (0, 0)))
        result = program.export(output)
        try:
            with patch.object(program._world, 'declare', lambda tasks: declarations.append([t.index for t in tasks])):
                program.realize()
            self.assertEqual(declarations, [[0], [1]])
            self.assertEqual(initial_count, [1])
            for value in (2, 5):
                with program.write(host) as data:
                    data[:] = value
                deadline = time.monotonic() + 2
                while not result.ready and time.monotonic() < deadline:
                    time.sleep(.001)
                self.assertTrue(result.ready, 'recurrent Program result did not complete')
                self.assertTrue(torch.all(result.tensor == 14 + value))
                result.consume()
            self.assertEqual(initial_count, [1])
        finally:
            program.close()

    def program(self, rank=0, size=3):
        program = mesh.Program()
        program.rank = program.node = rank
        program.size = size
        group = object()
        program._world = SimpleNamespace(group=group, of=lambda *_: group,
                                         union_cuts=lambda task, cuts: sorted(set(cuts)))
        return program

    def setUp(self):
        self.allocations = []
        def allocate(count, dtype, device, regional):
            self.allocations.append((count, dtype, regional))
            return torch.empty(max(1, count), dtype=dtype, device=device)
        self.alloc = patch.object(collective._native, 'allocate', allocate)
        self.alloc.start()
        self.calls = patch.object(collective._native, 'call', lambda kind, count, dtype, **rest:
                                 SimpleNamespace(kind=kind, count=count, dtype=dtype, **rest))
        self.calls.start()
        self.write_plans = []
        def prepare(deliveries, group):
            self.write_plans.append(deliveries)
            return SimpleNamespace(execute=lambda **_: None)
        self.writes = patch.object(collective._native, 'prepare_writes', prepare)
        self.writes.start()

    def tearDown(self):
        self.alloc.stop()
        self.calls.stop()
        self.writes.stop()

    def test_adjacent_contiguous_alias_has_no_pack_or_unpack(self):
        program = self.program()
        value = program.tensor((2, 9), contiguous=True, dtype=torch.int32)
        ref = value[0, 0]
        refs = (ref.slice(0, 0, 1, 9), ref.slice(1, 0, 1, 9))
        ref.tensor.copy_(torch.arange(18).view(2, 9))
        task = SimpleNamespace(index=0)
        pieces = collective._alias_fragments(program, task, refs)
        aliases = [collective._alias(r, at, count) for r, at, count in pieces]
        self.assertEqual(aliases[0].data_ptr(), ref.tensor.data_ptr())
        before = len(self.allocations)
        for _ in range(3):
            for alias in aliases:
                alias.add_(1)
        self.assertEqual(len(self.allocations), before)
        self.assertTrue(torch.equal(ref.tensor, torch.arange(18).view(2, 9) + 3))

    def test_strided_movement_binds_original_and_final_views_without_packing(self):
        program = self.program()
        value = program.tensor((3, 7), dtype=torch.int8, contiguous=True)[0, 0]
        value.tensor.fill_(77)
        selected = value.slice(0, 1, 3, 3)
        before = len(self.allocations)
        target = program.tensor((3, 8), dtype=torch.int8, contiguous=True)[0, 0]
        target.tensor.fill_(88)
        receiving = target.slice(0, 2, 3, 3)
        moves = [(selected, 0, receiving, 2)]
        task = SimpleNamespace(queue=0, members=(0, 1, 2))
        collective._prepare_moves(program, task, moves)
        sent = self.write_plans[-1][0][0]
        program.rank = 2
        collective._prepare_moves(program, task, moves)
        landed = self.write_plans[-1][0][0]
        self.assertEqual(sent.data_ptr(), selected.tensor.data_ptr())
        self.assertEqual(sent.stride(), (7, 1))
        self.assertEqual(landed.data_ptr(), receiving.tensor.data_ptr())
        self.assertEqual(landed.stride(), (8, 1))
        count = len(self.allocations)
        for turn in (3, 8, 11):
            selected.tensor.fill_(turn)
            landed.copy_(sent)  # emulate the exact logical native write
            self.assertTrue(torch.all(receiving.tensor == turn))
            self.assertTrue(torch.all(value.tensor[:, :1] == 77))
            self.assertTrue(torch.all(value.tensor[:, 4:] == 77))
            self.assertTrue(torch.all(target.tensor[:, :2] == 88))
            self.assertTrue(torch.all(target.tensor[:, 5:] == 88))
        self.assertEqual(len(self.allocations), count)
        self.assertEqual(count - before, 1, 'only the requested final tensor was allocated')

    def test_copy_is_one_declared_common_exchange_on_all_physical_ranks(self):
        for rank in range(4):
            program = self.program(rank, 4)
            source = program.tensor((1, 3), dtype=torch.uint8)[0, 0]
            target = program.tensor((1, 3), dtype=torch.uint8)[0, 0]
            program.copy(source.on(0), target.on(2), queue=7)
            task, = program._tasks
            self.assertEqual(task.owners, (0, 1, 2, 3))
            self.assertEqual(task.members, task.owners)
            self.assertEqual(task.participants, (0, 2))
            self.assertIs(task.run[0], collective._issue_exchange)
            task.prepared = task.run[0].prepare(program, task)
            call = task.run[0].declaration(task)[0][0]
            self.assertEqual((call.kind, call.count, call.root, call.destination), ('WRITE', 3, 0, 2))

    def test_subset_gather_keeps_nonparticipant_as_physical_member(self):
        program = self.program(1, 4)
        value = program.tensor((2, 5), block_shape=(1, 5), dtype=torch.int32)
        collective.all_gather(program, value, peers=(0, 2), owners={(0, 0): 0, (1, 0): 2})
        for task in program._tasks:
            self.assertEqual(program._roles(task, 1), ((), (), ()))
            task.prepared = task.run[0].prepare(program, task)
            call = task.run[0].declaration(task)[0][0]
            self.assertEqual(call.kind, 'WRITE')
            self.assertEqual({call.root, call.destination}, {0, 2})
            self.assertEqual(call.count, 5 * torch.int32.itemsize)

    def test_reduce_scatter_owner_can_be_outside_contributors(self):
        for rank in range(3):
            program = self.program(rank)
            value = program.tensor((1, 5), dtype=torch.float32)
            collective.reduce_scatter(program, value, peers=(0, 1), owners={(0, 0): 2})
            task, = program._tasks
            task.prepared = task.run[0].prepare(program, task)
            call = task.run[0].declaration(task)[0][0]
            self.assertEqual(set(call.contributors), {0, 1})
            self.assertEqual(call.kind, 'REDUCE')
            self.assertEqual(call.root, 2)
            self.assertEqual(call.count, 5)
            self.assertEqual(task.participants, (0, 1, 2))

    def test_repeated_allreduce_uses_prepared_alias_and_native_operator(self):
        program = self.program()
        value = program.tensor((1, 7), dtype=torch.float32)
        collective.all_reduce(program, value, peers=(0, 2), op=collective.MAX)
        task, = program._tasks
        task.prepared = task.run[0].prepare(program, task)
        original = value[0, 0].tensor
        seen = []
        def reduce(bindings, group):
            for call, tensor, destination in bindings:
                self.assertEqual(tensor.data_ptr(), destination.data_ptr())
                seen.append((tensor.data_ptr(), call.op, group))
                tensor.add_(4)
        before = len(self.allocations)
        with patch.object(collective._native, 'fragments', reduce):
            for turn in range(3):
                original.fill_(turn)
                task.run[0](program, task)
                self.assertTrue(torch.all(original == turn + 4))
        self.assertEqual(len(self.allocations), before)
        self.assertTrue(all(ptr == original.data_ptr() and op == collective.MAX for ptr, op, _ in seen))
        call = task.run[0].declaration(task)[0][0]
        self.assertEqual(call.contributors, (0, 2))
        self.assertEqual(call.recipients, (0, 2))

    def test_disconnected_strided_reductions_alias_each_common_fragment(self):
        program = self.program()
        left = program.tensor((2, 7), dtype=torch.float32, contiguous=True)[0, 0]
        right = program.tensor((2, 6), dtype=torch.float32, contiguous=True)[0, 0]
        left.tensor.fill_(91)
        right.tensor.fill_(83)
        refs = (left.slice(0, 1, 2, 3), right.slice(0, 2, 2, 3))
        for ref in refs:
            ref.tensor.fill_(2)
        # Another rank's backing ends within our first contiguous row.
        program._world.union_cuts = lambda task, cuts: sorted(set([*cuts, 2]))
        task = SimpleNamespace(index=9, run=(collective._issue_all_reduce, refs, collective.SUM),
                               participants=(0, 2), queue=0, members=(0, 1, 2))
        before = len(self.allocations)
        task.prepared = collective._prepare_all_reduce(program, task)
        self.assertEqual(self.allocations[before:], [(0, torch.float32, True)], 'only an empty inactive operand may be allocated')
        bindings, _ = task.prepared
        self.assertEqual([call.count for call, _, _ in bindings], [2, 1, 3, 3, 3])
        for call, source, target in bindings:
            self.assertEqual(source.data_ptr(), target.data_ptr())
            self.assertEqual(call.contributors, (0, 2))
            target.mul_(3)  # mock native arithmetic writes only the bound original intervals
        self.assertTrue(all(torch.all(ref.tensor == 6) for ref in refs))
        self.assertTrue(torch.all(left.tensor[:, :1] == 91))
        self.assertTrue(torch.all(left.tensor[:, 4:] == 91))
        self.assertTrue(torch.all(right.tensor[:, :2] == 83))
        self.assertTrue(torch.all(right.tensor[:, 5:] == 83))

    def test_excluded_reduction_rank_binds_no_original_tensor_storage(self):
        program = self.program(rank=1)
        value = program.tensor((2, 5), block_shape=(1, 5), dtype=torch.int32)
        collective.all_reduce(program, value, peers=(0, 2))
        before = len(self.allocations)
        for task in program._tasks:
            task.prepared = collective._prepare_all_reduce(program, task)
            self.assertTrue(all(source.numel() == destination.numel() == 0
                                for _, source, destination in task.prepared[0]))
        self.assertTrue(all(count == 0 for count, _, _ in self.allocations[before:]))

    def test_partials_keep_complete_contiguous_records_with_row_gaps(self):
        for columns in (66, 130, 514):  # attention O widths64,128,512 plus m,l
            program = self.program()
            operators = []
            backend = SimpleNamespace(partials_operator=lambda width: operators.append(width) or 42)
            group = SimpleNamespace(_get_backend=lambda _: backend)
            program._world.group = group
            program._world.of = lambda *_: group
            whole = program.tensor((3, columns + 7), dtype=torch.float32, contiguous=True)[0, 0]
            whole.tensor.fill_(99)
            records = whole.slice(0, 2, 3, columns)
            task = SimpleNamespace(index=4, run=(collective._issue_partials, (records,), (2,), (0, 1)),
                                   participants=(0, 1, 2), queue=0, members=(0, 1, 2))
            task.prepared = collective._prepare_partials(program, task)
            calls = collective._declaration(task)[0]
            self.assertEqual([call.count for call in calls], [columns] * 3)
            self.assertEqual(operators, [columns] * 3)
            for row, (call, source, target) in enumerate(task.prepared[0]):
                self.assertEqual(source.data_ptr(), records.tensor[row].data_ptr())
                self.assertEqual(target.numel(), 0)
                self.assertEqual(call.root, 2)
                self.assertEqual(call.op, 42)
            transposed = program.tensor((columns, 3), dtype=torch.float32)[0, 0].T
            task.run = (collective._issue_partials, (transposed,), (2,), (0, 1))
            with self.assertRaisesRegex(ValueError, 'contiguous float32'):
                collective._prepare_partials(program, task)


if __name__ == '__main__':
    unittest.main()
