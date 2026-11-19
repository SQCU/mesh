"""Common Call metadata reaches native AOT realization without losing scope."""
import ast
import collections
import contextlib
from pathlib import Path
from types import MethodType, SimpleNamespace
import unittest

import cffi

SOURCE = Path(__file__).resolve().parents[1] / 'torch-mesh/torch_mesh/backend.py'


class BackendPlanMetadataTests(unittest.TestCase):
    def environment(self, size=3, per=1):
        ffi = cffi.FFI()
        ffi.cdef('''
            typedef struct {
              int what, peer; size_t count; const size_t *counts; int type, op, group;
              unsigned repeat; int destination, fused;
              const size_t *matrix; const uint64_t *contributors, *recipients;
            } ncclMeshCall;
            typedef struct { int groups; int entries[128][2]; } ncclMeshPlan;
        ''')
        calls, active = [], []
        def prepare(comm, rows, count, result):
            calls.append([{'kind': rows[i].what, 'count': rows[i].count,
                           'counts': None if rows[i].counts == ffi.NULL else
                           [rows[i].counts[q] for q in range(size * (2 if rows[i].what == 5 else 1))],
                           'matrix': None if rows[i].matrix == ffi.NULL else [rows[i].matrix[q] for q in range(size * size)],
                           'contributors': None if rows[i].contributors == ffi.NULL else
                           [rows[i].contributors[q] for q in range((size + 63) // 64)],
                           'recipients': None if rows[i].recipients == ffi.NULL else
                           [rows[i].recipients[q] for q in range((size + 63) // 64)],
                           'repeat': rows[i].repeat, 'group': rows[i].group} for i in range(count)])
            return 0
        lib = SimpleNamespace(ncclSum=0, ncclProd=1, ncclMax=2, ncclMin=3,
                              ncclMeshCommPlan=prepare, ncclMeshCommPlanActive=lambda _comm, yes: active.append(yes) or 0)
        tree = ast.parse(SOURCE.read_text())
        call = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'Call' for t in n.targets))
        methods = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ProcessGroupMesh')
        functions = [n for n in methods.body if isinstance(n, ast.FunctionDef) and n.name in {'plan', 'plan_active', 'configuration'}]
        env = dict(ffi=ffi, LIB=lib, check=lambda result: self.assertEqual(result, 0), contextlib=contextlib,
                   collections=collections, ReduceOp=SimpleNamespace(SUM=0), WHAT=dict(enumerate(range(9))),
                   ALLREDUCE=0, REDUCE=1, REDUCE_SCATTER=4, typed=lambda dtype, reduces: (8, per),
                   torch=SimpleNamespace(bool='bool'), op=lambda value: value)
        exec(compile(ast.Module(body=[call, *functions], type_ignores=[]), str(SOURCE), 'exec'), env)
        group = SimpleNamespace(_size=size, comm='comm', _declaration=[], _plan_enabled=False)
        for name in ('plan', 'plan_active', 'configuration'):
            setattr(group, name, MethodType(env[name], group))
        return group, env['Call'], calls, active

    def test_exact_matrix_units_and_independent_masks(self):
        group, Call, calls, _ = self.environment(per=2)
        matrix = [0, 0, 3, 0, 0, 5, 0, 0, 0]
        group.plan([[Call(5, 3, 'complex-bits', counts=[0, 0, 3, 0, 0, 0], matrix=matrix,
                              contributors=(0, 1), recipients=(2,))]])
        row = calls[-1][0]
        self.assertEqual(row['count'], 6)
        self.assertEqual(row['counts'], [0, 0, 6, 0, 0, 0])
        self.assertEqual(row['matrix'], [x * 2 for x in matrix])
        self.assertEqual(row['contributors'], [3])
        self.assertEqual(row['recipients'], [4])

    def test_masks_preserve_ranks_above_one_machine_word_and_default_all(self):
        group, Call, calls, _ = self.environment(size=70)
        group.plan([[Call(0, 9, 'f32', contributors=(0, 65, 69), recipients=(64,))], [Call(0, 9, 'f32')]])
        self.assertEqual(calls[-1][0]['contributors'], [1, 34])
        self.assertEqual(calls[-1][0]['recipients'], [0, 1])
        self.assertIsNone(calls[-1][1]['contributors'])
        self.assertIsNone(calls[-1][1]['recipients'])

    def test_configuration_restores_paused_function_even_on_error(self):
        group, Call, calls, active = self.environment()
        graph = [([Call(0, 11, 'f32')], 4)]
        group.plan(graph)
        group.plan_active(False)
        with self.assertRaisesRegex(RuntimeError, 'producer failed'):
            with group.configuration([[Call(5, 3, 'u8', counts=[0, 0, 3, 0, 0, 0])]]):
                self.assertEqual(calls[-1][0]['kind'], 5)
                raise RuntimeError('producer failed')
        self.assertEqual(calls[-1][0]['kind'], 0)
        self.assertEqual(calls[-1][0]['repeat'], 4)
        self.assertFalse(active[-1])
        self.assertEqual(group._declaration, graph)

    def test_bad_matrix_cannot_expose_an_undersized_c_array(self):
        group, Call, calls, _ = self.environment()
        with self.assertRaisesRegex(ValueError, 'every ordered physical'):
            group.plan([[Call(5, 1, 'u8', matrix=[1])]])
        self.assertFalse(calls)


if __name__ == '__main__':
    unittest.main()
