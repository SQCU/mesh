"""Production function boundaries: feedback cannot change an in-flight tensor split."""
import ast
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'rdma'))
from controller import WholeFunctionController


class FunctionBoundaryTests(unittest.TestCase):
    def solver_type(self, clock):
        source = (ROOT / 'xonotic/solver/worker.py').read_text()
        cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'CtxSolver')
        # Compile the actual methods; avoid heavyweight model construction.
        namespace = {'np': np, 'time': types.SimpleNamespace(monotonic=clock), 'CTX': 8, 'REQ_WIDTH': 3}
        exec(compile(ast.Module(body=[cls], type_ignores=[]), 'worker.py', 'exec'), namespace)
        return namespace['CtxSolver']

    def test_prior_complete_latency_is_applied_after_numerical_completion(self):
        ticks = iter([10., 15.])
        Solver = self.solver_type(lambda: next(ticks))
        solver = Solver.__new__(Solver)
        controller = WholeFunctionController({'columns': {'parts': [64, 64], 'grain': 8}}, workload='solve')
        solver.balancer, solver.world, solver.T = controller, types.SimpleNamespace(size=2, native=object()), 8
        solver.columns, solver.solves, solver.pending_feedback = [64, 64], 0, None
        solver.previous_function = {'proposal': controller.proposal(), 'seconds': 1}
        solver.configure = lambda rows: controller.proposal()
        solver._insert = lambda X, Xn: np.arange(len(X))
        order = []
        def numerical(indices):
            solver.observe([3, 7])
            self.assertEqual(controller.flow.serial, 0)
            self.assertEqual(solver.columns, [64, 64])
            order.append('numerical complete')
            return np.ones((len(indices), 2), np.float32)
        solver._solve = numerical
        fake = types.SimpleNamespace(check=lambda status: self.assertEqual(status, 0),
                 lib=types.SimpleNamespace(ncclMeshCommPlanActive=lambda comm, active: order.append(('active', active)) or 0))
        with patch.dict(sys.modules, {'mesh': fake}):
            out = solver.solve(np.zeros((2, 3), np.float32))
        np.testing.assert_array_equal(out, np.ones((2, 2), np.float32))
        self.assertEqual(solver.previous_function['seconds'], 5)
        self.assertEqual(controller.flow.serial, 1)
        self.assertEqual(controller.flow.metric_contract[0], 'functions')
        self.assertEqual(order, ['numerical complete', ('active', 0)])
        self.assertEqual(solver.columns, [64, 64])

    def test_original_region_view_reaches_numpy_collective(self):
        source = (ROOT / 'xonotic/solver/worker.py').read_text()
        cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'MPIWorld')
        namespace = {'np': np}
        exec(compile(ast.Module(body=[cls], type_ignores=[]), 'worker.py', 'exec'), namespace)
        world = namespace['MPIWorld'].__new__(namespace['MPIWorld'])
        original = np.arange(6, dtype=np.float32).reshape(2, 3)
        def gather(send, target):
            self.assertTrue(np.shares_memory(send, original))
            target[0][:] = np.tile(send, 2)
        world.mpi = types.SimpleNamespace(empty=np.empty)
        world.comm = types.SimpleNamespace(Allgatherv=gather)
        parts = world.allgatherv(original, [6, 6])
        np.testing.assert_array_equal(parts, np.tile(original.reshape(1, -1), (2, 1)))


if __name__ == '__main__':
    unittest.main()
