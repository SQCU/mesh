"""Trace actual forward/backward graphs without transport execution."""
import sys
from pathlib import Path
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'torch-mesh')]
import torch
import torch.distributed as dist
from torch_mesh.declarations import Function, trace


class FunctionDeclarations(unittest.TestCase):
    def test_complete_gradient_outputs_survive_eager_and_aot_compilation(self):
        for compiler in (None, 'aot_eager'):
            with self.subTest(compiler=compiler):
                model = torch.nn.Linear(3, 2, bias=False)
                with torch.no_grad():
                    model.weight.copy_(torch.arange(6).reshape(2, 3))
                x = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
                previous = model.weight.grad = torch.full_like(model.weight, -7)

                def step():
                    model.zero_grad(set_to_none=True)
                    y = model(x)
                    y.square().sum().backward()
                    return {'result': y}

                function = Function(step, parameters=model.parameters(), compiler=compiler)
                self.assertEqual(function.declare(types.SimpleNamespace(group_name='unused')), [])
                self.assertIs(model.weight.grad, previous, 'configuration must preserve real gradient state')
                for delta in (0., .25):
                    with torch.no_grad():
                        model.weight.add_(delta)
                    expected = x @ model.weight.detach().T
                    torch.testing.assert_close(function()['result'], expected)
                    torch.testing.assert_close(model.weight.grad, 2 * expected.T @ x)

    def test_actual_ragged_tp_forward_backward_is_traced_without_payload_execution(self):
        self.ragged_tp([1, 2])

    def test_zero_compute_tp_rank_retains_global_shapes_through_forward_backward(self):
        self.ragged_tp([0, 3])

    def ragged_tp(self, shares):
        from torch.testing._internal.distributed.fake_pg import FakeProcessGroup  # noqa: F401
        from torch.distributed.device_mesh import DeviceMesh
        from torch.distributed.tensor import Shard
        from torch.distributed.tensor.parallel import parallelize_module
        from torch_mesh import attention, partition
        from torch_mesh.partition import ColwiseParallel, RowwiseParallel
        dist.init_process_group('fake', store=dist.HashStore(), rank=0, world_size=2)
        try:
            mesh = DeviceMesh.from_group(dist.group.WORLD, 'cpu', mesh_dim_names=('tp',))
            partition.attach(mesh, tp=3)
            partition._attach(mesh, 'tp', shares)
            model = torch.nn.Sequential(torch.nn.Linear(6, 6, bias=False), torch.nn.SiLU(),
                                        torch.nn.Linear(6, 6, bias=False))
            parallelize_module(model, mesh, {'0': ColwiseParallel(), '2': RowwiseParallel()}, src_data_rank=None)
            x = torch.randn(2, 6)

            def step():
                model.zero_grad(set_to_none=True)
                y = model(x)
                y.sum().backward()
                return y

            before = dict(attention._FREE), dict(attention._MASKS)
            graph, declarations = trace(step, parameters=model.parameters())
            self.assertGreater(len(list(graph.graph.nodes)), 20)
            self.assertTrue(declarations[dist.group.WORLD.group_name])
            self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
            self.assertEqual((attention._FREE, attention._MASKS), before)
        finally:
            dist.destroy_process_group()


if __name__ == '__main__':
    unittest.main()
