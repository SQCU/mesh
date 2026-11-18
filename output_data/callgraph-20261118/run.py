from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from compiler_emit import compile
from mesh_compiler.placement import place
from schedule_emit.driver import schedule_placed
from schedule_emit.export import export
from schedule_emit.runtime import Program


class Count(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("calls", torch.zeros(()))

    def forward(self):
        self.calls.add_(1)


def repeated():
    a = torch.nn.Linear(1, 1)
    b = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        a.weight.fill_(1)
        a.bias.fill_(2)
        b.weight.fill_(3)
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    graph.call_module("counter", ())
    first = graph.call_module("a", (x,))
    second = graph.call_module("b", (first,))
    last = graph.call_module("a", (second,))
    graph.output({"last": last, "aliases": (first, first)})
    source = torch.fx.GraphModule({"a": a, "b": b, "counter": Count()}, graph)
    sites = dict.fromkeys(graph.nodes, "cpu")
    placed = place(source, sites.__getitem__)
    for value in (1., -2.):
        t = torch.tensor([value], requires_grad=True)
        result = placed(t)
        torch.testing.assert_close(result, source(t))
        torch.testing.assert_close(torch.autograd.grad(result["last"], (t, a.bias)),
                                   (torch.tensor([3.]), torch.tensor([4.])))
        assert result["aliases"][0] is result["aliases"][1]
    assert source.counter.calls.item() == 4


class Product(torch.nn.Module):
    def forward(self, x, weight):
        return (x @ weight).relu()


def sharded():
    inputs = torch.arange(24.).reshape(3, 8), torch.arange(48.).reshape(8, 6) - 20
    model = Product()
    source = export(model, inputs)
    devices = dict.fromkeys(("left", "right"), torch.device("cpu"))

    def placement(node):
        if node.op == "placeholder" and node.target == "weight":
            return 1, tuple(devices), (1, 2)
        return "left"

    placed = place(source.graph_module, placement)
    scheduled = schedule_placed(source, placed, devices)
    assert scheduled.channels > 0
    with TemporaryDirectory() as directory:
        image = compile(scheduled, str(Path(directory) / "product.pt2"))
        for run in (Program(scheduled), Program(scheduled, image.functions)):
            for x in (inputs[0], -inputs[0], 2 * inputs[0]):
                torch.testing.assert_close(run(x, inputs[1]), model(x, inputs[1]))
    print(f"Two CPU placements, channels={scheduled.channels}, FX and compiled bodies matched three inputs")


if __name__ == "__main__":
    repeated()
    sharded()
