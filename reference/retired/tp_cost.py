# examples/tp.py at 1ac90a36, header of taken made unparseable by b25ec14b: lines 102-116 (Timed) and 119-126
# (taken). Dead reference source: see README.md


# design/unequal-shards-change-request.md item 7
class Timed(torch.fx.Interpreter):
    def __init__(self, module, device, times):
        super().__init__(module)
        self.device, self.times = device, times

    def run_node(self, node):
        started = time.perf_counter()
        result = super().run_node(node)
        if self.device.type == 'mps':
            torch.mps.synchronize()
        if node.op == 'call_function':
            self.times[node.meta['processor']].setdefault(shares.kind(node), {})[shares.units(node)[0]] = (
                time.perf_counter() - started)
        return result


# design/unequal-shards-change-request.md item 7; compiler/mesh_compiler/schedule.py assemble, members;
# compiler/mesh_compiler/runtime.py Arguments
taken(fixed, inputs):
    prepared = Arguments(fixed)
    module = fixed.modules[prepared.entry]
    times = {p: {} for p in module.meta['members']}
    Timed(module, prepared.device, times).run(*prepared.prepare(inputs, {}))
    return times
