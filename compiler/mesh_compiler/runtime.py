# design/data-flow.md
from queue import SimpleQueue
from threading import Thread

import torch
from torch.export.graph_signature import OutputKind
from torch.utils import _pytree as pytree

from . import ir


class Program:
    def __init__(self, scheduled, functions=None):
        self.exported, self.output_processor = scheduled.exported, scheduled.output_processor
        self.functions = functions or scheduled.modules
        self.devices = {processor: getattr(self.functions.get(processor), "device", device)
                        for processor, device in scheduled.devices.items()}
        self.feedforward = set(scheduled.feedforward)
        self.channels, self.received = [SimpleQueue() for _ in range(scheduled.channels)], {}
        machines = set(scheduled.machines.values())
        if len(machines) > 1:
            from . import bridge
            machines = {bridge.channels(scheduled, self.channels, self.received)}
        machine = machines.pop()
        inputs = {processor: names for processor, names in scheduled.inputs.items()
                  if scheduled.machines[processor] == machine}
        self.caller = self.output_processor if self.output_processor in inputs else next(iter(inputs), None)
        state = {name: tensor.detach() for name, (_, tensor) in self.exported.state.items()}
        state.update((name, torch.empty(0)) for name in self.exported.tokens)
        self.arguments = {
            processor: [self.placed(state[n], processor) if n in state else None for n in names]
            for processor, names in inputs.items()}
        self.positions = {}
        for processor, names in inputs.items():
            for position, name in enumerate(names):
                self.positions.setdefault(name, []).append((processor, position))
        self.key_of = {key: name for name, (key, _) in self.exported.state.items()}
        self.argument_queues = {processor: SimpleQueue() for processor in inputs
                                if processor != self.caller}
        self.finished = SimpleQueue()
        for processor in self.argument_queues:
            Thread(target=self.run, args=(processor,), daemon=True).start()

    def execute(self, processor, arguments):
        for channel in self.received.get(processor, ()):
            channel.start()
        return self.functions[processor](*arguments)

    def run(self, processor):
        ir.LOCAL.channels = self.channels
        while True:
            self.execute(processor, self.argument_queues[processor].get())
            if self.caller != self.output_processor:
                self.finished.put(processor)

    def placed(self, value, processor):
        return value if self.devices[processor] is None else value.to(self.devices[processor])

    def bind(self, name, value):
        for processor, position in self.positions.get(name, ()):
            self.arguments[processor][position] = self.placed(value, processor)

    def __call__(self, *args, **kwargs):
        flat, _ = pytree.tree_flatten((args, kwargs))
        for name, value in zip(self.exported.user_inputs, flat, strict=True):
            self.bind(name, value)
        for processor, argument_queue in self.argument_queues.items():
            arguments = self.arguments[processor]
            argument_queue.put(list(arguments) if processor in self.feedforward else arguments)
        if self.caller is None:
            return None
        ir.LOCAL.channels = self.channels
        results = self.execute(self.caller, self.arguments[self.caller])
        if self.caller != self.output_processor:
            for _ in self.argument_queues:
                self.finished.get()
            return None
        user = []
        for (kind, key), value in zip(self.exported.outputs, results):
            if kind == OutputKind.USER_OUTPUT:
                user.append(value)
            elif kind == OutputKind.BUFFER_MUTATION:
                self.bind(self.key_of[key], value)
        return pytree.tree_unflatten(user, self.exported.out_spec)
