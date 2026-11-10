# design/data-flow.md
from queue import SimpleQueue
from threading import Thread

import torch
from torch.export.graph_signature import OutputKind
from torch.utils import _pytree as pytree

from . import ir


class Program:
    def __init__(self, scheduled, functions=None):
        self.exported, self.devices = scheduled.exported, scheduled.devices
        self.output_processor = scheduled.output_processor
        self.functions = functions or scheduled.modules
        self.feedforward = set(scheduled.feedforward)
        self.channels = [SimpleQueue() for _ in range(scheduled.channels)]
        state = {name: tensor.detach() for name, (_, tensor) in self.exported.state.items()}
        state.update((name, torch.empty(0)) for name in self.exported.tokens)
        self.arguments = {
            processor: [state[n].to(self.devices[processor]) if n in state else None for n in names]
            for processor, names in scheduled.inputs.items()}
        self.positions = {}
        for processor, names in scheduled.inputs.items():
            for position, name in enumerate(names):
                self.positions.setdefault(name, []).append((processor, position))
        self.key_of = {key: name for name, (key, _) in self.exported.state.items()}
        self.argument_queues = {processor: SimpleQueue() for processor in scheduled.inputs
                                if processor != self.output_processor}
        for processor in self.argument_queues:
            Thread(target=self.run, args=(processor,), daemon=True).start()

    def run(self, processor):
        ir.LOCAL.channels = self.channels
        while True:
            self.functions[processor](*self.argument_queues[processor].get())

    def bind(self, name, value):
        for processor, position in self.positions.get(name, ()):
            self.arguments[processor][position] = value.to(self.devices[processor])

    def __call__(self, *args, **kwargs):
        flat, _ = pytree.tree_flatten((args, kwargs))
        for name, value in zip(self.exported.user_inputs, flat, strict=True):
            self.bind(name, value)
        for processor, argument_queue in self.argument_queues.items():
            arguments = self.arguments[processor]
            argument_queue.put(list(arguments) if processor in self.feedforward else arguments)
        ir.LOCAL.channels = self.channels
        results = self.functions[self.output_processor](*self.arguments[self.output_processor])
        user = []
        for (kind, key), value in zip(self.exported.outputs, results):
            if kind == OutputKind.USER_OUTPUT:
                user.append(value)
            elif kind == OutputKind.BUFFER_MUTATION:
                self.bind(self.key_of[key], value)
        return pytree.tree_unflatten(user, self.exported.out_spec)
