import types

import mlx.core as mlx
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from . import tensor as mx
from .inputs import ChorusArrays, frame_shapes
from .policy_math import clip_grad_norm
from .state_steering import sample, integrate
from .tensor_runtime import Executable


class PersistentPolicy:
    def __init__(self, model, forward, learner=None):
        self.model, self.forward, self.learner = model, forward, learner
        self.executable = None
        self.optimizer_inputs = {}
        self.initial_optimizer = {}

    def build(self, template):
        if self.model is not None:
            mlx.eval(self.model.parameters(), {} if self.learner is None else self.learner.optimizer.state)
        graph = mx.Graph()
        dimensions = tuple(mx.Dimension('axis', (axis,)) for axis in range(10))
        shapes = frame_shapes(template, dimensions)
        with graph:
            current = ChorusArrays(*(graph.input('current.' + name, shape, value.dtype)
                for name, shape, value in zip(template._fields, shapes, template)))
            model = mx.Model(graph, self.model) if self.model is not None else None
            output = self.forward(model, *current)
            key = graph.input('sampling_key', (2,), 'uint32')
            velocity, likelihood = sample(output.rate, key)
            residual = mx.where(output.rate.present, integrate(current.residual, velocity,
                               current.delta, current.relaxation_time), 0)
            imposed = graph.input('imposed_velocity', current.state.shape)
            integrated = integrate(current.residual, imposed, current.delta, current.relaxation_time)
            phases = {'infer': output, 'emit': (output, velocity, likelihood, residual), 'integrate': integrated}
            if self.learner is not None:
                successor = ChorusArrays(*(graph.input('successor.' + name, shape, value.dtype)
                    for name, shape, value in zip(template._fields, shapes, template)))
                learner = self.learner
                item = {name: graph.input('transition.' + name, self.item_shape(name, value, dimensions), value.dtype)
                        for name, value in learner.item_storage.values.items() if name != 'weights'}
                proxy = types.SimpleNamespace(wally=model, policy_forward=self.forward,
                    policy_arm=learner.policy_arm, loss_traces=0)
                loss, reports = type(learner)._tensor_loss(proxy, current, successor, item)
                names, parameters = zip(*((name, value) for name, (value, _) in graph.parameters.items()))
                parameter_ids = {value.index for value in parameters}
                parameter_owners = {value.index: owner for _, _, inputs, _, owner in graph.nodes if owner
                                    for value in inputs if value.index in parameter_ids}
                gradients = graph.vjp((loss,), (mx.ones(()),), parameters)
                reset = graph.input('accumulator_reset', (), 'bool')
                accumulated = {}
                updates = []
                for name, gradient in zip(names, gradients):
                    target = graph.input('accumulator.gradient.' + name, gradient.shape)
                    accumulated[name] = target
                    updates.append(graph.assign(target, mx.where(reset, gradient, target + gradient)))
                metric_weights = graph.input('metric_weights', reports[0].shape)
                for name, value in (('loss', loss), ('metrics', reports[0] * metric_weights)):
                    target = graph.input('accumulator.' + name, value.shape)
                    updates.append(graph.assign(target, mx.where(reset, value, target + value)))
                phases['contribute'] = (reports[1:], tuple(updates))
                for name, value in tree_flatten(learner.optimizer.state):
                    self.initial_optimizer[name] = np.asarray(value).copy()
                    self.optimizer_inputs[name] = graph.input('optimizer.' + name, value.shape, value.dtype)
                clipped, magnitude = clip_grad_norm(tree_unflatten(list(accumulated.items())), learner.gradient_clip)
                clipped = dict(tree_flatten(clipped))
                optimizer_updates = []
                optimizer = learner.optimizer
                step = self.optimizer_inputs['step'] + 1
                rate = self.optimizer_inputs['learning_rate']
                for name, parameter in zip(names, parameters):
                    graph.owner = parameter_owners.get(parameter.index, 0)
                    gradient = clipped[name]
                    m, v = self.optimizer_inputs[name + '.m'], self.optimizer_inputs[name + '.v']
                    first, second = optimizer.betas
                    next_m = first * m + (1 - first) * gradient
                    next_v = second * v + (1 - second) * mx.square(gradient)
                    if optimizer.bias_correction:
                        numerator = rate / (1 - first ** step) * next_m
                        denominator = mx.sqrt(next_v) * mx.rsqrt(1 - second ** step) + optimizer.eps
                    else:
                        numerator, denominator = rate * next_m, mx.sqrt(next_v) + optimizer.eps
                    updated = parameter * (1 - rate * optimizer.weight_decay) - numerator / denominator
                    optimizer_updates.extend((graph.assign(m, next_m), graph.assign(v, next_v), graph.assign(parameter, updated)))
                graph.owner = 0
                optimizer_updates.append(graph.assign(self.optimizer_inputs['step'], step))
                phases['update'] = (magnitude, tuple(optimizer_updates))
        self.graph = graph
        self.executable = Executable(graph, phases)

    def item_shape(self, name, value, dimensions):
        if name == 'velocity': return dimensions[0], dimensions[1] * self.model.w.d_x if hasattr(self.model, 'w') else dimensions[1] * self.model.state_width
        return (dimensions[0],) if value.ndim else ()

    def reserve(self, template, capacity):
        if self.executable is None:
            self.build(template)
        previous = self.executable.capacity
        self.executable.reserve(capacity)
        if previous != self.executable.capacity:
            if previous is None:
                for name, initial in self.initial_optimizer.items():
                    np.copyto(self.executable.arrays[self.optimizer_inputs[name].index], initial)
                self.initial_optimizer.clear()
            parameters = [(name, mlx.from_dlpack(self.executable.arrays[value.index], copy=False))
                          for name, (value, _) in self.graph.parameters.items()]
            if self.model is not None:
                self.model.update(tree_unflatten(parameters))
            if self.learner is not None:
                state = tree_unflatten([(name, mlx.from_dlpack(self.executable.arrays[value.index], copy=False))
                                        for name, value in self.optimizer_inputs.items()])
                self.learner.optimizer.state.clear()
                self.learner.optimizer.state.update(state)

    def frame(self, prefix, values):
        return {prefix + name: value for name, value in zip(ChorusArrays._fields, values)}

    def infer(self, values):
        return self.executable.run('infer', self.frame('current.', values))

    def emit(self, values, key):
        return self.executable.run('emit', self.frame('current.', values) | {'sampling_key': key})

    def integrate(self, values, velocity):
        return self.executable.run('integrate', self.frame('current.', values) | {'imposed_velocity': velocity})

    def contribute(self, before, after, item, weights, reset):
        arguments = self.frame('current.', before) | self.frame('successor.', after)
        arguments.update({'transition.' + name: value for name, value in item.items()})
        arguments.update(metric_weights=weights, accumulator_reset=np.asarray(reset))
        reports, _ = self.executable.run('contribute', arguments)
        return reports

    def update(self):
        magnitude, _ = self.executable.run('update')
        total, parts = (self.executable.arrays[self.graph.inputs['accumulator.' + name].index] for name in ('loss', 'metrics'))
        return total, parts, magnitude

    def report(self):
        return {} if self.executable is None else self.executable.report()
