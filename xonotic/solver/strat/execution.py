from typing import NamedTuple

import numpy as np
import torch
import torch_mesh
from torch_mesh.engines import host
from mesh_statistics import now, measure
from prepared import pages

from .inputs import ChorusArrays, capacity_for, frame_shapes, fill_frame
from .strategy import Strategy
from .state_steering import StateRate, writable_words, sample, integrate
from .storage import TensorStorage
from .policy_math import clip_grad_norm
from .policy_contract import POLICY_ACTOR_WEIGHT, POLICY_MOE_BALANCE_WEIGHT, POLICY_RATIO_CLIP


class PolicyEmission(NamedTuple):
    outputs: dict
    comparisons: dict
    velocity: np.ndarray
    logp: np.ndarray
    residual: np.ndarray
    preparation_seconds: float
    inference_seconds: float
    performance: dict


class InputArena:
    def __init__(self, device='cpu'):
        self.device = device
        self.capacity = (1,) * 10
        self.slots = []
        self.allocations = self.loads = 0

    def reserve(self, *frames, count, capacity=(1,) * 10):
        realized = tuple(max(a, b, c) for a, b, c in zip(self.capacity, capacity, capacity_for(*frames)))
        self.capacity = realized
        shapes = frame_shapes(frames[0], self.capacity)
        while len(self.slots) < count:
            self.slots.append((TensorStorage(device=self.device), np.empty((0, 0), dtype=bool)))
        for slot, (storage, scratch) in enumerate(self.slots):
            before = storage.allocations
            storage.realize({name: (shape, value.dtype)
                for name, shape, value in zip(frames[0]._fields, shapes, frames[0])})
            if scratch.shape != shapes.neighborhood_indices:
                scratch = np.empty(shapes.neighborhood_indices, dtype=bool)
                self.allocations += 1
            self.slots[slot] = storage, scratch
            self.allocations += storage.allocations - before

    def load(self, frame, slot=0):
        storage, scratch = self.slots[slot]
        fill_frame(ChorusArrays(**storage.host), frame, scratch)
        self.loads += 1
        return ChorusArrays(**storage.values)

    def stage(self, *frames, capacity=(1,) * 10):
        unique = dict((id(frame), frame) for frame in frames)
        self.reserve(*frames, count=len(unique), capacity=capacity)
        return {identity: self.load(frame, slot) for slot, (identity, frame) in enumerate(unique.items())}

    def report(self):
        return {'slots': len(self.slots), 'allocations': self.allocations, 'loads': self.loads,
            'bytes': sum(storage.nbytes + scratch.nbytes for storage, scratch in self.slots)}


def prepare_policies(programs, frame):
    started = now()
    owner = next(iter(programs.values()))
    capacity = tuple(map(max, zip(*(program.capacity for program in programs.values()))))
    values = owner.inputs.stage(frame, capacity=capacity)[id(frame)]
    for program in programs.values():
        program.bind_capacity(owner.inputs.capacity)
        program.reserve(frame)
    shape = values.state.shape
    if owner.command is None:
        owner.command = TensorStorage(device=owner.device)
    owner.command.realize({'velocity': (tuple(shape), np.float32), 'residual': (tuple(shape), np.float32),
        'logp': ((shape[0],), np.float32)})
    return values, owner.command, now() - started


def emit_policies(programs, prepared, frame, arms, key, recorded=None):
    values, command, preparation = prepared
    started = now()
    outputs, comparisons = {}, {}
    n, width = frame.state.shape
    velocity, residual = (command.host[name][:n, :width] for name in ('velocity', 'residual'))
    logp = command.host['logp'][:n]
    for value in command.host.values():
        value.fill(0)
    for arm, program in programs.items():
        output, sampled, likelihood, integrated = program.emit(values, frame, key)
        selected = arms == arm
        velocity[selected], logp[selected], residual[selected] = sampled[selected], likelihood[selected], integrated[selected]
        outputs[arm] = output
        comparisons[arm] = {'rate_mean': np.asarray(output.rate.mean), 'rate_log_scale': np.asarray(output.rate.log_scale),
            'velocity': sampled, 'residual': integrated, 'present': np.asarray(output.rate.present),
            'density': np.asarray(output.rate.density), 'winner_value': np.asarray(output.value_winnie),
            'loser_value': np.asarray(output.value_lou)}
    if recorded is not None:
        np.copyto(velocity, recorded['velocity'])
        np.copyto(logp, recorded['behavior_logp'])
        arm = next(iter(programs))
        integrated = programs[arm].integrate(values, command.values['velocity'])
        np.copyto(residual, np.asarray(integrated)[:n, :width])
    writable = writable_words(frame.layout, np).reshape(n, width)
    np.copyto(velocity, 0, where=~writable)
    np.copyto(residual, 0, where=~writable)
    performance = measure(started, now(), velocity.nbytes + logp.nbytes + residual.nbytes, n)
    return PolicyEmission(outputs, comparisons, velocity, logp, residual,
        preparation, performance['seconds'], performance)


LOSS_WEIGHTS = {"actor": POLICY_ACTOR_WEIGHT, "winnie": 0.5, "lou": 0.5, "moe_balance": POLICY_MOE_BALANCE_WEIGHT}


def tensor_loss(model, forward, policy_arm, current_inputs, next_inputs, item):
    """1e1d8fb5^:online.OnlineLearner._tensor_loss on torch: the decoupled PPO actor term with rollout importance
    weights, the winner and loser value regressions to the bootstrapped (or observed) return, the scale experts'
    balance term; its 16 metrics.  autograd derives its gradient."""
    from .policy_math import clipped_policy_surrogate, rollout_importance_weights
    from .strategy import logp_of
    value_scale, actor_scale = item['value_scale'], item['actor_scale']
    velocity = item["velocity"]
    behavior_logp = item["behavior_logp"]
    train_weight = item['train_mask'].to(torch.float32)
    train_count = torch.clamp(train_weight.sum(), min=1.0)
    actor_weight = train_weight * item['actor_mask'].to(torch.float32)
    actor_count = torch.clamp(actor_weight.sum(), min=1.0)
    reward = item["reward"]
    winner_mask = item["winner_mask"].to(torch.bool)
    next_winner_mask = item["next_winner_mask"].to(torch.bool)

    current = forward(model, *current_inputs)
    following = forward(model, *next_inputs)

    actor_active = (actor_weight > 0) & (actor_scale > 0)
    logpi = logp_of(current, velocity, actor_active)

    value = torch.where(winner_mask, current.value_winnie, current.value_lou)
    successor_rows = torch.clamp(item['successor_rows'].long(), min=0)
    next_values_w = following.value_winnie[successor_rows]
    next_values_l = following.value_lou[successor_rows]
    bootstrap = torch.where(next_winner_mask, next_values_w, next_values_l)
    bootstrap = torch.where(((winner_mask == next_winner_mask) | (policy_arm == "terminal_win")) & (item["discount"] > 0), bootstrap, 0.0)
    target = reward + item["discount"] * bootstrap.detach()
    target = torch.where(item['target_present'], item['value_target'], target)
    error = target - value

    proximal_logp = logpi.detach()
    rollout_log_ratio = proximal_logp - torch.where(actor_active, behavior_logp, 0.0)
    ratio = rollout_importance_weights(rollout_log_ratio)
    td_advantage = error.detach()
    actor = -(ratio * clipped_policy_surrogate(logpi - proximal_logp, td_advantage, POLICY_RATIO_CLIP) * actor_weight).sum() / actor_count

    present = torch.ones_like(current.rate.mean) if current.rate.present is None else current.rate.present.to(current.rate.mean.dtype)
    state_count = torch.clamp(present.sum(dim=-1), min=1)
    state_entropy = ((((current.rate.log_scale + 0.5 * np.log(2 * np.pi * np.e)) * present).sum(dim=-1) / state_count)
        * actor_weight).sum() / actor_count
    winner_weight = winner_mask.to(torch.float32) * train_weight
    loser_weight = (~winner_mask).to(torch.float32) * train_weight
    winner_loss = ((current.value_winnie - target).square() * winner_weight).sum() / torch.clamp(winner_weight.sum(), min=1.0)
    loser_loss = ((current.value_lou - target).square() * loser_weight).sum() / torch.clamp(loser_weight.sum(), min=1.0)

    total = (
        actor_scale * LOSS_WEIGHTS["actor"] * actor
        + value_scale * (LOSS_WEIGHTS["winnie"] * winner_loss
        + LOSS_WEIGHTS["lou"] * loser_loss
        + LOSS_WEIGHTS["moe_balance"] * current.scale_balance)
    )

    with torch.no_grad():
        winner = winner_mask.to(torch.float32)
        loser = (~winner_mask).to(torch.float32)
        advantage = (error * train_weight).sum() / train_count
        weighted_advantage = (ratio * error * train_weight).sum() / train_count
        winner_count = (winner * train_weight).sum()
        loser_count = (loser * train_weight).sum()
        winner_advantage = (ratio * error * winner * train_weight).sum() / torch.clamp(winner_count, min=1.0)
        loser_advantage = (ratio * error * loser * train_weight).sum() / torch.clamp(loser_count, min=1.0)
        winner_reward = (reward * winner * train_weight).sum() / torch.clamp(winner_count, min=1.0)
        loser_reward = (reward * loser * train_weight).sum() / torch.clamp(loser_count, min=1.0)
        role_change = ((winner_mask != next_winner_mask).to(torch.float32) * train_weight).sum() / train_count
        target_mean = (target * train_weight).sum() / train_count
        target_variance = ((target - target_mean).square() * train_weight).sum() / train_count
        metrics = torch.stack([actor, winner_loss, loser_loss, (ratio * train_weight).sum() / train_count,
            advantage, weighted_advantage, winner_advantage, loser_advantage, winner_reward,
            loser_reward, winner_count, loser_count, role_change,
            state_entropy, target_variance, current.scale_balance.reshape(())]).detach()
    return total, (metrics, rollout_log_ratio, ratio, actor_active)


class AdamW:
    """578de271^:persistent_policy.PersistentPolicy's update, as mlx.optimizers.AdamW computes it: betas (0.9,
    0.999), eps 1e-8, no bias correction, the decay applied to the parameter, p (1 - lr wd) - lr m / (sqrt(v) + eps).
    Its state is named as mlx's tree_flatten names it ("step", "learning_rate", "<parameter>.m", "<parameter>.v")."""

    def __init__(self, parameters, learning_rate, weight_decay, betas=(0.9, 0.999), eps=1e-8):
        self.parameters = dict(parameters)
        self.learning_rate, self.weight_decay, self.betas, self.eps = float(learning_rate), float(weight_decay), betas, eps
        self.step = pages(np.asarray(0, np.int64), device='cpu')
        self.moments = {name: (pages(np.zeros(tuple(value.shape), host(value).numpy().dtype), device=value.device.type),
                               pages(np.zeros(tuple(value.shape), host(value).numpy().dtype), device=value.device.type))
                        for name, value in self.parameters.items()}

    @torch.no_grad()
    def apply(self, gradients):
        first, second = self.betas
        rate = self.learning_rate
        for name, parameter in self.parameters.items():
            gradient = gradients.get(name)
            if gradient is None:
                gradient = torch.zeros_like(parameter)
            m, v = self.moments[name]
            m.mul_(first).add_(gradient, alpha=1 - first)
            v.mul_(second).add_(gradient.square(), alpha=1 - second)
            parameter.mul_(1 - rate * self.weight_decay).sub_(rate * m / (v.sqrt() + self.eps))
        self.step.add_(1)

    def state(self):
        out = {'step': host(self.step).numpy().copy(), 'learning_rate': np.asarray(self.learning_rate, np.float32)}
        for name, (m, v) in self.moments.items():
            out[name + '.m'] = host(m).numpy().copy()
            out[name + '.v'] = host(v).numpy().copy()
        return out

    @torch.no_grad()
    def load(self, state):
        host(self.step).numpy()[...] = state.get('step', host(self.step).numpy())
        if 'learning_rate' in state:
            self.learning_rate = float(np.asarray(state['learning_rate']))
        for name, (m, v) in self.moments.items():
            np.copyto(host(m).numpy(), state[name + '.m'])
            np.copyto(host(v).numpy(), state[name + '.v'])


class PolicyProgram:
    """Policy queries are whole Function declarations over persistent page operands.

    Capture includes the complete forward, gradient accumulation or optimizer
    update and its collectives. The application consumes the completed outputs;
    no tensor backend runs an eager copy of the function.
    """

    def __init__(self, model, forward, learner=None, world=None, arm=None, device='cpu', lead=0.0):
        self.model, self.forward = model, forward
        if model is not None:
            shared = {}
            for name, value in list(model.named_parameters(remove_duplicate=False)):
                if id(value) not in shared:
                    tensor = pages(value.detach().cpu().numpy(), device=device)
                    shared[id(value)] = torch.nn.Parameter(tensor, requires_grad=value.requires_grad)
                owner, _, field = name.rpartition('.')
                setattr(model.get_submodule(owner) if owner else model, field, shared[id(value)])
        self.capacity = (1,) * 10
        self.device = device
        self.inputs = InputArena(device)
        self.command = None
        self.capacity_changes = 0
        self.external = any(getattr(model, name, None) is not None for name in ('scale_executor', 'cross_executor'))
        self.learner = learner
        self.world, self.arm, self.lead = world, arm, float(lead)
        self.optimizer = None
        self.gradient_clip = None
        self.policy_arm = arm
        self.queries = {}
        self.programs, self.operand_pages, self.measurements = {}, {}, {}

    def announce(self, name, *operands):
        self.queries[name] = self.queries.get(name, 0) + 1
        if self.world is not None and self.world.size > 1 and self.world.rank == 0:
            self.world.announce(name, self.arm, *operands)

    def owned(self):
        if self.model is None:
            return {}
        sharded = getattr(self.model, 'sharded', frozenset())
        mine = self.world is None or self.world.rank == 0
        return {name: value for name, value in self.model.named_parameters() if mine or name in sharded}

    def replicated(self):
        sharded = getattr(self.model, 'sharded', frozenset())
        return [value for name, value in self.model.named_parameters() if name not in sharded]

    def train(self, learning_rate, weight_decay, gradient_clip, policy_arm):
        self.optimizer = AdamW(self.owned(), learning_rate, weight_decay)
        self.gradient_clip, self.policy_arm = abs(float(gradient_clip)), policy_arm

    def _query(self, name, source, operands=(), *, distribute=True):
        """Prepare an ordinary Function for these page bindings and consume its result."""
        from torch.utils._pytree import tree_flatten, tree_unflatten
        from torch_mesh import _stream
        from torch_mesh.declarations import Function
        import torch.distributed._functional_collectives as collective
        leaves, structure = tree_flatten(operands)
        bound, signature = [], []
        for index, value in enumerate(leaves):
            if isinstance(value, torch.Tensor):
                tensor = value
            else:
                array = np.asarray(value)
                key = (name, index, array.shape, array.dtype.str)
                if key not in self.operand_pages:
                    self.operand_pages = {k: v for k, v in self.operand_pages.items() if k[:2] != key[:2]}
                    self.operand_pages[key] = pages(array, device=self.device)
                tensor = self.operand_pages[key]
                np.copyto(host(tensor).numpy(), array)
            bound.append(tensor)
            signature.append((tuple(tensor.shape), tensor.dtype, tensor.device, tensor.data_ptr()))
        signature = (tuple(signature), structure)
        plan = self.programs.get(name)
        if plan is None or plan['signature'] != signature:
            group = self.world.group if self.world is not None else None
            def declared(*values):
                if distribute and group is not None:
                    values = tuple(collective.wait_tensor(collective.broadcast(value, 0, group.group_name)) for value in values)
                return source(*tree_unflatten(values, structure))
            parameters = tuple(self.model.parameters()) if self.model is not None else ()
            function = Function(declared, *bound, parameters=parameters,
                                source_scope={'function': 'policy-' + name, 'arm': self.arm})
            function.capture()
            output = next(node.args[0] for node in function.captured_graph.graph.nodes if node.op == 'output')
            destinations, outputs = [], []
            for node in output:
                value = node.meta['val'] if isinstance(node, torch.fx.Node) else node
                if isinstance(value, torch.Tensor):
                    destination = torch_mesh.empty(tuple(value.shape), dtype=value.dtype, device=value.device.type)
                    destinations.append(destination)
                    outputs.append(host(destination).numpy())
                else:
                    destinations.append(None); outputs.append(value)
            calls = function.declare(group, destinations=destinations)
            useful = sum(value.nbytes for value in outputs[:function.output_spec.num_leaves] if isinstance(value, np.ndarray))
            plan = dict(signature=signature, function=function, outputs=outputs, useful=useful, calls=calls)
            self.programs[name] = plan
        function = plan['function']
        if self.world is not None and self.world.group is not None:
            self.world.group.plan(plan['calls'])
        function.prepare()
        started = now()
        function.run()
        _stream.completion_end(function.completion())
        self.measurements[name] = measure(started, now(), plan['useful'], 1)
        return tree_unflatten(plan['outputs'][:function.output_spec.num_leaves], function.output_spec)

    def initialize_replicas(self):
        self.announce('initialize_replicas')
        if self.world is not None and self.world.size > 1 and self.model is not None:
            def source():
                with torch.no_grad():
                    self.world.broadcast_(self.replicated())
                return ()
            self._query('initialize_replicas', source, distribute=False)

    def bind_capacity(self, capacity):
        self.capacity_changes += self.capacity != tuple(capacity)
        self.capacity = tuple(capacity)

    def reserve(self, frame):
        if self.learner is not None:
            self.learner.realize_items(frame)

    def member(self):
        return self.world is not None and self.world.rank != 0

    def infer(self, values, frame):
        self.announce('infer', list(values), frame)
        def source(inputs):
            with torch.no_grad():
                return self.forward(self.model, *inputs)
        output = self._query('infer', source, (values,))
        return None if self.member() else self.crop(output, frame)

    def emit(self, values, frame, key):
        key = np.asarray(key, dtype=np.int64)
        self.announce('emit', list(values), frame, key)
        def source(inputs, sampling_key):
            with torch.no_grad():
                output = self.forward(self.model, *inputs)
                velocity, likelihood = sample(output.rate, sampling_key)
                residual = torch.where(output.rate.present, integrate(inputs.residual, velocity, inputs.delta,
                    inputs.relaxation_time, self.lead), 0.0)
                return output, velocity, likelihood, residual
        output, velocity, likelihood, residual = self._query('emit', source, (values, key))
        if self.member():
            return None
        n, width = frame.state.shape
        return self.crop(output, frame), velocity[:n, :width].copy(), likelihood[:n].copy(), residual[:n, :width].copy()

    def integrate(self, values, velocity):
        self.announce('integrate', list(values), velocity)
        def source(inputs, rate):
            return integrate(inputs.residual, rate, inputs.delta, inputs.relaxation_time, self.lead)
        return self._query('integrate', source, (values, velocity))

    def learn(self, items):
        """The entire supplied transition batch and optimizer update are one graph."""
        self.announce('learn', items)
        items = tuple((ChorusArrays(*before), ChorusArrays(*after), item, weights)
                      for before, after, item, weights in items)
        parameters = {name: value for name, value in self.model.named_parameters() if value.requires_grad}
        owned = self.owned()
        def source(batch):
            losses, metrics, observations, derivatives = [], [], [], []
            for before, after, item, weights in batch:
                loss, (metric, log_ratio, ratio, active) = tensor_loss(
                    self.model, self.forward, self.policy_arm, before, after, item)
                derivatives.append(torch.autograd.grad(loss, tuple(parameters.values()), allow_unused=True))
                losses.append(loss.detach()); metrics.append(metric * weights.to(metric.dtype))
                observations.append((log_ratio.detach(), ratio.detach(), active.detach()))
            gradients = {}
            for index, (name, parameter) in enumerate(parameters.items()):
                if name in owned:
                    values = tuple(torch.zeros_like(parameter) if derivative[index] is None else derivative[index]
                                   for derivative in derivatives)
                    gradients[name] = torch.stack(values).sum(0)
            reduce = self.world.reduce_norm if self.world is not None and self.world.size > 1 else None
            clipped, magnitude = clip_grad_norm(gradients, self.gradient_clip, reduce)
            self.optimizer.apply(clipped)
            if self.world is not None:
                with torch.no_grad():
                    self.world.broadcast_(self.replicated())
            return observations, torch.stack(losses).sum(0), torch.stack(metrics).sum(0), magnitude
        return self._query('learn', source, (items,))

    def export_shard(self):
        sharded = getattr(self.model, 'sharded', frozenset())
        payload = {name: host(value).numpy().copy()
                   for name, value in self.model.named_parameters() if name in sharded} if self.model is not None else {}
        if self.optimizer is not None:
            payload.update({'__opt__' + name: value for name, value in self.optimizer.state().items()})
        return payload

    def restore_shard(self, payload):
        with torch.no_grad():
            for name, value in self.model.named_parameters() if self.model is not None else ():
                if name in payload:
                    np.copyto(host(value).numpy(), payload[name])
        if self.optimizer is not None:
            self.optimizer.load({name[7:]: value for name, value in payload.items() if name.startswith('__opt__')})
        self.programs.clear()

    def crop(self, out, frame):
        n, width = frame.state.shape
        pages = 1 + frame.layout.shape[1]
        host = np.asarray
        ir = host(out.ir)
        latent_width = ir.shape[-1] // (1 + self.capacity[1])
        ir = ir[:n].reshape(n, 1 + self.capacity[1], latent_width)[:, :pages].reshape(n, pages * latent_width)
        return Strategy(StateRate(host(out.rate.mean)[:n, :width].copy(), host(out.rate.log_scale)[:n, :width].copy(),
            host(out.rate.density)[:n].copy(), host(out.rate.present)[:n, :width].copy()), ir.copy(),
            host(out.value_winnie)[:n].copy(), host(out.value_lou)[:n].copy(), host(out.coupling)[:n, :pages].copy(),
            host(out.local_neighborhood)[:n].copy(),
            host(out.scale_residual_stats).copy(), host(out.scale_expert_load).copy(), host(out.scale_balance).copy())

    def report(self):
        return {'capacity': dict(zip(('participants', 'state_pages', 'events', 'observations', 'carts', 'teams', 'navigation_nodes', 'navigation_edges', 'navigation_cells', 'neighbors'), self.capacity)),
                'capacity_changes': self.capacity_changes, 'input_storage': self.inputs.report(),
                'device': self.device, 'queries': dict(self.queries), 'functions': dict(self.measurements),
                'world': self.world.report() if self.world is not None else None}


def serve(world, initialize):
    """Every rank but 0: run the queries rank 0 announces until it stops (products.World.receive).  `initialize`
    (width, schema) makes this rank's programs as rank 0's responder makes its own."""
    programs = {}
    while (call := world.receive()) is not None:
        name, operands = call
        if name == 'initialize':
            programs = initialize(*operands)
            continue
        arm, operands = operands[0], operands[1:]
        program = programs[arm]
        if name in ('infer', 'emit'):
            values, frame = ChorusArrays(*operands[0]), ChorusArrays(*operands[1])
            getattr(program, name)(values, frame, *operands[2:])
        elif name == 'integrate':
            program.integrate(ChorusArrays(*operands[0]), operands[1])
        elif name == 'learn':
            program.learn(operands[0])
        else:
            getattr(program, name)()
    return programs
