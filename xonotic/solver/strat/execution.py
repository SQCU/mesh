import os
import time
from typing import NamedTuple

import numpy as np
import torch

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
    started = time.perf_counter()
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
    return values, owner.command, time.perf_counter() - started


def emit_policies(programs, prepared, frame, arms, key, recorded=None):
    values, command, preparation = prepared
    started = time.perf_counter()
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
    return PolicyEmission(outputs, comparisons, velocity, logp, residual,
        preparation, time.perf_counter() - started)


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
        self.step = 0
        self.moments = {name: (torch.zeros_like(value), torch.zeros_like(value)) for name, value in self.parameters.items()}

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
        self.step += 1

    def state(self):
        out = {'step': np.asarray(self.step), 'learning_rate': np.asarray(self.learning_rate, np.float32)}
        for name, (m, v) in self.moments.items():
            out[name + '.m'] = m.detach().cpu().numpy()
            out[name + '.v'] = v.detach().cpu().numpy()
        return out

    @torch.no_grad()
    def load(self, state):
        self.step = int(np.asarray(state.get('step', self.step)))
        if 'learning_rate' in state:
            self.learning_rate = float(np.asarray(state['learning_rate']))
        for name, (m, v) in self.moments.items():
            m.copy_(torch.as_tensor(np.asarray(state[name + '.m'])))
            v.copy_(torch.as_tensor(np.asarray(state[name + '.v'])))


class PolicyProgram:
    """1e1d8fb5^:execution.PolicyProgram with 578de271^:persistent_policy.PersistentPolicy's queries (infer, emit,
    integrate, contribute, update) as torch calls of one program: evaluation and training are queries of the same
    forward, autograd deriving the training query's backward.  On a world (products.World) rank 0 announces each query
    with its operands and every rank runs it (serve), so the partitioned products' collectives pair; a rank updates the
    parameters it owns (rank 0 the replicated ones and its shards, every other rank its shards) and rank 0's replicated
    values are then every rank's (World.broadcast_)."""

    def __init__(self, model, forward, learner=None, world=None, arm=None, device='cpu', lead=0.0):
        self.model, self.forward = model, forward
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
        self.accumulated = {}
        self.queries = {}

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

    def synchronize(self):
        self.announce('synchronize')
        if self.world is not None and self.model is not None:
            self.world.broadcast_([value.data for value in self.replicated()])

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
        with torch.no_grad():
            output = self.forward(self.model, *values)
        return None if self.member() else self.crop(output, frame)

    def emit(self, values, frame, key):
        self.announce('emit', list(values), frame, np.asarray(key))
        with torch.no_grad():
            output = self.forward(self.model, *values)
            if self.member():
                return None
            velocity, likelihood = sample(output.rate, key)
            residual = torch.where(output.rate.present, integrate(values.residual, velocity, values.delta,
                                                                  values.relaxation_time, self.lead), 0.0)
        n, width = frame.state.shape
        host = lambda value: value.detach().cpu().numpy()
        return self.crop(output, frame), host(velocity)[:n, :width].copy(), host(likelihood)[:n].copy(), host(residual)[:n, :width].copy()

    def integrate(self, values, velocity):
        with torch.no_grad():
            return integrate(values.residual, velocity, values.delta, values.relaxation_time, self.lead).cpu().numpy()

    def contribute(self, before, after, item, weights, reset):
        self.announce('contribute', list(before), list(after), item, weights, bool(reset))
        before, after = ChorusArrays(*before), ChorusArrays(*after)
        if reset:
            self.model.zero_grad(set_to_none=True)
            self.accumulated = {}
        loss, (metrics, log_ratio, ratio, active) = tensor_loss(self.model, self.forward, self.policy_arm, before, after, item)
        loss.backward()
        weights = torch.as_tensor(weights, dtype=metrics.dtype, device=metrics.device)
        self.accumulated['loss'] = self.accumulated.get('loss', 0.0) + loss.detach()
        self.accumulated['metrics'] = self.accumulated.get('metrics', 0.0) + metrics * weights
        return log_ratio.detach().cpu().numpy(), ratio.detach().cpu().numpy(), active.detach().cpu().numpy()

    def update(self):
        self.announce('update')
        owned = self.owned()
        gradients = {name: value.grad for name, value in owned.items()}
        reduce = self.world.reduce_norm if self.world is not None and self.world.size > 1 else None
        clipped, magnitude = clip_grad_norm(gradients, self.gradient_clip, reduce)
        self.optimizer.apply(clipped)
        if self.world is not None and self.world.size > 1:
            self.world.broadcast_([value.data for value in self.replicated()])
        total = self.accumulated.get('loss', torch.zeros(()))
        parts = self.accumulated.get('metrics', torch.zeros(()))
        return (np.asarray(total.detach().cpu().numpy() if torch.is_tensor(total) else total),
                np.asarray(parts.detach().cpu().numpy() if torch.is_tensor(parts) else parts), np.asarray(magnitude))

    def save_shard(self, path):
        """A rank's shard of the parameters and their moments beside rank 0's checkpoint (`path`.rank<r>.npz): the
        columns a world's parts give it, which no other rank holds."""
        self.announce('save_shard', path)
        sharded = getattr(self.model, 'sharded', frozenset())
        if not sharded or self.world is None or self.world.rank == 0:
            return
        payload = {name: value.detach().cpu().numpy() for name, value in self.model.named_parameters() if name in sharded}
        if self.optimizer is not None:
            payload.update({'__opt__' + name: value for name, value in self.optimizer.state().items()})
        payload['__parts__'] = np.asarray(self.world.products.get('scale', []), np.float64)
        target = f'{path}.rank{self.world.rank}.npz'
        os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
        np.savez(target + '.tmp.npz', **payload)
        os.replace(target + '.tmp.npz', target)

    def load_shard(self, path):
        """Every other rank loads its shard saved beside rank 0's checkpoint `path` (save_shard): rank 0 announces it
        once it has loaded `path` itself, so every rank's columns come from the same lineage."""
        self.announce('load_shard', path)
        sharded = getattr(self.model, 'sharded', frozenset())
        target = f'{path}.rank{self.world.rank}.npz' if self.world is not None else None
        if not sharded or target is None or self.world.rank == 0 or not os.path.isfile(target):
            return False
        with np.load(target, allow_pickle=False) as source:
            if list(source['__parts__']) != list(self.world.products.get('scale', [])):
                return False
            with torch.no_grad():
                for name, value in self.model.named_parameters():
                    if name in sharded:
                        value.copy_(torch.from_numpy(source[name]))
            if self.optimizer is not None:
                self.optimizer.load({name[7:]: source[name] for name in source.files if name.startswith('__opt__')})
        return True

    def crop(self, out, frame):
        n, width = frame.state.shape
        pages = 1 + frame.layout.shape[1]
        host = lambda value: value.detach().cpu().numpy()
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
                'device': self.device, 'queries': dict(self.queries),
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
        elif name == 'contribute':
            before, after, item, weights, reset = operands
            program.contribute(before, after, item, weights, reset)
        elif name in ('save_shard', 'load_shard'):
            getattr(program, name)(*operands)
        else:
            getattr(program, name)()
    return programs
