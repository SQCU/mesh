"""design/whole-function.md"""
import copy
import math
import statistics


def project(values, total, grain, low, high):
    """design/whole-function.md#finite-differences"""
    grain, total = int(grain), int(total)
    if grain <= 0 or not values or len(values) != len(low) or len(values) != len(high):
        raise ValueError('allocation dimensions and grain must be valid')
    lower = [math.ceil(x / grain) for x in low]
    upper = [math.floor(x / grain) for x in high]
    units = total // grain
    if total % grain or not sum(lower) <= units <= sum(upper):
        raise ValueError('infeasible allocation lattice')
    values = [float(x) / grain for x in values]
    left = min(x - hi for x, hi in zip(values, upper)) - 1
    right = max(x - lo for x, lo in zip(values, lower)) + 1
    for _ in range(64):
        middle = (left + right) / 2
        if sum(max(lo, min(hi, x - middle)) for x, lo, hi in zip(values, lower, upper)) > units:
            left = middle
        else:
            right = middle
    real = [max(lo, min(hi, x - right)) for x, lo, hi in zip(values, lower, upper)]
    result = [math.floor(x) for x in real]
    order = sorted(range(len(result)), key=lambda i: (-(real[i] - result[i]), i))
    for i in order:
        if sum(result) == units:
            break
        if result[i] < upper[i]:
            result[i] += 1
    if sum(result) != units:
        raise ArithmeticError('allocation projection lost units')
    return [x * grain for x in result]


class FunctionSearch:
    """design/whole-function.md#finite-differences"""

    def __init__(self, decisions, resolution=0.002, fraction=0.125):
        self.decisions = copy.deepcopy(decisions)
        self.centre = {}
        self.radius = {}
        for name, spec in self.decisions.items():
            p = list(map(int, spec['parts']))
            spec.update(total=sum(p), grain=int(spec.get('grain', 1)),
                        low=list(spec.get('low', [1] * len(p))), high=list(spec.get('high', [sum(p)] * len(p))))
            q = project(p, spec['total'], spec['grain'], spec['low'], spec['high'])
            if q != p:
                raise ValueError(f'{name}: initial allocation is outside its lattice')
            self.centre[name] = p
            self.radius[name] = max(spec['grain'], int(spec['total'] * fraction / spec['grain']) * spec['grain'])
        self.resolution = float(resolution)
        self.serial, self.round = 0, 0
        self.jobs, self.observations, self.history = [], [], []
        self.workload = None
        self.standing = False
        self._start()

    def state(self):
        return copy.deepcopy(self.__dict__)

    @classmethod
    def of(cls, state):
        obj = cls.__new__(cls)
        obj.__dict__.update(copy.deepcopy(state))
        return obj

    def ask(self):
        return {'id': self.serial, 'parts': copy.deepcopy(self.jobs[0]['parts']),
                'phase': self.jobs[0]['kind'], 'round': self.round}

    @property
    def parts(self):
        return self.ask()['parts']

    def _job(self, kind, parts=None, **extra):
        return {'kind': kind, 'parts': copy.deepcopy(self.centre if parts is None else parts), **extra}

    def _start(self):
        self.round += 1
        self.observations = []
        self.jobs = [self._job('baseline')]

    def _probes(self, observation):
        pressure = observation.get('queues', {})
        order = sorted(self.decisions, key=lambda k: -sum(
            float(pressure.get(str(e), {}).get('peak_bytes', 0)) for e in self.decisions[k].get('edges', [])))
        for name in order:
            spec, p = self.decisions[name], self.centre[name]
            pivot = len(p) - 1
            for i in range(pivot):
                for sign in (1, -1):
                    give, take = (pivot, i) if sign > 0 else (i, pivot)
                    amount = min(self.radius[name], p[give] - spec['low'][give], spec['high'][take] - p[take])
                    amount = int(amount // spec['grain']) * spec['grain']
                    if amount <= 0:
                        continue
                    parts = copy.deepcopy(self.centre)
                    parts[name][give] -= amount
                    parts[name][take] += amount
                    self.jobs.append(self._job('probe', parts, name=name, coordinate=i, offset=sign * amount))
        if self.jobs:
            self.jobs.append(self._job('baseline_end'))
        else:
            self.standing = True
            self._start()

    @staticmethod
    def _queue(observation):
        return sum(float(q.get('integral_byte_seconds', 0)) for q in observation.get('queues', {}).values()) / observation['seconds']

    def _candidate(self):
        first, last = self.observations[0], self.observations[-1]
        elapsed = last['serial'] - first['serial']
        slopes = {}
        for o in self.observations[1:-1]:
            job = o['job']
            drift = (last['seconds'] - first['seconds']) * (o['serial'] - first['serial']) / elapsed
            key = (job['name'], job['coordinate'])
            slopes.setdefault(key, []).append((job['offset'], o['seconds'] - drift, self._queue(o)))
        parts, derivatives = copy.deepcopy(self.centre), {}
        for name, spec in self.decisions.items():
            gradient, queue_gradient = [], []
            for i in range(len(parts[name]) - 1):
                samples = sorted(slopes.get((name, i), []))
                if len(samples) == 1:
                    samples.append((0, first['seconds'], self._queue(first)))
                    samples.sort()
                if len(samples) < 2:
                    gradient.append(0.0); queue_gradient.append(0.0)
                    continue
                a, b = samples[0], samples[-1]
                gradient.append((b[1] - a[1]) / (b[0] - a[0]))
                queue_gradient.append((b[2] - a[2]) / (b[0] - a[0]))
            gradient.append(0.0); queue_gradient.append(0.0)
            mean = statistics.mean(gradient)
            gradient = [g - mean for g in gradient]
            norm = max(map(abs, gradient), default=0)
            derivatives[name] = {'seconds_per_unit': list(gradient), 'queue_per_unit': queue_gradient}
            if norm * self.radius[name] <= self.resolution * last['seconds']:
                mean = statistics.mean(queue_gradient)
                gradient = [g - mean for g in queue_gradient]
                norm = max(map(abs, gradient), default=0)
            if norm:
                wanted = [p - self.radius[name] * g / norm for p, g in zip(parts[name], gradient)]
                parts[name] = project(wanted, spec['total'], spec['grain'], spec['low'], spec['high'])
        self.history.append({'round': self.round, 'centre': copy.deepcopy(self.centre), 'candidate': copy.deepcopy(parts),
                             'derivatives': derivatives, 'baseline_seconds': last['seconds']})
        if parts == self.centre:
            self._shrink()
            self._start()
        else:
            self.jobs = [self._job('control'), self._job('trial', parts), self._job('trial', parts), self._job('control')]

    def _shrink(self):
        self.standing = True
        for name, spec in self.decisions.items():
            self.radius[name] = max(spec['grain'], self.radius[name] // (2 * spec['grain']) * spec['grain'])
            self.standing &= self.radius[name] == spec['grain']

    def observe(self, observation):
        expected = self.ask()
        if observation.get('scope') != 'function' or not observation.get('complete'):
            raise ValueError('feedback requires a completed whole function')
        if observation.get('id') != expected['id'] or observation.get('parts') != expected['parts']:
            raise ValueError('feedback does not describe the pending compiled configuration')
        if self.workload is None:
            self.workload = observation.get('workload')
        if observation.get('workload') != self.workload:
            raise ValueError('finite differences require the same workload')
        seconds = float(observation['seconds'])
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('function latency must be finite and positive')
        job = self.jobs.pop(0)
        self.observations.append({**copy.deepcopy(observation), 'seconds': seconds, 'job': job, 'serial': self.serial})
        self.serial += 1
        if job['kind'] == 'baseline':
            self._probes(observation)
        elif job['kind'] == 'baseline_end':
            self._candidate()
        elif not self.jobs:
            trial = [o for o in self.observations if o['job']['kind'] == 'trial']
            control = [o for o in self.observations if o['job']['kind'] == 'control']
            t, c = (statistics.median(o['seconds'] for o in group) for group in (trial, control))
            tq, cq = (statistics.median(self._queue(o) for o in group) for group in (trial, control))
            accepted = t < c * (1 - self.resolution) or (t <= c * (1 + self.resolution) and tq < cq * (1 - self.resolution))
            self.history[-1].update(trial_seconds=t, control_seconds=c, accepted=accepted)
            if accepted:
                self.centre = copy.deepcopy(trial[0]['job']['parts'])
                self.standing = False
            else:
                self._shrink()
            self._start()
        return self.ask()
