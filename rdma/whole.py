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


def _sigmoid(x):
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))


def simplex(values, total, low, high):
    """A smooth bounded simplex: one multiplier conserves the total."""
    if not sum(low) <= total <= sum(high):
        raise ValueError('infeasible simplex')
    if total == sum(low):
        return list(map(float, low))
    if total == sum(high):
        return list(map(float, high))
    left, right = min(values) - 80, max(values) + 80
    for _ in range(80):
        multiplier = (left + right) / 2
        amount = sum(lo + (hi - lo) * _sigmoid(v - multiplier) for v, lo, hi in zip(values, low, high))
        if amount > total:
            left = multiplier
        else:
            right = multiplier
    return [lo + (hi - lo) * _sigmoid(v - (left + right) / 2) for v, lo, hi in zip(values, low, high)]


class FunctionFlow:
    """design/whole-function.md#finite-differences"""

    def __init__(self, decisions, learning_rate=0.15, perturbation=0.2, congestion_weight=1.0):
        self.decisions = copy.deepcopy(decisions)
        self.vector, self.momentum, self.energy = {}, {}, {}
        self.learning_rate, self.perturbation = float(learning_rate), float(perturbation)
        self.congestion_weight = float(congestion_weight)
        self.serial, self.round, self.workload = 0, 0, None
        self.samples, self.history = [], []
        for name, spec in self.decisions.items():
            parts = list(map(float, spec['parts']))
            total = sum(parts)
            spec.update(total=total, grain=int(spec.get('grain', 0)),
                        low=list(map(float, spec.get('low', [0] * len(parts)))),
                        high=list(map(float, spec.get('high', [total] * len(parts)))))
            if not all(lo <= v <= hi for v, lo, hi in zip(parts, spec['low'], spec['high'])):
                raise ValueError('initial vector is outside its simplex')
            fraction = [(v - lo) / (hi - lo) if hi > lo else 0.5 for v, lo, hi in zip(parts, spec['low'], spec['high'])]
            self.vector[name] = [math.log(max(v, 1e-12) / max(1 - v, 1e-12)) for v in fraction]
            self.momentum[name] = [0.0] * len(parts)
            self.energy[name] = [0.0] * len(parts)
        self.dimension = sum(map(len, self.vector.values()))
        self.order = 1 << max(1, (self.dimension - 1).bit_length())

    def state(self):
        return copy.deepcopy(self.__dict__)

    @classmethod
    def of(cls, state):
        obj = cls.__new__(cls)
        obj.__dict__.update(copy.deepcopy(state))
        return obj

    def _direction(self):
        row, at, result = 1 + self.round % (self.order - 1), 0, {}
        for name, values in self.vector.items():
            result[name] = [1.0 - 2.0 * ((row & (at + i)).bit_count() % 2) for i in range(len(values))]
            at += len(values)
        return result

    def _values(self, sign):
        c, direction = self.perturbation / (self.round + 1) ** 0.101, self._direction()
        return {name: simplex([v + sign * c * d for v, d in zip(values, direction[name])],
                              self.decisions[name]['total'], self.decisions[name]['low'], self.decisions[name]['high'])
                for name, values in self.vector.items()}

    @property
    def centre(self):
        return self._values(0)

    def ask(self):
        # A fixed +--+ finite-difference stencil cancels linear timing drift.
        # All four evaluations are ordinary complete functions. No winner is chosen.
        sign = (1, -1, -1, 1)[len(self.samples)]
        continuous = self._values(sign)
        parts = {}
        for name, values in continuous.items():
            spec = self.decisions[name]
            parts[name] = project(values, int(spec['total']), spec['grain'], spec['low'], spec['high']) if spec['grain'] else values
        return {'id': self.serial, 'parts': parts, 'continuous': continuous, 'phase': 'finite_difference',
                'offset': sign * self.perturbation / (self.round + 1) ** 0.101, 'round': self.round}

    @property
    def parts(self):
        return self.ask()['parts']

    def observe(self, observation):
        expected = self.ask()
        if observation.get('scope') != 'function' or not observation.get('complete'):
            raise ValueError('feedback requires a completed whole function')
        if observation.get('id') != expected['id'] or observation.get('parts') != expected['parts']:
            raise ValueError('feedback does not describe the compiled configuration')
        workload = observation.get('workload')
        if self.workload is not None and workload != self.workload:
            raise ValueError('finite differences require the same workload')
        seconds, congestion = float(observation['seconds']), float(observation.get('congestion_seconds', 0))
        if not math.isfinite(seconds) or seconds <= 0 or not math.isfinite(congestion) or congestion < 0:
            raise ValueError('function latency and congestion must be finite')
        self.workload = workload
        cost = math.log(seconds) + self.congestion_weight * congestion / seconds
        self.samples.append({'id': self.serial, 'seconds': seconds, 'cost': cost, 'offset': expected['offset']})
        self.serial += 1
        if len(self.samples) == 4:
            c = self.samples[0]['offset']
            derivative = (self.samples[0]['cost'] - self.samples[1]['cost'] - self.samples[2]['cost'] + self.samples[3]['cost']) / (4 * c)
            direction, gradient = self._direction(), {}
            before = self.centre
            rate = self.learning_rate / (self.round + 1) ** 0.3
            for name, values in self.vector.items():
                raw = [derivative * d for d in direction[name]]
                mean = statistics.mean(raw)
                gradient[name] = [g - mean for g in raw]
                for i, g in enumerate(gradient[name]):
                    self.momentum[name][i] = 0.8 * self.momentum[name][i] + 0.2 * g
                    self.energy[name][i] = 0.95 * self.energy[name][i] + 0.05 * g * g
                    m = self.momentum[name][i] / (1 - 0.8 ** (self.round + 1))
                    v = self.energy[name][i] / (1 - 0.95 ** (self.round + 1))
                    values[i] -= rate * m / math.sqrt(v + 1e-12)
                centre = statistics.mean(values)
                self.vector[name] = [v - centre for v in values]
            self.history.append({'round': self.round, 'before': before, 'after': self.centre,
                                 'gradient': gradient, 'learning_rate': rate, 'samples': self.samples})
            self.samples = []
            self.round += 1
        return self.ask()
