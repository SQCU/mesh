"""design/whole-function.md"""
import copy
import math
import statistics


def performance(observation):
    """Performance-only objective; design/objective-20261024.md.

    ``useful_work_bytes`` counts the fixed numerical function's logical output,
    never transport hops, padding, or intermediate collective deliveries. If it
    is absent, one completed function is the useful-work unit.
    """
    seconds = float(observation['seconds'])
    useful = float(observation.get('useful_work_bytes', 1))
    if not math.isfinite(seconds) or seconds <= 0 or not math.isfinite(useful) or useful <= 0:
        raise ValueError('function latency and fixed useful work must be finite and positive')
    return {'cost': math.log(seconds) - math.log(useful),
            'useful_work': useful,
            'useful_work_unit': 'logical_output_bytes' if 'useful_work_bytes' in observation else 'functions',
            'whole_function_seconds': seconds,
            'whole_function_throughput_per_second': useful / seconds}


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

    def __init__(self, decisions, learning_rate=0.15, perturbation=0.2, fit_compute=False):
        self.decisions = copy.deepcopy(decisions)
        self.vector, self.momentum, self.energy = {}, {}, {}
        self.learning_rate, self.perturbation = float(learning_rate), float(perturbation)
        self.objective_version, self.metric_contract = 'whole_function_throughput_v3', None
        self.fit_compute = bool(fit_compute)
        self.compute_samples, self.compute_model = [], None
        self.schedule_model = None
        self.serial, self.round, self.workload = 0, 0, None
        self.samples, self.history = [], []
        self.calibration_parts = {name: list(spec['parts']) for name, spec in self.decisions.items()}
        for name, spec in self.decisions.items():
            parts = list(map(float, spec['parts']))
            total = sum(parts)
            spec.update(total=total, grain=int(spec.get('grain', 0)),
                        low=list(map(float, spec.get('low', [0] * len(parts)))),
                        high=list(map(float, spec.get('high', [total] * len(parts)))))
            if not all(lo <= v <= hi for v, lo, hi in zip(parts, spec['low'], spec['high'])):
                raise ValueError('initial vector is outside its simplex')
            fraction = [(v - lo) / (hi - lo) if hi > lo else 0.5 for v, lo, hi in zip(parts, spec['low'], spec['high'])]
            quantum = spec.get('quantum', spec['grain'])
            quantum = [quantum] * len(parts) if isinstance(quantum, (float, int)) else quantum
            if len(quantum) != len(parts) or any(not math.isfinite(q) or q < 0 for q in quantum):
                raise ValueError('coordinate quantum must be finite, nonnegative and match its parts')
            # A quantized boundary must remain observable from both perturbation
            # signs. This seeds the latent coordinate at half a declared quantum;
            # it reserves no work and imposes no minimum physical allocation.
            fraction = [min(.25, q / (2 * (hi - lo))) if v == 0 and q and hi > lo else
                        1 - min(.25, q / (2 * (hi - lo))) if v == 1 and q and hi > lo else v
                        for v, q, lo, hi in zip(fraction, quantum, spec['low'], spec['high'])]
            self.vector[name] = [math.log(max(v, 1e-12) / max(1 - v, 1e-12)) for v in fraction]
            self.momentum[name] = [0.0] * len(parts)
            self.energy[name] = [0.0] * len(parts)
        self.dimension = sum(map(len, self.vector.values()))
        self.order = 1 << max(1, (self.dimension - 1).bit_length())

    def state(self):
        return copy.deepcopy(self.__dict__)

    @classmethod
    def of(cls, state):
        if state.get('objective_version') != 'whole_function_throughput_v3':
            raise ValueError('optimizer state of another objective; realize a new controller from its configuration')
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

    def ask(self, explore=True):
        # A fixed +--+ finite-difference stencil cancels linear timing drift.
        # All four evaluations are ordinary complete functions. No winner is chosen.
        sign = (1, -1, -1, 1)[len(self.samples)] if explore else 0
        continuous = self._values(sign)
        parts = {}
        for name, values in continuous.items():
            spec = self.decisions[name]
            if name == 'compute' and self.fit_compute and self.compute_model is not None:
                from allocate import min_max
                slopes = self.compute_model['slopes']
                parts[name], _ = min_max(int(spec['total']), spec['grain'], list(map(int, spec['low'])), list(map(int, spec['high'])),
                                          [lambda u, i=i: slopes[i] * (u - values[i]) for i in range(len(values))])
            else:
                parts[name] = project(values, int(spec['total']), spec['grain'], spec['low'], spec['high']) if spec['grain'] else values
        release = None
        if self.schedule_model is not None:
            previous = self.schedule_model
            release = list(previous['release'])
            if self.compute_model is not None:
                release = [t + b * (new - old) / previous.get('stages', 1) for t, b, new, old in
                           zip(release, self.compute_model['slopes'], parts['compute'], previous['parts'])]
            release = [t - min(release) for t in release]
        if not explore and self.serial == 0 and hasattr(self, 'calibration_parts'):
            parts = copy.deepcopy(self.calibration_parts)
        return {'id': self.serial, 'parts': parts, 'continuous': continuous, 'phase': 'finite_difference' if explore else 'centre',
                'producer_seconds': release,
                'offset': sign * self.perturbation / (self.round + 1) ** 0.101, 'round': self.round}

    @property
    def parts(self):
        return self.ask()['parts']

    def observe(self, observation):
        expected = self.ask(explore=observation.get('phase') != 'centre')
        if observation.get('scope') != 'function' or not observation.get('complete'):
            raise ValueError('feedback requires a completed whole function')
        if observation.get('id') != expected['id'] or observation.get('parts') != expected['parts']:
            raise ValueError('feedback does not describe the compiled configuration')
        workload = observation.get('workload')
        if self.workload is not None and workload != self.workload:
            raise ValueError('finite differences require the same workload')
        metrics = performance(observation)
        contract = [metrics['useful_work_unit'], metrics['useful_work']]
        if self.metric_contract is not None and contract != self.metric_contract:
            raise ValueError('finite differences require invariant useful work')
        seconds, congestion = metrics['whole_function_seconds'], float(observation.get('congestion_seconds', 0))
        if not math.isfinite(seconds) or seconds <= 0 or not math.isfinite(congestion) or congestion < 0:
            raise ValueError('function latency and congestion must be finite')
        finish = float(observation.get('finish_variance', 0))
        if not math.isfinite(finish) or finish < 0:
            raise ValueError('finish variance must be finite and nonnegative')
        self.workload, self.metric_contract = workload, contract
        if self.fit_compute:
            self._compute_observation(observation)
        if expected['phase'] == 'centre':
            # A measured calibration/repeat is evidence, not a zero-offset FD sample.
            self.serial += 1
            return self.ask()
        self.samples.append({'id': self.serial, 'seconds': seconds, 'cost': metrics['cost'],
                             'performance': metrics, 'offset': expected['offset']})
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
            if self.fit_compute:
                self._compute_step()
            self.history.append({'round': self.round, 'before': before, 'after': self.centre,
                                 'gradient': gradient, 'learning_rate': rate, 'samples': self.samples})
            self.samples = []
            self.round += 1
        return self.ask()

    def _compute_observation(self, observation):
        n = len(self.decisions['compute']['parts'])
        times = observation['compute_seconds']
        tails = observation['delivery_tail_seconds']
        if len(times) != n or len(tails) != n or any(not math.isfinite(t) or t <= 0 for t in times + tails):
            raise ValueError('one positive producer and delivery time per rank required')
        sample = {'parts': observation['parts']['compute'], 'seconds': list(times), 'tails': list(tails)}
        if 'producer_release_seconds' in observation:
            sample.update(release=list(observation['producer_release_seconds']), stages=len(observation['finishes']['stages']))
        self.compute_samples.append(sample)
        self.compute_samples = self.compute_samples[-64:]

    def _compute_step(self):
        """Historical equal_finish/min_max, driven only by completed functions.

        Fit fixed and work-dependent producer cost separately. A weak prior
        identifies the first single-share sample; later share variation supplies
        the slope. Delivery tails are the observation's own, not GPU contention.
        """
        import numpy as np
        from scipy.optimize import nnls
        from allocate import equal_finish
        spec = self.decisions['compute']
        total, n = spec['total'], len(spec['parts'])
        weights = np.asarray([0.97 ** (len(self.compute_samples) - 1 - k) for k in range(len(self.compute_samples))])
        fixed, slopes, tails = [], [], []
        for i in range(n):
            x = np.asarray([s['parts'][i] / total for s in self.compute_samples])
            y = np.asarray([s['seconds'][i] for s in self.compute_samples])
            prior = [0.0, float(y[0] / x[0])] if self.compute_model is None else [self.compute_model['fixed'][i], self.compute_model['slopes'][i] * total]
            matrix = np.column_stack((np.ones(len(x)), x)) * np.sqrt(weights)[:, None]
            coef, _ = nnls(np.vstack((matrix, np.eye(2) * 0.03)), np.r_[y * np.sqrt(weights), np.asarray(prior) * 0.03])
            fixed.append(float(coef[0])); slopes.append(max(float(coef[1]) / total, 1e-12))
            tails.append(float(np.average([s['tails'][i] for s in self.compute_samples], weights=weights)))
        # A smaller producer can still publish late because its own inputs
        # arrived later. Subtract its measured producer duration from observed
        # publication time to isolate that input-release delay. Do not add a
        # whole communication window (which also contains other producers).
        recent = self.compute_samples[-4:]
        start_delay = [0.0] * n
        if 'release' in recent[-1]:
            start_delay = [statistics.mean(s['release'][i] * s['stages'] - s['seconds'][i] for s in recent)
                           for i in range(n)]
            origin = min(start_delay)
            start_delay = [t - origin for t in start_delay]
        intercepts = [a + d for a, d in zip(fixed, start_delay)]
        tiles, target, finish = equal_finish(int(total), spec['grain'], list(map(int, spec['low'])), list(map(int, spec['high'])),
                                             intercepts, slopes)
        # Move the continuous centre each stencil; never keep/reject a measured
        # candidate. Tile realization uses the historical time-aware min_max.
        # This refitted affine model is an ongoing allocation input, not an
        # initialization prior whose influence should disappear during descent.
        guide = 0.5
        values = [(1 - guide) * a + guide * b for a, b in zip(self.centre['compute'], target)]
        fractions = [(v - lo) / (hi - lo) for v, lo, hi in zip(values, spec['low'], spec['high'])]
        self.vector['compute'] = [math.log(max(v, 1e-12) / max(1 - v, 1e-12)) for v in fractions]
        self.compute_model = {'fixed': fixed, 'slopes': slopes, 'start_delay': start_delay, 'tails': tails, 'target': target,
                              'tiles': tiles, 'predicted_finish': finish}
        if 'release' in self.compute_samples[-1]:
            self.schedule_model = {'release': [statistics.mean(s['release'][i] for s in recent) for i in range(n)],
                                   'parts': [statistics.mean(s['parts'][i] for s in recent) for i in range(n)],
                                   'stages': recent[-1]['stages']}

