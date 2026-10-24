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

    def __init__(self, decisions, learning_rate=0.15, perturbation=0.2, congestion_weight=1.0,
                 finish_weight=1.0, fit_compute=False, routing_model=None):
        self.decisions = copy.deepcopy(decisions)
        self.vector, self.momentum, self.energy = {}, {}, {}
        self.learning_rate, self.perturbation = float(learning_rate), float(perturbation)
        self.congestion_weight = float(congestion_weight)
        self.finish_weight, self.fit_compute = float(finish_weight), bool(fit_compute)
        self.compute_samples, self.compute_model = [], None
        self.schedule_model = None
        self.routing_model, self.routing_samples, self.routing_fit = copy.deepcopy(routing_model), [], None
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
        for name, value in {'finish_weight': 1.0, 'fit_compute': False,
                            'compute_samples': [], 'compute_model': None,
                            'routing_model': None, 'routing_samples': [], 'routing_fit': None, 'schedule_model': None}.items():
            obj.__dict__.setdefault(name, value)
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
        return {'id': self.serial, 'parts': parts, 'continuous': continuous, 'phase': 'finite_difference' if explore else 'centre',
                'producer_seconds': release,
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
        finish = float(observation.get('finish_variance', 0))
        if not math.isfinite(finish) or finish < 0:
            raise ValueError('finish variance must be finite and nonnegative')
        if self.fit_compute:
            self._compute_observation(observation)
        if self.routing_model is not None:
            self.routing_samples.append(copy.deepcopy(observation['finishes']['stages']))
        cost = math.log(seconds) + self.congestion_weight * congestion / seconds + self.finish_weight * finish
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
            if self.fit_compute:
                self._compute_step()
            if self.routing_model is not None:
                self._routing_step()
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
        the slope. Transport tails come from cable windows, not GPU contention.
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
        tiles, target, finish = equal_finish(int(total), spec['grain'], list(map(int, spec['low'])), list(map(int, spec['high'])),
                                             fixed, slopes)
        # Move the continuous centre each stencil; never keep/reject a measured
        # candidate. Tile realization uses the historical time-aware min_max.
        # This refitted affine model is an ongoing allocation input, not an
        # initialization prior whose influence should disappear during descent.
        guide = 0.5
        values = [(1 - guide) * a + guide * b for a, b in zip(self.centre['compute'], target)]
        fractions = [(v - lo) / (hi - lo) for v, lo, hi in zip(values, spec['low'], spec['high'])]
        self.vector['compute'] = [math.log(max(v, 1e-12) / max(1 - v, 1e-12)) for v in fractions]
        self.compute_model = {'fixed': fixed, 'slopes': slopes, 'tails': tails, 'target': target,
                              'tiles': tiles, 'predicted_finish': finish}
        if 'release' in self.compute_samples[-1]:
            recent = self.compute_samples[-4:]
            self.schedule_model = {'release': [statistics.mean(s['release'][i] for s in recent) for i in range(n)],
                                   'parts': [statistics.mean(s['parts'][i] for s in recent) for i in range(n)],
                                   'stages': recent[-1]['stages']}

    def _routing_step(self):
        """Measured finish residual + physical load Jacobian, after a stencil.

        A convex local correction complements the joint finite difference of
        whole-function latency/queues. Every correction moves the vector; no
        measured candidate is selected or rejected. Residuals are refreshed from
        actual landing times, so release delays need not match the fluid model.
        """
        import numpy as np
        from scipy.optimize import linprog, minimize
        model = self.routing_model
        columns = [(name, p) for name, spec in model['paths'].items() for p in range(len(spec['arcs']))]
        names = list(model['paths'])
        matrix = np.zeros((len(model['capacities']), len(columns)))
        conserve = np.zeros((len(names), len(columns)))
        centre = self.centre
        x0 = np.asarray([centre[name][p] / self.decisions[name]['total'] for name, p in columns])
        for k, (name, p) in enumerate(columns):
            conserve[names.index(name), k] = 1
            for e in model['paths'][name]['arcs'][p]:
                matrix[e, k] = model['paths'][name]['bytes'] / model['capacities'][e]
        measured, loads = [], []
        for stages in self.routing_samples:
            for stage in stages:
                arcs = stage['arcs']
                origin = min(v['start_ns'][0] for v in arcs.values())
                measured.append([(sum(arcs[str(e)]['finish_ns']) / 2 - origin) * 1e-9 for e in range(len(matrix))])
                loads.append([arcs[str(e)]['bytes'] / model['capacities'][e] for e in range(len(matrix))])
        measured, loads = np.mean(measured, axis=0), np.mean(loads, axis=0)
        scale = max(measured)
        residual, jacobian = (measured - loads) / scale, matrix / scale

        bounds = [(1e-9, 1)] * len(x0)
        first = linprog([0.0] * len(x0) + [1.0], A_ub=np.column_stack((jacobian, -np.ones(len(matrix)))),
                        b_ub=-residual, A_eq=np.column_stack((conserve, np.zeros(len(names)))), b_eq=np.ones(len(names)),
                        bounds=bounds + [(0, None)], method='highs')
        if not first.success:
            raise ArithmeticError(f'completion duration solve: {first.message}')
        target = first.x[-1]

        def cost(x):
            delta = residual + jacobian @ x - target
            return float(delta @ delta + 1e-4 * np.sum((x - x0) ** 2))

        def gradient(x):
            return 2 * jacobian.T @ (residual + jacobian @ x - target) + 2e-4 * (x - x0)

        solved = minimize(cost, first.x[:-1], jac=gradient, method='SLSQP', bounds=bounds,
                          constraints=[{'type': 'eq', 'fun': lambda x: conserve @ x - 1, 'jac': lambda x: conserve},
                                       {'type': 'ineq', 'fun': lambda x: target + 1e-8 - residual - jacobian @ x,
                                        'jac': lambda x: -jacobian}],
                          options={'ftol': 1e-11, 'maxiter': 300})
        if not solved.success:
            raise ArithmeticError(f'completion residual solve: {solved.message}')
        x = (x0 + solved.x) / 2
        for name in names:
            values = [x[k] * self.decisions[name]['total'] for k, (n, _) in enumerate(columns) if n == name]
            spec = self.decisions[name]
            fractions = [(v - lo) / (hi - lo) for v, lo, hi in zip(values, spec['low'], spec['high'])]
            self.vector[name] = [math.log(max(v, 1e-12) / max(1 - v, 1e-12)) for v in fractions]
        self.routing_fit = {'measured_finish_seconds': measured.tolist(),
                            'residual_seconds': (residual * scale).tolist(),
                            'predicted_finish_seconds': ((residual + jacobian @ x) * scale).tolist()}
        self.routing_samples = []
