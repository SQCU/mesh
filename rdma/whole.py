"""design/whole-function.md"""
import copy
import math
from routing import weight_ticks
import statistics

SHARE_TICKS = 1 << 40


def performance(observation, transaction_weight=0.0):
    """Performance-only objective; design/objective-20261024.md.

    ``useful_work_bytes`` counts the fixed numerical function's logical output,
    never transport hops, padding, or intermediate collective deliveries. If it
    is absent, one completed function is the useful-work unit. A separately
    declared transaction-window preference is a tradeoff, not a congestion tax.
    """
    seconds = float(observation['seconds'])
    useful = float(observation.get('useful_work_bytes', 1))
    weight = float(transaction_weight)
    if not 2 ** -40 <= seconds <= 2 ** 40 or not 2 ** -40 <= useful <= 2 ** 64:
        raise ValueError('rescale metrics: latency must be in [2^-40, 2^40] seconds and useful work in [2^-40, 2^64] units')
    if not 0 <= weight <= 2 ** 40:
        raise ValueError('transaction preference must be a finite value in [0, 2^40]')
    result = {'cost': math.log(seconds) - math.log(useful),
              'useful_work': useful,
              'useful_work_unit': 'logical_output_bytes' if 'useful_work_bytes' in observation else 'functions',
              'whole_function_seconds': seconds,
              'whole_function_throughput_per_second': useful / seconds,
              'transaction_weight': weight}
    if 'roofline' in observation:
        roof = observation['roofline']
        lower, upper = map(float, roof['window_seconds_bounds'])
        transaction_bytes = float(roof['useful_bytes'])
        if (not all(math.isfinite(x) for x in (lower, upper, transaction_bytes))
                or lower < 0 or lower > upper or not 2 ** -40 <= upper <= 2 ** 40
                or not 2 ** -40 <= transaction_bytes <= 2 ** 64):
            raise ValueError('transaction window bounds and useful bytes must describe completed transfers')
        result.update(transaction_window_seconds=upper, useful_transaction_bytes=transaction_bytes,
                      useful_transaction_bandwidth_bytes_per_second=transaction_bytes / upper)
        if weight:
            result['cost'] += weight * (math.log(upper) - math.log(transaction_bytes))
    elif weight:
        raise ValueError('a transaction preference requires complete transaction-window observations')
    return result


def delay_observations(cables, ranks, offsets):
    """Observed first-frame delay; never serialize whole relay transfers.

    Bracket midpoints/variances assume uniform timing uncertainty within each
    observed interval. Clock-offset uncertainty is shared between observations
    and supplied separately so the posterior retains that correlation.

    A SEND may be refined at relay dependencies while its destination COPY
    remains whole. Validate all directed page coverage, but fit only boundaries
    whose source eligibility and destination first landing were both observed.
    An interior landing bounded only by the enclosing COPY's entire duration
    is not a measured first-frame sample. Entry subdivision is not evidence.
    """
    from roofline import matched_transfers
    nodes = sorted(r['rank'] for r in ranks)
    labels = [f'node:{r}' for r in nodes] + [f'link:{e}' for e in range(2 * len(cables))] + ['function']
    matches = matched_transfers(ranks)
    design, values, variances, clock, groups, page_sizes, first_spans = [], [], [], [], [], set(), []
    explicit_spans = any('unit_bytes' in m['send'] or m['send'].get('unit_kind') == 'byte_extent' for m in matches)
    for match in matches:
        source, destination, sent = match['source'], match['destination'], match['send']
        lengths = match['unit_bytes']
        page_sizes.update(lengths)
        cable = cables[sent['cable']]
        if source == destination or set(cable['nodes']) != {source, destination}:
            raise ValueError('delay observation endpoints disagree with the physical cable')
        if not match['arrival_exact']:
            continue
        arc = 2 * sent['cable'] + (source == cable['nodes'][1])
        capacity = cable['bits_per_second'] / 8
        start = (sent['start_lo_ns'] + sent['start_hi_ns']) / 2 + sum(offsets[str(source)]) / 2
        arrival = sum(match['arrival_ns']) / 2 + sum(offsets[str(destination)]) / 2
        values.append((arrival - start) * 1e-9 - lengths[0] / capacity)
        first_spans.append(lengths[0])
        variances.append(((sent['start_hi_ns'] - sent['start_lo_ns']) ** 2
                          + (match['arrival_ns'][1] - match['arrival_ns'][0]) ** 2) * 1e-18 / 12)
        row, clocks = [0.0] * len(labels), [0.0] * len(nodes)
        row[nodes.index(source)] = row[nodes.index(destination)] = 1
        row[len(nodes) + arc] = row[-1] = 1
        clocks[nodes.index(source)], clocks[nodes.index(destination)] = -1, 1
        design.append(row); clock.append(clocks); groups.append(f'link:{arc}')
    if not design or (not explicit_spans and len(page_sizes) != 1):
        raise ValueError('delay inference requires a complete set of observed transfers')
    receive_count = sum(w['direction'] == 1 for rank in ranks for w in rank['windows'])
    result = {'labels': labels, 'design': design, 'seconds': values, 'variance': variances,
            'clock_design': clock, 'clock_variance': [(offsets[str(r)][1] - offsets[str(r)][0]) ** 2 * 1e-18 / 12 for r in nodes],
            'groups': groups,
            'coverage': {'send_entries': len(matches), 'receive_entries': receive_count,
                         'wire_bytes': sum(match['send']['bytes'] for match in matches),
                         'observed_shared_first_boundaries': len(design),
                         'send_entries_without_destination_first_observation': len(matches) - len(design),
                         'receive_entries_without_source_first_observation': receive_count - len(design)}}
    if explicit_spans:
        result.update(unit_kind='byte_extent', first_span_bytes=first_spans)
    else:
        result['page_bytes'] = page_sizes.pop()
    return result


class DelayPosterior:
    """Joint Gaussian random effects, updated only after a complete function.

    Nodes begin with a common mean (zero by default) and broad uncertainty. Full covariance
    preserves the ambiguity between node, link and function effects. The means
    are observed first-frame excess delays, not an arbitrary per-hop tax.
    """

    def __init__(self, labels, prior_stddev=1e-3, process_stddev=1e-5, noise_stddev=1e-6, prior_mean=0.0):
        if len(set(labels)) != len(labels) or not labels:
            raise ValueError('delay effects require distinct labels')
        if (not all(math.isfinite(x) for x in (prior_mean, prior_stddev, process_stddev, noise_stddev))
                or not (prior_stddev > 0 and process_stddev >= 0 and noise_stddev > 0)):
            raise ValueError('delay prior/noise must be positive and process noise nonnegative')
        self.labels = list(labels)
        self.mean = [float(prior_mean) if label.startswith('node:') else 0.0 for label in labels]
        self.covariance = [[prior_stddev ** 2 if i == j else 0.0 for j in range(len(labels))] for i in range(len(labels))]
        self.process_variance, self.noise_floor = process_stddev ** 2, noise_stddev ** 2
        self.noise_variance, self.updates, self.last, self.predictions = {}, 0, None, {}

    def state(self):
        return copy.deepcopy(self.__dict__)

    @classmethod
    def of(cls, state):
        obj = cls.__new__(cls)
        obj.__dict__.update(copy.deepcopy(state))
        return obj

    def update(self, batch):
        import numpy as np
        if batch['labels'] != self.labels:
            raise ValueError('delay observations changed their node/link/function effects')
        H, y = np.asarray(batch['design'], dtype=float), np.asarray(batch['seconds'], dtype=float)
        observed = np.asarray(batch['variance'], dtype=float)
        clocks, clock_variance = np.asarray(batch['clock_design'], dtype=float), np.asarray(batch['clock_variance'], dtype=float)
        groups = batch['groups']
        n, m = len(self.mean), len(y)
        if (H.shape != (m, n) or observed.shape != (m,) or clocks.shape != (m, len(clock_variance))
                or len(groups) != m or not m or any(not np.all(np.isfinite(a)) for a in (H, y, observed, clocks, clock_variance))
                or np.any(observed < 0) or np.any(clock_variance < 0)):
            raise ValueError('delay batch has invalid dimensions or nonfinite evidence')
        prior = np.asarray(self.covariance) + np.eye(n) * self.process_variance
        mean = np.asarray(self.mean)
        noise = np.asarray([self.noise_variance.get(g, self.noise_floor) for g in groups])
        inverse_diagonal = 1 / (observed + noise)
        clock_factor = clocks * np.sqrt(clock_variance)
        weighted_clock = inverse_diagonal[:, None] * clock_factor

        def precision(values):
            weighted = inverse_diagonal[:, None] * values
            return weighted - weighted_clock @ np.linalg.solve(np.eye(len(clock_variance)) + clock_factor.T @ weighted_clock,
                                                                clock_factor.T @ weighted)

        # Woodbury on the few shared clock offsets, then an information-form
        # update on effects. No cubic solve in the number of traced entries.
        innovation = y - H @ mean
        posterior = np.linalg.inv(np.linalg.inv(prior) + H.T @ precision(H))
        posterior = (posterior + posterior.T) / 2
        mean += posterior @ H.T @ precision(innovation[:, None])[:, 0]
        residual = y - H @ mean
        uncertainty = np.sum((H @ posterior) * H, axis=1)
        clock_diagonal = np.sum(clock_factor ** 2, axis=1)
        for group in sorted(set(groups)):
            ids = [i for i, g in enumerate(groups) if g == group]
            # Predictive residual moment, after accounting for observation
            # brackets. This is total link-observation variability, not a
            # claim that its node and function causes are separately known.
            moment = float(np.mean((residual ** 2 + uncertainty - observed - clock_diagonal)[ids]))
            old = self.noise_variance.get(group, self.noise_floor)
            self.noise_variance[group] = 0.95 * old + 0.05 * max(self.noise_floor, moment)
        self.mean, self.covariance = mean.tolist(), posterior.tolist()
        for group in sorted(set(groups)):
            row = H[groups.index(group)]
            parameter_variance = float(row @ posterior @ row)
            self.predictions[group] = {'mean_seconds': float(row @ mean),
                                      'parameter_stddev_seconds': math.sqrt(max(0, parameter_variance)),
                                      'predictive_stddev_seconds': math.sqrt(max(0, parameter_variance)
                                          + float(row @ row) * self.process_variance + self.noise_variance[group])}
        self.updates += 1
        self.last = {'observations': m, 'innovation_rms_seconds': float(np.sqrt(np.mean(innovation ** 2))),
                     'residual_rms_seconds': float(np.sqrt(np.mean(residual ** 2)))}
        if 'coverage' in batch:
            self.last['coverage'] = copy.deepcopy(batch['coverage'])
        return self.report()

    def report(self):
        return {'effects': {name: {'mean_seconds': self.mean[i], 'stddev_seconds': math.sqrt(max(0, self.covariance[i][i]))}
                            for i, name in enumerate(self.labels)},
                'labels': self.labels, 'covariance_seconds_squared': self.covariance,
                'link_observation_stddev_seconds': {name: math.sqrt(v) for name, v in self.noise_variance.items()},
                'first_frame_excess_delay': self.predictions,
                'updates': self.updates, 'last': self.last}


def pipeline_delay_prior(model, posterior, fractions):
    """Flow-weighted first-frame release prior and its routing Jacobian.

    Upstream first-frame delays add along a dependency path. Bulk transfer
    service remains in the separate shared-capacity load matrix and therefore
    overlaps across hops. Empirical finish residuals correct this release prior.
    """
    import numpy as np
    columns = [(name, p) for name, spec in model['paths'].items() for p in range(len(spec['arcs']))]
    size = len(model['capacities'])
    load, arrival = np.zeros((size, len(columns))), np.zeros((size, len(columns)))
    if posterior is None or 'directed_nodes' not in model:
        return np.zeros(size), load
    labels, mean = posterior['labels'], np.asarray(posterior['mean'])
    delay = []
    for e, (source, destination) in enumerate(model['directed_nodes']):
        delay.append(sum(mean[labels.index(label)] for label in
                         (f'node:{source}', f'node:{destination}', f'link:{e}', 'function')))
    for k, (name, p) in enumerate(columns):
        prefix = 0.0
        spec = model['paths'][name]
        arcs = spec['arcs'][p]
        if 'first_span_bytes' in spec or 'first_span_service_upper_bound_bytes' in spec:
            # A declared service upper bound is a conservative model prior.
            # Observed delay residuals always subtract actual trace lengths.
            key = 'first_span_bytes' if 'first_span_bytes' in spec else 'first_span_service_upper_bound_bytes'
            sizes = spec[key][p]
            if len(sizes) != len(arcs) or any(not math.isfinite(n) or n <= 0 for n in sizes):
                raise ValueError('routing first-span lengths must describe every path arc')
        elif model.get('unit_kind') == 'byte_extent':
            raise ValueError('byte-extent routing requires compiled first-span lengths')
        else:
            sizes = [model.get('page_bytes', 0)] * len(arcs)
        for e, span_bytes in zip(arcs, sizes):
            load[e, k] = model['paths'][name]['bytes']
            arrival[e, k] = load[e, k] * (prefix + delay[e])
            prefix += delay[e] + span_bytes / model['capacities'][e]
    bytes_on_edge, weighted_arrival = load @ fractions, arrival @ fractions
    offset = np.divide(weighted_arrival, bytes_on_edge, out=np.zeros(size), where=bytes_on_edge > 0)
    derivative = np.divide(arrival - offset[:, None] * load, bytes_on_edge[:, None],
                           out=np.zeros_like(load), where=bytes_on_edge[:, None] > 0)
    return offset, derivative


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


def simplex(values, total, low, high):
    """Project linear shares onto a bounded simplex with 40 fractional bits.

    The precision is explicit (9.1e-13 of a decision), with exact zero/one and
    conservation, without exponential tails or a positive allocation floor.
    The transport's coarser allocation is separate.
    """
    if not total > 0 or not all(math.isfinite(v) for v in [total, *values, *low, *high]):
        raise ValueError('simplex operands must be finite with a positive total')
    ticks = SHARE_TICKS
    from fractions import Fraction
    ratio = Fraction(ticks) / Fraction(total)
    # Tiny operands are explicitly below representation precision. Do not
    # divide them into subnormals on the way to an inevitable rounded zero.
    scaled = [0.0 if abs(v) < total / ticks else v / total * ticks for v in values]
    result = project(scaled, ticks, 1,
                     [math.ceil(Fraction(v) * ratio) for v in low],
                     [math.floor(Fraction(v) * ratio) for v in high])
    return [v / ticks * total for v in result]


class FunctionFlow:
    """design/whole-function.md#finite-differences"""

    def __init__(self, decisions, learning_rate=0.15, perturbation=0.025,
                 fit_compute=False, routing_model=None, transaction_weight=0.0):
        self.decisions = {name: copy.deepcopy(decisions[name]) for name in sorted(decisions)}
        self.vector = {}
        self.learning_rate, self.perturbation = float(learning_rate), float(perturbation)
        if not all(math.isfinite(v) and 0 < v <= 1 for v in (self.learning_rate, self.perturbation)):
            raise ValueError('step and perturbation must be finite fractions in (0, 1]')
        self.coordinate_version = 'linear-simplex-q40-v1'
        self.transaction_weight = float(transaction_weight)
        if not 0 <= self.transaction_weight <= 2 ** 40:
            raise ValueError('transaction preference must be a finite value in [0, 2^40]')
        self.objective_version, self.metric_contract = 'whole_function_throughput_v2', None
        self.fit_compute = bool(fit_compute)
        self.compute_samples, self.compute_model = [], None
        self.schedule_model = None
        self.routing_model, self.routing_samples, self.routing_fit = copy.deepcopy(routing_model), [], None
        self.routing_observations = []
        self.delay_model, self.delay_posterior = None, None
        self.serial, self.round, self.workload = 0, 0, None
        self.samples, self.history = [], []
        self.calibration_parts = {name: list(spec['parts']) for name, spec in self.decisions.items()}
        for name, spec in self.decisions.items():
            parts = list(map(float, spec['parts']))
            total = sum(parts)
            if not parts or not 2 ** -40 <= total <= 2 ** 52 or any(not math.isfinite(v) or v < 0 for v in parts):
                raise ValueError('decision shares must be finite, nonnegative; choose units with total in [2^-40, 2^52]')
            spec.update(total=total, grain=int(spec.get('grain', 0)),
                        low=list(map(float, spec.get('low', [0] * len(parts)))),
                        high=list(map(float, spec.get('high', [total] * len(parts)))))
            if (len(spec['low']) != len(parts) or len(spec['high']) != len(parts)
                    or any(not math.isfinite(v) or not 0 <= v <= total for v in spec['low'] + spec['high'])):
                raise ValueError('simplex bounds must match its coordinates and lie within its total')
            if not all(lo <= v <= hi for v, lo, hi in zip(parts, spec['low'], spec['high'])):
                raise ValueError('initial vector is outside its simplex')
            quantum = spec.get('quantum', spec['grain'])
            quantum = [quantum] * len(parts) if isinstance(quantum, (float, int)) else quantum
            if len(quantum) != len(parts) or any(not math.isfinite(q) or not 0 <= q <= total for q in quantum):
                raise ValueError('coordinate quantum must be in [0, total] and match its parts')
            spec['quantum'] = [round(q / total * SHARE_TICKS) / SHARE_TICKS * total for q in quantum]
            self.vector[name] = weight_ticks(simplex(parts, total, spec['low'], spec['high']))
            self.calibration_parts[name] = [v / SHARE_TICKS * total for v in self.vector[name]]
            if spec['grain']:
                self.calibration_parts[name] = project(self.calibration_parts[name], int(total), spec['grain'], spec['low'], spec['high'])
            spec['parts'] = list(self.calibration_parts[name])
        self.dimension = sum(map(len, self.vector.values()))
        self.order = 1 << max(1, (self.dimension - 1).bit_length())

    def state(self):
        return copy.deepcopy(self.__dict__)

    @classmethod
    def of(cls, state):
        if state.get('objective_version') != 'whole_function_throughput_v2':
            raise ValueError('optimizer state uses the obsolete congestion objective; realize a new controller from its configuration')
        if state.get('coordinate_version') != 'linear-simplex-q40-v1':
            raise ValueError('optimizer state uses obsolete logarithmic coordinates; realize from its recorded physical allocation')
        obj = cls.__new__(cls)
        obj.__dict__.update(copy.deepcopy(state))
        obj.decisions = {name: obj.decisions[name] for name in sorted(obj.decisions)}
        obj.vector = {name: obj.vector[name] for name in sorted(obj.vector)}
        for name, ticks in obj.vector.items():
            if (name not in obj.decisions or len(ticks) != len(obj.decisions[name]['parts'])
                    or any(type(v) is not int or v < 0 for v in ticks) or sum(ticks) != SHARE_TICKS):
                raise ValueError('saved simplex coordinates must be nonnegative integer ticks conserving 2^40')
        for name, value in {'fit_compute': False,
                            'compute_samples': [], 'compute_model': None,
                            'routing_model': None, 'routing_samples': [], 'routing_fit': None, 'schedule_model': None,
                            'routing_observations': [], 'delay_model': None, 'delay_posterior': None}.items():
            obj.__dict__.setdefault(name, value)
        return obj

    def _direction(self):
        row, at, result = 1 + self.round % (self.order - 1), 0, {}
        for name in sorted(self.vector):
            values = self.vector[name]
            result[name] = [1.0 - 2.0 * ((row & (at + i)).bit_count() % 2) for i in range(len(values))]
            at += len(values)
        return result

    def _values(self, sign):
        direction, result = self._direction(), {}
        for name, values in self.vector.items():
            spec, c = self.decisions[name], self._radius(name)
            result[name] = simplex([(v / SHARE_TICKS + sign * c * d) * spec['total'] for v, d in zip(values, direction[name])],
                                    spec['total'], spec['low'], spec['high'])
        return result

    def _radius(self, name):
        spec = self.decisions[name]
        # At a boundary, a projected +/- probe has only one inward half.
        # One declared allocation unit keeps that probe physically observable.
        return max(self.perturbation / (self.round + 1) ** 0.101,
                   max(spec['quantum']) / spec['total'], 2 ** -40)

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
        metrics = performance(observation, self.transaction_weight)
        contract = [metrics['useful_work_unit'], metrics['useful_work'],
                    metrics.get('useful_transaction_bytes') if self.transaction_weight else None]
        if self.metric_contract is not None and contract != self.metric_contract:
            raise ValueError('finite differences require invariant useful work and transaction accounting')
        seconds, congestion = metrics['whole_function_seconds'], float(observation.get('congestion_seconds', 0))
        if not math.isfinite(seconds) or seconds <= 0 or not math.isfinite(congestion) or congestion < 0:
            raise ValueError('function latency and congestion must be finite')
        finish = float(observation.get('finish_variance', 0))
        if not math.isfinite(finish) or finish < 0:
            raise ValueError('finish variance must be finite and nonnegative')
        self.workload, self.metric_contract = workload, contract
        if 'delay_observations' in observation:
            batch = observation['delay_observations']
            prior = (self.routing_model or {}).get('delay_prior', {})
            posterior = DelayPosterior.of(self.delay_model) if self.delay_model is not None else DelayPosterior(batch['labels'], **prior)
            self.delay_posterior = posterior.update(batch)
            self.delay_model = posterior.state()
            if self.routing_model is not None and 'page_bytes' in batch:
                self.routing_model['page_bytes'] = batch['page_bytes']
            if self.routing_model is not None and batch.get('unit_kind') == 'byte_extent':
                self.routing_model['unit_kind'] = 'byte_extent'
        if self.fit_compute:
            self._compute_observation(observation)
        if self.routing_model is not None:
            self.routing_samples.append(copy.deepcopy(observation['finishes']['stages']))
            self.routing_observations.append({'parts': copy.deepcopy(observation['parts']),
                                              'stages': copy.deepcopy(observation['finishes']['stages'])})
            self.routing_observations = self.routing_observations[-64:]
        if expected['phase'] == 'centre':
            # A measured calibration/repeat is evidence, not a zero-offset FD sample.
            self.serial += 1
            return self.ask()
        self.samples.append({'id': self.serial, 'seconds': seconds, 'cost': metrics['cost'],
                             'performance': metrics, 'offset': expected['offset']})
        self.serial += 1
        if len(self.samples) == 4:
            difference = (self.samples[0]['cost'] - self.samples[1]['cost'] - self.samples[2]['cost'] + self.samples[3]['cost']) / 4
            direction, gradient = self._direction(), {}
            before = self.centre
            rate = self.learning_rate / (self.round + 1) ** 0.3
            for name, values in self.vector.items():
                derivative = difference / self._radius(name)
                raw = [derivative * d for d in direction[name]]
                mean = statistics.mean(raw)
                gradient[name] = [g - mean for g in raw]
                # Bounded scaling, without a small denominator or momentum:
                # flat measurements produce exactly no movement of the centre.
                scale = 1 + max(map(abs, gradient[name]))
                spec = self.decisions[name]
                self.vector[name] = weight_ticks(simplex(
                    [(v / SHARE_TICKS - rate * g / scale) * spec['total'] for v, g in zip(values, gradient[name])],
                    spec['total'], spec['low'], spec['high']))
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
        self.vector['compute'] = weight_ticks(simplex(values, total, spec['low'], spec['high']))
        self.compute_model = {'fixed': fixed, 'slopes': slopes, 'start_delay': start_delay, 'tails': tails, 'target': target,
                              'tiles': tiles, 'predicted_finish': finish}
        if 'release' in self.compute_samples[-1]:
            self.schedule_model = {'release': [statistics.mean(s['release'][i] for s in recent) for i in range(n)],
                                   'parts': [statistics.mean(s['parts'][i] for s in recent) for i in range(n)],
                                   'stages': recent[-1]['stages']}

    def _routing_step(self):
        """Measured finish residual + physical load Jacobian, after a stencil.

        A convex local correction complements the joint finite difference of
        whole-function performance. Every correction moves the vector; no
        measured candidate is selected or rejected. Residuals are refreshed from
        actual landing times, so release delays need not match the fluid model.
        """
        import numpy as np
        from scipy.optimize import minimize
        from scipy.special import logsumexp
        model = self.routing_model
        edges = len(model['capacities'])
        stages = len(self.routing_observations[-1]['stages']) if model.get('serial_stages') else 1
        columns = [(name, p) for name, spec in model['paths'].items() for p in range(len(spec['arcs']))]
        names = list(model['paths'])
        matrix = np.zeros((edges, len(columns)))
        conserve = np.zeros((len(names), len(columns)))
        centre = self.centre
        x0 = np.asarray([centre[name][p] / self.decisions[name]['total'] for name, p in columns])
        for k, (name, p) in enumerate(columns):
            conserve[names.index(name), k] = 1
            for e in model['paths'][name]['arcs'][p]:
                matrix[e, k] = model['paths'][name]['bytes'] / model['capacities'][e]
        def finish(stages):
            measured = []
            for stage in stages:
                arcs = stage['arcs']
                origin = min(v['start_ns'][0] for v in arcs.values())
                measured.append([(sum(arcs[str(e)]['finish_ns']) / 2 - origin) * 1e-9 for e in range(edges)])
            return np.asarray(measured).reshape(-1) if model.get('serial_stages') else np.mean(measured, axis=0)

        # Learn residual sensitivities, not just offsets. A change to one
        # route can change another cable's upstream queue. Compute fractions
        # are joint features so their perturbations are not misattributed to
        # the path weights. Ridge regularization retains the physical prior
        # in directions that the completed calls have not identified yet.
        def features(parts):
            values = [parts[name][p] / self.decisions[name]['total'] for name, p in columns]
            if 'compute' in parts:
                values += [v / self.decisions['compute']['total'] for v in parts['compute']]
            return values

        observations = self.routing_observations
        X = np.asarray([features(o['parts']) for o in observations])
        Y = np.asarray([finish(o['stages']) for o in observations])
        physical = np.pad(np.tile(matrix, (stages, 1)), ((0, 0), (0, X.shape[1] - len(columns))))
        arrival = np.asarray([np.tile(pipeline_delay_prior(model, self.delay_model, x[:len(columns)])[0], stages) for x in X])
        residuals = Y - X @ physical.T - arrival
        weights = np.asarray([0.97 ** (len(X) - 1 - k) for k in range(len(X))])
        xmean, ymean = np.average(X, axis=0, weights=weights), np.average(residuals, axis=0, weights=weights)
        centred_x = (X - xmean) * np.sqrt(weights)[:, None]
        centred_y = (residuals - ymean) * np.sqrt(weights)[:, None]
        reliability = np.ones(edges)
        if self.delay_model is not None and 'directed_nodes' in model:
            labels = self.delay_model['labels']
            covariance = np.asarray(self.delay_model['covariance'])
            variances = []
            for e, (source, destination) in enumerate(model['directed_nodes']):
                row = np.zeros(len(labels))
                for label in (f'node:{source}', f'node:{destination}', f'link:{e}', 'function'):
                    row[labels.index(label)] = 1
                variances.append(float(row @ covariance @ row)
                                 + self.delay_model['noise_variance'].get(f'link:{e}', self.delay_model['noise_floor']))
            variances = np.asarray(variances)
            median = np.median(variances)
            reliability = median / (median + variances)
        correction = np.asarray([np.linalg.solve(centred_x.T @ centred_x * weight + np.eye(X.shape[1]) * 1e-3,
                                                 centred_x.T @ centred_y[:, e] * weight)
                                 for e, weight in enumerate(np.tile(reliability, stages))])
        start_at_centre, start_derivative = pipeline_delay_prior(model, self.delay_model, x0)
        start_derivative = np.pad(np.tile(start_derivative, (stages, 1)), ((0, 0), (0, X.shape[1] - len(columns))))
        sensitivity = physical + correction + start_derivative
        measured, observed_x = np.mean(Y[-4:], axis=0), np.mean(X[-4:], axis=0)
        at_centre = (measured + (physical + correction) @ (np.asarray(features(centre)) - observed_x)
                     + np.tile(start_at_centre, stages) - np.mean(arrival[-4:], axis=0))
        scale = max(measured)
        jacobian = sensitivity[:, :len(columns)] / scale
        residual = at_centre / scale - jacobian @ x0

        bounds = [(0, 1)] * len(x0)
        # A learned Jacobian is local. A proximal step evolves the vector
        # continuously; globally minimizing its affine extrapolation could
        # move an entire commodity on the strength of one noisy derivative.
        rate = self.learning_rate / (self.round + 1) ** 0.3
        # The smooth maximum differs from max by at most log(cables)/64
        # in units of this stencil's longest measured finish.
        sharpness = 64.0

        def cost(x):
            finish = (residual + jacobian @ x).reshape(stages, edges)
            return float(np.sum(logsumexp(sharpness * finish, axis=1)) / sharpness
                         + np.sum((x - x0) ** 2) / (2 * rate))

        def gradient(x):
            finish = (residual + jacobian @ x).reshape(stages, edges)
            probability = np.exp(sharpness * finish - logsumexp(sharpness * finish, axis=1)[:, None])
            return jacobian.T @ probability.reshape(-1) + (x - x0) / rate

        solved = minimize(cost, x0, jac=gradient, method='SLSQP', bounds=bounds,
                          constraints=[{'type': 'eq', 'fun': lambda x: conserve @ x - 1, 'jac': lambda x: conserve}],
                          options={'ftol': 1e-11, 'maxiter': 300})
        if not solved.success:
            raise ArithmeticError(f'completion residual solve: {solved.message}')
        x = solved.x
        for name in names:
            values = [x[k] * self.decisions[name]['total'] for k, (n, _) in enumerate(columns) if n == name]
            spec = self.decisions[name]
            self.vector[name] = weight_ticks(simplex(values, spec['total'], spec['low'], spec['high']))
        self.routing_fit = {'measured_finish_seconds': measured.tolist(),
                            'residual_seconds': (residual * scale).tolist(),
                            'residual_jacobian_seconds': correction.tolist(),
                            'residual_fit_rms_seconds': float(np.sqrt(np.mean((residuals - ymean - (X - xmean) @ correction.T) ** 2))),
                            'predicted_finish_seconds': ((residual + jacobian @ x) * scale).tolist(),
                            'serial_stages': stages, 'link_model_reliability': reliability.tolist(),
                            'first_frame_prior_seconds': start_at_centre.tolist(),
                            'first_frame_prior_jacobian_seconds': start_derivative[:, :len(columns)].tolist()}
        self.routing_samples = []
