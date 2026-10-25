"""design/whole-function.md"""
import copy
import math
from routing import weight_ticks
import statistics

SHARE_TICKS = 1 << 40


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
