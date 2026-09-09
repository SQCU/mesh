from collections import Counter, defaultdict
from itertools import combinations

import numpy as np

from .report_store import identity


class Ratings:
    def __init__(self, store):
        self.store = store
        self.previous = {source: value for source, key, value in store.records('exposure_cursor')}
        self.report = next((value for source, key, value in store.records('ratings')), {})
        self.dirty = False

    def ingest(self, source, frame):
        episode = frame.get('episode_id')
        context = frame.get('match_metadata', {})
        members = []
        counts = Counter(row['team'] for row in frame.get('assignments', ()))
        for row in frame.get('assignments', ()):
            configuration = row.get('bot_configuration')
            config_id = identity(configuration) if configuration is not None else None
            if config_id:
                self.store.put('bot_configuration', '', config_id, {'id': config_id,
                    'schema': row['bot_configuration_schema'], 'values': configuration})
            controller = {'arm': row.get('applied_policy_arm'), 'lineage': row.get('applied_policy_lineage')}
            controller_id = identity(controller) if row.get('applied_state_current') and controller['arm'] else None
            if controller_id:
                self.store.put('controller', '', controller_id, {'id': controller_id, **controller,
                    'scope': 'observed_learning_trajectory; versions retained in exposure records'})
            members.append({key: row.get(key) for key in ('edict', 'team', 'spawn_time', 'engine_time',
                'applied_response_seq', 'applied_policy_updates', 'outcome_totals')})
            members[-1].update(configuration=config_id, controller=controller_id,
                player_count=counts[row['team']], participant_kind=row.get('controller'))
        compositions = {}
        for team in counts:
            bots = sorted(member['configuration'] for member in members if member['team'] == team and member['configuration'])
            composition = {'bots': bots, 'unconfigured_players': counts[team] - len(bots)}
            key = identity(composition)
            self.store.put('composition', '', key, {'id': key, **composition})
            compositions[team] = key
        current = {'episode': episode, 'record_id': frame.get('record_id'),
            'engine_episode': frame.get('engine_episode'), 'context': context, 'members': members,
            'compositions': compositions, 'request_seq': frame.get('request_seq'),
            'producer_id': frame.get('producer_id'), 'team_count': frame.get('k')}
        previous = self.previous.get(source)
        before = {row['edict']: row for row in previous['members']} if previous and previous['episode'] == episode else {}
        for row in members:
            old = before.get(row['edict'])
            if old is None or row['engine_time'] is None or old['engine_time'] is None:
                continue
            duration = row['engine_time'] - old['engine_time']
            if duration <= 0:
                continue
            stable = all(row[key] == old[key] for key in ('team', 'spawn_time', 'configuration', 'controller', 'player_count'))
            known = stable and row['configuration'] is not None and row['controller'] is not None
            changes = {name: value - old.get('outcome_totals', {}).get(name, value)
                for name, value in (row.get('outcome_totals') or {}).items()}
            record = {'source': source, 'episode': episode, 'context': context,
                'seed': context.get('seed'), 'edict': row['edict'], 'team': row['team'],
                'start': old['engine_time'], 'end': row['engine_time'], 'seconds': duration,
                'before': old, 'after': row, 'composition': compositions[row['team']],
                'known_source_at_endpoints': bool(known),
                'controller_changed': row['controller'] != old['controller'],
                'version_changed': row['applied_policy_updates'] != old['applied_policy_updates'],
                'counter_deltas': {name: value if value >= 0 else None for name, value in changes.items()},
                'producer_changed': frame.get('producer_id') != previous.get('producer_id'),
                'request_interval': [previous.get('request_seq'), frame.get('request_seq')]}
            self.store.put('exposure', source, identity([episode, row['edict'], record['start'], record['end']]), record)
        self.previous[source] = current
        self.store.put('exposure_cursor', source, 'current', current)
        self.store.put('episodes', source, str(episode), current)
        self.dirty |= self.store.get('rating_outcome', source, str(episode)) is not None

    def outcome(self, source, event):
        episode = event.get('episode_id')
        if episode is None or not isinstance(episode, str):
            matches = [key for src, key, value in self.store.records('episodes')
                if src == source and value.get('engine_episode') == event.get('episode', episode)]
            episode = matches[0] if len(matches) == 1 else None
        if episode is None:
            self.store.put('unmatched_outcome', source, identity(event), event)
            return
        self.dirty |= self.store.put('rating_outcome', source, str(episode), event)

    def refresh(self):
        for source, key, event in self.store.records('unmatched_outcome'):
            self.outcome(source, event)
        if not self.dirty:
            return self.report
        episodes = {(source, key): value for source, key, value in self.store.records('episodes')}
        outcomes = {(source, key): value for source, key, value in self.store.records('rating_outcome')}
        aggregates = defaultdict(lambda: defaultdict(lambda: {'seconds': 0.0, 'unknown_seconds': 0.0, 'span': 0.0, 'features': Counter()}))
        for source, key, row in self.store.records('exposure'):
            bucket = aggregates[(source, str(row['episode']))][row['team']]
            end = outcomes.get((source, str(row['episode'])), {}).get('time', row['end'])
            seconds = max(0.0, min(row['end'], end) - row['start'])
            bucket['seconds'] += seconds
            if row['known_source_at_endpoints']:
                bucket['span'] += seconds / row['after']['player_count']
                bucket['features']['bot:' + row['after']['configuration']] += seconds
                bucket['features']['controller:' + row['after']['controller']] += seconds
            else:
                bucket['unknown_seconds'] += seconds
        groups, excluded = defaultdict(list), []
        for source, episode, outcome in self.store.records('rating_outcome'):
            record = episodes.get((source, episode), {})
            teams = aggregates[(source, episode)]
            if len(teams) != record.get('team_count') or any(row['span'] <= 0 or row['unknown_seconds'] for row in teams.values()):
                excluded.append({'source': source, 'episode': episode, 'reason': 'incomplete_configuration_or_controller_exposure',
                    'teams': {team: {key: value for key, value in row.items() if key != 'features'} for team, row in teams.items()}})
                continue
            vectors = {team: {key: value / row['span'] for key, value in row['features'].items()}
                for team, row in teams.items()}
            configuration = record.get('context', {}).get('configuration', {})
            context = {key: configuration.get(key) for key in ('map', 'teams', 'carts', 'score_limit', 'checkpoint_score_rate')}
            winner = outcome['actor_team']
            pairs = [(winner, other, 1.0) for other in sorted(teams) if other != winner] if winner > 0 else [(*pair, 0.5) for pair in combinations(sorted(teams), 2)]
            if winner > 0 and winner not in teams:
                excluded.append({'source': source, 'episode': episode, 'reason': 'winning_team_not_observed'})
                continue
            for left, right, result in pairs:
                difference = dict(vectors[left])
                for name, value in vectors[right].items():
                    difference[name] = difference.get(name, 0.0) - value
                groups[identity(context)].append({'context': context, 'features': difference, 'result': result,
                    'weight': 1.0 / len(pairs), 'source': source, 'episode': episode,
                    'teams': [{'composition': record['compositions'].get(str(team), record['compositions'].get(team)),
                        'features': vector} for team, vector in vectors.items()]})
        fitted = [fit(rows) for rows in groups.values()]
        self.report = {'contract': 'realized-configuration-controller-ratings-v1',
            'measure': 'regularized Bradley-Terry; one total likelihood weight per completed round',
            'scope': 'conditional observational estimates; bot and controller effects are not causal effects',
            'exposure_scope': 'same observed source at interval endpoints; intermediate invocations are not reconstructed',
            'groups': fitted, 'excluded_rounds': excluded,
            'completed_rounds': self.store.count('rating_outcome'),
            'exposure_records': self.store.count('exposure'),
            'configurations': self.store.count('bot_configuration'),
            'compositions': self.store.count('composition')}
        self.store.put('ratings', '', 'current', self.report)
        self.dirty = False
        return self.report


def fit(rows, precision=1.0, steps=32):
    names = sorted({name for row in rows for team in row['teams'] for name in team['features']})
    index = {name: i for i, name in enumerate(names)}
    x = np.zeros((len(rows), len(names)), dtype=np.float64)
    for i, row in enumerate(rows):
        for name, value in row['features'].items():
            if name in index:
                x[i, index[name]] = value
    u, singular, vt = np.linalg.svd(x, full_matrices=False)
    tolerance = np.finfo(np.float64).eps * max(x.shape) * (singular[0] if len(singular) else 0)
    rank = int(np.count_nonzero(singular > tolerance))
    basis, projection = u[:, :rank] * singular[:rank], vt[:rank]
    y = np.asarray([row['result'] for row in rows])
    weights = np.asarray([row['weight'] for row in rows])
    beta = np.zeros(rank)
    converged = rank == 0
    def objective(value):
        logits = basis @ value
        return np.sum(weights * (np.logaddexp(0, logits) - y * logits)) + precision * (value @ value) / 2
    for iteration in range(steps):
        probability = np.exp(-np.logaddexp(0, -(basis @ beta)))
        gradient = basis.T @ (weights * (probability - y)) + precision * beta
        curvature = (basis.T * (weights * probability * (1 - probability))) @ basis + precision * np.eye(rank)
        if np.linalg.norm(gradient) < 1e-8:
            converged = True
            break
        direction = np.linalg.solve(curvature, gradient)
        scale, current = 1.0, objective(beta)
        while scale > np.finfo(np.float64).eps and objective(beta - scale * direction) > current:
            scale *= 0.5
        beta -= scale * direction
    probability = np.exp(-np.logaddexp(0, -(basis @ beta)))
    curvature = (basis.T * (weights * probability * (1 - probability))) @ basis + precision * np.eye(rank)
    covariance = np.linalg.solve(curvature, np.eye(rank))
    coefficients = projection.T @ beta
    support = np.sum(projection ** 2, axis=0)
    variance = np.sum(projection * (covariance @ projection), axis=0) + (1 - support) / precision
    elo_scale = 400.0 / np.log(10.0)
    teams = {identity(team): team for row in rows for team in row['teams']}
    team_ratings = []
    for key, team in teams.items():
        vector = np.asarray([team['features'].get(name, 0.0) for name in names])
        projected = projection @ vector
        unknown = max(0.0, float(vector @ vector - projected @ projected))
        variance_team = float(projected @ covariance @ projected + unknown / precision)
        team_ratings.append({'id': key, 'composition': team['composition'], 'features': team['features'],
            'elo': float(1500 + elo_scale * (vector @ coefficients)),
            'posterior_sd_elo': float(elo_scale * np.sqrt(max(0.0, variance_team))),
            'unidentified_feature_mass': unknown})
    return {'context': rows[0]['context'], 'team_ratings': team_ratings, 'pair_observations': len(rows),
        'rounds': len({(row['source'], row['episode']) for row in rows}),
        'rank': rank, 'parameters': len(names), 'unidentified_dimensions': len(names) - rank,
        'prior_precision': precision, 'converged': converged,
        'ratings': [{'factor': name, 'elo_offset': float(elo_scale * coefficients[i]),
            'posterior_sd_elo': float(elo_scale * np.sqrt(max(0.0, variance[i]))),
            'identified_fraction': float(support[i])} for i, name in enumerate(names)]}
