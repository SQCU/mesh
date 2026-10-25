import argparse, json, os, select, signal, socket, sys, time, uuid
from itertools import chain

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "..", "rdma"))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))

from solver.strat.cast_header import Wally, parameter_seed
from solver.strat.scale_rpc import RemoteCross, RemoteScale
from solver.strat.baselines import BaselinePolicy, baseline_strategy, default_strategy
from solver.strat.inputs import native_inputs
from solver.strat.neighborhood import NavigationRows
from solver.strat.execution import PolicyProgram, prepare_policies, emit_policies, serve
from solver.strat.products import World
from solver.strat.action_history import ActionHistory, source_features, view_observation
from solver.strat.buffers import ObservationMemory
from solver.strat.replay import Frame
from solver.strat.checkpoint_state import Payload, encode_payload, decode_payload, open_checkpoint, load_policy, policy_source
from solver.strat.strategy import strategy
from solver.strat.state_steering import PAGE_WIDTH, STATE_HEADER, state_labels, latent_labels, advance_key
from solver.strat.policy_reports import compare_policies
from solver.xonwire import (
    Engine, RuntimeFrames, frames,
    CART_KIND, OBSERVATION_KIND, STRATEGY_KIND, TEAM_KIND, STATE_KIND,
)
from solver.strat.joracle.probe import LiteralJReporter
from solver.strat.policy_contract import architecture_arm, is_matrix_fusion_arm
from solver.strat.runtime import build_runtime_frame
from solver.strat.work_estimate import strategy_work
from solver.strat.game_value import formal_value_record, reward_contract
from solver.strat.scale_config import (
    SCALE_EXPERTS, SCALE_HIDDEN, SCALE_RANK, SCALE_TOPK,
    strategy_widths,
)
from workload import WorkloadMeter

from payload.tools.strategy_io_schema import (
    OBS,
    CS,
    TS,
    EVT,
    OBS_AMMO_COLUMNS,
    OBS_WEAPON_COLUMNS,
)

def team_resources(rows, teams):
    out = []
    for team in range(1, teams + 1):
        selected = rows[np.asarray(rows[:, OBS["TEAM"]], dtype=np.int64) == team]
        alive = selected[:, OBS["ALIVE"]] >= 0.5 if len(selected) else np.zeros(0, dtype=bool)
        speeds = np.linalg.norm(selected[:, OBS["VEL_X"]:OBS["VEL_Z"] + 1], axis=1) if len(selected) else np.zeros(0)
        out.append({
            "team": team,
            "players": len(selected),
            "alive": int(alive.sum()),
            "health": float(selected[:, OBS["HEALTH"]].sum()),
            "armor": float(selected[:, OBS["ARMOR"]].sum()),
            "ammo": {
                name.removeprefix("AMMO_").lower(): float(selected[:, OBS[name]].sum())
                for name in OBS_AMMO_COLUMNS
            },
            "weapon_words": {
                name.removeprefix("WEAPONS_").lower(): int(np.bitwise_or.reduce(
                    selected[:, OBS[name]].astype(np.int64), initial=0,
                ))
                for name in OBS_WEAPON_COLUMNS
            },
            "mean_speed": float(speeds.mean()) if len(speeds) else 0.0,
        })
    return out

def make_policy(arm, widths, hidden, world=None):
    if is_matrix_fusion_arm(arm):
        model = Wally(widths, scale_shard=world.span('scale', widths.scale_h) if world is not None and world.parts('scale') else None)
        model.participant_fusion_scale = float(arm != 'participant_fusion_ablated')
        model.residual_fusion_scale = float(arm != 'residual_fusion_ablated')
        return model, strategy
    if arm == 'default':
        return None, default_strategy
    return BaselinePolicy(arm, widths.d_x, widths.d_c, hidden), baseline_strategy


def place(model, arm, world):
    """The world's partitioned products on a matrix-fusion model (scale_rpc: the row-Gram cross and the scale
    experts divided by the placement's parts)."""
    remotes = {}
    if model is None or not is_matrix_fusion_arm(arm) or world.size == 1:
        return remotes
    if world.parts('cross') is not None:
        model.cross_executor = remotes['cross'] = RemoteCross(world, owner=arm)
    if world.parts('scale') is not None:
        model.scale_executor = remotes['scale'] = RemoteScale(world, owner=arm)
    return remotes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--placement",
                    help="the call's placement document (xonotic/README.md \"The policy learner\"): device, products, socket, framebytes, strategy_interval, relaxation_time, anticipatory_lead")
    ap.add_argument("--placement-json")
    ap.add_argument("--state-stdin", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--train-arms")
    ap.add_argument("--match-metadata", default="{}")
    ap.add_argument("--policy-arm", default="matrix_fusion")
    ap.add_argument("--baseline-hidden", type=int, default=256)
    ap.add_argument("--scale-rank", type=int, default=SCALE_RANK)
    ap.add_argument("--scale-hidden", type=int, default=SCALE_HIDDEN)
    ap.add_argument("--scale-experts", type=int, default=SCALE_EXPERTS)
    ap.add_argument("--scale-topk", type=int, default=SCALE_TOPK)
    ap.add_argument("--team-policy-arms")
    ap.add_argument("--arm-checkpoint", action="append", default=[])
    ap.add_argument("--off-policy-players", type=int, default=0)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--gradient-clip", type=float, default=1.0)
    ap.add_argument("--checkpoint")
    ap.add_argument("--environment", default="game2_server")
    ap.add_argument("--navigation-realization")
    ap.add_argument("--model-sample-every", type=int, default=50)
    ap.add_argument("--measure-rows", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=20260829)
    ap.add_argument("--control-weight", type=float, default=0.5)
    ap.add_argument("--exploration-weight", type=float, default=0.05)
    ap.add_argument("--replay-capacity", type=int, default=1024,
                    help="optional transition-count bound; zero leaves retention to the memory budget")
    ap.add_argument("--replay-precision", default="float32",
                    help="storage precision label recorded with the replay run")
    ap.add_argument("--replay-memory-mb", type=float, default=256.0,
                    help="byte budget for each retained-history and current-episode pool")
    ap.add_argument("--replay-batch", type=int, default=8,
                    help="historical value examples mixed into each shared fresh-state update")
    ap.add_argument("--replay-weight", type=float, default=0.5)
    ap.add_argument("--replay-steps", type=int, help="deprecated; both policies take one step per fresh attributed group")
    args = ap.parse_args()
    from solver.strat.online import OnlineLearner, OPTIMIZER_WEIGHT_DECAY
    packet = json.loads(sys.stdin.readline()) if args.state_stdin and int(os.environ.get('RANK', '0')) == 0 else {}
    if args.placement_json:
        placement = json.loads(args.placement_json)
    elif 'placement' in packet:
        placement = packet['placement']
    else:
        with open(args.placement) as source:
            placement = json.load(source)
    device = placement["device"]
    lead = float(placement.get("anticipatory_lead", 0.0))
    interval, relaxation = float(placement["strategy_interval"]), float(placement["relaxation_time"])
    world = World(device, placement.get("products"))
    state_store = world.dist.distributed_c10d._get_default_store() if world.dist is not None else None
    application = json.loads(os.environ.get('MESH_APPLICATION_IDENTITY', '{"state":"unbundled"}'))
    print(json.dumps({'event': 'application_runtime', 'application': application}), flush=True)

    stopping = {"signal": None}
    wake, woken = os.pipe()
    os.set_blocking(woken, False)
    signal.set_wakeup_fd(woken)
    def stop(signum, frame):
        stopping["signal"] = signum
        print(json.dumps({"event": "responder_stop", "signal": signum}), flush=True)
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, stop)
    retired = {key: getattr(args, key) for key in
               ("control_weight", "exploration_weight", "off_policy_players", "replay_steps")}
    print(json.dumps({"event": "state_rate_interface", "retired_candidate_options": retired,
                      "relaxation_time": relaxation,
                      "integration_interval": interval}), flush=True)
    assigned = tuple(filter(None, (args.team_policy_arms or args.policy_arm).split(',')))
    train_arms = tuple(filter(None, (args.train_arms or args.policy_arm).split(','))) if args.train else ()
    evaluation_arms = tuple(dict.fromkeys((*assigned, *train_arms)))
    checkpoint_sources = {}
    for value in args.arm_checkpoint:
        arm, separator, path = value.partition('=')
        checkpoint_sources[arm if separator else args.policy_arm] = path if separator else value
    states = {arm: decode_payload(value) for arm, value in packet.get('states', {}).items()}
    shards = {rank: {arm: decode_payload(value) for arm, value in values.items()}
              for rank, values in packet.get('shards', {}).items()}
    for arm in evaluation_arms:
        path = checkpoint_sources.get(arm) or args.checkpoint
        if arm not in states and path:
            with open_checkpoint(path) as source:
                states[arm] = Payload({name: source[name] for name in source.files})
            for rank in range(1, world.size):
                target = f'{path}.rank{rank}.npz'
                if os.path.isfile(target):
                    with np.load(target, allow_pickle=False) as source:
                        shards.setdefault(str(rank), {})[arm] = Payload({name: source[name] for name in source.files})
    navigation_data = packet.get('navigation_realization')
    if navigation_data is None and args.navigation_realization:
        with open(args.navigation_realization) as source:
            navigation_data = json.load(source)
    navigation = NavigationRows(navigation_data) if navigation_data is not None else None
    initial_states = {arm: decode_payload(value) for arm, value in packet.get('initial_states', {}).items()}
    def realize(width, schema_id):
        made_models, made_remotes = {}, {}
        widths = strategy_widths(width, args.scale_rank, args.scale_hidden,
                                 args.scale_experts, args.scale_topk)
        for arm in evaluation_arms:
            seed = parameter_seed(args.seed, architecture_arm(arm))
            torch.manual_seed(seed)
            model, forward = make_policy(arm, widths, args.baseline_hidden, world)
            if model is not None:
                model.parameter_seed = seed
                model.state_schema = str(schema_id)
                made_remotes.update({f'{arm}.{name}': remote for name, remote in place(model, arm, world).items()})
                model.to(device)
            made_models[arm] = (model, forward, seed)
        return made_models, made_remotes

    if world.rank:
        initial_shards = {}
        def initialize_member(width, schema_id, member_shards):
            made, _ = realize(width, tuple(schema_id))
            out = {}
            for arm, (model, forward, seed) in made.items():
                program = PolicyProgram(model, forward, world=world, arm=arm, device=device, lead=lead)
                if arm in train_arms:
                    program.train(args.learning_rate, OPTIMIZER_WEIGHT_DECAY, args.gradient_clip, arm)
                payload = member_shards.get(str(world.rank), {}).get(arm)
                if payload is not None:
                    program.restore_shard(payload)
                initial_shards.setdefault(arm, {name: np.array(value, copy=True)
                                               for name, value in program.export_shard().items()})
                out[arm] = program
            return out
        print(json.dumps({'event': 'policy_member', **world.report()}), flush=True)
        try:
            programs = serve(world, initialize_member)
            state_store.set(f'mesh/strat/completed/{world.rank}', json.dumps({
                'initial_shards': {arm: encode_payload(state) for arm, state in initial_shards.items()},
                'shards': {arm: encode_payload(program.export_shard()) for arm, program in programs.items()}}))
        finally:
            world.close()
        return
    engine = Engine(placement["socket"], placement["framebytes"])
    route = {}
    inbox = RuntimeFrames(placement["framebytes"])
    models, learners, provenance, programs = {}, {}, {}, {}
    schema = None
    key = np.asarray([(args.seed >> 32) & 0xffffffff, args.seed & 0xffffffff], dtype=np.uint32)
    sampling_key = np.empty_like(key)
    history = ActionHistory(args.measure_rows, args.replay_memory_mb * (1 << 20), interval)
    episode_key = None
    observation_memory = ObservationMemory()
    remotes = {}
    count = 0
    started = time.monotonic()
    measurements = []
    meter = WorkloadMeter('xonotic.strategy', {'environment': args.environment,
                          'policy_arm': 'mixed' if len(assigned) > 1 else args.policy_arm,
                          'host_role': 'responder', 'host': socket.gethostname()})
    reporter = LiteralJReporter(args.measure_rows).start()
    producer_id = uuid.uuid4().hex
    acknowledged = set()
    def outcomes():
        for identity, event in inbox.outcomes.items():
            if identity in acknowledged:
                continue
            learning = {arm: learner.outcomes[identity] for arm, learner in learners.items()
                        if identity in learner.outcomes}
            if not learners or len(learning) == len(learners):
                acknowledged.add(identity)
                print(json.dumps({'event': 'outcome_acknowledged', 'episode_id': identity,
                                  'outcome': event, 'learning_outcomes': learning, 'responses': count}), flush=True)
    print(json.dumps({'event': 'policy_navigation', 'navigation': navigation.report if navigation else
        {'source': None, 'event_support': 'native_team_observations', 'spatial_geometry': 'unavailable'}}), flush=True)
    def initialize(width, schema_id):
        nonlocal models, learners, provenance, remotes, programs
        models, learners, provenance, programs = {}, {}, {}, {}
        world.announce('initialize', width, schema_id, shards)
        made, remotes = realize(width, schema_id)
        for arm in evaluation_arms:
            model, forward, seed = made[arm]
            payload = states.get(arm)
            source = 'operand' if payload is not None else None
            if arm in train_arms:
                learning = OnlineLearner(
                    model, learning_rate=args.learning_rate, gradient_clip=args.gradient_clip,
                    state=payload,
                    replay_capacity=args.replay_capacity, replay_memory_mb=args.replay_memory_mb,
                    replay_precision=args.replay_precision, replay_batch=args.replay_batch,
                    seed=seed, policy_forward=forward, policy_arm=arm,
                    match_metadata=json.loads(args.match_metadata), replay_weight=args.replay_weight,
                    world=world, device=device, lead=lead,
                )
                learners[arm] = learning
                initial_states.setdefault(arm, {name: np.array(value, copy=True)
                                               for name, value in learning.export_payload().items()})
            elif model is not None and payload is not None:
                load_policy(model, payload, arm)
            models[arm] = model
            programs[arm] = learners[arm].program if arm in learners else PolicyProgram(
                model, forward, world=world, arm=arm, device=device, lead=lead)
            if model is not None:
                programs[arm].synchronize()
            provenance[arm] = policy_source(arm, model, source, 'online_train' if arm in learners else 'inference')
        history.clear()
        print(json.dumps({"event": "state_schema_realized", "state_width": width,
                          "state_schema": schema_id, "policies": list(models)}), flush=True)
    try:
        while stopping['signal'] is None:
            readable, _, _ = select.select([engine, wake], [], [])
            if wake in readable:
                os.read(wake, 64)
                continue
            for node, address, buffer in engine.receive():
                inbox.feed(buffer)
                route['engine'] = (node, address)
            ready = inbox.take()
            if ready is None:
                for learning in learners.values():
                    for identity in tuple(learning.awaiting_outcomes):
                        if identity in inbox.outcomes and identity not in learning.outcomes:
                            learning.end_episode(inbox.outcomes[identity]['actor_team'], identity)
                outcomes()
                continue
            (session, tick_id), records, event_batches = ready
            obs_rows = records[OBSERVATION_KIND][1]
            cart_rows = records[CART_KIND][1]
            team_rows = records[TEAM_KIND][1]
            current_episode = ObservationMemory.identity(session, team_rows[0, TS['EPISODE']])
            if observation_memory.observe(current_episode, event_batches):
                for learning in learners.values():
                    learning.switch_episode(current_episode)
                history.clear()
                reporter.reset()
                episode_key = current_episode
            event_rows = observation_memory.rows
            view_pages, chorus = native_inputs(records[STATE_KIND][1], obs_rows, cart_rows, team_rows,
                event_rows, interval, relaxation, navigation)
            state_header = view_pages.headers
            state_rows, residual_rows = chorus.state, chorus.residual
            if not len(state_rows):
                continue
            participant_ids = state_header[:, STATE_HEADER['ID']].astype(np.int64)
            rows = obs_rows[chorus.participant_rows]
            current_schema = (PAGE_WIDTH, tuple(np.unique(state_header[:, STATE_HEADER['SCHEMA']]).astype(np.int64).tolist()))
            if current_schema != schema:
                initialize(*current_schema)
                schema = current_schema
                for learning in learners.values():
                    learning.switch_episode(current_episode)
            k = len(team_rows)
            tick = build_runtime_frame(rows, cart_rows, team_rows)
            for learning in learners.values():
                for identity in tuple(learning.awaiting_outcomes):
                    if identity in inbox.outcomes:
                        learning.end_episode(inbox.outcomes[identity]['actor_team'], identity)
            teams = chorus.team_ids
            frame = Frame.capture(count, chorus)
            applied_sequences = state_header[:, STATE_HEADER['APPLIED_SEQUENCE']].astype(np.int64)
            realized_events = [{"kind": {5: 'damage', 6: 'kill', 7: 'pickup', 8: 'score_win' if row[EVT['TEAM']] > 0 else 'tie'}.get(int(row[EVT['KIND']]), str(int(row[EVT['KIND']]))),
                                "episode_id": owner,
                                "actor": int(row[EVT['OBSERVER']]), "actor_team": int(row[EVT['TEAM']]),
                                "subject": int(row[EVT['SUBJECT']]), "value": float(row[EVT['AMOUNT']]),
                                "time": float(row[EVT['TIME']]), "response_seq": int(row[EVT['RESPONSE_SEQ']])}
                               for owner, arrived in observation_memory.arrivals.items() for row in arrived if row[EVT['KIND']] >= 5]
            outcome = inbox.outcomes.get(current_episode)
            terminal = outcome['actor_team'] if outcome is not None and tick.cartstate.finished else None
            if terminal is not None:
                realized_events = [event for event in realized_events if event['episode_id'] != current_episode
                                   or event['kind'] not in ('score_win', 'tie')] + [outcome]
            active_learners = {arm: learning for arm, learning in learners.items() if current_episode not in learning.outcomes}
            attributed = history.advance(tick, frame, participant_ids, rows, applied_sequences, active_learners, terminal)
            versions = {arm: learning.updates for arm, learning in learners.items()}
            versions.update({arm: provenance[arm].get('updates') for arm in models if arm not in versions})
            arms = np.asarray([assigned[(int(team) - 1) % len(assigned)] for team in teams])
            advance_key(key, sampling_key)
            response_started = time.perf_counter()
            prepared = prepare_policies(programs, chorus)
            emission = emit_policies(programs, prepared, chorus, arms, sampling_key)
            outputs, comparisons = emission.outputs, emission.comparisons
            velocity, behavior_logp, integrated = emission.velocity, emission.logp, emission.residual
            preparation_elapsed, inference_elapsed = emission.preparation_seconds, emission.inference_seconds
            request = records[STATE_KIND][0]['req_id']
            response = view_pages.response(velocity, interval, relaxation, lead)
            if 'engine' in route:
                node, address = route['engine']
                engine.send(address, node, frames(STRATEGY_KIND, request, tick_id, response, placement["framebytes"], session))
            response_elapsed = time.perf_counter() - response_started
            reporting_started = time.perf_counter()
            assignments, decisions = history.assignments(rows, applied_sequences, request, velocity, integrated,
                                                         behavior_logp, arms, versions, provenance)
            labels = [f'page_word.{index}' for index in range(state_rows.shape[1])]
            participant_labels = [state_labels(layout) for layout in view_pages.layout]
            model_output = outputs[args.policy_arm] if args.policy_arm in outputs else next(iter(outputs.values()))
            ir = np.asarray(model_output.ir)
            input_features, input_labels = source_features(chorus)
            model_record = {'x': state_rows, 'x_labels': labels, 'server_state': state_rows,
                            'source_features': input_features, 'source_labels': input_labels,
                            'j': ir, 'j_labels': [f'ir.{index}' for index in range(ir.shape[1])],
                            'coupling': np.asarray(model_output.coupling),
                            'local_neighborhood': np.asarray(model_output.local_neighborhood), 'hierarchy': tick.semantics,
                            'input_residual': residual_rows, 'state_layout': view_pages.layout,
                            'event_rows': event_rows,
                            'observation_rows': obs_rows, 'cart_rows': cart_rows, 'team_rows': team_rows,
                            'navigation': navigation.report if navigation else {'spatial_geometry': 'unavailable'},
                            'coupling_representation': 'per_row_team_and_rival_gram_factors',
                            'execution': {arm: program.report() for arm, program in programs.items()},
                            'moe': {arm: {'residual_stats': np.asarray(output.scale_residual_stats).tolist(),
                                         'expert_load': np.asarray(output.scale_expert_load).tolist(),
                                         'balance': float(output.scale_balance)} for arm, output in outputs.items()},
                            'team_ids': teams, 'delta': interval,
                            'relaxation_time': relaxation, 'rate': velocity, 'residual': integrated}
            for index, item in enumerate(assignments):
                observed_state, observed_labels = view_observation(chorus, index)
                item['successor_state'] = observed_state.tolist()
                item['successor_state_labels'] = observed_labels
                own_ir = np.asarray(outputs[str(arms[index])].ir)[index]
                pages = view_pages.layout.shape[1]
                latent_width = len(own_ir) // (1 + pages)
                own_labels = latent_labels(view_pages.layout[index], latent_width, pages)
                latent_present = np.flatnonzero(np.asarray(own_labels, dtype=bool))
                own_ir = own_ir[latent_present]
                own_labels = [f'{arms[index]}.{own_labels[coordinate]}' for coordinate in latent_present]
                decisions[item['edict']].update(j=own_ir.copy(), j_labels=own_labels, episode_id=current_episode,
                                               features=input_features, feature_labels=input_labels,
                                               server_state=observed_state, state_labels=observed_labels)
            sources = [{"response_seq": sequence, "edict": participant, **decision,
                        "action": {"velocity": decision["velocity"], "residual": decision["residual"]}}
                       for sequence, saved in (*history.items(), (request, decisions))
                       for participant, decision in saved.items()]
            issued = dict(request_seq=request, participant_ids=participant_ids, context=tick.context,
                          snapshot=tick.cartstate, frame=frame, engine_time=float(np.median(rows[:, OBS['ENGINE_TIME']])),
                          velocity=velocity.copy(), behavior_logp=behavior_logp.copy(), behavior_arms=arms.copy(),
                          policy_updates=versions)
            history.put(request, decisions, set(applied_sequences), state=issued)
            source_reporting_elapsed = time.perf_counter() - reporting_started
            optimization_started = time.perf_counter()
            learning = {arm: value for arm, learner in learners.items()
                        if (value := learner.observe_attributed(attributed.get(arm, []))) is not None}
            if terminal is not None:
                for learner in learners.values():
                    learner.end_episode(winning_team=terminal, episode_id=current_episode)
            optimization_elapsed = time.perf_counter() - optimization_started
            learning_state = {arm: {'updates': learner.updates,
                              'schedule_updates': learner.updates - learner.schedule_start_updates,
                              'training_contract': learner.training_contract,
                              'pending_states': len(learner.episode), 'replay_size': len(learner.replay),
                              'awaiting_outcomes': len(learner.awaiting_outcomes), 'episode_id': current_episode,
                              'outcome': learner.outcomes.get(current_episode), 'execution': history.measure(),
                              'transport': inbox.report(),
                              'view_faults': int(rows[:, OBS['VIEW_FAULTS']].sum()),
                              'snapshots_superseded': int(rows[:, OBS['SNAPSHOTS_SUPERSEDED']].max()),
                              'completed_episodes': learner.completed_episodes,
                              'truncated_episodes': learner.truncated_episodes,
                              'reward_contract': reward_contract(arm), **learner.replay.report()}
                              for arm, learner in learners.items()}
            count += 1
            sampled = args.model_sample_every > 0 and count % args.model_sample_every == 0
            comparison_started = time.perf_counter()
            comparison = compare_policies(comparisons, assignments, versions, sampled)
            comparison.update(sampled_at=time.time(), response=count,
                              producer_pid=os.getpid(), elapsed_s=time.perf_counter() - comparison_started)
            line = {'at': time.time(), 'resp_id': count, 'request_seq': request,
                    'producer_id': producer_id, 'record_id': f'{producer_id}:{count}',
                    'engine_session': session, 'engine_episode': int(team_rows[0, TS['EPISODE']]),
                    'match_metadata': json.loads(args.match_metadata),
                    'application': application,
                    'episode_id': current_episode,
                    'interface': 'full_state_rate', 'state_width': state_rows.shape[1], 'state_schema': schema,
                    'observation_memory': observation_memory.report(),
                    'event_arrivals': [{'session': key[0], 'tick': key[1], 'request_seq': key[2], 'rows': rows}
                                       for key, rows in event_batches.items()],
                    'environment': args.environment, 'teams': k, 'players': len(rows), 'j': len(cart_rows),
                    'k': k, 'l': len(rows), 't': time.monotonic() - started,
                    'recorded_at': time.time(), 'producer_pid': os.getpid(),
                    'policy_arm': 'mixed' if len(assigned) > 1 else args.policy_arm,
                    'team_policy_arms': assigned, 'policy_provenance': provenance,
                    'mode': 'online_train' if learners else 'inference',
                    'updates': sum(learner.updates for learner in learners.values()),
                    'resources': team_resources(rows, k),
                    'carts': [{**{name.lower(): float(row[column]) for name, column in CS.items()},
                               'depth': float(tick.cartstate.pos[index])} for index, row in enumerate(cart_rows)],
                    'assignments': assignments, 'realized_events': realized_events, 'learning': learning_state,
                    'policy_updates': learning,
                    'game_value': formal_value_record(tick.game_value, tick.cartstate),
                    'policy_comparison': comparison,
                    'server_state_labels': labels if sampled else [], 'participant_state_labels': participant_labels if sampled else [], 'source_window': history.measure(),
                    'model': model_record, 'measure_sources': sources, 'moe': model_record['moe'],
                    'distributed_scale': {name: remote.last for name, remote in remotes.items()},
                    'world': world.report(), 'transport_local': engine.report()}
            line['execution'] = {arm: program.report() for arm, program in programs.items()}
            model_record['row_outputs'] = [{'row': index, 'arm': str(arms[index]),
                                           'j': decisions[int(participant)]['j'].tolist()}
                                          for index, participant in enumerate(participant_ids)]
            reporter.ingest(dict(line))
            _, line['j_measures'] = reporter.snapshot()
            numeric = lambda value: np.asarray(value).tolist()
            line['model'] = {'sampled': sampled, 'shapes': {name: list(np.shape(value))
                             for name, value in model_record.items() if isinstance(value, np.ndarray)},
                             **({name: numeric(value) if isinstance(value, np.ndarray) else value
                                 for name, value in model_record.items()} if sampled else {})}
            line.pop('measure_sources')
            response_work = [strategy_work(arm, models[arm].w if is_matrix_fusion_arm(arm) else
                             strategy_widths(PAGE_WIDTH, args.scale_rank, args.scale_hidden,
                                             args.scale_experts, args.scale_topk),
                             programs[arm].capacity[0], programs[arm].capacity[2], args.baseline_hidden,
                             remote_scale_operation='block' if world.parts('scale') else 'gram',
                             state_pages=programs[arm].capacity[1], observations=programs[arm].capacity[3],
                             carts=programs[arm].capacity[4], teams=programs[arm].capacity[5],
                             navigation_nodes=programs[arm].capacity[6], navigation_edges=programs[arm].capacity[7],
                             navigation_cells=programs[arm].capacity[8], neighbors=programs[arm].capacity[9])
                             for arm in models]
            totals = {name: sum(part[name] for part in response_work)
                      for name in ('lower_flops', 'upper_flops', 'lower_bytes', 'upper_bytes')}
            line['work'] = {**totals, 'elapsed_s': response_elapsed,
                            'preparation_elapsed_s': preparation_elapsed, 'inference_elapsed_s': inference_elapsed,
                            'source_reporting_elapsed_s': source_reporting_elapsed,
                            'strategy_interval_s': interval,
                            'optimization': {'elapsed_s': optimization_elapsed,
                                             'gradient_steps': sum(value['gradient_steps'] for value in learning.values())}}
            meter.record(response_elapsed,
                         flops_lower=totals['lower_flops'], flops_upper=totals['upper_flops'],
                         bytes_lower=totals['lower_bytes'], bytes_upper=totals['upper_bytes'],
                         rows=len(rows),
                         operations={'stage': 'state_rate_response', 'players': len(rows), 'teams': k,
                                     'state_width': state_rows.shape[1], 'host_role': 'responder'},
                         measures={**line['j_measures'], 'moe': model_record['moe'], **{f'learning.{arm}': {**state, 'last_update': learning.get(arm, {})}
                                   for arm, state in learning_state.items()}})
            measurements.append(line)
            outcomes()
    finally:
        reporter.stop()
        meter.close(reporter.snapshot()[1])
        world.stop()
        member_states = {str(rank): json.loads(state_store.get(f'mesh/strat/completed/{rank}'))
                         for rank in range(1, world.size)}
        world.close()
        engine.close()
    final_states = {arm: encode_payload(learning.export_payload()) for arm, learning in learners.items()}
    final_states.update({arm: encode_payload(states[arm]) for arm in models if arm not in learners and arm in states})
    result = {'event': 'responder_finished', 'states': final_states,
              'shards': {rank: value['shards'] for rank, value in member_states.items()},
              'initial_shards': {rank: value['initial_shards'] for rank, value in member_states.items()},
              'initial_states': {arm: encode_payload(state) for arm, state in initial_states.items()},
              'measurements': measurements,
              'j_measures': reporter.snapshot()[1]}
    print(json.dumps(result, default=lambda value: np.asarray(value).tolist()), flush=True)
    return result


if __name__ == '__main__':
    main()
