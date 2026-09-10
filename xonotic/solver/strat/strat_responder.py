import argparse, json, os, signal, socket, sys, time, uuid
from collections import deque
from itertools import chain

import numpy as np
import mlx.core as mx

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "..", "rdma"))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))

from solver.strat.cast_header import Wally, parameter_seed
from solver.strat.scale_rpc import RemoteCross, RemoteScale
from solver.strat.baselines import BaselinePolicy, baseline_strategy, default_strategy
from solver.strat.inputs import native_inputs
from solver.strat.neighborhood import NavigationRows
from solver.strat.execution import PolicyProgram, prepare_policies, emit_policies
from solver.strat.action_history import ActionHistory, source_features, view_observation
from solver.strat.buffers import ObservationMemory
from solver.strat.replay import Frame
from solver.strat.checkpoint_state import Payload, atomic_save, pack_state, unpack_state, load_policy, policy_source
from solver.strat.journal import TrainingJournal
from solver.strat.strategy import strategy
from solver.strat.state_steering import PAGE_WIDTH, STATE_HEADER, state_labels, latent_labels, advance_key
from solver.strat.policy_reports import compare_policies, publish_active_run
from solver.xonwire import (
    FrameStream, RuntimeFrames,
    CART_KIND, OBSERVATION_KIND, STRATEGY_KIND, TEAM_KIND, STATE_KIND,
)
from solver.strat.joracle.probe import LiteralJReporter
from solver.strat.policy_contract import architecture_arm, is_matrix_fusion_arm, checkpoint_path
from solver.strat.runtime import build_runtime_frame
from solver.strat.work_estimate import strategy_work
from solver.strat.game_value import formal_value_record, reward_contract
from solver.strat.scale_config import (
    SCALE_EXPERTS, SCALE_HIDDEN, SCALE_RANK, SCALE_TOPK,
    strategy_widths,
)
from workload import WorkloadMeter
from mesh import Mesh

from payload.tools.strategy_io_schema import (
    STRATEGY_DEADLINE_S,
    OBS,
    CS,
    TS,
    EVT,
    OBS_AMMO_COLUMNS,
    OBS_WEAPON_COLUMNS,
)

CKPT = os.path.join(_HERE, "runs", "policy_ckpt_v6.npz")
ONLINE_CKPT = os.path.join(_HERE, "runs", "policy_online_v6.npz")
TELEM = os.path.join(_HERE, "runs", "cartserver_telemetry.jsonl")
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

def make_policy(arm, widths, hidden):
    if is_matrix_fusion_arm(arm):
        model = Wally(widths)
        model.participant_fusion_scale = float(arm != 'participant_fusion_ablated')
        model.residual_fusion_scale = float(arm != 'residual_fusion_ablated')
        return model, strategy
    if arm == 'default':
        return None, default_strategy
    return BaselinePolicy(arm, widths.d_x, widths.d_c, hidden), baseline_strategy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peer-node", type=int, default=0)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--train-arms")
    ap.add_argument("--match-metadata", default="{}")
    ap.add_argument("--policy-arm", default="matrix_fusion")
    ap.add_argument("--baseline-hidden", type=int, default=256)
    ap.add_argument("--scale-rank", type=int, default=SCALE_RANK)
    ap.add_argument("--scale-hidden", type=int, default=SCALE_HIDDEN)
    ap.add_argument("--scale-experts", type=int, default=SCALE_EXPERTS)
    ap.add_argument("--scale-topk", type=int, default=SCALE_TOPK)
    ap.add_argument("--distributed-scale", action="store_true")
    ap.add_argument("--distributed-scale-operation", choices=("block", "gram"), default="gram")
    ap.add_argument("--team-policy-arms")
    ap.add_argument("--arm-checkpoint", action="append", default=[])
    ap.add_argument("--off-policy-players", type=int, default=0)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--gradient-clip", type=float, default=1.0)
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--save-secs", type=float, default=30.0)
    ap.add_argument("--checkpoint")
    ap.add_argument("--online-checkpoint", default=ONLINE_CKPT)
    ap.add_argument("--initial-checkpoint")
    ap.add_argument("--resume-checkpoint")
    ap.add_argument("--telemetry", default=TELEM)
    ap.add_argument("--environment", default="game2_server")
    ap.add_argument("--navigation-realization")
    ap.add_argument("--append-telemetry", action="store_true")
    ap.add_argument("--model-sample-every", type=int, default=50)
    ap.add_argument("--measure-rows", type=int, default=4000)
    ap.add_argument("--measure-interval", type=float, default=4.0)
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
    ap.add_argument("--relaxation-time", type=float, default=1.0)
    ap.add_argument("--integration-interval", type=float, default=STRATEGY_DEADLINE_S)
    args = ap.parse_args()
    from solver.strat.online import OnlineLearner
    application = json.loads(os.environ.get('MESH_APPLICATION_IDENTITY', '{"state":"unbundled"}'))
    print(json.dumps({'event': 'application_runtime', 'application': application}), flush=True)

    stopping = {"signal": None}
    def stop(signum, frame):
        stopping["signal"] = signum
        print(json.dumps({"event": "responder_stop", "signal": signum}), flush=True)
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, stop)
    retired = {key: getattr(args, key) for key in
               ("control_weight", "exploration_weight", "off_policy_players", "replay_steps")}
    print(json.dumps({"event": "state_rate_interface", "retired_candidate_options": retired,
                      "relaxation_time": args.relaxation_time,
                      "integration_interval": args.integration_interval}), flush=True)
    assigned = tuple(filter(None, (args.team_policy_arms or args.policy_arm).split(',')))
    train_arms = tuple(filter(None, (args.train_arms or args.policy_arm).split(','))) if args.train else ()
    evaluation_arms = tuple(dict.fromkeys((*assigned, *train_arms)))
    checkpoint_sources = {}
    for value in args.arm_checkpoint:
        arm, separator, path = value.partition('=')
        checkpoint_sources[arm if separator else args.policy_arm] = path if separator else value
    mesh = Mesh()
    tx = FrameStream(mesh)
    backlog = deque()
    inbox = RuntimeFrames(mesh.usable)
    models, learners, provenance, programs = {}, {}, {}, {}
    schema = None
    key = np.asarray(mx.random.key(args.seed)).copy()
    sampling_key = np.empty_like(key)
    history = ActionHistory(args.measure_rows, args.replay_memory_mb * (1 << 20),
                            os.path.abspath(args.online_checkpoint or CKPT) + '.actions')
    episode_key = None
    observation_memory = ObservationMemory()
    remotes = {}
    count = 0
    started = time.monotonic()
    last_status = started
    runstate_path = os.path.abspath(args.online_checkpoint or CKPT) + '.runstate.npz'
    saved, saved_payload = None, None
    if args.append_telemetry and os.path.isfile(runstate_path):
        try:
            with np.load(runstate_path, allow_pickle=False) as source:
                saved_payload = Payload({name: np.asarray(source[name]) for name in source.files})
            saved = unpack_state(saved_payload)
        except (OSError, ValueError, KeyError) as error:
            print(json.dumps({'event': 'runstate_recovery', 'error': str(error)}), flush=True)
    journal = TrainingJournal(saved['journal_path'] if saved else runstate_path + '.steps.' + str(time.time_ns()))
    journal_offset = int(saved['journal_offset']) if saved else 0
    recovery = deque(journal.read(journal_offset))
    journal.repair_tail(recovery[-1][0] if recovery else journal_offset)
    step_complete = True
    meter = WorkloadMeter('xonotic.strategy', {'environment': args.environment,
                          'policy_arm': 'mixed' if len(assigned) > 1 else args.policy_arm,
                          'host_role': 'responder', 'host': socket.gethostname()})
    last_save = time.monotonic()
    telemetry_path = os.path.abspath(args.telemetry)
    os.makedirs(os.path.dirname(telemetry_path), exist_ok=True)
    reporter = LiteralJReporter(args.measure_rows, args.measure_interval,
                               os.path.join(os.path.dirname(telemetry_path), "j-measures." + os.path.basename(telemetry_path))).start()
    published = False
    telemetry = open(telemetry_path, 'a')
    producer_id = uuid.uuid4().hex
    def save():
        nonlocal journal, journal_offset
        if schema is None:
            return
        previous_journal = journal
        next_journal = journal if recovery else TrainingJournal(runstate_path + '.steps.' + str(time.time_ns()))
        with open(next_journal.path, 'ab') as target:
            target.flush()
            os.fsync(target.fileno())
        state = {'model_key': key, 'resp_id': count, 'schema': schema, 'episode_key': episode_key,
                 'history': history.export_state(),
                 'observation_memory': observation_memory.export_state(),
                 'transport': inbox.export_state(), 'backlog': list(backlog),
                 'journal_path': next_journal.path, 'journal_offset': journal_offset if recovery else 0,
                 'assigned': assigned, 'train_arms': train_arms, 'at': time.time()}
        payload = pack_state(state)
        for arm, learning in learners.items():
            payload.update({f'__learner_{arm}__/{name}': value for name, value in learning.export_payload().items()})
        atomic_save(runstate_path, payload)
        history.collect_spills()
        journal = next_journal
        journal_offset = state['journal_offset']
        if previous_journal.path != journal.path and os.path.isfile(previous_journal.path):
            os.unlink(previous_journal.path)
        for learning in learners.values():
            learning.save()
        acknowledgements = {arm: learning.outcomes[episode_key] for arm, learning in learners.items() if episode_key in learning.outcomes}
        if acknowledgements:
            outcome_path = os.path.join(os.path.dirname(telemetry_path), 'outcome.json')
            with open(outcome_path + '.new', 'w') as target:
                json.dump({'episode_id': episode_key, 'event': inbox.outcomes[episode_key],
                           'learning_outcomes': acknowledgements, 'checkpoint': runstate_path, 'responses': count}, target)
                target.flush()
                os.fsync(target.fileno())
            os.replace(outcome_path + '.new', outcome_path)
    navigation = None
    if args.navigation_realization:
        with open(args.navigation_realization) as source:
            navigation = NavigationRows(json.load(source))
    print(json.dumps({'event': 'policy_navigation', 'navigation': navigation.report if navigation else
        {'source': None, 'event_support': 'native_team_observations', 'spatial_geometry': 'unavailable'}}), flush=True)
    def initialize(width, schema_id):
        nonlocal models, learners, provenance, remotes, programs
        models, learners, provenance, programs = {}, {}, {}, {}
        remotes = {}
        widths = strategy_widths(width, args.scale_rank, args.scale_hidden,
                                 args.scale_experts, args.scale_topk)
        for arm in evaluation_arms:
            seed = parameter_seed(args.seed, architecture_arm(arm))
            mx.random.seed(seed)
            model, forward = make_policy(arm, widths, args.baseline_hidden)
            source = checkpoint_sources.get(arm) or args.resume_checkpoint or args.checkpoint
            if model is not None:
                model.parameter_seed = seed
                model.state_schema = str(schema_id)
                if args.distributed_scale and is_matrix_fusion_arm(arm):
                    remote_type = RemoteScale if args.distributed_scale_operation == 'block' else RemoteCross
                    remote = remote_type(mesh, args.peer_node, backlog, stopping=stopping,
                        **({'owner': arm} if remote_type is RemoteScale else {}))
                    remotes[arm] = remote
                    setattr(model, 'scale_executor' if isinstance(remote, RemoteScale) else 'cross_executor', remote)
            if arm in train_arms:
                target = checkpoint_path(args.online_checkpoint, arm) if len(train_arms) > 1 else args.online_checkpoint
                learning = OnlineLearner(
                    model, learning_rate=args.learning_rate, gradient_clip=args.gradient_clip,
                    checkpoint=target, load_checkpoint=source or target,
                    replay_capacity=args.replay_capacity, replay_memory_mb=args.replay_memory_mb,
                    replay_precision=args.replay_precision, replay_batch=args.replay_batch,
                    seed=seed, policy_forward=forward, policy_arm=arm,
                    match_metadata=json.loads(args.match_metadata), replay_weight=args.replay_weight,
                )
                learners[arm] = learning
                prefix = f'__learner_{arm}__/'
                if saved_payload is not None and any(name.startswith(prefix) for name in saved_payload):
                    learning._load_full(runstate_path, {name[len(prefix):]: value for name, value in saved_payload.items() if name.startswith(prefix)})
                if args.initial_checkpoint:
                    learning.save(checkpoint_path(args.initial_checkpoint, arm) if len(train_arms) > 1 else args.initial_checkpoint)
            elif model is not None and source:
                load_policy(model, source, arm)
            models[arm] = model
            programs[arm] = learners[arm].program if arm in learners else PolicyProgram(model, forward)
            provenance[arm] = policy_source(arm, model, source, 'online_train' if arm in learners else 'inference')
        history.clear()
        print(json.dumps({"event": "state_schema_realized", "state_width": width,
                          "state_schema": schema_id, "policies": list(models)}), flush=True)
    if saved is not None:
        initialize(*saved['schema'])
        schema = saved['schema']
        if saved['assigned'] == assigned and saved['train_arms'] == train_arms and all(learning.continuation['mode'] == 'resumed' for learning in learners.values()):
            np.copyto(key, np.asarray(saved['model_key'], dtype=np.uint32))
            count, episode_key = int(saved['resp_id']), saved['episode_key']
            history.restore_state(saved['history'])
            observation_memory, memory_recovery = ObservationMemory.restore(saved)
            inbox.restore_state(saved)
            backlog.extend(saved.get('backlog', ()))
            print(json.dumps({'event': 'training_continuation', 'mode': 'resumed', 'responses': count,
                              'history': history.measure(), 'replayed_steps': len(recovery),
                              'observation_history': memory_recovery}), flush=True)
        else:
            print(json.dumps({'event': 'training_continuation', 'mode': 'changed_configuration',
                              'unreplayed_steps': len(recovery)}), flush=True)
            recovery.clear()
        saved_payload = None
    try:
        while stopping['signal'] is None:
            replaying = bool(recovery)
            recovered = None
            if replaying:
                journal_offset, recovered = recovery.popleft()
                session, tick_id = recovered['key']
                records = recovered['records']
                event_frames = recovered['event_frames']
                event_batches = event_frames if isinstance(event_frames, dict) else {(session, tick_id, None): recovered['events']}
                inbox.replay_snapshot(recovered['key'], event_frames, recovered['outcomes'])
            else:
                incoming = list(backlog)
                backlog.clear()
                for buffer, source in chain(incoming, mesh.read(np.uint8, max_batches=1)):
                    inbox.feed(buffer)
                ready = inbox.take()
                if ready is None:
                    resolved = False
                    for learning in learners.values():
                        for identity in tuple(learning.awaiting_outcomes):
                            if identity in inbox.outcomes and identity not in learning.outcomes:
                                learning.end_episode(inbox.outcomes[identity]['actor_team'], identity)
                                resolved = True
                    if resolved:
                        save()
                    if time.monotonic() - last_status >= 5:
                        print(json.dumps({'event': 'responder_status', 'responses': count,
                                          'transport': inbox.report(), 'state_schema': schema}), flush=True)
                        last_status = time.monotonic()
                    time.sleep(0.001)
                    continue
                (session, tick_id), records, event_batches = ready
            step_complete = False
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
                event_rows, args.integration_interval, args.relaxation_time, navigation)
            state_header = view_pages.headers
            state_rows, residual_rows = chorus.state, chorus.residual
            if not len(state_rows):
                step_complete = True
                save()
                continue
            participant_ids = state_header[:, STATE_HEADER['ID']].astype(np.int64)
            rows = obs_rows[chorus.participant_rows]
            current_schema = (PAGE_WIDTH, tuple(np.unique(state_header[:, STATE_HEADER['SCHEMA']]).astype(np.int64).tolist()))
            if current_schema != schema:
                save()
                initialize(*current_schema)
                schema = current_schema
                for learning in learners.values():
                    learning.switch_episode(current_episode)
                save()
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
            terminal_new = terminal is not None and bool(active_learners)
            attributed = history.advance(tick, frame, participant_ids, rows, applied_sequences, active_learners, terminal)
            versions = {arm: learning.updates for arm, learning in learners.items()}
            versions.update({arm: provenance[arm].get('updates') for arm in models if arm not in versions})
            arms = np.asarray([assigned[(int(team) - 1) % len(assigned)] for team in teams])
            advance_key(key, sampling_key)
            response_started = time.perf_counter()
            prepared = prepare_policies(programs, chorus)
            emission = emit_policies(programs, prepared, chorus, arms, sampling_key, recorded=recovered)
            outputs, comparisons = emission.outputs, emission.comparisons
            velocity, behavior_logp, integrated = emission.velocity, emission.logp, emission.residual
            preparation_elapsed, inference_elapsed = emission.preparation_seconds, emission.inference_seconds
            request = records[STATE_KIND][0]['req_id']
            response = view_pages.response(velocity, args.integration_interval, args.relaxation_time)
            if not replaying:
                journal_offset = journal.append({'key': (session, tick_id), 'records': records, 'event_frames': event_batches,
                    'outcomes': inbox.outcomes, 'velocity': velocity, 'behavior_logp': behavior_logp})
                tx.send(STRATEGY_KIND, request, tick_id, response, args.peer_node,
                        cancel=lambda: stopping['signal'] is not None)
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
                            'team_ids': teams, 'delta': args.integration_interval,
                            'relaxation_time': args.relaxation_time, 'rate': velocity, 'residual': integrated}
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
                              'transport': inbox.report(), 'journal_bytes': journal_offset,
                              'view_faults': int(rows[:, OBS['VIEW_FAULTS']].sum()),
                              'snapshots_superseded': int(rows[:, OBS['SNAPSHOTS_SUPERSEDED']].max()),
                              'completed_episodes': learner.completed_episodes,
                              'truncated_episodes': learner.truncated_episodes,
                              'reward_contract': reward_contract(arm), **learner.replay.report()}
                              for arm, learner in learners.items()}
            count += 1
            step_complete = True
            if replaying:
                print(json.dumps({'event': 'training_step_recovered', 'responses': count, 'request_seq': request,
                                  'updates': {arm: value.updates for arm, value in learners.items()}}), flush=True)
                continue
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
                    'distributed_scale': {arm: remote.last for arm, remote in remotes.items()}}
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
                             remote_scale_operation=args.distributed_scale_operation,
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
                            'deadline_s': args.integration_interval,
                            'optimization': {'elapsed_s': optimization_elapsed,
                                             'gradient_steps': sum(value['gradient_steps'] for value in learning.values())}}
            meter.record(response_elapsed,
                         flops_lower=totals['lower_flops'], flops_upper=totals['upper_flops'],
                         bytes_lower=totals['lower_bytes'], bytes_upper=totals['upper_bytes'],
                         deadline_s=args.integration_interval, rows=len(rows),
                         operations={'stage': 'state_rate_response', 'players': len(rows), 'teams': k,
                                     'state_width': state_rows.shape[1], 'host_role': 'responder'},
                         measures={**line['j_measures'], 'moe': model_record['moe'], **{f'learning.{arm}': {**state, 'last_update': learning.get(arm, {})}
                                   for arm, state in learning_state.items()}})
            learning_path = os.path.join(os.path.dirname(telemetry_path), 'learning.json')
            with open(learning_path + '.new', 'w') as target:
                json.dump({'at': time.time(), 'environment': args.environment, 'teams': k,
                           'players': len(rows), 'responses': count, 'learning': learning_state,
                           'policy_updates': learning}, target)
            os.replace(learning_path + '.new', learning_path)
            telemetry.write(json.dumps(line, default=lambda value: np.asarray(value).tolist()) + '\n')
            telemetry.flush()
            if not published:
                publish_active_run(telemetry_path)
                published = True
            if terminal_new or time.monotonic() - last_save >= args.save_secs or (args.save_every and count % args.save_every == 0):
                save()
                last_save = time.monotonic()
    finally:
        if step_complete:
            save()
        reporter.stop()
        meter.close(reporter.snapshot()[1])
        telemetry.close()


if __name__ == '__main__':
    main()
