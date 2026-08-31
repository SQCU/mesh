import argparse, json, os, signal, sys, time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "..", "rdma"))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "payload", "tools"))

from solver.xonwire import Mesh, Reassembler, TxWindow, parse_hdr, REQ, RESP
from solver.strat.estimator import StrategyEstimator, state_from_runtime
from solver.strat.game import succession
from solver.strat.instruments import CartTarget, Participant, build_instruments, decode_allocations, update_weight_table, weights_from_table
from solver.strat.live_belief import LiveBelief
from solver.strat.runtime import CartSnapshot, GameContext, carts as game_carts, winner
from strategy_io_schema import OBS, CS, EVT, SC

OBS_W, CART_W, EVT_W, RESP_W = 40, 12, 6, 8
CKPT = os.path.join(_HERE, "runs", "policy_ckpt_v3.npz")
ONLINE_CKPT = os.path.join(_HERE, "runs", "policy_online_v3.npz")
TELEM = os.path.join(_HERE, "runs", "cartserver_telemetry.jsonl")
L_LEVELS = 8


def team_resources(rows, teams):
    out = []
    for team in range(1, teams + 1):
        selected = rows[np.asarray(rows[:, OBS["TEAM"]], dtype=np.int64) == team]
        alive = selected[:, OBS["ALIVE"]] >= 0.5 if len(selected) else np.zeros(0, dtype=bool)
        weapons = [int(round(value)) for value in selected[:, OBS["WEAPONS"]]]
        speeds = np.linalg.norm(selected[:, OBS["VEL_X"]:OBS["VEL_Z"] + 1], axis=1) if len(selected) else np.zeros(0)
        out.append({
            "team": team,
            "players": len(selected),
            "alive": int(alive.sum()),
            "health": round(float(selected[:, OBS["HEALTH"]].sum()), 3),
            "armor": round(float(selected[:, OBS["ARMOR"]].sum()), 3),
            "ammo": round(float(selected[:, OBS["AMMO"]].sum()), 3),
            "weapon_slots": sum(bin(value).count("1") for value in weapons),
            "weapon_union": int(np.bitwise_or.reduce(weapons, initial=0)),
            "mean_speed": round(float(speeds.mean()) if len(speeds) else 0.0, 3),
            "power": round(float(selected[:, OBS["POWER"]].sum()), 3),
        })
    return out


def estimator_bundle(est):
    """The parameter tree a checkpoint covers — the same one `OnlineLearner` saves."""
    import mlx.nn as nn

    bundle = nn.Module()
    bundle.qkv = est.qkv
    bundle.encoder = est.encoder
    bundle.head = est.head
    bundle.value = est.value
    return bundle


def load_ckpt_into(est, path):
    """Load a checkpoint into `est`, or REFUSE — never a silent partial load.

    The estimator bundle here is a strict subset of the learner's bundle (the
    learner also owns `dynamics`), so the fingerprints differ by construction;
    what is checked is that every parameter the checkpoint carries for these
    four submodules exists here with the SAME SHAPE, and that the checkpoint's
    recorded architecture spec agrees on them. A checkpoint with no recorded
    architecture is a pre-fingerprint checkpoint and is refused.
    """
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from solver.strat.online import ARCH_KEY, ARCH_SPEC_KEY, CheckpointArchitectureMismatch

    saved = np.load(path, allow_pickle=False)
    keys = list(saved.files)
    if ARCH_KEY not in keys:
        raise CheckpointArchitectureMismatch(
            f"{path} records no architecture fingerprint (pre-fingerprint checkpoint); "
            "refusing to load it into the current model"
        )
    live = dict(tree_flatten(estimator_bundle(est).parameters()))
    mismatched = []
    weights = []
    for name in keys:
        if name.startswith("__"):
            continue
        if name.startswith("dynamics."):
            continue  # owned by the learner, not the estimator
        if name not in live:
            mismatched.append((name, list(saved[name].shape), None))
        elif tuple(live[name].shape) != tuple(saved[name].shape):
            mismatched.append((name, list(saved[name].shape), list(live[name].shape)))
        else:
            weights.append((name, mx.array(saved[name])))
    missing = sorted(set(live) - {name for name, _ in weights})
    if mismatched or missing:
        raise CheckpointArchitectureMismatch(
            f"{path} architecture {str(saved[ARCH_KEY])!r} does not match this estimator.\n"
            f"  shape/name mismatches: {mismatched}\n"
            f"  parameters the checkpoint does not supply: {missing}\n"
            f"  checkpoint spec: {str(saved[ARCH_SPEC_KEY]) if ARCH_SPEC_KEY in keys else '(none)'}"
        )
    estimator_bundle(est).load_weights(weights, strict=True)
    return [name for name, _ in weights]


class EstCache:
    def __init__(self, checkpoint, allow_mismatch=False):
        self.est = None
        self.trained = False
        self.checkpoint = checkpoint
        self.allow_mismatch = bool(allow_mismatch)

    def get(self, k, j, l):
        if self.est is not None:
            return self.est, self.trained
        est = StrategyEstimator.for_runtime(k, l, seed=20260829)
        loaded = []
        if os.path.exists(self.checkpoint):
            if self.allow_mismatch:
                try:
                    loaded = load_ckpt_into(est, self.checkpoint)
                except Exception as exc:
                    print(f"[responder] ARCHITECTURE MISMATCH -- running an initialized "
                          f"policy instead:\n{exc}", flush=True)
            else:
                loaded = load_ckpt_into(est, self.checkpoint)
        self.est, self.trained = est, bool(loaded)
        print(
            f"[responder] shared estimator initialized at k={k} j={j} l={l} loaded={len(loaded)}",
            flush=True,
        )
        return self.est, self.trained


def build_cartstate(cart_rows, k):
    j = cart_rows.shape[0]
    pos = np.zeros(j)
    control = np.full(j, -1, dtype=np.int64)
    for c in range(j):
        pos[c] = np.clip(float(cart_rows[c, CS["DEPTH"]]), 0, 1) * L_LEVELS
        ctrl = int(round(cart_rows[c, CS["CTRL"]]))
        control[c] = ctrl - 1 if ctrl >= 1 else -1
    return CartSnapshot(
        pos=pos,
        control=control,
        banked=np.zeros(k),
        levels=L_LEVELS,
    )


def array(value, dtype=None):
    out = np.asarray(value)
    return out.astype(dtype) if dtype is not None else out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peer-node", type=int, default=0)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--off-policy-players", type=int, default=0)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--save-secs", type=float, default=30.0)
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--online-checkpoint", default=ONLINE_CKPT)
    ap.add_argument("--resume-checkpoint")
    ap.add_argument("--telemetry", default=TELEM)
    ap.add_argument("--environment", default="game2_server")
    ap.add_argument("--append-telemetry", action="store_true")
    ap.add_argument("--model-sample-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260829)
    ap.add_argument("--replay-capacity", type=int, default=2048,
                    help="transitions kept in the replay ring (oldest evicted first)")
    ap.add_argument("--replay-memory-mb", type=float, default=256.0,
                    help="memory ceiling for the replay ring; evicts oldest when exceeded")
    ap.add_argument("--replay-batch", type=int, default=8,
                    help="transitions per replayed gradient step")
    ap.add_argument("--replay-steps", type=int, default=4,
                    help="replayed gradient steps taken after each fresh segment")
    ap.add_argument(
        "--allow-arch-mismatch", action="store_true",
        help="Explicit re-init path: run an initialized policy (and re-initialize the "
             "learner) when the checkpoint's architecture fingerprint disagrees. Without "
             "it an architecture mismatch is a loud refusal, never a partial load.",
    )
    args = ap.parse_args()

    runstate_path = (args.online_checkpoint or CKPT) + ".runstate.json"

    def save_runstate(rng, resp_id, nt_written, updates):
        try:
            tmp = runstate_path + ".new"
            with open(tmp, "w") as fh:
                json.dump({
                    "rng": rng.bit_generator.state,
                    "resp_id": int(resp_id),
                    "nt_written": int(nt_written),
                    "updates": int(updates),
                    "environment": args.environment,
                    "wrote_at": time.time(),
                }, fh)
            os.replace(tmp, runstate_path)
        except Exception as exc:
            print(f"[responder] runstate save failed: {exc}", flush=True)

    m = Mesh()
    print(f"[responder] mesh attached slots={m.slots} usable={m.usable}", flush=True)
    ra_obs = Reassembler(REQ, OBS_W, 4096, m.usable)
    ra_cart = Reassembler(REQ, CART_W, 4096, m.usable)
    ra_evt = Reassembler(REQ, EVT_W, 4096, m.usable)
    tx = TxWindow(m)
    est_cache = EstCache(args.checkpoint, allow_mismatch=args.allow_arch_mismatch)
    live_belief = LiveBelief()
    rng = np.random.default_rng(args.seed)
    learner = None
    last_saved_update = 0
    previous = None
    weight_table = {}
    weight_context = None
    carry_key = None
    resp_id = 0
    last_obs = None
    last_cart = None
    last_evt = None
    belief_depths = None
    belief_episode = 0
    nt_written = 0
    resumed_runstate = False
    if args.append_telemetry and os.path.exists(runstate_path):
        try:
            with open(runstate_path) as fh:
                rs = json.load(fh)
            rng.bit_generator.state = rs["rng"]
            resp_id = int(rs.get("resp_id", 0))
            nt_written = int(rs.get("nt_written", 0))
            resumed_runstate = True
            print(f"[responder] resumed runstate: resp_id={resp_id} nt_written={nt_written} "
                  f"updates={rs.get('updates')}", flush=True)
        except Exception as exc:
            print(f"[responder] runstate restore failed: {exc}; fresh RNG/cursor", flush=True)
    os.makedirs(os.path.dirname(args.telemetry) or ".", exist_ok=True)
    telem = open(args.telemetry, "a" if args.append_telemetry else "w")
    stats = dict(slots=0, obs=0, cart=0, evt=0, resp=0, updates=0)
    t0 = time.time()
    last_report = t0
    last_save_time = t0

    # The learner is a SERVICE with exactly ONE lifetime path: it runs until it
    # is signalled or killed. There is no wall-clock deadline and no flag that
    # reintroduces one -- match duration belongs to the MATCH. A learner that
    # stops on its own leaves the dedicated server running with nothing
    # attached; restarts are fine and exercise the resume contract.
    stopping = {"signal": None}

    def _stop(signum, _frame):
        stopping["signal"] = int(signum)
        print(f"[responder] signal {signum}: draining and checkpointing", flush=True)

    for _sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(_sig, _stop)
        except (ValueError, OSError):
            pass

    print("[responder] lifetime: service — runs until SIGINT/SIGTERM/SIGHUP", flush=True)

    try:
        while stopping["signal"] is None:
            got_any = False
            for buf, src in m.read(np.uint8):
                got_any = True
                stats["slots"] += 1
                header = parse_hdr(buf)
                if header is None:
                    continue
                width = header["width"]
                if width == OBS_W:
                    record = ra_obs.feed(buf)
                    if record:
                        last_obs = record, ra_obs.stage[:record["rows"]].copy()
                        stats["obs"] += 1
                elif width == CART_W:
                    record = ra_cart.feed(buf)
                    if record:
                        last_cart = record, ra_cart.stage[:record["rows"]].copy()
                        stats["cart"] += 1
                elif width == EVT_W:
                    record = ra_evt.feed(buf)
                    if record:
                        last_evt = record, ra_evt.stage[:record["rows"]].copy()
                        stats["evt"] += 1

            if last_cart is not None and last_obs is not None:
                ch, cart_rows = last_cart
                oh, obs_rows = last_obs
                last_cart = None
                all_teams = array(obs_rows[:, OBS["TEAM"]], int)
                active = np.flatnonzero(all_teams >= 1)
                if len(active):
                    rows = obs_rows[active]
                    participant_ids = array(rows[:, OBS["ID"]], int)
                    teams_present = all_teams[active]
                    cart_ctrl = array(cart_rows[:, CS["CTRL"]], int)
                    k = max([2] + teams_present.tolist() + cart_ctrl[cart_ctrl >= 1].tolist())
                    j = cart_rows.shape[0]
                    l = len(active)
                    team_of = (teams_present - 1).clip(0, k - 1).tolist()
                    est, trained = est_cache.get(k, j, l)
                    key = (k, j, tuple(participant_ids.tolist()), tuple(team_of))
                    map_key = tuple(
                        (int(round(cart_rows[c, CS["ID"]])),
                         round(float(cart_rows[c, CS["LENGTH"]]), 3))
                        for c in range(j)
                    )
                    depths = np.asarray(cart_rows[:, CS["DEPTH"]], dtype=np.float32)
                    if (belief_depths is not None and np.max(belief_depths, initial=0) > 0.5
                            and np.max(depths, initial=0) < 0.02):
                        belief_episode += 1
                    belief_depths = depths.copy()
                    episode_context = (map_key, belief_episode, tuple(participant_ids.tolist()), tuple(team_of))
                    belief_reset = live_belief.sync(episode_context, max(int(ch["tick"]), int(oh["tick"])))
                    if weight_context != episode_context:
                        weight_table = {}
                        weight_context = episode_context
                    deposited = 0
                    event_tick = None
                    if last_evt is not None:
                        eh, event_rows = last_evt
                        last_evt = None
                        event_tick = int(eh["tick"])
                        deposited = live_belief.ingest(event_rows, EVT)
                    beta, belief_diag = live_belief.beliefs(rows, OBS)
                    cartstate = build_cartstate(cart_rows, k)
                    context = GameContext(tuple(range(k)), tuple(team_of), L_LEVELS)
                    participants = [
                        Participant(
                            int(participant_ids[p]), int(teams_present[p]),
                            int(round(rows[p, OBS["CELL"]])),
                            tuple(float(v) for v in rows[p, OBS["POS_X"]:OBS["POS_Z"] + 1]),
                            float(rows[p, OBS["ALIVE"]]),
                            float(rows[p, OBS["HEALTH"]]) / 100.0,
                            float(rows[p, OBS["ARMOR"]]) / 100.0,
                            float(rows[p, OBS["AMMO"]]),
                            float(rows[p, OBS["TSS"]]),
                        )
                        for p in range(l)
                    ]
                    carts = [
                        CartTarget(
                            int(round(cart_rows[c, CS["ID"]])), int(round(cart_rows[c, CS["CTRL"]])),
                            float(cart_rows[c, CS["DEPTH"]]),
                            float(cart_rows[c, CS["SPEED"]]),
                            float(cart_rows[c, CS["PROGRESS"]]),
                        )
                        for c in range(j)
                    ]
                    items, rivals, cells = live_belief.instrument_targets(rows, OBS)
                    batch = build_instruments(participants, carts, items, rivals, cells)
                    batch_key = (
                        key,
                        tuple((instrument.kind.value, instrument.subject) for instrument in batch.instruments),
                    )
                    online_metrics = None
                    # A changed instrument set (or roster) CLOSES the credit
                    # segment; it does not discard the transition. The old code
                    # set `previous = None` here, so on a run where the
                    # instrument set turns over every tick the learner saw no
                    # transitions at all. The only thing that actually changes
                    # with the batch is the width of the integrated weight
                    # state, and that is re-derived from the weight table below
                    # in THIS state's instrument basis.
                    segment_break = carry_key is not None and carry_key != batch_key
                    carry_key = batch_key
                    w_in = weights_from_table(batch, weight_table)
                    state = state_from_runtime(context, cartstate, rows, OBS, beta, batch, w=w_in)
                    if args.train:
                        if learner is None:
                            from solver.strat.online import OnlineLearner

                            learner = OnlineLearner(
                                est,
                                learning_rate=args.learning_rate,
                                checkpoint=args.online_checkpoint,
                                load_checkpoint=args.resume_checkpoint or args.checkpoint,
                                on_architecture_mismatch=(
                                    "reinit" if args.allow_arch_mismatch else "refuse"
                                ),
                                replay_capacity=args.replay_capacity,
                                replay_memory_mb=args.replay_memory_mb,
                                replay_batch=args.replay_batch,
                                replay_steps=args.replay_steps,
                                seed=args.seed,
                            )
                        if previous is not None:
                            if segment_break:
                                # close the pending segment against the LAST
                                # state written in the old instrument basis,
                                # before anything in the new basis is credited
                                learner.flush(previous["state"], previous["snapshot"],
                                              terminal=True)
                            # the successor's weight state, in the successor's
                            # own instrument basis — so a change of instrument
                            # count is representable instead of discarded
                            previous["w_out"] = w_in.copy()
                            online_metrics = learner.observe(
                                previous, state, cartstate, terminal=segment_break
                            )
                        stats["updates"] = learner.updates
                        due_updates = (
                            args.save_every > 0
                            and learner.updates > last_saved_update
                            and learner.updates % args.save_every == 0
                        )
                        due_time = (
                            args.save_secs > 0
                            and learner.updates > last_saved_update
                            and time.time() - last_save_time >= args.save_secs
                        )
                        if due_updates or due_time:
                            learner.save()
                            save_runstate(rng, resp_id, nt_written, learner.updates)
                            last_saved_update = learner.updates
                            last_save_time = time.time()

                    result = est.forward(state)
                    actions = array(result.action, np.int64).reshape(l)
                    w_next = array(result.w_next, np.float32).reshape(l, len(batch.instruments))
                    dynamics_guidance = None
                    if learner is not None and learner.updates:
                        from solver.strat.dynamics import guided_actions

                        actions, dynamics_guidance = guided_actions(
                            learner.dynamics, state, array(result.score, np.float32), actions,
                            temperature=est.temperature,
                        )
                    n_off_policy = min(max(0, args.off_policy_players), l)
                    off_policy = np.zeros(l, dtype=bool)
                    if n_off_policy:
                        chosen = rng.choice(l, size=n_off_policy, replace=False)
                        for player in chosen:
                            eligible_actions = np.flatnonzero(batch.eligible[player])
                            actions[player] = rng.choice(eligible_actions)
                        off_policy[chosen] = True
                    import mlx.core as mx
                    from solver.strat.head import strategy_log_prob

                    target_logp = array(
                        strategy_log_prob(result.score, mx.array(actions), est.temperature),
                        np.float32,
                    )
                    behavior_logp = target_logp.copy()
                    behavior_logp[off_policy] = -np.log(np.maximum(1, batch.eligible[off_policy].sum(axis=1)))
                    weight_table = update_weight_table(batch, w_next, weight_table)
                    projected = winner(context, cartstate)
                    pw = 0 if projected is None else projected + 1
                    succ = [(team + 1, denial) for team, denial in succession(game_carts(cartstate), context.teams)]
                    response = np.zeros((len(obs_rows), RESP_W), dtype=np.float32)
                    intensities = np.clip(
                        1.0 + w_next[np.arange(l), actions], 0.1, 3.0
                    )
                    local_response = decode_allocations(
                        batch, actions, intensity=intensities,
                        commitments=np.clip(intensities, 0.25, 3.0),
                        spawn_delays=np.clip(intensities, 0.0, 3.0),
                        lead=teams_present == pw,
                    )
                    response[active] = local_response
                    model_arrays = {
                        "x": array(state.x, np.float32),
                        "beta": array(state.beta, np.float32),
                        "z": array(state.z, np.float32),
                        "relation": array(state.relation, np.float32),
                        "hierarchy": array(state.hierarchy, np.float32),
                        "w": array(state.w, np.float32),
                        "ir": array(result.ir, np.float32),
                        "gram": array(result.gram, np.float32),
                        "score": array(result.score, np.float32),
                        "winner_value": array(result.winner_value, np.float32),
                        "loser_value": array(result.loser_value, np.float32),
                    }
                    model = {
                        "shapes": {name: list(value.shape) for name, value in model_arrays.items()},
                        "finite": {name: bool(np.isfinite(value).all()) for name, value in model_arrays.items()},
                        "range": {
                            name: [round(float(value.min()), 6), round(float(value.max()), 6)]
                            for name, value in model_arrays.items()
                        },
                        "winner_value": np.round(model_arrays["winner_value"], 5).tolist(),
                        "loser_value": np.round(model_arrays["loser_value"], 5).tolist(),
                    }
                    sampled = args.model_sample_every > 0 and (resp_id + 1) % args.model_sample_every == 0
                    model["sampled"] = sampled
                    if sampled:
                        for name in ("x", "beta", "z", "relation", "hierarchy", "w", "ir", "gram", "score"):
                            model[name] = np.round(model_arrays[name], 5).tolist()
                    assignments = []
                    strategy_focus = np.zeros((k, k), dtype=np.int64)
                    for local, row_index in enumerate(active):
                        team = int(teams_present[local])
                        action = int(actions[local])
                        instrument = batch.instruments[action]
                        kind = instrument.kind.value
                        target = float(local_response[local, SC["TARGET"]])
                        gain = float(local_response[local, SC["GAIN"]])
                        lane = float(local_response[local, SC["LANE"]])
                        if instrument.team > 0 and instrument.team != team:
                            strategy_focus[team - 1, instrument.team - 1] += 1
                        assignments.append(
                            dict(
                                row=int(row_index), edict=int(participant_ids[local]), team=team,
                                controller="bot" if rows[local, OBS["CONTROL"]] >= 0.5 else "human",
                                behavior="uniform" if off_policy[local] else "policy",
                                action=action, kind=kind, subject=int(instrument.subject), target=target,
                                gain=round(gain, 4), lane=round(lane, 4),
                                commit=round(float(local_response[local, SC["COMMIT"]]), 4),
                                spawn=round(float(local_response[local, SC["SPAWN"]]), 4),
                                target_logp=round(float(target_logp[local]), 6),
                                behavior_logp=round(float(behavior_logp[local]), 6),
                            )
                        )
                    resp_id += 1
                    tx.send(RESP, ch["req_id"], ch["tick"], response, args.peer_node)
                    stats["resp"] += 1
                    previous = dict(
                        context=context, state=state, snapshot=cartstate, cartstate=cartstate, w_in=w_in.copy(),
                        w_out=w_next.copy(), actions=actions.copy(),
                        behavior_logp=behavior_logp.copy(),
                        teams_present=teams_present.copy(),
                    )
                    line = dict(
                        environment=args.environment,
                        t=round(time.time() - t0, 3),
                        mode="online_train" if args.train else "inference",
                        updates=learner.updates if learner is not None else 0,
                        resp_id=resp_id, req_tick=int(ch["tick"]), obs_tick=int(oh["tick"]),
                        k=k, j=j, l=l, trained=bool(trained or args.train),
                        off_policy_players=n_off_policy,
                        resources=team_resources(rows, k),
                        strategy_focus=strategy_focus.tolist(),
                        carts=[
                            dict(
                                id=int(round(cart_rows[c, CS["ID"]])),
                                depth=round(float(cart_rows[c, CS["DEPTH"]]), 5),
                                ctrl=int(round(cart_rows[c, CS["CTRL"]])),
                                speed=round(float(cart_rows[c, CS["SPEED"]]), 4),
                                progress=round(float(cart_rows[c, CS["PROGRESS"]]), 4),
                            )
                            for c in range(j)
                        ],
                        PW=int(pw), SUCC=[[int(a), round(float(b), 4)] for a, b in succ],
                        belief=dict(
                            reset=belief_reset, event_tick=event_tick,
                            deposited=deposited, **belief_diag,
                        ),
                        instrument_count=len(batch.instruments),
                        instrument_counts={
                            kind: sum(instrument.kind.value == kind for instrument in batch.instruments)
                            for kind in sorted({instrument.kind.value for instrument in batch.instruments})
                        },
                        assignments=assignments, update=online_metrics,
                        dynamics_guidance=dynamics_guidance,
                        model=model,
                    )
                    telem.write(json.dumps(line) + "\n")
                    telem.flush()
                    nt_written += 1

            if not got_any:
                time.sleep(0.003)
            if time.time() - last_report >= 5:
                print(
                    f"[responder] {int(time.time() - t0)}s stats={stats} "
                    f"telem_lines={nt_written} resp_id={resp_id}",
                    flush=True,
                )
                last_report = time.time()
    finally:
        if learner is not None:
            if previous is not None:
                learner.flush(previous["state"], previous["cartstate"], terminal=True)
            learner.save()
            save_runstate(rng, resp_id, nt_written, learner.updates)
        telem.close()
    print(f"[responder] STOPPED (signal {stopping['signal']}) stats={stats} "
          f"telem_lines={nt_written}", flush=True)


if __name__ == "__main__":
    main()
