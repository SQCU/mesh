import argparse, json, os, sys, time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "..", "rdma"))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "payload", "tools"))

from solver.xonwire import Mesh, Reassembler, TxWindow, parse_hdr, REQ, RESP
from solver.strat.cartsim import CartSim, CartState
from solver.strat.estimator import StrategyEstimator, state_from_cartsim, state_with_instruments, state_with_observations
from solver.strat.game_value import EmpiricalTransitionGraph
from solver.strat.instruments import CartTarget, Participant, build_instruments, decode_allocations, update_weight_table, weights_from_table
from solver.strat.live_belief import LiveBelief
from strategy_io_schema import OBS, CS, EVT, SC, projected_winner, succession, carts_from_rows

OBS_W, CART_W, EVT_W, RESP_W = 40, 12, 6, 8
CKPT = os.path.join(_HERE, "runs", "policy_ckpt_v3.npz")
ONLINE_CKPT = os.path.join(_HERE, "runs", "policy_online_v3.npz")
TELEM = os.path.join(_HERE, "runs", "cartserver_telemetry.jsonl")
L_LEVELS = 8


def load_ckpt_into(est, path):
    import mlx.core as mx

    saved = np.load(path, allow_pickle=True)
    values = {name: saved[name] for name in saved.files}
    leaves = {
        "qkv.W_q": (est.qkv, "W_q"),
        "qkv.W_k": (est.qkv, "W_k"),
        "qkv.W_v": (est.qkv, "W_v"),
        "head.norm_weight": (est.head, "norm_weight"),
        "head.w_gate": (est.head, "w_gate"),
        "head.w_up": (est.head, "w_up"),
        "head.w_down": (est.head, "w_down"),
        "value.winner.up.weight": (est.value.winner.up, "weight"),
        "value.winner.up.bias": (est.value.winner.up, "bias"),
        "value.winner.down.weight": (est.value.winner.down, "weight"),
        "value.winner.down.bias": (est.value.winner.down, "bias"),
        "value.loser.up.weight": (est.value.loser.up, "weight"),
        "value.loser.up.bias": (est.value.loser.up, "bias"),
        "value.loser.down.weight": (est.value.loser.down, "weight"),
        "value.loser.down.bias": (est.value.loser.down, "bias"),
    }
    loaded = []
    for name, target in leaves.items():
        value = getattr(target[0], target[1])
        if name in values and tuple(values[name].shape) == tuple(value.shape):
            setattr(target[0], target[1], mx.array(values[name]))
            loaded.append(name)
    return loaded


class EstCache:
    def __init__(self, checkpoint):
        self.est = None
        self.trained = False
        self.checkpoint = checkpoint

    def get(self, k, j, l):
        if self.est is not None:
            return self.est, self.trained
        sim = CartSim(k, j, l, team_of=[p % k for p in range(l)], L=L_LEVELS, seed=20260829)
        est = StrategyEstimator.for_cartsim(sim, seed=20260829)
        loaded = []
        if os.path.exists(self.checkpoint):
            try:
                loaded = load_ckpt_into(est, self.checkpoint)
            except Exception as exc:
                print(f"[responder] checkpoint load failed ({exc}); initialized policy", flush=True)
        self.est, self.trained = est, len(loaded) == 15
        print(
            f"[responder] shared estimator initialized at k={k} j={j} l={l} loaded={len(loaded)}",
            flush=True,
        )
        return self.est, self.trained


def build_cartstate(cart_rows, k):
    j = cart_rows.shape[0]
    pos = np.zeros(j)
    control = np.full(j, -1, dtype=np.int64)
    depth_frac = np.zeros(j)
    for c in range(j):
        depth_frac[c] = float(cart_rows[c, CS["DEPTH"]])
        pos[c] = np.clip(depth_frac[c], 0, 1) * L_LEVELS
        ctrl = int(round(cart_rows[c, CS["CTRL"]]))
        control[c] = ctrl - 1 if ctrl >= 1 else -1
    return CartState(
        pos=pos,
        control=control,
        banked=np.zeros(k),
        L=L_LEVELS,
        t=0,
        highwater=np.floor(pos).astype(np.int64),
    ), depth_frac


def array(value, dtype=None):
    out = np.asarray(value)
    return out.astype(dtype) if dtype is not None else out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--secs", type=float, default=90.0)
    ap.add_argument("--peer-node", type=int, default=0)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--off-policy-players", type=int, default=0)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--online-checkpoint", default=ONLINE_CKPT)
    ap.add_argument("--resume-checkpoint")
    ap.add_argument("--telemetry", default=TELEM)
    ap.add_argument("--seed", type=int, default=20260829)
    args = ap.parse_args()

    m = Mesh()
    print(f"[responder] mesh attached slots={m.slots} usable={m.usable}", flush=True)
    ra_obs = Reassembler(REQ, OBS_W, 4096, m.usable)
    ra_cart = Reassembler(REQ, CART_W, 4096, m.usable)
    ra_evt = Reassembler(REQ, EVT_W, 4096, m.usable)
    tx = TxWindow(m)
    est_cache = EstCache(args.checkpoint)
    live_belief = LiveBelief()
    empirical_game = EmpiricalTransitionGraph(())
    rng = np.random.default_rng(args.seed)
    learner = None
    last_saved_update = 0
    previous = None
    previous_game_state = None
    previous_game_teams = ()
    weight_table = {}
    carry_key = None
    resp_id = 0
    last_obs = None
    last_cart = None
    last_evt = None
    belief_depths = None
    belief_episode = 0
    os.makedirs(os.path.dirname(args.telemetry) or ".", exist_ok=True)
    telem = open(args.telemetry, "w")
    nt_written = 0
    stats = dict(slots=0, obs=0, cart=0, evt=0, resp=0, updates=0)
    t0 = time.time()
    last_report = t0

    try:
        while time.time() - t0 < args.secs:
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
                    belief_reset = live_belief.sync(
                        (map_key, belief_episode, tuple(participant_ids.tolist()), tuple(team_of)),
                        max(int(ch["tick"]), int(oh["tick"])),
                    )
                    deposited = 0
                    event_tick = None
                    if last_evt is not None:
                        eh, event_rows = last_evt
                        last_evt = None
                        event_tick = int(eh["tick"])
                        deposited = live_belief.ingest(event_rows, EVT)
                    beta, belief_diag = live_belief.beliefs(rows, OBS)
                    sim = CartSim(k, j, l, team_of=team_of, L=L_LEVELS, seed=args.seed)
                    cartstate, depth_frac = build_cartstate(cart_rows, k)
                    game_state = (
                        map_key,
                        belief_episode,
                        k,
                        tuple(np.floor(cartstate.pos).astype(np.int64).tolist()),
                        tuple(cartstate.control.tolist()),
                    )
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
                            c, int(round(cart_rows[c, CS["CTRL"]])),
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
                    if carry_key is not None and carry_key != batch_key:
                        if learner is not None and previous is not None:
                            if carry_key[0] == key:
                                online_metrics = learner.observe(
                                    previous, previous["state"], cartstate, terminal=True
                                )
                            else:
                                online_metrics = learner.flush(
                                    previous["state"], previous["cartstate"], terminal=True
                                )
                        previous = None
                    carry_key = batch_key
                    w_in = weights_from_table(batch, weight_table)
                    state = state_with_observations(state_from_cartsim(sim, cartstate), rows, OBS)
                    state.beta = beta
                    state = state_with_instruments(state, batch, w=w_in)
                    if args.train:
                        if learner is None:
                            from solver.strat.online import OnlineLearner

                            learner = OnlineLearner(
                                est,
                                learning_rate=args.learning_rate,
                                checkpoint=args.online_checkpoint,
                                load_checkpoint=args.resume_checkpoint or args.checkpoint,
                            )
                        if previous is not None:
                            online_metrics = learner.observe(previous, state, cartstate)
                        stats["updates"] = learner.updates
                        if (
                            args.save_every > 0
                            and learner.updates > last_saved_update
                            and learner.updates % args.save_every == 0
                        ):
                            learner.save()
                            last_saved_update = learner.updates

                    game_value = empirical_game.evaluate(game_state)
                    if (
                        previous_game_state is not None
                        and previous_game_state[:3] == game_state[:3]
                        and previous_game_state != game_state
                    ):
                        for team in sorted(set(previous_game_teams)):
                            empirical_game.observe(previous_game_state, game_state, int(team))
                        game_value = empirical_game.evaluate(previous_game_state)
                    if np.any(depth_frac >= 0.999):
                        empirical_game.observe_terminal(game_state)
                        game_value = empirical_game.evaluate(game_state)

                    result = est.forward(state)
                    actions = array(result.action, np.int64).reshape(l)
                    w_next = array(result.w_next, np.float32).reshape(l, len(batch.instruments))
                    dynamics_guidance = None
                    if learner is not None and learner.updates:
                        from solver.strat.dynamics import guided_actions

                        actions, dynamics_guidance = guided_actions(
                            learner.dynamics, state, w_next, actions,
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
                        strategy_log_prob(result.w_next, mx.array(actions), est.temperature),
                        np.float32,
                    )
                    behavior_logp = target_logp.copy()
                    behavior_logp[off_policy] = -np.log(np.maximum(1, batch.eligible[off_policy].sum(axis=1)))
                    weight_table = update_weight_table(batch, w_next, weight_table)
                    schema_carts = carts_from_rows(cart_rows)
                    pw = projected_winner(schema_carts)
                    succ = succession(schema_carts)
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
                    assignments = []
                    for local, row_index in enumerate(active):
                        team = int(teams_present[local])
                        action = int(actions[local])
                        instrument = batch.instruments[action]
                        kind = instrument.kind.value
                        target = float(local_response[local, SC["TARGET"]])
                        gain = float(local_response[local, SC["GAIN"]])
                        lane = float(local_response[local, SC["LANE"]])
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
                    tx.send(RESP, resp_id, ch["tick"], response, args.peer_node)
                    stats["resp"] += 1
                    previous = dict(
                        sim=sim, state=state, cartstate=cartstate, w_in=w_in.copy(),
                        w_out=w_next.copy(), actions=actions.copy(),
                        behavior_logp=behavior_logp.copy(), game_state=game_state,
                        teams_present=teams_present.copy(),
                    )
                    previous_game_state = game_state
                    previous_game_teams = teams_present.copy()
                    line = dict(
                        t=round(time.time() - t0, 3),
                        mode="online_train" if args.train else "inference",
                        resp_id=resp_id, req_tick=int(ch["tick"]), obs_tick=int(oh["tick"]),
                        k=k, j=j, l=l, trained=bool(trained or args.train),
                        off_policy_players=n_off_policy,
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
                        game_value=dict(
                            state=game_state, kind=game_value.kind, nimber=game_value.nimber,
                            reason=game_value.reason,
                            roles={
                                str(role): dict(
                                    mobility=value.mobility,
                                    complete=value.complete,
                                )
                                for role, value in game_value.role_values.items()
                            },
                        ),
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
        telem.close()
    print(f"[responder] DONE stats={stats} telem_lines={nt_written}", flush=True)


if __name__ == "__main__":
    main()
