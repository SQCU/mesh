// =============================================================================
//  sv_payload_strategy_io.qc  —  SCAFFOLDING  [BUILD]
// -----------------------------------------------------------------------------
//  Reference stub implementation of the strategy I/O contract declared in
//  sv_payload_strategy_io.qh. It compiles against the shipping payload symbols
//  (payload_carts[], plc_s, plc_length, plc_ctrl, plc_speed_now, plc_idle) and
//  the real mesh builtins (bridge/qc/mesh_ipc.qh), but it is NOT #included by
//  the generated _mod.inc and NOTHING here is called from the working
//  sv_payload.qc. Wiring it in is the one-line-per-callsite [BUILD] step in
//  STRATEGY-IO.md §6.
//
//  Style note (repo AGENTS.md): no rejection filtering. These stubs never
//  "decline to act" by class or condition; unbuilt bodies emit honest zeros and
//  are marked TODO, they do not guard a capability away. The perception gate is
//  a GAME rule (occlusion), not a capability refusal.
// =============================================================================

#include "sv_payload_strategy_io.qh"

float plc_str_h_obs  = -1;   // gather handle A (per-player)
float plc_str_h_cart = -1;   // gather handle B (cartstate, GUARANTEED)
float plc_str_h_evt  = -1;   // gather handle C (perception ring)
float plc_str_h_sc   = -1;   // scatter handle (8-col responses)
float plc_str_seq    = 0;    // last completed plan sequence from mesh_poll

// ---------------------------------------------------------------------------
//  payload_strategy_attach — lazy open of the four handles.
//  Retry-until-attached (playerbot-interface §2.1 payload_mesh_attach pattern).
//  No "give up" branch: if the coprocessor is not yet up we simply retry next
//  step; we never demote to a no-mesh fallback (AGENTS.md).
// ---------------------------------------------------------------------------
void payload_strategy_attach()
{
	if (plc_str_h_obs < 0)
		plc_str_h_obs  = mesh_open(PLC_STR_NODE, PLC_STR_OBS_W,  maxclients);
	if (plc_str_h_cart < 0)
		plc_str_h_cart = mesh_open(PLC_STR_NODE, PLC_STR_CART_W, PLC_MAX_CARTS);
	if (plc_str_h_evt < 0)
		plc_str_h_evt  = mesh_open(PLC_STR_NODE, PLC_STR_EVT_W,  PLC_STR_EVT_ROWS);
	if (plc_str_h_sc < 0)
		plc_str_h_sc   = mesh_open(PLC_STR_NODE, PLC_STR_RESP_W, maxclients);
}

// ---------------------------------------------------------------------------
//  payload_str_cell_of — V-cell id for a world origin (payload-spec §2.2.2).
//  [BUILD] Real impl: nearest navigable Voronoi cell over item/waypoint nodes,
//  cells fused until each bot's receptive field lands in the [5%,15%] band. Stub
//  returns a coarse spatial hash so the schema is exercisable end to end.
// ---------------------------------------------------------------------------
float payload_str_cell_of(vector org)
{
	// TODO[BUILD]: replace with the fused-Voronoi cell index. Coarse 256u grid.
	float gx = floor(org.x / 256);
	float gy = floor(org.y / 256);
	return (gx * 131 + gy) & 1023;
}

// ---------------------------------------------------------------------------
//  payload_perceive — the frustum + LOS + 2-V-cell perception gate.
//  (payload-spec §2.2.1; the foundational [BUILD] and the riskiest gap in
//  playerbot-interface §8.) For one bot, scan candidate map features and deposit
//  an edge-triggered event into the per-team ring ONLY when all three gates
//  pass. This is the ONLY producer of enemy/item rows; there is no omniscient
//  ITS_AVAILABLE feed anywhere in this path.
// ---------------------------------------------------------------------------
void payload_perceive(entity bot)
{
	if (IS_DEAD(bot))
		return;   // dead bots have no viewpoint — a physics fact, not a refusal

	vector eye = bot.origin + bot.view_ofs;
	float bcell = payload_str_cell_of(bot.origin);
	bot.plc_str_cell = bcell;

	// Gate 1 (frustum) + Gate 2 (LOS) reuse engine primitives:
	//   checkpvs(eye, target)               -> potentially-visible + frustum
	//   traceline(eye, target.origin, ...)  -> unoccluded line of sight
	// Gate 3 (range) — within 2 V-cells graph distance of the bot's cell.
	//
	// TODO[BUILD]: iterate item posts and enemy players:
	//   FOREACH item post p:
	//     if (checkpvs(eye, p) && los_clear(eye, p) && cell_hops(bcell,cell(p))<=2)
	//        deposit(cell(p), avail(p)?PLC..ITEM_HERE:PLC..ITEM_GONE, bot.team, p, phase, time)
	//   FOREACH_CLIENT enemy e of bot:
	//     if (checkpvs(eye, e) && los_clear(eye, e) && cell_hops(bcell,cell(e))<=2)
	//        deposit(cell(e), PLC..ENEMY_HERE, bot.team, etnum(e), threat, time)
	//
	// Edge-trigger: only deposit when the (cell,kind,subject) state DIFFERS from
	// this team's last stamp, so the ring carries events, not a per-tick dump.
}

// ---------------------------------------------------------------------------
//  payload_strategy_gather — fill per-edict fields then gather schemas A,B,C.
//  Cartstate (schema B) is filled UNCONDITIONALLY every step: it is the
//  guaranteed global feature (rl-training-spec §1). If this emit is ever skipped
//  that is a [BUILD] bug, per the spec.
// ---------------------------------------------------------------------------
void payload_strategy_gather(float nclients)
{
	// (B) cartstate rows — GUARANTEED. Fill scratch fields on each cart edict.
	// TODO[BUILD]: stage plc_s/plc_length/plc_ctrl/plc_speed_now/plc_idle into
	// the PLC_CS_* columns via per-column scratch fields, then:
	//   mesh_gather(plc_str_h_cart, PLC_CS_DEPTH, plc_cs_depth_fld, cart0, PLC_MAX_CARTS);
	//   ... one gather per PLC_CS_* column ...

	// (A) per-player observation rows x_b (own known state only).
	//   FOREACH_CLIENT: stage health/armor/ammo/pos/vel/weapons/powerup/tss/cell.
	//   mesh_gather(plc_str_h_obs, PLC_OBS_HEALTH, health, first, nclients); ...

	// (C) perception ring rows — already deposited this step by payload_perceive.
	//   mesh_gather(plc_str_h_evt, PLC_EVT_*, <ring fields>, ringbase, PLC_STR_EVT_ROWS);

	mesh_publish(plc_str_h_cart, plc_str_seq, PLC_MAX_CARTS);
	mesh_publish(plc_str_h_obs,  plc_str_seq, nclients);
	mesh_publish(plc_str_h_evt,  plc_str_seq, PLC_STR_EVT_ROWS);
}

// ---------------------------------------------------------------------------
//  payload_strategy_scatter — pull the latest completed plan down.
//  The 8 response columns are the absolute, mesh-integrated w_b. QC never
//  integrates (payload-spec §3.2); the optional prev/lerp is cosmetic smoothing.
// ---------------------------------------------------------------------------
void payload_strategy_scatter(float nclients)
{
	float seq = mesh_poll(plc_str_h_sc);
	if (seq <= plc_str_seq)
		return;   // no newer plan yet — hold last w_b (a hold, not a demotion)
	plc_str_seq = seq;

	// Preserve previous committed values for cosmetic lerp before overwrite.
	FOREACH_CLIENT(IS_PLAYER(it), {
		it.plc_str_gain_prev = it.plc_str_gain;
		it.plc_str_lane_prev = it.plc_str_lane;
	});

	mesh_scatter(plc_str_h_sc, PLC_SC_TARGET,  plc_str_target,  0, nclients);
	mesh_scatter(plc_str_h_sc, PLC_SC_GAIN,    plc_str_gain,    0, nclients);
	mesh_scatter(plc_str_h_sc, PLC_SC_LANE,    plc_str_lane,    0, nclients);
	mesh_scatter(plc_str_h_sc, PLC_SC_HUNT,    plc_str_hunt,    0, nclients);
	mesh_scatter(plc_str_h_sc, PLC_SC_EXPLORE, plc_str_explore, 0, nclients);
	mesh_scatter(plc_str_h_sc, PLC_SC_COMMIT,  plc_str_commit,  0, nclients);
	mesh_scatter(plc_str_h_sc, PLC_SC_SPAWN,   plc_str_spawn,   0, nclients);
	mesh_scatter(plc_str_h_sc, PLC_SC_LEAD,    plc_str_lead,    0, nclients);
}

// ---------------------------------------------------------------------------
//  havocbot_goalrating_strategy — the payoff: scattered weights become behavior.
//  ADDITIVE reference rater. It is a SEPARATE function from the shipping
//  havocbot_goalrating_payload (sv_payload.qc:604), which is left untouched.
//  Wiring choice (STRATEGY-IO.md §6): swap the role's rater call to this one.
//  Everything here is a navigation_routerating base-rating bias — the single
//  lever (navigation.qc:1220). It NEVER writes skill/bot_*aim* fields, so
//  skill-orthogonality holds by construction (playerbot-interface §4).
// ---------------------------------------------------------------------------
void havocbot_goalrating_strategy(entity this, float ratingscale)
{
	float tgt  = this.plc_str_target;
	float gain = this.plc_str_gain;

	// push / suppress cart k : committed cart gets the gain; lane fraction spreads
	// the bot along the golden path (ahead of the cart == suppress-position).
	if (tgt >= PLC_TGT_CART_BASE && tgt < PLC_TGT_ITEM_BASE)
	{
		float k = tgt - PLC_TGT_CART_BASE;
		if (k >= 0 && k < payload_cart_count)
			navigation_routerating(this, payload_carts[k], ratingscale * gain, 5000);
		// TODO[BUILD]: rate the g_payload_node at arclength (plc_str_lane * length)
		// along cart k, as ratingscale*gain, rangebias 3000 (path-slot spread).
	}
	// contest item post p
	else if (tgt >= PLC_TGT_ITEM_BASE && tgt < PLC_TGT_RIVAL_BASE)
	{
		// TODO[BUILD]: resolve item entity for post (tgt-PLC_TGT_ITEM_BASE) and
		// navigation_routerating(this, item, ratingscale*gain, 3000).
	}
	// hunt rival r  [BUILD must-build #1]
	else if (tgt >= PLC_TGT_RIVAL_BASE && tgt < PLC_TGT_CELL_BASE)
	{
		// TODO[BUILD]: entity r = findfloat(NULL, entnum, tgt-PLC_TGT_RIVAL_BASE);
		// if (IS_PLAYER(r)) navigation_routerating(this, r, ratingscale*this.plc_str_hunt, 5000);
		// Single-target rater — NOT havocbot_goalrating_enemyplayers' all-visible sweep.
	}
	// explore / grab at cell c  [BUILD must-build #2]
	else if (tgt >= PLC_TGT_CELL_BASE)
	{
		// TODO[BUILD]: rate g_waypoints within radius of the cell origin with
		// ratingscale*this.plc_str_explore — a NAMED cell, not roles.qc's random one.
	}

	// travel commitment  [BUILD must-build #4]: apply AFTER goalrating_end.
	//   navigation_goalrating_timeout_extend_if_needed(this, this.plc_str_commit);
	// spawn timing  [BUILD must-build #3]: at respawn calc, hold with plc_str_spawn:
	//   this.respawn_time = max(this.respawn_time, time + this.plc_str_spawn);
}

// ---------------------------------------------------------------------------
//  payload_strategy_step — one strategy step. Intended callsite: the EXISTING
//  lead-cart gate in payload_think (sv_payload.qc:414, `if (plc_cart_id == 0)`),
//  throttled to PLC_STRAT_TICK. Reuses the gate; adds no new think loop.
// ---------------------------------------------------------------------------
void payload_strategy_step(entity leadcart)
{
	payload_strategy_attach();

	float nclients = maxclients;
	FOREACH_CLIENT(IS_PLAYER(it), { payload_perceive(it); });
	payload_strategy_gather(nclients);
	payload_strategy_scatter(nclients);
	// The rater (havocbot_goalrating_strategy) runs later, per-bot, when each bot
	// holds the strategy token in havocbot_ai — reading the freshly scattered w_b.
}
