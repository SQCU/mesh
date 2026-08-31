// =============================================================================
//  sv_payload_strategy_io.qc  —  ACTIVE server I/O for the strategy stack
// -----------------------------------------------------------------------------
//  Real wiring of the STRATEGY-IO.md contract. Three flows:
//    (1) cartstate (GUARANTEED) + per-player observation rows gathered up to the
//        mesh every PLC_STRAT_TICK  (schemas B and A);
//    (2) perception-gated event rows (frustum + LOS + 2-cell), the ONLY producer
//        of enemy/item rows — never an omniscient findradius feed (schema C);
//    (3) the 8-column scatter pulled back down and applied as navigation_
//        routerating biases in havocbot_goalrating_strategy().
//
//  Determinism boundary (rl-training-spec §4): QC stages RAW numbers and reads an
//  already-integrated absolute weight. It never integrates and never learns.
//
//  AGENTS.md: no rejection filtering. The perception gate is a GAME rule
//  (occlusion), expressed by what a bot can see — not a capability refusal. Dead
//  bots have no viewpoint (a physics fact, early return), and a missing plan is
//  HELD (last absolute weight), never demoted to a no-mesh fallback.
// =============================================================================

#include "sv_payload_strategy_io.qh"

float plc_str_h_obs  = -1;   // gather handle A (per-player)
float plc_str_h_cart = -1;   // gather handle B (cartstate, GUARANTEED)
float plc_str_h_evt  = -1;   // gather handle C (perception ring)
float plc_str_h_sc   = -1;   // scatter handle (8-col responses)
float plc_str_seq    = 0;    // last completed plan sequence from mesh_poll

// Perception event-carrier pool: a contiguous block of edicts spawned back-to-
// back in attach() so mesh_gather can sweep it as one [firstedict,count] range.
entity plc_str_evt_pool[PLC_STR_EVT_ROWS];
float  plc_str_evt_w = 0;    // ring write cursor

// Per-team last-stamp for edge-triggered deposits (index by team, 0..NUM_TEAMS).
float plc_str_last_cell[NUM_TEAMS + 1];
float plc_str_last_kind[NUM_TEAMS + 1];
float plc_str_last_subj[NUM_TEAMS + 1];

// ---------------------------------------------------------------------------
//  payload_strategy_attach — lazy open of the four handles + event pool.
//  Retry-until-attached: if the coprocessor is not up yet we simply try again
//  next step (a hold, not a demotion). No no-mesh fallback branch exists.
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

	if (!plc_str_evt_pool[0])
	{
		// Spawn the ring as one contiguous edict block, nothing in between, so
		// etof(pool[0]) + i addresses row i for the whole gather sweep.
		for (int i = 0; i < PLC_STR_EVT_ROWS; ++i)
			plc_str_evt_pool[i] = spawn();
	}
}

// ---------------------------------------------------------------------------
//  payload_str_cell_of — V-cell id for a world origin (payload-spec §2.2.2).
//  Coarse 256u grid hash: a stable id per grid square. The 2-cell range gate
//  (payload_str_within2) works directly on grid coordinates, so the id here is
//  only a stamp for edge-trigger dedup and the observation CELL column.
// ---------------------------------------------------------------------------
float payload_str_cell_of(vector org)
{
	float gx = floor(org.x / 256);
	float gy = floor(org.y / 256);
	return (gx * 131 + gy) & 1023;
}

// Gate 3 (range): both origins within 2 grid cells (Chebyshev) on the 256u grid.
float payload_str_within2(vector a, vector b)
{
	float dx = fabs(floor(a.x / 256) - floor(b.x / 256));
	float dy = fabs(floor(a.y / 256) - floor(b.y / 256));
	return (max(dx, dy) <= 2);
}

// Deposit one edge-triggered event into the per-team ring. Only fires when the
// (cell,kind,subject) triple differs from this team's last stamp, so the ring
// carries events, not a per-tick dump.
void payload_str_deposit(float cell, float kind, float team, float subject, float value)
{
	float ti = team;
	if (ti < 0) ti = 0;
	if (ti > NUM_TEAMS) ti = NUM_TEAMS;

	if (cell == plc_str_last_cell[ti] && kind == plc_str_last_kind[ti]
	    && subject == plc_str_last_subj[ti])
		return;   // same event as last stamp — not an edge

	plc_str_last_cell[ti] = cell;
	plc_str_last_kind[ti] = kind;
	plc_str_last_subj[ti] = subject;

	entity e = plc_str_evt_pool[plc_str_evt_w];
	e.plc_ev_f_cell    = cell;
	e.plc_ev_f_kind    = kind;
	e.plc_ev_f_team    = team;
	e.plc_ev_f_subject = subject;
	e.plc_ev_f_value   = value;
	e.plc_ev_f_time    = time;

	plc_str_evt_w += 1;
	if (plc_str_evt_w >= PLC_STR_EVT_ROWS)
		plc_str_evt_w = 0;
}

// ---------------------------------------------------------------------------
//  payload_perceive — the frustum + LOS + 2-V-cell perception gate.
//  For one bot, scan enemy players and pickup items; deposit an edge-triggered
//  event ONLY when all three gates pass. This is the ONLY producer of enemy/item
//  rows — there is no omniscient feed anywhere in this path. Stealth is emergent:
//  an unobserved body or pickup is simply absent from the ring.
// ---------------------------------------------------------------------------
void payload_perceive(entity bot)
{
	if (IS_DEAD(bot))
		return;   // dead bots have no viewpoint — a physics fact, not a refusal

	vector eye = bot.origin + bot.view_ofs;
	float bcell = payload_str_cell_of(bot.origin);
	bot.plc_str_cell = bcell;
	float myteam = Team_TeamToIndex(bot.team);

	// Enemy players → rival_here (kind 3). Featurized ONLY here.
	FOREACH_CLIENT(IS_PLAYER(it) && it != bot && !IS_DEAD(it) && DIFF_TEAM(it, bot),
	{
		if (!payload_str_within2(bot.origin, it.origin))   // gate 3: range
			continue;
		if (!checkpvs(eye, it))                            // gate 1: frustum+PVS
			continue;
		traceline(eye, it.origin, MOVE_NOMONSTERS, bot);   // gate 2: line of sight
		if (trace_fraction < 1)
			continue;
		payload_str_deposit(payload_str_cell_of(it.origin), 3, myteam,
			etof(it), GetResource(it, RES_HEALTH));
	});

	// Pickup items → item_here (kind 1) / item_gone (kind 0).
	IL_EACH(g_items, it.bot_pickup,
	{
		vector o = (it.absmin + it.absmax) * 0.5;
		if (!payload_str_within2(bot.origin, o))
			continue;
		if (!checkpvs(eye, it))
			continue;
		traceline(eye, o, MOVE_NOMONSTERS, bot);
		if (trace_fraction < 1)
			continue;
		float kind  = it.solid ? 1 : 0;
		float phase = it.solid ? 0 : max(0, it.scheduledrespawntime - time);
		payload_str_deposit(payload_str_cell_of(o), kind, myteam, etof(it), phase);
	});
}

// ---------------------------------------------------------------------------
//  payload_strategy_gather — stage per-edict scratch columns, then gather A,B,C.
//  Cartstate (schema B) is staged and published UNCONDITIONALLY every step: it
//  is the guaranteed global feature (rl-training-spec §1).
// ---------------------------------------------------------------------------
void payload_strategy_gather(float nclients)
{
	// (B) cartstate rows — GUARANTEED. Stage every cart edict.
	for (int c = 0; c < payload_cart_count; ++c)
	{
		entity cart = payload_carts[c];
		cart.plc_cs_f_id       = cart.plc_cart_id;
		cart.plc_cs_f_depth    = (cart.plc_length > 0) ? cart.plc_s / cart.plc_length : 0;
		cart.plc_cs_f_length   = cart.plc_length;
		cart.plc_cs_f_ctrl     = cart.plc_ctrl;
		cart.plc_cs_f_speed    = cart.plc_speed_now;
		cart.plc_cs_f_idle     = cart.plc_idle;
		cart.plc_cs_f_bankmask = cart.plc_bank_team;
		cart.plc_cs_f_progress = cart.plc_banked;
	}
	if (payload_cart_count > 0)
	{
		float cbase = etof(payload_carts[0]);
		mesh_gather(plc_str_h_cart, PLC_CS_ID,       plc_cs_f_id,       cbase, payload_cart_count);
		mesh_gather(plc_str_h_cart, PLC_CS_DEPTH,    plc_cs_f_depth,    cbase, payload_cart_count);
		mesh_gather(plc_str_h_cart, PLC_CS_LENGTH,   plc_cs_f_length,   cbase, payload_cart_count);
		mesh_gather(plc_str_h_cart, PLC_CS_CTRL,     plc_cs_f_ctrl,     cbase, payload_cart_count);
		mesh_gather(plc_str_h_cart, PLC_CS_SPEED,    plc_cs_f_speed,    cbase, payload_cart_count);
		mesh_gather(plc_str_h_cart, PLC_CS_IDLE,     plc_cs_f_idle,     cbase, payload_cart_count);
		mesh_gather(plc_str_h_cart, PLC_CS_BANKMASK, plc_cs_f_bankmask, cbase, payload_cart_count);
		mesh_gather(plc_str_h_cart, PLC_CS_PROGRESS, plc_cs_f_progress, cbase, payload_cart_count);
	}

	// (A) per-player observation rows x_b — own, known self-state only.
	FOREACH_CLIENT(IS_PLAYER(it),
	{
		float nci = -1; float ncd = 999999999;
		for (int c = 0; c < payload_cart_count; ++c)
		{
			float d = vlen(payload_carts[c].origin - it.origin);
			if (d < ncd) { ncd = d; nci = c; }
		}
		float ammo = GetResource(it, RES_SHELLS) + GetResource(it, RES_BULLETS)
		           + GetResource(it, RES_ROCKETS) + GetResource(it, RES_CELLS)
		           + GetResource(it, RES_PLASMA)  + GetResource(it, RES_FUEL);

		it.plc_ob_f_id      = etof(it);
		it.plc_ob_f_team    = Team_TeamToIndex(it.team);
		it.plc_ob_f_health  = GetResource(it, RES_HEALTH);
		it.plc_ob_f_armor   = GetResource(it, RES_ARMOR);
		it.plc_ob_f_ammo    = ammo / 100;
		it.plc_ob_f_px      = it.origin.x / 1024;
		it.plc_ob_f_py      = it.origin.y / 1024;
		it.plc_ob_f_pz      = it.origin.z / 1024;
		it.plc_ob_f_vx      = it.velocity.x / 1024;
		it.plc_ob_f_vy      = it.velocity.y / 1024;
		it.plc_ob_f_vz      = it.velocity.z / 1024;
		it.plc_ob_f_weapons = STAT(WEAPONS, it).x;
		it.plc_ob_f_power   = max(0, it.strength_finished - time);
		it.plc_ob_f_tss     = (it.plc_str_spawn_stamp2 > 0) ? time - it.plc_str_spawn_stamp2 : 0;
		it.plc_ob_f_cell    = it.plc_str_cell;
		it.plc_ob_f_ncart   = nci;
		it.plc_ob_f_ncartd  = ncd / 1024;
		it.plc_ob_f_alive   = IS_DEAD(it) ? 0 : 1;
		it.plc_ob_f_control = IS_BOT_CLIENT(it) ? 1 : 0;
	});
	mesh_gather(plc_str_h_obs, PLC_OBS_ID,      plc_ob_f_id,      1, nclients);
	mesh_gather(plc_str_h_obs, PLC_OBS_TEAM,    plc_ob_f_team,    1, nclients);
	mesh_gather(plc_str_h_obs, PLC_OBS_HEALTH,  plc_ob_f_health,  1, nclients);
	mesh_gather(plc_str_h_obs, PLC_OBS_ARMOR,   plc_ob_f_armor,   1, nclients);
	mesh_gather(plc_str_h_obs, PLC_OBS_AMMO,    plc_ob_f_ammo,    1, nclients);
	mesh_gather(plc_str_h_obs, PLC_OBS_POS_X,   plc_ob_f_px,      1, nclients);
	mesh_gather(plc_str_h_obs, PLC_OBS_POS_Y,   plc_ob_f_py,      1, nclients);
	mesh_gather(plc_str_h_obs, PLC_OBS_POS_Z,   plc_ob_f_pz,      1, nclients);
	mesh_gather(plc_str_h_obs, PLC_OBS_VEL_X,   plc_ob_f_vx,      1, nclients);
	mesh_gather(plc_str_h_obs, PLC_OBS_VEL_Y,   plc_ob_f_vy,      1, nclients);
	mesh_gather(plc_str_h_obs, PLC_OBS_VEL_Z,   plc_ob_f_vz,      1, nclients);
	mesh_gather(plc_str_h_obs, PLC_OBS_WEAPONS, plc_ob_f_weapons, 1, nclients);
	mesh_gather(plc_str_h_obs, PLC_OBS_POWER,   plc_ob_f_power,   1, nclients);
	mesh_gather(plc_str_h_obs, PLC_OBS_TSS,     plc_ob_f_tss,     1, nclients);
	mesh_gather(plc_str_h_obs, PLC_OBS_CELL,    plc_ob_f_cell,    1, nclients);
	mesh_gather(plc_str_h_obs, PLC_OBS_NCART,   plc_ob_f_ncart,   1, nclients);
	mesh_gather(plc_str_h_obs, PLC_OBS_NCART_D, plc_ob_f_ncartd,  1, nclients);
	mesh_gather(plc_str_h_obs, PLC_OBS_ALIVE,   plc_ob_f_alive,   1, nclients);
	mesh_gather(plc_str_h_obs, PLC_OBS_CONTROL, plc_ob_f_control, 1, nclients);

	// (C) perception ring — rows already deposited this step by payload_perceive.
	float ebase = etof(plc_str_evt_pool[0]);
	mesh_gather(plc_str_h_evt, PLC_EVT_CELL,    plc_ev_f_cell,    ebase, PLC_STR_EVT_ROWS);
	mesh_gather(plc_str_h_evt, PLC_EVT_KIND,    plc_ev_f_kind,    ebase, PLC_STR_EVT_ROWS);
	mesh_gather(plc_str_h_evt, PLC_EVT_TEAM,    plc_ev_f_team,    ebase, PLC_STR_EVT_ROWS);
	mesh_gather(plc_str_h_evt, PLC_EVT_SUBJECT, plc_ev_f_subject, ebase, PLC_STR_EVT_ROWS);
	mesh_gather(plc_str_h_evt, PLC_EVT_VALUE,   plc_ev_f_value,   ebase, PLC_STR_EVT_ROWS);
	mesh_gather(plc_str_h_evt, PLC_EVT_TIME,    plc_ev_f_time,    ebase, PLC_STR_EVT_ROWS);

	mesh_publish(plc_str_h_cart, plc_str_seq, payload_cart_count);
	mesh_publish(plc_str_h_obs,  plc_str_seq, nclients);
	mesh_publish(plc_str_h_evt,  plc_str_seq, PLC_STR_EVT_ROWS);
}

// ---------------------------------------------------------------------------
//  payload_strategy_scatter — pull the latest completed plan down (8 cols).
//  The response columns are the absolute, mesh-integrated w_b. QC never
//  integrates (payload-spec §3.2). A missing newer plan is HELD, not demoted.
// ---------------------------------------------------------------------------
void payload_strategy_scatter(float nclients)
{
	float seq = mesh_poll(plc_str_h_sc);
	if (seq <= plc_str_seq)
		return;   // no newer plan yet — hold last absolute w_b (a hold)
	plc_str_seq = seq;

	FOREACH_CLIENT(IS_PLAYER(it),
	{
		it.plc_str_gain_prev = it.plc_str_gain;
		it.plc_str_lane_prev = it.plc_str_lane;
	});

	mesh_scatter(plc_str_h_sc, PLC_SC_TARGET,  plc_str_target,  1, nclients);
	mesh_scatter(plc_str_h_sc, PLC_SC_GAIN,    plc_str_gain,    1, nclients);
	mesh_scatter(plc_str_h_sc, PLC_SC_LANE,    plc_str_lane,    1, nclients);
	mesh_scatter(plc_str_h_sc, PLC_SC_HUNT,    plc_str_hunt,    1, nclients);
	mesh_scatter(plc_str_h_sc, PLC_SC_EXPLORE, plc_str_explore, 1, nclients);
	mesh_scatter(plc_str_h_sc, PLC_SC_COMMIT,  plc_str_commit,  1, nclients);
	mesh_scatter(plc_str_h_sc, PLC_SC_SPAWN,   plc_str_spawn,   1, nclients);
	mesh_scatter(plc_str_h_sc, PLC_SC_LEAD,    plc_str_lead,    1, nclients);
}

// ---------------------------------------------------------------------------
//  havocbot_goalrating_strategy — THIN TRANSLATOR: matmul target-weights become
//  navigation_routerating biases on REAL entities and nothing else. It decides
//  NOTHING (the committed target is a mesh output it reads) and computes NO
//  navigation (no node-finding, no path geometry, no "which point on the path").
//  Each objective in the vocabulary maps to one stock-navigable target entity:
//    push/suppress cart k -> the cart entity        (occupancy law resolves push/suppress)
//    contest item post p  -> the item entity
//    hunt rival r         -> the rival player entity
//    explore/gather cell c-> stock waypoint entities in the cell
//  Every output is a navigation_routerating base-rating bias — the single lever
//  (navigation.qc:1220). It NEVER writes skill/bot_*aim* fields, so skill-
//  orthogonality holds by construction (playerbot-interface §4).
// ---------------------------------------------------------------------------
void havocbot_goalrating_strategy(entity this, float ratingscale)
{
	float tgt  = this.plc_str_target;
	float gain = this.plc_str_gain;

	// push / suppress cart k: ONE routerating on the real CART entity. Push vs
	// suppress is NOT re-encoded here — it is resolved downstream by the cylinder
	// occupancy law (payload_occupancy, sv_payload.qc:102; design/cart-force-field.md):
	// a body present in the cart's cylinder pushes or contests by presence. The
	// adapter only says "want this cart, this much"; stock navigation finds the
	// route to it via the waypoint navmesh (navigation.qc:1220, moving-goal path).
	if (tgt >= PLC_TGT_CART_BASE && tgt < PLC_TGT_ITEM_BASE)
	{
		float k = tgt - PLC_TGT_CART_BASE;
		if (k >= 0 && k < payload_cart_count && gain != 0)
			navigation_routerating(this, payload_carts[k], ratingscale * gain, 5000);
	}
	// contest item post p: subject was written as etof(item); resolve it back.
	else if (tgt >= PLC_TGT_ITEM_BASE && tgt < PLC_TGT_RIVAL_BASE)
	{
		float subj = tgt - PLC_TGT_ITEM_BASE;
		IL_EACH(g_items, etof(it) == subj,
		{
			navigation_routerating(this, it, ratingscale * gain, 3000);
			break;
		});
	}
	// hunt rival r (must-build #1): single-target rate of the observed rival.
	else if (tgt >= PLC_TGT_RIVAL_BASE && tgt < PLC_TGT_CELL_BASE)
	{
		float subj = tgt - PLC_TGT_RIVAL_BASE;
		FOREACH_CLIENT(IS_PLAYER(it) && etof(it) == subj && DIFF_TEAM(it, this),
		{
			navigation_routerating(this, it, ratingscale * this.plc_str_hunt, 5000);
			break;
		});
	}
	// explore / grab at cell c (must-build #2): routerating the STOCK waypoint
	// entities whose hashed V-cell matches the selected cell. These are ordinary
	// g_waypoints navmesh nodes, not a project-specific path — the adapter names
	// target entities and stock navigation computes the route.
	else if (tgt >= PLC_TGT_CELL_BASE && this.plc_str_explore != 0)
	{
		float cell = tgt - PLC_TGT_CELL_BASE;
		IL_EACH(g_waypoints, payload_str_cell_of(it.origin) == cell,
		{
			navigation_routerating(this, it, ratingscale * this.plc_str_explore, 3000);
		});
	}

	// travel commitment (must-build #4): hold the strategy token this long so a
	// committed instrument is not re-diced every tick (navigation.qc:51 pattern).
	if (this.plc_str_commit > 0)
		this.bot_strategytime = max(this.bot_strategytime, time + this.plc_str_commit);

}

// ---------------------------------------------------------------------------
//  payload_strategy_step — one strategy step. Called from the EXISTING lead-cart
//  gate in payload_think (if plc_cart_id == 0), throttled to PLC_STRAT_TICK.
// ---------------------------------------------------------------------------
void payload_strategy_step(entity leadcart)
{
	payload_strategy_attach();

	float nclients = maxclients;
	FOREACH_CLIENT(IS_PLAYER(it), { payload_perceive(it); });
	payload_strategy_gather(nclients);
	payload_strategy_scatter(nclients);
	// The rater (havocbot_goalrating_strategy) runs later, per bot, when each bot
	// holds the strategy token in havocbot_ai — reading the freshly scattered w_b.
}
