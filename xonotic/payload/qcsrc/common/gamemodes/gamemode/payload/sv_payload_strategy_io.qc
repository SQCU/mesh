
#include "sv_payload_strategy_io.qh"

float plc_str_h_obs  = -1;
float plc_str_h_cart = -1;
float plc_str_h_evt  = -1;
float plc_str_h_sc   = -1;
float plc_str_req_seq  = 0;
float plc_str_resp_seq = 0;

entity plc_str_evt_pool[PLC_STR_EVT_ROWS];
entity plc_str_cart_pool[PLC_MAX_CARTS];
float  plc_str_evt_w = 0;
float  plc_str_evt_base = 0;
float  plc_str_cart_base = 0;

float plc_str_last_cell[NUM_TEAMS + 1];
float plc_str_last_kind[NUM_TEAMS + 1];
float plc_str_last_subj[NUM_TEAMS + 1];

// mesh_gather(h, col, .field, firstedict, count) reads `count` CONSECUTIVE
// edicts (bridge/engine/mesh_ipc.c:305-306:
//     m->req[row*width+col] = prog->edictsfields[(first+row)*stride + fld]).
// A staging pool is therefore only a valid gather source if its edict numbers
// are consecutive. spawn() gives no such guarantee: PRVM_ED_Alloc
// (darkplaces prvm_edict.c:267) scans from `reserved_edicts+1` upward and
// returns the FIRST reusable free slot, so a burst of spawn() calls after map
// load lands in whatever holes the map's self-removing entities left behind.
// This builds the run explicitly: keep spawning and restart the run whenever
// the edict number is not prev+1. Free holes are finite, so once they are
// consumed PRVM_ED_Alloc appends and the run completes. Rejected edicts are
// simply never removed, so they can never be recycled underneath the pool.
float payload_str_pool_run(float want, float is_cart)
{
	float have = 0;
	float prev = 0;
	float tries = 0;
	while (have < want && tries < 8192)
	{
		entity e = spawn();
		tries += 1;
		if (have > 0 && etof(e) != prev + 1)
			have = 0;
		if (is_cart)
			plc_str_cart_pool[have] = e;
		else
			plc_str_evt_pool[have] = e;
		prev = etof(e);
		have += 1;
	}
	if (have < want)
	{
		LOG_INFO(strcat("payload: FAILED to allocate a contiguous ",
			ftos(want), "-edict strategy pool in ", ftos(tries), " spawns"));
		return 0;
	}
	return etof(is_cart ? plc_str_cart_pool[0] : plc_str_evt_pool[0]);
}

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
		plc_str_evt_base  = payload_str_pool_run(PLC_STR_EVT_ROWS, 0);
		plc_str_cart_base = payload_str_pool_run(PLC_MAX_CARTS, 1);
		// Contiguity is the gather precondition; state it as a measurement, not
		// as a comment. Both runs are re-verified here against the live edict
		// numbers so a real run's log carries the proof.
		float okc = 1, oke = 1;
		for (int i = 0; i < PLC_STR_EVT_ROWS; ++i)
			if (etof(plc_str_evt_pool[i]) != plc_str_evt_base + i) oke = 0;
		for (int i = 0; i < PLC_MAX_CARTS; ++i)
			if (etof(plc_str_cart_pool[i]) != plc_str_cart_base + i) okc = 0;
		// One sprintf, not a strcat chain: DP's VM_strcat takes at most 8
		// arguments and silently drops the rest.
		LOG_INFO(sprintf("payload: [PLCPOOL] evt_base %d rows %d contiguous %d",
			plc_str_evt_base, PLC_STR_EVT_ROWS, oke));
		LOG_INFO(sprintf("payload: [PLCPOOL] cart_base %d rows %d contiguous %d obs_base 1 rows %d",
			plc_str_cart_base, PLC_MAX_CARTS, okc, maxclients));
	}
}

float payload_str_cell_of(vector org)
{
	float gx = floor(org.x / 256);
	float gy = floor(org.y / 256);
	return (gx * 131 + gy) & 1023;
}

float payload_str_within2(vector a, vector b)
{
	float dx = fabs(floor(a.x / 256) - floor(b.x / 256));
	float dy = fabs(floor(a.y / 256) - floor(b.y / 256));
	return (max(dx, dy) <= 2);
}

void payload_str_deposit(float cell, float kind, float team, float subject, float value)
{
	float ti = team;
	if (ti < 0) ti = 0;
	if (ti > NUM_TEAMS) ti = NUM_TEAMS;

	if (cell == plc_str_last_cell[ti] && kind == plc_str_last_kind[ti]
	    && subject == plc_str_last_subj[ti])
		return;

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

void payload_perceive(entity bot)
{
	if (IS_DEAD(bot))
		return;

	vector eye = bot.origin + bot.view_ofs;
	float bcell = payload_str_cell_of(bot.origin);
	bot.plc_str_cell = bcell;
	float myteam = Team_TeamToIndex(bot.team);

	FOREACH_CLIENT(IS_PLAYER(it) && it != bot && !IS_DEAD(it) && DIFF_TEAM(it, bot),
	{
		if (!payload_str_within2(bot.origin, it.origin))
			continue;
		if (!checkpvs(eye, it))
			continue;
		traceline(eye, it.origin, MOVE_NOMONSTERS, bot);
		if (trace_fraction < 1)
			continue;
		payload_str_deposit(payload_str_cell_of(it.origin), 3, myteam,
			etof(it), GetResource(it, RES_HEALTH));
	});

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

// Log the EXACT rows the mesh is about to carry, read back off the staged
// fields rather than recomputed, so the log IS the gather source. One line per
// row; tools/measure.py parses these into the per-player observation table
// (AGENDA E9/E10) and reconstructs the instrument descriptors z and the
// relation rows from them. Gated on g_payload_strategy_log because it is one
// line per player per strategy tick.
void payload_strategy_log(float nclients)
{
	LOG_INFO(strcat("[PLCPUB] ", ftos(plc_str_req_seq), " ", ftos(time),
		" carts ", ftos(payload_cart_count), " clients ", ftos(nclients),
		" evtrows ", ftos(PLC_STR_EVT_ROWS)));

	for (int c = 0; c < payload_cart_count; ++c)
	{
		entity row = plc_str_cart_pool[c];
		// At most 7 values per sprintf: the QuakeC calling convention carries
		// MAX_PARMS = 8 (darkplaces pr_comp.h:143) and the format string is
		// one of them, so an 8th value is silently dropped.
		LOG_INFO(strcat(
			sprintf("[PLCCART] %d %d %g %g %g", plc_str_req_seq, c,
				row.plc_cs_f_id, row.plc_cs_f_depth, row.plc_cs_f_length),
			sprintf(" %g %g %g %g %g",
				row.plc_cs_f_ctrl, row.plc_cs_f_speed, row.plc_cs_f_idle,
				row.plc_cs_f_bankmask, row.plc_cs_f_progress),
			sprintf(" %g %g %g",
				row.plc_cs_f_px, row.plc_cs_f_py, row.plc_cs_f_pz)));
	}

	FOREACH_CLIENT(IS_PLAYER(it),
	{
		string a = sprintf("[PLCOBS] %d %d %g %g %g", plc_str_req_seq, etof(it),
			it.plc_ob_f_id, it.plc_ob_f_team, it.plc_ob_f_health);
		string b = sprintf(" %g %g %g %g %g %g",
			it.plc_ob_f_armor, it.plc_ob_f_ammo, it.plc_ob_f_px,
			it.plc_ob_f_py, it.plc_ob_f_pz, it.plc_ob_f_vx);
		string c2 = sprintf(" %g %g %g %g %g %g",
			it.plc_ob_f_vy, it.plc_ob_f_vz, it.plc_ob_f_weapons,
			it.plc_ob_f_power, it.plc_ob_f_tss, it.plc_ob_f_cell);
		string d = sprintf(" %g %g %g %g",
			it.plc_ob_f_ncart, it.plc_ob_f_ncartd,
			it.plc_ob_f_alive, it.plc_ob_f_control);
		string e = sprintf(" %g %g %g %g %g %g",
			it.plc_ob_f_applied_target, it.plc_ob_f_target_resolved,
			it.plc_ob_f_goal_target, it.plc_ob_f_goal_distance,
			it.plc_ob_f_goal_match, it.plc_ob_f_target_touch);
		LOG_INFO(strcat(a, b, c2, d, e));
	});

	for (int r = 0; r < PLC_STR_EVT_ROWS; ++r)
	{
		entity e = plc_str_evt_pool[r];
		if (e.plc_ev_f_time == 0 && e.plc_ev_f_kind == 0 && e.plc_ev_f_subject == 0)
			continue;
		LOG_INFO(strcat(
			sprintf("[PLCEVT] %d %d %g %g %g", plc_str_req_seq, r,
				e.plc_ev_f_cell, e.plc_ev_f_kind, e.plc_ev_f_team),
			sprintf(" %g %g %g", e.plc_ev_f_subject, e.plc_ev_f_value,
				e.plc_ev_f_time)));
	}
}

float plc_str_link_cursor = 0;
float plc_str_link_pass = 0;
float plc_str_link_rows = 0;

// Stream the map's V-cell adjacency onto the perception ring: for each stock
// waypoint, every outgoing link whose endpoint hashes to a DIFFERENT V-cell is
// one navigable cell->cell edge. `waypoint_get_link` reads the stock nav graph
// (server/bot/default/waypoints.qh:28), so this reports the engine's own
// navigability and introduces no second navigation definition (SPEC 12).
//
// The sweep never latches "done": g_waypoints is still filling during the first
// seconds of a match (a one-shot pass caught 33 of 469 waypoints on a real
// fused run), and a map can gain waypoints later. It keeps cycling instead, at
// a rate low enough that link rows never crowd the perception events out of the
// 256-row ring.
void payload_emit_cell_links()
{
	float index = 0;
	float processed = 0;
	float total = 0;
	IL_EACH(g_waypoints, true,
	{
		total += 1;
		if (index >= plc_str_link_cursor && processed < PLC_STR_LINKS_PER_STEP)
		{
			float ca = payload_str_cell_of(it.origin);
			for (int j = 0; j < 32; ++j)
			{
				entity link = waypoint_get_link(it, j);
				if (!link)
					continue;
				float cb = payload_str_cell_of(link.origin);
				if (cb == ca)
					continue;
				entity row = plc_str_evt_pool[plc_str_evt_w];
				row.plc_ev_f_cell    = ca;
				row.plc_ev_f_kind    = PLC_EVT_KIND_CELL_LINK;
				row.plc_ev_f_team    = 0;
				row.plc_ev_f_subject = cb;
				row.plc_ev_f_value   = vlen(link.origin - it.origin) / 1024;
				row.plc_ev_f_time    = time;
				plc_str_evt_w += 1;
				if (plc_str_evt_w >= PLC_STR_EVT_ROWS)
					plc_str_evt_w = 0;
				plc_str_link_rows += 1;
			}
			processed += 1;
		}
		index += 1;
	});

	plc_str_link_cursor += processed;
	if (plc_str_link_cursor >= total)
	{
		plc_str_link_cursor = 0;
		plc_str_link_pass += 1;
		LOG_INFO(sprintf("payload: [PLCLINK] pass %d over %d waypoints, %d cell-link rows emitted",
			plc_str_link_pass, total, plc_str_link_rows));
	}
}

void payload_strategy_gather(float nclients)
{
	for (int c = 0; c < payload_cart_count; ++c)
	{
		entity cart = payload_carts[c];
		entity row = plc_str_cart_pool[c];
		row.plc_cs_f_id       = cart.plc_cart_id;
		row.plc_cs_f_depth    = (cart.plc_length > 0) ? cart.plc_s / cart.plc_length : 0;
		row.plc_cs_f_length   = cart.plc_length;
		row.plc_cs_f_ctrl     = cart.plc_ctrl;
		row.plc_cs_f_speed    = cart.plc_speed_now;
		row.plc_cs_f_idle     = cart.plc_idle;
		row.plc_cs_f_bankmask = cart.plc_bank_team;
		row.plc_cs_f_progress = cart.plc_banked;
		// Cart world origin, same 1/1024 scaling as the per-player position
		// columns. Without it no consumer can price the travel a cart
		// assignment costs (AGENDA F4).
		// A cart is a brush model: sv_payload.qc sets `view_ofs = mins` and
		// parks it at `payload_pos(...) - view_ofs`, so its WORLD position on
		// the golden path is origin + view_ofs, not origin.
		vector cpos = cart.origin + cart.view_ofs;
		row.plc_cs_f_px      = cpos.x / 1024;
		row.plc_cs_f_py      = cpos.y / 1024;
		row.plc_cs_f_pz      = cpos.z / 1024;
	}
	if (payload_cart_count > 0)
	{
		float cbase = plc_str_cart_base;
		mesh_gather(plc_str_h_cart, PLC_CS_ID,       plc_cs_f_id,       cbase, payload_cart_count);
		mesh_gather(plc_str_h_cart, PLC_CS_DEPTH,    plc_cs_f_depth,    cbase, payload_cart_count);
		mesh_gather(plc_str_h_cart, PLC_CS_LENGTH,   plc_cs_f_length,   cbase, payload_cart_count);
		mesh_gather(plc_str_h_cart, PLC_CS_CTRL,     plc_cs_f_ctrl,     cbase, payload_cart_count);
		mesh_gather(plc_str_h_cart, PLC_CS_SPEED,    plc_cs_f_speed,    cbase, payload_cart_count);
		mesh_gather(plc_str_h_cart, PLC_CS_IDLE,     plc_cs_f_idle,     cbase, payload_cart_count);
		mesh_gather(plc_str_h_cart, PLC_CS_BANKMASK, plc_cs_f_bankmask, cbase, payload_cart_count);
		mesh_gather(plc_str_h_cart, PLC_CS_PROGRESS, plc_cs_f_progress, cbase, payload_cart_count);
		mesh_gather(plc_str_h_cart, PLC_CS_POS_X,    plc_cs_f_px,       cbase, payload_cart_count);
		mesh_gather(plc_str_h_cart, PLC_CS_POS_Y,    plc_cs_f_py,       cbase, payload_cart_count);
		mesh_gather(plc_str_h_cart, PLC_CS_POS_Z,    plc_cs_f_pz,       cbase, payload_cart_count);
	}

	FOREACH_CLIENT(IS_PLAYER(it),
	{
		float nci = -1; float ncd = 999999999;
		for (int c = 0; c < payload_cart_count; ++c)
		{
			// Same brush-model correction: comparing a player's origin to a
			// cart's UNOFFSET origin made the nearest-cart columns wrong by
			// the cart brush's mins on every row of every run so far.
			float d = vlen((payload_carts[c].origin + payload_carts[c].view_ofs) - it.origin);
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
		it.plc_ob_f_applied_target = it.plc_str_applied_target;
		it.plc_ob_f_target_resolved = it.plc_str_resolved;
		it.plc_ob_f_goal_target = it.plc_str_goal_target;
		it.plc_ob_f_goal_distance = -1;
		if (it.plc_str_goal_entity != NULL)
			it.plc_ob_f_goal_distance = vlen(((it.plc_str_goal_entity.absmin
				+ it.plc_str_goal_entity.absmax) * 0.5) - it.origin) / 1024;
		it.plc_ob_f_goal_match = (it.plc_str_applied_target >= 0
			&& it.plc_str_goal_target == it.plc_str_applied_target);
		it.plc_ob_f_target_touch = 0;
		if (it.plc_ob_f_goal_match && it.plc_str_applied_target >= PLC_TGT_CELL_BASE)
			it.plc_ob_f_target_touch = (payload_str_cell_of(it.origin)
				== it.plc_str_applied_target - PLC_TGT_CELL_BASE);
		if (it.plc_ob_f_goal_match && it.plc_str_applied_target < PLC_TGT_CELL_BASE
			&& it.plc_ob_f_goal_distance >= 0 && it.plc_ob_f_goal_distance <= 0.16)
			it.plc_ob_f_target_touch = 1;
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
	mesh_gather(plc_str_h_obs, PLC_OBS_APPLIED_TARGET, plc_ob_f_applied_target, 1, nclients);
	mesh_gather(plc_str_h_obs, PLC_OBS_TARGET_RESOLVED, plc_ob_f_target_resolved, 1, nclients);
	mesh_gather(plc_str_h_obs, PLC_OBS_GOAL_TARGET, plc_ob_f_goal_target, 1, nclients);
	mesh_gather(plc_str_h_obs, PLC_OBS_GOAL_DISTANCE, plc_ob_f_goal_distance, 1, nclients);
	mesh_gather(plc_str_h_obs, PLC_OBS_GOAL_MATCH, plc_ob_f_goal_match, 1, nclients);
	mesh_gather(plc_str_h_obs, PLC_OBS_TARGET_TOUCH, plc_ob_f_target_touch, 1, nclients);

	float ebase = plc_str_evt_base;
	mesh_gather(plc_str_h_evt, PLC_EVT_CELL,    plc_ev_f_cell,    ebase, PLC_STR_EVT_ROWS);
	mesh_gather(plc_str_h_evt, PLC_EVT_KIND,    plc_ev_f_kind,    ebase, PLC_STR_EVT_ROWS);
	mesh_gather(plc_str_h_evt, PLC_EVT_TEAM,    plc_ev_f_team,    ebase, PLC_STR_EVT_ROWS);
	mesh_gather(plc_str_h_evt, PLC_EVT_SUBJECT, plc_ev_f_subject, ebase, PLC_STR_EVT_ROWS);
	mesh_gather(plc_str_h_evt, PLC_EVT_VALUE,   plc_ev_f_value,   ebase, PLC_STR_EVT_ROWS);
	mesh_gather(plc_str_h_evt, PLC_EVT_TIME,    plc_ev_f_time,    ebase, PLC_STR_EVT_ROWS);

	plc_str_req_seq += 1;
	if (autocvar_g_payload_strategy_log)
		payload_strategy_log(nclients);

	mesh_publish(plc_str_h_cart, plc_str_req_seq, payload_cart_count);
	mesh_publish(plc_str_h_obs,  plc_str_req_seq, nclients);
	mesh_publish(plc_str_h_evt,  plc_str_req_seq, PLC_STR_EVT_ROWS);
}

void payload_strategy_scatter(float nclients)
{
	float seq = mesh_poll(plc_str_h_sc);
	if (seq <= plc_str_resp_seq)
		return;
	plc_str_resp_seq = seq;

	mesh_scatter(plc_str_h_sc, PLC_SC_TARGET,  plc_str_target,  1, nclients);
	mesh_scatter(plc_str_h_sc, PLC_SC_GAIN,    plc_str_gain,    1, nclients);
	mesh_scatter(plc_str_h_sc, PLC_SC_COMMIT,  plc_str_commit,  1, nclients);
	mesh_scatter(plc_str_h_sc, PLC_SC_SPAWN,   plc_str_spawn,   1, nclients);
}

float payload_strategy_entity_target(entity e)
{
	if (!e)
		return -1;
	for (int c = 0; c < payload_cart_count; ++c)
		if (e == payload_carts[c])
			return PLC_TGT_CART_BASE + payload_carts[c].plc_cart_id;
	if (IS_PLAYER(e))
		return PLC_TGT_RIVAL_BASE + etof(e);
	if (e.bot_pickup)
		return PLC_TGT_ITEM_BASE + etof(e);
	if (e.classname == "waypoint")
		return PLC_TGT_CELL_BASE + payload_str_cell_of(e.origin);
	return -1;
}

void havocbot_goalrating_strategy(entity this, float ratingscale)
{
	float tgt  = this.plc_str_target;
	float gain = this.plc_str_gain;
	this.plc_str_applied_target = tgt;
	this.plc_str_resolved = 0;

	if (tgt >= PLC_TGT_CART_BASE && tgt < PLC_TGT_ITEM_BASE)
	{
		float k = tgt - PLC_TGT_CART_BASE;
		if (k >= 0 && k < payload_cart_count && gain != 0)
		{
			this.plc_str_resolved = 1;
			navigation_routerating(this, payload_carts[k], ratingscale * gain, 5000);
		}
	}
	else if (tgt >= PLC_TGT_ITEM_BASE && tgt < PLC_TGT_RIVAL_BASE)
	{
		float subj = tgt - PLC_TGT_ITEM_BASE;
		IL_EACH(g_items, etof(it) == subj,
		{
			this.plc_str_resolved = 1;
			navigation_routerating(this, it, ratingscale * gain, 3000);
			break;
		});
	}
	else if (tgt >= PLC_TGT_RIVAL_BASE && tgt < PLC_TGT_CELL_BASE)
	{
		float subj = tgt - PLC_TGT_RIVAL_BASE;
		FOREACH_CLIENT(IS_PLAYER(it) && etof(it) == subj && DIFF_TEAM(it, this),
		{
			this.plc_str_resolved = 1;
			navigation_routerating(this, it, ratingscale * gain, 5000);
			break;
		});
	}
	else if (tgt >= PLC_TGT_CELL_BASE && gain != 0)
	{
		float cell = tgt - PLC_TGT_CELL_BASE;
		IL_EACH(g_waypoints, payload_str_cell_of(it.origin) == cell,
		{
			this.plc_str_resolved = 1;
			navigation_routerating(this, it, ratingscale * gain, 3000);
		});
	}
}

void payload_strategy_step(entity leadcart)
{
	payload_strategy_attach();

	float nclients = maxclients;
	payload_emit_cell_links();
	FOREACH_CLIENT(IS_PLAYER(it), { payload_perceive(it); });
	payload_strategy_gather(nclients);
	payload_strategy_scatter(nclients);
}
