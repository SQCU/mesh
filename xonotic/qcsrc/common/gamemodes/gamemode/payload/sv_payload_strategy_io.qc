
#include "sv_payload_strategy_io.qh"

float plc_str_h_obs  = -1;
float plc_str_h_cart = -1;
float plc_str_h_evt  = -1;
float plc_str_h_sc   = -1;
float plc_str_req_seq  = 0;
float plc_str_resp_seq = 0;

entity plc_str_cart_pool[PLC_MAX_CARTS];
float  plc_str_evt_w = 0;
entity plc_str_evt_head;
entity plc_str_evt_tail;
float  plc_str_cart_base = 0;
const float PLC_STR_CELL_EXTENT = 256;

void payload_str_measure_reset(entity player)
{
	player.plc_str_outcome_a_used = 0;
	player.plc_str_outcome_a_seq = 0;
	player.plc_str_outcome_a_enemy_damage_dealt = 0;
	player.plc_str_outcome_a_enemy_damage_taken = 0;
	player.plc_str_outcome_a_enemy_kills = 0;
	player.plc_str_outcome_a_deaths = 0;
	player.plc_str_outcome_a_pickups = 0;
	player.plc_str_outcome_a_cart_push = 0;
	player.plc_str_outcome_a_cart_contest = 0;
	player.plc_str_outcome_b_used = 0;
	player.plc_str_outcome_b_seq = 0;
	player.plc_str_outcome_b_enemy_damage_dealt = 0;
	player.plc_str_outcome_b_enemy_damage_taken = 0;
	player.plc_str_outcome_b_enemy_kills = 0;
	player.plc_str_outcome_b_deaths = 0;
	player.plc_str_outcome_b_pickups = 0;
	player.plc_str_outcome_b_cart_push = 0;
	player.plc_str_outcome_b_cart_contest = 0;
}

void payload_strategy_reset_player(entity player)
{
	player.plc_str_target_kind = PLC_TARGET_NONE;
	player.plc_str_instrument_kind = PLC_INSTRUMENT_NONE;
	player.plc_str_target_id = -1;
	player.plc_str_target_cell_x = 0;
	player.plc_str_target_cell_y = 0;
	player.plc_str_applied_target_kind = PLC_TARGET_NONE;
	player.plc_str_applied_instrument_kind = PLC_INSTRUMENT_NONE;
	player.plc_str_applied_target_id = -1;
	player.plc_str_applied_target_cell_x = 0;
	player.plc_str_applied_target_cell_y = 0;
	player.plc_str_applied_gain = 0;
	player.plc_str_goal_target_kind = PLC_TARGET_NONE;
	player.plc_str_goal_target_id = -1;
	player.plc_str_goal_target_cell_x = 0;
	player.plc_str_goal_target_cell_y = 0;
	player.plc_str_response_seq = 0;
	player.plc_str_response_time = 0;
	player.plc_str_route_seq = 0;
	player.plc_str_route_latency = 0;
	player.plc_str_goal_seq = 0;
	player.plc_str_goal_latency = 0;
	player.plc_str_touch_seq = 0;
	player.plc_str_touch_latency = 0;
	player.plc_str_enemy_damage_dealt = 0;
	player.plc_str_enemy_damage_taken = 0;
	player.plc_str_enemy_kills = 0;
	player.plc_str_deaths = 0;
	player.plc_str_pickups = 0;
	player.plc_str_cart_push = 0;
	player.plc_str_cart_contest = 0;
	payload_str_measure_reset(player);
}

void payload_str_measure(entity player, float coordinate, float amount)
{
	bool a = !player.plc_str_outcome_a_used || player.plc_str_outcome_a_seq == player.plc_str_route_seq;
	if (a && !player.plc_str_outcome_a_used)
	{
		player.plc_str_outcome_a_used = 1;
		player.plc_str_outcome_a_seq = player.plc_str_route_seq;
	}
	if (!a && !player.plc_str_outcome_b_used)
	{
		player.plc_str_outcome_b_used = 1;
		player.plc_str_outcome_b_seq = player.plc_str_route_seq;
	}
	if (a)
	{
		if (coordinate == PLC_OUTCOME_ENEMY_DAMAGE_DEALT) player.plc_str_outcome_a_enemy_damage_dealt += amount;
		else if (coordinate == PLC_OUTCOME_ENEMY_DAMAGE_TAKEN) player.plc_str_outcome_a_enemy_damage_taken += amount;
		else if (coordinate == PLC_OUTCOME_ENEMY_KILLS) player.plc_str_outcome_a_enemy_kills += amount;
		else if (coordinate == PLC_OUTCOME_DEATHS) player.plc_str_outcome_a_deaths += amount;
		else if (coordinate == PLC_OUTCOME_PICKUPS) player.plc_str_outcome_a_pickups += amount;
		else if (coordinate == PLC_OUTCOME_CART_PUSH) player.plc_str_outcome_a_cart_push += amount;
		else player.plc_str_outcome_a_cart_contest += amount;
	}
	else
	{
		if (coordinate == PLC_OUTCOME_ENEMY_DAMAGE_DEALT) player.plc_str_outcome_b_enemy_damage_dealt += amount;
		else if (coordinate == PLC_OUTCOME_ENEMY_DAMAGE_TAKEN) player.plc_str_outcome_b_enemy_damage_taken += amount;
		else if (coordinate == PLC_OUTCOME_ENEMY_KILLS) player.plc_str_outcome_b_enemy_kills += amount;
		else if (coordinate == PLC_OUTCOME_DEATHS) player.plc_str_outcome_b_deaths += amount;
		else if (coordinate == PLC_OUTCOME_PICKUPS) player.plc_str_outcome_b_pickups += amount;
		else if (coordinate == PLC_OUTCOME_CART_PUSH) player.plc_str_outcome_b_cart_push += amount;
		else player.plc_str_outcome_b_cart_contest += amount;
	}
}

float payload_str_cart_pool_run()
{
	float have = 0;
	float prev = 0;
	while (have < PLC_MAX_CARTS)
	{
		entity e = spawn();
		if (have > 0 && etof(e) != prev + 1)
			have = 0;
		plc_str_cart_pool[have] = e;
		e.plc_cs_f_pool = etof(e);
		prev = etof(e);
		have += 1;
	}
	return prev - PLC_MAX_CARTS + 1;
}

void payload_strategy_cart_pool()
{
	bool valid = plc_str_cart_base > 0;
	for (int i = 0; i < PLC_MAX_CARTS && valid; ++i)
		if (!plc_str_cart_pool[i]
			|| etof(plc_str_cart_pool[i]) != plc_str_cart_base + i
			|| plc_str_cart_pool[i].plc_cs_f_pool != etof(plc_str_cart_pool[i]))
			valid = false;
	if (!valid)
	{
		plc_str_cart_base = payload_str_cart_pool_run();
		LOG_INFO(sprintf("payload: [PLCPOOL] cart_base %d rows %d capacity %d contiguous %d obs_base 1 rows %d",
			plc_str_cart_base, payload_cart_count, PLC_MAX_CARTS,
			boolean(plc_str_cart_base), maxclients));
	}
}

void payload_strategy_attach()
{
	if (plc_str_h_obs < 0)
		plc_str_h_obs  = mesh_open(PLC_STR_NODE, MESH_KIND_OBSERVATION, MESH_KIND_NONE, PLC_STR_OBS_W, maxclients);
	if (plc_str_h_cart < 0)
		plc_str_h_cart = mesh_open(PLC_STR_NODE, MESH_KIND_CART, MESH_KIND_NONE, PLC_STR_CART_W, PLC_MAX_CARTS);
	if (plc_str_h_evt < 0)
		plc_str_h_evt  = mesh_open(PLC_STR_NODE, MESH_KIND_EVENT, MESH_KIND_NONE, PLC_STR_EVT_W, max(1, maxclients));
	if (plc_str_h_sc < 0)
		plc_str_h_sc   = mesh_open(PLC_STR_NODE, MESH_KIND_NONE, MESH_KIND_STRATEGY, PLC_STR_RESP_W, maxclients);

	payload_strategy_cart_pool();
}

vector payload_str_cell_of(vector org)
{
	return vec2(floor(org.x / PLC_STR_CELL_EXTENT), floor(org.y / PLC_STR_CELL_EXTENT));
}

float payload_str_within2(vector a, vector b)
{
	float dx = fabs(floor(a.x / PLC_STR_CELL_EXTENT) - floor(b.x / PLC_STR_CELL_EXTENT));
	float dy = fabs(floor(a.y / PLC_STR_CELL_EXTENT) - floor(b.y / PLC_STR_CELL_EXTENT));
	return (max(dx, dy) <= 2);
}

void payload_str_deposit(float kind, float observer, float team, float subject, vector position, vector cell, vector target_cell, float respawn_time, float health, float link_length, float amount, float response_seq)
{
	entity e = spawn();
	e.plc_ev_f_kind    = kind;
	e.plc_ev_f_time    = time;
	e.plc_ev_f_observer = observer;
	e.plc_ev_f_team    = team;
	e.plc_ev_f_subject = subject;
	e.plc_ev_f_cell_x = cell.x;
	e.plc_ev_f_cell_y = cell.y;
	e.plc_ev_f_target_cell_x = target_cell.x;
	e.plc_ev_f_target_cell_y = target_cell.y;
	e.plc_ev_f_px      = position.x;
	e.plc_ev_f_py      = position.y;
	e.plc_ev_f_pz      = position.z;
	e.plc_ev_f_respawn_time = respawn_time;
	e.plc_ev_f_health = health;
	e.plc_ev_f_link_length = link_length;
	e.plc_ev_f_amount = amount;
	e.plc_ev_f_response_seq = response_seq;
	e.plc_ev_next = NULL;
	if (plc_str_evt_tail)
		plc_str_evt_tail.plc_ev_next = e;
	else
		plc_str_evt_head = e;
	plc_str_evt_tail = e;
	plc_str_evt_w += 1;
}

void payload_str_outcome(entity actor, float kind, entity subject, float value)
{
	payload_str_deposit(kind, etof(actor), Team_TeamToIndex(actor.team),
		etof(subject), actor.origin, payload_str_cell_of(actor.origin), '0 0 0', 0, 0, 0, value,
		actor.plc_str_route_seq);
}

void payload_str_round(float winner)
{
	payload_str_deposit(PLC_EVT_KIND_ROUND, 0, winner, 0, '0 0 0', '0 0 0', '0 0 0', 0, 0, 0, 1, 0);
}

void payload_perceive(entity bot)
{
	if (IS_DEAD(bot))
		return;

	vector eye = bot.origin + bot.view_ofs;
	vector bcell = payload_str_cell_of(bot.origin);
	bot.plc_str_cell_x = bcell.x;
	bot.plc_str_cell_y = bcell.y;
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
		payload_str_deposit(PLC_EVT_KIND_RIVAL_HERE, etof(bot), myteam, etof(it), it.origin,
			payload_str_cell_of(it.origin), '0 0 0', 0, GetResource(it, RES_HEALTH), 0, 0, 0);
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
		float kind = it.solid ? PLC_EVT_KIND_ITEM_HERE : PLC_EVT_KIND_ITEM_GONE;
		payload_str_deposit(kind, etof(bot), myteam, etof(it), o,
			payload_str_cell_of(o), '0 0 0', it.scheduledrespawntime, 0, 0, 0, 0);
	});
}

void payload_vcell_log()
{
	LOG_INFOF("[BOTWARP] rows %.0f kernel_rows %.0f cells %.0f waves %.0f working_sets %.0f peak_rows %.0f\n",
		bot_controller_stat_read(0), bot_controller_stat_read(1),
		bot_controller_stat_read(3), bot_controller_stat_read(2),
		bot_controller_stat_read(6),
		bot_controller_stat_read(7));
	LOG_INFOF("[VCELL] barriers %.0f total_rows %.0f total_kernel_rows %.0f wave_span %.0f cell_extent %.3f causal_cells %.0f\n",
		bot_controller_stat_read(4), bot_controller_stat_read(5),
		bot_controller_stat_read(11), bot_controller_stat_read(8),
		bot_controller_stat_read(9), bot_controller_stat_read(10));
	LOG_INFOF("[VCELLMERGE] input_coordinates %.0f output_coordinates %.0f buffer_bytes %.0f\n",
		bot_controller_stat_read(12), bot_controller_stat_read(13),
		bot_controller_stat_read(14));
	LOG_INFOF("[BOTQC] staged_rows %.0f staged_total %.0f scalar_total %.0f fallback_total %.0f\n",
		bot_controller_qc_stat(0), bot_controller_qc_stat(1),
		bot_controller_qc_stat(2), bot_controller_qc_stat(3));
}

void payload_strategy_log(float nclients)
{
	string pub_left = strcat("[PLCPUB] ", ftos(plc_str_req_seq), " ", ftos(time),
		" carts ", ftos(payload_cart_count), " clients ");
	string pub_right = strcat(ftos(nclients), " evtrows ", ftos(plc_str_evt_w));
	LOG_INFO(strcat(pub_left, pub_right));

	for (int c = 0; c < payload_cart_count; ++c)
	{
		entity row = plc_str_cart_pool[c];
		LOG_INFO(strcat(
			sprintf("[PLCCART] %d %d %g %g %g %g %g", plc_str_req_seq, c,
				row.plc_cs_f_id, row.plc_cs_f_path_position,
				row.plc_cs_f_path_length, row.plc_cs_f_control_team,
				row.plc_cs_f_speed),
				sprintf(" %g %g %g %g %g %g %g", row.plc_cs_f_idle_time,
				row.plc_cs_f_lead_team, row.plc_cs_f_second_team,
				row.plc_cs_f_home_team,
				row.plc_cs_f_px, row.plc_cs_f_py, row.plc_cs_f_pz),
				sprintf(" %g %g %g %g", row.plc_cs_f_supports_player,
				row.plc_cs_f_team_count, row.plc_cs_f_rollback_active,
				row.plc_cs_f_rollback_target)));
	}

	FOREACH_CLIENT(IS_PLAYER(it),
	{
		string a = sprintf("[PLCOBS] %d %d %g %g %g", plc_str_req_seq, etof(it),
			it.plc_ob_f_id, it.plc_ob_f_team, it.plc_ob_f_health);
		string b = sprintf(" %g %g %g %g %g %g %g", it.plc_ob_f_armor,
			it.plc_ob_f_ammo_shells, it.plc_ob_f_ammo_bullets,
			it.plc_ob_f_ammo_rockets, it.plc_ob_f_ammo_cells,
			it.plc_ob_f_ammo_plasma, it.plc_ob_f_ammo_fuel);
		string c2 = sprintf(" %g %g %g %g %g %g %g", it.plc_ob_f_px,
			it.plc_ob_f_py, it.plc_ob_f_pz, it.plc_ob_f_vx,
			it.plc_ob_f_vy, it.plc_ob_f_vz, it.plc_ob_f_weapons_x);
		string d = sprintf(" %g %g %g %g %g %g %g", it.plc_ob_f_weapons_y,
			it.plc_ob_f_weapons_z, it.plc_ob_f_strength_finished,
			it.plc_ob_f_spawn_time, it.plc_ob_f_engine_time,
			it.plc_ob_f_cell_x, it.plc_ob_f_cell_y);
		string e = sprintf(" %g %g %g %g %g %g %g", it.plc_ob_f_alive,
			it.plc_ob_f_control,
			it.plc_ob_f_applied_target_kind, it.plc_ob_f_applied_target_id,
			it.plc_ob_f_applied_target_cell_x, it.plc_ob_f_applied_target_cell_y,
			it.plc_ob_f_target_resolved);
		string f = sprintf(" %g %g %g %g %g %g %g",
			it.plc_ob_f_goal_target_kind, it.plc_ob_f_goal_target_id,
			it.plc_ob_f_goal_target_cell_x, it.plc_ob_f_goal_target_cell_y,
			it.plc_ob_f_goal_present, it.plc_ob_f_goal_px, it.plc_ob_f_goal_py);
		string g = sprintf(" %g %g %g %g %g %g %g", it.plc_ob_f_goal_pz,
			it.plc_ob_f_response_seq, it.plc_ob_f_response_time,
			it.plc_ob_f_route_seq, it.plc_ob_f_route_latency,
			it.plc_ob_f_goal_seq, it.plc_ob_f_goal_latency);
		string h = sprintf(" %g %g %g %g %g %g %g",
			it.plc_ob_f_touch_seq, it.plc_ob_f_touch_latency,
			it.plc_ob_f_enemy_damage_dealt, it.plc_ob_f_enemy_damage_taken,
			it.plc_ob_f_enemy_kills, it.plc_ob_f_deaths, it.plc_ob_f_pickups);
		string i = sprintf(" %g %g %g %g %g %g %g", it.plc_ob_f_cart_push,
			it.plc_ob_f_cart_contest, it.plc_ob_f_outcome_a_seq,
			it.plc_ob_f_outcome_a_enemy_damage_dealt,
			it.plc_ob_f_outcome_a_enemy_damage_taken, it.plc_ob_f_outcome_a_enemy_kills,
			it.plc_ob_f_outcome_a_deaths);
		string i2 = sprintf(" %g %g %g %g %g %g %g",
			it.plc_ob_f_outcome_a_pickups, it.plc_ob_f_outcome_a_cart_push,
			it.plc_ob_f_outcome_a_cart_contest, it.plc_ob_f_outcome_b_seq,
			it.plc_ob_f_outcome_b_enemy_damage_dealt,
			it.plc_ob_f_outcome_b_enemy_damage_taken,
			it.plc_ob_f_outcome_b_enemy_kills);
		string j = sprintf(" %g %g %g %g %g %g %g",
			it.plc_ob_f_outcome_b_deaths, it.plc_ob_f_outcome_b_pickups,
			it.plc_ob_f_outcome_b_cart_push, it.plc_ob_f_outcome_b_cart_contest,
			it.plc_ob_f_swizzle_active, it.plc_ob_f_swizzle_epoch,
			it.plc_ob_f_swizzle_player_count);
		string j2 = sprintf(" %g %g %g %g %g %g %g",
			it.plc_ob_f_swizzle_spot_count, it.plc_ob_f_swizzle_slot_count,
			it.plc_ob_f_swizzle_ticket, it.plc_ob_f_swizzle_cohort,
			it.plc_ob_f_swizzle_cohort_count, it.plc_ob_f_swizzle_generation,
			it.plc_ob_f_swizzle_scheduled_time);
		string k = sprintf(" %g %g %g", it.plc_ob_f_swizzle_actual_time,
			it.plc_ob_f_swizzle_lane, it.plc_ob_f_swizzle_spot);
		string left = strcat(a, b, c2, d, e, f, g);
		string right = strcat(h, i, i2, j, j2, k);
		LOG_INFO(strcat(left, right));
	});

	int r = 0;
	for (entity e = plc_str_evt_head; e; e = e.plc_ev_next)
	{
		LOG_INFO(strcat(
			sprintf("[PLCEVT] %d %d %g %g %g %g %g", plc_str_req_seq, r,
				e.plc_ev_f_kind, e.plc_ev_f_time, e.plc_ev_f_observer,
				e.plc_ev_f_team, e.plc_ev_f_subject),
			sprintf(" %g %g %g %g %g %g %g", e.plc_ev_f_cell_x, e.plc_ev_f_cell_y,
				e.plc_ev_f_target_cell_x, e.plc_ev_f_target_cell_y,
				e.plc_ev_f_px, e.plc_ev_f_py, e.plc_ev_f_pz),
			sprintf(" %g %g %g %g %g", e.plc_ev_f_respawn_time,
				e.plc_ev_f_health, e.plc_ev_f_link_length,
				e.plc_ev_f_amount, e.plc_ev_f_response_seq)));
		r += 1;
	}
}

float plc_str_link_cursor = 0;
float plc_str_link_pass = 0;
float plc_str_link_rows = 0;

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
			vector ca = payload_str_cell_of(it.origin);
			for (int j = 0; j < 32; ++j)
			{
				entity link = waypoint_get_link(it, j);
				if (!link)
					continue;
				vector cb = payload_str_cell_of(link.origin);
				if (cb == ca)
					continue;
				payload_str_deposit(PLC_EVT_KIND_CELL_LINK, 0, 0, 0,
					it.origin, ca, cb, 0, 0, vlen(link.origin - it.origin), 0, 0);
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
	int c = 0;
	IL_EACH(g_payload_carts, true,
	{
		entity row = plc_str_cart_pool[c];
		row.plc_cs_f_id            = it.plc_cart_id;
		row.plc_cs_f_path_position = it.plc_s;
		row.plc_cs_f_path_length   = it.plc_length;
		row.plc_cs_f_control_team  = it.plc_ctrl;
		row.plc_cs_f_speed         = it.plc_speed_now;
		row.plc_cs_f_idle_time     = it.plc_idle;
		row.plc_cs_f_lead_team     = it.plc_lead;
		row.plc_cs_f_second_team   = it.plc_second;
		row.plc_cs_f_home_team     = it.plc_home;
		vector cpos = it.origin + it.view_ofs;
		row.plc_cs_f_px           = cpos.x;
		row.plc_cs_f_py           = cpos.y;
		row.plc_cs_f_pz           = cpos.z;
		row.plc_cs_f_supports_player = payload_cart_supports_player(it);
		row.plc_cs_f_team_count = payload_teams;
		row.plc_cs_f_rollback_active = it.plc_rollback_active;
		row.plc_cs_f_rollback_target = it.plc_rollback_target;
		c += 1;
	});
	if (payload_cart_count > 0)
	{
		float cbase = plc_str_cart_base;
		mesh_gather_rows(plc_str_h_cart, plc_cs_f_id, cbase, payload_cart_count, PLC_STR_CART_W);
	}

	FOREACH_CLIENT(IS_PLAYER(it),
	{
		it.plc_ob_f_id      = etof(it);
		it.plc_ob_f_team    = Team_TeamToIndex(it.team);
		it.plc_ob_f_health  = GetResource(it, RES_HEALTH);
		it.plc_ob_f_armor   = GetResource(it, RES_ARMOR);
		it.plc_ob_f_ammo_shells = GetResource(it, RES_SHELLS);
		it.plc_ob_f_ammo_bullets = GetResource(it, RES_BULLETS);
		it.plc_ob_f_ammo_rockets = GetResource(it, RES_ROCKETS);
		it.plc_ob_f_ammo_cells = GetResource(it, RES_CELLS);
		it.plc_ob_f_ammo_plasma = GetResource(it, RES_PLASMA);
		it.plc_ob_f_ammo_fuel = GetResource(it, RES_FUEL);
		it.plc_ob_f_px      = it.origin.x;
		it.plc_ob_f_py      = it.origin.y;
		it.plc_ob_f_pz      = it.origin.z;
		it.plc_ob_f_vx      = it.velocity.x;
		it.plc_ob_f_vy      = it.velocity.y;
		it.plc_ob_f_vz      = it.velocity.z;
		it.plc_ob_f_weapons_x = STAT(WEAPONS, it).x;
		it.plc_ob_f_weapons_y = STAT(WEAPONS, it).y;
		it.plc_ob_f_weapons_z = STAT(WEAPONS, it).z;
		it.plc_ob_f_strength_finished = it.strength_finished;
		it.plc_ob_f_spawn_time = it.plc_str_spawn_stamp2;
		it.plc_ob_f_engine_time = time;
		it.plc_ob_f_cell_x = it.plc_str_cell_x;
		it.plc_ob_f_cell_y = it.plc_str_cell_y;
		it.plc_ob_f_alive   = IS_DEAD(it) ? 0 : 1;
		it.plc_ob_f_control = IS_BOT_CLIENT(it) ? 1 : 0;
		it.plc_ob_f_applied_target_kind = it.plc_str_applied_target_kind;
		it.plc_ob_f_applied_target_id = it.plc_str_applied_target_id;
		it.plc_ob_f_applied_target_cell_x = it.plc_str_applied_target_cell_x;
		it.plc_ob_f_applied_target_cell_y = it.plc_str_applied_target_cell_y;
		it.plc_ob_f_target_resolved = it.plc_str_resolved;
		it.plc_ob_f_goal_target_kind = it.plc_str_goal_target_kind;
		it.plc_ob_f_goal_target_id = it.plc_str_goal_target_id;
		it.plc_ob_f_goal_target_cell_x = it.plc_str_goal_target_cell_x;
		it.plc_ob_f_goal_target_cell_y = it.plc_str_goal_target_cell_y;
		it.plc_ob_f_goal_present = (it.plc_str_goal_entity != NULL);
		it.plc_ob_f_goal_px = 0;
		it.plc_ob_f_goal_py = 0;
		it.plc_ob_f_goal_pz = 0;
		vector goalpos = '0 0 0';
		if (it.plc_str_goal_entity != NULL)
		{
			goalpos = (it.plc_str_goal_entity.absmin + it.plc_str_goal_entity.absmax) * 0.5;
			it.plc_ob_f_goal_px = goalpos.x;
			it.plc_ob_f_goal_py = goalpos.y;
			it.plc_ob_f_goal_pz = goalpos.z;
		}
		bool goalmatch = it.plc_str_applied_target_kind != PLC_TARGET_NONE
			&& it.plc_str_goal_target_kind == it.plc_str_applied_target_kind
			&& ((it.plc_str_applied_target_kind == PLC_TARGET_CELL
			&& it.plc_str_goal_target_cell_x == it.plc_str_applied_target_cell_x
			&& it.plc_str_goal_target_cell_y == it.plc_str_applied_target_cell_y)
			|| (it.plc_str_applied_target_kind != PLC_TARGET_CELL
			&& it.plc_str_goal_target_id == it.plc_str_applied_target_id));
		bool targettouch = goalmatch && ((it.plc_str_applied_target_kind == PLC_TARGET_CELL
			&& it.plc_str_cell_x == it.plc_str_applied_target_cell_x
			&& it.plc_str_cell_y == it.plc_str_applied_target_cell_y)
			|| (it.plc_str_applied_target_kind != PLC_TARGET_CELL && it.plc_str_goal_entity != NULL
			&& vlen(goalpos - it.origin) <= 163.84));
		if (targettouch)
		{
			it.plc_str_touch_seq = it.plc_str_goal_seq;
			it.plc_str_touch_latency = max(0, time - it.plc_str_response_time);
		}
		it.plc_ob_f_response_seq = it.plc_str_response_seq;
		it.plc_ob_f_response_time = it.plc_str_response_time;
		it.plc_ob_f_route_seq = it.plc_str_route_seq;
		it.plc_ob_f_route_latency = it.plc_str_route_latency;
		it.plc_ob_f_goal_seq = it.plc_str_goal_seq;
		it.plc_ob_f_goal_latency = it.plc_str_goal_latency;
		it.plc_ob_f_touch_seq = it.plc_str_touch_seq;
		it.plc_ob_f_touch_latency = it.plc_str_touch_latency;
		it.plc_ob_f_enemy_damage_dealt = it.plc_str_enemy_damage_dealt;
		it.plc_ob_f_enemy_damage_taken = it.plc_str_enemy_damage_taken;
		it.plc_ob_f_enemy_kills = it.plc_str_enemy_kills;
		it.plc_ob_f_deaths = it.plc_str_deaths;
		it.plc_ob_f_pickups = it.plc_str_pickups;
		it.plc_ob_f_cart_push = it.plc_str_cart_push;
		it.plc_ob_f_cart_contest = it.plc_str_cart_contest;
		if (!it.plc_str_outcome_a_used)
		{
			it.plc_str_outcome_a_used = 1;
			it.plc_str_outcome_a_seq = it.plc_str_route_seq;
		}
		else if (it.plc_str_outcome_a_seq != it.plc_str_route_seq && !it.plc_str_outcome_b_used)
		{
			it.plc_str_outcome_b_used = 1;
			it.plc_str_outcome_b_seq = it.plc_str_route_seq;
		}
		it.plc_ob_f_outcome_a_seq = it.plc_str_outcome_a_seq;
		it.plc_ob_f_outcome_a_enemy_damage_dealt = it.plc_str_outcome_a_enemy_damage_dealt;
		it.plc_ob_f_outcome_a_enemy_damage_taken = it.plc_str_outcome_a_enemy_damage_taken;
		it.plc_ob_f_outcome_a_enemy_kills = it.plc_str_outcome_a_enemy_kills;
		it.plc_ob_f_outcome_a_deaths = it.plc_str_outcome_a_deaths;
		it.plc_ob_f_outcome_a_pickups = it.plc_str_outcome_a_pickups;
		it.plc_ob_f_outcome_a_cart_push = it.plc_str_outcome_a_cart_push;
		it.plc_ob_f_outcome_a_cart_contest = it.plc_str_outcome_a_cart_contest;
		it.plc_ob_f_outcome_b_seq = it.plc_str_outcome_b_seq;
		it.plc_ob_f_outcome_b_enemy_damage_dealt = it.plc_str_outcome_b_enemy_damage_dealt;
		it.plc_ob_f_outcome_b_enemy_damage_taken = it.plc_str_outcome_b_enemy_damage_taken;
		it.plc_ob_f_outcome_b_enemy_kills = it.plc_str_outcome_b_enemy_kills;
		it.plc_ob_f_outcome_b_deaths = it.plc_str_outcome_b_deaths;
		it.plc_ob_f_outcome_b_pickups = it.plc_str_outcome_b_pickups;
		it.plc_ob_f_outcome_b_cart_push = it.plc_str_outcome_b_cart_push;
		it.plc_ob_f_outcome_b_cart_contest = it.plc_str_outcome_b_cart_contest;
		it.plc_ob_f_swizzle_active = it.spawn_swizzle_active;
		it.plc_ob_f_swizzle_epoch = it.spawn_swizzle_epoch;
		it.plc_ob_f_swizzle_player_count = it.spawn_swizzle_player_count;
		it.plc_ob_f_swizzle_spot_count = it.spawn_swizzle_spot_count;
		it.plc_ob_f_swizzle_slot_count = it.spawn_swizzle_slot_count;
		it.plc_ob_f_swizzle_ticket = it.spawn_swizzle_ticket;
		it.plc_ob_f_swizzle_cohort = it.spawn_swizzle_cohort;
		it.plc_ob_f_swizzle_cohort_count = it.spawn_swizzle_cohort_count;
		it.plc_ob_f_swizzle_generation = it.spawn_swizzle_generation;
		it.plc_ob_f_swizzle_scheduled_time = it.spawn_swizzle_scheduled_time;
		it.plc_ob_f_swizzle_actual_time = it.spawn_swizzle_actual_time;
		it.plc_ob_f_swizzle_lane = it.spawn_swizzle_lane;
		it.plc_ob_f_swizzle_spot = it.spawn_swizzle_spot;
	});
	mesh_gather_rows(plc_str_h_obs, plc_ob_f_id, 1, nclients, PLC_STR_OBS_W);
	FOREACH_CLIENT(IS_PLAYER(it), { payload_str_measure_reset(it); });

	float event_rows = mesh_gather_list(plc_str_h_evt, plc_ev_f_kind, plc_str_evt_head,
		plc_ev_next, plc_str_evt_w, PLC_STR_EVT_W);

	plc_str_req_seq += 1;
	if (autocvar_g_payload_vcell_log)
		payload_vcell_log();
	if (autocvar_g_payload_strategy_log)
		payload_strategy_log(nclients);

	mesh_publish(plc_str_h_cart, plc_str_req_seq, payload_cart_count);
	mesh_publish(plc_str_h_obs,  plc_str_req_seq, nclients);
	mesh_publish(plc_str_h_evt,  plc_str_req_seq, event_rows);
	entity event = plc_str_evt_head;
	while (event)
	{
		entity next = event.plc_ev_next;
		remove(event);
		event = next;
	}
	plc_str_evt_head = NULL;
	plc_str_evt_tail = NULL;
	plc_str_evt_w = 0;
}

void payload_strategy_scatter(float nclients)
{
	float seq = mesh_poll(plc_str_h_sc);
	if (seq <= plc_str_resp_seq)
		return;
	float committed = mesh_scatter_rows(plc_str_h_sc, plc_str_instrument_kind, 1, nclients, PLC_STR_RESP_W);
	if (committed < nclients)
	{
		LOG_INFOF("payload: [PLCBARRIER] seq %.0f rows %.0f required %.0f committed 0\n", seq, committed, nclients);
		return;
	}
	plc_str_resp_seq = seq;
	FOREACH_CLIENT(IS_PLAYER(it),
	{
		it.plc_str_response_seq = seq;
		it.plc_str_response_time = time;
		if(IS_BOT_CLIENT(it) && (it.plc_str_instrument_kind != it.plc_str_applied_instrument_kind
			|| it.plc_str_target_kind != it.plc_str_applied_target_kind
			|| it.plc_str_target_id != it.plc_str_applied_target_id
			|| it.plc_str_target_cell_x != it.plc_str_applied_target_cell_x
			|| it.plc_str_target_cell_y != it.plc_str_applied_target_cell_y
			|| it.plc_str_gain != it.plc_str_applied_gain))
			navigation_goalrating_timeout_force(it);
	});
	if (autocvar_g_payload_strategy_log)
		LOG_INFOF("payload: [PLCBARRIER] seq %.0f rows %.0f barriers %.0f total_rows %.0f\n", seq, committed, mesh_stat(plc_str_h_sc, MESH_STAT_BARRIERS), mesh_stat(plc_str_h_sc, MESH_STAT_SCATTER_ROWS));
}

void payload_strategy_entity_target(entity player, entity target)
{
	player.plc_str_goal_target_kind = PLC_TARGET_NONE;
	player.plc_str_goal_target_id = -1;
	player.plc_str_goal_target_cell_x = 0;
	player.plc_str_goal_target_cell_y = 0;
	if (!target)
		return;
	IL_EACH(g_payload_carts, target == it,
	{
		player.plc_str_goal_target_kind = PLC_TARGET_CART;
		player.plc_str_goal_target_id = it.plc_cart_id;
		return;
	});
	if (IS_PLAYER(target))
		player.plc_str_goal_target_kind = PLC_TARGET_RIVAL;
	else if (target.bot_pickup)
		player.plc_str_goal_target_kind = PLC_TARGET_ITEM;
	else if (target.classname == "waypoint")
		player.plc_str_goal_target_kind = PLC_TARGET_CELL;
	if (player.plc_str_goal_target_kind != PLC_TARGET_NONE)
	{
		if (player.plc_str_goal_target_kind == PLC_TARGET_CELL)
		{
			vector cell = payload_str_cell_of(target.origin);
			player.plc_str_goal_target_cell_x = cell.x;
			player.plc_str_goal_target_cell_y = cell.y;
		}
		else
			player.plc_str_goal_target_id = etof(target);
	}
}

void havocbot_goalrating_strategy(entity this, float ratingscale)
{
	float kind = this.plc_str_target_kind;
	float instrument = this.plc_str_instrument_kind;
	float tgt  = this.plc_str_target_id;
	float cell_x = this.plc_str_target_cell_x;
	float cell_y = this.plc_str_target_cell_y;
	float gain = this.plc_str_gain;
	this.plc_str_applied_target_kind = kind;
	this.plc_str_applied_instrument_kind = instrument;
	this.plc_str_applied_target_id = tgt;
	this.plc_str_applied_target_cell_x = cell_x;
	this.plc_str_applied_target_cell_y = cell_y;
	this.plc_str_applied_gain = gain;
	this.plc_str_resolved = 0;

	if (kind == PLC_TARGET_NONE || (kind != PLC_TARGET_CELL && tgt < 0) || gain == 0)
	{
		this.plc_str_resolved = 1;
	}
	else if (kind == PLC_TARGET_CART)
	{
		entity cart = payload_cart_by_id(tgt);
		if (cart && gain != 0)
		{
			this.plc_str_resolved = 1;
			if (instrument == PLC_INSTRUMENT_SUPPRESS_CART)
			{
				float rivals = 0;
				FOREACH_CLIENT(IS_PLAYER(it) && DIFF_TEAM(it, this) && vlen(it.origin - cart.origin) <= 1024,
				{
					navigation_routerating(this, it, ratingscale * gain, 5000);
					rivals += 1;
				});
				if (!rivals)
					navigation_routerating(this, cart, ratingscale * gain, 5000);
			}
			else
				navigation_routerating(this, cart, ratingscale * gain, 5000);
		}
	}
	else if (kind == PLC_TARGET_ITEM)
	{
		IL_EACH(g_items, etof(it) == tgt && gain != 0,
		{
			this.plc_str_resolved = 1;
			navigation_routerating(this, it, ratingscale * gain, 3000);
			break;
		});
	}
	else if (kind == PLC_TARGET_RIVAL)
	{
		FOREACH_CLIENT(IS_PLAYER(it) && etof(it) == tgt && DIFF_TEAM(it, this) && gain != 0,
		{
			this.plc_str_resolved = 1;
			navigation_routerating(this, it, ratingscale * gain, 5000);
			break;
		});
	}
	else if (kind == PLC_TARGET_CELL && gain != 0)
	{
		vector target_cell = cell_x * '1 0 0' + cell_y * '0 1 0';
		IL_EACH(g_waypoints, payload_str_cell_of(it.origin) == target_cell,
		{
			this.plc_str_resolved = 1;
			navigation_routerating(this, it, ratingscale * gain, 3000);
		});
	}
	if (this.plc_str_resolved)
	{
		this.plc_str_route_seq = this.plc_str_response_seq;
		this.plc_str_route_latency = max(0, time - this.plc_str_response_time);
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
