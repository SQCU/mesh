#include "sv_payload.qh"

#include <common/notifications/all.qh>
#include <common/net_linked.qh>
#include <common/mapobjects/defs.qh>
#include <common/mapobjects/platforms.qh>
#include <common/mapobjects/subs.qh>
#include <common/mapobjects/triggers.qh>
#include <common/mutators/mutator/waypoints/waypointsprites.qh>
#include <server/bot/api.qh>
#include <server/mutators/_mod.qh>
#include <server/client.qh>
#include <server/command/vote.qh>
#include <server/damage.qh>
#include <server/gamelog.qh>
#include <server/teamplay.qh>
#include <server/world.qh>

entity payload_cart_by_id(int id)
{
	entity found = NULL;
	IL_EACH(g_payload_carts, it.plc_cart_id == id,
	{
		found = it;
		break;
	});
	return found;
}
int autocvar_g_payload_default_teams = 3;
int autocvar_g_payload_teams_override;
float autocvar_g_payload_speed = 30;
float autocvar_g_payload_max_speed = 200;
float autocvar_g_payload_push_radius = 160;
float autocvar_g_payload_push_height = 96;
int autocvar_g_payload_push_cap = 3;
float autocvar_g_payload_push_falloff = 1;
float autocvar_g_payload_contest_speed = 24;
float autocvar_g_payload_reverse_speed = 14;
float autocvar_g_payload_score_rate = 0.01;
float autocvar_g_payload_block_rate = 1;
float autocvar_g_payload_stall_eps = 1;
float autocvar_g_payload_checkpoint_rate = 1;
float autocvar_g_payload_warmup = 5;
float autocvar_g_payload_node_fade_dist = 1100;
int autocvar_g_payload_pathlinks = 1;
bool autocvar_g_payload_cart_procedural = true;
int autocvar_g_payload_node_sprites = 0;

.float plc_in;
.float plc_block_accum;

void payload_EventLog(string mode, float t)
{
	if(autocvar_sv_eventlog)
		GameLogEcho(strcat(":plc:", mode, ":", ftos(t)));
}

vector payload_bezier(vector p0, vector p1, vector p2, float t)
{
	float u = 1 - t;
	return p0 * (u * u) + p1 * (2 * u * t) + p2 * (t * t);
}

float payload_seglen(entity a, entity b)
{
	if(!a.plc_curve)
		return vlen(b.origin - a.origin);
	float d = 0;
	vector p = a.origin;
	for(float i = 1; i <= PLC_CURVE_STEPS; ++i)
	{
		vector q = payload_bezier(a.origin, a.plc_curve.origin, b.origin, i / PLC_CURVE_STEPS);
		d += vlen(q - p);
		p = q;
	}
	return d;
}

vector payload_pos(entity cart, float s)
{
	s = bound(0, s, cart.plc_length);
	entity a = cart.plc_first;
	if(!a)
		return '0 0 0';
	while(a.plc_next && a.plc_next.plc_dist <= s)
		a = a.plc_next;
	entity b = a.plc_next;
	if(!b)
		return a.origin;
	float seg = b.plc_dist - a.plc_dist;
	float want = s - a.plc_dist;
	if(!a.plc_curve)
		return a.origin + (b.origin - a.origin) * ((seg > 0) ? want / seg : 0);
	float acc = 0;
	vector p = a.origin;
	for(float i = 1; i <= PLC_CURVE_STEPS; ++i)
	{
		vector q = payload_bezier(a.origin, a.plc_curve.origin, b.origin, i / PLC_CURVE_STEPS);
		float l = vlen(q - p);
		if(acc + l >= want)
			return p + (q - p) * ((l > 0) ? (want - acc) / l : 0);
		acc += l;
		p = q;
	}
	return b.origin;
}

void payload_occupancy(entity cart)
{
	vector c = (cart.absmin + cart.absmax) * 0.5;
	float r = cart.radius;
	float h = cart.height;
	for(int j = 0; j <= NUM_TEAMS; ++j)
	{
		payload_w[j] = 0;
		payload_n[j] = 0;
	}
	FOREACH_CLIENT(true, { it.plc_in = 0; });
	FOREACH_CLIENT(IS_PLAYER(it) && !IS_DEAD(it),
	{
		int j = Team_TeamToIndex(it.team);
		if(j < 1)
			continue;
		vector d = it.origin - c;
		if(fabs(d.z) > h)
			continue;
		if(vlen(vec2(d)) > r)
			continue;
		it.plc_in = j;
		payload_n[j] += 1;
	});
	for(int j = 1; j <= NUM_TEAMS; ++j)
	{
		float w = 0, term = 1;
		float cnt = min(payload_n[j], autocvar_g_payload_push_cap);
		for(int i = 0; i < cnt; ++i)
		{
			w += term;
			term *= autocvar_g_payload_push_falloff;
		}
		payload_w[j] = w;
	}
}

void payload_spawn_link(entity cart, entity a, entity b)
{
	entity lk = new(plc_link);
	lk.goalentity = a;
	lk.enemy = b;
	lk.plc_cart = cart;
	IL_PUSH(g_payload_links, lk);
	++payload_link_count;
	Net_LinkEntity(lk, false, 0, payload_link_send);
	lk.SendFlags = 7;
}

bool payload_link_send(entity this, entity to, int sendflags)
{
	WriteHeader(MSG_ENTITY, ENT_CLIENT_PAYLOAD_LINK);
	WriteByte(MSG_ENTITY, sendflags);
	if(sendflags & 1)
		WriteVector(MSG_ENTITY, this.goalentity.origin);
	if(sendflags & 2)
		WriteVector(MSG_ENTITY, this.enemy.origin);
	if(sendflags & 4)
	{
		WriteShort(MSG_ENTITY, this.plc_link_team_a);
		WriteShort(MSG_ENTITY, this.plc_link_team_b);
	}
	return true;
}

float payload_projected_time;
int payload_projected_ties;

int payload_projected_winner()
{
	int leader = 0;
	payload_projected_time = 0;
	payload_projected_ties = 0;
	for(int j = 1; j <= payload_teams; ++j)
	{
		float rate = payload_checkpoints_held[j] * autocvar_g_payload_checkpoint_rate;
		if(rate > 0 || payload_score[j] >= autocvar_g_payload_point_limit)
		{
			float remaining = payload_score[j] >= autocvar_g_payload_point_limit ? 0 : (autocvar_g_payload_point_limit - payload_score[j]) / rate;
			if(!payload_projected_ties || remaining < payload_projected_time)
			{
				payload_projected_time = remaining;
				leader = j;
				payload_projected_ties = 1;
			}
			else if(remaining == payload_projected_time)
				++payload_projected_ties;
		}
	}
	return payload_projected_ties == 1 ? leader : 0;
}

bool payload_cart_state_send(entity this, entity to, int sendflags)
{
	entity cart = this.owner;
	vector cartpos = cart.origin + cart.view_ofs;
	WriteHeader(MSG_ENTITY, ENT_CLIENT_PAYLOAD_CART);
	WriteByte(MSG_ENTITY, cart.plc_cart_id);
	WriteVector(MSG_ENTITY, cartpos);
	WriteFloat(MSG_ENTITY, cart.plc_length > 0 ? cart.plc_s / cart.plc_length : 0);
	WriteShort(MSG_ENTITY, cart.plc_ctrl);
	WriteShort(MSG_ENTITY, cart.plc_lead);
	WriteShort(MSG_ENTITY, cart.plc_second);
	WriteShort(MSG_ENTITY, cart.plc_home);
	WriteByte(MSG_ENTITY, bound(0, cart.plc_contest * 255, 255));
	WriteByte(MSG_ENTITY, boolean(cart.plc_speed_now < -autocvar_g_payload_stall_eps) | (boolean(fabs(cart.plc_speed_now) <= autocvar_g_payload_stall_eps) << 1));
	if(cart.plc_cart_id == 0)
	{
		int leader = payload_projected_winner();
		if(payload_score_finished)
			leader = payload_score_winner;
		WriteShort(MSG_ENTITY, leader);
		WriteShort(MSG_ENTITY, payload_projected_ties);
		WriteByte(MSG_ENTITY, payload_score_finished);
		WriteFloat(MSG_ENTITY, payload_score[leader]);
		WriteFloat(MSG_ENTITY, autocvar_g_payload_point_limit);
		WriteFloat(MSG_ENTITY, payload_projected_time);
	}
	return true;
}

int payload_node_color(entity n)
{
	return n.plc_banked ? n.plc_bank_team : 0;
}

void payload_update_links(entity cart)
{
	int contest = cart.plc_ctrl;
	IL_EACH(g_payload_links, it.plc_cart == cart,
	{
		entity a = it.goalentity;
		entity b = it.enemy;
		int ca;
		int cb;
		if(a.plc_dist <= cart.plc_s && b.plc_dist > cart.plc_s)
			ca = cb = contest;
		else
		{
			ca = payload_node_color(a);
			cb = payload_node_color(b);
		}
		if(ca != it.plc_link_team_a || cb != it.plc_link_team_b)
		{
			it.plc_link_team_a = ca;
			it.plc_link_team_b = cb;
			it.SendFlags |= 4;
		}
	});
}

void payload_integrate_score()
{
	float dt = max(0, time - payload_score_time);
	payload_score_time = time;
	if(game_stopped || !round_handler_IsRoundStarted() || payload_score_finished)
		return;
	int winner = payload_projected_winner();
	if(payload_projected_ties && payload_projected_time <= dt)
	{
		dt = payload_projected_time;
		payload_score_finished = true;
		payload_score_winner = winner;
	}
	for(int j = 1; j <= payload_teams; ++j)
	{
		float before = payload_score[j];
		payload_score[j] += dt * payload_checkpoints_held[j] * autocvar_g_payload_checkpoint_rate;
		TeamScore_AddToTeam(Team_IndexToTeam(j), ST_PAYLOAD_POINTS, floor(payload_score[j]) - floor(before));
	}
}

void payload_checkpoint_control(entity this, int ctrl, float s)
{
	this.plc_checkpoint_depth = 0;
	IL_EACH(g_payload_nodes, it.plc_cart == this && (it.spawnflags & PLC_CHECKPOINT),
	{
		int owner = it.plc_dist <= s ? ctrl : 0;
		this.plc_checkpoint_depth += owner > 0;
		if(owner != it.plc_bank_team)
		{
			payload_checkpoints_held[it.plc_bank_team] -= it.plc_bank_team > 0;
			payload_checkpoints_held[owner] += owner > 0;
			it.plc_bank_team = owner;
			it.plc_banked = owner > 0;
			LOG_INFO(sprintf("payload: checkpoint cart %d node %d owner %d", this.plc_cart_id, it.plc_idx, owner));
			if(it.sprite)
				WaypointSprite_UpdateTeamRadar(it.sprite, RADARICON_WAYPOINT,
					owner ? Team_ColorRGB(Team_IndexToTeam(owner)) : '0.5 0.5 0.5');
		}
	});
}

void payload_clear_progress(entity this)
{
	IL_EACH(g_payload_nodes, it.plc_cart == this,
	{
		it.plc_banked = 0;
		it.plc_bank_team = 0;
		if(it.sprite)
			WaypointSprite_UpdateTeamRadar(it.sprite, RADARICON_WAYPOINT, WP_Waypoint.m_color);
	});
}

void payload_think(entity this)
{
	this.nextthink = this.ltime + PLC_TICK;
	payload_integrate_score();

	bool live = !game_stopped && !payload_score_finished && time >= game_starttime
		&& (!round_handler_IsActive() || round_handler_IsRoundStarted());

	payload_occupancy(this);

	float total = 0, best = 0, second = 0;
	int ctrl = 0, lead = 0, second_team = 0;
	bool tie = false;
	for(int j = 1; j <= NUM_TEAMS; ++j)
	{
		total += payload_w[j];
		if(payload_w[j] > best)
		{
			second = best;
			second_team = lead;
			best = payload_w[j];
			lead = j;
			tie = false;
		}
		else if(payload_w[j] == best && best > 0)
		{
			tie = true;
			second = best;
			second_team = j;
		}
		else if(payload_w[j] > second)
		{
			second = payload_w[j];
			second_team = j;
		}
	}
	ctrl = tie ? 0 : lead;

	int color = this.plc_ctrl;

	if(color == 0 && ctrl)
		color = this.plc_ctrl = ctrl;

	float wA = color ? payload_w[color] : 0;
	float wopp = total - wA;

	int opp = 0; float oppw = 0;
	for(int j = 1; j <= NUM_TEAMS; ++j)
		if(j != color && payload_w[j] > oppw) { oppw = payload_w[j]; opp = j; }
	this.plc_lead = lead;
	this.plc_second = second_team;
	this.plc_home = opp;
	this.plc_contest = best > 0 ? bound(0, second / best, 1) : 0;

	if(total <= 0)
		this.plc_idle += PLC_TICK;
	else
		this.plc_idle = 0;

	float v;
	if(color && wA > 0)
	{

		v = this.speed * (wA - wopp) / (1 + wopp * wopp);
		v = bound(-autocvar_g_payload_contest_speed, v, autocvar_g_payload_max_speed);
	}
	else if(color && oppw > 0)
	{

		v = -autocvar_g_payload_reverse_speed * oppw;
		v = bound(-autocvar_g_payload_max_speed, v, 0);
	}
	else
	{

		v = 0;
	}

	v = live ? v : 0;
	float s_prev = this.plc_s;
	float s_new = bound(0, this.plc_s + v * PLC_TICK, this.plc_length);
	this.plc_s = s_new;
	this.plc_speed_now = (s_new - s_prev) / PLC_TICK;

	vector want = payload_pos(this, s_new);
	this.velocity = (want - (this.origin + this.view_ofs)) * (1 / PLC_TICK);

	if(this.spawnflags & PLC_CART_TURN)
	{
		vector ahead = payload_pos(this, min(this.plc_length, s_new + 16)) - payload_pos(this, max(0, s_new - 16));
		if(vlen(ahead) > 0)
			this.angles = vectoangles(ahead);
	}

	payload_checkpoint_control(this, color, s_new);
	if(s_new <= 0 && s_prev > 0)
	{
		payload_clear_progress(this);
		this.plc_ctrl = 0;
		LOG_INFO(strcat("payload: cart ", ftos(this.plc_cart_id), " regressed to origin, recolored neutral"));
		payload_EventLog("origin", this.plc_cart_id);
	}

	float gain = (s_new > s_prev && color && payload_n[color] > 0)
		? (s_new - s_prev) * autocvar_g_payload_score_rate / payload_n[color] : 0;
	bool stalled = (total > 0) && (fabs(this.plc_speed_now) < autocvar_g_payload_stall_eps);
	FOREACH_CLIENT(IS_PLAYER(it) && it.plc_in,
	{
		if (total > payload_w[it.plc_in])
		{
			it.plc_str_cart_contest += PLC_TICK;
			payload_str_measure(it, PLC_OUTCOME_CART_CONTEST, PLC_TICK);
		}
		if(gain > 0 && it.plc_in == color)
		{
			it.plc_str_cart_push += (s_new - s_prev) / payload_n[color];
			payload_str_measure(it, PLC_OUTCOME_CART_PUSH, (s_new - s_prev) / payload_n[color]);
			GameRules_scoring_add_float2int(it, PAYLOAD_PUSH, gain, plc_push_accum, 1);
		}
		if(stalled)
			GameRules_scoring_add_float2int(it, PAYLOAD_BLOCK, autocvar_g_payload_block_rate * PLC_TICK, plc_block_accum, 1);
	});

	if(this.plc_state)
		this.plc_state.SendFlags |= 1;
	WaypointSprite_UpdateHealth(this.sprite, s_new);
	payload_update_links(this);

	if(this.plc_cart_id == 0)
	{

		if(time >= plc_str_next_step)
		{
			plc_str_next_step = time + PLC_STRAT_TICK;
			payload_strategy_step(this);
		}
	}

}

void payload_cart_reset(entity this)
{
	this.plc_s = 0;
	this.plc_idle = 0;
	this.plc_ctrl = 0;
	this.plc_checkpoint_depth = 0;
	this.velocity = '0 0 0';
	this.plc_speed_now = 0;
	payload_clear_progress(this);
	setorigin(this, payload_pos(this, 0) - this.view_ofs);
}

void payload_node_sprite(entity node, float maxd)
{
	if(node.sprite)
		return;
	WaypointSprite_Spawn(WP_Waypoint, 0, maxd, NULL, node.origin + '0 0 48',
		NULL, 0, node, sprite, true, RADARICON_WAYPOINT);
}

void payload_path_configure(entity this, entity first, bool register_cart)
{
	if(register_cart)
	{
		this.plc_cart_id = payload_cart_count;
		IL_PUSH(g_payload_carts, this);
		++payload_cart_count;
		this.plc_state = new_pure(plc_cart_state);
		this.plc_state.owner = this;
		Net_LinkEntity(this.plc_state, false, 0, payload_cart_state_send);
	}
	if(!first)
	{
		first = new_pure(plc_path);
		setorigin(first, this.origin + this.view_ofs);
		LOG_INFO("payload: func_plc_cart has no reachable plc_path, using a single node at its own origin");
	}
	this.plc_first = first;
	this.plc_checkpoint_count = 0;

	entity prev = NULL;
	entity node = first;
	float d = 0;
	int guard = 0;
	int interior = 0;
	while(node)
	{
		IL_PUSH(g_payload_nodes, node);
		if(interior < autocvar_g_payload_node_sprites)
		{
			payload_node_sprite(node, autocvar_g_payload_node_fade_dist);
			++interior;
		}
		node.plc_prev = prev;
		node.plc_cart = this;
		node.plc_idx = guard;
		this.plc_checkpoint_count += (node.spawnflags & PLC_CHECKPOINT) != 0;
		++guard;
		if(node.curvetarget != "")
			node.plc_curve = find(NULL, targetname, node.curvetarget);
		if(prev)
		{
			d += payload_seglen(prev, node);
			prev.plc_next = node;
		}
		node.plc_dist = d;
		entity nxt = node.plc_next;
		if(!nxt && node.target != "")
			nxt = find(NULL, targetname, node.target);
		if(nxt == first || (nxt && nxt.plc_prev))
			nxt = NULL;
		node.plc_next = nxt;
		prev = node;
		node = nxt;
	}
	this.plc_length = d;

	if(autocvar_g_payload_pathlinks && guard > 1)
	{
		int stride = max(1, floor((guard - 1) / PLC_RIBBON_SEG));
		entity samp = first;
		int lastk = 0, kk = 0;
		for(entity nd = first.plc_next; nd; nd = nd.plc_next)
		{
			++kk;
			if(kk - lastk >= stride || nd.plc_next == NULL)
			{
				payload_spawn_link(this, samp, nd);
				samp = nd;
				lastk = kk;
			}
		}
	}

	this.plc_s = 0;
	setorigin(this, payload_pos(this, 0) - this.view_ofs);

	entity wp = WP_PayloadCart;
	switch(this.plc_cart_id)
	{
		case 1: wp = WP_PayloadCart2; break;
		case 2: wp = WP_PayloadCart3; break;
		case 3: wp = WP_PayloadCart4; break;
	}
	if(!this.sprite)
		WaypointSprite_Spawn(wp, 0, 0, this, this.view_ofs + PLC_SPRITE_OFS, NULL, 0, this, sprite, true, RADARICON_OBJECTIVE);

	if(this.sprite)
		this.sprite.wp_extra = this.plc_cart_id;

	if(autocvar_g_payload_cart_procedural)
		this.alpha = -1;
	if(this.sprite)
		WaypointSprite_UpdateSprites(this.sprite, wp, wp, wp);
	WaypointSprite_UpdateMaxHealth(this.sprite, this.plc_length);
	WaypointSprite_UpdateHealth(this.sprite, this.plc_s);

	setthink(this, payload_think);
	this.nextthink = this.ltime + PLC_TICK;

	LOG_INFO(strcat("payload: cart ", ftos(this.plc_cart_id), ": ", ftos(guard),
		" path nodes, length ", ftos(this.plc_length), " support ",
		ftos(payload_cart_supports_player(this))));
}

void payload_path_build(entity this)
{
	entity first = (this.target != "") ? find(NULL, targetname, this.target) : NULL;
	payload_path_configure(this, first, true);
}

void payload_cart_configure(entity cart)
{
	cart.effects |= EF_LOWPRECISION;
	if(!cart.speed)
		cart.speed = autocvar_g_payload_speed * (cart.plc_speed_scale ? cart.plc_speed_scale : 1);
	if(!cart.radius)
		cart.radius = autocvar_g_payload_push_radius;
	if(!cart.height)
		cart.height = autocvar_g_payload_push_height;
	if(cart.dmg && !cart.dmgtime)
		cart.dmgtime = 0.25;
	if(cart.dmg && cart.message == "")
		cart.message = " was squished";
	if(cart.dmg && cart.message2 == "")
		cart.message2 = "was squished by";
	cart.dmgtime2 = time;

	precache_model(PLC_CART_MODEL);
	_setmodel(cart, PLC_CART_MODEL);
	cart.effects |= EF_NODRAW;
	cart.solid = SOLID_BBOX;
	set_movetype(cart, MOVETYPE_PUSH);
	setsize(cart, PLC_CART_MIN, PLC_CART_MAX);
	cart.view_ofs = -PLC_CART_RIDE_OFS;
	setblocked(cart, generic_plat_blocked);
	cart.reset = payload_cart_reset;
}

bool payload_cart_supports_player(entity cart)
{
	vector stand = cart.origin + PLC_CART_STAND_OFS;
	tracebox(stand + '0 0 8', PL_MIN_CONST, PL_MAX_CONST, stand - '0 0 8', MOVE_NORMAL, NULL);
	return !trace_startsolid && trace_ent == cart && fabs(trace_endpos.z - stand.z) <= 1;
}

void payload_RoundEnd(int winner_team)
{
	mesh_round_outcome(payload_episode, payload_score_winner);
	string outcome = sprintf("{\"event\":\"server_outcome\",\"contract\":\"checkpoint-control-integral-v1\",\"map\":\"%s\",\"episode\":%d,\"time\":%.9g,\"actor_team\":%d,\"score_limit\":%.9g,\"scores\":[", mapname, payload_episode, time, payload_score_winner, autocvar_g_payload_point_limit);
	for(int j = 1; j <= payload_teams; ++j)
		outcome = strcat(outcome, (j > 1 ? "," : ""), sprintf("%.9g", payload_score[j]));
	print("MESH_OUTCOME ", outcome, "]}\n");
	payload_str_round(winner_team > 0 ? Team_TeamToIndex(winner_team) : 0);
	if(winner_team > 0)
	{
		if(winner_team <= NUM_TEAM_5)
		{
			Send_Notification(NOTIF_ALL, NULL, MSG_CENTER, APP_TEAM_NUM(winner_team, CENTER_ROUND_TEAM_WIN));
			Send_Notification(NOTIF_ALL, NULL, MSG_INFO, APP_TEAM_NUM(winner_team, INFO_ROUND_TEAM_WIN));
		}
		else
		{
			string result = strcat(Team_ColorCode(winner_team), Team_ColorName_Upper(winner_team), "^7 team wins the round");
			FOREACH_CLIENT(true, { centerprint(it, result); });
			bprint(result, "\n");
		}
		payload_EventLog("score_win", winner_team);
	}
	else
	{
		Send_Notification(NOTIF_ALL, NULL, MSG_CENTER, CENTER_ROUND_TIED);
		Send_Notification(NOTIF_ALL, NULL, MSG_INFO, INFO_ROUND_TIED);
		payload_EventLog("tie", 0);
	}

	game_stopped = true;
	round_handler_Init(5, autocvar_g_payload_warmup, 0);
}

bool Payload_CheckWinner()
{
	payload_integrate_score();
	if(payload_score_finished)
	{
		payload_RoundEnd(payload_score_winner ? Team_IndexToTeam(payload_score_winner) : 0);
		return true;
	}
	return false;
}

bool Payload_CheckPlayers()
{
	return true;
}

void Payload_RoundStart()
{
	FOREACH_CLIENT(IS_PLAYER(it),
	{
		it.player_blocked = false;
		payload_strategy_reset_player(it);
	});
	for(int j = 0; j <= payload_teams; ++j)
	{
		if(j > 0)
			TeamScore_AddToTeam(Team_IndexToTeam(j), ST_PAYLOAD_POINTS, -floor(payload_score[j]));
		payload_checkpoints_held[j] = 0;
		payload_score[j] = 0;
	}
	payload_score_time = time;
	payload_score_winner = 0;
	payload_score_finished = false;
	++payload_episode;
	IL_EACH(g_payload_carts, true, { payload_cart_reset(it); });
}

MUTATOR_HOOKFUNCTION(plc, TeamBalance_CheckAllowedTeams)
{
	TeamBalance_AllowEntityClass(M_ARGV(0, entity), "plc_team");

	return true;
}

MUTATOR_HOOKFUNCTION(plc, reset_map_players)
{
	FOREACH_CLIENT(true, {
		if(IS_PLAYER(it))
		{
			PutClientInServer(it);
			it.player_blocked = 1;
		}
	});
	return true;
}

MUTATOR_HOOKFUNCTION(plc, PlayerSpawn)
{
	entity player = M_ARGV(0, entity);

	player.player_blocked = !round_handler_IsRoundStarted();
	player.plc_str_spawn_stamp2 = time;
	player.plc_str_death_stamp = 0;
}

MUTATOR_HOOKFUNCTION(plc, PlayerDies)
{
	entity attacker = M_ARGV(1, entity);
	entity player = M_ARGV(2, entity);
	player.plc_str_death_stamp = time;
	player.plc_str_deaths += 1;
	payload_str_measure(player, PLC_OUTCOME_DEATHS, 1);
	if (IS_PLAYER(attacker) && attacker != player && DIFF_TEAM(attacker, player))
	{
		attacker.plc_str_enemy_kills += 1;
		payload_str_measure(attacker, PLC_OUTCOME_ENEMY_KILLS, 1);
		payload_str_outcome(attacker, PLC_EVT_KIND_KILL, player, 1);
	}
}

MUTATOR_HOOKFUNCTION(plc, PlayerDamaged)
{
	entity attacker = M_ARGV(0, entity);
	entity player = M_ARGV(1, entity);
	float dealt = max(0, M_ARGV(2, float)) + max(0, M_ARGV(3, float));
	if (IS_PLAYER(attacker) && IS_PLAYER(player) && attacker != player && DIFF_TEAM(attacker, player))
	{
		attacker.plc_str_enemy_damage_dealt += dealt;
		player.plc_str_enemy_damage_taken += dealt;
		payload_str_measure(attacker, PLC_OUTCOME_ENEMY_DAMAGE_DEALT, dealt);
		payload_str_measure(player, PLC_OUTCOME_ENEMY_DAMAGE_TAKEN, dealt);
		payload_str_outcome(attacker, PLC_EVT_KIND_DAMAGE, player, dealt);
	}
}

MUTATOR_HOOKFUNCTION(plc, ItemTouched)
{
	entity item = M_ARGV(0, entity);
	entity player = M_ARGV(1, entity);
	if (IS_PLAYER(player))
	{
		player.plc_str_pickups += 1;
		payload_str_measure(player, PLC_OUTCOME_PICKUPS, 1);
		payload_str_outcome(player, PLC_EVT_KIND_PICKUP, item, 1);
	}
}

MUTATOR_HOOKFUNCTION(plc, ClientConnect)
{
	entity player = M_ARGV(0, entity);

	payload_strategy_reset_player(player);
}

spawnfunc(func_plc_cart)
{
	if(!g_payload)
	{
		delete(this);
		return;
	}

	payload_cart_configure(this);
	InitializeEntity(this, payload_path_build, INITPRIO_FINDTARGET);
}

spawnfunc(plc_path)
{
	if(!g_payload)
	{
		delete(this);
		return;
	}
	this.spawnfunc_checked = true;
}

spawnfunc(plc_team)
{
	if(!g_payload || autocvar_g_payload_teams_override >= 2)
	{
		delete(this);
		return;
	}
	this.spawnfunc_checked = true;
	this.team = Team_IndexToTeam(this.cnt + 1);
}

void ScoreRules_payload(int teams)
{
	GameRules_scoring(teams, SFL_SORT_PRIO_PRIMARY, 0, {
		field_team(ST_PAYLOAD_POINTS, "points", SFL_SORT_PRIO_PRIMARY);
		field(SP_PAYLOAD_PUSH, "push", 0);
		field(SP_PAYLOAD_BLOCK, "block", 0);
	});
}

void payload_SpawnTeam(int teamcolor)
{
	entity e = new_pure(plc_team);
	e.netname = Team_ColoredFullName(teamcolor);
	e.cnt = Team_TeamToIndex(teamcolor) - 1;
	e.team = teamcolor;
	e.spawnfunc_checked = true;
}

int payload_team_count()
{
	int teams = 0;
	for(entity t = findchain(classname, "plc_team"); t; t = t.chain)
		teams = max(teams, Team_TeamToIndex(t.team));
	return teams;
}

void payload_DelayedInit(entity this)
{
	payload_teams = payload_team_count();

	if(find(NULL, classname, "plc_team") == NULL || autocvar_g_payload_teams_override >= 2)
	{
		int n = autocvar_g_payload_teams_override;
		if(n < 2)
			n = autocvar_g_payload_default_teams;
		payload_teams = bound(2, n, NUM_TEAMS);
		for(int j = 1; j <= payload_teams; ++j)
			payload_SpawnTeam(Team_IndexToTeam(j));
	}
	else
	{
		entity balance = TeamBalance_CheckAllowedTeams(NULL);
		payload_teams = payload_team_count();
		TeamBalance_Destroy(balance);
	}

	ScoreRules_payload(payload_teams);
	LOG_INFO(strcat("payload: teams ", ftos(payload_teams), ", carts ", ftos(payload_cart_count)));

	round_handler_Spawn(Payload_CheckPlayers, Payload_CheckWinner, Payload_RoundStart);
	round_handler_Init(5, autocvar_g_payload_warmup, 0);
}

void payload_Initialize()
{
	g_payload = true;
	g_payload_nodes = IL_NEW();
	g_payload_links = IL_NEW();
	g_payload_carts = IL_NEW();
	payload_link_count = 0;

	InitializeEntity(NULL, payload_DelayedInit, INITPRIO_GAMETYPE);
}
