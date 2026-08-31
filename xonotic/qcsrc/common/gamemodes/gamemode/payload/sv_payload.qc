#include "sv_payload.qh"

#include <common/notifications/all.qh>
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

bool g_payload;

int autocvar_g_payload_default_teams = 2;
int autocvar_g_payload_teams_override;
float autocvar_g_payload_speed = 30;
float autocvar_g_payload_max_speed = 200;
float autocvar_g_payload_push_radius = 160;
float autocvar_g_payload_push_height = 96;
int autocvar_g_payload_push_cap = 3;
float autocvar_g_payload_push_falloff = 1;
float autocvar_g_payload_idle_time = 8;
float autocvar_g_payload_rollback_speed = 20;
float autocvar_g_payload_score_rate = 0.01;
float autocvar_g_payload_block_rate = 1;
float autocvar_g_payload_stall_eps = 1;
float autocvar_g_payload_round_timelimit = 300;
float autocvar_g_payload_warmup = 5;

entity payload_first;
float payload_gain[NUM_TEAMS + 1];
float payload_speed_now;

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

vector payload_pos(float s)
{
	s = bound(0, s, payload_length);
	entity a = payload_first;
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

float payload_nearest_node_s(vector org)
{
	float best = 0;
	float bestd = -1;
	IL_EACH(g_payload_nodes, true,
	{
		float d = vlen(it.origin - org);
		if(bestd < 0 || d < bestd)
		{
			bestd = d;
			best = it.plc_dist;
		}
	});
	return best;
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
		traceline(it.origin + it.view_ofs, c, MOVE_NOMONSTERS, it);
		if(trace_fraction < 1 && trace_ent != cart)
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

float payload_rollback_target(entity cart)
{
	float dir = (cart.plc_home_s >= cart.plc_s) ? 1 : -1;
	float best = cart.plc_home_s;
	IL_EACH(g_payload_nodes, it.spawnflags & PLC_CHECKPOINT,
	{
		float d = (it.plc_dist - cart.plc_s) * dir;
		if(d > 0 && d < (best - cart.plc_s) * dir)
			best = it.plc_dist;
	});
	return best;
}

int payload_pack_push()
{
	int packed = 0;
	for(int j = 1; j <= NUM_TEAMS; ++j)
	{
		int q = bound(0, floor(payload_w[j] * PLC_PUSH_SCALE), PLC_PUSH_MASK);
		packed |= (q << ((j - 1) * PLC_PUSH_BITS));
	}
	return packed;
}

int payload_pack_goals()
{
	int packed = 0;
	for(int j = 1; j <= NUM_TEAMS; ++j)
	{
		if(!(payload_goal_mask & Team_IndexToBit(j)))
			continue;
		float f = (payload_length > 0) ? payload_goal_s[j] / payload_length : 0;
		int q = 1 + bound(0, floor(f * (PLC_PUSH_MASK - 1)), PLC_PUSH_MASK - 1);
		packed |= (q << ((j - 1) * PLC_PUSH_BITS));
	}
	return packed;
}

void set_payload_state(entity e)
{
	STAT(PAYLOAD_PROGRESS, e) = (payload_length > 0 && payload_cart) ? payload_cart.plc_s / payload_length : 0;
	STAT(PAYLOAD_SPEED, e) = payload_speed_now;
	STAT(PAYLOAD_PUSH_PACKED, e) = payload_pack_push();
	STAT(PAYLOAD_GOALS_PACKED, e) = payload_pack_goals();
}

void payload_think(entity this)
{
	this.nextthink = this.ltime + PLC_TICK;

	bool live = !game_stopped && time >= game_starttime
		&& (!round_handler_IsActive() || round_handler_IsRoundStarted());

	payload_occupancy(this);

	float pplus = 0, pminus = 0, total = 0;
	for(int j = 1; j <= NUM_TEAMS; ++j)
	{
		total += payload_w[j];
		if(!(payload_goal_mask & Team_IndexToBit(j)))
			continue;
		float d = payload_goal_s[j] - this.plc_s;
		pplus += (d > 0) ? payload_w[j] : 0;
		pminus += (d < 0) ? payload_w[j] : 0;
	}

	float v = bound(-autocvar_g_payload_max_speed, this.speed * (pplus - pminus), autocvar_g_payload_max_speed);
	float target = this.plc_s;
	bool rolling = false;

	if(total <= 0)
	{
		this.plc_idle += PLC_TICK;
		if(this.plc_idle >= autocvar_g_payload_idle_time)
		{
			target = payload_rollback_target(this);
			rolling = true;
			v = ((target > this.plc_s) ? 1 : -1) * autocvar_g_payload_rollback_speed;
		}
	}
	else
		this.plc_idle = 0;

	v = live ? v : 0;
	float s_prev = this.plc_s;
	float s_new = bound(0, this.plc_s + v * PLC_TICK, payload_length);
	if(rolling)
		s_new = (target > s_prev) ? min(s_new, target) : max(s_new, target);
	this.plc_s = s_new;
	payload_speed_now = (s_new - s_prev) / PLC_TICK;

	vector want = payload_pos(s_new);
	this.velocity = (want - (this.origin + this.view_ofs)) * (1 / PLC_TICK);

	if(this.spawnflags & PLC_CART_TURN)
	{
		vector ahead = payload_pos(min(payload_length, s_new + 16)) - payload_pos(max(0, s_new - 16));
		if(vlen(ahead) > 0)
			this.angles = vectoangles(ahead);
	}

	for(int j = 1; j <= NUM_TEAMS; ++j)
	{
		payload_gain[j] = 0;
		if(!(payload_goal_mask & Team_IndexToBit(j)))
			continue;
		float g = (fabs(s_prev - payload_goal_s[j]) - fabs(s_new - payload_goal_s[j])) * autocvar_g_payload_score_rate;
		if(g <= 0)
			continue;
		payload_gain[j] = g;
	}

	bool stalled = (total > 0) && (fabs(payload_speed_now) < autocvar_g_payload_stall_eps);
	FOREACH_CLIENT(IS_PLAYER(it) && it.plc_in,
	{
		int j = it.plc_in;
		if(payload_gain[j] > 0)
			GameRules_scoring_add_float2int(it, PAYLOAD_PUSH, payload_gain[j] / payload_n[j], plc_push_accum, 1);
		if(stalled)
			GameRules_scoring_add_float2int(it, PAYLOAD_BLOCK, autocvar_g_payload_block_rate * PLC_TICK, plc_block_accum, 1);
	});

	WaypointSprite_UpdateHealth(this.sprite, s_new);

	FOREACH_CLIENT(IS_REAL_CLIENT(it), { set_payload_state(it); });
}

void payload_cart_reset(entity this)
{
	this.plc_s = this.plc_home_s;
	this.plc_idle = 0;
	this.velocity = '0 0 0';
	payload_speed_now = 0;
	setorigin(this, payload_pos(this.plc_s) - this.view_ofs);
	FOREACH_CLIENT(IS_REAL_CLIENT(it), { set_payload_state(it); });
}

void payload_path_build(entity this)
{
	payload_cart = this;
	entity first = (this.target != "") ? find(NULL, targetname, this.target) : NULL;
	if(!first)
	{
		first = new_pure(plc_path);
		setorigin(first, this.origin + this.view_ofs);
		LOG_INFO("payload: func_plc_cart has no reachable plc_path, using a single node at its own origin");
	}
	payload_first = first;

	entity prev = NULL;
	entity node = first;
	float d = 0;
	int guard = 0;
	while(node && guard < 4096)
	{
		++guard;
		IL_PUSH(g_payload_nodes, node);
		node.plc_prev = prev;
		if(node.curvetarget != "")
			node.plc_curve = find(NULL, targetname, node.curvetarget);
		if(prev)
		{
			d += payload_seglen(prev, node);
			prev.plc_next = node;
		}
		node.plc_dist = d;
		entity nxt = (node.target != "") ? find(NULL, targetname, node.target) : NULL;
		if(nxt == first || (nxt && nxt.plc_prev))
			nxt = NULL;
		prev = node;
		node = nxt;
	}
	payload_length = d;

	payload_goal_mask = 0;
	for(int j = 0; j <= NUM_TEAMS; ++j)
	{
		payload_goal_s[j] = 0;
		payload_goal_r[j] = 0;
	}
	for(entity g = findchain(classname, "plc_goal"); g; g = g.chain)
	{
		int j = Team_TeamToIndex(g.team);
		if(j < 1)
			continue;
		entity n = (g.target != "") ? find(NULL, targetname, g.target) : NULL;
		payload_goal_s[j] = n ? n.plc_dist : payload_nearest_node_s(g.origin);
		payload_goal_r[j] = (g.radius > 0) ? g.radius : 64;
		payload_goal_mask |= Team_IndexToBit(j);
	}
	payload_goal_mask &= payload_teams;

	entity home = (this.plc_start != "") ? find(NULL, targetname, this.plc_start) : NULL;
	this.plc_home_s = home ? home.plc_dist : payload_nearest_node_s(this.origin + this.view_ofs);
	this.plc_s = this.plc_home_s;
	setorigin(this, payload_pos(this.plc_s) - this.view_ofs);

	WaypointSprite_SpawnFixed(WP_PayloadCart, '0 0 64', this, sprite, RADARICON_OBJECTIVE);
	WaypointSprite_UpdateMaxHealth(this.sprite, payload_length);
	WaypointSprite_UpdateHealth(this.sprite, this.plc_s);

	setthink(this, payload_think);
	this.nextthink = this.ltime + PLC_TICK;

	LOG_INFO(strcat("payload: ", ftos(guard), " path nodes, length ", ftos(payload_length),
		", goal mask ", ftos(payload_goal_mask), ", home ", ftos(this.plc_home_s)));
}

int payload_goal_reached_team()
{
	if(!payload_cart)
		return 0;
	for(int j = 1; j <= NUM_TEAMS; ++j)
	{
		if(!(payload_goal_mask & Team_IndexToBit(j)))
			continue;
		if(fabs(payload_cart.plc_s - payload_goal_s[j]) <= payload_goal_r[j])
			return Team_IndexToTeam(j);
	}
	return 0;
}

int payload_nearest_goal_team()
{
	if(!payload_cart)
		return 0;
	int best = 0;
	float bestd = -1;
	int ties = 0;
	for(int j = 1; j <= NUM_TEAMS; ++j)
	{
		if(!(payload_goal_mask & Team_IndexToBit(j)))
			continue;
		float d = fabs(payload_cart.plc_s - payload_goal_s[j]);
		if(bestd < 0 || d < bestd)
		{
			bestd = d;
			best = Team_IndexToTeam(j);
			ties = 1;
		}
		else if(d == bestd)
			++ties;
	}
	return (ties == 1) ? best : 0;
}

void payload_RoundEnd(int winner_team)
{
	if(winner_team > 0)
	{
		Send_Notification(NOTIF_ALL, NULL, MSG_CENTER, APP_TEAM_NUM(winner_team, CENTER_ROUND_TEAM_WIN));
		Send_Notification(NOTIF_ALL, NULL, MSG_INFO, APP_TEAM_NUM(winner_team, INFO_ROUND_TEAM_WIN));
		TeamScore_AddToTeam(winner_team, ST_PAYLOAD_CAPS, +1);
		payload_EventLog("capture", winner_team);
	}
	else
	{
		Send_Notification(NOTIF_ALL, NULL, MSG_CENTER, CENTER_ROUND_TIED);
		Send_Notification(NOTIF_ALL, NULL, MSG_INFO, INFO_ROUND_TIED);
		payload_EventLog("tie", 0);
	}

	game_stopped = true;
	round_handler_Init(5, autocvar_g_payload_warmup, autocvar_g_payload_round_timelimit);
}

bool Payload_CheckWinner()
{
	if(round_handler_GetEndTime() > 0 && round_handler_GetEndTime() - time <= 0)
	{
		payload_RoundEnd(payload_nearest_goal_team());
		return true;
	}

	int t = payload_goal_reached_team();
	if(t <= 0)
		return false;

	payload_RoundEnd(t);
	return true;
}

bool Payload_CheckPlayers()
{
	return true;
}

void Payload_RoundStart()
{
	FOREACH_CLIENT(IS_PLAYER(it), { it.player_blocked = false; });
	if(payload_cart)
		payload_cart_reset(payload_cart);
}

void havocbot_goalrating_payload(entity this, float ratingscale)
{
	if(!payload_cart)
		return;
	int j = Team_TeamToIndex(this.team);
	navigation_routerating(this, payload_cart, ratingscale, 5000);
	if(j < 1 || !(payload_goal_mask & Team_IndexToBit(j)))
		return;
	float d = payload_goal_s[j] - payload_cart.plc_s;
	float lead = bound(-256, d, 256);
	IL_EACH(g_payload_nodes, true,
	{
		float rel = (it.plc_dist - payload_cart.plc_s) / ((lead >= 0) ? 256 : -256);
		if(rel > 0 && rel < 1)
			navigation_routerating(this, it, ratingscale * 0.5, 3000);
	});
}

void havocbot_role_payload(entity this)
{
	if(IS_DEAD(this))
		return;

	if(navigation_goalrating_timeout(this))
	{
		navigation_goalrating_start(this);
		havocbot_goalrating_payload(this, 20000);
		havocbot_goalrating_items(this, 10000, this.origin, 8000);
		havocbot_goalrating_waypoints(this, 1, this.origin, 3000);
		navigation_goalrating_end(this);

		navigation_goalrating_timeout_set(this);
	}
}

MUTATOR_HOOKFUNCTION(plc, HavocBot_ChooseRole)
{
	entity bot = M_ARGV(0, entity);

	bot.havocbot_role = havocbot_role_payload;
	return true;
}

MUTATOR_HOOKFUNCTION(plc, TeamBalance_CheckAllowedTeams)
{
	M_ARGV(0, float) = payload_teams;
	entity head = find(NULL, classname, "plc_team");
	while(head)
	{
		if(Team_IsValidTeam(head.team))
			M_ARGV(0, float) |= Team_TeamToBit(head.team);
		head = find(head, classname, "plc_team");
	}

	M_ARGV(1, string) = string_null;

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
		if(IS_REAL_CLIENT(it))
			set_payload_state(it);
	});
	return true;
}

MUTATOR_HOOKFUNCTION(plc, PlayerSpawn)
{
	entity player = M_ARGV(0, entity);

	player.player_blocked = !round_handler_IsRoundStarted();
}

MUTATOR_HOOKFUNCTION(plc, ClientConnect)
{
	entity player = M_ARGV(0, entity);

	set_payload_state(player);
}

spawnfunc(func_plc_cart)
{
	if(!g_payload)
	{
		delete(this);
		return;
	}

	this.effects |= EF_LOWPRECISION;
	this.view_ofs = this.mins;
	if(!this.speed)
		this.speed = autocvar_g_payload_speed;
	if(!this.radius)
		this.radius = autocvar_g_payload_push_radius;
	if(!this.height)
		this.height = autocvar_g_payload_push_height;
	if(this.dmg && !this.dmgtime)
		this.dmgtime = 0.25;
	if(this.dmg && this.message == "")
		this.message = " was squished";
	if(this.dmg && this.message2 == "")
		this.message2 = "was squished by";
	this.dmgtime2 = time;

	InitMovingBrushTrigger(this);
	setblocked(this, generic_plat_blocked);
	this.reset = payload_cart_reset;

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

spawnfunc(plc_goal)
{
	if(!g_payload)
	{
		delete(this);
		return;
	}
	this.spawnfunc_checked = true;
	if(this.cnt)
		this.team = this.cnt + 1;
}

spawnfunc(plc_team)
{
	if(!g_payload || autocvar_g_payload_teams_override >= 2)
	{
		delete(this);
		return;
	}
	this.spawnfunc_checked = true;
	if(this.cnt)
		this.team = this.cnt + 1;
}

void ScoreRules_payload(int teams)
{
	GameRules_scoring(teams, SFL_SORT_PRIO_PRIMARY, 0, {
		field_team(ST_PAYLOAD_CAPS, "caps", SFL_SORT_PRIO_PRIMARY);
		field(SP_PAYLOAD_PUSH, "push", 0);
		field(SP_PAYLOAD_BLOCK, "block", 0);
	});
}

void payload_SpawnTeam(int teamcolor)
{
	entity e = new_pure(plc_team);
	e.netname = Team_ColoredFullName(teamcolor);
	e.cnt = teamcolor - 1;
	e.team = teamcolor;
	e.spawnfunc_checked = true;
}

int payload_goal_team_bits()
{
	int bits = 0;
	for(entity g = findchain(classname, "plc_goal"); g; g = g.chain)
		bits |= Team_TeamToBit(g.team);
	return bits;
}

void payload_DelayedInit(entity this)
{
	payload_teams = payload_goal_team_bits();

	if(find(NULL, classname, "plc_team") == NULL || autocvar_g_payload_teams_override >= 2)
	{
		int n = autocvar_g_payload_teams_override;
		if(n < 2)
			n = autocvar_g_payload_default_teams;
		payload_teams |= BITS(bound(2, n, NUM_TEAMS));
		for(int j = 1; j <= NUM_TEAMS; ++j)
			if(payload_teams & Team_IndexToBit(j))
				payload_SpawnTeam(Team_IndexToTeam(j));
	}
	else
	{
		entity balance = TeamBalance_CheckAllowedTeams(NULL);
		payload_teams = TeamBalance_GetAllowedTeams(balance);
		TeamBalance_Destroy(balance);
	}

	ScoreRules_payload(payload_teams);

	LOG_INFO(strcat("payload: teams mask ", ftos(payload_teams)));

	round_handler_Spawn(Payload_CheckPlayers, Payload_CheckWinner, Payload_RoundStart);
	round_handler_Init(5, autocvar_g_payload_warmup, autocvar_g_payload_round_timelimit);
}

void payload_Initialize()
{
	g_payload = true;
	g_payload_nodes = IL_NEW();

	InitializeEntity(NULL, payload_DelayedInit, INITPRIO_GAMETYPE);
}
