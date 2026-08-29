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

bool g_payload;

int autocvar_g_payload_default_teams = 2;
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
float autocvar_g_payload_point_score = 1;
float autocvar_g_payload_capture_radius = 64;
float autocvar_g_payload_round_timelimit = 300;
float autocvar_g_payload_warmup = 5;
float autocvar_g_payload_node_fade_dist = 1100;
int autocvar_g_payload_pathlinks = 1;
int autocvar_g_payload_ribbon_max = 32;
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

int payload_pack_carts()
{
	int packed = 0;
	for(int c = 0; c < payload_cart_count; ++c)
	{
		entity cart = payload_carts[c];
		float f = (cart.plc_length > 0) ? cart.plc_s / cart.plc_length : 0;
		int q = 1 + bound(0, floor(f * (PLC_PUSH_MASK - 1)), PLC_PUSH_MASK - 1);
		packed |= (q << (c * PLC_PUSH_BITS));
	}
	return packed;
}

int payload_pack_cart_state()
{
	int packed = 0;
	for(int c = 0; c < payload_cart_count; ++c)
	{
		entity cart = payload_carts[c];
		int f = bound(0, cart.plc_ctrl, 7);
		if(cart.plc_speed_now < -autocvar_g_payload_stall_eps)
			f |= PLC_CS_REGRESS;
		else if(fabs(cart.plc_speed_now) <= autocvar_g_payload_stall_eps)
			f |= PLC_CS_STALL;
		packed |= (f << (c * PLC_CS_BITS));
	}
	return packed;
}

void payload_spawn_link(entity cart, entity a, entity b)
{
	if(payload_link_count >= autocvar_g_payload_ribbon_max)
		return;
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
	WriteHeader(MSG_ENTITY, ENT_CLIENT_RADARLINK);
	WriteByte(MSG_ENTITY, sendflags);
	if(sendflags & 1)
		WriteVector(MSG_ENTITY, this.goalentity.origin);
	if(sendflags & 2)
		WriteVector(MSG_ENTITY, this.enemy.origin);
	if(sendflags & 4)
		WriteByte(MSG_ENTITY, this.clientcolors);
	return true;
}

int payload_node_color(entity n)
{
	return n.plc_banked ? Team_IndexToTeam(n.plc_bank_team) - 1 : 0;
}

void payload_update_links(entity cart)
{
	int contest = cart.plc_ctrl ? Team_IndexToTeam(cart.plc_ctrl) - 1 : 0;
	IL_EACH(g_payload_links, it.plc_cart == cart,
	{
		entity a = it.goalentity;
		entity b = it.enemy;
		int cc;
		if(a.plc_dist <= cart.plc_s && b.plc_dist > cart.plc_s)
			cc = contest | (contest << 4);
		else
			cc = payload_node_color(a) | (payload_node_color(b) << 4);
		if(cc != it.clientcolors)
		{
			it.clientcolors = cc;
			it.SendFlags |= 4;
		}
	});
}

void set_payload_state(entity e)
{
	entity c0 = (payload_cart_count > 0) ? payload_carts[0] : NULL;
	STAT(PAYLOAD_PROGRESS, e) = (c0 && c0.plc_length > 0) ? c0.plc_s / c0.plc_length : 0;
	STAT(PAYLOAD_SPEED, e) = c0 ? c0.plc_speed_now : 0;
	STAT(PAYLOAD_PUSH_PACKED, e) = payload_pack_push();
	STAT(PAYLOAD_GOALS_PACKED, e) = payload_pack_carts();
	STAT(PAYLOAD_CARTS_STATE, e) = payload_pack_cart_state();
}

void payload_bank_points(entity this, int ctrl, float s_prev, float s_new)
{
	int bit = Team_IndexToBit(ctrl);
	int t = Team_IndexToTeam(ctrl);
	IL_EACH(g_payload_nodes, it.plc_cart == this && it.plc_dist > s_prev && it.plc_dist <= s_new,
	{
		if(it.plc_banked & bit)
			continue;
		it.plc_banked |= bit;
		it.plc_bank_team = ctrl;
		payload_round_banks[ctrl] += 1;
		TeamScore_AddToTeam(t, ST_PAYLOAD_POINTS, autocvar_g_payload_point_score);
		LOG_INFO(strcat("payload: bank cart ", ftos(this.plc_cart_id), " team ", ftos(t),
			" point ", ftos(it.plc_idx), " s ", ftos(it.plc_dist)));
		payload_EventLog(strcat("bank:", ftos(this.plc_cart_id), ":", ftos(it.plc_idx)), t);
		if(it.sprite)
			WaypointSprite_UpdateTeamRadar(it.sprite, RADARICON_WAYPOINT, Team_ColorRGB(t));
	});
}

void payload_unbank_points(entity this, float s_prev, float s_new)
{
	// The cart retreated from s_prev down to s_new (s_new < s_prev). Every banked
	// control point it uncrosses is freed — a retreat destroys banked position,
	// not just tempo (design/payload-strategy-spec.md §2). Frees any team's bank,
	// since the progress that earned it is being erased.
	IL_EACH(g_payload_nodes, it.plc_cart == this && it.plc_dist <= s_prev && it.plc_dist > s_new,
	{
		if(!it.plc_banked)
			continue;
		int ct = it.plc_bank_team;
		int t = Team_IndexToTeam(ct);
		it.plc_banked = 0;
		it.plc_bank_team = 0;
		if(payload_round_banks[ct] > 0)
			payload_round_banks[ct] -= 1;
		TeamScore_AddToTeam(t, ST_PAYLOAD_POINTS, -autocvar_g_payload_point_score);
		LOG_INFO(strcat("payload: unbank cart ", ftos(this.plc_cart_id), " team ", ftos(t),
			" point ", ftos(it.plc_idx), " s ", ftos(it.plc_dist)));
		payload_EventLog(strcat("unbank:", ftos(this.plc_cart_id), ":", ftos(it.plc_idx)), t);
		if(it.sprite)
			WaypointSprite_UpdateTeamRadar(it.sprite, RADARICON_WAYPOINT, '0.5 0.5 0.5');
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

	bool live = !game_stopped && time >= game_starttime
		&& (!round_handler_IsActive() || round_handler_IsRoundStarted());

	payload_occupancy(this);

	float total = 0, best = 0;
	int ctrl = 0;
	bool tie = false;
	for(int j = 1; j <= NUM_TEAMS; ++j)
	{
		total += payload_w[j];
		if(payload_w[j] > best)
		{
			best = payload_w[j];
			ctrl = j;
			tie = false;
		}
		else if(payload_w[j] == best && best > 0)
			tie = true;
	}
	if(tie)
		ctrl = 0;
	// --- Two-regime signed velocity ---
	// design/payload-strategy-spec.md §2 (the game), design/cart-force-field.md
	// (the continuous/differentiable form). plc_ctrl is the cart's *sticky color*
	// — the team it banks for, changed only at origin; `ctrl` is this tick's
	// instantaneous plurality.
	int color = this.plc_ctrl;

	// A neutral cart (fresh, or just collapsed to origin) is claimed by whoever
	// holds the plurality now.
	if(color == 0 && ctrl)
		color = this.plc_ctrl = ctrl;

	float wA = color ? payload_w[color] : 0;   // the color team's presence
	float wopp = total - wA;                    // every other team present

	// strongest present opposing team — the one that walks an abandoned cart home
	int opp = 0; float oppw = 0;
	for(int j = 1; j <= NUM_TEAMS; ++j)
		if(j != color && payload_w[j] > oppw) { oppw = payload_w[j]; opp = j; }

	float v;
	if(color && wA > 0)
	{
		// Regime A — contested: bounded local tug-of-war. Backward drift is
		// capped at contest_speed while defenders are present, so a cart being
		// fought over stays near where the fight is; it is never driven home
		// while its own team is still swinging.
		v = this.speed * (wA - wopp) / (1 + wopp * wopp);
		v = bound(-autocvar_g_payload_contest_speed, v, autocvar_g_payload_max_speed);
	}
	else if(color && oppw > 0)
	{
		// Regime B — abandoned: the color team has left; the strongest opponent
		// walks the cart back toward origin, linear in its player count (this is
		// capture, not contest — no quadratic damping).
		v = -autocvar_g_payload_reverse_speed * oppw;
		v = bound(-autocvar_g_payload_max_speed, v, 0);
	}
	else
	{
		// empty, or a colored cart with no opposition present: it waits.
		v = 0;
	}

	if(total <= 0)
		this.plc_idle += PLC_TICK;
	else
		this.plc_idle = 0;

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

	if(color && s_new > s_prev)
		payload_bank_points(this, color, s_prev, s_new);
	if(s_new < s_prev)
		payload_unbank_points(this, s_prev, s_new);
	if(s_new <= 0 && s_prev > 0)
	{
		payload_clear_progress(this);
		this.plc_ctrl = 0;   // recolor at origin: neutral, next plurality claims it
		LOG_INFO(strcat("payload: cart ", ftos(this.plc_cart_id), " regressed to origin, recolored neutral"));
		payload_EventLog("origin", this.plc_cart_id);
	}

	float gain = (s_new > s_prev && color && payload_n[color] > 0)
		? (s_new - s_prev) * autocvar_g_payload_score_rate / payload_n[color] : 0;
	bool stalled = (total > 0) && (fabs(this.plc_speed_now) < autocvar_g_payload_stall_eps);
	FOREACH_CLIENT(IS_PLAYER(it) && it.plc_in,
	{
		if(gain > 0 && it.plc_in == color)
			GameRules_scoring_add_float2int(it, PAYLOAD_PUSH, gain, plc_push_accum, 1);
		if(stalled)
			GameRules_scoring_add_float2int(it, PAYLOAD_BLOCK, autocvar_g_payload_block_rate * PLC_TICK, plc_block_accum, 1);
	});

	WaypointSprite_UpdateHealth(this.sprite, s_new);
	payload_update_links(this);

	if(this.plc_cart_id == 0)
	{
		FOREACH_CLIENT(IS_REAL_CLIENT(it), { set_payload_state(it); });
	}

}

void payload_cart_reset(entity this)
{
	this.plc_s = 0;
	this.plc_idle = 0;
	this.plc_ctrl = 0;
	this.velocity = '0 0 0';
	this.plc_speed_now = 0;
	payload_clear_progress(this);
	setorigin(this, payload_pos(this, 0) - this.view_ofs);
	FOREACH_CLIENT(IS_REAL_CLIENT(it), { set_payload_state(it); });
}

void payload_node_sprite(entity node, float maxd)
{
	if(node.sprite)
		return;
	WaypointSprite_Spawn(WP_Waypoint, 0, maxd, NULL, node.origin + '0 0 48',
		NULL, 0, node, sprite, true, RADARICON_WAYPOINT);
}

void payload_path_build(entity this)
{
	this.plc_cart_id = payload_cart_count;
	if(payload_cart_count < PLC_MAX_CARTS)
	{
		payload_carts[payload_cart_count] = this;
		++payload_cart_count;
	}

	entity first = (this.target != "") ? find(NULL, targetname, this.target) : NULL;
	if(!first)
	{
		first = new_pure(plc_path);
		setorigin(first, this.origin + this.view_ofs);
		LOG_INFO("payload: func_plc_cart has no reachable plc_path, using a single node at its own origin");
	}
	this.plc_first = first;

	entity prev = NULL;
	entity node = first;
	float d = 0;
	int guard = 0;
	int interior = 0;
	while(node && guard < 4096)
	{
		IL_PUSH(g_payload_nodes, node);
		if(node.spawnflags & PLC_CHECKPOINT)
			payload_node_sprite(node, 0);
		else if(interior < autocvar_g_payload_node_sprites)
		{
			payload_node_sprite(node, autocvar_g_payload_node_fade_dist);
			++interior;
		}
		node.plc_prev = prev;
		node.plc_cart = this;
		node.plc_idx = guard;
		++guard;
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

	entity wp = WP_PayloadCart2;
	if(this.plc_cart_id == 0)
		wp = WP_PayloadCart;
	WaypointSprite_Spawn(wp, 0, 0, this, '0 0 64', NULL, 0, this, sprite, true, RADARICON_OBJECTIVE);
	WaypointSprite_UpdateMaxHealth(this.sprite, this.plc_length);
	WaypointSprite_UpdateHealth(this.sprite, this.plc_s);

	setthink(this, payload_think);
	this.nextthink = this.ltime + PLC_TICK;

	LOG_INFO(strcat("payload: cart ", ftos(this.plc_cart_id), ": ", ftos(guard),
		" path nodes, length ", ftos(this.plc_length)));
}

int payload_leading_team()
{
	int bestj = 0;
	float bestb = 0;
	int ties = 0;
	for(int j = 1; j <= NUM_TEAMS; ++j)
	{
		if(payload_round_banks[j] > bestb)
		{
			bestb = payload_round_banks[j];
			bestj = j;
			ties = 1;
		}
		else if(payload_round_banks[j] == bestb && bestb > 0)
			++ties;
	}
	return (bestj && ties == 1) ? Team_IndexToTeam(bestj) : 0;
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
		payload_RoundEnd(payload_leading_team());
		return true;
	}

	for(int c = 0; c < payload_cart_count; ++c)
	{
		entity cart = payload_carts[c];
		if(cart.plc_ctrl && cart.plc_length > 0
			&& cart.plc_s >= cart.plc_length - autocvar_g_payload_capture_radius)
		{
			int t = Team_IndexToTeam(cart.plc_ctrl);
			LOG_INFO(strcat("payload: cart ", ftos(cart.plc_cart_id), " delivered by team ", ftos(t)));
			payload_RoundEnd(t);
			return true;
		}
	}
	return false;
}

bool Payload_CheckPlayers()
{
	return true;
}

void Payload_RoundStart()
{
	FOREACH_CLIENT(IS_PLAYER(it), { it.player_blocked = false; });
	for(int j = 0; j <= NUM_TEAMS; ++j)
		payload_round_banks[j] = 0;
	for(int c = 0; c < payload_cart_count; ++c)
		payload_cart_reset(payload_carts[c]);
}

void havocbot_goalrating_payload(entity this, float ratingscale)
{
	// Bare stub. Rates carts and near-cart nodes so bots engage the cart
	// demonstrator. Strategy-weighted objective selection removed: there is no
	// strategy implementation; design/payload-spec.md is the reference.
	for(int c = 0; c < payload_cart_count; ++c)
		navigation_routerating(this, payload_carts[c], ratingscale, 5000);
	IL_EACH(g_payload_nodes, it.plc_cart != NULL,
	{
		float rel = (it.plc_dist - it.plc_cart.plc_s) / 256;
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
	this.view_ofs = this.mins;
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
		field_team(ST_PAYLOAD_POINTS, "points", 0);
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

	LOG_INFO(strcat("payload: teams mask ", ftos(payload_teams), ", carts ", ftos(payload_cart_count)));

	round_handler_Spawn(Payload_CheckPlayers, Payload_CheckWinner, Payload_RoundStart);
	round_handler_Init(5, autocvar_g_payload_warmup, autocvar_g_payload_round_timelimit);
}

void payload_Initialize()
{
	g_payload = true;
	g_payload_nodes = IL_NEW();
	g_payload_links = IL_NEW();
	payload_link_count = 0;

	InitializeEntity(NULL, payload_DelayedInit, INITPRIO_GAMETYPE);
}
