#include "sv_payload_mesh.qh"

#include "sv_payload.qh"

#include <common/mutators/mutator/status_effects/_mod.qh>
#include <common/mutators/mutator/waypoints/waypointsprites.qh>
#include <common/resources/_mod.qh>
#include <common/stats.qh>
#include <common/weapons/_all.qh>
#include <common/resources/sv_resources.qh>
#include <server/bot/api.qh>
#include <server/utils.qh>

float(float node, float width, float maxrows) mesh_open = #644;
void(float h, float col, .float fld, float first, float n) mesh_gather = #648;
void(float h, float col, .float fld, float first, float n) mesh_scatter = #649;
float(float h, float tick, float nrows) mesh_publish = #650;
float(float h) mesh_poll = #651;
float(float h, float sel) mesh_stat = #653;

int autocvar_g_payload_mesh_node = 1;
float autocvar_g_payload_mesh_retry = 60;

void payload_mesh_attach()
{
	payload_mesh_retry = time + autocvar_g_payload_mesh_retry;
	payload_mesh_h = mesh_open(autocvar_g_payload_mesh_node, PLC_MESH_WIDTH, maxclients);
	LOG_INFO(strcat("payload: mesh handle ", ftos(payload_mesh_h), " peer ",
		ftos((payload_mesh_h >= 0) ? mesh_stat(payload_mesh_h, PLC_MESH_STAT_PEER) : -1)));
}

void payload_mesh_mark_objectives()
{
	IL_EACH(g_payload_nodes, it.sprite != NULL,
	{
		WaypointSprite_UpdateTeamRadar(it.sprite, RADARICON_WAYPOINT,
			it.plc_bank_team ? Team_ColorRGB(Team_IndexToTeam(it.plc_bank_team)) : WP_Waypoint.m_color);
	});
	for(int j = 1; j <= NUM_TEAMS; ++j)
	{
		if(!(payload_teams & Team_IndexToBit(j)))
			continue;
		entity n = payload_mesh_node(payload_mesh_objective[j]);
		if(n && n.sprite)
			WaypointSprite_UpdateTeamRadar(n.sprite, RADARICON_OBJECTIVE, Team_ColorRGB(Team_IndexToTeam(j)));
	}
}

float payload_mesh_weapons(entity e)
{
	float n = 0;
	FOREACH(Weapons, it != WEP_Null, {
		if(STAT(WEAPONS, e) & (it.m_wepset))
			n += 1;
	});
	return n;
}

float payload_mesh_weight(entity b, int o)
{
	float c, v;
	switch(o)
	{
		case 0: c = b.plc_mesh_w0; v = b.plc_mesh_v0; break;
		case 1: c = b.plc_mesh_w1; v = b.plc_mesh_v1; break;
		case 2: c = b.plc_mesh_w2; v = b.plc_mesh_v2; break;
		case 3: c = b.plc_mesh_w3; v = b.plc_mesh_v3; break;
		case 4: c = b.plc_mesh_w4; v = b.plc_mesh_v4; break;
		default: return 0;
	}
	float lt = (payload_mesh_plan_span > 0)
		? bound(0, (time - payload_mesh_plan_time) / payload_mesh_plan_span, 1) : 1;
	return v + (c - v) * lt;
}

bool payload_mesh_weights_valid(entity b)
{
	float s = b.plc_mesh_w0 + b.plc_mesh_w1 + b.plc_mesh_w2 + b.plc_mesh_w3 + b.plc_mesh_w4;
	if(s < 0.5 || s > 1.5)
		return false;
	if(b.plc_mesh_w0 < -0.01 || b.plc_mesh_w1 < -0.01 || b.plc_mesh_w2 < -0.01
		|| b.plc_mesh_w3 < -0.01 || b.plc_mesh_w4 < -0.01)
		return false;
	return true;
}

entity payload_mesh_node(float idx)
{
	if(payload_cart_count < 1)
		return NULL;
	int c = bound(0, floor(idx / PLC_MESH_OBJECTIVES), payload_cart_count - 1);
	entity a = payload_carts[c].plc_first;
	float want = idx - c * PLC_MESH_OBJECTIVES;
	for(float k = 0; a && k < want && a.plc_next; ++k)
		a = a.plc_next;
	return a;
}

void payload_mesh_tick(entity cart)
{
	if(cart.plc_cart_id != 0)
		return;
	if(payload_mesh_h < 0)
	{
		if(time >= payload_mesh_retry)
			payload_mesh_attach();
		return;
	}

	float c0p = 0, c0t = 0, c0r = 0, c1p = 0, c1t = 0, c1r = 0;
	if(payload_cart_count > 0)
	{
		entity ca = payload_carts[0];
		c0p = (ca.plc_length > 0) ? ca.plc_s / ca.plc_length : 0;
		c0t = ca.plc_ctrl;
		c0r = (ca.plc_speed_now < 0) ? 1 : 0;
	}
	if(payload_cart_count > 1)
	{
		entity cb = payload_carts[1];
		c1p = (cb.plc_length > 0) ? cb.plc_s / cb.plc_length : 0;
		c1t = cb.plc_ctrl;
		c1r = (cb.plc_speed_now < 0) ? 1 : 0;
	}

	FOREACH_CLIENTSLOT(true,
	{
		bool p = IS_PLAYER(it);
		float j = p ? Team_TeamToIndex(it.team) : 0;
		float fr = 0;
		float en = 0;
		for(entity o = findradius(it.origin, PLC_MESH_RADIUS); p && o; o = o.chain)
		{
			if(o == it || !IS_PLAYER(o))
				continue;
			if(o.team == it.team)
				fr += 1;
			else
				en += 1;
		}
		it.plc_mrow_id = p ? i : 0;
		it.plc_mrow_team = j;
		it.plc_mrow_health = p ? GetResource(it, RES_HEALTH) : 0;
		it.plc_mrow_armor = p ? GetResource(it, RES_ARMOR) : 0;
		it.plc_mrow_ammo = p ? bound(0, (GetResource(it, RES_SHELLS) + GetResource(it, RES_BULLETS)
			+ GetResource(it, RES_ROCKETS) + GetResource(it, RES_CELLS)) / 400, 1) : 0;
		it.plc_mrow_ox = p ? it.origin.x / 1024 : 0;
		it.plc_mrow_oy = p ? it.origin.y / 1024 : 0;
		it.plc_mrow_oz = p ? it.origin.z / 1024 : 0;
		it.plc_mrow_vx = p ? it.velocity.x / 1024 : 0;
		it.plc_mrow_vy = p ? it.velocity.y / 1024 : 0;
		it.plc_mrow_vz = p ? it.velocity.z / 1024 : 0;
		float dbest = 0;
		float pbest = 0;
		for(int c = 0; c < payload_cart_count; ++c)
		{
			entity ct = payload_carts[c];
			float dd = vlen(it.origin - (ct.origin + ct.view_ofs));
			if(c == 0 || dd < dbest)
			{
				dbest = dd;
				pbest = (ct.plc_length > 0) ? ct.plc_s / ct.plc_length : 0;
			}
		}
		it.plc_mrow_dist = p ? dbest : 0;
		it.plc_mrow_prog = pbest;
		it.plc_mrow_friends = fr;
		it.plc_mrow_enemies = en;
		it.plc_mrow_obj = (j >= 1 && j <= NUM_TEAMS) ? payload_mesh_objective[j] : 0;
		it.plc_mrow_weps = p ? payload_mesh_weapons(it) : 0;
		it.plc_mrow_pow = (p && (StatusEffects_active(STATUSEFFECT_Strength, it)
			|| StatusEffects_active(STATUSEFFECT_Shield, it))) ? 1 : 0;
		it.plc_mrow_tss = p ? bound(0, time - it.plc_spawn_stamp, 999) : 0;
		it.plc_mrow_pad = 0;
		it.plc_mrow_c0p = c0p;
		it.plc_mrow_c0t = c0t;
		it.plc_mrow_c0r = c0r;
		it.plc_mrow_c1p = c1p;
		it.plc_mrow_c1t = c1t;
		it.plc_mrow_c1r = c1r;
	});

	mesh_gather(payload_mesh_h, 0, plc_mrow_id, 1, maxclients);
	mesh_gather(payload_mesh_h, 1, plc_mrow_team, 1, maxclients);
	mesh_gather(payload_mesh_h, 2, plc_mrow_health, 1, maxclients);
	mesh_gather(payload_mesh_h, 3, plc_mrow_armor, 1, maxclients);
	mesh_gather(payload_mesh_h, 4, plc_mrow_ammo, 1, maxclients);
	mesh_gather(payload_mesh_h, 5, plc_mrow_ox, 1, maxclients);
	mesh_gather(payload_mesh_h, 6, plc_mrow_oy, 1, maxclients);
	mesh_gather(payload_mesh_h, 7, plc_mrow_oz, 1, maxclients);
	mesh_gather(payload_mesh_h, 8, plc_mrow_vx, 1, maxclients);
	mesh_gather(payload_mesh_h, 9, plc_mrow_vy, 1, maxclients);
	mesh_gather(payload_mesh_h, 10, plc_mrow_vz, 1, maxclients);
	mesh_gather(payload_mesh_h, 11, plc_mrow_dist, 1, maxclients);
	mesh_gather(payload_mesh_h, 12, plc_mrow_prog, 1, maxclients);
	mesh_gather(payload_mesh_h, 13, plc_mrow_friends, 1, maxclients);
	mesh_gather(payload_mesh_h, 14, plc_mrow_enemies, 1, maxclients);
	mesh_gather(payload_mesh_h, 15, plc_mrow_obj, 1, maxclients);
	mesh_gather(payload_mesh_h, 16, plc_mrow_weps, 1, maxclients);
	mesh_gather(payload_mesh_h, 17, plc_mrow_pow, 1, maxclients);
	mesh_gather(payload_mesh_h, 18, plc_mrow_tss, 1, maxclients);
	mesh_gather(payload_mesh_h, 19, plc_mrow_pad, 1, maxclients);
	mesh_gather(payload_mesh_h, 20, plc_mrow_c0p, 1, maxclients);
	mesh_gather(payload_mesh_h, 21, plc_mrow_c0t, 1, maxclients);
	mesh_gather(payload_mesh_h, 22, plc_mrow_c0r, 1, maxclients);
	mesh_gather(payload_mesh_h, 23, plc_mrow_c1p, 1, maxclients);
	mesh_gather(payload_mesh_h, 24, plc_mrow_c1t, 1, maxclients);
	mesh_gather(payload_mesh_h, 25, plc_mrow_c1r, 1, maxclients);

	payload_mesh_seq += 1;
	mesh_publish(payload_mesh_h, payload_mesh_seq, maxclients);

	float done = mesh_poll(payload_mesh_h);
	if(done <= payload_mesh_plan)
		return;
	payload_mesh_plan = done;

	FOREACH_CLIENTSLOT(true,
	{
		it.plc_mesh_v0 = payload_mesh_weight(it, 0);
		it.plc_mesh_v1 = payload_mesh_weight(it, 1);
		it.plc_mesh_v2 = payload_mesh_weight(it, 2);
		it.plc_mesh_v3 = payload_mesh_weight(it, 3);
		it.plc_mesh_v4 = payload_mesh_weight(it, 4);
	});
	payload_mesh_plan_span = bound(0.05, time - payload_mesh_plan_time, 1.0);
	payload_mesh_plan_time = time;

	mesh_scatter(payload_mesh_h, 1, plc_mesh_pick, 1, maxclients);
	mesh_scatter(payload_mesh_h, 2, plc_mesh_w0, 1, maxclients);
	mesh_scatter(payload_mesh_h, 3, plc_mesh_w1, 1, maxclients);
	mesh_scatter(payload_mesh_h, 4, plc_mesh_w2, 1, maxclients);
	mesh_scatter(payload_mesh_h, 5, plc_mesh_w3, 1, maxclients);
	mesh_scatter(payload_mesh_h, 6, plc_mesh_w4, 1, maxclients);
	mesh_scatter(payload_mesh_h, 7, plc_mesh_lead, 1, maxclients);

	FOREACH_CLIENT(IS_PLAYER(it),
	{
		bool lead = (it.plc_mesh_lead > 0.5 && it.plc_mesh_lead < 1.5);
		if(lead && !it.wps_plcleader)
			WaypointSprite_Spawn(WP_PayloadLeader, 0, 0, it, '0 0 64', NULL,
				it.team, it, wps_plcleader, true, RADARICON_FLAG);
		if(!lead && it.wps_plcleader)
			WaypointSprite_Kill(it.wps_plcleader);
	});

	int nobj = bound(1, payload_cart_count * PLC_MESH_OBJECTIVES, PLC_MESH_COMBINED);
	for(int k = 0; k < (NUM_TEAMS + 1) * PLC_MESH_COMBINED; ++k)
		payload_mesh_vote[k] = 0;

	FOREACH_CLIENT(IS_PLAYER(it),
	{
		int j = Team_TeamToIndex(it.team);
		int o = bound(0, floor(it.plc_mesh_pick), nobj - 1);
		if(j >= 1 && j <= NUM_TEAMS)
			payload_mesh_vote[j * PLC_MESH_COMBINED + o] += 1;
	});

	for(int j = 1; j <= NUM_TEAMS; ++j)
	{
		float best = 0;
		int pick = payload_mesh_objective[j];
		for(int o = 0; o < nobj; ++o)
		{
			float w = payload_mesh_vote[j * PLC_MESH_COMBINED + o];
			if(w > best)
			{
				best = w;
				pick = o;
			}
		}
		payload_mesh_objective[j] = pick;
	}

	payload_mesh_mark_objectives();
}
