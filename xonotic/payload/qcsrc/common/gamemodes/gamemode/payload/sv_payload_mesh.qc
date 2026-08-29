#include "sv_payload_mesh.qh"

#include "sv_payload.qh"

#include <common/items/_mod.qh>
#include <common/mutators/mutator/powerups/_mod.qh>
#include <common/mutators/mutator/status_effects/_mod.qh>
#include <common/mutators/mutator/waypoints/waypointsprites.qh>
#include <common/resources/_mod.qh>
#include <common/stats.qh>
#include <common/weapons/_all.qh>
#include <common/resources/sv_resources.qh>
#include <server/bot/api.qh>
#include <server/items/items.qh>
#include <server/items/spawning.qh>
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

float payload_mesh_item_class_rank(entity e)
{
	if(e.itemdef == ITEM_Strength) return 6;
	if(e.itemdef == ITEM_Shield) return 5;
	if(e.itemdef == ITEM_ArmorMega) return 4;
	if(e.itemdef == ITEM_HealthMega) return 3;
	if(e.itemdef == ITEM_ArmorBig) return 2;
	return 0;
}

void payload_mesh_scan_items()
{
	payload_mesh_items_scanned = true;
	for(int p = 0; p < PLC_MESH_ITEMS; ++p)
	{
		payload_mesh_items[p] = NULL;
		payload_mesh_item_rank[p] = 0;
		payload_mesh_item_org[p] = '0 0 0';
		payload_mesh_item_avail[p] = 0;
	}
	IL_EACH(g_items, it.itemdef != NULL && !Item_IsLoot(it),
	{
		float r = payload_mesh_item_class_rank(it);
		if(r > 0)
		{
			for(int p = 0; p < PLC_MESH_ITEMS; ++p)
			{
				if(r > payload_mesh_item_rank[p])
				{
					for(int q = PLC_MESH_ITEMS - 1; q > p; --q)
					{
						payload_mesh_items[q] = payload_mesh_items[q - 1];
						payload_mesh_item_rank[q] = payload_mesh_item_rank[q - 1];
						payload_mesh_item_org[q] = payload_mesh_item_org[q - 1];
					}
					payload_mesh_items[p] = it;
					payload_mesh_item_rank[p] = r;
					payload_mesh_item_org[p] = it.origin;
					break;
				}
			}
		}
	});
	float np = 0;
	for(int p = 0; p < PLC_MESH_ITEMS; ++p)
		if(payload_mesh_item_rank[p] > 0)
			np += 1;
	LOG_INFO(strcat("payload: mesh item posts ", ftos(np)));
}

entity payload_mesh_obj_ent(float idx)
{
	if(idx >= PLC_MESH_ITEM_BASE)
	{
		int p = bound(0, idx - PLC_MESH_ITEM_BASE, PLC_MESH_ITEMS - 1);
		entity ie = payload_mesh_items[p];
		if(payload_mesh_item_rank[p] > 0 && ie != NULL && ie.itemdef != NULL)
			return ie;
		return NULL;
	}
	return payload_mesh_node(idx);
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
		if(payload_mesh_objective[j] >= PLC_MESH_ITEM_BASE)
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
	float nn = 0;
	for(entity t = a; t; t = t.plc_next)
		nn += 1;
	float want = rint((idx - c * PLC_MESH_OBJECTIVES) * (nn - 1) / (PLC_MESH_OBJECTIVES - 1));
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
	if(!payload_mesh_items_scanned)
		payload_mesh_scan_items();
	for(int p = 0; p < PLC_MESH_ITEMS; ++p)
	{
		entity ie = payload_mesh_items[p];
		payload_mesh_item_avail[p] = (payload_mesh_item_rank[p] > 0 && ie != NULL
			&& ie.itemdef != NULL && (ie.ItemStatus & ITS_AVAILABLE)) ? 1 : 0;
	}

	float c0p = 0, c0t = 0, c0r = 0, c1p = 0, c1t = 0, c1r = 0;
	float c2p = 0, c2t = 0, c2r = 0, c3p = 0, c3t = 0, c3r = 0;
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
	if(payload_cart_count > 2)
	{
		entity cc = payload_carts[2];
		c2p = (cc.plc_length > 0) ? cc.plc_s / cc.plc_length : 0;
		c2t = cc.plc_ctrl;
		c2r = (cc.plc_speed_now < 0) ? 1 : 0;
	}
	if(payload_cart_count > 3)
	{
		entity cd = payload_carts[3];
		c3p = (cd.plc_length > 0) ? cd.plc_s / cd.plc_length : 0;
		c3t = cd.plc_ctrl;
		c3r = (cd.plc_speed_now < 0) ? 1 : 0;
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
		it.plc_mrow_c2p = c2p;
		it.plc_mrow_c2t = c2t;
		it.plc_mrow_c2r = c2r;
		it.plc_mrow_c3p = c3p;
		it.plc_mrow_c3t = c3t;
		it.plc_mrow_c3r = c3r;
		it.plc_mrow_i0r = payload_mesh_item_rank[0];
		it.plc_mrow_i0x = payload_mesh_item_org[0].x / 1024;
		it.plc_mrow_i0y = payload_mesh_item_org[0].y / 1024;
		it.plc_mrow_i0z = payload_mesh_item_org[0].z / 1024;
		it.plc_mrow_i0a = payload_mesh_item_avail[0];
		it.plc_mrow_i1r = payload_mesh_item_rank[1];
		it.plc_mrow_i1x = payload_mesh_item_org[1].x / 1024;
		it.plc_mrow_i1y = payload_mesh_item_org[1].y / 1024;
		it.plc_mrow_i1z = payload_mesh_item_org[1].z / 1024;
		it.plc_mrow_i1a = payload_mesh_item_avail[1];
		it.plc_mrow_i2r = payload_mesh_item_rank[2];
		it.plc_mrow_i2x = payload_mesh_item_org[2].x / 1024;
		it.plc_mrow_i2y = payload_mesh_item_org[2].y / 1024;
		it.plc_mrow_i2z = payload_mesh_item_org[2].z / 1024;
		it.plc_mrow_i2a = payload_mesh_item_avail[2];
		it.plc_mrow_i3r = payload_mesh_item_rank[3];
		it.plc_mrow_i3x = payload_mesh_item_org[3].x / 1024;
		it.plc_mrow_i3y = payload_mesh_item_org[3].y / 1024;
		it.plc_mrow_i3z = payload_mesh_item_org[3].z / 1024;
		it.plc_mrow_i3a = payload_mesh_item_avail[3];
		it.plc_mrow_i4r = payload_mesh_item_rank[4];
		it.plc_mrow_i4x = payload_mesh_item_org[4].x / 1024;
		it.plc_mrow_i4y = payload_mesh_item_org[4].y / 1024;
		it.plc_mrow_i4z = payload_mesh_item_org[4].z / 1024;
		it.plc_mrow_i4a = payload_mesh_item_avail[4];
		it.plc_mrow_i5r = payload_mesh_item_rank[5];
		it.plc_mrow_i5x = payload_mesh_item_org[5].x / 1024;
		it.plc_mrow_i5y = payload_mesh_item_org[5].y / 1024;
		it.plc_mrow_i5z = payload_mesh_item_org[5].z / 1024;
		it.plc_mrow_i5a = payload_mesh_item_avail[5];
		it.plc_mrow_i6r = payload_mesh_item_rank[6];
		it.plc_mrow_i6x = payload_mesh_item_org[6].x / 1024;
		it.plc_mrow_i6y = payload_mesh_item_org[6].y / 1024;
		it.plc_mrow_i6z = payload_mesh_item_org[6].z / 1024;
		it.plc_mrow_i6a = payload_mesh_item_avail[6];
		it.plc_mrow_i7r = payload_mesh_item_rank[7];
		it.plc_mrow_i7x = payload_mesh_item_org[7].x / 1024;
		it.plc_mrow_i7y = payload_mesh_item_org[7].y / 1024;
		it.plc_mrow_i7z = payload_mesh_item_org[7].z / 1024;
		it.plc_mrow_i7a = payload_mesh_item_avail[7];
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
	mesh_gather(payload_mesh_h, 26, plc_mrow_c2p, 1, maxclients);
	mesh_gather(payload_mesh_h, 27, plc_mrow_c2t, 1, maxclients);
	mesh_gather(payload_mesh_h, 28, plc_mrow_c2r, 1, maxclients);
	mesh_gather(payload_mesh_h, 29, plc_mrow_c3p, 1, maxclients);
	mesh_gather(payload_mesh_h, 30, plc_mrow_c3t, 1, maxclients);
	mesh_gather(payload_mesh_h, 31, plc_mrow_c3r, 1, maxclients);
	mesh_gather(payload_mesh_h, 32, plc_mrow_i0r, 1, maxclients);
	mesh_gather(payload_mesh_h, 33, plc_mrow_i0x, 1, maxclients);
	mesh_gather(payload_mesh_h, 34, plc_mrow_i0y, 1, maxclients);
	mesh_gather(payload_mesh_h, 35, plc_mrow_i0z, 1, maxclients);
	mesh_gather(payload_mesh_h, 36, plc_mrow_i0a, 1, maxclients);
	mesh_gather(payload_mesh_h, 37, plc_mrow_i1r, 1, maxclients);
	mesh_gather(payload_mesh_h, 38, plc_mrow_i1x, 1, maxclients);
	mesh_gather(payload_mesh_h, 39, plc_mrow_i1y, 1, maxclients);
	mesh_gather(payload_mesh_h, 40, plc_mrow_i1z, 1, maxclients);
	mesh_gather(payload_mesh_h, 41, plc_mrow_i1a, 1, maxclients);
	mesh_gather(payload_mesh_h, 42, plc_mrow_i2r, 1, maxclients);
	mesh_gather(payload_mesh_h, 43, plc_mrow_i2x, 1, maxclients);
	mesh_gather(payload_mesh_h, 44, plc_mrow_i2y, 1, maxclients);
	mesh_gather(payload_mesh_h, 45, plc_mrow_i2z, 1, maxclients);
	mesh_gather(payload_mesh_h, 46, plc_mrow_i2a, 1, maxclients);
	mesh_gather(payload_mesh_h, 47, plc_mrow_i3r, 1, maxclients);
	mesh_gather(payload_mesh_h, 48, plc_mrow_i3x, 1, maxclients);
	mesh_gather(payload_mesh_h, 49, plc_mrow_i3y, 1, maxclients);
	mesh_gather(payload_mesh_h, 50, plc_mrow_i3z, 1, maxclients);
	mesh_gather(payload_mesh_h, 51, plc_mrow_i3a, 1, maxclients);
	mesh_gather(payload_mesh_h, 52, plc_mrow_i4r, 1, maxclients);
	mesh_gather(payload_mesh_h, 53, plc_mrow_i4x, 1, maxclients);
	mesh_gather(payload_mesh_h, 54, plc_mrow_i4y, 1, maxclients);
	mesh_gather(payload_mesh_h, 55, plc_mrow_i4z, 1, maxclients);
	mesh_gather(payload_mesh_h, 56, plc_mrow_i4a, 1, maxclients);
	mesh_gather(payload_mesh_h, 57, plc_mrow_i5r, 1, maxclients);
	mesh_gather(payload_mesh_h, 58, plc_mrow_i5x, 1, maxclients);
	mesh_gather(payload_mesh_h, 59, plc_mrow_i5y, 1, maxclients);
	mesh_gather(payload_mesh_h, 60, plc_mrow_i5z, 1, maxclients);
	mesh_gather(payload_mesh_h, 61, plc_mrow_i5a, 1, maxclients);
	mesh_gather(payload_mesh_h, 62, plc_mrow_i6r, 1, maxclients);
	mesh_gather(payload_mesh_h, 63, plc_mrow_i6x, 1, maxclients);
	mesh_gather(payload_mesh_h, 64, plc_mrow_i6y, 1, maxclients);
	mesh_gather(payload_mesh_h, 65, plc_mrow_i6z, 1, maxclients);
	mesh_gather(payload_mesh_h, 66, plc_mrow_i6a, 1, maxclients);
	mesh_gather(payload_mesh_h, 67, plc_mrow_i7r, 1, maxclients);
	mesh_gather(payload_mesh_h, 68, plc_mrow_i7x, 1, maxclients);
	mesh_gather(payload_mesh_h, 69, plc_mrow_i7y, 1, maxclients);
	mesh_gather(payload_mesh_h, 70, plc_mrow_i7z, 1, maxclients);
	mesh_gather(payload_mesh_h, 71, plc_mrow_i7a, 1, maxclients);

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
	for(int k = 0; k < (NUM_TEAMS + 1) * PLC_MESH_TOTAL; ++k)
		payload_mesh_vote[k] = 0;

	FOREACH_CLIENT(IS_PLAYER(it),
	{
		int j = Team_TeamToIndex(it.team);
		int o = floor(it.plc_mesh_pick);
		if(!(o >= PLC_MESH_ITEM_BASE && o < PLC_MESH_TOTAL
			&& payload_mesh_item_rank[o - PLC_MESH_ITEM_BASE] > 0))
			o = bound(0, o, nobj - 1);
		if(j >= 1 && j <= NUM_TEAMS)
			payload_mesh_vote[j * PLC_MESH_TOTAL + o] += 1;
	});

	for(int j = 1; j <= NUM_TEAMS; ++j)
	{
		float best = 0;
		int pick = payload_mesh_objective[j];
		for(int o = 0; o < PLC_MESH_TOTAL; ++o)
		{
			if(o >= nobj && o < PLC_MESH_ITEM_BASE)
				continue;
			float w = payload_mesh_vote[j * PLC_MESH_TOTAL + o];
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
