#include "sv_payload_mesh.qh"

#include "sv_payload.qh"

#include <common/resources/_mod.qh>
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

entity payload_mesh_node(float idx)
{
	entity a = payload_first;
	for(float k = 0; a && k < idx && a.plc_next; ++k)
		a = a.plc_next;
	return a;
}

void payload_mesh_tick(entity cart)
{
	if(payload_mesh_h < 0)
	{
		if(time >= payload_mesh_retry)
			payload_mesh_attach();
		return;
	}

	float prog = (payload_length > 0) ? cart.plc_s / payload_length : 0;
	vector cpos = cart.origin + cart.view_ofs;

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
		it.plc_mrow_dist = p ? vlen(it.origin - cpos) : 0;
		it.plc_mrow_prog = prog;
		it.plc_mrow_friends = fr;
		it.plc_mrow_enemies = en;
		it.plc_mrow_obj = (j >= 1 && j <= NUM_TEAMS) ? payload_mesh_objective[j] : 0;
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

	payload_mesh_seq += 1;
	mesh_publish(payload_mesh_h, payload_mesh_seq, maxclients);

	float done = mesh_poll(payload_mesh_h);
	if(done <= payload_mesh_plan)
		return;
	payload_mesh_plan = done;

	mesh_scatter(payload_mesh_h, 1, plc_mesh_pick, 1, maxclients);

	for(int k = 0; k < (NUM_TEAMS + 1) * PLC_MESH_OBJECTIVES; ++k)
		payload_mesh_vote[k] = 0;

	FOREACH_CLIENT(IS_PLAYER(it),
	{
		int j = Team_TeamToIndex(it.team);
		int o = bound(0, floor(it.plc_mesh_pick), PLC_MESH_OBJECTIVES - 1);
		if(j >= 1 && j <= NUM_TEAMS)
			payload_mesh_vote[j * PLC_MESH_OBJECTIVES + o] += 1;
	});

	for(int j = 1; j <= NUM_TEAMS; ++j)
	{
		float best = 0;
		int pick = payload_mesh_objective[j];
		for(int o = 0; o < PLC_MESH_OBJECTIVES; ++o)
		{
			float w = payload_mesh_vote[j * PLC_MESH_OBJECTIVES + o];
			if(w > best)
			{
				best = w;
				pick = o;
			}
		}
		payload_mesh_objective[j] = pick;
	}
}
