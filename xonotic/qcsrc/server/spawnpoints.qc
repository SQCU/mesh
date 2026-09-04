#include "spawnpoints.qh"

#include <common/constants.qh>
#include <common/gamemodes/_mod.qh>
#include <common/mapobjects/subs.qh>
#include <common/mapobjects/target/spawnpoint.qh>
#include <common/mapobjects/triggers.qh>
#include <common/net_linked.qh>
#include <common/stats.qh>
#include <common/teams.qh>
#include <common/util.qh>
#include <common/weapons/_all.qh>
#include <lib/warpzone/common.qh>
#include <lib/warpzone/util_server.qh>
#include <server/command/vote.qh>
#include <server/intermission.qh>
#include <server/mutators/_mod.qh>
#include <server/race.qh>
#include <server/utils.qh>
#include <server/world.qh>

bool SpawnPoint_Send(entity this, entity to, int sf)
{
	WriteHeader(MSG_ENTITY, ENT_CLIENT_SPAWNPOINT);

	WriteShort(MSG_ENTITY, this.team);
	WriteVector(MSG_ENTITY, this.origin);

	return true;
}

bool SpawnEvent_Send(entity this, entity to, int sf)
{
	float send;

	WriteHeader(MSG_ENTITY, ENT_CLIENT_SPAWNEVENT);

	if(autocvar_g_spawn_alloweffects)
	{
		WriteShort(MSG_ENTITY, etof(this.owner));
		WriteVector(MSG_ENTITY, this.owner.origin);
		send = true;
	}
	else if((to == this.owner) || (IS_SPEC(to) && (to.enemy == this.owner)) )
	{
		WriteShort(MSG_ENTITY, 0);
		send = true;
	}
	else { send = false; }

	return send;
}

bool Spawn_HasTeamSpawn(int team_num)
{
	return team_num >= 0 && team_num <= NUM_TEAMS && have_team_spawns_for_team[team_num];
}

.vector spawnpoint_prevorigin;
void spawnpoint_think(entity this)
{
	this.nextthink = time + 0.1;
	if(this.origin != this.spawnpoint_prevorigin)
	{
		this.spawnpoint_prevorigin = this.origin;
		this.SendFlags |= 1;
	}
}

void spawnpoint_use(entity this, entity actor, entity trigger)
{
	if(teamplay)
	if(have_team_spawns > 0)
	{
		this.team = actor.team;
		some_spawn_has_been_used = true;
		this.SendFlags |= 1;
	}

}

void spawnpoint_reset(entity this)
{
	this.SendFlags |= 1;
}

void link_spawnpoint(entity this)
{
	bool anypoint = (autocvar_g_spawn_useallspawns || (teamplay && have_team_spawns <= 0));

	if(anypoint || !((teamplay && !Team_IsValidTeam(this.team)) || (!teamplay && Team_IsValidTeam(this.team))))
		Net_LinkEntity(this, false, 0, SpawnPoint_Send);
}

void relocate_spawnpoint(entity this)
{

    setorigin(this, this.origin + '0 0 1');

    tracebox(this.origin, PL_MIN_CONST, PL_MAX_CONST, this.origin, true, this);
    if (trace_startsolid)
    {
        vector o;
        o = this.origin;
        this.mins = PL_MIN_CONST;
        this.maxs = PL_MAX_CONST;
        if (!move_out_of_solid(this))
            objerror(this, "could not get out of solid at all!");
        LOG_INFOF(
            "^1NOTE: this map needs FIXING. Spawnpoint at %s needs to be moved out of solid, e.g. by %s",
            vtos(o - '0 0 1'),
            vtos(this.origin - o)
        );
        if (autocvar_g_spawnpoints_auto_move_out_of_solid)
        {
            if (!spawnpoint_nag)
                LOG_INFO("\{1}^1NOTE: this map needs FIXING (it contains spawnpoints in solid, see server log)");
            spawnpoint_nag = 1;
        }
        else
        {
            setorigin(this, o);
            this.mins = this.maxs = '0 0 0';
            objerror(this, "player spawn point in solid, mapper sucks!\n");
            return;
        }
    }

    this.use = spawnpoint_use;
    setthink(this, spawnpoint_think);
    this.nextthink = time + 0.5 + random() * 2;
    this.reset2 = spawnpoint_reset;
    this.team_saved = this.team;
    IL_PUSH(g_saved_team, this);
    if (!this.cnt)
        this.cnt = 1;

    if (have_team_spawns != 0)
        if (this.team)
            have_team_spawns = 1;
    have_team_spawns_for_team[Team_IsValidTeam(this.team) ? this.team : 0] = true;

    if (autocvar_r_showbboxes)
    {

        vector forward, right, up;
        MAKE_VECTORS(this.angles, forward, right, up);
        entity e = new(info_player_foo);
        setorigin(e, this.origin + forward * 24);
        setsize(e, '-8 -8 -8', '8 8 8');
        e.solid = SOLID_TRIGGER;
    }

	InitializeEntity(this, link_spawnpoint, INITPRIO_FINDTARGET);
}

spawnfunc(info_player_survivor)
{
	spawnfunc_info_player_deathmatch(this);
}

spawnfunc(info_player_start)
{
	spawnfunc_info_player_deathmatch(this);
}

spawnfunc(info_player_deathmatch)
{
	IL_PUSH(g_spawnpoints, this);
	relocate_spawnpoint(this);
}

spawnfunc(info_player_team1)
{
	this.team = NUM_TEAM_1;
	spawnfunc_info_player_deathmatch(this);
}

spawnfunc(info_player_team2)
{
	this.team = NUM_TEAM_2;
	spawnfunc_info_player_deathmatch(this);
}

spawnfunc(info_player_team3)
{
	this.team = NUM_TEAM_3;
	spawnfunc_info_player_deathmatch(this);
}

spawnfunc(info_player_team4)
{
	this.team = NUM_TEAM_4;
	spawnfunc_info_player_deathmatch(this);
}

spawnfunc(info_player_team5)
{
	this.team = NUM_TEAM_5;
	spawnfunc_info_player_deathmatch(this);
}

spawnfunc(info_player_team6)
{
	this.team = NUM_TEAM_6;
	spawnfunc_info_player_deathmatch(this);
}

spawnfunc(info_player_team7)
{
	this.team = NUM_TEAM_7;
	spawnfunc_info_player_deathmatch(this);
}

spawnfunc(info_player_team8)
{
	this.team = NUM_TEAM_8;
	spawnfunc_info_player_deathmatch(this);
}

vector Spawn_Score(entity this, entity spot, float mindist, float teamcheck, bool targetcheck)
{

	if(teamcheck >= 0)
		if(spot.team != teamcheck)
			return '-1 0 0';

	if(race_spawns)
		if(!spot.target || spot.target == "")
			return '-1 0 0';

	if(IS_REAL_CLIENT(this))
	{
		if(spot.restriction == 1)
			return '-1 0 0';
	}
	else
	{
		if(spot.restriction == 2)
			return '-1 0 0';
	}

	float prio = 0;
	float shortest = vlen(world.maxs - world.mins);
	FOREACH_CLIENT(IS_PLAYER(it) && !IS_DEAD(it) && it != this, {
		float thisdist = vlen(it.origin - spot.origin);
		if (thisdist < shortest)
			shortest = thisdist;
	});
	if(shortest > mindist)
		prio += SPAWN_PRIO_GOOD_DISTANCE;

	vector spawn_score = prio * '1 0 0' + shortest * '0 1 0';

	if(spot.target && spot.target != "" && targetcheck)
	{
		int found = 0;
		for(entity targ = findchain(targetname, spot.target); targ; targ = targ.chain)
		{
			++found;
			if(targ.spawn_evalfunc)
			{
				spawn_score = targ.spawn_evalfunc(targ, this, spot, spawn_score);
				if(spawn_score.x < 0)
					return spawn_score;
			}
		}

		if(!found && !g_cts)
		{
			LOG_TRACE("WARNING: spawnpoint at ", vtos(spot.origin), " could not find its target ", spot.target);
			return '-1 0 0';
		}
	}

	MUTATOR_CALLHOOK(Spawn_Score, this, spot, spawn_score);
	spawn_score = M_ARGV(2, vector);
	return spawn_score;
}

void Spawn_ScoreAll(entity this, entity firstspot, float mindist, float teamcheck, bool targetcheck)
{
	entity spot;
	for(spot = firstspot; spot; spot = spot.chain)
		spot.spawnpoint_score = Spawn_Score(this, spot, mindist, teamcheck, targetcheck);
}

entity Spawn_FilterOutBadSpots(entity this, entity firstspot, float mindist, float teamcheck, bool targetcheck)
{
	entity spot, spotlist, spotlistend;

	spotlist = NULL;
	spotlistend = NULL;

	Spawn_ScoreAll(this, firstspot, mindist, teamcheck, targetcheck);

	for(spot = firstspot; spot; spot = spot.chain)
	{
		if(spot.spawnpoint_score.x >= 0)
		{
			if(spotlistend)
				spotlistend.chain = spot;
			spotlistend = spot;
			if(!spotlist)
				spotlist = spot;
		}
	}
	if(spotlistend)
		spotlistend.chain = NULL;

	return spotlist;
}

entity Spawn_WeightedPoint(entity firstspot, float lower, float upper, float exponent)
{

	entity spot;

	RandomSelection_Init();
	for(spot = firstspot; spot; spot = spot.chain)
		RandomSelection_AddEnt(spot, (bound(lower, spot.spawnpoint_score.y, upper) ** exponent) * spot.cnt, (spot.spawnpoint_score.y >= lower) * 0.5 + spot.spawnpoint_score.x);

	return RandomSelection_chosen_ent;
}

int Spawn_PointCount(entity firstspot)
{
	int count = 0;
	for(entity spot = firstspot; spot; spot = spot.chain)
		++count;
	return count;
}

int Spawn_SwizzlePlayerCount()
{
	int count = 0;
	FOREACH_CLIENT(IS_PLAYER(it) && Team_IsValidTeam(it.team), { ++count; });
	return max(1, count);
}

int Spawn_SwizzleTeamOrdinal(entity player)
{
	int ordinal = 0;
	FOREACH_CLIENT(IS_PLAYER(it) && it.team == player.team,
	{
		if(etof(it) < etof(player))
			++ordinal;
	});
	return ordinal;
}

int Spawn_SwizzleTicket(entity player)
{
	int playerteam = Team_TeamToIndex(player.team);
	int playerordinal = Spawn_SwizzleTeamOrdinal(player);
	int ticket = 0;
	FOREACH_CLIENT(IS_PLAYER(it) && Team_IsValidTeam(it.team),
	{
		int ordinal = Spawn_SwizzleTeamOrdinal(it);
		int candidate_team = Team_TeamToIndex(it.team);
		if(ordinal < playerordinal || (ordinal == playerordinal && candidate_team < playerteam))
			++ticket;
	});
	return ticket;
}

int Spawn_SwizzleSlots(int spots)
{
	int configured = floor(autocvar_g_spawn_swizzle_slots);
	return configured > 0 ? min(max(1, spots), configured) : max(1, spots);
}

float Spawn_SwizzleStep()
{
	return max(sys_frametime, autocvar_g_spawn_swizzle_step);
}

bool Spawn_SwizzleReserved(entity player, float scheduled, int lane)
{
	bool reserved = false;
	FOREACH_CLIENT(it != player && it.spawn_swizzle_active && it.spawn_swizzle_actual_time < 0,
	{
		if(fabs(it.spawn_swizzle_scheduled_time - scheduled) < 0.001
			&& floor(it.spawn_swizzle_lane) == lane)
			reserved = true;
	});
	return reserved;
}

float Spawn_SwizzleTime(entity player, float earliest)
{
	player.spawn_swizzle_active = 0;
	if(!autocvar_g_spawn_swizzle || !Team_IsValidTeam(player.team))
		return earliest;
	int spots = 0;
	IL_EACH(g_spawnpoints, true, { ++spots; });
	spots = max(1, spots);
	int players = Spawn_SwizzlePlayerCount();
	int slots = Spawn_SwizzleSlots(spots);
	float step = Spawn_SwizzleStep();
	int cohorts = max(1, ceil(players / slots));
	int ticket = Spawn_SwizzleTicket(player);
	int cohort = floor(ticket / slots);
	float cycle = cohorts * step;
	int generation = max(0, ceil((earliest - cohort * step) / cycle));
	float scheduled = generation * cycle + cohort * step;
	int lane = ticket % slots;
	while(Spawn_SwizzleReserved(player, scheduled, lane))
	{
		++cohort;
		if(cohort >= cohorts)
		{
			cohort = 0;
			++generation;
		}
		scheduled = generation * cycle + cohort * step;
	}
	player.spawn_swizzle_active = 1;
	player.spawn_swizzle_epoch += 1;
	player.spawn_swizzle_player_count = players;
	player.spawn_swizzle_spot_count = spots;
	player.spawn_swizzle_slot_count = slots;
	player.spawn_swizzle_ticket = ticket;
	player.spawn_swizzle_cohort = cohort;
	player.spawn_swizzle_cohort_count = cohorts;
	player.spawn_swizzle_generation = generation;
	player.spawn_swizzle_scheduled_time = scheduled;
	player.spawn_swizzle_actual_time = -1;
	player.spawn_swizzle_lane = lane;
	player.spawn_swizzle_spot = -1;
	LOG_INFOF("[SPAWNSWIZZLE] team=%d players=%d spots=%d slots=%d ticket=%d cohort=%d cohorts=%d generation=%d scheduled=%.3f delay=%.3f\n",
		Team_TeamToIndex(player.team), players, spots, slots, ticket, cohort,
		cohorts, generation, scheduled, scheduled - time);
	return scheduled;
}

entity Spawn_SwizzledPoint(entity firstspot, entity player)
{
	int count = Spawn_PointCount(firstspot);
	if(!count)
		return NULL;
	if(!player.spawn_swizzle_active)
	{
		int players = Spawn_SwizzlePlayerCount();
		int slots = Spawn_SwizzleSlots(count);
		int ticket = Spawn_SwizzleTicket(player);
		player.spawn_swizzle_active = 1;
		player.spawn_swizzle_epoch += 1;
		player.spawn_swizzle_player_count = players;
		player.spawn_swizzle_spot_count = count;
		player.spawn_swizzle_slot_count = slots;
		player.spawn_swizzle_ticket = ticket;
		player.spawn_swizzle_cohort = floor(ticket / slots);
		player.spawn_swizzle_cohort_count = max(1, ceil(players / slots));
		player.spawn_swizzle_generation = floor(time / (player.spawn_swizzle_cohort_count * Spawn_SwizzleStep()));
		player.spawn_swizzle_scheduled_time = time;
		player.spawn_swizzle_lane = ticket % slots;
	}
	int slots = min(count, max(1, floor(player.spawn_swizzle_slot_count)));
	int generation = floor(player.spawn_swizzle_generation);
	int ticket = floor(player.spawn_swizzle_ticket);
	int lane = floor(player.spawn_swizzle_lane) % slots;
	int slot = (generation + lane) % count;
	int selected = slot;
	entity spot = firstspot;
	while(slot-- > 0 && spot.chain)
		spot = spot.chain;
	player.spawn_swizzle_actual_time = time;
	player.spawn_swizzle_lane = lane;
	player.spawn_swizzle_spot = etof(spot);
	LOG_INFOF("[SPAWNSWIZZLE] team=%d ticket=%d lane=%d generation=%d slot=%d spots=%d slots=%d\n",
		Team_TeamToIndex(player.team), ticket, lane, generation, selected, count, slots);
	return spot;
}

bool testspawn_checked;
entity testspawn_point;
entity SelectSpawnPoint(entity this, bool anypoint)
{
	float teamcheck;
	entity spot = NULL;

	if(!testspawn_checked)
	{
		testspawn_point = find(NULL, classname, "testplayerstart");
		testspawn_checked = true;
	}

	if(testspawn_point)
		return testspawn_point;

	if(this.spawnpoint_targ)
		return this.spawnpoint_targ;

	if(anypoint || autocvar_g_spawn_useallspawns || autocvar_g_spawn_swizzle)
		teamcheck = -1;
	else if(have_team_spawns > 0)
	{
		if(!Spawn_HasTeamSpawn(this.team))
		{

			if(Spawn_HasTeamSpawn(0))

				teamcheck = 0;
			else

				teamcheck = -1;
		}
		else
			teamcheck = this.team;
	}
	else if(have_team_spawns == 0 && Spawn_HasTeamSpawn(0))
		teamcheck = 0;
	else
		teamcheck = -1;

	entity firstspot = IL_FIRST(g_spawnpoints);
	entity prev = NULL;
	IL_EACH(g_spawnpoints, true,
	{
		if(prev)
			prev.chain = it;
		it.chain = NULL;
		prev = it;
	});

	if(anypoint)
	{
		spot = Spawn_WeightedPoint(firstspot, 1, 1, 1);
	}
	else if(autocvar_g_spawn_swizzle && Team_IsValidTeam(this.team))
	{
		firstspot = Spawn_FilterOutBadSpots(this, firstspot, 100, -1, true);
		spot = Spawn_SwizzledPoint(firstspot, this);
	}
	else
	{
		firstspot = Spawn_FilterOutBadSpots(this, firstspot, 100, teamcheck, true);

		if(!firstspot)
		{
			firstspot = IL_FIRST(g_spawnpoints);
			prev = NULL;
			IL_EACH(g_spawnpoints, true,
			{
				if(prev)
					prev.chain = it;
				it.chain = NULL;
				prev = it;
			});
			firstspot = Spawn_FilterOutBadSpots(this, firstspot, 100, teamcheck, false);
		}

		if (random() > autocvar_g_spawn_furthest)
			spot = Spawn_WeightedPoint(firstspot, 1, 1, 1);
		else
			spot = Spawn_WeightedPoint(firstspot, 1, 5000, 5);
	}

	if (!spot)
	{
		if(autocvar_spawn_debug)
			GotoNextMap(0);
		else
		{
			if(some_spawn_has_been_used)
				return NULL;
			else
				error("Cannot find a spawn point - please fix the map!");
		}
	}

	return spot;
}
