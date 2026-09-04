#include "teamplay.qh"

#include <common/deathtypes/all.qh>
#include <common/gamemodes/_mod.qh>
#include <common/teams.qh>
#include <server/bot/api.qh>
#include <server/bot/default/cvars.qh>
#include <server/campaign.qh>
#include <server/client.qh>
#include <server/command/vote.qh>
#include <server/damage.qh>
#include <server/gamelog.qh>
#include <server/mutators/_mod.qh>
#include <server/race.qh>
#include <server/scores.qh>
#include <server/scores_rules.qh>

enum
{
	TEAM_BALANCE_UNINITIALIZED,

	TEAM_BALANCE_TEAMS_CHECKED,

	TEAM_BALANCE_TEAM_COUNTS_FILLED
};

const int TEAM_NOT_ALLOWED = -1;

.int m_team_balance_state;
.float m_team_balance_score[NUM_TEAMS];
.int m_team_balance_players[NUM_TEAMS];
.int m_team_balance_bots[NUM_TEAMS];

.float m_team_score;
.int m_num_players;
.int m_num_bots;
.int m_num_players_alive;
.int m_num_owned_items;

string autocvar_g_forced_team_red;
string autocvar_g_forced_team_blue;
string autocvar_g_forced_team_yellow;
string autocvar_g_forced_team_pink;
string autocvar_g_forced_team_green;

entity g_team_entities[NUM_TEAMS];

void Team_InitTeams()
{
	if (g_team_entities[0])
		return;
	for (int i = 0; i < NUM_TEAMS; ++i)
	{
		g_team_entities[i] = new_pure(team_entity);
	}
}

entity Team_GetTeamFromIndex(int index)
{
	if (!Team_IsValidIndex(index))
	{
		LOG_FATALF("Team_GetTeamFromIndex: Index is invalid: %f", index);
	}
	return g_team_entities[index - 1];
}

entity Team_GetTeam(int team_num)
{
	if (!Team_IsValidTeam(team_num))
	{
		LOG_FATALF("Team_GetTeam: Value is invalid: %f", team_num);
	}
	return g_team_entities[Team_TeamToIndex(team_num) - 1];
}

float Team_GetTeamScore(entity team_ent)
{
	return team_ent.m_team_score;
}

void Team_SetTeamScore(entity team_ent, float score)
{
	team_ent.m_team_score = score;
}

int Team_GetNumberOfAlivePlayers(entity team_ent)
{
	return team_ent.m_num_players_alive;
}

void Team_SetNumberOfAlivePlayers(entity team_ent, int number)
{
	team_ent.m_num_players_alive = number;
}

int Team_GetWinnerAliveTeam()
{
	int winner = 0;
	for (int i = 0; i < NUM_TEAMS; ++i)
	{
		if (g_team_entities[i].m_num_players_alive > 0)
		{
			if (winner)
				return 0;
			winner = Team_IndexToTeam(i + 1);
		}
	}
	return (winner ? winner : -1);
}

int Team_GetNumberOfAliveTeams()
{
	int result = 0;
	for (int i = 0; i < NUM_TEAMS; ++i)
	{
		if (g_team_entities[i].m_num_players_alive > 0)
		{
			++result;
		}
	}
	return result;
}

int Team_GetWinnerTeam_WithOwnedItems(int min_control_points)
{
	int winner = 0;
	for (int i = 0; i < NUM_TEAMS; ++i)
	{
		if (g_team_entities[i].m_num_owned_items >= min_control_points)
		{
			if (winner)
				return 0;
			winner = Team_IndexToTeam(i + 1);
		}
	}
	return (winner ? winner : -1);
}

int Team_GetNumberOfOwnedItems(entity team_ent)
{
	return team_ent.m_num_owned_items;
}

void Team_SetNumberOfOwnedItems(entity team_ent, int number)
{
	team_ent.m_num_owned_items = number;
}

int Team_GetNumberOfTeamsWithOwnedItems()
{
	int result = 0;
	for (int i = 0; i < NUM_TEAMS; ++i)
	{
		if (g_team_entities[i].m_num_owned_items > 0)
		{
			++result;
		}
	}
	return result;
}

void setcolor(entity this, int clr)
{
#if 1
	this.clientcolors = clr;
#else
	builtin_setcolor(this, clr);
#endif
}

bool Entity_HasValidTeam(entity this)
{
	return Team_IsValidTeam(this.team);
}

int Entity_GetTeamIndex(entity this)
{
	return Team_TeamToIndex(this.team);
}

entity Entity_GetTeam(entity this)
{
	int index = Entity_GetTeamIndex(this);
	if (!Team_IsValidIndex(index))
	{
		return NULL;
	}
	return Team_GetTeamFromIndex(index);
}

void SetPlayerColors(entity player, float _color)
{
	float pants = _color & 0x0F;
	float shirt = (floor(_color / 16) & 0x0F) * 16;
	if (teamplay)
	{
		setcolor(player, shirt + pants);
	}
	else
	{
		setcolor(player, shirt + pants);
	}
}

bool Player_SetTeamIndex(entity player, int index)
{
	int new_team = Team_IndexToTeam(index);
	if (player.team == new_team)
	{
		if (new_team != -1)
		{

			SetPlayerColors(player, new_team - 1);
		}
		return true;
	}
	int old_index = Team_TeamToIndex(player.team);
	if (MUTATOR_CALLHOOK(Player_ChangeTeam, player, old_index, index) == true)
	{

		return false;
	}
	if (new_team == -1)
	{
		player.team = -1;
	}
	else
	{
		SetPlayerColors(player, new_team - 1);
		player.team = new_team;
	}
	MUTATOR_CALLHOOK(Player_ChangedTeam, player, old_index, index);
	return true;
}

bool SetPlayerTeam(entity player, int team_index, int type)
{
	int old_team_index = Entity_GetTeamIndex(player);

	if (!Player_SetTeamIndex(player, team_index))
		return false;

	LogTeamChange(player.playerid, player.team, type);

	if (team_index != old_team_index)
	{
		KillPlayerForTeamChange(player);
		PlayerScore_Clear(player);

		if (!IS_BOT_CLIENT(player))
			TeamBalance_AutoBalanceBots();

		if (team_index != -1)
		{
			if(player.team <= NUM_TEAM_5)
				Send_Notification(NOTIF_ALL, NULL, MSG_INFO, APP_TEAM_NUM(player.team, INFO_JOIN_PLAY_TEAM), player.netname);
			else
				bprint("^7", player.netname, "^3 is now playing on the ", Team_ColoredFullName(player.team), "\n");
		}
	}

	if (team_index == -1)
	{
		if (autocvar_sv_maxidle_playertospectator > 0 && CS(player).idlekick_lasttimeleft)
		{

			Kill_Notification(NOTIF_ONE_ONLY, player, MSG_CENTER, CPID_IDLING);
			CS(player).idlekick_lasttimeleft = 0;
		}
		else if (!CS(player).just_joined && player.frags != FRAGS_SPECTATOR)
		{
			Send_Notification(NOTIF_ALL, NULL, MSG_INFO, INFO_QUIT_SPECTATE, player.netname);
		}
	}

	return true;
}

void Player_SetTeamIndexChecked(entity player, int team_index)
{
	if (!teamplay)
	{
		return;
	}
	if (!Team_IsValidIndex(team_index))
	{
		return;
	}
	if ((autocvar_g_campaign) || (autocvar_g_changeteam_banned &&
		CS(player).wasplayer))
	{
		Send_Notification(NOTIF_ONE, player, MSG_INFO,
			INFO_TEAMCHANGE_NOTALLOWED);
		return;
	}
	entity balance = TeamBalance_CheckAllowedTeams(player);
	if (!TeamBalance_IsTeamAllowedInternal(balance, team_index))
	{
		for (int offset = 1; offset < NUM_TEAMS; ++offset)
		{
			int candidate = (team_index - 1 + offset) % NUM_TEAMS + 1;
			if (TeamBalance_IsTeamAllowedInternal(balance, candidate))
			{
				team_index = candidate;
				break;
			}
		}
	}

	if (autocvar_g_balance_teams && autocvar_g_balance_teams_prevent_imbalance)
	{
		TeamBalance_GetTeamCounts(balance, player);
		if (!TeamBalance_IsBestTeam(balance, team_index, player, false))
		{
			Send_Notification(NOTIF_ONE, player, MSG_INFO,
				INFO_TEAMCHANGE_LARGERTEAM);
			TeamBalance_Destroy(balance);
			return;
		}
	}
	TeamBalance_Destroy(balance);
	SetPlayerTeam(player, team_index, TEAM_CHANGE_MANUAL);
}

bool MoveToTeam(entity client, int team_index, int type)
{

	int lockteams_backup = lockteams;
	lockteams = 0;
	if (!SetPlayerTeam(client, team_index, type))
	{
		lockteams = lockteams_backup;
		return false;
	}
	lockteams = lockteams_backup;
	return true;
}

bool Player_HasRealForcedTeam(entity player)
{
	return player.team_forced > TEAM_FORCE_DEFAULT;
}

int Player_GetForcedTeamIndex(entity player)
{
	return player.team_forced;
}

void Player_SetForcedTeamIndex(entity player, int team_index)
{
	switch (team_index)
	{
		case TEAM_FORCE_SPECTATOR:
		case TEAM_FORCE_DEFAULT:
		{
			player.team_forced = team_index;
			break;
		}
		default:
		{
			if (!Team_IsValidIndex(team_index))
			{
				LOG_FATAL("Player_SetForcedTeamIndex: Invalid team index.");
			}
			else
			{
				player.team_forced = team_index;
				break;
			}
		}
	}
}

void Player_DetermineForcedTeam(entity player)
{
	if (autocvar_g_campaign)
	{
		if (IS_REAL_CLIENT(player))
		{
			if (Team_IsValidIndex(autocvar_g_campaign_forceteam))
			{
				player.team_forced = autocvar_g_campaign_forceteam;
			}
			else
			{
				player.team_forced = TEAM_FORCE_DEFAULT;
			}
		}
	}
	else if (PlayerInList(player, autocvar_g_forced_team_red))
	{
		player.team_forced = 1;
	}
	else if (PlayerInList(player, autocvar_g_forced_team_blue))
	{
		player.team_forced = 2;
	}
	else if (PlayerInList(player, autocvar_g_forced_team_yellow))
	{
		player.team_forced = 3;
	}
	else if (PlayerInList(player, autocvar_g_forced_team_pink))
	{
		player.team_forced = 4;
	}
	else if (PlayerInList(player, autocvar_g_forced_team_green))
	{
		player.team_forced = 5;
	}
	else
	{
		switch (autocvar_g_forced_team_otherwise)
		{
			case "red":
			{
				player.team_forced = 1;
				break;
			}
			case "blue":
			{
				player.team_forced = 2;
				break;
			}
			case "yellow":
			{
				player.team_forced = 3;
				break;
			}
			case "pink":
			{
				player.team_forced = 4;
				break;
			}
			case "green":
			{
				player.team_forced = 5;
				break;
			}
			case "spectate":
			case "spectator":
			{
				player.team_forced = TEAM_FORCE_SPECTATOR;
				break;
			}
			default:
			{
				player.team_forced = TEAM_FORCE_DEFAULT;
				break;
			}
		}
	}
	if (!teamplay && Player_HasRealForcedTeam(player))
	{
		player.team_forced = TEAM_FORCE_DEFAULT;
	}
}

void TeamBalance_JoinBestTeam(entity player)
{

	if (!teamplay)
	{
		return;
	}
	if (player.bot_forced_team)
	{
		return;
	}
	entity balance = TeamBalance_CheckAllowedTeams(player);
	if (Player_HasRealForcedTeam(player))
	{
		int forced_team_index = player.team_forced;
		bool is_team_allowed = TeamBalance_IsTeamAllowedInternal(balance,
			forced_team_index);
		TeamBalance_Destroy(balance);
		if (!is_team_allowed)
		{
			return;
		}
		if (!SetPlayerTeam(player, forced_team_index, TEAM_CHANGE_AUTO))
		{
			return;
		}
		return;
	}
	int best_team_index = TeamBalance_FindBestTeam(balance, player, true);
	TeamBalance_Destroy(balance);
	if (!SetPlayerTeam(player, best_team_index, TEAM_CHANGE_AUTO))
	{
		return;
	}
}

entity TeamBalance_CheckAllowedTeams(entity for_whom)
{
	entity balance = spawn();
	for (int i = 0; i < NUM_TEAMS; ++i)
	{
		balance.m_team_balance_score[i] = g_team_entities[i].m_team_score;
		balance.m_team_balance_players[i] = TEAM_NOT_ALLOWED;
		balance.m_team_balance_bots[i] = 0;
	}
	setthink(balance, TeamBalance_Destroy);
	balance.nextthink = time;

	MUTATOR_CALLHOOK(TeamBalance_CheckAllowedTeams, balance, for_whom);

	if (autocvar_bot_vs_human && AVAILABLE_TEAMS == 2 && for_whom)
	{
		if (autocvar_bot_vs_human > 0)
		{

			if (IS_BOT_CLIENT(for_whom))
			{
				if (TeamBalance_IsTeamAllowedInternal(balance, 4))
				{
					TeamBalance_BanTeamsExcept(balance, 4);
				}
				else if (TeamBalance_IsTeamAllowedInternal(balance, 3))
				{
					TeamBalance_BanTeamsExcept(balance, 3);
				}
				else
				{
					TeamBalance_BanTeamsExcept(balance, 2);
				}

			}
			else
			{
				if (TeamBalance_IsTeamAllowedInternal(balance, 1))
				{
					TeamBalance_BanTeamsExcept(balance, 1);
				}
				else if (TeamBalance_IsTeamAllowedInternal(balance, 2))
				{
					TeamBalance_BanTeamsExcept(balance, 2);
				}
				else
				{
					TeamBalance_BanTeamsExcept(balance, 3);
				}

			}
		}
		else
		{

			if (IS_BOT_CLIENT(for_whom))
			{
				if (TeamBalance_IsTeamAllowedInternal(balance, 1))
				{
					TeamBalance_BanTeamsExcept(balance, 1);
				}
				else if (TeamBalance_IsTeamAllowedInternal(balance, 2))
				{
					TeamBalance_BanTeamsExcept(balance, 2);
				}
				else
				{
					TeamBalance_BanTeamsExcept(balance, 3);
				}

			}
			else
			{
				if (TeamBalance_IsTeamAllowedInternal(balance, 4))
				{
					TeamBalance_BanTeamsExcept(balance, 4);
				}
				else if (TeamBalance_IsTeamAllowedInternal(balance, 3))
				{
					TeamBalance_BanTeamsExcept(balance, 3);
				}
				else
				{
					TeamBalance_BanTeamsExcept(balance, 2);
				}

			}
		}
	}

	if (!for_whom)
	{
		balance.m_team_balance_state = TEAM_BALANCE_TEAMS_CHECKED;
		return balance;
	}

	for (int i = 1; i <= NUM_TEAMS; ++i)
	{
		if (for_whom.team_forced == i &&
			TeamBalance_IsTeamAllowedInternal(balance, i))
		{
			TeamBalance_BanTeamsExcept(balance, i);
			break;
		}
	}
	balance.m_team_balance_state = TEAM_BALANCE_TEAMS_CHECKED;
	return balance;
}

void TeamBalance_Destroy(entity balance)
{
	if (balance == NULL)
	{
		return;
	}
	delete(balance);
}

void TeamBalance_AllowTeam(entity balance, int index)
{
	if (Team_IsValidIndex(index))
		balance.m_team_balance_players[index - 1] = 0;
}

void TeamBalance_AllowEntityClass(entity balance, string team_classname)
{
	for(entity row = find(NULL, classname, team_classname); row;
		row = find(row, classname, team_classname))
		TeamBalance_AllowTeam(balance, Team_TeamToIndex(row.team));
}

string TeamBalance_GetAllowedTeamRows(entity balance)
{
	string rows = "";
	for(int i = 1; i <= NUM_TEAMS; ++i)
		if(TeamBalance_IsTeamAllowedInternal(balance, i))
			rows = strcat(rows, (rows == "" ? "" : ","), ftos(i));
	return rows;
}

int TeamBalance_GetAllowedTeamCount(entity balance)
{
	if (balance == NULL)
	{
		LOG_FATAL("TeamBalance_GetAllowedTeamCount: Team balance entity is NULL.");
	}
	if (balance.m_team_balance_state == TEAM_BALANCE_UNINITIALIZED)
	{
		LOG_FATAL("TeamBalance_GetAllowedTeamCount: Team balance entity is not initialized.");
	}
	int result = 0;
	for (int i = 1; i <= NUM_TEAMS; ++i)
		result += TeamBalance_IsTeamAllowedInternal(balance, i);
	return result;
}

bool TeamBalance_IsTeamAllowed(entity balance, int index)
{
	if (balance == NULL)
	{
		LOG_FATAL("TeamBalance_IsTeamAllowed: Team balance entity is NULL.");
	}
	if (balance.m_team_balance_state == TEAM_BALANCE_UNINITIALIZED)
	{
		LOG_FATAL("TeamBalance_IsTeamAllowed: "
			"Team balance entity is not initialized.");
	}
	if (!Team_IsValidIndex(index))
	{
		LOG_FATALF("TeamBalance_IsTeamAllowed: Team index is invalid: %f",
			index);
	}
	return TeamBalance_IsTeamAllowedInternal(balance, index);
}

void TeamBalance_GetTeamCounts(entity balance, entity ignore)
{
	if (balance == NULL)
	{
		LOG_FATAL("TeamBalance_GetTeamCounts: Team balance entity is NULL.");
	}
	if (balance.m_team_balance_state == TEAM_BALANCE_UNINITIALIZED)
	{
		LOG_FATAL("TeamBalance_GetTeamCounts: "
			"Team balance entity is not initialized.");
	}
	if (MUTATOR_CALLHOOK(TeamBalance_GetTeamCounts) == true)
	{

		for (int i = 1; i <= NUM_TEAMS; ++i)
		{
			if (TeamBalance_IsTeamAllowedInternal(balance, i))
			{
				MUTATOR_CALLHOOK(TeamBalance_GetTeamCount, i, ignore);
				balance.m_team_balance_players[i - 1] = M_ARGV(2, float);
				balance.m_team_balance_bots[i - 1] = M_ARGV(3, float);
			}
		}
	}
	else
	{

		FOREACH_CLIENT(true,
		{
			if (it == ignore)
			{
				continue;
			}
			int team_num;

			if (IS_CLIENT(it) || INGAME(it))
			{
				team_num = it.team;
			}
			else if (Player_HasRealForcedTeam(it))
			{

				team_num = Team_IndexToTeam(it.team_forced);
			}
			else
			{
				continue;
			}
			if (!Team_IsValidTeam(team_num))
			{
				continue;
			}
			int team_index = Team_TeamToIndex(team_num);
			if (!TeamBalance_IsTeamAllowedInternal(balance, team_index))
			{
				continue;
			}
			++balance.m_team_balance_players[team_index - 1];
			if (IS_BOT_CLIENT(it))
			{
				++balance.m_team_balance_bots[team_index - 1];
			}
		});
	}

	if (autocvar_g_campaign)
	{
		if (Team_IsValidIndex(autocvar_g_campaign_forceteam))
		{
			int team_index = autocvar_g_campaign_forceteam - 1;
			if (balance.m_team_balance_players[team_index] ==
				balance.m_team_balance_bots[team_index])
			{
				++balance.m_team_balance_players[team_index];
			}
		}
	}
	balance.m_team_balance_state = TEAM_BALANCE_TEAM_COUNTS_FILLED;
}

int TeamBalance_GetNumberOfPlayers(entity balance, int index)
{
	if (balance == NULL)
	{
		LOG_FATAL("TeamBalance_GetNumberOfPlayers: "
			"Team balance entity is NULL.");
	}
	if (balance.m_team_balance_state != TEAM_BALANCE_TEAM_COUNTS_FILLED)
	{
		LOG_FATAL("TeamBalance_GetNumberOfPlayers: "
			"TeamBalance_GetTeamCounts has not been called.");
	}
	if (!Team_IsValidIndex(index))
	{
		LOG_FATALF("TeamBalance_GetNumberOfPlayers: Team index is invalid: %f",
			index);
	}
	return balance.m_team_balance_players[index - 1];
}

int TeamBalance_FindBestTeam(entity balance, entity player, bool ignore_player)
{
	if (balance == NULL)
	{
		LOG_FATAL("TeamBalance_FindBestTeam: Team balance entity is NULL.");
	}
	if (balance.m_team_balance_state == TEAM_BALANCE_UNINITIALIZED)
	{
		LOG_FATAL("TeamBalance_FindBestTeam: "
			"Team balance entity is not initialized.");
	}

	if (ignore_player)
	{
		TeamBalance_GetTeamCounts(balance, player);
	}
	else
	{
		TeamBalance_GetTeamCounts(balance, NULL);
	}
	int best_team_index = 0;
	int ties = 0;
	for (int i = 1; i <= NUM_TEAMS; ++i)
	{
		if (!TeamBalance_IsTeamAllowedInternal(balance, i))
			continue;
		if (!best_team_index)
		{
			best_team_index = i;
			ties = 1;
			continue;
		}
		int compare = TeamBalance_CompareTeams(balance, i, best_team_index, player, true);
		if (compare == TEAMS_COMPARE_LESS)
		{
			best_team_index = i;
			ties = 1;
		}
		else if (compare == TEAMS_COMPARE_EQUAL)
		{
			++ties;
			if (random() * ties < 1)
				best_team_index = i;
		}
	}
	if (!best_team_index)
	{
		LOG_FATALF("TeamBalance_FindBestTeam: No teams available for %s\n",
			MapInfo_Type_ToString(MapInfo_CurrentGametype()));
	}
	return best_team_index;
}

bool TeamBalance_IsBestTeam(entity balance, int team_index, entity player, bool use_score)
{
	if(!TeamBalance_IsTeamAllowed(balance, team_index))
		return false;
	for(int i = 1; i <= NUM_TEAMS; ++i)
	{
		if(!TeamBalance_IsTeamAllowedInternal(balance, i))
			continue;
		if(TeamBalance_CompareTeams(balance, i, team_index, player, use_score) == TEAMS_COMPARE_LESS)
			return false;
	}
	return true;
}

int TeamBalance_CompareTeams(entity balance, int team_index_a, int team_index_b,
	entity player, bool use_score)
{
	if (balance == NULL)
	{
		LOG_FATAL("TeamBalance_CompareTeams: Team balance entity is NULL.");
	}
	if (balance.m_team_balance_state != TEAM_BALANCE_TEAM_COUNTS_FILLED)
	{
		LOG_FATAL("TeamBalance_CompareTeams: "
			"TeamBalance_GetTeamCounts has not been called.");
	}
	if (!Team_IsValidIndex(team_index_a))
	{
		LOG_FATALF("TeamBalance_CompareTeams: team_index_a is invalid: %f",
			team_index_a);
	}
	if (!Team_IsValidIndex(team_index_b))
	{
		LOG_FATALF("TeamBalance_CompareTeams: team_index_b is invalid: %f",
			team_index_b);
	}
	if (team_index_a == team_index_b)
	{
		return TEAMS_COMPARE_EQUAL;
	}
	if (!TeamBalance_IsTeamAllowedInternal(balance, team_index_a) ||
		!TeamBalance_IsTeamAllowedInternal(balance, team_index_b))
	{
		return TEAMS_COMPARE_INVALID;
	}
	int num_players_team_a = balance.m_team_balance_players[team_index_a - 1];
	int num_players_team_b = balance.m_team_balance_players[team_index_b - 1];
	if (IS_REAL_CLIENT(player) && bots_would_leave)
	{
		num_players_team_a -= balance.m_team_balance_bots[team_index_a - 1];
		num_players_team_b -= balance.m_team_balance_bots[team_index_b - 1];
	}
	if (num_players_team_a < num_players_team_b)
		return TEAMS_COMPARE_LESS;
	if (num_players_team_a > num_players_team_b)
		return TEAMS_COMPARE_GREATER;
	if (!use_score)
		return TEAMS_COMPARE_EQUAL;
	if (balance.m_team_balance_score[team_index_a - 1] < balance.m_team_balance_score[team_index_b - 1])
		return TEAMS_COMPARE_LESS;
	if (balance.m_team_balance_score[team_index_a - 1] > balance.m_team_balance_score[team_index_b - 1])
		return TEAMS_COMPARE_GREATER;
	return TEAMS_COMPARE_EQUAL;
}

void TeamBalance_AutoBalanceBots()
{

	entity balance = TeamBalance_CheckAllowedTeams(NULL);
	TeamBalance_GetTeamCounts(balance, NULL);
	int smallest_team_index = 0;
	int smallest_team_player_count = 0;
	for (int i = 1; i <= NUM_TEAMS; ++i)
	{
		if (!TeamBalance_IsTeamAllowedInternal(balance, i))
		{
			continue;
		}
		int playercount = balance.m_team_balance_players[i - 1];
		if (smallest_team_index == 0)
		{
			smallest_team_index = i;
			smallest_team_player_count = playercount;
		}
		else if (playercount < smallest_team_player_count)
		{
			smallest_team_index = i;
			smallest_team_player_count = playercount;
		}
	}
	entity switchable_bot = NULL;
	int largest_team_player_count = -1;
	for (int i = 1; i <= NUM_TEAMS; ++i)
	{
		if (i == smallest_team_index)
			continue;
		if (!TeamBalance_IsTeamAllowedInternal(balance, i))
			continue;
		int playercount = balance.m_team_balance_players[i - 1];
		if (playercount <= largest_team_player_count)
			continue;
		entity candidate = TeamBalance_GetPlayerForTeamSwitch(i,
			smallest_team_index, true);
		if (candidate == NULL)
			continue;
		largest_team_player_count = playercount;
		switchable_bot = candidate;
	}
	TeamBalance_Destroy(balance);
	if (switchable_bot == NULL || largest_team_player_count - smallest_team_player_count < 2)
		return;
	SetPlayerTeam(switchable_bot, smallest_team_index, TEAM_CHANGE_AUTO);
}

entity TeamBalance_GetPlayerForTeamSwitch(int source_team_index,
	int destination_team_index, bool is_bot)
{
	if (MUTATOR_CALLHOOK(TeamBalance_GetPlayerForTeamSwitch, source_team_index,
		destination_team_index, is_bot))
	{
		return M_ARGV(3, entity);
	}
	entity lowest_player = NULL;
	float lowest_score = FLOAT_MAX;
	FOREACH_CLIENT(Entity_GetTeamIndex(it) == source_team_index,
	{
		if (IS_BOT_CLIENT(it) != is_bot)
		{
			continue;
		}
		float temp_score = PlayerScore_Get(it, SP_SCORE);
		if (temp_score >= lowest_score)
		{
			continue;
		}

		entity balance = TeamBalance_CheckAllowedTeams(it);
		if (TeamBalance_IsTeamAllowed(balance, source_team_index))
		{

			lowest_player = it;
			lowest_score = temp_score;
		}
		else
		{

		}
		TeamBalance_Destroy(balance);
	});
	return lowest_player;
}

void LogTeamChange(float player_id, float team_number, int type)
{
	if (!autocvar_sv_eventlog)
	{
		return;
	}
	if (player_id < 1)
	{
		return;
	}
	GameLogEcho(sprintf(":team:%d:%d:%d", player_id, team_number, type));
}

void KillPlayerForTeamChange(entity player)
{
	if (IS_DEAD(player))
	{
		return;
	}
	if (MUTATOR_CALLHOOK(Player_ChangeTeamKill, player) == true)
	{
		return;
	}
	Damage(player, player, player, 100000, DEATH_TEAMCHANGE.m_id, DMG_NOWEP,
		player.origin, '0 0 0');
}

bool TeamBalance_IsTeamAllowedInternal(entity balance, int index)
{
	return balance.m_team_balance_players[index - 1] !=
		TEAM_NOT_ALLOWED;
}

void TeamBalance_BanTeamsExcept(entity balance, int index)
{
	for (int i = 1; i <= NUM_TEAMS; ++i)
	{
		if (i != index)
		{
			balance.m_team_balance_players[i - 1] = TEAM_NOT_ALLOWED;
		}
	}
}

void SV_ChangeTeam(entity player, int new_color)
{
	if (!teamplay)
	{
		SetPlayerColors(player, new_color);
	}
	if(!IS_CLIENT(player))
	{
		return;
	}
	if (!teamplay)
	{
		return;
	}
	Player_SetTeamIndexChecked(player, Team_TeamToIndex((new_color & 0x0F) + 1));
}
