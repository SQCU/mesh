#include "scores_rules.qh"

#include <common/gamemodes/rules.qh>
#include <common/stats.qh>
#include <common/weapons/_all.qh>
#include <server/client.qh>
#include <server/scores.qh>
#include <server/teamplay.qh>

int ScoreRules_team_count;

bool IsTeamAvailable(int team_num)
{
	return TeamScore_IsAvailable(team_num);
}

int LegacyTeamBitCount(int team_bits)
{
	int n = 0;
	while(team_bits)
	{
		n += team_bits & 1;
		team_bits >>= 1;
	}
	return n;
}

void ScoreRules_basics(int team_count, float sprio, float stprio, float score_enabled)
{
	FOREACH(Scores, true, {
		ScoreInfo_SetLabel_PlayerScore(it, "", 0);
	});
	for(int i = 0; i < MAX_TEAMSCORE; ++i)
		ScoreInfo_SetLabel_TeamScore(i, "", 0);

	ScoreRules_team_count = team_count;
	AVAILABLE_TEAMS = team_count;

	if(score_enabled)
		ScoreInfo_SetLabel_TeamScore(ST_SCORE, "score", stprio);

	if (!INDEPENDENT_PLAYERS)
		ScoreInfo_SetLabel_PlayerScore(SP_KILLS, "kills", 0);

	ScoreInfo_SetLabel_PlayerScore(SP_DEATHS, "deaths", SFL_LOWER_IS_BETTER);

	if (!INDEPENDENT_PLAYERS)
	{
		ScoreInfo_SetLabel_PlayerScore(SP_SUICIDES, "suicides", SFL_LOWER_IS_BETTER);
		if (teamplay)
			ScoreInfo_SetLabel_PlayerScore(SP_TEAMKILLS, "teamkills", SFL_LOWER_IS_BETTER);
	}

	if(score_enabled)
		ScoreInfo_SetLabel_PlayerScore(SP_SCORE, "score", sprio);

	if (!INDEPENDENT_PLAYERS)
	{
		ScoreInfo_SetLabel_PlayerScore(SP_DMG, "dmg", 0);
		ScoreInfo_SetLabel_PlayerScore(SP_DMGTAKEN, "dmgtaken", SFL_LOWER_IS_BETTER);

		ScoreInfo_SetLabel_PlayerScore(SP_ELO, "elo", SFL_NOT_SORTABLE);
	}

	ScoreInfo_SetLabel_PlayerScore(SP_ROUNDS_PL, "rounds_pl", 0);

	if(STAT(SHOWFPS))
		ScoreInfo_SetLabel_PlayerScore(SP_FPS, "fps", SFL_NOT_SORTABLE);
}

void ScoreRules_basics_end()
{
	ScoreInfo_InitCount(ScoreRules_team_count);
}

void ScoreInfo_InitTeamRows(entity balance)
{
	ScoreInfo_Init();
	for(int i = 1; i <= NUM_TEAMS; ++i)
		if(TeamBalance_IsTeamAllowed(balance, i))
			TeamScore_Spawn(Team_IndexToTeam(i), Static_Team_ColorName(Team_IndexToTeam(i)));
}

void ScoreRules_generic()
{
	bool has_teams = teamplay;
	GameRules_scoring(0, SFL_SORT_PRIO_PRIMARY, SFL_SORT_PRIO_PRIMARY, {});
	if(has_teams)
	{
		entity balance = TeamBalance_CheckAllowedTeams(NULL);
		ScoreInfo_InitTeamRows(balance);
		teamplay = TeamBalance_GetAllowedTeamCount(balance);
		TeamBalance_Destroy(balance);
	}
}
