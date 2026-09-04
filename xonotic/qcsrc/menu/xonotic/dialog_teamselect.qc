#include "dialog_teamselect.qh"

#include "bigcommandbutton.qh"
#include "inputbox.qh"
#include "textlabel.qh"

entity makeTeamButton_T(string theName, vector theColor, string commandtheName, string theTooltip)
{
	entity b;
	b = makeXonoticBigCommandButton_T(theName, theColor, commandtheName, COMMANDBUTTON_CLOSE, theTooltip);
	return b;
}
entity makeTeamButton(string theName, vector theColor, string commandtheName)
{
	return makeTeamButton_T(theName, theColor, commandtheName, string_null);
}

void XonoticTeamSelectDialog_showNotify(entity me)
{
	SUPER(XonoticTeamSelectDialog).showNotify(me);
	string rows = strcat(",", cvar_string("_teams_available_rows"), ",");
	me.team1.disabled = strstrofs(rows, ",1,", 0) < 0;
	me.team2.disabled = strstrofs(rows, ",2,", 0) < 0;
	me.team3.disabled = strstrofs(rows, ",3,", 0) < 0;
	me.team4.disabled = strstrofs(rows, ",4,", 0) < 0;
	me.team5.disabled = strstrofs(rows, ",5,", 0) < 0;
}

void XonoticTeamSelectDialog_fill(entity me)
{
	entity e;
	me.TR(me);
		me.TD(me, 2, 5, e = makeTeamButton_T(_("join 'best' team (auto-select)"), '0 0 0', "cmd selectteam auto; cmd join",
			_("Autoselect team (recommended)")));
			e.preferredFocusPriority = 1;
	me.TR(me);
	me.TR(me);
		me.TD(me, 2, 1, e = me.team1 = makeTeamButton(_("red"), '1 0.5 0.5', "cmd selectteam red; cmd join"));
		me.TD(me, 2, 1, e = me.team2 = makeTeamButton(_("blue"), '0.5 0.5 1', "cmd selectteam blue; cmd join"));
		me.TD(me, 2, 1, e = me.team3 = makeTeamButton(_("yellow"), '1 1 0.5', "cmd selectteam yellow; cmd join"));
		me.TD(me, 2, 1, e = me.team4 = makeTeamButton(_("pink"), '1 0.5 1', "cmd selectteam pink; cmd join"));
		me.TD(me, 2, 1, e = me.team5 = makeTeamButton(_("green"), '0.5 1 0.5', "cmd selectteam green; cmd join"));
	me.TR(me);
		me.TD(me, 1, 1.5, makeXonoticTextLabel(0.5, _("team number:")));
		me.TD(me, 1, 2, e = makeXonoticInputBox(0, "_team_select_number"));
			e.saveImmediately = 1;
		me.TD(me, 1, 1.5, makeXonoticBigCommandButton(_("join team"), '0 0 0', "cmd selectteam $_team_select_number; cmd join", COMMANDBUTTON_CLOSE));
	me.TR(me);
		me.TD(me, 1, 5, makeXonoticCommandButton(_("spectate"), '0 0 0', "cmd spectate", COMMANDBUTTON_CLOSE));
}
