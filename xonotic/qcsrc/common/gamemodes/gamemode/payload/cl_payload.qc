#include "cl_payload.qh"

#include "payload.qh"

#include <client/draw.qh>
#include <client/hud/panel/modicons.qh>

void HUD_Mod_Payload_Export(int fh)
{
	HUD_Write_Cvar("hud_panel_modicons_payload_layout");
}

int autocvar_hud_panel_modicons_payload_layout;

void HUD_Mod_Payload(vector myPos, vector mySize)
{
	mod_active = 1;

	float progress = bound(0, STAT(PAYLOAD_PROGRESS), 1);
	float spd = STAT(PAYLOAD_SPEED);
	int push = STAT(PAYLOAD_PUSH_PACKED);
	int goals = STAT(PAYLOAD_GOALS_PACKED);

	float barh = mySize.y * 0.34;
	vector barPos = myPos + vec2(0, (mySize.y - barh) * 0.5);
	vector barSize = vec2(mySize.x, barh);

	drawfill(barPos, barSize, '0.1 0.1 0.1', panel_fg_alpha * 0.5, DRAWFLAG_NORMAL);

	for(int j = 1; j <= NUM_TEAMS; ++j)
	{
		float f = ((goals >> ((j - 1) * PLC_PUSH_BITS)) & PLC_PUSH_MASK) / PLC_PUSH_MASK;
		float w = payload_unpack_push(push, j);
		vector col = Team_ColorRGB(Team_IndexToTeam(j));
		drawfill(barPos + vec2(f * barSize.x - barSize.y * 0.1, 0), vec2(barSize.y * 0.2, barSize.y), col, panel_fg_alpha, DRAWFLAG_NORMAL);
		if(w > 0)
			drawfill(barPos + vec2(f * barSize.x - barSize.y * 0.1, -barSize.y * 0.5), vec2(barSize.y * 0.2, barSize.y * 0.4 * min(1, w)), col, panel_fg_alpha, DRAWFLAG_ADDITIVE);
	}

	vector cartCol = (spd == 0) ? '0.6 0.6 0.6' : '1 0.8 0.2';
	drawfill(barPos + vec2(progress * barSize.x - barSize.y * 0.25, -barSize.y * 0.15), vec2(barSize.y * 0.5, barSize.y * 1.3), cartCol, panel_fg_alpha, DRAWFLAG_NORMAL);

	if(autocvar_hud_panel_modicons_payload_layout)
		drawstring_aspect(myPos, strcat(ftos(floor(progress * 100 + 0.5)), "%"), vec2(mySize.x * 0.25, mySize.y * 0.3), '1 1 1', panel_fg_alpha, DRAWFLAG_NORMAL);
}
