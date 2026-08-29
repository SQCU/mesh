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
	int carts = STAT(PAYLOAD_GOALS_PACKED);

	float barh = mySize.y * 0.3;
	vector barPos = myPos + vec2(0, mySize.y * 0.2);
	vector barSize = vec2(mySize.x, barh);

	drawfill(barPos, barSize, '0.1 0.1 0.1', panel_fg_alpha * 0.5, DRAWFLAG_NORMAL);

	for(int j = 1; j <= NUM_TEAMS; ++j)
	{
		float w = payload_unpack_push(push, j);
		if(w > 0)
			drawfill(barPos + vec2(progress * barSize.x + (j - 2.5) * barSize.y * 0.25, -barSize.y * 0.5),
				vec2(barSize.y * 0.2, barSize.y * 0.4 * min(1, w)),
				Team_ColorRGB(Team_IndexToTeam(j)), panel_fg_alpha, DRAWFLAG_ADDITIVE);
	}

	vector cartCol = (spd == 0) ? '0.6 0.6 0.6' : ((spd < 0) ? '1 0.3 0.2' : '1 0.8 0.2');
	drawfill(barPos + vec2(progress * barSize.x - barSize.y * 0.25, -barSize.y * 0.15), vec2(barSize.y * 0.5, barSize.y * 1.3), cartCol, panel_fg_alpha, DRAWFLAG_NORMAL);

	float suby = barPos.y + barh * 1.4;
	for(int c = 1; c < 4; ++c)
	{
		int q = (carts >> (c * PLC_PUSH_BITS)) & PLC_PUSH_MASK;
		if(!q)
			continue;
		float f = (q - 1) / (PLC_PUSH_MASK - 1);
		drawfill(vec2(barPos.x, suby), vec2(mySize.x, barh * 0.4), '0.1 0.1 0.1', panel_fg_alpha * 0.5, DRAWFLAG_NORMAL);
		drawfill(vec2(barPos.x + f * barSize.x - barh * 0.15, suby - barh * 0.1), vec2(barh * 0.3, barh * 0.6), '0.4 0.8 1', panel_fg_alpha, DRAWFLAG_NORMAL);
		suby += barh * 0.7;
	}

	if(autocvar_hud_panel_modicons_payload_layout)
		drawstring_aspect(myPos, strcat(ftos(floor(progress * 100 + 0.5)), "%"), vec2(mySize.x * 0.25, mySize.y * 0.3), '1 1 1', panel_fg_alpha, DRAWFLAG_NORMAL);
}
