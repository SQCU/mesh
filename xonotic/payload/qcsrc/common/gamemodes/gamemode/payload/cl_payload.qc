#include "cl_payload.qh"

#include "payload.qh"

#include <client/draw.qh>
#include <client/main.qh>
#include <client/hud/panel/modicons.qh>
#include <lib/color.qh>

void HUD_Mod_Payload_Export(int fh)
{
	HUD_Write_Cvar("hud_panel_modicons_payload_layout");
}

int autocvar_hud_panel_modicons_payload_layout;
float autocvar_hud_panel_modicons_payload_ribbon = 1;
float autocvar_hud_panel_modicons_payload_ribbon_width = 12;

entity payload_ribbon_mgr;
float payload_ribbon_stamp;

void Payload_Ribbon_Draw(entity this)
{
	if(!autocvar_hud_panel_modicons_payload_ribbon || time - payload_ribbon_stamp > 0.5)
		return;
	float hw = autocvar_hud_panel_modicons_payload_ribbon_width;
	IL_EACH(g_radarlinks, true,
	{
		vector a = it.origin + '0 0 6';
		vector b = it.velocity + '0 0 6';
		vector dir = b - a;
		dir.z = 0;
		if(vlen(dir) < 1)
			continue;
		vector perp = normalize(vec2(-dir.y, dir.x)) * hw;
		vector c0 = colormapPaletteColor(it.team & 0x0F, false);
		vector c1 = colormapPaletteColor((it.team & 0xF0) / 0x10, false);
		R_BeginPolygon("", DRAWFLAG_ADDITIVE, false);
		R_PolygonVertex(a - perp, '0 0 0', c0, 0.28);
		R_PolygonVertex(a + perp, '1 0 0', c0, 0.28);
		R_PolygonVertex(b + perp, '1 1 0', c1, 0.28);
		R_PolygonVertex(b - perp, '0 1 0', c1, 0.28);
		R_EndPolygon();
	});
}

void Payload_Ribbon_Setup()
{
	payload_ribbon_stamp = time;
	if(payload_ribbon_mgr)
		return;
	payload_ribbon_mgr = new(payload_ribbon);
	payload_ribbon_mgr.draw = Payload_Ribbon_Draw;
	IL_PUSH(g_drawables, payload_ribbon_mgr);
}

void HUD_Mod_Payload(vector myPos, vector mySize)
{
	mod_active = 1;
	Payload_Ribbon_Setup();

	int carts = STAT(PAYLOAD_GOALS_PACKED);
	int cstate = STAT(PAYLOAD_CARTS_STATE);
	int push = STAT(PAYLOAD_PUSH_PACKED);
	float c0prog = bound(0, STAT(PAYLOAD_PROGRESS), 1);

	int n = 0;
	for(int c = 0; c < PLC_MAX_CARTS; ++c)
		if((carts >> (c * PLC_PUSH_BITS)) & PLC_PUSH_MASK)
			++n;
	if(n < 1)
		n = 1;

	float pipband = mySize.y * 0.16;
	float rowarea = mySize.y - pipband;
	float rowh = rowarea / n;
	float barh = rowh * 0.5;

	for(int j = 1; j <= NUM_TEAMS; ++j)
	{
		float w = payload_unpack_push(push, j);
		if(w > 0)
			drawfill(myPos + vec2((j - 1) * mySize.x / NUM_TEAMS, pipband * (1 - min(1, w))),
				vec2(mySize.x / NUM_TEAMS - mySize.x * 0.01, pipband * min(1, w)),
				Team_ColorRGB(Team_IndexToTeam(j)), panel_fg_alpha, DRAWFLAG_ADDITIVE);
	}

	int slot = 0;
	for(int c = 0; c < PLC_MAX_CARTS; ++c)
	{
		int q = (carts >> (c * PLC_PUSH_BITS)) & PLC_PUSH_MASK;
		if(!q && !(n == 1 && c == 0))
			continue;

		float f = (c == 0) ? c0prog : (q > 0 ? (q - 1) / (PLC_PUSH_MASK - 1) : 0);
		int st = payload_unpack_cart_state(cstate, c);
		int ctrl = st & 7;
		bool regress = boolean(st & PLC_CS_REGRESS);
		bool stall = boolean(st & PLC_CS_STALL);

		vector teamcol = ctrl ? Team_ColorRGB(Team_IndexToTeam(ctrl)) : '0.55 0.55 0.55';
		vector motcol = regress ? '1 0.3 0.2' : (stall ? '0.6 0.6 0.6' : teamcol);

		vector rp = myPos + vec2(0, pipband + slot * rowh + (rowh - barh) * 0.5);
		vector rs = vec2(mySize.x, barh);

		drawfill(rp, rs, '0.08 0.08 0.08', panel_fg_alpha * 0.6, DRAWFLAG_NORMAL);
		if(f > 0)
			drawfill(rp, vec2(f * rs.x, rs.y), teamcol, panel_fg_alpha * 0.45, DRAWFLAG_ADDITIVE);

		float hx = rp.x + bound(barh * 0.25, f * rs.x, rs.x - barh * 0.25);
		drawfill(vec2(hx - barh * 0.22, rp.y - barh * 0.15), vec2(barh * 0.44, barh * 1.3),
			motcol, panel_fg_alpha, DRAWFLAG_NORMAL);

		float gy = rp.y + barh * 0.5 - barh * 0.18;
		vector gc = myPos + vec2(mySize.x - barh * 0.4, gy);
		if(regress)
			drawfill(gc, vec2(barh * 0.36, barh * 0.36), '1 0.3 0.2', panel_fg_alpha, DRAWFLAG_ADDITIVE);
		else if(!stall)
			drawfill(gc, vec2(barh * 0.36, barh * 0.36), teamcol, panel_fg_alpha, DRAWFLAG_ADDITIVE);

		++slot;
	}

	if(autocvar_hud_panel_modicons_payload_layout)
		drawstring_aspect(myPos, strcat(ftos(floor(c0prog * 100 + 0.5)), "%"),
			vec2(mySize.x * 0.22, pipband), '1 1 1', panel_fg_alpha, DRAWFLAG_NORMAL);
}
