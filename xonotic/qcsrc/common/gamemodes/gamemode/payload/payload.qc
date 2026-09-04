#include "payload.qh"

#ifdef CSQC
bool plc_cart_live[PLC_MAX_CARTS];
vector plc_cart_net_origin[PLC_MAX_CARTS];
float plc_cart_net_frac[PLC_MAX_CARTS];
float plc_cart_net_contest[PLC_MAX_CARTS];
int plc_cart_net_ctrl[PLC_MAX_CARTS];
int plc_cart_net_lead[PLC_MAX_CARTS];
int plc_cart_net_second[PLC_MAX_CARTS];
int plc_cart_net_home[PLC_MAX_CARTS];
int plc_cart_net_flags[PLC_MAX_CARTS];

vector payload_link_color(int team_index)
{
	return team_index ? Team_ColorRGB(Team_IndexToTeam(team_index)) : '1 1 1';
}

void payload_cart_read(int c)
{
	plc_r_present = plc_cart_live[c];
	if(!plc_r_present)
		return;
	plc_r_frac = plc_cart_net_frac[c];
	plc_r_ctrl = plc_cart_net_ctrl[c];
	plc_r_lead = plc_cart_net_lead[c];
	plc_r_second = plc_cart_net_second[c];
	plc_r_home = plc_cart_net_home[c];
	plc_r_contest = plc_cart_net_contest[c];
	plc_r_regress = boolean(plc_cart_net_flags[c] & 1);
	plc_r_stall = boolean(plc_cart_net_flags[c] & 2);
	plc_r_color = plc_r_ctrl ? Team_ColorRGB(Team_IndexToTeam(plc_r_ctrl)) : PLC_GREY;
	plc_r_color2 = plc_r_second ? Team_ColorRGB(Team_IndexToTeam(plc_r_second)) : plc_r_color;
	if(plc_r_regress && plc_r_home)
		plc_r_inkcolor = Team_ColorRGB(Team_IndexToTeam(plc_r_home));
	else
		plc_r_inkcolor = plc_r_color * (1 - 0.5 * plc_r_contest) + plc_r_color2 * (0.5 * plc_r_contest);
	plc_r_glow = (plc_r_ctrl ? 1 : 0.45) * (plc_r_stall ? 0.6 : 1);
	plc_r_haveorigin = true;
	plc_r_origin = plc_cart_net_origin[c];
}

void payload_cart_remove(entity this)
{
	plc_cart_live[this.plc_cart_id] = false;
}

NET_HANDLE(ENT_CLIENT_PAYLOAD_CART, bool isnew)
{
	make_pure(this);
	int c = ReadByte();
	this.plc_cart_id = c;
	plc_cart_live[c] = true;
	plc_cart_net_origin[c] = ReadVector();
	plc_cart_net_frac[c] = ReadFloat();
	plc_cart_net_ctrl[c] = ReadShort();
	plc_cart_net_lead[c] = ReadShort();
	plc_cart_net_second[c] = ReadShort();
	plc_cart_net_home[c] = ReadShort();
	plc_cart_net_contest[c] = ReadByte() / 255.0;
	plc_cart_net_flags[c] = ReadByte();
	this.entremove = payload_cart_remove;
	return = true;
}

NET_HANDLE(ENT_CLIENT_PAYLOAD_LINK, bool isnew)
{
	make_pure(this);
	int sendflags = ReadByte();
	if(isnew)
	{
		this.plc_payload_link = true;
		IL_PUSH(g_radarlinks, this);
	}
	if(sendflags & 1)
	{
		this.origin = ReadVector();
		setorigin(this, this.origin);
	}
	if(sendflags & 2)
		this.velocity = ReadVector();
	if(sendflags & 4)
	{
		this.plc_link_team_a = ReadShort();
		this.plc_link_team_b = ReadShort();
	}
	return = true;
}

#endif
