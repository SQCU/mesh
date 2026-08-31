#include "payload.qh"

#ifdef CSQC
	#include <client/main.qh>
	#include <common/stats.qh>
	#include <common/mutators/mutator/waypoints/all.qh>
	#include <common/mutators/mutator/waypoints/waypointsprites.qh>
#endif

float payload_unpack_push(int packed, int idx)
{
	return ((packed >> ((idx - 1) * PLC_PUSH_BITS)) & PLC_PUSH_MASK) / PLC_PUSH_SCALE;
}

int payload_unpack_cart_state(int packed, int c)
{
	return (packed >> (c * PLC_CS_BITS)) & PLC_CS_MASK;
}

#ifdef CSQC

// Each cart's occupancy weights ride their own stat: four teams at PLC_PUSH_BITS
// each do not fit two carts in one 32-bit int, and a cart whose cylinder the
// client cannot see cannot be drawn as contested.
int payload_cart_push_packed(int c)
{
	switch (c)
	{
		case 1:  return STAT(PAYLOAD_PUSH_PACKED1);
		case 2:  return STAT(PAYLOAD_PUSH_PACKED2);
		case 3:  return STAT(PAYLOAD_PUSH_PACKED3);
		default: return STAT(PAYLOAD_PUSH_PACKED);
	}
}

// The cart's exact world origin comes from the cart's own waypointsprite, which
// is spawned with ref = the cart and therefore re-sends the cart's origin every
// time it moves.  Its spare wp_extra byte carries the cart id, so a sprite
// identifies which cart it is without a second entity or a second packing.
//
// Scanned once per frame for all carts rather than once per cart per caller.
float plc_scan_frame = -1;
vector plc_scan_pos[PLC_MAX_CARTS];
bool plc_scan_ok[PLC_MAX_CARTS];

void payload_cart_scan()
{
	if (plc_scan_frame == time)
		return;
	plc_scan_frame = time;
	for (int c = 0; c < PLC_MAX_CARTS; ++c)
		plc_scan_ok[c] = false;
	IL_EACH(g_radaricons,
		it.netname == WP_PayloadCart.netname || it.netname == WP_PayloadCart2.netname
		|| it.netname == WP_PayloadCart3.netname || it.netname == WP_PayloadCart4.netname,
	{
		int c = it.wp_extra;
		if (c >= 0 && c < PLC_MAX_CARTS)
		{
			// the sprite sits above the cart by its spawn offset; the cart body
			// and its ink want the cart, so take the offset back off
			plc_scan_pos[c] = it.origin - PLC_SPRITE_OFS;
			plc_scan_ok[c] = true;
		}
	});
}

// A path-link colour byte carries a team id minus one per nibble (the RADARLINK
// convention Onslaught already uses), 0 meaning "not yet anyone's".  Decoded
// through the same Team_ColorRGB every other cart colour comes from, so the
// world ribbon, the panel row, the body and the ink cannot drift apart.
vector payload_link_color(int nibble)
{
	return nibble ? Team_ColorRGB(nibble + 1) : '1 1 1';
}

// Control state is a colour.  Uncontrolled is grey; a plurality is that team's
// colour; a contested cylinder carries a second colour that the body streaks
// with and the ink muddies toward; a cart being driven home is inked by the team
// driving it, so the losing side's paint is overpainted as the cart goes back.
void payload_cart_read(int c)
{
	int carts = STAT(PAYLOAD_GOALS_PACKED);
	int q = (carts >> (c * PLC_PUSH_BITS)) & PLC_PUSH_MASK;
	int st = payload_unpack_cart_state(STAT(PAYLOAD_CARTS_STATE), c);
	int push = payload_cart_push_packed(c);
	float bestw = 0, secondw = 0, homew = 0;

	plc_r_present = (q != 0);
	// cart 0's arclength arrives at full precision; the rest carry the 6-bit field
	plc_r_frac = (c == 0) ? bound(0, STAT(PAYLOAD_PROGRESS), 1)
	                      : (q > 0 ? (q - 1) / (PLC_PUSH_MASK - 1) : 0);
	plc_r_ctrl = st & 7;
	plc_r_regress = boolean(st & PLC_CS_REGRESS);
	plc_r_stall = boolean(st & PLC_CS_STALL);

	plc_r_lead = plc_r_second = plc_r_home = 0;
	for (int j = 1; j <= NUM_TEAMS; ++j)
	{
		float w = payload_unpack_push(push, j);
		if (w <= 0)
			continue;
		if (w > bestw) { secondw = bestw; plc_r_second = plc_r_lead; bestw = w; plc_r_lead = j; }
		else if (w > secondw) { secondw = w; plc_r_second = j; }
		// whoever is driving it home is the strongest team that is NOT in control
		if (j != plc_r_ctrl && w > homew) { homew = w; plc_r_home = j; }
	}
	plc_r_contest = (bestw > 0) ? bound(0, secondw / bestw, 1) : 0;

	if (plc_r_ctrl)
		plc_r_color = Team_ColorRGB(Team_IndexToTeam(plc_r_ctrl));
	else if (plc_r_lead)
		// occupied but tied: nobody controls it, so it only half-wears the colour
		plc_r_color = Team_ColorRGB(Team_IndexToTeam(plc_r_lead)) * 0.5 + PLC_GREY * 0.5;
	else
		plc_r_color = PLC_GREY;

	plc_r_color2 = plc_r_second ? Team_ColorRGB(Team_IndexToTeam(plc_r_second)) : plc_r_color;

	if (plc_r_regress && plc_r_home)
		plc_r_inkcolor = Team_ColorRGB(Team_IndexToTeam(plc_r_home));
	else
		plc_r_inkcolor = plc_r_color * (1 - 0.5 * plc_r_contest) + plc_r_color2 * (0.5 * plc_r_contest);

	plc_r_glow = (plc_r_ctrl ? 1 : 0.45) * (plc_r_stall ? 0.6 : 1);

	payload_cart_scan();
	plc_r_haveorigin = plc_scan_ok[c];
	plc_r_origin = plc_scan_pos[c];
}

#endif
