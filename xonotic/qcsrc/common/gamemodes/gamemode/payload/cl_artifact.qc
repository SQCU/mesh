#include "cl_artifact.qh"

#include "payload.qh"
#include "ink.qh"

#include <client/draw.qh>
#include <client/main.qh>
#include <common/teams.qh>
#include <lib/macro.qh>
#include <lib/warpzone/mathlib.qh>

// ===========================================================================
// The artifact
//
// A knot curve lives in R^5.  Every frame it is rotated by a product of SO(5)
// plane (Givens) rotations with mutually incommensurate rates, then projected
// down to R^3 by two successive perspective divides - 5D->4D on the fifth
// coordinate, then 4D->3D on the fourth.  Because the rotation mixes the two
// hidden axes into the visible ones, the projected body continuously turns
// itself inside out while its 5D shape never changes.
//
// Cost is a handful of trig per frame: ART_SEG samples, each 5 sin/cos for the
// curve, five 2x2 Givens applications (20 multiply-adds), two divides, and a
// frame construction.  The polygon emission dominates, so ART_SEG and ART_SIDES
// are the tuning knobs, not the mathematics.
// ===========================================================================

float autocvar_cl_artifact = 1;
float autocvar_cl_artifact_segments = 40;
float autocvar_cl_artifact_sides = 6;
float autocvar_cl_artifact_scale = 110;
float autocvar_cl_artifact_tube = 13;
float autocvar_cl_artifact_speed = 0.11;
float autocvar_cl_artifact_glob_interval = 0.20;
float autocvar_cl_artifact_glob_radius = 420;
// the drift is a low-amplitude wash: it saturates coverage without seizing hue
float autocvar_cl_artifact_glob_amount = 0.14;
// the carts are the opposite - a narrow, decisive, high-amplitude write
float autocvar_cl_artifact_cart_ink_radius = 220;
float autocvar_cl_artifact_cart_ink_amount = 0.55;
float autocvar_cl_artifact_glob_reach = 4096;
float autocvar_cl_artifact_catchup = 3000;
float autocvar_cl_artifact_sky = 1;
float autocvar_cl_artifact_sky_distance = 26000;
float autocvar_cl_artifact_sky_scale = 26;
// the sky body's tube is thickened relative to its curve so it reads as a solid
// volume at 26,000 units rather than as wireframe
float autocvar_cl_artifact_sky_thickness = 2.6;
float autocvar_cl_artifact_alpha = 0.62;
float autocvar_cl_artifact_drift = 1;

// the neutral drift colour: an off-key sour green that reads as "not a team"
vector autocvar_cl_artifact_color = '0.42 0.95 0.30';

// Capture harness: pin the view so two screenshots taken minutes apart frame the
// identical geometry.  0 off, -1 report the live view each second, 1 pin to
// cl_artifact_cam_origin / cl_artifact_cam_angles.
float autocvar_cl_artifact_camera = 0;
vector autocvar_cl_artifact_cam_origin = '0 0 0';
vector autocvar_cl_artifact_cam_angles = '0 0 0';

const int ART_MAXSIDES = 12;

// ---------------------------------------------------------------------------
// 5D curve, SO(5) rotation, projection
// ---------------------------------------------------------------------------

// The knot, as five coordinates.  x/y/z are the standard (2,3) torus knot
// (trefoil); w and v carry two further harmonics so that the rotation has real
// structure to fold into the visible three.
// QuakeC has no 5-vectors, so a sample is returned as a vector plus two floats
// through these globals.
vector art_p3;
float art_p4, art_p5;

void Artifact_Curve(float t)
{
	art_p3_x = sin(t) + 2 * sin(2 * t);
	art_p3_y = cos(t) - 2 * cos(2 * t);
	art_p3_z = -sin(3 * t);
	art_p4 = 1.30 * cos(5 * t);
	art_p5 = 1.30 * sin(4 * t);
}

// One Givens rotation in the (a,b) plane.  Written out per plane because QC has
// no arrays of locals worth the indirection.
#define ART_GIVENS(A, B, ANG) \
	MACRO_BEGIN \
		float _c = cos(ANG), _s = sin(ANG); \
		float _a = (A), _b = (B); \
		(A) = _a * _c - _b * _s; \
		(B) = _a * _s + _b * _c; \
	MACRO_END

// Rotate the current sample by R(phase) in SO(5) and project to R^3.
// The five rates are irrational multiples of each other, so the orbit in SO(5)
// is dense and the body never repeats a pose.
vector Artifact_Project(float phase)
{
	float x = art_p3_x, y = art_p3_y, z = art_p3_z, w = art_p4, v = art_p5;
	float d4, d5;

	ART_GIVENS(x, y, phase * 1.000);
	ART_GIVENS(z, w, phase * 0.6180);
	ART_GIVENS(x, v, phase * 0.4142);
	ART_GIVENS(y, w, phase * 0.7320);
	ART_GIVENS(z, v, phase * 0.3178);

	// 5D -> 4D, then 4D -> 3D.  The eye distances are comfortably larger than the
	// curve's extent in those axes, so the divides never blow up.
	d5 = 4.2 / (4.2 - v);
	x *= d5; y *= d5; z *= d5; w *= d5;
	d4 = 4.2 / (4.2 - w);
	return vec3(x * d4, y * d4, z * d4);
}

// ---------------------------------------------------------------------------
// Where the artifact is
// ---------------------------------------------------------------------------

vector art_worldmin, art_worldsize;
bool art_bounds_known;

void Artifact_Bounds()
{
	if (art_bounds_known)
		return;
	if (!ink_stat(INK_ACTIVE))
		return;
	art_worldmin = vec3(ink_stat(INK_MINS_X), ink_stat(INK_MINS_Y), ink_stat(INK_MINS_Z));
	art_worldsize = vec3(ink_stat(INK_SIZE_X), ink_stat(INK_SIZE_Y), ink_stat(INK_SIZE_Z));
	art_bounds_known = true;
}

// Drift path.  A Lissajous through the level's bounding volume, evaluated purely
// from match time, so every client places the artifact in the same spot on the
// same frame without a single byte of network traffic.
vector Artifact_Lissajous(float t)
{
	vector c = art_worldmin + art_worldsize * 0.5;
	vector r = art_worldsize * 0.36;
	float s = t * autocvar_cl_artifact_speed;
	return c + vec3(r.x * sin(s * 1.000 + 0.7),
	                r.y * sin(s * 0.7320 + 2.1),
	                r.z * 0.55 * sin(s * 1.6180 + 4.3));
}

// A bounding-box Lissajous spends much of its time in the void outside the sealed
// hull, where a glob has nothing to land on.  Clipping the ray from the level's
// centre out to the Lissajous point keeps the artifact inside the map, sliding
// along the inner faces as it orbits - which is what "traverses regions" wants
// anyway.  One traceline per frame.
vector Artifact_Origin(float t)
{
	vector c = art_worldmin + art_worldsize * 0.5;
	vector l = Artifact_Lissajous(t);
	traceline(c, l, MOVE_WORLDONLY, NULL);
	if (trace_fraction < 1)
		return c + (l - c) * trace_fraction * 0.90;
	return l;
}

// ---------------------------------------------------------------------------
// Tube sweep
// ---------------------------------------------------------------------------

// Emit the projected knot as a tube.  `origin` places it, `scale` sizes the
// curve, `radius` the tube, `phase` drives the SO(5) rotation, `col` tints it,
// `col2` is the contesting colour and `contest` (0..1) how strongly it is
// claimed - when it is nonzero the two colours alternate along a helical stripe
// so a contested cart reads as visibly streaked in a still frame rather than as
// a flicker.  `band` (0..1) is a bright travelling marker used to show a cart's
// progress along its path (negative disables it).
void Artifact_DrawTube(vector origin, float scale, float radius, float phase,
                       vector col, vector col2, float contest,
                       float alpha, float band, int segments, int sides)
{
	vector prev = '0 0 0', prevu = '0 0 0', prevr = '0 0 0';
	float i, j, tau = 2 * M_PI;
	bool have_prev = false;

	segments = bound(6, segments, 96);
	sides = bound(3, sides, ART_MAXSIDES);

	for (i = 0; i <= segments; ++i)
	{
		float u = i / segments;
		vector p, tangent, up, rt;

		Artifact_Curve(u * tau);
		p = origin + Artifact_Project(phase) * scale;

		// forward difference for the tangent; one extra curve evaluation per
		// sample is cheaper and steadier than differencing the projected points
		Artifact_Curve((u + 1 / (segments * 4)) * tau);
		tangent = (origin + Artifact_Project(phase) * scale) - p;
		if (vlen(tangent) < 0.0001)
			tangent = '1 0 0';
		tangent = normalize(tangent);

		// stable-ish frame: reference up, swapped near the degenerate pole
		up = (fabs(tangent.z) > 0.95) ? '1 0 0' : '0 0 1';
		rt = normalize(cross(tangent, up));
		up = cross(rt, tangent);

		if (have_prev)
		{
			for (j = 0; j < sides; ++j)
			{
				float a0 = tau * j / sides, a1 = tau * (j + 1) / sides;
				vector o0 = prevr * cos(a0) + prevu * sin(a0);
				vector o1 = prevr * cos(a1) + prevu * sin(a1);
				vector n0 = rt * cos(a0) + up * sin(a0);
				vector n1 = rt * cos(a1) + up * sin(a1);
				vector base = (contest > 0 && ((j + i) & 1)) ? (col * (1 - contest) + col2 * contest) : col;
				vector ca = base, cb = base;

				// a bright band travelling along the tube reads as "how far"
				if (band >= 0)
				{
					float du = fabs(u - band);
					du = min(du, 1 - du);
					float k = max(0, 1 - du * 8);
					ca = base + (('1 1 1' - base) * k);
					cb = ca;
				}

				R_BeginPolygon("", DRAWFLAG_ADDITIVE, false);
				R_PolygonVertex(prev + o0 * radius, vec3(j / sides, 0, 0), ca, alpha);
				R_PolygonVertex(prev + o1 * radius, vec3((j + 1) / sides, 0, 0), ca, alpha);
				R_PolygonVertex(p + n1 * radius, vec3((j + 1) / sides, 1, 0), cb, alpha);
				R_PolygonVertex(p + n0 * radius, vec3(j / sides, 1, 0), cb, alpha);
				R_EndPolygon();
			}
		}
		prev = p;
		prevu = up;
		prevr = rt;
		have_prev = true;
	}
}

// ---------------------------------------------------------------------------
// Glob dropping
// ---------------------------------------------------------------------------

// Globs are dropped on a fixed cadence keyed to match time, so the sequence of
// (position, tick) pairs is identical on every client - the accumulated ink is
// consistent without replicating splats.
float art_lastglob = -1;

// The drift and the carts write the SAME voxels, so they have to mean different
// things in them or one erases the other.  The volume already separates them:
// coverage (alpha) drives how wet and rubbery a surface is, colour (rgb) drives
// what it is tinted.  The carts own colour - they deposit saturated team paint
// at a high rate along a track.  The drift owns coverage: it deposits at a low
// per-glob amount over a wide radius, and its colour is the mean colour the
// world ALREADY carries (ink_stat's cached global tint, the same pair the engine
// tints the sky with), pulled toward the sour green only as far as the world is
// still unpainted.  So the drift wets everything and repaints nothing: it makes
// the world rubberier and off-colour without arguing with a team about whose
// territory a corridor is.
vector Artifact_DriftColor()
{
	float cov = bound(0, ink_stat(INK_COVERAGE), 1);
	vector mean = vec3(ink_stat(INK_TINT_R), ink_stat(INK_TINT_G), ink_stat(INK_TINT_B));
	return autocvar_cl_artifact_color * (1 - cov) + mean * cov;
}

void Artifact_DropGlob(float tick, vector col, float amount)
{
	float t = tick * autocvar_cl_artifact_glob_interval;
	vector src = Artifact_Origin(t);
	vector dst = src - '0 0 1' * autocvar_cl_artifact_glob_reach;

	traceline(src, dst, MOVE_WORLDONLY, NULL);
	if (trace_fraction >= 1)
	{
		// nothing underneath: throw it at the nearest wall instead, so a glob
		// released in a shaft or over a pit still lands somewhere
		vector c = art_worldmin + art_worldsize * 0.5;
		vector d = c - src;
		if (vlen(d) < 1)
			return;
		traceline(src, src + normalize(d) * autocvar_cl_artifact_glob_reach, MOVE_WORLDONLY, NULL);
		if (trace_fraction >= 1)
			return;
	}
	// lift the deposit off the surface by a touch: the shader biases its lookup
	// along the surface normal, and the two together keep paint off the far side
	// of thin geometry
	ink_splat(trace_endpos + trace_plane_normal * 8, autocvar_cl_artifact_glob_radius, col, amount);
}

void Artifact_DropGlobs()
{
	float now = floor(time / autocvar_cl_artifact_glob_interval);
	float first;

	if (art_lastglob < 0)
	{
		// A client that joins mid-match replays the globs it missed, so it sees the
		// same painted world as everyone else.  Bounded, so a long match costs a
		// one-off burst rather than an unbounded stall.
		first = max(0, now - autocvar_cl_artifact_catchup);
	}
	else
	{
		first = art_lastglob + 1;
		if (now - first > autocvar_cl_artifact_catchup)
			first = now - autocvar_cl_artifact_catchup;
	}
	vector col = Artifact_DriftColor();
	for (float k = first; k <= now; ++k)
		Artifact_DropGlob(k, col, autocvar_cl_artifact_glob_amount);
	art_lastglob = now;
}

// ---------------------------------------------------------------------------
// Carts, rendered as instances of the same body
// ---------------------------------------------------------------------------

// A cart's world position is NOT re-derived here.  An earlier revision walked the
// radar-link polyline with the cart's arclength fraction, which is wrong twice
// over: g_radarlinks holds every cart's links in one undifferentiated list, and
// those links are subsampled to at most PLC_RIBBON_SEG straight hops, so the
// reconstructed point left the track on every curve.  The cart's exact origin is
// already networked - payload_cart_read() reads it off the cart's own
// waypointsprite - so this file only draws it.

float art_cart_lastink = -1;

void Artifact_DrawCarts(float phase)
{
	float inktick = floor(time / PLC_TICK);
	bool doink = (inktick != art_cart_lastink);

	for (int c = 0; c < PLC_MAX_CARTS; ++c)
	{
		payload_cart_read(c);
		if (!plc_r_present || !plc_r_haveorigin)
			continue;

		vector pos = plc_r_origin;
		float glow = autocvar_cl_artifact_alpha * plc_r_glow;
		// The body.  Same curve, same SO(5) rotation, same sweep as the drifting
		// artifact - a cart IS one of these, only smaller, tinted by who holds it
		// and pinned to a track instead of drifting.  Its size comes from cvars,
		// not from whatever brush the source map handed the cart, so every cart on
		// every fused tile has the same silhouette and the same bounds.
		Artifact_DrawTube(pos + '0 0 40', autocvar_cl_artifact_scale * 0.42,
			autocvar_cl_artifact_tube * 0.8, phase * 1.7 + c * 2.0,
			plc_r_color, plc_r_color2, plc_r_contest, glow,
			-1, autocvar_cl_artifact_segments * 0.6, autocvar_cl_artifact_sides);

		// Ink is cart territory.  An advancing cart lays its controller's colour
		// on the ground it has just covered; a contested cart lays the muddied
		// blend of the two claimants; a cart being driven back has its own paint
		// overpainted by the team pushing it home (payload_cart_read resolves
		// which of those plc_r_inkcolor is).  Laid on the cart's own PLC_TICK, so
		// the paint advances at exactly the rate the cart's control law runs at.
		if (doink)
		{
			vector inkat = pos;
			traceline(pos + '0 0 40', pos - '0 0 512', MOVE_WORLDONLY, NULL);
			if (trace_fraction < 1)
				inkat = trace_endpos + trace_plane_normal * 8;
			ink_splat(inkat, autocvar_cl_artifact_cart_ink_radius, plc_r_inkcolor,
				autocvar_cl_artifact_cart_ink_amount * (plc_r_stall ? 0.2 : 1));
		}

		// Supermassive sky counterpart.  The fused world is ~152,000 units across;
		// nobody can see the far theatre, so each cart also hangs against the sky
		// in the direction it actually lies, wearing its control colour and its
		// contest streak, with the travelling band showing how deep along its path
		// it is.  Whose cart, where, what state and how far - without a HUD.
		if (autocvar_cl_artifact_sky)
		{
			vector dir = pos - view_origin;
			dir.z = 0;
			if (vlen(dir) < 1)
				dir = '1 0 0';
			dir = normalize(dir);
			// each cart gets its own elevation band so k carts read as k bodies
			// rather than as one tangle in the same patch of sky
			vector skyat = view_origin + dir * autocvar_cl_artifact_sky_distance
				+ '0 0 1' * (autocvar_cl_artifact_sky_distance * (0.30 + 0.14 * c));
			Artifact_DrawTube(skyat,
				autocvar_cl_artifact_scale * autocvar_cl_artifact_sky_scale,
				autocvar_cl_artifact_tube * autocvar_cl_artifact_sky_scale * autocvar_cl_artifact_sky_thickness,
				phase * 0.55 + c * 2.0, plc_r_color, plc_r_color2, plc_r_contest,
				glow * 0.8, plc_r_frac, autocvar_cl_artifact_segments * 0.5, 6);
		}
	}

	if (doink)
		art_cart_lastink = inktick;
}

// ---------------------------------------------------------------------------

entity art_mgr;

float art_camreport;

void Artifact_Camera()
{
	if (autocvar_cl_artifact_camera == 0)
		return;
	if (autocvar_cl_artifact_camera < 0)
	{
		if (time > art_camreport)
		{
			art_camreport = time + 1;
			print(sprintf("^3view origin %v angles %v\n", view_origin, view_angles));
		}
		return;
	}
	setproperty(VF_ORIGIN, autocvar_cl_artifact_cam_origin);
	setproperty(VF_ANGLES, autocvar_cl_artifact_cam_angles);
	view_origin = autocvar_cl_artifact_cam_origin;
	view_angles = autocvar_cl_artifact_cam_angles;
	MAKE_VECTORS(view_angles, view_forward, view_right, view_up);
}

void Artifact_Draw(entity this)
{
	float phase;

	Artifact_Camera();

	if (!autocvar_cl_artifact || !ink_stat(INK_ACTIVE))
		return;

	Artifact_Bounds();
	if (!art_bounds_known)
		return;

	phase = time * 0.45;

	if (autocvar_cl_artifact_drift)
	{
		vector at = Artifact_Origin(time);
		vector dcol = Artifact_DriftColor();
		Artifact_DrawTube(at, autocvar_cl_artifact_scale, autocvar_cl_artifact_tube,
			phase, dcol, dcol, 0, autocvar_cl_artifact_alpha, -1,
			autocvar_cl_artifact_segments, autocvar_cl_artifact_sides);

		// the same body, supermassive, dropped against the sky
		if (autocvar_cl_artifact_sky)
		{
			vector dir = at - view_origin;
			dir.z = 0;
			if (vlen(dir) < 1)
				dir = '1 0 0';
			dir = normalize(dir);
			Artifact_DrawTube(view_origin + dir * autocvar_cl_artifact_sky_distance
					+ '0 0 1' * (autocvar_cl_artifact_sky_distance * 0.5),
				autocvar_cl_artifact_scale * autocvar_cl_artifact_sky_scale,
				autocvar_cl_artifact_tube * autocvar_cl_artifact_sky_scale * autocvar_cl_artifact_sky_thickness,
				phase * 0.4, dcol, dcol, 0,
				autocvar_cl_artifact_alpha * 0.75, -1,
				autocvar_cl_artifact_segments * 0.5, 6);
		}

		Artifact_DropGlobs();
	}

	Artifact_DrawCarts(phase);
}

void Artifact_Setup()
{
	if (art_mgr)
		return;
	art_mgr = new(payload_artifact);
	art_mgr.draw = Artifact_Draw;
	IL_PUSH(g_drawables, art_mgr);
}

// The artifact is a world effect, not a HUD widget, so it registers itself once
// at CSQC init rather than waiting for a gamemode panel to wake it up.
// STATIC_INIT_LATE runs after the STATIC_INIT that creates g_drawables.
STATIC_INIT_LATE(payload_artifact)
{
	Artifact_Setup();
}

