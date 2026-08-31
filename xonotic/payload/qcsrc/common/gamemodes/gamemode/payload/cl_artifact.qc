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
float autocvar_cl_artifact_glob_interval = 0.35;
float autocvar_cl_artifact_glob_radius = 190;
float autocvar_cl_artifact_glob_amount = 0.5;
float autocvar_cl_artifact_glob_reach = 4096;
float autocvar_cl_artifact_catchup = 3000;
float autocvar_cl_artifact_sky = 1;
float autocvar_cl_artifact_sky_distance = 26000;
float autocvar_cl_artifact_sky_scale = 62;
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
// `band` (0..1) is a bright travelling marker used to show a cart's progress
// along its path (negative disables it).
void Artifact_DrawTube(vector origin, float scale, float radius, float phase,
                       vector col, float alpha, float band, int segments, int sides)
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
				vector ca = col, cb = col;

				// a bright band travelling along the tube reads as "how far"
				if (band >= 0)
				{
					float du = fabs(u - band);
					du = min(du, 1 - du);
					float k = max(0, 1 - du * 8);
					ca = col + (('1 1 1' - col) * k);
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
	for (float k = first; k <= now; ++k)
		Artifact_DropGlob(k, autocvar_cl_artifact_color, autocvar_cl_artifact_glob_amount);
	art_lastglob = now;
}

// ---------------------------------------------------------------------------
// Carts, rendered as instances of the same body
// ---------------------------------------------------------------------------

// The cart's position along its path comes from the radar-link polyline that the
// payload gamemode already networks: each link's .origin is its start node and
// .velocity its end node.  Walking that polyline with the cart's arclength
// fraction reproduces the cart's world position without a second source of truth.
float art_pathlen;
float art_pathvalid;

float Artifact_PathLength()
{
	float total = 0;
	IL_EACH(g_radarlinks, true, { total += vlen(it.velocity - it.origin); });
	return total;
}

vector Artifact_PathPoint(float frac)
{
	float want, acc = 0;
	vector best = '0 0 0';
	bool found = false;
	if (art_pathlen <= 0)
		return '0 0 0';
	want = bound(0, frac, 1) * art_pathlen;
	IL_EACH(g_radarlinks, !found,
	{
		float l = vlen(it.velocity - it.origin);
		if (acc + l >= want)
		{
			best = it.origin + (it.velocity - it.origin) * ((l > 0) ? (want - acc) / l : 0);
			found = true;
		}
		else
		{
			acc += l;
			best = it.velocity;
		}
	});
	return best;
}

// Colour a cart by its control state.  Grey when nobody holds it, the team colour
// under a plurality, and a streak between the two strongest teams when the
// activation cylinder is contested.
vector Artifact_CartColor(int ctrl, int push, bool regress, float streakphase)
{
	int j, bestj = 0, secondj = 0;
	float best = 0, second = 0;
	vector a, b;

	for (j = 1; j <= NUM_TEAMS; ++j)
	{
		float w = payload_unpack_push(push, j);
		if (w > best) { second = best; secondj = bestj; best = w; bestj = j; }
		else if (w > second) { second = w; secondj = j; }
	}

	if (!ctrl && !bestj)
		return '0.55 0.55 0.55';

	a = ctrl ? Team_ColorRGB(Team_IndexToTeam(ctrl)) : Team_ColorRGB(Team_IndexToTeam(bestj));
	if (second <= 0 || !secondj)
		return regress ? (a * 0.5 + '0.5 0.15 0.1') : a;

	// contested: alternate between the two claimants so it reads as streaked
	b = Team_ColorRGB(Team_IndexToTeam(secondj));
	return (sin(streakphase * 9) > 0) ? a : b;
}

float art_cart_lastink = -1;

void Artifact_DrawCarts(float phase)
{
	int carts = STAT(PAYLOAD_GOALS_PACKED);
	int cstate = STAT(PAYLOAD_CARTS_STATE);
	int push = STAT(PAYLOAD_PUSH_PACKED);
	float lead = bound(0, STAT(PAYLOAD_PROGRESS), 1);
	float inktick = floor(time / PLC_TICK);
	bool doink = (inktick != art_cart_lastink);

	if (!art_pathvalid)
	{
		art_pathlen = Artifact_PathLength();
		art_pathvalid = (art_pathlen > 0);
		if (!art_pathvalid)
			return;
	}

	for (int c = 0; c < PLC_MAX_CARTS; ++c)
	{
		int q = (carts >> (c * PLC_PUSH_BITS)) & PLC_PUSH_MASK;
		if (!q)
			continue;

		// Cart 0's arclength arrives as a full-precision float; the others are the
		// 6-bit field, which is all that is currently networked.
		float f = (c == 0) ? lead : ((q - 1) / (PLC_PUSH_MASK - 1));
		int st = payload_unpack_cart_state(cstate, c);
		int ctrl = st & 7;
		bool regress = boolean(st & PLC_CS_REGRESS);
		bool stall = boolean(st & PLC_CS_STALL);
		vector pos = Artifact_PathPoint(f);
		vector col = Artifact_CartColor(ctrl, push, regress, time + c);

		if (pos == '0 0 0')
			continue;

		// the body itself, sized consistently regardless of what brush model the
		// map happened to hand the cart
		Artifact_DrawTube(pos + '0 0 40', autocvar_cl_artifact_scale * 0.42,
			autocvar_cl_artifact_tube * 0.8, phase * 1.7 + c * 2.0, col,
			(stall ? autocvar_cl_artifact_alpha * 0.6 : autocvar_cl_artifact_alpha),
			-1, autocvar_cl_artifact_segments * 0.6, autocvar_cl_artifact_sides);

		// Ink is cart territory: an advancing cart lays its controller's colour
		// along the path it has just covered, a contested cart lays the streak, and
		// a cart driven back gets overpainted by whoever is pushing it home.
		if (doink)
		{
			vector inkat = pos;
			traceline(pos + '0 0 40', pos - '0 0 512', MOVE_WORLDONLY, NULL);
			if (trace_fraction < 1)
				inkat = trace_endpos + trace_plane_normal * 8;
			ink_splat(inkat, autocvar_cl_artifact_glob_radius * 0.8, col,
				(stall ? 0.05 : 0.30));
		}

		// Supermassive sky counterpart: on a world you cannot see across, this is
		// how a cart stays legible.  It sits far enough out to read as sky, so it
		// is occluded by level geometry exactly where the sky would be, and the
		// travelling band shows how deep along its path the cart is.
		if (autocvar_cl_artifact_sky)
		{
			vector dir = pos - view_origin;
			dir.z = 0;
			if (vlen(dir) < 1)
				dir = '1 0 0';
			dir = normalize(dir);
			vector skyat = view_origin + dir * autocvar_cl_artifact_sky_distance
				+ '0 0 1' * (autocvar_cl_artifact_sky_distance * 0.42);
			Artifact_DrawTube(skyat,
				autocvar_cl_artifact_scale * autocvar_cl_artifact_sky_scale,
				autocvar_cl_artifact_tube * autocvar_cl_artifact_sky_scale * 0.9,
				phase * 0.55 + c * 2.0, col, autocvar_cl_artifact_alpha * 0.8,
				f, autocvar_cl_artifact_segments * 0.5, 4);
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
		Artifact_DrawTube(at, autocvar_cl_artifact_scale, autocvar_cl_artifact_tube,
			phase, autocvar_cl_artifact_color, autocvar_cl_artifact_alpha, -1,
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
				autocvar_cl_artifact_tube * autocvar_cl_artifact_sky_scale,
				phase * 0.4, autocvar_cl_artifact_color,
				autocvar_cl_artifact_alpha * 0.75, -1,
				autocvar_cl_artifact_segments * 0.5, 4);
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
