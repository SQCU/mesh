#include "cl_artifact.qh"

#include "payload.qh"
#include "ink.qh"

#include <client/draw.qh>
#include <client/main.qh>
#include <common/teams.qh>
#include <lib/macro.qh>
#include <lib/warpzone/mathlib.qh>

float autocvar_cl_artifact = 1;
float autocvar_cl_artifact_segments = 40;
float autocvar_cl_artifact_sides = 6;
float autocvar_cl_artifact_scale = 110;
float autocvar_cl_artifact_tube = 13;
float autocvar_cl_artifact_speed = 0.11;
float autocvar_cl_artifact_glob_interval = 0.20;
float autocvar_cl_artifact_glob_radius = 420;

float autocvar_cl_artifact_glob_amount = 0.14;

float autocvar_cl_artifact_cart_ink_radius = 220;
float autocvar_cl_artifact_cart_ink_amount = 0.55;
float autocvar_cl_artifact_glob_reach = 4096;
float autocvar_cl_artifact_catchup = 3000;
float autocvar_cl_artifact_sky = 1;
float autocvar_cl_artifact_sky_distance = 26000;
float autocvar_cl_artifact_sky_scale = 26;

float autocvar_cl_artifact_sky_thickness = 2.6;
float autocvar_cl_artifact_alpha = 0.62;
float autocvar_cl_artifact_drift = 1;

vector autocvar_cl_artifact_color = '0.42 0.95 0.30';

float autocvar_cl_artifact_camera = 0;
vector autocvar_cl_artifact_cam_origin = '0 0 0';
vector autocvar_cl_artifact_cam_angles = '0 0 0';

const int ART_MAXSIDES = 12;

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

#define ART_GIVENS(A, B, ANG) \
	MACRO_BEGIN \
		float _c = cos(ANG), _s = sin(ANG); \
		float _a = (A), _b = (B); \
		(A) = _a * _c - _b * _s; \
		(B) = _a * _s + _b * _c; \
	MACRO_END

vector Artifact_Project(float phase)
{
	float x = art_p3_x, y = art_p3_y, z = art_p3_z, w = art_p4, v = art_p5;
	float d4, d5;

	ART_GIVENS(x, y, phase * 1.000);
	ART_GIVENS(z, w, phase * 0.6180);
	ART_GIVENS(x, v, phase * 0.4142);
	ART_GIVENS(y, w, phase * 0.7320);
	ART_GIVENS(z, v, phase * 0.3178);

	d5 = 8 / (8 - v);
	x *= d5; y *= d5; z *= d5; w *= d5;
	d4 = 16 / (16 - w);
	return vec3(x * d4, y * d4, z * d4);
}

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

vector Artifact_Lissajous(float t)
{
	vector c = art_worldmin + art_worldsize * 0.5;
	vector r = art_worldsize * 0.36;
	float s = t * autocvar_cl_artifact_speed;
	return c + vec3(r.x * sin(s * 1.000 + 0.7),
	                r.y * sin(s * 0.7320 + 2.1),
	                r.z * 0.55 * sin(s * 1.6180 + 4.3));
}

vector Artifact_Origin(float t)
{
	vector c = art_worldmin + art_worldsize * 0.5;
	vector l = Artifact_Lissajous(t);
	traceline(c, l, MOVE_WORLDONLY, NULL);
	if (trace_fraction < 1)
		return c + (l - c) * trace_fraction * 0.90;
	return l;
}

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

		Artifact_Curve((u + 1 / (segments * 4)) * tau);
		tangent = (origin + Artifact_Project(phase) * scale) - p;
		if (vlen(tangent) < 0.0001)
			tangent = '1 0 0';
		tangent = normalize(tangent);

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

float art_lastglob = -1;

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

		vector c = art_worldmin + art_worldsize * 0.5;
		vector d = c - src;
		if (vlen(d) < 1)
			return;
		traceline(src, src + normalize(d) * autocvar_cl_artifact_glob_reach, MOVE_WORLDONLY, NULL);
		if (trace_fraction >= 1)
			return;
	}

	ink_splat(trace_endpos + trace_plane_normal * 8, autocvar_cl_artifact_glob_radius, col, amount);
}

void Artifact_DropGlobs()
{
	float now = floor(time / autocvar_cl_artifact_glob_interval);
	float first;

	if (art_lastglob < 0)
	{

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

float art_cart_lastink = -1;

void Artifact_DrawCartHull(vector pos, vector col, float alpha)
{
	vector a = pos + '-32 -32 0';
	vector b = pos + '32 -32 0';
	vector c = pos + '32 32 0';
	vector d = pos + '-32 32 0';
	vector e = a + '0 0 80';
	vector f = b + '0 0 80';
	vector g = c + '0 0 80';
	vector h = d + '0 0 80';
	R_BeginPolygon("", DRAWFLAG_ADDITIVE, false);
	R_PolygonVertex(e, '0 0 0', col, alpha);
	R_PolygonVertex(f, '1 0 0', col, alpha);
	R_PolygonVertex(g, '1 1 0', col, alpha);
	R_PolygonVertex(h, '0 1 0', col, alpha);
	R_EndPolygon();
	R_BeginPolygon("", DRAWFLAG_ADDITIVE, false);
	R_PolygonVertex(a, '0 0 0', col, alpha * 0.45);
	R_PolygonVertex(b, '1 0 0', col, alpha * 0.45);
	R_PolygonVertex(f, '1 1 0', col, alpha * 0.45);
	R_PolygonVertex(e, '0 1 0', col, alpha * 0.45);
	R_EndPolygon();
	R_BeginPolygon("", DRAWFLAG_ADDITIVE, false);
	R_PolygonVertex(b, '0 0 0', col, alpha * 0.45);
	R_PolygonVertex(c, '1 0 0', col, alpha * 0.45);
	R_PolygonVertex(g, '1 1 0', col, alpha * 0.45);
	R_PolygonVertex(f, '0 1 0', col, alpha * 0.45);
	R_EndPolygon();
	R_BeginPolygon("", DRAWFLAG_ADDITIVE, false);
	R_PolygonVertex(c, '0 0 0', col, alpha * 0.45);
	R_PolygonVertex(d, '1 0 0', col, alpha * 0.45);
	R_PolygonVertex(h, '1 1 0', col, alpha * 0.45);
	R_PolygonVertex(g, '0 1 0', col, alpha * 0.45);
	R_EndPolygon();
	R_BeginPolygon("", DRAWFLAG_ADDITIVE, false);
	R_PolygonVertex(d, '0 0 0', col, alpha * 0.45);
	R_PolygonVertex(a, '1 0 0', col, alpha * 0.45);
	R_PolygonVertex(e, '1 1 0', col, alpha * 0.45);
	R_PolygonVertex(h, '0 1 0', col, alpha * 0.45);
	R_EndPolygon();
}

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
		Artifact_DrawCartHull(pos, plc_r_color, glow * 0.8);

		Artifact_DrawTube(pos + '0 0 40', autocvar_cl_artifact_scale * 0.06,
			autocvar_cl_artifact_tube * 0.5, phase * 1.7 + c * 2.0,
			plc_r_color, plc_r_color2, plc_r_contest, glow,
			-1, autocvar_cl_artifact_segments * 0.6, autocvar_cl_artifact_sides);

		if (doink)
		{
			vector inkat = pos;
			traceline(pos + '0 0 40', pos - '0 0 512', MOVE_WORLDONLY, NULL);
			if (trace_fraction < 1)
				inkat = trace_endpos + trace_plane_normal * 8;
			ink_splat(inkat, autocvar_cl_artifact_cart_ink_radius, plc_r_inkcolor,
				autocvar_cl_artifact_cart_ink_amount * (plc_r_stall ? 0.2 : 1));
		}

		if (autocvar_cl_artifact_sky)
		{
			vector dir = pos - view_origin;
			dir.z = 0;
			if (vlen(dir) < 1)
				dir = '1 0 0';
			dir = normalize(dir);

			vector skyat = view_origin + dir * autocvar_cl_artifact_sky_distance
				+ '0 0 1' * (autocvar_cl_artifact_sky_distance * (0.30 + 0.08 * (c % 8)));
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

STATIC_INIT_LATE(payload_artifact)
{
	Artifact_Setup();
}
