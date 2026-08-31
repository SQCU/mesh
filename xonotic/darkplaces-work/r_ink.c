#include "quakedef.h"
#include "r_ink.h"

r_ink_state_t r_ink_state;

cvar_t r_ink = {CVAR_SAVE, "r_ink", "1", "accumulate world ink from the paint artifact and payload carts"};
cvar_t r_ink_texelsize = {CVAR_SAVE, "r_ink_texelsize", "64", "target world units per ink voxel"};
cvar_t r_ink_maxresolution = {CVAR_SAVE, "r_ink_maxresolution", "96", "hard cap on ink volume dimension per axis (96^3 RGBA8 = 3.4 MB)"};
cvar_t r_ink_normalbias = {CVAR_SAVE, "r_ink_normalbias", "0.75", "how far along the surface normal, in voxels, the shader samples the ink volume - suppresses bleed through thin walls"};
cvar_t r_ink_roughness = {CVAR_SAVE, "r_ink_roughness", "0.14", "roughness a fully inked surface is pulled toward (low = wet lacquer, high = dry rubber)"};
cvar_t r_ink_tint = {CVAR_SAVE, "r_ink_tint", "0.85", "how far a fully inked surface's albedo moves toward the ink colour"};
cvar_t r_ink_f0 = {CVAR_SAVE, "r_ink_f0", "0.09", "normal-incidence reflectance of the ink film (0.05 rubber, 0.09 wet)"};
cvar_t r_ink_intensity = {CVAR_SAVE, "r_ink_intensity", "1", "global multiplier on accumulated coverage"};
cvar_t r_ink_noisescale = {CVAR_SAVE, "r_ink_noisescale", "170", "splatter cell frequency across the ink volume"};
cvar_t r_ink_noiseedge = {CVAR_SAVE, "r_ink_noiseedge", "0.22", "softness of the splatter edge (0 = hard cut)"};
cvar_t r_ink_skytint = {CVAR_SAVE, "r_ink_skytint", "1", "let accumulated ink tint the sky as well as the world"};

static mempool_t *r_ink_mempool;
static rtexturepool_t *r_ink_texturepool;
static char r_ink_worldname[MAX_QPATH];

// ---------------------------------------------------------------------------

static void R_Ink_FreeVolume(void)
{
	if (r_ink_state.texture)
		R_FreeTexture(r_ink_state.texture);
	if (r_ink_state.pixels)
		Mem_Free(r_ink_state.pixels);
	if (r_ink_state.scratch)
		Mem_Free(r_ink_state.scratch);
	memset(&r_ink_state, 0, sizeof(r_ink_state));
	r_ink_worldname[0] = 0;
}

void R_Ink_Clear(void)
{
	if (!r_ink_state.pixels)
		return;
	memset(r_ink_state.pixels, 0, r_ink_state.numtexels * 4);
	r_ink_state.uploadall = true;
	r_ink_state.dirty = true;
	r_ink_state.splatstotal = 0;
	r_ink_state.texelswritten = 0;
}

static void R_Ink_Clear_f(void)
{
	R_Ink_Clear();
	Con_Printf("ink cleared\n");
}

/// Debug lever: drop one glob at the current view origin.
///   r_ink_splat <radius> [r g b] [amount]
static void R_Ink_Splat_f(void)
{
	vec3_t color;
	float radius, amount;
	if (Cmd_Argc() < 2)
	{
		Con_Printf("r_ink_splat <radius> [r g b] [amount] : drop a glob at the view origin\n");
		return;
	}
	radius = atof(Cmd_Argv(1));
	VectorSet(color, 1.0f, 0.15f, 0.65f);
	if (Cmd_Argc() >= 5)
		VectorSet(color, atof(Cmd_Argv(2)), atof(Cmd_Argv(3)), atof(Cmd_Argv(4)));
	amount = Cmd_Argc() >= 6 ? atof(Cmd_Argv(5)) : 1.0f;
	R_Ink_Splat(r_refdef.view.origin, radius, color, amount);
}

/// Debug lever: flood the whole volume, so the material response can be inspected
/// without waiting for an artifact to traverse the level.
static void R_Ink_Fill_f(void)
{
	int i;
	vec3_t color;
	float amount;
	unsigned char cb[3];
	if (!r_ink_state.enabled)
	{
		Con_Printf("no ink volume\n");
		return;
	}
	VectorSet(color, 1.0f, 0.15f, 0.65f);
	if (Cmd_Argc() >= 4)
		VectorSet(color, atof(Cmd_Argv(1)), atof(Cmd_Argv(2)), atof(Cmd_Argv(3)));
	amount = Cmd_Argc() >= 5 ? atof(Cmd_Argv(4)) : 1.0f;
	for (i = 0; i < 3; i++)
		cb[i] = (unsigned char)bound(0, (int)(bound(0.0f, color[i], 1.0f) * amount * 255.0f + 0.5f), 255);
	for (i = 0; i < r_ink_state.numtexels; i++)
	{
		unsigned char *px = r_ink_state.pixels + i * 4;
		px[0] = cb[2];
		px[1] = cb[1];
		px[2] = cb[0];
		px[3] = (unsigned char)bound(0, (int)(amount * 255.0f + 0.5f), 255);
	}
	r_ink_state.uploadall = true;
	r_ink_state.dirty = true;
	Con_Printf("ink volume flooded\n");
}

static void R_Ink_Stats_f(void)
{
	if (!r_ink_state.enabled)
	{
		Con_Printf("ink volume: inactive\n");
		return;
	}
	Con_Printf("ink volume: %ix%ix%i (%.1f KB), %.1f units/voxel, %i splats, %i voxel writes, %i voxels uploaded\n",
		r_ink_state.resolution[0], r_ink_state.resolution[1], r_ink_state.resolution[2],
		r_ink_state.numtexels * 4 / 1024.0f,
		r_ink_state.spacing[0], r_ink_state.splatstotal,
		r_ink_state.texelswritten, r_ink_state.texelsuploaded);
}

// ---------------------------------------------------------------------------

/// Build (or rebuild) the volume to cover the current world model.
static void R_Ink_UpdateVolume(void)
{
	int i, res[3];
	dp_model_t *world = cl.worldmodel;
	float texelsize, m[16];
	vec3_t mins, maxs;

	if (!r_ink.integer || !world || !world->loaded || cls.state == ca_dedicated)
	{
		Con_DPrintf("ink: skip (cvar %i world %p loaded %i state %i)\n", r_ink.integer, (void *)world, world ? world->loaded : -1, (int)cls.state);
		if (r_ink_state.texture)
			R_Ink_FreeVolume();
		return;
	}

	// keep the existing volume if it is already the right one
	if (r_ink_state.enabled && !strcmp(r_ink_worldname, world->name))
		return;

	R_Ink_FreeVolume();

	VectorCopy(world->normalmins, mins);
	VectorCopy(world->normalmaxs, maxs);
	// a little slack so surfaces exactly on the bounds are not clamped
	for (i = 0; i < 3; i++)
	{
		mins[i] -= 32.0f;
		maxs[i] += 32.0f;
		if (maxs[i] - mins[i] < 1.0f)
			maxs[i] = mins[i] + 1.0f;
	}

	texelsize = max(1.0f, r_ink_texelsize.value);
	for (i = 0; i < 3; i++)
	{
		res[i] = (int)((maxs[i] - mins[i]) / texelsize + 0.5f);
		res[i] = bound(4, res[i], max(4, r_ink_maxresolution.integer));
	}

	r_ink_state.numtexels = res[0] * res[1] * res[2];
	r_ink_state.pixels = (unsigned char *)Mem_Alloc(r_ink_mempool, r_ink_state.numtexels * 4);
	memset(r_ink_state.pixels, 0, r_ink_state.numtexels * 4);
	VectorCopy(mins, r_ink_state.mins);
	VectorCopy(maxs, r_ink_state.maxs);
	VectorSubtract(maxs, mins, r_ink_state.size);
	for (i = 0; i < 3; i++)
	{
		r_ink_state.resolution[i] = res[i];
		r_ink_state.spacing[i] = r_ink_state.size[i] / res[i];
		r_ink_state.ispacing[i] = res[i] / r_ink_state.size[i];
	}

	// world space -> [0,1]^3, same shape as the bouncegrid matrix
	memset(m, 0, sizeof(m));
	m[0]  = 1.0f / r_ink_state.size[0];
	m[3]  = -r_ink_state.mins[0] * m[0];
	m[5]  = 1.0f / r_ink_state.size[1];
	m[7]  = -r_ink_state.mins[1] * m[5];
	m[10] = 1.0f / r_ink_state.size[2];
	m[11] = -r_ink_state.mins[2] * m[10];
	m[15] = 1.0f;
	Matrix4x4_FromArrayFloatD3D(&r_ink_state.matrix, m);

	r_ink_state.texture = R_LoadTexture3D(r_ink_texturepool, "inkvolume", res[0], res[1], res[2], r_ink_state.pixels, TEXTYPE_BGRA, TEXF_CLAMP | TEXF_ALPHA | TEXF_FORCELINEAR, 0, NULL);
	if (!r_ink_state.texture)
	{
		Con_DPrintf("ink: R_LoadTexture3D failed\n");
		Mem_Free(r_ink_state.pixels);
		memset(&r_ink_state, 0, sizeof(r_ink_state));
		return;
	}
	r_ink_state.enabled = true;
	strlcpy(r_ink_worldname, world->name, sizeof(r_ink_worldname));
	Con_DPrintf("ink volume %ix%ix%i (%.1f KB) at %.1f units/voxel for %s\n",
		res[0], res[1], res[2], r_ink_state.numtexels * 4 / 1024.0f, r_ink_state.spacing[0], world->name);
}

// ---------------------------------------------------------------------------

void R_Ink_Splat(const vec3_t origin, float radius, const vec3_t color, float amount)
{
	int i, x, y, z, lo[3], hi[3], stridey, stridez;
	float r2, ir2, cx, cy, cz, dx, dy, dz, w, sa, oa;
	float cf[3];

	if (!r_ink_state.enabled || radius <= 0.0f || amount <= 0.0f)
		return;

	for (i = 0; i < 3; i++)
	{
		lo[i] = (int)floor((origin[i] - radius - r_ink_state.mins[i]) * r_ink_state.ispacing[i]);
		hi[i] = (int)ceil ((origin[i] + radius - r_ink_state.mins[i]) * r_ink_state.ispacing[i]);
		lo[i] = bound(0, lo[i], r_ink_state.resolution[i]);
		hi[i] = bound(0, hi[i], r_ink_state.resolution[i]);
		if (lo[i] >= hi[i])
			return;
	}

	for (i = 0; i < 3; i++)
		cf[i] = bound(0.0f, color[i], 1.0f);
	r2 = radius * radius;
	ir2 = 1.0f / r2;
	amount = bound(0.0f, amount, 1.0f);
	stridey = r_ink_state.resolution[0];
	stridez = r_ink_state.resolution[0] * r_ink_state.resolution[1];

	for (z = lo[2]; z < hi[2]; z++)
	{
		cz = r_ink_state.mins[2] + (z + 0.5f) * r_ink_state.spacing[2];
		dz = cz - origin[2];
		dz *= dz;
		if (dz > r2)
			continue;
		for (y = lo[1]; y < hi[1]; y++)
		{
			cy = r_ink_state.mins[1] + (y + 0.5f) * r_ink_state.spacing[1];
			dy = cy - origin[1];
			dy = dy * dy + dz;
			if (dy > r2)
				continue;
			for (x = lo[0]; x < hi[0]; x++)
			{
				unsigned char *px;
				cx = r_ink_state.mins[0] + (x + 0.5f) * r_ink_state.spacing[0];
				dx = cx - origin[0];
				dx = dx * dx + dy;
				if (dx > r2)
					continue;
				// smooth radial falloff, squared so the core stays solid
				w = 1.0f - dx * ir2;
				w = w * w * amount;
				if (w <= (1.0f / 512.0f))
					continue;
				// source-over with premultiplied destination: a later team's paint
				// covers an earlier one, and repeated passes saturate toward opaque
				px = r_ink_state.pixels + (z * stridez + y * stridey + x) * 4;
				oa = px[3] * (1.0f / 255.0f);
				// BGRA in memory; every term here is 0..1 and is scaled to bytes once
				sa = w;
				px[0] = (unsigned char)bound(0, (int)((cf[2] * sa + px[0] * (1.0f / 255.0f) * (1.0f - sa)) * 255.0f + 0.5f), 255);
				px[1] = (unsigned char)bound(0, (int)((cf[1] * sa + px[1] * (1.0f / 255.0f) * (1.0f - sa)) * 255.0f + 0.5f), 255);
				px[2] = (unsigned char)bound(0, (int)((cf[0] * sa + px[2] * (1.0f / 255.0f) * (1.0f - sa)) * 255.0f + 0.5f), 255);
				px[3] = (unsigned char)bound(0, (int)((sa + oa * (1.0f - sa)) * 255.0f + 0.5f), 255);
				r_ink_state.texelswritten++;
			}
		}
	}

	// grow the dirty box
	if (!r_ink_state.dirty)
	{
		for (i = 0; i < 3; i++)
		{
			r_ink_state.dirtymins[i] = lo[i];
			r_ink_state.dirtymaxs[i] = hi[i];
		}
		r_ink_state.dirty = true;
	}
	else
	{
		for (i = 0; i < 3; i++)
		{
			if (lo[i] < r_ink_state.dirtymins[i]) r_ink_state.dirtymins[i] = lo[i];
			if (hi[i] > r_ink_state.dirtymaxs[i]) r_ink_state.dirtymaxs[i] = hi[i];
		}
	}
	r_ink_state.splatstotal++;
}

// ---------------------------------------------------------------------------

void R_Ink_GlobalTint(vec3_t out_color, float *out_coverage)
{
	VectorCopy(r_ink_state.globaltint, out_color);
	*out_coverage = r_ink_state.globalcoverage;
}

/// Sampling every voxel every frame would be pointless work, so this walks a
/// strided subset (about 4096 voxels), which is far more than a sky tint needs.
static void R_Ink_RecomputeGlobalTint(void)
{
	int i, step, n = 0;
	float acc[3] = {0, 0, 0}, aacc = 0;
	VectorClear(r_ink_state.globaltint);
	r_ink_state.globalcoverage = 0;
	if (!r_ink_state.enabled)
		return;
	step = max(1, r_ink_state.numtexels / 4096);
	for (i = 0; i < r_ink_state.numtexels; i += step)
	{
		const unsigned char *px = r_ink_state.pixels + i * 4;
		float a = px[3] * (1.0f / 255.0f);
		acc[0] += px[2] * (1.0f / 255.0f);
		acc[1] += px[1] * (1.0f / 255.0f);
		acc[2] += px[0] * (1.0f / 255.0f);
		aacc += a;
		n++;
	}
	if (n < 1 || aacc <= 0.0f)
		return;
	// premultiplied, so dividing the colour sum by the alpha sum gives the mean tint
	r_ink_state.globaltint[0] = acc[0] / aacc;
	r_ink_state.globaltint[1] = acc[1] / aacc;
	r_ink_state.globaltint[2] = acc[2] / aacc;
	r_ink_state.globalcoverage = aacc / n;
}

// ---------------------------------------------------------------------------

void R_Ink_Frame(void)
{
	int x, y, z, w, h, d, stridey, stridez, need;
	unsigned char *out;

	R_Ink_UpdateVolume();
	if (!r_ink_state.enabled || !r_ink_state.dirty)
		return;

	R_Ink_RecomputeGlobalTint();
	if (r_ink_state.uploadall)
	{
		R_UpdateTexture(r_ink_state.texture, r_ink_state.pixels, 0, 0, 0, r_ink_state.resolution[0], r_ink_state.resolution[1], r_ink_state.resolution[2]);
		r_ink_state.texelsuploaded += r_ink_state.numtexels;
		r_ink_state.uploadall = false;
		r_ink_state.dirty = false;
		return;
	}

	w = r_ink_state.dirtymaxs[0] - r_ink_state.dirtymins[0];
	h = r_ink_state.dirtymaxs[1] - r_ink_state.dirtymins[1];
	d = r_ink_state.dirtymaxs[2] - r_ink_state.dirtymins[2];
	if (w < 1 || h < 1 || d < 1)
	{
		r_ink_state.dirty = false;
		return;
	}

	// pack the dirty box contiguously; the partial-upload path takes a tight block
	need = w * h * d * 4;
	if (need > r_ink_state.scratchsize)
	{
		if (r_ink_state.scratch)
			Mem_Free(r_ink_state.scratch);
		r_ink_state.scratch = (unsigned char *)Mem_Alloc(r_ink_mempool, need);
		r_ink_state.scratchsize = need;
	}
	stridey = r_ink_state.resolution[0];
	stridez = r_ink_state.resolution[0] * r_ink_state.resolution[1];
	out = r_ink_state.scratch;
	for (z = 0; z < d; z++)
		for (y = 0; y < h; y++, out += w * 4)
			memcpy(out, r_ink_state.pixels + (((z + r_ink_state.dirtymins[2]) * stridez + (y + r_ink_state.dirtymins[1]) * stridey + r_ink_state.dirtymins[0]) * 4), w * 4);
	x = 0; (void)x;
	R_UpdateTexture(r_ink_state.texture, r_ink_state.scratch, r_ink_state.dirtymins[0], r_ink_state.dirtymins[1], r_ink_state.dirtymins[2], w, h, d);
	r_ink_state.texelsuploaded += w * h * d;
	r_ink_state.dirty = false;
}

// ---------------------------------------------------------------------------

static void R_Ink_Start(void)
{
	r_ink_texturepool = R_AllocTexturePool();
	memset(&r_ink_state, 0, sizeof(r_ink_state));
	r_ink_worldname[0] = 0;
}

static void R_Ink_Stop(void)
{
	if (r_ink_state.pixels)
		Mem_Free(r_ink_state.pixels);
	if (r_ink_state.scratch)
		Mem_Free(r_ink_state.scratch);
	memset(&r_ink_state, 0, sizeof(r_ink_state));
	r_ink_worldname[0] = 0;
	R_FreeTexturePool(&r_ink_texturepool);
}

static void R_Ink_Newmap(void)
{
	// the volume itself is rebuilt lazily by R_Ink_UpdateVolume when the world
	// model name changes; nothing to do here beyond dropping the old one
	if (r_ink_state.texture)
		R_Ink_FreeVolume();
}

void R_Ink_Init(void)
{
	r_ink_mempool = Mem_AllocPool("ink volume", 0, NULL);
	Cvar_RegisterVariable(&r_ink);
	Cvar_RegisterVariable(&r_ink_texelsize);
	Cvar_RegisterVariable(&r_ink_maxresolution);
	Cvar_RegisterVariable(&r_ink_normalbias);
	Cvar_RegisterVariable(&r_ink_roughness);
	Cvar_RegisterVariable(&r_ink_tint);
	Cvar_RegisterVariable(&r_ink_f0);
	Cvar_RegisterVariable(&r_ink_intensity);
	Cvar_RegisterVariable(&r_ink_noisescale);
	Cvar_RegisterVariable(&r_ink_noiseedge);
	Cvar_RegisterVariable(&r_ink_skytint);
	Cmd_AddCommand("r_ink_clear", R_Ink_Clear_f, "erase all accumulated world ink");
	Cmd_AddCommand("r_ink_stats", R_Ink_Stats_f, "report ink volume size and update cost");
	Cmd_AddCommand("r_ink_splat", R_Ink_Splat_f, "drop one ink glob at the view origin: r_ink_splat <radius> [r g b] [amount]");
	Cmd_AddCommand("r_ink_fill", R_Ink_Fill_f, "flood the whole ink volume: r_ink_fill [r g b] [amount]");
	R_RegisterModule("R_Ink", R_Ink_Start, R_Ink_Stop, R_Ink_Newmap, NULL, NULL);
}
