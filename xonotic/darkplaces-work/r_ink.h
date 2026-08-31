/*
r_ink.h - accumulated world ink volume.

A single RGBA8 3D texture covering the world bounds.  Globs dropped by the
traversing artifact (and by payload carts) splat into it; the surface shader
samples it once per fragment and uses the coverage to drag roughness down,
F0 up and albedo toward the ink tint - the "rubberier, wetter, off-colour"
drift, plus team-coloured cart territory, in one channel set.

The volume is deliberately coarse.  High-frequency splatter edges come from
thresholding the coverage against a procedural field in the shader, not from
resolution, which is what keeps both the memory and the per-splat update cost
negligible.  Only the axis-aligned box a splat actually touched is re-uploaded.
*/

#ifndef R_INK_H
#define R_INK_H

typedef struct r_ink_state_s
{
	qboolean			enabled;		///< a volume exists and the shader should sample it
	rtexture_t			*texture;
	matrix4x4_t			matrix;			///< world space -> [0,1]^3 volume space
	int					resolution[3];
	int					numtexels;
	vec3_t				mins, maxs, size;
	float				spacing[3];		///< world units per texel
	float				ispacing[3];
	unsigned char		*pixels;		///< BGRA, premultiplied tint in RGB, coverage in A
	unsigned char		*scratch;		///< packing buffer for the dirty sub-box upload
	int					scratchsize;
	int					dirtymins[3];	///< inclusive
	int					dirtymaxs[3];	///< exclusive
	qboolean			dirty;
	qboolean			uploadall;
	vec3_t				globaltint;		///< mean ink colour, recomputed only when the volume changes
	float				globalcoverage;
	// stats, reported by the r_ink command
	int					splatstotal;
	int					texelswritten;
	int					texelsuploaded;
}
r_ink_state_t;

extern r_ink_state_t r_ink_state;

extern cvar_t r_ink;
extern cvar_t r_ink_texelsize;
extern cvar_t r_ink_maxresolution;
extern cvar_t r_ink_normalbias;
extern cvar_t r_ink_roughness;
extern cvar_t r_ink_tint;
extern cvar_t r_ink_f0;
extern cvar_t r_ink_intensity;
extern cvar_t r_ink_noisescale;
extern cvar_t r_ink_noiseedge;
extern cvar_t r_ink_skytint;

void R_Ink_Init(void);
/// (re)build the volume for the current world; safe to call every frame
void R_Ink_Frame(void);
/// drop one glob.  amount is 0..1 opacity at the centre, colour is linear 0..1 rgb
void R_Ink_Splat(const vec3_t origin, float radius, const vec3_t color, float amount);
/// wipe all accumulated ink (match reset)
void R_Ink_Clear(void);
/// mean ink colour and coverage over the whole volume, for tinting the sky.
/// Cached; recomputed by R_Ink_Frame only on frames where the volume changed.
void R_Ink_GlobalTint(vec3_t out_color, float *out_coverage);

#endif
