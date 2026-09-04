

#ifndef R_INK_H
#define R_INK_H

typedef struct r_ink_state_s
{
	qboolean			enabled;
	rtexture_t			*texture;
	matrix4x4_t			matrix;
	int					resolution[3];
	int					numtexels;
	vec3_t				mins, maxs, size;
	float				spacing[3];
	float				ispacing[3];
	unsigned char		*pixels;
	unsigned char		*scratch;
	int					scratchsize;
	int					dirtymins[3];
	int					dirtymaxs[3];
	qboolean			dirty;
	qboolean			uploadall;
	vec3_t				globaltint;
	float				globalcoverage;

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

void R_Ink_Frame(void);

void R_Ink_Splat(const vec3_t origin, float radius, const vec3_t color, float amount);

void R_Ink_Clear(void);

void R_Ink_GlobalTint(vec3_t out_color, float *out_coverage);

#endif
