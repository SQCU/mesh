

#if defined(USESKELETAL) || defined(USEOCCLUDE)
#  ifdef GL_ARB_uniform_buffer_object
#    extension GL_ARB_uniform_buffer_object : enable
#  endif
#endif

#ifdef USESHADOWMAP2D
# ifdef GL_EXT_gpu_shader4
#   extension GL_EXT_gpu_shader4 : enable
# endif
# ifdef GL_ARB_texture_gather
#   extension GL_ARB_texture_gather : enable
# else
#   ifdef GL_AMD_texture_texture4
#     extension GL_AMD_texture_texture4 : enable
#   endif
# endif
#endif

#ifdef USECELSHADING
# define SHADEDIFFUSE myhalf diffuse = cast_myhalf(min(max(float(dot(surfacenormal, lightnormal)) * 2.0, 0.0), 1.0));
# ifdef USEEXACTSPECULARMATH
#  define SHADESPECULAR(specpow) myhalf specular = pow(cast_myhalf(max(float(dot(reflect(lightnormal, surfacenormal), eyenormal))*-1.0, 0.0)), 1.0 + specpow);specular = max(0.0, specular * 10.0 - 9.0);
# else
#  define SHADESPECULAR(specpow) myhalf3 specularnormal = normalize(lightnormal + eyenormal);myhalf specular = pow(cast_myhalf(max(float(dot(surfacenormal, specularnormal)), 0.0)), 1.0 + specpow);specular = max(0.0, specular * 10.0 - 9.0);
# endif
#else
# define SHADEDIFFUSE myhalf diffuse = cast_myhalf(max(float(dot(surfacenormal, lightnormal)), 0.0));
# ifdef USEEXACTSPECULARMATH
#  define SHADESPECULAR(specpow) myhalf specular = pow(cast_myhalf(max(float(dot(reflect(lightnormal, surfacenormal), eyenormal))*-1.0, 0.0)), 1.0 + specpow);
# else
#  define SHADESPECULAR(specpow) myhalf3 specularnormal = normalize(lightnormal + eyenormal);myhalf specular = pow(cast_myhalf(max(float(dot(surfacenormal, specularnormal)), 0.0)), 1.0 + specpow);
# endif
#endif

#if (defined(GLSL120) || defined(GLSL130) || defined(GLSL140) || defined(GLES)) && defined(VERTEX_SHADER)
invariant gl_Position;
# endif
#if defined(GLSL130) || defined(GLSL140)
precision highp float;
# ifdef VERTEX_SHADER
#  define dp_varying out
#  define dp_attribute in
# endif
# ifdef FRAGMENT_SHADER
out vec4 dp_FragColor;
#  define dp_varying in
#  define dp_attribute in
# endif
# define dp_offsetmapping_dFdx dFdx
# define dp_offsetmapping_dFdy dFdy
# define dp_textureGrad textureGrad
# define dp_textureOffset(a,b,c,d) textureOffset(a,b,ivec2(c,d))
# define dp_texture2D texture
# define dp_texture3D texture
# define dp_textureCube texture
# define dp_shadow2D(a,b) float(texture(a,b))
#else
# ifdef FRAGMENT_SHADER
#  define dp_FragColor gl_FragColor
# endif
# define dp_varying varying
# define dp_attribute attribute
# define dp_offsetmapping_dFdx(a) vec2(0.0, 0.0)
# define dp_offsetmapping_dFdy(a) vec2(0.0, 0.0)
# define dp_textureGrad(a,b,c,d) texture2D(a,b)
# define dp_textureOffset(a,b,c,d) texture2DOffset(a,b,ivec2(c,d))
# define dp_texture2D texture2D
# define dp_texture3D texture3D
# define dp_textureCube textureCube
# define dp_shadow2D(a,b) float(shadow2D(a,b))
#endif

#ifndef GL_ES
#define lowp
#define mediump
#define highp
#endif

#ifdef USEDEPTHRGB

#define decodedepthmacro(d) dot((d).rgb, vec3(1.0, 255.0 / 65536.0, 255.0 / 16777215.0))
#define encodedepthmacro(d) (vec4(d, d*256.0, d*65536.0, 0.0) - floor(vec4(d, d*256.0, d*65536.0, 0.0)))
#endif

#ifdef VERTEX_SHADER
dp_attribute vec4 Attrib_Position;
dp_attribute vec4 Attrib_Color;
dp_attribute vec4 Attrib_TexCoord0;
dp_attribute vec3 Attrib_TexCoord1;
dp_attribute vec3 Attrib_TexCoord2;
dp_attribute vec3 Attrib_TexCoord3;
dp_attribute vec4 Attrib_TexCoord4;
#ifdef USESKELETAL

uniform Skeletal_Transform12_UniformBlock
{
	vec4 Skeletal_Transform12[768];
};
dp_attribute vec4 Attrib_SkeletalIndex;
dp_attribute vec4 Attrib_SkeletalWeight;
#endif
#endif
dp_varying mediump vec4 VertexColor;

#if defined(USEFOGINSIDE) || defined(USEFOGOUTSIDE) || defined(USEFOGHEIGHTTEXTURE)
# define USEFOG
#endif
#if defined(MODE_LIGHTMAP) || defined(MODE_LIGHTDIRECTIONMAP_MODELSPACE) || defined(MODE_LIGHTDIRECTIONMAP_TANGENTSPACE) || defined(MODE_LIGHTDIRECTIONMAP_FORCED_LIGHTMAP)
# define USELIGHTMAP
#endif
#if defined(USESPECULAR) || defined(USEOFFSETMAPPING) || defined(USEREFLECTCUBE) || defined(MODE_FAKELIGHT) || defined(USEFOG)
# define USEEYEVECTOR
#endif

# define myhalf mediump float
# define myhalf2 mediump vec2
# define myhalf3 mediump vec3
# define myhalf4 mediump vec4
# define cast_myhalf float
# define cast_myhalf2 vec2
# define cast_myhalf3 vec3
# define cast_myhalf4 vec4

#ifdef VERTEX_SHADER
uniform highp mat4 ModelViewProjectionMatrix;
#endif

#ifdef VERTEX_SHADER
#ifdef USETRIPPY

uniform highp float ClientTime;
vec4 TrippyVertex(vec4 position)
{
	float worldTime = ClientTime;

	worldTime *= 10.0;
	position *= 0.125;

	float distanceSquared = (position.x * position.x + position.z * position.z);
	position.y += 5.0*sin(distanceSquared*sin(worldTime/143.0)/1000.0);
	float y = position.y;
	float x = position.x;
	float om = sin(distanceSquared*sin(worldTime/256.0)/5000.0) * sin(worldTime/200.0);
	position.y = x*sin(om)+y*cos(om);
	position.x = x*cos(om)-y*sin(om);
	return position;
}
#endif
#endif

#ifdef MODE_DEPTH_OR_SHADOW
dp_varying highp float Depth;
#ifdef VERTEX_SHADER
void main(void)
{
#ifdef USESKELETAL
	ivec4 si0 = ivec4(Attrib_SkeletalIndex * 3.0);
	ivec4 si1 = si0 + ivec4(1, 1, 1, 1);
	ivec4 si2 = si0 + ivec4(2, 2, 2, 2);
	vec4 sw = Attrib_SkeletalWeight;
	vec4 SkeletalMatrix1 = Skeletal_Transform12[si0.x] * sw.x + Skeletal_Transform12[si0.y] * sw.y + Skeletal_Transform12[si0.z] * sw.z + Skeletal_Transform12[si0.w] * sw.w;
	vec4 SkeletalMatrix2 = Skeletal_Transform12[si1.x] * sw.x + Skeletal_Transform12[si1.y] * sw.y + Skeletal_Transform12[si1.z] * sw.z + Skeletal_Transform12[si1.w] * sw.w;
	vec4 SkeletalMatrix3 = Skeletal_Transform12[si2.x] * sw.x + Skeletal_Transform12[si2.y] * sw.y + Skeletal_Transform12[si2.z] * sw.z + Skeletal_Transform12[si2.w] * sw.w;
	mat4 SkeletalMatrix = mat4(SkeletalMatrix1, SkeletalMatrix2, SkeletalMatrix3, vec4(0.0, 0.0, 0.0, 1.0));
	vec4 SkeletalVertex = Attrib_Position * SkeletalMatrix;
#define Attrib_Position SkeletalVertex
#endif
	gl_Position = ModelViewProjectionMatrix * Attrib_Position;
#ifdef USETRIPPY
	gl_Position = TrippyVertex(gl_Position);
#endif
	Depth = gl_Position.z;
}
#endif

#ifdef FRAGMENT_SHADER
void main(void)
{
#ifdef USEDEPTHRGB
	dp_FragColor = encodedepthmacro(Depth);
#else
	dp_FragColor = vec4(1.0,1.0,1.0,1.0);
#endif
}
#endif
#else

#ifdef MODE_POSTPROCESS
dp_varying mediump vec2 TexCoord1;
dp_varying mediump vec2 TexCoord2;

#ifdef VERTEX_SHADER
void main(void)
{
	gl_Position = ModelViewProjectionMatrix * Attrib_Position;
	TexCoord1 = Attrib_TexCoord0.xy;
#ifdef USEBLOOM
	TexCoord2 = Attrib_TexCoord4.xy;
#endif
}
#endif

#ifdef FRAGMENT_SHADER
uniform sampler2D Texture_First;
#ifdef USEBLOOM
uniform sampler2D Texture_Second;
uniform mediump vec4 BloomColorSubtract;
#endif
#ifdef USEGAMMARAMPS
uniform sampler2D Texture_GammaRamps;
#endif
#ifdef USESATURATION
uniform mediump float Saturation;
#endif
#ifdef USEVIEWTINT
uniform mediump vec4 ViewTintColor;
#endif

uniform mediump vec4 UserVec1;
uniform mediump vec4 UserVec2;

uniform mediump vec2 PixelSize;

#ifdef USEFXAA

vec4 fxaa(vec4 inColor, float maxspan)
{
	vec4 ret = inColor;
	float mulreduct = 1.0/maxspan;
	float minreduct = (1.0 / 128.0);

	vec3 NW = dp_texture2D(Texture_First, TexCoord1 + (vec2(-1.0, -1.0) * PixelSize)).xyz;
	vec3 NE = dp_texture2D(Texture_First, TexCoord1 + (vec2(+1.0, -1.0) * PixelSize)).xyz;
	vec3 SW = dp_texture2D(Texture_First, TexCoord1 + (vec2(-1.0, +1.0) * PixelSize)).xyz;
	vec3 SE = dp_texture2D(Texture_First, TexCoord1 + (vec2(+1.0, +1.0) * PixelSize)).xyz;
	vec3 M = dp_texture2D(Texture_First, TexCoord1).xyz;

	vec3 luma = vec3(0.299, 0.587, 0.114);
	float lNW = dot(NW, luma);
	float lNE = dot(NE, luma);
	float lSW = dot(SW, luma);
	float lSE = dot(SE, luma);
	float lM = dot(M, luma);
	float lMin = min(lM, min(min(lNW, lNE), min(lSW, lSE)));
	float lMax = max(lM, max(max(lNW, lNE), max(lSW, lSE)));

	vec2 dir = vec2(-((lNW + lNE) - (lSW + lSE)), ((lNW + lSW) - (lNE + lSE)));
	float rcp = 1.0/(min(abs(dir.x), abs(dir.y)) + max((lNW + lNE + lSW + lSE) * (0.25 * mulreduct), minreduct));

	dir = min(vec2(maxspan, maxspan), max(vec2(-maxspan, -maxspan), dir * rcp)) * PixelSize;

	vec3 rA = (1.0/2.0) * (
		dp_texture2D(Texture_First, TexCoord1 + dir * (1.0/3.0 - 0.5)).xyz +
		dp_texture2D(Texture_First, TexCoord1 + dir * (2.0/3.0 - 0.5)).xyz);
	vec3 rB = rA * (1.0/2.0) + (1.0/4.0) * (
		dp_texture2D(Texture_First, TexCoord1 + dir * (0.0/3.0 - 0.5)).xyz +
		dp_texture2D(Texture_First, TexCoord1 + dir * (3.0/3.0 - 0.5)).xyz);
	float lB = dot(rB, luma);

	ret.xyz = ((lB < lMin) || (lB > lMax)) ? rA : rB;
	ret.a = 1.0;
	return ret;
}
#endif

void main(void)
{
	dp_FragColor = dp_texture2D(Texture_First, TexCoord1);

#ifdef USEFXAA
	dp_FragColor = fxaa(dp_FragColor, 8.0);
#endif

#ifdef USEPOSTPROCESSING

#if defined(USERVEC1) || defined(USERVEC2)
	float sobel = 1.0;

	vec2 px = PixelSize;
	vec3 x1 = dp_texture2D(Texture_First, TexCoord1 + vec2(-px.x, px.y)).rgb;
	vec3 x2 = dp_texture2D(Texture_First, TexCoord1 + vec2(-px.x,  0.0)).rgb;
	vec3 x3 = dp_texture2D(Texture_First, TexCoord1 + vec2(-px.x,-px.y)).rgb;
	vec3 x4 = dp_texture2D(Texture_First, TexCoord1 + vec2( px.x, px.y)).rgb;
	vec3 x5 = dp_texture2D(Texture_First, TexCoord1 + vec2( px.x,  0.0)).rgb;
	vec3 x6 = dp_texture2D(Texture_First, TexCoord1 + vec2( px.x,-px.y)).rgb;
	vec3 y1 = dp_texture2D(Texture_First, TexCoord1 + vec2( px.x,-px.y)).rgb;
	vec3 y2 = dp_texture2D(Texture_First, TexCoord1 + vec2(  0.0,-px.y)).rgb;
	vec3 y3 = dp_texture2D(Texture_First, TexCoord1 + vec2(-px.x,-px.y)).rgb;
	vec3 y4 = dp_texture2D(Texture_First, TexCoord1 + vec2( px.x, px.y)).rgb;
	vec3 y5 = dp_texture2D(Texture_First, TexCoord1 + vec2(  0.0, px.y)).rgb;
	vec3 y6 = dp_texture2D(Texture_First, TexCoord1 + vec2(-px.x, px.y)).rgb;
	float px1 = -1.0 * dot(vec3(0.3, 0.59, 0.11), x1);
	float px2 = -2.0 * dot(vec3(0.3, 0.59, 0.11), x2);
	float px3 = -1.0 * dot(vec3(0.3, 0.59, 0.11), x3);
	float px4 =  1.0 * dot(vec3(0.3, 0.59, 0.11), x4);
	float px5 =  2.0 * dot(vec3(0.3, 0.59, 0.11), x5);
	float px6 =  1.0 * dot(vec3(0.3, 0.59, 0.11), x6);
	float py1 = -1.0 * dot(vec3(0.3, 0.59, 0.11), y1);
	float py2 = -2.0 * dot(vec3(0.3, 0.59, 0.11), y2);
	float py3 = -1.0 * dot(vec3(0.3, 0.59, 0.11), y3);
	float py4 =  1.0 * dot(vec3(0.3, 0.59, 0.11), y4);
	float py5 =  2.0 * dot(vec3(0.3, 0.59, 0.11), y5);
	float py6 =  1.0 * dot(vec3(0.3, 0.59, 0.11), y6);
	sobel = 0.25 * abs(px1 + px2 + px3 + px4 + px5 + px6) + 0.25 * abs(py1 + py2 + py3 + py4 + py5 + py6);
	dp_FragColor += dp_texture2D(Texture_First, TexCoord1 + PixelSize*UserVec1.x*vec2(-0.987688, -0.156434)) * UserVec1.y;
	dp_FragColor += dp_texture2D(Texture_First, TexCoord1 + PixelSize*UserVec1.x*vec2(-0.156434, -0.891007)) * UserVec1.y;
	dp_FragColor += dp_texture2D(Texture_First, TexCoord1 + PixelSize*UserVec1.x*vec2( 0.891007, -0.453990)) * UserVec1.y;
	dp_FragColor += dp_texture2D(Texture_First, TexCoord1 + PixelSize*UserVec1.x*vec2( 0.707107,  0.707107)) * UserVec1.y;
	dp_FragColor += dp_texture2D(Texture_First, TexCoord1 + PixelSize*UserVec1.x*vec2(-0.453990,  0.891007)) * UserVec1.y;
	dp_FragColor /= (1.0 + 5.0 * UserVec1.y);
	dp_FragColor.rgb = dp_FragColor.rgb * (1.0 + UserVec2.x) + vec3(max(0.0, sobel - UserVec2.z))*UserVec2.y;
#endif
#endif

#ifdef USEBLOOM
	dp_FragColor += max(vec4(0,0,0,0), dp_texture2D(Texture_Second, TexCoord2) - BloomColorSubtract);
#endif

#ifdef USEVIEWTINT
	dp_FragColor = mix(dp_FragColor, ViewTintColor, ViewTintColor.a);
#endif

#ifdef USESATURATION

	float y = dot(dp_FragColor.rgb, vec3(0.299, 0.587, 0.114));

	#ifdef SATURATION_REDCOMPENSATE
		float rboost = max(0.0, (dp_FragColor.r - max(dp_FragColor.g, dp_FragColor.b))*(1.0 - Saturation));
		dp_FragColor.rgb = mix(vec3(y), dp_FragColor.rgb, Saturation);
		dp_FragColor.r += rboost;
	#else

		dp_FragColor.rgb = mix(vec3(y), dp_FragColor.rgb, Saturation);
	#endif
#endif

#ifdef USEGAMMARAMPS
	dp_FragColor.r = dp_texture2D(Texture_GammaRamps, vec2(dp_FragColor.r, 0)).r;
	dp_FragColor.g = dp_texture2D(Texture_GammaRamps, vec2(dp_FragColor.g, 0)).g;
	dp_FragColor.b = dp_texture2D(Texture_GammaRamps, vec2(dp_FragColor.b, 0)).b;
#endif
}
#endif
#else

#ifdef MODE_GENERIC
#ifdef USEDIFFUSE
dp_varying mediump vec2 TexCoord1;
#endif
#ifdef USESPECULAR
dp_varying mediump vec2 TexCoord2;
#endif
uniform myhalf Alpha;
#ifdef VERTEX_SHADER
void main(void)
{
#ifdef USESKELETAL
	ivec4 si0 = ivec4(Attrib_SkeletalIndex * 3.0);
	ivec4 si1 = si0 + ivec4(1, 1, 1, 1);
	ivec4 si2 = si0 + ivec4(2, 2, 2, 2);
	vec4 sw = Attrib_SkeletalWeight;
	vec4 SkeletalMatrix1 = Skeletal_Transform12[si0.x] * sw.x + Skeletal_Transform12[si0.y] * sw.y + Skeletal_Transform12[si0.z] * sw.z + Skeletal_Transform12[si0.w] * sw.w;
	vec4 SkeletalMatrix2 = Skeletal_Transform12[si1.x] * sw.x + Skeletal_Transform12[si1.y] * sw.y + Skeletal_Transform12[si1.z] * sw.z + Skeletal_Transform12[si1.w] * sw.w;
	vec4 SkeletalMatrix3 = Skeletal_Transform12[si2.x] * sw.x + Skeletal_Transform12[si2.y] * sw.y + Skeletal_Transform12[si2.z] * sw.z + Skeletal_Transform12[si2.w] * sw.w;
	mat4 SkeletalMatrix = mat4(SkeletalMatrix1, SkeletalMatrix2, SkeletalMatrix3, vec4(0.0, 0.0, 0.0, 1.0));
	vec4 SkeletalVertex = Attrib_Position * SkeletalMatrix;
#define Attrib_Position SkeletalVertex
#endif
	VertexColor = Attrib_Color;
#ifdef USEDIFFUSE
	TexCoord1 = Attrib_TexCoord0.xy;
#endif
#ifdef USESPECULAR
	TexCoord2 = Attrib_TexCoord1.xy;
#endif
	gl_Position = ModelViewProjectionMatrix * Attrib_Position;
#ifdef USETRIPPY
	gl_Position = TrippyVertex(gl_Position);
#endif
}
#endif

#ifdef FRAGMENT_SHADER
#ifdef USEDIFFUSE
uniform sampler2D Texture_First;
#endif
#ifdef USESPECULAR
uniform sampler2D Texture_Second;
#endif
#ifdef USEGAMMARAMPS
uniform sampler2D Texture_GammaRamps;
#endif

void main(void)
{
#ifdef USEVIEWTINT
	dp_FragColor = VertexColor;
#else
	dp_FragColor = vec4(1.0, 1.0, 1.0, 1.0);
#endif
#ifdef USEDIFFUSE
# ifdef USEREFLECTCUBE

	dp_FragColor.rgb *= dp_texture2D(Texture_First, TexCoord1).rgb;
# else
	dp_FragColor *= dp_texture2D(Texture_First, TexCoord1);
# endif
#endif

#ifdef USESPECULAR
	vec4 tex2 = dp_texture2D(Texture_Second, TexCoord2);
# ifdef USECOLORMAPPING
	dp_FragColor *= tex2;
# endif
# ifdef USEGLOW
	dp_FragColor += tex2;
# endif
# ifdef USEVERTEXTEXTUREBLEND
	dp_FragColor = mix(dp_FragColor, tex2, tex2.a);
# endif
#endif
#ifdef USEGAMMARAMPS
	dp_FragColor.r = dp_texture2D(Texture_GammaRamps, vec2(dp_FragColor.r, 0)).r;
	dp_FragColor.g = dp_texture2D(Texture_GammaRamps, vec2(dp_FragColor.g, 0)).g;
	dp_FragColor.b = dp_texture2D(Texture_GammaRamps, vec2(dp_FragColor.b, 0)).b;
#endif
#ifdef USEALPHAKILL
	dp_FragColor.a *= Alpha;
#endif
}
#endif
#else

#ifdef MODE_BLOOMBLUR
dp_varying mediump vec2 TexCoord;
#ifdef VERTEX_SHADER
void main(void)
{
	VertexColor = Attrib_Color;
	TexCoord = Attrib_TexCoord0.xy;
	gl_Position = ModelViewProjectionMatrix * Attrib_Position;
}
#endif

#ifdef FRAGMENT_SHADER
uniform sampler2D Texture_First;
uniform mediump vec4 BloomBlur_Parameters;

void main(void)
{
	int i;
	vec2 tc = TexCoord;
	vec3 color = dp_texture2D(Texture_First, tc).rgb;
	tc += BloomBlur_Parameters.xy;
	for (i = 1;i < SAMPLES;i++)
	{
		color += dp_texture2D(Texture_First, tc).rgb;
		tc += BloomBlur_Parameters.xy;
	}
	dp_FragColor = vec4(color * BloomBlur_Parameters.z + vec3(BloomBlur_Parameters.w), 1);
}
#endif
#else
#ifdef MODE_REFRACTION
dp_varying mediump vec2 TexCoord;
dp_varying highp vec4 ModelViewProjectionPosition;
uniform highp mat4 TexMatrix;
#ifdef VERTEX_SHADER

void main(void)
{
#ifdef USESKELETAL
	ivec4 si0 = ivec4(Attrib_SkeletalIndex * 3.0);
	ivec4 si1 = si0 + ivec4(1, 1, 1, 1);
	ivec4 si2 = si0 + ivec4(2, 2, 2, 2);
	vec4 sw = Attrib_SkeletalWeight;
	vec4 SkeletalMatrix1 = Skeletal_Transform12[si0.x] * sw.x + Skeletal_Transform12[si0.y] * sw.y + Skeletal_Transform12[si0.z] * sw.z + Skeletal_Transform12[si0.w] * sw.w;
	vec4 SkeletalMatrix2 = Skeletal_Transform12[si1.x] * sw.x + Skeletal_Transform12[si1.y] * sw.y + Skeletal_Transform12[si1.z] * sw.z + Skeletal_Transform12[si1.w] * sw.w;
	vec4 SkeletalMatrix3 = Skeletal_Transform12[si2.x] * sw.x + Skeletal_Transform12[si2.y] * sw.y + Skeletal_Transform12[si2.z] * sw.z + Skeletal_Transform12[si2.w] * sw.w;
	mat4 SkeletalMatrix = mat4(SkeletalMatrix1, SkeletalMatrix2, SkeletalMatrix3, vec4(0.0, 0.0, 0.0, 1.0));
	vec4 SkeletalVertex = Attrib_Position * SkeletalMatrix;
#define Attrib_Position SkeletalVertex
#endif
#ifdef USEALPHAGENVERTEX
	VertexColor = Attrib_Color;
#endif
	TexCoord = vec2(TexMatrix * Attrib_TexCoord0);
	gl_Position = ModelViewProjectionMatrix * Attrib_Position;
	ModelViewProjectionPosition = gl_Position;
#ifdef USETRIPPY
	gl_Position = TrippyVertex(gl_Position);
#endif
}
#endif

#ifdef FRAGMENT_SHADER
uniform sampler2D Texture_Normal;
uniform sampler2D Texture_Refraction;

uniform mediump vec4 DistortScaleRefractReflect;
uniform mediump vec4 ScreenScaleRefractReflect;
uniform mediump vec4 ScreenCenterRefractReflect;
uniform mediump vec4 RefractColor;
uniform mediump vec4 ReflectColor;
uniform highp float ClientTime;
#ifdef USENORMALMAPSCROLLBLEND
uniform highp vec2 NormalmapScrollBlend;
#endif

void main(void)
{
	vec2 ScreenScaleRefractReflectIW = ScreenScaleRefractReflect.xy * (1.0 / ModelViewProjectionPosition.w);

	vec2 SafeScreenTexCoord = ModelViewProjectionPosition.xy * ScreenScaleRefractReflectIW + ScreenCenterRefractReflect.xy;
#ifdef USEALPHAGENVERTEX
	vec2 distort = DistortScaleRefractReflect.xy * VertexColor.a;
	vec4 refractcolor = mix(RefractColor, vec4(1.0, 1.0, 1.0, 1.0), VertexColor.a);
#else
	vec2 distort = DistortScaleRefractReflect.xy;
	vec4 refractcolor = RefractColor;
#endif
	#ifdef USENORMALMAPSCROLLBLEND
		vec3 normal = dp_texture2D(Texture_Normal, (TexCoord + vec2(0.08, 0.08)*ClientTime*NormalmapScrollBlend.x*0.5)*NormalmapScrollBlend.y).rgb - vec3(1.0);
		normal += dp_texture2D(Texture_Normal, (TexCoord + vec2(-0.06, -0.09)*ClientTime*NormalmapScrollBlend.x)*NormalmapScrollBlend.y*0.75).rgb;
		vec2 ScreenTexCoord = SafeScreenTexCoord + vec3(normalize(cast_myhalf3(normal))).xy * distort;
	#else
		vec2 ScreenTexCoord = SafeScreenTexCoord + vec3(normalize(cast_myhalf3(dp_texture2D(Texture_Normal, TexCoord)) - cast_myhalf3(0.5))).xy * distort;
	#endif

	float f = min(1.0, length(dp_texture2D(Texture_Refraction, ScreenTexCoord + vec2(0.01, 0.01)).rgb) / 0.05);
	f      *= min(1.0, length(dp_texture2D(Texture_Refraction, ScreenTexCoord + vec2(0.01, -0.01)).rgb) / 0.05);
	f      *= min(1.0, length(dp_texture2D(Texture_Refraction, ScreenTexCoord + vec2(-0.01, 0.01)).rgb) / 0.05);
	f      *= min(1.0, length(dp_texture2D(Texture_Refraction, ScreenTexCoord + vec2(-0.01, -0.01)).rgb) / 0.05);
	ScreenTexCoord = mix(SafeScreenTexCoord, ScreenTexCoord, f);
	dp_FragColor = vec4(dp_texture2D(Texture_Refraction, ScreenTexCoord).rgb, 1.0) * refractcolor;
}
#endif
#else

#ifdef MODE_WATER
dp_varying mediump vec2 TexCoord;
dp_varying highp vec3 EyeVector;
dp_varying highp vec4 ModelViewProjectionPosition;
#ifdef VERTEX_SHADER
uniform highp vec3 EyePosition;
uniform highp mat4 TexMatrix;

void main(void)
{
#ifdef USESKELETAL
	ivec4 si0 = ivec4(Attrib_SkeletalIndex * 3.0);
	ivec4 si1 = si0 + ivec4(1, 1, 1, 1);
	ivec4 si2 = si0 + ivec4(2, 2, 2, 2);
	vec4 sw = Attrib_SkeletalWeight;
	vec4 SkeletalMatrix1 = Skeletal_Transform12[si0.x] * sw.x + Skeletal_Transform12[si0.y] * sw.y + Skeletal_Transform12[si0.z] * sw.z + Skeletal_Transform12[si0.w] * sw.w;
	vec4 SkeletalMatrix2 = Skeletal_Transform12[si1.x] * sw.x + Skeletal_Transform12[si1.y] * sw.y + Skeletal_Transform12[si1.z] * sw.z + Skeletal_Transform12[si1.w] * sw.w;
	vec4 SkeletalMatrix3 = Skeletal_Transform12[si2.x] * sw.x + Skeletal_Transform12[si2.y] * sw.y + Skeletal_Transform12[si2.z] * sw.z + Skeletal_Transform12[si2.w] * sw.w;
	mat4 SkeletalMatrix = mat4(SkeletalMatrix1, SkeletalMatrix2, SkeletalMatrix3, vec4(0.0, 0.0, 0.0, 1.0));
	mat3 SkeletalNormalMatrix = mat3(cross(SkeletalMatrix[1].xyz, SkeletalMatrix[2].xyz), cross(SkeletalMatrix[2].xyz, SkeletalMatrix[0].xyz), cross(SkeletalMatrix[0].xyz, SkeletalMatrix[1].xyz));
	vec4 SkeletalVertex = Attrib_Position * SkeletalMatrix;
	vec3 SkeletalSVector = normalize(Attrib_TexCoord1.xyz * SkeletalNormalMatrix);
	vec3 SkeletalTVector = normalize(Attrib_TexCoord2.xyz * SkeletalNormalMatrix);
	vec3 SkeletalNormal  = normalize(Attrib_TexCoord3.xyz * SkeletalNormalMatrix);
#define Attrib_Position SkeletalVertex
#define Attrib_TexCoord1 SkeletalSVector
#define Attrib_TexCoord2 SkeletalTVector
#define Attrib_TexCoord3 SkeletalNormal
#endif
#ifdef USEALPHAGENVERTEX
	VertexColor = Attrib_Color;
#endif
	TexCoord = vec2(TexMatrix * Attrib_TexCoord0);
	vec3 EyeRelative = EyePosition - Attrib_Position.xyz;
	EyeVector.x = dot(EyeRelative, Attrib_TexCoord1.xyz);
	EyeVector.y = dot(EyeRelative, Attrib_TexCoord2.xyz);
	EyeVector.z = dot(EyeRelative, Attrib_TexCoord3.xyz);
	gl_Position = ModelViewProjectionMatrix * Attrib_Position;
	ModelViewProjectionPosition = gl_Position;
#ifdef USETRIPPY
	gl_Position = TrippyVertex(gl_Position);
#endif
}
#endif

#ifdef FRAGMENT_SHADER
uniform sampler2D Texture_Normal;
uniform sampler2D Texture_Refraction;
uniform sampler2D Texture_Reflection;

uniform mediump vec4 DistortScaleRefractReflect;
uniform mediump vec4 ScreenScaleRefractReflect;
uniform mediump vec4 ScreenCenterRefractReflect;
uniform mediump vec4 RefractColor;
uniform mediump vec4 ReflectColor;
uniform mediump float ReflectFactor;
uniform mediump float ReflectOffset;
uniform highp float ClientTime;
#ifdef USENORMALMAPSCROLLBLEND
uniform highp vec2 NormalmapScrollBlend;
#endif

void main(void)
{
	vec4 ScreenScaleRefractReflectIW = ScreenScaleRefractReflect * (1.0 / ModelViewProjectionPosition.w);

	vec4 SafeScreenTexCoord = ModelViewProjectionPosition.xyxy * ScreenScaleRefractReflectIW + ScreenCenterRefractReflect;

#ifdef USEALPHAGENVERTEX
	vec4 distort = DistortScaleRefractReflect * VertexColor.a;
	float reflectoffset = ReflectOffset * VertexColor.a;
	float reflectfactor = ReflectFactor * VertexColor.a;
	vec4 refractcolor = mix(RefractColor, vec4(1.0, 1.0, 1.0, 1.0), VertexColor.a);
#else
	vec4 distort = DistortScaleRefractReflect;
	float reflectoffset = ReflectOffset;
	float reflectfactor = ReflectFactor;
	vec4 refractcolor = RefractColor;
#endif
	#ifdef USENORMALMAPSCROLLBLEND
		vec3 normal = dp_texture2D(Texture_Normal, (TexCoord + vec2(0.08, 0.08)*ClientTime*NormalmapScrollBlend.x*0.5)*NormalmapScrollBlend.y).rgb - vec3(1.0);
		normal += dp_texture2D(Texture_Normal, (TexCoord + vec2(-0.06, -0.09)*ClientTime*NormalmapScrollBlend.x)*NormalmapScrollBlend.y*0.75).rgb;
		vec4 ScreenTexCoord = SafeScreenTexCoord + vec2(normalize(normal) + vec3(0.15)).xyxy * distort;
	#else
		vec4 ScreenTexCoord = SafeScreenTexCoord + vec2(normalize(vec3(dp_texture2D(Texture_Normal, TexCoord)) - vec3(0.5))).xyxy * distort;
	#endif

	float f  = min(1.0, length(dp_texture2D(Texture_Refraction, ScreenTexCoord.xy + vec2(0.005, 0.01)).rgb) / 0.002);
	f       *= min(1.0, length(dp_texture2D(Texture_Refraction, ScreenTexCoord.xy + vec2(0.005, -0.01)).rgb) / 0.002);
	f       *= min(1.0, length(dp_texture2D(Texture_Refraction, ScreenTexCoord.xy + vec2(-0.005, 0.01)).rgb) / 0.002);
	f       *= min(1.0, length(dp_texture2D(Texture_Refraction, ScreenTexCoord.xy + vec2(-0.005, -0.01)).rgb) / 0.002);
	ScreenTexCoord.xy = mix(SafeScreenTexCoord.xy, ScreenTexCoord.xy, f);
	f  = min(1.0, length(dp_texture2D(Texture_Reflection, ScreenTexCoord.zw + vec2(0.005, 0.005)).rgb) / 0.002);
	f *= min(1.0, length(dp_texture2D(Texture_Reflection, ScreenTexCoord.zw + vec2(0.005, -0.005)).rgb) / 0.002);
	f *= min(1.0, length(dp_texture2D(Texture_Reflection, ScreenTexCoord.zw + vec2(-0.005, 0.005)).rgb) / 0.002);
	f *= min(1.0, length(dp_texture2D(Texture_Reflection, ScreenTexCoord.zw + vec2(-0.005, -0.005)).rgb) / 0.002);
	ScreenTexCoord.zw = mix(SafeScreenTexCoord.zw, ScreenTexCoord.zw, f);
	float Fresnel = pow(min(1.0, 1.0 - float(normalize(EyeVector).z)), 2.0) * reflectfactor + reflectoffset;
	dp_FragColor = mix(vec4(dp_texture2D(Texture_Refraction, ScreenTexCoord.xy).rgb, 1) * refractcolor, vec4(dp_texture2D(Texture_Reflection, ScreenTexCoord.zw).rgb, 1) * ReflectColor, Fresnel);
}
#endif
#else

dp_varying mediump vec4 TexCoordSurfaceLightmap;
#ifdef USEVERTEXTEXTUREBLEND
dp_varying mediump vec2 TexCoord2;
#endif

#ifdef MODE_LIGHTSOURCE
dp_varying mediump vec3 CubeVector;
#endif

#if (defined(MODE_LIGHTSOURCE) || defined(MODE_LIGHTDIRECTION)) && defined(USEDIFFUSE)
dp_varying mediump vec3 LightVector;
#endif

#ifdef USEEYEVECTOR
dp_varying highp vec4 EyeVectorFogDepth;
#endif

#if defined(MODE_LIGHTDIRECTIONMAP_MODELSPACE) || defined(MODE_DEFERREDGEOMETRY) || defined(USEREFLECTCUBE) || defined(USEBOUNCEGRIDDIRECTIONAL)
dp_varying highp vec4 VectorS;
dp_varying highp vec4 VectorT;
dp_varying highp vec4 VectorR;
#else
# ifdef USEFOG
dp_varying highp vec3 EyeVectorModelSpace;
# endif
#endif

#ifdef USEREFLECTION
dp_varying highp vec4 ModelViewProjectionPosition;
#endif
#ifdef MODE_DEFERREDLIGHTSOURCE
uniform highp vec3 LightPosition;
dp_varying highp vec4 ModelViewPosition;
#endif

#ifdef MODE_LIGHTSOURCE
uniform highp vec3 LightPosition;
#endif
uniform highp vec3 EyePosition;
#ifdef MODE_LIGHTDIRECTION
uniform highp vec3 LightDir;
#endif
uniform highp vec4 FogPlane;

#ifdef USESHADOWMAPORTHO
dp_varying highp vec3 ShadowMapTC;
#endif

#ifdef USEBOUNCEGRID
dp_varying highp vec3 BounceGridTexCoord;
#endif

#ifdef MODE_DEFERREDGEOMETRY
dp_varying highp float Depth;
#endif

#ifdef FRAGMENT_SHADER

uniform sampler2D Texture_Normal;
uniform sampler2D Texture_Color;
uniform sampler2D Texture_Gloss;
#ifdef USEGLOW
uniform sampler2D Texture_Glow;
#endif
#ifdef USEVERTEXTEXTUREBLEND
uniform sampler2D Texture_SecondaryNormal;
uniform sampler2D Texture_SecondaryColor;
uniform sampler2D Texture_SecondaryGloss;
#ifdef USEGLOW
uniform sampler2D Texture_SecondaryGlow;
#endif
#endif
#ifdef USECOLORMAPPING
uniform sampler2D Texture_Pants;
uniform sampler2D Texture_Shirt;
#endif
#ifdef USEFOG
#ifdef USEFOGHEIGHTTEXTURE
uniform sampler2D Texture_FogHeightTexture;
#endif
uniform sampler2D Texture_FogMask;
#endif
#ifdef USELIGHTMAP
uniform sampler2D Texture_Lightmap;
#endif
#if defined(MODE_LIGHTDIRECTIONMAP_MODELSPACE) || defined(MODE_LIGHTDIRECTIONMAP_TANGENTSPACE)
uniform sampler2D Texture_Deluxemap;
#endif
#ifdef USEREFLECTION
uniform sampler2D Texture_Reflection;
#endif

#ifdef MODE_DEFERREDLIGHTSOURCE
uniform sampler2D Texture_ScreenNormalMap;
#endif
#ifdef USEDEFERREDLIGHTMAP
#ifdef USECELOUTLINES
uniform sampler2D Texture_ScreenNormalMap;
#endif
uniform sampler2D Texture_ScreenDiffuse;
uniform sampler2D Texture_ScreenSpecular;
#endif

uniform mediump vec3 Color_Pants;
uniform mediump vec3 Color_Shirt;
uniform mediump vec3 FogColor;

#ifdef USEFOG
uniform highp float FogRangeRecip;
uniform highp float FogPlaneViewDist;
uniform highp float FogHeightFade;
vec3 FogVertex(vec4 surfacecolor)
{
#if defined(MODE_LIGHTDIRECTIONMAP_MODELSPACE) || defined(MODE_DEFERREDGEOMETRY) || defined(USEREFLECTCUBE) || defined(USEBOUNCEGRIDDIRECTIONAL)
	vec3 EyeVectorModelSpace = vec3(VectorS.w, VectorT.w, VectorR.w);
#endif
	float FogPlaneVertexDist = EyeVectorFogDepth.w;
	float fogfrac;
       vec3 fc = FogColor;
#ifdef USEFOGALPHAHACK
	fc *= surfacecolor.a;
#endif
#ifdef USEFOGHEIGHTTEXTURE
	vec4 fogheightpixel = dp_texture2D(Texture_FogHeightTexture, vec2(1,1) + vec2(FogPlaneVertexDist, FogPlaneViewDist) * (-2.0 * FogHeightFade));
	fogfrac = fogheightpixel.a;
	return mix(fogheightpixel.rgb * fc, surfacecolor.rgb, dp_texture2D(Texture_FogMask, cast_myhalf2(length(EyeVectorModelSpace)*fogfrac*FogRangeRecip, 0.0)).r);
#else
# ifdef USEFOGOUTSIDE
	fogfrac = min(0.0, FogPlaneVertexDist) / (FogPlaneVertexDist - FogPlaneViewDist) * min(1.0, min(0.0, FogPlaneVertexDist) * FogHeightFade);
# else
	fogfrac = FogPlaneViewDist / (FogPlaneViewDist - max(0.0, FogPlaneVertexDist)) * min(1.0, (min(0.0, FogPlaneVertexDist) + FogPlaneViewDist) * FogHeightFade);
# endif
	return mix(fc, surfacecolor.rgb, dp_texture2D(Texture_FogMask, cast_myhalf2(length(EyeVectorModelSpace)*fogfrac*FogRangeRecip, 0.0)).r);
#endif
}
#endif

#ifdef USEOFFSETMAPPING
uniform mediump vec4 OffsetMapping_ScaleSteps;
uniform mediump float OffsetMapping_Bias;
#ifdef USEOFFSETMAPPING_LOD
uniform mediump float OffsetMapping_LodDistance;
#endif
vec2 OffsetMapping(vec2 TexCoord, vec2 dPdx, vec2 dPdy)
{
	float i;

#ifdef USEOFFSETMAPPING_LOD

	mediump float GuessLODFactor = min(1.0, OffsetMapping_LodDistance / EyeVectorFogDepth.z);
#ifdef USEOFFSETMAPPING_RELIEFMAPPING

	mediump float LODSteps = max(3.0, ceil(GuessLODFactor * OffsetMapping_ScaleSteps.y));
#else
	mediump float LODSteps = ceil(GuessLODFactor * OffsetMapping_ScaleSteps.y);
#endif
	mediump float LODFactor = LODSteps / OffsetMapping_ScaleSteps.y;
	mediump vec4 ScaleSteps = vec4(OffsetMapping_ScaleSteps.x, LODSteps, 1.0 / LODSteps, OffsetMapping_ScaleSteps.w * LODFactor);
#else
	#define ScaleSteps OffsetMapping_ScaleSteps
#endif
#ifdef USEOFFSETMAPPING_RELIEFMAPPING
	float f;

	vec3 OffsetVector = vec3(normalize(EyeVectorFogDepth.xyz).xy * ScaleSteps.x * vec2(-1, 1), -1);
	vec3 RT = vec3(vec2(TexCoord.xy - OffsetVector.xy*OffsetMapping_Bias), 1);
	OffsetVector *= ScaleSteps.z;
	for(i = 1.0; i < ScaleSteps.y; ++i)
		RT += OffsetVector *  step(dp_textureGrad(Texture_Normal, RT.xy, dPdx, dPdy).a, RT.z);
	for(i = 0.0, f = 1.0; i < ScaleSteps.w; ++i, f *= 0.5)
		RT += OffsetVector * (step(dp_textureGrad(Texture_Normal, RT.xy, dPdx, dPdy).a, RT.z) * f - 0.5 * f);
	return RT.xy;
#else

	vec2 OffsetVector = vec2(normalize(EyeVectorFogDepth.xyz).xy * ScaleSteps.x * vec2(-1, 1));
	OffsetVector *= ScaleSteps.z;
	for(i = 0.0; i < ScaleSteps.y; ++i)
		TexCoord += OffsetVector * ((1.0 - OffsetMapping_Bias) - dp_textureGrad(Texture_Normal, TexCoord, dPdx, dPdy).a);
	return TexCoord;
#endif
}
#endif

#if defined(MODE_LIGHTSOURCE) || defined(MODE_DEFERREDLIGHTSOURCE)
uniform sampler2D Texture_Attenuation;
uniform samplerCube Texture_Cube;
#endif

#if defined(MODE_LIGHTSOURCE) || defined(MODE_DEFERREDLIGHTSOURCE) || defined(USESHADOWMAPORTHO)

#ifdef USESHADOWMAP2D
# ifdef USESHADOWSAMPLER
uniform sampler2DShadow Texture_ShadowMap2D;
# else
uniform sampler2D Texture_ShadowMap2D;
# endif
#endif

#ifdef USESHADOWMAPVSDCT
uniform samplerCube Texture_CubeProjection;
#endif

#if defined(USESHADOWMAP2D)
uniform mediump vec4 ShadowMap_TextureScale;
uniform mediump vec4 ShadowMap_Parameters;
#endif

#if defined(USESHADOWMAP2D)
# ifdef USESHADOWMAPORTHO
#  define GetShadowMapTC2D(dir) (max(vec3(0.0, 0.0, 0.0), min(dir, ShadowMap_Parameters.xyz)))
# else
#  ifdef USESHADOWMAPVSDCT
vec3 GetShadowMapTC2D(vec3 dir)
{
	vec3 adir = abs(dir);
	float m = max(max(adir.x, adir.y), adir.z);
	vec4 proj = dp_textureCube(Texture_CubeProjection, dir);
#ifdef USEDEPTHRGB
	return vec3(mix(dir.xy, dir.zz, proj.xy) * (ShadowMap_Parameters.x / m) +  proj.zw * ShadowMap_Parameters.z, m + 64.0 * ShadowMap_Parameters.w);
#else
	vec2 mparams = ShadowMap_Parameters.xy / m;
	return vec3(mix(dir.xy, dir.zz, proj.xy) * mparams.x + proj.zw * ShadowMap_Parameters.z, mparams.y + ShadowMap_Parameters.w);
#endif
}
#  else
vec3 GetShadowMapTC2D(vec3 dir)
{
	vec3 adir = abs(dir);
	float m; vec4 proj;
	if (adir.x > adir.y) { m = adir.x; proj = vec4(dir.zyx, 0.5); } else { m = adir.y; proj = vec4(dir.xzy, 1.5); }
	if (adir.z > m) { m = adir.z; proj = vec4(dir, 2.5); }
#ifdef USEDEPTHRGB
	return vec3(proj.xy * (ShadowMap_Parameters.x / m) + vec2(0.5,0.5) + vec2(proj.z < 0.0 ? 1.5 : 0.5, proj.w) * ShadowMap_Parameters.z, m + 64.0 * ShadowMap_Parameters.w);
#else
	vec2 mparams = ShadowMap_Parameters.xy / m;
	return vec3(proj.xy * mparams.x + vec2(proj.z < 0.0 ? 1.5 : 0.5, proj.w) * ShadowMap_Parameters.z, mparams.y + ShadowMap_Parameters.w);
#endif
}
#  endif
# endif
#endif

# ifdef USESHADOWMAP2D
float ShadowMapCompare(vec3 dir)
{
	vec3 shadowmaptc = GetShadowMapTC2D(dir) + vec3(ShadowMap_TextureScale.zw, 0.0f);
	float f;

#  ifdef USEDEPTHRGB
#   ifdef USESHADOWMAPPCF
#    define texval(x, y) decodedepthmacro(dp_texture2D(Texture_ShadowMap2D, center + vec2(x, y)*ShadowMap_TextureScale.xy))
#    if USESHADOWMAPPCF > 1
	vec2 center = shadowmaptc.xy - 0.5, offset = fract(center);
	center *= ShadowMap_TextureScale.xy;
	vec4 row1 = step(shadowmaptc.z, vec4(texval(-1.0, -1.0), texval( 0.0, -1.0), texval( 1.0, -1.0), texval( 2.0, -1.0)));
	vec4 row2 = step(shadowmaptc.z, vec4(texval(-1.0,  0.0), texval( 0.0,  0.0), texval( 1.0,  0.0), texval( 2.0,  0.0)));
	vec4 row3 = step(shadowmaptc.z, vec4(texval(-1.0,  1.0), texval( 0.0,  1.0), texval( 1.0,  1.0), texval( 2.0,  1.0)));
	vec4 row4 = step(shadowmaptc.z, vec4(texval(-1.0,  2.0), texval( 0.0,  2.0), texval( 1.0,  2.0), texval( 2.0,  2.0)));
	vec4 cols = row2 + row3 + mix(row1, row4, offset.y);
	f = dot(mix(cols.xyz, cols.yzw, offset.x), vec3(1.0/9.0));
#    else
	vec2 center = shadowmaptc.xy*ShadowMap_TextureScale.xy, offset = fract(shadowmaptc.xy);
	vec3 row1 = step(shadowmaptc.z, vec3(texval(-1.0, -1.0), texval( 0.0, -1.0), texval( 1.0, -1.0)));
	vec3 row2 = step(shadowmaptc.z, vec3(texval(-1.0,  0.0), texval( 0.0,  0.0), texval( 1.0,  0.0)));
	vec3 row3 = step(shadowmaptc.z, vec3(texval(-1.0,  1.0), texval( 0.0,  1.0), texval( 1.0,  1.0)));
	vec3 cols = row2 + mix(row1, row3, offset.y);
	f = dot(mix(cols.xy, cols.yz, offset.x), vec2(0.25));
#    endif
#   else
	f = step(shadowmaptc.z, decodedepthmacro(dp_texture2D(Texture_ShadowMap2D, shadowmaptc.xy*ShadowMap_TextureScale.xy)));
#   endif
#  else
#   ifdef USESHADOWSAMPLER
#     ifdef USESHADOWMAPPCF
#       define texval(off) dp_shadow2D(Texture_ShadowMap2D, vec3(off, shadowmaptc.z))
	vec2 offset = fract(shadowmaptc.xy - 0.5);
   vec4 size = vec4(offset + 1.0, 2.0 - offset);
#       if USESHADOWMAPPCF > 1
   vec2 center = (shadowmaptc.xy - offset + 0.5)*ShadowMap_TextureScale.xy;
   vec4 weight = (vec4(-1.5, -1.5, 2.0, 2.0) + (shadowmaptc.xy - 0.5*offset).xyxy)*ShadowMap_TextureScale.xyxy;
	f = (1.0/25.0)*dot(size.zxzx*size.wwyy, vec4(texval(weight.xy), texval(weight.zy), texval(weight.xw), texval(weight.zw))) +
		(2.0/25.0)*dot(size, vec4(texval(vec2(weight.z, center.y)), texval(vec2(center.x, weight.w)), texval(vec2(weight.x, center.y)), texval(vec2(center.x, weight.y)))) +
		(4.0/25.0)*texval(center);
#       else
	vec4 weight = (vec4(1.0, 1.0, -0.5, -0.5) + (shadowmaptc.xy - 0.5*offset).xyxy)*ShadowMap_TextureScale.xyxy;
	f = (1.0/9.0)*dot(size.zxzx*size.wwyy, vec4(texval(weight.zw), texval(weight.xw), texval(weight.zy), texval(weight.xy)));
#       endif
#     else
	f = dp_shadow2D(Texture_ShadowMap2D, vec3(shadowmaptc.xy*ShadowMap_TextureScale.xy, shadowmaptc.z));
#     endif
#   else
#     ifdef USESHADOWMAPPCF
#      if defined(GL_ARB_texture_gather) || defined(GL_AMD_texture_texture4)
#       ifdef GL_ARB_texture_gather
#         define texval(x, y) textureGatherOffset(Texture_ShadowMap2D, center, ivec2(x, y))
#       else
#         define texval(x, y) texture4(Texture_ShadowMap2D, center + vec2(x, y)*ShadowMap_TextureScale.xy)
#       endif
	vec2 offset = fract(shadowmaptc.xy - 0.5), center = (shadowmaptc.xy - offset)*ShadowMap_TextureScale.xy;
#       if USESHADOWMAPPCF > 1
   vec4 group1 = step(shadowmaptc.z, texval(-2.0, -2.0));
   vec4 group2 = step(shadowmaptc.z, texval( 0.0, -2.0));
   vec4 group3 = step(shadowmaptc.z, texval( 2.0, -2.0));
   vec4 group4 = step(shadowmaptc.z, texval(-2.0,  0.0));
   vec4 group5 = step(shadowmaptc.z, texval( 0.0,  0.0));
   vec4 group6 = step(shadowmaptc.z, texval( 2.0,  0.0));
   vec4 group7 = step(shadowmaptc.z, texval(-2.0,  2.0));
   vec4 group8 = step(shadowmaptc.z, texval( 0.0,  2.0));
   vec4 group9 = step(shadowmaptc.z, texval( 2.0,  2.0));
	vec4 locols = vec4(group1.ab, group3.ab);
	vec4 hicols = vec4(group7.rg, group9.rg);
	locols.yz += group2.ab;
	hicols.yz += group8.rg;
	vec4 midcols = vec4(group1.rg, group3.rg) + vec4(group7.ab, group9.ab) +
				vec4(group4.rg, group6.rg) + vec4(group4.ab, group6.ab) +
				mix(locols, hicols, offset.y);
	vec4 cols = group5 + vec4(group2.rg, group8.ab);
	cols.xyz += mix(midcols.xyz, midcols.yzw, offset.x);
	f = dot(cols, vec4(1.0/25.0));
#      else
	vec4 group1 = step(shadowmaptc.z, texval(-1.0, -1.0));
	vec4 group2 = step(shadowmaptc.z, texval( 1.0, -1.0));
	vec4 group3 = step(shadowmaptc.z, texval(-1.0,  1.0));
	vec4 group4 = step(shadowmaptc.z, texval( 1.0,  1.0));
	vec4 cols = vec4(group1.rg, group2.rg) + vec4(group3.ab, group4.ab) +
				mix(vec4(group1.ab, group2.ab), vec4(group3.rg, group4.rg), offset.y);
	f = dot(mix(cols.xyz, cols.yzw, offset.x), vec3(1.0/9.0));
#       endif
#      else
#       ifdef GL_EXT_gpu_shader4
#         define texval(x, y) dp_textureOffset(Texture_ShadowMap2D, center, x, y).r
#       else
#         define texval(x, y) dp_texture2D(Texture_ShadowMap2D, center + vec2(x, y)*ShadowMap_TextureScale.xy).r
#       endif
#       if USESHADOWMAPPCF > 1
	vec2 center = shadowmaptc.xy - 0.5, offset = fract(center);
	center *= ShadowMap_TextureScale.xy;
	vec4 row1 = step(shadowmaptc.z, vec4(texval(-1.0, -1.0), texval( 0.0, -1.0), texval( 1.0, -1.0), texval( 2.0, -1.0)));
	vec4 row2 = step(shadowmaptc.z, vec4(texval(-1.0,  0.0), texval( 0.0,  0.0), texval( 1.0,  0.0), texval( 2.0,  0.0)));
	vec4 row3 = step(shadowmaptc.z, vec4(texval(-1.0,  1.0), texval( 0.0,  1.0), texval( 1.0,  1.0), texval( 2.0,  1.0)));
	vec4 row4 = step(shadowmaptc.z, vec4(texval(-1.0,  2.0), texval( 0.0,  2.0), texval( 1.0,  2.0), texval( 2.0,  2.0)));
	vec4 cols = row2 + row3 + mix(row1, row4, offset.y);
	f = dot(mix(cols.xyz, cols.yzw, offset.x), vec3(1.0/9.0));
#       else
	vec2 center = shadowmaptc.xy*ShadowMap_TextureScale.xy, offset = fract(shadowmaptc.xy);
	vec3 row1 = step(shadowmaptc.z, vec3(texval(-1.0, -1.0), texval( 0.0, -1.0), texval( 1.0, -1.0)));
	vec3 row2 = step(shadowmaptc.z, vec3(texval(-1.0,  0.0), texval( 0.0,  0.0), texval( 1.0,  0.0)));
	vec3 row3 = step(shadowmaptc.z, vec3(texval(-1.0,  1.0), texval( 0.0,  1.0), texval( 1.0,  1.0)));
	vec3 cols = row2 + mix(row1, row3, offset.y);
	f = dot(mix(cols.xy, cols.yz, offset.x), vec2(0.25));
#       endif
#      endif
#     else
	f = step(shadowmaptc.z, dp_texture2D(Texture_ShadowMap2D, shadowmaptc.xy*ShadowMap_TextureScale.xy).r);
#     endif
#   endif
#  endif
#  ifdef USESHADOWMAPORTHO
	return mix(ShadowMap_Parameters.w, 1.0, f);
#  else
	return f;
#  endif
}
# endif
#endif
#endif

#ifdef MODE_DEFERREDGEOMETRY
#ifdef VERTEX_SHADER
uniform highp mat4 TexMatrix;
#ifdef USEVERTEXTEXTUREBLEND
uniform highp mat4 BackgroundTexMatrix;
#endif
uniform highp mat4 ModelViewMatrix;
void main(void)
{
#ifdef USESKELETAL
	ivec4 si0 = ivec4(Attrib_SkeletalIndex * 3.0);
	ivec4 si1 = si0 + ivec4(1, 1, 1, 1);
	ivec4 si2 = si0 + ivec4(2, 2, 2, 2);
	vec4 sw = Attrib_SkeletalWeight;
	vec4 SkeletalMatrix1 = Skeletal_Transform12[si0.x] * sw.x + Skeletal_Transform12[si0.y] * sw.y + Skeletal_Transform12[si0.z] * sw.z + Skeletal_Transform12[si0.w] * sw.w;
	vec4 SkeletalMatrix2 = Skeletal_Transform12[si1.x] * sw.x + Skeletal_Transform12[si1.y] * sw.y + Skeletal_Transform12[si1.z] * sw.z + Skeletal_Transform12[si1.w] * sw.w;
	vec4 SkeletalMatrix3 = Skeletal_Transform12[si2.x] * sw.x + Skeletal_Transform12[si2.y] * sw.y + Skeletal_Transform12[si2.z] * sw.z + Skeletal_Transform12[si2.w] * sw.w;
	mat4 SkeletalMatrix = mat4(SkeletalMatrix1, SkeletalMatrix2, SkeletalMatrix3, vec4(0.0, 0.0, 0.0, 1.0));
	mat3 SkeletalNormalMatrix = mat3(cross(SkeletalMatrix[1].xyz, SkeletalMatrix[2].xyz), cross(SkeletalMatrix[2].xyz, SkeletalMatrix[0].xyz), cross(SkeletalMatrix[0].xyz, SkeletalMatrix[1].xyz));
	vec4 SkeletalVertex = Attrib_Position * SkeletalMatrix;
	vec3 SkeletalSVector = normalize(Attrib_TexCoord1.xyz * SkeletalNormalMatrix);
	vec3 SkeletalTVector = normalize(Attrib_TexCoord2.xyz * SkeletalNormalMatrix);
	vec3 SkeletalNormal  = normalize(Attrib_TexCoord3.xyz * SkeletalNormalMatrix);
#define Attrib_Position SkeletalVertex
#define Attrib_TexCoord1 SkeletalSVector
#define Attrib_TexCoord2 SkeletalTVector
#define Attrib_TexCoord3 SkeletalNormal
#endif
	TexCoordSurfaceLightmap = vec4((TexMatrix * Attrib_TexCoord0).xy, 0.0, 0.0);
#ifdef USEVERTEXTEXTUREBLEND
	VertexColor = Attrib_Color;
	TexCoord2 = vec2(BackgroundTexMatrix * Attrib_TexCoord0);
#endif

#ifdef USEOFFSETMAPPING
	vec3 EyeRelative = EyePosition - Attrib_Position.xyz;
	EyeVectorFogDepth.x = dot(EyeRelative, Attrib_TexCoord1.xyz);
	EyeVectorFogDepth.y = dot(EyeRelative, Attrib_TexCoord2.xyz);
	EyeVectorFogDepth.z = dot(EyeRelative, Attrib_TexCoord3.xyz);
	EyeVectorFogDepth.w = 0.0;
#endif

	VectorS = (ModelViewMatrix * vec4(Attrib_TexCoord1.xyz, 0));
	VectorT = (ModelViewMatrix * vec4(Attrib_TexCoord2.xyz, 0));
	VectorR = (ModelViewMatrix * vec4(Attrib_TexCoord3.xyz, 0));
	gl_Position = ModelViewProjectionMatrix * Attrib_Position;
#ifdef USETRIPPY
	gl_Position = TrippyVertex(gl_Position);
#endif
	Depth = (ModelViewMatrix * Attrib_Position).z;
}
#endif

#ifdef FRAGMENT_SHADER
void main(void)
{
#ifdef USEOFFSETMAPPING

	vec2 dPdx = dp_offsetmapping_dFdx(TexCoordSurfaceLightmap.xy);
	vec2 dPdy = dp_offsetmapping_dFdy(TexCoordSurfaceLightmap.xy);
	vec2 TexCoordOffset = OffsetMapping(TexCoordSurfaceLightmap.xy, dPdx, dPdy);
# define offsetMappedTexture2D(t) dp_textureGrad(t, TexCoordOffset, dPdx, dPdy)
#else
# define offsetMappedTexture2D(t) dp_texture2D(t, TexCoordSurfaceLightmap.xy)
#endif

#ifdef USEALPHAKILL
	if (offsetMappedTexture2D(Texture_Color).a < 0.5)
		discard;
#endif

#ifdef USEVERTEXTEXTUREBLEND
	float alpha = offsetMappedTexture2D(Texture_Color).a;
	float terrainblend = clamp(float(VertexColor.a) * alpha * 2.0 - 0.5, float(0.0), float(1.0));

#endif

#ifdef USEVERTEXTEXTUREBLEND
	vec3 surfacenormal = mix(vec3(dp_texture2D(Texture_SecondaryNormal, TexCoord2)), vec3(offsetMappedTexture2D(Texture_Normal)), terrainblend) - vec3(0.5, 0.5, 0.5);
	float a = mix(dp_texture2D(Texture_SecondaryGloss, TexCoord2).a, offsetMappedTexture2D(Texture_Gloss).a, terrainblend);
#else
	vec3 surfacenormal = vec3(offsetMappedTexture2D(Texture_Normal)) - vec3(0.5, 0.5, 0.5);
	float a = offsetMappedTexture2D(Texture_Gloss).a;
#endif

	vec3 pixelnormal = normalize(surfacenormal.x * VectorS.xyz + surfacenormal.y * VectorT.xyz + surfacenormal.z * VectorR.xyz);
	dp_FragColor = vec4(pixelnormal.x, pixelnormal.y, Depth, a);
}
#endif
#else

#ifdef MODE_DEFERREDLIGHTSOURCE
#ifdef VERTEX_SHADER
uniform highp mat4 ModelViewMatrix;
void main(void)
{
	ModelViewPosition = ModelViewMatrix * Attrib_Position;
	gl_Position = ModelViewProjectionMatrix * Attrib_Position;
}
#endif

#ifdef FRAGMENT_SHADER
uniform highp mat4 ViewToLight;

uniform highp vec2 ScreenToDepth;
uniform myhalf3 DeferredColor_Ambient;
uniform myhalf3 DeferredColor_Diffuse;
#ifdef USESPECULAR
uniform myhalf3 DeferredColor_Specular;
uniform myhalf SpecularPower;
#endif
uniform myhalf2 PixelToScreenTexCoord;
void main(void)
{

	vec2 ScreenTexCoord = gl_FragCoord.xy * PixelToScreenTexCoord;
	vec3 position;

	myhalf4 normalmap = dp_texture2D(Texture_ScreenNormalMap, ScreenTexCoord);

	myhalf3 surfacenormal = myhalf3(normalmap.rg, sqrt(1.0-dot(normalmap.rg, normalmap.rg)));

	position.z = normalmap.b;

	position.xy = ModelViewPosition.xy * (position.z / ModelViewPosition.z);

	vec3 CubeVector = vec3(ViewToLight * vec4(position,1));
	myhalf fade = cast_myhalf(dp_texture2D(Texture_Attenuation, vec2(length(CubeVector), 0.0)));
#ifdef USEDIFFUSE

	myhalf3 lightnormal = cast_myhalf3(normalize(LightPosition - position));
SHADEDIFFUSE
#endif
#ifdef USESPECULAR

	myhalf3 eyenormal = -normalize(cast_myhalf3(position));
SHADESPECULAR(SpecularPower * normalmap.a)
#endif

#if defined(USESHADOWMAP2D)
	fade *= ShadowMapCompare(CubeVector);
#endif

#ifdef USESPECULAR
	gl_FragData[0] = vec4((DeferredColor_Ambient + DeferredColor_Diffuse * diffuse) * fade, 1.0);
	gl_FragData[1] = vec4(DeferredColor_Specular * (specular * fade), 1.0);
# ifdef USECUBEFILTER
	vec3 cubecolor = dp_textureCube(Texture_Cube, CubeVector).rgb;
	gl_FragData[0].rgb *= cubecolor;
	gl_FragData[1].rgb *= cubecolor;
# endif
#else
# ifdef USEDIFFUSE
	gl_FragColor = vec4((DeferredColor_Ambient + DeferredColor_Diffuse * diffuse) * fade, 1.0);
# else
	gl_FragColor = vec4(DeferredColor_Ambient * fade, 1.0);
# endif
# ifdef USECUBEFILTER
	vec3 cubecolor = dp_textureCube(Texture_Cube, CubeVector).rgb;
	gl_FragColor.rgb *= cubecolor;
# endif
#endif
}
#endif
#else

#ifdef VERTEX_SHADER
uniform highp mat4 TexMatrix;
#ifdef USEVERTEXTEXTUREBLEND
uniform highp mat4 BackgroundTexMatrix;
#endif
#ifdef MODE_LIGHTSOURCE
uniform highp mat4 ModelToLight;
#endif
#ifdef USESHADOWMAPORTHO
uniform highp mat4 ShadowMapMatrix;
#endif
#ifdef USEBOUNCEGRID
uniform highp mat4 BounceGridMatrix;
#endif
void main(void)
{
#ifdef USESKELETAL
	ivec4 si0 = ivec4(Attrib_SkeletalIndex * 3.0);
	ivec4 si1 = si0 + ivec4(1, 1, 1, 1);
	ivec4 si2 = si0 + ivec4(2, 2, 2, 2);
	vec4 sw = Attrib_SkeletalWeight;
	vec4 SkeletalMatrix1 = Skeletal_Transform12[si0.x] * sw.x + Skeletal_Transform12[si0.y] * sw.y + Skeletal_Transform12[si0.z] * sw.z + Skeletal_Transform12[si0.w] * sw.w;
	vec4 SkeletalMatrix2 = Skeletal_Transform12[si1.x] * sw.x + Skeletal_Transform12[si1.y] * sw.y + Skeletal_Transform12[si1.z] * sw.z + Skeletal_Transform12[si1.w] * sw.w;
	vec4 SkeletalMatrix3 = Skeletal_Transform12[si2.x] * sw.x + Skeletal_Transform12[si2.y] * sw.y + Skeletal_Transform12[si2.z] * sw.z + Skeletal_Transform12[si2.w] * sw.w;
	mat4 SkeletalMatrix = mat4(SkeletalMatrix1, SkeletalMatrix2, SkeletalMatrix3, vec4(0.0, 0.0, 0.0, 1.0));

	mat3 SkeletalNormalMatrix = mat3(cross(SkeletalMatrix[1].xyz, SkeletalMatrix[2].xyz), cross(SkeletalMatrix[2].xyz, SkeletalMatrix[0].xyz), cross(SkeletalMatrix[0].xyz, SkeletalMatrix[1].xyz));
	vec4 SkeletalVertex = Attrib_Position * SkeletalMatrix;
	SkeletalVertex.w = 1.0;
	vec3 SkeletalSVector = normalize(Attrib_TexCoord1.xyz * SkeletalNormalMatrix);
	vec3 SkeletalTVector = normalize(Attrib_TexCoord2.xyz * SkeletalNormalMatrix);
	vec3 SkeletalNormal  = normalize(Attrib_TexCoord3.xyz * SkeletalNormalMatrix);
#define Attrib_Position SkeletalVertex
#define Attrib_TexCoord1 SkeletalSVector
#define Attrib_TexCoord2 SkeletalTVector
#define Attrib_TexCoord3 SkeletalNormal
#endif

#if defined(MODE_VERTEXCOLOR) || defined(USEVERTEXTEXTUREBLEND) || defined(MODE_LIGHTDIRECTIONMAP_FORCED_VERTEXCOLOR) || defined(USEALPHAGENVERTEX)
	VertexColor = Attrib_Color;
#endif

#ifdef USELIGHTMAP
	TexCoordSurfaceLightmap = vec4((TexMatrix * Attrib_TexCoord0).xy, Attrib_TexCoord4.xy);
#else
	TexCoordSurfaceLightmap = vec4((TexMatrix * Attrib_TexCoord0).xy, 0.0, 0.0);
#endif
#ifdef USEVERTEXTEXTUREBLEND
	TexCoord2 = vec2(BackgroundTexMatrix * Attrib_TexCoord0);
#endif

#ifdef USEBOUNCEGRID
	BounceGridTexCoord = vec3(BounceGridMatrix * Attrib_Position);
#ifdef USEBOUNCEGRIDDIRECTIONAL
	BounceGridTexCoord.z *= 0.125;
#endif
#endif

#ifdef MODE_LIGHTSOURCE

	CubeVector = vec3(ModelToLight * Attrib_Position);

# ifdef USEDIFFUSE

	vec3 lightminusvertex = LightPosition - Attrib_Position.xyz;
	LightVector.x = dot(lightminusvertex, Attrib_TexCoord1.xyz);
	LightVector.y = dot(lightminusvertex, Attrib_TexCoord2.xyz);
	LightVector.z = dot(lightminusvertex, Attrib_TexCoord3.xyz);
# endif
#endif

#if defined(MODE_LIGHTDIRECTION) && defined(USEDIFFUSE)
	LightVector.x = dot(LightDir, Attrib_TexCoord1.xyz);
	LightVector.y = dot(LightDir, Attrib_TexCoord2.xyz);
	LightVector.z = dot(LightDir, Attrib_TexCoord3.xyz);
#endif

#ifdef USEEYEVECTOR
	vec3 EyeRelative = EyePosition - Attrib_Position.xyz;
	EyeVectorFogDepth.x = dot(EyeRelative, Attrib_TexCoord1.xyz);
	EyeVectorFogDepth.y = dot(EyeRelative, Attrib_TexCoord2.xyz);
	EyeVectorFogDepth.z = dot(EyeRelative, Attrib_TexCoord3.xyz);
#ifdef USEFOG
	EyeVectorFogDepth.w = dot(FogPlane, Attrib_Position);
#else
	EyeVectorFogDepth.w = 0.0;
#endif
#endif

#if defined(MODE_LIGHTDIRECTIONMAP_MODELSPACE) || defined(USEREFLECTCUBE) || defined(USEBOUNCEGRIDDIRECTIONAL)
# ifdef USEFOG
	VectorS = vec4(Attrib_TexCoord1.xyz, EyePosition.x - Attrib_Position.x);
	VectorT = vec4(Attrib_TexCoord2.xyz, EyePosition.y - Attrib_Position.y);
	VectorR = vec4(Attrib_TexCoord3.xyz, EyePosition.z - Attrib_Position.z);
# else
	VectorS = vec4(Attrib_TexCoord1, 0);
	VectorT = vec4(Attrib_TexCoord2, 0);
	VectorR = vec4(Attrib_TexCoord3, 0);
# endif
#else
# ifdef USEFOG
	EyeVectorModelSpace = EyePosition - Attrib_Position.xyz;
# endif
#endif

	gl_Position = ModelViewProjectionMatrix * Attrib_Position;

#ifdef USESHADOWMAPORTHO
	ShadowMapTC = vec3(ShadowMapMatrix * gl_Position);
#endif

#ifdef USEREFLECTION
	ModelViewProjectionPosition = gl_Position;
#endif
#ifdef USETRIPPY
	gl_Position = TrippyVertex(gl_Position);
#endif
}
#endif

#ifdef FRAGMENT_SHADER
#ifdef USEDEFERREDLIGHTMAP
uniform myhalf2 PixelToScreenTexCoord;
uniform myhalf3 DeferredMod_Diffuse;
uniform myhalf3 DeferredMod_Specular;
#endif
uniform myhalf3 Color_Ambient;
uniform myhalf3 Color_Diffuse;
uniform myhalf3 Color_Specular;
uniform myhalf SpecularPower;
#ifdef USEGLOW
uniform myhalf3 Color_Glow;
#endif
uniform myhalf Alpha;
#ifdef USEREFLECTION
uniform mediump vec4 DistortScaleRefractReflect;
uniform mediump vec4 ScreenScaleRefractReflect;
uniform mediump vec4 ScreenCenterRefractReflect;
uniform mediump vec4 ReflectColor;
#endif
#ifdef USEREFLECTCUBE
uniform highp mat4 ModelToReflectCube;
uniform sampler2D Texture_ReflectMask;
uniform samplerCube Texture_ReflectCube;
#endif
#ifdef USEBOUNCEGRID
uniform sampler3D Texture_BounceGrid;
uniform float BounceGridIntensity;
uniform highp mat4 BounceGridMatrix;
#endif
uniform highp float ClientTime;
#ifdef USENORMALMAPSCROLLBLEND
uniform highp vec2 NormalmapScrollBlend;
#endif
#ifdef USEOCCLUDE
uniform occludeQuery {
    uint visiblepixels;
    uint allpixels;
};
#endif
void main(void)
{
#ifdef USEOFFSETMAPPING

	vec2 dPdx = dp_offsetmapping_dFdx(TexCoordSurfaceLightmap.xy);
	vec2 dPdy = dp_offsetmapping_dFdy(TexCoordSurfaceLightmap.xy);
	vec2 TexCoordOffset = OffsetMapping(TexCoordSurfaceLightmap.xy, dPdx, dPdy);
# define offsetMappedTexture2D(t) dp_textureGrad(t, TexCoordOffset, dPdx, dPdy)
# define TexCoord TexCoordOffset
#else
# define offsetMappedTexture2D(t) dp_texture2D(t, TexCoordSurfaceLightmap.xy)
# define TexCoord TexCoordSurfaceLightmap.xy
#endif

	myhalf4 color = cast_myhalf4(offsetMappedTexture2D(Texture_Color));
#ifdef USEALPHAKILL
	if (color.a < 0.5)
		discard;
#endif
	color.a *= Alpha;
#ifdef USECOLORMAPPING
	color.rgb += cast_myhalf3(offsetMappedTexture2D(Texture_Pants)) * Color_Pants + cast_myhalf3(offsetMappedTexture2D(Texture_Shirt)) * Color_Shirt;
#endif
#ifdef USEVERTEXTEXTUREBLEND
#ifdef USEBOTHALPHAS
	myhalf4 color2 = cast_myhalf4(dp_texture2D(Texture_SecondaryColor, TexCoord2));
	myhalf terrainblend = clamp(cast_myhalf(VertexColor.a) * color.a, cast_myhalf(1.0 - color2.a), cast_myhalf(1.0));
	color.rgb = mix(color2.rgb, color.rgb, terrainblend);
#else
	myhalf terrainblend = clamp(cast_myhalf(VertexColor.a) * color.a * 2.0 - 0.5, cast_myhalf(0.0), cast_myhalf(1.0));

	color.rgb = mix(cast_myhalf3(dp_texture2D(Texture_SecondaryColor, TexCoord2)), color.rgb, terrainblend);
#endif
	color.a = 1.0;

#endif
#ifdef USEALPHAGENVERTEX
	color.a *= VertexColor.a;
#endif

#ifdef USEVERTEXTEXTUREBLEND
	myhalf3 surfacenormal = normalize(mix(cast_myhalf3(dp_texture2D(Texture_SecondaryNormal, TexCoord2)), cast_myhalf3(offsetMappedTexture2D(Texture_Normal)), terrainblend) - cast_myhalf3(0.5, 0.5, 0.5));
#else
	myhalf3 surfacenormal = normalize(cast_myhalf3(offsetMappedTexture2D(Texture_Normal)) - cast_myhalf3(0.5, 0.5, 0.5));
#endif

	myhalf3 diffusetex = color.rgb;
#if defined(USESPECULAR) || defined(USEDEFERREDLIGHTMAP)
# ifdef USEVERTEXTEXTUREBLEND
	myhalf4 glosstex = mix(cast_myhalf4(dp_texture2D(Texture_SecondaryGloss, TexCoord2)), cast_myhalf4(offsetMappedTexture2D(Texture_Gloss)), terrainblend);
# else
	myhalf4 glosstex = cast_myhalf4(offsetMappedTexture2D(Texture_Gloss));
# endif
#endif

#ifdef USEREFLECTCUBE
	vec3 TangentReflectVector = reflect(-EyeVectorFogDepth.xyz, surfacenormal);
	vec3 ModelReflectVector = TangentReflectVector.x * VectorS.xyz + TangentReflectVector.y * VectorT.xyz + TangentReflectVector.z * VectorR.xyz;
	vec3 ReflectCubeTexCoord = vec3(ModelToReflectCube * vec4(ModelReflectVector, 0));
	diffusetex += cast_myhalf3(offsetMappedTexture2D(Texture_ReflectMask)) * cast_myhalf3(dp_textureCube(Texture_ReflectCube, ReflectCubeTexCoord));
#endif

#ifdef USESPECULAR
	myhalf3 eyenormal = normalize(cast_myhalf3(EyeVectorFogDepth.xyz));
#endif

#ifdef MODE_LIGHTSOURCE

#ifdef USEDIFFUSE
	myhalf3 lightnormal = cast_myhalf3(normalize(LightVector));
SHADEDIFFUSE
	color.rgb = diffusetex * (Color_Ambient + diffuse * Color_Diffuse);
#ifdef USESPECULAR
SHADESPECULAR(SpecularPower * glosstex.a)
	color.rgb += glosstex.rgb * (specular * Color_Specular);
#endif
#else
	color.rgb = diffusetex * Color_Ambient;
#endif
	color.rgb *= cast_myhalf(dp_texture2D(Texture_Attenuation, vec2(length(CubeVector), 0.0)));
#if defined(USESHADOWMAP2D)
	color.rgb *= ShadowMapCompare(CubeVector);
#endif
# ifdef USECUBEFILTER
	color.rgb *= cast_myhalf3(dp_textureCube(Texture_Cube, CubeVector));
# endif
#endif

#ifdef MODE_LIGHTDIRECTION
	#define SHADING
	#ifdef USEDIFFUSE
		myhalf3 lightnormal = cast_myhalf3(normalize(LightVector));
	#endif
	#define lightcolor 1
#endif
#ifdef MODE_LIGHTDIRECTIONMAP_MODELSPACE
   #define SHADING

	myhalf3 lightnormal_modelspace = cast_myhalf3(dp_texture2D(Texture_Deluxemap, TexCoordSurfaceLightmap.zw)) * 2.0 + cast_myhalf3(-1.0, -1.0, -1.0);
	myhalf3 lightcolor = cast_myhalf3(dp_texture2D(Texture_Lightmap, TexCoordSurfaceLightmap.zw));

	myhalf3 lightnormal;
	lightnormal.x = dot(lightnormal_modelspace, cast_myhalf3(VectorS));
	lightnormal.y = dot(lightnormal_modelspace, cast_myhalf3(VectorT));
	lightnormal.z = dot(lightnormal_modelspace, cast_myhalf3(VectorR));
	lightnormal = normalize(lightnormal);

	lightcolor *= 1.0 / max(0.25, lightnormal.z);
#endif
#ifdef MODE_LIGHTDIRECTIONMAP_TANGENTSPACE
   #define SHADING

	myhalf3 lightnormal = cast_myhalf3(dp_texture2D(Texture_Deluxemap, TexCoordSurfaceLightmap.zw)) * 2.0 + cast_myhalf3(-1.0, -1.0, -1.0);
	myhalf3 lightcolor = cast_myhalf3(dp_texture2D(Texture_Lightmap, TexCoordSurfaceLightmap.zw));
#endif
#if defined(MODE_LIGHTDIRECTIONMAP_FORCED_LIGHTMAP) || defined(MODE_LIGHTDIRECTIONMAP_FORCED_VERTEXCOLOR)
	#define SHADING

	myhalf3 lightnormal = cast_myhalf3(0.0, 0.0, 1.0);
   #ifdef USELIGHTMAP
		myhalf3 lightcolor = cast_myhalf3(dp_texture2D(Texture_Lightmap, TexCoordSurfaceLightmap.zw));
   #else
		myhalf3 lightcolor = cast_myhalf3(VertexColor.rgb);
   #endif
#endif
#ifdef MODE_FAKELIGHT
	#define SHADING
	myhalf3 lightnormal = cast_myhalf3(normalize(EyeVectorFogDepth.xyz));
	#define lightcolor 1
#endif

#ifdef MODE_LIGHTMAP
	color.rgb = diffusetex * (Color_Ambient + cast_myhalf3(dp_texture2D(Texture_Lightmap, TexCoordSurfaceLightmap.zw)) * Color_Diffuse);
#endif
#ifdef MODE_VERTEXCOLOR
	color.rgb = diffusetex * (Color_Ambient + cast_myhalf3(VertexColor.rgb) * Color_Diffuse);
#endif
#ifdef MODE_FLATCOLOR
	color.rgb = diffusetex * Color_Ambient;
#endif

#ifdef SHADING
# ifdef USEDIFFUSE
SHADEDIFFUSE
#  ifdef USESPECULAR
SHADESPECULAR(SpecularPower * glosstex.a)
	color.rgb = diffusetex * Color_Ambient + (diffusetex * Color_Diffuse * diffuse + glosstex.rgb * Color_Specular * specular) * lightcolor;
#  else
	color.rgb = diffusetex * (Color_Ambient + Color_Diffuse * diffuse * lightcolor);
#  endif
# else
	color.rgb = diffusetex * Color_Ambient;
# endif
#endif

#ifdef USESHADOWMAPORTHO
	color.rgb *= ShadowMapCompare(ShadowMapTC);
#endif

#ifdef USEDEFERREDLIGHTMAP
	vec2 ScreenTexCoord = gl_FragCoord.xy * PixelToScreenTexCoord;
	color.rgb += diffusetex * cast_myhalf3(dp_texture2D(Texture_ScreenDiffuse, ScreenTexCoord)) * DeferredMod_Diffuse;
	color.rgb += glosstex.rgb * cast_myhalf3(dp_texture2D(Texture_ScreenSpecular, ScreenTexCoord)) * DeferredMod_Specular;

#endif

#ifdef USEBOUNCEGRID
#ifdef USEBOUNCEGRIDDIRECTIONAL

	myhalf4 bouncegrid_coeff3 = cast_myhalf4(dp_texture3D(Texture_BounceGrid, BounceGridTexCoord + vec3(0.0, 0.0, 0.250)));
	myhalf4 bouncegrid_coeff4 = cast_myhalf4(dp_texture3D(Texture_BounceGrid, BounceGridTexCoord + vec3(0.0, 0.0, 0.375)));
	myhalf4 bouncegrid_coeff5 = cast_myhalf4(dp_texture3D(Texture_BounceGrid, BounceGridTexCoord + vec3(0.0, 0.0, 0.500)));
	myhalf4 bouncegrid_coeff6 = cast_myhalf4(dp_texture3D(Texture_BounceGrid, BounceGridTexCoord + vec3(0.0, 0.0, 0.625)));
	myhalf4 bouncegrid_coeff7 = cast_myhalf4(dp_texture3D(Texture_BounceGrid, BounceGridTexCoord + vec3(0.0, 0.0, 0.750)));
	myhalf4 bouncegrid_coeff8 = cast_myhalf4(dp_texture3D(Texture_BounceGrid, BounceGridTexCoord + vec3(0.0, 0.0, 0.875)));
	myhalf3 bouncegrid_dir = normalize(mat3(BounceGridMatrix) * (surfacenormal.x * VectorS.xyz + surfacenormal.y * VectorT.xyz + surfacenormal.z * VectorR.xyz));
	myhalf3 bouncegrid_dirp = max(cast_myhalf3(0.0, 0.0, 0.0), bouncegrid_dir);
	myhalf3 bouncegrid_dirn = max(cast_myhalf3(0.0, 0.0, 0.0), -bouncegrid_dir);

	myhalf3 bouncegrid_light = cast_myhalf3(
		dot(bouncegrid_coeff3.xyz, bouncegrid_dirp) + dot(bouncegrid_coeff6.xyz, bouncegrid_dirn),
		dot(bouncegrid_coeff4.xyz, bouncegrid_dirp) + dot(bouncegrid_coeff7.xyz, bouncegrid_dirn),
		dot(bouncegrid_coeff5.xyz, bouncegrid_dirp) + dot(bouncegrid_coeff8.xyz, bouncegrid_dirn));
	color.rgb += diffusetex * bouncegrid_light * BounceGridIntensity;

#else
	color.rgb += diffusetex * cast_myhalf3(dp_texture3D(Texture_BounceGrid, BounceGridTexCoord)) * BounceGridIntensity;
#endif
#endif

#ifdef USEGLOW
#ifdef USEVERTEXTEXTUREBLEND
	color.rgb += mix(cast_myhalf3(dp_texture2D(Texture_SecondaryGlow, TexCoord2)), cast_myhalf3(offsetMappedTexture2D(Texture_Glow)), terrainblend) * Color_Glow;
#else
	color.rgb += cast_myhalf3(offsetMappedTexture2D(Texture_Glow)) * Color_Glow;
#endif
#endif

#ifdef USECELOUTLINES
# ifdef USEDEFERREDLIGHTMAP

	vec4 ScreenTexCoordStep = vec4(PixelToScreenTexCoord.x, 0.0, 0.0, PixelToScreenTexCoord.y);
	vec4 DepthNeighbors;

	float DepthCenter = dp_texture2D(Texture_ScreenNormalMap, ScreenTexCoord).b;

	float DepthScale1 = 4.0 / DepthCenter;

	float DepthScale2 = DepthScale1 / 2.0;

	float DepthBias1 = -DepthCenter * DepthScale1;
	float DepthBias2 = -DepthCenter * DepthScale2;

	float DepthShadow = max(0.0, dp_texture2D(Texture_ScreenNormalMap, ScreenTexCoord + PixelToScreenTexCoord * vec2(-1.0,  0.0)).b * DepthScale1 + DepthBias1)
	                  + max(0.0, dp_texture2D(Texture_ScreenNormalMap, ScreenTexCoord + PixelToScreenTexCoord * vec2( 1.0,  0.0)).b * DepthScale1 + DepthBias1)
	                  + max(0.0, dp_texture2D(Texture_ScreenNormalMap, ScreenTexCoord + PixelToScreenTexCoord * vec2( 0.0, -1.0)).b * DepthScale1 + DepthBias1)
	                  + max(0.0, dp_texture2D(Texture_ScreenNormalMap, ScreenTexCoord + PixelToScreenTexCoord * vec2( 0.0,  1.0)).b * DepthScale1 + DepthBias1)
	                  + max(0.0, dp_texture2D(Texture_ScreenNormalMap, ScreenTexCoord + PixelToScreenTexCoord * vec2(-2.0,  0.0)).b * DepthScale2 + DepthBias2)
	                  + max(0.0, dp_texture2D(Texture_ScreenNormalMap, ScreenTexCoord + PixelToScreenTexCoord * vec2( 2.0,  0.0)).b * DepthScale2 + DepthBias2)
	                  + max(0.0, dp_texture2D(Texture_ScreenNormalMap, ScreenTexCoord + PixelToScreenTexCoord * vec2( 0.0, -2.0)).b * DepthScale2 + DepthBias2)
	                  + max(0.0, dp_texture2D(Texture_ScreenNormalMap, ScreenTexCoord + PixelToScreenTexCoord * vec2( 0.0,  2.0)).b * DepthScale2 + DepthBias2)

	                  - 0.0;
	color.rgb *= 1.0 - max(0.0, min(DepthShadow, 1.0));

# endif
#endif

#ifdef USEFOG
	color.rgb = FogVertex(color);
#endif

#ifdef USEREFLECTION
	vec4 ScreenScaleRefractReflectIW = ScreenScaleRefractReflect * (1.0 / ModelViewProjectionPosition.w);

	vec2 SafeScreenTexCoord = ModelViewProjectionPosition.xy * ScreenScaleRefractReflectIW.zw + ScreenCenterRefractReflect.zw;
	#ifdef USENORMALMAPSCROLLBLEND
# ifdef USEOFFSETMAPPING
		vec3 normal = dp_textureGrad(Texture_Normal, (TexCoord + vec2(0.08, 0.08)*ClientTime*NormalmapScrollBlend.x*0.5)*NormalmapScrollBlend.y, dPdx*NormalmapScrollBlend.y, dPdy*NormalmapScrollBlend.y).rgb - vec3(1.0);
# else
		vec3 normal = dp_texture2D(Texture_Normal, (TexCoord + vec2(0.08, 0.08)*ClientTime*NormalmapScrollBlend.x*0.5)*NormalmapScrollBlend.y).rgb - vec3(1.0);
# endif
		normal += dp_texture2D(Texture_Normal, (TexCoord + vec2(-0.06, -0.09)*ClientTime*NormalmapScrollBlend.x)*NormalmapScrollBlend.y*0.75).rgb;
		vec2 ScreenTexCoord = SafeScreenTexCoord + vec3(normalize(cast_myhalf3(normal))).xy * DistortScaleRefractReflect.zw;
	#else
		vec2 ScreenTexCoord = SafeScreenTexCoord + vec3(normalize(cast_myhalf3(offsetMappedTexture2D(Texture_Normal)) - cast_myhalf3(0.5))).xy * DistortScaleRefractReflect.zw;
	#endif

	float f = min(1.0, length(dp_texture2D(Texture_Reflection, ScreenTexCoord + vec2(0.01, 0.01)).rgb) / 0.05);
	f      *= min(1.0, length(dp_texture2D(Texture_Reflection, ScreenTexCoord + vec2(0.01, -0.01)).rgb) / 0.05);
	f      *= min(1.0, length(dp_texture2D(Texture_Reflection, ScreenTexCoord + vec2(-0.01, 0.01)).rgb) / 0.05);
	f      *= min(1.0, length(dp_texture2D(Texture_Reflection, ScreenTexCoord + vec2(-0.01, -0.01)).rgb) / 0.05);
	ScreenTexCoord = mix(SafeScreenTexCoord, ScreenTexCoord, f);
	color.rgb = mix(color.rgb, cast_myhalf3(dp_texture2D(Texture_Reflection, ScreenTexCoord)) * ReflectColor.rgb, ReflectColor.a);
#endif
#ifdef USEOCCLUDE
   color.rgb *= clamp(float(visiblepixels) / float(allpixels), 0.0, 1.0);
#endif

	dp_FragColor = vec4(color);
}
#endif

#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
