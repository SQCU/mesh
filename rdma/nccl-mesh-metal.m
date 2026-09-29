#import <Metal/Metal.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

/* The GPU side of nccl-mesh.c.  Every buffer a program binds is a Metal buffer over exactly one
   window allocation (made once with the allocation, the one Metal object for its pages, which the
   caller's kernels bind too) or, for memory outside the window, over the host pages that hold it
   (newBufferWithBytesNoCopy on the containing pages: no copy, the program's blits read and write the
   caller's memory).  A program: one command buffer of a group on its stream's queue (the NULL stream's
   on the communicator's own), whose kernels combine, premultiply and post-divide, and whose blits copy,
   between waits for shared-event values (a region's recorded writer or reader done, a piece arrived)
   and signals (the operands staged, a piece combined, the group done). */
static const char *source =
  "#include <metal_stdlib>\n"
  "using namespace metal;\n"
  "// ncclDataType_t 0 int8 1 uint8 2 int32 3 uint32 4 int64 5 uint64 6 float16 7 float32 8 float64 9 bfloat16\n"
  "// 10 float8e4m3 11 float8e5m2; ncclRedOp_t 0 sum 1 prod 2 max 3 min.  dst and src: byte offsets into buffers 0 and 1.\n"
  "struct args { ulong dst, src, n, scalar; uint type, op, nranks, pad; };\n"
  "\n"
  "// half and the fp8 formats decoded exactly to float, a float rounded to nearest even into them (one\n"
  "// rounding; fp8 saturating as NCCL's __NV_SATFINITE); bfloat16 by its float bits\n"
  "struct small { int e, m; bool fn, saturate; };\n"
  "constant small F16 = {5, 10, false, false}, E4M3 = {4, 3, true, true}, E5M2 = {5, 2, false, true};\n"
  "static float decode(uint bits, small f) {\n"
  "  uint top = (1u << f.e) - 1, exponent = (bits >> f.m) & top, mantissa = bits & ((1u << f.m) - 1);\n"
  "  int bias = (1 << (f.e - 1)) - 1;\n"
  "  float sign = (bits >> (f.e + f.m)) & 1 ? -1.0f : 1.0f;\n"
  "  if (!f.fn && exponent == top) return mantissa ? NAN : sign * INFINITY;\n"
  "  if (f.fn && exponent == top && mantissa == (1u << f.m) - 1) return NAN;\n"
  "  if (!exponent) return sign * ldexp(float(mantissa), 1 - bias - f.m);\n"
  "  return sign * ldexp(float(mantissa | (1u << f.m)), int(exponent) - bias - f.m);\n"
  "}\n"
  "static uint encode(float x, small f) {\n"
  "  uint sign = signbit(x) ? 1u << (f.e + f.m) : 0u, top = (1u << f.e) - 1;\n"
  "  int bias = (1 << (f.e - 1)) - 1;\n"
  "  uint largest = f.fn ? (top << f.m) | ((1u << f.m) - 2) : ((top - 1) << f.m) | ((1u << f.m) - 1);\n"
  "  if (isnan(x)) return f.fn ? sign | (top << f.m) | ((1u << f.m) - 1) : sign | (top << f.m) | (1u << (f.m - 1));\n"
  "  float a = fabs(x);\n"
  "  if (isinf(a)) return f.saturate ? sign | largest : sign | (top << f.m);\n"
  "  if (a == 0.0f) return sign;\n"
  "  int exponent;\n"
  "  frexp(a, exponent);\n"
  "  exponent -= 1;\n"
  "  if (exponent < 1 - bias) exponent = 1 - bias;\n"
  "  float q = rint(ldexp(a, f.m - exponent));\n"
  "  if (q >= ldexp(1.0f, f.m + 1)) { q = ldexp(1.0f, f.m); exponent++; }\n"
  "  uint field = q < ldexp(1.0f, f.m) ? 0u : uint(exponent + bias), mantissa = uint(q) & ((1u << f.m) - 1);\n"
  "  uint bits = (field << f.m) | mantissa;\n"
  "  if (field > top || (!f.fn && field == top) || (f.fn && field == top && mantissa == (1u << f.m) - 1) || bits > largest)\n"
  "    return f.saturate ? sign | largest : sign | (top << f.m);\n"
  "  return sign | bits;\n"
  "}\n"
  "static uint bf16_encode(float x) {\n"
  "  uint u = as_type<uint>(x);\n"
  "  return isnan(x) ? (u >> 16) | 0x40u : (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;\n"
  "}\n"
  "\n"
  "// binary64, which the GPU lacks: add and multiply rounded to nearest even on its bits (Hauser's\n"
  "// SoftFloat 3e f64_add / f64_mul, softfloat_roundPackToF64), max and min by the ordered bits\n"
  "static uint clz64(ulong x) { uint high = uint(x >> 32); return high ? clz(high) : 32 + clz(uint(x)); }\n"
  "static ulong jam(ulong a, uint distance) { return distance < 63 ? a >> distance | ulong((a << (-distance & 63)) != 0) : ulong(a != 0); }\n"
  "static ulong pack(ulong sign, long exponent, ulong sig) { return (sign << 63) + (ulong(exponent) << 52) + sig; }\n"
  "static ulong quiet(ulong a, ulong b) { return ((a & 0x7fffffffffffffffUL) > 0x7ff0000000000000UL ? a : b) | 0x0008000000000000UL; }\n"
  "static ulong round_pack(ulong sign, long exponent, ulong sig) {\n"
  "  uint bits = uint(sig & 0x3ff);\n"
  "  if (ulong(exponent) >= 0x7fd) {\n"
  "    if (exponent < 0) { sig = jam(sig, uint(-exponent)); exponent = 0; bits = uint(sig & 0x3ff); }\n"
  "    else if (exponent > 0x7fd || sig + 0x200 >= 0x8000000000000000UL) return pack(sign, 0x7ff, 0);\n"
  "  }\n"
  "  sig = (sig + 0x200) >> 10;\n"
  "  if (bits == 0x200) sig &= ~ulong(1);\n"
  "  if (!sig) exponent = 0;\n"
  "  return pack(sign, exponent, sig);\n"
  "}\n"
  "static ulong norm_round_pack(ulong sign, long exponent, ulong sig) {\n"
  "  int shift = int(clz64(sig)) - 1;\n"
  "  exponent -= shift;\n"
  "  if (shift >= 10 && ulong(exponent) < 0x7fd) return pack(sign, sig ? exponent : 0, sig << (shift - 10));\n"
  "  return round_pack(sign, exponent, sig << shift);\n"
  "}\n"
  "static ulong f64_add(ulong a, ulong b) {\n"
  "  ulong sign = a >> 63;\n"
  "  long ea = long((a >> 52) & 0x7ff), eb = long((b >> 52) & 0x7ff), d = ea - eb;\n"
  "  ulong sa = a & 0x000fffffffffffffUL, sb = b & 0x000fffffffffffffUL;\n"
  "  if (sign == b >> 63) {\n"
  "    long ez; ulong sz;\n"
  "    if (!d) {\n"
  "      if (!ea) return a + sb;\n"
  "      if (ea == 0x7ff) return sa | sb ? quiet(a, b) : a;\n"
  "      ez = ea; sz = (0x0020000000000000UL + sa + sb) << 9;\n"
  "    } else {\n"
  "      sa <<= 9; sb <<= 9;\n"
  "      if (d < 0) {\n"
  "        if (eb == 0x7ff) return sb ? quiet(a, b) : pack(sign, 0x7ff, 0);\n"
  "        ez = eb; sa = ea ? sa + 0x2000000000000000UL : sa << 1; sa = jam(sa, uint(-d));\n"
  "      } else {\n"
  "        if (ea == 0x7ff) return sa ? quiet(a, b) : a;\n"
  "        ez = ea; sb = eb ? sb + 0x2000000000000000UL : sb << 1; sb = jam(sb, uint(d));\n"
  "      }\n"
  "      sz = 0x2000000000000000UL + sa + sb;\n"
  "      if (sz < 0x4000000000000000UL) { ez--; sz <<= 1; }\n"
  "    }\n"
  "    return round_pack(sign, ez, sz);\n"
  "  }\n"
  "  if (!d) {\n"
  "    if (ea == 0x7ff) return sa | sb ? quiet(a, b) : 0x7ff8000000000000UL;\n"
  "    long difference = long(sa - sb);\n"
  "    if (!difference) return 0;\n"
  "    if (ea) ea--;\n"
  "    if (difference < 0) { sign ^= 1; difference = -difference; }\n"
  "    int shift = int(clz64(ulong(difference))) - 11;\n"
  "    long ez = ea - shift;\n"
  "    if (ez < 0) { shift = int(ea); ez = 0; }\n"
  "    return pack(sign, ez, ulong(difference) << shift);\n"
  "  }\n"
  "  sa <<= 10; sb <<= 10;\n"
  "  long ez; ulong sz;\n"
  "  if (d < 0) {\n"
  "    sign ^= 1;\n"
  "    if (eb == 0x7ff) return sb ? quiet(a, b) : pack(sign, 0x7ff, 0);\n"
  "    sa += ea ? 0x4000000000000000UL : sa; sa = jam(sa, uint(-d));\n"
  "    sb |= 0x4000000000000000UL; ez = eb; sz = sb - sa;\n"
  "  } else {\n"
  "    if (ea == 0x7ff) return sa ? quiet(a, b) : a;\n"
  "    sb += eb ? 0x4000000000000000UL : sb; sb = jam(sb, uint(d));\n"
  "    sa |= 0x4000000000000000UL; ez = ea; sz = sa - sb;\n"
  "  }\n"
  "  return norm_round_pack(sign, ez - 1, sz);\n"
  "}\n"
  "static ulong f64_mul(ulong a, ulong b) {\n"
  "  ulong sign = (a ^ b) >> 63;\n"
  "  long ea = long((a >> 52) & 0x7ff), eb = long((b >> 52) & 0x7ff);\n"
  "  ulong sa = a & 0x000fffffffffffffUL, sb = b & 0x000fffffffffffffUL;\n"
  "  if (ea == 0x7ff) { if (sa || (eb == 0x7ff && sb)) return quiet(a, b); return eb | sb ? pack(sign, 0x7ff, 0) : 0x7ff8000000000000UL; }\n"
  "  if (eb == 0x7ff) { if (sb) return quiet(a, b); return ea | sa ? pack(sign, 0x7ff, 0) : 0x7ff8000000000000UL; }\n"
  "  if (!ea) { if (!sa) return pack(sign, 0, 0); int s = int(clz64(sa)) - 11; ea = 1 - s; sa <<= s; }\n"
  "  if (!eb) { if (!sb) return pack(sign, 0, 0); int s = int(clz64(sb)) - 11; eb = 1 - s; sb <<= s; }\n"
  "  long ez = ea + eb - 0x3ff;\n"
  "  sa = (sa | 0x0010000000000000UL) << 10; sb = (sb | 0x0010000000000000UL) << 11;\n"
  "  ulong a0 = sa & 0xffffffffUL, a1 = sa >> 32, b0 = sb & 0xffffffffUL, b1 = sb >> 32;\n"
  "  ulong p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;\n"
  "  ulong middle = (p00 >> 32) + (p01 & 0xffffffffUL) + (p10 & 0xffffffffUL);\n"
  "  ulong low = (p00 & 0xffffffffUL) | (middle << 32), high = p11 + (p01 >> 32) + (p10 >> 32) + (middle >> 32);\n"
  "  ulong sz = high | ulong(low != 0);\n"
  "  if (sz < 0x4000000000000000UL) { ez--; sz <<= 1; }\n"
  "  return round_pack(sign, ez, sz);\n"
  "}\n"
  "static bool f64_nan(ulong a) { return (a & 0x7fffffffffffffffUL) > 0x7ff0000000000000UL; }\n"
  "static ulong f64_key(ulong a) { return a >> 63 ? ~a : a | 0x8000000000000000UL; }\n"
  "\n"
  "// binary32 and bfloat16 by their float bits: the GPU flushes binary32 subnormals, so an operand or\n"
  "// result below 2^-101 goes exactly through binary64, rounded once to nearest even into the type\n"
  "// (m its mantissa bits: 23 binary32, 7 bfloat16); max and min by the ordered bits\n"
  "static ulong widen(uint x) {\n"
  "  ulong sign = ulong(x >> 31) << 63;\n"
  "  int exponent = int((x >> 23) & 0xff);\n"
  "  uint mantissa = x & 0x7fffffu;\n"
  "  if (exponent == 0xff) return sign | 0x7ff0000000000000UL | (ulong(mantissa) << 29);\n"
  "  if (!exponent) {\n"
  "    if (!mantissa) return sign;\n"
  "    int shift = int(clz(mantissa)) - 8;\n"
  "    exponent = 1 - shift; mantissa = (mantissa << shift) & 0x7fffffu;\n"
  "  }\n"
  "  return sign | (ulong(exponent - 127 + 1023) << 52) | (ulong(mantissa) << 29);\n"
  "}\n"
  "static uint narrow(ulong x, int m) {\n"
  "  uint sign = uint(x >> 63) << (8 + m), infinity = 0xffu << m;\n"
  "  long exponent = long((x >> 52) & 0x7ff);\n"
  "  ulong sig = x & 0x000fffffffffffffUL;\n"
  "  if (exponent == 0x7ff) return sig ? sign | infinity | (1u << (m - 1)) : sign | infinity;\n"
  "  if (!exponent) return sign;\n"
  "  sig |= 0x0010000000000000UL;\n"
  "  long e = exponent - 1023 + 127;\n"
  "  int shift = 52 - m + (e < 1 ? int(1 - e) : 0);\n"
  "  if (shift > 54) return sign;\n"
  "  ulong q = sig >> shift, rest = sig & ((1UL << shift) - 1), tie = 1UL << (shift - 1);\n"
  "  if (rest > tie || (rest == tie && (q & 1))) q++;\n"
  "  ulong bits = e >= 1 ? (ulong(e - 1) << m) + q : q;\n"
  "  return bits >= infinity ? sign | infinity : sign | uint(bits);\n"
  "}\n"
  "static bool subnormal(uint x) { return !(x & 0x7f800000u) && (x & 0x7fffffu); }\n"
  "static uint field(uint x) { return (x >> 23) & 0xffu; }\n"
  "static uint rounded(float r, int m) { return m == 23 ? as_type<uint>(r) : bf16_encode(r); }\n"
  "static uint add32(uint a, uint b, int m) {\n"
  "  if (subnormal(a) || subnormal(b) || (field(a) < 26 && field(b) < 26)) return narrow(f64_add(widen(a), widen(b)), m);\n"
  "  return rounded(as_type<float>(a) + as_type<float>(b), m);\n"
  "}\n"
  "static uint mul32(uint a, uint b, int m) {\n"
  "  if (subnormal(a) || subnormal(b) || (field(a) && field(b) && field(a) + field(b) < 129)) return narrow(f64_mul(widen(a), widen(b)), m);\n"
  "  return rounded(as_type<float>(a) * as_type<float>(b), m);\n"
  "}\n"
  "static bool nan32(uint a) { return (a & 0x7fffffffu) > 0x7f800000u; }\n"
  "static uint key32(uint a) { return a >> 31 ? ~a : a | 0x80000000u; }\n"
  "static uint f32_apply(uint a, uint b, uint op, int m) {\n"
  "  if (op == 0) return add32(a, b, m);\n"
  "  if (op == 1) return mul32(a, b, m);\n"
  "  uint x = nan32(b) ? b : nan32(a) ? a : (op == 2 ? key32(b) > key32(a) : key32(b) < key32(a)) ? b : a;\n"
  "  return m == 23 ? x : x >> 16;\n"
  "}\n"
  "\n"
  "// NCCL's FuncSum/FuncProd/FuncMax/FuncMin on each type: integers wrap; max and min of floating values\n"
  "// propagate a NaN and order -0 below +0\n"
  "static float max_of(float a, float b) { return isnan(b) || b > a || (b == a && !signbit(b) && signbit(a)) ? b : a; }\n"
  "static float min_of(float a, float b) { return isnan(b) || b < a || (b == a && signbit(b) && !signbit(a)) ? b : a; }\n"
  "static float apply(float a, float b, uint op) { return op == 0 ? a + b : op == 1 ? a * b : op == 2 ? max_of(a, b) : min_of(a, b); }\n"
  "template <typename T, typename U> static T integer(T a, T b, uint op) {\n"
  "  return op == 0 ? T(U(a) + U(b)) : op == 1 ? T(U(a) * U(b)) : op == 2 ? (b > a ? b : a) : (b < a ? b : a);\n"
  "}\n"
  "static ulong f64_apply(ulong a, ulong b, uint op) {\n"
  "  if (op == 0) return f64_add(a, b);\n"
  "  if (op == 1) return f64_mul(a, b);\n"
  "  if (f64_nan(b)) return b;\n"
  "  if (f64_nan(a)) return a;\n"
  "  return (op == 2 ? f64_key(b) > f64_key(a) : f64_key(b) < f64_key(a)) ? b : a;\n"
  "}\n"
  "#define D(T) ((device T *)(d + p.dst))\n"
  "#define S(T) ((device T *)(s + p.src))\n"
  "#define EACH for (ulong i = first; i < p.n; i += grid)\n"
  "#define GRID uint first [[thread_position_in_grid]], uint grid [[threads_per_grid]]\n"
  "\n"
  "// dst = dst op src\n"
  "kernel void combine(device uchar *d [[buffer(0)]], device uchar *s [[buffer(1)]], constant args &p [[buffer(2)]], GRID) {\n"
  "  switch (p.type) {\n"
  "  case 0: EACH D(char)[i] = integer<char, uint>(D(char)[i], S(char)[i], p.op); break;\n"
  "  case 1: EACH D(uchar)[i] = integer<uchar, uint>(D(uchar)[i], S(uchar)[i], p.op); break;\n"
  "  case 2: EACH D(int)[i] = integer<int, uint>(D(int)[i], S(int)[i], p.op); break;\n"
  "  case 3: EACH D(uint)[i] = integer<uint, uint>(D(uint)[i], S(uint)[i], p.op); break;\n"
  "  case 4: EACH D(long)[i] = integer<long, ulong>(D(long)[i], S(long)[i], p.op); break;\n"
  "  case 5: EACH D(ulong)[i] = integer<ulong, ulong>(D(ulong)[i], S(ulong)[i], p.op); break;\n"
  "  case 6: EACH D(ushort)[i] = ushort(encode(apply(decode(D(ushort)[i], F16), decode(S(ushort)[i], F16), p.op), F16)); break;\n"
  "  case 7: EACH D(uint)[i] = f32_apply(D(uint)[i], S(uint)[i], p.op, 23); break;\n"
  "  case 8: EACH D(ulong)[i] = f64_apply(D(ulong)[i], S(ulong)[i], p.op); break;\n"
  "  case 9: EACH D(ushort)[i] = ushort(f32_apply(uint(D(ushort)[i]) << 16, uint(S(ushort)[i]) << 16, p.op, 7)); break;\n"
  "  case 10: EACH D(uchar)[i] = uchar(encode(apply(decode(D(uchar)[i], E4M3), decode(S(uchar)[i], E4M3), p.op), E4M3)); break;\n"
  "  default: EACH D(uchar)[i] = uchar(encode(apply(decode(D(uchar)[i], E5M2), decode(S(uchar)[i], E5M2), p.op), E5M2)); break;\n"
  "  }\n"
  "}\n"
  "// dst = src x scalar (a premultiplication: ncclAvg on floating types, PreMulSum), one operation of the type\n"
  "kernel void premultiply(device uchar *d [[buffer(0)]], device uchar *s [[buffer(1)]], constant args &p [[buffer(2)]], GRID) {\n"
  "  const ulong k = p.scalar;\n"
  "  switch (p.type) {\n"
  "  case 0: EACH D(char)[i] = char(uint(S(char)[i]) * uint(char(k))); break;\n"
  "  case 1: EACH D(uchar)[i] = uchar(uint(S(uchar)[i]) * uint(uchar(k))); break;\n"
  "  case 2: EACH D(int)[i] = int(uint(S(int)[i]) * uint(k)); break;\n"
  "  case 3: EACH D(uint)[i] = S(uint)[i] * uint(k); break;\n"
  "  case 4: EACH D(long)[i] = long(ulong(S(long)[i]) * k); break;\n"
  "  case 5: EACH D(ulong)[i] = S(ulong)[i] * k; break;\n"
  "  case 6: EACH D(ushort)[i] = ushort(encode(decode(S(ushort)[i], F16) * decode(uint(k & 0xffff), F16), F16)); break;\n"
  "  case 7: EACH D(uint)[i] = mul32(S(uint)[i], uint(k), 23); break;\n"
  "  case 8: EACH D(ulong)[i] = f64_mul(S(ulong)[i], k); break;\n"
  "  case 9: EACH D(ushort)[i] = ushort(mul32(uint(S(ushort)[i]) << 16, uint(k & 0xffff) << 16, 7)); break;\n"
  "  case 10: EACH D(uchar)[i] = uchar(encode(decode(S(uchar)[i], E4M3) * decode(uint(k & 0xff), E4M3), E4M3)); break;\n"
  "  default: EACH D(uchar)[i] = uchar(encode(decode(S(uchar)[i], E5M2) * decode(uint(k & 0xff), E5M2), E5M2)); break;\n"
  "  }\n"
  "}\n"
  "// ncclAvg on integer types: the wrapped sum over nranks, truncated toward zero (NCCL's\n"
  "// FuncSumPostDiv::divide: the magnitude divided, the sign kept)\n"
  "template <typename T, typename U> static T divide(T x, U n) {\n"
  "  U u = x < 0 ? U(0) - U(x) : U(x), q = u / n;\n"
  "  return x < 0 ? T(U(0) - q) : T(q);\n"
  "}\n"
  "kernel void postdivide(device uchar *d [[buffer(0)]], device uchar *s [[buffer(1)]], constant args &p [[buffer(2)]], GRID) {\n"
  "  switch (p.type) {\n"
  "  case 0: EACH D(char)[i] = divide<char, uint>(D(char)[i], p.nranks); break;\n"
  "  case 1: EACH D(uchar)[i] = uchar(uint(D(uchar)[i]) / p.nranks); break;\n"
  "  case 2: EACH D(int)[i] = divide<int, uint>(D(int)[i], p.nranks); break;\n"
  "  case 3: EACH D(uint)[i] = D(uint)[i] / p.nranks; break;\n"
  "  case 4: EACH D(long)[i] = divide<long, ulong>(D(long)[i], ulong(p.nranks)); break;\n"
  "  case 5: EACH D(ulong)[i] = D(ulong)[i] / ulong(p.nranks); break;\n"
  "  default: break;\n"
  "  }\n"
  "}\n"
  "// bytes a blit does not take (an offset or length not a multiple of 4)\n"
  "kernel void copy(device uchar *d [[buffer(0)]], device uchar *s [[buffer(1)]], constant args &p [[buffer(2)]], GRID) {\n"
  "  EACH d[p.dst + i] = s[p.src + i];\n"
  "}\n"
  "// one thread's n atomic adds: the GPU kept out of its idle state (nccl_mesh_hold)\n"
  "kernel void hold(device atomic_uint *c [[buffer(0)]], constant uint &n [[buffer(1)]], uint i [[thread_position_in_grid]]) {\n"
  "  if (i) return;\n"
  "  for (uint k = 0; k < n; k++) atomic_fetch_add_explicit(c, 1u, memory_order_relaxed);\n"
  "}\n";

enum { KERNELS = 5, KERNEL_HOLD = 4 };
static struct { id<MTLDevice> device; id<MTLComputePipelineState> kernels[KERNELS]; } gpu;
static _Atomic int failed;
#define HIDDEN __attribute__((visibility("hidden")))

/* The device and the kernels; 0, or -1 with the reason in `error`. */
HIDDEN int nccl_mesh_gpu_attach(char *error,size_t size){
  @autoreleasepool {
    if(gpu.device)return 0;
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();
    if(!device){snprintf(error,size,"no Metal device");return -1;}
    MTLCompileOptions *options=[[MTLCompileOptions new] autorelease];
    if(@available(macOS 15.0,*))options.mathMode=MTLMathModeSafe;
    NSError *failure=nil;
    id<MTLLibrary> library=[[device newLibraryWithSource:@(source) options:options error:&failure] autorelease];
    static const char *const names[KERNELS]={"combine","premultiply","postdivide","copy","hold"};
    for(int k=0;library && k<KERNELS;k++){
      id<MTLFunction> function=[[library newFunctionWithName:@(names[k])] autorelease];
      if(!(gpu.kernels[k]=[device newComputePipelineStateWithFunction:function error:&failure]))library=nil;
    }
    if(!library){
      snprintf(error,size,"%s",failure.localizedDescription.UTF8String);
      for(int k=0;k<KERNELS;k++){[gpu.kernels[k] release];gpu.kernels[k]=nil;}
      [device release];
      return -1;
    }
    gpu.device=device;
    return 0;
  }
}
/* A GPU program failed (its command buffer's status an error) since the process started. */
HIDDEN int nccl_mesh_gpu_failed(void){return atomic_load(&failed);}

/* A Metal buffer over `bytes` at `pointer`, no copy: the pages holding them (from the page below
   `pointer`), `*offset` the pointer's place in it.  A window allocation's pages are its own. */
HIDDEN void *nccl_mesh_buffer(const void *pointer,size_t bytes,uint64_t *offset){
  const uintptr_t page=(uintptr_t)getpagesize(),at=(uintptr_t)pointer,first=at&~(page-1),end=(at+(bytes?bytes:1)+page-1)&~(page-1);
  *offset=at-first;
  return [gpu.device newBufferWithBytesNoCopy:(void *)first length:end-first options:MTLResourceStorageModeShared deallocator:nil];
}
HIDDEN void *nccl_mesh_queue_create(void){
  return [gpu.device newCommandQueueWithMaxCommandBufferCount:4096];
}
HIDDEN void nccl_mesh_retain(void *object){[(id)object retain];}
HIDDEN void nccl_mesh_release(void *object){[(id)object release];}
HIDDEN void *nccl_mesh_event_create(void *queue){
  @autoreleasepool {
    id<MTLDevice> device=queue?[(id<MTLCommandQueue>)queue device]:gpu.device?[gpu.device retain]:MTLCreateSystemDefaultDevice();
    if(!device)return NULL;
    id<MTLSharedEvent> event=[device newSharedEvent];
    if(!queue)[device release];
    event.signaledValue=0;
    return event;
  }
}
HIDDEN uint64_t nccl_mesh_event_value(void *event){return [(id<MTLSharedEvent>)event signaledValue];}
HIDDEN void nccl_mesh_event_signal(void *event,uint64_t value){
  id<MTLSharedEvent> shared=event;
  if(shared.signaledValue<value)shared.signaledValue=value;
}

/* ---- holds ----
   A GPU left idle for about a millisecond drops into a state it leaves slowly: on the M4 Pro a command
   buffer parked 2 ms on a shared event started 433 us (median) after the host signalled it and ran its
   combine at 340 us, against 68 us after a 200 us park, and 51 us and 160 us while one thread of another
   queue kept the GPU busy (metal-microbench output_data/perf-torch-20260929/probe, wake2).  So while a
   worker waits on the network for a part whose GPU work waits on it, holds keep the GPU busy: a hold is
   one thread's atomic adds on the holds' own queue, then a signal of the worker's hold event. */
static struct { pthread_mutex_t lock; id<MTLCommandQueue> queue; id<MTLBuffer> counter; uint32_t adds; } holding={.lock=PTHREAD_MUTEX_INITIALIZER};
/* One hold of `adds` adds committed, `event` signalled `value` once it ends. */
static void hold_commit(void *event,uint64_t value,uint32_t adds){
  @autoreleasepool {
    id<MTLCommandBuffer> buffer=[holding.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder=[buffer computeCommandEncoder];
    [encoder setComputePipelineState:gpu.kernels[KERNEL_HOLD]];
    [encoder setBuffer:holding.counter offset:0 atIndex:0];
    [encoder setBytes:&adds length:sizeof adds atIndex:1];
    [encoder dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
    [encoder endEncoding];
    if(event)[buffer encodeSignalEvent:(id<MTLSharedEvent>)event value:value];
    [buffer commit];
  }
}
/* The holds' queue made, and a hold's adds taken for about `us` microseconds: timed once on a GPU the
   first hold has woken.  0 where the GPU is not attached. */
HIDDEN uint32_t nccl_mesh_hold_adds(double us){
  pthread_mutex_lock(&holding.lock);
  if(!holding.queue && gpu.device){
    holding.queue=[gpu.device newCommandQueue];
    holding.counter=[gpu.device newBufferWithLength:64 options:MTLResourceStorageModeShared];
    double seconds=0;const uint32_t trial=1u<<16;
    for(int k=0;k<2;k++){
      @autoreleasepool {
        id<MTLCommandBuffer> buffer=[holding.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder=[buffer computeCommandEncoder];
        [encoder setComputePipelineState:gpu.kernels[KERNEL_HOLD]];
        [encoder setBuffer:holding.counter offset:0 atIndex:0];
        [encoder setBytes:&trial length:sizeof trial atIndex:1];
        [encoder dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
        [encoder endEncoding];
        [buffer commit];[buffer waitUntilCompleted];
        seconds=buffer.GPUEndTime-buffer.GPUStartTime;
      }
    }
    double adds=seconds>0?trial*(us*1e-6)/seconds:trial;
    holding.adds=adds<1024?1024:adds>(1u<<26)?(1u<<26):(uint32_t)adds;
  }
  uint32_t adds=holding.adds;
  pthread_mutex_unlock(&holding.lock);
  return adds;
}
HIDDEN void nccl_mesh_hold(void *event,uint64_t value,uint32_t adds){hold_commit(event,value,adds);}

/* ---- programs ---- */
HIDDEN void *nccl_mesh_program_begin(void *queue){
  @autoreleasepool {return [[(id<MTLCommandQueue>)queue commandBuffer] retain];}
}
HIDDEN void nccl_mesh_program_wait(void *program,void *event,uint64_t value){
  [(id<MTLCommandBuffer>)program encodeWaitForEvent:(id<MTLSharedEvent>)event value:value];
}
HIDDEN void nccl_mesh_program_signal(void *program,void *event,uint64_t value){
  [(id<MTLCommandBuffer>)program encodeSignalEvent:(id<MTLSharedEvent>)event value:value];
}
/* Kernel k (0 combine: dst op= src, 1 premultiply: dst = src x scalar, 2 postdivide: dst /= nranks, 3
   copy: n bytes) over n elements of `type`, dst a byte offset into buffer `to`, src into `from`. */
struct args { uint64_t dst,src,n,scalar; uint32_t type,op,nranks,pad; };
HIDDEN void nccl_mesh_program_kernel(void *program,int k,void *to,uint64_t dst,void *from,uint64_t src,uint64_t n,int type,int op,
  int nranks,uint64_t scalar){
  @autoreleasepool {
    struct args a={dst,src,n,scalar,(uint32_t)type,(uint32_t)op,(uint32_t)nranks,0};
    id<MTLComputeCommandEncoder> encoder=[(id<MTLCommandBuffer>)program computeCommandEncoder];
    [encoder setComputePipelineState:gpu.kernels[k]];
    [encoder setBuffer:(id<MTLBuffer>)to offset:0 atIndex:0];
    [encoder setBuffer:(id<MTLBuffer>)from offset:0 atIndex:1];
    [encoder setBytes:&a length:sizeof a atIndex:2];
    NSUInteger width=gpu.kernels[k].maxTotalThreadsPerThreadgroup<256?gpu.kernels[k].maxTotalThreadsPerThreadgroup:256;
    [encoder dispatchThreads:MTLSizeMake(n<(1u<<20)?(NSUInteger)n:(1u<<20),1,1) threadsPerThreadgroup:MTLSizeMake(width,1,1)];
    [encoder endEncoding];
  }
}
/* `bytes` from offset src of buffer `from` to offset dst of buffer `to`: a blit, or the copy kernel
   where an offset or the length is not a multiple of 4 (a blit's rule on macOS). */
HIDDEN void nccl_mesh_program_copy(void *program,void *to,uint64_t dst,void *from,uint64_t src,uint64_t bytes){
  if((dst|src|bytes)&3){nccl_mesh_program_kernel(program,3,to,dst,from,src,bytes,0,0,0,0);return;}
  @autoreleasepool {
    id<MTLBlitCommandEncoder> encoder=[(id<MTLCommandBuffer>)program blitCommandEncoder];
    [encoder copyFromBuffer:(id<MTLBuffer>)from sourceOffset:src toBuffer:(id<MTLBuffer>)to destinationOffset:dst size:bytes];
    [encoder endEncoding];
  }
}
/* `done(argument, failed)` once the GPU has run the command buffer (a failed one marks the process's
   GPU failed, so no host wait on its values goes on): the library's own program, or a caller's that a
   stream's deferred programs were encoded into (ncclMeshStreamEncodeWait). */
HIDDEN void nccl_mesh_program_handler(void *program,void (*done)(void *,int),void *argument){
  [(id<MTLCommandBuffer>)program addCompletedHandler:^(id<MTLCommandBuffer> ran){
    int error=ran.status==MTLCommandBufferStatusError;
    if(error){atomic_store(&failed,1);fprintf(stderr,"nccl-mesh: a GPU program failed: %s\n",ran.error.localizedDescription.UTF8String);}
    done(argument,error);
  }];
}
/* Committed, with `done` as above. */
HIDDEN void nccl_mesh_program_commit(void *program,void (*done)(void *,int),void *argument){
  nccl_mesh_program_handler(program,done,argument);
  [(id<MTLCommandBuffer>)program commit];
  [(id<MTLCommandBuffer>)program release];
}
