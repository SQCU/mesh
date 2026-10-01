#import <Metal/Metal.h>
#include "mesh.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

/* The GPU side of nccl-mesh.c.  A call's program is a run of dispatches in one serial compute encoder of a
   command buffer (the caller's, for a deferred stream, or the communicator's own queue's), so each dispatch runs
   after the ones before it.  The network and the GPU meet in mapped memory: the program writes each message's
   descriptor into a stream's ring and its ready word last (mesh.h mesh_ring), system-coherent and fenced at system
   scope, and polls the word the bridge stores as a message's last request completes, with system-coherent loads and
   system-scope fences, as the prepared programs' crossings do (metal-microbench metal_recording.m).  A sequence
   number is the ring's counter the GPU keeps (one word a ring, advanced by the post kernel in stream order), so a
   recording of a program replays as it ran.  What a SEND reads the program stores system-coherent first (a kernel's
   plain stores reach the NIC only when its command buffer completes: on the M5 a word published mid-command-buffer
   after 16 MB of plain stores found 2,041,216 of 4,194,304 words stale on the host, none with system-coherent
   stores, metal-microbench output_data/handoffs-20260929/probe/coh-m5.jsonl); what the NIC wrote it loads
   system-coherent.  A wait ends once its word is set, the message's sequence + 1 or the session's cancellation
   (UINT64_MAX), or once it has polled its bound (MESH_REMOTE_POLLS, 2^20 unset), a leaky maybe (docs/standards.md
   S2, S3): it then sets its stream's give-up words (the flag, its step, where a post stopped) and ends, and every
   kernel of the stream's programs after it returns at once while the flag is set, so the command buffer ends; the
   host encodes the program again from that step (nccl-mesh.c).  A post waits for its ring's room the same way, before
   it takes a sequence number.  No kernel here waits without a bound: a GPU's work outlives its process and a kernel
   polling without end holds the GPU from every other process (2026-10-01 14:03: WindowServer's watchdog killed it
   three times behind this library's unbounded wait).  The reductions are the tag pre-revert-20261001's kernels,
   verbatim but for their predicates, which are now the give-up gate. */
static const char *source =
  "#pragma METAL internals : enable\n"
  "#include <metal_stdlib>\n"
  "using namespace metal;\n"
  "#define FENCE atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, static_cast<thread_scope>(3))\n"
  "#define SYS volatile coherent(system) device\n"
  "// ncclDataType_t 0 int8 1 uint8 2 int32 3 uint32 4 int64 5 uint64 6 float16 7 float32 8 float64 9 bfloat16\n"
  "// 10 float8e4m3 11 float8e5m2; ncclRedOp_t 0 sum 1 prod 2 max 3 min.  dst and src: byte offsets into buffers 0 and 1;\n"
  "// width: a copy's unit (16, 4 or 1 bytes), received: its source the NIC wrote; published: its stores\n"
  "// system-coherent and fenced (a SEND reads them), else plain (the GPU reads them); fresh: a combine's first operand\n"
  "// is buffer 3 at byte offset `other` (the caller's send buffer, read in place), not the destination.\n"
  "struct args { ulong dst, src, n, scalar; uint type, op, nranks, width, received, published, fresh, want; ulong other, pred; };\n"
  "// the give-up gate: buffer 4 is the region, p.pred its stream's give-up flag (nccl-mesh.c)\n"
  "#define GATE if (*(device const ulong *)(q + p.pred)) return;\n"
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
  "// D: the operand read (or, fresh, the send buffer); O: the operand stored, system-coherent; W: stored plain;\n"
  "// R: a piece the NIC wrote, loaded system-coherent; S: a source the GPU or the host wrote before the program\n"
  "#define D(T) ((device T *)(p.fresh ? x + p.other : d + p.dst))\n"
  "#define W(T) ((device T *)(d + p.dst))\n"
  "#define O(T) ((SYS T *)(d + p.dst))\n"
  "#define R(T) ((SYS T *)(s + p.src))\n"
  "#define S(T) ((device T *)(s + p.src))\n"
  "#define EACH for (ulong i = first; i < p.n; i += grid)\n"
  "#define EACH_N(n) for (ulong i = first; i < (n); i += grid)\n"
  "#define GRID uint first [[thread_position_in_grid]], uint grid [[threads_per_grid]]\n"
  "\n"
  "// dst = dst op src (fresh: send op src), src a received piece; stored OUT (O published, W plain)\n"
  "#define COMBINE(OUT) switch (p.type) {\\\n"
  "  case 0: EACH OUT(char)[i] = integer<char, uint>(D(char)[i], R(char)[i], p.op); break;\\\n"
  "  case 1: EACH OUT(uchar)[i] = integer<uchar, uint>(D(uchar)[i], R(uchar)[i], p.op); break;\\\n"
  "  case 2: EACH OUT(int)[i] = integer<int, uint>(D(int)[i], R(int)[i], p.op); break;\\\n"
  "  case 3: EACH OUT(uint)[i] = integer<uint, uint>(D(uint)[i], R(uint)[i], p.op); break;\\\n"
  "  case 4: EACH OUT(long)[i] = integer<long, ulong>(D(long)[i], R(long)[i], p.op); break;\\\n"
  "  case 5: EACH OUT(ulong)[i] = integer<ulong, ulong>(D(ulong)[i], R(ulong)[i], p.op); break;\\\n"
  "  case 6: EACH OUT(ushort)[i] = ushort(encode(apply(decode(D(ushort)[i], F16), decode(R(ushort)[i], F16), p.op), F16)); break;\\\n"
  "  case 7: EACH OUT(uint)[i] = f32_apply(D(uint)[i], R(uint)[i], p.op, 23); break;\\\n"
  "  case 8: EACH OUT(ulong)[i] = f64_apply(D(ulong)[i], R(ulong)[i], p.op); break;\\\n"
  "  case 9: EACH OUT(ushort)[i] = ushort(f32_apply(uint(D(ushort)[i]) << 16, uint(R(ushort)[i]) << 16, p.op, 7)); break;\\\n"
  "  case 10: EACH OUT(uchar)[i] = uchar(encode(apply(decode(D(uchar)[i], E4M3), decode(R(uchar)[i], E4M3), p.op), E4M3)); break;\\\n"
  "  default: EACH OUT(uchar)[i] = uchar(encode(apply(decode(D(uchar)[i], E5M2), decode(R(uchar)[i], E5M2), p.op), E5M2)); break;\\\n"
  "  }\n"
  "kernel void combine(device uchar *d [[buffer(0)]], device uchar *s [[buffer(1)]], constant args &p [[buffer(2)]],\n"
  "                    device uchar *x [[buffer(3)]], device uchar *q [[buffer(4)]], GRID) {\n"
  "  GATE\n"
  "  if (p.published) { COMBINE(O) FENCE; } else { COMBINE(W) }\n"
  "}\n"
  "// dst = src x scalar (a premultiplication: ncclAvg on floating types, PreMulSum), one operation of the type\n"
  "kernel void premultiply(device uchar *d [[buffer(0)]], device uchar *s [[buffer(1)]], constant args &p [[buffer(2)]],\n"
  "                        device uchar *q [[buffer(4)]], GRID) {\n"
  "  GATE\n"
  "  const ulong k = p.scalar;\n"
  "  switch (p.type) {\n"
  "  case 0: EACH O(char)[i] = char(uint(S(char)[i]) * uint(char(k))); break;\n"
  "  case 1: EACH O(uchar)[i] = uchar(uint(S(uchar)[i]) * uint(uchar(k))); break;\n"
  "  case 2: EACH O(int)[i] = int(uint(S(int)[i]) * uint(k)); break;\n"
  "  case 3: EACH O(uint)[i] = S(uint)[i] * uint(k); break;\n"
  "  case 4: EACH O(long)[i] = long(ulong(S(long)[i]) * k); break;\n"
  "  case 5: EACH O(ulong)[i] = S(ulong)[i] * k; break;\n"
  "  case 6: EACH O(ushort)[i] = ushort(encode(decode(S(ushort)[i], F16) * decode(uint(k & 0xffff), F16), F16)); break;\n"
  "  case 7: EACH O(uint)[i] = mul32(S(uint)[i], uint(k), 23); break;\n"
  "  case 8: EACH O(ulong)[i] = f64_mul(S(ulong)[i], k); break;\n"
  "  case 9: EACH O(ushort)[i] = ushort(mul32(uint(S(ushort)[i]) << 16, uint(k & 0xffff) << 16, 7)); break;\n"
  "  case 10: EACH O(uchar)[i] = uchar(encode(decode(S(uchar)[i], E4M3) * decode(uint(k & 0xff), E4M3), E4M3)); break;\n"
  "  default: EACH O(uchar)[i] = uchar(encode(decode(S(uchar)[i], E5M2) * decode(uint(k & 0xff), E5M2), E5M2)); break;\n"
  "  }\n"
  "  FENCE;\n"
  "}\n"
  "// ncclAvg on integer types: the wrapped sum over nranks, truncated toward zero (NCCL's\n"
  "// FuncSumPostDiv::divide: the magnitude divided, the sign kept)\n"
  "template <typename T, typename U> static T divide(T x, U n) {\n"
  "  U u = x < 0 ? U(0) - U(x) : U(x), q = u / n;\n"
  "  return x < 0 ? T(U(0) - q) : T(q);\n"
  "}\n"
  "kernel void postdivide(device uchar *d [[buffer(0)]], device uchar *s [[buffer(1)]], constant args &p [[buffer(2)]],\n"
  "                       device uchar *q [[buffer(4)]], GRID) {\n"
  "  GATE\n"
  "  switch (p.type) {\n"
  "  case 0: EACH W(char)[i] = divide<char, uint>(W(char)[i], p.nranks); break;\n"
  "  case 1: EACH W(uchar)[i] = uchar(uint(W(uchar)[i]) / p.nranks); break;\n"
  "  case 2: EACH W(int)[i] = divide<int, uint>(W(int)[i], p.nranks); break;\n"
  "  case 3: EACH W(uint)[i] = W(uint)[i] / p.nranks; break;\n"
  "  case 4: EACH W(long)[i] = divide<long, ulong>(W(long)[i], ulong(p.nranks)); break;\n"
  "  case 5: EACH W(ulong)[i] = W(ulong)[i] / ulong(p.nranks); break;\n"
  "  default: break;\n"
  "  }\n"
  "}\n"
  "// n units of `width` bytes (16: uint4, 4: uint, 1: uchar), loaded system-coherent where the NIC wrote them\n"
  "// (received), stored system-coherent where published\n"
  "#define COPY(T) EACH { T v = p.received ? ((SYS T *)(s + p.src))[i] : ((device T *)(s + p.src))[i];\\\n"
  "  if (p.published) ((SYS T *)(d + p.dst))[i] = v; else ((device T *)(d + p.dst))[i] = v; }\n"
  "kernel void copy(device uchar *d [[buffer(0)]], device uchar *s [[buffer(1)]], constant args &p [[buffer(2)]],\n"
  "                 device uchar *q [[buffer(4)]], GRID) {\n"
  "  GATE\n"
  "  if (p.width == 16) COPY(uint4) else if (p.width == 4) COPY(uint) else COPY(uchar)\n"
  "  if (p.published) FENCE;\n"
  "}\n"
  "// A post's, a wait's, a publish's and a clear's control: n records, the rings' entries R, the op's step (kept in\n"
  "// the give-up words where it gives up), the give-up words' byte offset in the region, the record a post starts at,\n"
  "// the polls a wait makes before it gives up.\n"
  "struct control { ulong n, entries, step, giveup, first, polls; };\n"
  "static void give_up(SYS ulong *g, ulong step, ulong at) { g[1] = step; g[2] = at; FENCE; g[0] = 1; FENCE; }\n"
  "// The messages of one post (nccl-mesh.c), records of 6: its ring and counter (byte offsets in the region), its window\n"
  "// offset and bytes, its ring's R words and where its sequence is kept (byte offsets in the region).  Its sequence k\n"
  "// is the counter; once the bridge has taken message k - R (the ring's taken past it: the entry free) the entry\n"
  "// k mod R is written, ready = k + 1 last, then k kept and the counter k + 1, so a post that gives up while it\n"
  "// waits for room has taken nothing and starts again at that record.\n"
  "kernel void post(device uchar *region [[buffer(0)]], constant ulong *list [[buffer(1)]], constant control &c [[buffer(2)]],\n"
  "                 uint i [[thread_position_in_grid]]) {\n"
  "  if (i) return;\n"
  "  SYS ulong *g = (SYS ulong *)(region + c.giveup);\n"
  "  if (g[0]) return;\n"
  "  FENCE;\n"
  "  for (ulong m = c.first; m < c.n; m++) {\n"
  "    constant ulong *r = list + 6 * m;\n"
  "    SYS ulong *counter = (SYS ulong *)(region + r[1]);\n"
  "    const ulong k = *counter;\n"
  "    SYS ulong *ring = (SYS ulong *)(region + r[0]);\n"
  "    for (ulong polls = 0; k >= c.entries && ring[0] <= k - c.entries; polls++) {\n"
  "      if (polls >= c.polls) { give_up(g, c.step, m); return; }\n"
  "      FENCE;\n"
  "    }\n"
  "    SYS ulong *entry = (SYS ulong *)(region + r[0] + 128 + (k & (c.entries - 1)) * 32);\n"
  "    entry[1] = r[2]; entry[2] = r[3]; entry[3] = r[4] + (k & (c.entries - 1)) * 8;\n"
  "    FENCE;\n"
  "    entry[0] = k + 1;\n"
  "    *(SYS ulong *)(region + r[5]) = k;\n"
  "    *counter = k + 1;\n"
  "    FENCE;\n"
  "  }\n"
  "}\n"
  "// One thread waits for each listed message, pairs (where its sequence k is kept, its ring's words): its word, of its\n"
  "// ring's R, set to at least k + 1 (the bridge's k + 1, or the session's cancellation); past c.polls polls it gives up.\n"
  "kernel void wait(device uchar *region [[buffer(0)]], constant ulong *list [[buffer(1)]], constant control &c [[buffer(2)]],\n"
  "                 uint i [[thread_position_in_grid]]) {\n"
  "  if (i) return;\n"
  "  SYS ulong *g = (SYS ulong *)(region + c.giveup);\n"
  "  if (g[0]) return;\n"
  "  for (ulong m = 0; m < c.n; m++) {\n"
  "    const ulong k = *(SYS ulong *)(region + list[2 * m]);\n"
  "    SYS ulong *word = (SYS ulong *)(region + list[2 * m + 1] + (k & (c.entries - 1)) * 8);\n"
  "    for (ulong polls = 0;; polls++) {\n"
  "      FENCE;\n"
  "      if (*word > k) break;\n"
  "      if (polls >= c.polls) { give_up(g, c.step, 0); return; }\n"
  "    }\n"
  "  }\n"
  "  FENCE;\n"
  "}\n"
  "// a word (c.first, a byte offset in the region) set to c.n after the dispatches before it, unless they gave up\n"
  "kernel void publish(device uchar *region [[buffer(0)]], constant control &c [[buffer(1)]], uint i [[thread_position_in_grid]]) {\n"
  "  if (i) return;\n"
  "  if (*(SYS ulong *)(region + c.giveup)) return;\n"
  "  FENCE;\n"
  "  *(SYS ulong *)(region + c.first) = c.n;\n"
  "  FENCE;\n"
  "}\n"
  "// the give-up words cleared: a program encoded again from where it gave up starts with it\n"
  "kernel void clear(device uchar *region [[buffer(0)]], constant control &c [[buffer(1)]], uint i [[thread_position_in_grid]]) {\n"
  "  if (i) return;\n"
  "  SYS ulong *g = (SYS ulong *)(region + c.giveup);\n"
  "  FENCE;\n"
  "  g[0] = 0; g[1] = 0; g[2] = 0;\n"
  "  FENCE;\n"
  "}\n";

enum { KERNEL_COMBINE, KERNEL_PREMULTIPLY, KERNEL_POSTDIVIDE, KERNEL_COPY, KERNEL_POST, KERNEL_WAIT, KERNEL_PUBLISH, KERNEL_CLEAR, KERNELS };
struct control { uint64_t n,entries,step,giveup,first,polls; };
struct args { uint64_t dst,src,n,scalar; uint32_t type,op,nranks,width,received,published,fresh,want; uint64_t other,pred; };
static struct { id<MTLDevice> device; id<MTLComputePipelineState> kernels[KERNELS]; } gpu;
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
    static const char *const names[KERNELS]={"combine","premultiply","postdivide","copy","post","wait","publish","clear"};
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

/* A Metal buffer over `bytes` at page-aligned `pointer`, no copy (a slab's, or a caller's allocation made with
   `options`, `gone(argument)` once it is deallocated: NULL, none). */
HIDDEN void *nccl_mesh_buffer(const void *pointer,size_t bytes,uint64_t options,void (*gone)(void *),void *argument){
  return [gpu.device newBufferWithBytesNoCopy:(void *)pointer length:bytes options:(MTLResourceOptions)options
    deallocator:gone?^(void *at,NSUInteger length){(void)at;(void)length;gone(argument);}:nil];
}
/* A Metal buffer over host memory outside the window: the pages holding `bytes` at `pointer`, `*offset` the
   pointer's place in it. */
HIDDEN void *nccl_mesh_buffer_over(const void *pointer,size_t bytes,uint64_t *offset){
  const uintptr_t page=(uintptr_t)getpagesize(),at=(uintptr_t)pointer,first=at&~(page-1),end=(at+(bytes?bytes:1)+page-1)&~(page-1);
  *offset=at-first;
  return [gpu.device newBufferWithBytesNoCopy:(void *)first length:end-first options:MTLResourceStorageModeShared deallocator:nil];
}
HIDDEN void *nccl_mesh_queue_create(void){return [gpu.device newCommandQueue];}
HIDDEN void nccl_mesh_release(void *object){[(id)object release];}
HIDDEN void *nccl_mesh_buffer_contents(void *buffer){return [(id<MTLBuffer>)buffer contents];}

/* ---- programs: a command buffer's serial compute encoder, each dispatch after the ones before it ---- */
HIDDEN void *nccl_mesh_command_buffer(void *queue){
  @autoreleasepool {return [[(id<MTLCommandQueue>)queue commandBuffer] retain];}
}
HIDDEN void *nccl_mesh_encoder(void *commandBuffer){
  @autoreleasepool {return [[(id<MTLCommandBuffer>)commandBuffer computeCommandEncoder] retain];}
}
HIDDEN void nccl_mesh_encoder_end(void *encoder){[(id<MTLComputeCommandEncoder>)encoder endEncoding];[(id)encoder release];}
HIDDEN void nccl_mesh_commit_wait(void *commandBuffer,char *error,size_t size){
  id<MTLCommandBuffer> command=commandBuffer;
  [command commit];[command waitUntilCompleted];
  if(command.status==MTLCommandBufferStatusError)snprintf(error,size,"%s",command.error.localizedDescription.UTF8String);
  else if(size)error[0]=0;
  [command release];
}
/* Kernel k (0 combine: dst op= src, or with `other` dst = other op src; 1 premultiply: dst = src x scalar; 2
   postdivide: dst /= nranks) over n elements of `type`, dst a byte offset into buffer `to`, src into `from`; it returns
   at once while the give-up flag at byte `giveup` of `region` is set. */
HIDDEN void nccl_mesh_kernel(void *encoder,int k,void *to,uint64_t dst,void *from,uint64_t src,uint64_t n,int type,int op,
  int nranks,uint64_t scalar,int published,void *other,uint64_t at,void *region,uint64_t giveup){
  if(!n)return;
  @autoreleasepool {
    id<MTLComputeCommandEncoder> e=encoder;
    struct args a={.dst=dst,.src=src,.n=n,.scalar=scalar,.type=(uint32_t)type,.op=(uint32_t)op,.nranks=(uint32_t)nranks,.width=1,
      .published=(uint32_t)(published!=0),.fresh=(uint32_t)(other!=NULL),.other=at,.pred=giveup};
    [e setComputePipelineState:gpu.kernels[k]];
    [e setBuffer:(id<MTLBuffer>)to offset:0 atIndex:0];
    [e setBuffer:(id<MTLBuffer>)from offset:0 atIndex:1];
    [e setBytes:&a length:sizeof a atIndex:2];
    [e setBuffer:(id<MTLBuffer>)(other?other:to) offset:0 atIndex:3];
    [e setBuffer:(id<MTLBuffer>)region offset:0 atIndex:4];
    NSUInteger width=gpu.kernels[k].maxTotalThreadsPerThreadgroup<256?gpu.kernels[k].maxTotalThreadsPerThreadgroup:256;
    uint64_t threads=n<(1u<<20)?n:(1u<<20);
    [e dispatchThreads:MTLSizeMake((NSUInteger)threads,1,1) threadsPerThreadgroup:MTLSizeMake(width,1,1)];
  }
}
/* `bytes` from offset src of `from` to offset dst of `to`, stored system-coherent and fenced where a SEND reads
   them (`published`), loaded system-coherent where the NIC wrote them (`received`), in the widest unit the offsets
   and length allow; gated as a kernel is. */
HIDDEN void nccl_mesh_copy(void *encoder,void *to,uint64_t dst,void *from,uint64_t src,uint64_t bytes,int received,int published,
  void *region,uint64_t giveup){
  if(!bytes)return;
  @autoreleasepool {
    id<MTLComputeCommandEncoder> e=encoder;
    const uint32_t width=!((dst|src|bytes)&15)?16:!((dst|src|bytes)&3)?4:1;
    struct args a={.dst=dst,.src=src,.n=bytes/width,.width=width,.received=(uint32_t)(received!=0),.published=(uint32_t)(published!=0),.pred=giveup};
    [e setComputePipelineState:gpu.kernels[KERNEL_COPY]];
    [e setBuffer:(id<MTLBuffer>)to offset:0 atIndex:0];
    [e setBuffer:(id<MTLBuffer>)from offset:0 atIndex:1];
    [e setBytes:&a length:sizeof a atIndex:2];
    [e setBuffer:(id<MTLBuffer>)region offset:0 atIndex:4];
    uint64_t threads=a.n<(1u<<20)?a.n:(1u<<20);
    [e dispatchThreads:MTLSizeMake((NSUInteger)threads,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
  }
}
/* A one-thread kernel (post or wait) over `count` records of `width` words, its control set: the records as bytes up
   to setBytes' 4 KB, else from `*held`, a buffer made at the first encoding and kept by the caller. */
static int list_kernel(id<MTLComputeCommandEncoder> e,int k,void *region,const uint64_t *list,size_t words,const struct control *c,void **held){
  const size_t bytes=(words?words:1)*sizeof *list;
  [e setComputePipelineState:gpu.kernels[k]];
  [e setBuffer:(id<MTLBuffer>)region offset:0 atIndex:0];
  if(bytes<=4096)[e setBytes:list length:bytes atIndex:1];
  else{
    if(!*held && !(*held=[gpu.device newBufferWithBytes:list length:bytes options:MTLResourceStorageModeShared]))return -1;
    [e setBuffer:(id<MTLBuffer>)*held offset:0 atIndex:1];
  }
  [e setBytes:c length:sizeof *c atIndex:2];
  [e dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
  return 0;
}
HIDDEN int nccl_mesh_post(void *encoder,void *region,const uint64_t *list,uint64_t n,uint64_t entries,uint64_t step,uint64_t giveup,
  uint64_t first,uint64_t polls,void **held){
  struct control c={n,entries,step,giveup,first,polls};
  @autoreleasepool {return list_kernel(encoder,KERNEL_POST,region,list,6*n,&c,held);}
}
HIDDEN int nccl_mesh_wait(void *encoder,void *region,const uint64_t *list,uint64_t n,uint64_t entries,uint64_t step,uint64_t giveup,
  uint64_t polls,void **held){
  struct control c={n,entries,step,giveup,0,polls};
  @autoreleasepool {return list_kernel(encoder,KERNEL_WAIT,region,list,2*n,&c,held);}
}
/* The word at byte `at` of `region` set to `value` (unless the dispatches before it gave up), or (clear) the give-up
   words cleared. */
static void control_kernel(void *encoder,int k,void *region,const struct control *c){
  @autoreleasepool {
    id<MTLComputeCommandEncoder> e=encoder;
    [e setComputePipelineState:gpu.kernels[k]];
    [e setBuffer:(id<MTLBuffer>)region offset:0 atIndex:0];
    [e setBytes:c length:sizeof *c atIndex:1];
    [e dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
  }
}
HIDDEN void nccl_mesh_publish(void *encoder,void *region,uint64_t at,uint64_t value,uint64_t giveup){
  struct control c={.n=value,.giveup=giveup,.first=at};
  control_kernel(encoder,KERNEL_PUBLISH,region,&c);
}
HIDDEN void nccl_mesh_clear(void *encoder,void *region,uint64_t giveup){
  struct control c={.giveup=giveup};
  control_kernel(encoder,KERNEL_CLEAR,region,&c);
}
/* The buffers a program's commands use but do not bind (a slab a SEND reads or a RECV lands in: the NIC's), declared
   to the encoder, so Metal orders the queue's later work on them after it. */
HIDDEN void nccl_mesh_use(void *encoder,void *buffer){
  [(id<MTLComputeCommandEncoder>)encoder useResource:(id<MTLBuffer>)buffer usage:MTLResourceUsageRead|MTLResourceUsageWrite];
}
/* A completion handler on `commandBuffer` (an id<MTLCommandBuffer> not yet committed): `done(argument, error)` once it
   has completed, `error` its failure or NULL. */
HIDDEN void nccl_mesh_on_completion(void *commandBuffer,void (*done)(void *,const char *),void *argument){
  [(id<MTLCommandBuffer>)commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> command){
    done(argument,command.status==MTLCommandBufferStatusError?command.error.localizedDescription.UTF8String:NULL);
  }];
}
HIDDEN void nccl_mesh_commit(void *commandBuffer){[(id<MTLCommandBuffer>)commandBuffer commit];}
