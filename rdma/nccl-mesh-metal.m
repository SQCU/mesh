#import <Metal/Metal.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

/* The GPU side of nccl-mesh.c.  Every buffer a program binds is a Metal buffer over exactly one
   window allocation (made once with the allocation, the one Metal object for its pages, which the
   caller's kernels bind too) or, for memory outside the window, over the host pages that hold it
   (newBufferWithBytesNoCopy on the containing pages: no copy, the program's kernels read and write the
   caller's memory).  A program: one command buffer of a group on its stream's queue (the NULL stream's
   on the communicator's own), or a deferred stream's commands in the caller's, whose kernels copy,
   combine, premultiply and post-divide, wait on completion words and publish them, between waits for
   shared-event values (a region's recorded writer or reader done, another stream's work) and signals
   (the group done).
     The network and the GPU meet in mapped memory, as the prepared programs' crossings do (metal-microbench
   metal_recording.m MetalRemoteSpin, MetalPayloadRead, mesh_coherent_store): the bridge stores each
   request's end into its completion word (mesh.h mesh_net_request), which a kernel polls with
   system-coherent loads and system-scope fences.  That spin is leaky: it polls at most `spin` times (a
   performance setting, MESH_NCCL_SPIN, never a failure) and then writes not-yet (its flag word stays 0, the
   call's leaked word 1) and returns; the work that consumes what it waited for is dispatched twice, once
   predicated on the flag (it runs now) and once after the program's gate predicated on its absence (it runs
   there instead).  A gate is where a command buffer ends and the next is committed only once an event the host
   signals says every word the work after it needs is set (nccl-mesh.c's dispatcher, a caller's ncclMeshGate_t),
   so no command buffer waits on the network: Metal ends one that waits past its watchdog, running kernels or
   waiting on an event at its start (about 5 s), and a Metal 4 queue's wait before a commit passes silently some
   4 s after it is submitted (metal-microbench output_data/maybe-20260930/probe).  What the NIC wrote is loaded system-coherent, and what a later
   SEND reads a kernel stores system-coherent, so the NIC sees it once the kernel's publication word is seen;
   a spin's declared buffers order the queue's later work after it.  A store that is not reaches another agent
   only when its command buffer completes: on the M5 a word published mid-command-buffer after 16 MB of plain
   stores found 2,041,216 of their 4,194,304 words stale on the host, none with system-coherent stores
   (metal-microbench output_data/handoffs-20260929/probe/coh-m5.jsonl).  coherent(system) needs Metal's
   internals pragma, as the recorder's generated sources begin. */
static const char *source =
  "#pragma METAL internals : enable\n"
  "#include <metal_stdlib>\n"
  "using namespace metal;\n"
  "#define FENCE atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, static_cast<thread_scope>(3))\n"
  "#define SYS volatile coherent(system) device\n"
  "// ncclDataType_t 0 int8 1 uint8 2 int32 3 uint32 4 int64 5 uint64 6 float16 7 float32 8 float64 9 bfloat16\n"
  "// 10 float8e4m3 11 float8e5m2; ncclRedOp_t 0 sum 1 prod 2 max 3 min.  dst and src: byte offsets into buffers 0 and 1;\n"
  "// width: a copy's unit (16, 4 or 1 bytes), received: its source the NIC wrote; published: its stores\n"
  "// system-coherent and fenced, a word published after them (a SEND reads them), else plain (the GPU reads them);\n"
  "// fresh: a combine's first operand is buffer 3 at byte offset `other` (the caller's send buffer, read in place),\n"
  "// not the destination; pred (0: none): 1 + the byte offset in buffer 4 of the flag word a spin wrote, the kernel\n"
  "// running only where that flag is `want` (1: the spin saw its words; 0: it did not, the gate passed since).\n"
  "struct args { ulong dst, src, n, scalar; uint type, op, nranks, width, received, published, fresh, want; ulong other, pred; };\n"
  "#define PRED(q, at, want) if ((at) && ((*(SYS ulong *)((q) + (at) - 1) != 0ul) != ((want) != 0))) return;\n"
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
  "  PRED(q, p.pred, p.want)\n"
  "  if (p.published) { COMBINE(O) FENCE; } else { COMBINE(W) }\n"
  "}\n"
  "// dst = src x scalar (a premultiplication: ncclAvg on floating types, PreMulSum), one operation of the type\n"
  "kernel void premultiply(device uchar *d [[buffer(0)]], device uchar *s [[buffer(1)]], constant args &p [[buffer(2)]],\n"
  "                        device uchar *q [[buffer(4)]], GRID) {\n"
  "  PRED(q, p.pred, p.want)\n"
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
  "  PRED(q, p.pred, p.want)\n"
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
  "  PRED(q, p.pred, p.want)\n"
  "  if (p.width == 16) COPY(uint4) else if (p.width == 4) COPY(uint) else COPY(uchar)\n"
  "  if (p.published) FENCE;\n"
  "}\n"
  "// The leaky spin: one thread waits for each listed word (a byte offset into buffer 0) to reach its value: a\n"
  "// system-coherent load about every 0.5 us (a dependent chain of 128 multiply-adds between them: a wait needs\n"
  "// microseconds, and each load is a trip to the system's coherence point, which the NIC's writes share), a\n"
  "// system-scope fence every 16 (one before every load slowed a concurrent host copy 3x, one every 64 did not and\n"
  "// saw a host store within 3 us: metal-microbench output_data/handoffs-20260929/probe/spin*), and one once it is\n"
  "// seen, before the loads it orders.  It polls at most `bound` times in all, a performance setting: every word\n"
  "// seen, its flag word is set (the work predicated on it runs now); else its call's leaked word is set, so the\n"
  "// call's later spins give up at once, and the work runs after the gate instead; the first to give up counts\n"
  "// itself in buffer 2's word (a statistic).  list: n, bound, flag, leaked (byte offsets into buffer 0), then n\n"
  "// (offset, value) pairs.\n"
  "kernel void wait(device uchar *w [[buffer(0)]], constant ulong *list [[buffer(1)]], device atomic_uint *given [[buffer(2)]],\n"
  "                 uint i [[thread_position_in_grid]]) {\n"
  "  if (i) return;\n"
  "  SYS ulong *flag = (SYS ulong *)(w + list[2]), *leaked = (SYS ulong *)(w + list[3]);\n"
  "  ulong total = 0;\n"
  "  float pace = float(list[0]);\n"
  "  bool seen = !*leaked, fresh = seen;\n"
  "  for (ulong k = 0; seen && k < list[0]; k++)\n"
  "    for (SYS ulong *word = (SYS ulong *)(w + list[4 + 2 * k]);; total++) {\n"
  "      if (*word >= list[5 + 2 * k]) break;\n"
  "      if (total >= list[1]) { seen = false; break; }\n"
  "      for (uint a = 0; a < 128; a++) pace = fma(pace, 0.999f, 1e-3f);\n"
  "      if (total & 15) continue;\n"
  "      FENCE;\n"
  "    }\n"
  "  if (pace < 0.0f) seen = false;\n"
  "  FENCE;\n"
  "  if (seen) *flag = 1ul; else { *leaked = 1ul; if (fresh) atomic_fetch_add_explicit(given, 1u, memory_order_relaxed); }\n"
  "  FENCE;\n"
  "}\n"
  "// a counted all-to-all's own rows (nccl-mesh.c counted_copy): a: dst and src (byte offsets into buffers 0 and 1),\n"
  "// the counts' and the landed segments' (into buffers 2 and 3, int64), row bytes, the copy's unit, ranks, this rank,\n"
  "// the caller's landed counts' (into buffer 5; all ones: none), then a rank each: its sent segment's first and length,\n"
  "// its received segment's first and length; the offsets the rows of the ranks before this one (sent: its counts;\n"
  "// received: their segments, which the NIC wrote); each rank's segment for this one also written to buffer 5; then\n"
  "// its predicate (args' pred and want, buffer 6)\n"
  "kernel void counted(device uchar *d [[buffer(0)]], device uchar *s [[buffer(1)]], device uchar *c [[buffer(2)]],\n"
  "                    device uchar *g [[buffer(3)]], constant ulong *a [[buffer(4)]], device uchar *out [[buffer(5)]],\n"
  "                    device uchar *q [[buffer(6)]], GRID) {\n"
  "  PRED(q, a[9 + 4 * a[6]], a[10 + 4 * a[6]])\n"
  "  device const long *counts = (device const long *)(c + a[2]);\n"
  "  volatile coherent(system) device const long *landed = (volatile coherent(system) device const long *)(g + a[3]);\n"
  "  ulong from = 0, to = 0, own = 0;\n"
  "  for (ulong q = 0; q < a[6]; q++) {\n"
  "    ulong sent = 0, got = 0;\n"
  "    for (ulong i = 0; i < a[10 + 4 * q]; i++) sent += ulong(counts[a[9 + 4 * q] + i]);\n"
  "    if (a[8] != ~0ul && first < a[12 + 4 * q])\n"
  "      ((device long *)(out + a[8]))[a[11 + 4 * q] + first] = q == a[7] ? counts[a[9 + 4 * q] + first] : landed[a[11 + 4 * q] + first];\n"
  "    if (q == a[7]) { own = sent; continue; }\n"
  "    if (own || q > a[7]) continue;\n"
  "    for (ulong i = 0; i < a[12 + 4 * q]; i++) got += ulong(landed[a[11 + 4 * q] + i]);\n"
  "    from += sent; to += got;\n"
  "  }\n"
  "  const ulong units = own * a[4] / a[5], at = a[0] + to * a[4], fromat = a[1] + from * a[4];\n"
  "  if (a[5] == 16) EACH_N(units) ((device uint4 *)(d + at))[i] = ((device uint4 *)(s + fromat))[i];\n"
  "  else if (a[5] == 4) EACH_N(units) ((device uint *)(d + at))[i] = ((device uint *)(s + fromat))[i];\n"
  "  else EACH_N(units) (d + at)[i] = (s + fromat)[i];\n"
  "}\n"
  "// n words from a byte offset into buffer 0 set to 0 after the dispatches before it (a persistent call's\n"
  "// completion words once its program has waited for them all: its next iteration's waits see only its own)\n"
  "kernel void zero(device uchar *w [[buffer(0)]], constant ulong *a [[buffer(1)]], uint i [[thread_position_in_grid]]) {\n"
  "  if (i) return;\n"
  "  FENCE;\n"
  "  for (ulong k = 0; k < a[1]; k++) ((SYS ulong *)(w + a[0]))[k] = 0ul;\n"
  "  FENCE;\n"
  "}\n"
  "// a word (a byte offset into buffer 0) set to `value`, system-coherent, after the dispatches before it; a: the\n"
  "// offset, the value, then its predicate (args' pred and want, buffer 2)\n"
  "kernel void publish(device uchar *w [[buffer(0)]], constant ulong *a [[buffer(1)]], device uchar *q [[buffer(2)]],\n"
  "                    uint i [[thread_position_in_grid]]) {\n"
  "  if (i) return;\n"
  "  PRED(q, a[2], a[3])\n"
  "  FENCE;\n"
  "  *(SYS ulong *)(w + a[0]) = a[1];\n"
  "  FENCE;\n"
  "}\n";

enum { KERNELS = 8, KERNEL_COPY = 3, KERNEL_WAIT = 4, KERNEL_PUBLISH = 5, KERNEL_COUNTED = 6, KERNEL_ZERO = 7 };
/* A kernel's arguments (the source's struct args): byte offsets dst and src into buffers 0 and 1, `other`
   into buffer 3 (a fresh combine's first operand), `pred` its predicate's flag in buffer 4 (1 + the offset; 0:
   none) and the flag's value it runs at, `want`. */
struct args { uint64_t dst,src,n,scalar; uint32_t type,op,nranks,width,received,published,fresh,want; uint64_t other,pred; };
static struct { id<MTLDevice> device; id<MTLComputePipelineState> kernels[KERNELS]; uint64_t spin; } gpu;
#define HIDDEN __attribute__((visibility("hidden")))

/* The device and the kernels; 0, or -1 with the reason in `error`.  The spin's bound, polls a spin makes in all
   (MESH_NCCL_SPIN, default 65536: tens of milliseconds, about 0.4 to 1.2 us a poll as measured, metal-microbench
   output_data/handoffs-20260929/t1): past it the waiting work runs after its gate, so it sets only how long the GPU
   polls before it parks there.  At most SPIN_MOST (about 1 s at the slowest poll): a call spins once in full, and
   the spins of a command buffer's calls stay under Metal's watchdog. */
#define SPIN_MOST (UINT64_C(1)<<20)
HIDDEN int nccl_mesh_gpu_attach(char *error,size_t size){
  @autoreleasepool {
    if(gpu.device)return 0;
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();
    if(!device){snprintf(error,size,"no Metal device");return -1;}
    MTLCompileOptions *options=[[MTLCompileOptions new] autorelease];
    if(@available(macOS 15.0,*))options.mathMode=MTLMathModeSafe;
    NSError *failure=nil;
    id<MTLLibrary> library=[[device newLibraryWithSource:@(source) options:options error:&failure] autorelease];
    static const char *const names[KERNELS]={"combine","premultiply","postdivide","copy","wait","publish","counted","zero"};
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
    const char *spin=getenv("MESH_NCCL_SPIN");
    gpu.spin=spin && strtoull(spin,NULL,10)?strtoull(spin,NULL,10):UINT64_C(65536);
    if(gpu.spin>SPIN_MOST)gpu.spin=SPIN_MOST;
    gpu.device=device;
    return 0;
  }
}


/* A Metal buffer over `bytes` at `pointer`, no copy: the pages holding them (from the page below
   `pointer`), `*offset` the pointer's place in it.  A window allocation's pages are its own. */
HIDDEN void *nccl_mesh_buffer(const void *pointer,size_t bytes,uint64_t *offset){
  const uintptr_t page=(uintptr_t)getpagesize(),at=(uintptr_t)pointer,first=at&~(page-1),end=(at+(bytes?bytes:1)+page-1)&~(page-1);
  *offset=at-first;
  return [gpu.device newBufferWithBytesNoCopy:(void *)first length:end-first options:MTLResourceStorageModeShared deallocator:nil];
}
/* As nccl_mesh_buffer, the buffer made with `options` for a caller that owns it: `gone(argument)` once it is
   deallocated. */
HIDDEN void *nccl_mesh_buffer_owned(const void *pointer,size_t bytes,uint64_t options,void (*gone)(void *),void *argument){
  return [gpu.device newBufferWithBytesNoCopy:(void *)pointer length:bytes options:(MTLResourceOptions)options
                                  deallocator:^(void *at,NSUInteger length){(void)at;(void)length;gone(argument);}];
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

/* ---- programs ----
   A program's consecutive kernels share one serial compute encoder, so each dispatch runs after the ones
   before it (a wait before the combine it guards, a publication after the stores it publishes); an event's
   wait or signal, a commit, or the caller's own encoding ends it (nccl_mesh_program_end).  Programs are
   encoded one at a time (nccl-mesh.c's stream_lock). */
static struct { void *program; id<MTLComputeCommandEncoder> encoder; } encoding;
static id<MTLComputeCommandEncoder> encoder_of(void *program){
  if(encoding.program!=program || !encoding.encoder){
    if(encoding.encoder){[encoding.encoder endEncoding];[encoding.encoder release];}
    encoding.program=program;
    encoding.encoder=[[(id<MTLCommandBuffer>)program computeCommandEncoder] retain];
  }
  return encoding.encoder;
}
HIDDEN void nccl_mesh_program_end(void *program){
  if(encoding.encoder && (!program || encoding.program==program)){[encoding.encoder endEncoding];[encoding.encoder release];encoding.encoder=nil;encoding.program=NULL;}
}
HIDDEN void *nccl_mesh_program_begin(void *queue){
  @autoreleasepool {return [[(id<MTLCommandQueue>)queue commandBuffer] retain];}
}
HIDDEN void nccl_mesh_program_wait(void *program,void *event,uint64_t value){
  nccl_mesh_program_end(program);
  [(id<MTLCommandBuffer>)program encodeWaitForEvent:(id<MTLSharedEvent>)event value:value];
}
HIDDEN void nccl_mesh_program_signal(void *program,void *event,uint64_t value){
  nccl_mesh_program_end(program);
  [(id<MTLCommandBuffer>)program encodeSignalEvent:(id<MTLSharedEvent>)event value:value];
}
/* Kernel k (0 combine: dst op= src, or with `other` dst = other op src, other at byte offset `at` of that
   buffer; 1 premultiply: dst = src x scalar, 2 postdivide: dst /= nranks, 3 copy: n units of `width`
   bytes) over n elements of `type`, dst a byte offset into buffer `to`, src into `from`; `pred` (nil: none) the
   buffer holding its predicate's flag at `a.pred` - 1; at most `grid` threads (0: one an element, at most
   2^20). */
static void dispatch(void *program,int k,void *to,void *from,void *other,void *pred,uint64_t grid,struct args a){
  @autoreleasepool {
    id<MTLComputeCommandEncoder> encoder=encoder_of(program);
    [encoder setComputePipelineState:gpu.kernels[k]];
    [encoder setBuffer:(id<MTLBuffer>)to offset:0 atIndex:0];
    [encoder setBuffer:(id<MTLBuffer>)from offset:0 atIndex:1];
    [encoder setBytes:&a length:sizeof a atIndex:2];
    if(k==0)[encoder setBuffer:(id<MTLBuffer>)(other?other:to) offset:0 atIndex:3];
    [encoder setBuffer:(id<MTLBuffer>)(pred?pred:to) offset:0 atIndex:4];
    NSUInteger width=gpu.kernels[k].maxTotalThreadsPerThreadgroup<256?gpu.kernels[k].maxTotalThreadsPerThreadgroup:256;
    uint64_t threads=a.n<(1u<<20)?a.n:(1u<<20);
    if(grid && threads>grid)threads=grid;
    [encoder dispatchThreads:MTLSizeMake((NSUInteger)threads,1,1) threadsPerThreadgroup:MTLSizeMake(width,1,1)];
  }
}
HIDDEN void nccl_mesh_program_kernel(void *program,int k,void *to,uint64_t dst,void *from,uint64_t src,uint64_t n,int type,int op,
  int nranks,uint64_t scalar,int published,void *other,uint64_t at,void *pred,uint64_t pred_at,int want,uint64_t grid){
  if(n)dispatch(program,k,to,from,other,pred,grid,(struct args){.dst=dst,.src=src,.n=n,.scalar=scalar,.type=(uint32_t)type,.op=(uint32_t)op,
    .nranks=(uint32_t)nranks,.width=1,.published=(uint32_t)(published!=0),.fresh=(uint32_t)(other!=NULL),.other=at,
    .pred=pred?pred_at+1:0,.want=(uint32_t)(want!=0)});
}
/* `bytes` from offset src of buffer `from` to offset dst of buffer `to`, stored system-coherent where a word
   published after them lets a SEND read them (`published`), loaded system-coherent where the NIC wrote them
   (`received`): the copy kernel in the widest unit the offsets and length allow; predicated and capped as a
   kernel's. */
HIDDEN void nccl_mesh_program_copy(void *program,void *to,uint64_t dst,void *from,uint64_t src,uint64_t bytes,int received,int published,
  void *pred,uint64_t pred_at,int want,uint64_t grid){
  const uint32_t width=!((dst|src|bytes)&15)?16:!((dst|src|bytes)&3)?4:1;
  if(bytes)dispatch(program,KERNEL_COPY,to,from,NULL,pred,grid,(struct args){.dst=dst,.src=src,.n=bytes/width,.width=width,
    .received=(uint32_t)(received!=0),.published=(uint32_t)(published!=0),.pred=pred?pred_at+1:0,.want=(uint32_t)(want!=0)});
}
/* The leaky spin for words of `buffer` to reach their values (`at`: n pairs of a byte offset and a value), after
   the dispatches before it and before those after it: at most the spin's bound of polls, then its flag word (at
   byte offset `flag`) set where it saw them all, else its call's leaked word (`leaked`).  The buffers the waited
   requests read or write (`touch`) are declared used, so Metal orders a later command buffer's work on them after
   the spin, as it did after an event wait: it orders a queue's command buffers only where they share a buffer. */
HIDDEN int nccl_mesh_program_spin(void *program,void *buffer,const uint64_t *at,uint32_t n,uint64_t flag,uint64_t leaked,void *given,
  void *const *touch,int ntouch,void **held){
  @autoreleasepool {
    const size_t bytes=(4+2*(size_t)n)*sizeof(uint64_t);
    uint64_t *list=malloc(bytes);
    if(!list)return -1;
    list[0]=n;list[1]=gpu.spin;list[2]=flag;list[3]=leaked;
    memcpy(list+4,at,2*(size_t)n*sizeof *at);
    id<MTLComputeCommandEncoder> encoder=encoder_of(program);
    for(int t=0;t<ntouch;t++)if(touch[t])[encoder useResource:(id<MTLBuffer>)touch[t] usage:MTLResourceUsageRead|MTLResourceUsageWrite];
    [encoder setComputePipelineState:gpu.kernels[KERNEL_WAIT]];
    [encoder setBuffer:(id<MTLBuffer>)buffer offset:0 atIndex:0];
    [encoder setBuffer:(id<MTLBuffer>)(given?given:buffer) offset:0 atIndex:2];
    /* a list past setBytes' 4 KB in a buffer of its own, retained by the caller's recording until it is freed (a
       caller's command buffer may not retain what it binds) */
    if(bytes<=4096)[encoder setBytes:list length:bytes atIndex:1];
    else{
      id<MTLBuffer> made=[gpu.device newBufferWithBytes:list length:bytes options:MTLResourceStorageModeShared];
      if(!made){free(list);return -1;}
      [encoder setBuffer:made offset:0 atIndex:1];
      if(held)*held=made;else [made autorelease];
    }
    [encoder dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
    free(list);
    return 0;
  }
}
/* The word at byte offset `at` of `buffer` set to `value`, system-coherent, after the dispatches before it;
   predicated as a kernel is. */
HIDDEN void nccl_mesh_program_publish(void *program,void *buffer,uint64_t at,uint64_t value,void *pred,uint64_t pred_at,int want){
  @autoreleasepool {
    const uint64_t a[4]={at,value,pred?pred_at+1:0,(uint64_t)(want!=0)};
    id<MTLComputeCommandEncoder> encoder=encoder_of(program);
    [encoder setComputePipelineState:gpu.kernels[KERNEL_PUBLISH]];
    [encoder setBuffer:(id<MTLBuffer>)buffer offset:0 atIndex:0];
    [encoder setBytes:a length:sizeof a atIndex:1];
    [encoder setBuffer:(id<MTLBuffer>)(pred?pred:buffer) offset:0 atIndex:2];
    [encoder dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
  }
}
/* `n` words of `buffer` from byte offset `at` set to 0 after the dispatches before it. */
HIDDEN void nccl_mesh_program_zero(void *program,void *buffer,uint64_t at,uint64_t n){
  if(!n)return;
  @autoreleasepool {
    const uint64_t a[2]={at,n};
    id<MTLComputeCommandEncoder> encoder=encoder_of(program);
    [encoder setComputePipelineState:gpu.kernels[KERNEL_ZERO]];
    [encoder setBuffer:(id<MTLBuffer>)buffer offset:0 atIndex:0];
    [encoder setBytes:a length:sizeof a atIndex:1];
    [encoder dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
  }
}
/* The counted kernel (a counted all-to-all's own rows): `a` its arguments (na of them, its predicate's two
   appended), over `units` copy units at most (and `grid` threads at most, 0: 2^20). */
HIDDEN void nccl_mesh_program_counted(void *program,void *to,void *from,void *counts,void *got,void *landed,const uint64_t *a,uint32_t na,uint64_t units,
  void *pred,uint64_t grid){
  if(!units)return;
  @autoreleasepool {
    id<MTLComputeCommandEncoder> encoder=encoder_of(program);
    [encoder setComputePipelineState:gpu.kernels[KERNEL_COUNTED]];
    [encoder setBuffer:(id<MTLBuffer>)to offset:0 atIndex:0];
    [encoder setBuffer:(id<MTLBuffer>)from offset:0 atIndex:1];
    [encoder setBuffer:(id<MTLBuffer>)counts offset:0 atIndex:2];
    [encoder setBuffer:(id<MTLBuffer>)got offset:0 atIndex:3];
    [encoder setBytes:a length:na*sizeof *a atIndex:4];
    [encoder setBuffer:(id<MTLBuffer>)landed offset:0 atIndex:5];
    [encoder setBuffer:(id<MTLBuffer>)(pred?pred:got) offset:0 atIndex:6];
    uint64_t threads=units<(1u<<20)?units:(1u<<20);
    if(grid && threads>grid)threads=grid;
    [encoder dispatchThreads:MTLSizeMake((NSUInteger)threads,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
  }
}
/* `done(argument, status)` once the GPU has run the command buffer, its status as the device reported it: 0 it ran;
   1 it failed by no fault of its own (Metal's watchdog, a victim of another process's fault or of a GPU recovery,
   memory another process held: the same commands run again from the same inputs give the same result); 2 it
   faulted itself (a page fault, an invalid resource, a stack overflow, access refused, the device gone).  The
   library's own program, or a caller's that a stream's deferred programs were encoded into
   (ncclMeshStreamEncodeWait). */
HIDDEN void nccl_mesh_program_handler(void *program,void (*done)(void *,int),void *argument){
  [(id<MTLCommandBuffer>)program addCompletedHandler:^(id<MTLCommandBuffer> ran){
    int status=0;
    if(ran.status==MTLCommandBufferStatusError){
      const NSInteger code=ran.error.code;
      status=code==MTLCommandBufferErrorPageFault || code==MTLCommandBufferErrorInvalidResource || code==MTLCommandBufferErrorStackOverflow ||
             code==MTLCommandBufferErrorNotPermitted || code==MTLCommandBufferErrorAccessRevoked || code==MTLCommandBufferErrorDeviceRemoved?2:1;
      fprintf(stderr,"nccl-mesh: a GPU program failed (%s): %s\n",status==1?"to be run again":"its own fault",ran.error.localizedDescription.UTF8String);
    }
    done(argument,status);
  }];
}
/* Committed (its encoding ended by the caller), with `done` as above. */
HIDDEN void nccl_mesh_program_commit(void *program,void (*done)(void *,int),void *argument){
  nccl_mesh_program_handler(program,done,argument);
  [(id<MTLCommandBuffer>)program commit];
  [(id<MTLCommandBuffer>)program release];
}
/* A command buffer never committed, released (a program run again from its start drops what it had not run). */
HIDDEN void nccl_mesh_program_drop(void *program){
  nccl_mesh_program_end(program);
  [(id<MTLCommandBuffer>)program release];
}

