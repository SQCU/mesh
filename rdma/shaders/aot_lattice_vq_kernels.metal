#include <metal_stdlib>
using namespace metal;

inline uint hash_u32(uint x) {
    x ^= x >> 16u;
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    x ^= x >> 16u;
    return x;
}

inline float u01(uint seed, uint idx) {
    return float(hash_u32(seed ^ (idx * 0x9e3779b9u)) >> 8u) * (1.0f / 16777216.0f);
}

inline void round_d8(const thread float x[8], thread float z[8]) {
    int parity = 0;
    int worst = 0;
    float worst_err = -1.0f;
    float worst_step = 0.0f;
    for (int i = 0; i < 8; ++i) {
        float zi = rint(x[i]);
        z[i] = zi;
        parity ^= (int(zi) & 1);
        float d = x[i] - zi;
        float ad = fabs(d);
        if (ad > worst_err) {
            worst_err = ad;
            worst = i;
            worst_step = (d >= 0.0f) ? 1.0f : -1.0f;
        }
    }
    if (parity != 0) {
        z[worst] += worst_step;
    }
}

inline uint quantize_e8(const thread float x[8], thread float q[8]) {
    float z0[8], z1[8], xs[8];
    round_d8(x, z0);
    for (int i = 0; i < 8; ++i) xs[i] = x[i] - 0.5f;
    round_d8(xs, z1);
    float d0 = 0.0f, d1 = 0.0f;
    for (int i = 0; i < 8; ++i) {
        float e0 = x[i] - z0[i];
        float e1 = x[i] - (z1[i] + 0.5f);
        d0 += e0 * e0;
        d1 += e1 * e1;
    }
    if (d0 <= d1) {
        for (int i = 0; i < 8; ++i) q[i] = z0[i];
        return 0u;
    } else {
        for (int i = 0; i < 8; ++i) q[i] = z1[i] + 0.5f;
        return 1u;
    }
}

inline void sample_e8_dither(uint seed, uint vec_id, thread float u[8]) {
    float y[8], qy[8];
    for (int i = 0; i < 8; ++i) {
        y[i] = u01(seed, vec_id * 8u + uint(i)) * 2.0f;
    }
    quantize_e8(y, qy);
    for (int i = 0; i < 8; ++i) u[i] = y[i] - qy[i];
}

kernel void e8_dither_encode(
    device const half* X [[buffer(0)]],
    device int8_t* out_z [[buffer(1)]],
    device uint8_t* out_coset [[buffer(2)]],
    constant uint& num_vecs [[buffer(3)]],
    constant float& inv_delta [[buffer(4)]],
    constant uint& seed [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= num_vecs) return;
    float u[8], x[8], q[8];
    sample_e8_dither(seed, gid, u);
    device const half* xp = X + ulong(gid) * 8u;
    for (int i = 0; i < 8; ++i) {
        x[i] = float(xp[i]) * inv_delta + u[i];
    }
    uint coset = quantize_e8(x, q);
    out_coset[gid] = uint8_t(coset);
    device int8_t* zp = out_z + ulong(gid) * 8u;
    float shift = coset ? 0.5f : 0.0f;
    for (int i = 0; i < 8; ++i) {
        int zi = clamp(int(rint(q[i] - shift)), -127, 127);
        zp[i] = int8_t(zi);
    }
}

kernel void e8_dither_decode(
    device const int8_t* in_z [[buffer(0)]],
    device const uint8_t* in_coset [[buffer(1)]],
    device half* out_X [[buffer(2)]],
    constant uint& num_vecs [[buffer(3)]],
    constant float& delta [[buffer(4)]],
    constant uint& seed [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= num_vecs) return;
    float u[8];
    sample_e8_dither(seed, gid, u);
    float shift = in_coset[gid] ? 0.5f : 0.0f;
    device const int8_t* zp = in_z + ulong(gid) * 8u;
    device half* yp = out_X + ulong(gid) * 8u;
    for (int i = 0; i < 8; ++i) {
        float qv = float(zp[i]) + shift;
        yp[i] = half((qv - u[i]) * delta);
    }
}

kernel void aot_rvq4_encode(
    device const half4* X [[buffer(0)]],
    device const half4* C1 [[buffer(1)]],
    device const half4* C2 [[buffer(2)]],
    device uchar2* out_idx [[buffer(3)]],
    constant uint& num_vecs4 [[buffer(4)]],
    constant float& inv_std [[buffer(5)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float4 tg_c1[256];
    threadgroup float4 tg_c2[256];
    if (tid < 256u) {
        tg_c1[tid] = float4(C1[tid]);
        tg_c2[tid] = float4(C2[tid]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (gid >= num_vecs4) return;

    float4 x = float4(X[gid]) * inv_std;
    uint best1 = 0u;
    float bd1 = 1e30f;
    for (uint k = 0u; k < 256u; ++k) {
        float4 d = x - tg_c1[k];
        float d2 = dot(d, d);
        if (d2 < bd1) { bd1 = d2; best1 = k; }
    }
    float4 rem = x - tg_c1[best1];
    uint best2 = 0u;
    float bd2 = 1e30f;
    for (uint k = 0u; k < 256u; ++k) {
        float4 d = rem - tg_c2[k];
        float d2 = dot(d, d);
        if (d2 < bd2) { bd2 = d2; best2 = k; }
    }
    out_idx[gid] = uchar2(uchar(best1), uchar(best2));
}

kernel void aot_rvq4_decode(
    device const uchar2* in_idx [[buffer(0)]],
    device const half4* C1 [[buffer(1)]],
    device const half4* C2 [[buffer(2)]],
    device half4* out_X [[buffer(3)]],
    constant uint& num_vecs4 [[buffer(4)]],
    constant float& res_std [[buffer(5)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup half4 tg_c1[256];
    threadgroup half4 tg_c2[256];
    if (tid < 256u) {
        tg_c1[tid] = C1[tid];
        tg_c2[tid] = C2[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (gid >= num_vecs4) return;

    uchar2 idx = in_idx[gid];
    float4 rec = (float4(tg_c1[idx.x]) + float4(tg_c2[idx.y])) * res_std;
    out_X[gid] = half4(rec);
}
