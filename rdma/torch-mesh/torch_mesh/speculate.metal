// Speculative decoding's step boundary on the device (metal-microbench docs/kernels.md#speculative-decoding): one
// stream verified at R rows a step [Leviathan, Kalman & Matias 2023; Chen et al. 2023], the accept rule and the drafter
// run by every member on the tokens every member selected, so no draft crosses and no host is in the step.
//
// The stream's state (words): step (invocations begun), p0 (row 0's position this step), length (tokens in the
// history), flags (bit 0 pending: the first step's row 0 is the previous phase's selection, appended to the history;
// bit 1 stopped: the context or the teacher is spent and the rows repeat their positions), order (the drafter's match
// this step), n (the drafts the previous step accepted), then the R tokens of this step's rows. The history is the
// stream's tokens, the prompt's and every committed one.
//
// Accept (greedy verification): row i's selected token y_i is the target's next token after row i's; the drafts d_1 ..
// d_{R-1} (rows 1 ..) are accepted while d_{i+1} = y_i, n of them; y_0 .. y_n are committed (the drafts accepted and the
// target's own next token), and the next step's row 0 is y_n at p0 + n + 1. A teacher (taught > 0 tokens by position)
// replaces y_i by the teacher's token at p0 + 1 + i (docs/measurement.md, the pair's divergence).
//
// Draft (prompt lookup [Saxena 2023]: an n-gram transition table over the stream's own tokens, the most recent earlier
// occurrence of the longest suffix of at most ORDER tokens): the scan scores each end position p <= t - 2 of the
// history's t tokens by (its match length, p), the greatest wins; its continuation is copied, extended periodically
// past the history's end (period t - 1 - p). No match, or ORDER 0 (a network drafter's rows replace them: drafter.metal
// mesh_spec_drafts): the rows repeat the last token (any token verifies exactly).
//
// Each row's step words (token, position p0 + r, length p0 + r + 1, KV-write skip 0) for the step's own apply, and the
// step's record: p0, n, the order (bit 8 stopped), the R row tokens.
#ifndef TORCH_MESH_SPECULATE
#define TORCH_MESH_SPECULATE
#include <metal_stdlib>
using namespace metal;

constant constexpr uint MESH_SPEC_STATE = 8;

template <uint R, uint ORDER, uint T>
__attribute__((always_inline)) static inline void mesh_spec_step(device uint *state, device uint *history, device const uint *selected,
                                                                 device const uint *teacher, uint taught, uint capacity, uint context,
                                                                 device uint *words, uint stride, device uint *record,
                                                                 threadgroup uint *shared, uint lane) {
    if (lane == 0) {
        uint step = state[0], p0 = state[1], length = state[2], flags = state[3], n = 0;
        if (step == 0) {
            if ((flags & 1u) != 0u) { history[length] = selected[0]; length += 1u; }
            flags &= ~1u;
        } else if ((flags & 2u) == 0u) {
            uint y[R];
            for (uint i = 0; i < R; i++)
                y[i] = taught == 0u ? selected[i] : p0 + 1u + i < taught ? teacher[p0 + 1u + i] : 0xffffffffu;
            while (n + 1u < R && state[MESH_SPEC_STATE + n + 1u] == y[n]) n++;
            if (y[n] == 0xffffffffu || length + n + 1u + R > capacity || p0 + n + 1u + R > context) {
                flags |= 2u;
                n = 0u;
            } else {
                for (uint i = 0; i <= n; i++) history[length + i] = y[i];
                length += n + 1u;
                p0 += n + 1u;
            }
        }
        state[1] = p0; state[2] = length; state[3] = flags; state[5] = n;
        shared[0] = length; shared[1] = p0; shared[2] = n; shared[3] = flags;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    const uint t = shared[0], p0 = shared[1];
    uint best = 0;
    if (R > 1 && ORDER > 0) {
        for (uint p = lane; p + 2u <= t; p += T) {
            uint m = 0;
            for (uint k = 0; k < ORDER && k <= p; k++) {
                if (history[p - k] != history[t - 1u - k]) break;
                m = k + 1u;
            }
            if (m > 0u) best = max(best, (m << 24) | p);
        }
        best = simd_max(best);
        if (lane % 32u == 0u) shared[4u + lane / 32u] = best;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0) {
            for (uint q = 1; q < T / 32u; q++) best = max(best, shared[4u + q]);
            shared[4u] = best;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        best = shared[4u];
    }
    if (lane < R) {
        const uint last = history[t - 1u];
        uint token = last;
        if (lane > 0u && best != 0u) {
            const uint p = best & 0xffffffu, period = t - 1u - p;
            token = history[p + 1u + (lane - 1u) % period];
        }
        words[lane * stride + 0u] = token;
        words[lane * stride + 1u] = p0 + lane;
        words[lane * stride + 2u] = p0 + lane + 1u;
        words[lane * stride + 3u] = 0u;
        state[MESH_SPEC_STATE + lane] = token;
        record[3u + lane] = token;
    }
    if (lane == 0) {
        record[0] = p0;
        record[1] = shared[2];
        record[2] = (best >> 24) | ((shared[3] & 2u) << 7);
        state[0] += 1u;
        state[4] = best >> 24;
    }
}
#endif
