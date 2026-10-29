// Speculative decoding's step boundary on the device (metal-microbench docs/kernels.md#speculative-decoding): one
// stream verified at R rows a step [Leviathan, Kalman & Matias 2023; Chen et al. 2023], the accept rule and the drafter
// run by every member on the tokens every member selected, so no draft crosses and no host is in the step.
//
// The stream's state (words): step (invocations begun), p0 (row 0's position this step), length (tokens in the
// history), flags (bit 0 pending: the first step's row 0 is the previous phase's selection, appended to the history;
// bit 1 stopped: the context or the teacher is spent and the rows repeat their positions; bit 2 given: this step's rows
// replay the history), order (the drafter's match this step), n (the drafts the previous step accepted), given (the
// history's positions below it are given: a step whose R rows all lie below it takes them from the history, accepts
// and appends nothing and moves p0 on by R, and leaves step at 0, so a program whose drafter reads the target's own
// rows computes them for a prompt's last positions before its first draft: metal-microbench litert_program.m), the
// stream's table row (0 on a pair's members; a served stream's own row), then the R tokens of this step's rows. The history is the stream's tokens, the prompt's and every committed one.
//
// Accept (verification against the target's draws): row i's selected token y_i is the target's own draw after row i's
// (its selection at temperature 1, keyed by row i's position: argmax.metal mesh_gumbel), so a row computed again draws the
// same token; the drafts d_1 .. d_{R-1} (rows 1 ..) are accepted while d_{i+1} = y_i, n of them; y_0 .. y_n are committed
// (the drafts accepted and the target's own next draw: exact samples of the target, the tokens a decode of one row a step
// draws with the same seed), and the next step's row 0 is y_n at p0 + n + 1. A teacher (taught > 0 tokens by position)
// replaces y_i by the teacher's token at p0 + 1 + i (docs/measurement.md, the pair's divergence).
//
// Draft (prompt lookup [Saxena 2023]: an n-gram transition table over the stream's own tokens, the most recent earlier
// occurrence of the longest suffix of at most ORDER tokens): the scan scores each end position p <= t - 2 of the
// history's t tokens by (its match length, p), the greatest wins; its continuation is copied, extended periodically
// past the history's end (period t - 1 - p). No match, or ORDER 0 (a network drafter's rows replace them: drafter.metal
// mesh_spec_drafts): the rows repeat the last token (any token verifies exactly).
//
// Each row's descriptor words (token, position p0 + r, the stream's table row, word 7: stream 0, live, writing its KV, selecting; metal-microbench
// docs/serving.md#the-batch-descriptor) for the step's own apply, and the step's record: p0, n, the order (bit 8 stopped, bit 9 given),
// the R row tokens.
#ifndef TORCH_MESH_SPECULATE
#define TORCH_MESH_SPECULATE
#include <metal_stdlib>
using namespace metal;

constant constexpr uint MESH_SPEC_STATE = MESH_SPEC_STATE_WORDS;

template <uint R, uint ORDER, uint T>
__attribute__((always_inline)) static inline void mesh_spec_step(device uint *state, device uint *history, device const uint *selected,
                                                                 device const uint *teacher, uint taught, uint capacity, uint context,
                                                                 device uint *words, uint stride, device uint *record,
                                                                 threadgroup uint *shared, uint lane) {
    if (lane == 0) {
        uint step = state[0], p0 = state[1], length = state[2], flags = state[3] & ~4u, n = 0;
        if (p0 + R <= state[6]) {
            flags |= 4u;
        } else if (step == 0) {
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
    const bool given = (shared[3] & 4u) != 0u;
    uint best = 0;
    if (R > 1 && ORDER > 0 && !given) {
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
        uint token = given ? history[p0 + lane] : last;
        if (!given && lane > 0u && best != 0u) {
            const uint p = best & 0xffffffu, period = t - 1u - p;
            token = history[p + 1u + (lane - 1u) % period];
        }
        words[lane * stride + 0u] = token;
        words[lane * stride + 1u] = p0 + lane;
        words[lane * stride + 2u] = state[7];
        words[lane * stride + 3u] = 7u;
        state[MESH_SPEC_STATE + lane] = token;
        record[3u + lane] = token;
    }
    if (lane == 0) {
        record[0] = p0;
        record[1] = shared[2];
        record[2] = (best >> 24) | ((shared[3] & 6u) << 7);
        state[0] += given ? 0u : 1u;
        state[4] = best >> 24;
        if (given) state[1] = p0 + R;
    }
}

// Speculative speculative decoding's cache probed at the step's start (metal-microbench
// docs/kernels.md#speculative-speculative-decoding; Kumar, Dao & May 2026, SSD and Saguaro, arXiv:2603.03251, Alg. 1): after
// mesh_spec_step's accept, the step's outcome (n of the previous step's drafts accepted, state word 5, and the target's own
// token after them, row 0's, state word 8) against the N slots the walking member's fan-out filled in the step before
// (drafter.metal mesh_saguaro_keys: slot i the outcome (slots[i], keys[i]), its chain chains[i R ..] the block drafted
// from it, row 0 its key): a hit's chain is this step's rows, a miss leaves the rows the step drafted (the prompt lookup's:
// plain speculative decoding, the backup, never worse than it: Cor 8). The cache serves the step after the one that filled
// it (keys[N] + 1, the step count after this step's accept); a stopped or given step takes none. `drafts` the step's R
// rows for the draft point's broadcast (drafter.metal mesh_spec_drafts writes them into every member's rows); `hit` (a word
// an invocation) the slot + 1, 0 a miss.
template <uint R, uint N>
kernel void mesh_saguaro_probe(device const uint *state [[buffer(0)]], device const uint *keys [[buffer(1)]],
                               device const uint *slots [[buffer(2)]], device const uint *chains [[buffer(3)]],
                               device uint *drafts [[buffer(4)]], device uint *hit [[buffer(5)]], uint r [[thread_position_in_grid]]) {
    if (r >= R) return;
    const uint n = state[5], y = state[MESH_SPEC_STATE];
    const bool live = (state[3] & 6u) == 0u && keys[N] + 1u == state[0];
    uint found = 0u;
    for (uint i = 0; live && i < N; i++) {
        if (slots[i] == n && keys[i] == y) {
            found = i + 1u;
            break;
        }
    }
    drafts[r] = found != 0u ? chains[(found - 1u) * R + r] : state[MESH_SPEC_STATE + r];
    if (r == 0u) hit[0] = found;
}

// The step as a kernel, one threadgroup of T lanes: metal-microbench's engine instantiates it (decode_crossings.swift
// specStepCommand), LiteRT's program by MESH_SPEC_ROWS and MESH_SPEC_ORDER (litert_program.m).
template <uint R, uint ORDER, uint T>
kernel void mesh_spec_steps(device uint *state [[buffer(0)]], device uint *history [[buffer(1)]], device const uint *selected [[buffer(2)]],
                            device const uint *teacher [[buffer(3)]], device uint *words [[buffer(4)]], device uint *record [[buffer(5)]],
                            constant uint &taught [[buffer(6)]], constant uint &capacity [[buffer(7)]], constant uint &context [[buffer(8)]],
                            constant uint &stride [[buffer(9)]], uint lane [[thread_index_in_threadgroup]]) {
    threadgroup uint shared[4 + T / 32];
    mesh_spec_step<R, ORDER, T>(state, history, selected, teacher, taught, capacity, context, words, stride, record, shared, lane);
}
#ifdef MESH_SPEC_ROWS
template [[host_name("mesh_spec_steps_rows")]] [[kernel]] decltype(mesh_spec_steps<MESH_SPEC_ROWS, MESH_SPEC_ORDER, MESH_SPEC_THREADS>)
    mesh_spec_steps<MESH_SPEC_ROWS, MESH_SPEC_ORDER, MESH_SPEC_THREADS>;
#endif
#endif
