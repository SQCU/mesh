// The GPU side of the words mesh-metal.m prepares (design/prepared-machine.md M04, M07, M10, M12, M30), defined once:
// a kernel that waits on a word, reads a landed slot or releases a cell includes this file and calls these from its own
// source, and no source splices them into another kernel's text (metal-microbench docs/acceptance-20261014.md A40).
// libnccl-mesh's kernels include it, and so does every kernel that hands words to another agent (the host, a Neural
// Engine partner's thread). install.sh installs it with the library ($MESH_ROOT/rdma/mesh.metal).
#ifndef MESH_METAL
#define MESH_METAL
#pragma METAL internals : enable
#include <metal_stdlib>

// The word at `word` once it holds at least `target`, with a system-scope fence before each load (without it the M4
// did not observe the NIC's or the host's store: metal-microbench d8566be). The wait has no bound of its own: whoever
// owns the word ends it (M12's cancellation, a host watcher's store), since every spun-on word is in a canceller's set
// (design/recovery.md Q16). A reader orders its later loads after the observation with a system-scope fence of its own,
// after the barrier that gives the observation to its threads.
template <typename W>
__attribute__((always_inline)) static inline W mesh_wait(volatile coherent(system) device const W *word, ulong target) {
    W seen;
    do {
        metal::atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst, static_cast<metal::thread_scope>(3));
        seen = *word;
    } while (seen < target);
    return seen;
}

// A landing's completion word (M07) once it holds 1 plus the invocation's cycle (M30; 0 in a one-pass program), or
// UINT64_MAX once its link was cancelled (M12); the word returned tells the two apart.
__attribute__((always_inline)) static inline ulong mesh_remote_wait(volatile coherent(system) device const ulong *completion, ulong cycle) {
    return mesh_wait(completion, 1 + cycle);
}

// A landed slot's value, loaded through volatile coherent(system) (the NIC wrote it) in the widest words that divide it.
template <typename T>
__attribute__((always_inline)) static inline T mesh_payload(volatile coherent(system) device const T *at) {
    T value;
    if (sizeof(T) % 4 == 0) {
        thread uint *to = (thread uint *)&value;
        for (uint w = 0u; w < sizeof(T) / 4; w++) to[w] = ((volatile coherent(system) device const uint *)at)[w];
    } else if (sizeof(T) % 2 == 0) {
        thread ushort *to = (thread ushort *)&value;
        for (uint w = 0u; w < sizeof(T) / 2; w++) to[w] = ((volatile coherent(system) device const ushort *)at)[w];
    } else {
        thread uchar *to = (thread uchar *)&value;
        for (uint w = 0u; w < sizeof(T); w++) to[w] = ((volatile coherent(system) device const uchar *)at)[w];
    }
    return value;
}

// Releases a cell: a system-scope fence orders every store before it (the payload, a SEND's slot) before `value` is
// stored into `cell`. A SEND cell's ready word takes its prepared argument plus the invocation's cycle (M04, M10, M30;
// mesh-collective.c mesh_host_publish is the host's release).
template <typename W>
__attribute__((always_inline)) static inline void mesh_publish(volatile coherent(system) device W *cell, ulong value) {
    metal::atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst, static_cast<metal::thread_scope>(3));
    *cell = W(value);
}
#endif
