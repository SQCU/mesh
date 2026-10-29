#include "mesh-isl-memory.h"
#include "nccl.h"
#include <errno.h>
#include <limits.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* ISL's small objects share region pages instead of each taking a whole lease.
   Power-of-two slots reuse freed objects during a solve. Lists and headers live
   in those same pages; only the roots and lock are static. The last object freed
   returns all preparation leases. No ISL object is retained by execution. */
struct IslSlab;
union IslSlot {
    struct { struct IslSlab *slab; size_t bytes; } value;
    union IslSlot *next;
    max_align_t alignment;
};
struct IslSlab {
    struct IslSlab *all, *next, *previous;
    union IslSlot *free;
    size_t used, issued, count, step;
    unsigned bin;
    max_align_t alignment;
};
static pthread_mutex_t isl_memory_lock = PTHREAD_MUTEX_INITIALIZER;
static struct IslSlab *isl_slabs, *isl_available[sizeof(size_t) * CHAR_BIT];
static size_t isl_objects;

static void available_add(struct IslSlab *slab) {
    slab->previous = NULL;
    slab->next = isl_available[slab->bin];
    if (slab->next) slab->next->previous = slab;
    isl_available[slab->bin] = slab;
}
static void available_remove(struct IslSlab *slab) {
    if (slab->previous) slab->previous->next = slab->next;
    else isl_available[slab->bin] = slab->next;
    if (slab->next) slab->next->previous = slab->previous;
}
void *mesh_isl_malloc(size_t bytes) {
    size_t step = sizeof(union IslSlot), need;
    unsigned bin = 0;
    if (__builtin_add_overflow(bytes, sizeof(union IslSlot), &need)) goto overflow;
    while (step < need) {
        if (step > SIZE_MAX / 2) goto overflow;
        step *= 2;
        ++bin;
    }
    pthread_mutex_lock(&isl_memory_lock);
    struct IslSlab *slab = isl_available[bin];
    if (!slab) {
        size_t extent;
        if (__builtin_add_overflow(sizeof *slab, step, &extent)) {
            pthread_mutex_unlock(&isl_memory_lock);
            goto overflow;
        }
        if (extent < MESH_ISL_SLAB_BYTES) extent = MESH_ISL_SLAB_BYTES;
        void *pages = NULL;
        if (ncclMemAlloc(&pages, extent) != ncclSuccess) {
            pthread_mutex_unlock(&isl_memory_lock);
            fprintf(stderr, "ISL preparation pages: %s\n", ncclGetLastError(NULL));
            errno = ENOMEM;
            return NULL;
        }
        slab = pages;
        *slab = (struct IslSlab){.all = isl_slabs, .step = step, .bin = bin,
            .count = (extent - sizeof *slab) / step};
        isl_slabs = slab;
        available_add(slab);
    }
    union IslSlot *slot = slab->free;
    if (slot) slab->free = slot->next;
    else slot = (union IslSlot *)((char *)(slab + 1) + slab->issued++ * slab->step);
    slot->value.slab = slab;
    slot->value.bytes = bytes;
    if (++slab->used == slab->count) available_remove(slab);
    ++isl_objects;
    pthread_mutex_unlock(&isl_memory_lock);
    return slot + 1;
overflow:
    errno = ENOMEM;
    return NULL;
}
void mesh_isl_free(void *memory) {
    if (!memory) return;
    union IslSlot *slot = (union IslSlot *)memory - 1;
    struct IslSlab *slab = slot->value.slab, *released = NULL;
    pthread_mutex_lock(&isl_memory_lock);
    if (slab->used-- == slab->count) available_add(slab);
    slot->next = slab->free;
    slab->free = slot;
    if (!--isl_objects) {
        released = isl_slabs;
        isl_slabs = NULL;
        memset(isl_available, 0, sizeof isl_available);
    }
    pthread_mutex_unlock(&isl_memory_lock);
    while (released) {
        struct IslSlab *next = released->all;
        ncclMemFree(released);
        released = next;
    }
}
void *mesh_isl_calloc(size_t count, size_t bytes) {
    size_t total;
    if (__builtin_mul_overflow(count, bytes, &total)) { errno = ENOMEM; return NULL; }
    void *memory = mesh_isl_malloc(total);
    if (memory) memset(memory, 0, total);
    return memory;
}
void *mesh_isl_realloc(void *memory, size_t bytes) {
    if (!memory) return mesh_isl_malloc(bytes);
    if (!bytes) { mesh_isl_free(memory); return NULL; }
    union IslSlot *slot = (union IslSlot *)memory - 1;
    size_t previous = slot->value.bytes;
    if (bytes <= slot->value.slab->step - sizeof *slot) {
        slot->value.bytes = bytes;
        return memory;
    }
    void *replacement = mesh_isl_malloc(bytes);
    if (replacement) {
        memcpy(replacement, memory, previous < bytes ? previous : bytes);
        mesh_isl_free(memory);
    }
    return replacement;
}
char *mesh_isl_strdup(const char *text) {
    size_t bytes = strlen(text) + 1;
    char *copy = mesh_isl_malloc(bytes);
    if (copy) memcpy(copy, text, bytes);
    return copy;
}
