#pragma once
#include <stddef.h>
#include <stdint.h>

/* A monotone page publication can be observed after several producers have
   advanced it. Observers never choose or unblock the numerical schedule. */
static inline uint64_t MeshHostReadPublication(const uint64_t *value) {
    return __atomic_load_n(value, __ATOMIC_ACQUIRE);
}
static inline void MeshHostWritePublication(uint32_t *value, uint32_t next) {
    __atomic_store_n(value, next, __ATOMIC_RELEASE);
}

/* A declaration prepares one invocation and its completion operand. Submission
   calls only this retained task; it neither constructs work nor means done. */
typedef void (*MeshHostCompletion)(void *context, const char *error);
typedef struct {
    const void *identity;
    size_t capacity;
} MeshHostResource;
/* A host instruction names the same page bytes as a device instruction.
   These records are consumed only during preparation. access uses read=1,
   written=2 and observe=32 (a monotone read without an anti-dependency). */
typedef struct {
    const void *pages;
    size_t offset, length;
    uint32_t access;
} MeshHostAccess;
typedef struct MeshHostTask MeshHostTask;
typedef const char *(*MeshHostPrepare)(void *declaration, MeshHostCompletion,
                                     void *completion_context, MeshHostTask *);
typedef struct {
    MeshHostPrepare prepare;
    void *declaration;
    const MeshHostAccess *accesses;
    size_t access_count;
} MeshHostCommand;
struct MeshHostTask {
    void *context;
    void (*submit)(void *context);
    void (*release)(void *context);
    const MeshHostResource *resources;
    size_t resource_count;
    /* A compound declaration returns commands instead of submit. The common
       dependency compiler expands them before scheduling. There is no runtime
       compound executor and no implicit order between child instructions. */
    const MeshHostCommand *commands;
    size_t command_count;
    /* Activity belongs to the numerical source's page operands. Submission
       always enters that prepared source, including its declared publications. */
};
