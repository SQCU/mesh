#pragma once
#include <stddef.h>

/* A declaration prepares one invocation and its completion operand. Submission
   calls only this retained task; it neither constructs work nor means done. */
typedef void (*MeshHostCompletion)(void *context, const char *error);
typedef struct {
    const void *identity;
    size_t capacity;
} MeshHostResource;
typedef struct {
    void *context;
    void (*submit)(void *context);
    void (*release)(void *context);
    const MeshHostResource *resources;
    size_t resource_count;
} MeshHostTask;
typedef const char *(*MeshHostPrepare)(void *declaration, MeshHostCompletion,
                                     void *completion_context, MeshHostTask *);
