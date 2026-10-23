#ifndef NCCL_MESH_METAL_H
#define NCCL_MESH_METAL_H
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include "mesh.h"

#pragma GCC visibility push(hidden)

#define MESH_POLLS 4500000u
typedef struct mesh_pages mesh_pages;
static inline mesh_pages pages_of(uint64_t bytes, uint64_t page) { return (mesh_pages){(bytes + page - 1) / page}; }
static inline mesh_pages pages_add(mesh_pages a, mesh_pages b) { return (mesh_pages){a.n + b.n}; }
static inline mesh_pages pages_sub(mesh_pages a, mesh_pages b) { return (mesh_pages){a.n - b.n}; }
static inline int pages_below(mesh_pages a, mesh_pages b) { return a.n < b.n; }
static inline int pages_lost(mesh_pages a) { return a.n == UINT64_MAX; }
static inline uint64_t pages_to_word(mesh_pages a) { return a.n; }
static inline mesh_pages pages_from_word(uint64_t word) { return (mesh_pages){word}; }

#define MESH_PEER(P, U64, U32) struct mesh_peer { P landing; P landed; U64 landed_target, landed_at, published, ready_value; U32 channel, pad; }
#define MESH_TABLE(P, U64, U32, ATOMIC) struct mesh_table { P part; ATOMIC arrivals; U32 peers; ATOMIC ended; U32 pad; }
#define MESH_CROSSING(T, P, U32, ...) struct mesh_crossing { __VA_ARGS__ T land; T publish; P failure; U32 groups, polls; }
MESH_PEER(uint64_t, uint64_t, uint32_t);
MESH_TABLE(uint64_t, uint64_t, uint32_t, uint32_t);
enum { METAL_PLAIN, METAL_LAND };
struct metal_program;
struct metal_rows;
int metal_rows_take(struct metal_rows *, size_t bytes, void **buffer, size_t *offset);

struct metal_result { int status; const char *message; };
#define METAL_DONE ((struct metal_result){0, NULL})
static inline struct metal_result metal_failed(int status) { return (struct metal_result){status, strerror(status)}; }

void *metal_device(void);

void *metal_wrap(void *address, size_t bytes);
uint64_t metal_address(void *buffer);

void *metal_shared(size_t bytes);
void *metal_event(void);
uint64_t metal_signaled(void *event);

int metal_wait(void *event, uint64_t value, uint64_t ns);
void metal_release(void *object);
void *metal_retain(void *object);

void *metal_command(const struct metal_program *);

int metal_ended(void *command);

#define METAL_BOUND 1
#define METAL_LOST 2
#define METAL_REVOKED 4

struct metal_program *metal_begin(void *command_buffer, void *encoder, void *violation, const char **message);
struct metal_program *metal_begin4(void *encoder, void *table, void *residency, void *violation, const char **message);

void metal_resident(struct metal_program *, void *buffer);
struct metal_rows *metal_rows(struct metal_program *, struct metal_rows *);

void metal_gate(struct metal_program *, void *violation);
struct metal_result metal_copy(struct metal_program *, int mode, void *dst, size_t dst_offset, void *src, size_t src_offset, size_t bytes);

#define METAL_MERGE 12

struct metal_result metal_combine3(struct metal_program *, int type, int op, void *dst, size_t dst_offset, void *src, size_t src_offset,
                                   void *with, size_t with_offset, size_t n);
struct metal_result metal_premultiply(struct metal_program *, int type, void *dst, size_t dst_offset, void *src, size_t src_offset, size_t n,
                                      uint64_t scalar);
struct metal_result metal_truncdiv(struct metal_program *, int type, void *dst, size_t dst_offset, void *src, size_t src_offset, size_t n,
                                   uint64_t divisor);

struct metal_wait { void *words, *violation, *data; size_t word, sent_word, data_offset; mesh_pages target, sent_target; uint64_t at; uint32_t channel; };

struct metal_release { void *words; size_t word; mesh_pages value; };
struct metal_result metal_publish(struct metal_program *, const struct metal_release *);

void metal_uses(struct metal_program *, void *const *buffers, size_t n, int write);

struct metal_result metal_stamp(struct metal_program *, void *words, size_t word);

struct metal_result metal_spin(struct metal_program *, const struct metal_wait *);

struct metal_result metal_land(struct metal_program *, int type, int op, void *dst, size_t dst_offset, size_t n, const struct metal_wait *);

struct metal_result metal_bind(void *encoder, void *table, void *residency, struct metal_rows *rows, uint64_t land, uint64_t publish, void *violation,
                               uint32_t groups, uint32_t index, void *const *buffers, size_t n);

void *metal_contents(void *buffer);

#define METAL_NONE UINT32_MAX
enum { METAL_PUBLISH, METAL_CONSUME };
struct metal_result metal_record(struct metal_program *, uint32_t track, int cursor, uint64_t position, uint64_t need);
struct metal_result metal_depend(struct metal_program *, uint32_t other, uint64_t count);
struct metal_result metal_tracks(struct metal_program *, uint32_t tracks, const uint64_t *consumed);

const char *metal_prelude(void);
void metal_end(struct metal_program *, void *event, uint64_t value);
#pragma GCC visibility pop
#endif
