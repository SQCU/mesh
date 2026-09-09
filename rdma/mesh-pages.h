#ifndef MESH_PAGES_H
#define MESH_PAGES_H
#include "mesh.h"

#define MESH_PAGES_LOCAL UINT16_MAX
#define MESH_PAGES_DEPENDENCIES 8
struct mesh_pages_slot { uint32_t sid, pages; uint16_t peer; uint8_t receive, depends, pagewise; uint32_t dependency[MESH_PAGES_DEPENDENCIES], lag[MESH_PAGES_DEPENDENCIES]; };
struct mesh_pages_policy { uint64_t open_retry_ns, fault_seed; uint32_t fault_period, control_pages; };
#define MESH_REDUCE_MATERIALIZED 0
#define MESH_REDUCE_PARTIAL 1
struct mesh_pages_reduce { uint32_t output; uint32_t input[MESH_PAGES_DEPENDENCIES]; uint8_t inputs, kind; uint32_t group, offset, bytes; };
typedef struct mesh_pages mesh_pages;
typedef void (*mesh_pages_hook)(void *capture, uint32_t slot, uint64_t generation);

mesh_pages *mesh_pages_compile(struct mesh_ctx *context, struct mesh_epoch epoch, const unsigned char plan[32],
  const struct mesh_pages_slot *slots, size_t count, uint32_t versions, struct mesh_pages_policy policy);
int mesh_pages_reduces(mesh_pages *p, const struct mesh_pages_reduce *reduces, size_t count);
int mesh_pages_start(mesh_pages *p);
void mesh_pages_stop(mesh_pages *p);
int mesh_pages_progress(mesh_pages *p);
int mesh_pages_free(mesh_pages *p);

size_t mesh_pages_header(const mesh_pages *p);
size_t mesh_pages_payload(const mesh_pages *p);
const uint32_t *mesh_pages_entries(const mesh_pages *p, uint32_t slot);
const uint64_t *mesh_pages_stamps(const mesh_pages *p, uint32_t slot);
static inline uint64_t mesh_pages_stamp(const uint64_t *stamps, uint32_t page){ return __atomic_load_n(stamps+page,__ATOMIC_ACQUIRE); }
static inline uint32_t mesh_pages_entry(const uint32_t *entries, uint32_t page){ return __atomic_load_n(entries+page,__ATOMIC_ACQUIRE); }
static inline void mesh_pages_store(uint64_t *word, uint64_t value){ __atomic_store_n(word,value,__ATOMIC_RELEASE); }
static inline uint64_t mesh_pages_load(const uint64_t *word){ return __atomic_load_n(word,__ATOMIC_ACQUIRE); }
void *mesh_pages_data(const mesh_pages *p, uint32_t slot, uint32_t page);
size_t mesh_pages_select(const mesh_pages *p, const uint32_t *slots, size_t count,
  uint32_t group, uint64_t generation, uint64_t *consumed, uint32_t *indices);
uint32_t mesh_pages_filled(const mesh_pages *p, uint32_t slot, uint64_t generation);
uint64_t mesh_pages_highest(const mesh_pages *p, uint32_t slot);

void mesh_pages_produce_hook(mesh_pages *p, mesh_pages_hook hook, void *capture);
uint64_t mesh_pages_producible(const mesh_pages *p, uint32_t slot);
int mesh_pages_publish(mesh_pages *p, uint32_t slot, uint32_t first, uint32_t count, uint64_t generation);
int mesh_pages_consume(mesh_pages *p, uint32_t slot, uint32_t first, uint32_t count, uint64_t generation);
int mesh_pages_faulted(const mesh_pages *p, uint32_t slot, uint64_t generation, uint32_t *first, uint32_t *second);
uint64_t mesh_pages_hash(const void *data, size_t bytes, uint64_t seed);
int mesh_pages_agreed(const mesh_pages *p, uint32_t slot);
int mesh_pages_digest(const mesh_pages *p, uint32_t slot, uint64_t generation, uint64_t *hash);
int mesh_pages_status(const mesh_pages *p);
int mesh_pages_settled(const mesh_pages *p);
int mesh_pages_recover(mesh_pages *p);
uint64_t mesh_pages_incarnation(const mesh_pages *p);
void mesh_pages_fail(mesh_pages *p, int error);
size_t mesh_pages_describe(const mesh_pages *p, char *out, size_t bytes);
#endif
