#ifndef MESH_SHM_H
#define MESH_SHM_H

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdatomic.h>

#define MESH_MAGIC     0x4853454dU
#define MESH_VERSION   3U
#define MESH_MAX_REGIONS 8
#define MESH_DEPTH     4U

typedef struct mesh_slot_s
{
	_Atomic uint64_t s0;
	_Atomic uint64_t s1;
	uint64_t pad[6];
}
mesh_slot_t;

typedef struct mesh_hdr_s
{
	uint32_t magic;
	uint32_t version;
	uint32_t nreq;
	uint32_t nresp;
	uint32_t depth;
	uint32_t pad0;
	uint64_t pad1[5];
	_Atomic uint64_t req_seq;
	uint64_t pad2[7];
	_Atomic uint64_t resp_seq;
	uint64_t pad3[7];
	_Atomic uint64_t solver_alive;
	uint64_t pad4[7];
	_Atomic uint64_t epoch;
	uint64_t pad5[7];
}
mesh_hdr_t;

static inline size_t mesh_slotstride(uint32_t n)
{
	return ((sizeof(mesh_slot_t) + (size_t)n * sizeof(float)) + 63u) & ~(size_t)63u;
}

static inline size_t mesh_bytes(uint32_t nreq, uint32_t nresp)
{
	return sizeof(mesh_hdr_t) + (size_t)MESH_DEPTH * (mesh_slotstride(nreq) + mesh_slotstride(nresp));
}

static inline mesh_slot_t *mesh_reqslot(mesh_hdr_t *h, uint64_t seq)
{
	return (mesh_slot_t *)((char *)h + sizeof(mesh_hdr_t) + mesh_slotstride(h->nreq) * (size_t)(seq % MESH_DEPTH));
}

static inline mesh_slot_t *mesh_respslot(mesh_hdr_t *h, uint64_t seq)
{
	return (mesh_slot_t *)((char *)h + sizeof(mesh_hdr_t) + mesh_slotstride(h->nreq) * MESH_DEPTH + mesh_slotstride(h->nresp) * (size_t)(seq % MESH_DEPTH));
}

static inline float *mesh_payload(mesh_slot_t *s)
{
	return (float *)((char *)s + sizeof(mesh_slot_t));
}

static inline void mesh_slot_write(mesh_slot_t *s, uint64_t seq, const float *src, uint32_t n)
{
	atomic_store_explicit(&s->s0, seq, memory_order_relaxed);
	atomic_thread_fence(memory_order_release);
	memcpy(mesh_payload(s), src, (size_t)n * sizeof(float));
	atomic_store_explicit(&s->s1, seq, memory_order_release);
}

static inline int mesh_slot_read(mesh_slot_t *s, uint64_t seq, float *dst, uint32_t n)
{
	if (atomic_load_explicit(&s->s1, memory_order_acquire) != seq)
		return 0;
	memcpy(dst, mesh_payload(s), (size_t)n * sizeof(float));
	atomic_thread_fence(memory_order_acquire);
	return atomic_load_explicit(&s->s0, memory_order_relaxed) == seq;
}

static inline void mesh_put_req(mesh_hdr_t *h, uint64_t seq, const float *src)
{
	mesh_slot_write(mesh_reqslot(h, seq), seq, src, h->nreq);
	atomic_store_explicit(&h->req_seq, seq, memory_order_release);
}

static inline void mesh_put_resp(mesh_hdr_t *h, uint64_t seq, const float *src)
{
	mesh_slot_write(mesh_respslot(h, seq), seq, src, h->nresp);
	atomic_store_explicit(&h->resp_seq, seq, memory_order_release);
}

static inline uint64_t mesh_get_req(mesh_hdr_t *h, uint64_t last, float *dst, uint64_t *misses)
{
	uint64_t s;
	int t;
	for (t = 0; t < (int)MESH_DEPTH; t++)
	{
		s = atomic_load_explicit(&h->req_seq, memory_order_acquire);
		if (s <= last)
			return last;
		if (mesh_slot_read(mesh_reqslot(h, s), s, dst, h->nreq))
			return s;
		if (misses)
			(*misses)++;
	}
	return last;
}

static inline uint64_t mesh_get_resp(mesh_hdr_t *h, uint64_t last, float *dst, uint64_t *misses)
{
	uint64_t s;
	int t;
	for (t = 0; t < (int)MESH_DEPTH; t++)
	{
		s = atomic_load_explicit(&h->resp_seq, memory_order_acquire);
		if (s <= last)
			return last;
		if (mesh_slot_read(mesh_respslot(h, s), s, dst, h->nresp))
			return s;
		if (misses)
			(*misses)++;
	}
	return last;
}

static inline void mesh_reset(mesh_hdr_t *h, uint32_t nreq, uint32_t nresp)
{
	uint64_t e = 0;
	uint32_t i;
	if (h->magic == MESH_MAGIC)
		e = atomic_load_explicit(&h->epoch, memory_order_acquire);
	h->magic = 0;
	atomic_thread_fence(memory_order_release);
	h->nreq = nreq;
	h->nresp = nresp;
	h->depth = MESH_DEPTH;
	h->version = MESH_VERSION;
	for (i = 0; i < MESH_DEPTH; i++)
	{
		atomic_store_explicit(&mesh_reqslot(h, i)->s0, 0, memory_order_relaxed);
		atomic_store_explicit(&mesh_reqslot(h, i)->s1, 0, memory_order_relaxed);
		atomic_store_explicit(&mesh_respslot(h, i)->s0, 0, memory_order_relaxed);
		atomic_store_explicit(&mesh_respslot(h, i)->s1, 0, memory_order_relaxed);
	}
	atomic_store_explicit(&h->req_seq, 0, memory_order_relaxed);
	atomic_store_explicit(&h->resp_seq, 0, memory_order_relaxed);
	atomic_store_explicit(&h->solver_alive, 0, memory_order_relaxed);
	atomic_store_explicit(&h->epoch, e + 1, memory_order_relaxed);
	atomic_thread_fence(memory_order_release);
	h->magic = MESH_MAGIC;
}

#endif
