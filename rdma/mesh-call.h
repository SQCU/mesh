#ifndef MESH_CALL_H
#define MESH_CALL_H
#include <stddef.h>
#include <stdint.h>
#include "mesh-view.h"
#ifdef __cplusplus
extern "C" {
#endif

/* The client's interface to its node's page table (libmesh: mesh-call.c, mesh-dataflow.c): plain C declarations and
   integer constants, which C, Objective-C and C++ include and Python binds through mesh_c_build.py.

   A node's page table maps an index to a page of the node's pool: an index's word holds the page it names (or none)
   and `landed`, the count of pages that have landed at it.  A client allocates indices (mesh_pages), writes the page an
   index names to indices of any nodes (mesh_write: the bridges carry it, and each destination's landing makes that
   index name the page and counts the landing), reads an index (mesh_entry, mesh_page_data) and drops its pages
   (mesh_invalidate).  A consumer that waits for a landing reads the index until `landed` reaches what it expects; that
   wait is the consumer's own, and nothing in the transport waits.  What lands at an index is whatever page reached it:
   the transport checks no write's bytes and reports nothing about them (a write may land wrong, or not at all). */

/* A page: MESH_HEAD bytes of in-band header the bridge writes, then the payload; a page is MESH_CLASS_SHORT or
   MESH_CLASS_LONG frames of MESH_FRAME bytes.  A write names at most MESH_DESTS destinations; a node's table holds
   MESH_TABLE indices. */
#define MESH_FRAME 4096u
#define MESH_HEAD 256u
#define MESH_CLASS_SHORT 2u
#define MESH_CLASS_LONG 9u
#define MESH_SHORT_PAYLOAD (MESH_CLASS_SHORT*MESH_FRAME-MESH_HEAD)
#define MESH_LONG_PAYLOAD (MESH_CLASS_LONG*MESH_FRAME-MESH_HEAD)
#define MESH_DESTS 8u
#define MESH_TABLE (1u<<20)

/* A process's attach to its node's region (struct hdr, mesh.h): the region mapped, its client slot (`client`, the
   owner word) and the cursors of its index and arena scans.  Zeroed before mesh_attach. */
struct hdr;
struct mesh_ctx { struct hdr *M; size_t len; uint64_t client; uint32_t index_cursor,arena_cursor; int fd; };
/* The region mapped writable and a vacant client slot taken by one compare-exchange, stamped with this process's pid
   and start: 0, EBUSY where no slot is vacant, EINVAL where the region is of another layout or expired as the slot was
   taken (the slot is let go), else the errno of the open. */
int mesh_attach(struct mesh_ctx *);
/* The slot left (mesh.h mesh_leave: its indices freed and their pages dropped; writes already on its ring are still
   carried) and the region unmapped; 0, or the errno of the unmap or close. */
int mesh_detach(struct mesh_ctx *);

/* A destination of a write: a node and an index of its table. */
struct mesh_dest { uint32_t node,index; };
/* `count` consecutive free indices, owned by this client.  bytes > 0: each index given a pool page of the least class
   whose payload holds `bytes` (its uses 1, its landed 0); bytes 0: indices alone, landed 0, each naming no page until
   one lands.  The first index; UINT32_MAX with errno EINVAL (not attached, count 0 or past the table, bytes past
   MESH_LONG_PAYLOAD), ENOSPC (no run of `count` free indices), ENOMEM (the pool holds fewer free pages of the class). */
uint32_t mesh_pages(struct mesh_ctx *,uint32_t count,uint32_t bytes);
/* The page at `index` (`bytes` of payload) written to each of dests[0 .. n): one cell of this client's write ring, the
   page's uses raised by one until the bridge takes it.  0; EBUSY where the page has a SEND or a write out (its uses
   past 1: the client must not change its bytes); EAGAIN where the ring is full; EINVAL where the index is not this
   client's or holds no page, n is 0 or past MESH_DESTS, a destination index is past the table, or `bytes` is past the
   page's payload.  The ring has one producer: a client's calls of mesh_write are made one at a time. */
int mesh_write(struct mesh_ctx *,uint32_t index,uint32_t bytes,const struct mesh_dest *dests,uint32_t n);
/* Each index of [first, first + count) left naming no page (its landed kept), its page dropped; indices != 0: the
   indices freed too.  0; EINVAL where the range is past the table or holds an index that is not this client's (then
   nothing changes). */
int mesh_invalidate(struct mesh_ctx *,uint32_t first,uint32_t count,int indices);
/* An index as one acquire load of its word reads it: its landed count, and its page's payload bytes, uses and payload
   address (all 0 and NULL where it names no page).  0, or EINVAL where not attached or the index is past the table. */
struct mesh_view_entry { uint32_t landed,bytes,uses,unused; void *data; };
int mesh_entry(struct mesh_ctx *,uint32_t index,struct mesh_view_entry *);
/* The payload of the page the index names; NULL where it names none (or not attached, or past the table). */
void *mesh_page_data(struct mesh_ctx *,uint32_t index);
/* The nodes this node's bridge reaches, through cables or relays, each with the bandwidth of its own cables toward it
   (0: through relays alone), at most `room`; their count. */
uint32_t mesh_nodes_reached(struct hdr *,struct mesh_reach *,uint32_t room);

#ifdef __cplusplus
}
#endif
#endif
