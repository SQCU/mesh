/* bridge.h: a node's bridge. It owns the node's RDMA devices and its one registered region of pages, and it moves
   bytes. It decides nothing: what moves, where and in what order is the collective's plan, compiled before the call.
   Sources: design/data-model.md sections 3 and 6; the rubric's interface table (rubrics.md:14-18): between bridge and
   collective cross only "rows a client owns, positions in them, supplied and landed words"; between bridge and
   program, nothing. */
#ifndef CORE_BRIDGE_H
#define CORE_BRIDGE_H
#include <stdatomic.h>
#include <stdint.h>

enum { BRIDGE_PAGE = 16384 };            /* bytes a page */
typedef uint32_t bridge_page;            /* a page's name: its row in this node's page table */

/* A step of a page list (data-model.md section 6): for a send, a page on this node and the link to the next node;
   for a receive, the page on this node it lands in. A mask (offset, bytes) stays within its page. */
struct bridge_step {
  uint32_t link;                         /* the link (cable) it moves on */
  uint32_t receive;                      /* 1: lands into `page`; 0: sends from `page` */
  bridge_page page;
  uint32_t offset, bytes;                /* the mask within the page */
  uint64_t position;                     /* a send: posted once its link's supplied word reaches this */
};

/* The words a client and the bridge share, one set a link. The client stores `supplied` (how far its producers have
   made data); the bridge stores `sent` and `landed` (how many of the link's sends completed, receives landed). A lost
   link stores ~0 in both. */
struct bridge_words { _Atomic uint64_t supplied, sent, landed; };

/* The region's page table: a page's address on this node. */
void *bridge_address(bridge_page page);

/* Hand the bridge a client's page lists once, before the call: every receive is posted at once, into this node's own
   pages; each send is posted when its link's supplied word reaches its position. The bridge sends no header,
   receipt or acknowledgement. 0, or an errno. */
int bridge_load(const struct bridge_step *steps, uint32_t count, struct bridge_words *words, uint32_t links);
#endif
