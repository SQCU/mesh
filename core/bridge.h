/* bridge.h: a node's bridge. It owns the node's RDMA devices and its one registered region of pages, and it moves
   bytes. It decides nothing: what moves, where and in what order is the collective's plan, compiled before the call.
   Sources: design/data-model.md sections 3 and 6; the rubric's interface table (rubrics.md:14-18): between bridge and
   collective cross only "rows a client owns, positions in them, supplied and landed words"; between bridge and
   program, nothing.

   The driver (data-model.md section 3): SEND and RECV on unreliable-connected queue pairs, nothing else; a SEND fills
   the oldest receive posted on its queue pair, both spanning the same 4 KB credit frames; one range a request, within
   one registration; at most 100 registrations a device, none across a 4 GiB boundary (the driver keeps an address's
   low 32 bits); no shared receive queue; and "a process that exits while it holds queue pairs can hang in the driver
   until the node resets". So the bridge is its own resident process (bridge_serve), one a node, holding the queue
   pairs for good; a program maps the region and hands the bridge its page lists and words through it.

   Header-only: define BRIDGE_IMPLEMENTATION in one translation unit. Build the bridge with
       cc -O2 -DBRIDGE_IMPLEMENTATION -DBRIDGE_MAIN -x c core/bridge.h -o bridge -lrdma
   and run it with its configuration (bridge_serve's comment). */
#ifndef CORE_BRIDGE_H
#define CORE_BRIDGE_H
#include <stdatomic.h>
#include <stdint.h>

enum { BRIDGE_PAGE = 16384, BRIDGE_LINKS = 16, BRIDGE_STEPS = 1 << 20 };
typedef uint32_t bridge_page;            /* a page's name: its row in this node's page table */

/* A step of a page list (data-model.md section 6): for a send, a page on this node and the link to the next node;
   for a receive, the page on this node it lands in. A mask (offset, bytes) stays within its page; a send and the
   receive it fills move the same bytes. */
struct bridge_step {
  uint32_t link;                         /* the link (cable) it moves on */
  uint32_t receive;                      /* 1: lands into `page`; 0: sends from `page` */
  bridge_page page;
  uint32_t offset, bytes;                /* the mask within the page */
  uint32_t after_link, after_landed;     /* a send waits on this link's word: landed (1, a relay's forward: "a relay
                                            passes each part on as it arrives", data-model.md section 5) or supplied */
  uint64_t position;                     /* a send: posted once that word reaches this */
};

/* The words a client and the bridge share, one set a link. The client stores `supplied` (how far its producers have
   made data); the bridge stores `sent` and `landed` (how many of the link's sends completed, receives landed). A lost
   link stores ~0 in both. */
struct bridge_words { _Atomic uint64_t supplied, sent, landed; };

/* The region as a client maps it: the table the bridge reads, the words, then the pages. */
struct bridge_region {
  uint64_t magic, pages, links;
  _Atomic uint64_t loaded;               /* the client's count of tables handed over (bridge_load) */
  _Atomic uint64_t taken;                /* the bridge's count of tables it has posted */
  uint64_t count;                        /* the steps in the table */
  struct bridge_words words[BRIDGE_LINKS];
  struct bridge_step table[BRIDGE_STEPS];
};

/* Map this node's region (the bridge's, by name): 0, or an errno. */
int bridge_attach(const char *name);
/* The region's page table: a page's address on this node. */
void *bridge_address(bridge_page page);
/* A link's words. */
struct bridge_words *bridge_words(uint32_t link);

/* Hand the bridge a client's page lists once, before the call: every receive is posted ahead, into this node's own
   pages, in list order a link; each send is posted when the word it names reaches its position, in list order a
   link. The bridge sends no header, receipt or acknowledgement. 0, or an errno. */
int bridge_load(const struct bridge_step *steps, uint32_t count);

/* The bridge process. Its configuration, one line each: "region <name> <GiB>", then for each link "link <device>
   <gid index> <peer host> <tcp port> <listen 0|1>". It makes the region (1 GiB extents, 1 GiB aligned, so none crosses
   a 4 GiB boundary), registers it with each device, makes one unreliable-connected queue pair a link, exchanges its
   number and GID with the peer's bridge over TCP (the end with listen 1 accepts), then serves tables for good. */
int bridge_serve(const char *configuration);
#endif

#ifdef BRIDGE_IMPLEMENTATION
#include <errno.h>
#include <fcntl.h>
#include <infiniband/verbs.h>
#include <netdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <unistd.h>

static struct bridge_region *bridge_region_;
static char *bridge_pages_;
enum { BRIDGE_EXTENT = 1 << 30, BRIDGE_MAGIC = 0x6d65736862726467ull };

static size_t bridge_header_bytes(void) { return (sizeof(struct bridge_region) + BRIDGE_PAGE - 1) / BRIDGE_PAGE * BRIDGE_PAGE; }

int bridge_attach(const char *name) {
  const int fd = shm_open(name, O_RDWR, 0);
  if (fd < 0) return errno;
  struct bridge_region head;
  if (pread(fd, &head, sizeof head, 0) != (ssize_t)sizeof head || head.magic != BRIDGE_MAGIC) { close(fd); return EINVAL; }
  const size_t bytes = bridge_header_bytes() + head.pages * BRIDGE_PAGE;
  void *m = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  if (m == MAP_FAILED) return errno;
  bridge_region_ = m; bridge_pages_ = (char *)m + bridge_header_bytes();
  return 0;
}
void *bridge_address(bridge_page page) { return bridge_pages_ + (size_t)page * BRIDGE_PAGE; }
struct bridge_words *bridge_words(uint32_t link) { return &bridge_region_->words[link]; }

int bridge_load(const struct bridge_step *steps, uint32_t count) {
  if (count > BRIDGE_STEPS) return E2BIG;
  memcpy(bridge_region_->table, steps, count * sizeof *steps);
  bridge_region_->count = count;
  for (uint32_t l = 0; l < bridge_region_->links; l++) {
    atomic_store(&bridge_region_->words[l].supplied, 0);
    atomic_store(&bridge_region_->words[l].sent, 0);
    atomic_store(&bridge_region_->words[l].landed, 0);
  }
  atomic_fetch_add_explicit(&bridge_region_->loaded, 1, memory_order_release);
  return 0;
}

/* ---- the bridge process ---- */
struct bridge_link {
  struct ibv_context *context; struct ibv_pd *pd; struct ibv_cq *cq; struct ibv_qp *qp;
  struct ibv_mr *mr[100]; uint32_t mrs; int gid_index;
  uint32_t depth;                        /* the queue pair's receive and send depth (max_qp_wr): the hardware's limit */
  uint64_t posted_receives, posted_sends;/* this table's, in list order */
  uint64_t next_receive, next_send;      /* the table index past the last one posted (a cursor a link) */
};
static struct bridge_link bridge_link_[BRIDGE_LINKS];

static int bridge_exchange(const char *host, const char *port, int listen_end, const void *mine, void *theirs, size_t n) {
  struct addrinfo hints = { .ai_socktype = SOCK_STREAM, .ai_flags = listen_end ? AI_PASSIVE : 0 }, *ai;
  if (getaddrinfo(listen_end ? NULL : host, port, &hints, &ai)) return EHOSTUNREACH;
  int s = socket(ai->ai_family, SOCK_STREAM, 0), c = -1, one = 1;
  setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
  if (listen_end) { if (!bind(s, ai->ai_addr, ai->ai_addrlen) && !listen(s, 1)) c = accept(s, NULL, NULL); }
  else { while (connect(s, ai->ai_addr, ai->ai_addrlen)) sleep(1); c = s; }
  freeaddrinfo(ai);
  if (c < 0) { close(s); return ECONNREFUSED; }
  const int ok = write(c, mine, n) == (ssize_t)n && read(c, theirs, n) == (ssize_t)n;
  if (c != s) close(c);
  close(s);
  return ok ? 0 : EIO;
}

struct bridge_end { uint32_t qpn, psn; union ibv_gid gid; };

static int bridge_link_open(struct bridge_link *L, const char *device, int gid_index, const char *host, const char *port,
                            int listen_end, char *region, size_t bytes) {
  int n = 0;
  struct ibv_device **list = ibv_get_device_list(&n), *dev = NULL;
  for (int i = 0; i < n; i++) if (!strcmp(ibv_get_device_name(list[i]), device)) dev = list[i];
  if (!dev) return ENODEV;
  L->context = ibv_open_device(dev);
  ibv_free_device_list(list);
  if (!L->context) return ENODEV;
  struct ibv_device_attr attr;
  ibv_query_device(L->context, &attr);
  L->depth = (uint32_t)attr.max_qp_wr;
  L->pd = ibv_alloc_pd(L->context);
  for (size_t at = 0; at < bytes; at += BRIDGE_EXTENT) {
    const size_t span = bytes - at < BRIDGE_EXTENT ? bytes - at : BRIDGE_EXTENT;
    if (L->mrs == 100 || !(L->mr[L->mrs++] = ibv_reg_mr(L->pd, region + at, span, IBV_ACCESS_LOCAL_WRITE))) return ENOMEM;
  }
  L->cq = ibv_create_cq(L->context, (int)(2 * L->depth), NULL, NULL, 0);
  struct ibv_qp_init_attr qa = { .send_cq = L->cq, .recv_cq = L->cq, .qp_type = IBV_QPT_UC,
                                 .cap = { .max_send_wr = L->depth, .max_recv_wr = L->depth, .max_send_sge = 1, .max_recv_sge = 1 } };
  if (!(L->qp = ibv_create_qp(L->pd, &qa))) return ENOMEM;
  struct ibv_qp_attr init = { .qp_state = IBV_QPS_INIT, .port_num = 1, .pkey_index = 0, .qp_access_flags = IBV_ACCESS_LOCAL_WRITE };
  if (ibv_modify_qp(L->qp, &init, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS)) return EIO;
  struct bridge_end mine = { .qpn = L->qp->qp_num, .psn = 0 }, theirs;
  ibv_query_gid(L->context, 1, gid_index, &mine.gid);
  L->gid_index = gid_index;
  int e = bridge_exchange(host, port, listen_end, &mine, &theirs, sizeof mine);
  if (e) return e;
  struct ibv_port_attr pa;
  ibv_query_port(L->context, 1, &pa);
  struct ibv_qp_attr rtr = { .qp_state = IBV_QPS_RTR, .path_mtu = pa.active_mtu, .dest_qp_num = theirs.qpn, .rq_psn = theirs.psn,
                             .ah_attr = { .is_global = 1, .port_num = 1, .grh = { .dgid = theirs.gid, .sgid_index = (uint8_t)gid_index, .hop_limit = 1 } } };
  if (ibv_modify_qp(L->qp, &rtr, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN)) return EIO;
  struct ibv_qp_attr rts = { .qp_state = IBV_QPS_RTS, .sq_psn = mine.psn };
  return ibv_modify_qp(L->qp, &rts, IBV_QP_STATE | IBV_QP_SQ_PSN) ? EIO : 0;
}

static uint32_t bridge_lkey(struct bridge_link *L, size_t offset) { return L->mr[offset / BRIDGE_EXTENT]->lkey; }

int bridge_serve(const char *configuration) {
  FILE *f = fopen(configuration, "r");
  if (!f) return errno;
  char name[128] = "/bridge", device[64], host[256], port[16];
  unsigned gib = 1; int gid, listen_end; uint32_t links = 0;
  char line[512];
  size_t bytes = 0; char *region = NULL; int fd = -1;
  while (fgets(line, sizeof line, f)) {
    if (sscanf(line, "region %127s %u", name, &gib) == 2) {
      bytes = (size_t)gib * BRIDGE_EXTENT;
      const size_t total = bridge_header_bytes() + bytes;
      shm_unlink(name);
      fd = shm_open(name, O_RDWR | O_CREAT, 0600);
      if (fd < 0 || ftruncate(fd, (off_t)total)) return errno;
      /* the pages on a 1 GiB boundary, the header just below them: no 1 GiB registration crosses a 4 GiB boundary */
      char *reserve = mmap(NULL, total + 2 * (size_t)BRIDGE_EXTENT, PROT_NONE, MAP_ANON | MAP_PRIVATE, -1, 0);
      if (reserve == MAP_FAILED) return errno;
      const uintptr_t pages_at = ((uintptr_t)reserve + bridge_header_bytes() + BRIDGE_EXTENT - 1) / BRIDGE_EXTENT * BRIDGE_EXTENT;
      char *m = mmap((char *)pages_at - bridge_header_bytes(), total, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, fd, 0);
      if (m == MAP_FAILED) return errno;
      bridge_region_ = (struct bridge_region *)m;
      region = (char *)pages_at;
      memset(bridge_region_, 0, sizeof *bridge_region_);
      bridge_region_->magic = BRIDGE_MAGIC; bridge_region_->pages = bytes / BRIDGE_PAGE;
    } else if (sscanf(line, "link %63s %d %255s %15s %d", device, &gid, host, port, &listen_end) == 5) {
      if (!region || links == BRIDGE_LINKS) return EINVAL;
      int e = bridge_link_open(&bridge_link_[links], device, gid, host, port, listen_end, region, bytes);
      if (e) { fprintf(stderr, "bridge: link %u (%s): %s\n", links, device, strerror(e)); return e; }
      links++;
    }
  }
  fclose(f);
  if (!bridge_region_) return EINVAL;
  bridge_region_->links = links;
  fprintf(stderr, "bridge: region %s, %llu pages, %u links\n", name, (unsigned long long)bridge_region_->pages, links);
  uint64_t taken = 0;
  for (;;) {
    /* a new table: its receives posted ahead and its sends from the start, link by link */
    if (atomic_load_explicit(&bridge_region_->loaded, memory_order_acquire) != taken) {
      taken = atomic_load(&bridge_region_->loaded);
      for (uint32_t l = 0; l < links; l++) {
        bridge_link_[l].posted_receives = bridge_link_[l].posted_sends = 0;
        bridge_link_[l].next_receive = bridge_link_[l].next_send = 0;
      }
      atomic_store(&bridge_region_->taken, taken);
    }
    const struct bridge_step *t = bridge_region_->table;
    const uint64_t count = bridge_region_->count;
    for (uint32_t l = 0; l < links; l++) {
      struct bridge_link *L = &bridge_link_[l];
      struct bridge_words *w = &bridge_region_->words[l];
      /* every receive of the link, in list order, as far as the queue pair holds (its depth: the hardware's limit) */
      const uint64_t landed = atomic_load_explicit(&w->landed, memory_order_relaxed);
      for (uint64_t k = L->next_receive; k < count && L->posted_receives - landed < L->depth; k++) {
        if (t[k].link != l || !t[k].receive) continue;
        const size_t at = (size_t)t[k].page * BRIDGE_PAGE + t[k].offset;
        struct ibv_sge sge = { .addr = (uintptr_t)(region + at), .length = t[k].bytes, .lkey = bridge_lkey(L, at) };
        struct ibv_recv_wr wr = { .wr_id = k, .sg_list = &sge, .num_sge = 1 }, *bad;
        if (ibv_post_recv(L->qp, &wr, &bad)) break;
        L->posted_receives++; L->next_receive = k + 1;
      }
      /* each send of the link, in list order, once the word it names (its producer's supplied, or the landed of the
         link a relayed page came in on) reaches its position */
      const uint64_t sent = atomic_load_explicit(&w->sent, memory_order_relaxed);
      for (uint64_t k = L->next_send; k < count && L->posted_sends - sent < L->depth; k++) {
        if (t[k].link != l || t[k].receive) continue;
        const struct bridge_words *a = &bridge_region_->words[t[k].after_link];
        const uint64_t reached = atomic_load_explicit(t[k].after_landed ? &a->landed : &a->supplied, memory_order_acquire);
        if (t[k].position > reached) break;
        const size_t at = (size_t)t[k].page * BRIDGE_PAGE + t[k].offset;
        struct ibv_sge sge = { .addr = (uintptr_t)(region + at), .length = t[k].bytes, .lkey = bridge_lkey(L, at) };
        struct ibv_send_wr wr = { .wr_id = k, .sg_list = &sge, .num_sge = 1, .opcode = IBV_WR_SEND, .send_flags = IBV_SEND_SIGNALED }, *bad;
        if (ibv_post_send(L->qp, &wr, &bad)) break;
        L->posted_sends++; L->next_send = k + 1;
      }
      /* completions: a receive's lands, a send's has been read */
      struct ibv_wc wc[64];
      const int got = ibv_poll_cq(L->cq, 64, wc);
      for (int i = 0; i < got; i++) {
        if (wc[i].status != IBV_WC_SUCCESS) { atomic_store(&w->sent, ~0ull); atomic_store(&w->landed, ~0ull); continue; }
        atomic_fetch_add_explicit(wc[i].opcode & IBV_WC_RECV ? &w->landed : &w->sent, 1, memory_order_release);
      }
    }
  }
}

#ifdef BRIDGE_MAIN
int main(int argc, char **argv) {
  if (argc != 2) { fprintf(stderr, "usage: bridge <configuration>\n"); return 2; }
  const int e = bridge_serve(argv[1]);
  fprintf(stderr, "bridge: %s\n", strerror(e));
  return 1;
}
#endif
#endif
