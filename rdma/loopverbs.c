/* loopverbs (mesh 800fe6bc^, on the resident bridge; design/transport.md): the ibverbs entry
   points the bridge uses (mesh-verbs.h, mesh-link.c, mesh-program.c), over shared memory between the processes of
   one machine, so N bridges on one host are N nodes (make mesh-flow-loop: the bridge's sources built with
   MESH_LOOP_FABRIC and linked with this in place of -lrdma; the deployed bridge is built from the same source without
   either).  Every queue pair has, in a
   shared fabric segment (its name the bridge's configuration's, loop_fabric), a byte ring of the SENDs addressed to it,
   one producer (its peer's process) and one consumer (its owner).  A SEND is copied into its destination's ring and
   completes on the sender as it is delivered there (a SEND to a queue pair gone or made again since its peer connected
   is dropped and completes, as UC drops it); the owner drains its ring into its receives in the order posted, a SEND
   waiting for its receive as TB5's credit flow control holds it (TN3205), and completes each receive with its byte
   count, or IBV_WC_LOC_LEN_ERR where the SEND is longer than the receive.  A queue pair moved to the error state
   flushes its posted receives and pending SENDs (IBV_WC_WR_FLUSH_ERR), as mesh-verbs.h pair_flush expects.  A device
   holds at most 10 queue pairs (a TB5 device's max_qp 11, 10 usable: design/transport.md), a SEND at most half a ring
   (the port's max_msg_sz).  Pairing is the bridges' own (their control sockets on the loopback interface).  Only the
   bridge's verbs are provided: one scatter-gather entry a request, SEND and RECV on unreliable-connected queue pairs,
   one port. */
#include <infiniband/verbs.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

enum { LOOP_QPS = 1024, LOOP_DEVICES = 64, LOOP_DEVICE_QPS = 10 };
#define LOOP_RING ((uint64_t)8 << 20)

/* a queue pair's incoming SENDs: `head` the bytes written, `tail` the bytes consumed, each record an 8-byte length then
   the payload rounded up to 8 bytes; `used` nonzero while a queue pair holds the ring, `generation` moved at each
   making */
struct loop_ring {
  _Alignas(128) _Atomic uint64_t head;
  _Alignas(128) _Atomic uint64_t tail;
  _Atomic uint32_t used, generation;
  unsigned char data[LOOP_RING];
};
struct loop_fabric {
  _Atomic uint32_t next;
  struct loop_ring rings[LOOP_QPS];
};

struct loop_cq {
  struct ibv_cq cq;
  struct ibv_wc *entries;
  int capacity, head, count, idle;
};
struct loop_receive { uint64_t wr_id, address; uint32_t length; };
struct loop_send { uint64_t wr_id, address; uint32_t length; };
struct loop_qp {
  struct ibv_qp qp;
  struct loop_cq *send_cq, *recv_cq;
  struct loop_receive *posted;
  struct loop_send *pending;
  uint32_t receive_capacity, posted_head, posted_count;
  uint32_t send_capacity, pending_head, pending_count;
  uint32_t destination, generation;
  int connected;
  struct loop_qp *next;
};

static pthread_mutex_t lock_ = PTHREAD_MUTEX_INITIALIZER;
static struct loop_fabric *fabric_;
static char fabric_name_[64];
static struct loop_qp *qps_;
static struct ibv_device devices_[LOOP_DEVICES], *list_[LOOP_DEVICES + 1];
static uint32_t keys_ = 1;

/* the fabric segment's name, the bridge's configuration's (mesh-link.c configuration_read, its `loop` `fabric`), before
   any device is opened */
void loop_fabric(const char *name) { snprintf(fabric_name_, sizeof fabric_name_, "%s", name ? name : ""); }

/* the fabric segment, made and sized by the first process to need it (a shared memory object is sized once), which the
   others wait for */
static struct loop_fabric *fabric(void) {
  if (fabric_) return fabric_;
  if (!fabric_name_[0]) { errno = ENOENT; return NULL; }
  int file = shm_open(fabric_name_, O_CREAT | O_EXCL | O_RDWR, 0600);
  if (file >= 0 && ftruncate(file, sizeof(struct loop_fabric))) { close(file); return NULL; }
  if (file < 0) {
    if (errno != EEXIST || (file = shm_open(fabric_name_, O_RDWR, 0600)) < 0) return NULL;
    struct stat info;
    for (int polls = 0; !fstat(file, &info) && info.st_size < (off_t)sizeof(struct loop_fabric); polls++)
      if (polls == 100000) { close(file); errno = ETIMEDOUT; return NULL; }
      else usleep(100);
  }
  void *mapped = mmap(NULL, sizeof(struct loop_fabric), PROT_READ | PROT_WRITE, MAP_SHARED, file, 0);
  close(file);
  if (mapped == MAP_FAILED) return NULL;
  return fabric_ = mapped;
}

static int complete(struct loop_cq *cq, struct ibv_wc entry) {
  if (cq->count == cq->capacity) return ENOSPC;
  cq->entries[(cq->head + cq->count++) % cq->capacity] = entry;
  return 0;
}

static void ring_copy(struct loop_ring *ring, uint64_t at, const void *from, uint64_t bytes) {
  const uint64_t start = at % LOOP_RING, first = bytes < LOOP_RING - start ? bytes : LOOP_RING - start;
  memcpy(ring->data + start, from, first);
  if (first < bytes) memcpy(ring->data, (const unsigned char *)from + first, bytes - first);
}

static void ring_read(struct loop_ring *ring, uint64_t at, void *to, uint64_t bytes) {
  const uint64_t start = at % LOOP_RING, first = bytes < LOOP_RING - start ? bytes : LOOP_RING - start;
  memcpy(to, ring->data + start, first);
  if (first < bytes) memcpy((unsigned char *)to + first, ring->data, bytes - first);
}

/* the queue pair's SENDs posted and not yet delivered, into its destination's ring while it has room; one whose
   destination is gone or made again completes undelivered */
static int deliver(struct loop_qp *q) {
  if (!q->connected) return 0;
  struct loop_ring *ring = &fabric_->rings[q->destination];
  while (q->pending_count) {
    struct loop_send *send = q->pending + q->pending_head;
    const int gone = !atomic_load_explicit(&ring->used, memory_order_acquire) ||
                     atomic_load_explicit(&ring->generation, memory_order_acquire) != q->generation;
    if (!gone) {
      const uint64_t record = 8 + ((send->length + 7) & ~(uint64_t)7);
      const uint64_t head = atomic_load_explicit(&ring->head, memory_order_relaxed), tail = atomic_load_explicit(&ring->tail, memory_order_acquire);
      if (record > LOOP_RING - (head - tail)) break;
      const uint64_t length = send->length;
      ring_copy(ring, head, &length, 8);
      ring_copy(ring, head + 8, (const void *)(uintptr_t)send->address, send->length);
      atomic_store_explicit(&ring->head, head + record, memory_order_release);
    }
    int error = complete(q->send_cq, (struct ibv_wc){.wr_id = send->wr_id, .status = IBV_WC_SUCCESS, .opcode = IBV_WC_SEND,
                                                    .byte_len = send->length, .qp_num = q->qp.qp_num});
    if (error) return error;
    q->pending_head = (q->pending_head + 1) % q->send_capacity;
    q->pending_count--;
  }
  return 0;
}

/* the queue pair's ring into its posted receives, in order */
static int drain(struct loop_qp *q) {
  struct loop_ring *ring = &fabric_->rings[q->qp.qp_num];
  while (q->posted_count) {
    const uint64_t tail = atomic_load_explicit(&ring->tail, memory_order_relaxed), head = atomic_load_explicit(&ring->head, memory_order_acquire);
    if (tail == head) break;
    uint64_t length;
    ring_read(ring, tail, &length, 8);
    struct loop_receive *receive = q->posted + q->posted_head;
    struct ibv_wc entry = {.wr_id = receive->wr_id, .opcode = IBV_WC_RECV, .byte_len = (uint32_t)length, .qp_num = q->qp.qp_num};
    if (length > receive->length) entry.status = IBV_WC_LOC_LEN_ERR;
    else ring_read(ring, tail + 8, (void *)(uintptr_t)receive->address, length);
    atomic_store_explicit(&ring->tail, tail + 8 + ((length + 7) & ~(uint64_t)7), memory_order_release);
    int error = complete(q->recv_cq, entry);
    if (error) return error;
    q->posted_head = (q->posted_head + 1) % q->receive_capacity;
    q->posted_count--;
  }
  return 0;
}

static int poll_cq(struct ibv_cq *cq, int entries, struct ibv_wc *wc) {
  struct loop_cq *queue = (struct loop_cq *)cq;
  int error = 0, taken = 0;
  pthread_mutex_lock(&lock_);
  for (struct loop_qp *q = qps_; q && !error; q = q->next) {
    error = deliver(q);
    if (!error && q->recv_cq == queue && q->connected) error = drain(q);
  }
  while (!error && taken < entries && queue->count) {
    wc[taken++] = queue->entries[queue->head];
    queue->head = (queue->head + 1) % queue->capacity;
    queue->count--;
  }
  const int idle = taken ? (queue->idle = 0) : ++queue->idle;
  pthread_mutex_unlock(&lock_);
  /* a bridge's progress threads poll without pause while anything moves, as a device's completion queue lets them; N
     bridges' on one host's cores yield once a queue has stayed empty, so the processes with work run */
  if (idle > 64) usleep(20);
  else if (idle > 8) sched_yield();
  return error ? -1 : taken;
}

static int post_send(struct ibv_qp *qp, struct ibv_send_wr *wr, struct ibv_send_wr **bad) {
  struct loop_qp *q = (struct loop_qp *)qp;
  int error = 0;
  pthread_mutex_lock(&lock_);
  for (; wr && !error; wr = wr->next) {
    if (wr->opcode != IBV_WR_SEND || wr->num_sge != 1 || q->pending_count == q->send_capacity || wr->sg_list[0].length > LOOP_RING / 2) {
      error = q->pending_count == q->send_capacity ? ENOMEM : EINVAL;
      break;
    }
    q->pending[(q->pending_head + q->pending_count++) % q->send_capacity] =
        (struct loop_send){.wr_id = wr->wr_id, .address = wr->sg_list[0].addr, .length = wr->sg_list[0].length};
  }
  if (!error) error = deliver(q);
  pthread_mutex_unlock(&lock_);
  if (error && bad) *bad = wr;
  return error;
}

static int post_recv(struct ibv_qp *qp, struct ibv_recv_wr *wr, struct ibv_recv_wr **bad) {
  struct loop_qp *q = (struct loop_qp *)qp;
  int error = 0;
  pthread_mutex_lock(&lock_);
  for (; wr && !error; wr = wr->next) {
    if (wr->num_sge != 1 || q->posted_count == q->receive_capacity) { error = ENOMEM; break; }
    q->posted[(q->posted_head + q->posted_count++) % q->receive_capacity] =
        (struct loop_receive){.wr_id = wr->wr_id, .address = wr->sg_list[0].addr, .length = wr->sg_list[0].length};
  }
  pthread_mutex_unlock(&lock_);
  if (error && bad) *bad = wr;
  return error;
}

struct ibv_device **ibv_get_device_list(int *count) {
  for (int i = 0; i < LOOP_DEVICES; i++) {
    devices_[i].node_type = IBV_NODE_CA;
    devices_[i].transport_type = IBV_TRANSPORT_IB;
    snprintf(devices_[i].name, sizeof devices_[i].name, "rdma_loop%d", i);
    list_[i] = devices_ + i;
  }
  list_[LOOP_DEVICES] = NULL;
  if (count) *count = LOOP_DEVICES;
  return list_;
}

void ibv_free_device_list(struct ibv_device **list) { (void)list; }

const char *ibv_get_device_name(struct ibv_device *device) { return device->name; }

/* each open device's queue pairs, refused past LOOP_DEVICE_QPS as the TB5 device refuses them */
static struct { struct ibv_context *context; int qps; } opened_[LOOP_DEVICES * 4];
static int *device_qps(struct ibv_context *context) {
  for (size_t i = 0; i < sizeof opened_ / sizeof *opened_; i++)
    if (opened_[i].context == context) return &opened_[i].qps;
  for (size_t i = 0; i < sizeof opened_ / sizeof *opened_; i++)
    if (!opened_[i].context) { opened_[i].context = context; opened_[i].qps = 0; return &opened_[i].qps; }
  return NULL;
}

struct ibv_context *ibv_open_device(struct ibv_device *device) {
  if (!fabric()) return NULL;
  struct ibv_context *context = calloc(1, sizeof *context);
  if (!context) return NULL;
  context->device = device;
  context->ops.poll_cq = poll_cq;
  context->ops.post_send = post_send;
  context->ops.post_recv = post_recv;
  pthread_mutex_init(&context->mutex, NULL);
  return context;
}

int ibv_close_device(struct ibv_context *context) {
  pthread_mutex_lock(&lock_);
  for (size_t i = 0; i < sizeof opened_ / sizeof *opened_; i++)
    if (opened_[i].context == context) opened_[i].context = NULL;
  pthread_mutex_unlock(&lock_);
  pthread_mutex_destroy(&context->mutex);
  free(context);
  return 0;
}

int ibv_query_device(struct ibv_context *context, struct ibv_device_attr *attributes) {
  (void)context;
  memset(attributes, 0, sizeof *attributes);
  attributes->max_mr_size = (uint64_t)1 << 30;
  attributes->max_qp = LOOP_DEVICE_QPS + 1;
  attributes->max_qp_wr = 4095;
  attributes->max_sge = 1;
  attributes->max_cq = 64;
  attributes->max_cqe = 65535;
  attributes->max_mr = 1024;
  attributes->max_pd = 64;
  attributes->phys_port_cnt = 1;
  return 0;
}

/* the port: active, 4x of 25 Gb/s (mesh-verbs.h's speeds and widths: 100 Gb/s), one GID, a SEND at most half a ring */
#undef ibv_query_port
int ibv_query_port(struct ibv_context *context, uint8_t port, struct _compat_ibv_port_attr *compat) {
  (void)context; (void)port;
  struct ibv_port_attr *attributes = (struct ibv_port_attr *)compat;
  attributes->state = IBV_PORT_ACTIVE;
  attributes->max_mtu = attributes->active_mtu = IBV_MTU_4096;
  attributes->lid = 1;
  attributes->gid_tbl_len = 1;
  attributes->max_msg_sz = (uint32_t)(LOOP_RING / 2);
  attributes->active_width = 2;
  attributes->active_speed = 32;
  attributes->phys_state = 5;
  return 0;
}

/* the port's one address, link-local and nonzero (mesh-verbs.h device_gid takes the first nonzero entry) */
int ibv_query_gid(struct ibv_context *context, uint8_t port, int index, union ibv_gid *gid) {
  (void)port;
  memset(gid, 0, sizeof *gid);
  if (index) return 0;
  gid->raw[0] = 0xfe; gid->raw[1] = 0x80;
  for (int i = 0; i < LOOP_DEVICES; i++)
    if (context && context->device == devices_ + i) gid->raw[15] = (uint8_t)(i + 1);
  if (!gid->raw[15]) gid->raw[15] = 1;
  return 0;
}

struct ibv_pd *ibv_alloc_pd(struct ibv_context *context) {
  struct ibv_pd *pd = calloc(1, sizeof *pd);
  if (pd) pd->context = context;
  return pd;
}

int ibv_dealloc_pd(struct ibv_pd *pd) {
  free(pd);
  return 0;
}

struct ibv_mr *ibv_reg_mr_iova2(struct ibv_pd *pd, void *address, size_t length, uint64_t iova, unsigned int access) {
  (void)iova; (void)access;
  struct ibv_mr *mr = calloc(1, sizeof *mr);
  if (!mr) return NULL;
  pthread_mutex_lock(&lock_);
  *mr = (struct ibv_mr){.context = pd->context, .pd = pd, .addr = address, .length = length, .handle = keys_, .lkey = keys_, .rkey = keys_};
  keys_++;
  pthread_mutex_unlock(&lock_);
  return mr;
}

#undef ibv_reg_mr
struct ibv_mr *ibv_reg_mr(struct ibv_pd *pd, void *address, size_t length, int access) {
  return ibv_reg_mr_iova2(pd, address, length, (uintptr_t)address, (unsigned int)access);
}

int ibv_dereg_mr(struct ibv_mr *mr) {
  free(mr);
  return 0;
}

struct ibv_cq *ibv_create_cq(struct ibv_context *context, int entries, void *cq_context, struct ibv_comp_channel *channel, int vector) {
  (void)channel; (void)vector;
  struct loop_cq *cq = calloc(1, sizeof *cq);
  if (!cq || !(cq->entries = calloc((size_t)entries + 1, sizeof *cq->entries))) { free(cq); errno = ENOMEM; return NULL; }
  cq->capacity = entries + 1;
  cq->cq.context = context;
  cq->cq.cq_context = cq_context;
  cq->cq.cqe = entries;
  return &cq->cq;
}

int ibv_destroy_cq(struct ibv_cq *cq) {
  struct loop_cq *queue = (struct loop_cq *)cq;
  free(queue->entries);
  free(queue);
  return 0;
}

/* a queue pair: its number a ring of the fabric no other queue pair holds (each making moves the ring's generation) */
struct ibv_qp *ibv_create_qp(struct ibv_pd *pd, struct ibv_qp_init_attr *init) {
  struct loop_qp *q = calloc(1, sizeof *q);
  if (!q || !fabric()) { free(q); errno = ENOMEM; return NULL; }
  q->receive_capacity = init->cap.max_recv_wr ? init->cap.max_recv_wr : 1;
  q->send_capacity = init->cap.max_send_wr ? init->cap.max_send_wr : 1;
  q->posted = calloc(q->receive_capacity, sizeof *q->posted);
  q->pending = calloc(q->send_capacity, sizeof *q->pending);
  if (!q->posted || !q->pending) { free(q->posted); free(q->pending); free(q); errno = ENOMEM; return NULL; }
  q->send_cq = (struct loop_cq *)init->send_cq;
  q->recv_cq = (struct loop_cq *)init->recv_cq;
  q->qp.context = pd->context;
  q->qp.pd = pd;
  q->qp.send_cq = init->send_cq;
  q->qp.recv_cq = init->recv_cq;
  q->qp.qp_type = init->qp_type;
  q->qp.state = IBV_QPS_RESET;
  pthread_mutex_lock(&lock_);
  int *held = device_qps(pd->context);
  uint32_t number = LOOP_QPS;
  for (uint32_t k = 0; held && *held < LOOP_DEVICE_QPS && k < LOOP_QPS && number == LOOP_QPS; k++) {
    const uint32_t at = (atomic_fetch_add(&fabric_->next, 1)) % LOOP_QPS;
    uint32_t vacant = 0;
    if (atomic_compare_exchange_strong(&fabric_->rings[at].used, &vacant, 1)) number = at;
  }
  if (number == LOOP_QPS) {
    pthread_mutex_unlock(&lock_);
    free(q->posted); free(q->pending); free(q);
    errno = ENOMEM;
    return NULL;
  }
  ++*held;
  q->qp.qp_num = number;
  struct loop_ring *ring = &fabric_->rings[number];
  atomic_store(&ring->head, 0);
  atomic_store(&ring->tail, 0);
  atomic_fetch_add(&ring->generation, 1);
  q->next = qps_;
  qps_ = q;
  pthread_mutex_unlock(&lock_);
  return &q->qp;
}

/* a queue pair's flush, moved to the error state: its posted receives and pending SENDs completed IBV_WC_WR_FLUSH_ERR */
static void flush(struct loop_qp *q) {
  for (; q->posted_count; q->posted_count--, q->posted_head = (q->posted_head + 1) % q->receive_capacity)
    complete(q->recv_cq, (struct ibv_wc){.wr_id = q->posted[q->posted_head].wr_id, .status = IBV_WC_WR_FLUSH_ERR, .opcode = IBV_WC_RECV,
                                        .qp_num = q->qp.qp_num});
  for (; q->pending_count; q->pending_count--, q->pending_head = (q->pending_head + 1) % q->send_capacity)
    complete(q->send_cq, (struct ibv_wc){.wr_id = q->pending[q->pending_head].wr_id, .status = IBV_WC_WR_FLUSH_ERR, .opcode = IBV_WC_SEND,
                                        .qp_num = q->qp.qp_num});
  q->connected = 0;
}

int ibv_modify_qp(struct ibv_qp *qp, struct ibv_qp_attr *attributes, int mask) {
  struct loop_qp *q = (struct loop_qp *)qp;
  pthread_mutex_lock(&lock_);
  if (mask & IBV_QP_DEST_QPN) {
    q->destination = attributes->dest_qp_num % LOOP_QPS;
    q->generation = atomic_load(&fabric_->rings[q->destination].generation);
    q->connected = 1;
  }
  if (mask & IBV_QP_STATE) {
    qp->state = attributes->qp_state;
    if (attributes->qp_state == IBV_QPS_ERR) flush(q);
  }
  pthread_mutex_unlock(&lock_);
  return 0;
}

int ibv_query_qp(struct ibv_qp *qp, struct ibv_qp_attr *attributes, int mask, struct ibv_qp_init_attr *init) {
  (void)mask;
  struct loop_qp *q = (struct loop_qp *)qp;
  memset(attributes, 0, sizeof *attributes);
  memset(init, 0, sizeof *init);
  init->cap = (struct ibv_qp_cap){.max_send_wr = q->send_capacity, .max_recv_wr = q->receive_capacity, .max_send_sge = 1, .max_recv_sge = 1};
  attributes->cap = init->cap;
  attributes->qp_state = qp->state;
  return 0;
}

int ibv_destroy_qp(struct ibv_qp *qp) {
  struct loop_qp *q = (struct loop_qp *)qp;
  pthread_mutex_lock(&lock_);
  for (struct loop_qp **at = &qps_; *at; at = &(*at)->next)
    if (*at == q) { *at = q->next; break; }
  int *held = device_qps(q->qp.context);
  if (held && *held) --*held;
  struct loop_ring *ring = &fabric_->rings[q->qp.qp_num];
  atomic_fetch_add(&ring->generation, 1);
  atomic_store(&ring->used, 0);
  pthread_mutex_unlock(&lock_);
  free(q->posted);
  free(q->pending);
  free(q);
  return 0;
}
