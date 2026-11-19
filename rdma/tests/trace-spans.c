/* Exact payload accounting and observer-owned descriptor lifetime. */
#include <assert.h>
#include <errno.h>
#include <pthread.h>
#include <sched.h>
#include <stdlib.h>
#include <string.h>
#include "mesh-call.h"
#include "nccl.h"
#include "mesh-byte-account.h"

struct mesh_entry { uint64_t spans, units, word, base; };
#define CHANNEL_LOG 4096u
struct channel_view { struct mesh_entry *list[2]; _Atomic uint64_t *presented[2], *ready, *sent, *landed; };
struct channel {
  uint32_t cable;
  int peer;
  uint64_t *wire;
  struct channel_view view;
  struct mesh_units out, in;
  struct mesh_byte_account payload[2];
  unsigned trace_readers;
};
struct attach { struct mesh_ctx context; };
struct ncclComm { uint32_t count; struct channel *channels; struct attach *attach; struct mesh_fabric fabric; };
static ncclResult_t fail(ncclResult_t result, const char *format, ...) { (void)format; return result; }
static pthread_mutex_t group_lock = PTHREAD_MUTEX_INITIALIZER;
static void channel_account_prune(struct channel *c) {
  if (c->trace_readers) return;
  uint64_t ready = atomic_load(c->view.ready), sent = atomic_load(c->view.sent);
  assert(!mesh_byte_account_checkpoint(c->payload, ready));
  mesh_byte_account_prune(c->payload, sent);
  mesh_byte_account_prune(c->payload + 1, atomic_load(c->view.landed));
}
#include "nccl-mesh-trace.h"

int main(void) {
  struct hdr *region = calloc(1, 65536);
  assert(region);
  region->length = 65536;
  region->data_off = 4096;
  struct mesh_dma_span *spans = (struct mesh_dma_span *)((char *)region + 4096);
  spans[0] = (struct mesh_dma_span){8192, 1};
  spans[1] = (struct mesh_dma_span){8193, 4097};
  spans[2] = (struct mesh_dma_span){12290, 17};
  struct mesh_entry lists[2] = {{4096, 3, 0, 0}, {4096, 3, 0, 0}};
  _Atomic uint64_t presented[2] = {1, 1}, ready = 3, sent = 1, landed = 0;
  uint64_t wire = 17;
  struct channel channel = {.cable = 0, .peer = 1, .wire = &wire,
    .view = {.list = {lists, lists + 1}, .presented = {presented, presented + 1},
             .ready = &ready, .sent = &sent, .landed = &landed}};
  struct attach attach = {.context = {.M = region}};
  const struct mesh_cable link = {.wire = 17, .bandwidth = 1, .node = {0, 1}};
  struct ncclComm comm = {.count = 1, .channels = &channel, .attach = &attach, .fabric = {2, 1, &link}};
  for (int direction = 0; direction < 2; direction++)
    assert(!mesh_byte_account_append(channel.payload + direction, 0, &spans[0].bytes, 3, sizeof *spans));
  /* First observation may be after the original table has been reclaimed. */
  spans[1].bytes = 999;
  struct ncclMeshTrace *trace = calloc(1, sizeof *trace);
  assert(trace);
  trace->comm = &comm;
  trace->room = 4;
  trace->joined = 1; /* Deterministic observer samples; no worker in this fixture. */
  channel.trace_readers = 1;
  trace->entries = calloc(4, sizeof *trace->entries);
  trace->cursor = calloc(2, sizeof *trace->cursor);
  trace->units = calloc(2, sizeof *trace->units);
  trace->queued = calloc(2, sizeof *trace->queued);
  trace->peak = calloc(2, sizeof *trace->peak);
  trace->previous_ns = mesh_clock_ns();
  ncclMeshTrace_t another = NULL;
  assert(ncclMeshTraceStart(&comm, 4, &another) == ncclSuccess);
  assert(channel.trace_readers == 2);
  trace_read(trace);
  assert(!trace->status && trace->count == 2);
  assert(trace->entries[0].window.bytes == 4115);
  assert(trace->entries[0].window.first == 0 && trace->entries[0].window.last == 3);
  assert(trace->entries[0].previous_queue == 4114);
  assert(trace->entries[0].window.peak_queue_bytes == 4114);

  /* Subsequent observations also use immutable host-owned payload prefixes. */
  atomic_store(&sent, 2);
  atomic_store(&landed, 1);
  trace_read(trace);
  assert(trace->entries[0].previous_queue == 17);
  assert(trace->entries[1].window.start_hi_ns);
  atomic_store(&sent, 3);
  atomic_store(&landed, 3);
  trace_read(trace);
  assert(trace->entries[0].previous_queue == 0);
  channel_account_prune(&channel);
  assert(atomic_load(&channel.payload[0].head)); /* both observers retain history */

  size_t count = 0;
  uint64_t lengths[6] = {0};
  assert(ncclMeshTraceSpanLengths(trace, NULL, 0, &count) == ncclSuccess && count == 6);
  assert(ncclMeshTraceSpanLengths(trace, lengths, 5, &count) == ncclInvalidArgument);
  assert(ncclMeshTraceSpanLengths(trace, lengths, 6, &count) == ncclSuccess);
  const uint64_t expected[] = {1, 4097, 17};
  for (size_t i = 0; i < 6; i++) assert(lengths[i] == expected[i % 3]);
  struct ncclMeshWindow windows[4];
  struct ncclMeshTraceStats stats;
  int entries;
  assert(ncclMeshTraceStop(trace, windows, 4, &entries, &stats) == ncclSuccess);
  assert(entries == 2 && !stats.incomplete);
  assert(windows[0].bytes == 4115 && windows[1].bytes == 4115);
  assert(windows[0].channel_peak_queue_bytes == 4114 && windows[0].pair == 17);
  assert(channel.trace_readers == 1 && atomic_load(&channel.payload[0].head));
  assert(ncclMeshTraceStop(another, windows, 4, &entries, &stats) == ncclSuccess);
  assert(!channel.trace_readers && !atomic_load(&channel.payload[0].head) && !atomic_load(&channel.payload[1].head));
  uint64_t total;
  assert(!mesh_byte_account_at(channel.payload, 3, &total) && total == 4115);
  for (int direction = 0; direction < 2; direction++) mesh_byte_account_destroy(channel.payload + direction);
  free(region);
  return 0;
}
