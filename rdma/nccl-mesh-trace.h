/* design/whole-function.md#measurements. An observer of existing stream words,
   never a transport gate. The caller retains comm and stops after the complete
   function's downstream consumers finish. Nothing on a data path calls this. */
struct traced_entry {
  struct mesh_entry entry;
  struct ncclMeshWindow window;
  uint64_t previous_queue, previous_ns;
  uint32_t channel;
};
struct ncclMeshTrace {
  struct ncclComm *comm;
  pthread_t thread;
  _Atomic int stop;
  int status, count, room;
  uint64_t *cursor, *page, *queued, *peak;
  uint64_t previous_ns, polls, max_gap_ns;
  struct traced_entry *entries;
};

static void trace_read(struct ncclMeshTrace *trace) {
  struct ncclComm *comm = trace->comm;
  const uint64_t before = mesh_clock_ns();
  memset(trace->queued, 0, (comm->count + 1) * sizeof *trace->queued);
  for (uint32_t h = 0; h < comm->count && !trace->status; h++) {
    const struct channel *channel = comm->channels + h;
    for (uint32_t direction = 0; direction < 2 && !trace->status; direction++) {
    const uint32_t slot = 2 * h + direction;
    const uint64_t count = atomic_load_explicit(channel->view.presented[direction], memory_order_acquire);
    if (count - trace->cursor[slot] > MESH_STREAM_ENTRIES) { trace->status = EOVERFLOW; break; }
    while (trace->cursor[slot] < count) {
      if (trace->count == trace->room) { trace->status = ENOSPC; break; }
      struct traced_entry *record = trace->entries + trace->count++;
      record->entry = channel->view.list[direction][trace->cursor[slot] % MESH_STREAM_ENTRIES];
      record->channel = h;
      record->previous_ns = trace->previous_ns;
      record->window = (struct ncclMeshWindow){.cable = channel->cable, .peer = (uint32_t)channel->peer, .direction = direction,
        .wire = *channel->wire, .first = trace->page[slot], .last = trace->page[slot] + record->entry.pages,
        .bytes = record->entry.pages * comm->page};
      trace->page[slot] += record->entry.pages;
      trace->cursor[slot]++;
    }
    }
  }
  for (int k = 0; k < trace->count && !trace->status; k++) {
    struct traced_entry *record = trace->entries + k;
    struct ncclMeshWindow *w = &record->window;
    if (w->end_hi_ns) continue;
    const struct channel *channel = comm->channels + record->channel;
    if (w->direction) {
      const uint64_t read_before = mesh_clock_ns();
      const uint64_t landed = atomic_load_explicit(channel->view.landed, memory_order_acquire);
      if (landed == UINT64_MAX) { trace->status = ENOLINK; break; }
      const uint64_t now = mesh_clock_ns();
      if (!w->start_hi_ns && landed > w->first) {
        w->start_lo_ns = record->previous_ns;
        w->start_hi_ns = now;
      }
      if (landed >= w->last) {
        w->end_lo_ns = record->previous_ns;
        w->end_hi_ns = now;
      }
      record->previous_ns = read_before;
      continue;
    }
    const _Atomic uint64_t *word = record->entry.word ?
      (const _Atomic uint64_t *)((const char *)comm->attach->context.M + record->entry.word) : channel->view.ready;
    const uint64_t read_before = mesh_clock_ns();
    const uint64_t ready = atomic_load_explicit(word, memory_order_acquire);
    const uint64_t sent = atomic_load_explicit(channel->view.sent, memory_order_acquire);
    if (ready == UINT64_MAX || sent == UINT64_MAX) { trace->status = ENOLINK; break; }
    const uint64_t eligible = ready > record->entry.base ? ready - record->entry.base : 0;
    const uint64_t supplied = eligible < record->entry.pages ? eligible : record->entry.pages;
    const uint64_t done = sent > w->first ? sent - w->first : 0;
    const uint64_t queued = supplied > done ? (supplied - done) * comm->page : 0;
    const uint64_t now = mesh_clock_ns();
    if (!w->start_hi_ns && (supplied || done)) {
      w->start_lo_ns = record->previous_ns;
      w->start_hi_ns = now;
    }
    if (sent >= w->last) {
      w->end_lo_ns = record->previous_ns;
      w->end_hi_ns = now;
    }
    w->queue_byte_seconds += 0.5 * (double)(record->previous_queue + queued) * (now - record->previous_ns) * 1e-9;
    if (queued > w->peak_queue_bytes) w->peak_queue_bytes = queued;
    record->previous_queue = queued;
    record->previous_ns = read_before;
    trace->queued[record->channel] += queued;
  }
  for (uint32_t h = 0; h < comm->count; h++)
    if (trace->queued[h] > trace->peak[h]) trace->peak[h] = trace->queued[h];
  const uint64_t after = mesh_clock_ns();
  if (after - trace->previous_ns > trace->max_gap_ns) trace->max_gap_ns = after - trace->previous_ns;
  trace->previous_ns = before;
  trace->polls++;
}

static void *trace_run(void *argument) {
  struct ncclMeshTrace *trace = argument;
  while (!atomic_load_explicit(&trace->stop, memory_order_acquire) && !trace->status) {
    trace_read(trace);
    sched_yield();
  }
  trace_read(trace);
  return NULL;
}

uint64_t ncclMeshTraceClock(void) { return mesh_clock_ns(); }

ncclResult_t ncclMeshTraceStart(ncclComm_t comm, int room, ncclMeshTrace_t *out) {
  if (!comm || !out || room <= 0 || (unsigned)room > MESH_STREAM_ENTRIES)
    return fail(ncclInvalidArgument, "trace capacity must be within the stream entry ring");
  struct ncclMeshTrace *trace = calloc(1, sizeof *trace);
  if (!trace) return fail(ncclSystemError, "out of memory");
  trace->comm = comm; trace->room = room;
  atomic_init(&trace->stop, 0);
  trace->entries = calloc((size_t)room, sizeof *trace->entries);
  trace->cursor = calloc(2 * comm->count + 1, sizeof *trace->cursor);
  trace->page = calloc(2 * comm->count + 1, sizeof *trace->page);
  trace->queued = calloc(comm->count + 1, sizeof *trace->queued);
  trace->peak = calloc(comm->count + 1, sizeof *trace->peak);
  if (!trace->entries || !trace->cursor || !trace->page || !trace->queued || !trace->peak) {
    free(trace->entries); free(trace->cursor); free(trace->page); free(trace->queued); free(trace->peak); free(trace);
    return fail(ncclSystemError, "out of memory");
  }
  for (uint32_t h = 0; h < comm->count; h++) {
    for (uint32_t direction = 0; direction < 2; direction++)
      trace->cursor[2 * h + direction] = atomic_load_explicit(comm->channels[h].view.presented[direction], memory_order_acquire);
    trace->page[2 * h] = pages_to_word(comm->channels[h].out);
    trace->page[2 * h + 1] = pages_to_word(comm->channels[h].in);
  }
  trace->previous_ns = mesh_clock_ns();
  const int status = pthread_create(&trace->thread, NULL, trace_run, trace);
  if (status) {
    free(trace->entries); free(trace->cursor); free(trace->page); free(trace->queued); free(trace->peak); free(trace);
    return fail(ncclSystemError, "trace observer: %s", strerror(status));
  }
  *out = trace;
  return ncclSuccess;
}

ncclResult_t ncclMeshTraceStop(ncclMeshTrace_t trace, struct ncclMeshWindow *out, int room, int *count,
                             struct ncclMeshTraceStats *stats) {
  if (!trace || !out || !count || !stats || room < trace->room) return fail(ncclInvalidArgument, "trace result storage");
  atomic_store_explicit(&trace->stop, 1, memory_order_release);
  pthread_join(trace->thread, NULL);
  *count = trace->count;
  *stats = (struct ncclMeshTraceStats){.polls = trace->polls, .max_gap_ns = trace->max_gap_ns};
  for (int k = 0; k < trace->count; k++) {
    out[k] = trace->entries[k].window;
    out[k].cable_peak_queue_bytes = trace->peak[trace->entries[k].channel];
    stats->incomplete += !out[k].end_hi_ns;
  }
  const int status = trace->status;
  free(trace->entries); free(trace->cursor); free(trace->page); free(trace->queued); free(trace->peak); free(trace);
  return status ? fail(ncclSystemError, "trace observer: %s", strerror(status)) : ncclSuccess;
}
