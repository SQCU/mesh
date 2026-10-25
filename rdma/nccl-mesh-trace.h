/* design/whole-function.md#measurements. An observer of existing stream words,
   never a transport gate. The caller retains comm and stops after the complete
   function's downstream consumers finish. Nothing on a data path calls this. */
struct traced_entry {
  struct mesh_entry entry;
  struct ncclMeshWindow window;
  uint64_t previous_queue, previous_ns;
  uint64_t *payload_prefix;
  uint32_t channel;
};
struct ncclMeshTrace {
  struct ncclComm *comm;
  pthread_t thread;
  _Atomic int stop;
  int status, count, room, joined;
  uint64_t *cursor, *units, *queued, *peak;
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
      const uint64_t units = record->entry.units;
      if (!units || units > SIZE_MAX / sizeof(uint64_t) - 1 || trace->units[slot] > UINT64_MAX - units) {
        trace->status = EINVAL; break;
      }
      record->payload_prefix = malloc(((size_t)units + 1) * sizeof *record->payload_prefix);
      if (!record->payload_prefix) { trace->status = ENOMEM; break; }
      /* The compiler records lengths before presenting the entry. Its region
         table may have retired before this observer's first sample. */
      trace->status = mesh_byte_account_prefix(channel->payload + direction, trace->units[slot], units, record->payload_prefix);
      if (trace->status) break;
      record->window = (struct ncclMeshWindow){.cable = channel->cable, .peer = (uint32_t)channel->peer, .direction = direction,
        .wire = *channel->wire, .first = trace->units[slot], .last = trace->units[slot] + units,
        .bytes = record->payload_prefix[units], .relay = !direction && record->entry.word != 0};
      trace->units[slot] += units;
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
    const uint64_t supplied = eligible < record->entry.units ? eligible : record->entry.units;
    const uint64_t done = sent > w->first ? sent - w->first : 0;
    const uint64_t queued = supplied > done ? record->payload_prefix[supplied] - record->payload_prefix[done] : 0;
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
  trace->units = calloc(2 * comm->count + 1, sizeof *trace->units);
  trace->queued = calloc(comm->count + 1, sizeof *trace->queued);
  trace->peak = calloc(comm->count + 1, sizeof *trace->peak);
  if (!trace->entries || !trace->cursor || !trace->units || !trace->queued || !trace->peak) {
    free(trace->entries); free(trace->cursor); free(trace->units); free(trace->queued); free(trace->peak); free(trace);
    return fail(ncclSystemError, "out of memory");
  }
  pthread_mutex_lock(&group_lock);
  for (uint32_t h = 0; h < comm->count; h++) {
    comm->channels[h].trace_readers++;
    for (uint32_t direction = 0; direction < 2; direction++)
      trace->cursor[2 * h + direction] = atomic_load_explicit(comm->channels[h].view.presented[direction], memory_order_acquire);
    trace->units[2 * h] = comm->channels[h].out.n;
    trace->units[2 * h + 1] = comm->channels[h].in.n;
  }
  trace->previous_ns = mesh_clock_ns();
  const int status = pthread_create(&trace->thread, NULL, trace_run, trace);
  if (status) {
    for (uint32_t h = 0; h < comm->count; h++) comm->channels[h].trace_readers--;
    pthread_mutex_unlock(&group_lock);
    free(trace->entries); free(trace->cursor); free(trace->units); free(trace->queued); free(trace->peak); free(trace);
    return fail(ncclSystemError, "trace observer: %s", strerror(status));
  }
  pthread_mutex_unlock(&group_lock);
  *out = trace;
  return ncclSuccess;
}

static void trace_finish(ncclMeshTrace_t trace) {
  if (trace->joined) return;
  atomic_store_explicit(&trace->stop, 1, memory_order_release);
  pthread_join(trace->thread, NULL);
  trace->joined = 1;
}

ncclResult_t ncclMeshTraceSpanLengths(ncclMeshTrace_t trace, uint64_t *out, size_t room, size_t *count) {
  if (!trace || !count) return fail(ncclInvalidArgument, "trace span storage");
  trace_finish(trace);
  size_t needed = 0;
  for (int k = 0; k < trace->count; k++) {
    if (trace->entries[k].entry.units > SIZE_MAX - needed) return fail(ncclSystemError, "trace span count overflow");
    needed += trace->entries[k].entry.units;
  }
  *count = needed;
  if (trace->status) return fail(ncclSystemError, "trace observer: %s", strerror(trace->status));
  if (!out) return ncclSuccess;
  if (room < needed) return fail(ncclInvalidArgument, "trace span storage has %zu values, requires %zu", room, needed);
  size_t at = 0;
  for (int k = 0; k < trace->count; k++) {
    const struct traced_entry *record = trace->entries + k;
    for (uint64_t u = 0; u < record->entry.units; u++)
      out[at++] = record->payload_prefix[u + 1] - record->payload_prefix[u];
  }
  return ncclSuccess;
}

ncclResult_t ncclMeshTraceStop(ncclMeshTrace_t trace, struct ncclMeshWindow *out, int room, int *count,
                             struct ncclMeshTraceStats *stats) {
  if (!trace || !out || !count || !stats || room < trace->room) return fail(ncclInvalidArgument, "trace result storage");
  trace_finish(trace);
  *count = trace->count;
  *stats = (struct ncclMeshTraceStats){.polls = trace->polls, .max_gap_ns = trace->max_gap_ns};
  for (int k = 0; k < trace->count; k++) {
    out[k] = trace->entries[k].window;
    out[k].cable_peak_queue_bytes = trace->peak[trace->entries[k].channel];
    stats->incomplete += !out[k].end_hi_ns;
    free(trace->entries[k].payload_prefix);
  }
  const int status = trace->status;
  pthread_mutex_lock(&group_lock);
  for (uint32_t h = 0; h < trace->comm->count; h++) {
    struct channel *channel = trace->comm->channels + h;
    channel->trace_readers--;
    channel_account_prune(channel);
  }
  pthread_mutex_unlock(&group_lock);
  free(trace->entries); free(trace->cursor); free(trace->units); free(trace->queued); free(trace->peak); free(trace);
  return status ? fail(ncclSystemError, "trace observer: %s", strerror(status)) : ncclSuccess;
}
