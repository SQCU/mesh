/* The trainer's telemetry: a table of named 64-bit counters in shared memory (a file mapped MAP_SHARED), which the
   trainer's threads update in place with relaxed atomic adds and route-stat reads where it lies (zero copy). Nothing is
   formatted, written or flushed; a reader takes its own samples and computes the rates between them. Counters only
   grow (a reader's two samples give any interval's rate); gauges hold a current value; phases hold the monotonic time
   (ns) each startup phase was reached, 0 until then. */
#ifndef ROUTE_TELEMETRY_H
#define ROUTE_TELEMETRY_H
#include <stdint.h>

#define ROUTE_TEL_MAGIC 0x4c455452u   /* "RTEL" */
#define ROUTE_TEL_VERSION 1
enum route_tel {
  /* counters */
  RT_STEPS, RT_OPTIMIZER_STEPS, RT_TRANSACTIONS, RT_GROUPS, RT_EVALUATIONS, RT_FORWARDS, RT_FORWARD_ROWS, RT_FORWARD_NS,
  RT_EVENTS, RT_SENDS, RT_DRAWS, RT_ENTRIES, RT_WORKER_NS, RT_CREDIT_NS, RT_LEARN_NS, RT_EXCHANGE_ROUNDS, RT_EXCHANGE_NS,
  RT_LAG_SUM, RT_LAG_GROUPS,
  /* gauges */
  RT_WORKERS, RT_RUNS_IN_FLIGHT, RT_EXCHANGE_IN_FLIGHT, RT_LAG_MAX,
  /* phases: the monotonic ns each was reached */
  RT_PHASE_START, RT_PHASE_NETWORKS, RT_PHASE_JOINED, RT_PHASE_FIRST_FORWARD, RT_PHASE_FIRST_STEP, RT_PHASE_DONE,
  RT_COUNT
};
static const char *const route_tel_names[RT_COUNT] = {
  "steps", "optimizer_steps", "transactions", "groups", "evaluations", "forwards", "forward_rows", "forward_ns",
  "events", "sends", "draws", "entries", "worker_ns", "credit_ns", "learn_ns", "exchange_rounds", "exchange_ns",
  "lag_sum", "lag_groups",
  "workers", "runs_in_flight", "exchange_in_flight", "lag_max",
  "phase_start", "phase_networks", "phase_joined", "phase_first_forward", "phase_first_step", "phase_done",
};
#define RT_FIRST_GAUGE RT_WORKERS
#define RT_FIRST_PHASE RT_PHASE_START

struct route_tel_file {
  uint32_t magic, version, rank, world;
  uint64_t pid, clock_hz;   /* clock_hz: the monotonic clock's ticks a second (ns: 1e9) */
  uint64_t v[RT_COUNT];
};

extern struct route_tel_file *route_tel;   /* the mapped table (a private one until route_tel_open) */
/* maps `path` (created or reused, its size the table's) as this process's table; 0, or -1 (the private table kept) */
int route_tel_open(const char *path, int rank, int world);
uint64_t route_tel_now(void);   /* monotonic ns */
static inline void route_tel_add(int k, uint64_t n) { __atomic_fetch_add(&route_tel->v[k], n, __ATOMIC_RELAXED); }
static inline void route_tel_set(int k, uint64_t n) { __atomic_store_n(&route_tel->v[k], n, __ATOMIC_RELAXED); }
static inline void route_tel_max(int k, uint64_t n) {
  uint64_t o = __atomic_load_n(&route_tel->v[k], __ATOMIC_RELAXED);
  while (o < n && !__atomic_compare_exchange_n(&route_tel->v[k], &o, n, 1, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {}
}
static inline void route_tel_phase(int k) { uint64_t z = 0; __atomic_compare_exchange_n(&route_tel->v[k], &z, route_tel_now(), 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED); }
#endif
