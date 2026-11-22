/* The live routing loop's native part (rdma/tests/route/live/README.md): a node's agent attaches its bridge's region
   (read and write, as its clients do), and through these calls
   - reads the view and its contexts (mesh_switch_contexts: the split tables' keys, alike in every bridge of a view),
   - publishes a w for each context into the bridge's split tables (mesh_switch_splits_publish: every piece here then
     draws its split from them, mesh_switch_choose), or withdraws them (the bridge's own rule again),
   - turns the bridge's records on or off and drains them (every draw with its exact behaviour log-probability, every
     piece delivered with the cost it carried: memory only, a ring a process, never a wait: full rings drop and count);
   and the learner, through the same library, scores draws (route_draw_score: d logp / d w at a context's w, split.h)
   and sums their policy gradients. Arrays only; the schedule is the Python's. */
#include "../../../mesh-switched.h"
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static struct hdr *M;
static struct mesh_switch *SW;

int live_attach(void) {
  const int fd = shm_open(mesh_name(), O_RDWR, 0);
  if (fd < 0) return -1;
  struct stat st;
  if (fstat(fd, &st) || st.st_size < (off_t)sizeof(struct hdr)) { close(fd); return -2; }
  void *m = mmap(NULL, (size_t)st.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  if (m == MAP_FAILED) return -3;
  M = m;
  if (M->magic != MESH_MAGIC || M->version != MESH_LAYOUT) return -4;
  SW = mesh_switch_of(M);
  return SW->magic == MESH_SWITCH_MAGIC ? 0 : -5;
}
uint32_t live_record_bytes(void) { return (uint32_t)sizeof(struct mesh_switch_record); }
uint32_t live_draw_bytes(void) { return (uint32_t)sizeof(struct route_draw); }
uint32_t live_self(void) { return M->node; }
/* the view's nodes (ascending: a node's view index its place) and its serial (odd while written) */
uint32_t live_view(uint32_t *nodes, uint64_t *serial) {
  struct mesh_switch_view v; mesh_switch_view_read(SW, &v);
  for (uint32_t i = 0; i < v.nodes; i++) nodes[i] = v.node[i];
  *serial = atomic_load_explicit(&SW->view.serial, memory_order_acquire);
  return v.nodes;
}
/* the contexts, each (source, wanted, owed, pattern) as four uint32 */
uint32_t live_contexts(struct mesh_switch_context *out) {
  struct mesh_switch_view v; mesh_switch_view_read(SW, &v);
  return mesh_switch_contexts(&v, out);
}
void live_publish(const struct mesh_switch_context *keys, uint32_t n, const float *w, uint64_t point) {
  struct mesh_switch_view v; mesh_switch_view_read(SW, &v);
  const float unit = (float)mesh_switch_unit(&v, (double)mesh_switch_piece(M));
  mesh_switch_splits_publish(SW, keys, n, (const float (*)[SPLIT_FS])w, point, unit, 0);
}
void live_withdraw(void) {
  struct mesh_switch_splits *x = &SW->splits;
  atomic_fetch_add_explicit(&x->serial, 1, memory_order_acq_rel);
  x->active = 0;
  atomic_fetch_add_explicit(&x->serial, 1, memory_order_release);
}
void live_logging(int on) { atomic_store_explicit(&SW->logging, on ? 1u : 0u, memory_order_release); }
/* up to `most` records from every ring into out; *dropped the rings' drops so far */
uint32_t live_drain(struct mesh_switch_record *out, uint32_t most, uint64_t *dropped) {
  uint32_t n = 0; uint64_t d = 0;
  for (uint32_t r = 0; r < MESH_SWITCH_LOGS; r++) {
    while (n < most && mesh_switch_log_take(SW, r, out + n)) n++;
    d += atomic_load_explicit(&SW->log[r].dropped, memory_order_relaxed);
  }
  *dropped = d;
  return n;
}

/* rdma/tests/route/sim.c's, over split.h: a draw's log-probability under w and its score (d logp / d w) */
float live_score(const struct route_draw *d, const float *w, float *score) {
  const struct split_set a = split_set_of(d);
  struct split_lse acc; split_lse_init(&acc);
  for (int f = 0; f < SPLIT_FS; f++) score[f] = 0;
  const struct split tsp = split_of(d, d->token);
  if (!tsp.ok || !a.n) { for (int f = 0; f < SPLIT_FS; f++) score[f] = 0; return -INFINITY; }
  const float tth = split_tanh(split_dot(&tsp, w));
  split_add(&tsp, split_slope(tth), score);
  struct split_walk k = split_walk_at(&a, 0, a.n);
  for (; k.left; split_walk_next(&k)) {
    const struct split sp = split_of(d, k.token);
    const float th = split_tanh(split_dot(&sp, w));
    split_lse_add(&acc, split_logit(th), split_slope(th), &sp);
  }
  for (int f = 0; f < SPLIT_FS; f++) score[f] -= acc.e[f] / acc.s;
  return split_logit(tth) - (acc.m + logf(acc.s));
}
/* the policy gradient of n draws (each a route_draw at `stride` bytes from the last), each at row[i] of the table w
   ([rows][FS]) with advantage adv[i]: g[row] += adv x score, clipped to the behaviour's support by its importance
   ratio (exp(logp now - logp behaved), at most `clip`); returns the draws scored (their logp finite) */
uint32_t live_grad(const char *draws, uint32_t stride, const float *behaved, uint32_t n, const uint32_t *row, const float *adv,
                   const float *w, float *g, float clip, float *logp_now) {
  uint32_t used = 0; float score[SPLIT_FS];
  for (uint32_t i = 0; i < n; i++) {
    const struct route_draw *d = (const struct route_draw *)(draws + (size_t)i * stride);
    const float lp = live_score(d, w + (size_t)row[i] * SPLIT_FS, score);
    logp_now[i] = lp;
    if (!isfinite(lp) || !isfinite(adv[i])) continue;
    float ratio = isfinite(behaved[i]) ? expf(lp - behaved[i]) : 1.0f;
    if (ratio > clip) ratio = clip;
    for (int f = 0; f < SPLIT_FS; f++) g[(size_t)row[i] * SPLIT_FS + f] += adv[i] * ratio * score[f];
    used++;
  }
  return used;
}
