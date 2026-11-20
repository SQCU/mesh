/* The native kernels the Python driver (py/route) calls: each a whole array at a call, none with a schedule, a thread
   or a lock of its own. The data flow and its schedule are the driver's (design/dataflow.png).

   sims (sims.c): n simulators and their arena, held from route_sims_new: each slot's refreshes as made (lanes,
   tokens), which a forward reads packed and a learning batch reads by the contexts drawn; its run's costs.
   net (net.swift): the int8 network on the GPU; batches are its input and output buffers in shared memory (the
   driver's numpy views); forward, learn and the optimizer's step are committed, never waited on, except by
   route_net_wait (a batch's results, where the driver reads them). */
#ifndef ROUTE_KERNELS_H
#define ROUTE_KERNELS_H
#include "route.h"

/* a window group: four rollouts of one stream from one state in one window, rollout q drawing with key[q] (key0[q]
   the run before the window); mode 0 the policy, 2 the shortest-path tree, 3 uniform, 4 switch.c's fixed router; the
   collectives' floors filled at begin */
struct route_case {
  struct route_fabric f;
  double background[ROUTE_L];
  int ops, mode;
  struct route_op op[ROUTE_OPS];
  struct route_window w;
  float h[2 * ROUTE_D];
  uint32_t key0[4], key[4];
};
static inline void route_case_stream(const struct route_case *c, struct route_stream *s) { route_stream_build(&c->f, c->background, c->ops, c->op, s); }

/* ---- the simulators ---- */
struct route_sims;
struct route_sims *route_sims_new(int n, int refreshes);   /* refreshes: a slot's kept, at most (the learning window's) */
void route_sims_free(struct route_sims *x);
/* slot[i] begins rollout[i] (0..3) of case c[which[i]] */
void route_sims_begin(struct route_sims *x, const int *slot, int m, const struct route_case *c, const int *which, const int *rollout);
/* each slot answers its last refresh with w rows from wrow[i] (-1 or w NULL: none; copied before the call returns)
   and runs, as a task of its own on the system's thread pool, to its next refresh or its end */
void route_sims_launch(struct route_sims *x, const int *slot, int m, const float *w, const int *wrow);
/* the runs that have since reached a refresh (status 1, nc its contexts) or ended (status 0): their count; wait: first
   sleeps until something is ready (a run, or a forward or step done: route_wake) */
int route_sims_collect(struct route_sims *x, int wait, int *slot, int *status, int *nc);
void route_wake(void);
/* in parallel, the slots' current refreshes as one forward's input: lanes [m][L][IN], mask [m][L], tokens at rows
   offset[i] .., pos[row] the row's state (its run's last lane row: i L + L - 1) */
void route_sims_pack(struct route_sims *x, const int *slot, int m, const int *offset, float *lanes, float *mask, float *tokens, int *pos);
/* ended slots: each refresh's learning rows (the contexts drawn from) and draws, [m][refreshes]; their costs and counts */
void route_sims_sizes(struct route_sims *x, const int *slot, int m, int *rows, int *draws);
void route_sims_costs(struct route_sims *x, const int *slot, int m, struct route_costs *costs, uint64_t *counts);
/* in parallel, ended slots as learning-batch runs b0 + i: their refreshes' lanes [b][K][L][IN] and masks, each refresh's
   drawn contexts' tokens at rows row0[i][k] .., pos (the refresh's lane row (b K + k) L + L - 1), the behaviour w a row,
   the draws at draw0[i][k] .. (their rows the batch's), first[row] a row's first draw (the caller sets first[R]) */
void route_sims_export(struct route_sims *x, const int *slot, int m, int b0, int K, const int *row0, const int *draw0, float *lanes, float *mask,
                       float *tokens, int *pos, float *w, struct route_draw *draws, uint32_t *first);
/* one slot's current refresh, for the checks: its lanes and tokens (as packed), and its stored w at refresh k */
struct route_sim *route_sims_sim(struct route_sims *x, int slot);

/* ---- the network ---- */
/* a batch's buffers (shared memory): a forward's input (lanes, mask, h0, tokens, pos) and output (w, next); a learning
   step's input besides (w: each row's behaviour w; seg: each refresh's rows; draws, first, scale0), its forward's w
   (wcur) and its draws' log importance ratios (ll) */
struct route_batch { float *lanes, *mask, *h0, *next, *tokens, *w, *wcur, *scale0, *ll; int *pos; uint32_t *first, *seg; struct route_draw *draws;
                     int runs, refreshes, rows, draws_room; };
void *route_net_new(const char *metallib, int runs, int refreshes, int rows, uint64_t seed, const char *theta_in);   /* NULL: route_net_failure() */
const char *route_net_failure(void);
struct route_batch *route_net_batch(void *net, int runs, int refreshes, int rows);   /* buffers held from now */
void route_net_reserve(void *net, struct route_batch *b, int draws);                 /* its draws' room, grown */
void route_net_forward(void *net, struct route_batch *b, int B, int R);            /* w [R][FS] and next [B][2D] */
void route_net_wait(void *net, struct route_batch *b);                              /* the batch's last call done */
int route_net_done(void *net, struct route_batch *b);                               /* the same, without waiting */
/* the forward again, the credit (draws' score times importance ratio), the backward: grads; then (world > 1) the
   grads summed across the ranks through the mesh's lease, the Gram matrix, UPGrad, Adam, V pinned, the codes: one
   command buffer. ndraws: the draws; scale: the tasks' scale */
void route_net_step(void *net, struct route_batch *b, int B, int K, int R, int ndraws, float scale, double rate, int world);
/* theta from `from` into `to`, in their queues' order (a shared event between them), the codes rebuilt */
void route_net_publish(void *from, void *to);
int route_net_steps(void *net);
/* this rank's communicator and lease for the step's exchange (exchange.c), declared on the network's queue; 0, or -1
   (route_exchange_failure) */
int route_net_join(void *net, int rank, int world, const char *id);
int route_net_save(void *net, const char *fp32, const char *int8);
float *route_net_theta(void *net, int *count);                                      /* the fp32 master copy (shared) */
float *route_net_grads(void *net);                                                  /* the last step's gradients [3][P] */
float *route_net_dw(void *net);                                                     /* the last step's credit [3][R][FS] */
void route_net_exchange(void *net);                                                 /* the gradients' sum alone (world > 1), committed */
/* the checks: the fp32 reference (learn.swift) on a learned batch: w [R][FS], grads [3][P]; each kernel's GPU time */
void route_net_reference(void *net, struct route_batch *b, int B, int K, int R, float *w, float *grads);
int route_net_profile(void *net, int on, char *out, size_t room);   /* on: from the next call; off: that call's, as JSON */
int route_net_probes(void *net, int on, char *out, size_t room);    /* fused producers' codes against int8mm.h's (JSON) */
/* the checks' view of the network's own buffers (model.swift's names: "gq", "gs", "partial", ...): bytes from offset,
   copied into out once the queue is idle; 0, or -1 (no such buffer, or past its end) */
int route_net_read(void *net, const char *buffer, size_t offset, size_t bytes, void *out);
void route_net_free(void *net);

/* ---- the exchange (exchange.c): world > 1 ---- */
void *route_exchange_metal(size_t *offset);   /* the lease's Metal buffer (id<MTLBuffer>) and its offset */
#endif
