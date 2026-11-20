/* The trainer (trainer.swift) as a library: the Python driver (py/route) configures it, supplies every window it runs
   and reads its telemetry table. What crosses: the configuration, window groups (a case: a stream's fabric and
   collectives, the window, the state the policy starts from, the draws' keys), and back the states runs ended with and
   the evaluation's completions. The trainer draws nothing: every choice of data is the driver's.

   The producers (runs.swift's workers on every hardware thread) run the pushed groups, four rollouts each; a thread
   of the trainer's serves their forwards; the learner thread learns from every finished group, exchanges gradients
   through the mesh (exchange.c) when world > 1, and publishes theta to the producers. */
#ifndef ROUTE_TRAINER_H
#define ROUTE_TRAINER_H
#include "route.h"

struct route_config {
  int rank, world, steps;       /* steps: learning steps (0: none, the checks and measurements alone) */
  double rate;                  /* Adam's step */
  uint64_t seed;                /* theta's initialization, every rank's the same */
  int forward_rows, learn_runs; /* a forward's context rows; a learning batch's runs (both buffers held from open) */
  int refreshes;                /* a learning window's refreshes, at most (a case's w.points; ROUTE_POINTS at most) */
  const char *mesh_id;          /* the bootstrap root's host:port (world > 1) */
  const char *metallib, *theta_in, *telemetry;   /* theta_in NULL: from the seed */
};
/* a window group: four rollouts of one stream from one state in one window, rollout q drawing with key[q] (key0[q]
   the run before the window: a training group's four the same); mode 0 the policy, 2 the shortest-path tree, 3 uniform, 4 switch.c's fixed router. The
   collectives' floors are the trainer's to fill. */
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

void *route_trainer_open(const struct route_config *c);   /* NULL: route_trainer_failure() says why */
const char *route_trainer_failure(void);
int route_trainer_groups(void *t);                          /* window groups in flight at most */
void route_trainer_start(void *t);                          /* producers, the learner, the exchange, the forwards */
/* the cases that can be pushed now (the queue holds one for every group), waiting up to `timeout` seconds for room;
   -1 once the learning is over */
int route_trainer_room(void *t, double timeout);
void route_trainer_push(void *t, const struct route_case *c, int n);
int route_trainer_ended(void *t, float *states, int room);  /* the states runs ended with since the last call, [n][2 D] */
int route_trainer_learned(void *t);                         /* learning steps taken */
void route_trainer_finish(void *t);                         /* after the last step: the exchange's means in flight applied */
/* each case's four rollouts on a snapshot of theta (a runner and network of its own, so the training goes on): their
   completions over the floor, [n][4] */
int route_trainer_evaluate(void *t, const struct route_case *c, int n, double *throughput);
int route_trainer_save(void *t, const char *fp32, const char *int8);   /* NULL: that one not written */
/* one refresh on the GPU (the producers' network): lanes [L][IN] (nl used), tokens [nc][IN], state [2 D] advanced, w [nc][FS] */
void route_trainer_forward(void *t, const float *lanes, int nl, const float *tokens, int nc, float *state, float *w);
/* the exchange alone: `rounds` means of known values; each round's seconds, and the means wrong */
int route_trainer_xbench(void *t, int rounds, double *seconds, long *wrong);
/* reports as JSON text into `out` (its length; 0, or -1 if it did not fit): check, the credit against central
   differences on the first case whose draws had a choice (graph: also the int8 network against the fp32 reference);
   bench, each call's time and the environment's on the cases */
int route_trainer_check(void *t, const struct route_case *c, int n, double eps, int graph, char *out, size_t room);
int route_trainer_bench(void *t, const struct route_case *c, int n, char *out, size_t room);
void route_trainer_close(void *t);
#endif
