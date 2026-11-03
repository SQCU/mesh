#ifndef MESH_STATISTICS_H
#define MESH_STATISTICS_H

#ifdef __cplusplus
extern "C" {
#endif

/* Ordinary host observation of a whole function. No GPU counter, submission,
   synchronization or peer rendezvous is performed by this interface. */
struct mesh_statistics {
  double seconds;
  double useful_gigabytes_per_second;
  double units_per_second;
};
double mesh_statistics_now(void);
struct mesh_statistics mesh_statistics_measure(double started, double observed,
                                                double useful_bytes, double units);
/* A run's own stream words beside it, read from a mapping of the bridge's region (the words its bridge and its GPU
   store; the run is not touched): a spin over every stream, each move of a word kept as four values, moves[4 k ..]:
   the pass's clock (CLOCK_UPTIME_RAW nanoseconds), the stream, the word (0 owner, 1 ready, 2 sent, 3 landed) and its
   value, until *stop is nonzero or `capacity` moves are kept. Their count, or -errno (no region: no bridge runs).
   The balancer's observation of each rank's compute in the run it balances (examples/balance-service.py): a light
   GPU trace's stamps doubled the LSTM's period (metal-microbench docs/facts.md, light-1), these words cost it none
   (tools/mesh/tick_words.py reads the same words from mesh-stat). */
long mesh_statistics_streams(unsigned long long *moves, unsigned long capacity, const volatile int *stop);

#ifdef __cplusplus
}
#endif
#endif
