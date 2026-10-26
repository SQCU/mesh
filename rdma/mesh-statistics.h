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

#ifdef __cplusplus
}
#endif
#endif
