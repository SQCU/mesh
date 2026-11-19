#ifndef MESH_DIAGNOSTICS_H
#define MESH_DIAGNOSTICS_H
/* Operator diagnostics implemented by the same versioned libmesh as normal
   transport. They are not an application route-selection interface.
   quiet: argv = {"quiet", "list", any} (the nodes reached) or
                 {"quiet", "run", role, node, rounds, gaps_us_csv}.
   It measures the idle/wake latency of the order toward a node; the bridge
   alone chooses the cables.
   Arguments are mutable strings; results are JSON lines on stdout. */
int mesh_diagnostic_quiet(int argc, char **argv);
/* Measure local bridge reclamation after n fresh diagnostic clients exit.
   This read-only observer changes no transport/recovery policy. */
int mesh_diagnostic_retire(int n);
#endif
