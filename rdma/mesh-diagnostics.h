#ifndef MESH_DIAGNOSTICS_H
#define MESH_DIAGNOSTICS_H
/* Operator diagnostics implemented by the same versioned libmesh as normal
   transport. They are not an application route-selection interface.
   quiet: argv = {"quiet", "list", peer} or
                 {"quiet", "run", role, physical_wire, rounds, gaps_us_csv}.
   The explicit physical wire measures one selected cable's idle/wake latency.
   Arguments are mutable strings; results are JSON lines on stdout. */
int mesh_diagnostic_quiet(int argc, char **argv);
/* Measure local bridge reclamation after n fresh diagnostic clients exit.
   This read-only observer changes no transport/recovery policy. */
int mesh_diagnostic_retire(int n);
#endif
