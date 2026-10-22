#ifndef MESH_DISK_H
#define MESH_DISK_H
/* The one capacity guard of every writer of the mesh stack, in both repositories: the bridges' logs,
   torch-mesh's trace, the drivers' records and dumps, prepared programs and shards, the services'
   runtimes.  A node's source generation is not under it: a node fetches its intent whenever the
   generation fits, since below the floor the source is what carries the repair (bootstrap.sh).  A write of `bytes` is made only while the filesystem it lands in keeps
   MESH_DISK_FLOOR_GIB free after it; below the floor it is skipped with one line on stderr naming what
   was skipped and why, and the computation it serves goes on: nothing fails because a record of it
   could not be written.  The two numbers are stated here only: rdma/mesh_disk.py reads them from this
   file, the services' shell scripts with awk. */
#define MESH_DISK_FLOOR_GIB 20
/* A log a long-lived process writes on its stderr (a bridge's): past MESH_LOG_CAP_MIB it moves to
   <path>.1, the one before it dropped, and starts again, so it holds at most twice this. */
#define MESH_LOG_CAP_MIB 16
#endif
