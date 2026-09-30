#ifndef MESH_DISK_H
#define MESH_DISK_H
/* The one capacity guard of every writer of the mesh stack, in both repositories: the bridges' logs,
   torch-mesh's trace, the drivers' records and dumps, prepared programs and shards, the services'
   runtimes and sources.  A write of `bytes` is made only while the filesystem it lands in keeps
   MESH_DISK_FLOOR_GIB free after it; below the floor it is skipped with one line on stderr naming what
   was skipped and why, and the computation it serves goes on: nothing fails because a record of it
   could not be written.  The two numbers are stated here only: rdma/mesh_disk.py reads them from this
   file, the services' shell scripts with awk. */
#define MESH_DISK_FLOOR_GIB 20
/* A log a long-lived process writes on its stderr (a bridge's): past MESH_LOG_CAP_MIB it moves to
   <path>.1, the one before it dropped, and starts again, so it holds at most twice this. */
#define MESH_LOG_CAP_MIB 16
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <string.h>
#include <sys/mount.h>
#include <sys/stat.h>
#include <unistd.h>

#define MESH_DISK_FLOOR ((unsigned long long)MESH_DISK_FLOOR_GIB << 30)
#define MESH_LOG_CAP ((unsigned long long)MESH_LOG_CAP_MIB << 20)

/* The free bytes of the filesystem holding `path` (or its nearest existing ancestor), its mount point
   in `mount`; ULLONG_MAX where none is found (the write then goes ahead and reports its own error). */
static inline unsigned long long mesh_disk_free(const char *path, char *mount, size_t size) {
  char at[PATH_MAX];
  snprintf(at, sizeof at, "%s", path && *path ? path : ".");
  for (;;) {
    struct statfs s;
    if (!statfs(at, &s)) {
      snprintf(mount, size, "%s", s.f_mntonname);
      return (unsigned long long)s.f_bavail * s.f_bsize;
    }
    char *slash = strrchr(at, '/');
    if (errno != ENOENT || !strcmp(at, ".")) break;
    if (!slash) snprintf(at, sizeof at, ".");
    else if (slash == at) slash[1] = 0;
    else *slash = 0;
  }
  snprintf(mount, size, "%s", path ? path : "");
  return ULLONG_MAX;
}

/* 1 when `bytes` more may be written under `path`; else 0, after one line on stderr naming `what`. */
static inline int mesh_disk_room(const char *path, unsigned long long bytes, const char *what) {
  char mount[MNAMELEN];
  const unsigned long long free = mesh_disk_free(path, mount, sizeof mount);
  if (free == ULLONG_MAX || free >= MESH_DISK_FLOOR + bytes) return 1;
  fprintf(stderr, "mesh-disk: %s not written (%llu bytes): %s has %llu bytes free, under the floor of %d GiB (mesh-disk.h)\n",
          what, bytes, mount, free, MESH_DISK_FLOOR_GIB);
  return 0;
}

/* Before a line of `bytes` on this process's stderr: where stderr is a file past MESH_LOG_CAP, it moves
   to <path>.1 and stderr (stdout too, where it is the same file) starts a new <path>; 1 when the line
   may be written, 0 under the floor, the first line of a run of dropped ones saying so. */
static inline int mesh_log_room(size_t bytes) {
  static int rotating, dropping;
  struct stat was;
  if (fstat(2, &was) || !S_ISREG(was.st_mode)) return 1;
  if ((unsigned long long)was.st_size + bytes > MESH_LOG_CAP && !__atomic_exchange_n(&rotating, 1, __ATOMIC_ACQ_REL)) {
    char path[PATH_MAX], old[PATH_MAX + 2];
    struct stat out;
    if (!fcntl(2, F_GETPATH, path) && snprintf(old, sizeof old, "%s.1", path) < (int)sizeof old && !rename(path, old)) {
      const int file = open(path, O_WRONLY | O_CREAT | O_APPEND, 0644);
      if (file >= 0) {
        if (!fstat(1, &out) && out.st_dev == was.st_dev && out.st_ino == was.st_ino) dup2(file, 1);
        dup2(file, 2);
        close(file);
      }
    }
    __atomic_store_n(&rotating, 0, __ATOMIC_RELEASE);
  }
  struct statfs s;
  if (fstatfs(2, &s)) return 1;
  const unsigned long long free = (unsigned long long)s.f_bavail * s.f_bsize;
  if (free >= MESH_DISK_FLOOR + bytes) {
    __atomic_store_n(&dropping, 0, __ATOMIC_RELAXED);
    return 1;
  }
  if (!__atomic_exchange_n(&dropping, 1, __ATOMIC_RELAXED))
    dprintf(2, "mesh-disk: this log's lines dropped from here: %s has %llu bytes free, under the floor of %d GiB (mesh-disk.h)\n",
            s.f_mntonname, free, MESH_DISK_FLOOR_GIB);
  return 0;
}
#endif
