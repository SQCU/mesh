#!/bin/sh
set -e
DP="$1"
HERE="$(cd "$(dirname "$0")" && pwd)"
RDMA="$(cd "$HERE/../../rdma" && pwd)"
cp "$RDMA/mesh.h" "$RDMA/mesh-client.c" "$HERE/engine/mesh_ipc.c" "$DP/"
rm -f "$DP/mesh_shm.h"

python3 - "$DP" <<'PY'
import sys
dp = sys.argv[1]

live = {644: "open", 648: "gather", 649: "scatter", 650: "publish", 651: "poll", 653: "stat"}

p = dp + "/prvm_cmds.c"
s = open(p).read()
if '#include "mesh_ipc.c"' not in s:
    open(p, "w").write(s.rstrip() + '\n\n#include "mesh_ipc.c"\n')

p = dp + "/prvm_cmds.h"
s = open(p).read()
anchor = "void VM_uri_get (prvm_prog_t *prog);\n"
decls = "".join("void VM_mesh_%s(prvm_prog_t *prog);\n" % live[n] for n in sorted(live))
if "VM_mesh_open" not in s:
    assert anchor in s, "prvm_cmds.h anchor missing"
    open(p, "w").write(s.replace(anchor, anchor + decls, 1))

entries = "".join(
    ("VM_mesh_%s,\t\t\t\t\t// #%d\n" % (live[n], n)) if n in live else ("NULL,\t\t\t\t\t\t\t// #%d\n" % n)
    for n in range(644, 654))

p = dp + "/svvm_cmds.c"
s = open(p).read()
anchor = "NULL,\t\t\t\t\t\t\t// #643\n"
if "VM_mesh_open" not in s:
    assert anchor in s, "svvm_cmds.c anchor missing"
    open(p, "w").write(s.replace(anchor, anchor + entries, 1))

p = dp + "/clvm_cmds.c"
s = open(p).read()
anchor = "VM_coverage,\t\t\t\t\t\t// #642\n"
if "VM_mesh_open" not in s:
    assert anchor in s, "clvm_cmds.c anchor missing"
    open(p, "w").write(s.replace(anchor, anchor + "NULL,\t\t\t\t\t\t\t// #643\n" + entries, 1))

print("patched", dp)
PY
