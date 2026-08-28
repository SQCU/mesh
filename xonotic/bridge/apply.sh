#!/bin/sh
set -e
DP="$1"
HERE="$(cd "$(dirname "$0")" && pwd)"
cp "$HERE/engine/mesh_shm.h" "$HERE/engine/mesh_ipc.c" "$DP/"

python3 - "$DP" <<'PY'
import sys, io
dp = sys.argv[1]

names = ["open","close","set","get","gather","scatter","publish","poll","wait","stat"]

p = dp + "/prvm_cmds.c"
s = open(p).read()
if '#include "mesh_ipc.c"' not in s:
    s = s.rstrip() + '\n\n#include "mesh_ipc.c"\n'
    open(p, "w").write(s)

p = dp + "/prvm_cmds.h"
s = open(p).read()
anchor = "void VM_uri_get (prvm_prog_t *prog);\n"
decls = "".join("void VM_mesh_%s(prvm_prog_t *prog);\n" % n for n in names)
if "VM_mesh_open" not in s:
    assert anchor in s, "prvm_cmds.h anchor missing"
    s = s.replace(anchor, anchor + decls, 1)
    open(p, "w").write(s)

entries = "".join("VM_mesh_%s,\t\t\t\t\t// #%d\n" % (n, 644 + i) for i, n in enumerate(names))

p = dp + "/svvm_cmds.c"
s = open(p).read()
anchor = "NULL,\t\t\t\t\t\t\t// #643\n"
if "VM_mesh_open" not in s:
    assert anchor in s, "svvm_cmds.c anchor missing"
    s = s.replace(anchor, anchor + entries, 1)
    open(p, "w").write(s)

p = dp + "/clvm_cmds.c"
s = open(p).read()
anchor = "VM_coverage,\t\t\t\t\t\t// #642\n"
if "VM_mesh_open" not in s:
    assert anchor in s, "clvm_cmds.c anchor missing"
    s = s.replace(anchor, anchor + "NULL,\t\t\t\t\t\t\t// #643\n" + entries, 1)
    open(p, "w").write(s)

print("patched", dp)
PY
