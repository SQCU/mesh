#!/usr/bin/env mesh-python
import sys, pathlib
src = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else
                   pathlib.Path(__file__).parent / "glsl" / "default.glsl")
dst = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else
                   pathlib.Path(__file__).parent.parent / "darkplaces-work" / "shader_glsl.h")
out = []
for line in src.read_text().split("\n"):
    out.append('"%s\\n",\n' % line.replace("\\", "\\\\").replace('"', '\\"'))

if out and out[-1] == '"\\n",\n':
    out.pop()
dst.write_text("".join(out))
print("%s -> %s (%d lines)" % (src, dst, len(out)))
