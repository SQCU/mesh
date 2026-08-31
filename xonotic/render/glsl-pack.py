#!/usr/bin/env python3
"""glsl/default.glsl  ->  darkplaces-work/shader_glsl.h

DarkPlaces embeds the surface shader as a C array of one-line string literals.
r_glsl_dumpshader emits the reverse direction, so together these two give a real
GLSL file as the editable source of truth.
"""
import sys, pathlib
src = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else
                   pathlib.Path(__file__).parent / "glsl" / "default.glsl")
dst = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else
                   pathlib.Path(__file__).parent.parent / "darkplaces-work" / "shader_glsl.h")
out = []
for line in src.read_text().split("\n"):
    if line == "" and len(out) and src.read_text().endswith("\n") and line is src:
        pass
    out.append('"%s\\n",\n' % line.replace("\\", "\\\\").replace('"', '\\"'))
# the dumper writes a trailing newline that the header does not carry
if out and out[-1] == '"\\n",\n':
    out.pop()
dst.write_text("".join(out))
print("%s -> %s (%d lines)" % (src, dst, len(out)))
