"""Actual GPU masked-frame checks, with null transport bindings (no NIC claim)."""
import ast
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from migration import RowWrite

root = Path(__file__).resolve().parents[1]
package = root / 'torch-mesh/torch_mesh'
tree = ast.parse((package / 'partition.py').read_text())
source = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == '_MIGRATION_SOURCE' for t in n.targets))
word_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_migration_words')
namespace = {}
exec(compile(ast.Module(body=[word_node], type_ignores=[]), str(package / 'partition.py'), 'exec'), namespace)
with tempfile.TemporaryDirectory(prefix='mesh-masked-prelude-') as temporary:
    main, executable = Path(temporary) / 'prelude.m', Path(temporary) / 'prelude'
    main.write_text('#include <stdio.h>\n#include "nccl-mesh-kernels.h"\n'
                    'int main(void) { fputs(STEPS.UTF8String, stdout); return 0; }\n')
    subprocess.run(['clang', '-fobjc-arc', '-framework', 'Foundation', '-I', str(root), str(main), '-o', str(executable)], check=True)
    steps = subprocess.check_output([str(executable)], text=True)
prelude = (root / 'mesh.metal').read_text() + steps + (package / 'argmax.metal').read_text()


def instance(name, *args):
    values = ', '.join(map(str, args))
    symbol = '_'.join([name, *map(str, args)])
    return symbol, f'template [[host_name("{symbol}")]] [[kernel]] decltype({name}<{values}>) {name}<{values}>;'


dummy = torch.zeros(1, dtype=torch.int32, device='mps')
checks = []
for channels, vector, inline in itertools.product((1, 3, 5, 6), (1, 2, 4, 8, 16), (0, 1)):
    declarations = [instance('migration_publish', vector, channels, 256),
                    instance('migration_land', vector, channels, inline, 256),
                    instance('migration_ready', channels, 256), instance('migration_local', vector, channels, 256)]
    shader = torch.mps.compile_shader(prelude + source + '\n'.join(line for _, line in declarations))
    publish, land, ready, retain = [getattr(shader, name) for name, _ in declarations]
    for row, rows in ((1, 513), (257, 3), (65, 17)):
        run = RowWrite(0, 1, 3 * vector, 5 * vector, row * vector, rows,
                       (row + 7) * vector, (row + 11) * vector)
        frame_bytes = 256 * vector
        count = run.frames(frame_bytes)
        source_bytes = run.source_byte(run.bytes - 1) + 1 + 32
        output_bytes = run.destination_byte(run.bytes - 1) + 1 + 32
        original = torch.arange(source_bytes, dtype=torch.int32).remainder(251).to(torch.uint8)
        original[run.source_offset:run.source_offset + vector] = 255  # includes NaN payloads when viewed as float
        expected = torch.full((output_bytes,), 173, dtype=torch.uint8)
        expected_frames = torch.zeros(count * frame_bytes + 32, dtype=torch.uint8)
        expected_frames[-32:] = 239
        for i in range(run.bytes):
            expected[run.destination_byte(i)] = original[run.source_byte(i)]
            expected_frames[i] = original[run.source_byte(i)]
        x = original.to('mps')
        frames = torch.full((count * frame_bytes + 32,), 239, dtype=torch.uint8, device='mps')
        out = torch.full((output_bytes,), 173, dtype=torch.uint8, device='mps')
        local = out.clone()
        arrivals = torch.zeros(1, dtype=torch.int32, device='mps')
        for j in range(count):
            constants = torch.tensor(namespace['_migration_words'](run, frame_bytes, vector, j) + [0] * (24 * channels),
                                     dtype=torch.int32, device='mps')
            publish(x, frames, arrivals, *([dummy] * 12), constants, threads=2048, group_size=256)
            if not inline:
                ready(*([dummy] * 15), constants, threads=256, group_size=256)
            land(frames, out, *([dummy] * 13), constants, threads=256 if inline else 2048, group_size=256)
        retain(x, local, *([dummy] * 13), constants, threads=2048, group_size=256)
        assert torch.equal(frames.cpu(), expected_frames), ('publication bits/padding/guard', channels, vector, row, rows)
        assert torch.equal(out.cpu(), expected), ('masked output/neighbor poison', channels, vector, row, rows)
        assert torch.equal(local.cpu(), expected), ('local retention bits/neighbor poison', channels, vector, row, rows)
        assert torch.equal(x.cpu(), original), 'original input overwritten'
        checks.append(dict(channels=channels, vector_bytes=vector, inline_wait=inline, row_bytes=run.row_bytes,
                           rows=rows, frames=count, final_valid_bytes=run.bytes % frame_bytes or frame_bytes))

report = dict(cases=len(checks), checks=checks, torch=torch.__version__,
              scope='Actual GPU bit preservation, masked padding, neighboring-byte guards, affine row/frame boundaries; null transport bindings')
if len(sys.argv) > 1:
    Path(sys.argv[1]).write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: v for k, v in report.items() if k != 'checks'}))
