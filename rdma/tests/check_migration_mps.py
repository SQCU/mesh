"""Actual GPU exact-byte migration kernels; null transport bindings, no NIC claim."""
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
for channels, vector in itertools.product((1, 3, 5, 6), (1, 2, 4, 8, 16)):
    rows_cases = ((1, 513), (257, 3), (65, 17), (3, 5003))
    declarations = [instance('migration_ready', channels, 256),
                    instance('migration_publish_existing', channels, 256)]
    declarations += [instance('migration_local', vector, channels, row, 256) for row, _ in rows_cases]
    shader = torch.mps.compile_shader(prelude + source + '\n'.join(line for _, line in declarations))
    ready, publish = [getattr(shader, name) for name, _ in declarations[:2]]
    for case, (row, rows) in enumerate(rows_cases):
        retain = getattr(shader, declarations[case + 2][0])
        run = RowWrite(0, 1, 3 * vector, 5 * vector, row * vector, rows,
                       (row + 7) * vector, (row + 11) * vector)
        source_bytes = run.source_byte(run.bytes - 1) + 1 + 32
        output_bytes = run.destination_byte(run.bytes - 1) + 1 + 32
        original = torch.arange(source_bytes, dtype=torch.int32).remainder(251).to(torch.uint8)
        original[run.source_offset:run.source_offset + vector] = 255
        expected = torch.full((output_bytes,), 173, dtype=torch.uint8)
        for r in range(rows):
            expected.narrow(0, run.destination_offset + r * run.destination_stride, run.row_bytes).copy_(
                original.narrow(0, run.source_offset + r * run.source_stride, run.row_bytes))
        x, local = original.to('mps'), torch.full((output_bytes,), 173, dtype=torch.uint8, device='mps')
        constants = torch.tensor(namespace['_migration_words'](run, run.bytes, vector) + [0] * (24 * channels),
                                 dtype=torch.int32, device='mps')
        publish(*([dummy] * 15), constants, threads=256, group_size=256)
        ready(*([dummy] * 15), constants, threads=256, group_size=256)
        retain(x, local, *([dummy] * 13), constants, threads=2048, group_size=256)
        assert torch.equal(local.cpu(), expected), ('local retention bits/neighbor poison', channels, vector, row, rows)
        assert torch.equal(x.cpu(), original), 'original input overwritten'
        checks.append(dict(channels=channels, vector_bytes=vector, row_bytes=run.row_bytes, rows=rows))

report = dict(cases=len(checks), checks=checks, torch=torch.__version__,
              scope='Actual GPU local-retention bit/guard checks and release-only/ready kernel compilation; null transport bindings')
if len(sys.argv) > 1:
    Path(sys.argv[1]).write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: v for k, v in report.items() if k != 'checks'}))
