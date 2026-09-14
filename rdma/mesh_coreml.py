import json
import errno
from collections import OrderedDict
from pathlib import Path
import shutil
import sys
import tempfile

import coremltools as ct
import numpy as np
from coremltools.converters.mil import Builder as mb, Function, Program
from coremltools.converters.mil.mil import types


# design/algorithm-sources.md#coreml-partial-execution
def compile_part(request, destination):
    specification = json.loads(Path(request).read_text())
    arguments = OrderedDict()
    for i, (rows, columns, depth, left_half, right_half) in enumerate(specification['rectangles']):
        arguments[f'x{i}'] = mb.placeholder(shape=(rows, depth), dtype=types.fp16 if left_half else types.fp32)
        arguments[f'w{i}'] = mb.placeholder(shape=(depth, columns), dtype=types.fp16 if right_half else types.fp32)
    program = Program()
    with Function(arguments, opset_version=ct.target.macOS15) as function:
        parts = []
        for i in range(len(specification['rectangles'])):
            x, w = function.inputs[f'x{i}'], function.inputs[f'w{i}']
            product = mb.matmul(x=mb.cast(x=x, dtype='fp32'), y=mb.cast(x=w, dtype='fp32'))
            scaled = mb.mul(x=product, y=np.float32(specification['alpha']))
            parts.append(mb.reshape(x=scaled, shape=(-1,)))
        joined = parts[0] if len(parts) == 1 else mb.concat(values=parts, axis=0)
        output = mb.cast(x=joined, dtype='fp16' if specification['output_half'] else 'fp32', name='z')
        function.set_outputs([output])
    program.add_function('main', function)
    model = ct.convert(program, minimum_deployment_target=ct.target.macOS15,
                       compute_precision=ct.precision.FLOAT32)
    destination = Path(destination).with_suffix('.mlmodelc')
    with tempfile.TemporaryDirectory(prefix=destination.name + '.', dir=destination.parent) as directory:
        staged = Path(directory) / destination.name
        shutil.copytree(model.get_compiled_model_path(), staged)
        try:
            staged.rename(destination)
        except OSError as error:
            if error.errno not in (errno.EEXIST, errno.ENOTEMPTY) or not destination.is_dir():
                raise


if __name__ == '__main__':
    compile_part(*sys.argv[1:])
