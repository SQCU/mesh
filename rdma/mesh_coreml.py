import json
from collections import OrderedDict
from pathlib import Path
import shutil
import sys

import coremltools as ct
import numpy as np
from coremltools.converters.mil import Builder as mb, Function, Program
from coremltools.converters.mil.mil import types


# design/algorithm-sources.md#coreml-partial-execution
def compile_part(request, destination):
    specification = json.loads(Path(request).read_text())
    arguments = OrderedDict()
    for i, (rows, columns, depth, half) in enumerate(specification['rectangles']):
        dtype = types.fp16 if half else types.fp32
        arguments[f'x{i}'] = mb.placeholder(shape=(rows, depth), dtype=dtype)
        arguments[f'w{i}'] = mb.placeholder(shape=(depth, columns), dtype=dtype)
    program = Program()
    with Function(arguments, opset_version=ct.target.macOS15) as function:
        parts = []
        for i in range(len(specification['rectangles'])):
            product = mb.matmul(x=function.inputs[f'x{i}'], y=function.inputs[f'w{i}'])
            scalar = np.float16 if specification['rectangles'][i][3] else np.float32
            scaled = mb.mul(x=product, y=scalar(specification['alpha']))
            parts.append(mb.reshape(x=scaled, shape=(-1,)))
        joined = parts[0] if len(parts) == 1 else mb.concat(values=parts, axis=0)
        output = mb.cast(x=joined, dtype='fp16' if specification['output_half'] else 'fp32', name='z')
        function.set_outputs([output])
    program.add_function('main', function)
    model = ct.convert(program, minimum_deployment_target=ct.target.macOS15,
                       compute_precision=ct.precision.FLOAT16)
    destination = Path(destination).with_suffix('.mlmodelc')
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(model.get_compiled_model_path(), destination)


if __name__ == '__main__':
    compile_part(*sys.argv[1:])
