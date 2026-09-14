import argparse
import signal
import sys
from contextlib import ExitStack

import numpy as np

from mesh import BlockSpec, Program, ShapeDtypeStruct, kernels
from mesh.nn import linear
from mesh.collective import reduce_scatter, all_gather


# design/algorithm-sources.md#nnffn
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('up_weight')
    parser.add_argument('down_weight')
    parser.add_argument('consumer_weight')
    parser.add_argument('--root', type=int, required=True)
    parser.add_argument('--peer', type=int, required=True)
    parser.add_argument('--split', type=int, required=True)
    parser.add_argument('--output-split', type=int, required=True)
    parser.add_argument('--instances', type=int, default=2)
    parser.add_argument('--region')
    parser.add_argument('--numerics')
    parser.add_argument('--model')
    parser.add_argument('--gate-weight')
    parser.add_argument('--normalize', nargs='?', const='', metavar='MODEL_TENSOR')
    parser.add_argument('--backend', choices=('cpu', 'metal'), default='cpu')
    parser.add_argument('--tile-rows', type=int, default=128)
    parser.add_argument('--tile-k', type=int, default=128)
    parser.add_argument('--tile-columns', type=int, default=128)
    args = parser.parse_args()
    if args.instances < 1:
        parser.error('--instances must be positive')
    if (args.normalize is not None or args.model or args.gate_weight) and not args.numerics:
        parser.error('--normalize and --model require the engine numerical library')
    if args.normalize and not args.model:
        parser.error('A normalization tensor name requires --model')
    values = np.load(args.input)
    weight_names = (args.up_weight, args.down_weight) + ((args.gate_weight,) if args.gate_weight else ())
    weights = None if args.model else tuple(np.load(name, mmap_mode='r') for name in weight_names)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    functions = {}
    if args.numerics:
        import ctypes as C
        from mesh import check
        from mesh._native import View
        library = C.CDLL(args.numerics)
        for kernel, symbol in ((kernels.add, 'gemma_mesh_add'), (kernels.dot, 'gemma_mesh_mps_dot')):
            native = getattr(library, symbol)
            native.argtypes = [C.c_void_p, C.POINTER(View), C.c_int32]
            native.restype = C.c_int32

            # design/algorithm-sources.md#programkernel_call
            def bind(program, inputs, outputs, native=native):
                refs = (*inputs, *outputs)
                if len(inputs) != 2 or len(outputs) != 1 or any(ref.dtype != inputs[0].dtype for ref in refs):
                    raise TypeError('The supplied numerical function requires two inputs and one output of one dtype')
                scalar = (np.dtype('float16'), np.dtype('float32')).index(inputs[0].dtype)
                check(native(program.handle, (View * 3)(*(ref.view for ref in refs)), scalar))

            functions[kernel] = bind
        if args.gate_weight:
            gelu_native = library.gemma_mesh_gelu_mul
            gelu_native.argtypes = [C.c_void_p, C.POINTER(View)]
            gelu_native.restype = C.c_int32

            # design/algorithm-sources.md#nnffn
            def gated_activation(program, inputs, outputs):
                refs = (*inputs, *outputs)
                if any(ref.dtype != np.dtype('float16') for ref in refs):
                    raise TypeError('The engine gated activation requires FP16 operands')
                check(gelu_native(program.handle, (View * 3)(*(ref.view for ref in refs))))

        if args.normalize is not None:
            normalize_native = library.gemma_mesh_rmsnorm
            normalize_native.argtypes = [C.c_void_p, C.POINTER(View), C.c_int32]
            normalize_native.restype = C.c_int32

            # design/algorithm-sources.md#programkernel_call
            def normalize(program, inputs, outputs):
                scalar = (np.dtype('float16'), np.dtype('float32')).index(inputs[0].dtype)
                check(normalize_native(program.handle,
                    (View * 3)(*(ref.view for ref in (*inputs, *outputs))), scalar))

    with ExitStack() as resources, Program(backend=args.backend, region=args.region, functions=functions) as program:
        if args.model:
            library.gemma_mesh_model_open.argtypes = [C.c_char_p]
            library.gemma_mesh_model_open.restype = C.c_void_p
            library.gemma_mesh_model_close.argtypes = [C.c_void_p]
            library.gemma_mesh_model_close.restype = None
            library.gemma_mesh_model_shape.argtypes = [C.c_void_p, C.c_char_p, C.POINTER(C.c_size_t)]
            library.gemma_mesh_model_shape.restype = C.c_int32
            library.gemma_mesh_model_load.argtypes = [C.c_void_p, C.c_char_p, C.c_void_p, C.POINTER(View), C.c_int32, C.c_size_t, C.c_size_t]
            library.gemma_mesh_model_load.restype = C.c_int32
            model = library.gemma_mesh_model_open(args.model.encode())
            if not model:
                raise OSError(f'Could not open model file {args.model}')
            resources.callback(library.gemma_mesh_model_close, model)
            shapes = []
            for name in weight_names:
                shape = (C.c_size_t * 2)()
                check(library.gemma_mesh_model_shape(model, name.encode(), shape))
                shapes.append(tuple(reversed(shape)))
            dtypes = (values.dtype,) * len(weight_names)
        else:
            shapes, dtypes = tuple(w.shape for w in weights), tuple(w.dtype for w in weights)
        start, end = (0, args.split) if program.node == args.root else (args.split, shapes[0][1])
        up_weight = program.tensor((shapes[0][0], end - start), dtype=dtypes[0])
        down_weight = program.tensor((end - start, shapes[1][1]), dtype=dtypes[1])
        tensors = (up_weight, down_weight)
        if args.gate_weight:
            if shapes[2] != shapes[0]:
                raise ValueError('Gate and up projection weights must have the same shape')
            gate_weight = program.tensor(up_weight.shape, dtype=dtypes[2])
            tensors += (gate_weight,)
        for index, tensor in enumerate(tensors):
            if args.model:
                ref = tensor[0, 0].T
                scalar = (np.dtype('float16'), np.dtype('float32')).index(ref.dtype)
                check(library.gemma_mesh_model_load(model, weight_names[index].encode(),
                    program.handle, C.byref(ref.view), scalar, start if index == 1 else 0, 0 if index == 1 else start))
                program.constant(tensor[0, 0])
            else:
                program.constant(tensor[0, 0], weights[index][:, start:end] if index != 1 else weights[index][start:end, :])
        if program.node == args.root:
            weight = np.load(args.consumer_weight, mmap_mode='r')
            consumer_weight = program.tensor(weight.shape, dtype=weight.dtype)
            program.constant(consumer_weight[0, 0], weight)
            if args.normalize is not None:
                gamma = program.tensor((1, down_weight.shape[1]), dtype=np.float16)
                if args.normalize:
                    ref = gamma[0, 0]
                    check(library.gemma_mesh_model_load(model, args.normalize.encode(),
                        program.handle, C.byref(ref.view), 0, 0, 0))
                    program.constant(ref)
                else:
                    program.constant(gamma[0, 0], np.ones(gamma.shape, dtype=np.float16))
        inputs, outputs = [], {}
        for instance in range(args.instances):
            x = program.tensor(values.shape, (args.tile_rows, args.tile_k), dtype=values.dtype)
            inputs.append(x)
            up = linear(program, x, up_weight, tile_rows=args.tile_rows,
                        tile_k=args.tile_k, tile_columns=args.tile_columns)
            spec = BlockSpec(up.block_shape, lambda i, j: (i, j))
            if args.gate_weight:
                gate = linear(program, x, gate_weight, tile_rows=args.tile_rows,
                              tile_k=args.tile_k, tile_columns=args.tile_columns)
                hidden = program.kernel_call(gated_activation,
                    grid=up.grid, in_specs=(spec, spec), out_specs=spec,
                    out_shape=ShapeDtypeStruct(up.shape, up.dtype))(gate, up)
            else:
                hidden = program.kernel_call(kernels.swish,
                    grid=up.grid, in_specs=(spec,), out_specs=spec,
                    out_shape=ShapeDtypeStruct(up.shape, up.dtype))(up)
            down = linear(program, hidden, down_weight, tile_rows=args.tile_rows,
                          tile_k=args.tile_k, tile_columns=args.tile_columns)
            peers = (args.root, args.peer)
            owners = {index: args.root if index[0] * down.block_shape[0] < args.output_split else args.peer
                      for index in down.blocks}
            scattered = reduce_scatter(program, down, peers=peers, owners=owners)
            reduced = all_gather(program, scattered, peers=peers, owners=owners)
            if program.node == args.root:
                if args.normalize is not None:
                    if reduced.grid[1] != 1:
                        raise ValueError('--normalize requires --tile-columns to cover the full output width')
                    spec = BlockSpec(reduced.block_shape, lambda i, j: (i, j))
                    reduced = program.kernel_call(normalize, grid=reduced.grid,
                        in_specs=(spec, BlockSpec(gamma.shape, lambda i, j: (0, 0))),
                        out_specs=spec, out_shape=ShapeDtypeStruct(reduced.shape, reduced.dtype))(reduced, gamma)
                spec = BlockSpec(down.block_shape, lambda i, j: (i, j))
                activated = program.kernel_call(kernels.swish,
                    grid=reduced.grid, in_specs=(spec,), out_specs=spec,
                    out_shape=ShapeDtypeStruct(reduced.shape, reduced.dtype))(reduced)
                consumed = linear(program, activated, consumer_weight, tile_rows=args.tile_rows,
                                  tile_k=args.tile_k, tile_columns=args.tile_columns)
                outputs.update({(instance, *index): program.export(ref) for index, ref in consumed.blocks.items()})
        program.realize()
        for i, j in inputs[0].blocks:
            for x in inputs:
                ref = x[i, j]
                row, column = i * x.block_shape[0], j * x.block_shape[1]
                with program.write(ref) as destination:
                    destination[...] = values[row:row + ref.shape[0], column:column + ref.shape[1]]
        if program.node != args.root:
            signal.pause()
            return
        while outputs:
            for index, output in tuple(outputs.items()):
                if output.ready:
                    print(index, output.array.tolist(), flush=True)
                    output.consume()
                    del outputs[index]
            if program.report.code:
                raise RuntimeError(f'Mesh execution error: {program.report.code}')


if __name__ == '__main__':
    main()
