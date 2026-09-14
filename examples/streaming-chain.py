import argparse
import signal
import sys

import numpy as np

from mesh import BlockSpec, Program, ShapeDtypeStruct, kernels
from mesh.nn import linear
from mesh.collective import reduce_scatter, all_gather


# design/algorithm-sources.md#pallas-panel-composition
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
    parser.add_argument('--region')
    parser.add_argument('--numerics')
    parser.add_argument('--backend', choices=('cpu', 'metal'), default='cpu')
    parser.add_argument('--tile-rows', type=int, default=128)
    parser.add_argument('--tile-k', type=int, default=128)
    parser.add_argument('--tile-columns', type=int, default=128)
    args = parser.parse_args()
    values = np.load(args.input)
    weights = (np.load(args.up_weight, mmap_mode='r'), np.load(args.down_weight, mmap_mode='r'))
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
    with Program(backend=args.backend, region=args.region, functions=functions) as program:
        shard = slice(0, args.split) if program.node == args.root else slice(args.split, weights[0].shape[1])
        weights = (weights[0][:, shard], weights[1][shard, :])
        x = program.tensor(values.shape, (args.tile_rows, args.tile_k), dtype=values.dtype)
        up_weight, down_weight = tuple(program.tensor(w.shape, dtype=w.dtype) for w in weights)
        for tensor, value in zip((up_weight, down_weight), weights):
            program.constant(tensor[0, 0], value)
        up = linear(program, x, up_weight, tile_rows=args.tile_rows,
                    tile_k=args.tile_k, tile_columns=args.tile_columns)
        spec = BlockSpec(up.block_shape, lambda i, j: (i, j))
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
        outputs = {}
        if program.node == args.root:
            spec = BlockSpec(down.block_shape, lambda i, j: (i, j))
            activated = program.kernel_call(kernels.swish,
                grid=reduced.grid, in_specs=(spec,), out_specs=spec,
                out_shape=ShapeDtypeStruct(reduced.shape, reduced.dtype))(reduced)
            weight = np.load(args.consumer_weight, mmap_mode='r')
            consumer_weight = program.tensor(weight.shape, dtype=weight.dtype)
            program.constant(consumer_weight[0, 0], weight)
            consumed = linear(program, activated, consumer_weight, tile_rows=args.tile_rows,
                              tile_k=args.tile_k, tile_columns=args.tile_columns)
            outputs = {index: program.export(ref) for index, ref in consumed.blocks.items()}
        program.realize()
        for (i, j), ref in x.blocks.items():
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
