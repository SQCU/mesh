# The Metal Shading Language specification, version 4.1: sources, quoted verbatim

Retrieved 2026-11-06. Each excerpt below was copied by script from the cited lines of the text file named; nothing in a
block is retyped.

- **Source:** Apple, "Metal Shading Language Specification", Version 4.1, a 383-page PDF created 2026-06-04,
  https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf, SHA-256
  41538b30d2f1140a5b2a0c84ce0a9f7b67bf0c707e224cfea0bfe5a44aa26cf5 →
  `scratchpad/lit/d-msl/Metal-Shading-Language-Specification.pdf`; the excerpts are lines of the text file beside it,
  `Metal-Shading-Language-Specification.txt`, made from the PDF on 2026-11-06.

## 1. §2.22.3, the cooperative tensor type

L4785–4812:

```text
2.22.3       Cooperative Tensor Type
The header <metal_cooperative_tensor> defines the
cooperative_tensor<ElementType, Extents, Layout> type. The
cooperative_tensor represents a tensor with elements that are partitioned across a set of
participating threads in thread memory. Each thread has access to only the elements in its
partition. These threads belong to the same threadgroup and may be spread across
consecutive SIMD-groups. You can’t define a cooperative_tensor in an aggregate type
(see section 2.12).


                 Table 2.31. Cooperative tensor template parameters
    Template         Description
    parameter
    ElementType      ElementType is the type of the underlying type in the tensor. For cooperative
                     tensor, the address space is thread.
    Extents          Extents describes the dimensions of the tensor using extents<…> (see
                     section 2.22.1).
    Layout           Layout specifies the mapping of the multidimensional coordinate space of
                     the tensor to the prepartitioned storage for each thread.


You typically don’t construct cooperative_tensor directly as the Layout is device
specific. Instead, you use libraries such as Metal Performance Primitives (see section 7), a
library of optimized primitives that include operators that work on tensors such as matrix
multiplication and convolution. You create them using the tensor operations, which use them to
store intermediate results. The tensor operation determines an efficient and performant
Layout for a cooperative_tensor based on its usage and the GPU.

```

## 2. §4.4, the threadgroup address space

L5512–5514:

```text
4.4 Threadgroup Address Space
A GPU compute unit can execute multiple threads concurrently in a threadgroup, and a GPU can
execute a separate threadgroup for each of its compute units.
```

## 3. §5.2.3, SIMD-groups within a threadgroup

L7316–7319:

```text
[[dispatch_quadgroups_per_threadgroup]] are similarly computed for threads
specified at dispatch.
SIMD-groups execute concurrently within a given threadgroup and make independent forward
progress with respect to each other, in the absence of threadgroup barrier operations. The
```

## 4. §6.10.1, threadgroup and SIMD-group synchronization functions

L10150–10158:

```text

     Table 6.12. Synchronization compute function in the Metal standard library
    Built-in threadgroup function                       Description

    void threadgroup_barrier(                           All threads in a threadgroup executing the
           mem_flags flags)                             kernel, fragment, mesh, or object need to
                                                        execute this function before any thread can
                                                        continue execution beyond the
                                                        threadgroup_barrier.
```

L10185–10192:

```text
A barrier function (threadgroup_barrier or simdgroup_barrier) acts as an execution
and memory barrier. All threads in a threadgroup (or SIMD-group) executing the kernel must
encounter the threadgroup_barrier (or simdgroup_barrier) function. The
threadgroup_barrier (or simdgroup_barrier) also supports a variant that takes a
thread scope parameter. When no thread scope is specified, threadgroup_barrier
defaults to thread_scope_threadgroup and simdgroup_barrier defaults to
thread_scope_simdgroup. On Apple silicon, a thread that has ended no longer participates
or blocks remaining threads at a barrier.
```

## 5. §6.17.2, compute commands encoded into indirect command buffers

L14944–14962:

```text
6.17.2      Encoding Compute Commands in Indirect Command Buffers
iOS: Metal 2.2 and later support indirect command buffers for compute commands.
macOS: Metal 2.3 and later support indirect command buffers for compute commands.
iPadOS and visionOS: Metal supports indirect command buffers for compute commands.



                   2026-06-04 | Copyright © 2026 Apple Inc. | All Rights Reserved.
                                         Page 315 of 383
ICBs allow the encoding of dispatch commands into a Metal buffer for subsequent execution on
the GPU.
In a shading language function, use the command_buffer type to encode commands for ICBs
into a Metal buffer object that provides indexed access to a compute_command structure:
struct arguments {
    command_buffer cmd_buffer;
};
[[kernel]] void producer(device arguments &args,
                         ushort cmd_idx [[thread_position_in_grid]])
{
```
