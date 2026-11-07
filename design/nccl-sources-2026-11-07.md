# NCCL User Guide: sources, quoted verbatim

Retrieved 2026-11-07 with curl into the session's download directory `scratchpad/lit4-nccl/`. Each quotation was taken
from the page's text with its markup removed.

## 1. "Collective Operations"

- **Source:** NVIDIA, *NCCL User Guide*, "Collective Operations", page title "Collective Operations — NCCL 2.32.3
  documentation", https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html
- **Read:** yes, `lit4-nccl/collectives.html`.

> Collective operations have to be called for each rank (hence CUDA device), using the same count and the same
> datatype, to form a complete collective operation. Failure to do so will result in undefined behavior, including
> hangs, crashes, or data corruption.

## 2. "Using NCCL with CUDA Graphs"

- **Source:** NVIDIA, *NCCL User Guide*, "Using NCCL with CUDA Graphs", page title "Using NCCL with CUDA Graphs — NCCL
  2.32.3 documentation", https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/cudagraph.html
- **Read:** yes, `lit4-nccl/cudagraph.html`.

> Starting with NCCL 2.9, NCCL operations can be captured by CUDA Graphs. CUDA Graphs provide a way to define workflows
> as graphs rather than single operations. They may reduce overhead by launching multiple GPU operations through a
> single CPU operation.
