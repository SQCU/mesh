# Page arrival and storage reuse, 23 November 2026

## A: Before the revision

For each participating node pair, `plan_endpoints` assigned a link number and a rate of one. The C scheduler and Python program scheduler both treated that number as an exclusive resource. Two independent page writes between the same nodes could acquire a dependency from the selected resource order. The bridge independently selected the physical cables and relays.

## B: Endpoint compilation

For each transferred page, the scheduler multiplies its byte count by the transfer operand’s cost per byte. It assigns that delay to arrival, records both participating nodes on the instruction, and adds the producer and reader edges. Independent pages can arrive concurrently. The solver still orders numerical tasks that require the same exclusive resource. The allocator uses the resulting dependency relation to decide whether all accesses to one value finish before another value begins. [Disjoint pages, sections C3 and D3](disjoint-pages-change-request.md) · [Flow order and lowering](flow-order-and-lowering-memorandum.md)

## Observed on this revision

Mesh `c7a45b3a6` and compiler-emit `ed9e232` compiled `sin(x) + cos(y)` for two sites. Both input tensors had 16,384 float32 values. The first site computed both operands. The second site added them. With a 16 KiB page and a transfer weight of two per page, both 64 KiB operands generated four page arrivals. All eight carried delay two and the same two-node mask. No exclusive resource cycle contained an arrival. Neither destination lifetime preceded the other. The compiled page table assigned pages 0–3 to the sine result and pages 4–7 to the cosine result. Loading the serialized compilation reproduced both page assignments. The threads-and-queues execution matched eager PyTorch for three distinct input pairs.

The physical execution and timing records for the DNN workloads belong to their respective compiler revisions. [Fleet DNN measurements](../../../compiler-emit/artifacts/fleet-dnn-20261123/measurements-20261123.md)
