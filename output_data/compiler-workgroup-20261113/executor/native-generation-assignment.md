# Preserve native sending generation — active connected unit

SAME native e48f4bc7-7cef-4afd-ae50-fc1b4618fa73, explicit claude-opus-5-5. Read current compiler-workgroup, composition CR items 2/3/6/7, plan-to-bridge R4 and synchronization-signature clauses. Previous owned-result unit is committed metal ee02e6c4 / mesh0dcdcd15 and complete source-ready. Do not reopen unrelated files.

## Exact defect

metal_transport.mm Steps inserts METAL_IR_SENT before CPU/ANE writes: prior use of that physical copy must complete. MetalHostEntries reads actual mesh_metal_sending.sent.copies, but retains only its completion pointer and combined call residue period. NativeTransfer lacks the prior-copy distance. LowerProgram currently waits mesh_host_sent(current producer execution) before that execution is written/published, unconditionally even startup. This loses the prepared generation operand and can deadlock. Do not treat pointer-table LCM copies as actual transport copies.

## Source and fields

Own only metal_program.h NativeTransfer, metal_recording.m transport preparation/signature emission, metal_lower.mm fixed NATIVE_DISPATCH emission, and exact archive field if it is genuinely serialized (check current native-binary archive before changing). Compiler confirms these hunks free; no shared build until ready.

Read mesh_metal_sending.sent{counter,sends,copies}, mesh_metal_range.input.copies and generation; metal_recording.m MetalHostEntries/MetalNativeHost; mesh-collective.c mesh_host_sent/arrived/publish; actual NativeTransfer source. Add the smallest retained physical-copy/generation operands to NativeTransfer, separate from existing pointer table residue count. SENT uses actual sent.copies as prior execution distance. If owner execution E < actual copies, no wait occurs. Otherwise wait on E−copies, matching counter slot and threshold (round+1)*sends in the existing canonical helper. ARRIVED and PUBLISH keep their existing actual producer execution mapping. No new counter, scheduler, delay inference or runtime instrumentation. Confirm multiple calls/every/ordinal meanings rather than replacing them by t. Existing completion pointer table is indexed by its retained consumer execution period; prior whole-copy delay selects the same physical slot.

## Signature and persistence consumer

The same retained operands must select startup omission, source execution, physical role and ring slot in MetalSitesEmit. MetalTransportObject currently keys only plan+transfer and falsely merges publication, arrival, completion; retain role and exact copy slot. Publication is its supplied stream, arrival/completion are separate ring words. Preserve actual source occurrence/dependence delay; do not fabricate bridge raises or use PLAN_COUNT+transfer (PLAN_COUNT is BSL added-edge identity). Normalized generation threshold may be used only with explicit physical role/slot correspondence; do not claim a matching bridge event that was never emitted.

Use the newly existing mesh_graph_signature_span(planned->signature,executions,&expanded) before mesh_graph_compose in MetalProgramSignatureSpans, free owned expanded result. Source spans have cumulative first/count as current code; no repeated plan solve or extrapolation from finite records. Verify this exact API against current header. Old finite records without templates must retain existing ENODATA truth, no bypass. This does not solve independent saved binding-recipe reconstruction; report that as separate obligation.

NativeTransfer is currently a preparation struct embedded into final generated code immediates. Archive saves final executable+symbol and rebuilds preparation from saved typed/send records under library build identity. Trace that new fields are either present in saved canonical sending/range records and final binary or add only the genuinely missing persisted operand. Do not introduce a general serializer/runtime opcode reader. Explain actual archive path and its known independent correspondence gap.

## Review/evidence

Make exact source diff; private strict compile of changed Metal files and all public header consumers using existing Makefile flags. No shared build; root combines owned graph/extent ABI checkpoint. Check generated entry source for first-call omission and later E−copies, including every>1/ordinal via existing source emission if feasible without new evaluator. Actual live bridge waits remain root-coordinated evidence, not assumed from compilation. No fake native context execution. Report exact fields/consumer status promptly to executor/native-generation-status.md. No source reset/stash/backups or worker commits.
