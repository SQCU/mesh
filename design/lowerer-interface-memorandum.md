# Memorandum: compatible lowerer interfaces

The runnable `schedule_emit` lowerer and `Program(scheduled, functions)` supply a concrete function-and-operand contract for lowering and executing the assigned program domain.
Threads-and-queues, MPS, ANE and combined CPU/MPS/ANE lowerers consume the same complete produced callgraph and optimization results at each shared boundary, with their original owners accessible by view.
Target-specific lowerers extend that interface with explicit additional operands and retain their target artifacts alongside callable results compatible with the reference consumer.
Codex review demonstrates this compatibility by tracing each changed producer and its direct consumers and executing the resulting program through the original reference runtime, using the actual transformed result to assess adherence to the assignment.

Authority: the operator's [16 November compatibility direction](../output_data/standalone-replacement-20261115/workgroup.md#compatible-operands-and-the-next-integration-work),
the [composition request's numerical lowerer boundary](compilation-composition-change-request.md#numerical-lowerer-boundary),
and the executable `schedule_emit/{driver,ir,lower_inductor,runtime}.py` interfaces.
