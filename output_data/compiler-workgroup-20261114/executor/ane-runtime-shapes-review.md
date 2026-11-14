# Runtime MIL shape consumer: bounded source checkpoint

Metal `38f67112` changes only `metal_neural.mm` (+336/-51). Executor and compiler reviewed the 65-line operand/result
boundary before SAME native e48 implemented it. Public ABI, fixed NeuralIO, placement, numerical source, saved native
framing and original poisoned files are unchanged. This is source progress, not an ANE execution acceptance result.

The lowerer now consumes actual Operation symbol ValueIDs through Need and keeps producers observable when a later
kernel uses them as symbols. Runtime extents feed unknown MIL result dimensions, actual-size slices, coordinate
ranges, reshapes, matmul dimensions and scatter updates into prior fixed output tensors. True MIL conditional blocks
guard zero domains; the parallel-empty branch returns prior outputs and common reduction-empty branches produce
the source combiner identity before existing ACCUMULATE/epilogue/Store processing. No full-bound masked substitute,
new scheduler, variant cache, placement reassignment or public shape ABI was introduced.

Explicit incomplete consumers:

- Coupled REGION reduction with any runtime loop domain fails, including a dynamic parallel axis with static
  reduction width. The current pairwise tree still consumes static dimensions.
- ARGMIN, ARGMAX and ASSIGN with a runtime reduction fail during construction of the required empty branch,
  irrespective of a particular invocation's positive extent. Their MIL empty-domain result is not connected.
- Core ML compile/predict success would not establish ANE placement, support for internal data-dependent shapes,
  or work proportional to the runtime extent. The existing MLComputePlan device results remain the evidence owner.

Worker and supervisor independently ran Objective-C++ strict syntax using the actual Makefile TINT_CFLAGS,
`-fobjc-arc -O2 -Wall -Wextra -Werror -fsyntax-only -I . -iquote .build -I ../dox/mesh/rdma metal_neural.mm`.
Both exited zero. The supervisor log is empty; `git diff --check` passed. No shared build, import, target invocation
or new evaluator was run. The source is held for compiler co-review and the coordinated native build/client gate.

The existing SAME session is `e48f4bc7-7cef-4afd-ae50-fc1b4618fa73`, actual model `claude-opus-5-5`; resume75a ended
successfully in 25 turns. An initial resume with the default config could not find the session and changed nothing;
the original `.claude-personal` session store was then used. Raw logs remain local; compact hashes are in the JSON.
