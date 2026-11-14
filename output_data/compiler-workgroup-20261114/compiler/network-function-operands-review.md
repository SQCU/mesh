# Prepared network operands reach Function

The first unchanged five-node Crossing caller failed with ENOMEM before Program creation. Source inspection found a mandatory missing handoff: Target.functions constructed only numerical values/Actors, and solve_program left Function.outstanding/count/declaration fields zero. plan_function rejects a zero bound, and its transfer-count validation also requires the prepared rows. This proves a producer defect; it is not a live observation excluding every earlier ENOMEM return.

The network branch now installs Prepared.arrays[j]'s existing Declared/Weight pointers and actual count into Function[plan_calls[j]]. It materializes the retained canonical Dependence entries once and passes the original source function's analysis.outstanding bound. Original source_calls/outstanding still drive plan_graph. Native numerical values/Actors, source grouping, roots and uses stay with their existing owner. The same LocalOwner retains Prepared and the dependence arrays; declarations are not reconstructed.

SAME979 implementation and supervisor review agree on these exact fields. Supervisor Python AST and diff checks pass. No C/API/build, capacity change, new solve or diagnostic target was needed. Executor owns the next unchanged Crossing retry; successful distributed execution is not claimed here.
