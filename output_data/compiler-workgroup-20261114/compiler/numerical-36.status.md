# numerical-36: retain the existing typed triangle owner

Reviewed source: both formulation tracing sites use one fresh default-decomposition table with only tril/triu omitted. Existing operations._triangular_part consumes those operators using typed loop coordinates. The actual agenda construction declares its index ranges int32, matching its existing int32 state domain.

The initial worker draft included eye and cached the table. Supervisor removed both: installed torch/_refs/__init__.py5972ff reduces ordinary integer eye dimensions through _strength_reduce_integer; the actual 2×2 eye trace uses uint8 ranges. No eye failure justified another hook. Installed tril at6355ff does create default-int64 arange/subtract mask data, matching the live value44 source observation. No generic integer narrowing was added.

Validation: operations.py/decompose.py AST and diff whitespace pass. The actual 2×2 _qdwh_eigh_state fake trace with this table retains aten.tril and contains zero int64 nodes; eye remains uint8 and agenda ranges are int32. Full node evidence: ane-triangle-source.json. Source trace does not establish ANE construction or numerical execution.

Prior same-client attempt3 exited1 during construction: constant value44 has no Core ML tensor kind. executor/ane-value44.json retains live source/root facts; value44 is the triangle-mask arange view, not the surviving BOOL full operation. No Program or invocation occurred. Next: the same requested-ANE2×2 client on current matched libraries, without a rebuild.

Only operations.py and decompose.py changed. No native ABI, original poisoned path, placement or input change. The unrelated integration status file is excluded.
