# Ordinary tall fixed-result least-squares: actual native result

The unchanged public caller in lstsq-fixed-command.md passed with no strict kwarg or custom op in user source: compile_module → native artifact → Program reload (34 calls/positions, no relation solved) → invocation. The F32 MPS two-member tall batch contains a full-rank and a rank-deficient matrix. Original residual dtype/device source selection, keyword arguments and rank assertion remain in the connected captured function.

Ranks are [2,1], matching CPU-double GELSD. Relative solution errors are [2.4749481791158465e-08,3.9736429850260414e-08]; singular-value errors [1.5391152724464332e-07,5.3237706593483106e-08]. Returned shapes are [2,2,1], [2], [2,2], all finite. Process53685 exited0 with empty stderr. Source, installed Torch patch and loaded binary hashes were unchanged across the process (lstsq-fixed-native-1-source.json).

This closes the observed tall batched fixed-result producer-to-native boundary. It does not claim consumed genuinely dynamic residual tensors are static, or extrapolate to all precision/processor combinations. Static empty residual descriptors and fixed-solution derivatives have separate frontend source evidence. The same run used the Bound completion simplification now committed d3002abb; network paging C drafts were not built into these held libraries.
