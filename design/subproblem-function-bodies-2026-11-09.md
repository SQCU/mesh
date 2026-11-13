# Memorandum: what the functions already written for each subproblem do

2026-11-09. Written by Claude at the operator's request, from the session
`.claude-personal/projects/-Users-mdot-metal-microbench/55974018-f56d-4c8b-8973-6fc61b49140c.jsonl`. It is the detail
behind [already-solved-subproblems-2026-11-09.md](already-solved-subproblems-2026-11-09.md), which summarises it.
Like the other memoranda of this date it is an aid and not a source.

## The turn

The operator, 2026-11-09, quoting the earlier manifest's limit:

> "- "Written" means a function of that purpose exists. I did not read the bodies, run them, or establish their correctness or coverage.' use haiku-parallel repoistory parsing yet again...

## What was done

Eleven readers (Claude Haiku), one per subproblem, each read-only, ran in parallel. Each was given the names of the
functions the manifest lists and told to read each whole body and state: what the body does, from the code and not
from its name; what it takes and returns; the source it cites for its method, quoted; every case it refuses, with
the message and line; who calls it; where two or more functions compute the same thing; and which parts of the
subproblem have no function. Each was told to make no judgment of correctness and no proposal. Nothing was built or
run.

## What this is and is not

- Code was read at mesh 7df87fe4 and the metal-microbench working tree on 7483572e. Where a reader read a file from
  an earlier commit, the commit is named in the location.
- The statements are the readers' own, reformatted. Claude checked these against the repositories: that
  `mesh_graph_schedule` and `mesh_graph_unfold` have no call site; that `mesh_graph_compose` composes signatures;
  that no `torch_mesh` source is tracked at mesh HEAD; that metal-microbench has no `configs/links`; that
  `rdma/allocate.py` is imported nowhere. Each held.
- One class of the readers' statements is unreliable: "no callers found". Claude found 18 call sites of `_layout`,
  4 of `_aliases` and 1 of `_view` in `rdma/mesh_compile.py`, where a reader reported none. Treat every "none found"
  under **callers** as unchecked.
- "Part of the body" and "declaration only" mean what they say: that function's refusals may be incomplete.
- Reading a body supplies a source observation. It does not establish reachability, executed behavior or
  correctness. Nothing here records an end-to-end run.

## Selecting what a supplier of numeric functions offers

Read: 15 whole body, 5 part of the body, 2 declaration only. 22 functions.

- **`_contract`** — mesh:rdma/mesh_compile.py:2975-3008 (whole body)
  - does: Computes output sizes from out.shape and operand shapes (max per label); summed = operand labels not in output;
    domain = output+summed; if out.is_complex() returns _complex_contract(...). Else t=_compute(out.dtype, operand
    dtypes, float32 if out floating); direct = alpha==1 and (addend None or beta==0); acc = out if direct else a
    temporary. Builds one _Region that loads each operand and multiplies them, r.done(product); maps each operand's
    rows; emits one _Generic over the domain with PARALLEL for output labels, REDUCTION for summed labels, combiner
    [SUM], precision=REASSOCIATE|CONTRACT. If direct returns that list; otherwise appends _epilogue applying alpha
    (mul) and beta*addend (add).
  - cites: Comment above the function at mesh_compile.py:2973: 'PyTorch 2.14 documentation, torch.einsum: the output's
    labels PARALLEL, the summed REDUCTION (SUM in the operands'...' (truncated in my read).
  - refuses:
    - No raise, error return or assertion in 2975-3008.
  - callers: _complex_contract (1 call, mesh_compile.py:2984); _matrix (3044); _product (3061, 3069, 3082); _summation (3106,
    3116). 7 call sites in mesh_compile.py. None found in mesh:bin, metal-microbench:tools or
    metal-microbench:provider.
- **`_complex_contract`** — mesh:rdma/mesh_compile.py:3010-3041 (whole body)
  - does: Computes the sum over k of complex products as separate real and imaginary parts: t=F64 if any of out or
    operands is complex128/float64 else F32; a complex operand goes through _pair into inputs and _complex_value
    'mul' with the running product; a real operand is loaded and multiplies both parts. Emits one _Generic with
    outputs (acc.real, acc.imag), combiners SUM and SUM. If alpha!=1 scales via _complex('mul'); returns made +
    _complex('add', out, acc, addend, beta) when addend is given and beta!=0, else _complex('copy', out, acc).
  - cites: Comment at mesh_compile.py:3009: '# sum_k (ar_k br_k - ai_k bi_k, ar_k bi_k + ai_k br_k)' (formula only).
  - refuses:
    - No raise or error return in 3010-3041.
  - callers: 1 call site: _contract at mesh_compile.py:2984.
- **`_reduced`** — mesh:rdma/mesh_compile.py:5246 through the line before the comment at 5259 (whole body)
  - does: Calls _typed(target,[x],outs,REAL); dims=_dims(given,x); t=_compute(out.dtype, x.dtype); acc = temporary if
    packet is A.mean else out; kept = acc squeezed over dims if keepdim; made=[_reduction(packet,x,dims,kept,t)]; if
    packet is not A.mean returns made; else divides the accumulator by total = product of reduced sizes via
    _epilogue with r('div').
  - cites: Comment at mesh_compile.py:5213 (above REDUCTIONS, line 5216): 'PyTorch 2.14 documentation, torch.sum,
    torch.mean, torch.amax and torch.amin: the free dimensions PARALLEL and the ...' (truncated). The FOLDS comment
    at 5259-5262 is not used by _reduced.
  - refuses:
    - Via _typed (2170-2181): mesh_compile.py:2173-2174 NotImplementedError('%s of %s: PyTorch 2.14 defines no form
      of it for that type (its CPU kernel admits none)') for complex dtype (or bool with CONTRACTED results);
      mesh_compile.py:2177-2178 NotImplementedError('%s of %s: its lowering into the structured basis is unfinished
      (design/compilation-composition-change-request.md item 1)') for any other non-REAL dtype.
    - Via _reduction (5228-5238): packets other than A.sum/A.mean take the extremum path with _identity(packet,
      x.dtype) (not read).
  - callers: Dispatch table SOURCES at mesh_compile.py:6626 (dict.fromkeys(REDUCTIONS, _reduced)); REDUCTIONS = {sum, mean,
    amax, amin} at 5216. Direct internal calls (15): mesh_compile.py 3214, 3405, 3697, 3706, 3715, 3875, 3881, 3882,
    3920, 4122, 4302, 4308, 4461, 4689, 4708.
- **`_selected`** — mesh:rdma/mesh_compile.py:5637-5702 (whole body)
  - does: Calls _typed(target,[x],outs[:1],REAL). For median/nanmedian with no 'dim' flattens x to 1-D with a copy; else
    dim d normalized and a 0-d input viewed as 1-D; outputs squeezed when keepdim. Adds a _Sort(x,d,n,False,sv,si)
    op. Inner picked(values, indices, nan) builds a Region selecting value and index. Dispatch in order: kthvalue ->
    picked at k-1; nanmedian on floating -> _ranked_at; median/nanmedian -> k=(n-1)//2; integer dtype picked
    directly; floating dtype: max over the line via _extreme(A.max.dim) and picked with a NaN select. The remaining
    packet, mode (SELECTIONS = kthvalue, median, nanmedian, mode), builds run marks/starts/counts over sorted
    neighbours, takes the longest run via _extreme(A.max.dim), and extracts value and index at that position.
  - cites: Comment at mesh_compile.py:5629-5633 (above SELECTIONS at 5634): 'PyTorch 2.14 documentation, torch.kthvalue,
    torch.median and torch.mode: order statistics of each line from its...' (truncated).
  - refuses:
    - No raise in 5637-5702. Refusals only through _typed (mesh_compile.py:2173-2174, 2177-2178).
  - callers: Dispatch table SOURCES at mesh_compile.py:6642 (dict.fromkeys(SELECTIONS, _selected)); no direct call sites
    found.
- **`_lowering`** — mesh:rdma/mesh_compile.py:6694-6723 (whole body)
  - does: packet = overloadpacket of local.target. Ordered branches: scaled_dot_product_attention -> lambda calling
    _attended; _linalg_det/_linalg_slogdet -> _determinant with transpose flag from input meta (contiguous and
    real); linalg_pinv -> _pseudoinverse with float32 when device type is mps/metal else float64; target in SYMBOLIC
    -> _elementwise(SYMBOLIC[target], outs[0], *args); packet in LINEAR -> lifted (calls _lifted when any argument
    or output is complex, else SOURCES[packet]); packet in SOURCES -> SOURCES[packet]; packet in VIEWS ->
    _copy(outs[0], _call(...)); otherwise raise.
  - cites: VIEWS comment mesh_compile.py:6644 'PyTorch 2.14 documentation, Tensor Views'; LINEAR comment 6651-6654 'PyTorch
    2.14 documentation, Complex Numbers (and torch.nan_to_num ...)'; SYMBOLIC comment 6685-6686 'Python's operator
    module on a held scalar (torch.fx symbolic arithmetic ...)'; SOURCES comment 6609-6610 cites
    design/torch-source-lowering-change-request.md item 1 and design/compilation-composition-change-request.md.
  - refuses:
    - mesh_compile.py:6722-6723 raise NotImplementedError('%s: its lowering into the structured basis is unfinished
      (design/torch-source-lowering-change-request.md item 1)' % (local.target,)) for any target no earlier branch
      matches.
  - callers: 1 call site: _called at mesh_compile.py:6882 (_called called once at 7069).
- **`_lifted`** — mesh:rdma/mesh_compile.py:6665-6679 (whole body)
  - does: For k in (0,1) (real, imaginary part): recursively maps arguments: complex tensor ->
    torch.view_as_real(v)[...,k]; real tensor with a datum flag and k=1 -> zero temporary filled with 0; python
    complex/float/int datum -> _component(v,k); others unchanged. Calls lower(target, made, parts-of-outs) and
    accumulates ops.
  - cites: Via _lowering's LINEAR comment at 6651-6654 (each-component rule); no citation in the body.
  - refuses:
    - None in the body.
  - callers: 2 call sites: _lowering at mesh_compile.py:6715; mesh_compile.py:4001 (in _taken, return
    _lifted(A.gather.default, _taken, ...) if source.is_complex()).
- **`_lowered`** — mesh:rdma/mesh_compile.py:8361-8363 (whole body)
  - does: packet = getattr(op,'overloadpacket',op); returns True when packet is in SOURCES or VIEWS, else False.
  - cites: Comment at mesh_compile.py:8357-8360 sits above it (torch 2.14.0 torch/export/decomp_utils.py
    default_decompositions; torch/_decomp/__init__.py decomposition_table; torch/export/exported_program.py
    run_decompositions).
  - refuses:
    - None.
  - callers: 2 call sites: _decomposed at mesh_compile.py:8390 and 8395.
- **`_decomposed`** — mesh:rdma/mesh_compile.py:8371-8399 (ends before def _metadata at 8402) (whole body)
  - does: Collects storage identities of exported state (state_dict tensors and non-persistent buffers). For each
    GraphModule: get_attr constants not in state are added to exported.constants; call_function nodes with target
    linalg_lstsq.default get driver rewritten via _lstsq_driver; module.recompile(). Builds table =
    torch.export.default_decompositions() minus OpOverloads for which _lowered is true. known = decomposition_table
    merged with LAZY. Runs exported.run_decompositions({table, LAZY}), then loops: wanted = non-lowered
    call_function targets in known and not yet done; if none, returns exported; else run_decompositions on known[op]
    for the wanted ops.
  - cites: Comment at 8357-8360 (torch 2.14.0 decomp_utils.py, decomposition_table, exported_program.py
    run_decompositions); LAZY comment at 8366-8367 (PyTorch 2.14 documentation, torch.conj and torch.conj_physical;
    Tensor.neg negative-bit view).
  - refuses:
    - None in the body.
  - callers: 1 call site: _exported at mesh_compile.py:8419 (_exported defined at 8411).
- **`_typed (dependency of _reduced and _selected)`** — mesh:rdma/mesh_compile.py:2170-2181 (whole body)
  - does: Collects dtypes of tensor operands not in REAL and of outs not in results; raises if any found and (first found
    is complex, or results is CONTRACTED and dtype is bool); otherwise raises for any remaining found dtype. Returns
    nothing.
  - cites: Refusal texts cite 'PyTorch 2.14 defines no form of it for that type (its CPU kernel admits none)' and
    '(design/compilation-composition-change-request.md item 1)'.
  - refuses:
    - mesh_compile.py:2173-2174 NotImplementedError('%s of %s: PyTorch 2.14 defines no form of it for that type (its
      CPU kernel admits none)')
    - mesh_compile.py:2177-2178 NotImplementedError('%s of %s: its lowering into the structured basis is unfinished
      (design/compilation-composition-change-request.md item 1)')
  - callers: Called from _reduced (5248), _selected (5638) and other op functions not counted here.
- **`_exported (caller of _decomposed, for context)`** — mesh:rdma/mesh_compile.py:8411-8425 (read to 8425) (part of the body)
  - does: torch.export.export(function.module, args, kwargs, strict=False); if function.loss is not None runs
    _export_forward_backward under sdpa_kernel(MATH); then _decomposed(exported) at 8419; then builds TrainingResult
    when loss is set.
  - cites: Comment at 8408-8410 (torch 2.14.0 torch/export/experimental/__init__.py _export_forward_backward).
  - refuses:
    - None read.
  - callers: 1 call site: mesh_compile.py:8494 (enclosing function not identified).
- **`Unary`** — metal-microbench:provider/litert_mesh.cc:1012-1071 (whole body)
  - does: kind=r.KindOf(x); fl = kind is F32 or F16; f = r.F(x) when fl else x. switch(md::OperationType): kAbs -> Call
    abs(x); kCeil/kFloor/kLog/kRound/kExp/kSqrt -> Call ceil/floor/log/round/exp/sqrt(f); kCos,kSin -> Call on
    r.F(x); kCopy -> x; kElu -> select(f, exp(f)-1, f<0); kGelu -> 0.5*f*(1+Erf(f*0.7071...)); kGeluTanhApprox ->
    0.5*f*(1+Tanh(0.79788*(f+0.044715*f^3))); kHardSwish -> f*clamp(f*0.16666667+0.5, 0, 1); kLogicalNot -> NOT if
    BOOL else COMPLEMENT; kNeg -> NEGATION; kRsqrt -> Call inverseSqrt(f); kSigmoid -> 1/(1+exp(-f)); kSign ->
    sign(f) when float else clamp(x,-1,1); kSquare -> x*x; kTanh -> Tanh(r,f); kMish -> f*tanh(softplus) via exp and
    a select at f>8. default returns false.
  - cites: None in code.
  - refuses:
    - litert_mesh.cc:1070 'default: return false;' for any md::OperationType not listed (writes nothing, returns
      false).
  - callers: Graph::Elementwise at litert_mesh.cc:1275; Builtin::Elementwise at 3446; direct call at 2249
    (Unary(md::OperationType::kGeluTanhApprox, r, x, &out)) in a function not identified.
- **`Binary`** — metal-microbench:provider/litert_mesh.cc:1074-1131 (whole body)
  - does: kind = F32 if either operand kind is F16 or F32, else the kind of a; fl = kind==F32; a,b converted to kind via
    r.Cvt. switch: kAdd/kSub/kMul/kDiv -> ADD/SUBTRACT/MULTIPLY/DIVIDE (kDiv is a plain DIVIDE in the shown branch);
    kMaximum/kMinimum/kPow/kAtan2 -> Call max/min/pow/atan2; kSquaredDiff -> d*d with d=a-b; kFloorDiv: float ->
    floor(a/b); integer -> c=a/b, c-1 when inexact and signs differ, else c; kFloorMod: float -> a-floor(a/b)*b;
    integer -> c=a-b*(a/b) with a fix add when inexact and signs differ; kMod/kRemainder -> MODULO; comparisons ->
    LESS/LESS_EQUAL/GREATER/GREATER_EQUAL/EQUAL/NOT_EQUAL; kLogicalAnd/kLogicalOr -> LOGICAL_AND/OR when BOOL else
    AND/OR; kLogicalXor -> NOT_EQUAL for BOOL else XOR; kShiftLeft/kShiftRight -> SHIFT_LEFT/RIGHT with b converted
    to U32. default returns false.
  - cites: None in code.
  - refuses:
    - litert_mesh.cc:1131 'default: return false;' for any md::OperationType not listed.
  - callers: Graph::Elementwise at litert_mesh.cc:1278 and 1280; Builtin::Elementwise at 3446.
- **`Elementwise (Graph class)`** — metal-microbench:provider/litert_mesh.cc:1260-1289 (whole body)
  - does: Reads attr = Attr<md::ElementwiseAttributes>(node). If attr and in.size()==1: Param(attr->param,&scalar); the
    value UINT32_MAX-2 is refused; otherwise the param value is inserted as second input. Calls l_.Map over the
    inputs; in the lambda: one input and no scalar -> Unary; one input with scalar -> Binary with the scalar on the
    left or right per attr->runtime_tensor_is_second; two inputs -> Binary. Result is r.Cvt(Compute(out scalar
    kind), result) on success, else r.Arg(0). Returns ok.
  - cites: None in code.
  - refuses:
    - litert_mesh.cc:1267 'if (p == UINT32_MAX - 2) return false;'.
    - Unary/Binary false returns (1070, 1131) set ok=false, returned from the function.
  - callers: Graph class Lower at litert_mesh.cc:2853 (return Elementwise(node, type, in, out[0]) for the listed
    md::OperationType values).
- **`Builtin::Elementwise`** — metal-microbench:provider/litert_mesh.cc:3441 to the end of the function (about 3452) (part of the body)
  - does: kind = Kind(out); l_.Map(in, out, lambda): the lambda calls Unary when in.size()==1 else Binary; on failure
    returns r.Arg(0); applies Activate(r, r.F(v), act) when act != kTfLiteActNone; returns r.Cvt(kind, v). Returns
    ok.
  - cites: None in code.
  - refuses:
    - Returns ok=false when Unary/Binary return false (litert_mesh.cc:1070, 1131 via the ok flag).
  - callers: Builtin::Lower at litert_mesh.cc:4060 (unary table), 4067 (binary table), 4072 (kGelu/kGeluTanhApprox).
- **`Builtin::Parsed`** — metal-microbench:provider/litert_mesh.cc:3972-4015 (whole body)
  - does: Returns false at once when index<0. Collects input tensor indices (not kTfLiteMmapRo); builds
    TfLiteDelegateParams with the single node index; patches tfl_->GetNodeAndRegistration; calls
    litert::ml_drift::BuildFinalModel(tfl_, &params, ..., graph.get(), ..., &factory) building md::GraphFloat32 with
    CustomOperationParserFactory; restores the function pointer and frees arrays. If status is not ok or
    graph->nodes() is empty, returns false with selected=false. Otherwise selected=true, Val() the inputs and
    outputs, l_.Retain(graph), constructs Graph lowering, binds tensors and quantized-to-floating maps, returns
    lowering.Lower(*graph).
  - cites: None in code.
  - refuses:
    - litert_mesh.cc:3974 'if (index < 0) return false;'
    - litert_mesh.cc:4005 'if (!status.ok() || graph->nodes().empty()) return false;' (selected stays false;
      Builtin::Lower then continues to its hand-written table).
  - callers: 1 call site: Builtin::Lower at litert_mesh.cc:4022.
- **`Builtin::Lower`** — metal-microbench:provider/litert_mesh.cc:4017-~4824 (about 800 lines) (part of the body)
  - does: First: if the node is not a delegate-with-builtin-data node, calls Parsed (4022) and returns if selected. If the
    node is a delegate with builtin data and sub_ is set, lowers each node in nodes_to_replace recursively
    (missing.insert Name on failure). Otherwise: static maps unary (4040-4048) and binary (4049-4059) from
    kTfLiteBuiltin codes to md::OperationType; unary -> Elementwise; binary -> Elementwise with fused activation for
    Add/Sub/Mul/Div; then switch(code) over about 100 TFLite builtin codes to individual handlers (Gelu, Relu
    family, Cast/Quantize, Select, Reshape/Squeeze/ExpandDims, Transpose, Concatenation, Pack/Unpack, Slice, Pad
    family, StridedSlice, Stablehlo*, Gather*, Reduce*, Softmax, Range, Shape, FullyConnected with optional
    activation, etc.). default returns false (4818-4819). Read: lines 4017-4070 and 4800-4830; middle handlers
    4070-4800 not read, their refusals taken from grep.
  - cites: None in code.
  - refuses:
    - litert_mesh.cc:4818-4819 'default: return false;' for any unhandled builtin code (the caller records it as
      'builtin N').
    - Per-handler 'return false' with no message at, for example, 4114, 4172, 4205, 4211, 4220, 4228, 4241, 4248
      (function not found), 4255, 4269 (nested Inline fails), 4326, 4336, 4361, 4372, 4388, 4430, 4444, 4474, 4493,
      4556, 4611, 4621, 4678, 4706, 4812 (MatMul fails). The message comes from the caller.
    - Caller-recorded refusal text: litert_mesh.cc:4950-4951 unconverted.insert(Builtin::Name(...) + ' (' +
      cpu.error + ')'); the final error at litert_mesh.cc:4973-4975 is ENOTSUP 'TFLite nodes outside the delegated
      graph without a lowering in this provider revision: ...'.
  - callers: Builtin::Inline at litert_mesh.cc:3236-3243 (nested, per node of a subgraph); LiteRtMeshPublish at
    litert_mesh.cc:4945-4950 (per step with no kernel); Lower's own delegate branch (recursive).
- **`Builtin::Name`** — metal-microbench:provider/litert_mesh.cc:3211-3215 (whole body)
  - does: Returns 'custom <name>' for a custom registration with a name, else 'builtin <code>'.
  - cites: None.
  - refuses:
    - None (label function).
  - callers: Builtin::Inline (3243); Builtin::Lower delegate branch; LiteRtMeshPublish (4951).
- **`LowerNeural`** — metal-microbench:metal_neural.mm:1250-1358 (whole body)
  - does: For kernel k: Lowering(p); for each non-erased operation of the kernel calls lowering.Lower(op)
    (metal_neural.mm:772) and fails on a missing op; fails if lowering.missing is non-empty. outside(value) =
    observable, or read by an operation outside the kernel with the same storage. Creates a Neural object (k.kind =
    METAL_IR_NEURAL, k.neural = Invoke). For each lowering.inputs value calls bind(value, 1, 'in_<id>', n->inputs):
    checks Layout (strides) and that the scalar is F16/F32/I32/U32; records a Binding (slot, storage, access,
    offset, bytes) and an IO entry. Builds the Core ML protobuf description for inputs. For each kernel op output
    not in lowering.outputs: if not outside, sets p.values[value].internal = kernel; else appends to
    lowering.outputs. For each output: adds an MIL identity op named out_<id> and bind(value, 2, ..., n->outputs).
    Builds the MIL program (block, function, specialization CoreML8, entry 'main', model with description at field
    502), sets n->program, n->weights, n->summary and returns 0.
  - cites: Comment at metal_neural.mm:1246-1249: '../dox/mesh/design/compilation-composition-change-request.md item 5: a
    kernel of ANE-placed operations (consecutive in program order, Lower's grouping) as one Core ML function'.
  - refuses:
    - metal_neural.mm:1255 LowerFail(error, room, 'ANE operation ' + op + ': ' + lowering.missing) when an op's
      Lower fails.
    - metal_neural.mm:1256 LowerFail(error, room, 'ANE kernel ' + kernel + ': ' + lowering.missing) when
      lowering.missing is non-empty.
    - metal_neural.mm:1278 LowerFail(... 'ANE kernel ' + kernel + ': ' + why) when Layout fails (why from
      metal_neural.mm:1148-1149: 'a bound value whose layout is not a positive stride per dimension (MLMultiArray
      strides)').
    - metal_neural.mm:1280-1281 LowerFail(... 'value ' + value + ' binds a kind MLMultiArray has no type for') when
      scalar is not F16/F32/I32/U32.
    - bind returns EINVAL after a LowerFail (metal_neural.mm:1278-1281).
  - callers: 1 call site: metal_lower.mm:2574 ('if (int status = LowerNeural(p, k, error, room)) return status;'), in the
    Lower pass for kernels where neural(p.kernels[k]) holds (metal_lower.mm:2521-2526). Declared at
    metal_lower.h:186.
- **`Lowering::Lower(uint32_t) (ANE per-operation lowering, called by LowerNeural)`** — metal-microbench:metal_neural.mm:772-~860 (read 772-800) (part of the body)
  - does: Reads op = p.operations[index]; Contract(p, op) for the contraction description; sets weightInput when the
    contraction element type matches the accumulator and K>1 and sizes fit INT32; for each input: unused parameter
    -> empty Tensor; materialized constant -> embedded sources, shape with op.bounds on prepared axes, and Tensor;
    otherwise Need(value) and values[prepared->value] with permutation. Body beyond line 800 not read.
  - cites: None read in body.
  - refuses:
    - metal_neural.mm:775 Fail('operation N without a body')
    - metal_neural.mm:846 Fail('REGION reduction without REASSOCIATE')
    - metal_neural.mm:916 Fail('floating reduction without REASSOCIATE (Core ML associates reductions freely)')
    - metal_neural.mm:940 Fail('combiner without a MIL reduction')
    - Fail (metal_neural.mm:337-340) keeps only the first message in 'missing' and returns false.
  - callers: 1 call site: LowerNeural at metal_neural.mm:1254 (per op).
- **`Lowering::Contraction (ANE matmul, for context)`** — metal-microbench:metal_neural.mm:948-960 (read through 960) (part of the body)
  - does: Takes Contract(p, op); requires element type to match the region's compute type; MilType(element) must be
    nonzero; casts operands to dtype when not materialized; returns false when x.shape[k]==1 or y.shape[k]==1; then
    forms batch/rows/cols/rest dimensions for one MIL matmul over [batch,rows,k] x [batch,k,cols].
  - cites: Comment at metal_neural.mm:948-950: 'A contraction (one REDUCTION loop k, SUM, the region the product of two
    input parameters, reassociable) as one MIL matmul ... (the Neural Engine's matrix operation; no rows x cols x k
    product tensor)'.
  - refuses:
    - metal_neural.mm:952 return false (element type mismatch or no contraction description)
    - metal_neural.mm:955 return false (no MIL type)
    - metal_neural.mm:959 return false when a contracted extent is 1
  - callers: Inside Lowering::Lower's loop (metal_neural.mm:772-860); no other call sites read.
- **`_fold (dispatch target, not read)`** — mesh:rdma/mesh_compile.py:5267 (declaration only)
  - does: declaration only
  - cites: Comment at 5259-5262 (torch.prod, any, all, argmax, argmin, max, min, var; SharedReduceOps.h
    LessOrNan/GreaterOrNan).
  - refuses:
    - not read
  - callers: SOURCES dict at mesh_compile.py:6633 (dict.fromkeys(FOLDS, _fold)).
- **`_ranked (dispatch target, not read)`** — mesh:rdma/mesh_compile.py:5618 (declaration only)
  - does: declaration only
  - cites: Comment at 5610-5611 (torch.sort and torch.topk; metal_ir.h metal_ir_sort).
  - refuses:
    - not read
  - callers: SOURCES dict at mesh_compile.py:6639 (A.sort: _ranked, A.topk: _ranked).

**Documents**

- mesh:design/already-solved-subproblems-2026-11-09.md: Line 96, a table row for this subproblem: '| 1. selecting what the supplier offers | `rdma/mesh_compile.py`:
  `_contract`, `_reduced`, `_selected`, `_lowered`, `_decomposed`; `provider/litert_mesh.cc`: the lowering of LiteRT
  nodes; `metal_neural.mm`: `LowerNeural` | 11-07 (`_contract`), 11-09 (the rest) | in three modules, each also
  emitting the bodies of numeric functions | one function on tensor operators, before scheduling |'. Found by grep;
  only that line was read.
- mesh:rdma/mesh_compile.py: Comments cite PyTorch 2.14 documentation for each op family (2907 fill_/arange; 2973 einsum; 3054
  mm/bmm/matmul/addmm/linear; 3087 einsum pairwise; 5213 sum/mean/amax/amin; 5259 prod/any/all/argmax/min/max/var;
  5284 complex prod/any/all; 5553 cummax/cummin; 5574 searchsorted; 5629 kthvalue/median/mode; 5704 nanmedian; 6644
  Tensor Views; 6651 Complex Numbers; 6685 operator module; 8366 conj). Refusals cite
  design/compilation-composition-change-request.md item 1 (2177-2178) and
  design/torch-source-lowering-change-request.md item 1 (6609-6610, 6723). Other citations: torch 2.14.0
  aten/src/ATen/native/SharedReduceOps.h (5261); torch 2.14.0 torch/export/decomp_utils.py (8357).
  - sources it names: PyTorch 2.14 documentation (einsum, mm/bmm/matmul/addmm, sum/mean/amax/amin,
    prod/any/all/argmax/argmin/max/min/var, cumsum/cumprod/logcumsumexp, cummax/cummin, searchsorted, sort/topk,
    kthvalue/median/mode, nanmedian, Tensor Views, Complex Numbers, conj/conj_physical); torch 2.14.0
    aten/src/ATen/native/SharedReduceOps.h; torch 2.14.0 aten/src/ATen/OpMathType.h; torch 2.14.0
    torch/export/decomp_utils.py; torch 2.14.0 torch/_decomp/__init__.py; torch 2.14.0
    torch/export/exported_program.py; design/torch-source-lowering-change-request.md;
    design/compilation-composition-change-request.md
- metal-microbench:provider/litert_mesh.cc: No comment above Unary, Binary, Parsed, Builtin::Lower or Graph::Elementwise cites a paper, standard or document
  (read at 1012, 1074, 1260, 3972, 4017). The file names the MLDrift parser (litert::ml_drift::BuildFinalModel and
  CustomOperationParserFactory, 3999-4001) and TFLite builtin codes (kTfLiteBuiltin*).
  - sources it names: LiteRT / TFLite builtin codes (kTfLiteBuiltin*); litert::ml_drift (BuildFinalModel,
    CustomOperationParserFactory)
- metal-microbench:metal_neural.mm: Comment at 1246-1249 on LowerNeural cites '../dox/mesh/design/compilation-composition-change-request.md item 5: a
  kernel of ANE-placed operations ... as one Core ML function'. Comment at 948-950 on Contraction: 'the Neural
  Engine's matrix operation; no rows x cols x k product tensor'. Refusal messages at 640-657 name 'MIL has no
  bitwise operations'.
  - sources it names: ../dox/mesh/design/compilation-composition-change-request.md item 5; Core ML / MIL (MLMultiArray, CoreML8)

**The same thing computed in more than one place**

- Mapping an operator name to a pointwise operation: mesh_compile.py POINTWISE (2352-2360) and SYMBOLIC (6687-6692)
  map torch operator names (A.gelu, A.sigmoid, A.floor_divide, operator.mod -> 'remainder', and so on) to pointwise
  names; litert_mesh.cc Builtin::Lower static maps unary (4040-4048) and binary (4049-4059) map TFLite builtin codes
  to md::OperationType, then Unary (1012-1071) and Binary (1074-1131) produce METAL_IR operations. Same subproblem
  in two sources: torch operators and TFLite builtins.
- GELU and sigmoid: litert_mesh.cc Unary kGelu and kSigmoid (about 1080-1110) with the GELU formula via Erf;
  mesh_compile.py POINTWISE includes A.gelu and A.sigmoid (2352-2356) whose lowering (_arithmetic/_elementwise) was
  not read.
- Contraction/matrix multiply: mesh_compile.py _contract (2975) via _matrix (3043) and _product (3056); ANE
  Lowering::Contraction (metal_neural.mm:948-960) forms a MIL matmul from the same Contract description;
  litert_mesh.cc Builtin::MatMul (l_.MatMul, 3180s, body not read) and the FullyConnected handler
  (litert_mesh.cc:4800-4812) also lower matrix multiplies.
- Sum/mean reduction: mesh_compile.py _reduced (5246) with _reduction (5228) and _dims (5240); litert_mesh.cc
  Builtin::Reduce (declaration at 3454, body not read) lowering kTfLiteBuiltinSum/Mean (4747-4748).
- Floor division and modulo: mesh_compile.py SYMBOLIC maps operator.floordiv and operator.mod to
  'floordiv'/'remainder' (6687-6692); litert_mesh.cc Binary kFloorDiv and kFloorMod (1074-1131) define float and
  integer forms; the mesh_compile side implementation was not read.
- Logical and/or/xor: mesh_compile.py POINTWISE bitwise_*/logical_* entries (about 2352-2360); litert_mesh.cc Binary
  kLogicalAnd/kLogicalOr/kLogicalXor (1120-1130) choose BOOL vs integer forms by kind; ANE lowering
  (metal_neural.mm:640-651) refuses integer bitwise and shift operations with 'MIL has no bitwise operations'
  messages.

**No function found, or only a refusal**

- No function selects an operator implementation by cost, device or measurement. Each operator in mesh_compile.py
  _lowering (6694-6723) takes the first matching branch of an ordered if-chain; no alternative is tried after a
  branch returns. The only fallback found is litert_mesh.cc Builtin::Parsed returning selected=false (4005) and
  Builtin::Lower continuing to its hand-written tables (4022-4023).
- ANE placement is not selected in metal_neural.mm or mesh_compile.py: place == METAL_IR_ANE is set elsewhere
  (metal_lower.mm neural lambda 2521-2526, read; the setter not read).
- mesh_compile.py _lowering has no branch for a packet in none of SOURCES, VIEWS, LINEAR, SYMBOLIC and the
  attention/determinant/pinv cases beyond the raise at 6722-6723.
- metal_neural.mm Lowering::Lower(uint32_t) beyond line 800 (argument building, the ANE contraction branch,
  reductions, region evaluation) was not read in full; only its Fail refusal lines were captured by grep.
- litert_mesh.cc Builtin::Lower handlers between lines 4070 and 4800 were not read; their 'return false' lines were
  captured by grep only.
- Tanh, Erf, Activate, Builtin::Reduce, Builtin::MatMul, Lowering::MatMul, _elementwise, _arithmetic
  (mesh_compile.py), and the bodies of _fold, _ranked and _extreme were not read, so their refusals are unknown.

**Not determined**

- Whether POINTWISE, SYMBOLIC or SOURCES cover every operator the frontend emits: no enumeration of torch.ops was
  made.
- The contents of the citations to design/torch-source-lowering-change-request.md and
  design/compilation-composition-change-request.md (not opened).
- The enclosing function of the Unary call at litert_mesh.cc:2249, the enclosing function of mesh_compile.py:8494,
  and the setter of place == METAL_IR_ANE.
- Whether Builtin::Lower's unconverted nodes fall back to CPU at run time: litert_mesh.cc:4950-4975 records them and
  returns ENOTSUP; the run-time behaviour beyond that was not traced.

## Composing operators into one function per launch

Read: 11 whole body. 11 functions.

- **`Together`** — metal-microbench:metal_transform.mm:211-217 (whole body) (whole body)
  - does: For every control range c in p.controls, tests two arms [c.first,c.middle) and [c.middle,c.last). Returns false
    if exactly one of operations a, b lies inside an arm (the XOR at line 215); otherwise returns true. Loops,
    comparisons, no allocation.
  - takes: const Program &p, uint32_t a, uint32_t b
  - returns: bool
  - cites: none; comment at lines 207-209 gives rationale only (fusing ops held by different arms or WHILE ranges would
    change how many times or whether an op runs)
  - refuses:
    - metal_transform.mm:215 return false (no message) when a and b are in different arms of the same control range
  - callers: 4 call sites, all in metal_transform.mm: 492 (Inline), 570 (Epilogue), 1117 (Locals/Conflict area), 1392
    (Transform). None elsewhere found.
- **`Fuse`** — metal-microbench:metal_transform.mm:344-444 (whole body, 101 lines) (whole body)
  - does: Producer P's single output must be read through unit-coefficient maps, one distinct loop per output dimension
    with matching bound (351-356), and every producer loop must be covered (359). Composes the consumer's input map
    at slot with the producer's input maps via Compose (357-375; loops of the producer's inputs are rewritten by
    Symbols/Carry/Carried). If apply is false, returns true at 376. If apply is true: Append(C,P), builds an ir
    function named "prologue" computing the producer element from the producer's inputs (382-405, EmitExpr, Element,
    Simplify), rewrites consumer inputs/maps/terms (408-433), copies the producer's absorbed list and adds the
    producer id (439-441), calls Changed, returns true. Consumer input slot becomes C.prologue[slot]=pre.
  - takes: Program &p, uint32_t producer, uint32_t consumer, size_t slot, bool apply
  - returns: bool
  - cites: metal_transform.mm:339-342 "MLIR Linalg ElementwiseOpFusion with the producer kept as the consumer operand's
    prologue" (names MLIR, no paper or year)
  - refuses:
    - metal_transform.mm:354 return false (no message): output map not a unit map, loop repeated, or bound differs
      from value shape
    - metal_transform.mm:359 return false (no message): a producer loop not covered by the output map
    - metal_transform.mm:372 return false (no message): Compose fails (the affine composition is not closed in the
      given box)
  - callers: 2 call sites: metal_transform.mm:521 (Inline, dry run with apply=false), 530 (Inline, apply=true).
- **`Inline`** — metal-microbench:metal_transform.mm:469-539 (whole body) (whole body)
  - does: Rejects producer unless Generic, not erased, has region, one output, inputs, not ACCUMULATE, no states/scatters,
    not Clipped, and all iterators PARALLEL (471-475); rejects if its output is Observable (477). Scans later
    operations o: a write conflict returns false (489); readers are operations reading v (490-509), each must be
    Generic with region, same place, no scatters, Together, not Clipped, read v as input and not as gather
    (492-496); other inputs of the reader on the same storage reject (498); v already used as a prologue or epilogue
    input of the reader rejects (499-507). Requires readers non-empty (510) and no conflicting reads in between
    (511-512). For each reader, dry-runs Fuse (521) and accumulates Evaluations(...) (522); rejects if added *
    Points(v shape) * Work(P) > Points(C.bounds) * Work(C) (524, arithmetic comparison, only when work is nonzero).
    Then calls Fuse with apply=true for each slot of v in each reader (526-531), sets P.erased, Changed, Discards
    P.region and prologues (532-538).
  - takes: Program &p, uint32_t producer
  - returns: bool (true when producer was inlined and erased)
  - cites: metal_transform.mm:466-468 "S. Williams, A. Waterman and D. Patterson, CACM 52(4), 2009: what it removes is a
    launch and the intermediate's memory traffic"
  - refuses:
    - metal_transform.mm:471-473 return false (no message): not Generic, erased, no region, outputs != 1, no inputs,
      ACCUMULATE start, states, scatters, Clipped
    - metal_transform.mm:475 return false: an iterator not PARALLEL
    - metal_transform.mm:477 return false: output Observable
    - metal_transform.mm:489 return false: a later operation has a write conflict with the producer's written view
    - metal_transform.mm:492-496 return false: reader not Generic, no region, place differs, scatters, not Together,
      Clipped, does not take v, or reads v as gather
    - metal_transform.mm:498 return false: another input of the reader on the same storage
    - metal_transform.mm:500-507 return false: reader's v input already has a prologue, prologueInputs, or
      epilogueInputs use
    - metal_transform.mm:510 return false: no readers
    - metal_transform.mm:512 return false: an intervening operation conflicts with the reads
    - metal_transform.mm:521 return false: Fuse dry run fails
    - metal_transform.mm:524 return false: recomputation cost exceeds the readers' own cost (no message)
  - callers: 1 call site: metal_transform.mm:1374 in Transform, inside the loop over all operations.
- **`Epilogue`** — metal-microbench:metal_transform.mm:547-758 (whole body, 212 lines, read in four segments) (whole body)
  - does: For each output j of producer P that is not Observable and not ACCUMULATE (551-553): finds the single later
    non-erased operation that reads it (558-566; more than one reader sets single to false). Requires reader C:
    Generic, region, same place, one output, no states, Together, not Clipped, no scatters, no existing epilogue,
    combiner ASSIGN, start IDENTITY, y in inputs exactly once, not a gather, all iterators PARALLEL (569-576).
    Rejects if the reader's other inputs alias y's storage or y's slot has a prologue (577-585). Checks y is read
    through a permutation onto the reader's domain (586-595). Checks that moving the reader up to the producer does
    not conflict with intervening operations (597-606). Composes the reader's output map over the producer's loops
    (607-623) and the reader's other needed inputs (634-648). Builds an ir function named "epilogue" (654-657+) that
    takes the producer contribution (and any prior epilogue) and the reader's other inputs, emits the reader's
    region and stores the result (Stored, Spliced, Return) (658-747), then rewrites P's outputs[j]=C.outputs[0],
    maps, terms, inputs, prologueInputs, epilogue[j]=post, epilogueInputs[j], gathers, precision, absorbed list
    (748-752 region), sets C.erased, Changed, discards C's region and prologues, returns true (753-757). Loops while
    it returns true (Transform 1375).
  - takes: Program &p, uint32_t producer
  - returns: bool (true when one output's reader was fused into the producer)
  - cites: metal_transform.mm:541-546 "MLIR Linalg producer-consumer fusion with the consumer as the producer's output
    epilogue (the old MetalFuseEpilogue's register epilogue)" (names MLIR, no paper or year)
  - refuses:
    - metal_transform.mm:549 return false (no message): producer not Generic, erased, no region, scatters, or
      Clipped
    - metal_transform.mm:553 continue (not return): output Observable or ACCUMULATE
    - metal_transform.mm:562-567 continue: more than one reader, or reader not unique
    - metal_transform.mm:569-575 continue: reader kind, region, place, outputs!=1, states, Together, Clipped,
      scatters, existing epilogue, combiner, start, y count in inputs, gather
    - metal_transform.mm:576 continue: a reader iterator not PARALLEL (producer iterators are not checked in this
      function)
    - metal_transform.mm:580-585 continue: aliased storage or prologue on y's slot
    - metal_transform.mm:595 continue: y not read through a permutation onto reader domain
    - metal_transform.mm:606 continue: move not legal past intervening operations
    - metal_transform.mm:620 and 648 continue: Compose not closed
    - metal_transform.mm:758 return false when no output is fusable
  - callers: 1 call site: metal_transform.mm:1375 in Transform, as while (Epilogue(p,o)) {} for each operation not inlined.
- **`Groups`** — metal-microbench:metal_transform.mm:761-765 (whole body) (whole body)
  - does: Sets groups[0..2]=1; for each loop l below min(op.bounds.size(), m.axis.size()) with m.axis[l]<3, multiplies
    groups[m.axis[l]] by ceil(op.bounds[l]/m.tile[l]) computed as (bounds+tile-1)/tile. No check on m.tile[l] being
    nonzero in the body.
  - takes: const Operation &op, const Mapping &m, uint64_t groups[3]
  - returns: void (fills groups[3])
  - cites: metal_transform.mm:760 comment "(4's Map: loops sharing an axis linearize row-major)" (refers to an unnamed
    source "4"; no external citation)
  - refuses:
    - none (no return value, no error path)
  - callers: 8 call sites in metal_transform.mm: 785 (Partitioned), 899, 1001 (Time), 1049 and 1050 (Joins), 1155, 1245,
    1442.
- **`Profitable`** — metal-microbench:metal_transform.mm:1011-1029 (whole body) (whole body)
  - does: Collects the prior members' mappings and the max threads of the group (1013-1018). If the candidate mapping m
    has the same axis and tile as own and m.threads <= group threads, returns true (1019). Returns false if
    p.machines is null (1020). Otherwise computes combined = Time(joined group+op under together) and apart =
    Time(group under prior) + Time(op under own), each with a per-call complete flag (1021-1027), returns a && b &&
    c && combined <= apart (1028).
  - takes: const Program &p, const std::vector<uint32_t> &group, const std::vector<Mapping> &mapping, uint32_t op, const
    Mapping &own, const Mapping &m
  - returns: bool
  - cites: metal_transform.mm:1010 "docs/provider.md#composed-resource-costs"; that section (docs/provider.md:585-610)
    states the cost DAG and "Structural fusion with unchanged member mappings and thread geometry remains justified
    independently of machine measurements; a changed mapping needs a complete cost comparison" (lines 608-609)
  - refuses:
    - metal_transform.mm:1020 return false (no message) when p.machines is null
    - metal_transform.mm:1028 return false when any Time call was incomplete (ENODATA-type lower bound)
  - callers: 2 call sites in Transform: metal_transform.mm:1394, 1401.
- **`Joins`** — metal-microbench:metal_transform.mm:1034-1057 (whole body) (whole body)
  - does: Sums threads as max and memory and extraBindings as sums over group members plus candidate m (1036-1043).
    Rejects if gpu and memory > p.memory, or m.threads != threads (1045). Rejects if any member's threads differ
    (1046-1047). Compares Groups of op o under m with Groups of group[0] under its mapping, per axis (1048-1051).
    Requires Tiled(p, member, mapping[member], o, m) for every member (1052-1053). Returns !gpu ||
    Bindings(p,members)+TransportBindings(p,members)+extraBindings <= kBindings (1054-1056).
  - takes: const Program &p, const std::vector<uint32_t> &group, const std::vector<Mapping> &mapping, uint32_t o, const
    Mapping &m
  - returns: bool
  - cites: metal_transform.mm:1031-1033 cites "4's Bindings" and "the transport's TransportBindings, kBindings" (no
    external paper, "4" unnamed)
  - refuses:
    - metal_transform.mm:1045 return false (no message): GPU memory over p.memory, or thread mismatch
    - metal_transform.mm:1047 return false: member threads differ
    - metal_transform.mm:1051 return false: threadgroup grids differ
    - metal_transform.mm:1053 return false: Tiled fails for a member
    - metal_transform.mm:1056 return false: binding table exceeds kBindings
  - callers: 2 call sites in Transform: metal_transform.mm:1394, 1400.
- **`Time`** — metal-microbench:metal_transform.mm:989-1008 (whole body; related helper used by Profitable) (whole body)
  - does: Selects machine p.cpu or p.gpu by place of members[0] (991). Calls mesh_graph_actor_time on a launch record with
    launches (992-994). Computes Locals for the members (996), then for each member Groups and Actor (1001-1002),
    calls mesh_graph_actor_time on that actor and sums into time (1003-1005). Returns time; *complete is ANDed with
    each call's success.
  - takes: const Program &p, const std::vector<uint32_t> &members, const std::vector<Mapping> &mappings, uint64_t launches,
    bool *complete
  - returns: double
  - cites: metal_transform.mm:985-988 "OpenXLA's GPU performance model, Roofline" (no paper citation details beyond the
    names)
  - refuses:
    - none in body; failures are reported only through *complete
  - callers: 2 call sites: metal_transform.mm:1026, 1027 (Profitable).
- **`_epilogue`** — mesh:rdma/mesh_compile.py:2928-2933 (whole body) (whole body)
  - does: Creates a _Region r; loads the accumulator acc at input 0 (_load), loads each extra (value, _) at input 1+j;
    stores apply(r, x, *loaded) if apply else x into out (_store, r.done); returns a one-element list
    [_Generic(out.shape, [(acc, _same)] + extra, [(out, _same)], r)]. It emits a structured generic; it does not
    itself fuse. Fusion happens in metal_transform.mm Epilogue (per its comment at line 2926-2927).
  - takes: out, acc, t, extra=(), apply=None
  - returns: list of one _Generic
  - cites: mesh_compile.py:2926-2927 "an output kept in its arithmetic kind: the structured map out = convert(f(acc,
    operands...)), fused by the transformation with its producer" (no external citation)
  - refuses:
    - none (no raise, no assert)
  - callers: 7 call sites in mesh_compile.py: 3006, 4813, 4970, 5101, 5256, 5371, 6236. A separate function named
    _epilogue(args) exists at
    dox/mesh/rdma/torch-mesh/build/lib.macosx-11.0-arm64-cpython-312/torch_mesh/streamed.py:1618 (not read; in a
    build directory).
- **`mesh_graph_contract`** — mesh:rdma/mesh-graph.c:714-741 (whole body, 28 lines); declaration mesh-graph.h:243-244 (whole body)
  - does: Rejects any edge endpoint >= vertices (716). Computes keeps = count of kept[v] != 0 (718-719). Builds source[]
    (kept vertices) and target[] (vertices+v) (725-726). Calls mesh_graph_rows (729) to order edges by tail, then
    mesh_graph_lags over 2*vertices with contract_head (kept targets map to vertices+to) and contract_lag (edge
    delay) to fill lags[keeps*keeps] (730). Writes each lags[p] != UINT64_MAX as {source[p/keeps], source[p%keeps],
    lags[p]} into contracted up to room (733-737), counts all such in *made (738), returns ERANGE if count > room.
  - takes: uint32_t vertices, size_t edges, const struct mesh_graph_dependence *edge, const unsigned char *kept, size_t
    room, struct mesh_graph_dependence *contracted, size_t *made
  - returns: int status (0 or errno-style)
  - cites: mesh-graph.h:236-242 (no literature citation); the comment names metal_recording.m MetalPlannedOrder for
    unrolled copies. No paper cited.
  - refuses:
    - mesh-graph.c:716 return EINVAL: edge endpoint out of range
    - mesh-graph.c:717 return ENOMEM: vertices >= UINT32_MAX/2 or edges too large
    - mesh-graph.c:720 return ENOMEM: keeps squared overflow
    - mesh-graph.c:724 result ENOMEM on allocation failure
    - mesh-graph.c:738 return ERANGE: count > room
  - callers: 1 call site: metal-microbench:metal_recording.m:2815 (in an unrolled-plan contraction). Not called from
    mesh-graph.c itself.
- **`mesh_graph_compose`** — mesh:rdma/mesh-graph.c:2660-2881 (whole body, read across segments to the closing brace); declaration mesh-graph.h:314-315 (whole body)
  - does: Composes two synchronization signatures a and b into one signature s. Allocates arrays (2662-2680), unites
    processors by identity via compose_processor (2679-2680), concatenates tasks with b's task indices shifted by
    a->tasks (2681-2685) and dependences shifted (2686-2691), matches b's BRIDGE_START operations against a's
    (2693-2705, qsort and graph_lower), splices a's operations before each matched point and b's operation
    (2706-2725), remaps synchronized edges (2733-2745), then composes controls, events, scope, spans, selections,
    guards, choices, feedbacks, submissions, actions, conditions and instructions (2746-2880 in segments), and
    finally calls signature_derive(s) (2876). This composes program signatures (sync graph and scheduling records),
    not operator bodies.
  - takes: const struct mesh_graph_signature *a, const struct mesh_graph_signature *b, struct mesh_graph_signature *s
  - returns: int status (0 or errno-style); on error the signature is freed (2878)
  - cites: mesh-graph.h:309-313 "design/synchronization-signature-change-request.md item 1; L. Lamport, CACM 21(7), 1978:
    one program's signature from two producers' (a: the plan's, b: the replay's)"; metal_recording.m:4432 cites
    "design/compilation-composition-change-request.md item 6.2"
  - refuses:
    - mesh-graph.c:2663 return ENOMEM: a->tasks + b->tasks >= UINT32_MAX
    - mesh-graph.c:2677-2678 result ENOMEM on any allocation failure
    - mesh-graph.c:2750, 2768, 2787 ENOMEM on allocation failure of control/span/selection arrays
    - mesh-graph.c:2769 result EINVAL: a->spans and b->spans both nonzero and differ
    - mesh-graph.c:2775 result EINVAL: a span's first or count differs from b's at the same index
    - mesh-graph.c:2788 result EINVAL: both selections and choices nonzero
  - callers: 1 call site: metal-microbench:metal_recording.m:4562.

**Documents**

- metal-microbench:docs/provider.md: Section "## Composed resource costs" (lines 585-610). Quoted: line 587 "The compiler retains a cost DAG alongside
  its selected construction."; line 596 "This refines the existing [OpenXLA GPU performance model](...)"; lines
  608-609 "Structural fusion with unchanged member mappings and thread geometry remains justified independently of
  machine measurements; a changed mapping needs a complete cost comparison." External sources named: OpenXLA GPU
  performance model (github.com/openxla/xla, gpu_performance_model_base.cc). It specifies the cost DAG, machine
  rates and the fusion justification used by Profitable (metal_transform.mm:1010). It does not name any
  operator-fusion paper.
  - sources it names: OpenXLA GPU performance model
    (github.com/openxla/xla/blob/main/xla/service/gpu/model/gpu_performance_model_base.cc)
- mesh:rdma/mesh-graph.h: Line 236-242 (contract comment): "For kept u and w, one output {u, w, d} where a walk u -> ... -> w of at least
  one edge has every interior vertex contracted, d the least sum of its delays"; lines 241-242 name
  metal_recording.m MetalPlannedOrder for unrolled copies. Lines 309-313 (compose comment): "one program's signature
  from two producers' (a: the plan's, b: the replay's)"; a native BRIDGE_START operation both list is b's. Lines
  316-324 cite Bhattacharyya, Sriram, Lee (IEEE Trans. Signal Processing 45(6), 1997) for the resynchronization
  function at line 325, not for compose or contract.
  - sources it names: L. Lamport, CACM 21(7), 1978 (line 309 comment); design/synchronization-signature-change-request.md item 1 (line
    309); metal_recording.m MetalPlannedOrder (line 241-242)
- mesh:rdma/mesh_compile.py: Lines 2926-2927: "an output kept in its arithmetic kind: the structured map out = convert(f(acc, operands...)),
  fused by the transformation with its producer". Lines 2973-2974 cite PyTorch 2.14 torch.einsum documentation for
  _contract (not for _epilogue).
  - sources it names: PyTorch 2.14 documentation, torch.einsum (line 2973, for _contract)
- metal-microbench:metal_transform.mm: Lines 339-342: MLIR Linalg ElementwiseOpFusion (fuse as prologue). Lines 466-468: S. Williams, A. Waterman and D.
  Patterson, CACM 52(4), 2009 (Inline's recomputation admission). Lines 541-546: MLIR Linalg producer-consumer
  fusion (Epilogue). Lines 985-988: OpenXLA GPU performance model, Roofline (Time). Line 1010:
  docs/provider.md#composed-resource-costs (Profitable). Lines 1031-1033: "4's Bindings" and transport
  TransportBindings (Joins). Lines 207-209: Together rationale, no citation. Line 1-2 cite
  ../dox/mesh/design/compilation-composition-change-request.md worker 3 (file header, not one of the named
  functions).
  - sources it names: MLIR Linalg ElementwiseOpFusion; S. Williams, A. Waterman and D. Patterson, CACM 52(4), 2009; MLIR Linalg
    producer-consumer fusion; OpenXLA GPU performance model; Roofline; docs/provider.md#composed-resource-costs

**The same thing computed in more than one place**

- Operand-map composition loop (Compose over each producer/consumer map dimension, pushing composed expressions into
  `composed`/`added`/`fed` vectors): metal_transform.mm:368-375 (Fuse), 617-623 (Epilogue output map), 641-646
  (Epilogue fed inputs). Same operation: substituting one op's loop index expressions through another's maps.
- Dry run then apply of the same fusion: metal_transform.mm:521 (Fuse apply=false inside Inline's reader loop) and
  metal_transform.mm:530 (Fuse apply=true) both compute the same composition for the same reader slot.
- Reader search over later operations (write-conflict check plus 'reads v as input' test):
  metal_transform.mm:485-509 (Inline) and 558-566 (Epilogue). Same scan, different accept conditions.
- Per-operation operand list construction Operands(p, op): called at metal_transform.mm:348, 517, 586 for the same
  purpose (operand index to value and map offset).
- Group geometry: Groups (metal_transform.mm:761) is re-implemented inline in Joins at 1048-1051 via two Groups
  calls compared per axis; Partitioned (780-785) also uses Groups. Same threadgroup-count computation, one
  definition plus per-caller comparisons.
- Epilogue sequence in two repositories: mesh_compile.py:2928 (_epilogue, builds the structured reader generic) and
  metal_transform.mm:547 (Epilogue, fuses it into the producer). The python side emits the form the C++ side later
  fuses; same subproblem split across files, not duplicated code. A third _epilogue(args) exists at
  dox/mesh/rdma/torch-mesh/build/lib.macosx-11.0-arm64-cpython-312/torch_mesh/streamed.py:1618 (not read).

**No function found, or only a refusal**

- No function found that fuses one operator into a consumer's region body for non-Generic operations (reductions
  with combiners other than ASSIGN): Epilogue requires combiner[0]==METAL_IR_ASSIGN (metal_transform.mm:572) and
  Inline requires no states and PARALLEL iterators (metal_transform.mm:472, 475); the only path for other forms is
  the refusal.
- No function fuses more than one producer into a single consumer in one call; Transform applies Inline/Epilogue per
  operation in order (metal_transform.mm:1373-1375).
- No fusion across different places (GPU vs CPU): refused at metal_transform.mm:492 and 569 (place mismatch).
- No fusion across scatters, gathers of the fused value, accumulate, or state-carrying operations: refused at
  metal_transform.mm:472, 492-495, 553, 569-574.
- No iterator check on the producer in Epilogue: metal_transform.mm:547-758 checks iterators only for the reader
  (576); Inline checks producer iterators (474-475).
- mesh-graph.c contains no operator-fusion function: mesh_graph_compose (mesh-graph.c:2660) composes synchronization
  signatures and mesh_graph_contract (mesh-graph.c:714) contracts dependence edges; neither composes operator
  bodies.
- Python side: _epilogue (mesh_compile.py:2928) only builds a structured generic; no Python function in
  mesh_compile.py was found that fuses two generics into one.

**Not determined**

- Bodies of helpers called but not read: Compose, Symbols, Carry, Carried, Append, EmitExpr, Element, Simplify,
  Stored, Gathered, Spliced, Discard, Changed, Observable, Accesses, Conflict, Clipped, Arithmetic, Work,
  Evaluations (read only at call site, metal_transform.mm:449-464: read whole), Tiled, Bindings, TransportBindings,
  Map, Follow, Actor, Locals, mesh_graph_actor_time, mesh_graph_lags, mesh_graph_rows, compose_processor
  (mesh-graph.c:2646), signature_derive, graph_lower.
- Whether Epilogue's reader-chain order ever produces an output requiring both prologue and epilogue on one buffer:
  depends on Compose closure and Conflict, not read.
- Transform (metal_transform.mm:1361) body beyond line 1409 was not read; the kernel-group construction after 1409
  is unread.
- provider/litert_mesh.cc:3217 has a function named Inline (inlining control-flow subgraphs into one graph); it is a
  different subproblem and was not read.
- mesh-graph.h:229-235 (start of contract comment) and mesh-graph.h:243 comment fully: only 236-242 were read.
- Caller search covered mesh:rdma, dox/mesh/bin, metal-microbench/metal_transform.mm, tools, provider,
  metal_recording.m, metal_lower.mm, metal_program.h, and server (excluding .build, third_party, .venv). A full
  search of the metal-microbench top level for _epilogue and of all of dox/mesh was not completed (search timed
  out); top-level callers of Fuse/Inline/Groups etc. in other .mm/.m files were not exhaustively confirmed.

## Loop and index analysis

Read: 16 whole body, 2 part of the body, 1 declaration only. 19 functions.

- **`_extent`** — mesh:rdma/mesh_compile.py:752-765 (whole body)
  - does: Two branches on window.transposed. None: starts[q] = q*o*window.stride - window.low for q in range(parts);
    length = (o-1)*window.stride + window.extent; after = [0]*parts. Else: (padding, reach) = window.transposed, s =
    window.stride; starts[q] = (q*o + padding - reach)//s; after[q] = q*o + padding - a*s where a = starts[q];
    length = max of ((q+1)*o-1+padding)//s - a + 1 over q, -(-(f+o-1-reach)//s)+1 over f in after, and 1. Returns
    (starts, length, after, left, right) with left = max(q*n - starts[q], 0) and right = max(starts[q] + length -
    (q+1)*n, 0), where n is the shard extent argument.
  - takes: window (fields transposed, stride, low, extent), parts (shard count), n (per-shard extent), o (per-shard output
    extent).
  - returns: 5-tuple (starts list, length int, after list, left int, right int).
  - cites: mesh_compile.py:751 "# GSPMD Appendix A.2 and Figure 9b"
  - refuses:
    - none: no raise, assert or error return in lines 752-765
  - callers: 1 call site: mesh_compile.py:772 (inside _region). No other caller found in dox/mesh, metal-microbench/tools,
    provider, bin.
- **`_region`** — mesh:rdma/mesh_compile.py:768-809 (whole body)
  - does: Calls _extent (line 772). When whole, sets left = max(-a) and right = max(a+length-size) over starts. Builds
    region key (node, key[1], 'region', consumer, dim) with shape length along dim. fill = window.fill if any
    start<0 or start+length>size else None. offsets[q] = starts[q] - (0 if whole else q*n) + left. If left or right
    nonzero: emits a 'copy' of key into a 'middle' value. For each side (left, right) and each j in
    1..ceil(width/n): emits a 'narrow' of middle to a 'halo' piece of width min(n, width-(j-1)*n) and a
    'collective-permute' with pairs (p, p+j) for left or (p+j, p) for right over range(parts-j) into an 'arrival'
    value; reads is the arrivals then key. If left or right: emits a 'landed' value of width left+n+right, records
    ir.added and ir.inside maps for each read piece and for the region; finally emits a 'window' item (reads,
    region, (dim, m, offsets, length, starts, size, fill)). Returns (region, after).
  - takes: ir, consumer, key (value key with key[0] node and key[1] placement), window (dim, fill, label), m (mesh axis
    index), o (shard output extent), whole (bool).
  - returns: (region key, after) where after is the _extent after list.
  - cites: mesh_compile.py:767 "# GSPMD sections 4.1, 4.3 and 4.5, Appendix A.2, Figures 5a and 9b;
    design/aot-compilation-change-request.md, appendix item 7; design/plan-to-bridge-change-request.md R1; MPICH
    allgather_intra_ring.c, MPIR_Localcopy"
  - refuses:
    - none: no raise, assert or error return in lines 768-809. When whole is true the halo loop runs over an empty
      tuple (line 780 'for side, width in () if whole else ...'), so no halo pieces are emitted.
  - callers: 1 call site: mesh_compile.py:867 (inside _partition).
- **`_partition`** — mesh:rdma/mesh_compile.py:811-912 (whole body)
  - does: Walks ir.order. placeholder/get_attr: emits 'input' or 'constant' with whole sharding (-1 per dim) and reshards
    to the declared sharding. output: reshards each input to unsharded and emits 'output'. Operator with rule.labels
    None: reshards operands and emits 'call'. Otherwise: target sharding per output label (-1 for labels in free
    set); partial = mesh axes of contracted labels held by exactly one operand (not fused, first only); local =
    target with partial axes set to -1; wants per operand from _carry; for each window whose label is local and
    whose operand is an input/constant or not already sharded identically, sets wants[operand][dim] = -1 (whole);
    reshards operands; for each window dim calls _region (line 867) with the sharded extent; transposed windows
    append to cuts; attention annotation (AttentionOrigin) computed from coords and extents; emits 'call'
    (optionally a merged call); emits trims (ir.trims) for contracted labels whose operand extent is not divisible
    by the mesh size; emits 'window' for cuts; then reduce-scatter if one partial axis is in used, all-reduce if
    partial and not used; finally reshards to the declared sharding if target differs.
  - takes: ir, operators (node -> rule with operands, out, labels, contracting, windows, merged, configure, kind, fused).
  - returns: None (emits into ir).
  - cites: mesh_compile.py:810 "# GSPMD sections 2.1, 3, 3.2, 4, 4.2, 4.5 and 5.1, Figures 6 and 7;
    design/aot-compilation-change-request.md A4 and appendix item 7; XLA spmd_partitioner.cc,
    PartitionedHlo::ReshardAsWindowedInput"
  - refuses:
    - none: no raise, assert or NotImplemented in lines 811-912
  - callers: 1 call site: mesh_compile.py:953 (inside partition, def at line 914).
- **`_paged`** — mesh:rdma/mesh_compile.py:1149-1172 (whole body)
  - does: Returns a bool. Returns False if reads is empty, if any key or read's meta val is not a torch.Tensor, if
    places[key] exists with a different owner, or if places has no entry for key and the op is not
    scaled_dot_product_attention (lines 1151-1154). Then: REDUCTIONS -> _whole(reads[0]); SOFTMAX -> same layout as
    reads[0] and _whole(reads[0]); scaled_dot_product_attention -> checks nk % ks[-1] == 0, nv % vs[-1] == 0,
    nk//ks[-1] == nv//vs[-1] (nk, nv = page // itemsize) and _whole for reads[1] and reads[2]; other packets not in
    POINTWISE, copy, or COPIES -> False (line 1166); for each operand, itemsize must equal dtype.itemsize and
    element count (COPIES) or shape (others) must match, else False (line 1170); finally all _whole(key and reads).
  - takes: ir, places, items, key, reads (list of value keys), local (the operation's local target with overloadpacket).
  - returns: bool: True means the call is classed as paged.
  - cites: none found at the def (line 1148 not read)
  - refuses:
    - mesh_compile.py:1154 return False (non-tensor operand, owner mismatch, or no place for a non-SDPA op)
    - mesh_compile.py:1166 return False (operation packet not in POINTWISE, copy, or COPIES)
    - mesh_compile.py:1170 return False (itemsize or size mismatch across operands)
  - callers: 1 call site: mesh_compile.py:1178 (inside _kinds, def at 1174). _kinds is called once at mesh_compile.py:969
    (inside partition).
- **`mesh_graph_dependences`** — mesh:rdma/mesh-graph.c:1572-1607 (whole body)
  - does: Checks supplied edges (from < to), qsorts access[] by (key, start, vertex). For every pair i<j with same key and
    access[j].start < access[i].end (sorted sweep), skips same vertex, non-conflicting pairs (access_conflict:
    writes, excluding write==2 with write==2), and empty j ranges; appends an unordered vertex pair packed in
    uint64. Grows buffer by doubling with realloc. qsorts pairs by completion_edge_before, dedups adjacent, writes
    up to room entries to from/to, sets *edges to the distinct count.
  - takes: accesses count, access array (sorted in place), supplied edges (from, to arrays), room, output from/to arrays,
    edges out-param.
  - returns: 0, or EINVAL/ENOMEM/ERANGE.
  - cites: mesh-graph.h:147-150 (comment: "design/compilation-composition-change-request.md item 6.2: the relation is the
    supplied edges ... united with the accesses' conflicts ... sorted by (from, to)") and mesh-graph.c:1562 "/*
    design/torch-source-lowering-change-request.md item 3 */" above access_conflict
  - refuses:
    - mesh-graph.c:1575 EINVAL when supplied_from[e] >= supplied_to[e]
    - mesh-graph.c:1576 ENOMEM when supplied >= SIZE_MAX/sizeof(uint64_t)-64
    - mesh-graph.c:1580 and 1589 ENOMEM on malloc/realloc failure
    - mesh-graph.c:1601 ERANGE when *edges (count) > room
  - callers: none found: mesh_graph_dependences appears only at mesh-graph.h:151 and mesh-graph.c:1572 (and a comment at
    mesh-graph.h:190).
- **`mesh_graph_dependences_views`** — mesh:rdma/mesh-graph.c:1737-1750 (whole body)
  - does: Calls view_conflicts (self=0) into a conflict array, then copies the first min(edges, room) records to from[],
    to[], and cause[] (key, kind, given). Sets ERANGE when *edges > room. Frees the array.
  - takes: views array (struct mesh_graph_view), supplied edges, room, output arrays from/to/cause, edges out-param.
  - returns: 0, or the error from view_conflicts, or ERANGE.
  - cites: mesh-graph.h:161-170 comment citing S. Verdoolaege, isl (ICMS 2010, LNCS 6327) and
    design/compilation-composition-change-request.md, "Complete replacement workgroup"; header comment at
    mesh-graph.h:155 cites MPI 4.1 section 6.1.3 and torch c10 TensorImpl.h
  - refuses:
    - mesh-graph.c:1746 ERANGE when *edges > room
  - callers: 4 call sites in 2 files: metal_ir.mm:772 (Conflict), metal_recording.m:2755 (MetalHostSide, room 1),
    metal_recording.m:3065 and 3071 (row dependence construction).
- **`mesh_graph_conflicts_views`** — mesh:rdma/mesh-graph.c:1731-1734 (whole body)
  - does: Calls view_conflicts with self=1 (self-conflicts included) and returns its status; conflict array and count out.
  - takes: views, supplied edges, out conflict (caller frees), count.
  - returns: 0 or error from view_conflicts.
  - cites: mesh-graph.h:177-179 "design/compilation-composition-change-request.md: retained exact intersections, including
    self conflicts"
  - refuses:
    - none directly; errors come from view_conflicts (mesh-graph.c:1674 EINVAL, 1675 ENOMEM)
  - callers: 1 call site: metal_ir.mm:1755 (Relation).
- **`view_conflicts (static)`** — mesh:rdma/mesh-graph.c:1670-1729 (whole body)
  - does: Validates supplied edges; builds view_hull array (key, start=offset, end from view_hull_end or 2^126 when that
    fails) and an isl context; qsorts hulls by (key, start); for pairs i<j (j from i+!self) with same key and hull
    start < hull end: skips same-vertex pairs when not self, pairs with no writes, both-write-2 pairs, and pairs
    with a zero element or zero-count dimension; calls view_meets (exact isl test) unless same view; for a meeting
    pair records edges with kind READ_AFTER_WRITE / WRITE_AFTER_READ / WRITE_AFTER_WRITE by lower vertex's write
    flag, and two directions when same vertex and not WAW; qsorts edges by view_edge_before and dedups.
  - takes: views, supplied edges, self flag, out conflict array, out count.
  - returns: 0, or EINVAL/ENOMEM/-ENOMEM.
  - cites: mesh-graph.c:1606 "/* S. Verdoolaege, isl: An Integer Set Library for the Polyhedral Model, ICMS 2010, LNCS 6327
    (isl 0.27, its exact integer arithmetic imath): a view's dimensions [0, dims) at `offset` */"
  - refuses:
    - mesh-graph.c:1674 EINVAL when supplied_from[e] >= supplied_to[e]
    - mesh-graph.c:1675 ENOMEM when supplied or views too large
    - mesh-graph.c:1679 and 1690 ENOMEM (malloc or isl_ctx_alloc failure)
    - mesh-graph.c:1699 result = -meets on view_meets failure (ENOMEM from isl)
    - mesh-graph.c:1711 and 1713 ENOMEM growth failure
  - callers: called by mesh_graph_conflicts_views (mesh-graph.c:1734) and mesh_graph_dependences_views (mesh-graph.c:1739).
- **`view_meets (static)`** — mesh:rdma/mesh-graph.c:1626-1655 (whole body)
  - does: Returns 0 immediately if both hulls are computed and the offsets are disjoint (a.offset >= eb or b.offset >=
    ea). Otherwise builds an isl basic set over n = a.dims + b.dims + 2 integer variables: one equality a.offset -
    b.offset + sum_k i_k a.stride_k - sum_l j_l b.stride_l + u - v = 0 (u = a.element variable, v = b.element
    variable), bounds 0 <= x <= size-1 for each index and element variable, then isl_basic_set_is_empty. Returns 1
    if not empty, 0 if empty, -ENOMEM on isl error.
  - takes: isl ctx, two view_part structs (offset, element, pairs {size, stride}, dims).
  - returns: 1 meets, 0 disjoint, negative errno.
  - cites: mesh-graph.c:1606-1608 (isl, Verdoolaege) and mesh-graph.h:161-162
  - refuses:
    - mesh-graph.c:1654 returns -ENOMEM when isl_basic_set_is_empty returns isl_bool_error
  - callers: called from view_conflicts only.
- **`view_hull_end (static)`** — mesh:rdma/mesh-graph.c:1611-1620 (whole body)
  - does: Computes w = element + sum over dims with pairs[2d] != 0 of (pairs[2d]-1)*pairs[2d+1] using __int128 with
    overflow checks; end = offset + w. Returns 0 on any overflow (no hull), 1 otherwise.
  - takes: view_part.
  - returns: int status; *end set on success.
  - cites: mesh-graph.c:1606-1608 comment (isl) only
  - refuses:
    - mesh-graph.c:1615 returns 0 on mul or add overflow (caller then uses end = 2^126 at mesh-graph.c:1686, no
      pruning)
  - callers: called from view_meets (mesh-graph.c:1628) and view_conflicts (mesh-graph.c:1685).
- **`mesh_graph_intersect`** — mesh:rdma/mesh-graph.c:3126-3132 (whole body)
  - does: qsorts interval[] by (key, start) ascending, then returns 1 if any adjacent pair has the same key and
    interval[k].start < interval[k-1].end (half-open overlap of sorted neighbors), else 0.
  - takes: count, array of struct mesh_graph_interval {key, start, end} (sorted in place).
  - returns: 1 if any two intervals with the same key overlap, 0 otherwise.
  - cites: mesh-graph.h:337 "/* MPI 4.1 section 6.1.11 */"
  - refuses:
    - none: no error return; result 1 means overlap, which the caller maps to EINVAL
  - callers: 1 call site: mesh-collective.c:1475 (const int status = mesh_graph_intersect(count,interval)?EINVAL:0).
- **`mesh_graph_arcs_overlap`** — mesh:rdma/mesh-graph.c:3135-3141 (whole body)
  - does: Returns overlap of two arcs on a ring of 64-bit positions; an arc with end <= start wraps. Both wrap: 1. One
    wraps: end of wrapping arc > start of other or other's end > its start. Neither wraps: a_end > b_start && b_end
    > a_start.
  - takes: a_start, a_end, b_start, b_end (uint64).
  - returns: int 1 overlap, 0 not.
  - cites: mesh-graph.c:3133-3134 "/* GaijinEntertainment/DagorEngine 2baf1947,
    prog/gameLibs/render/daFrameGraph/backend/resourceScheduling/heapLayout.cpp:13-23, do_arcs_overlap */"
  - refuses:
    - none
  - callers: 1 call site: mesh-collective.c:556 (plan_intersect).
- **`Footprints (static)`** — metal-microbench:metal_ir.mm:946-1059 (whole body)
  - does: For each kernel member mapping of a Generic op (not erased): box.hi[l] = min(tile, bound)-1 for axes < 3; full =
    all bounds divisible by tile; tiles[l] = ceil(bound/tile); inner[l] = product of later tile counts on same axis.
    For each operand j (inputs then outputs) not Internal, finds the non-PAGES binding with the same storage; if
    none, skip. Creates mesh_graph_footprint f with exact 0 and whole-binding extent. If the operand is separable
    (not gathered, no joint composition): builds mesh_graph_term entries from Atoms (Fold, tile/divisor arithmetic),
    setting separable=false on non-divisible atoms or bit-misalignment. If separable: ByteViews of
    ElementViews(Compose(...)) over box; empty views give exact=1, extent=0; else computes spans, contiguity, start
    = min span start - binding offset, extent = max end - min start, exact = full && contiguous && word-aligned (for
    Words scalars). Appends terms to k.footprintTerms and f.terms. Then Accesses(p, op) views for storages not yet
    covered get whole-binding footprints (exact 0). Then k.added accesses and control predicates (controlOwners,
    access bit 1) produce exact=1 footprints with start = access start - binding offset.
  - takes: Program &p, kernel index.
  - returns: void; appends to k.footprints and k.footprintTerms.
  - cites: metal_ir.mm:938-945 "// J. Lee, Ph.D. dissertation, University of Michigan, 2015, section 3.2.2 (mesh-graph.h
    struct mesh_graph_footprint): the bytes threadgroup g of a kernel's mapped operation touches through each
    binding ... `exact` needs full tiles, separable atoms, word-aligned packed kinds and one contiguous run;
    otherwise the record bounds every group by the whole binding."
  - refuses:
    - none: no raise or assert. Non-separable or non-exact operands are not rejected; they get exact=0 and the whole
      binding (metal_ir.mm:944-945 comment; default at 978).
  - callers: 1 call site: metal_ir.mm:1998 (inside Artifact, def at 1982).
- **`Effects (both overloads) and OperandEffects (static)`** — metal-microbench:metal_ir.mm:890-927 (OperandEffects), 929-936 (Effects) (whole body)
  - does: OperandEffects: for each operand (inputs then outputs) not Internal: if input and not gathered and not Loaded,
    skip. reads = input, or output with start ACCUMULATE. gathered/scattered operands produce Whole views; others
    ElementViews(Compose(v, map, terms, box), box) -> ByteViews for reads and/or writes. Quantization scale and
    zero/folded operands: Whole for gathered else ComposeGroup -> ByteViews. Then Whole views for op.gathers not in
    inputs and op.symbols not Internal. Effects(box) passes box; Effects(op, vertex, kernel) passes Domain(op).
  - takes: Program, operation, box (Box), vertex, kernel, output vector of View.
  - returns: void; appends View records to into.
  - cites: metal_ir.mm:194 "// docs/provider.md#target-owned-control" (above Accesses, which caches Effects)
  - refuses:
    - none: no raise or error return in 890-936
  - callers: Effects: metal_ir.mm:203 (Accesses cache), 1593, 1600 (Actor overloads), 2008 (Artifact). Accesses
    (metal_ir.mm:195): called at metal_ir.mm:340, 345, 1032 (Footprints).
- **`Relation (static)`** — metal-microbench:metal_ir.mm:1727-1765 (whole body)
  - does: Clears givens. For each later pass: adds edge first->compaction (if first < compaction) and compaction->later
    (if compaction < later) with given kinds METAL_IR_GIVEN_COMPACTION / METAL_IR_GIVEN_LATER. For each kernel with
    a subject task != itself: adds edge (min, max) with METAL_IR_GIVEN_STEP. Views = p.records plus one view per
    control predicate (key, start, size, vertex = dispatches + index). Calls mesh_graph_conflicts_views with those
    views and edges. Stores p.conflicts. On status: sets p.error and returns. Stores dependence from/to/cause for
    conflicts with from < to and to < dispatches.
  - takes: Program &p.
  - returns: void; sets p.conflicts, p.dependenceFrom/To/Cause, p.givens, or p.error.
  - cites: none stated in the function; mesh-graph.h:161-175 comment (isl,
    design/compilation-composition-change-request.md) describes the callee
  - refuses:
    - metal_ir.mm:1760 p.error = "the dispatch conflict relation: " + strerror(status); return (on any nonzero
      status)
  - callers: 1 call site: metal_ir.mm:2052 (inside Artifact).
- **`Conflict`** — metal-microbench:metal_ir.mm:763-777 (part of the body)
  - does: Builds views from view sets a and b (only writes if aWrites/bWrites), tagged vertex 0 and 1; calls
    mesh_graph_dependences_views with no supplied edges and room 4; returns status != 0 or edges > 0.
  - takes: Program, two View vectors, aWrites, bWrites flags.
  - returns: bool: true if any dependence (or on error).
  - cites: none stated
  - refuses:
    - metal_ir.mm:774 p.error = "the exact dependence test of N views failed: " + strerror(status) when status not 0
      or ERANGE
  - callers: not determined (not searched beyond its definition).
- **`Span (static) and Contiguous (static)`** — metal-microbench:metal_ir.mm:731 (Span); 736-743 (Contiguous) (part of the body)
  - does: Contiguous: starting from extent = element, walks pairs from innermost outward; returns false if a stride
    exceeds the running extent, else extends extent by (size-1)*stride; returns true. Span: declaration only read at
    line 731; returns the first and last byte offsets of the view.
  - takes: View.
  - returns: bool (Contiguous); pair of uint64 (Span).
  - cites: none stated
  - refuses:
    - Contiguous: returns false (metal_ir.mm:739) when a stride exceeds the running extent (not an error; marks
      non-contiguous)
  - callers: Contiguous and Span: called from Footprints (metal_ir.mm:1011-1016); other callers not determined.
- **`mask_bounds (static, C)`** — mesh:rdma/mesh-metal.m:136-148 (whole body)
  - does: Per threadgroup g: low = footprint.start - r.at, plus for each of the footprint's terms t: coefficient * (q %
    modulus or q), q = g[t.dim] / divisor. high = low + extent. Clamps low to >= 0 and high to <= partitions*page.
    If high <= low or page is 0: lo=1, hi=0. Otherwise lo = low/page, hi = (high-1)/page: the page index range.
  - takes: g[3], footprint, term array, range r (at, partitions, page).
  - returns: void; writes lo and hi page indices.
  - cites: none stated in the body; mesh-graph.h:144 struct footprint comment
  - refuses:
    - none: empty range is encoded as lo=1, hi=0 (mesh-metal.m:146)
  - callers: called from mesh_metal_mask (mesh-metal.m:158, body not read).
- **`mesh_metal_mask (signature only)`** — mesh:rdma/mesh-metal.m:158 (signature); declared mesh-metal.h:90 (declaration only)
  - does: Not read beyond the signature. Its use of mask_bounds indicates it maps threadgroups to page masks.
  - takes: groups[3], footprints, footprint array, term array, range array, readiness out.
  - returns: int
  - cites: none read
  - refuses:
    - not read
  - callers: 1 call site: metal_transport.mm:268.

**Documents**

- mesh:design/already-solved-subproblems-2026-11-09.md: Line 98, row "3. loop and index analysis": names rdma/mesh_compile.py `_extent`, `_paged`, `_partition`'s regions;
  rdma/mesh-graph.h `mesh_graph_intersect`, `mesh_graph_dependences_views`, `mesh_graph_conflicts_views`;
  metal_ir.mm `Footprints`, `Effects`, `Relation`; dated "11-05 (`_extent`), 11-06 (`mesh_graph_intersect`), 11-09
  (the rest)"; its "where it is computed today" column says "computed in the frontend, again in the graph module,
  and again in the compiler core"; its target column says "once, in the graph module, with the frontend supplying
  the accesses". The document makes this claim; it was not checked against code beyond the functions read here.
  Source names: none in row 98.
- mesh:rdma/mesh-graph.h: Lines 147-150 (mesh_graph_dependences comment): "the relation is the supplied edges (supplied_from[e] ->
  supplied_to[e]) united with the accesses' conflicts (lower vertex first), each pair once, sorted by (from, to) ...
  `access` is sorted in place". Lines 159-170 (view struct comment): "Two views of one key, of different vertices,
  at least one written (both 2 do not conflict), conflict exactly where some byte is in both ... no index or byte
  enumerated and no hull taken for a conflict." Lines 177-179 for mesh_graph_conflicts_views: "retained exact
  intersections, including self conflicts". Line 337: "MPI 4.1 section 6.1.11" for mesh_graph_intersect with no
  further text.
  - sources it names: design/compilation-composition-change-request.md item 6.2 (line 147); S. Verdoolaege, isl, ICMS 2010, LNCS 6327
    (line 161); MPI 4.1 section 6.1.3 (line 155); MPI 4.1 section 6.1.11 (line 337); J. Lee, Ph.D. dissertation,
    Univ. of Michigan 2015, Chapter III section 3.2.2 (lines 141-143); torch 2.14.0 c10/core/TensorImpl.h (line
    155); GaijinEntertainment/DagorEngine heapLayout.cpp (line ~340-341)
- mesh:rdma/mesh_compile.py (comments at lines 751, 767, 810): These are the source comments above _extent, _region and _partition. Each cites the method's source; no sentence
  describes the index-space arithmetic beyond the citation.
  - sources it names: GSPMD Appendix A.2, Figure 9b (line 751); GSPMD sections 4.1, 4.3, 4.5, Appendix A.2, Figures 5a, 9b; MPICH
    allgather_intra_ring.c, MPIR_Localcopy (line 767); GSPMD sections 2.1, 3, 3.2, 4, 4.2, 4.5, 5.1, Figures 6, 7;
    XLA spmd_partitioner.cc PartitionedHlo::ReshardAsWindowedInput (line 810)

**The same thing computed in more than one place**

- Half-open same-key interval overlap is computed in mesh-graph.c:3128-3130 (mesh_graph_intersect: interval[k].start
  < interval[k-1].end), mesh-graph.c:1582-1583 (mesh_graph_dependences sweep: access[j].start < access[i].end),
  mesh-graph.c:3140 (mesh_graph_arcs_overlap: a_end > b_start && b_end > a_start), and mesh-graph.c:1628 (view_meets
  disjoint-hull test). All decide whether two byte or position ranges share a point.
- Pairwise access conflict is computed twice on different inputs: mesh-graph.c:1572-1607 (mesh_graph_dependences,
  interval accesses with access_conflict at mesh-graph.c:1563-1565) and mesh-graph.c:1670-1729 (view_conflicts via
  view_meets, exact strided views). Both produce the (from, to) conflict relation; the first is interval-only, the
  second exact.
- Dependence relation from access records: metal_ir.mm:763-777 (Conflict), metal_ir.mm:1727-1765 (Relation),
  metal_recording.m:2743-2760 (MetalHostSide) and metal_recording.m:3059-3071 all call mesh_graph_dependences_views
  or mesh_graph_conflicts_views on view sets built from the same struct mesh_graph_view records.
- Operand byte extent: metal_ir.mm:890-927 (OperandEffects, ElementViews then ByteViews over Compose) and
  metal_ir.mm:1003-1005 (Footprints, ElementViews(c, box) then ByteViews) both compute the bytes an operand touches
  over a box; Footprints additionally recomputes spans (metal_ir.mm:1011-1014) and contiguity (metal_ir.mm:1016).
- Halo width: mesh_compile.py:766-767 (_extent: left = max(q*n - a), right = max(a + length - (q+1)*n)) and
  mesh_compile.py:776 (_region whole branch: left = max(-a), right = max(a + length - size)) both compute how far a
  window extends beyond a shard or the full value.
- Footprint term evaluation: metal_ir.mm:946-1059 (Footprints builds mesh_graph_term entries from Atom/Fold) and
  mesh-metal.m:136-148 (mask_bounds evaluates the same terms per threadgroup: q = g/divisor, coefficient * (q %
  modulus)). The divisor/modulus meaning is stated in the struct mesh_graph_term comment at mesh-graph.h:139-141
  (MLIR affine_map) and metal_ir.mm:938-945.
- Ceiling-to-multiple padding -(-x // m) * m appears in _reshard (mesh_compile.py, exact lines not read) and at
  mesh_compile.py:902 (_partition reduce-scatter pad), with the same arithmetic as the padded size in _partition_pad
  (mesh_compile.py:673-680).

**No function found, or only a refusal**

- A mesh-compile function that computes the index interval of a general non-window shard (offset per mesh
  coordinate) is not among the named functions; it appears inside _reshard (ir.inside entries with
  ir.coords[e][m]*shape[d]) which was not read in full.
- Page coverage: _paged decides paging but no function in the read set maps a byte range to page indices; that
  mapping is in mesh-metal.m mask_bounds (136-148) for threadgroup footprints, and the page-value path is a ctypes
  call at mesh_compile.py:7342 (metal_ir_value_paged), not read.
- Exact dependence for host-side transfers in mesh_compile.py: _dependences (mesh_compile.py:1477) was not read
  beyond its first lines.
- Callers of metal_ir_footprints (metal_ir.mm:3582): none found in metal-microbench or dox/mesh beyond the
  declaration (metal_ir.h:297) and definition.
- No function found for an index-space (non-byte) dependence test between two loops of a Generic operation;
  Footprints works on the tile grid only.

**Not determined**

- mesh_compile.py _whole (def at line 1140) and _paging (def at line 7246): bodies not read, so the exact condition
  used in _paged is not determined.
- Callers of Conflict (metal_ir.mm:763) and of Span (metal_ir.mm:731) beyond the sites named: not determined.
- Callers of mesh_metal_mask beyond metal_transport.mm:268: not determined.
- Whether mesh_graph_dependences (no callers found) is reached through any binding in a language other than C or
  Python: not searched beyond grep over dox/mesh and metal-microbench.
- The exact line of the _reshard definition in mesh_compile.py: not read (it is after line 673 and before line 752).

## View, layout, stride and alias analysis

Read: 14 whole body, 1 part of the body. 15 functions.

- **`_layout`** — mesh:rdma/mesh_compile.py:249-252 (whole body)
  - does: If key is in ir.metas, returns (metas[key].dtype, tuple(metas[key].shape)); otherwise returns
    (_value(key[0]).dtype, tuple(_shape(ir, key))). Metadata read, no loop.
  - takes: ir, key
  - returns: (dtype, shape tuple)
  - cites: none (no comment above the def in lines read; line 248 not read)
  - callers: about 20 call sites on 19 lines, all in mesh_compile.py (1145, 1155, 1159, 1161 x2, 1167, 1223, 1237, 1254, 1303
    x2, 1333, 1373, 1453, 1561, 1602, 6986, 6989, 7033, 7061). None found outside mesh_compile.py.
- **`_view_rule`** — mesh:rdma/mesh_compile.py:493-510 (whole body)
  - does: For aten.expand: each output dim maps to an input dim when sizes match at the trailing alignment, else label
    ('expanded', d). For view/_unsafe_view: boundaries dict from (prod(shape[:d]), size) for input dims with size>1;
    each output dim maps to an input dim label when (prod(out[:d]), size) matches, else ('reshaped', d). Returns
    Rule('format', 0, ...) whose configure() rewrites the node's view args with sharded sizes.
  - takes: node, local, operands, shapes, out
  - returns: Rule object
  - cites: mesh_compile.py:491: "# GSPMD sections 3.5 and 4.1: a reshape's unchanged dimension boundaries preserve their
    partition;" and mesh_compile.py:492: "# PyTorch Tensor Views, aten::view, aten::_unsafe_view and aten::expand"
  - callers: 1 call site: mesh_compile.py:618 (operator-rule dispatch, for packet in (view, _unsafe_view, expand) with one
    operand). None found outside.
- **`_viewed`** — mesh:rdma/mesh_compile.py:1095-1096 (whole body)
  - does: Returns True iff item[0] == 'window', len(item[1]) == 1 and item[3][6] is None.
  - takes: item
  - returns: bool
  - cites: mesh_compile.py:1094: "# design/aot-compilation-change-request.md, appendix item 7"
  - callers: 1 call site: mesh_compile.py:1100 inside _sliced. None found outside.
- **`_sliced`** — mesh:rdma/mesh_compile.py:1099-1100 (whole body)
  - does: Returns True iff item[0] in ('slice','narrow','pad') or _viewed(item).
  - takes: item
  - returns: bool
  - cites: none (no comment read)
  - callers: used in _placement (mesh_compile.py ~1278: _sliced(item) and reads[0] is not None) and elsewhere in
    mesh_compile.py; exact count not enumerated (unknown).
- **`_contiguous`** — mesh:rdma/mesh_compile.py:1218-1219 (whole body)
  - does: Returns tuple(math.prod(shape[d+1:]) for d in range(len(shape))): the row-major dense stride vector of shape, in
    elements.
  - takes: shape
  - returns: tuple of ints
  - cites: mesh_compile.py:1212-1214 comment block above Place: "# design/plan-to-bridge-change-request.md R1;
    design/aot-compilation-change-request.md, appendix item 7; mesh-collective.h struct mesh_strided; torch 2.14.0
    torch/_prims/__init__.py, collapse_view and _collapse_view_helper"
  - callers: 3 call sites in mesh_compile.py: 1146 (_whole), 1227 (_own fallback), 1238 (_part). None found outside.
- **`_view`** — mesh:rdma/mesh_compile.py:1298-1329 (whole body)
  - does: Takes _based(local) (index of the argument the op's schema aliases into its result). Returns None if that index
    is None, reads[k] is None, or key not in ir.metas. Builds stand-ins: for each operand, _meta(ir, r) if its place
    p is None, else a meta tensor of numel(owner) .as_strided(p.sizes, p.strides, p.offset). Re-runs the op through
    _call with args rebuilt by map_arg. NotImplementedError -> None; RuntimeError -> add key to ir.contiguous,
    return _own(ir, key); result not a Tensor or dtype differs -> None; result not an alias of metas[k]
    (torch._C._is_alias_of) -> add key to ir.contiguous, return _own(ir, key); otherwise returns Place(owner of
    reads[k], result.storage_offset(), result.shape, result.stride(), reads[k].writer). So offset and strides of a
    view are taken from torch's result.
  - takes: ir, key, local, keys, reads
  - returns: Place namedtuple (owner, offset, sizes, strides, writer), None, or an _own Place
  - cites: none directly above _view; the comment "# torch 2.14.0 torch/_tensor_docs.py, view" sits above _based (about
    line 1280), not _view
  - refuses:
    - mesh_compile.py:1301 'return None' (k is None, reads[k] is None, or key not in ir.metas)
    - mesh_compile.py:1306 'return None' (any meta is UNKNOWN)
    - mesh_compile.py:1311-1312 except NotImplementedError: return None
    - mesh_compile.py:1313-1315 except RuntimeError: ir.contiguous.add(key); return _own(ir, key) (fallback, no
      message)
    - mesh_compile.py:1317 'return None' (result not Tensor or dtype differs)
    - mesh_compile.py:1318-1320 not _is_alias_of -> ir.contiguous.add(key); return _own(ir, key) (fallback)
  - callers: 1 call site: mesh_compile.py:1279 in _placement (kind == 'call', form None, op is OpOverload and
    _aliases(local)). None found outside.
- **`_strided`** — mesh:rdma/mesh_compile.py:1332-1343 (whole body)
  - does: size = itemsize of the owner's dtype. Walks (n, s) of place.sizes/strides, skipping n == 1; merges into the
    previous (count, stride) when prev_count*... i.e. dims[-1][1] == n*s (dims[-1] = (prev_n*n, s)), else appends
    (n, s). Builds a ctypes array of (count, stride*size) pairs and returns mesh.Strided(offset=place.offset*size,
    element=size, dims=len(dims), pairs=pairs).
  - takes: ir, place
  - returns: mesh.Strided struct
  - cites: mesh_compile.py:1331: "# MPI 4.1 section 6.1.3, MPI_TYPE_CREATE_SUBARRAY" (the comment block at 1212-1214 also
    cites mesh-collective.h struct mesh_strided)
  - callers: about 9 call sites in mesh_compile.py: 1376, 1377, 1378, 1379, 1405, 1406, 1422, 1423, 1567. None found outside
    mesh_compile.py.
- **`_aliases`** — mesh:rdma/mesh_compile.py:6853-6854 (whole body)
  - does: True iff local.target is operator.getitem, or (local.target is an OpOverload and any return of its schema has
    alias_info). Precedence: A or (B and C).
  - takes: local
  - returns: bool
  - cites: none (no comment above the def)
  - callers: 4 call sites in mesh_compile.py: 950, 1109, 1278, 6898. None found outside.
- **`_whole`** — mesh:rdma/mesh_compile.py:1144-1146 (body lines read; def line within 1138-1143 not separately captured) (part of the body)
  - does: If key has no place: True iff items[key][0] in ('input','call'), len(key)==2, key[0].meta['val'] is a Tensor and
    is_contiguous(). Else True iff place.offset == 0 and place.sizes == layout shape and place.strides ==
    _contiguous(shape).
  - takes: ir, places, items, key
  - returns: bool
  - cites: mesh_compile.py:1136-1137 comment above COPIES: "# ACM Comput. Surv. 25(2), 1993; PyTorch 2.14 documentation,
    Broadcasting semantics" (attached to COPIES, not to _whole)
  - callers: in _paged, mesh_compile.py 1155 (REDUCTIONS branch), 1159 (SOFTMAX branch), and the scaled_dot_product_attention
    branch after 1161 (two calls). None found outside mesh_compile.py.
- **`Span (C++ free function)`** — metal-microbench:metal_ir.mm:731-734 (declared metal_ir.h:422) (whole body)
  - does: end = v.offset + v.element; for each pair k (pairs[k] = count, pairs[k+1] = stride) adds (count-1)*stride to
    end. Returns {offset, end}. Arithmetic only.
  - takes: const View &v
  - returns: std::pair<uint64_t,uint64_t> {first, end} in bytes
  - cites: metal_ir.mm:729-730: "A byte view's span [first, end) and whether its bytes are one contiguous run (each
    dimension, smallest stride first, no further apart than the extent inside it)." (a description; no external
    citation)
  - callers: metal_ir.mm:752 (Distinct, twice) and 1015 (Footprints). Declared metal_ir.h:422.
- **`Contiguous (C++ free function)`** — metal-microbench:metal_ir.mm:736-744 (declared metal_ir.h:423) (whole body)
  - does: extent = v.element; iterates pairs from the last pair down: if the pair's stride > extent returns false; else
    extent += (count-1)*stride. Returns true if the loop completes.
  - takes: const View &v
  - returns: bool
  - cites: none (the comment at 728-730 is shared with Span)
  - refuses:
    - metal_ir.mm:739 'return false' when a dimension's stride exceeds the extent inside it (boolean result, no
      message)
  - callers: 1 call site: metal_ir.mm:1016 inside Footprints (contiguous &= Contiguous(x)). None found outside metal_ir.mm.
- **`Distinct (static)`** — metal-microbench:metal_ir.mm:746-757 (whole body)
  - does: Walks dims smallest stride first while stride <= extent to get the contiguous part (extent). Then for remaining
    dims: if a dim's stride < span, returns Span(v).second - Span(v).first (the full span); else multiplies distinct
    by count and extends span. Returns distinct bytes.
  - takes: const View &v
  - returns: uint64_t bytes
  - cites: metal_ir.mm:744-745: "Its distinct bytes: the contiguous part (dimensions within the extent inside them) times
    the counts of the dimensions outside it where they do not overlap (each stride at least the span inside it); the
    span otherwise." (no external citation)
  - refuses:
    - metal_ir.mm:752 returns the full span instead of a distinct count when a stride overlaps the span (fallback,
      not a refusal)
  - callers: metal_ir.mm:1547 and metal_ir.mm:1560. None found outside metal_ir.mm.
- **`ByteViews (static)`** — metal-microbench:metal_ir.mm:695-728 (whole body)
  - does: For each ElementView e: if the scalar is not packed (Words false), emits one View {storage, base*eb, eb} with
    pairs (count, stride*eb) per digit. For packed (sub-32-bit) scalars: lanes = 32/bits; drops trailing digits
    whose stride-run fits within one lane word; splits remaining digits into whole (stride % lanes == 0) and
    partial; enumerates partial digits recursively; for each base emits a View of 4-byte words covering
    [FloorDiv(base, lanes), FloorDiv(base+run-1, lanes)] with pairs (count, stride/lanes*4) for whole digits.
  - takes: const Value &v, const vector<ElementView> &views, vertex, write, into
  - returns: void; appends to vector<View> &into
  - cites: none (the comment above Span at 728-730 is the nearest; not a citation)
  - callers: metal_ir.mm:851, 911, 912, 920, 1005. None found outside metal_ir.mm.
- **`Normalize (static, ElementView)`** — metal-microbench:metal_ir.mm:616-624 (whole body)
  - does: Drops digits with count<=1 or stride==0; for negative stride moves base by stride*(count-1) and negates the
    stride; stable-sorts remaining digits by stride descending (outermost first).
  - takes: int64_t base, const vector<Digit> &digits
  - returns: ElementView {base, digits}
  - cites: metal_ir.mm:614-615: "One strided element view: base + sum of digits (count > 1, stride > 0), outermost (largest
    stride) first: negative strides normalized by moving the base to the least offset, unit counts and zero strides
    dropped." (no external citation)
  - callers: not enumerated in this pass (unknown)
- **`Bind (C++, anonymous namespace)`** — metal-microbench:provider/litert_mesh.cc:4861-4910 (whole body)
  - does: Validates program and values non-null; gets the interpreter; gets the signature's subgraph by key; calls
    AllocateTensors; checks count == inputs + outputs; appends one step per execution_plan node (delegated ones
    tagged by the LITERT_WEBGPU custom_name). For each subgraph tensor k: if values[k] == UINT32_MAX, reuses an
    existing bound value for that tensor or creates one with metal_ir_value(role, scalar, rank, shape, rank, layout,
    0, tensor pointer). The layout at 4897-4902 is layout[d] = {d, 0, stride, 1, 0} with stride starting at 1 and
    multiplied by shape[d] from the innermost dim outward: row-major dense strides computed from the TfLite dims,
    not read from the tensor's own strides. Records out->tensors[tensor] = values[k]. Returns 0.
  - takes: LiteRtCompiledModel, signature index, metal_ir_program*, uint32_t *values, roles, count, error buffer, room,
    Bound* out
  - returns: int (0, or EINVAL with error text)
  - cites: none (no comment inside Bind; comments above 4861 not read)
  - refuses:
    - litert_mesh.cc:4863 EINVAL "program and values are required"
    - litert_mesh.cc:4865 EINVAL interpreter.Error().Message() (GetInterpreter failure)
    - litert_mesh.cc:4868 EINVAL "signature index past the model's signatures"
    - litert_mesh.cc:4871 EINVAL "the signature's subgraph does not allocate at its current input shapes"
      (AllocateTensors != kTfLiteOk)
    - litert_mesh.cc:4874 EINVAL "values: one per subgraph input then output" (count mismatch)
    - litert_mesh.cc:4894 EINVAL "subgraph tensor without a stored kind" (returns when Scalar(t->type, &scalar) is
      nonzero)
  - callers: litert_mesh.cc:4917 (LiteRtMeshDeclare) and 4924 (LiteRtMeshPublish); both are exported C entry points, called
    from tools/e2b_replay.m:495-496 and declared in provider/litert_mesh.h:13,16.

**The same thing computed in more than one place**

- Row-major dense stride vector of a shape: mesh_compile.py:1218-1219 (_contiguous, prod of later dims);
  litert_mesh.cc:4897-4902 (Bind, stride *= shape[d] from innermost outward); litert_mesh.cc:4932-4938
  (LiteRtMeshPublish lambda, same loop over Builtin::Dimensions(t)).
- Dense-contiguity test on an operand: mesh_compile.py:1144-1146 (_whole: offset 0, sizes equal, strides ==
  _contiguous(shape)) and metal_ir.mm:736-744 (Contiguous: each stride no larger than the extent inside it).
- Merging adjacent dims with compatible stride: mesh_compile.py:1336-1339 (_strided merges when dims[-1][1] == n*s)
  and metal_ir.mm:616-624 (Normalize drops unit dims and sorts by stride).
- Alias classification from the op schema: mesh_compile.py:6853-6854 (_aliases: any return with alias_info) and the
  _based function in mesh_compile.py (schema argument alias_info.before_set intersected with return sets, body at
  lines ~1281-1292 not fully captured), both reading the same schema alias_info.
- View dimension mapping for the same aten view/expand ops by two methods: mesh_compile.py:493-510 (_view_rule,
  labels from shape products) and mesh_compile.py:1298-1329 (_view, strides taken from torch's as_strided result).
- Byte extent of a View: metal_ir.mm:731-734 (Span) and metal_ir.mm:746-757 (Distinct, which calls Span at 752) and
  metal_ir.mm:1547/1560 (both consume Distinct).

**No function found, or only a refusal**

- No function in mesh_compile.py checks stride-based aliasing against actual storage beyond torch._C._is_alias_of
  inside _view (mesh_compile.py:1318).
- metal_ir.mm has no function named Layout or Alias; value strides are carried as metal_ir_term arrays and their
  bounding code was not read in this pass.
- The explicit-copy emitter (_assign, and the _called branch that calls _assign at ~6898-6915) was only partly read,
  so the part of the subproblem that makes copies explicit was not fully traced.

**Not determined**

- ElementViews (metal_ir.mm:629) body was not read.
- Callers of Normalize and ByteViews beyond the listed lines were not enumerated.
- Exact call counts of _sliced and _whole in mesh_compile.py were not enumerated beyond the listed lines.
- Design documents cited by mesh_compile.py (design/plan-to-bridge-change-request.md R1,
  design/aot-compilation-change-request.md appendix item 7, GSPMD sections, Alpa section 4.1, MPI 4.1, torch 2.14.0
  source files) were named in comments but not opened, so their content for this subproblem is unknown.
- Whether Scalar's nonzero return at litert_mesh.cc:4894 covers exactly the unsupported-type case: the
  Scalar(TfLiteType) body at litert_mesh.cc:158 was not read in full.
- The function that contains mesh_compile.py:617-618 (operator-rule dispatch) was not enumerated for callers.

## Folding static operands into the program

Read: 19 whole body, 2 part of the body. 21 functions.

- **`_folded`** — mesh:rdma/mesh_compile.py:3047-3051 (whole body)
  - does: Returns value.view(-1, value.shape[-1]), a 2-D view of the tensor's leading dims. It is a reshape-for-matmul
    helper, not constant folding. On RuntimeError (a non-viewable tensor) it returns None.
  - cites: none directly above the def. The comment at 3054-3055 cites PyTorch matmul docs and LinearAlgebra.cpp
    _matmul_impl, but it sits above _product (3056), not _folded.
  - refuses:
    - returns None on RuntimeError (mesh_compile.py:3050-3051)
  - callers: 3 statements, 6 call expressions: mesh_compile.py:3065 (2), 3075 (2), 3076 (2).
- **`_fold`** — mesh:rdma/mesh_compile.py:5267-5285 (whole body)
  - does: Lowers torch prod/any/all (dispatch set FOLDS = (A.prod, A.any, A.all), line 5263) to a _Generic reduction over
    the input's loops. Complex input is delegated to _complex_fold (5270). For real input it calls
    _typed(target,[x],outs,REAL) (5271), builds a REGION that loads each element, and uses mul for prod, or for any,
    and for all as the combine. Its states use identities.
  - cites: none; lines 5265-5266 above the def are blank.
  - refuses:
    - _typed(target, [x], outs, REAL) at mesh_compile.py:5271 can refuse non-real input; its body was not read
  - callers: 7 direct calls: mesh_compile.py:3600, 4017, 4053, 4190, 4217, 4421, 5979. It is also dispatched through the
    FOLDS table at mesh_compile.py:6633.
- **`_complex_fold`** — mesh:rdma/mesh_compile.py:5287-5305 (whole body)
  - does: Complex prod/any/all. For any/all it reduces to a boolean: OR (any) or AND (all) of (re!=0) or (im!=0). For prod
    it reduces the (re, im) pair under a complex-multiply REGION monoid, returning two real outputs via
    view_as_real.
  - cites: comment directly above def, lines 5285-5286: "PyTorch 2.14 documentation, torch.prod, torch.any and torch.all of
    complex input: the product the REGION monoid of the (re, im) states under the complex product, each element read
    at both components by constant rows; a truth an element with either component nonzero". No external URL.
  - refuses:
    - none explicit
  - callers: 1: mesh_compile.py:5270 (in _fold).
- **`_constant_origins`** — mesh:rdma/mesh_compile.py:6993-7003 (whole body)
  - does: Collects the storage data_ptr of every torch.Tensor in exported.constants whose name is not in
    graph_signature.non_persistent_buffers. It also adds the attributes named in each submodule's _mesh_constants
    (via operator.attrgetter). It then subtracts the storages written by buffers_to_mutate and parameters_to_mutate
    (via _storages). Returns a frozenset of storage pointers that are immutable constants.
  - cites: comment directly above def, line 6992: "# torch/export/graph_signature.py InputKind.CONSTANT_TENSOR;
    mesh_code.Matrix._mesh_constants"
  - refuses:
    - none; non-Tensor values are skipped by isinstance (mesh_compile.py:6999)
  - callers: 1: mesh_compile.py:8510.
- **`_initial_values`** — mesh:rdma/mesh_compile.py:7005-7029 (not a named function; same subproblem) (whole body)
  - does: If ir.initial is set it returns it. Otherwise: loaded = ir.loads & ir.homes.keys(); needed = mesh_graph.reach
    over attributes; host = mesh_graph.forward evaluating each attribute via _hosted; fixed[key] is true for
    'constant' attributes whose host value's _key is in the immutable argument. It allocates a zero CPU tensor per
    layout value, copies each host value into a strided view at its home, and records mutable values. It sets
    ir.initial and ir.constant_values = set(initial) - mutable.
  - cites: comment directly above def, line 7004: "# metal-microbench
    docs/provider.md#source-constants-before-target-preparation; torch/export/graph_signature.py
    ExportGraphSignature.buffers_to_mutate, parameters_to_mutate"
  - refuses:
    - none explicit
  - callers: 1: mesh_compile.py:8673.
- **`_specialized`** — mesh:rdma/mesh_compile.py:7039-7075 (whole body)
  - does: Computes one item (kind, reads, key, operand) of a function's program over placed stand-ins. Returns early with
    the placed value if key is in ir.attributes. Otherwise by kind: 'input' sets value=given[operand]; 'constant'
    with value None sets value=attrgetter(operand)(ir.gm); 'copy' appends _copied ops; 'pad' fills and assigns the
    source; 'window' fills out-of-range slices with fill; 'slice'/'narrow'/'window' narrow the operand; 'call' trims
    operands per ir.trims, calls _called, appends its ops; 'collective' appends _collected ops. Returns value.
    Unknown kinds fall through and return the placed value.
  - cites: comment directly above def, lines 7037-7038: "N. D. Jones, C. K. Gomard and P. Sestoft, 1993, Definition 4.1:
    one item of the function's program over its placed stand-ins (`placed`) and the call's input leaves (`given`):
    its value and its structured operations"
  - refuses:
    - no raise or error return; unknown kind is not rejected (falls through, line 7075)
  - callers: 1: mesh_compile.py:7229.
- **`_specialize`** — mesh:rdma/mesh_compile.py:8065-8090 (whole body)
  - does: For each residue p in range(_period(copies, horizon)) (_period = math.lcm(horizon, all copy-ring lengths;
    mesh_compile.py:8038-8039), builds a _Program(gm, ...) fx Interpreter run. Its call hook maps each output tree
    of a node to _TensorLocation via leaf(): tensors whose _key is in owners become a location (copy k, value v,
    index (p if rated else p*every[k]+j) mod len(copies)). Program inputs become locations via input(i,p) (holders
    None uses constants[i]). Returns a list, one entry per residue, of run results.
  - cites: none directly above def (lines 8063-8064 blank). The Jones et al. comment at 8042-8045 sits above class
    _TensorLocation (8046), not above _specialize.
  - refuses:
    - none explicit; missing keys would raise KeyError (not guarded)
  - callers: 1: mesh_compile.py:8573 (in the output_recipe path).
- **`_constant_storage`** — mesh:rdma/mesh_compile.py:8809-8811 (whole body)
  - does: If info.buffer is set it returns (('buffer', info.buffer), OBJECT(info.buffer, _SEL(b'contents')),
    VALUE(info.buffer, _SEL(b'length'))). Otherwise it returns (('data', info.data, info.data_bytes), info.data,
    info.data_bytes). The OBJECT, VALUE and _SEL helpers are defined elsewhere and were not read.
  - cites: none; line 8808 above the def is blank. The comment at mesh_compile.py:8091 ("mesh-call.h mesh_transfers_start
    ...") is above _execution_error (8092).
  - refuses:
    - none
  - callers: 2: mesh_compile.py:8612 and 8843.
- **`Constants`** — metal-microbench:metal_transform.mm:36-40 (whole body)
  - does: Returns true iff every operand of inst is null or an ir::Constant.
  - cites: none
  - refuses:
    - none; returns false on any non-constant operand (line 38)
  - callers: 1: Fold, metal_transform.mm:125.
- **`Identity`** — metal-microbench:metal_transform.mm:51-76 (whole body)
  - does: For a CoreBinary, returns the non-constant operand when the op is an exact identity given a constant operand:
    x*1 and x/1 (multiply/divide); x+0 (integer, or float when the constant is -0); x-0 (integer, or float when +0);
    x<<0, x>>0, x|0, x^0 (integers only). It also handles a constant left operand for x*1 (multiply) and 0+x (add)
    when the other operand has the result type. Returns nullptr otherwise.
  - cites: comment at metal_transform.mm:48-49: "MLIR arith dialect folders (ArithOps.cpp:
    addi/subi/muli/divui/divsi/shli/shrui x op identity, mulf x * 1, divf x / 1, subf x - (+0), addf x + (-0)): each
    an exact identity, so it holds under every precision flag."
  - refuses:
    - returns nullptr when the operand type differs from the result type (line 53), when the constant is non-scalar,
      or for any other op (default case)
  - callers: Fold only (metal_transform.mm:108-141). Callers outside this file were not searched.
- **`Select`** — metal-microbench:metal_transform.mm:78-106 (whole body)
  - does: Replaces an ir::If by its taken branch. If the branch ends in an ExitIf, the If's results are replaced by the
    ExitIf args and the chosen block's instructions are moved before the If. The If is destroyed. Otherwise the
    block is spliced and all instructions from the If onward are destroyed.
  - cites: none
  - refuses:
    - none
  - callers: Fold (metal_transform.mm:110-115, when the If condition is a Constant).
- **`Fold`** — metal-microbench:metal_transform.mm:108-141 (whole body)
  - does: Walks one block's instructions. An If with a Constant condition is replaced by Select. Recurses into
    ControlInstruction blocks. A Let whose value is a Constant is replaced by that constant. A pure instruction
    (Pure, not Load/LoadVectorElement, with all-Constant operands and at least one operand) is evaluated with
    ir::Evaluator, and its result is replaced by the folded constant on success. Otherwise a CoreBinary is replaced
    by Identity's result when one exists.
  - cites: none
  - refuses:
    - no raise; on evaluator failure the instruction is left in place (line 127-128)
  - callers: Fold recurses on itself (metal_transform.mm:119); Simplify calls it (metal_transform.mm:158).
- **`Prune`** — metal-microbench:metal_transform.mm:143-155 (whole body)
  - does: Walks a block backward. It recurses into control blocks. It destroys any non-Terminator Pure instruction whose
    result is unused.
  - cites: none
  - refuses:
    - none
  - callers: Prune recurses on itself (metal_transform.mm:147); Simplify calls it (metal_transform.mm:159).
- **`Simplify`** — metal-microbench:metal_transform.mm:157-160 (declared metal_lower.h:149) (whole body)
  - does: Runs Fold on the function's block, then Prune on the same block.
  - cites: none
  - refuses:
    - none
  - callers: 4 sites: metal_transform.mm:405 (inside Fuse, 344-448), metal_transform.mm:715 (inside Epilogue, 547-760),
    metal_transform.mm:1366 (inside Transform, 1361-), metal_native.mm:529 (inside Specialize,
    metal_native.mm:508-).
- **`ZeroPoints`** — metal-microbench:metal_transform.mm:168-197 (whole body)
  - does: For each Value whose quantization has a zero-point index (not UINT32_MAX) and scheme METAL_IR_AFFINE, if the
    zero value is a CONSTANT with data, it enumerates every element of the zero tensor with Constant(p, q.zero,
    index, x). If all elements are equal to shared, it clears q.zero. Then, if shared == 0 it sets the scheme to
    METAL_IR_SYMMETRIC; otherwise it sets q.folded = true and q.offset = shared.
  - cites: comment directly above def, lines 164-167: "S6/SCCP of a quantization's zero points: a CONSTANT zero tensor
    whose elements ... all equal z makes every element read (x − z)·scale ... (the old MetalSpecializeScalars'
    MN_ZERO_POINTS fold, and 4's per-kernel Uniform scan)." No external source; the names are internal.
  - refuses:
    - continue (no fold) when q.zero == UINT32_MAX or scheme != METAL_IR_AFFINE (line 171); when the zero value is
      not CONSTANT or has no data (line 173); when the elements are not uniform (line 185, uniform=false) or
      Constant() fails on any element (line 187)
  - callers: 1: Transform, metal_transform.mm:1362.
- **`Constant (helper used by ZeroPoints)`** — metal-microbench:metal_ir.mm:147-~170 (declared metal_ir.h:791) (part of the body)
  - does: Reads one scalar element from the retained byte storage of value: computes the element offset through
    Element(v,index), reads the bit-packed word from Contents(p,value).data, masks it by ScalarBits, and converts to
    double x by scalar type (bool, int, float).
  - cites: none
  - refuses:
    - returns false when storage.data is null (metal_ir.mm:150), when the element index e < 0 (line 153), or when
      the read goes past storage.bytes (line 155)
  - callers: ZeroPoints (metal_transform.mm:~180). Other callers not searched.
- **`PrepareConstants`** — metal-microbench:metal_lower.mm:1806-2075 (whole body)
  - does: Builds one batch CompileTask of 'mesh_prepare' kernels that fill argument constants before the program runs. It
    first returns 0 if no CPU or ANE operation is live. It computes writers per value and a relation (op reads a
    value with a single writer), sorted and deduplicated, and uses mesh_graph_rows and mesh_graph_preorder.
    prepareDefinition admits a single-writer Generic producer whose outputs are unobservable, fully covered by its
    writes, and whose reads are known constants. It compiles the producer into the batch, gives the output values
    held bytes, and records materializations. demand(root) walks the graph to the needed producers. request(op,
    input, scalar, axes) builds a mesh_constant kernel that evaluates op's argument over the axis domain (groups =
    ceil(count/4096)), stores it into a new CONSTANT value, and records a Materialization. Per kernel: CPU ops
    request the left/weight argument (host.stage), and ANE ops request each used Region parameter (scalar chosen by
    ComputeType, contract element for contractions). After the loop it runs CompileTask on the batch, calls
    batch.task on the binding pointers, releases owners, and on success marks prepared roots CONSTANT (internal =
    UINT32_MAX), erases definition ops, drops matching locals and host entries, and erases kernels whose ops are all
    erased.
  - cites: comment directly above def, line 1805: "// docs/provider.md#compiled-invariant-arguments". See docs/provider.md
    below.
  - refuses:
    - LowerFail "constant argument domain overflows" at metal_lower.mm:1938 (count overflow)
    - LowerFail "constant argument storage overflows" at metal_lower.mm:1941
    - LowerFail(e.missing) at metal_lower.mm:1972 (the builder's missing-message)
    - silent return 0 (no error) at: 1809 (no CPU/ANE op live), 1930 (operation,input already tried), 1934
      (dimension has static extent), 1937 (zero bound on an axis), 1976 (binding is PAGES), 1977 (writes or
      undemanded binding), 1980 (pack false), 2049 (empty batch)
    - prepareDefinition returns false (silent, no error) at its guard sites, which begin after the `known` lambda
      near lines 1884-1915 (approximate; not individually numbered)
    - CPU/ANE branches skip silently when host.selected is false (metal_lower.mm:~2014) or scalar == UINT32_MAX
      (~2041)
  - callers: 1: metal_lower.mm:2543 (inside the enclosing lowering function, name not read).
- **`Specialize (activation with weight = 1; not in the named set, same subproblem)`** — metal-microbench:metal_native.mm:508-530 (body continues beyond 530) (part of the body)
  - does: If the projection is not dense it builds a Kernel 'mesh_activation' from the op's region. It inlines the region
    with the weight parameter replaced by e.One(type) (the constant 1), then calls Simplify on the new function and
    stores it as selected->body.
  - cites: comment directly above def, line 507: "// docs/provider.md#composed-resource-costs"
  - refuses:
    - returns false with p.error = e.missing when the builder has a missing entry (metal_native.mm:527)
  - callers: not searched
- **`_static (control and constant predicates; same subproblem)`** — mesh:rdma/mesh_compile.py:8332-8359 (whole body)
  - does: For every GraphModule reached, it runs call_function nodes whose target is A._assert_tensor_metadata.default,
    sets their meta['val'] to None. For cond and while_loop higher-order nodes it collects the data inputs (for
    while_loop only the test's read placeholders). If any such input is a non-tensor Node (a host scalar), it
    raises. It does not fold any constant predicate.
  - cites: comment above the constant-predicate text at mesh_compile.py:8323-8328 ("A constant predicate is a static input:
    its arm is taken at compilation (N. D. Jones, C. K. Gomard and P. Sestoft, 1993, Definition 4.1)"), and the
    raise message cites "design/torch-source-lowering-change-request.md item 2; PyTorch docs/source/notes/cuda.md
    L1683".
  - refuses:
    - ValueError at mesh_compile.py:8352-8354: "%s (%s): control decided by the host scalar %s, which a captured
      program cannot replay (design/torch-source-lowering-change-request.md item 2; PyTorch
      docs/source/notes/cuda.md L1683)"
  - callers: not searched
- **`_period`** — mesh:rdma/mesh_compile.py:8038-8039 (whole body)
  - does: Returns math.lcm(horizon, the lengths of all copy rings across all functions).
  - cites: none
  - refuses:
    - none
  - callers: 1 (in _specialize, mesh_compile.py:8067).
- **`_Program`** — mesh:rdma/mesh_compile.py:8279-8302 (whole body)
  - does: torch.fx.Interpreter subclass. For call_module nodes it calls self.execute(n, args, kwargs). For call_function
    nodes whose first argument is None in the env it returns None. Otherwise it defers to the base class.
  - cites: none directly above; the Jones/aot-compilation comment is at 8042-8045.
  - refuses:
    - none
  - callers: _specialize (mesh_compile.py:8089).

**Documents**

- metal-microbench:docs/provider.md: Section '## Compiled invariant arguments' (line 807): 'One call fills ordinary artifact-owned constant values,
  after which its temporary library is released. This is the same CPU target and typed arithmetic lowering used for
  numerical execution.' (lines 816-817). 'No importer expands coded weights and no per-invocation constant
  preparation is added.' (line 818-819). 'Source effects disappear only when the actual numerical bindings no longer
  read that storage root; other consumers of an alias keep their binding and effects.' (lines 824-825). 'Dense
  constant literals use their retained bytes directly instead of interpreting their scalar elements again.' (line
  831-832). Section '## Invariant producer definitions' (line 841): 'Target-demanded preparation follows unique
  defining writes through the existing operation access cache.' (line 843). 'No source decoder or scalar evaluator
  is introduced.' (line 855). Section '## Source constants before target preparation' (line 871): 'A placed meta
  tensor does not erase the source's storage role.' (line 873). 'Buffer names or a get_attr instruction alone are
  not evidence of immutability.' (line 878). 'Nested captured tensor literals also belong to
  ExportedProgram.constants.' (line 879). '_initial_values realizes the existing attribute/view/slice initialization
  once into the exact planned storage layout.' (line 889). 'Its immutable subset reaches _Builder as CONSTANT roots
  with their bytes before target lowering' (lines 890-891). 'The immutable declaration means a compiled program owns
  a snapshot of that encoded definition.' (line 896). External sources named: torch 2.14.0 torch/_export/utils.py
  _collect_and_set_constant_attrs and aten/src/ATen/native/native_functions.yaml (fresh copy semantics), lines
  ~880-884; torch 2.14.0 torch/export/graph_signature.py (InputKind.CONSTANT_TENSOR). Also cites Core ML ct.convert
  constant folding (line 829), not read as source.
  - sources it names: PyTorch 2.14.0 torch/_export/utils.py; PyTorch 2.14.0 aten/src/ATen/native/native_functions.yaml; PyTorch 2.14.0
    torch/export/graph_signature.py; Core ML ct.convert (constant folding)

**The same thing computed in more than one place**

- Simplify (metal_transform.mm:157-160) is called at four sites that each run Fold then Prune:
  metal_transform.mm:405 (Fuse), :715 (Epilogue), :1366 (Transform) and metal_native.mm:529 (Specialize).
- Constant-predicate arm selection: metal_transform.mm Fold/Select (lines 110-115, constant If condition replaced by
  the taken arm) and mesh_compile.py _static (8332-8359, which does not fold but takes constant predicates as static
  input, per the comment at 8323-8328). Both cite the same Jones et al. 1993 Definition 4.1 idea.
- Replacing a value computed from constant operands by its constant result: metal_transform.mm Fold (108-141, via
  ir::Evaluator on IR instructions); metal_transform.mm ZeroPoints (168-197, replaces a quantization zero point with
  a folded offset); metal_lower.mm PrepareConstants (1806-2075, compiles a kernel that materializes argument
  constants into bytes). Three layers, same goal.
- Immutable-constant classification: mesh_compile.py _constant_origins (6993-7003) classifies constant storages by
  subtracting written storages; mesh_compile.py _initial_values (7005-7029) classifies attribute keys as
  fixed/mutable via the immutable argument and sets ir.constant_values. Both decide which constants are immutable.
- Constant-element read: metal_ir.mm Constant (147-) reads a scalar from retained bytes for ZeroPoints.
  PrepareConstants reads the same kind of values by compiling a kernel (metal_lower.mm:1930-1990), not by a scalar
  reader.

**No function found, or only a refusal**

- Python-side folding of constant operands in the FX graph (torch constant propagation) is not in the named set;
  _specialize and _specialized handle constants by placement, not by folding. No other Python function was found in
  the searched names.
- Specialization of constant operands in metal_transform.mm beyond Fold/Select/Identity/ZeroPoints: none found in
  the read bodies.
- Folding of Core ML constants: provider.md:829 says that the constant folding of ct.convert 'was formerly supplied'
  and is now replaced by PrepareConstants; no current Core ML folding function was found by name.
- metal_ir.mm Fold (line 410, static bool Fold(const metal_ir_term&...)) has the same identifier as
  metal_transform.mm Fold but is an affine-term helper for address terms; its body was not read and it is not
  constant folding by name or by signature.
- Per-element evaluation for ZeroPoints and PrepareConstants is compiled in part by Constant and CompileTask
  (metal_lower.mm:2061, CompileTask body not read).

**Not determined**

- Callers of Identity beyond Fold, of Constant beyond ZeroPoints (metal_ir.mm:147), of Specialize
  (metal_native.mm:508), of _static, and of the enclosing function of metal_lower.mm:2543. Not searched.
- Whether ir::Evaluator (metal_transform.mm:126) can fail on any instruction type accepted by Pure: its body is not
  in this repo subset and was not read.
- Whether the mesh_compile.py _specialized kind 'constant' attributes (ir.attributes, ir.gm) come from the same
  constants as _constant_origins: the linkage is not traced through _Builder or the producer.
- The enclosing function name for metal_lower.mm:2543 and the behavior of CompileTask, mesh_graph_rows,
  mesh_graph_preorder: not read.
- Whether the design documents named by the comments (dox/mesh/design/compilation-composition-change-request.md,
  design/torch-source-lowering-change-request.md, design/aot-compilation-change-request.md) specify folding: not
  read.

## Sharding a tensor across a collective, with sends and receives

Read: 10 whole body, 1 part of the body, 1 declaration only. 12 functions.

- **`partition`** — mesh:rdma/mesh_compile.py:914-990 (whole body)
  - does: Checks that placement is a Placement. Builds IR(gm, placement, nodes, region) and sets ir.page. For each
    call_function node stores a Local and an operator rule (_operator), and sets ir.sharding[node] to all -1 for
    every non-output node with a tensor prototype, else None. Merges a pad operator into its single-user slice
    operator (_merged). Applies placement.splits as sharding overrides, checked for length and uniqueness. Runs
    mesh_graph.forward replication, then _complete, then _partition(ir, operators) (line 953). Then computes
    ir.producer, ir.sent (mesh_graph.reach), _forms, bases, ir.attributes, ir.loads, _inside, ir.places, _over,
    _kinds, _port_values. Recurses over ir.region.children (line 978) with splits copied into child placements.
    Returns ir.
  - cites: Comment at line 913: '# design/aot-compilation-change-request.md A2 and A4; GSPMD sections 3, 3.1 and 4.6'.
  - refuses:
    - line 916: TypeError 'the placement function returned %r, not a Placement (A2)'
    - lines 941-943: ValueError 'splits[%r] = %s: no tensor result\'s dims_mapping over a mesh of %d dimensions,
      each at most once (GSPMD section 3.1)' when a split names no node with a sharding, the length differs, an
      entry is outside -1..len(mesh shape), or a mesh dim repeats
  - callers: 2 in rdma/mesh_compile.py: line 8661 (inside _built) and line 978 (recursive). No callers in mesh:bin, the
    metal-microbench top level, tools or provider.
- **`_partition`** — mesh:rdma/mesh_compile.py:811-910 (whole body)
  - does: Walks ir.order. placeholder/get_attr: emit 'input' or 'constant' with whole (-1s) sharding, then _reshard from
    whole to ir.sharding[node]. output: _reshard each input value to whole, emit one 'output' item. Opaque call
    (rule.labels None): reshard operands to whole, emit 'call', reshard the result. Otherwise: target = sharding
    with factory-created labels set to -1; for each contracting label, collects the mesh dims of operands holding
    it; a single held mesh dim (only when not fused and no earlier partial) becomes a partial dim, else -1; local =
    target with partial dims set to -1; wants per operand from _carry(local, out, own) with contracted labels set;
    windowed operand dims set to -1 when the window label is whole. Each operand is _reshard to its want. For each
    sharded window dim, calls _region (line 867). For kind 'attention', writes an AttentionOrigin annotation. Emits
    'call' (line 897) and 'window' items for transposed cuts. If a partial mesh dim is in used target:
    _partition_pad to a multiple of the mesh extent, then emit 'collective' reduce-scatter (lines 903-905). Else if
    partial is non-empty: emit all-reduce (lines 907-908). If target != sharding, _reshard target to sharding (line
    910).
  - cites: Comment at lines 809-810: '# GSPMD sections 2.1, 3, 3.2, 4, 4.2, 4.5 and 5.1, Figures 6 and 7;
    design/aot-compilation-change-request.md A4 and appendix item 7; XLA spmd_partitioner.cc,
    PartitionedHlo::ReshardAsWindowedInput'.
  - refuses:
    - No raise, assert or NotImplementedError in the body (lines 811-910). The unpacking (m,) = partial at line 901
      is safe because line 843 appends to partial only while it is empty; no explicit message exists.
  - callers: 1 in rdma/mesh_compile.py: line 953 inside partition(). None elsewhere.
- **`_reshard`** — mesh:rdma/mesh_compile.py:684-748 (whole body)
  - does: Takes a tensor key node with have (current mesh dim per tensor dim, -1 = unsharded) and want. Returns (node,
    want) if (node, want) is already in ir.made. If node is placeholder/get_attr, have becomes all -1. Loop 1: for
    each mesh dim m in have but not in want, gather(have, m). Loop 2: while a mesh dim sits at a different tensor
    dim than in want, exchange (all-to-all) when the target slot is free, else gather of the blocking dim. Loop 3:
    for each m in want but not in have, _partition_pad the source to a multiple of the mesh extent (line 745) and
    emit a 'slice' item (line 746). gather: copies the local buffer to an 'own' buffer and emits an all-gather over
    mesh dim m (line 703) into a buffer of shape[d]*P (a 'gather' buffer when it differs from the target), then
    emits 'narrow' to the true size (line 705). exchange: pads the source along j to ceil(shape[j]/P)*P (line 715),
    emits an all-to-all with Collective('all-to-all', node, (m,), (i, j), ...) (lines 723-724), and narrows along i
    (line 726). step() emits an item only if its output key is not already in ir.made.
  - cites: none on this function. The comment at lines 669-672 sits above _partition_pad (line 673), not above _reshard.
  - refuses:
    - No raise, assert or NotImplementedError in the body (lines 684-748). Extents are not checked for divisibility;
      padding is by multiplication (line 695) and ceil division (lines 715, 745).
  - callers: 6 call sites, all in rdma/mesh_compile.py _partition: lines 820, 824, 831, 835, 859, 910. None elsewhere.
- **`_partition_pad`** — mesh:rdma/mesh_compile.py:673-681 (whole body)
  - does: If shape[dimension] equals extent, returns key. Otherwise made = key + ('padded', dimension, extent); if made is
    not in ir.made, records ir.shapes[made] with that dimension set to extent and emits ('pad', (key,), made,
    (dimension, extent)). Returns made. The function does not set the pad's contents.
  - cites: Comment at lines 669-672: '# GSPMD section 4.5; MPICH allgather_intra_ring.c and alltoall_intra_pairwise.c,
    MPIR_Localcopy; design/aot-compilation-change-request.md, appendix item 7: a mesh dimension the result does not
    shard is gathered; one it shards along another dimension moves there by an AllToAll once that dimension is free
    ...'.
  - refuses:
    - No raise, assert or error return in the body (lines 673-681). It does not check that extent is at least
      shape[dimension]; callers supply extent as ceil(n / P) * P (lines 715, 745, 903).
  - callers: 3 in rdma/mesh_compile.py: line 715 (_reshard.exchange), line 745 (_reshard), line 903 (_partition). None
    elsewhere.
- **`_region`** — mesh:rdma/mesh_compile.py:768-806 (whole body)
  - does: For a window dim on mesh dim m with o outputs per part: computes starts, length, after, left, right via _extent
    (line 772). If whole, recomputes left and right as the max overhang over all parts (line 774). Sets the region's
    shape. If left or right is nonzero: copies key to a 'middle' buffer (line 783), and for each side and halo step
    j = 1..ceil(width/n) narrows to a 'halo' piece (line 791) and emits a 'collective-permute' with pairs (p, p+j)
    for left or (p+j, p) for right (lines 792-794), storing 'arrival'. Builds a 'landed' buffer of extent
    left+n+right along dim, records ir.inside, and emits a 'window' item with (dim, m, offsets, length, starts,
    size, fill) (line 805). fill is window.fill when any start is negative or start+length exceeds size (line 777).
    Returns (region, after).
  - cites: Comment at lines 766-767: '# GSPMD sections 4.1, 4.3 and 4.5, Appendix A.2, Figures 5a and 9b;
    design/aot-compilation-change-request.md, appendix item 7; design/plan-to-bridge-change-request.md R1; MPICH
    allgather_intra_ring.c, MPIR_Localcopy'.
  - refuses:
    - No raise, assert or NotImplementedError in the body (lines 768-806).
  - callers: 1 in rdma/mesh_compile.py: line 867 inside _partition. None elsewhere.
- **`_extent`** — mesh:rdma/mesh_compile.py:752-763 (whole body)
  - does: Arithmetic only. If window.transposed is None: starts[q] = q*o*stride - low, length = (o-1)*stride + extent,
    after = [0]*parts. Otherwise uses (padding, reach) and stride s: starts[q] = (q*o + padding - reach) // s,
    after[q] = q*o + padding - a*s, length = the maximum covering count with a floor of 1. Returns (starts, length,
    after, left, right), where left = max over q of (q*n - starts[q]) and right = max over q of (starts[q] + length
    - (q+1)*n), each floored at 0.
  - cites: Comment at line 751: '# GSPMD Appendix A.2 and Figure 9b'.
  - refuses:
    - No raise, assert or error return in the body (lines 752-763).
  - callers: 1 in rdma/mesh_compile.py: line 772 inside _region.
- **`IR.local`** — mesh:rdma/mesh_compile.py:197-198 (whole body)
  - does: Returns, per tensor dim of _value(node).shape, ceil(s / self.shape[m]) when the sharding entry m is >= 0, else
    s.
  - cites: Comment at line 196: '# GSPMD section 3.1'.
  - refuses:
    - No raise in the body. _value (line 230) raises: line 233 NotImplementedError 'no tensor value', line 236
      ValueError 'symbolic sizes', line 239 NotImplementedError 'no binding of its tensors to the placed pages'.
  - callers: About 10 sites in rdma/mesh_compile.py (lines 690, 694, 698, 712, 720, 745, 760, 854, 860, 866, 875), within
    _reshard, _partition and _region.
- **`_value`** — mesh:rdma/mesh_compile.py:230-241 (whole body)
  - does: Returns the node's tensor prototype from _prototype, after the checks below.
  - cites: none in the function; the comment at lines 221-223 cites J. Ansel et al., PyTorch 2, ASPLOS 2024, sections 3.5
    and 5.1, and torch 2.14.0 FakeTensor.
  - refuses:
    - line 233: NotImplementedError '%s (%s): no tensor value (torch.export: its meta[\'val\'])'
    - line 236: ValueError '%s: symbolic sizes %s; A1 compiles shapes, so nothing at run time changes one'
    - line 239: NotImplementedError '%s on %s: no binding of its tensors to the placed pages (a DLPack tensor of
      kDLCPU or kDLMetal alone: K1)'
  - callers: Used throughout rdma/mesh_compile.py, including by IR.local and _partition (lines 893, 896).
- **`_shape`** — mesh:rdma/mesh_compile.py:245-246 (whole body)
  - does: Returns ir.shapes[key] if present, else ir.local(key[0], key[1]).
  - cites: Comment at line 244: '# GSPMD section 3.1'.
  - refuses:
    - none in the body
  - callers: About 5 sites in rdma/mesh_compile.py: lines 674, 769, 903 and within _partition.
- **`_mesh`** — mesh:rdma/mesh_compile.py:262-288 (whole body)
  - does: Builds the mesh shape, a device-to-coordinate map and the node list from placement.mesh and placement.nodes.
    Used by IR.__init__ (line 167) to set ir.shape, ir.coords and ir.map.
  - cites: Comment at lines 260-261: 'GSPMD section 3.1; Alpa section 4.1; design/aot-compilation-change-request.md A10'.
  - refuses:
    - line 273: ValueError 'the placement operand\'s mesh holds an empty sequence (A2)'
    - line 279: ValueError 'the placement operand\'s mesh at %s: %r is no new device id (A2)'
    - line 282: ValueError 'the placement operand\'s mesh at %s is not of shape %s (A2)'
    - lines 286-287: ValueError 'the placement operand: devices %s onto map indices %s ... each device 0 .. D - 1
      once, each onto its own index (A2)'
  - callers: 1 in rdma/mesh_compile.py: line 167 (IR.__init__).
- **`_merged`** — mesh:rdma/mesh_compile.py:636-644 (part of the body)
  - does: Merges a pad operator's padding and a slice operator's slicing into one operator: its windows and configure
    combine both, and the merged result replaces the slice. Lines 640-644 were read; lines 636-639 were not.
  - cites: none in the lines read.
  - refuses:
    - none in the lines read
  - callers: 1 in rdma/mesh_compile.py: line 929 in partition().
- **`partition_bounds (commit be5d8920 only, not at HEAD)`** — commit be5d8920, rdma/allocate.py, added 'def partition_bounds(parts, extent)' (git show be5d8920 -- rdma/allocate.py); absent from HEAD (declaration only)
  - does: At be5d8920: returns tuple(extent * prefix // total for prefix in itertools.accumulate(parts, initial=0)), exact
    adjacent integer cuts weighted by rank shares. Its docstring says 'Integer prefix rounding preserves coverage,
    including tails and empty shares.' Used at be5d8920 by rdma/migration.py _geometry, which HEAD no longer has.
  - cites: none (docstring only).
  - refuses:
    - none
  - callers: At be5d8920: rdma/migration.py (1). At HEAD: none.

**Documents**

- mesh:design/heterogeneity.md: Line 64, R6: 'Tensors that meet in one operation split their shared extent alike.' Line 66, R8: 'Equal counts run
  the regular collective. Unequal counts run the v-collective with each block at its own size, never padded to the
  largest.' Line 72, R14: 'Parts change without shapes changing. Each rank's local buffer is shaped to the largest
  share the bounds allow it (`allocate.capacity`) ... Padding holds the identity of its next consumer ... Kernels
  are stock: they compute garbage in padding that nothing reads unmasked.' Lines 15 and 177 cite veScale RaggedShard
  (pytorch#169320).
  - sources it names: Zhang et al. 2024; veScale RaggedShard (pytorch#169320); Megatron-LM 2019 section 3; HAP 2024 sections 2.4 and
    2.5.1; pytorch#198344; MPI 4.1 sections 6.5-6.10; Hunold & Carpen-Amarie 2017; GSPMD 2021 section 3.3
- mesh:design/already-solved-subproblems-2026-11-09.md: Line 108: 'sharding a tensor across a collective, with sends and receives | rdma/mesh_compile.py: partition,
  _partition, _reshard (GSPMD is cited there) | 11-04 | the frontend'.
  - sources it names: GSPMD (cited in mesh_compile.py)
- mesh:design/compilation-coordination.md: Line 1552: 'The partitioner inserts neither collective (code fact: _reshard, "no CollectivePermute arises";
  _partition replicates a contracting dimension the output's mesh dimension holds, where GSPMD section 5 "replace[s]
  the AllReduce on the output BSM with a ReduceScatter")'. Line 2939: 'placed followed _reshard's and _region's
  alone' (OV-3, view operators).
  - sources it names: GSPMD section 5

**The same thing computed in more than one place**

- Pad a tensor dimension up to a multiple of the mesh extent, ceil(n / P) * P: rdma/mesh_compile.py:715
  (_reshard.exchange source pad), :745 (_reshard slice branch) and :903 (_partition reduce-scatter source pad).
- Shard extent ceil(n / P) of a dimension split over mesh dim m: rdma/mesh_compile.py:198 (IR.local, general form),
  :717 and :720 (_reshard.exchange, new layout and 'own' buffer); and the inverse in _reshard.gather at :694-696
  (padded = local * P, size = local extent of after).
- Pad-then-slice or pad-then-collective over the same dim: :745-746 (_reshard emits 'slice' after pad) and :903-905
  (_partition emits reduce-scatter after pad).

**No function found, or only a refusal**

- Padding contents (what the padded region holds, zero or identity for a reduction): _partition_pad (673-681) emits
  a 'pad' item with (dimension, extent) only. The identity masking that design/heterogeneity.md R14 (line 72)
  describes is not found in the bodies read (_reshard, _region, _partition, _partition_pad, _extent).
- Unequal-count (v-variant) collectives with per-block sizes: no function read emits a per-rank size list. All
  collectives use one regular shape (Collective kinds 'all-gather', 'all-to-all', 'collective-permute',
  'reduce-scatter', 'all-reduce'). design/heterogeneity.md R8 (line 66) says 'Unequal counts run the v-collective
  with each block at its own size, never padded to the largest.'
- Validation of a ragged input extent for placeholder/get_attr: _partition (lines 814-820) emits the input and calls
  _reshard; no raise on the input's extent is found in the bodies read.
- Lowering of the emitted collectives into runtime calls is outside this subproblem. Lines 1468 ('a
  CollectivePermute's pairs ... over a group of %d') and 1525 ('%s of %s: no lowering (A-T3\')' NotImplementedError)
  in rdma/mesh_compile.py were seen only as grep matches, not read.

**Not determined**

- Whether HEAD's ragged-tail path matches what the commit messages describe: be5d8920 ('Lower ragged tails through
  shared exact partition boundaries') changed rdma/torch-mesh/torch_mesh/partition.py (286 lines), which was not
  read.
- 444d4fd9 ('Input domains are ragged relations') concerns the streamed/structured Hadamard relations in
  rdma/torch-mesh, not mesh_compile.py. Its hunks were not read (stat only).
- 27b8b77c ('Restore solved artifacts and preserve ragged typed partitioning') changed rdma/mesh_compile.py by 640
  lines per stat; its hunks were not read, so which functions above it introduced or changed is unknown.
- design/operand-contract.md, which be5d8920 and 444d4fd9 modify, is not in /Users/mdot/dox/mesh (find returned
  nothing), so it was not read.
- Whether metal-microbench has its own copy of the mesh partitioner: grep for mesh_compile, partition_bounds and
  _reshard (--include=*.py) over /Users/mdot/metal-microbench returned no files.

## The scheduling solve

Read: 14 whole body. 14 functions.

- **`mesh_graph_order`** — mesh:rdma/mesh-graph.c:37-48 (whole body)
  - does: Zeroes indegree and ordinal. Counts indegree[head(k)] over all row entries. Then for place = 0..vertices-1:
    scans for the lowest-index vertex with indegree 0 and no ordinal yet (line 42); if none, returns EDEADLK (line
    44); sets ordinal[next]=place and decrements indegree of every head of next's row (line 46).
  - takes: vertices; first[] (row starts, size vertices+1); head(context,k) row entries; indegree[] and ordinal[] outputs
  - returns: 0, or EDEADLK
  - cites: none (no comment at the definition; header mesh-graph.h:11-13 has no citation)
  - refuses:
    - mesh-graph.c:44 return EDEADLK when no unplaced vertex has zero indegree (a cycle)
  - callers: 3 call sites: mesh-collective.c:323 (inside plan_acyclic, mesh-collective.c:311); mesh-graph.c:233 (inside
    mesh_graph_depth, mesh-graph.c:225); mesh-graph.c:857 (inside mesh_graph_chains, mesh-graph.c:821).
- **`mesh_graph_levels`** — mesh:rdma/mesh-graph.c:1885-1901 (whole body)
  - does: Groups edges by to[] into rows with mesh_graph_rows(vertices, edges, depth_head, to, first, order). For v
    ascending: level[v]=0, then for each in-edge with tail u=from[order[k]]: if u>=v it is an error, else
    level[v]=max(level[v], level[u]+1). So level is the longest-path depth assuming every edge has from<to.
  - takes: vertices; edges count; from[], to[] (dependence e: from[e] -> to[e]); level[] output
  - returns: 0 or EINVAL/ENOMEM/EDEADLK
  - cites: Comment above mesh_graph_schedule, not above this function: mesh-graph.c:1884
    "design/torch-source-lowering-change-request.md item 3; M. Cowan et al., MSCCLang, ASPLOS 2023, §5.2"; header
    mesh-graph.h:199-200 cites design/torch-source-lowering-change-request.md item 3 and
    design/compilation-composition-change-request.md item 6.3.
  - refuses:
    - mesh-graph.c:1886 EINVAL when any from[e] or to[e] >= vertices
    - mesh-graph.c:1887 ENOMEM when vertices==UINT32_MAX or edges too large
    - mesh-graph.c:1889 ENOMEM when malloc of first/order fails
    - mesh-graph.c:1895 EDEADLK when an edge has from >= to (not a cycle test in general, a test of index order)
  - callers: 2: mesh-graph.c:1905 (mesh_graph_schedule); metal_ir.mm:1903 (Order).
- **`mesh_graph_schedule`** — mesh:rdma/mesh-graph.c:1903-1912 (whole body)
  - does: Calls mesh_graph_levels into level[]. Allocates priority[v] = -(double)level[v] for every v (line 1907 ENOMEM on
    failure). If no error, calls mesh_graph_heap_order(vertices, priority, edges, from, to, place). Frees priority.
  - takes: vertices; edges; from[], to[]; level[] and place[] outputs
  - returns: int status
  - cites: mesh-graph.h:199-201 "levels and places over the caller's dependence relation (from[e] < to[e]); level the
    longest path to v, place the heap order by -level (mesh_graph_heap_order)", citing
    design/torch-source-lowering-change-request.md item 3 and design/compilation-composition-change-request.md item
    6.3; mesh-graph.c:1884 cites MSCCLang §5.2.
  - refuses:
    - mesh-graph.c:1907 ENOMEM when priority malloc fails; otherwise errors from mesh_graph_levels or
      mesh_graph_heap_order propagate in result
  - callers: none found (only the declaration mesh-graph.h:202 and the definition; mesh-graph.h:98 mentions it in a comment).
- **`mesh_graph_heap_order`** — mesh:rdma/mesh-graph.c:1222-1252 (whole body)
  - does: Groups edges by from into rows (mesh_graph_rows(vertices, edges, depth_head, from, first, order)).
    indegree[to[e]] counted per edge. Pushes every zero-indegree vertex into a binary max-heap ordered by
    order_before (higher priority first; equal priority, smaller index first; mesh-graph.c:1216-1219) using heap_up.
    Loop: pop heap[0] as v, place[v]=placed++, for each out-edge of v decrement indegree of its head and push it
    when it reaches 0. Ends with EINVAL if placed < vertices (a cycle).
  - takes: vertices; priority[v] (double); edges; from[e], to[e]; place[v] output (a permutation of 0..vertices-1)
  - returns: 0, EINVAL, or ENOMEM
  - cites: mesh-graph.c:1215 "design/plan-to-bridge-change-request.md R5, 'The order of each cable's list, by the reference
    algorithm'"; mesh-graph.c:1220-1221 "M. Cowan et al., MSCCLang, ASPLOS 2023, §5.2, step 3;
    design/plan-to-bridge-change-request.md R5; F. B. Schneider, ACM Comput. Surv. 1990"; header mesh-graph.h:89-90
    cites the same MSCCLang section.
  - refuses:
    - mesh-graph.c:1224 EINVAL when from[e] or to[e] >= vertices
    - mesh-graph.c:1225 ENOMEM when edges >= SIZE_MAX/sizeof(size_t)
    - mesh-graph.c:1230 ENOMEM when a malloc fails
    - mesh-graph.c:1249 EINVAL when placed < vertices (cycle; not all vertices reached)
  - callers: 7 call sites in 5 files: mesh-collective.c:936 (plan_schedule); mesh-graph.c:1262 (mesh_graph_heap_cuts,
    mesh-graph.c:1257); mesh-graph.c:1909 (mesh_graph_schedule); mesh-graph.c:3088 (inside mesh_graph_space_time,
    mesh-graph.c:3073, which turns EINVAL into EDEADLK at 3088); metal_ir.mm:1929 (Order) and metal_ir.mm:1975
    (ReplayOrder); plus the Python wrapper mesh_graph.py:819-824 (heap_order), which has no callers found in
    rdma/*.py.
- **`mesh_graph_completion_times`** — mesh:rdma/mesh-graph.c:1104-1220 (end of function after the free() line and return result) (whole body)
  - does: Builds a HiGHS LP (Highs_create, simplex, one thread, minimize): variables C[v] with lower bound length[v], plus
    one extra column M (the makespan, cost 1). Rows: C[v]-M... (for each v a row with coefficients +1 on M and -1 on
    v, lower bound 0) and, for each distinct edge (i,j) sorted by qsort, C[j]-C[i] >= length[j]. Then an iterative
    loop: solve; per resource r, collect members with length>0, sort by LP value column[j], accumulate prefix sums
    of length and squares, and for every prefix whose (sum^2+squares)/2 minus the weighted left term exceeds
    tolerance, add a cut row (the Queyranne-type inequality) for the most-violated prefix. When no cut is violated
    on the first pass, it fixes M to its value, sets all lower bounds to 1 and all costs to 1 (sum of completion
    times) and re-solves once more; a second empty pass ends the loop. Finally copies column[0..vertices-1] to
    completion.
  - takes: vertices; length[v] (lower bounds, >=0); resource[v] in [0,resources); edges; from[e], to[e]; completion[v]
    output
  - returns: 0, EINVAL (infeasible model or bad input), EDOM (solver error or non-optimal status), ENOMEM
  - cites: mesh-graph.c:1089 "design/plan-to-bridge-change-request.md R5, 'How: the program' and 'How: solving it'";
    mesh-graph.c:1100-1103 "C.-Y. Chen, arXiv 2205.02474, 2022, (1), (1b) to (1e), (14) and (14a); L. A. Hall, A. S.
    Schulz, D. B. Shmoys and J. Wein, Math. Oper. Res. 22(3), 1997, (1), (3) and (5); M. Queyranne, Math.
    Programming 58, 1993; Q. Huangfu and J. A. J. Hall, Math. Program. Comput. 10(1), 2018; H. Isermann, OR Spektrum
    4, 1982; HiGHS 1.15.1, highs_c_api.h and HighsOptions.h"; header mesh-graph.h:86-87 "design/joint-compilation.md
    §3, row 8".
  - refuses:
    - mesh-graph.c:1107 EINVAL when length[j] is not in [0, DBL_MAX] or resource[j] >= resources
    - mesh-graph.c:1109 EINVAL when from[e] or to[e] >= vertices or from[e]==to[e]
    - mesh-graph.c:1112 ENOMEM when room >= INT32_MAX/2
    - mesh-graph.c:1135 EDOM when a HiGHS option or objective setting fails (inside the options block)
    - within the while loop (mesh-graph.c ~1140-1200): EDOM on Highs_run error, EINVAL when model status is
      infeasible, EDOM when status is not optimal, ENOMEM when realloc fails, EDOM on Highs_getSolution error or
      addRows/changeCol errors
  - callers: 1 call site: mesh-collective.c:927 (plan_schedule).
- **`mesh_graph_list_schedule`** — mesh:rdma/mesh-graph.c:520-537 (whole body)
  - does: Zeroes carried[]. For each unit k in index order: among machines with rate>0, picks the one minimizing
    (carried[l]+bytes[k])/rate[l] (first minimum kept), sets machine[k] to it and adds bytes[k] to carried[best]. If
    no machine has rate>0, machine[k]=UINT32_MAX and a lost flag is set. Returns INFINITY if lost, else the maximum
    carried[l]/rate[l].
  - takes: machines; rate[l] (>0 usable); units; bytes[k] per unit; machine[k] and carried[l] outputs
  - returns: double: the largest carried[l]/rate[l] over machines, or INFINITY
  - cites: mesh-graph.c:520 "design/page-queue-change-request.md, 'The rule both ends compute'"; mesh-graph.h:57-58
    "design/joint-compilation.md §3, row 10"; mesh-flow.c:682 names it as "Graham's list scheduling, mesh-graph.h".
  - refuses:
    - mesh-graph.c:535 returns INFINITY (no error code) when any unit had no machine with rate>0
  - callers: 1 call site: mesh-flow.c:858 (inside link_prepare, mesh-flow.c:796), which returns EHOSTUNREACH on isinf.
    Comments only: mesh-flow.c:682 and 725.
- **`plan_schedule`** — mesh:rdma/mesh-collective.c:807-946 (static) (whole body)
  - does: Builds an instruction graph: demand-produce tasks and optional reader tasks; consumer tasks per call (one per
    page for MESH_PAGES); a route per tree path (mesh_graph_climb) giving link ids and communication ids;
    per-instruction weight w[] (produced/read/consumed weight, transfer occupancy cost[e]/rate[link], rate =
    capacity/max capacity) and resource tag (processor n, or link n+link). Dependence edges before[]/after[] include
    route hops and declared dependences. Calls mesh_graph_completion_times for the LP completion times; sets
    p->completed[i] to the max completion over call i's consumer tasks; negates completion and calls
    mesh_graph_heap_order to get place[]. Finally sets each entry's place from its communication task's place and
    sorts entries by plan_entry_order (comparator, mesh-collective.c around line 800, not fully read).
  - takes: struct plan *p; weight[]; dependence[] and count; reads p->demand, p->of, p->run, p->tree, p->parent, p->pairs,
    p->capacity, p->from, p->to, p->comms, p->entries
  - returns: int status (0 or ENOMEM); writes p->entry[j].place and sorts p->entry; sets p->completed[i]
  - cites: mesh-collective.c:803-806 "design/plan-to-bridge-change-request.md R1, R3 to R5;
    design/aot-compilation-change-request.md A3, A6; O. Sinnen, Task Scheduling for Parallel Systems, Wiley, 2007;
    O. Sinnen and L. A. Sousa, IEEE TPDS 16(6), 2005, Definitions 1, 2 and 4; M. Cowan et al., MSCCLang, ASPLOS
    2023, §4.2, §5.2; F. B. Schneider, ACM Comput. Surv. 22(4), 1990"; further design citations at lines 857, 919,
    926, 940 (design/plan-to-bridge-change-request.md R5).
  - refuses:
    - mesh-collective.c:814 ENOMEM when reader/consumer allocation fails
    - mesh-collective.c:819, 828, 837, 840 ENOMEM when task, entry, edge or instruction counts reach UINT32_MAX
    - mesh-collective.c:855 ENOMEM when any of the instruction arrays fails to allocate
    - mesh-collective.c:931 ENOMEM when the completed array fails to allocate
    - status from mesh_graph_completion_times and mesh_graph_heap_order is returned unchanged
  - callers: 1 call site: mesh-collective.c:1945 (inside mesh_plan, mesh-collective.c:1846), guarded by p->demands.
- **`plan_graph`** — mesh:rdma/mesh-collective.c:1115-~1345 (static; ends with `done:` cleanup and return status) (whole body)
  - does: Builds a program graph over 2J call vertices plus per-node part vertices (actor[j]); edges for submission order,
    per-node chains, outstanding lags (mesh_graph_rows over plan_at keys), transfers (child[] per demand), given
    orders (orders[] with plan_ordered), and chains (plan_chain). Computes cycle mean lambda
    (mesh_graph_cycle_mean), converts to a synchronization graph (mesh_graph_convert_sc), determines delays
    (mesh_graph_determine_delays) and appends added edges. Builds wait and tell lists per node per plan
    (mesh_graph_rows over wait keys), a plan_signature, and room/submit_room lags via mesh_graph_lags. The place[]
    array (dispatch position) is recomputed from the order at several points.
  - takes: struct plan *plans; count; calls[] and ncalls (J); program; self; row[]/order[]/orders[]/causes[]/norders (given
    orders); chains[]; struct mesh_planned *planned
  - returns: int status; fills planned (call[], order[] and cause[] for this node, calls, signature) and per-plan sync,
    wait_row, waits, tells, room and submit_room fields
  - cites: mesh-collective.c:1109-1110 "design/compilation-composition-change-request.md item 6, unit 6D";
    mesh-collective.c:1177 "design/aot-compilation-change-request.md A10"; mesh-collective.c:1099
    "design/aot-compilation-change-request.md A10; A. J. Bernstein, 1966; M. C. Rinard and P. C. Diniz, 1997, §2";
    mesh-collective.c:1240-1243 "Bernstein's order between calls ... as BSL §4.1 closes a processor's order";
    mesh-collective.c:1258 "design/compilation-composition-change-request.md item 6, unit 6D";
    mesh-collective.c:1276-1277 "design/compilation-composition-change-request.md item 6, unit 6A ... BSL Fig. 8's
    lambda"; mesh-collective.c:1322 cites plan_signature; mesh-collective.c:1313 cites S. S. Bhattacharyya, S.
    Sriram and E. A. Lee, IEEE Trans. Signal Processing 45(6), 1997, (5).
  - refuses:
    - mesh-collective.c:1129 EINVAL when a given order cause is not MESH_ORDER_FLOW or MESH_ORDER_CONFLICT
    - mesh-collective.c:1141 EINVAL when an order pair has a >= b or b >= J
    - mesh-collective.c:1146 EINVAL when a chain head h > j or chain of the head is not itself
    - mesh-collective.c:1150 ENOMEM when vertices/edges/allocation bounds overflow
    - mesh-collective.c:1175 ENOMEM when any allocation fails
    - mesh-collective.c:1201 ENOMEM when task allocation fails
    - mesh-collective.c:1262 ENOMEM when planned arrays fail
    - mesh-collective.c:1279 ENOMEM when cycle mean is INFINITY
    - mesh-collective.c:1282 EDEADLK or ERANGE from convert_sc/determine_delays is mapped to ENOMEM
    - mesh-collective.c:1288 ENOMEM when an added delay plus offset overflows
    - lags loop near the end of the function: ENOMEM when lags[0] is UINT64_MAX or 0 or the lag for a here node is
      UINT64_MAX or 0
  - callers: 1 call site: mesh-collective.c:1926 (inside mesh_plan, mesh-collective.c:1846). Comments only:
    mesh-records.c:920 and 1024, mesh-flow.c:777, mesh-collective.c:352.
- **`Choose`** — metal-microbench:metal_native.mm:317-369 (whole body)
  - does: Computes weightBytes = KS*NS*2*bits. Sets spl (2 if bits==2 and rows<=4, else 1) and nsg=4. Chooses target
    threadgroup count from weightBytes/cores (small: max(2*cores,64); otherwise 4*cores or max(4*cores,128) under 4
    rows). Halves lanes from 32 while blocks < wanted (wanted depends on rows and family). Reduces nsg to 2 when
    lanes do not fit. Returns false if depth(KS) < 96 (line 339). Otherwise computes parts (capped by
    depth/(workers*unroll) and 32), piece, reduceRow, chunk, and fills every NativePlan field (rows, bits, lanes,
    slicesPerLane, simdgroups, parts, blocks, piece, unroll, chunk, groups[], threads[], memory, partialBytes,
    counterBytes, slices, span). It is a closed-form plan, not a search over orderings.
  - takes: const Projection &x (fields R, bits, NS, KS); uint32_t cores; uint32_t family; NativePlan &plan (output)
  - returns: bool: true with plan filled, false when the contraction is too shallow
  - cites: metal_native.mm:314-316 "The retained plan (native_projection's measured choices, metal_native.m at 69f99dc7)";
    metal_native.mm:322-323 "tools/native_bench NATIVE_COLD_CODE, both GPUs"; metal_native.mm:337-338 "measured cold
    on both GPUs: the PLE projection". No paper cited.
  - refuses:
    - metal_native.mm:339 `if (depth < 96) return false;` (refusal by return value, no message; the comment at
      337-338 says the threshold is 128 K slices (512 inputs))
  - callers: 1 call site: metal_native.mm:1184, inside MapTuned (metal_native.mm:1173).
- **`Order`** — metal-microbench:metal_ir.mm:1900-1933 (static void) (whole body)
  - does: Calls mesh_graph_levels over the dependence pairs into p.levels. Builds cut points orderCuts = {0, count} plus
    each un-owned control dispatch's first/middle/last, each control entry's writer and writer+1, and d and d+1 for
    every METAL_IR_CHECK dispatch; sorts and dedups. priority[d]=-1 for each later pass's later dispatch and each
    ARRIVED or SENT dispatch, else 0. For each consecutive cut segment [first,last): collects dependence edges with
    both ends inside the segment (rebased to first), calls mesh_graph_heap_order, and writes
    p.order[first+place[d-first]]=d.
  - takes: Program &p (dispatches, dependenceFrom/To, controlDispatches, controlOwners, controlEntries, later_passes,
    dispatches[].kind)
  - returns: void; sets p.levels, p.orderCuts, p.order, or p.error
  - cites: none at the function (line 1899 comment belongs to ControlFeedback, metal_ir.mm:1880
    "docs/provider.md#control-feedback-and-guarded-operands" is ControlFeedback's).
  - refuses:
    - metal_ir.mm:1904 sets p.error "the compiled dispatch levels: <strerror>" on mesh_graph_levels failure and
      returns
    - metal_ir.mm:1930 sets p.error "the compiled dispatch order: <strerror>" on mesh_graph_heap_order failure and
      returns
  - callers: 1 call site: metal_ir.mm:2055, inside Artifact (metal_ir.mm:1982).
- **`ReplayOrder`** — metal-microbench:metal_ir.mm:1942-1980 (static int) (whole body)
  - does: replayOrder = order copy; orderTimes zeroed. If no INDIRECT (movable) dispatch exists, returns 0 (identity).
    Requires orderParts/Earliest/Latest sizes equal count (else ENODATA). rank[order[at]]=at. For each movable later
    pass: completion = max over its origins of received input.time (EINVAL if not finite);
    orderTimes[later]=completion; priority[rank[later]] = -1 - completion. For each cut segment: edges from the
    dependence pairs with both ends in the segment (rebased), plus a chain of consecutive non-movable dispatches in
    order; calls mesh_graph_heap_order; writes replayOrder[first+place[at-first]] = order[at].
  - takes: Program &p (order, orderCuts, orderParts, orderEarliest, orderLatest, later_passes,
    values[].received[].input.time, dependenceFrom/To, dispatches movable)
  - returns: 0 or an error code; sets p.replayOrder and p.orderTimes
  - cites: none
  - refuses:
    - metal_ir.mm:1948 ENODATA when orderParts/orderEarliest/orderLatest sizes differ from count
    - metal_ir.mm:1957 EINVAL when a received input.time is not finite
    - metal_ir.mm:1976 returns the mesh_graph_heap_order status
  - callers: 1 call site: metal_ir.mm:3317, inside metal_ir_finalize (defined before line 3317, metal_ir.mm ~3311).
- **`MetalOrder`** — metal-microbench:metal_recording.m:374-437 (static void) (whole body)
  - does: Reads the compiled order (D), levels and earliest/latest bounds. Walks positions pos in order: an entry record
    goes to tasks, a later record to laters[l] (updates lastLater), otherwise a recorded dispatch (barrier set when
    the level differs from the previous dispatch's). For guards and bindings, extends each MetalStandIn's
    value.begin/end by [begin,end] at pos. Counts crossings (publishing or received-writing commands) and the
    longest local stretch between them. Sets capture->levels = max(D,1) and gives unranged values [0, levels-1].
    printf reports counts.
  - takes: MetalCapture *capture (artifact order and levels, program bounds, dispatches, tasks, guards, bindings, values,
    laters, boundaries)
  - returns: void; fills capture->starts, capture->parts, capture->levels, MetalStandIn value ranges; prints statistics
  - cites: metal_recording.m:373 "../dox/mesh/design/compilation-composition-change-request.md#requirements";
    metal_recording.m:786 comment refers to MetalOrder.
  - refuses:
    - metal_recording.m:380 NSInvalidArgumentException "The compiled order and levels do not cover every dispatch"
      when D, levels or bounds sizes disagree
  - callers: 1 call site: metal_recording.m:1151, inside Objective-C method `- (id)prepare:` (metal_recording.m:827).
- **`MetalPlannedOrder`** — metal-microbench:metal_recording.m:2782-2837 (static NSArray *) (whole body)
  - does: For each planned order edge with both endpoint calls having rows: maps the endpoint to a row (lastRow for odd
    processors, row otherwise); unrolls each edge into lag copies (delay d gives copy c from c-d), placing work
    parts at c*P+at[] and submission parts at parts+c*count+. Builds a keep[] set (submission vertices, plus work
    rows' parts when MetalWork) and calls mesh_graph_contract to contract the unrolled graph onto those vertices.
    Splits the contracted edges: work-to-work edges go to tail/head; work-to-submission edges with delay become
    start waits (only for the last copy); edges from a submission are dropped.
  - takes: planned (mesh_planned: orders, order[], signature tasks/label); row[], lastRow[] (call -> row), calls, count,
    at[], P (positions), lag
  - returns: NSArray @[tail, head, starts]: supplied order edges (position pairs) and start waits (from, delay)
  - cites: metal_recording.m:2770-2780 comment cites mesh-graph.h mesh_graph_contract; no design document cited in the
    lines read.
  - refuses:
    - metal_recording.m:2792 NSInvalidArgumentException "Plan order edge %zu: vertex %u or %u names no call of %u"
      when a vertex or call is out of range
    - metal_recording.m:2816 NSMallocException "The plan's order at this node: %s" when mesh_graph_contract fails
    - metal_recording.m:2826 NSInvalidArgumentException "Plan order: row %u starts after position %u's work of the
      same execution"
    - metal_recording.m:2832 NSInvalidArgumentException "Plan order: position %u's work at copy %u follows position
      %u's at copy %u"
  - callers: 1 call site: metal_recording.m:3032, inside MetalProgramPrepare.
- **`MetalProgramPrepare`** — metal-microbench:metal_recording.m:2839-3159 (static void; 321 lines, read in two parts 2839-3000 and 3000-3160) (whole body)
  - does: (1) Builds the position list: per call row k one position, plus one per range of its sent records; at[k] is row
    k's first position; residues = least period over rows (replay footprint count, or residues for native rows). (2)
    Window lag = 2*residues unless stepped. (3) For each call's WHILE loops: computes copies, binds predicate and
    guard words per copy, builds the loop table (start, body, end, control, parents, writers per phase), qsorts it
    with MetalLoopOrder, links parents, and builds loopRows/loopActions with mesh_graph_rows. (4) If saved: restores
    fences and waits from the dictionary after checks. (5) Otherwise: builds activation scope and parent arrays;
    planOrder = MetalPlannedOrder(...); adds supplied relation edges from each replay capture's dependences
    (metal_ir_dependences projected onto actors); for each residue r, collects accesses (MetalPositionAccesses),
    calls mesh_graph_dependences_views, dedups pairs, calls mesh_graph_synchronization_scoped to drop redundant
    edges, builds MetalNewest and per-position wait rows with mesh_graph_rows. (6) For each phase in [0,
    warm+period): for each position, start waits from planOrder[2] set signals and startWaits; for work rows
    mesh_graph_delayed yields (source, slot) waits, each becoming a send-completion wait, a queued fence wait
    (fences[source][slot]) or a host wait. (7) Stores results, prints counts, calls MetalProgramEvents.
  - takes: MetalProgram *program (calls, count, lag, stepped); id<MTLDevice> device; const struct mesh_planned *planned;
    NSDictionary *saved (a restored program's solved synchronization, or NULL)
  - returns: void; sets program->order, positions, residues, window, period, warm, loops, loopWords, loopRows, loopActions,
    fences, waits, fenceLists, hostWaits, startWaits, sentWaits, dependences, signals; prints statistics
  - cites: metal_recording.m:2982 "../dox/mesh/design/aot-compilation-change-request.md A1"; metal_recording.m:3033
    "../dox/mesh/design/compilation-composition-change-request.md item 6.2"; metal_recording.m:15-16 and 2566 and
    4004 refer to MetalProgramPrepare. The window comment (metal_recording.m ~2934-2940) cites no paper.
  - refuses:
    - metal_recording.m:2850 NSInvalidArgumentException "Row %u names plan call %llu of %u" when a call index is
      past planned->calls
    - metal_recording.m:2857 NSInvalidArgumentException "Call %u's native transfer task names row %llu, no planned
      row of the %u"
    - metal_recording.m:2882 NSInvalidArgumentException "%llu residues" when residues > UINT32_MAX/2
    - metal_recording.m:2900 NSInvalidArgumentException "Row %u's WHILE over parts [%u, %u): its parts are not the
      call's, or its predicate is not placed where the host reads it"
    - metal_recording.m:2927 NSInvalidArgumentException "Loop %u has invalid control ancestry"
    - metal_recording.m:2955 NSInvalidArgumentException "Loop %u's parent %u is absent"
    - metal_recording.m:2977 NSMallocException "A fence of call %u" when newFence fails
    - metal_recording.m:2988 NSInvalidArgumentException "The saved program's window, residues or positions (...) are
      not its rows' (...)"
    - metal_recording.m:2999 NSInvalidArgumentException "A saved fence wait on fence %u"
    - metal_recording.m:3007 NSInvalidArgumentException "%lu saved phases for %llu"
    - metal_recording.m:3084 NSMallocException "The calls' synchronization graph: %s" when
      mesh_graph_synchronization_scoped fails
  - callers: 1 call site: metal_recording.m:3564, inside MetalProgramMake (metal_recording.m:3519).

**The same thing computed in more than one place**

- Topological placement by repeatedly removing a zero-indegree vertex: mesh_graph_order (mesh-graph.c:37-48, picks
  lowest index) and mesh_graph_heap_order (mesh-graph.c:1222-1252, picks max priority then lowest index). Both take
  the same (vertices, edges) graph shape and yield a permutation or ordinal.
- Longest-path level per vertex: mesh_graph_levels (mesh-graph.c:1885-1901) is computed inside mesh_graph_schedule
  (mesh-graph.c:1905) and separately in Order (metal_ir.mm:1903).
- Heap-order per cut segment: Order (metal_ir.mm:1924-1931) and ReplayOrder (metal_ir.mm:1967-1977) each loop over
  orderCuts segments, collect the dependence edges inside each segment, call mesh_graph_heap_order and scatter
  results into an order array.
- Priority -completion into heap order: plan_schedule negates completion (mesh-collective.c:935-936) before
  mesh_graph_heap_order; mesh_graph_schedule uses -level (mesh-graph.c:1907-1909); ReplayOrder uses -1 - completion
  (metal_ir.mm:1960) and Order uses -1 or 0 (metal_ir.mm:1915-1918). Same concept (earlier-completing first) with
  different sign conventions and inputs.
- Greedy earliest-finish assignment: mesh_graph_list_schedule (mesh-graph.c:521-537) assigns each unit to the
  machine minimizing (carried+bytes)/rate; mesh_graph_fill (mesh-graph.c ~505-516, read only partly) assigns each
  unit to the part minimizing latency+(count+1)/rate. Same rule over different units (bytes vs unit count).
- Position-of-row computation in plan_graph (mesh-collective.c:1126 and 1268-1270 place[] recomputed) and in
  MetalProgramPrepare (at[] and row/lastRow arrays, metal_recording.m:2839-2920 and 3016-3018): both map plan calls
  to rows and positions.

**No function found, or only a refusal**

- No function found that orders transfers per processor on the device side other than plan_schedule's entry sort
  (mesh-collective.c:935-944, comparator plan_entry_order near mesh-collective.c:800, not fully read) and
  MetalPlannedOrder/MetalProgramPrepare's wait lists (metal_recording.m:2782, 2839).
- No function found named mesh_graph_schedule with any caller (only definition and header).
- mesh_graph_list_schedule has no error return; its only refusal is the INFINITY return (mesh-graph.c:535).
- mesh_graph_levels refuses edges with from >= to by EDEADLK (mesh-graph.c:1895), not cycles in general; no separate
  cycle detector in this function.
- No Metal-side function found that calls mesh_graph_order or mesh_graph_completion_times; the Metal ordering path
  uses mesh_graph_levels and mesh_graph_heap_order only (metal_ir.mm:1903, 1929, 1975).

**Not determined**

- Contents of design documents cited by the source (design/plan-to-bridge-change-request.md,
  design/aot-compilation-change-request.md, design/compilation-composition-change-request.md,
  design/joint-compilation.md, design/page-queue-change-request.md, design/torch-source-lowering-change-request.md,
  design/compilation-coordination.md, docs/provider.md) were not opened; no statements about what they specify are
  made.
- AGENTS.md's mandated reads for mesh/TP work (docs/principles.md, docs/measurement.md) were not opened; they are
  not named in the subproblem's document list.
- Not read: mesh_graph_contract (mesh-graph.c:714), mesh_graph_delayed, mesh_graph_synchronization_scoped,
  mesh_graph_dependences_views, mesh_graph_cycle_mean, mesh_graph_convert_sc, mesh_graph_determine_delays,
  mesh_graph_lags, mesh_graph_heap_cuts (mesh-graph.c:1257, no callers found outside its file), plan_entry_order,
  plan_rows, plan_communication, plan_place, MetalLoopOrder, MetalNewest.
- Exact line of the Highs_run/infeasible/non-optimal refusals inside mesh_graph_completion_times was given as a
  range (mesh-graph.c ~1140-1200), not pinned.
- Exact end line of plan_graph given as ~1345 from the read window; the closing brace was not separately located.
- Whether the Python test or driver code in mesh:bin calls any of these functions: grep found no matches there.
- Enclosing function of metal_ir.mm:3317 was inferred as metal_ir_finalize from the surrounding read
  (metal_ir.mm:3311-3320); the function header was not pinned by line.

## Storage decisions and the cost of a function on a processor

Read: 17 whole body, 1 part of the body. 18 functions.

- **`mesh_graph_buffer_bound`** — mesh:rdma/mesh-graph.c:1448-1485 (whole body read) (whole body)
  - does: Builds a graph of 2*actors vertices: host part j = vertex j (time host[j]), device part = vertex actors+j (time
    work[j]). Per processor j it adds 3 edges: j->next and actors+j->actors+next (delay 1 when next==0), and
    j->actors+j (delay 0). Appends the caller's edges at index 3*actors+e. lambda = mesh_graph_cycle_mean(V,...).
    sum = sum of all times. For each j: if !(lambda>0) bound[j]=1; else most=ceil(sum/lambda), adds reverse edge
    actors+j -> j at index E with delay 0, calls mesh_graph_min_delay(V,time,E+1,...,E,lambda,
    most>=1?most:1,&least), bound[j]=least?least:1.
  - takes: actors; host[actors]; work[actors]; edges and struct mesh_graph_dependence *edge; output uint64_t bound[actors]
  - returns: int status; bound filled per actor
  - cites: mesh-graph.c:1447: "/* BSL 4.2-4.3, (6): Bff of each feedforward edge of the two-processor IPC graph */".
    mesh-graph.h:88-96 (design/torch-capture-change-request.md item 2; "BSL 4, p. 9, 4.2-4.3, (6) and Lemma 3; M.
    Lam, PLDI 1988, 7"; "bound[j] is Bff(e_j) ... 1 where every t is 0"). BSL = S. S. Bhattacharyya, S. Sriram, E.
    A. Lee, IEEE Trans. Signal Processing 45(6), 1997.
  - refuses:
    - mesh-graph.c:1450 EINVAL if actors==0 or actors>=UINT32_MAX/2 or edges overflow the size bound
    - mesh-graph.c:1452 EINVAL if any host[j] or work[j] is negative, NaN or > DBL_MAX
    - mesh-graph.c:1454 EINVAL if an edge endpoint >= 2*actors
    - mesh-graph.c:1459 ENOMEM on allocation failure
    - mesh-graph.c:1473 EINVAL if lambda == INFINITY (a cycle of zero total delay; header mesh-graph.h:96 says the
      same)
    - errors from mesh_graph_min_delay propagate (its ERANGE at mesh-graph.c:1435)
  - callers: mesh-collective.c:1890 (inside mesh_plan, defined at mesh-collective.c:1846);
    metal-microbench/metal_recording.m:3312 (inside MetalHorizon, metal_recording.m:3259). Python wrapper
    mesh_graph.py:839 buffer_bound: no caller found in the searched trees. 2 C call sites plus 1 Python wrapper.
- **`mesh_graph_min_delay`** — mesh:rdma/mesh-graph.c:1422-1446 (whole body read) (whole body)
  - does: Sets changed[] = delay[] with changed[e]=bound, checks that delays_pass holds at the bound (else ERANGE), then
    binary-searches the least delay in [0,bound] for edge e for which delays_pass holds. delays_pass = no zero-delay
    cycle and no positive cycle at lambda.
  - takes: graph (vertices, time, edges from/to, delay), edge index e, lambda, bound
  - returns: int status; *least = least delay
  - cites: mesh-graph.c:1421: "/* BSL 9, Fig. 8, MinDelay */". mesh-graph.h:83-84 cites "design/joint-compilation.md 3, row
    6".
  - refuses:
    - mesh-graph.c:1425 EINVAL if e>=edges or lambda not in [0, DBL_MAX]
    - graph_check EINVAL (time, edge range); mesh-graph.c:1430 ENOMEM
    - mesh-graph.c:1435 ERANGE if the pass fails at the initial bound
  - callers: mesh-graph.c:1480 (mesh_graph_buffer_bound), mesh-graph.c:1553 (mesh_graph_determine_delays, body at 1528). 2
    call sites.
- **`mesh_graph_cycle_mean`** — mesh:rdma/mesh-graph.c:1398-1420 (whole body read) (whole body)
  - does: Checks graph; if a zero-delay cycle exists returns INFINITY. Otherwise, if positive_cycle at lambda=0, bisects
    lambda in [0, time_sum] for up to 200 steps using positive_cycle (Bellman-Ford style relaxation over edges with
    weight time[from]-lambda*delay), then mean = hi + vertices*tolerance; else mean = 0.
  - takes: graph with time[], edges (from,to,delay)
  - returns: int status; *mean double (INFINITY for zero-delay cycle)
  - cites: mesh-graph.c:1397: "/* BSL 9, BellmanFord; E. L. Lawler, Combinatorial Optimization: Networks and Matroids,
    1976, pp. 94-97 */". mesh-graph.h:80-81 cites "design/joint-compilation.md 3, row 6".
  - refuses:
    - graph_check EINVAL: negative/non-finite time, sum > DBL_MAX, edge endpoint >= vertices
      (mesh-graph.c:1311-1314)
    - ENOMEM from scratch_alloc (mesh-graph.c:1293-1304)
  - callers: mesh-graph.c:1472 (mesh_graph_buffer_bound); mesh-collective.c:1278 (plan_graph, defined
    mesh-collective.c:1115). 2 call sites.
- **`positive_cycle / zero_delay_cycle / delays_pass / cycle_tolerance / time_sum / graph_check / scratch_alloc / components_search (helpers)`** — mesh:rdma/mesh-graph.c:1293-1381 (read whole) (whole body)
  - does: positive_cycle: up to `vertices` rounds of relaxing distance[to] = distance[from] + time[from] - lambda*delay
    over edges (skipping delay==UINT64_MAX); returns 1 if still relaxing after the last round, else 0.
    zero_delay_cycle: builds rows of zero-delay edges, runs components_search (Tarjan SCC), returns 1 if a
    zero-delay edge lies within one SCC. delays_pass = !zero_delay_cycle && !positive_cycle. time_sum: sum of
    time[]. cycle_tolerance = 1e-9*time_sum. graph_check validates time and endpoints.
  - takes: graph arrays and lambda/tolerance
  - returns: int flags / double sums
  - cites: mesh-graph.c:~1315 "/* R. E. Tarjan, SIAM J. Comput. 1(2), 1972 */" (components_search); mesh-graph.c:~1352 "/*
    BSL Lemma 1 */" (zero_delay_cycle); mesh-graph.c:~1362 "/* R. Bellman, Quart. Appl. Math. 16, 1958; L. R. Ford,
    RAND P-923, 1956 */" (positive_cycle); mesh-graph.c:~1377 "/* BSL Fig. 8 */" (delays_pass). Line numbers marked
    ~ are approximate.
  - refuses:
    - graph_check: mesh-graph.c:1312 EINVAL for negative/non-finite time; 1313 EINVAL for sum > DBL_MAX; 1314 EINVAL
      for endpoint >= vertices
    - scratch_alloc: mesh-graph.c:1297 ENOMEM for vertices==UINT32_MAX or oversize edges; 1304 ENOMEM on allocation
      failure
  - callers: internal to mesh-graph.c (cycle_mean 1398, min_delay 1422, buffer_bound 1448, determine_delays 1528). Helpers
    only.
- **`mesh_graph_actor_time`** — mesh:rdma/mesh-graph.c:561-591 (read whole) (whole body)
  - does: Per actor: compute = max over (resource,width) of work/(rate.peak * (min(blocks,cores)/cores) *
    (min(threads,lanes)/lanes)) for each nonzero work operand. memory (ns, x1e9) =
    written/actor_bandwidth(bandwidth,blocks) plus, per read, actor_read (DRAM vs cache split using bandwidth,
    utilization, l2/l1 sizes and speedups), only when moved>0 and banded. time = compute + memory -
    min(compute,memory)*overlap + launches*launch + direct*direct + blits*blit. Sets missing for absent rates,
    absent bandwidth, unknown work, or zero utilization with bytes moved; returns ENODATA if missing.
  - takes: struct mesh_graph_machine (rates, cores, bandwidth, overlap, launch...), struct mesh_graph_actor (work, blocks,
    threads, written, launches, reads, direct, blits)
  - returns: int status; *time double (ns)
  - cites: mesh-graph.c:546 "/* OpenXLA gpu_performance_model_base.cc: ReadTimeWithDRAMHeuristic, in seconds */"
    (actor_read); mesh-graph.c:558-560 "/* OpenXLA gpu_performance_model_base.cc: CalculateEffectiveFlopsPerNs,
    ComputeTime, WriteTime and CombineComputeAndMemoryAccessTime; gpu_performance_model.cc
    EstimateRunTimeForInstructionImpl and EstimateRunTimes' kKernelLaunchOverhead a launch */". Header
    mesh-graph.h:72 declares it with no citation.
  - refuses:
    - No raise. mesh-graph.c:590 returns ENODATA when any needed rate, bandwidth, utilization or unknown work is
      absent; *time is still written (a lower bound)
    - mesh-graph.c:569-570: a missing rate (peak<=0, lanes==0, cores==0, threads==0) sets missing and skips that
      operand
  - callers: mesh-graph.c:601 (mesh_graph_cost_times); metal-microbench/metal_transform.mm:994 and :1004 (Time,
    metal_transform.mm:989); mesh_graph.py:829 (actor_time wrapper, which mesh_compile.py:7761 calls). Python _check
    raises OSError for other statuses (mesh_graph.py:808-810, 832).
- **`actor_bandwidth / actor_read (helpers)`** — mesh:rdma/mesh-graph.c:542-560 (whole body)
  - does: actor_bandwidth: min(bandwidth, blocks*block_bandwidth) when block_bandwidth>0. actor_read: splits a read into
    DRAM bytes (rate bandwidth*utilization) and cache bytes (rate scaled by l2_speedup/l1_speedup when net bytes <
    l2 or l1*cores); returns from_dram/dram + from_cache/rest.
  - takes: machine, blocks, read record
  - returns: double seconds / bandwidth
  - cites: mesh-graph.c:546 "OpenXLA gpu_performance_model_base.cc: ReadTimeWithDRAMHeuristic, in seconds".
  - refuses:
    - none in body
  - callers: mesh_graph_actor_time only
- **`mesh_graph_cost_times`** — mesh:rdma/mesh-graph.c:594-616 (read whole) (whole body)
  - does: Evaluates a cost DAG node by node in index order: ACTOR node -> mesh_graph_actor_time on actor[first]; SUM =
    value[first]+value[second]; MAX = fmax; REPEAT = value[first]*count (count 0 -> value 0, status ENODATA). Status
    propagates from children; a child index >= own index is rejected.
  - takes: machine, actors array, cost nodes (kind, first, second, count)
  - returns: int 0 (per-node values[] and status[] filled)
  - cites: mesh-graph.c:593: "/* docs/provider.md#composed-resource-costs */". metal-microbench/docs/provider.md:585-594
    (section "Composed resource costs"): "`mesh_graph_cost_times` evaluates the DAG once per supplied machine".
  - refuses:
    - mesh-graph.c:596 EINVAL if nodes>UINT32_MAX or null arrays with nodes>0
    - mesh-graph.c:601 EINVAL if actor index >= actors
    - mesh-graph.c:604 EINVAL if a child index >= i (or second >= i for non-REPEAT)
    - mesh-graph.c:607 ENODATA for REPEAT count 0
    - mesh-graph.c:613 EINVAL for unknown node kind
  - callers: metal-microbench/metal_ir.mm:3713 (inside metal_ir_dispatch_costs, metal_ir.mm:3708). 1 call site.
- **`Actor (metal_ir.mm, dispatch-level cost DAG builder)`** — metal-microbench:metal_ir.mm:1607-1693 (read whole) (whole body)
  - does: For dispatch `vertex`: appends a launch leaf (launches=1 for KERNEL/TASK/NEURAL, direct=1 for INDIRECT, blits=1
    for COPY/FILL), then one leaf per operation of the kernel (Sequence over k.operations; blocks overridden by
    hostEntries groups), IF controls -> SUM(test, MAX(first, second)), WHILE controls -> SUM(test, REPEAT(body+test,
    bound)), and a trailing leaf for kernels with no operations, CHECK, NEURAL, or added accesses. Root = SUM(root,
    issued); stored in p.costRoots[vertex]; p.costIssued[vertex] is UINT32_MAX for TASK/NEURAL/CHECK.
  - takes: Program& p, uint32_t vertex
  - returns: void (fills p.actors, p.costs, p.costRoots[vertex], p.costIssued[vertex])
  - cites: Body has no citation. metal_ir.h:353 comment: "The actor facts of dispatch `d` for its execution-time estimate
    (mesh-graph.h mesh_graph_actor_time; OpenXLA ...". docs/provider.md:585-594 describes the same DAG.
  - refuses:
    - No raise or error return. NEURAL kernels set facts.work.unknown=1 instead of refusing (metal_ir.mm:1643
      region)
  - callers: metal_ir.mm:2059 (inside Artifact, metal_ir.mm:1982) for every dispatch. 1 call site.
- **`Actor (metal_ir.mm, per-operation facts overloads)`** — metal-microbench:metal_ir.mm:1576-1605 (lines 1560-1605 read; the Assemble signature at 1557-1559 was not read) (part of the body)
  - does: Static Actor(p,op,m,groups,views,a,reads): Facts via OperationFacts; a={}; Assemble(views,facts) fills a.work,
    reads and a.written; a.blocks and a.threads = UINT64_MAX when the host construction is selected, else groups and
    m.threads; a.launches=0. Overload at 1590-1595 takes kernel and calls Effects(p,op,0,kernel,views). Overload at
    1597-1605 takes an `internal` list and erases those values from views.
  - takes: Program, op index, Mapping, group count, views or kernel or internal list
  - returns: void (mesh_graph_actor and reads filled)
  - cites: none in these lines.
  - refuses:
    - none in these lines
  - callers: metal_ir.mm:1594, 1604, 1641 (Actor(Program,vertex)); metal_transform.mm:1002 (Time, the internal-list
    overload). 4 call sites.
- **`Realize`** — metal-microbench:metal_ir.mm:1698-1724 (read whole) (whole body)
  - does: Marks kept[storage] for: every value that is not (TEMPORARY or CONSTANT with no placement, sent, received, or
    non-DATA word); every input/output/gather/symbol of each live operation; every kernel binding that is not a
    PAGES binding; every output of a control range that is Carried. Then p.realized[k] = Bytes(p,k) for each value
    with internal==UINT32_MAX and kept; then p.realized[root] = max over aliases sharing a storage root.
  - takes: Program& p
  - returns: void (fills p.realized)
  - cites: metal_ir.mm:1695-1697 comment: "The storage each value needs (metal_ir_value_info `bytes`), in one pass over
    values, operations and kernels: inputs, outputs, state, fixed placements and transport operands always;
    temporaries and constants where a remaining operation or kernel binding uses them". No external source.
  - refuses:
    - No raise or error. Values with no realized bytes stay 0.
  - callers: metal_ir.mm:2056 (inside Artifact, metal_ir.mm:1982). Consumers: Assemble (metal_ir.mm:1562) and
    metal_ir.mm:3743 (metal_ir_value_info sets info->bytes = realized). 1 call site.
- **`Time (metal_transform.mm)`** — metal-microbench:metal_transform.mm:989-1009 (read whole) (whole body)
  - does: machine = p.cpu if members[0] is CPU else p.gpu. Launch actor with `launches` set -> mesh_graph_actor_time into
    time. internal = Locals(p,members,mappings) values. For each member: Groups -> group count; Actor overload with
    internal list; mesh_graph_actor_time into member; time += member. complete is cleared when any call returns
    non-zero (ENODATA). Returns the sum.
  - takes: Program, member op list, mappings, launches, complete flag
  - returns: double time
  - cites: metal_transform.mm:986-988 comment: "the time of one kernel ... by mesh_graph_actor_time (OpenXLA's GPU
    performance model, Roofline) on the given machine; *complete false where it is a lower bound over the measured
    operands only (ENODATA)".
  - refuses:
    - No explicit refusal. members[0] is read without a size check (metal_transform.mm:991), so empty `members` is
      not handled.
    - ENODATA is reported through *complete (metal_transform.mm:994, 1004), not as an error.
  - callers: metal_transform.mm:1026 and :1027 (Profitable). 2 call sites (no other callers found by grep of
    metal-microbench).
- **`Profitable (metal_transform.mm)`** — metal-microbench:metal_transform.mm:1011-1029 (read whole) (whole body)
  - does: prior = mappings of current group; threads = max threads. If m has the same axis and tile as own and m.threads
    <= threads, returns true with no cost query (1018-1019). Else if !p.machines returns false (1020). Else combined
    = Time(group+op under the joined mapping); apart = Time(group under prior) + Time(op alone under own); returns a
    && b && c && combined <= apart.
  - takes: Program, group, mapping vector, op, own Mapping, candidate Mapping
  - returns: bool
  - cites: metal_transform.mm:1010 comment: "// docs/provider.md#composed-resource-costs".
    metal-microbench/docs/provider.md:585-609.
  - refuses:
    - returns false when p.machines is absent (metal_transform.mm:1020). No error value.
  - callers: metal_transform.mm:1394 (Transform, body at 1361) and :1401. 2 call sites.
- **`copies (method of _Instantiated)`** — mesh:rdma/mesh_compile.py:7130-7142 (read whole; enclosing class `class _Instantiated` at 7077, header not read) (whole body)
  - does: For each link in `links`: source,(scope,key) = self.linked(link); target =
    projection.homes_by_scope[scope][0][key][0]. If source is a tensor identical in key, storage_offset, shape and
    stride to target, skips it. Otherwise appends _assign(target, source) ops and calls self.remember(...) for the
    target. Then self.locate(ops, ir, node, segment) tags each op with scope, segment, node and place (CPU or GPU,
    from _device), and returns ops.
  - takes: links, ir, node, segment
  - returns: list of ops (copy operations)
  - cites: none in lines 7130-7142.
  - refuses:
    - No raise in the method. Failures in linked/_assign/_device are not visible here.
  - callers: mesh_compile.py:7157 (control, initial zero-exit copies), mesh_compile.py:7175 (control, copies of child yield
    exits). 2 call sites.
- **`_actors`** — mesh:rdma/mesh_compile.py:7738-7762 (read whole) (whole body)
  - does: For each program row with a builder (and machines present): calls lib.MetalRecorderActors twice, first with null
    buffers to get part and edge counts, then to fill host[], work[] (c_double arrays) and dependence edges. Then
    written = sum of input section bytes for the function. If parts>0 and written>0, calls
    mesh_graph.actor_time(CPU, Actor(0.0,1,1,written,0,0,None)) and adds the spent time to host[0]. Stores made[k] =
    (host, work, edges, parts, count, complete).
  - takes: programs dict (rows, lib, machines), irs
  - returns: dict k -> (host, work, edges, parts, count, complete)
  - cites: mesh_compile.py:7733-7737 comment ("function's call at this node part by part (host p its CPU tasks', work p its
    GPU dispatches' mesh_graph_actor_time on ..."). Cites no paper.
  - refuses:
    - mesh.check(status) raises OSError(status, strerror) (mesh.py:272-274) for statuses outside (0, ERANGE,
      ENODATA) on the first call (mesh_compile.py:7752) and outside (0, ENODATA) on the second (7758). ENODATA sets
      complete=False.
  - callers: mesh_compile.py:8722 (inside _built, def at 8657). 1 call site.
- **`_lifetimes`** — mesh:rdma/mesh_compile.py:7771-7811 (read whole; comment at 7765-7770) (whole body)
  - does: For each program row with a builder: reads the recorder's levels and values (values:level:into:) into MetalValue
    records. For each value vi: if the builder planned it to an EXECUTION value of this function that has no lag,
    widens that irs value's begin/end to the union of recorded intervals (spanned dict). Else if the value has
    bytes, appends a new mesh.Value (section bytes, kind, state, begin, end, run=1) to own[k] and maps
    row['own'][vi]. Then for each projection standin root without a [2] element, sets projection.allocations[root]
    to point at the function and own index. Stores row['levels']. Returns own.
  - takes: programs dict, irs
  - returns: own: list per irs of new mesh.Value lists
  - cites: mesh_compile.py:7765-7767 comment: "M. Lam, PLDI 1988, p. 322; ExecuTorch exir/memory_planning.py:348-354;
    metal_recording.h levels: and values:level:into::".
  - refuses:
    - No raise or error return in the body.
  - callers: mesh_compile.py:8702 (inside _built, def at 8657). 1 call site.
- **`mesh_graph_greedy_by_size`** — mesh:rdma/mesh-graph.c:3159-3196 (read whole; helper greedy_align_by_n 3153-3156 read; comparator greedy_compare_by_size ends at 3150, not read in full) (whole body)
  - does: Sorts records by size with qsort (greedy_compare_by_size). For each record in that order: among already-placed
    allocations that intersect it (intersect callback), finds prev_offset (the highest end, aligned by
    greedy_align_by_n to `align`) and best_diff: the smallest gap between prev_offset and a placed offset that is at
    least size[idx]. Places at the best-gap offset if found, else at prev_offset. Keeps ordered_allocs sorted by
    offset (memmove insertion). offset[idx]=best; total = max(offset+size).
  - takes: records count, size[], intersect callback and context, align, output offset[], total
  - returns: int status; offset[] and *total
  - cites: mesh-graph.c:3157-3158: "/* TensorFlow Lite 6cf18d88,
    tensorflow/lite/delegates/gpu/common/memory_management/greedy_by_size_assignment.cc:68-143,
    GreedyBySizeAssignment, and internal.h:31, kNotAssigned; Y. Pisarchyk and J. Lee, arXiv:2001.03288, 2020, 5.2
    */". mesh-graph.c:3152: TensorFlow Lite util.h:26-44 for AlignByN. Header mesh-graph.h:343-344 cites
    GreedyBySizeAssignment.
  - refuses:
    - mesh-graph.c:3163 ENOMEM on allocation failure
    - mesh-graph.c:3184 EINVAL if *total + allocs*(align-1) < prev_offset (in-body refusal)
    - no check of align==0 in greedy_align_by_n (3153-3156); not handled in body
  - callers: dox/mesh/rdma/mesh-collective.c:591 (inside plan_block, defined at mesh-collective.c:562), passing
    plan_intersect and quantum as align. 1 call site.
- **`mesh_graph_lags`** — mesh:rdma/mesh-graph.c:663-700 (read whole), with heap_up/heap_down 620-643 and lags_before/lags_relax 644-661 read (whole body)
  - does: For each source s: distance[] = UINT64_MAX, then seeds the binary heap from the source's out-edges with lag
    weights, then pops minimum-distance vertices (Dijkstra) relaxing out-edges, stopping early when every distinct
    target has been popped. Writes lags[s*targets+x] = distance[target[x]]. Edge lags of UINT64_MAX are absent;
    lags_relax returns without relaxing when at+weight would reach UINT64_MAX (silently, no error).
  - takes: graph (vertices, first[], head callback, lag callback), sources[], targets[]
  - returns: int status; lags[sources*targets] uint64
  - cites: mesh-graph.c:660-662: "/* E. W. Dijkstra, Numerische Mathematik 1, 1959, problem 2: lags[s * targets + x] the
    least sum of lags over a walk of at least one edge from source[s] to target[x], UINT64_MAX where there is none
    ...". mesh-graph.h:77 cites "design/joint-compilation.md 3, row 5".
  - refuses:
    - mesh-graph.c:667 ENOMEM if edges >= SIZE_MAX/8
    - mesh-graph.c:672 ENOMEM on allocation failure
    - No range check on head(context,k) or on source/target values inside the body (heads[] are stored as returned)
    - lags_relax (mesh-graph.c:650) drops relaxations with weight==UINT64_MAX or overflow, with no error
  - callers: mesh-graph.c:730 (mesh_graph_contract, body 714); mesh-graph.c:2323 and :2330 (mesh_graph_signature, body 2182);
    mesh-collective.c:381 (plan_unroll, body 357); mesh-collective.c:1333 (plan_graph, body 1115). 5 call sites.
- **`mesh_graph_unfold`** — mesh:rdma/mesh-graph.c:76-100 (read whole); declaration mesh-graph.h:21-28 (whole body)
  - does: Iterative depth-first walk from root over rows (first, head): for each path, calls visit(consumer, w, k, parent
    instance); an explicit stack (at, next, instance, open) tracks the current path; open[] marks vertices on the
    path. A vertex reached by several paths is visited once per path. Returns when all rows are exhausted or visit
    returns nonzero.
  - takes: graph rows, root, visit callback
  - returns: int status (0 or the visit's nonzero return)
  - cites: mesh-graph.h:21-25 comment ("the depth-first unfolding of rows (first, head) from `root`") cites no source.
    Cormen et al. 22.3-22.4 is cited for mesh_graph_preorder at mesh-graph.h:14-17, not this function.
  - refuses:
    - mesh-graph.c:79 EINVAL if root >= vertices
    - mesh-graph.c:83 ENOMEM on allocation failure
    - mesh-graph.c:92 EINVAL if head returns >= vertices; EDEADLK if the target is on the current path (cycle);
      EOVERFLOW if the next instance number would be UINT32_MAX
  - callers: none found. Only the declaration (mesh-graph.h:26) and the definition match in the searched trees.

**Documents**

- mesh:design/joint-compilation.md: Section 3 table rows: line 85 row 5 "`mesh_graph_lags` | none (computed) | least total lag over walks |
  sequential, after row 4 | E. W. Dijkstra, Numerische Mathematik 1, 1959"; line 86 row 6 "`mesh_graph_cycle_mean`,
  `mesh_graph_min_delay`, `mesh_graph_convert_sc`, `mesh_graph_determine_delays` | added synchronization edges and
  their delays | the period T = max over cycles of Σt(v)/Delay(C) (CP §4, Lemma 3) held, with the least added delays
  | sequential, after row 5 | BSL 1997 §9, Figs. 6 and 8 (CP §4); E. L. Lawler, Combinatorial Optimization: Networks
  and Matroids, 1976". Row 6 is the only row naming cycle mean, min delay and the delays. Row 7 (line 87) covers
  chains and fat cover, not buffer bounds.
  - sources it names: E. W. Dijkstra, Numerische Mathematik 1, 1959; BSL 1997 (Bhattacharyya, Sriram, Lee, IEEE Trans. Signal
    Processing 45(6), 1997), §9, Figs. 6 and 8; CP §4 (not identified further in the file); E. L. Lawler,
    Combinatorial Optimization: Networks and Matroids, 1976; Dilworth 1950, Fulkerson 1956, Hopcroft-Karp 1973,
    Dinic 1970, Mukerji 1994 (row 7, not read in detail)
- metal-microbench:docs/provider.md: Section "## Composed resource costs" at line 585: "The compiler retains a cost DAG alongside its selected
  construction ... `mesh_graph_cost_times` evaluates the DAG once per supplied machine; `metal_ir_dispatch_costs`
  projects both roots for every dispatch" (lines 587-593). Line 596 cites the OpenXLA GPU performance model
  (gpu_performance_model_base.cc). Lines 600-601: "Within a member, independent resource demands combine by their
  maximum; sequential members still sum. No resource-sharing coefficient is inferred." Lines 604-609: missing rates
  remain missing; unknown contributions are retained, not borrowed.
  - sources it names: OpenXLA gpu_performance_model_base.cc (github.com/openxla/xla, link at line 596)

**The same thing computed in more than one place**

- Cycle-ratio bound sum(t)/lambda: mesh-graph.c:1476 (buffer_bound, ceil(sum/lambda)) and mesh-graph.c:1547
  (mesh_graph_determine_delays, `time_sum(vertices, time) / lambda`) compute the same quotient from the same lambda.
- Composed actor time over a set of actors: mesh-graph.c:594-616 (mesh_graph_cost_times, SUM/MAX/REPEAT over
  mesh_graph_actor_time values) and metal-microbench/metal_transform.mm:989-1009 (Time, sum of mesh_graph_actor_time
  over a launch and each member) both produce a composed time for a set of actors via the same mesh_graph_actor_time
  call. metal_ir.mm:1607 (Actor builds the DAG that cost_times evaluates) versus metal_transform.mm:989 (Time
  evaluates per-operation facts directly).
- Host-side CPU time: mesh_compile.py:7761 (_actors builds one synthetic mesh_graph.Actor for the host write and
  calls mesh_graph.actor_time) and the mesh-graph.c:601 path in cost_times both call mesh_graph_actor_time for a
  host actor.
- Per-value storage bytes: metal_ir.mm:1698-1724 (Realize, p.realized) and metal_ir.mm:3743 (metal_ir_value_info
  info->bytes = realized) feed mesh_compile.py:_lifetimes (value.bytes read from the recorder,
  mesh_compile.py:7787-7790). One producer and consumers, not duplicated computation.
- Combination of an actor's time terms: mesh-graph.c:588-590 (compute, memory, launches, direct, blits combined with
  overlap) is the only place that combines these terms; metal_transform.mm:994 and :1004 instead add launch and
  member times, a second combination rule for the same cost quantity.

**No function found, or only a refusal**

- Storage lifetimes as a single graph function: no C function for interval lifetimes was found in the named files;
  _lifetimes (mesh_compile.py:7771) is the only one and it takes lifetimes from recorder intervals.
- Number of copies of a value: mesh_compile.py copies (7130) only builds copy ops for links; the count of locations
  per value is in mesh-graph.h:53-57 (mesh_graph_chains, mesh_graph_fat_cover; design/joint-compilation.md row 7),
  not read in its body here.
- Buffer bound on the Metal side: no Metal-side function computes Bff; metal_recording.m:3259 MetalHorizon is the
  caller of mesh_graph_buffer_bound.
- Python wrapper mesh_graph.py:buffer_bound (839-846): no caller found, so it is unused in the searched trees.
- mesh_graph_unfold: no caller in the searched trees, so no use of it for storage or cost was found.
- Per-processor cost in Python: none found beyond mesh_graph.py:829 actor_time.

**Not determined**

- Bodies of helpers not read: Bytes(p,k) (called by Realize, metal_ir.mm:1719), OperationFacts, Effects, Sequence,
  Groups, the Locals body beyond metal_transform.mm:1030, Tiled, Bindings, the Assemble signature
  (metal_ir.mm:1557-1559), greedy_compare_by_size (mesh-graph.c ~3140-3150), _assign, _link_values, _device, the
  _Instantiated class header (mesh_compile.py:7077), MetalRecorderActors (native), LEVELS and VALUES.
- Whether mesh_compile.py _actors' call mesh_graph.Actor(0.0, 1, 1, written, 0, 0, None) (7 positional arguments,
  first a float) matches the mesh_graph.py Actor struct (9 fields; first field `work` of type Work,
  mesh_graph.py:664-666 and 678-681) at runtime: only the argument list was read.
- Whether the mesh_graph_min_delay bisection gives the least delay as documented: not determined (correctness not
  assessed, as instructed).
- Caller counts for metal-microbench top level: the metal-microbench/*.py glob matched nothing; Python callers in
  that tree were not searched beyond grep -F over .c, .h, .py, .m, .mm and .swift.

## Saving and loading, and the synchronization signature

Read: 30 whole body, 5 part of the body, 2 declaration only. 37 functions.

- **`_saved_bytes`** — mesh:rdma/mesh_compile.py:8751-8759 (whole body)
  - does: Allocates two C out-parameters (void* data, size_t size) and a 4096-byte error buffer; calls the given ctypes
    function with the arguments plus &data, &size and (if diagnostic) error and its length; on nonzero status raises
    OSError; otherwise copies size bytes from data with C.string_at; in finally calls _FREE(data) (libc free).
  - takes: a ctypes function, its arguments, keyword diagnostic (default True)
  - returns: bytes (a copy of the malloc'd C buffer)
  - cites: none
  - refuses:
    - mesh_compile.py:8756 raise OSError(status, error text if diagnostic else function.__name__) on any nonzero C
      status (diagnostic=False passes the function name as the message)
  - callers: 4 call sites, all in mesh_compile.py: 8825 (mesh.LIB.mesh_plan_save, diagnostic=False), 8860
    (programs['compiler'].metal_ir_save), 8861 (programs['lib'].MetalRecorderSave), 8872
    (programs['lib'].MetalProgramSave).
- **`_save_program`** — mesh:rdma/mesh_compile.py:8821-8873 (whole body)
  - does: _finish(replay) runs MetalProgramEnd first (8822). Serializes: the plan via mesh.LIB.mesh_plan_save (8825); the
    arrays as (_declared_record per declared entry, bytes of weights, values, edges) (8828); actors as bytes tuples
    (8830-8831). For each row in programs['rows'] (8835): records k, j and node name; if row['builder'] is not None,
    walks builder.source_values with metal_ir_value_info, keeps constant values whose storage==v and that have data
    or buffer, deduplicates them by _constant_storage key into a storage list (state_data overrides bytes), records
    (v, storage index, info.at, info.data_bytes, offset); copies received marks (Range with input reset to Remote())
    and sent marks (Release with supplied/status zeroed); then stores program=metal_ir_save bytes,
    prepared=MetalRecorderSave bytes, and the scalar fields. Builds payload dict with header (_artifact_header),
    plan, arrays, actors, rows, storage, irs (layouts, inputs, device), analysed tuple, native=MetalProgramSave
    bytes of replay['program'], and signature=replay['signature'] (8866-8872). Writes b'MESH-AOT\0' +
    pickle.dumps(payload, protocol=5) via mesh_disk.write(path, ..., 'a compiled mesh program', 'w') (8873).
  - takes: context (ctypes mesh context), state dict, path
  - returns: the value returned by mesh_disk.write(path, ...), which this function returns unchanged
  - cites: none in body; the module header cites design/aot-compilation-change-request.md A1 and metal_ir.h /
    metal_recording.h (mesh_compile.py:8744-8745, comment above _saved_bytes)
  - refuses:
    - No raise in the body itself; failures surface through _saved_bytes (8756), _finish (mesh_compile.py:8818), and
      mesh_disk.write (mesh_disk.py:52, return value not checked here)
  - callers: 1 call site: run.save closure, mesh_compile.py:9018. No other caller found in dox/mesh/rdma, dox/mesh/bin,
    metal-microbench/tools/mesh.
- **`load`** — mesh:rdma/mesh_compile.py:8932-8968 (whole body)
  - does: Opens path in binary mode; reads 9 bytes and compares with b'MESH-AOT\0'; pickle.load(stream) of the rest;
    compares saved['header'] with _artifact_header(); creates context (_context()) and a state dict; defines make()
    which: retires an existing state if mapping or owned is set; mesh.Nodes + mesh_nodes_read; rebuilds arrays
    (_declared_restore, _records of Weight/Value/Dependence), actors (_records of double host/work,
    mesh_graph.Dependence edges), analysed Calls, irs as SimpleNamespace; retained dict from saved keys plus
    programs=_restore_programs(saved, state['owned']); _bound(...) with a monotonic deadline; sets stale=False; on
    BaseException retires and re-raises. Returns _runner(context, state, make).
  - takes: path
  - returns: run closure from _runner (mesh_compile.py:8971-9020)
  - cites: none
  - refuses:
    - mesh_compile.py:8935 ValueError('not a mesh compiled artifact') when the 9-byte magic differs
    - mesh_compile.py:8938-8939 ValueError('the saved program has different Torch, record layout, or machine
      configuration; compile for the current configuration') when saved['header'] != _artifact_header()
    - mesh_compile.py:8960-8964 except BaseException: _retire(context, state); raise (propagates whatever make()
      raised)
  - callers: No call of mesh_compile.load found in dox/mesh/rdma/*.py, dox/mesh/bin, metal-microbench/tools/mesh (grep for
    'mesh_compile' imports and load( found only the definition). Name collisions: mesh_code.py:85 def load(path) and
    dox/mesh/bin/mesh-sync-monitor.py:101 def load(path) are different functions.
- **`_load (name collision, not the artifact loader)`** — mesh:rdma/mesh_compile.py:2203-2204 (whole body)
  - does: _load(r, k, value, kind) returns _convert(r, r.arg(k, value), _class(value), kind): converts a call argument of
    a torch graph to a class; not an artifact loader.
  - takes: r, k, value, kind
  - returns: converted value
  - cites: mesh_compile.py:2199-2200 comment cites PyTorch v2.11.0 PowKernel.cpp for _power only
  - refuses:
    - none
  - callers: not searched in full; it is an internal argument converter
- **`_restore_programs`** — mesh:rdma/mesh_compile.py:8876-8930 (whole body)
  - does: Opens a recorder (lib.MetalRecorderOpen on a new device), releases the device; for each saved storage entry
    creates a SHARED MTLBuffer (fixed) or a ctypes buffer (non-fixed) and memmoves contents; for each saved row with
    'program': compiler.metal_ir_load(data, device) to get p; builds SimpleNamespace builder; for each saved
    constant, sets its storage (metal_ir_value_data when offset given, else metal_ir_value_storage) and records
    state_data for STATE role; sets row marks (Range/Release from_buffer_copy). Returns dict with lib, compiler,
    device, recorder, rows, rated (per-IR input sets), native_record, signature, state_data.
  - takes: saved (unpickled payload dict), owned (dict of cleanup callables)
  - returns: dict (see does)
  - cites: none
  - refuses:
    - mesh_compile.py:8882 OSError(errno.ENODEV, error text) if MetalRecorderOpen returns null
    - mesh_compile.py:8889 OSError(errno.ENOMEM, 'saved constant storage') if a buffer is null
    - mesh_compile.py:8904 OSError(errno.EINVAL, 'saved call %s: %s') if metal_ir_load returns null
  - callers: 1 call site: mesh_compile.py:8939-8940 inside load()'s make() (the programs= argument).
- **`_runner`** — mesh:rdma/mesh_compile.py:8971-9020 (whole body)
  - does: Wraps make() and execution. timed(work) raises OSError(ETIMEDOUT) if a previous run timed out (8974-8975) and
    records a first ETIMEDOUT. Defines writable/execution/inputs/run. Attaches run.inputs, run.ended, run.save
    (8015-9019: if stale calls timed(make), then timed(lambda: _save_program(context, state, path))),
    run.synchronize (_finish), run.signature (returns state['replay']['signature'] or None). weakref.finalize(run,
    _dropped, context, state).
  - takes: context, state, make
  - returns: run closure with attributes inputs, ended, save, synchronize, signature
  - cites: mesh_compile.py:8998 comment '# design/compilation-composition-change-request.md item 5' and 9002-ish comment '#
    MTL4CommandQueue.h signalEvent:value:' cite the sources for execution and run.inputs
  - refuses:
    - mesh_compile.py:8974-8975 OSError(ETIMEDOUT, "the program's run ended at its first ETIMEDOUT (R-7): ...") on
      any call after a timeout
    - mesh_compile.py:8998-9004 execution(): re-raises OSError; on ESTALE or ECANCELED sets stale=True
  - callers: called by load (8968) and mesh_compile.py _runner use only
- **`_finish`** — mesh:rdma/mesh_compile.py:8814-8818 (whole body)
  - does: Calls replay['lib'].MetalProgramEnd(replay['program'], replay['ended'], error, len) and raises _execution_error
    on nonzero status.
  - takes: replay dict
  - returns: None
  - cites: none
  - refuses:
    - mesh_compile.py:8818 raise _execution_error(replay, status, error text)
  - callers: _save_program (8822), run.synchronize (9011-ish).
- **`_artifact_header`** — mesh:rdma/mesh_compile.py:8801-8806 (whole body)
  - does: Returns tuple (1, sys.byteorder, sizeof(c_void_p), torch.__version__, tuple of sizeof for mesh.Declared,
    mesh.Weight, mesh.Value, mesh.Dependence, Range, Release, ValueInfo, mesh_graph.Dependence, tuple of (k,
    bytes(v)) for sorted machines from _machines()). Compared as equality in load().
  - takes: none
  - returns: tuple
  - cites: none
  - refuses:
    - none
  - callers: _save_program (8866 payload header), load (8937 comparison)
- **`_declared_record / _declared_restore / _records / _initial / _constant_storage`** — mesh:rdma/mesh_compile.py:8762-8769 (_records), 8766-8771 (_declared_record), 8773-8780 (_declared_restore), 8783-8799 (_initial), 8809-8812 (_constant_storage) (whole body)
  - does: _records(kind, data): ctypes array of kind from bytes via from_buffer_copy. _declared_record: copies a
    mesh.Declared to bytes after clearing from_/to pairs and returns (bytes, pairs). _declared_restore: rebuilds
    mesh.Declared array from (bytes, pairs). _initial: collects copy contents (C.string_at) for invariant or lagged
    values. _constant_storage(info): returns key ('buffer', info.buffer) with contents/length, or ('data',
    info.data, info.data_bytes).
  - takes: see does
  - returns: see does
  - cites: none
  - refuses:
    - none
  - callers: _records, _declared_record, _declared_restore: used in _save_program and load; _initial: not searched further;
    _constant_storage: _save_program 8856.
- **`metal_ir_save`** — metal-microbench:metal_ir.mm:3359-3402 (whole body)
  - does: Rejects non-finalized program (3362). Out-stream: Header (kArtifact magic, Build() id from LC_UUID of the dylib,
    device string, machines, gpu, cpu), six uint64 counts (values, operations, kernels, views, later_passes, held,
    actorReads), Values(o,*p), then per value: owner index and byte offset into held buffers (computed from
    materializations and liveness flags, lines 3370-3390), then per operation identity/place/erased/kind,
    Records(o,*p,false,&held), then per kernel: kind, flags, arguments, later, fill, subject, control, original,
    groups, threads, allocations, operations, function and source strings, devices, counts, counted, and blob from
    SaveTarget (3396). Copies to malloc'd *bytes.
  - takes: p, bytes, length, error, room
  - returns: int status; *bytes malloc'd, *length set
  - cites: metal_ir.h:269-271 comment: '...each kernel's target binary (SaveTarget)'
  - refuses:
    - metal_ir.mm:3362 EINVAL 'metal_ir_save before metal_ir_finalize'
    - metal_ir.mm:3396 propagates SaveTarget status
    - metal_ir.mm:3400 ENOMEM 'metal_ir_save: no memory for the artifact'
  - callers: mesh_compile.py:8860 (via ctypes, programs['compiler']); metal-microbench/tools/native_bench.m:72
- **`metal_ir_load`** — metal-microbench:metal_ir.mm:3406-3467 (whole body)
  - does: Creates a new program; reads the Header (mismatch -> fail); reads 7 counts; rejects counts > 2^32 or stream
    error (3420); resizes vectors; Values(in,*p); reads per-value (owner, offset) pairs; operations; Records(in,*p);
    sets device (CFRetain) and memory from maxThreadgroupMemoryLength; for each kernel reads fields and blob
    (in.Vec(blob)) then LoadTarget(*p, k, device, blob) (3444); after loop checks in.status or trailing bytes
    (3446); resolves value data pointers into held buffers (3450-3455 raises 'a compiler-created constant outside
    its bytes'); resets buffer/fields/prepared/release/released; rebuilds records from views; sets actor reads;
    Laters(*p); sets lowered and finalized; returns p.release().
  - takes: bytes, length, device, error, room
  - returns: struct metal_ir_program * (nullptr on failure, error text written to error)
  - cites: none
  - refuses:
    - metal_ir.mm:3414 fail(EINVAL, mismatch) with Header's message: 'not a metal_ir artifact of this format',
      'saved by another metal_ir library build', 'saved for device X, loading on Y', 'a truncated artifact header'
      (metal_ir.mm:2609-2624)
    - metal_ir.mm:3420 fail(EINVAL, 'a truncated or malformed artifact') for counts above 2^32 or stream status
    - metal_ir.mm:3444 returns nullptr when LoadTarget fails (its message is set by LowerFail)
    - metal_ir.mm:3446 fail(EINVAL, 'a truncated or malformed artifact') on trailing bytes or stream error
    - metal_ir.mm:3452 fail(EINVAL, 'a compiler-created constant outside its bytes')
  - callers: mesh_compile.py:8902 (compiler.metal_ir_load in _restore_programs); metal-microbench/tools/native_bench.m:77 and
    :472.
- **`metal_ir_loaded`** — metal-microbench:metal_ir.mm:3469- (part of the body)
  - does: Reports p->error if set; relocate lambda rebinds a Readiness's status and checks origin count consistency
    (returns EINVAL 'a later pass has inconsistent origins'). Not read beyond line 3475.
  - takes: p, error, room
  - returns: int status
  - cites: none
  - refuses:
    - metal_ir.mm:3470 Report(EINVAL, p->error) if p->error nonempty
  - callers: mesh_compile.py:7919 (compiler.metal_ir_loaded when 'prepared_record' in row)
- **`SaveTarget`** — metal-microbench:metal_lower.mm:2085-2105 (whole body)
  - does: If kind is METAL_IR_NEURAL, returns SaveNeural. Otherwise blob = k.binary. For METAL_IR_TASK: requires binary,
    object and nativeSymbol nonempty (else LowerFail); builds blob = uint64[3] sizes {binary, object, symbol}
    followed by the three byte sections via memcpy. Finally, for KERNEL/INDIRECT/TASK requires nonempty blob (else
    LowerFail with kernel index).
  - takes: Program, index, blob, error, room
  - returns: int status; blob via reference
  - cites: metal_ir.h:857-861 comment: 'A kernel's target binary for metal_ir_save ... (metal_lower.mm, the target's)'
  - refuses:
    - metal_lower.mm:2091 LowerFail 'native target lacks its linked code, lowerable object, or symbol identity'
    - metal_lower.mm:2103 LowerFail 'kernel N: no compiled target bytes'
  - callers: metal_ir.mm:3396 only (metal_ir_save loop); declared metal_ir.h:860
- **`LoadTarget`** — metal-microbench:metal_lower.mm:2107-2149 (whole body)
  - does: Neural kinds go to LoadNeural. Other kinds other than KERNEL/INDIRECT/TASK return 0. For TASK: parses the
    3-uint64 size header, checks sizes against remaining bytes, sets k.object, k.nativeSymbol (rejects embedded
    NUL), and shifts blob to the binary section. Sets k.binary = blob bytes; writes the bytes to a TargetFile
    (2131-2133); for TASK dlopen()s that file, records the handle in p.owners, dlsym's k.nativeSymbol into k.task.
    Otherwise: newArchiveWithURL (MTL4Archive) on the file, newLibraryWithURL,
    newComputePipelineStateWithDescriptor(PipelineDescriptor(k, library, index)) and stores the pipeline in
    k.pipeline.
  - takes: Program, index, device, blob, bytes, error, room
  - returns: int status; sets k.pipeline or k.task
  - cites: none
  - refuses:
    - metal_lower.mm:2111 'kernel N: empty compiled target'
    - metal_lower.mm:2114 'native target header is truncated'
    - metal_lower.mm:2118 'native target section is truncated'
    - metal_lower.mm:2121 'native target has trailing bytes'
    - metal_lower.mm:2125 'native target symbol contains a null byte'
    - metal_lower.mm:2132 'target restore: <NSError or strerror>'
    - metal_lower.mm:2135 'task restore: <dlerror>'
    - metal_lower.mm:2138 'restored task entry missing'
    - metal_lower.mm:2141 'Metal archive load: ...'
    - metal_lower.mm:2143 'Metal archive library: ...' (or 'compiled library is absent')
    - metal_lower.mm:2146 'Metal archive pipeline <name>: ...' (or 'named pipeline is absent')
  - callers: metal_ir.mm:3444 only (metal_ir_load loop)
- **`Archive`** — metal-microbench:metal_lower.mm:1512-1522 (whole body)
  - does: Serializes target.serializer as an MTL4 archive to a TargetFile URL (serializeAsArchiveAndFlushToURL), then
    reads the file back with NSData dataWithContentsOfURL and assigns its bytes to k.binary.
  - takes: Kernel &k, Target &target, error, room
  - returns: int status
  - cites: none
  - refuses:
    - metal_lower.mm:1515-1516 'Metal archive: <error>' if the file path or serialization fails
    - metal_lower.mm:1517 'Metal archive read: <error>' if the read fails
  - callers: metal_lower.mm:2653 (inside the lowering path, not the save path)
- **`SaveNeural`** — metal-microbench:metal_neural.mm:1420-1446 (read through the return at about 1446) (whole body)
  - does: Requires n->compiled. Walks the compiled Core ML model directory (enumeratorAtPath), reads each regular file
    into an NSDictionary 'files' keyed by relative path; requires coremldata.bin; builds a binary property list
    dictionary {files, inputs, outputs} where inputs/outputs are arrays of [name, slot, type, shape, strides];
    writes it to blob.
  - takes: Program, kernel index, blob, error, room
  - returns: int status, blob
  - cites: none
  - refuses:
    - metal_neural.mm:1422 'Core ML artifact has no compiled model'
    - metal_neural.mm:1439 'Core ML compiled directory has no coremldata.bin: <path>'
    - metal_neural.mm:1430-1436 read failure 'Core ML artifact read: ...' (via LowerFail)
    - metal_neural.mm:1443-1444 'Core ML artifact: <error>' if plist serialization fails
  - callers: metal_lower.mm:2087 (SaveTarget for METAL_IR_NEURAL)
- **`LoadNeural`** — metal-microbench:metal_neural.mm:1448-1475 (read to the line that creates a temporary directory, about 1475; the rest not read) (part of the body)
  - does: Deserializes the plist with NSPropertyListImmutable; requires NSDictionary with NSDictionary 'files', NSArray
    'inputs' and 'outputs'; allocates a Neural owned via p.owners; validates each IO record (5 fields, types, equal
    shape and strides counts) and pushes IO structs; then creates a temporary directory named
    /tmp/mesh-neural-load-XXXXXX (line 1475).
  - takes: Program, kernel index, blob, bytes, error, room
  - returns: int status
  - cites: none
  - refuses:
    - metal_neural.mm:1453 'Core ML artifact is not a model record'
    - metal_neural.mm:1456 'Core ML artifact is missing model files or IO records'
    - metal_neural.mm:1474 'Core ML artifact has malformed IO'
    - metal_neural.mm:1486 'Core ML artifact has malformed file records' (in the part read)
  - callers: metal_lower.mm:2109 (LoadTarget for METAL_IR_NEURAL)
- **`MetalRecorderSave`** — metal-microbench:metal_recording.m:2292-2352 (whole body)
  - does: Builds NSData with header MetalPreparedIdentity(capture->program) (struct with magic, layout, values,
    dispatches, SHA-256 digest of the program identity, lines 679-690). Appends tagged sections (uint32 tag + uint64
    length + data) in order: commands and tasks (MetalCommandsEncoded over dispatches/tasks),
    starts/boundaries/parts/artifact/actors/guards/loops, effects (with view pairs rewritten to offsets into a pairs
    section), pairs, values (per stand-in value), laters (MetalLaterRecord + length + encoded command), ends (sorted
    by MetalReference), scalars (buffers, memory, lastLater, levels, stepState). Copies to malloc'd bytes.
  - takes: prepared (id), bytes, length, error, room
  - returns: int status; malloc'd bytes
  - cites: metal_recording.m:665-666 comment: 'design/aot-compilation-change-request.md A1: the saved records of a prepared
    program (MetalRecorderSave, -restore:records:length: reads them)'
  - refuses:
    - metal_recording.m:2350-2351 @catch NSException: MetalError(exception.reason) and return EINVAL
    - metal_recording.m:2344 return ENOMEM if malloc fails
  - callers: mesh_compile.py:8861 (via ctypes); metal-microbench/tools/native_bench.m:73
- **`MetalRecorderLoad`** — metal-microbench:metal_recording.m:2354-2358 (whole body)
  - does: Calls [recorder restore:program records:bytes length:length] inside @try; on NSException, MetalError and returns
    nil. The restore method (metal_recording.m:1283) was not read.
  - takes: recorder, program, bytes, length, error, room
  - returns: id (restored prepared program) or nil
  - cites: metal_recording.m:2352-2353 comment: none (the save-side comment is at 665-666)
  - refuses:
    - metal_recording.m:2357 MetalError(exception.reason) and nil on any NSException from restore
  - callers: metal-microbench/tools/native_bench.m:93; mesh_compile.py ctypes binding at mesh_compile.py ~1893-1900
    (lib.MetalRecorderLoad.argtypes)
- **`MetalProgramSave`** — metal-microbench:metal_recording.m:3579-3596 (whole body)
  - does: Builds NSDictionary {version 3, key (NSData), lag, window, completionLookback, residues, positions, waits,
    hostWaits, sentWaits, startWaits, dependences, signals (or empty NSData)} from the MetalProgram; serializes as a
    binary property list; malloc's *bytes and copies.
  - takes: handle, bytes, length, error, room
  - returns: int status; malloc'd bytes
  - cites: metal_recording.m:3576-3578 comment: 'design/aot-compilation-change-request.md A1, the composed artifact's
    [native] section: the program's solved horizon and synchronization with the key of the inputs they were solved
    from (a binary property list of numbers and data, no address)'
  - refuses:
    - metal_recording.m:3590 MetalError(failure.description) and return EINVAL if plist serialization fails
    - metal_recording.m:3592 return ENOMEM if malloc fails
  - callers: mesh_compile.py:8872 (via ctypes, programs['lib'].MetalProgramSave); metal-microbench/tools/native_bench.m:74
- **`MetalProgramLoad`** — metal-microbench:metal_recording.m:3597-3610 (whole body)
  - does: Parses bytes as a property list (NSPropertyListImmutable, no copy); requires NSDictionary with version == 3 (as
    unsignedInt), completionLookback NSNumber, key NSData; otherwise raises NSInvalidArgumentException (3606);
    returns [MetalProgramMake(recorder, calls, count, machines, planned, saved) retain].
  - takes: recorder, calls, count, machines, planned, bytes, length, error, room
  - returns: id (program) or nil
  - cites: none
  - refuses:
    - metal_recording.m:3606 [NSException raise:NSInvalidArgumentException format:@'No saved native program: %@']
      when the dictionary is not a version-3 record or fields are missing/mistyped
    - metal_recording.m:3608 MetalError and nil on any NSException
  - callers: metal-microbench/tools/native_bench.m:65; mesh_compile.py ctypes binding (~1893-1900)
- **`MetalProgramMake (saved-key check, referenced by MetalProgramLoad)`** — metal-microbench:metal_recording.m:3519-3531 (part of the body)
  - does: Sets program->key to MetalProgramKey(program, machines, planned) retained; if saved and saved['key'] differs
    from program->key, refuses.
  - takes: recorder, calls, count, machines, planned, saved
  - returns: MetalProgram *
  - cites: none
  - refuses:
    - metal_recording.m:3531 refusal path when saved[@'key'] is not equal to the key (exact message not read; the
      condition is at line 3531)
  - callers: MetalProgramLoad (3600), MetalRecorderProgram path (3573)
- **`MetalProgramSignatureSpans`** — metal-microbench:metal_recording.m:4430-4450 (head read) (part of the body)
  - does: Zeroes the output signature; sums span counts into executions with overflow check (EOVERFLOW); then builds
    MetalEmitted tables and calls mesh_graph_signature (4487), mesh_graph_signature_program (4507), _feedback
    (4538), _spans (4540), _choices (4541), _execution (4549); on error frees. Body beyond line 4450 not read.
  - takes: handle, context, spans, counts, signature, error, room
  - returns: int status; fills *signature
  - cites: metal_recording.m:4430-4433 comment cites ../dox/mesh/design/synchronization-signature-change-request.md items 1
    and 6 and ../dox/mesh/rdma/mesh-graph.h mesh_graph_signature
  - refuses:
    - metal_recording.m:4434-4436 path: EINVAL when spans>0 and counts is NULL; EOVERFLOW on count overflow or size
      overflow
  - callers: tools/mesh_signature.h:17 (signature_write_spans); the non-spans MetalProgramSignature (metal_recording.m:4568)
    is called from mesh_compile.py:8013 (lib.MetalProgramSignature)
- **`mesh_graph.signature (Python)`** — mesh:rdma/mesh_graph.py:905-914 (whole body)
  - does: Constructs a Signature; converts four sequences to ctypes arrays via _array (Processor, Task, Dependence,
    Operation); calls LIB.mesh_graph_signature with C.byref(made); on success returns forms(made) and frees made in
    finally.
  - takes: processors, tasks, dependences, operations (sequences of tuples/ctypes)
  - returns: forms(made): dict of the three printed forms (not read) and record data via forms
  - cites: mesh_graph.py:904 comment: '# design/synchronization-signature-change-request.md items 1, 2, 4 and 6;
    mesh-graph.h mesh_graph_signature'
  - refuses:
    - mesh_graph.py:809-810 _check raises OSError(status, os.strerror(status)) on nonzero status
  - callers: dox/mesh/bin/mesh-sync-monitor.py:696-697 (two calls: expected and watched)
- **`mesh_graph.record (Python)`** — mesh:rdma/mesh_graph.py:926-933 (whole body)
  - does: Calls LIB.mesh_graph_record once with a NULL buffer and room 0 to get the size; if status is not ERANGE,
    _check(status or EINVAL) raises; allocates a ctypes buffer of that size; calls again; returns buffer.raw[:size].
  - takes: made (a Signature)
  - returns: bytes
  - cites: mesh_graph.py:918-919 comment: 'mesh-graph.h struct mesh_graph_record, mesh_graph_record,
    mesh_graph_record_read: the compiled program's signature at its external boundary (the frontend's and E2B's
    record, the monitor's input)'
  - refuses:
    - mesh_graph.py:931 _check(status or EINVAL) when the size query returns a status other than ERANGE
    - mesh_graph.py:809-810 OSError from _check on the write call
  - callers: dox/mesh/rdma/mesh_compile.py:8016 (dict(mesh_graph.forms(replayed), record=mesh_graph.record(replayed))). Also
    indirectly mesh_compile.py:8026-8027 writes signed['record'] to MESH_SIGNATURE + '.record'.
- **`mesh_graph.from_record (Python)`** — mesh:rdma/mesh_graph.py:936-939 (whole body)
  - does: Copies data into a ctypes buffer; makes a Signature; calls LIB.mesh_graph_record_read(addressof(buffer),
    byref(made)) with _check; returns made.
  - takes: data (bytes)
  - returns: Signature
  - cites: none
  - refuses:
    - mesh_graph.py:809-810 OSError via _check
  - callers: dox/mesh/bin/mesh-sync-monitor.py:102
- **`mesh_graph_signature (C)`** — mesh:rdma/mesh-graph.c:2182-2200 (head read; the derivation continues) (part of the body)
  - does: Zeroes output; validates processor kinds, task processor indices, dependence from/to indices, operation
    processor/primitive/vertex indices (EINVAL); overflow checks (ENOMEM); allocates processor, task, operation,
    dependence, edge, least, alias arrays; allocates key, row, image, slot, sinks, lags working arrays.
  - takes: processors, processor, tasks, task, dependences, dependence, operations, operation, signature
  - returns: int status; fills *signature
  - cites: none in the body read; the header comment at mesh-graph.h:299-301 and mesh-graph.h near 281 cite
    synchronization-signature-change-request.md; the Lamport citation appears at mesh-graph.c:2637 for the composer,
    not this function
  - refuses:
    - mesh-graph.c:2187 EINVAL for processor[p].kind >= MESH_GRAPH_KINDS
    - mesh-graph.c:2188 EINVAL for task processor >= processors
    - mesh-graph.c:2189 EINVAL for dependence from/to >= tasks
    - mesh-graph.c:2190-2192 EINVAL for operation processor, primitive, or vertex out of range
    - mesh-graph.c:2194-2196 ENOMEM for size overflow or tasks == UINT32_MAX
  - callers: mesh_graph.py:911 (ctypes), tools/mesh_signature.h not direct, metal_recording.m:4487 (in
    MetalProgramSignatureSpans)
- **`signature_derive (C, static)`** — mesh:rdma/mesh-graph.c:2137-2181 (whole body)
  - does: Allocates position, previous, waited, first, order, count, unknown arrays (ENOMEM on failure);
    mesh_graph_rows(... signature_processor ...) gives row order per processor; sets position and previous per
    operation; sums count per (processor, primitive) cell (static primitives add value, unknown values add object);
    for each synchronized edge sets waited[wait]=k, used[signal], states[state]++ and redundant; counts unwaited
    signal operations; counts uncovered dependences where least[k] > delay; then result = signature_digest(s).
  - takes: struct mesh_graph_signature *s
  - returns: int status; sets s->digest through signature_digest
  - cites: none
  - refuses:
    - mesh-graph.c:2140-2145 result ENOMEM when any allocation fails
  - callers: mesh-graph.c:2334 (in mesh_graph_signature), and record_restore with derive=1 (mesh-graph.c:2566-2567)
- **`signature_digest (C, static)`** — mesh:rdma/mesh-graph.c:2015-2034 (whole body)
  - does: Calls mesh_graph_matrix_market twice (size query, then text); then computes SHA-256
    (CC_SHA256_Init/Update/Final, CommonCrypto) over the matrix market text, updating in chunks of at most 2^30
    bytes, excluding the final NUL byte (at+1<made); stores digest in s->digest.
  - takes: struct mesh_graph_signature *s
  - returns: int status
  - cites: none
  - refuses:
    - mesh-graph.c:2018 result if the size query returns other than ERANGE: returns result or EINVAL
    - mesh-graph.c:2019 ENOMEM if malloc fails
  - callers: signature_derive (mesh-graph.c:2176); also mesh-graph.c:2049, 2060, 2072, 2086, 2096, 2133
    (signature_program/spans/conditions/choices/feedback/execution setters each end with return signature_digest(s))
- **`mesh_graph_record (C)`** — mesh:rdma/mesh-graph.c:2462-2481 (whole body)
  - does: Computes primary (record_parts: 7 arrays, aligned to 8 bytes, header first) and metadata (program_parts: 12
    arrays plus a program_record header with magic GRAPH_PROGRAM_RECORD and counts) sizes; *made = primary +
    metadata; if *made > room returns ERANGE without writing; memset zero; writes struct mesh_graph_record header
    (bytes, processors, tasks, operations, dependences, synchronized) and digest copied from s->digest; memcpy of
    the 7 primary arrays with alignment; if metadata, memcpy of the program_record body and the 12 program arrays.
  - takes: signature, record, room, made
  - returns: int status; *made set to the size
  - cites: mesh-graph.c:2425-2426 comment: 'design/synchronization-signature-change-request.md item 1';
    mesh-graph.h:299-301 comment: 'its SHA-256 is part of the compiled artifact; ... mesh_graph_record_read derives
    the rest as mesh_graph_signature does'
  - refuses:
    - mesh-graph.c:2466-2467 ERANGE when the output room is smaller than the record (no output written)
  - callers: mesh-graph.c:2602 (mesh_graph_archive); mesh_graph.py:928 and :932 (ctypes, via record);
    tools/mesh_signature.h:22 and :25 (signature_write_spans)
- **`record_restore (C, static)`** — mesh:rdma/mesh-graph.c:2483-2569 (whole body)
  - does: Reads the header; allocates processor, task, dependence, operation, edge, least, alias arrays sized from the
    header (ENOMEM on failure); EINVAL if primary size > head->bytes; copies 7 primary arrays; if bytes remain,
    reads a program_record magic and selects the field count for legacy magics (values 0x31..0x37 and
    GRAPH_PROGRAM_RECORD); reads extra fields; validates each count against remaining room (EINVAL); allocates
    control, event, scope, span, selection, guard, choice, submission, action, instruction, condition, feedback
    arrays (ENOMEM); checks that remaining payload size equals room (EINVAL); copies the 12 program arrays, with
    legacy control and choice element sizes handled separately. Then if derive: signature_derive(s) (recomputes
    digest). Else copies head->digest into s->digest.
  - takes: record (const void*), s, derive flag
  - returns: int status; fills *s (on error frees s)
  - cites: none in the body; the comment above mesh_graph_record (mesh-graph.c:2425) applies
  - refuses:
    - mesh-graph.c:2498 EINVAL if primary > head->bytes
    - mesh-graph.c:2504 EINVAL if the trailing program header is shorter than 4 uint64
    - mesh-graph.c:2512 EINVAL if the extra fields do not fit
    - mesh-graph.c:2524 EINVAL if the magic is not a known version or any count exceeds the remaining room
    - mesh-graph.c:2544 ENOMEM when a program array allocation is null
    - mesh-graph.c:2545 EINVAL if payload size does not match the room
    - mesh-graph.c:2495 ENOMEM when a primary allocation is null
  - callers: mesh-graph.c:2571 (mesh_graph_record_read, derive=1); mesh-graph.c:2617 (mesh_graph_archive_read, derive=0)
- **`mesh_graph_record_read (C)`** — mesh:rdma/mesh-graph.c:2571 (whole body)
  - does: Returns record_restore(record, s, 1).
  - takes: record, s
  - returns: int
  - cites: none
  - refuses:
    - the rejects of record_restore apply
  - callers: mesh_graph.py:938 (ctypes from_record)
- **`mesh_graph_archive (C)`** — mesh:rdma/mesh-graph.c:2592-2606 (with archive_parts 2579-2590) (whole body)
  - does: Computes primary = record_size(s) and derived = archive_parts (7 arrays: position, previous, waited, first,
    order, count, unknown, sized from operations and processors*MESH_GRAPH_PRIMITIVES); writes struct
    signature_archive header (magic SIGNATURE_ARCHIVE, bytes, record size, uncovered, redundant, unwaited, states,
    primitives); calls mesh_graph_record into bytes after the header; then copies the 7 derived arrays aligned by
    part size (no record_align).
  - takes: signature, bytes, room, made
  - returns: int status; *made set
  - cites: mesh-graph.c:2573 comment: 'design/aot-compilation-change-request.md A1; the signature and its already-derived
    indices.'
  - refuses:
    - mesh-graph.c:2596 ERANGE if room < *made (the total size)
    - mesh-graph.c:2603 propagates mesh_graph_record status
  - callers: none found in dox/mesh/rdma/*.py, dox/mesh/bin, tools/, metal-microbench top level or provider (grep for
    mesh_graph_archive found only the definition and declaration)
- **`mesh_graph_archive_read (C)`** — mesh:rdma/mesh-graph.c:2608-2636 (whole body)
  - does: Checks length >= header; reads signature_archive header; requires magic, primitives, head.bytes == length,
    record size bounds, record->bytes == head.record (EINVAL); calls record_restore(record, s, 0) (no derivation;
    digest copied from record); checks archive_parts size equals remaining length (EINVAL, after free); then
    allocates 7 owned arrays (ENOMEM), memcpy's each derived array from bytes, assigns s->position .. s->unknown,
    copies uncovered, redundant, unwaited and states from the header.
  - takes: bytes, length, s
  - returns: int status; fills *s
  - cites: mesh-graph.c:2573 comment (above struct signature_archive, applies to the pair)
  - refuses:
    - mesh-graph.c:2611 EINVAL if length < header size
    - mesh-graph.c:2614 EINVAL if magic, primitives, bytes or record bounds are wrong
    - mesh-graph.c:2616 EINVAL if the record's bytes field differs from head.record
    - mesh-graph.c:2620 EINVAL if derived sizes do not match the remaining length
    - mesh-graph.c:2625 ENOMEM if an owned allocation fails
  - callers: none found in the searched directories
- **`mesh_graph_matrix_market (C) (digest input, not read)`** — mesh:rdma/mesh-graph.c:2918 (declaration only)
  - does: Printer that writes the Matrix Market text of the signature; its text is the SHA-256 input in signature_digest
    (mesh-graph.c:2015-2034). Body not read.
  - takes: s, text, room, made
  - returns: int status
  - cites: not read
  - refuses:
    - not read
  - callers: signature_digest (2017, 2021); tools/mesh_signature.h:29-30 (printers array)
- **`mesh_plan_save / mesh_plan_load (C) (plan bytes, not read)`** — mesh:rdma/mesh-collective.c:2207 (mesh_plan_save), 2268 (mesh_plan_load); declared mesh-collective.h:122 and :125; ctypes in mesh.py:244 and :246 (declaration only)
  - does: Plan serialization, called by mesh_compile.py:8825 through _saved_bytes with diagnostic=False. Bodies not read.
  - takes: context, program, functions, count, ...
  - returns: int status
  - cites: not read
  - refuses:
    - not read
  - callers: mesh_compile.py:8825 (save); mesh_compile.py load path does not call mesh_plan_load directly (saved_plan kept in
    retained, mesh_compile.py:8960-8961 region)
- **`SaveTarget/LoadTarget callers in Archive path`** — metal-microbench:metal_ir.h:857-861 (declarations and comment) (whole body)
  - does: Declares SaveTarget and LoadTarget and documents them as the kernel target binary save/restore.
  - takes: see does
  - returns: declarations
  - cites: metal_ir.h:857-858 comment: 'A kernel's target binary for metal_ir_save, and its restoration at metal_ir_load
    (metal_lower.mm, the target's)'
  - refuses:
    - none
  - callers: metal_ir.mm:3396 and :3444

**Documents**

- mesh:design/aot-compilation-change-request.md: Line 109-110 quotes PyTorch v2.11.0 documentation: 'Since torch.export produces a full graph disjoint from any
  Python features or runtime, this graph can then be saved, loaded, and run in different environments and languages'
  (cited as PyTorch v2.11.0 docs/source/user_guide/torch_compiler/export.md, 'JIT vs. AOT' and 'Partial vs. Full
  Graph Capture'). Line 79 heading 'A1. The input is each program's captured dataflow graph, with shapes and types'.
  The doc's text on the serialized artifact is in the quoted A1 passage; the artifact sections themselves are cited
  by the code comments, not by this doc's quoted lines read here.
  - sources it names: PyTorch v2.11.0 docs/source/user_guide/torch_compiler/export.md
- mesh:design/synchronization-signature-change-request.md: Line 181: 'It is printed in the three forms (Matrix Market, DOT, and the text space-time diagram), and its SHA-256
  is part of the compiled artifact.' Line 232 (item 1, Write): 'It is printed in the three forms of item 2, and its
  SHA-256 is part of the compiled artifact.' Lines 216-224 (item 1, Source) name Bhattacharyya, Sriram and Lee 1997;
  Lee and Ha; Lee and Messerschmitt 1987 (synchronous dataflow); Lam 1988 (unrolled copies); Schneider 1990. Line
  226-228 says 'The graph module computes it, through mesh_graph_dependences, mesh_graph_synchronization,
  mesh_graph_redundant and mesh_graph_schedule.' Line 278-279 (item 4) says the digest changes for the same captured
  program are appended to metal-microbench output_data/change-request-reviews.jsonl.
  - sources it names: Bhattacharyya, Sriram and Lee 1997; Lee and Ha; Lee and Messerschmitt 1987; Lam 1988; Schneider 1990;
    design/plan-to-bridge-change-request.md R5; metal-microbench output_data/change-request-reviews.jsonl
- mesh:rdma/mesh-graph.h: Lines 299-301: '/* design/synchronization-signature-change-request.md item 1, "its SHA-256 is part of the compiled
  artifact": the signature's one serialized form, its header then processor, task, dependence, operation, edge,
  least and alias in that order; mesh_graph_record_read derives the rest as mesh_graph_signature does */'. Line
  302-304 struct mesh_graph_record of 72 bytes (static assert). Lines 305-308 declare mesh_graph_record,
  mesh_graph_record_read, mesh_graph_archive, mesh_graph_archive_read. Line 314-315 comment cites L. Lamport, CACM
  21(7), 1978 for mesh_graph_compose (not in the subproblem's named list).
  - sources it names: L. Lamport, CACM 21(7), 1978
- metal-microbench:metal_ir.h: Lines 269-271: '...each kernel's target binary (SaveTarget)' describing metal_ir_save contents. Lines 272-279
  describe metal_ir_load and metal_ir_loaded. Lines 857-861: 'A kernel's target binary for metal_ir_save, and its
  restoration at metal_ir_load (metal_lower.mm, the target's): the bytes from which LoadTarget sets Kernel::pipeline
  (KERNEL/INDIRECT), ::task (TASK) or ::neural/context (NEURAL)'.
- metal-microbench:metal_recording.h: Lines 96-99: MetalRecorderSave/MetalRecorderLoad declarations with comment '(metal_ir_load) they were prepared
  from, nothing re-solved. Records of another program, or of another record...'. Lines 168-172:
  MetalProgramSave/MetalProgramLoad declarations with comment 'MetalProgramLoad builds the same program from the
  same operands as MetalRecorderProgram and...'. Lines 186-189: MetalProgramSignature declarations.
- metal-microbench:metal_recording.m: Lines 665-666: comment cites aot-compilation-change-request.md A1 for MetalRecorderSave. Lines 1280-1281 comment
  for restore. Lines 2982-2983: restored program's synchronization (MetalProgramSave) and MetalProgramLoad checked
  the inputs. Lines 3443-3444: 'a native program of the rows: its horizon and synchronization solved, or restored
  from saved (MetalProgramSave)'. Lines 3576-3578: comment cites aot-compilation A1 for the composed [native]
  section. Lines 4430-4433: cites synchronization-signature-change-request.md items 1 and 6 and mesh-graph.h
  mesh_graph_signature.
  - sources it names: design/aot-compilation-change-request.md A1; design/synchronization-signature-change-request.md items 1 and 6;
    design/compilation-composition-change-request.md item 6.2
- mesh:rdma/mesh-graph.c: Line 2425-2426: '/* design/synchronization-signature-change-request.md item 1 */' above record_align and
  record_parts. Line 2573: '/* design/aot-compilation-change-request.md A1; the signature and its already-derived
  indices. */' above signature_archive. Line 2637: '/* design/synchronization-signature-change-request.md item 1; L.
  Lamport, CACM 21(7), 1978 */' above the compose helpers (not in the named list).
  - sources it names: L. Lamport, CACM 21(7), 1978 (compose only)
- mesh:rdma/mesh_compile.py: Lines 8744-8745: comment '# design/aot-compilation-change-request.md A1; metal_ir.h metal_ir_save/load;
  metal_recording.h MetalRecorderSave/Load and MetalProgramSave/Load; mesh-collective.h mesh_plan_save/load'. Lines
  8013-8016 comment: the graph module's forms and its record's bytes kept with the compiled program. Lines 8026-8027
  comment: tools/e2b_replay.m signature_write writes the record and its three printed forms beside the
  MESH_SIGNATURE prefix.
  - sources it names: design/aot-compilation-change-request.md A1; metal_ir.h; metal_recording.h; mesh-collective.h;
    tools/e2b_replay.m
- mesh:rdma/mesh_graph.py: Line 904: comment '# design/synchronization-signature-change-request.md items 1, 2, 4 and 6; mesh-graph.h
  mesh_graph_signature'. Lines 918-919: comment on mesh_graph_record, mesh_graph_record_read: 'the compiled
  program's signature at its external boundary (the frontend's and E2B's record, the monitor's input)'.
  - sources it names: design/synchronization-signature-change-request.md items 1, 2, 4 and 6

**The same thing computed in more than one place**

- Digest of the signature: signature_digest (mesh:rdma/mesh-graph.c:2015-2034) computes SHA-256 over the Matrix
  Market text; the same Matrix Market text is printed separately for the monitor/tool via mesh_graph_matrix_market
  (mesh-graph.c:2918) and mesh_signature.h:29-40 prints the three forms, and mesh_graph.forms (mesh_graph.py, not
  read) produces matrix_market in Python.
- Two serializations of one signature: mesh_graph_record (mesh-graph.c:2462-2481) writes the record (header + 7
  primary arrays + program_record and 12 program arrays), and mesh_graph_archive (mesh-graph.c:2592-2606) writes the
  record plus 7 derived arrays. mesh_graph_archive embeds mesh_graph_record's output (mesh-graph.c:2602).
- Recomputing vs storing the digest on read: record_restore with derive=1 (mesh-graph.c:2566-2567, via
  mesh_graph_record_read at 2571) recomputes the digest through signature_derive -> signature_digest (2137, 2176),
  while derive=0 (mesh_graph_archive_read, 2617) copies the stored digest (mesh-graph.c record_restore final
  statement). Both paths exist for the same record format.
- Kernel binary stored twice in-process: Archive (metal_lower.mm:1512-1522) sets k.binary from the serialized MTL4
  archive; SaveTarget (metal_lower.mm:2085-2105) then copies k.binary into the blob (for TASK, a sized concatenation
  of binary, object and symbol).
- Synchronization data in the saved artifact is stored twice: MetalProgramSave (metal_recording.m:3579-3596) saves
  the native waits, dependences and signals as a plist, and mesh_compile._save_program stores replay['signature'] as
  a Python object in the pickle (mesh_compile.py:8872). The signature's record bytes are produced separately by
  mesh_graph.record (mesh_graph.py:926-933) and written by mesh_compile.py:8026-8027 for MESH_SIGNATURE.
- Two MetalProgramSignature producers: MetalProgramSignatureSpans (metal_recording.m:4430-4450 head read) and
  MetalProgramSignature (metal_recording.m:4568), which mesh_compile.py:8013 calls; tools/mesh_signature.h:17 calls
  the spans variant.
- Record size/write pattern duplicated: mesh_graph.py:926-933 (size query with NULL, then write) and
  tools/mesh_signature.h:22-25 (same two-call pattern in C).
- Name collisions with the same word: mesh_compile.py:2203 _load (torch graph argument converter),
  mesh_compile.py:8932 load (artifact loader), dox/mesh/rdma/mesh_code.py:85 load(path),
  dox/mesh/bin/mesh-sync-monitor.py:101 load(path). Only mesh_compile.load is the compiled program loader.

**No function found, or only a refusal**

- mesh_graph_archive and mesh_graph_archive_read: no Python ctypes binding found in dox/mesh/rdma/*.py
  (mesh_graph.py bindings are only signature 905, record 926, from_record 936) and no caller found in the searched
  directories.
- A save entry point for the signature alone (a file of the signature) exists only as mesh_graph_record through
  mesh_compile.py:8026-8027 and tools/mesh_signature.h:16-27; no reader of the MESH_SIGNATURE .record file was found
  besides mesh-sync-monitor.py:102 (from_record).
- No caller of mesh_compile.load (the compiled-program loader) was found in dox/mesh/bin, tools/mesh, dox/mesh/rdma,
  or metal-microbench tests.
- No caller of the returned run.save was found in the searched directories.
- Version or compatibility check for the MetalProgram native section: MetalProgramLoad checks version == 3
  (metal_recording.m:3600-3606); no older version is read. For the pickle, the only check is the header equality
  (mesh_compile.py:8937-8939).

**Not determined**

- Body of metal_recording.m restore:records:length: (metal_recording.m:1283) and MetalProgramMake beyond 3531
  (metal_recording.m:3519-3531, partly read).
- Bodies of Values, Records, Laters in metal_ir.mm (called from metal_ir_save/load at 3359-3467) were not read;
  metal_ir_loaded beyond line 3475 not read.
- Body of SaveNeural after line 1446 is partly read; LoadNeural after line 1475 not read.
- Body of mesh_graph_matrix_market (mesh-graph.c:2918) and mesh_graph_signature beyond line 2200 (first 20 lines
  read) and of mesh_graph.py forms() were not read.
- Body of mesh_plan_save/mesh_plan_load (mesh-collective.c:2207, 2268) not read.
- The definition of signature_write used at tools/e2b_replay.m:1558 was not located in the read set;
  tools/mesh_signature.h defines signature_write_spans, not signature_write.
- Whether the digest stored in a record is checked against the derived digest on mesh_graph_record_read (derive=1
  overwrites the stored digest via signature_derive; no comparison found in record_restore at
  mesh-graph.c:2483-2569).
- Callers of mesh_disk.write beyond its definition (mesh_disk.py:52) were not searched, and its body was not read.
- Whether the saved 'signature' object in the pickle equals the record bytes of the same signature was not
  determined from code read here.

## Allocating unequal work to nodes of different speeds

Read: 17 whole body, 1 part of the body, 1 not found. 19 functions.

- **`allocate.Allocator (class: __init__ and __call__)`** — mesh:rdma/allocate.py:5-44 (whole body)
  - does: __init__ (7-18) stores shares, grain, low, floor as lists, total=sum(shares), forget, step; validates (see
    rejects); sets s=float shares, a and b from prior[i][0] and prior[i][1], covariance P[i]=(prior[i][2], 0,
    prior[i][3]), trace. __call__ (20-44): for each rank i, x=scale*shares[i]; a recursive-least-squares update of
    a_i, b_i with forget factor lam (29-35) using residual e=times[i]-a_i-b_i*x; covariance update with trace capped
    at its prior trace (33-35); b=max(b_i, floor_i) (37); g_i=times[i]+b*(next_scale*s_i-x) (38);
    w_i=1/(b*next_scale) (39); B=1/w; calls equal_finish(total, grain, low, [inf]*n, [eta*g_i-B_i*s_i], B) (41-42);
    stores shares and s (43); returns list(c) (44). The per-rank cost is a linear model a_i+b_i*x fit online, and
    the solve is an equal-finish on the step-damped targets.
  - cites: allocate.py:1 module docstring """design/joint-compilation.md §3, row 9""" (file not read); no other citation
  - refuses:
    - allocate.py:13-14 raise ValueError('allocator: shares and lower bounds on the grain with room above the
      bounds, one prior and positive floor per rank, forget and step in (0, 1]') when any of allocate.py:10-12
      holds: the four lists differ in length; sum(low)>=total; any shares or low not a multiple of grain, low<grain,
      or shares<low; min(floor)<=0; forget not in (0,1]; step not in (0,1].
  - callers: none found in dox/mesh (rdma, bin), metal-microbench (tools, provider, top level, docs) besides the class's own
    file; heterogeneity.md:69 names allocate.Allocator in prose only
- **`equal_finish`** — mesh:rdma/allocate.py:47-64 (whole body)
  - does: Loop (49-61): free = ranks not yet held; T=(total - sum(held values) + sum(a_i/b_i over free)) / sum(1/b_i over
    free) (51-52); s_i = held value or (T-a_i)/b_i (53); collects free ranks below low (54) or above high (55); if
    none, break (56-57); else holds all below-low ranks at low, or all above-high ranks at high, whichever total
    excess is larger (58-61). Then c_i = low_i + grain*floor((s_i-low_i)/grain) (62), and c is replaced by
    min_max(total, grain, c, high, [lambda x: a_i+b_i*x]) (63). Returns (c, s, T) (64).
  - cites: none
  - refuses:
    - none explicit; 1/b_i with b_i=0 would raise ZeroDivisionError implicitly (51-52, no check)
  - callers: 1 in code: allocate.py:41 (Allocator.__call__). Named in heterogeneity.md:42 in prose only.
- **`min_max`** — mesh:rdma/allocate.py:67-84 (whole body)
  - does: n=len(low). fill(cap) (70-78): c=list(low); repeats (total-sum(low))//grain times: fits = ranks with c_i+grain
    <= min(high_i, cap); if none, break; picks min over fits of (cost_i(c_i+grain), i) and adds grain to it. Any
    remainder of (total-sum(low)) not divisible by grain is left unassigned. Returns (c, max_i cost_i(c_i)). If
    shared is None (80-81): returns fill(inf). Otherwise (82-84): fills = [fill(cap) for cap in range(max(low),
    total+1, grain)]; picks among fills whose sum equals the sum of the last fill the one minimizing
    f[1]+shared(f[0]) (83); returns (c, t+shared(c)). cost[i] is any callable of a count; the greedy assigns each
    grain to the rank whose cost after adding it is least.
  - cites: none (no comment or docstring in the body)
  - refuses:
    - none explicit; min() at 83 over an empty sequence (when range(max(low), total+1, grain) is empty, i.e.
      max(low)>total) would raise ValueError from Python, with no message of this module
  - callers: 1 in code: allocate.py:63 (equal_finish). shared argument: no caller in code supplies it. heterogeneity.md:119
    and measurement.md:75,84 name allocate.min_max in prose.
- **`capacity`** — mesh:rdma/allocate.py:87-88 (whole body)
  - does: Returns, for each rank i, min(high_i, total - sum(low) + low_i) floored to a multiple of grain (list
    comprehension over range(len(low))). It bounds each rank's buffer by its own high bound and by the largest share
    left after the others' lower bounds.
  - cites: none
  - refuses:
    - none
  - callers: none found in code (dox/mesh, metal-microbench tools/provider/docs). Named in heterogeneity.md:72 (R14) and :121
    (step 5) in prose.
- **`grid.main`** — metal-microbench:tools/mesh/grid.py:123-153 (whole body)
  - does: Parses --nodes, --mesh, --programs, --shares, --placement, --round and related flags; builds node objects via
    nodes_from (line 144); sets args.shares to '1,1,...' (one per rank) when --shares is absent (146-147); checks
    the share count equals the rank count (148-149); calls prepared(args.node_list, 'round' or 'decode', 'tp', args)
    (151); prints json {directory, prepare command} (152).
  - cites: grid.py:2-12 module docstring: 'one share per rank (an even split by default)'; 'Share r is rank r's: a rank is
    its node's index in its bridge's map (run_pair.py reads it from mesh-stat --map), not its place in --nodes.';
    comments grid.py:125-126 '../dox/mesh/design/page-queue-change-request.md#collective-size'
  - refuses:
    - grid.py:142 parser.error('--placement or --round')
    - grid.py:149 parser.error(f'--shares {args.shares} gives {n} shares; --nodes names {ranks} ranks')
  - callers: CLI entry (grid.py:156-157 __main__); tools/mesh/run_pair.py:32 imports grid and names grid.nodes_from and
    grid.Node in a comment; run_pair.py:24 names grid.py as the preparer
- **`grid.prepared`** — metal-microbench:tools/mesh/grid.py:82-118 (whole body)
  - does: Computes directory name (86-90), the prepare command via prepare_command (91) and stamp = command + preparer
    hash (92). ensure(node) (94-103): reads node's prepared.cmd; reuses if it equals the stamp and operands.json
    exists; else runs over ssh/local a room check, rm -rf, mkdir and the prepare command, then writes the stamp.
    Runs ensure on all nodes in a ThreadPoolExecutor (105-106). Then links embeddings .tflite files into a shared/
    directory by sha256 (109-115). Reads operands.json from the first node (116-117). Returns (directory, manifest,
    command) (118).
  - cites: grid.py:74-76 preparer docstring: 'a prepared directory is reused only when the command and the preparer that
    ran it both match (every node checks out one commit).'; grid.py:98 comment on mesh_disk.room
  - refuses:
    - grid.py:108 raise SystemExit('prepare failed: ' + failures) when any node's prepare command fails
  - callers: grid.py:151 (main) only
- **`grid.prepare_command`** — metal-microbench:tools/mesh/grid.py:59-71 (whole body)
  - does: Builds the prepare_e2b.py argument string: model, directory, batch magic, prefill, context magic; for 'round'
    adds --draft; for ranks>1 adds --placement {args.shares} (the per-rank share list) and --layer-placement or
    --verify-placement/--draft-placement.
  - cites: grid.py:60-61 docstring: 'the committed prepare command'
  - refuses:
    - none
  - callers: grid.py:91 (prepared) only
- **`grid.nodes_from / grid.Node (share-to-rank mapping inputs)`** — metal-microbench:tools/mesh/grid.py:27-54 (whole body)
  - does: nodes_from splits the comma-joined host text into hosts and builds Node(rank=index in --nodes, host, mesh path)
    (50-54); Node stores rank; argv builds zsh or ssh argv (37-40). The rank here is the index in --nodes, which
    grid.py:8-11 says is not the share index.
  - cites: grid.py:28-30 docstring 'run_pair.py sets its rank, its index in its bridge's map'
  - refuses:
    - none
  - callers: grid.py:144 (main); tools/mesh/run_pair.py:32 comment
- **`prepare_e2b.column_cuts`** — metal-microbench:tools/prepare_e2b.py:466-468 (whole body)
  - does: cuts = width * cumsum([0, *shares]) // sum(shares) (floor of cumulative share fraction times width); returns
    cuts rounded down to a multiple of PACKED_COLUMNS if align else cuts.
  - cites: prepare_e2b.py:464-465 '# ../dox/mesh/design/algorithm-sources.md#nnffn' and '#
    ../dox/mesh/design/prepared-machine.md#M22'
  - refuses:
    - none
  - callers: 2: prepare_e2b.py:476 (packed_cuts), prepare_e2b.py:1283 (vocabulary, align=True)
- **`prepare_e2b.packed_cuts`** — metal-microbench:tools/prepare_e2b.py:473-479 (whole body)
  - does: Divides width by unit, calls column_cuts(width//unit, shares)*unit (476), and checks each cut is a multiple of
    PACKED_COLUMNS (477-478).
  - cites: prepare_e2b.py:471-472 '# ../dox/mesh/design/algorithm-sources.md#nnffn' and '#
    ../dox/mesh/design/prepared-machine.md#M22'
  - refuses:
    - prepare_e2b.py:475 ValueError f"{label} width {width} is not a whole number of {unit}-column units"
    - prepare_e2b.py:478 ValueError f"placement {shares} does not partition {label} width {width} at packed
      boundaries"
  - callers: 2: prepare_e2b.py:1657 (attention, unit=head_dim), prepare_e2b.py:1659 (mlp and others), both in prepare()
- **`prepare_e2b.partition`** — metal-microbench:tools/prepare_e2b.py:817-873 (whole body)
  - does: Given a fragment (a model graph) and [start,end) columns: heads=True calls shard_heads (822); else finds the
    down-projection's width and rejects ambiguity (826-829), builds axes for tensors carrying that width (830-831),
    for fed inputs copies them and inserts SLICE operators to cut the columns (832-847), slices all tensors on their
    axes via slice_tensors (848), then inserts a dequantize and rewires the last two operators to emit an fp32
    partial (849-870). Returns the shard fragment.
  - cites: prepare_e2b.py:815-816 '# ../dox/mesh/design/algorithm-sources.md#nnffn' and '#
    ../dox/mesh/design/prepared-machine.md#prepared-operands'
  - refuses:
    - prepare_e2b.py:828-829 ValueError f"{ambiguous} carry width {width} on more than one axis; the column axis of
      the partition is ambiguous"
    - shard_heads' raises propagate (prepare_e2b.py:768, 781)
  - callers: 1: prepare_e2b.py:1682 (prepare)
- **`prepare_e2b.shard_heads`** — metal-microbench:tools/prepare_e2b.py:765-815 (whole body)
  - does: heads, head_dim from head_geometry; requires start and end multiples of head_dim (767-768); computes head_axes
    marks; cuts the fragment to owned=(end-start)//head_dim heads (770-772); rewrites shape tensors, constant int32
    shape/extent buffers and op inputs for the owned count (773-806); for a RESHAPE with one input, replaces it with
    a head_join_shape tensor (800-806).
  - cites: prepare_e2b.py:763-764 '# ../dox/mesh/design/algorithm-sources.md#prefill-kv' and '#
    ../dox/mesh/design/prepared-machine.md#prepared-operands'
  - refuses:
    - prepare_e2b.py:768 ValueError f"attention cut [{start},{end}) does not land on {head_dim}-column heads"
    - prepare_e2b.py:781 ValueError f"decomposition constant {tensor.name} carries the head axis"
    - head_geometry (535) and head_axes (572) raises at prepare_e2b.py:543-725 (not read whole)
  - callers: 1: prepare_e2b.py:822 (partition)
- **`prepare_e2b.combine`** — metal-microbench:tools/prepare_e2b.py:876-918 (whole body)
  - does: Builds a ModelT with one subgraph whose inputs are the contributions; chains ADD operators over them
    (accumulated sum, 886-895); then QUANTIZE of the accumulated tensor using the original quantize op, and a
    DEQUANTIZE copy producing the output (896-911). Returns the result model.
  - cites: prepare_e2b.py:874-875 '# ../dox/mesh/design/algorithm-sources.md#nnffn' and '#
    ../dox/mesh/design/prepared-machine.md#prepared-operands'
  - refuses:
    - none
  - callers: 1: prepare_e2b.py:1710 (prepare)
- **`prepare_e2b.kv_pattern`** — metal-microbench:tools/prepare_e2b.py:920-938 (whole body)
  - does: If every share is an integer, counts=ints (923-924). Otherwise parts = Fraction(str(s)) normalised to sum 1
    (926-927), period = lcm of denominators (928); if period<=limit, counts = int(p*period) (929-930); else counts =
    floor(p*limit) plus one for each of the largest (limit - sum) fractional remainders (931-936) (largest remainder
    rounding). Returns counts divided by their gcd (937-938).
  - cites: prepare_e2b.py:918-919 '# ../dox/mesh/design/algorithm-sources.md#prefill-kv'
  - refuses:
    - prepare_e2b.py:922 ValueError f"KV shares {shares} give some rank no positions" when any share <= 0
  - callers: 1: prepare_e2b.py:946 (kv_layers)
- **`prepare_e2b.kv_layers`** — metal-microbench:tools/prepare_e2b.py:941-953 (whole body)
  - does: Selects layer_placement entries whose key ends '.kv' and value != 'replicated' (942-944); returns ([], None) if
    none; takes one pattern from kv_pattern for all such layers (946-949); requires counts to cover ranks (950-951);
    returns (sorted layer indices, counts).
  - cites: prepare_e2b.py:939-940 '# ../dox/mesh/design/algorithm-sources.md#prefill-kv'
  - refuses:
    - prepare_e2b.py:948 ValueError f"split KV layers take one ownership pattern, not {sorted(patterns)}"
    - prepare_e2b.py:951 ValueError f"KV shares {counts} do not cover the {ranks} ranks of the placement" when
      ranks<2 or len(counts)!=ranks
  - callers: 1: prepare_e2b.py:1368 (prepare)
- **`prepare_e2b.prepare (share-consuming parts only)`** — metal-microbench:tools/prepare_e2b.py:1325-~2012 (read: 1325-1372, 1576-1600, 1648-1690, 1960-2012; the rest not read) (part of the body)
  - does: ranks=len(placement) (1330); ffn_columns broadcast and checked (1332-1336); kv_layers (1368). Per layer and
    section (1588-1594): shares = layer_placement['L.section'] or ['L'] or placement; computing = ranks with nonzero
    share. Checks len(shares)==len(placement), no negatives, positive sum (1651-1652); attention cuts
    packed_cuts(unit=head_dim) (1657), mlp cuts packed_cuts (1659); owned heads per rank (1663-1664); per piece
    partition (1682) and combine (1710).
  - cites: prepare_e2b.py:1366-1367 and 1581,1656 '../dox/mesh/design/algorithm-sources.md#prefill-kv' /
    'prepared-machine.md#M22' comments (approximate lines)
  - refuses:
    - prepare_e2b.py:1334 ValueError 'FFN tile widths must be positive and cover the placement'
    - prepare_e2b.py:1336 ValueError f'FFN tile widths {ffn_columns} do not tile at packed boundaries'
    - prepare_e2b.py:1374 ValueError 'split KV serves the one-row decode...'; 1376 ValueError 'split KV merges the
      ranks' attention sections; --replicate-attention has none'
    - prepare_e2b.py:1582 ValueError (layer ranks would carry on a residual stream they do not hold); 1598
      ValueError (layer splits its KV positions...)
    - prepare_e2b.py:1652 ValueError f'placement {shares} does not cover layer {layer}'
    - prepare_e2b.py:1664 ValueError f'layer {layer} assigns {owned} heads, not {assigned["attention"]}'
    - prepare_e2b.py:2021, 2023, 2029 ValueError in compose (round checks)
  - callers: compose (prepare_e2b.py:2024); the CLI __main__ block (prepare_e2b.py:2070-2130); grid.py:70 and :91 pass the
    command text (subprocess, not import)
- **`partition.attach (partition module, build copy only)`** — mesh:rdma/torch-mesh/build/lib.macosx-11.0-arm64-cpython-312/torch_mesh/partition.py:589-599 (whole body)
  - does: For each name=units in dims: n = mesh.size(index of name in mesh_dim_names) or 0 (592); if n<=0 or units<=0
    raises (593-596); else _attach(mesh, name, [units//n + (r < units % n) for r in range(n)]) (597-599): an even
    integer split of units over the n coordinates, the first units mod n one higher.
  - cites: none in the function; module docstring (partition.py:1-30) cites design/heterogeneity.md §4, R3, R14, veScale
    RaggedShard [pytorch#169320], Dao et al. 2023 Flash-Decoding
  - refuses:
    - partition.py:594-596 ValueError f'mesh: partition {name}: {units} units over a mesh dimension of
      {mesh.mesh_dim_names}, positive units and a named mesh dimension are required'
  - callers: none found in source trees (dox/mesh/rdma/*.py, metal-microbench tools): the only match 'mesh_attach' in
    dox/mesh/rdma/mesh_compile.py:1665 is the C library call, unrelated. The torch_mesh source file partition.py is
    absent from dox/mesh/rdma/torch-mesh/torch_mesh/ (only build/ and __pycache__ exist).
- **`partition._attach / partition.operand / _Operand / _blocks / split (partition module, build copy)`** — partition.py:601-606 (_attach), 608-611 (operand), 125-147 (_Operand), 149-152 (_blocks), 1043-1046 (split) (whole body)
  - does: _attach stores _Operand(key, parts) in _OPERANDS and _NAMED[mesh,name] (602-605). _Operand stores
    capacity=tuple(parts), units=sum(parts), and a tensor of capacity (127-131). _blocks(o,N) calls
    partition_bounds(o.capacity, N) (151) and returns [b-a ...] local sizes and origins (152). split(mesh,name,N)
    returns _blocks(operand(mesh,name), N)[0] (1043-1046): the local block sizes in rank order.
  - cites: none
  - refuses:
    - operand (608-611) has no check; a missing name raises KeyError from the dict lookup, implicitly
  - callers: split and _blocks: partition.py:942 (follow) and split itself; none outside the build copy
- **`mesh_collective_choose / mesh_collective_time`** — not found (not found)
  - does: not found: no definition of either name in mesh:rdma/mesh-collective.c (2383 lines; grep for both names and for
    any function beginning mesh_collective_ returned none); the only mention in the searched trees is
    heterogeneity.md:43 and :47-48
  - cites: n/a
  - refuses:
    - n/a
  - callers: none found

**Documents**

- mesh:design/heterogeneity.md: The subproblem's specification (186 lines, read in full). Vocabulary: lines 18-36 (Rate, Link, Parts, Share,
  Grain, Counts, Finish time). Requirements table lines 57-73, R1-R14; R5 (line 63): 'Integer rounding minimises the
  largest finish time: each grain goes to the rank that finishes earliest with it.' R14 (line 72): 'Parts change
  without shapes changing. Each rank's local buffer is shaped to the largest share the bounds allow it
  (allocate.capacity)'. Section 2 table lines 40-53: 'Node rates ... the derivation (allocate.min_max)' and 'Rates
  decide the parts. The link map decides the algorithm.' (45). Section 4 operand lines 85-98 names
  partition.attach(mesh, tp=(capacity, parts)) (86), partition.write (87), partition.sizes (88). Derivation steps
  lines 100-121: step 2 cost[r](c) (107), step 3 shared(c) (116-118), step 4 'Call allocate.min_max(P, 1, [1] * n,
  high, cost, shared)' (119), step 5 allocate.capacity (121). Links' beta estimator lines 126-135. Limits lines
  142-150: 'One set of parts per mesh dimension' (143). Line 22 names configs/links/pair-ring.txt. Line 43 names
  mesh_collective_choose.
  - sources it names: Beaumont et al. 2001 (ScaLAPACK); Song et al. 2020 (AccPar); Zhang et al. 2024 (HAP); pytorch#169320
    (RaggedShard); MPI 4.1 sections 6.5-6.10; Lastovetsky & Reddy 2007; Roofline 2009; HexGen 2024; HeteroMPI 2006;
    Träff, Gropp & Thakur 2010; Hunold & Carpen-Amarie 2017; Megatron-LM 2019; Ibaraki & Katoh 1988; Whale 2022;
    Metis 2024; Bitran & Hax 1981; LB-BSP 2020; Striped Attention 2023; DFPA 2011; Meta-Balancer 2012; Wang et al.
    2023; Domino 2024; FLUX 2024; Hockney 1994; GSPMD 2021; Charm++ 1993; Patarasuk & Yuan 2009; MagPIe 1999;
    BlueConnect 2019; Blink 2020; torch.chunk docs; pytorch#198344
- mesh:rdma/allocate.py: Module docstring line 1: '"""design/joint-compilation.md §3, row 9"""'. No other comments or docstrings in the
  file (88 lines).
  - sources it names: design/joint-compilation.md (not read)
- metal-microbench:tools/mesh/grid.py: Docstring lines 2-12: 'one share per rank (an even split by default)' (line 7); 'Share r is rank r's: a rank is
  its node's index in its bridge's map ... not its place in --nodes.' (lines 8-11). Defaults lines 146-147 (shares =
  '1,...,1' per rank). Lines 134-135 help: 'the TP placement operand: one share per rank, share r the rank at map
  index r'.
  - sources it names: design/page-queue-change-request.md#collective-size (line 125, comment);
    design/page-queue-change-request.md#the-functions (line 126, comment)
- metal-microbench:tools/mesh/placements/even-5.json: Dict keyed 'L.attention' and 'L.mlp' for L=0..19 (read through layer 14 in the output). Values are integer share
  lists per rank. Layer 0: attention [2,2,2,1,1] (sum 8); mlp [1232,1228,1228,1228,1228] (sum 6144). Layers 1-14 as
  layer 0 (read).
- metal-microbench:tools/mesh/placements/attn-7-1-mlp-ledger-bal.json: Dict keyed 'L.attention' and 'L.mlp' for L=0..19 (read through layer 18). Layers 0-14: attention [7,1], mlp
  [4660,1484] (sum 6144). Layers 15-18: attention [7,1], mlp [9392,2896] (sum 12288). Two ranks.
- metal-microbench:docs/tensor_parallel_execution.md: Section 'Unequal assignment candidates' at line 51: 'Each pair is the best complementary cut among the measured
  candidates. Times are µs.' and 'The maximum combines separate local medians; it is **not measured distributed
  latency**.' (lines 51-55). Table rows 58-63 (M5/M4 extents per operation and rows). Later in the section: 'This is
  evidence for selecting work, extents and backends from live cost curves. It does not establish that TP is
  unsuitable for the machines...'. No citation of an external source in the section.
- metal-microbench:docs/hardware_roofline.md: Section '3. Use measured, unequal views; tiny shards need repeatability checks' at line 177. Quoted: 'Consequently
  no single automatically selected partition is installed from these samples.' and 'Plan from shape-specific service
  curves, clock state and publication cost, and validate the chosen workload's duty cycle.' Also: 'Every tested
  distributed projection still loses to its same-cohort whole-M5 baseline.' Exact sentence line numbers within
  177-200 not pinned.
- metal-microbench:docs/measurement.md: Lines 75-76: 'MESH_PARTITION="tp=3,13 cp=4,12" ... derived from the link map's proj rates by allocate.min_max (a
  head a unit for TP; ...)'. Line 84: 'at which min_max gives cp=5,11'. Lines 80-86 note the map's proj rates (fp16,
  4.1x per FMA) differ from the fp32 MPS programs' rates (2.3-2.4x).
- mesh:rdma/torch-mesh/build/lib.macosx-11.0-arm64-cpython-312/torch_mesh/partition.py: Module docstring (lines 1-30ish, read in part): 'a program declares a mesh dimension's P relative units,
  attach(mesh, tp=P)' and 'Each rank holds the interval between two integer-rounded cumulative shares of the
  declared extent.' 'The initial partition is even.' Line 100: 'from allocate import partition_bounds'. Only the
  build copy exists; the source torch_mesh/partition.py is absent.
  - sources it names: design/heterogeneity.md §4, R3, R14; veScale RaggedShard (pytorch#169320); Dao et al. 2023 Flash-Decoding

**The same thing computed in more than one place**

- Cumulative cut of an extent by shares: metal-microbench:tools/prepare_e2b.py:466-468 column_cuts
  (width*cumsum//sum, floor) and
  mesh:rdma/torch-mesh/build/lib.macosx-11.0-arm64-cpython-312/torch_mesh/partition.py:149-152 _blocks via
  partition_bounds (defined in no source file; partition.py docstring lines 5-8 say 'two integer-rounded cumulative
  shares of the declared extent').
- Converting fractional shares to whole units or grains: prepare_e2b.py:920-938 kv_pattern (largest-remainder
  rounding when the period exceeds 256), prepare_e2b.py:466-468 column_cuts (floor), mesh:rdma/allocate.py:62
  equal_finish (floor to grain, then min_max refinement at 63). Three rounding rules for the same conversion.
- Per-rank upper bound in whole grains (the buffer or share bound): mesh:rdma/allocate.py:87-88 capacity (min(high,
  total-sum(low)+low), floored to grain) and mesh:rdma/allocate.py:72-73 min_max.fill (c_i+grain <= min(high_i,
  cap)); and the per-rank buffer sizes from partition.py:126-131 _Operand.capacity (= the attached parts) and
  partition.py:599 attach (even integer split). Two mechanisms yield per-rank local sizes.
- Even integer split of units over ranks: partition.py:599 (units//n + (r < units%n)) and the default shares in
  grid.py:146-147 (one unit per rank); the second is an operand value, not a computation.

**No function found, or only a refusal**

- Deriving cost[r](c) from node rate constants (proj=, attn=, mm32=, sdpa32=) as heterogeneity.md:20-26 and :100-107
  describe: no function found that reads rates; grep for 'mm32', 'sdpa32', 'proj=' in tools/mesh/*.py and tools/*.py
  returned nothing.
- The link map: heterogeneity.md:22 names configs/links/pair-ring.txt; metal-microbench:configs contains only
  compute, draft, structure.json; no configs/links exists. No reader of link costs found.
- shared(c), the collective time at the largest share (heterogeneity.md:116-118): no function computes it.
  mesh_collective_time not found; allocate.min_max's shared argument has no caller in code.
- The rank-to-node mapping the derivation needs (heterogeneity.md:108-109, tools/torch_parallel/pair.py RANKS):
  tools/torch_parallel/pair.py and parts.py do not exist.
- partition.write and partition.sizes (heterogeneity.md:86-88): no def named write or sizes in the partition build
  copy; the nearest is split (partition.py:1043).
- partition.attach with tp=(capacity, parts) (heterogeneity.md:86): the code's attach takes tp=units, an integer,
  and derives parts itself (partition.py:589-599).
- from_local without shape means a capacity block (heterogeneity.md:92): no def from_local in partition.py; a class
  _FromLocal exists at partition.py:1135 (not read).
- attach raising when a mesh dimension precedes a partitioned one (heterogeneity.md:94-95): the attach body
  (partition.py:589-599) and _attach (601-606) contain no such check; the only raise is 594-596.
- Every extent split must be a multiple of P, other extents raise ValueError (heterogeneity.md:93-94): no such check
  in partition.py's _blocks/split/attach; partition_bounds (where a check would sit) is not defined in any source
  file.
- Causal or nonlinear cost functions (R10, heterogeneity.md:68): no function found; cost callables are supplied by
  callers (allocate.py:63) and none exists in the repo.
- Online estimation and step control of parts (R11): Allocator.__call__ (allocate.py:20-44) is the only code; it has
  no caller in the repo. The partition.py docstring mentions 'Asynchronous rank-max evidence' updates, with no
  function found in the lines read.
- Padding masks and identity fills (R14, heterogeneity.md:72): no function found; partition.py docstring says 'No
  block holds padding'.
- Per-tensor parts (heterogeneity.md:143-146): not built in partition.py per its own text; per-section shares do
  exist in prepare_e2b.py (layer_placement keys 'L.attention', 'L.mlp', prepare_e2b.py:1588-1589).

**Not determined**

- partition_bounds: the name is imported by partition.py:100 (build copy) and used at partition.py:151, and is
  absent from mesh:rdma/allocate.py. A stale bytecode file mesh:rdma/__pycache__/allocate.cpython-312.pyc matched a
  grep for partition_bounds (binary match); its contents were not read. Whether any other source defines it: not
  found.
- The rest of prepare() in metal-microbench:tools/prepare_e2b.py (about lines 1372-1576 and 1690-1960) was not read;
  the helpers it calls (head_geometry, head_axes, vocabulary, kv_sections) were not read whole.
- partition.py beyond the lines read (follow at 893, plan at 1021, _local_redistribute, the RaggedShard methods
  481-587, distribute_tensor at 1193) was not read.
- Callers of prepare_e2b.column_cuts, packed_cuts, partition and combine outside prepare_e2b.py were not searched
  exhaustively (tools/draft/build_draft.py imports only value, operation and LAYER from prepare_e2b, per its line
  67).
- Callers of grid.nodes_from and grid.Node inside tools/mesh/run_pair.py were not read (only its line 32 comment).
- mesh:design/joint-compilation.md (named in allocate.py:1, '§3, row 9') exists but was not read.
- metal-microbench:docs/hardware_roofline.md section 'Use measured, unequal views' (line 177 onward): quoted
  sentences were read; exact sentence line numbers within 177-200 were not pinned.
- Whether allocate.min_max always places every grain: fill (allocate.py:72) runs (total-sum(low))//grain steps and
  leaves any remainder unassigned; whether callers ever pass such totals was not determined.

## Ragged tensors and ragged relations

Read: 15 whole body, 4 declaration only, 2 part of the body. 21 functions.

- **`RaggedShard (class)`** — dox/mesh git blob be5d8920:rdma/torch-mesh/torch_mesh/partition.py:481-587 (whole class read); same class also on disk at rdma/torch-mesh/build/lib.macosx-11.0-arm64-cpython-312/torch_mesh/partition.py:481 (different text, e.g. its _shard_tensor docstring differs) (whole body)
  - does: Subclass of torch Shard holding (dim, key). _operand(num_chunks) (:493-500) looks up _OPERANDS[key] and raises
    if num_chunks != len(o.capacity). local_shard_size_and_offset (:502-505) and _local_shard_size_and_offset
    (:507-508) return sizes[rank], offsets[rank] from _blocks. _split_tensor (:510-512) torch Tensor.split on the
    blocks along dim and returns contiguous copies with zero offsets. _select_split_tensor (:514-518) narrows dim to
    the index's block. _shard_tensor (:520-554) computes counts = block size x prod(rest), tries _scatter_spans,
    otherwise stages pieces into an _empty buffer and calls dist.scatter inside _configuration_exchange with an
    alltoall count matrix. _reduce_shard_tensor (:556-562) calls _reduce_scatterv with the block sizes.
    _to_replicate_tensor (:564-567) calls _allgatherv with the block sizes, so the blocks laid end to end are the
    logical tensor. _to_partial_tensor (:569-576) zero-pads the local block to its logical place with torch.cat.
    _to_new_shard_dim (:578-581) replicates then calls RaggedShard(new_dim, key)._replicate_to_shard. No padding is
    added: block sizes come from _blocks.
  - cites: none in body. Class docstring at :481-485 says: 'Shard(dim) over a partitioned mesh dimension, `key` its operand
    (attach): each rank's local buffer its capacity block, its valid elements (its part's) first... It equals
    Shard(dim), as DTensorSpec requires of a shard placement (_verify_shard_order); this module compares the key too
    (_same).' Module docstring line 13 cites 'veScale's RaggedShard(local_units) [pytorch#169320]'. Line 6 cites
    'design/heterogeneity.md §4, R3, R14'.
  - refuses:
    - NotImplementedError f'mesh: {self!r} over {num_chunks} coordinates, a flattened or unflattened partitioned
      mesh dimension of {len(o.capacity)}' at partition.py:503 (in _operand, reached by every method above)
  - callers: Within partition.py at be5d8920: constructed at :922/:938/:966/:1124/:1332/:1393 (RaggedShard(...) from
    _same/_redistribute/_keys/_partitioned helpers), used via isinstance at ~15 sites (lines 922, 981, 1118, 1120,
    1131, 1197, 1248, 1322, 1331, 1346, 1350, 1393, 1426, 1471, 1532, 1551, 1607). Outside partition.py in the
    tracked tree at be5d8920: none found in examples/ or rdma/ (git grep 'RaggedShard' limited to
    rdma/examples/tools at be5d8920 returned only partition.py and the README). Working tree: none (only build/
    copy).
- **`_blocks`** — be5d8920:rdma/torch-mesh/torch_mesh/partition.py:149-152 (whole body)
  - does: bounds = partition_bounds(o.capacity, N) (the parts of the operand taken as capacities); returns ([b - a for
    consecutive bounds], list(bounds[:-1])), i.e. local sizes and origins of extent N.
  - cites: none in body; docstring: 'Exact local sizes and origins of the declared logical extent.' (:150)
  - refuses:
    - none in body
  - callers: partition.py: _split_tensor (:512), _select_split_tensor (:516, :~519), _shard_tensor (:535),
    _reduce_shard_tensor (:560), _to_replicate_tensor (:566), _to_partial_tensor (:571), local_shard_size_and_offset
    (:504), split() (:1046). About 8 call sites in partition.py.
- **`partition_bounds`** — be5d8920:rdma/allocate.py:8-19 (also present at 444d4fd9:rdma/allocate.py:8) (whole body)
  - does: total = sum(parts); returns tuple(extent * prefix // total for prefix in itertools.accumulate(parts,
    initial=0)): integer prefix-rounded cumulative cut points of extent over the parts' weights. Exact coverage:
    first cut 0, last cut extent.
  - cites: docstring :9-16: 'Exact adjacent cuts of a declared extent, weighted by rank shares. Integer prefix rounding
    preserves coverage, including tails and empty shares. Divisible extents retain the original capacity-block
    boundaries. This relation is consumed during preparation, never by numerical replay.' File line 1: '#
    design/whole-function.md' (a document name, not a citation for the method).
  - refuses:
    - none in body; if sum(parts) == 0 the division raises ZeroDivisionError implicitly (no explicit message).
      migration.py:~112-116 checks 'not sum(old_parts) or not sum(new_parts)' and raises ValueError('partitions must
      have positive total weights and the same ranks') before calling it.
  - callers: be5d8920: partition.py:151 (_blocks), partition.py:942 (_reconfigure path, direct call for old and new
    operands), migration.py:20 (import) and migration.py:117 (before/after bounds). At HEAD in dox/mesh rdma/: git
    grep finds no partition_bounds call site (allocate.py exists at HEAD, migration.py does not).
- **`attach`** — be5d8920:rdma/torch-mesh/torch_mesh/partition.py:589-599 (whole body)
  - does: For each name=units: n = mesh.size(index of name in mesh_dim_names) or 0; raises if n <= 0 or units <= 0;
    otherwise calls _attach(mesh, name, [units // n + (r < units % n) for r in range(n)]), i.e. an even integer
    split of the declared units over the n coordinates in rank order.
  - cites: docstring :590-592 'Partition mesh dimensions by name: attach(mesh, tp=units), the relative work units, the same
    on every rank. The backend splits them evenly over the coordinates in rank order (the first units mod n one
    more).' Module docstring :2-7 describes the attach(mesh, tp=P) parts as relative units with interval boundaries
    (design/heterogeneity.md §4, R3, R14 at :6).
  - refuses:
    - ValueError f'mesh: partition {name}: {units} units over a mesh dimension of {mesh.mesh_dim_names}, positive
      units and a named mesh dimension are required' at partition.py:596-597
  - callers: examples/chain-service.py:59, examples/cp.py:31, examples/ep.py:60, examples/fabric-dnn.py:19,
    examples/lstm-service.py:51, examples/tp.py:102 (be5d8920): 6 sites.
- **`split`** — be5d8920:rdma/torch-mesh/torch_mesh/partition.py:1043-1046 (whole body)
  - does: o = operand(mesh, name); returns _blocks(o, N)[0], the local sizes of extent N in rank order (raw tensors; the
    list collectives take these).
  - cites: docstring :1044 'The valid blocks of an extent N along partitioned mesh dimension `name` under the current
    parts, in rank order.' Module docstring :10-12 cites MPI v-collectives in torch spelling (dist.all_gather of
    unequal outputs, reduce_scatter of unequal inputs).
  - refuses:
    - none explicit; a name never attached raises KeyError from _OPERANDS[_NAMED[mesh, name]] inside operand()
      (partition.py:~254), with no message
  - callers: examples/chain-service.py:76, examples/ep.py:39, examples/fabric-dnn.py:25, examples/lstm-service.py:91,
    examples/lstm.py:27 (be5d8920): 5 sites.
- **`from_local`** — be5d8920:rdma/torch-mesh/torch_mesh/partition.py:1217-1238 (whole body)
  - does: placements normalized; if no RaggedShard, delegates to torch DTensor.from_local. Otherwise requires shape and
    stride, moves the local tensor to the mesh device type if needed, and if run_check broadcasts each Replicate
    mesh dimension via mesh_broadcast; returns _register(_FromLocal.apply(local, mesh, placements,
    torch.Size(shape), tuple(stride), grad_placements)).
  - cites: docstring :1218-1221 'Bind a local page view to its declared global shape, stride and placement. Ragged cuts
    cannot determine a global extent from one rank's local size. The frontend supplies that metadata, including on
    empty ranks.' No literature cited.
  - refuses:
    - ValueError('a ragged local operand requires its declared global shape and stride') at partition.py:1228 (only
      when placements are ragged and shape or stride is None)
  - callers: examples/tp.py:81 (partition.from_local(held, ...)) and rdma/torch-mesh/torch_mesh/declarations.py:547
    (be5d8920). 2 sites.
- **`_keys`** — be5d8920:rdma/torch-mesh/torch_mesh/partition.py:1341-1354 (whole body)
  - does: Over the input specs, records {mesh dimension i: key of its RaggedShard} and raises when two RaggedShards on the
    same dimension carry different keys; then raises if a plain Shard or _StridedShard meets a RaggedShard on a
    dimension in keys. Returns keys.
  - cites: message at :1351-1352 cites 'design/heterogeneity.md R6: tensors that meet split their shared extent alike'.
  - refuses:
    - ValueError f'mesh: tensors partitioned by {keys[i]} and {p.key} meet on mesh dimension {i}' at
      partition.py:1347
    - ValueError f'mesh: a {p} meets {keys[i]}\'s RaggedShard on mesh dimension {i} (design/heterogeneity.md R6:
      ...)' at partition.py:1351-1352
  - callers: partition.py only (not searched further).
- **`_ragged and _same`** — be5d8920:rdma/torch-mesh/torch_mesh/partition.py:1049-1050 (_ragged), :1053-1056 (_same) (whole body)
  - does: _ragged: any placement isinstance RaggedShard. _same: tuples equal in length, and each pair has equal type,
    equal placement, and equal getattr(key).
  - cites: _same docstring :1054 'Placements alike, a RaggedShard's key and type included (torch's equality takes a
    RaggedShard for Shard(dim)).' none for _ragged.
  - refuses:
    - none
  - callers: _ragged: partition.py:1131, 1197, 1214 region, 1224, 1248 and others (about 8 in partition.py). _same:
    partition.py:~1062, 1102 region, plus comment reference at :~1102.
- **`RaggedTensor (class)`** — be5d8920:rdma/torch-mesh/torch_mesh/partition.py:1253 (class body runs to ~1800; not read in full) (declaration only)
  - does: DTensor subclass; declaration and docstring at :1254-1255: 'A DTensor holding a RaggedShard. Its ops are
    dispatched here (_dispatch) and its to_local, redistribute and full_tensor keep its gradients ragged; every
    DTensor one makes that holds a RaggedShard is one.' Only its raise sites were read.
  - cites: docstring only (no literature).
  - refuses:
    - NotImplementedError('mesh: redistribute\'s forward_dtype and backward_dtype on a RaggedShard') at
      partition.py:1285
    - NotImplementedError('mesh: ragged SDPA takes no attn_mask, dropout or enable_gqa') at partition.py:1557
  - callers: from_local/distribute_tensor/_FromLocal paths in partition.py (:1131, 1136, 1197, 1214); not counted further.
- **`ragged SDPA entry`** — be5d8920:rdma/torch-mesh/torch_mesh/partition.py:1543-1566 (name not given in source; docstring at :1543 'SDPA on query, key and value split along the sequence (dimension 2) by one RaggedShard, on any mesh dimension...') (part of the body)
  - does: Checks exactly one RaggedShard on query along dim 2 (:1551), key and value share placements (:1552), other
    placements not Partial and not Shard past dim 1 (:1552-1553); takes held.key; calls _ring_attention(...) at
    :1561 and wraps outputs as RaggedTensor (:1563, :1566). Not read beyond these lines.
  - cites: docstring cites ring attention per the module docstring (:29 'Attention: SDPA ... is ring attention, each
    source's keys past its valid extents masked (score -inf)').
  - refuses:
    - NotImplementedError('mesh: ragged SDPA takes no attn_mask, dropout or enable_gqa') at partition.py:1557-1560
      (line 1560 in grep)
    - raise UnsupportedOperatorException(op_call) at partition.py:1554 (condition on the placement check)
  - callers: dispatched from RaggedTensor dispatch; not searched further.
- **`_partition_pad`** — rdma/mesh_compile.py:673-681 (dox/mesh HEAD) (whole body)
  - does: If shape[dimension] == extent returns key. Otherwise makes key + ('padded', dimension, extent), records its
    shape, and emits ('pad', (key,), made, (dimension, extent)) into the IR. Lowering of 'pad' is at
    rdma/mesh_compile.py:1261 and :6979 (eval side constant_pad_nd with extent - shape[dim] trailing zeros, per the
    27b8b77c diff).
  - cites: none in body.
  - refuses:
    - none in body
  - callers: rdma/mesh_compile.py:715 (exchange), :745 (_reshard slice path), :903 (_partition). 3 sites.
- **`_reshard`** — rdma/mesh_compile.py:684-752 (dox/mesh HEAD; nested helpers gather() and exchange() at :688-705 and :708-736) (whole body)
  - does: Moves a value from sharding `have` to `want` over mesh dimensions. gather(): all-gather a mesh dimension: padded
    = local extent x mesh size; if padded != size it emits an all-gather into a padded buffer then 'narrow' to size.
    exchange(): all-to-all between two dims with source padded to ceil(shape[j]/P)*P via _partition_pad (:715).
    Loops until no dimension moves; slice path (:745) pads the local extent to ceil(local/P)*P before 'slice'.
    Padding in the sense of ceil(n/P)*P is produced by the -(-n // P) * P expressions at :715, :718, :745 and then
    narrowed back.
  - cites: none in body. A comment '# GSPMD Appendix A.2 and Figure 9b' sits above _extent (mesh_compile.py ~line 753), not
    above _reshard.
  - refuses:
    - none in body
  - callers: rdma/mesh_compile.py:820, 824, 831, 835, 859, 910 inside _partition (6 sites).
- **`_extent (window extents, neighbour of the GSPMD comment)`** — rdma/mesh_compile.py:~754-775 (declaration only)
  - does: Computes per-part window starts and lengths for conv windows; not read in full.
  - cites: comment 'GSPMD Appendix A.2 and Figure 9b' directly above it.
  - refuses:
    - not read
  - callers: not searched.
- **`Matrix.__new__ (streamed.py input relation)`** — dox/mesh git blob 444d4fd9:rdma/torch-mesh/torch_mesh/streamed.py:683-700 (part read) (part of the body)
  - does: Sets out.input_sources = tuple(range(columns)) if None, out.input_starts = tuple(range(columns+1)) if None;
    stores them as the relation coordinate i -> logical columns input_sources[input_starts[i]:input_starts[i+1]]
    (comment :696-697).
  - cites: inline comment :696-697: 'the input's relation to the rotated coordinates: coordinate i sums the logical input
    columns input_sources[input_starts[i]:input_starts[i + 1]], each coordinate its own extent'.
  - refuses:
    - none
  - callers: Matrix.like (:729-740), Matrix.selected (:776-812).
- **`Matrix.selected`** — 444d4fd9:rdma/torch-mesh/torch_mesh/streamed.py:776-820 (read whole) (whole body)
  - does: Composes a row/column selection with retained blocks; when rotated, builds new sources/starts: for each
    coordinate i, appends inverse[source] for each source in input_sources[starts[i]:starts[i+1]], then
    starts.append(len(sources)) (:800-806), so an empty group where no source is selected. Produces the new Matrix
    via self.like(...) (:808-809) with input_sources and input_starts. Sets out.label from TensorLayout spans.
  - cites: inline comment: 'a coordinate's group becomes the selected columns of its sources: its own extent, empty where
    none' (:800).
  - refuses:
    - none explicit
  - callers: streamed.py:773 (slicing), streamed.py:1773-1774 (streaming plan), and internal. 3 sites.
- **`expand`** — 444d4fd9:rdma/torch-mesh/torch_mesh/structured.py:94-105 (whole body)
  - does: If starts is empty, _validate(x, signs, width). Then for each (local, step) in plan(width): x = stage(x, signs,
    local, step, normalization..., signed and step == 1, list(sources) if step == 1 else [], list(starts) if step ==
    1 else []). Returns x.
  - cites: docstring :95-99: 'Expose local factors before graph partition and collective placement. Where `starts` is
    given, coordinate i of the first factor's input is the sum of x's columns sources[starts[i]:starts[i + 1]], each
    coordinate its own extent (an empty one is zero); otherwise x is already in coordinates.' No literature cited.
  - refuses:
    - ValueError from _validate at structured.py:89 'fwht binds floating point rows and one int8 or fp32 sign per
      column' and :91 'fwht blocks must be power-of-two factors of the row width' (only when starts is empty)
  - callers: structured.py:42 (_backward), :113 (prepare), :219 (Rotation); streamed.py:872, :934, and selected path. About 5
    sites.
- **`stage (torch_mesh::fwht_stage custom op)`** — 444d4fd9:rdma/torch-mesh/torch_mesh/structured.py:48-51 (op body), :53-55 (fake), :57-76 (context and backward) (whole body)
  - does: Op body raises. Backward (:64-78): result = stage(grad, ..., signed False, [], []); if signed multiply by signs;
    if starts: builds coordinates list (each coordinate i repeated starts[i+1]-starts[i] times), index tensor from
    sources, and returns zeros(ctx.shape).index_add(1, index, result[:, coordinates]), i.e. each coordinate's
    gradient is added back into its own source columns.
  - cites: none; backward comment at :70 'each coordinate's gradient returns to its own sources: x[:, sources[k]] +=
    result[:, i] for k in its group'.
  - refuses:
    - RuntimeError('butterfly stages require whole-function preparation') at structured.py:50, raised by the eager
      op body
  - callers: expand (:103), prepare_stage path, streamed.py via expand. Several sites.
- **`_validate`** — 444d4fd9:rdma/torch-mesh/torch_mesh/structured.py:87-91 (whole body)
  - does: Checks x is 2-D, signs shape equals x.shape[-1:], dtypes in (float16, bfloat16, float32) and signs in (int8,
    float32), width > 0 power of two dividing x.shape[-1].
  - cites: none.
  - refuses:
    - ValueError 'fwht binds floating point rows and one int8 or fp32 sign per column' at structured.py:89
    - ValueError 'fwht blocks must be power-of-two factors of the row width' at structured.py:91
  - callers: expand (:97), prepare (:109).
- **`prepare_stage`** — 444d4fd9:rdma/torch-mesh/torch_mesh/structured.py:126-~170 (declaration only)
  - does: Not read beyond its signature; it is the function that produces the commands native.py:125 consumes (output,
    commands = prepare_stage(*args, out=out)).
  - cites: not read
  - refuses:
    - not read
  - callers: structured.py:113, native.py:125.
- **`butterflyCommands (Swift)`** — metal-microbench git blob 3f308655:matrix_operations.swift:544-592 (read whole); comment above it at :541-543: 'The same factor plan and numerical body used by the FX frontend. Each local factor reads and publishes declared tiles of preassigned page storage.' (whole body)
  - does: plan = MetalButterflyPlan(width, min(MESH_FWHT_BLOCK_WIDTH, device threadgroup memory / 4)) stages. For each
    stage builds an output TensorView [rows, columns] and coordinate views. When signed and sources/starts are
    non-empty: builds coordinates per tile, present = sources[starts[c]..<starts[c+1]] flattened, first = prefix
    sums of group sizes, extents = select over first (the per-element extents), left = value view gathered at
    present (nil when present is empty). Then per tile: binds binding 0 (left), 1 (right), 2 (signs), 4 (extents as
    Int32 layout), instances library.instance('structured_butterfly', [local, normalization, signed, value.type,
    signs.type]), appends a MetalDispatch with reaches. Returns (value, commands).
  - cites: comment at :541-543 (see above). No literature.
  - refuses:
    - no explicit refusal; force-unwrap of MetalButterflyPlan(...)! at :545 with no message
    - KernelLibrary/indexed preconditions in the called library (not read): metal-microbench matrix_operations.swift
      precondition at :172 'binding ... has no declared view' and :176 are in another function
  - callers: At HEAD (metal-microbench): git grep 'butterflyCommands' HEAD returns nothing; the function is not in HEAD.
    Present at 3f308655 (commit that changed it) and in 3f308655^ at :544.
- **`streamed_hadamard and structured_butterfly kernels (.metal)`** — dox/mesh 444d4fd9:rdma/torch-mesh/torch_mesh/streamed.metal and structured.metal (hunks only in --stat: 14 and 20 lines) (declaration only)
  - does: Not read. Commit message 444d4fd9 states: streamed_hadamard iterates [mesh_position<13>(i),
    mesh_position<13>(i+1)) of the sources table mesh_position<12>; structured_butterfly reads binding 4 (element
    extents) over a read view holding only the tile's sources.
  - cites: commit message
  - refuses:
    - not read
  - callers: streamed.py (444d4fd9) selected/expand path; not counted.

**The same thing computed in more than one place**

- Block sizes and origins of an extent over parts: _blocks
  (be5d8920:rdma/torch-mesh/torch_mesh/partition.py:149-152) calls partition_bounds (rdma/allocate.py:8);
  partition.py:942 calls partition_bounds directly on old and new operands; migration.py:117 calls partition_bounds
  for old and new parts and differences them into sizes (be5d8920:rdma/migration.py:117-118). Same computation
  (cumulative-share cuts of an extent, differenced to sizes) in three places.
- Ceil-to-multiple padded extent -(-n // P) * P: mesh_compile.py:715 (exchange source), :718 (exchange padded
  shape), :745 (slice), :903 (_partition) and the gather padded = shape[d] * P at :692; the same padded-extent rule
  written inline at four or more sites.
- Coordinate-to-sources relation sum over sources[starts[i]:starts[i+1]]: 444d4fd9 streamed.py:803 (selected), :877
  (matrix sums), :897-898 (equality), structured.py:99 docstring and expand, and the Swift butterflyCommands
  'present' computation at 3f308655 matrix_operations.swift:568-570 (same relation, now in Swift).
- Ragged-placement detection: _ragged (partition.py:1049-1050) and _keys (partition.py:1341-1354) both scan
  placements for RaggedShard; _same (:1053) and _keys both compare keys per mesh dimension.
- Per-coordinate gradient return to sources: structured.py:70-77 (_stage_backward index_add) and Matrix backward in
  streamed.py (commit 444d4fd9 message: 'The stage backward returns each coordinate's gradient to its own sources').

**No function found, or only a refusal**

- Ragged representation at HEAD of dox/mesh: no torch-mesh Python source is tracked at HEAD (git ls-tree -r HEAD |
  grep torch_mesh is empty). RaggedShard, RaggedTensor, partition.py, streamed.py and structured.py are not in HEAD.
  Last tracked partition.py is at 55f8a1af (git log -1 3981ca80^ -- rdma/torch-mesh/torch_mesh/partition.py);
  3981ca80 deletes partition.py (830 lines, git show --stat 3981ca80); 0b3e78d7 deletes 8 files (2867 deletions)
  under rdma/torch-mesh.
- Ragged code in metal-microbench HEAD: no tracked .mm, .m, .metal, .h, .swift, .py or .c file contains 'ragged'
  (git grep at HEAD). No 'StreamedInputDomain', 'inputDomain' or 'input_domain' tracked at HEAD (0 hits).
  butterflyCommands and structured_butterfly are not referenced at HEAD (0 hits).
- RaggedShard in the tracked tree: absent at 55f8a1af (git grep for 'ragged' in *.py at 55f8a1af returned nothing;
  grep for attach/operand/_partitioned returned attach at :222, operand at :249, _partitioned at :506 only). The
  class exists at be5d8920 (partition.py:481) and in the gitignored build/ copy.
- Lowering of a ragged relation to Metal kernels in dox/mesh HEAD: none found (mesh_compile.py has no 'ragged' hit).
- Sending without padding for uneven shards at HEAD: the only send-side code at HEAD is mesh_compile.py _reshard
  (padding by ceil) and 27b8b77c's mesh-graph.c/mesh-graph.h changes (70 and 2 lines), which were not read.
- Refusals listed in be5d8920 partition.py for unsupported ragged cases, each with a message: NotImplementedError at
  :503 (flattened/unflattened dim), :1076 (_StridedShard with partitioned dim), :1285 (forward_dtype/backward_dtype
  on RaggedShard), :1330 (_StridedShard FSDP2 with TP), :1557 (ragged SDPA with attn_mask/dropout/enable_gqa), :1604
  (LocalTensor RNG on RaggedShard), :1590 (DTensor RNG over _StridedShard), :1733 (BlockMask on partitioned dim);
  ValueError at :1228 (ragged from_local without shape/stride), :1347, :1351 (mismatched keys), :596 (attach units),
  :735, :856 (migration). The module docstring at be5d8920:partition.py:82-84 lists the NotImplementedError cases as
  'NotImplementedError on every rank alike'.
- Mesh_compile.py ragged-refusal: 27b8b77c diff removed a NotImplementedError at the slice path ('a dimension of %d
  over a mesh dimension of %d (GSPMD section 4.1 pads ...') and replaced it with ceil division; the HEAD file has no
  such raise (grep at HEAD returns none for 'GSPMD section 4.1').

**Not determined**

- Whether the gitignored build/ copy of partition.py
  (rdma/torch-mesh/build/lib.macosx-11.0-arm64-cpython-312/torch_mesh/partition.py) matches any tracked commit; its
  RaggedShard text differs from be5d8920:partition.py:481-587 (for example its _shard_tensor docstring at build copy
  :530-535 differs). Not determined which commit it was built from.
- Order of commits 1f794b8f and eecfb049 relative to be5d8920 (git log -S 'class RaggedShard' returned 1f794b8f and
  eecfb049; the ancestry between them and be5d8920 was not checked).
- Callers of RaggedShard, _blocks, _keys and _partition-related functions outside partition.py in the tracked tree
  at be5d8920, beyond the sites listed (not exhaustively searched; examples/ and rdma/ were searched by name only
  for the functions listed).
- RaggedTensor body (partition.py:1253-~1800), _ring_attention (:1802), _reshard's full contract with mesh-graph.c
  (27b8b77c), streamed.metal and structured.metal kernel bodies (444d4fd9), prepare_stage body
  (structured.py:126-~170), Matrix.selected's remaining lines after 820 (streamed.py), and metal-microbench
  streamed_weights.swift (3f308655 and d3d0f4bc: 'ragged halves', 'ragged' in commit message only).
- Whether allocate.py at HEAD is identical to be5d8920:rdma/allocate.py (file exists at HEAD; content not diffed).
- The meaning of 'ragged' in metal-microbench docs beyond the three hits (docs/provider.md:759 'ragged thirteen-row
  FFN shares', docs/soc_compute_backends.md:404 'optimal ragged TP split') and the
  'docs/placement.md#the-records-contract-for-every-ragged-axis' reference in the build copy of streamed.py:1712
  (docs/placement.md does not exist: ls failed).
