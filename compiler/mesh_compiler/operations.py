# design/compilation-composition-change-request.md, numerical lowerer boundary and "Complete replacement";
# design/torch-source-lowering-change-request.md items 5 and 6: the explicitly selected custom numerical methods (the
# CPU LAPACK drivers, and on the placed processor QDWH polar, eigh and SVD, the pseudoinverse and least-squares
# drivers, the eigenvalue drivers (zgeev.f, dgeev.f) and the matrix exponential) as ATen and torch HOP bodies, their
# custom-operator metadata, the typed LAPACK routine operators (LAPACK) and the program generator's state (Generator,
# GENERATOR, Advance). Every other operation, and every compiler pass over these bodies, is the installed upstream
# compiler's. Each definition carries its citation; this module's earlier per-operator construction and its region
# record compiler are in its git history
import collections
import functools
import math
import pathlib
import sys

import torch
from torch._higher_order_ops.cond import cond_op
from torch._higher_order_ops.switch import switch_op
from torch._higher_order_ops.while_loop import while_loop_op
from torch._prims_common import corresponding_complex_dtype, get_computation_dtype
from torch.fx.graph_module import _get_attr

sys.path.append(str(pathlib.Path(__file__).parents[2] / "rdma"))
import mesh_code

# torch 2.14.0 torch/export/__init__.py, export; torch/_ops.py, OpOverload
A = torch.ops.aten


# metal-microbench metal_ir.h element kinds (metal_ir_value's kind codes) by torch dtype (SCALAR), and the placement
# kinds GPU, CPU and ANE, as compiler/mesh_compiler/metal.py, lower_metal.py and schedule.py read them
BOOL, I8, U8, I16, U16, I32, U32, I64, U64, F16, BF16, F32, F64 = 0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16


SCALAR = {torch.bool: BOOL, torch.int8: I8, torch.uint8: U8, torch.int16: I16, torch.uint16: U16, torch.int32: I32,
          torch.uint32: U32, torch.int64: I64, torch.uint64: U64, torch.float16: F16, torch.bfloat16: BF16,
          torch.float32: F32, torch.float64: F64}
GPU, CPU, ANE = 0, 1, 2


def _operands(target, args, kwargs):
    schema = target._schema.arguments
    return {**{a.name: a.default_value for a in schema if a.has_default_value()},
            **dict(zip((a.name for a in schema), args)), **kwargs}


def _linalg_vector_rhs(a, b):
    return b.dim() == 1 or b.dim() == a.dim() - 1 and b.shape == a.shape[:-1]


# PyTorch v2.14.0 aten/src/ATen/native/BatchLinearAlgebra.cpp, linalg_lstsq_out_info and get_default_lstsq_driver;
# PyTorch 2.14 torch.library.register_fake
def _lstsq_driver(a, driver):
    return driver.lower() if driver is not None else ('gels' if a.device.type == 'cuda' else 'gelsy')


@torch.library.register_fake('aten::geqrf')
def _geqrf_metadata(self):
    torch._check(self.dim() >= 2, lambda: 'torch.geqrf: input must have at least 2 dimensions')
    m, n = self.shape[-2:]
    return (self.new_empty((*self.shape[:-2], n, m)).transpose(-2, -1),
            self.new_empty((*self.shape[:-2], min(m, n))))


def _lstsq_shapes(self, b, driver):
    torch._check(self.dim() >= 2 and b.dim() >= 1,
                 lambda: 'torch.linalg.lstsq: input must have at least 2 dimensions and other at least 1')
    torch._check(0 <= self.dim() - b.dim() <= 1,
                 lambda: 'torch.linalg.lstsq: input.dim() - other.dim() must be 0 or 1')
    torch._check(self.dtype == b.dtype and self.device == b.device,
                 lambda: 'torch.linalg.lstsq: input and other must have the same dtype and device')
    selected = _lstsq_driver(self, driver)
    torch._check(selected in ('gels', 'gelsy', 'gelsd', 'gelss'),
                 lambda: 'torch.linalg.lstsq: driver must be gels, gelsy, gelsd or gelss')
    # Rejection filters are not part of even temporary integration validation.
    # PyTorch's CUDA-provider gels-only check (lstsq-pytorch.reference.cpp 3499-3503) is not Mesh's: a placed lstsq
    # is lowered by Mesh's own _lapack_least_squares (CPU), _least_squares_qr and _least_squares_svd, each driver
    vector = _linalg_vector_rhs(self, b)
    rhs = b.unsqueeze(-1) if vector else b
    m, n = self.shape[-2:]
    torch._check(m == rhs.shape[-2], lambda: 'torch.linalg.lstsq: input and other row sizes must match')
    batch = torch.broadcast_shapes(self.shape[:-2], rhs.shape[:-2])
    return selected, vector, m, n, batch, rhs.shape[-1]


# the results whose shapes the data does not decide: solution, rank and singular_values, as PyTorch gives each driver's
def _lstsq_fixed(self, shapes):
    selected, vector, m, n, batch, nrhs = shapes
    solution = self.new_empty((*batch, nrhs, max(m, n))).transpose(-2, -1)[..., :n, :]
    if vector:
        solution = solution.squeeze(-1)
    rank = self.new_empty(self.shape[:-2] if selected != 'gels' else (0,), dtype=torch.int64)
    singular = self.real.new_empty((*self.shape[:-2], min(m, n)) if selected in ('gelsd', 'gelss') else (0,))
    return solution, rank, singular


@torch.library.register_fake('aten::linalg_lstsq')
def _lstsq_metadata(self, b, rcond=None, *, driver=None):
    shapes = _lstsq_shapes(self, b, driver)
    (selected, _, m, n, batch, nrhs), (solution, rank, singular) = shapes, _lstsq_fixed(self, shapes)
    residual = self.real.new_empty((*batch, nrhs) if m > n and (selected == 'gels' or n == 0 and selected != 'gelsy')
                                   else (0,))
    # PyTorch v2.14.0 BatchLinearAlgebra.cpp linalg_lstsq_out_info, gelsd and gelss with m > n > 0: the residuals
    # computed only where every rank is n. An empty batch's every rank is n, so (*batch, nrhs); unbatched with nrhs 0
    # they are (0,) whatever the rank (nrhs or 0 long); otherwise their shape is the data's (unbatched nrhs or 0 long,
    # batched (*batch, nrhs) or (0,))
    if m > n > 0 and selected in ('gelsd', 'gelss') and batch and not math.prod(batch):
        residual = self.real.new_empty((*batch, nrhs))
    elif m > n > 0 and selected in ('gelsd', 'gelss') and (batch or nrhs):
        from torch._subclasses.fake_tensor import DynamicOutputShapeException
        raise DynamicOutputShapeException(A.linalg_lstsq.default)
    return solution, residual, rank, singular


# PyTorch 2.14 torch.linalg.lstsq, residuals: "computed if m > n and every matrix in A is full rank; otherwise, it's an
# empty tensor": for m > n > 0 under gelsd and gelss their shape is the data's (unbatched nrhs or 0 long, batched
# (*batch, nrhs) or (0,)). torch.linalg.lstsq's results a source selects among solution, rank and singular_values alone
# (compiler/patches/torch-lstsq-fixed-results.patch, its Dynamo product and strict export selection) are this one
# operator's, lowered by the same drivers (_lapack_least_squares, _least_squares_qr, _least_squares_svd)
torch.library.define('mesh::linalg_lstsq_fixed', '(Tensor self, Tensor b, float? rcond=None, *, str? driver=None) -> '
                                                 '(Tensor solution, Tensor rank, Tensor singular_values)')


@torch.library.register_fake('mesh::linalg_lstsq_fixed')
def _lstsq_fixed_metadata(self, b, rcond=None, *, driver=None):
    return _lstsq_fixed(self, _lstsq_shapes(self, b, driver))


LSTSQ_FIXED = torch.ops.mesh.linalg_lstsq_fixed.default


# PyTorch 08187d9e0fba026dc8217405802ab5381dc88d90 (v2.14.0) torch/csrc/autograd/FunctionsManual.cpp 4009-4063,
# linalg_lstsq_backward(gX_, gL, A, B_, X_, grad_input_mask): its solution branch takes the default pinvA =
# at::linalg_pinv(A), A's gradient pinv_backward(gX.matmul(B.mH()), pinvA, A) and B's pinvA.mH().matmul(gX). Here no
# residuals are produced (gL undefined), so the gradient is that branch: the vector-Jacobian product of
# (A, B) -> linalg_pinv(A) @ B through linalg_pinv's own derivative (pinv_backward). tools/autograd/derivatives.yaml
# 958-962, linalg_lstsq output_differentiability [True, True, False, False]: rank and singular_values are not
# differentiable
def _lstsq_fixed_context(ctx, inputs, keyword_only_inputs, output):
    ctx.save_for_backward(inputs[0], inputs[1])
    ctx.mark_non_differentiable(output[1], output[2])


def _lstsq_fixed_backward(ctx, solution, rank, singular):
    a, b = ctx.saved_tensors
    vector = _linalg_vector_rhs(a, b)
    _, pullback = torch.func.vjp(lambda a, b: torch.linalg.pinv(a) @ (b.unsqueeze(-1) if vector else b), a, b)
    a_grad, b_grad = pullback(solution.unsqueeze(-1) if vector else solution)
    return a_grad, b_grad, None


torch.library.register_autograd('mesh::linalg_lstsq_fixed', _lstsq_fixed_backward, setup_context=_lstsq_fixed_context)


# design/torch-source-lowering-change-request.md item 5: the CPU drivers, each one Reference-LAPACK driver routine the
# native owner calls once (metal-microbench metal_ir.h METAL_IR_GESDD .. METAL_IR_GELSY, struct metal_ir_lapack_call;
# metal_native.mm LapackStages, LapackWorkspace32/64 and LapackMember, 9404b7b2 and 633068d4): its symbol by the
# operand's kind (s, d, c, z), its Accelerate ILP64 prototype order, its workspace queries and buffers (WORK, IWORK,
# RWORK, gelsy's JPVT zeroed per batch member), gels' fixed TRANS 'N', and INFO and RANK staged at __LAPACK_int and
# converted once on copy-out to the result's own dtype are the native owner's alone. Each routine family is one typed
# operator (mesh::lapack_*), its operands and results the routine's own: a matrix operand LAPACK reads and overwrites
# (A, B) is the call's own column-major buffer, a result matrix is that buffer (column-major, as the fake gives it),
# INFO the routine's per batch member. The operators have no eager kernel, never a Torch kernel. LAPACK[packet] is
# the routine's selection and its operands, by the operator's schema names (Call): `routine` the driver the native
# owner selects (geev for dgeev and zgeev), `jobs` its str operands in job order, `rcond` its float operand (gelsy,
# gelsd, gelss) or None, `inputs` its matrix operands, `matrices` its matrix results (a matrix per batch member where
# the job computes it, else the fake's (0,) placeholder), `info` its INFO result; every result, in schema order, is
# the call's own output. Error: INFO as PyTorch v2.14.0 BatchLinearAlgebra.cpp 1610-1690 _linalg_check_errors(info,
# api, is_matrix) raises it (INFO < 0 its illegal-argument RuntimeError, INFO > 0 its torch.linalg.LinAlgError for
# `api`), after the call, in each formulation below
Call = collections.namedtuple('Call', 'routine jobs rcond inputs matrices info')
LAPACK = {}


def _lapack(name, schema, fake, *call):
    torch.library.define('mesh::lapack_' + name, schema)
    torch.library.register_fake('mesh::lapack_' + name)(fake)
    target = getattr(torch.ops.mesh, 'lapack_' + name).default
    LAPACK[target.overloadpacket] = Call(*call)
    return target


def _column_major(like, shape, dtype=None):
    return like.new_empty((*shape[:-2], shape[-1], shape[-2]), dtype=dtype or like.dtype).mT


def _info(a, batch):
    return a.new_empty(batch, dtype=torch.int32)


def _gesdd_fake(a, jobz):
    batch, (m, n) = a.shape[:-2], a.shape[-2:]
    k = min(m, n)
    s = a.real.new_empty((*batch, k))
    if jobz == 'N':
        return s, a.new_empty((0,)), a.new_empty((0,)), _info(a, batch)
    full = jobz == 'A'
    return (s, _column_major(a, (*batch, m, m if full else k)), _column_major(a, (*batch, n if full else k, n)),
            _info(a, batch))


def _heevd_fake(a, jobz, uplo):
    batch = a.shape[:-2]
    return (a.real.new_empty(a.shape[:-1]), _column_major(a, a.shape) if jobz == 'V' else a.new_empty((0,)),
            _info(a, batch))


def _dgeev_fake(a, jobvr):
    vr = _column_major(a, a.shape) if jobvr == 'V' else a.new_empty((0,))
    return a.new_empty(a.shape[:-1]), a.new_empty(a.shape[:-1]), vr, _info(a, a.shape[:-2])


def _zgeev_fake(a, jobvr):
    return a.new_empty(a.shape[:-1]), _column_major(a, a.shape) if jobvr == 'V' else a.new_empty((0,)), \
        _info(a, a.shape[:-2])


def _least_buffer(a, b):
    return _column_major(b, (*b.shape[:-2], max(a.shape[-2:]), b.shape[-1]))


def _gels_fake(a, b):
    return _least_buffer(a, b), _info(a, a.shape[:-2])


def _gelsy_fake(a, b, rcond):
    return _least_buffer(a, b), _info(a, a.shape[:-2]), _info(a, a.shape[:-2])


def _gelsd_fake(a, b, rcond):
    return (_least_buffer(a, b), a.real.new_empty((*a.shape[:-2], min(a.shape[-2:]))), _info(a, a.shape[:-2]),
            _info(a, a.shape[:-2]))


LAPACK_GESDD = _lapack(
    'gesdd', '(Tensor a, str jobz) -> (Tensor s, Tensor u, Tensor vt, Tensor info)', _gesdd_fake,
    'gesdd', ('jobz',), None, ('a',), ('u', 'vt'), 'info')
LAPACK_HEEVD = _lapack(
    'heevd', '(Tensor a, str jobz, str uplo) -> (Tensor w, Tensor z, Tensor info)', _heevd_fake,
    'heevd', ('jobz', 'uplo'), None, ('a',), ('z',), 'info')
LAPACK_DGEEV = _lapack(
    'dgeev', '(Tensor a, str jobvr) -> (Tensor wr, Tensor wi, Tensor vr, Tensor info)', _dgeev_fake,
    'geev', ('jobvr',), None, ('a',), ('vr',), 'info')
LAPACK_ZGEEV = _lapack(
    'zgeev', '(Tensor a, str jobvr) -> (Tensor w, Tensor vr, Tensor info)', _zgeev_fake,
    'geev', ('jobvr',), None, ('a',), ('vr',), 'info')
LAPACK_GELS = _lapack(
    'gels', '(Tensor a, Tensor b) -> (Tensor x, Tensor info)', _gels_fake,
    'gels', (), None, ('a', 'b'), ('x',), 'info')
LAPACK_GELSY = _lapack(
    'gelsy', '(Tensor a, Tensor b, float rcond) -> (Tensor x, Tensor rank, Tensor info)', _gelsy_fake,
    'gelsy', (), 'rcond', ('a', 'b'), ('x',), 'info')
LAPACK_GELSD = _lapack(
    'gelsd', '(Tensor a, Tensor b, float rcond) -> (Tensor x, Tensor s, Tensor rank, Tensor info)', _gelsd_fake,
    'gelsd', (), 'rcond', ('a', 'b'), ('x',), 'info')
LAPACK_GELSS = _lapack(
    'gelss', '(Tensor a, Tensor b, float rcond) -> (Tensor x, Tensor s, Tensor rank, Tensor info)', _gelsd_fake,
    'gelss', (), 'rcond', ('a', 'b'), ('x',), 'info')


# the CPU formulations, each torch.linalg operator's own result through its routine: torch 2.14.0 aten _linalg_svd
# (linalg_svd_out: an empty input's U and Vh the identity where full), _linalg_eigh, linalg_eig/linalg_eigvals
# (linalg_eig_out_info) and linalg_lstsq (BatchLinearAlgebra.cpp linalg_lstsq_out_info 3366-3483); `shapes` the
# node's outputs' (shape, dtype). After each routine call, its INFO goes to torch's own LinAlgError boundary,
# aten._linalg_check_errors(info, api, is_matrix), with the api name and is_matrix rule of the call site in
# BatchLinearAlgebra.cpp (lstsq-pytorch.reference.cpp: linalg.svd 3188, linalg.eigh 2820, torch.linalg.eig 3048,
# torch.linalg.eigvals 3109, torch.linalg.lstsq 3681)
def _singular_empty(a, compute_uv, full_matrices, shapes):
    return tuple(torch.eye(shape[-1], dtype=dtype, device=a.device).expand(shape).clone()
                 if compute_uv and full_matrices and k != 1 and shape[-1] and shape[-1] == shape[-2]
                 else a.new_zeros(shape, dtype=dtype) for k, (shape, dtype) in enumerate(shapes))


def _lapack_singular(A, full_matrices=True, compute_uv=True, driver=None, shapes=()):
    if not A.numel():
        return _singular_empty(A, compute_uv, full_matrices, shapes)
    s, u, vt, info = LAPACK_GESDD(A, 'N' if not compute_uv else 'A' if full_matrices else 'S')
    torch.ops.aten._linalg_check_errors.default(info, 'linalg.svd', is_matrix=A.dim() == 2)
    if not compute_uv:
        return A.new_empty(shapes[0][0], dtype=shapes[0][1]), s, A.new_empty(shapes[2][0], dtype=shapes[2][1])
    return u, s, vt


def _lapack_hermitian(A, UPLO='L', compute_v=True, shapes=()):
    if not A.numel():
        return tuple(A.new_empty(shape, dtype=dtype) for shape, dtype in shapes)
    w, z, info = LAPACK_HEEVD(A, 'V' if compute_v else 'N', UPLO.upper())
    torch.ops.aten._linalg_check_errors.default(info, 'linalg.eigh', is_matrix=A.dim() == 2)
    return w, z if compute_v else A.new_empty(shapes[1][0], dtype=shapes[1][1])


# torch 2.14.0 linalg_eig (A) -> (eigenvalues, eigenvectors) and linalg_eigvals / _linalg_eigvals (A) ->
# eigenvalues from a GEEV routine's results (`zgeev`, `dgeev`: LAPACK_ZGEEV and LAPACK_DGEEV on the CPU, _geev_routine
# else): INFO to _linalg_check_errors after the results (BatchLinearAlgebra.cpp linalg_eig_out_info), a real input's
# eigenvalues WR + i WI and its vectors linalg_eig_make_complex_eigenvectors' (Reference-LAPACK dgeev.f: a complex
# pair, WI(j) > 0 and WI(j+1) < 0, has v(j) = VR(:,j) + i VR(:,j+1) and v(j+1) = VR(:,j) - i VR(:,j+1); a real
# eigenvalue's vector VR(:,j) itself)
def _eigen(self, vectors, zgeev, dgeev):
    a = self
    complex_dtype = a.dtype if a.is_complex() else corresponding_complex_dtype(a.dtype)
    if not a.numel():
        values = a.new_empty(a.shape[:-1], dtype=complex_dtype)
        return (values, _column_major(a, a.shape, complex_dtype)) if vectors else values
    api = 'torch.linalg.eig' if vectors else 'torch.linalg.eigvals'
    if a.is_complex():
        w, vr, info = zgeev(a, 'V' if vectors else 'N')
        A._linalg_check_errors.default(info, api, is_matrix=a.dim() == 2)
        return (w, vr) if vectors else w
    wr, wi, vr, info = dgeev(a, 'V' if vectors else 'N')
    A._linalg_check_errors.default(info, api, is_matrix=a.dim() == 2)
    values = torch.complex(wr, wi)
    if not vectors:
        return values
    zero = torch.zeros_like(vr[..., :, :1])
    before, after = torch.cat([zero, vr[..., :, :-1]], -1), torch.cat([vr[..., :, 1:], zero], -1)
    first, second = (wi > 0)[..., None, :], (wi < 0)[..., None, :]
    real = torch.where(second, before, vr)
    imaginary = torch.where(first, after, torch.where(second, -vr, 0))
    return values, torch.complex(real, imaginary)


def _lapack_eigen(self, vectors=True):
    return _eigen(self, vectors, LAPACK_ZGEEV, LAPACK_DGEEV)


# BatchLinearAlgebra.cpp linalg_lstsq_out_info: the routine over the broadcast batch (the right-hand sides' rows past m
# zero), rank and singular_values those of the input's own batch (its first member along each broadcast dimension);
# residuals where m > n and the driver is not gelsy, the squared magnitudes of B's rows n on, summed (a shape the data
# decides, gelsd and gelss with m > n > 0 batched, is LSTSQ_FIXED's, which has none); the solution B's first n rows,
# zero where m is 0. `shapes` (solution, residuals or None for LSTSQ_FIXED, rank, singular_values)
def _lapack_least_squares(self, b, rcond=None, driver=None, shapes=()):
    a, (m, n) = self, self.shape[-2:]
    selected = _lstsq_driver(a, driver)
    vector = _linalg_vector_rhs(a, b)
    rhs = b.unsqueeze(-1) if vector else b
    batch, own = torch.broadcast_shapes(a.shape[:-2], rhs.shape[:-2]), a.shape[:-2]
    (solution, _), produced, (rank, _), (singular, real) = shapes
    residual = produced and produced[0]
    if not math.prod(batch):
        return (a.new_zeros(solution), None if residual is None else a.new_empty(residual, dtype=real),
                a.new_zeros(rank, dtype=torch.int64), a.new_zeros(singular, dtype=real))
    rcond = torch.finfo(real).eps * max(m, n) if rcond is None else rcond
    a, rhs = a.expand(*batch, m, n), rhs.expand(*batch, m, rhs.shape[-1])
    ranks = values = None
    if selected == 'gels':
        x, info = LAPACK_GELS(a, rhs)
    elif selected == 'gelsy':
        x, ranks, info = LAPACK_GELSY(a, rhs, rcond)
    else:
        x, values, ranks, info = (LAPACK_GELSD if selected == 'gelsd' else LAPACK_GELSS)(a, rhs, rcond)
    A._linalg_check_errors.default(info, 'torch.linalg.lstsq', is_matrix=info.numel() <= 1)
    first = (0,) * (len(batch) - len(own)) + tuple(slice(0, 1) if size == 1 else slice(None) for size in own)
    rank_made = a.new_zeros(rank, dtype=torch.int64) if ranks is None else ranks[first].to(torch.int64)
    singular_made = a.new_zeros(singular, dtype=real) if values is None else values[first]
    if residual is not None and math.prod(residual):
        raw = x[..., n:, :]
        residual = (raw * raw.conj()).real.sum(-2) if raw.is_complex() else (raw * raw).sum(-2)
    elif residual is not None:
        residual = a.new_empty(residual, dtype=real)
    made = x[..., :n, :]
    if m == 0:
        made = torch.zeros_like(made)
    return made.squeeze(-1) if vector else made, residual, rank_made, singular_made


# design/torch-source-lowering-change-request.md items 5 and 6; PyTorch v2.14.0 aten/src/ATen/native/LinearAlgebra.cpp,
# get_atol_rtol and linalg_pinv (each overload: atol and rtol tensors or floats, or rcond the relative tolerance with atol
# zero): an empty input's result the empty (n x m); a tolerance a tensor as given, a float or the default one of the
# selected device's tolerance dtype (`tolerance_dtype`: float32 on MPS, float64 else); the default rtol
# eps max(m, n), zero where an atol > 0 is given; tol = max(atol, rtol max |S|) per matrix (each tolerance's last
# dimension added); the reciprocal of each value above tol, zero else, in the input's dtype; Hermitian through
# linalg_eigh (|eigenvalues| the singular values, U diag(S+) U^H), else linalg_svd (Vh^H diag(S+) U^H). The one pinv
# formulation, expanded before the eigh outline (compiler/mesh_compiler/decompose.py outline)
def _pseudoinverse(self, atol=None, rtol=None, hermitian=False, rcond=None, tolerance_dtype=torch.float64):
    a, (m, n) = self, self.shape[-2:]
    if not a.numel():
        return a.new_zeros((*a.shape[:-2], n, m))
    if rcond is not None:
        atol, rtol = 0.0, rcond

    def held(value):
        return value if isinstance(value, torch.Tensor) else torch.full((), value, dtype=tolerance_dtype,
                                                                         device=a.device)
    given, atol = atol is not None, held(0.0 if atol is None else atol)
    if rtol is None:
        rtol = held(torch.finfo(a.real.dtype).eps * max(m, n))
        rtol = torch.where(atol > 0, held(0.0), rtol) if given else rtol
    rtol = held(rtol)
    if hermitian:
        values, vectors = torch.linalg.eigh(a)
        magnitude = values.abs()
        tolerance = torch.maximum(atol.unsqueeze(-1), rtol.unsqueeze(-1) * magnitude.amax(-1, keepdim=True))
        inverse = torch.where(magnitude > tolerance, values.reciprocal(), 0).to(a.dtype)
        return torch.matmul(vectors * inverse.unsqueeze(-2), vectors.mH)
    u, values, vh = torch.linalg.svd(a, full_matrices=False)
    tolerance = torch.maximum(atol.unsqueeze(-1), rtol.unsqueeze(-1) * values.narrow(-1, 0, 1))
    inverse = torch.where(values > tolerance, values.reciprocal(), 0).to(a.dtype)
    return torch.matmul(vh.mH * inverse.unsqueeze(-2), u.mH)


# design/torch-source-lowering-change-request.md item 6; JAX jax/_src/scipy/linalg.py, expm, _calc_P_Q, _solve_P_Q,
# _squaring, _pade3, _pade5, _pade7, _pade9 and _pade13
PADE = {3: (120., 60., 12., 1.), 5: (30240., 15120., 3360., 420., 30., 1.),
        7: (17297280., 8648640., 1995840., 277200., 25200., 1512., 56., 1.),
        9: (17643225600., 8821612800., 2075673600., 302702400., 30270240., 2162160., 110880., 3960., 90., 1.),
        13: (64764752532480000., 32382376266240000., 7771770303897600., 1187353796428800., 129060195264000.,
             10559470521600., 670442572800., 33522128640., 1323241920., 40840800., 960960., 16380., 182., 1.)}
SQUARING = {torch.float64: (5.371920351148152, (1.495585217958292e-002, 2.539398330063230e-001,
                                                9.504178996162932e-001, 2.097847961257068e+000), (3, 5, 7, 9, 13)),
            torch.float32: (3.925724783138660, (4.258730016922831e-001, 1.880152677804762e+000), (3, 5, 7))}


# torch 2.14.0 torch/linalg/__init__.py matrix_exp ("Supports input of bfloat16, float, double, cfloat and cdouble")
# and torch/_meta_registrations.py linalg_matrix_exp (checkFloatingOrComplex): the composition below runs in the input's
# computation dtype (torch/_prims_common get_computation_dtype, aten OpMathType: bfloat16 and float16 in float32,
# complex32 in complex64, the others as given), and the result is cast back to the input's dtype before the assertion
# takes it as its dep_token
#
# JAX jax/_src/scipy/linalg.py _calc_P_Q: idx = jnp.digitize(A_L1, conds), U, V = lax.switch(idx, [_pade3, ...], A);
# expm: R = lax.cond(n_squarings > max_squarings, _nan, _compute, (A, P, Q)), and an input of ndim > 2 jnp.vectorize'd,
# where jax/_src/lax/control_flow/conditionals.py _cond_batching_rule evaluates every branch and selects by the batched
# index. One branch formulation (_pade, the solve-and-squaring and NaN branches) serves both: a single matrix through
# torch/_higher_order_ops/switch.py switch_op (one branch runs), a batch through _branch_selected
def _matrix_exponential(a, max_squarings):
    if not a.numel():
        return torch.empty_like(a)
    given, a = a.dtype, a.to(get_computation_dtype(a.dtype))
    # Rejection filters are not part of even temporary integration validation.
    # SQUARING's lookup is the formulation's own requirement: a computation kind without JAX's table (float8, which
    # get_computation_dtype keeps) raises at it
    maxnorm, conds, degrees = SQUARING[a.real.dtype]
    norm = a.abs().sum(-2).amax(-1)
    count = (norm / maxnorm).log2().floor().clamp(min=0)
    scaled = a / count.exp2()[..., None, None]
    identity = torch.eye(a.shape[-1], dtype=a.dtype, device=a.device)
    batched = a.dim() > 2

    index = _digitized(norm, conds)
    if batched:
        powers = _powers(degrees, scaled, identity)
        u, v = _branch_selected(index, [_pade(degree, scaled, powers) for degree in degrees])
    else:
        u, v = switch_op(index, tuple(functools.partial(_pade_branch, degree) for degree in degrees),
                         [scaled, identity])
    over = count > max_squarings
    squares = functools.partial(_solved_squares, max_squarings=max_squarings)
    operands = [u + v, v - u, count]
    result, = _branch_selected(over, [squares(*operands), _overflowed(*operands)]) if batched else \
        switch_op(over.to(torch.int32), (squares, _overflowed), operands)
    # JAX jax/_src/scipy/linalg.py expm, `_nan` where n_squarings > max_squarings; then the operator's runtime error,
    # torch 2.14.0 torch/_export/passes/functionalize_side_effectful_ops_pass.py: _assert_async.msg as
    # _functional_assert_async.msg, its `dep_token` the result, so the error follows the result's write, and the
    # operator's result the assertion's (torch/_meta_registrations.py functional_assert_async_meta: `return dep_token`)
    return A._functional_assert_async.msg(over.logical_not(), 'linalg.matrix_exp', result.to(given))


# jnp.digitize(A_L1, conds) (right=False, searchsorted side 'right'): the count of cutoffs not above A_L1, each
# logical_not(norm < cutoff), so an equal norm takes the higher degree and a NaN norm (no cutoff compares) the highest;
# int32, as lax.switch converts its index (np.int32)
def _digitized(norm, conds):
    index = None
    for cutoff in conds:
        above = (norm < cutoff).logical_not().to(torch.int32)
        index = above if index is None else index + above
    return index


# _cond_batching_rule's batched index: every branch's results `made` (in branch order), each result component
# made[index]'s (select_n); a boolean index the two-branch cond's
def _branch_selected(index, made):
    chosen = list(made[0])
    for k, outs in enumerate(made[1:], 1):
        at = (index == k)[..., None, None]
        chosen = [torch.where(at, out, held) for out, held in zip(outs, chosen)]
    return tuple(chosen)


def _combined(terms, made=None):
    for coefficient, matrix in terms:
        made = coefficient * matrix if made is None else made + coefficient * matrix
    return made


# the powers of the scaled A the Pade `degrees` read: A^2 .. A^(d-1) (A^2, A^4 and A^6 for 13), each A^k = A^(k-2) A^2;
# one selected degree's alone inside its branch, the union of a batch's degrees once
def _powers(degrees, scaled, identity):
    powers = {0: identity, 2: torch.matmul(scaled, scaled)}
    for k in range(4, max(6 if degree == 13 else degree - 1 for degree in degrees) + 1, 2):
        powers[k] = torch.matmul(powers[k - 2], powers[2])
    return powers


# JAX _pade3, _pade5, _pade7, _pade9 and _pade13 of the scaled A over its `powers`: (U, V)
def _pade(degree, scaled, powers):
    b, parts = PADE[degree], []
    for parity in (1, 0):
        low = [(b[k + parity], powers[k]) for k in range(6 if degree == 13 else degree - 1, -1, -2)]
        if degree == 13:
            high = _combined([(b[k + 6 + parity], powers[k]) for k in (6, 4, 2)])
            parts.append(_combined(low, torch.matmul(powers[6], high)))
        else:
            parts.append(_combined(low))
    return torch.matmul(scaled, parts[0]), parts[1]


# one switch branch: that degree's own powers, then its (U, V)
def _pade_branch(degree, scaled, identity):
    return _pade(degree, scaled, _powers((degree,), scaled, identity))


# expm's _compute: _solve_P_Q (jnp.linalg.solve(Q, P)) and _squaring
def _solved_squares(p, q, count, max_squarings):
    return _squaring(torch.linalg.solve_ex(q, p).result.contiguous(), count, max_squarings),


# JAX jax/_src/scipy/linalg.py _squaring(R, n_squarings, max_squarings):
#   _scan_f(c, i): lax.cond(i < n_squarings, _squaring_precise, _identity, c)  (_squaring_precise: x @ x)
#   res, _ = lax.scan(_scan_f, R, jnp.arange(max_squarings, dtype=n_squarings.dtype))
# The scan's xs, arange(max_squarings), are compile-time, so its max_squarings steps are emitted at that static bound
# and no loop continuation is carried (the bounded predication of 1989a938, which removed the host repeat waits).
# Step i is torch/_higher_order_ops/cond.py cond_op on i < n_squarings, so only the selected branch's work runs. A
# batch (expm's jnp.vectorize) takes jax/_src/lax/control_flow/conditionals.py _cond_batching_rule's batched
# predicate, both branches and the select. JAX's `_identity` branch is `_squaring_kept`, a copy, since a HOP branch
# returns no alias of its operand
def _squaring(r, count, max_squarings):
    for i in range(max_squarings):
        if r.dim() > 2:
            r = torch.where((count > i)[..., None, None], torch.matmul(r, r), r)
        else:
            r, = cond_op(count > i, _squaring_precise, _squaring_kept, (r,))
    return r


def _squaring_precise(x):
    return torch.matmul(x, x),


def _squaring_kept(x):
    return x.clone(),


# expm's _nan, jnp.full_like(A, nan): in _compute's contiguous result layout, so both branches yield one common
# result metadata
def _overflowed(p, q, count):
    return q.new_full(q.shape, math.nan),


# JAX jax/_src/tpu/linalg/qdwh.py _qdwh: the static coefficient schedule from eps (l the lower singular value bound,
# CHOLESKY_CUTOFF 100), and the convergence tolerance cbrt(10 eps / 2)
def _qdwh_coefficients(eps, max_iterations):
    l, tol_l = eps, 10.0 * eps / 2.0
    qr_coefs, chol_coefs = [], []
    while l + tol_l < 1 and len(qr_coefs) + len(chol_coefs) < max_iterations:
        l2 = l * l
        dd = (4 * (1 / l2 - 1) / l2) ** (1 / 3)
        sqd = (1.0 + dd) ** (1 / 2)
        a = sqd + (2 - dd + 2 * (2 - l2) / (l2 * sqd)) ** (1 / 2)
        b = (a - 1) ** 2 / 4
        c = a + b - 1
        l = l * (a + b * l2) / (1 + c * l2)
        e = b / c
        if c > 100:
            qr_coefs.append(((a - e) / c ** (1 / 2), c ** (1 / 2), e))
        else:
            chol_coefs.append((a - e, c, e))
    return tol_l ** (1 / 3), qr_coefs, chol_coefs


# design/torch-source-lowering-change-request.md item 6; JAX jax/_src/tpu/linalg/qdwh.py, qdwh, _qdwh, _use_qr and
# _use_cholesky. With qdwh's dynamic_shape the operand is zero outside its leading block (_mask), and every step keeps
# it so: the QR step's columns past the block are the stacked identity's own unit columns, the Cholesky step's padded
# corner the identity. _qdwh_eigh_pop's blocks are such operands.
#
# _qdwh's cond_fun and body_fun: from k = len(qr_coefs) + len(chol_coefs), while is_not_converged and
# k < max_iterations, Halley's method, get_chol_params(3, 1, 3) = (3 - 1/3, 3, 1/3), each step tested. The bound
# max_iterations is compile-time, so the loop is emitted as its static copies k .. max_iterations - 1 (a bounded
# continuation, with no host repeat wait). A single matrix's copy is
# torch/_higher_order_ops/cond.py cond_op on is_converged, so a copy after convergence runs no step. A batch takes
# jax/_src/lax/control_flow/loops.py _while_loop_batching_rule: every copy steps, and a member that has converged keeps
# its iterate
def _polar(x, max_iterations=10):
    batch, real = x.shape[:-2], x.real.dtype
    m, n = x.shape[-2:]
    eps = float(torch.finfo(real).eps)
    magnitude = x.abs()
    one_norm, inf_norm = magnitude.sum(-2).amax(-1), magnitude.sum(-1).amax(-1)
    alpha_inverse = torch.where(one_norm == 0, 1, one_norm.rsqrt() * inf_norm.rsqrt())
    current = x * alpha_inverse[..., None, None]
    identity = torch.eye(n, dtype=x.dtype, device=x.device)
    tol_norm, qr_coefs, chol_coefs = _qdwh_coefficients(eps, max_iterations)
    for a_minus_e_by_sqrt_c, sqrt_c, e in qr_coefs:
        q = torch.linalg.qr(torch.cat([sqrt_c * current, identity.expand(*batch, n, n)], -2)).Q
        current = e * current + a_minus_e_by_sqrt_c * torch.matmul(q[..., :m, :], q[..., m:, :].mH)
    # without Cholesky coefficients `iterate` returns is_not_converged True; with them, the last step's test
    converged = torch.zeros(batch, dtype=torch.bool, device=x.device)
    for a_minus_e, c, e in chol_coefs:
        step = _cholesky_step(current, a_minus_e, c, e)
        converged, current = _converged(step, current, tol_norm), step
    current = current.contiguous()
    for _ in range(len(qr_coefs) + len(chol_coefs), max_iterations):
        if batch:
            step = _cholesky_step(current, 3 - 1 / 3, 3, 1 / 3)
            passed = _converged(step, current, tol_norm)
            current = torch.where(converged[..., None, None], current, step)
            converged = converged | passed
        else:
            current, converged = cond_op(converged, _halley_kept, functools.partial(_halley_continued, tol_norm),
                                         (current, converged))
    u = 1.5 * current - 0.5 * torch.matmul(current, torch.matmul(current.mH, current))
    product = torch.matmul(u.mH, x)
    return u, (product + product.mH) / 2, converged


# _use_cholesky's step: e u + (a - e) (L^-H L^-1 u^H)^H, L the lower Cholesky factor of c u^H u + I
def _cholesky_step(u, a_minus_e, c, e):
    uh = u.mH
    identity = torch.eye(u.shape[-1], dtype=u.dtype, device=u.device)
    y = torch.linalg.cholesky_ex(c * torch.matmul(uh, u) + identity).L
    z = torch.linalg.solve_triangular(y, uh, upper=False)
    w = torch.linalg.solve_triangular(y.mH, z, upper=True)
    return e * u + a_minus_e * w.mH


# _qdwh's test of a step: converged unless norm(u - u_prev) > tol_norm
def _converged(step, u, tol_norm):
    distance = (step - u).abs()
    return ((distance * distance).sum((-2, -1)).sqrt() > tol_norm).logical_not()


def _halley_continued(tol_norm, current, converged):
    step = _cholesky_step(current, 3 - 1 / 3, 3, 1 / 3).contiguous()
    return step, _converged(step, current, tol_norm)


def _halley_kept(current, converged):
    return current.clone(), converged.clone()


# design/torch-source-lowering-change-request.md item 6; JAX jax/_src/tpu/linalg/eigh.py _eigh_work, split_spectrum
# and _projector_subspace, and jax/_src/tpu/linalg/qdwh.py _qdwh, _use_qr and _use_cholesky; Y. Nakatsukasa and N. J.
# Higham, SIAM J. Sci. Comput. 35(3), 2013, sections 5.1 to 5.4: QDWH-eigh with termination_size 1, its recursion
# JAX's explicit agenda. Per batch member the state is `blocks` (the ragged stack of Hermitian blocks, each
# left-aligned at its row offset), `vectors` (the isometries, initially the identity), `agenda` ((n + 1) x 2 int32:
# row 0 the empty entry (0, 0), rows 1..top the pushed (offset, size)), `top` and the input's Frobenius norm. The
# state before the first step (the UPLO triangle reflected; `site` the eigh's source draw site, carried to every step
# as a one-element int32 tensor) and the result after the last (the first column's
# eigenvalues ascending, their vectors' columns in that order): the prefix and suffix formulations
# compiler/mesh_compiler/decompose.py outline traces
def _qdwh_eigh_state(a, upper=False, site=0):
    n, batch = a.shape[-1], a.shape[:-2]
    keep = torch.ones(n, n, dtype=torch.bool, device=a.device)
    mask = keep.triu() if upper else keep.tril()
    if a.is_complex():
        # JAX jax/_src/tpu/linalg/eigh.py _eigh_qdwh_impl: the real part reflected through UPLO's triangle, the
        # imaginary part kept strictly inside it and reflected negated (the diagonal's imaginary part zero)
        inside = keep.triu(1) if upper else keep.tril(-1)
        imaginary = torch.where(inside, a.imag, 0)
        h = torch.complex(torch.where(mask, a.real, a.real.mT), torch.where(mask, imaginary, -imaginary.mT))
    else:
        h = torch.where(mask, a, a.mT)
    vectors = torch.eye(n, dtype=a.dtype, device=a.device).expand(*batch, n, n).clone()
    entry = (torch.arange(n + 1, dtype=torch.int32, device=a.device)[:, None] == 1) \
        & (torch.arange(2, dtype=torch.int32, device=a.device) == 1)
    size = torch.scalar_tensor(n, dtype=torch.int32, device=a.device)
    zero = torch.scalar_tensor(0, dtype=torch.int32, device=a.device)
    agenda = torch.where(entry, size, zero).expand(*batch, n + 1, 2).clone()
    top = torch.ones(*batch, 1, dtype=torch.int32, device=a.device)
    magnitude = h.abs()
    return (h, vectors, agenda, top, (magnitude * magnitude).sum((-2, -1)).sqrt()[..., None],
            torch.full((1,), site, dtype=torch.int32, device=a.device))


# eigh's sort_eigenvalues (jnp.argsort, stable, NaN last; `empty` the vectors' shape without compute_v), after the
# outlined 2n - 1 steps (each split's children of positive size, pub13 section 5.2: n leaves, n - 1 splits)
def _qdwh_eigh_result(blocks, vectors, empty=None):
    values = blocks[..., :, 0]
    values, order = torch.sort(values.real if values.is_complex() else values, stable=True, dim=-1)
    if empty is not None:
        return values, vectors.new_empty(empty)
    return values, vectors.gather(-1, order[..., None, :].expand(vectors.shape))


# the body's Generator words (operations.Generator `state`: seed low, seed high, execution, 0) as read by a draw at the
# source draw site `site` (one-element int32, compiler/mesh_compiler/export.py capture's `site`). This is the explicit
# operand through which a method body's own Philox draws read the program's generator. It has only its fake (four int32
# words on site's device) and no eager kernel. Its value is to be Generator.state's root, never a fresh allocation
# or Inductor's RandomSeeds, with Generator.site(site) recording the draw and Generator.advanced()'s Advance operand
# the execution word's advance after the draws (design/compilation-composition-change-request.md items 6 and 7). That
# state consumer is unfinished: nothing calls Generator.site or Generator.advanced, lower_metal's
# ConstructedMachine.generated stays (), and no once-per-original-call advance is established
torch.library.define('mesh::generator', '(Tensor site) -> Tensor')


@torch.library.register_fake('mesh::generator')
def _generator_words(site):
    return site.new_empty((4,), dtype=torch.int32)


GENERATOR = torch.ops.mesh.generator.default


# One step of JAX _eigh_work's loop_body, the outlined step's body (compiler/mesh_compiler/decompose.py outline traces
# it, once per step shape): pop (offset, b), row 0's (0, 0) once the agenda is empty. Then either the popped block's
# eigenvalues go into its first column, where it is terminal (nearly diagonal: off-diagonal norm <= 5 eps norm; tiny:
# norm < eps H0_norm; or b <= 1, termination_size 1), or the block is split (_qdwh_split). The block is JAX's
# dynamic-shape operand, _slice(blocks, (offset, 0), (b, b), (N, N)): zero outside its leading b x b (_mask), with each
# extent the member's own runtime value. lax.cond(nearly_diagonal | tiny, nearly_diagonal_case, default_case): a single
# matrix's cond_op runs one branch; a batch takes _cond_batching_rule, both branches and the select per member
def _qdwh_eigh_pop(blocks, vectors, agenda, top, norm, site, max_iterations=10):
    n, eps = blocks.shape[-1], float(torch.finfo(blocks.real.dtype).eps)
    words = GENERATOR(site)
    height = top[..., 0]
    entry = agenda.gather(-2, height.long()[..., None, None].expand(*height.shape, 1, 2))[..., 0, :]
    offset, size = entry[..., 0], entry[..., 1]
    below = torch.where(height < 1, 0, height - 1)
    h = _block_taken(blocks, offset, size)
    magnitude = _squared(h)
    squares = magnitude.sum((-2, -1))
    off_diagonal = torch.where(torch.eye(n, dtype=torch.bool, device=h.device), 0, magnitude).sum((-2, -1))
    frobenius = squares.sqrt()
    stop = (off_diagonal.sqrt() <= 5 * eps * frobenius) | (frobenius < eps * norm[..., 0]) | (size <= 1)
    e, d = torch.where(stop, 0, size), torch.where(stop, size, 0)
    operands = (blocks, vectors, agenda, h, squares, offset, e, d, below, words, site)
    split = functools.partial(_qdwh_split, max_iterations)
    if blocks.dim() == 2:
        return tuple(cond_op(e > 0, split, _qdwh_terminal, operands))
    return _selected(e > 0, split(*operands), _qdwh_terminal(*operands))


# each result where `predicate` holds (per member, broadcast over the result's own dimensions), else `held`'s
def _selected(predicate, chosen, held):
    return tuple(torch.where(predicate[(...,) + (None,) * (a.dim() - predicate.dim())], a, b)
                 for a, b in zip(chosen, held))


def _squared(x):
    return x.real * x.real + x.imag * x.imag if x.is_complex() else x * x


# JAX eigh.py _mask: the leading `rows` (x `columns`) of each member's n (x n), the extents runtime int tensors
def _within(extent, n):
    return torch.arange(n, device=extent.device) < extent[..., None]


def _square_within(rows, n, columns=None):
    return _within(rows, n)[..., :, None] & _within(rows if columns is None else columns, n)[..., None, :]


def _unit(like):
    return torch.eye(like.shape[-1], dtype=like.dtype, device=like.device)


# JAX eigh.py _slice(blocks, (offset, 0), (b, b), (N, N)): x[row + i, j] for i, j < extent, zero else
def _block_taken(x, row, extent):
    n = x.shape[-1]
    at = (row[..., None] + torch.arange(n, device=x.device)).clamp(max=n - 1)
    return torch.where(_square_within(extent, n), x.gather(-2, at[..., :, None].expand(x.shape)), 0)


# JAX eigh.py _slice(eigenvectors, (0, offset), (n, b), (N, B)): x[:, column + j] for j < extent, zero else
def _columns_taken(x, column, extent):
    n = x.shape[-1]
    at = (column[..., None] + torch.arange(n, device=x.device)).clamp(max=n - 1)
    return torch.where(_within(extent, n)[..., None, :], x.gather(-1, at[..., None, :].expand(x.shape)), 0)


# JAX eigh.py _update_slice: target[place + i, j] = source[at + i, at + j] for i, j < width, the rest kept
def _block_put(target, source, place, at, width):
    n = target.shape[-1]
    index = torch.arange(n, device=target.device)
    i = index - place[..., None]
    rows = (at[..., None] + i).clamp(0, n - 1)
    columns = (at[..., None] + index).clamp(max=n - 1)
    moved = source.gather(-2, rows[..., :, None].expand(source.shape)).gather(-1, columns[..., None, :].expand(
        source.shape))
    held = ((i >= 0) & (i < width[..., None]))[..., :, None] & _within(width, n)[..., None, :]
    return torch.where(held, moved, target)


# JAX eigh.py _update_slice: target[:, place + j] = source[:, at + j] for j < width, the rest kept
def _columns_put(target, source, place, at, width):
    n = target.shape[-1]
    j = torch.arange(n, device=target.device) - place[..., None]
    moved = source.gather(-1, (at[..., None] + j).clamp(0, n - 1)[..., None, :].expand(source.shape))
    return torch.where(((j >= 0) & (j < width[..., None]))[..., None, :], moved, target)


# nearly_diagonal_case: _update_slice(blocks, diag(H)[:, None], (offset, 0), (b, 1)); the agenda's top the entry below
# the popped one
def _qdwh_terminal(blocks, vectors, agenda, h, squares, offset, e, d, below, words, site):
    n = blocks.shape[-1]
    at = torch.arange(n, device=blocks.device) - offset[..., None]
    values = h.diagonal(dim1=-2, dim2=-1).gather(-1, at.clamp(0, n - 1))
    first = torch.where((at >= 0) & (at < d[..., None]), values, blocks[..., :, 0])
    return blocks.select_scatter(first, -1, 0), vectors.clone(), agenda.clone(), below[..., None].clone()


# default_case: split_spectrum at the diagonal's nanmedian (the leading b entries sorted, NaN last, the median of the
# non-NaN count's middle pair). U is the polar factor of H - sigma I, with the side dimensions of pub13 section 5.4.
# Where neither side is positive, section 5.1 case (ii)'s remedy reruns steps 1-4 at the diagonal's mean, which lies
# strictly between the extreme eigenvalues of a block that is not a multiple of I (section 5.2's nearly diagonal test
# has ended those): a single matrix's cond_op, a batch's select. Then Algorithm 3's subspace iteration on the projector
# of the side split (_subspace_procedure) and, while it is unaccepted, both cited remedies (_qdwh_remedied, a
# while_loop with no bound, as the source's). H_minus and H_plus (Q^H H Q's diagonal blocks) and V0 V_minus and
# V0 V_plus are written at their offsets, and the children, each of positive size, pushed
def _qdwh_split(max_iterations, blocks, vectors, agenda, h, squares, offset, e, d, below, words, site):
    n = h.shape[-1]
    diagonal = torch.where(_within(e, n), h.diagonal(dim1=-2, dim2=-1).real, math.nan)
    ordered = torch.sort(diagonal, dim=-1, stable=True).values
    count = diagonal.isnan().logical_not().sum(-1)
    last = (count - 1).clamp(min=0)
    high = torch.minimum(count // 2, last)
    sigma = (ordered.gather(-1, (last // 2)[..., None]) + ordered.gather(-1, high[..., None]))[..., 0] * 0.5
    u, total, trace = _qdwh_attempt(max_iterations, h, sigma, e)
    _, _, plus_valid, minus_valid = _dimensions(e, total, trace)
    failed = (plus_valid | minus_valid).logical_not()
    attempts, mean = (h, e, u, total, trace), functools.partial(_qdwh_mean_attempt, max_iterations)
    u, total, trace = cond_op(failed, mean, _qdwh_attempt_kept, attempts) if h.dim() == 2 else \
        _selected(failed, mean(*attempts), attempts[2:])
    state = _partition(e, offset, total, trace)
    projector = _projected(u, state[-1], e)
    q, rotated, error, unaccepted = _subspace_procedure(_initial_subspace(projector, e, state[2]), projector, h, e,
                                                        state[2], state[3], squares)
    carried = (u, projector, *state, torch.zeros_like(e), q, rotated, error, unaccepted)
    made = while_loop_op(_remedy_pending, functools.partial(_qdwh_remedied, max_iterations), carried,
                         (h, e, offset, squares, words, site))
    minus, plus, _, _, at_minus, at_plus, offset_plus, _ = made[2:10]
    q, rotated = made[11:13]
    turned = torch.matmul(_columns_taken(vectors, offset, e), q)
    for width, source, place in ((minus, at_minus, offset), (plus, at_plus, offset_plus)):
        blocks = _block_put(blocks, rotated, place, source, width)
        vectors = _columns_put(vectors, turned, place, source, width)
    return (blocks, vectors, *_pushed(agenda, below, offset, minus, offset_plus, plus))


# QDWH-eig steps 1-4 at `shift` over the leading extent block: U, the polar factor of H - shift I (split_spectrum's
# qdwh.qdwh(H_shift, is_hermitian=True, dynamic_shape=(n, n)), its operand masked), with ||U||_F^2 and trace U's real
# part
def _qdwh_attempt(max_iterations, h, shift, extent):
    shifted = torch.where(_square_within(extent, h.shape[-1]), h - shift[..., None, None] * _unit(h), 0)
    u = _polar(shifted, max_iterations)[0].contiguous()
    return (u, *_unitary(u, extent))


def _qdwh_mean_attempt(max_iterations, h, e, u, total, trace):
    diagonal = h.diagonal(dim1=-2, dim2=-1).real
    mean = torch.where(_within(e, h.shape[-1]), diagonal, 0).sum(-1) / e.to(diagonal.dtype)
    return _qdwh_attempt(max_iterations, h, mean, e)


def _qdwh_attempt_kept(h, e, u, total, trace):
    return u.clone(), total.clone(), trace.clone()


def _unitary(u, extent):
    n = u.shape[-1]
    total = torch.where(_square_within(extent, n), _squared(u), 0).sum((-2, -1))
    return total, torch.where(_within(extent, n), u.diagonal(dim1=-2, dim2=-1).real, 0).sum(-1)


# pub13 section 5.4: at an exactly singular shift U is the partial isometry, and C+ = (U + I) / 2 and C- = (I - U) / 2
# have eigenvalues 1, 0.5 and 0. The subspace for C's 1 and 0.5 is the invariant subspace of the nonnegative (C+) or
# nonpositive (C-) eigenvalues of A - sigma I. With S = ||U||_F^2 and T = trace U over the b x b block, the trace and
# ||.||_F^2 of C+- give those dimensions exactly (3 trace - 2 ||C||_F^2): b + (T - S) / 2 and b - (T + S) / 2, rounded.
# A side that gives both children positive dimension (section 5.2) splits
def _dimensions(b, total, trace):
    size = b.to(total.dtype)
    nonnegative = torch.round(size + 0.5 * (trace - total)).to(torch.int32)
    nonpositive = torch.round(size - 0.5 * (trace + total)).to(torch.int32)
    return (nonnegative, nonpositive, (nonnegative > 0) & (nonnegative < b), (nonpositive > 0) & (nonpositive < b))


# the side split by subspace iteration: the valid one, the smaller where both are (JAX's rank_plus < rank_minus), its
# dimension `rank`. Returns V_minus's column count among Q's (`minus`), V_plus's (`plus`), rank, the rest, their
# columns' first indices in Q (at_minus, at_plus), `offset_plus` and `swap` (the side C+)
def _partition(b, off, total, trace):
    nonnegative, nonpositive, plus_valid, minus_valid = _dimensions(b, total, trace)
    swap = plus_valid & (minus_valid.logical_not() | (nonnegative < nonpositive))
    smaller = torch.where(swap, nonnegative, nonpositive)
    lower = torch.where(swap, b - nonnegative, nonpositive)
    zero = torch.zeros_like(smaller)
    return (lower, b - lower, smaller, b - smaller, torch.where(swap, smaller, zero), torch.where(swap, zero, smaller),
            off + lower, swap)


# C, the projector of the side split, over the leading block: (U + I) / 2 where `swap`, else (I - U) / 2
# (split_spectrum's P_plus and P_minus, I = _mask(eye, (n, n)))
def _projected(u, swap, e):
    unit = _unit(u)
    return torch.where(_square_within(e, u.shape[-1]), 0.5 * torch.where(swap[..., None, None], u + unit, unit - u), 0)


# _projector_subspace's initial guess: the `rank` columns of C of largest norm (each row's norm, as C is Hermitian;
# the negated norms argsorted, stable, with those past the block NaN so they sort last), X = C[:, sort_idxs] masked
# to (e, rank)
def _initial_subspace(projector, e, rank):
    n = projector.shape[-1]
    norms = torch.where(_square_within(e, n), _squared(projector), 0).sum(-1).sqrt().neg()
    columns = torch.sort(torch.where(_within(e, n), norms, math.nan), dim=-1, stable=True).indices
    return torch.where(_square_within(e, n, rank), projector.gather(-1, columns[..., None, :].expand(projector.shape)),
                       0)


# pub13 section 5.1, Algorithm 3, accepting ||E||_F <= 10 u ||H||_F (u = eps / 2, the unit roundoff). A pass is
# _subspace_pass. The procedure is a pass, then X := C V1 and a pass again while E is above tolerance and strictly
# below the previous pass's (otherwise it has stagnated: "further subspace iteration does not further reduce
# ||E||_F"), as a while_loop (a batch's members selected where their predicate held, _while_loop_batching_rule).
# Returns the last pass's Q, Q^H H Q and E, and `unaccepted`, E still above tolerance
def _subspace_procedure(x, projector, h, e, rank, rest, squares):
    tolerance = 10 * float(torch.finfo(squares.dtype).eps) / 2
    q, rotated, error = _subspace_pass(x, h, rank, rest)
    carried = (q, rotated, error, error.clone(), _above(error, squares, tolerance))
    q, rotated, error, _, _ = while_loop_op(_iterating, functools.partial(_subspace_repeated, tolerance), carried,
                                            (projector, h, e, rank, rest, squares))
    return q, rotated, error, _above(error, squares, tolerance)


def _above(error, squares, tolerance):
    return error.sqrt() > tolerance * squares.sqrt()


def _iterating(q, rotated, error, previous, again, projector, h, e, rank, rest, squares):
    return again.any()


def _subspace_repeated(tolerance, q, rotated, error, previous, again, projector, h, e, rank, rest, squares):
    x = torch.where(_square_within(e, h.shape[-1], rank), torch.matmul(projector, q), 0)
    made_q, made_rotated, made = _subspace_pass(x, h, rank, rest)
    passed = _above(made, squares, tolerance) & (made < previous)
    return _selected(again, (made_q, made_rotated, made, made, passed), (q, rotated, error, previous, again))


# _projector_subspace's body_f_after_matmul over the padded operand with pub13 Algorithm 3's complete QR of the
# leading e x rank block (Reference-LAPACK dgeqr2.f and dorg2r.f: _geqr2, and _householder_applied over the identity;
# the zero columns past `rank` give identity reflectors, so Q is the e x e factor's direct sum with the identity),
# then Q^H H Q and E = V2^H H V1, rotated[rank:rank + rest, :rank], with ||E||_F^2 the error
def _subspace_pass(x, h, rank, rest):
    n = x.shape[-1]
    packed, tau, _ = _geqr2(x)
    q = _householder_applied(packed, tau, _unit(x).expand(x.shape)).contiguous()
    rotated = torch.matmul(q.mH, torch.matmul(h, q)).contiguous()
    index = torch.arange(n, device=x.device)
    rows = (index >= rank[..., None]) & (index < (rank + rest)[..., None])
    error = torch.where(rows[..., :, None] & _within(rank, n)[..., None, :], _squared(rotated), 0).sum((-2, -1))
    return q, rotated, error


def _remedy_pending(*carried_and_given):
    return carried_and_given[14].any()


# while unaccepted, the split applies both cited remedies of pub13 section 5.1: (ii) the polar factor of U again
# (the QDWH owner, its max_iterations), with its sides' dimensions, partition and C rebuilt from it; and (i)
# X := C W1, W1 the first `rank` columns of a Haar unitary W (the complete QR of a standard normal b x rank matrix,
# _haar_drawn, each column signed by its R diagonal: Mezzadri 5.12, R's diagonal zgeqr2's real beta, so its phase is
# its sign), with the procedure run again. `stream` counts the restarts, the draws' fourth counter word. A batch's
# members are selected where they were unaccepted
def _qdwh_remedied(max_iterations, u, projector, minus, plus, rank, rest, at_minus, at_plus, offset_plus, swap,
                   stream, q, rotated, error, unaccepted, h, e, offset, squares, words, site):
    n = h.shape[-1]
    refined = _polar(torch.where(_square_within(e, n), u, 0), max_iterations)[0].contiguous()
    state = _partition(e, offset, *_unitary(refined, e))
    remade = _projected(refined, state[-1], e)
    restart = stream + 1
    drawn = torch.where(_square_within(e, n, state[2]), _haar_drawn(words, site, restart, h), 0)
    packed, tau, _ = _geqr2(drawn)
    rotation = _householder_applied(packed, tau, _unit(drawn).expand(drawn.shape))
    signed = torch.where((packed.diagonal(dim1=-2, dim2=-1).real < 0)[..., None, :], -rotation, rotation)
    x = torch.where(_square_within(e, n, state[2]), torch.matmul(remade, signed), 0)
    made = (refined, remade, *state, restart, *_subspace_procedure(x, remade, h, e, state[2], state[3], squares))
    held = (u, projector, minus, plus, rank, rest, at_minus, at_plus, offset_plus, swap, stream, q, rotated, error,
            unaccepted)
    return tuple(value.contiguous() for value in _selected(unaccepted, made, held))


# J. K. Salmon, M. A. Moraes, R. O. Dror and D. E. Shaw, SC 2011, §3.1 and §4.2: Philox4x32-10 (PHILOX's multipliers
# and Weyl key increments, ten rounds) over uint32 words held in int64, each product formed from its 16-bit halves as
# the 32-bit word arithmetic does; the counter (c0, c1, c2, c3) under the key (k0, k1)
def _philox_words(counter, key):
    c0, c1, c2, c3 = counter
    k0, k1 = key
    for step in range(10):
        hi0, lo0 = _mulhilo_words(c0, PHILOX[0])
        hi1, lo1 = _mulhilo_words(c2, PHILOX[1])
        c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
        if step < 9:
            k0, k1 = (k0 + PHILOX[2]) & 0xFFFFFFFF, (k1 + PHILOX[3]) & 0xFFFFFFFF
    return c0, c1, c2, c3


def _mulhilo_words(a, m):
    low, high = m & 0xFFFF, m >> 16
    al, ah = a & 0xFFFF, a >> 16
    t0, t1, t2, t3 = al * low, ah * low, al * high, ah * high
    middle = (t0 >> 16) + (t1 & 0xFFFF) + (t2 & 0xFFFF)
    return t3 + (t1 >> 16) + (t2 >> 16) + (middle >> 16), (t0 + ((t1 + t2) << 16)) & 0xFFFFFFFF


# G. E. P. Box and M. E. Muller, Ann. Math. Statist. 29(2), 1958: a standard normal draw from two Philox words, in
# float32 (each word's high 24 bits a uniform; the first offset by 2^-24, so its logarithm is finite)
def _normal_words(first, second):
    radius = ((first >> 8).to(torch.float32) * 2.0 ** -24 + 2.0 ** -24).log().mul(-2).sqrt()
    return radius * ((second >> 8).to(torch.float32) * 2.0 ** -24).mul(2 * math.pi).cos()


# F. Mezzadri, arXiv:math-ph/0609050, section 5 (5.12, haar_measure): the restart's standard normal matrix (`like`'s
# shape and dtype). Each element is its own Philox draw: the counter is its row-major index in the member's n x n
# (a complex one's real pairs, n x n x 2, each part of variance 1/2) offset by member * n * n * parts, then the
# execution word (Generator.state word 2, the per-call counter), the eigh's source draw site and the restart count;
# the key is the seed's two words. Box-Muller in float32, then the matrix's dtype
def _haar_drawn(words, site, stream, like):
    batch, n = like.shape[:-2], like.shape[-1]
    parts = 2 if like.is_complex() else 1
    inner = (n, n, 2) if parts == 2 else (n, n)
    spread = (...,) + (None,) * len(inner)
    linear = torch.arange(n * n * parts, dtype=torch.int64, device=like.device).view(inner)
    member = torch.arange(math.prod(batch), dtype=torch.int64, device=like.device).view(batch)
    element = (member[spread] * (n * n * parts) + linear) & 0xFFFFFFFF
    drawn = _philox_words((element, _unsigned(words[2]), _unsigned(site[0]), _unsigned(stream)[spread]),
                          (_unsigned(words[0]), _unsigned(words[1])))
    normal = _normal_words(drawn[0], drawn[1])
    if parts == 1:
        return normal.to(like.dtype)
    normal = normal * math.sqrt(0.5)
    return torch.complex(normal[..., 0], normal[..., 1]).to(like.dtype)


# the children, each of positive size, pushed above `below`: (offset, minus) at below + 1, then (offset_plus, plus)
# above it, so the plus child is popped first
def _pushed(agenda, below, offset, minus, offset_plus, plus):
    first, second = minus > 0, plus > 0
    at = below + 1
    rows = torch.arange(agenda.shape[-2], device=agenda.device)
    for row, held, made in ((at, first, (offset, minus)), (at + first.to(at.dtype), second, (offset_plus, plus))):
        chosen = (rows == row[..., None]) & held[..., None]
        agenda = torch.where(chosen[..., None], torch.stack(made, -1)[..., None, :], agenda)
    return agenda, (below + first.to(below.dtype) + second.to(below.dtype))[..., None]


# JAX jax/_src/tpu/linalg/svd.py svd and _svd_tall_and_square_input; pub13
# sections 4 and 5.5; torch 2.14.0 aten _linalg_svd (A, full_matrices, compute_uv, driver) -> (U, S, Vh), `shapes`
# its outputs' (shape, dtype), an empty input's U and Vh the identity where full (linalg_svd_out)
def _singular_qdwh(A, full_matrices=False, compute_uv=True, driver=None, shapes=(), max_iterations=10):
    a, (m, n) = A, A.shape[-2:]
    if not a.numel():
        return _singular_empty(a, compute_uv, full_matrices, shapes)
    flip = m < n
    if flip:
        a, (m, n) = a.mH, (n, m)
    q = null = None
    if full_matrices and m > n:
        complete, reduced = torch.linalg.qr(a, mode='complete')
        q, null, a = complete[..., :n], complete[..., n:], reduced[..., :n, :]
    elif m > 1.15 * n:
        q, a = torch.linalg.qr(a)
    u_p, h, _ = _polar(a, max_iterations)
    if not compute_uv:
        s_out = torch.sort(torch.linalg.eigvalsh(h).clamp(min=0), stable=True, dim=-1, descending=True).values
        return A.new_empty(shapes[0][0], dtype=shapes[0][1]), s_out, A.new_empty(shapes[2][0], dtype=shapes[2][1])
    s, v = torch.linalg.eigh(h)
    s_out, order = torch.sort(s.clamp(min=0), stable=True, dim=-1, descending=True)
    v_out = v.gather(-1, order[..., None, :].expand(v.shape))
    u_out = _rank_corrected(torch.matmul(u_p, v_out), s_out, n)
    if q is not None:
        u_out = torch.matmul(q, u_out)
    if null is not None:
        u_out = torch.cat([u_out, null], -1)
    finite = torch.isfinite(a).all(-1).all(-1)
    u_out = torch.where(finite[..., None, None], u_out, math.nan)
    s_out = torch.where(finite[..., None], s_out, math.nan)
    v_out = torch.where(finite[..., None, None], v_out, math.nan)
    return (v_out, s_out, u_out.mH) if flip else (u_out, s_out, v_out.mH)


# JAX jax/_src/tpu/linalg/svd.py 104-118 (pub13 section 5.5), the correction where u from qdwh is not unitary:
#   correct_rank_deficiency(u_out): u_out, r = qr(u_out, full_matrices=False); u_out @ diag(where(diag(r) >= 0, 1, -1))
#   do_correction = s_out[-1] <= a.shape[1] * eps * s_out[0]
#   u_out, _ = lax.while_loop(lambda args: args[1], lambda args: (correct_rank_deficiency(args[0]), False),
#                             (u_out, do_correction))
# as torch/_higher_order_ops/while_loop.py while_loop_op over the carry (u_out, do_correction). A batch is
# jax/_src/lax/control_flow/loops.py _while_loop_batching_rule's batched predicate: the loop runs while any member's
# predicate holds and each member's body result is selected where its own predicate held. The carry is contiguous
# (while_loop_fake_tensor_mode's check_meta_consistency compares carried and body strides); diag(r) >= 0 of a
# complex r is its real part's (zgeqr2's beta, R's diagonal, is real)
def _rank_corrected(u_out, s_out, n):
    eps = float(torch.finfo(u_out.real.dtype).eps)
    do_correction = s_out[..., -1] <= n * eps * s_out[..., 0]
    u_out, _ = while_loop_op(_correction_pending, _rank_correction, (u_out.contiguous(), do_correction), ())
    return u_out


def _correction_pending(u_out, do_correction):
    return do_correction.any()


def _rank_correction(u_out, do_correction):
    q, r = torch.linalg.qr(u_out)
    diagonal = r.diagonal(dim1=-2, dim2=-1)
    sign = torch.where((diagonal.real if diagonal.is_complex() else diagonal) >= 0, 1, -1).to(q.dtype)
    corrected = torch.where(do_correction[..., None, None], q * sign[..., None, :], u_out).contiguous()
    return corrected, torch.zeros_like(do_correction)


# Reference-LAPACK SRC/dgels.f, zgels.f, dgelsy.f and zgelsy.f on the placed processor (a CPU lstsq is LAPACK's,
# _lapack_least_squares): the range scaling (_range_scaled) over the broadcast batch, then gels' QR solve (dgeqrf,
# dormqr, dtrtrs: a zero diagonal of the triangular factor INFO, its first index, as torch.linalg.lstsq's
# _linalg_check_errors; a zero matrix's solution and residual rows zero) or gelsy's column-pivoted QR, incremental rank
# and complete orthogonal factorization (_gelsy_rank, _gelsy_solution), then the scaling's undo. Each factorization
# step is functional ATen (slice_scatter, gather, matmul, where), unrolled over the static min(m, n) columns.
# `shapes` (solution, residuals or None for LSTSQ_FIXED, rank, singular_values) as _lapack_least_squares'
def _least_squares_qr(self, b, rcond=None, driver=None, shapes=()):
    a, (m, n) = self, self.shape[-2:]
    selected = _lstsq_driver(a, driver)
    vector = _linalg_vector_rhs(a, b)
    rhs = b.unsqueeze(-1) if vector else b
    batch, own, nrhs = torch.broadcast_shapes(a.shape[:-2], rhs.shape[:-2]), a.shape[:-2], rhs.shape[-1]
    (solution, _), produced, (rank, _), (singular, real) = shapes
    residual = produced and produced[0]
    singular_made = a.new_zeros(singular, dtype=real)
    if not min(m, n) or not nrhs:
        if residual is not None:
            residual = a.new_zeros(residual, dtype=real)
        return a.new_zeros(solution), residual, a.new_zeros(rank, dtype=torch.int64), singular_made
    rcond = torch.finfo(real).eps * max(m, n) if rcond is None else rcond
    a, rhs = a.expand(*batch, m, n), rhs.expand(*batch, m, nrhs)
    scaled, right, a_scale, b_scale = _range_scaled(a, rhs)
    raw, rank_made = None, a.new_zeros(rank, dtype=torch.int64)
    if selected == 'gels':
        zero = (_component_maximum(a) == 0)[..., None, None]
        k = min(m, n)
        packed, tau, _ = _geqr2(scaled.mH if m < n else scaled)
        triangle = packed[..., :k, :]
        diagonal = torch.where(zero[..., 0], 1, triangle.diagonal(dim1=-2, dim2=-1))
        triangle = triangle.diagonal_scatter(diagonal, dim1=-2, dim2=-1)
        singular_at = diagonal == 0
        info = torch.where(singular_at.any(-1), singular_at.to(torch.int32).argmax(-1) + 1, 0).to(torch.int32)
        A._linalg_check_errors.default(info.contiguous(), 'torch.linalg.lstsq', is_matrix=info.numel() <= 1)
        if m >= n:
            transformed = _householder_applied(packed, tau, right, adjoint=True)
            x = torch.linalg.solve_triangular(triangle, transformed[..., :n, :], upper=True)
            raw = torch.where(zero, 0, transformed[..., n:, :]) / b_scale[..., None, None]
        else:
            middle = torch.linalg.solve_triangular(triangle.mH, right, upper=False)
            x = _householder_applied(packed, tau, torch.cat([middle, middle.new_zeros((*batch, n - m, nrhs))], -2))
        x = torch.where(zero, 0, x)
    else:
        packed, tau, pivots = _geqr2(scaled, pivoted=True)
        ranks = _gelsy_rank(packed, rcond)
        x = _gelsy_solution(packed, tau, pivots, ranks, right)
        first = (0,) * (len(batch) - len(own)) + tuple(slice(0, 1) if size == 1 else slice(None) for size in own)
        rank_made = ranks[first]
    x = _range_unscaled(x, a_scale, b_scale)
    if residual is not None and math.prod(residual):
        residual = (raw * raw.conj()).real.sum(-2) if raw.is_complex() else (raw * raw).sum(-2)
    elif residual is not None:
        residual = a.new_zeros(residual, dtype=real)
    return x.squeeze(-1) if vector else x, residual, rank_made, singular_made


# Reference-LAPACK SRC/dlarfg.f and zlarfg.f over the last dimension of `column`: (v, tau, beta), v[0] = 1, with
# H = I - tau v v^H taking column to beta e_1; the column first scaled by its largest modulus, a zero tail (and a
# real alpha) the identity reflector (tau 0, beta alpha's real part)
def _reflector(column):
    magnitudes = column.abs()
    scale = magnitudes.amax(-1)
    safe = torch.where(scale == 0, 1, scale)
    normalized, magnitudes = column / safe[..., None], magnitudes / safe[..., None]
    norm = (magnitudes * magnitudes).sum(-1).sqrt()
    alpha = normalized[..., 0]
    signed = torch.where(alpha.real < 0, norm, -norm)
    inactive = magnitudes[..., 1:].amax(-1) <= 0 if column.shape[-1] > 1 else torch.ones_like(norm, dtype=torch.bool)
    if column.is_complex():
        inactive = inactive & (alpha.imag == 0)
    v = normalized / torch.where(inactive, 1, alpha - signed)[..., None]
    v = torch.cat([torch.ones_like(v[..., :1]), v[..., 1:]], -1)
    tau = torch.where(inactive, 0, (signed - alpha) / torch.where(inactive, 1, signed))
    beta = torch.where(inactive, column[..., 0].real, signed * scale)
    return v, tau, beta


# Reference-LAPACK SRC/dlarf.f: H block (left) or block H (right), H = I - tau v v^H (tau's conjugate: H^H)
def _dlarf(block, v, tau, left=True, adjoint=False):
    v, scalar = v[..., :, None], (tau.conj() if adjoint else tau)[..., None, None]
    if left:
        return block - torch.matmul(v, torch.matmul(v.mH, block) * scalar)
    return block - torch.matmul(torch.matmul(block, v) * scalar, v.mH)


# `block` written into x at (row, column), functionally (aten slice_scatter)
def _put(x, block, row, column):
    rows = x[..., row:row + block.shape[-2], :]
    return x.slice_scatter(rows.slice_scatter(block, -1, column, column + block.shape[-1]), -2, row,
                           row + block.shape[-2])


# rows (dim -2) or columns (dim -1) j and p (a batch index tensor) of x exchanged (dlaswp, dswap)
def _exchanged(x, p, j, dim):
    index = torch.arange(x.shape[dim], device=x.device)
    p = p[..., None]
    index = torch.where(index == j, p, torch.where(index == p, j, index))
    return x.gather(dim, (index[..., :, None] if dim == -2 else index[..., None, :]).expand(x.shape))


# Reference-LAPACK SRC/dgeqr2.f and zgeqr2.f (dgeqrf's unblocked form): the packed factor (R on and above the
# diagonal, each reflector's tail below it), tau and, `pivoted`, dgeqp3's column pivots (each step the remaining
# column of largest 2-norm, recomputed over the remaining rows, moved to the front; 0-based, one per step)
def _geqr2(a, pivoted=False):
    packed, (m, n) = a, a.shape[-2:]
    taus, pivots = [], []
    for j in range(min(m, n)):
        if pivoted:
            p = _column_lengths(packed[..., j:, j:]).argmax(-1) + j
            packed = _exchanged(packed, p, j, -1)
            pivots.append(p)
        v, tau, beta = _reflector(packed[..., j:, j])
        if j + 1 < n:
            packed = _put(packed, _dlarf(packed[..., j:, j + 1:], v, tau, adjoint=True), j, j + 1)
        packed = _put(packed, torch.cat([beta.to(packed.dtype)[..., None], v[..., 1:]], -1)[..., :, None], j, j)
        taus.append(tau)
    return packed, torch.stack(taus, -1), pivots


# each column's 2-norm, scaled by its largest modulus
def _column_lengths(x):
    magnitude = x.abs()
    maximum = magnitude.amax(-2)
    scaled = magnitude / torch.where(maximum == 0, 1, maximum)[..., None, :]
    return (scaled * scaled).sum(-2).sqrt() * maximum


# Reference-LAPACK SRC/dorm2r.f and zunm2r.f, side 'L': Q b (Q^H b `adjoint`), Q = H_1 .. H_k of the packed reflectors
def _householder_applied(packed, tau, b, adjoint=False):
    for j in range(tau.shape[-1]) if adjoint else reversed(range(tau.shape[-1])):
        v = torch.cat([torch.ones_like(packed[..., j:j + 1, j]), packed[..., j + 1:, j]], -1)
        b = b.slice_scatter(_dlarf(b[..., j:, :], v, tau[..., j], adjoint=adjoint), -2, j)
    return b


# dgelsy.f 10-20 with Reference-LAPACK SRC/dlaic1.f and zlaic1.f: the largest and smallest singular value estimates
# of the leading triangle (each its approximate singular vector), the rank the leading columns whose estimate ratio
# SMAXPR RCOND <= SMINPR holds throughout; |R(1,1)| = 0 rank 0
def _gelsy_rank(packed, rcond):
    k = min(packed.shape[-2:])
    start = packed[..., 0, 0].abs()
    unit = torch.zeros_like(packed[..., 0, :k])
    unit = torch.cat([torch.ones_like(unit[..., :1]), unit[..., 1:]], -1)
    estimates, vectors = [start, start], [unit, unit]
    active = start != 0
    rank = active.to(torch.int64)
    for j in range(1, k):
        for largest in (0, 1):
            estimates[largest], vectors[largest] = _condition(packed, j, estimates[largest], vectors[largest],
                                                              largest)
        active = active & (estimates[1] * rcond <= estimates[0])
        rank = rank + active.to(torch.int64)
    return rank


# dlaic1's update by the 2 x 2 Hermitian eigenproblem of [sest^2 + |alpha|^2, alpha conj(gamma); ., |gamma|^2]
# (alpha = x^H w over the column above the diagonal, gamma the diagonal), scaled by max(|alpha|, |gamma|, sest),
# through its Jacobi rotation (_rotation): the estimate (`largest`: the larger value's root; else sest |gamma| over
# it) and the vector's update (its leading j entries by the rotation's coefficient, entry j the other)
def _condition(packed, j, estimate, vector, largest):
    alpha = torch.matmul(vector[..., None, :j].conj(), packed[..., :j, j:j + 1])[..., 0, 0]
    gamma = packed[..., j, j]
    absa, absg = alpha.abs(), gamma.abs()
    scale = torch.maximum(torch.maximum(absa, absg), estimate)
    scale = torch.where(scale == 0, 1, scale)
    absa, absg, sest = absa / scale, absg / scale, estimate / scale
    app, aqq = sest * sest + absa * absa, absg * absg
    apq = alpha / scale * (gamma / scale).conj()
    first = app >= aqq
    cosine, sine, tangent, magnitude = _rotation(app, aqq, apq, apq != 0)
    delta = tangent * magnitude
    upper = torch.where(first, app - delta, aqq + delta).sqrt()
    if largest:
        estimate = scale * upper
    else:
        estimate = estimate * (absg / torch.where(upper == 0, 1, upper))
        first = first.logical_not()
    coefficient = torch.where(first, cosine, sine)
    last = torch.where(first, -sine.conj(), cosine)
    vector = torch.cat([vector[..., :j] * coefficient[..., None], last.to(vector.dtype)[..., None],
                        vector[..., j + 1:]], -1)
    return estimate, vector


# G. H. Golub and C. F. Van Loan, Matrix Computations, 3rd ed., section 8.4, Algorithms 8.4.1 and 8.4.3; Z. Drmac and
# K. Veselic, LAPACK Working Note 196, sections 2 and 3: the symmetric Schur rotation of [app, apq; conj(apq), aqq]
# (cosine, sine, tangent and |apq|), the tangent zero where `active` does not hold
def _rotation(app, aqq, apq, active):
    magnitude = apq.abs()
    delta = aqq * 0.5 - app * 0.5
    scale = torch.maximum(delta.abs(), magnitude)
    zero = scale == 0
    scale = torch.where(zero, 1, scale)
    d, g = delta / scale, magnitude / scale
    tangent = g / torch.where(zero, 1, (d * d + g * g).sqrt() + d.abs())
    tangent = torch.where(active, torch.where(delta < 0, -tangent, tangent), 0)
    cosine = (tangent * tangent + 1).rsqrt()
    sine = apq / torch.where(magnitude == 0, 1, magnitude) * tangent * cosine
    return cosine, sine, tangent, magnitude


# Reference-LAPACK SRC/dtzrzf.f, dlatrz.f, ztzrzf.f, zlatrz.f and dgelsy.f 30-70: the leading rank rows of R reduced
# by RZ reflectors (row j's entry j and its entries past the rank, conjugated, each reflector applied from the right
# to the rows above), the triangular solve of Q^H b's leading rank rows, the RZ reflectors applied back and the pivot
# exchanges undone in reverse
def _gelsy_solution(packed, tau, pivots, rank, b):
    n, k = packed.shape[-1], min(packed.shape[-2:])
    matrix, indices = packed[..., :k, :].triu(), torch.arange(n, device=packed.device)
    reflectors = []
    for j in reversed(range(k)):
        active = j < rank
        tail = indices[j:] >= rank[..., None]
        tail = torch.cat([torch.ones_like(tail[..., :1]), tail[..., 1:]], -1)
        v, scalar, beta = _reflector(torch.where(tail, matrix[..., j, j:], 0).conj())
        scalar = torch.where(active, scalar, 0)
        if j:
            matrix = _put(matrix, _dlarf(matrix[..., :j, j:], v, scalar, left=False), 0, j)
        matrix = _put(matrix, torch.where(active, beta.to(matrix.dtype), matrix[..., j, j])[..., None, None], j, j)
        reflectors.append((j, v, scalar))
    keep = indices[:k] < rank[..., None]
    triangle = torch.where(keep[..., :, None] & keep[..., None, :], matrix[..., :k], 0)
    triangle = triangle.diagonal_scatter(torch.where(keep, triangle.diagonal(dim1=-2, dim2=-1), 1), dim1=-2, dim2=-1)
    transformed = _householder_applied(packed, tau, b, adjoint=True)
    rhs = torch.where(keep[..., None], transformed[..., :k, :], 0)
    solved = torch.linalg.solve_triangular(triangle, rhs, upper=True)
    work = torch.cat([solved, solved.new_zeros((*solved.shape[:-2], n - k, solved.shape[-1]))], -2)
    for j, v, scalar in reversed(reflectors):
        work = work.slice_scatter(_dlarf(work[..., j:, :], v, scalar), -2, j)
    for j in reversed(range(k)):
        work = _exchanged(work, pivots[j], j, -2)
    return work


# Reference-LAPACK SRC/dgelss.f 385-435 and 737-768, dgelsd.f 379-431 and 621-645: the range scaling, ANRM = DLANGE('M')
# and BNRM brought into [SMLNUM, BIGNUM] (SMLNUM = DLAMCH('S') / DLAMCH('P'), BIGNUM its reciprocal; a zero norm
# unscaled), each matrix's component magnitudes (a complex value's real and imaginary parts); an empty B unscaled
def _range_scaled(a, b):
    limits = torch.finfo(a.real.dtype)
    small = limits.tiny / limits.eps
    a_scale = _range_scale(_component_maximum(a), small)
    b_scale = _range_scale(_component_maximum(b), small) if min(b.shape[-2:]) else b.real.new_ones(b.shape[:-2])
    return a * a_scale[..., None, None], b * b_scale[..., None, None], a_scale, b_scale


def _component_maximum(x):
    return torch.view_as_real(x).abs().amax((-3, -2, -1)) if x.is_complex() else x.abs().amax((-2, -1))


def _range_scale(maximum, small):
    safe = torch.where(maximum == 0, 1, maximum)
    return torch.minimum(torch.maximum(safe, safe.new_full((), small)), safe.new_full((), 1 / small)) / safe


# the scaling's undo (DLASCL(CTO, CFROM) of the solution by ANRM / BNRM): one multiply by numerator / denominator where
# both lie on one side of 1, else by the numerator, then the division by the denominator, so no ratio overflows
def _range_unscaled(x, numerator, denominator):
    same = (numerator >= 1) == (denominator >= 1)
    multiply = torch.where(same, numerator / denominator, numerator)
    divide = torch.where(same, 1, denominator)
    return x * multiply[..., None, None] / divide[..., None, None]


# Reference-LAPACK SRC/dgelss.f 385-435, 509-521 and 737-768, dgelsd.f 379-431 and 621-645, dlalsd.f 235-276 and
# 329-340; PyTorch v2.14.0 BatchLinearAlgebra.cpp linalg_lstsq_out_info 3366-3483: the SVD drivers through the QDWH
# SVD (_singular_qdwh), `shapes` the node's outputs' (shape, dtype)
def _least_squares_svd(self, b, rcond=None, driver=None, shapes=()):
    a, (m, n) = self, self.shape[-2:]
    rhs = b.unsqueeze(-1) if _linalg_vector_rhs(a, b) else b
    (solution, _), produced, (rank, _), (singular, real) = shapes
    residual = produced and produced[0]
    if not min(m, n):
        raw = (rhs.abs() * rhs.abs()).sum(-2)
        return (a.new_zeros(solution), None if residual is None else raw.expand(residual).clone()
                if math.prod(residual) else a.new_empty(residual, dtype=real), a.new_zeros(rank, dtype=torch.int64),
                a.new_zeros(singular, dtype=real))
    limits = torch.finfo(real)
    rcond = limits.eps * max(m, n) if rcond is None else rcond
    scaled, right, a_scale, b_scale = _range_scaled(a, rhs)
    u, s, vh = torch.linalg.svd(scaled, full_matrices=False)
    if driver.lower() == 'gelss':
        keep = s > (s[..., :1] * (rcond if rcond >= 0 else limits.eps)).clamp(min=limits.tiny)
    elif min(m, n) == 1:
        keep = s != 0
    else:
        keep = s > (limits.eps / 2 if rcond <= 0 or rcond >= 1 else rcond) * s.amax(-1, keepdim=True)
    inverse = torch.where(keep, 1 / torch.where(keep, s, 1), 0)
    made = _range_unscaled(torch.matmul(vh.mH, inverse.unsqueeze(-1) * torch.matmul(u.mH, right)), a_scale, b_scale)
    return (made.reshape(solution), None if residual is None else a.new_empty(residual, dtype=real), keep.sum(-1),
            s / a_scale.unsqueeze(-1))


# Reference-LAPACK SRC/zgeev.f and dgeev.f (JOBVL = 'N', JOBVR = 'V' or 'N') on a non-CPU engine as upstream ATen
# and torch HOP composition: each routine below is the cited routine's body over the batch, its DO loops static
# positions with their actual per-member extents (aten where/gather masks), its DO WHILE and GO TO loops
# torch/_higher_order_ops/while_loop.py while_loop_op under JAX's batching rule (jax/_src/lax/control_flow/loops.py
# _while_loop_batching_rule: the loop runs while any member's predicate holds, each member's step selected where its
# own does). K, L, I, J, ILO and IHI are the source's 1-based values as int64 batch tensors. DZNRM2/DNRM2 is the
# scaled norm (largest modulus, then the root of the sum of squared ratios); CABS1 |Re| + |Im|; DLADIV and ZLADIV
# dladiv.f's scaled division (_divided); a real factor of a complex entry multiplies each component (ZDSCAL, Fortran's
# mixed real and complex product); Fortran SIGN(A, B) takes B's sign bit, a -0 B negative

# 1-based positions along the last dimension
def _counted(x):
    return torch.arange(1, x.shape[-1] + 1, device=x.device)


# a batch scalar broadcast over a vector (_vec) or a matrix (_mat); a Python number as itself
def _vec(s):
    return s if isinstance(s, (int, float)) else s[..., None]


def _mat(s):
    return s if isinstance(s, (int, float)) else s[..., None, None]


def _first_any(live, *rest):
    return live.any()


def _abs1(x):
    return x.real.abs() + x.imag.abs() if x.is_complex() else x.abs()


def _re(x):
    return x.real if x.is_complex() else x


def _fsign(a, b):
    negative = (b < 0) | ((b == 0) & (1 / b < 0))
    return torch.where(negative, -a.abs(), a.abs())


def _scaled(x, s):
    return torch.complex(x.real * s, x.imag * s) if x.is_complex() else x * s


def _over(x, s):
    return torch.complex(x.real / s, x.imag / s) if x.is_complex() else x / s


def _plus_real(x, s):
    return torch.complex(s + x.real, x.imag) if x.is_complex() else s + x


# Reference-LAPACK SRC/dladiv.f (3.7 on; zladiv.f is it on the parts): P + i Q = (A + i B) / (C + i D), the operands
# halved where their larger part is at least OV / 2 and raised by BE = 2 / EPS**2 where it is at most UN 2 / EPS (OV,
# UN, EPS = DLAMCH 'O', 'S', 'E'), then DLADIV1 (or on the swapped parts, Q negated, where |D| > |C|) with DLADIV2's
# R = 0 and B R = 0 branches, P and Q times S
def _divided(a, b, c, d):
    limits = torch.finfo(a.dtype)
    eps = limits.eps / 2
    be, small = 2 / (eps * eps), limits.tiny * 2 / eps
    ab, cd = torch.maximum(a.abs(), b.abs()), torch.maximum(c.abs(), d.abs())
    s = torch.ones_like(a)
    halved = ab >= 0.5 * limits.max
    aa, bb, s = torch.where(halved, 0.5 * a, a), torch.where(halved, 0.5 * b, b), torch.where(halved, 2 * s, s)
    halved = cd >= 0.5 * limits.max
    cc, dd, s = torch.where(halved, 0.5 * c, c), torch.where(halved, 0.5 * d, d), torch.where(halved, 0.5 * s, s)
    raised = ab <= small
    aa, bb, s = torch.where(raised, aa * be, aa), torch.where(raised, bb * be, bb), torch.where(raised, s / be, s)
    raised = cd <= small
    cc, dd, s = torch.where(raised, cc * be, cc), torch.where(raised, dd * be, dd), torch.where(raised, s * be, s)
    plain = d.abs() <= c.abs()
    p, q = _dladiv1(aa, bb, cc, dd)
    p_swapped, q_swapped = _dladiv1(bb, aa, dd, cc)
    return torch.where(plain, p, p_swapped) * s, torch.where(plain, q, -q_swapped) * s


def _dladiv1(a, b, c, d):
    r = d / c
    t = 1 / (c + d * r)
    return _dladiv2(a, b, c, d, r, t), _dladiv2(b, -a, c, d, r, t)


def _dladiv2(a, b, c, d, r, t):
    br = b * r
    return torch.where(r != 0, torch.where(br != 0, (a + br) * t, a * t + (b * t) * r), (a + d * (b / c)) * t)


def _cdiv(x, y):
    return torch.complex(*_divided(x.real, x.imag, y.real, y.imag))


# x(row, col) of each member (*batch, n, n); positions clamped into range where unused
def _entry(x, row, col):
    n = x.shape[-1]
    at = (row - 1).clamp(0, n - 1) * n + (col - 1).clamp(0, n - 1)
    return x.flatten(-2).gather(-1, at[..., None])[..., 0]


def _entry_put(x, row, col, value, when):
    k = _counted(x)
    hit = (k[:, None] == _mat(row)) & (k == _mat(col)) & _mat(when)
    return torch.where(hit, value if isinstance(value, (int, float)) else _mat(value).to(x.dtype), x)


def _at(v, i):
    return v.gather(-1, (i - 1).clamp(0, v.shape[-1] - 1)[..., None])[..., 0]


def _at_put(v, i, value, when):
    hit = (_counted(v) == _vec(i)) & _vec(when)
    return torch.where(hit, value if isinstance(value, (int, float)) else _vec(value).to(v.dtype), v)


def _row(x, i):
    index = (i - 1).clamp(0, x.shape[-2] - 1)[..., None, None].expand(*x.shape[:-2], 1, x.shape[-1])
    return x.gather(-2, index)[..., 0, :]


def _column(x, j):
    index = (j - 1).clamp(0, x.shape[-1] - 1)[..., None, None].expand(*x.shape[:-2], x.shape[-2], 1)
    return x.gather(-1, index)[..., 0]


def _span(k, low, high):
    return (k >= _vec(low)) & (k <= _vec(high))


# ZSWAP of columns a and b over the rows `rows` (a row mask), and of rows a and b over the columns `columns`
def _columns_swapped(x, a, b, rows, when):
    k, first, second = _counted(x), _column(x, a), _column(x, b)
    rows = (rows & _vec(when))[..., :, None]
    x = torch.where(rows & (k == _mat(a)), second[..., :, None], x)
    return torch.where(rows & (k == _mat(b)), first[..., :, None], x)


def _rows_swapped(x, a, b, columns, when):
    k, first, second = _counted(x), _row(x, a), _row(x, b)
    columns = (columns & _vec(when))[..., None, :]
    x = torch.where(columns & (k[:, None] == _mat(a)), second[..., None, :], x)
    return torch.where(columns & (k[:, None] == _mat(b)), first[..., None, :], x)


# DZNRM2 over the entries `within` (a mask over the last dimension)
def _norm2(x, within):
    magnitude = torch.where(within, x.abs(), 0)
    largest = magnitude.amax(-1)
    q = magnitude / _vec(torch.where(largest == 0, 1, largest))
    return (q * q).sum(-1).sqrt() * largest


# DNRM2 of a few batch scalars (xLARFG's tail)
def _listed_norm(xs):
    if len(xs) == 1:
        return xs[0].abs()
    return _norm2(torch.stack(xs, -1), torch.ones(len(xs), dtype=torch.bool, device=xs[0].device))


# IZAMAX/IDAMAX over `within`: the first position of the largest CABS1
def _izamax(x, within):
    return torch.where(within, _abs1(x), -1).argmax(-1) + 1


# Reference-LAPACK SRC/dlapy2.f (a NaN operand itself, y's where both are) and dlapy3.f
def _dlapy2(x, y):
    huge = torch.finfo(x.dtype).max
    a, b = x.abs(), y.abs()
    w, z = torch.where(a >= b, a, b), torch.where(a >= b, b, a)
    plain = (z == 0) | (w > huge)
    q = z / torch.where(plain, 1, w)
    made = torch.where(plain, w, w * (1 + q * q).sqrt())
    return torch.where(y.isnan(), y, torch.where(x.isnan(), x, made))


def _dlapy3(x, y, z):
    huge = torch.finfo(x.dtype).max
    a, b, c = x.abs(), y.abs(), z.abs()
    w = torch.where(a >= b, a, b)
    w = torch.where(w >= c, w, c)
    plain = (w == 0) | (w > huge)
    safe = torch.where(plain, 1, w)
    qa, qb, qc = a / safe, b / safe, c / safe
    return torch.where(plain, a + b + c, w * (qa * qa + qb * qb + qc * qc).sqrt())


def _broadcast(s, x):
    return s[(...,) + (None,) * (x.dim() - s.dim())]


# Reference-LAPACK SRC/zlascl.f and dlascl.f (TYPE 'G') 10 loop: each pass's MUL (SMLNUM = DLAMCH('S'), BIGNUM its
# reciprocal; CFROMC an infinity, CTOC 0 or an infinity, the SMLNUM and BIGNUM steps, else CTOC / CFROMC) applied to
# x where `when`, until DONE
def _lascl(x, cfrom, cto, when):
    tiny = torch.finfo(x.real.dtype).tiny
    carried = (when.clone(), x.contiguous(), cfrom.contiguous(), cto.contiguous())
    return while_loop_op(_first_any, functools.partial(_lascl_pass, tiny), carried, ())[1]


def _lascl_pass(smlnum, pending, x, cfromc, ctoc):
    bignum = 1 / smlnum
    cfrom1 = cfromc * smlnum
    infinite = cfrom1 == cfromc
    cto1 = ctoc / bignum
    ends = ~infinite & (cto1 == ctoc)
    down = ~infinite & ~ends & (cfrom1.abs() > ctoc.abs()) & (ctoc != 0)
    up = ~infinite & ~ends & ~down & (cto1.abs() > cfromc.abs())
    mul = torch.where(ends, ctoc, torch.where(down, smlnum, torch.where(up, bignum, ctoc / cfromc)))
    x = torch.where(_broadcast(pending, x), _scaled(x, _broadcast(mul, x)), x)
    return (pending & (down | up), x, torch.where(pending & down, cfrom1, cfromc),
            torch.where(pending & up, cto1, ctoc))


# Reference-LAPACK SRC/zgebal.f and dgebal.f (JOB = 'B', lines 225-418) where `proceed`: the 242-286 row search (one
# pass one I of DO I = L..1, its count fixed at the round's entry; a swap with L = 1 returns ILO = IHI = 1), the
# 288-315 column search (one pass one J of DO J = K..L), SCALE(K..L) = 1 and the 330-410 norm balancing (one pass one
# I of DO I = K..L, its two DO WHILE loops nested while_loops over the member's scalars). Returns the balanced matrix,
# SCALE's factors (real) and interchanges (`perm`, int64), ILO and IHI. ZGEEV ignores the 363-367 NaN return
# (INFO = -3), which ends the balancing with ILO = K and IHI = L
def _gebal(h, proceed):
    n, batch = h.shape[-1], h.shape[:-2]
    limits = torch.finfo(h.real.dtype)
    sfmin1 = limits.tiny / limits.eps
    sfmax1, sfmin2 = 1 / sfmin1, sfmin1 * 2
    sfmax2 = 1 / sfmin2
    size = torch.full(batch, n, dtype=torch.int64, device=h.device)
    one, no = torch.ones_like(size), torch.zeros(batch, dtype=torch.bool, device=h.device)
    perm = _counted(h).expand(*batch, n).contiguous()
    _, h, perm, _, l, _, done = while_loop_op(_first_any, _gebal_rows, (
        proceed.clone(), h.contiguous(), perm, size, size.clone(), no, no.clone()), ())
    _, h, perm, _, low, _ = while_loop_op(_first_any, _gebal_columns, (
        proceed & ~done, h, perm, one, one.clone(), no.clone()), (l,))
    scale = torch.ones((*batch, n), dtype=h.real.dtype, device=h.device)
    _, h, scale, _, _ = while_loop_op(_first_any, functools.partial(_gebal_scaled, (sfmin1, sfmax1, sfmin2, sfmax2)), (
        proceed & ~done, h, scale, low.clone(), no.clone()), (low, l))
    return h, scale, perm, torch.where(done, 1, low), torch.where(done, 1, l)


def _gebal_rows(live, h, perm, i, l, noconv, done):
    k = _counted(h)
    active = live & (i >= 1)
    canswap = ((k == _vec(i)) | (k > _vec(l)) | (_row(h, i) == 0)).all(-1)
    hit = active & canswap
    h = _rows_swapped(_columns_swapped(h, i, l, k <= _vec(l), hit), i, l, k >= 1, hit)
    perm = _at_put(perm, l, i, hit)
    finish = hit & (l == 1)
    noconv = noconv | hit
    l = torch.where(hit & ~finish, l - 1, l)
    i = torch.where(active, i - 1, i)
    restart = live & ~active & noconv
    return (live & ~finish & (active | restart), h, perm, torch.where(restart, l, i), l, noconv & ~restart,
            done | finish)


def _gebal_columns(live, h, perm, j, low, noconv, l):
    k = _counted(h)
    active = live & (j <= l)
    canswap = ((k == _vec(j)) | (k < _vec(low)) | (k > _vec(l)) | (_column(h, j) == 0)).all(-1)
    hit = active & canswap
    h = _rows_swapped(_columns_swapped(h, j, low, k <= _vec(l), hit), j, low, k >= _vec(low), hit)
    perm = _at_put(perm, low, j, hit)
    noconv = noconv | hit
    low = torch.where(hit, low + 1, low)
    j = torch.where(active, j + 1, j)
    restart = live & ~active & noconv
    return live & (active | restart), h, perm, torch.where(restart, low, j), low, noconv & ~restart


def _gebal_scaled(limits, live, h, scale, i, noconv, low, l):
    sfmin1, sfmax1, sfmin2, sfmax2 = limits
    k = _counted(h)
    active = live & (i <= l)
    column, row = _column(h, i), _row(h, i)
    inner = _span(k, low, l)
    c, r = _norm2(column, inner), _norm2(row, inner)
    ca = _at(column, _izamax(column, k <= _vec(l))).abs()
    ra = _at(row, _izamax(row, k >= _vec(low))).abs()
    considered = active & (c != 0) & (r != 0)
    failed = considered & (c + ca + r + ra).isnan()
    going = considered & ~failed
    s, bounds = c + r, (sfmin2, sfmax2)
    f, c, ca, r, g, ra = while_loop_op(functools.partial(_gebal_pending, _gebal_up_holds, bounds), functools.partial(
        _gebal_step, _gebal_up_holds, bounds, 2), (torch.ones_like(c), c, ca, r, r / 2, ra), (going,))
    f, c, ca, r, g, ra = while_loop_op(functools.partial(_gebal_pending, _gebal_down_holds, bounds), functools.partial(
        _gebal_step, _gebal_down_holds, bounds, 0.5), (f, c, ca, r, c / 2, ra), (going,))
    at = _at(scale, i)
    skipped = ((c + r) >= 0.95 * s) | ((f < 1) & (at < 1) & (f * at <= sfmin1)) | (
        (f > 1) & (at > 1) & (at >= sfmax1 / f))
    applied = going & ~skipped
    scale = _at_put(scale, i, at * f, applied)
    h = torch.where(_mat(applied) & (k[:, None] == _mat(i)) & (k >= _mat(low)), _scaled(h, _mat(1 / f)), h)
    h = torch.where(_mat(applied) & (k == _mat(i)) & (k[:, None] <= _mat(l)), _scaled(h, _mat(f)), h)
    noconv = noconv | applied
    i = torch.where(active, i + 1, i)
    restart = live & ~active & noconv
    return live & ~failed & (active | restart), h, scale, torch.where(restart, low, i), noconv & ~restart


# zgebal.f 377-385 and 387-395: the DO WHILE tests over the member's (F, C, CA, R, G, RA), and one step (F, C, CA by
# SCLFAC = 2, R, G, RA by its reciprocal; the second loop's G scaled with C)
def _gebal_up_holds(bounds, f, c, ca, r, g, ra, alive):
    sfmin2, sfmax2 = bounds
    return alive & (c < g) & (torch.maximum(torch.maximum(f, c), ca) < sfmax2) & (
        torch.minimum(torch.minimum(r, g), ra) > sfmin2)


def _gebal_down_holds(bounds, f, c, ca, r, g, ra, alive):
    sfmin2, sfmax2 = bounds
    return alive & (g >= r) & (torch.maximum(r, ra) < sfmax2) & (
        torch.minimum(torch.minimum(torch.minimum(f, c), g), ca) > sfmin2)


def _gebal_pending(holds, bounds, *values):
    return holds(bounds, *values).any()


def _gebal_step(holds, bounds, factor, f, c, ca, r, g, ra, alive):
    held = holds(bounds, f, c, ca, r, g, ra, alive)
    up, down = (f * 2, c * 2, ca * 2, r / 2, g / 2, ra / 2), (f / 2, c / 2, ca / 2, r * 2, g / 2, ra * 2)
    return tuple(torch.where(held, made, x) for made, x in zip(up if factor == 2 else down, (f, c, ca, r, g, ra)))


# Reference-LAPACK 3.12.1 SRC/zlarf1f.f 196-299 (dlarf1f.f 193-288 alike): C = H C (SIDE = 'L', `left`) or C H
# ('R'), H = I - TAU v v^H with v(1) = 1 (never read), on C = x(R0:R0+M-1, C0:C0+N-1), v aligned with x's rows (left)
# or columns from R0 (C0), where `when`. With TAU /= 0 LASTV from M (N) down while
# V(LASTV) = 0, past V(1), and LASTC = ILAZLC(LASTV, N, C) / ILAZLR(M, LASTV, C) (ilazlc.f, ilazlr.f, iladlc.f,
# iladlr.f: the last column or row holding a nonzero, a NaN nonzero, else 0), else LASTC = 0; the LASTC = 0 return.
# Left: LASTV = 1, ZSCAL of C(1, 1:LASTC) by ONE - TAU; else WORK = C(2:LASTV, 1:LASTC)^H v(2:LASTV) (ZGEMV), WORK(J) +
# CONJG(C(1,J)), C(1,J) - TAU CONJG(WORK(J)) (dlarf1f.f's two DAXPYs), and ZGERC's C(I,J) + v(I) (-TAU CONJG(WORK(J)))
# where WORK(J) /= 0. Right: LASTV = 1, ZSCAL of C(1:LASTC, 1); else WORK = C(1:LASTC, 2:LASTV) v(2:LASTV) (ZGEMV),
# the ZAXPYs WORK + C(:,1) and C(:,1) + (-TAU) WORK, and ZGERC's C(I,J) + WORK(I) (-TAU CONJG(v(J))) where v(J) /= 0
def _larf1f(left, m, n, v, tau, x, r0, c0, when):
    rows, cols = torch.arange(1, x.shape[-2] + 1, device=x.device), _counted(x)
    along, start, extent = (rows, r0, m) if left else (cols, c0, n)
    lastv = torch.where((along > _vec(start)) & (along < _vec(start + extent)) & (v != 0), along - _vec(start) + 1,
                        1).amax(-1)
    held = (_span(rows, r0, r0 + (lastv if left else m) - 1)[..., :, None]
            & _span(cols, c0, c0 + (n if left else lastv) - 1)[..., None, :] & (x != 0))
    if left:
        lastc = torch.where(held.any(-2), cols - _vec(c0) + 1, 0).amax(-1)
    else:
        lastc = torch.where(held.any(-1), rows - _vec(r0) + 1, 0).amax(-1)
    go = when & (tau != 0) & (lastc > 0)
    multi, t = go & (lastv > 1), _vec(tau)
    tail = _span(along, start + 1, start + lastv - 1)
    vt = torch.where(tail, v, 0)
    if left:
        reached = _span(cols, c0, c0 + lastc - 1)
        first = _row(x, r0)
        work = torch.matmul(vt[..., None, :], _cj(torch.where(tail[..., :, None], x, 0)))[..., 0, :] + _cj(first)
        lead = torch.where(_vec(multi), first - t * _cj(work), _vec(1 - tau) * first)
        x = torch.where((rows[:, None] == _mat(r0)) & (reached & _vec(go))[..., None, :], lead[..., None, :], x)
        made = x + vt[..., :, None] * (-t * _cj(work))[..., None, :]
        hit = tail[..., :, None] & (reached & (work != 0))[..., None, :]
    else:
        reached = _span(rows, r0, r0 + lastc - 1)
        first = _column(x, c0)
        work = torch.matmul(torch.where(tail[..., None, :], x, 0), vt[..., :, None])[..., 0] + first
        lead = torch.where(_vec(multi), first + -t * work, _vec(1 - tau) * first)
        x = torch.where((cols == _mat(c0)) & (reached & _vec(go))[..., :, None], lead[..., :, None], x)
        made = x + work[..., :, None] * (-t * _cj(v))[..., None, :]
        hit = reached[..., :, None] & (tail & (v != 0))[..., None, :]
    return torch.where(hit & _mat(multi), made, x)


# Reference-LAPACK 3.12.1 SRC/zlarfb.f 235-366 (dlarfb.f 232-362 alike: 'Transpose', no ZLACGV), DIRECT = 'F' and
# STOREV = 'C', on local blocks (each argument from its (1,1)): C = H C or H^H C (`left`; `adjoint` for TRANS = 'C'),
# else C H or C H^H, H = I - V T V^H, V = v's leading M (N) x K (V1 its unit lower K x K, V's diagonal and upper part
# unread, over V2), T = t's upper K x K; the M <= 0 or N <= 0 return. Left (TRANST the other of 'N' and 'C'): W =
# C1^H (ZCOPY, ZLACGV), W V1 (ZTRMM), + C2^H V2 where M > K (ZGEMM), W T^H or W T (ZTRMM by TRANST), C2 - V2 W^H
# where M > K, W V1^H, DO 30 C1 - W^H. Right: W = C1, W V1, + C2 V2 where N > K, W T or W T^H, C2 - W V2^H where
# N > K, W V1^H, DO 60 C1 - W. Each product's operands zero past their extents
def _larfb(left, adjoint, m, n, k, v, t, c):
    p, j = torch.arange(1, v.shape[-2] + 1, device=v.device), torch.arange(1, v.shape[-1] + 1, device=v.device)
    go = (m > 0) & (n > 0)
    first, rest, within = _span(p, 1, k), _span(p, k + 1, m if left else n), _within(k, v.shape[-1])
    v1 = torch.where(first[..., :, None] & within[..., None, :],
                     torch.where(p[:, None] > j, v, (p[:, None] == j).to(v.dtype)), 0)
    v2 = torch.where(rest[..., :, None] & within[..., None, :], v, 0)
    upper = torch.where(_square_within(k, t.shape[-1]) & (j[:, None] <= j), t, 0)
    if left:
        across = _within(n, c.shape[-1]) & _vec(go)
        kept = across[..., :, None] & within[..., None, :]
        c1 = torch.where(first[..., :, None] & across[..., None, :], c, 0)
        c2 = torch.where(rest[..., :, None] & across[..., None, :], c, 0)
        w = torch.where(kept, torch.matmul(c1.mH, v1), 0)
        w = torch.where(_mat(m > k) & kept, w + torch.matmul(c2.mH, v2), w)
        w = torch.where(kept, torch.matmul(w, upper if adjoint else upper.mH), 0)
        c = torch.where(_mat(m > k) & rest[..., :, None] & across[..., None, :], c - torch.matmul(v2, w.mH), c)
        w = torch.matmul(w, v1.mH)
        return torch.where(first[..., :, None] & across[..., None, :], c - w.mH, c)
    down = _within(m, c.shape[-2]) & _vec(go)
    kept = down[..., :, None] & within[..., None, :]
    c1 = torch.where(down[..., :, None] & first[..., None, :], c, 0)
    c2 = torch.where(down[..., :, None] & rest[..., None, :], c, 0)
    w = torch.where(kept, torch.matmul(c1, v1), 0)
    w = torch.where(_mat(n > k) & kept, w + torch.matmul(c2, v2), w)
    w = torch.where(kept, torch.matmul(w, upper.mH if adjoint else upper), 0)
    c = torch.where(_mat(n > k) & down[..., :, None] & rest[..., None, :], c - torch.matmul(w, v2.mH), c)
    w = torch.matmul(w, v1.mH)
    return torch.where(down[..., :, None] & first[..., None, :], c - w, c)


# Reference-LAPACK 3.12.1 SRC/zlarft.f 211-345 (dlarft.f 207-341 alike, V^T), DIRECT = 'F' and STOREV = 'C', the
# recursive routine: T (t's upper K x K) of H(1) ... H(K) = I - V T V^H, V = v's V(1:N,1:K) (unit lower, its diagonal
# and upper part unread), TAU along `tau`, where `when`: the N = 0 or K = 0 return; T(1,1) = TAU(1) where N = 1 or
# K = 1; else L = K / 2, ZLARFT(N, L) on V, ZLARFT(N - L, K - L) on V(L+1,L+1) into T(L+1,L+1), then T(1:L,L+1:K)
# (_larft_joined). The calls are the complete call tree over K's static bound, in the source's order, each position
# with the member's offset O (its V(1,1) at V(O+1,O+1), its N = N - O) and K, present where its parent recursed
def _larft(n, k, v, tau, when):
    def call(t, o, s, present, most):
        base = present & ((n - o == 1) | (s == 1))
        t = _entry_put(t, o + 1, o + 1, _at(tau, o + 1), base)
        if most > 1:
            split, half = present & ~base, s // 2
            t = call(t, o, half, split, most // 2)
            t = call(t, o + half, s - half, split, most - most // 2)
            t = _larft_joined(t, n, o, s, half, v, split)
        return t
    bound = v.shape[-1]
    return call(v.new_zeros((*v.shape[:-2], bound, bound)), torch.zeros_like(k), k, when & (n > 0) & (k > 0), bound)


# zlarft.f 314-345 (dlarft.f 310-341) for the call at offset O with its K = S and L: DO J, DO I T(J,L+I) =
# CONJG(V(L+I,J)); ZTRMM('Right', 'Lower', 'No transpose', 'Unit') by V(L+1:K,L+1:K); + V(K+1:N,1:L)^H V(K+1:N,L+1:K)
# where N > K (ZGEMM's K = N - K = 0 return); ZTRMM('Left', 'Upper', 'No transpose', 'Non-unit', NEG_ONE) by
# T(1:L,1:L); ZTRMM('Right', 'Upper', 'No transpose', 'Non-unit') by T(L+1:K,L+1:K)
def _larft_joined(t, n, o, s, half, v, when):
    bound = t.shape[-1]
    p, rows = torch.arange(1, bound + 1, device=t.device), torch.arange(1, v.shape[-2] + 1, device=v.device)
    first, second = _span(p, o + 1, o + half), _span(p, o + half + 1, o + s)
    kept, square = first[..., :, None] & second[..., None, :], second[..., :, None] & second[..., None, :]
    leading = v[..., :bound, :]
    unit = torch.where(square, torch.where(p[:, None] > p, leading, (p[:, None] == p).to(v.dtype)), 0)
    t12 = torch.where(kept, torch.matmul(torch.where(kept, leading.mH, 0), unit), 0)
    below = _span(rows, o + s + 1, n)
    v31 = torch.where(below[..., :, None] & first[..., None, :], v, 0)
    v32 = torch.where(below[..., :, None] & second[..., None, :], v, 0)
    t12 = torch.where(_mat(n - o > s) & kept, t12 + torch.matmul(v31.mH, v32), t12)
    t11 = torch.where(first[..., :, None] & first[..., None, :] & (p[:, None] <= p), t, 0)
    t12 = torch.where(kept, -torch.matmul(t11, t12), 0)
    t12 = torch.matmul(t12, torch.where(square & (p[:, None] <= p), t, 0))
    return torch.where(_mat(when) & kept, t12, t)


# Reference-LAPACK 3.12.1 SRC/zgehrd.f 249-353 (dgehrd.f alike, 'Transpose') on the leading ORDER x ORDER block of h
# with LWORK `lwork` (batch ints): 249-255 TAU(1:ILO-1) and TAU(MAX(1,IHI):N-1) zero; 259-262 the NH <= 1 return;
# 266-294 NB = MIN(NBMAX = 64, ILAENV(1, 'ZGEHRD') = 32, ilaenv.f 340-345), NBMIN = 2, and where NB > 1 and NB < NH
# NX = MAX(NB, ILAENV(3) = 128, ilaenv.f 622-627), where also NX < NH and LWORK < LWKOPT = N NB + TSIZE (TSIZE = LDT
# NBMAX = 65 64) NBMIN = MAX(2, ILAENV(2) = 2, ilaenv.f 532-537) and NB = (LWORK - TSIZE) / N where LWORK >= N NBMIN
# + TSIZE, else 1; 296-349 I = ILO where NB < NBMIN or NB >= NH, else DO 40 I = ILO, IHI - 1 - NX, NB (_gehrd_block;
# the loop present where the static extent admits its first pass, NH >= 130); 353 ZGEHD2(N, I, IHI) from that I:
# zgehd2.f 190-207 DO 10 over static I = 1..n-1 where it is from that I to IHI - 1: ZLARFG(IHI - I, A(I+1,I),
# A(MIN(I+2,N),I)) (_larfg_vector over rows I+2..IHI, A(I+1,I) its BETA), ZLARF1F('Right', IHI, IHI - I, A(I+1,I),
# TAU(I), A(1,I+1)) and ZLARF1F('Left', IHI - I, N - I, A(I+1,I), CONJG(TAU(I)), A(I+1,I+1)). 222's LWORK < MAX(1, N)
# return is unreached: zgeev.f's LWORK - N and dgeev.f's LWORK - 2 N exceed 32 N, and xLAQR3's LWORK - JW >= JW (JW
# <= NW <= NWMAX <= LWORK / 2). Returns the packed matrix and TAU(1:n-1)
def _gehrd(order, ilo, ihi, h, lwork):
    n, k = h.shape[-1], _counted(h)
    one = torch.ones_like(ilo)
    nh, nx, nbmin, tsize = ihi - ilo + 1, 128, 2, 65 * 64
    nb = torch.full_like(ilo, 32)
    short = (nb > 1) & (nb < nh) & (nx < nh) & (lwork < order * nb + tsize)
    fits = short & (lwork >= order * nbmin + tsize)
    nb = torch.where(fits, torch.div(lwork - tsize, torch.where(fits, order, 1), rounding_mode='trunc'),
                     torch.where(short, 1, nb))
    prior, start = h.new_zeros((*h.shape[:-2], max(n - 1, 0))), ilo
    if n >= 130:
        carried = ((nb >= nbmin) & (nb < nh) & (ilo <= ihi - 1 - nx), h.contiguous(), prior, ilo.clone())
        _, h, prior, start = while_loop_op(_first_any, _gehrd_block, carried, (nb, ihi, order))
    taus = []
    for i in range(1, n):
        active = (start <= i) & (i <= ihi - 1)
        column = h[..., :, i - 1]
        tail = _span(k, i + 2, ihi)
        beta, tau, made = _larfg_vector(column[..., i], torch.where(tail, column, 0))
        tau = torch.where(active, tau, 0)
        column = torch.where(_vec(active) & tail, made, torch.where(_vec(active) & (k == i + 1), _vec(beta), column))
        h = _column_put(h, i - 1, column)
        h = _larf1f(False, ihi, ihi - i, column, tau, h, one, one + i, active)
        h = _larf1f(True, ihi - i, order - i, column, _cj(tau), h, one + i, one + i, active)
        taus.append(torch.where(active, tau, prior[..., i - 1]))
    return h, torch.stack(taus, -1) if taus else prior


# zgehrd.f 307-348, DO 40's pass for I: IB = MIN(NB, IHI - I); ZLAHR2(IHI, I, IB, A(1,I), TAU(I), T, Y) (_lahr2);
# EI = A(I+IB,I+IB-1), ONE there for ZGEMM('No transpose', 'Conjugate transpose', IHI, IHI - I - IB + 1, IB, -ONE, Y,
# A(I+IB,I), ONE, A(1,I+IB)), then EI back; ZTRMM('Right', 'Lower', 'Conjugate transpose', 'Unit', I, IB - 1,
# A(I+1,I), Y); DO 30 J = 0, IB - 2 ZAXPY(I, -ONE, Y(1,J+1), A(1,I+J+1)); ZLARFB('Left', 'Conjugate transpose',
# 'Forward', 'Columnwise', IHI - I, N - I - IB + 1, IB, A(I+1,I), T, A(I+1,I+IB)) (_larfb)
def _gehrd_block(live, h, taus, i, nb, ihi, order):
    size, k = h.shape[-1], _counted(h)
    one = torch.ones_like(i)
    ib = torch.minimum(nb, ihi - i)
    h, taus, t, y = _lahr2(live, ihi, i, ib, h, taus, 32)
    bound = y.shape[-1]
    j = torch.arange(1, bound + 1, device=h.device)
    ei = _entry(h, i + ib, i + ib - 1)
    h = _entry_put(h, i + ib, i + ib - 1, 1, live)
    rows, reached = (k <= _vec(ihi))[..., :, None], _span(k, i + ib, ihi)
    factor = torch.where(rows & _within(ib, bound)[..., None, :], y, 0)
    v = torch.where(reached[..., :, None] & _within(ib, bound)[..., None, :], _region_at(h, one, i, size, bound), 0)
    h = torch.where(_mat(live) & rows & reached[..., None, :], h - torch.matmul(factor, v.mH), h)
    h = _entry_put(h, i + ib, i + ib - 1, ei, live)
    unit = torch.where(_square_within(ib - 1, bound) & (j[:, None] >= j),
                       torch.where(j[:, None] > j, _region_at(h, i + 1, i, bound, bound), 1), 0)
    top = (k <= _vec(i))[..., :, None] & _within(ib - 1, bound)[..., None, :]
    y = torch.where(_mat(live) & top, torch.matmul(torch.where(top, y, 0), unit.mH), y)
    for jj in range(bound - 1):
        hit = _mat(live & (jj <= ib - 2)) & (k <= _vec(i))[..., :, None] & (k == _mat(i + jj + 1))
        h = torch.where(hit, h - y[..., :, jj:jj + 1], h)
    made = _larfb(True, True, ihi - i, order - i - ib + 1, ib, _region_at(h, i + 1, i, size, bound), t,
                  _region_at(h, i + 1, i + ib, size, size))
    h = _region_put(h, i + 1, i + ib, made, _square_within(ihi - i, size, order - i - ib + 1) & _mat(live))
    i = torch.where(live, i + nb, i)
    return live & (i <= ihi - 1 - 128), h, taus, i


# Reference-LAPACK 3.12.1 SRC/zlahr2.f 216-321 (dlahr2.f 216-319 alike, no ZLACGV), ZLAHR2(N, K, NB, A, TAU, T, Y) on
# h with its A(1,1) at h(1,K), where `live`: the N <= 1 return; DO 10 I = 1, NB as static I up to `bound` where I <=
# NB: for I > 1, A(K+1:N,I) - Y(K+1:N,1:I-1) CONJG(A(K+I-1,1:I-1)) (ZLACGV, ZGEMV; A(K+I-1,I-1) is then ONE) and
# I - V T^H V^H on it as b: w = V1^H b1 (ZCOPY, ZTRMV; w is T(1:I-1,NB) there, the last column of T as workspace,
# here its own vector: ZGEMV with BETA = ZERO writes T(1:NB-1,NB) at I = NB before any read), + V2^H b2, T^H w, b2 -
# V2 w, V1 w, b1 - w (ZAXPY), A(K+I-1,I-1) = EI; ZLARFG(N - K - I + 1, A(K+I,I), A(MIN(K+I+1,N),I)), EI its BETA,
# A(K+I,I) = ONE; Y(K+1:N,I) = A(K+1:N,I+1:N-K+1) v (ZGEMV, BETA = ZERO), T(1:I-1,I) = V2^H v, Y(K+1:N,I) -
# Y(K+1:N,1:I-1) T(1:I-1,I), times TAU(I) (ZSCAL); T(1:I-1,I) times -TAU(I), then by the upper T(1:I-1,1:I-1)
# (ZTRMV), T(I,I) = TAU(I); after DO 10 A(K+NB,NB) = EI; Y(1:K,1:NB) = A(1:K,2:NB+1) (ZLACPY), by the unit lower
# A(K+1:K+NB,1:NB) (ZTRMM), + A(1:K,NB+2:N-K+1) A(K+NB+1:N,1:NB) where N > K + NB (ZGEMM), by the upper T (ZTRMM).
# TAU(I) is tau's K + I - 1; returns h, tau, T (bound x bound) and Y (h's rows x bound)
def _lahr2(live, n, k, nb, h, tau, bound):
    size, r = h.shape[-1], _counted(h)
    p = torch.arange(1, bound + 1, device=h.device)
    one = torch.ones_like(k)
    live = live & (n > 1)
    y = h.new_zeros((*h.shape[:-2], size, bound))
    t = h.new_zeros((*h.shape[:-2], bound, bound))
    ei, below = h.new_zeros(k.shape), _span(r, k + 1, n)
    for i in range(1, bound + 1):
        active = live & (i <= nb)
        c, rest = k + i - 1, _span(r, k + i, n)
        v2 = torch.where(rest[..., :, None], _region_at(h, one, k, size, i - 1), 0)
        if i > 1:
            q = p[:i - 1]
            v1 = torch.where(q[:, None] > q, _region_at(h, k + 1, k, i - 1, i - 1), (q[:, None] == q).to(h.dtype))
            b = _column(h, c)
            x = _cj(_vec_at(_row(h, k + i - 1), k, i - 1))
            b = torch.where(below, b - torch.matmul(y[..., :, :i - 1], x[..., :, None])[..., 0], b)
            w = torch.matmul(v1.mH, _vec_at(b, k + 1, i - 1)[..., :, None])[..., 0]
            w = w + torch.matmul(v2.mH, torch.where(rest, b, 0)[..., :, None])[..., 0]
            w = torch.matmul(t[..., :i - 1, :i - 1].triu().mH, w[..., :, None])[..., 0]
            b = torch.where(rest, b - torch.matmul(v2, w[..., :, None])[..., 0], b)
            w = torch.matmul(v1, w[..., :, None])[..., 0]
            b = _vec_put(b, k + 1, _vec_at(b, k + 1, i - 1) - w, torch.ones_like(w, dtype=torch.bool))
            h = torch.where(_mat(active) & (r == _mat(c)), b[..., :, None], h)
            h = _entry_put(h, k + i - 1, k + i - 2, ei, active)
        tail = _span(r, k + i + 1, n)
        beta, taui, x = _larfg_vector(_entry(h, k + i, c), torch.where(tail, _column(h, c), 0))
        b = torch.where(tail, x, torch.where(r == _vec(k + i), 1, _column(h, c)))
        h = torch.where(_mat(active) & (r == _mat(c)), b[..., :, None], h)
        ei = torch.where(active, beta, ei)
        tau = _at_put(tau, c, taui, active)
        v = torch.where(rest, _column(h, c), 0)
        made = torch.matmul(torch.where(below[..., :, None] & rest[..., None, :], h, 0), v[..., :, None])[..., 0]
        column = torch.zeros_like(t[..., :, 0])
        if i > 1:
            ti = torch.matmul(v2.mH, v[..., :, None])[..., 0]
            made = made - torch.matmul(y[..., :, :i - 1], ti[..., :, None])[..., 0]
            ti = torch.matmul(t[..., :i - 1, :i - 1].triu(), (ti * _vec(-taui))[..., :, None])[..., 0]
            column = torch.cat([ti, column[..., i - 1:]], -1)
        made = made * _vec(taui)
        y = _column_put(y, i - 1, torch.where(_vec(active) & below, made, y[..., :, i - 1]))
        column = torch.where(p == i, _vec(taui), column)
        t = _column_put(t, i - 1, torch.where(_vec(active), column, t[..., :, i - 1]))
    h = _entry_put(h, k + nb, k + nb - 1, ei, live)
    upper = (r <= _vec(k))[..., :, None]
    kept, far = upper & _within(nb, bound)[..., None, :], _span(r, k + nb + 1, n)
    unit = torch.where(_square_within(nb, bound), torch.where(p[:, None] > p, _region_at(h, k + 1, k, bound, bound),
                                                              (p[:, None] == p).to(h.dtype)), 0)
    top = torch.where(kept, torch.matmul(torch.where(kept, _region_at(h, one, k + 1, size, bound), 0), unit), 0)
    added = torch.matmul(torch.where(upper & far[..., None, :], h, 0),
                         torch.where(far[..., :, None] & _within(nb, bound)[..., None, :],
                                     _region_at(h, one, k, size, bound), 0))
    top = torch.where(_mat(n > k + nb) & kept, top + added, top)
    top = torch.matmul(top, torch.where(_square_within(nb, bound), t, 0).triu())
    return h, tau, t, torch.where(_mat(live) & kept, top, y)


# Reference-LAPACK 3.12.1 SRC/zunghr.f 194-233 (dorghr.f alike) on `a` (zgeev.f 396's ZLACPY('L') of the reduced
# matrix, its upper part unread and here zero) with TAU `taus` and LWORK `lwork`: DO 40 J = IHI, ILO + 1, -1
# (A(1:J-1,J) = ZERO, A(J+1:IHI,J) = A(J+1:IHI,J-1), column J - 1 read before its own pass, A(IHI+1:N,J) = ZERO),
# DO 60 and DO 80 (columns 1..ILO and IHI+1..N the identity's), then ZUNGQR(NH, NH, NH, A(ILO+1,ILO+1), TAU(ILO))
# where NH = IHI - ILO > 0 (_ungqr). 175's LWORK < MAX(1, NH) return is unreached (zgeev.f's LWORK - N, dgeev.f's
# LWORK - 2 N)
def _unghr(ilo, ihi, a, taus, lwork):
    k = _counted(a)
    rows, cols = k[:, None], k
    band = _span(k, ilo + 1, ihi)[..., None, :]
    shifted = torch.cat([a[..., :, :1], a[..., :, :-1]], -1)
    a = torch.where(band & (rows < cols), 0, a)
    a = torch.where(band & (rows > cols) & (rows <= _mat(ihi)), shifted, a)
    a = torch.where(band & (rows > _mat(ihi)), 0, a)
    a = torch.where(~band, (rows == cols).to(a.dtype), a)
    nh = ihi - ilo
    return _ungqr(nh, nh, nh, a, ilo, taus, nh > 0, lwork)


# Reference-LAPACK 3.12.1 SRC/zungqr.f 190-282 (dorgqr.f alike), ZUNGQR(M, N, K, A, TAU) on a with its A(1,1) at
# a(BASE+1,BASE+1), its TAU(I) taus' BASE + I - 1, LWORK `lwork`, where `when`: the N <= 0 return; NB = ILAENV(1,
# 'ZUNGQR') = 32 (ilaenv.f 406-411, 420-425), NBMIN = 2, NX = 0, and where NB > 1 and NB < K NX = MAX(0, ILAENV(3) =
# 128, ilaenv.f 650-663), where also NX < K and LWORK < IWS = LDWORK NB (LDWORK = N) NB = LWORK / LDWORK and NBMIN =
# MAX(2, ILAENV(2) = 2, ilaenv.f 573-578, 587-592); where NB >= NBMIN, NB < K and NX < K, KI = ((K - NX - 1) / NB)
# NB, KK = MIN(K, KI + NB) and DO 20 A(1:KK,KK+1:N) = ZERO, else KK = 0; ZUNG2R(M - KK, N - KK, K - KK,
# A(KK+1,KK+1)) where KK < N; where KK > 0 DO 50 I = KI + 1, 1, -NB (_ungqr_block; the loop present where the static
# extent admits K > 128)
def _ungqr(m, n, k, a, base, taus, when, lwork):
    q = _counted(a)
    go = when & (n > 0)
    nb = torch.full_like(k, 32)
    considered = (nb > 1) & (nb < k)
    nx = torch.where(considered, 128, 0)
    short = considered & (nx < k) & (lwork < n * nb)
    nb = torch.where(short, torch.div(lwork, torch.where(short, n, 1), rounding_mode='trunc'), nb)
    blocked = go & (nb >= 2) & (nb < k) & (nx < k)
    ki = torch.where(blocked, torch.div(k - nx - 1, torch.where(blocked, nb, 1), rounding_mode='trunc') * nb, 0)
    kk = torch.where(blocked, torch.minimum(k, ki + nb), 0)
    a = torch.where(_mat(blocked) & _span(q, base + 1, base + kk)[..., :, None]
                    & _span(q, base + kk + 1, base + n)[..., None, :], 0, a)
    a = _ung2r(m - kk, n - kk, k - kk, a, base + kk, taus, go & (kk < n), a.shape[-1] - 1)
    if a.shape[-1] >= 130:
        _, a, _ = while_loop_op(_first_any, _ungqr_block, (blocked & (kk > 0), a.contiguous(), ki + 1),
                                (nb, k, m, n, base, taus))
    return a


# zungqr.f 250-281, DO 50's pass for I: IB = MIN(NB, K - I + 1); where I + IB <= N ZLARFT('Forward', 'Columnwise',
# M - I + 1, IB, A(I,I), TAU(I), WORK) (_larft) and ZLARFB('Left', 'No transpose', 'Forward', 'Columnwise', M - I + 1,
# N - I - IB + 1, IB, A(I,I), WORK, A(I,I+IB)) (_larfb); ZUNG2R(M - I + 1, IB, IB, A(I,I), TAU(I)); DO 40
# A(1:I-1,I:I+IB-1) = ZERO
def _ungqr_block(live, a, i, nb, k, m, n, base, taus):
    size, q = a.shape[-1], _counted(a)
    ib = torch.minimum(nb, k - i + 1)
    g = base + i
    apply = live & (i + ib <= n)
    v = _region_at(a, g, g, size, 32)
    t = _larft(m - i + 1, ib, v, _vec_at(taus, g - 1, 32), apply)
    made = _larfb(True, False, m - i + 1, n - i - ib + 1, ib, v, t, _region_at(a, g, g + ib, size, size))
    a = _region_put(a, g, g + ib, made, _square_within(m - i + 1, size, n - i - ib + 1) & _mat(apply))
    a = _ung2r(m - i + 1, ib, ib, a, g - 1, taus, live, 32)
    a = torch.where(_mat(live) & _span(q, base + 1, g - 1)[..., :, None] & _span(q, g, g + ib - 1)[..., None, :], 0, a)
    i = torch.where(live, i - nb, i)
    return live & (i >= 1), a, i


# Reference-LAPACK 3.12.1 SRC/zung2r.f 163-192 (dorg2r.f 162-191 alike), ZUNG2R(M, N, K, A, TAU) on a with its
# A(1,1) at a(R0+1,R0+1), its TAU(I) taus' R0 + I - 1, where `when`: the N <= 0 return; DO 20 J = K + 1, N (column J of
# the identity over rows 1..M); DO 40 I = K, 1, -1 as static I from `bound` where I <= K: ZLARF1F('Left', M - I + 1,
# N - I, A(I,I), TAU(I), A(I,I+1)) where I < N, ZSCAL(M - I, -TAU(I), A(I+1,I)) where I < M, A(I,I) = ONE - TAU(I),
# DO 30 A(1:I-1,I) = ZERO
def _ung2r(m, n, k, a, r0, taus, when, bound):
    q = _counted(a)
    rows, cols = q[:, None], q
    go = when & (n > 0)
    fresh = _mat(go) & _span(q, r0 + 1, r0 + m)[..., :, None] & _span(q, r0 + k + 1, r0 + n)[..., None, :]
    a = torch.where(fresh, (rows == cols).to(a.dtype), a)
    for i in range(bound, 0, -1):
        active = go & (i <= k)
        g = r0 + i
        tau = _at(taus, g - 1)
        a = _larf1f(True, m - i + 1, n - i, _column(a, g), tau, a, g, g + 1, active & (i < n))
        a = torch.where(_mat(active & (i < m)) & (cols == _mat(g)) & _span(q, g + 1, r0 + m)[..., :, None],
                        a * _mat(-tau), a)
        a = _entry_put(a, g, g, 1 - tau, active)
        a = torch.where(_mat(active) & (cols == _mat(g)) & _span(q, r0 + 1, g - 1)[..., :, None], 0, a)
    return a


# Reference-LAPACK 3.12.1 SRC/dlarfg.f and zlarfg.f on batch scalars (_larfg, N = len(xs) + 1) or on a vector X
# whose entries past the member's N - 1 are zero (_larfg_vector, its result there unused): (ALPHA on exit, the beta
# where TAU is nonzero; TAU; X on exit). XNORM = DNRM2/DZNRM2 of X; TAU = 0 where XNORM = 0 (and ALPHI = 0,
# complex), else BETA = -SIGN(DLAPY2(ALPHA, XNORM), ALPHA) real, -SIGN(DLAPY3(ALPHR, ALPHI, XNORM), ALPHR) complex;
# the label 10 KNT loop (at most 20 passes, X by RSAFMN through DSCAL/ZDSCAL) and the DO 20 undo are static passes,
# SAFMIN = DLAMCH('S') / DLAMCH('E'); X times ONE / (ALPHA - BETA) (DSCAL) or ZLADIV(ONE, ALPHA - BETA) (ZSCAL)
def _larfg(alpha, xs):
    beta, tau, made = _larfg_vector(alpha, torch.stack(xs, -1))
    return beta, tau, list(made.unbind(-1))


def _larfg_vector(alpha, x):
    whole = alpha.is_complex()
    limits = torch.finfo(alpha.real.dtype)
    safmin = limits.tiny / (limits.eps / 2)
    rsafmn = 1 / safmin
    parts = [alpha.real, alpha.imag] if whole else [alpha]

    def norm(values):
        if values.shape[-1] == 1:
            return values[..., 0].abs()
        return _norm2(values, torch.ones_like(values, dtype=torch.bool))

    def beta_of(ps, xnorm):
        return -_fsign(_dlapy3(ps[0], ps[1], xnorm) if whole else _dlapy2(ps[0], xnorm), ps[0])
    xnorm = norm(x)
    trivial = (xnorm == 0) & (parts[1] == 0) if whole else xnorm == 0
    beta = beta_of(parts, xnorm)
    small = beta.abs() < safmin
    running, knt, sb, sp, sx = small, torch.zeros_like(beta, dtype=torch.int64), beta, list(parts), x
    for _ in range(20):
        knt = torch.where(running, knt + 1, knt)
        sx = torch.where(_vec(running), _scaled(sx, rsafmn), sx)
        sb = torch.where(running, sb * rsafmn, sb)
        sp = [torch.where(running, p * rsafmn, p) for p in sp]
        running = running & (sb.abs() < safmin)
    fb = torch.where(small, beta_of(sp, norm(sx)), beta)
    if whole:
        tau = torch.complex((fb - sp[0]) / fb, -sp[1] / fb)
        made = _vec(torch.complex(*_divided(torch.ones_like(fb), torch.zeros_like(fb), sp[0] - fb, sp[1]))) * sx
    else:
        tau = (fb - sp[0]) / fb
        made = sx * _vec(1 / (sp[0] - fb))
    undone = fb
    for j in range(20):
        undone = torch.where(knt > j, undone * safmin, undone)
    if whole:
        undone = torch.complex(undone, torch.zeros_like(undone))
    return torch.where(trivial, alpha, undone), torch.where(trivial, 0, tau), torch.where(_vec(trivial), x, made)


# Reference-LAPACK SRC/dlanv2.f 159-306 on batch scalars: the Schur factorization of the real 2 x 2 [A B; C D] to
# (A, B, C, D, RT1R, RT1I, RT2R, RT2I, CS, SN), each source branch selected. The label 10 rescaling loop is its static
# passes (COUNT <= 20, at most 21), and the 251-257 products stay separate (Reference-LAPACK issue 1031). SAFMIN/EPS
# is an exact power of two for each floating kind, so SAFMN2 = BASE**INT(LOG(SAFMIN / EPS) / LOG(BASE) / TWO)
def _dlanv2(a, b, c, d):
    limits = torch.finfo(a.dtype)
    safmn2 = math.ldexp(1.0, math.trunc(math.log2(limits.tiny / limits.eps) / 2))
    safmx2, eps = 1 / safmn2, limits.eps
    one, zero = torch.ones_like(a), torch.zeros_like(a)
    first = c == 0
    second = ~first & (b == 0)
    third = ~first & ~second & ((a - d) == 0) & (_fsign(one, b) != _fsign(one, c))
    general = ~first & ~second & ~third
    temp = a - d
    p = 0.5 * temp
    ab, ac = b.abs(), c.abs()
    bcmax = torch.where(ab >= ac, ab, ac)
    bcmis = torch.where(ab <= ac, ab, ac) * _fsign(one, b) * _fsign(one, c)
    scale = torch.where(p.abs() >= bcmax, p.abs(), bcmax)
    z = (p / scale) * p + (bcmax / scale) * bcmis
    real_pair = general & (z >= 4 * eps)
    zr = p + _fsign(scale.sqrt() * z.sqrt(), p)
    ar, dr = d + zr, d - (bcmax / zr) * bcmis
    tau = _dlapy2(c, zr)
    csr, snr, br = zr / tau, c / tau, b - c
    sigma, running = b + c, torch.ones_like(first)
    for count in range(1, 22):
        size = torch.where(temp.abs() >= sigma.abs(), temp.abs(), sigma.abs())
        down = running & (size >= safmx2)
        up = running & ~down & (size <= safmn2)
        sigma = torch.where(down, sigma * safmn2, torch.where(up, sigma * safmx2, sigma))
        temp = torch.where(down, temp * safmn2, torch.where(up, temp * safmx2, temp))
        running = (down | up) & (count <= 20)
    pc = 0.5 * temp
    tau = _dlapy2(sigma, temp)
    cs = (0.5 * (1 + sigma.abs() / tau)).sqrt()
    sn = -(pc / (tau * cs)) * _fsign(one, sigma)
    aa, bb = a * cs + b * sn, -a * sn + b * cs
    cc, dd = c * cs + d * sn, -c * sn + d * cs
    a2, b2 = aa * cs + cc * sn, bb * cs + dd * sn
    c2, d2 = -aa * sn + cc * cs, -bb * sn + dd * cs
    mean = 0.5 * (a2 + d2)
    split = (c2 != 0) & (b2 != 0) & (_fsign(one, b2) == _fsign(one, c2))
    turned = (c2 != 0) & (b2 == 0)
    sab, sac = b2.abs().sqrt(), c2.abs().sqrt()
    pp = _fsign(sab * sac, c2)
    tau1 = 1 / (b2 + c2).abs().sqrt()
    cs1, sn1 = sab * tau1, sac * tau1
    ax, dx = torch.where(split, mean + pp, mean), torch.where(split, mean - pp, mean)
    bx = torch.where(split, b2 - c2, torch.where(turned, -c2, b2))
    cx = torch.where(split | turned, zero, c2)
    csx = torch.where(split, cs * cs1 - sn * sn1, torch.where(turned, -sn, cs))
    snx = torch.where(split, cs * sn1 + sn * cs1, torch.where(turned, cs, sn))
    complex_ = general & ~real_pair
    made = []
    for kept, swapped, pair, other in ((a, d, ar, ax), (b, -c, br, bx), (c, zero, zero, cx), (d, a, dr, dx),
                                       (one, zero, csr, csx), (zero, one, snr, snx)):
        made.append(torch.where(second, swapped, torch.where(real_pair, pair, torch.where(complex_, other, kept))))
    a_, b_, c_, d_, cs_, sn_ = made
    rt1i = torch.where(c_ == 0, zero, b_.abs().sqrt() * c_.abs().sqrt())
    return a_, b_, c_, d_, a_, rt1i, d_, torch.where(c_ == 0, zero, -rt1i), cs_, sn_

def _row_put(x, r, row):
    return _put(x, row[..., None, :], r, 0)


def _column_put(x, c, column):
    return _put(x, column[..., :, None], 0, c)


# zlahqr.f 297-322 / dlahqr.f 298-323 (DO 40 / DO 30): K, the largest k in L+1..I whose H(k,k-1) is negligible (the
# SMLNUM test, or the TST and Ahues-Tisseur tests), else L
def _deflation(h, i, l, ilo, ihi, smlnum, ulp):
    diagonal, sub, sup = (h.diagonal(offset, dim1=-2, dim2=-1) for offset in (0, -1, 1))
    kk = _counted(h)[1:]
    p, q = diagonal[..., :-1], diagonal[..., 1:]
    pad = torch.zeros_like(sub[..., :1])
    tst = _abs1(p) + _abs1(q)
    vanished = tst == 0
    tst = torch.where(vanished & (kk - 2 >= _vec(ilo)), tst + _re(torch.cat([pad, sub[..., :-1]], -1)).abs(), tst)
    tst = torch.where(vanished & (kk + 1 <= _vec(ihi)), tst + _re(torch.cat([sub[..., 1:], pad], -1)).abs(), tst)
    ab, ba = torch.maximum(_abs1(sub), _abs1(sup)), torch.minimum(_abs1(sub), _abs1(sup))
    aa, bb = torch.maximum(_abs1(q), _abs1(p - q)), torch.minimum(_abs1(q), _abs1(p - q))
    s = aa + ab
    negligible = (_abs1(sub) <= _vec(smlnum)) | ((_re(sub).abs() <= ulp * tst) & (
        ba * (ab / s) <= torch.maximum(_vec(smlnum), ulp * (bb * (aa / s)))))
    found = torch.where((kk > _vec(l)) & (kk <= _vec(i)) & negligible, kk, 0).amax(-1)
    return torch.where(found > 0, found, l)


# zlahqr.f 271-276 / dlahqr.f 265-270 (DO 10): the trash below the second subdiagonal ZGEHRD's reflectors left
def _cleared(h, ilo, ihi, proceed):
    k = _counted(h)
    below = k[:, None] - k
    trash = ((below == 2) | (below == 3)) & _span(k, ilo, ihi - 3)[..., None, :]
    trash = trash | ((k[:, None] == _mat(ihi)) & (k == _mat(ihi - 2)) & _mat(ilo <= ihi - 2))
    return torch.where(trash & _mat(proceed), 0, h)


# Reference-LAPACK SRC/zlahqr.f 258-573, ZLAHQR(WANTT, WANTZ, N, ILO, IHI, H, W, ILOZ, IHIZ, Z) where `proceed`, N
# the member's ORDER within h's static extent: ILO = IHI's W(ILO) = H(ILO,ILO) return; the trash, the DO 20
# subdiagonal made real (each static I in order, its row scaling over I..JHI and column scaling over JLO..MIN(JHI,
# I+1), JLO, JHI = 1, N where WANTT, else ILO, IHI; Z's rows ILOZ..IHIZ where WANTZ), then the iteration as one
# while_loop whose pass is one DO 130 iteration (labels 30 and 140 flattened: on deflation W(I) = H(I,I), KDEFL = 0,
# I = L - 1, L = ILO, ITS = 0; INFO = I past ITMAX). Returns H, W (positions ILO..IHI), Z and INFO
def _zlahqr(wantt, wantz, order, ilo, ihi, h, w, iloz, ihiz, z, proceed):
    n, k = h.shape[-1], _counted(h)
    w = _at_put(w, ilo, _entry(h, ilo, ilo), proceed & (ilo == ihi))
    proceed = proceed & (ilo != ihi)
    h = _cleared(h, ilo, ihi, proceed)
    jlo, jhi = (torch.ones_like(ilo), order) if wantt else (ilo, ihi)
    rows_z = _span(k, iloz, ihiz)
    for ii in range(2, n + 1):
        sub = h[..., ii - 1, ii - 2]
        hit = proceed & (ilo + 1 <= ii) & (ii <= ihi) & (sub.imag != 0)
        sc = _over(sub, _abs1(sub))
        sc = _over(sc.conj(), sc.abs())
        h = _entry_put(h, torch.full_like(ilo, ii), torch.full_like(ilo, ii - 1), sub.abs(), hit)
        cols = _vec(hit) & (k >= ii) & (k <= _vec(jhi))
        h = _row_put(h, ii - 1, torch.where(cols, h[..., ii - 1, :] * _vec(sc), h[..., ii - 1, :]))
        rows = _vec(hit) & _span(k, jlo, torch.clamp(jhi, max=ii + 1))
        h = _column_put(h, ii - 1, torch.where(rows, h[..., :, ii - 1] * _vec(sc.conj()), h[..., :, ii - 1]))
        if wantz:
            z = _column_put(z, ii - 1, torch.where(_vec(hit) & rows_z, z[..., :, ii - 1] * _vec(sc.conj()),
                                                   z[..., :, ii - 1]))
    smlnum, itmax = _iteration_bounds(h, ilo, ihi)
    zero = torch.zeros_like(ilo)
    carried = (proceed & (ihi >= ilo), h.contiguous(), z.contiguous() if wantz else h.new_zeros(()), w.contiguous(),
               ihi.clone(), ilo.clone(), zero, zero.clone(), zero.clone())
    _, h, z, w, _, _, _, _, info = while_loop_op(_first_any, functools.partial(
        _zlahqr_pass, wantt, wantz, torch.finfo(h.real.dtype).eps), carried, (ilo, ihi, smlnum, itmax, order, iloz,
                                                                              ihiz))
    return h, w, z, info


# SMLNUM = SAFMIN (NH / ULP) and ITMAX = 30 MAX(10, NH), NH = IHI - ILO + 1
def _iteration_bounds(h, ilo, ihi):
    limits, nh = torch.finfo(h.real.dtype), ihi - ilo + 1
    return limits.tiny * (nh.to(h.real.dtype) / limits.eps), 30 * torch.clamp(nh, min=10)


def _zlahqr_pass(wantt, wantz, ulp, live, h, z, w, i, l, its, kdefl, info, ilo, ihi, smlnum, itmax, order,
                 iloz, ihiz):
    n, k = h.shape[-1], _counted(h)
    l = torch.where(live, _deflation(h, i, l, ilo, ihi, smlnum, ulp), l)
    h = _entry_put(h, l, l - 1, 0, live & (l > ilo))
    deflate = live & (l >= i)
    sweep = live & ~deflate
    w = _at_put(w, i, _entry(h, i, i), deflate)
    kdefl = torch.where(sweep, kdefl + 1, kdefl)
    i1, i2 = (torch.ones_like(l), order) if wantt else (l, i)
    rows_z = _span(k, iloz, ihiz)

    # 355-386: the exceptional shifts (KDEFL a multiple of 2 KEXSH at I, of KEXSH at L; DAT1 = 3/4), else Wilkinson's
    hii, hii1 = _entry(h, i, i), _entry(h, i, i - 1)
    u = _entry(h, i - 1, i).sqrt() * hii1.sqrt()
    us = _abs1(u)
    x = _scaled(_entry(h, i - 1, i - 1) - hii, 0.5)
    sx = _abs1(x)
    s = torch.maximum(us, sx)
    xs, uu = _over(x, s), _over(u, s)
    y = _scaled((xs * xs + uu * uu).sqrt(), s)
    xn = _over(x, sx)
    y = torch.where((sx > 0) & (xn.real * y.real + xn.imag * y.imag < 0), -y, y)
    t = torch.where(kdefl % 20 == 0, _plus_real(hii, 0.75 * hii1.real.abs()), torch.where(
        kdefl % 10 == 0, _plus_real(_entry(h, l, l), 0.75 * _entry(h, l + 1, l).real.abs()),
        torch.where(us != 0, hii - u * _cdiv(u, x + y), hii)))

    # 390-414: M, the largest m in L+1..I-1 where two consecutive small subdiagonals hold, else L; V at M
    m = l
    if n > 2:
        diagonal, sub = h.diagonal(dim1=-2, dim2=-1), h.diagonal(-1, dim1=-2, dim2=-1)
        mm, h11, h22 = k[1:-1], diagonal[..., 1:-1], diagonal[..., 2:]
        h11s, h21 = h11 - _vec(t), sub[..., 1:].real
        s = _abs1(h11s) + h21.abs()
        h11s, h21 = _over(h11s, s), h21 / s
        ok = sub[..., :-1].real.abs() * h21.abs() <= ulp * (_abs1(h11s) * (_abs1(h11) + _abs1(h22)))
        found = torch.where((mm > _vec(l)) & (mm < _vec(i)) & ok, mm, 0).amax(-1)
        m = torch.where(found > 0, found, l)
    h11s, h21 = _entry(h, m, m) - t, _entry(h, m + 1, m).real
    s = _abs1(h11s) + h21.abs()
    v1, v2 = _over(h11s, s), torch.complex(h21 / s, torch.zeros_like(h21))

    # 418-482: the bulge chase, each static K = M..I-1
    for kk in range(1, n):
        act = sweep & (m <= kk) & (kk <= i - 1)
        later = act & (kk > m)
        if kk > 1:
            v1 = torch.where(later, h[..., kk - 1, kk - 2], v1)
            v2 = torch.where(later, h[..., kk, kk - 2], v2)
        v1, t1, (v2,) = _larfg(v1, [v2])
        if kk > 1:
            column = h[..., :, kk - 2]
            column = torch.where(_vec(later) & (k == kk), _vec(v1), torch.where(_vec(later) & (k == kk + 1), 0, column))
            h = _column_put(h, kk - 2, column)
        t2 = (t1 * v2).real
        cols = _vec(act) & (k >= kk) & (k <= _vec(i2))
        hk, hk1 = h[..., kk - 1, :], h[..., kk, :]
        total = _vec(t1.conj()) * hk + _scaled(hk1, _vec(t2))
        h = _put(h, torch.stack([torch.where(cols, hk - total, hk), torch.where(cols, hk1 - total * _vec(v2), hk1)],
                                -2), kk - 1, 0)
        rows = _vec(act) & (k >= _vec(i1)) & (k <= _vec(torch.clamp(i, max=kk + 2)))
        hk, hk1 = h[..., :, kk - 1], h[..., :, kk]
        total = _vec(t1) * hk + _scaled(hk1, _vec(t2))
        h = _put(h, torch.stack([torch.where(rows, hk - total, hk), torch.where(
            rows, hk1 - total * _vec(v2.conj()), hk1)], -1), 0, kk - 1)
        if wantz:
            zk, zk1 = z[..., :, kk - 1], z[..., :, kk]
            total = _vec(t1) * zk + _scaled(zk1, _vec(t2))
            moved = _vec(act) & rows_z
            z = _put(z, torch.stack([torch.where(moved, zk - total, zk), torch.where(
                moved, zk1 - total * _vec(v2.conj()), zk1)], -1), 0, kk - 1)

        # 458-480: at K = M > L, TEMP = (1 - T1) / |1 - T1| scales H(M+1,M), H(M+2,M+1) and rows/columns J = M..I
        first = act & (kk == m) & (m > l)
        temp = torch.complex(1 - t1.real, 0 - t1.imag)
        temp = _over(temp, temp.abs())
        h = _entry_put(h, m + 1, m, _entry(h, m + 1, m) * temp.conj(), first)
        h = _entry_put(h, m + 2, m + 1, _entry(h, m + 2, m + 1) * temp, first & (m + 2 <= i))
        chosen = _span(k, m, i) & (k != _vec(m + 1))
        h = torch.where(_mat(first) & chosen[..., :, None] & (k > k[:, None]) & (k <= _mat(i2)), h * _mat(temp), h)
        h = torch.where(_mat(first) & chosen[..., None, :] & (k[:, None] >= _mat(i1)) & (k[:, None] < k),
                        h * _mat(temp.conj()), h)
        if wantz:
            z = torch.where(_mat(first) & chosen[..., None, :] & rows_z[..., :, None], z * _mat(temp.conj()), z)

    # 486-498: H(I,I-1) made real, row I and column I by its phase
    temp = _entry(h, i, i - 1)
    phased = sweep & (temp.imag != 0)
    rtemp = temp.abs()
    h = _entry_put(h, i, i - 1, rtemp, phased)
    temp = _over(temp, rtemp)
    h = torch.where(_mat(phased) & (k[:, None] == _mat(i)) & (k > _mat(i)) & (k <= _mat(i2)), h * _mat(temp.conj()), h)
    h = torch.where(_mat(phased) & (k == _mat(i)) & (k[:, None] >= _mat(i1)) & (k[:, None] < _mat(i)), h * _mat(temp),
                    h)
    if wantz:
        z = torch.where(_mat(phased) & (k == _mat(i)) & rows_z[..., :, None], z * _mat(temp), z)
    return _iterated(live, deflate, sweep, h, z if wantz else z.clone(), (w,), i, l, its, kdefl, info, ilo, itmax)


# the pass's ITS, KDEFL, I, L and INFO: a deflation resets ITS and KDEFL, I = L - 1 and L = ILO; a sweep counts
# ITS, the sweep at ITS = ITMAX the last (INFO = I)
def _iterated(live, deflate, sweep, h, z, eigen, i, l, its, kdefl, info, ilo, itmax):
    fail = sweep & (its >= itmax)
    following = torch.where(deflate, l - 1, i)
    return (live & ~fail & (following >= ilo), h, z, *eigen, following, torch.where(deflate, ilo, l),
            torch.where(deflate, 0, torch.where(sweep, its + 1, its)), torch.where(deflate, 0, kdefl),
            torch.where(fail, i, info))


# Reference-LAPACK SRC/dlahqr.f 258-570, DLAHQR(WANTT, WANTZ, N, ILO, IHI, H, WR, WI, ILOZ, IHIZ, Z) as _zlahqr (no
# DO 20; ILO = IHI's WR(ILO) = H(ILO,ILO), WI(ILO) = 0 return): a deflation of one (WR(I) = H(I,I), WI(I) = 0) or
# two (DLANV2's standardized block, its DROTs of H's rows I-1, I over I+1..I2 and columns over I1..I-2 where WANTT,
# Z's columns over ILOZ..IHIZ where WANTZ) at L >= I-1; the 343-379 exceptional (DAT1 = 3/4, DAT2 = -0.4375) or
# Francis double shift, the 383-400 M search over L+1..I-2, and the 404-481 bulge chase of NR = MIN(3, I-K+1)
# reflectors. Returns H, WR, WI, Z and INFO
def _dlahqr(wantt, wantz, order, ilo, ihi, h, wr, wi, iloz, ihiz, z, proceed):
    single = proceed & (ilo == ihi)
    wr, wi = _at_put(wr, ilo, _entry(h, ilo, ilo), single), _at_put(wi, ilo, 0, single)
    proceed = proceed & (ilo != ihi)
    h = _cleared(h, ilo, ihi, proceed)
    smlnum, itmax = _iteration_bounds(h, ilo, ihi)
    zero = torch.zeros_like(ilo)
    carried = (proceed & (ihi >= ilo), h.contiguous(), z.contiguous() if wantz else h.new_zeros(()), wr.contiguous(),
               wi.contiguous(), ihi.clone(), ilo.clone(), zero, zero.clone(), zero.clone())
    _, h, z, wr, wi, _, _, _, _, info = while_loop_op(_first_any, functools.partial(
        _dlahqr_pass, wantt, wantz, torch.finfo(h.dtype).eps), carried, (ilo, ihi, smlnum, itmax, order, iloz, ihiz))
    return h, wr, wi, z, info


# dlahqr.f 387-397: V for the double shift at M, of H(M,M), H(M+1,M), H(M,M+1), H(M+1,M+1) and H(M+2,M+1)
def _double_shifted(hmm, hm1m, hmm1, hm1m1, hm2m1, rt1r, rt1i, rt2r, rt2i):
    s = (hmm - rt2r).abs() + rt2i.abs() + hm1m.abs()
    h21s = hm1m / s
    v1 = h21s * hmm1 + (hmm - rt1r) * ((hmm - rt2r) / s) - rt1i * (rt2i / s)
    v2 = h21s * (hmm + hm1m1 - rt1r - rt2r)
    v3 = h21s * hm2m1
    s = v1.abs() + v2.abs() + v3.abs()
    return v1 / s, v2 / s, v3 / s


def _dlahqr_pass(wantt, wantz, ulp, live, h, z, wr, wi, i, l, its, kdefl, info, ilo, ihi, smlnum, itmax, order,
                 iloz, ihiz):
    n, k = h.shape[-1], _counted(h)
    l = torch.where(live, _deflation(h, i, l, ilo, ihi, smlnum, ulp), l)
    h = _entry_put(h, l, l - 1, 0, live & (l > ilo))
    deflate = live & (l >= i - 1)
    sweep = live & ~deflate
    rows_z = _span(k, iloz, ihiz)

    # 150: one eigenvalue, or the 2 x 2 block's
    single, pair = deflate & (l == i), deflate & (l == i - 1)
    wr, wi = _at_put(wr, i, _entry(h, i, i), single), _at_put(wi, i, 0, single)
    a, b, c, d, rt1r, rt1i, rt2r, rt2i, cs, sn = _dlanv2(_entry(h, i - 1, i - 1), _entry(h, i - 1, i),
                                                         _entry(h, i, i - 1), _entry(h, i, i))
    for row, col, value in ((i - 1, i - 1, a), (i - 1, i, b), (i, i - 1, c), (i, i, d)):
        h = _entry_put(h, row, col, value, pair)
    wr = _at_put(_at_put(wr, i - 1, rt1r, pair), i, rt2r, pair)
    wi = _at_put(_at_put(wi, i - 1, rt1i, pair), i, rt2i, pair)
    if wantt:
        upper, lower = _row(h, i - 1), _row(h, i)
        cols = (_vec(pair) & (k > _vec(i)) & (k <= _vec(order)))[..., None, :]
        h = torch.where(cols & (k[:, None] == _mat(i - 1)), (_vec(cs) * upper + _vec(sn) * lower)[..., None, :], h)
        h = torch.where(cols & (k[:, None] == _mat(i)), (_vec(cs) * lower - _vec(sn) * upper)[..., None, :], h)
        left, right = _column(h, i - 1), _column(h, i)
        rows = (_vec(pair) & (k <= _vec(i - 2)))[..., :, None]
        h = torch.where(rows & (k == _mat(i - 1)), (_vec(cs) * left + _vec(sn) * right)[..., :, None], h)
        h = torch.where(rows & (k == _mat(i)), (_vec(cs) * right - _vec(sn) * left)[..., :, None], h)
    if wantz:
        left, right = _column(z, i - 1), _column(z, i)
        rows = (_vec(pair) & rows_z)[..., :, None]
        z = torch.where(rows & (k == _mat(i - 1)), (_vec(cs) * left + _vec(sn) * right)[..., :, None], z)
        z = torch.where(rows & (k == _mat(i)), (_vec(cs) * right - _vec(sn) * left)[..., :, None], z)

    kdefl = torch.where(sweep, kdefl + 1, kdefl)
    i1, i2 = (torch.ones_like(l), order) if wantt else (l, i)

    # 343-379: the shifts
    exceptional_i = kdefl % 20 == 0
    exceptional = exceptional_i | (kdefl % 10 == 0)
    s = torch.where(exceptional_i, _entry(h, i, i - 1).abs() + _entry(h, i - 1, i - 2).abs(),
                    _entry(h, l + 1, l).abs() + _entry(h, l + 2, l + 1).abs())
    shifted = 0.75 * s + torch.where(exceptional_i, _entry(h, i, i), _entry(h, l, l))
    h11 = torch.where(exceptional, shifted, _entry(h, i - 1, i - 1))
    h12 = torch.where(exceptional, -0.4375 * s, _entry(h, i - 1, i))
    h21 = torch.where(exceptional, s, _entry(h, i, i - 1))
    h22 = torch.where(exceptional, shifted, _entry(h, i, i))
    s = h11.abs() + h12.abs() + h21.abs() + h22.abs()
    vanished = s == 0
    safe = torch.where(vanished, 1, s)
    h11, h21, h12, h22 = h11 / safe, h21 / safe, h12 / safe, h22 / safe
    tr = (h11 + h22) / 2
    det = (h11 - tr) * (h22 - tr) - h12 * h21
    rtdisc = det.abs().sqrt()
    conjugate = ~vanished & (det >= 0)
    first, second = tr + rtdisc, tr - rtdisc
    nearer = torch.where((first - h22).abs() <= (second - h22).abs(), first * s, second * s)
    rt1r = torch.where(vanished, 0, torch.where(conjugate, tr * s, nearer))
    rt2r = rt1r
    rt1i = torch.where(conjugate, rtdisc * s, 0)
    rt2i = torch.where(conjugate, -(rtdisc * s), 0)

    # 383-400: M
    m = l
    if n > 3:
        diagonal, sub, sup = (h.diagonal(offset, dim1=-2, dim2=-1) for offset in (0, -1, 1))
        hmm, hm1m1 = diagonal[..., 1:-2], diagonal[..., 2:-1]
        v1, v2, v3 = _double_shifted(hmm, sub[..., 1:-1], sup[..., 1:-1], hm1m1, sub[..., 2:], _vec(rt1r), _vec(rt1i),
                                     _vec(rt2r), _vec(rt2i))
        ok = sub[..., :-2].abs() * (v2.abs() + v3.abs()) <= ulp * v1.abs() * (
            diagonal[..., :-3].abs() + hmm.abs() + hm1m1.abs())
        mm = k[1:-2]
        found = torch.where((mm > _vec(l)) & (mm <= _vec(i) - 2) & ok, mm, 0).amax(-1)
        m = torch.where(found > 0, found, l)
    v1, v2, v3 = _double_shifted(_entry(h, m, m), _entry(h, m + 1, m), _entry(h, m, m + 1), _entry(h, m + 1, m + 1),
                                 _entry(h, m + 2, m + 1), rt1r, rt1i, rt2r, rt2i)

    # 404-481: each static K = M..I-1, NR = 3 where K <= I-2 (a K = n-1 is NR = 2 alone)
    for kk in range(1, n):
        act = sweep & (m <= kk) & (kk <= i - 1)
        three = act & (kk <= i - 2) if kk + 1 < n else torch.zeros_like(act)
        later = act & (kk > m)
        if kk > 1:
            v1 = torch.where(later, h[..., kk - 1, kk - 2], v1)
            v2 = torch.where(later, h[..., kk, kk - 2], v2)
            if kk + 1 < n:
                v3 = torch.where(later & three, h[..., kk + 1, kk - 2], v3)
        b2, t2_, (x2,) = _larfg(v1, [v2])
        if kk + 1 < n:
            b3, t3_, (y2, y3) = _larfg(v1, [v2, v3])
            v1, t1, v2, v3 = (torch.where(three, b3, b2), torch.where(three, t3_, t2_), torch.where(three, y2, x2),
                              torch.where(three, y3, v3))
        else:
            v1, t1, v2 = b2, t2_, x2
        if kk > 1:
            column = h[..., :, kk - 2]
            reduced = act & (kk == m) & (m > l)
            column = torch.where(_vec(reduced) & (k == kk), column * _vec(1 - t1), column)
            column = torch.where(_vec(later) & (k == kk), _vec(v1), column)
            column = torch.where(_vec(later) & ((k == kk + 1) | ((k == kk + 2) & _vec(kk < i - 1))), 0, column)
            h = _column_put(h, kk - 2, column)
        t2, t3 = t1 * v2, t1 * v3
        cols = _vec(act) & (k >= kk) & (k <= _vec(i2))
        rows = _vec(act) & (k >= _vec(i1)) & (k <= _vec(torch.where(three, torch.clamp(i, max=kk + 3), i)))
        h = _reflected3(h, kk, cols, rows, three, v2, v3, t1, t2, t3)
        if wantz:
            z = _reflected3(z, kk, None, _vec(act) & rows_z, three, v2, v3, t1, t2, t3)
    return _iterated(live, deflate, sweep, h, z if wantz else z.clone(), (wr, wi), i, l, its, kdefl, info, ilo, itmax)


# dlahqr.f 424-479: the NR = 3 (`three`) or NR = 2 reflector at static K from the left to rows K..K+NR-1 over the
# columns `cols` (skipped where None, Z's) and from the right to columns K..K+NR-1 over the rows `rows`
def _reflected3(x, kk, cols, rows, three, v2, v3, t1, t2, t3):
    wide = kk + 1 < x.shape[-1]
    if cols is not None:
        hk, hk1 = x[..., kk - 1, :], x[..., kk, :]
        total = hk + _vec(v2) * hk1
        if wide:
            hk2 = x[..., kk + 1, :]
            total = torch.where(_vec(three), total + _vec(v3) * hk2, total)
        made = [torch.where(cols, hk - total * _vec(t1), hk), torch.where(cols, hk1 - total * _vec(t2), hk1)]
        if wide:
            made.append(torch.where(cols & _vec(three), hk2 - total * _vec(t3), hk2))
        x = _put(x, torch.stack(made, -2), kk - 1, 0)
    hk, hk1 = x[..., :, kk - 1], x[..., :, kk]
    total = hk + _vec(v2) * hk1
    if wide:
        hk2 = x[..., :, kk + 1]
        total = torch.where(_vec(three), total + _vec(v3) * hk2, total)
    made = [torch.where(rows, hk - total * _vec(t1), hk), torch.where(rows, hk1 - total * _vec(t2), hk1)]
    if wide:
        made.append(torch.where(rows & _vec(three), hk2 - total * _vec(t3), hk2))
    return _put(x, torch.stack(made, -1), 0, kk - 1)

# Reference-LAPACK SRC/dlaln2.f 276-601 with LTRANS = false, CA = 1, D1 = D2 = 1 on batch tensors: the NA x NA (1 or
# 2, static) system (A - (WR + i WI)) X = SCALE B, NW = 1 (real WR) or 2 (WR + i WI). `a` is (A11,) or (A11, A12,
# A21, A22) and `b` (and the returned X) B(1,1), B(2,1) [NA = 2], then B(1,2), B(2,2) [NW = 2]; returns (X, SCALE,
# XNORM). SMINI = max(SMIN, 2 SAFMIN), the complete pivoting of CRV/CIV by the IPIVOT, RSWAP and ZSWAP tables, the
# SMINI replacements and every SCALE/BNORM/BBND/XNORM overflow test; DLADIV is _divided
def _dlaln2(na, nw, smin, a, b, wr, wi):
    limits = torch.finfo(smin.dtype)
    smlnum = 2 * limits.tiny
    bignum = 1 / smlnum
    zero = torch.zeros_like(smin)

    def bigger(x, y):
        return torch.where(x >= y, x, y)

    def bounded(norm, size):
        return torch.where((size < 1) & (norm > 1) & (norm > bignum * size), 1 / norm, 1)
    smini = torch.where(smin >= smlnum, smin, smlnum)
    if na == 1:
        csr = a[0] - wr
        if nw == 1:
            small = csr.abs() < smini
            cnorm = torch.where(small, smini, csr.abs())
            csr = torch.where(small, smini, csr)
            scale = bounded(b[0].abs(), cnorm)
            x = (b[0] * scale) / csr
            return (x,), scale, x.abs()
        csi = -wi
        cnorm = csr.abs() + csi.abs()
        small = cnorm < smini
        csr, csi, cnorm = torch.where(small, smini, csr), torch.where(small, 0, csi), torch.where(small, smini, cnorm)
        scale = bounded(b[0].abs() + b[1].abs(), cnorm)
        xr, xi = _divided(scale * b[0], scale * b[1], csr, csi)
        return (xr, xi), scale, xr.abs() + xi.abs()
    a11, a12, a21, a22 = a
    crv = [a11 - wr, a21, a12, a22 - wr]
    civ = [-wi, zero, zero, -wi] if nw == 2 else [zero] * 4
    cmax, ic = zero, torch.zeros_like(smin, dtype=torch.int64)
    for j in range(4):
        size = crv[j].abs() if nw == 1 else crv[j].abs() + civ[j].abs()
        larger = size > cmax
        cmax, ic = torch.where(larger, size, cmax), torch.where(larger, j + 1, ic)
    pivot = {1: (1, 2, 3, 4), 2: (2, 1, 4, 3), 3: (3, 4, 1, 2), 4: (4, 3, 2, 1)}

    def picked(values, row):
        made = values[pivot[4][row] - 1]
        for column in (3, 2, 1):
            made = torch.where(ic == column, values[pivot[column][row] - 1], made)
        return made
    rswap, zswap = (ic == 2) | (ic == 4), (ic == 3) | (ic == 4)
    identity = cmax < smini
    if nw == 1:
        b11, b21 = b
        bnorm = bigger(b11.abs(), b21.abs())
        held = bounded(bnorm, smini)
        temp = held / smini
        plain = ((temp * b11, temp * b21), held, temp * bnorm)
        ur11, cr21, ur12, cr22 = (picked(crv, row) for row in range(4))
        ur11r = 1 / ur11
        lr21 = ur11r * cr21
        ur22 = cr22 - ur12 * lr21
        ur22 = torch.where(ur22.abs() < smini, smini, ur22)
        br1, br2 = torch.where(rswap, b21, b11), torch.where(rswap, b11, b21)
        br2 = br2 - lr21 * br1
        bbnd = bigger((br1 * (ur22 * ur11r)).abs(), br2.abs())
        scale = torch.where((bbnd > 1) & (ur22.abs() < 1) & (bbnd >= bignum * ur22.abs()), 1 / bbnd, 1)
        xr2 = (br2 * scale) / ur22
        xr1 = (scale * br1) * ur11r - xr2 * (ur11r * ur12)
        x = [torch.where(zswap, xr2, xr1), torch.where(zswap, xr1, xr2)]
        xnorm = bigger(xr1.abs(), xr2.abs())
    else:
        b11, b21, b12, b22 = b
        bnorm = bigger(b11.abs() + b12.abs(), b21.abs() + b22.abs())
        held = bounded(bnorm, smini)
        temp = held / smini
        plain = (tuple(temp * v for v in b), held, temp * bnorm)
        ur11, cr21, ur12, cr22 = (picked(crv, row) for row in range(4))
        ui11, ci21, ui12, ci22 = (picked(civ, row) for row in range(4))
        wide = ur11.abs() > ui11.abs()
        ratio = torch.where(wide, ui11 / ur11, ur11 / ui11)
        grown = 1 + ratio * ratio
        ur11r_a = torch.where(wide, 1 / (ur11 * grown), -(ratio * (-1 / (ui11 * grown))))
        ui11r_a = torch.where(wide, -(ratio * (1 / (ur11 * grown))), -1 / (ui11 * grown))
        offreal = (ic == 1) | (ic == 4)
        ur11r = torch.where(offreal, ur11r_a, 1 / ur11)
        ui11r = torch.where(offreal, ui11r_a, zero)
        lr21 = cr21 * ur11r
        li21 = torch.where(offreal, cr21 * ui11r, ci21 * ur11r)
        ur12s = ur12 * ur11r
        ui12s = torch.where(offreal, ur12 * ui11r, ui12 * ur11r)
        ur22 = torch.where(offreal, cr22 - ur12 * lr21, (cr22 - ur12 * lr21) + ui12 * li21)
        ui22 = torch.where(offreal, ci22 - ur12 * li21, -(ur12 * li21) - ui12 * lr21)
        u22abs = ur22.abs() + ui22.abs()
        small = u22abs < smini
        ur22, ui22 = torch.where(small, smini, ur22), torch.where(small, 0, ui22)
        br1, br2 = torch.where(rswap, b21, b11), torch.where(rswap, b11, b21)
        bi1, bi2 = torch.where(rswap, b22, b12), torch.where(rswap, b12, b22)
        br2 = (br2 - lr21 * br1) + li21 * bi1
        bi2 = (bi2 - li21 * br1) - lr21 * bi1
        bbnd = bigger((br1.abs() + bi1.abs()) * (u22abs * (ur11r.abs() + ui11r.abs())), br2.abs() + bi2.abs())
        rescaled = (bbnd > 1) & (u22abs < 1) & (bbnd >= bignum * u22abs)
        scale = torch.where(rescaled, 1 / bbnd, 1)
        br1, bi1, br2, bi2 = (torch.where(rescaled, scale * v, v) for v in (br1, bi1, br2, bi2))
        xr2, xi2 = _divided(br2, bi2, ur22, ui22)
        xr1 = ((ur11r * br1 - ui11r * bi1) - ur12s * xr2) + ui12s * xi2
        xi1 = ((ui11r * br1 + ur11r * bi1) - ui12s * xr2) - ur12s * xi2
        x = [torch.where(zswap, xr2, xr1), torch.where(zswap, xr1, xr2), torch.where(zswap, xi2, xi1),
             torch.where(zswap, xi1, xi2)]
        xnorm = bigger(xr1.abs() + xi1.abs(), xr2.abs() + xi2.abs())
    further = (xnorm > 1) & (cmax > 1) & (xnorm > bignum / cmax)
    temp = cmax / bignum
    x = [torch.where(further, temp * v, v) for v in x]
    xnorm, scale = torch.where(further, temp * xnorm, xnorm), torch.where(further, temp * scale, scale)
    return (tuple(torch.where(identity, p, v) for p, v in zip(plain[0], x)), torch.where(identity, plain[1], scale),
            torch.where(identity, plain[2], xnorm))


# Reference-LAPACK 3.12.1 SRC/dtrevc3.f 383-398 and 406-842 (SIDE = 'R', HOWMNY = 'B') on the real quasi-triangular
# Schur form t with LWORK `lwork`. IP by 423-438 for KI = N..1: the KI after a pair's second is its first (IP = 1,
# skipped), else KI = 1 (`sub`'s leading zero) or T(KI,KI-1) = 0 real (IP = 0), else the pair's second (IP = -1, its
# WORK(1+(IV-1)N:) in xr and WORK(1+IV N:) in xi); WR, WI and SMIN (452-457), the real right-hand side (464-470) and
# the pair's initial solve (603-618). All the solves run together in rounds J = N-1..1, each column by its own DO 60
# (DO 90) JNXT recurrence from KI-1 (KI-2) (475-487, 623-635): J > JNXT skipped, else the 1 x 1 block at J or, where
# T(J,J-1) /= 0, the 2 x 2 block ending at J: DLALN2, the BIGNUM / XNORM rescaling of X and SCALE (a pair's 2 x 2 by
# REC = 1 / XNORM), DSCAL of WORK(1:KI) by SCALE, X into WORK and the DAXPYs above the block. NB = MIN((LWORK - N) /
# (2 N), NBMAX = 128) (373-379; dgeev.f's LWORK - N >= 129 N >= N + 2 N NBMIN, so 378's NB = 1, version 1, is
# unreached and NB >= 64). Version 2 (584-593, 775-787, 790-837): IV from NB (416-419), a pair at IV - 1 and IV, KI2
# = KI - 1 (else KI), and where IV <= 2 or KI2 = 1 the DGEMM of VR(1:N,1:KI2+NB-IV) by the block's columns IV..NB and
# their normalization (810-829: a real column by its largest |.|, a pair's two by EMAX of |VR(K,KI-1)| + |VR(K,KI)|).
# The per-member blocks (at most NB columns, at least NB - 1 before the last) are each static block b's superset of
# columns, Z zero past the member's own KI2 + NB - IV (0 times X's zero rows there)
def _dtrevc(t, z, lwork):
    n, k = t.shape[-1], _counted(t)
    limits = torch.finfo(t.dtype)
    ulp = limits.eps
    smlnum = limits.tiny * (n / ulp)
    bignum = (1 - ulp) / smlnum
    pad = torch.zeros_like(t[..., 0, :1])
    sub = torch.cat([pad, t.diagonal(-1, dim1=-2, dim2=-1)], -1)
    sup = torch.cat([pad, t.diagonal(1, dim1=-2, dim2=-1)], -1)
    codes, previous = [None] * n, torch.zeros_like(sub[..., 0], dtype=torch.int64)
    for ki in range(n, 0, -1):
        previous = torch.where(previous == -1, 1, torch.where(sub[..., ki - 1] != 0, -1, 0))
        codes[ki - 1] = previous
    ip = torch.stack(codes, -1)
    second, first, isreal = ip == -1, ip == 1, ip == 0
    wr = t.diagonal(dim1=-2, dim2=-1)
    wi = torch.where(second, sub.abs().sqrt() * sup.abs().sqrt(), 0)
    size = ulp * (wr.abs() + wi.abs())
    smin = torch.where(size >= smlnum, size, smlnum)
    wide = sup.abs() >= sub.abs()
    s1, s2 = torch.where(wide, 1, -wi / sub), torch.where(wide, wi / sup, 1)
    cnorm = torch.where(k[:, None] < k, t.abs(), 0).sum(-2)
    row, col = k[:, None], k
    tcol = t
    xr = torch.where(isreal[..., None, :], torch.where(row < col, -t, torch.where(row == col, 1, 0)), 0)
    prev = torch.cat([torch.zeros_like(t[..., :, :1]), t[..., :, :-1]], -1)
    xr = torch.where(second[..., None, :], torch.where(row < col - 1, -(s1[..., None, :] * prev), torch.where(
        row == col - 1, s1[..., None, :], 0)), xr)
    xi = torch.where(second[..., None, :], torch.where(row < col - 1, -(s2[..., None, :] * tcol), torch.where(
        row == col, s2[..., None, :], 0)), torch.zeros_like(t))
    jnxt = torch.where(second, k - 2, k - 1)
    for j in range(n - 1, 0, -1):
        two = (t[..., j - 1, j - 2] != 0) if j >= 2 else torch.zeros_like(second[..., 0])
        active = (isreal | second) & (j <= jnxt)
        single, double = active & _vec(~two), active & _vec(two)
        jnxt = torch.where(active, torch.where(_vec(two), j - 2, j - 1), jnxt)
        rows = [j - 1] if j < 2 else [j - 2, j - 1]
        made = {}
        for na in (1, 2) if j >= 2 else (1,):
            at = rows[-na:]
            given = [xr[..., r, :] for r in at] + [xi[..., r, :] for r in at]
            a = [_vec(t[..., p, q]) for p in at for q in at]
            (real_x, real_s, real_n) = _dlaln2(na, 1, smin, a, given[:na], wr, torch.zeros_like(wr))
            (pair_x, pair_s, pair_n) = _dlaln2(na, 2, smin, a, given, wr, wi)
            x = [torch.where(second, p, q) for p, q in zip(pair_x, list(real_x) + given[na:])]
            scale = torch.where(second, pair_s, real_s)
            xnorm = torch.where(second, pair_n, real_n)
            beta = _vec(cnorm[..., j - 1]) if na == 1 else _vec(torch.where(
                cnorm[..., j - 2] >= cnorm[..., j - 1], cnorm[..., j - 2], cnorm[..., j - 1]))
            over = (xnorm > 1) & (beta > bignum / xnorm)
            if na == 1:
                x, scale = [torch.where(over, e / xnorm, e) for e in x], torch.where(over, scale / xnorm, scale)
            else:
                rec = 1 / xnorm
                x = [torch.where(over, torch.where(second, e * rec, e / xnorm), e) for e in x]
                scale = torch.where(over, torch.where(second, scale * rec, scale / xnorm), scale)
            made[na] = (at, x, scale)
        factor = torch.where(single, made[1][2], 1)
        if j >= 2:
            factor = torch.where(double, made[2][2], factor)
        taking = (single | (double if j >= 2 else single)) & (factor != 1)
        scaled = taking[..., None, :] & (row <= col)
        xr, xi = torch.where(scaled, xr * factor[..., None, :], xr), torch.where(scaled, xi * factor[..., None, :], xi)
        for na, used in ((1, single), (2, double if j >= 2 else None)):
            if used is None:
                continue
            at, x, _ = made[na]
            for p, r in enumerate(at):
                xr = _row_put(xr, r, torch.where(used, x[p], xr[..., r, :]))
                xi = _row_put(xi, r, torch.where(used, x[na + p], xi[..., r, :]))
            above = used[..., None, :] & (row <= at[0])
            for p, r in enumerate(at):
                xr = torch.where(above, xr + (-x[p])[..., None, :] * t[..., :, r][..., :, None], xr)
            for p, r in enumerate(at):
                xi = torch.where(above, xi + (-x[na + p])[..., None, :] * t[..., :, r][..., :, None], xi)
    packed = torch.where(first[..., None, :], torch.cat([xr[..., :, 1:], torch.zeros_like(xr[..., :, :1])], -1),
                         torch.where(second[..., None, :], xi, xr))
    vr = torch.matmul(z, packed)
    after = torch.cat([vr[..., :, 1:], torch.zeros_like(vr[..., :, :1])], -1)
    before = torch.cat([torch.zeros_like(vr[..., :, :1]), vr[..., :, :-1]], -1)
    partner = torch.where(first[..., None, :], after, before)
    emax = torch.where((first | second)[..., None, :], vr.abs() + partner.abs(), vr.abs()).amax(-2)
    return vr * (1 / emax)[..., None, :]


# Fortran MAX(A, B) and MIN(A, B) as the source's comparisons select them
def _larger(a, b):
    return torch.where(a >= b, a, b)


def _smaller(a, b):
    return torch.where(a <= b, a, b)


# CABS2(Z) = |DBLE(Z) / 2| + |DIMAG(Z) / 2| (zlatrs.f's statement function)
def _cabs2(x):
    return (x.real / 2).abs() + (x.imag / 2).abs()


# Reference-LAPACK SRC/ztrevc.f 323-409 (ztrevc3.f's SIDE = 'R', HOWMNY = 'B' right vectors, its blocked
# back-transformation VR = Z X) with zlatrs.f 324-1025 (UPLO = 'U', TRANS = 'N', DIAG = 'N', NORMIN = 'Y', N = KI-1)
# for each KI = N..1 of the upper triangular Schur form t. Column KI of X solves (T(1:KI-1,1:KI-1) - T(KI,KI)) X =
# SCALE (-T(1:KI-1,KI)), the shifted diagonal T(K,K) - T(KI,KI) raised to SMIN where its CABS1 is below it (ztrevc
# 366-370, the original diagonal restored after each KI). RWORK (CNORM, DZASUM of each column above the diagonal) is
# ZLATRS's in-out operand: each KI's call scales CNORM(1:KI-1) by TSCAL (case 1; case 2 recomputing an unrepresentable
# norm as the sum of (2 TSCAL) CABS2 terms) and returns it by DSCAL(N, 1 / TSCAL, CNORM) where TSCAL /= 1 (zlatrs
# 1021-1023; not after case 2's ZTRSV return), so the next KI reads the norms the previous calls left: the static scan
# KI = N..2 below. GROW and XBND (zlatrs 467-504) and the solves (ZTRSV, ztrsv.f 159-170, where GROW TSCAL > SMLNUM or
# case 2's return; else 606-743's careful solve with its REC scalings, A(J,J) = 0's null vector and ZAXPY, zaxpy.f's
# DCABS1(ZA) = 0 return) run for every KI together in the source's row order, round J = N-1..1 for each KI > J.
# X(KI,KI) = SCALE (the careful solve's SCALE / TSCAL, ZTRSV's 1; KI = 1: WORK(1) = 1); each column of Z X by 1 / its
# largest CABS1 (IZAMAX, ZDSCAL)
def _ztrevc(t, z):
    n, k = t.shape[-1], _counted(t)
    limits = torch.finfo(t.real.dtype)
    unfl, ulp, ovfl = limits.tiny, limits.eps, limits.max
    smlnum_t, smlnum = unfl * (n / ulp), unfl / ulp
    bignum = 1 / smlnum
    row, col = k[:, None], k
    upper = row < col
    diagonal = t.diagonal(dim1=-2, dim2=-1)
    smin = _larger(ulp * _abs1(diagonal), torch.full_like(diagonal.real, smlnum_t))
    shifted = diagonal[..., :, None] - diagonal[..., None, :]
    shifted = torch.where(_abs1(shifted) < smin[..., None, :], smin[..., None, :].to(t.dtype), shifted)
    cabs2 = _cabs2(t)
    colmax = torch.where(upper, _larger(t.real.abs(), t.imag.abs()), 0).amax(-2)

    # zlatrs 358-431 for KI = N..2 over the CNORM the previous KI returned
    current = torch.where(upper, _abs1(t), 0).sum(-2)
    seen, scales = [current] * n, [torch.ones_like(current[..., 0])] * n
    returned = [torch.zeros_like(current[..., 0], dtype=torch.bool)] * n
    for ki in range(n, 1, -1):
        within = k <= ki - 1
        tmax = torch.where(within, current, 0).amax(-1)
        tmax2 = torch.where(within, colmax, 0).amax(-1)
        case1 = ~(tmax <= bignum * 0.5) & (tmax <= ovfl)
        rest = ~(tmax <= bignum * 0.5) & ~case1
        case2, case3 = rest & (tmax2 <= ovfl), rest & ~(tmax2 <= ovfl)
        tscal = torch.where(case1, 0.5 / (smlnum * tmax), torch.where(case2, 1 / (smlnum * tmax2), torch.ones_like(
            tmax)))
        recomputed = torch.where(upper, _mat(2 * tscal) * cabs2, 0).sum(-2)
        read = torch.where(_vec(case1), current * _vec(tscal), torch.where(_vec(case2), torch.where(
            current <= ovfl, current * _vec(tscal), recomputed), current))
        seen[ki - 1], scales[ki - 1], returned[ki - 1] = read, tscal, case3
        current = torch.where(within & _vec((case1 | case2) & (tscal != 1)), read * _vec(1 / tscal), current)
    cn, tscal, fallback = torch.stack(seen, -1), torch.stack(scales, -1), torch.stack(returned, -1)

    # zlatrs 436-504: XMAX = XBND the largest CABS2 of X = -T(1:KI-1,KI); where TSCAL = 1, GROW = HALF / MAX(XBND,
    # SMLNUM) and XBND = GROW, then each J = KI-1..1: GROW <= SMLNUM leaves the loop (GO TO 60, GROW kept), XBND =
    # MIN(XBND, MIN(ONE, TJJ) GROW) (ZERO where TJJ < SMLNUM), GROW = GROW (TJJ / (TJJ + CNORM(J))) (ZERO where
    # TJJ + CNORM(J) < SMLNUM); a completed loop GROW = XBND; TSCAL /= 1 GROW = ZERO
    rhs = torch.where(upper, -t, 0)
    xmax = torch.where(upper, _cabs2(rhs), 0).amax(-2)
    grow = 0.5 / _larger(xmax, torch.full_like(xmax, smlnum))
    xbnd, stopped = grow, torch.zeros_like(fallback)
    for j in range(n - 1, 0, -1):
        live = (k >= j + 1) & (tscal == 1) & ~stopped
        stop = live & (grow <= smlnum)
        step = live & ~stop
        tjj = _abs1(shifted[..., j - 1, :])
        bounded = torch.where(tjj >= smlnum, _smaller(xbnd, _smaller(torch.ones_like(tjj), tjj) * grow), 0)
        total = tjj + cn[..., j - 1, :]
        grow, xbnd = torch.where(step, torch.where(total >= smlnum, grow * (tjj / total), 0), grow), torch.where(
            step, bounded, xbnd)
        stopped = stopped | stop
    grow = torch.where(tscal != 1, 0, torch.where(stopped, grow, xbnd))

    # zlatrs 606-617: ZTRSV where GROW TSCAL > SMLNUM (or case 2's return), SCALE = 1; else SCALE = (BIGNUM HALF) /
    # XMAX, X by it and XMAX = BIGNUM where XMAX > BIGNUM HALF, else XMAX = 2 XMAX
    plain = fallback | (grow * tscal > smlnum)
    big = ~plain & (xmax > bignum * 0.5)
    scale = torch.where(big, (bignum * 0.5) / xmax, torch.ones_like(xmax))
    xmax = torch.where(plain, xmax, torch.where(big, bignum, xmax * 2))
    x = _scaled(rhs, scale[..., None, :])
    for j in range(n - 1, 0, -1):
        part = k >= j + 1
        careful = part & ~plain
        xj, dj, norm = x[..., j - 1, :], shifted[..., j - 1, :], cn[..., j - 1, :]
        present = xj != 0
        direct = torch.where(present, _cdiv(xj, dj), xj)
        xa = _abs1(xj)
        tjjs = _scaled(dj, tscal)
        tjj = _abs1(tjjs)
        large = tjj > smlnum
        tiny = ~large & (tjj > 0)
        singular = ~large & ~tiny
        over = xa > tjj * bignum
        held = (tjj * bignum) / xa
        rec1 = torch.where(large & (tjj < 1) & over, 1 / xa, torch.where(tiny & over, torch.where(
            norm > 1, held / norm, held), torch.ones_like(xa)))
        xn = torch.where(singular, 1, _cdiv(_scaled(xj, rec1), tjjs))
        xjn = torch.where(singular, 1, _abs1(xn))
        s1, t1 = torch.where(singular, 0, scale * rec1), torch.where(singular, 0, xmax * rec1)
        room, recip = bignum - t1, 1 / xjn
        rec2 = torch.where((xjn > 1) & (norm > room * recip), recip * 0.5, torch.where(
            ~(xjn > 1) & (xjn * norm > room), 0.5, torch.ones_like(xjn)))
        xf = _scaled(xn, rec2)
        rows = careful[..., None, :] & upper
        x = torch.where(rows, torch.where(singular[..., None, :], 0, _scaled(_scaled(x, rec1[..., None, :]), rec2[
            ..., None, :])), x)
        x = _row_put(x, j - 1, torch.where(careful, xf, torch.where(part, direct, xj)))
        coefficient = torch.where(careful, -_scaled(xf, tscal), -direct)
        updating = part & torch.where(careful, _abs1(coefficient) != 0, present)
        x = torch.where(updating[..., None, :] & (row <= j - 1), x + coefficient[..., None, :] * t[..., :, j - 1][
            ..., :, None], x)
        latest = torch.where(row <= j - 1, _abs1(x), 0).amax(-2) if j > 1 else t1
        xmax, scale = torch.where(careful, latest, xmax), torch.where(careful, s1 * rec2, scale)
    finished = torch.where((k == 1) | plain, torch.ones_like(scale), scale / tscal)
    x = torch.where(row == col, finished[..., None, :].to(x.dtype), x)
    vr = torch.matmul(z, x)
    return _scaled(vr, (1 / _abs1(vr).amax(-2))[..., None, :])

# zgebak.f / dgebak.f 205-261 (JOB = 'B', SIDE = 'R', M = N): rows ILO..IHI by SCALE(I) where ILO /= IHI (ZDSCAL,
# each component), then for II = 1..N outside ILO..IHI (I = ILO - II below ILO) rows I and K = SCALE(I) (`perm`)
# interchanged where K /= I, in that order
def _gebak(v, scale, perm, ilo, ihi):
    n, k = v.shape[-1], _counted(v)
    v = torch.where((_span(k, ilo, ihi) & _vec(ilo != ihi))[..., :, None], _scaled(v, scale[..., :, None]), v)
    every = torch.ones(n, dtype=torch.bool, device=v.device)
    for ii in range(1, n + 1):
        i = torch.where(ii < ilo, ilo - ii, torch.full_like(ilo, ii))
        target = _at(perm, i)
        v = _rows_swapped(v, i, target, every, ((ii < ilo) | (ii > ihi)) & (target != i))
    return v


# column j's DZNRM2 / DNRM2 (_norm2 over the column)
def _column_norms(v):
    return _norm2(v.mT, torch.ones(v.shape[-2], dtype=torch.bool, device=v.device))


# zgeev.f 492-503: each column by SCL = 1 / DZNRM2 (ZDSCAL), then RWORK(K) = DBLE(VR(K,I))**2 + AIMAG(VR(K,I))**2, K
# = IDAMAX (the first largest), TMP = CONJG(VR(K,I)) / SQRT(RWORK(K)) (each component), the column by TMP (ZSCAL)
# and VR(K,I) = DCMPLX(DBLE(VR(K,I)), ZERO)
def _zgeev_normalized(v):
    k = _counted(v)
    v = _scaled(v, (1 / _column_norms(v))[..., None, :])
    squares = v.real * v.real + v.imag * v.imag
    at = squares.argmax(-2, keepdim=True)
    tmp = _over(v.gather(-2, at).conj(), squares.gather(-2, at).sqrt())
    v = tmp * v
    return torch.where(k[:, None] == at + 1, torch.complex(v.real, torch.zeros_like(v.real)), v)


# Reference-LAPACK SRC/dlartg.f90 (E. Anderson, Algorithm 978: Safe Scaling in the Level 1 BLAS, ACM TOMS 44(1),
# 2017; not among this directory's references): C, S with SAFMIN = DLAMCH('S') (radix**max(minexponent - 1,
# 1 - maxexponent)), SAFMAX = 1 / SAFMIN, RTMIN = SQRT(SAFMIN), RTMAX = SQRT(SAFMAX / 2); G = 0: (1, 0); F = 0:
# (0, SIGN(1, G)); F1 and G1 within (RTMIN, RTMAX): D = SQRT(F F + G G), C = F1 / D, R = SIGN(D, F), S = G / R; else
# by U = MIN(SAFMAX, MAX(SAFMIN, F1, G1)): FS = F / U, GS = G / U, D = SQRT(FS FS + GS GS), C = |FS| / D,
# S = GS / SIGN(D, F)
def _dlartg(f, g):
    limits = torch.finfo(f.dtype)
    safmin = limits.tiny
    safmax = 1 / safmin
    rtmin, rtmax = math.sqrt(safmin), math.sqrt(safmax / 2)
    f1, g1 = f.abs(), g.abs()
    plain = (f1 > rtmin) & (f1 < rtmax) & (g1 > rtmin) & (g1 < rtmax)
    d = (f * f + g * g).sqrt()
    u = _smaller(torch.full_like(f1, safmax), _larger(_larger(torch.full_like(f1, safmin), f1), g1))
    fs, gs = f / u, g / u
    du = (fs * fs + gs * gs).sqrt()
    c = torch.where(plain, f1 / d, fs.abs() / du)
    s = torch.where(plain, g / _fsign(d, f), gs / _fsign(du, f))
    one = torch.ones_like(f)
    c = torch.where(g == 0, one, torch.where(f == 0, 0, c))
    return c, torch.where(g == 0, 0, torch.where(f == 0, _fsign(one, g), s))


# dgeev.f 505-524: WI(I) = 0, the column by SCL = 1 / DNRM2; WI(I) > 0, columns I and I+1 by SCL = 1 / DLAPY2(their
# DNRM2s), K = IDAMAX(VR(K,I)**2 + VR(K,I+1)**2), DLARTG(VR(K,I), VR(K,I+1)) and DROT (DTEMP = C X + S Y,
# Y = C Y - S X) of the two columns, VR(K,I+1) = ZERO
def _dgeev_normalized(v, wi):
    k = _counted(v)
    zero, flag = torch.zeros_like(v[..., :, :1]), torch.zeros_like(wi[..., :1], dtype=torch.bool)
    norms = _column_norms(v)
    paired = 1 / _dlapy2(norms, torch.cat([norms[..., 1:], torch.zeros_like(norms[..., :1])], -1))
    first = wi > 0
    tail = torch.cat([flag, first[..., :-1]], -1)
    factor = torch.where(first, paired, torch.where(wi == 0, 1 / norms, torch.ones_like(norms)))
    factor = torch.where(tail, torch.cat([torch.ones_like(paired[..., :1]), paired[..., :-1]], -1), factor)
    v = v * factor[..., None, :]
    after, before = torch.cat([v[..., :, 1:], zero], -1), torch.cat([zero, v[..., :, :-1]], -1)
    squares = v * v + after * after
    at = squares.argmax(-2, keepdim=True)
    c, s = _dlartg(v.gather(-2, at)[..., 0, :], after.gather(-2, at)[..., 0, :])
    pc = torch.cat([torch.ones_like(c[..., :1]), c[..., :-1]], -1)
    ps = torch.cat([torch.zeros_like(s[..., :1]), s[..., :-1]], -1)
    placed = torch.cat([torch.zeros_like(at[..., :1]), at[..., :-1]], -1) + 1
    rotated = torch.where(first[..., None, :], c[..., None, :] * v + s[..., None, :] * after, torch.where(
        tail[..., None, :], pc[..., None, :] * v - ps[..., None, :] * before, v))
    return torch.where(tail[..., None, :] & (k[:, None] == placed), 0, rotated)


# Reference-LAPACK SRC/zlaqr5.f and dlaqr5.f (with zlaqr1.f and dlaqr1.f) in the batch convention above, for
# xLAQR0's sweep: WANTT and WANTZ static; N, KTOP, KBOT, NSHFTS, ILOZ, IHIZ, KACC22 = 2 (`accum`, iparmq.f ISPEC
# 16: NS >= 14) and the workspace xLAQR0 places in H (U = H(KU,1), WV = H(KWV,1), WH = H(KU,KWH), NV, NH) each
# member's own. The shifts are a static-length tuple, (S,) complex or (SR, SI) real, the member's first NSHFTS used;
# V is 3 x NBMPS and U is KDU x KDU within their static bounds. xLAQR5 reaches H(KU,1) only as its argument U: held
# as its own array through ZLASET and DO 145 and put into H(KU,1) before DO 150; DO 150, 160 and 170 then write each
# chunk into H's own WH and WV positions and copy it back from there. DO 180 and DO 145 are nested while_loop
# passes, and within DO 145's pass DO 80, DO 100 and DO 120/140 are three while_loops in the source's order, each its
# M = MBOT, MTOP, -1. Real only: dlaqr5.f's DO 10 pairing of the shifts, ABS for
# CABS1, no conjugates, and DO 50/60 as REFSUM = X(J,K+1) + V2 X(J,K+2) times T1, T2; complex DO 50/60 is
# REFSUM = V1 (X(J,K+1) + V2 X(J,K+2)), X(J,K+2) -= REFSUM CONJG(V2)

def _cj(x):
    return x.conj() if x.is_complex() else x


# x at the 0-based positions `rows` (*batch, R) by `cols` (*batch, C), each clamped into x
def _taken(x, rows, cols):
    rows, cols = rows.clamp(0, x.shape[-2] - 1), cols.clamp(0, x.shape[-1] - 1)
    x = x.gather(-2, rows[..., :, None].expand(*rows.shape, x.shape[-1]))
    return x.gather(-1, cols[..., None, :].expand(*cols.shape[:-1], rows.shape[-1], cols.shape[-1]))


# the static ROWS x COLS block of x at 1-based (R0, C0) (_region_at), and x with `block` placed there where `mask`
# holds (_region_put; block positions past x unplaced)
def _region_at(x, r0, c0, rows, cols):
    return _taken(x, (r0 - 1)[..., None] + torch.arange(rows, device=x.device),
                  (c0 - 1)[..., None] + torch.arange(cols, device=x.device))


def _region_put(x, r0, c0, block, mask):
    rows, cols = block.shape[-2:]
    down = torch.arange(1, x.shape[-2] + 1, device=x.device) - _vec(r0)
    across = torch.arange(1, x.shape[-1] + 1, device=x.device) - _vec(c0)
    hit = ((down >= 0) & (down < rows))[..., :, None] & ((across >= 0) & (across < cols))[..., None, :]
    down, across = down.clamp(0, rows - 1), across.clamp(0, cols - 1)

    def spread(b):
        b = b.expand(*x.shape[:-2], rows, cols).gather(-2, down[..., :, None].expand(*down.shape, cols))
        return b.gather(-1, across[..., None, :].expand(*x.shape))
    return torch.where(hit & spread(mask), spread(block.to(x.dtype)), x)


# one pass, for J, of a chunked DO J = FIRST, LAST, CHUNK with JLEN = MIN(CHUNK, LAST - J + 1), as zlaqr3.f DO 60
# and DO 80 and zlaqr5.f DO 160 and DO 170 write it (dlaqr3.f and dlaqr5.f with DGEMM and DLACPY): xGEMM('N', 'N',
# JLEN, NU, NU, ONE, X(J,C0), M, ZERO, W) into the workspace W = H(WR,1), then xLACPY('ALL', JLEN, NU, W, X(J,C0)).
# X is h, or z where `onz`; M is static SIZE x SIZE, its leading NU x NU the source's
def _rows_chunk(onz, live, h, z, j, last, chunk, c0, nu, m, wr):
    x = z if onz else h
    rows, size, one = x.shape[-2], m.shape[-1], torch.ones_like(j)
    used = _within(torch.minimum(chunk, last - j + 1), rows)[..., :, None] & _within(nu, size)[..., None, :]
    made = torch.matmul(torch.where(used, _region_at(x, j, c0, rows, size), 0),
                        torch.where(_square_within(nu, size), m, 0))
    used = used & _mat(live)
    h = _region_put(h, wr, one, made, used)
    copied = _region_at(h, wr, one, rows, size)
    if onz:
        z = _region_put(z, j, c0, copied, used)
    else:
        h = _region_put(h, j, c0, copied, used)
    j = torch.where(live, j + chunk, j)
    return live & (j <= last), h, z if onz else z.clone(), j


# the same pass for zlaqr3.f DO 70 and zlaqr5.f DO 150: xGEMM('C', 'N', NU, JLEN, NU, ONE, M, H(R0,J), ZERO, W) into
# the workspace W = H(WR,WC), then xLACPY('ALL', NU, JLEN, W, H(R0,J))
def _columns_chunk(live, h, j, last, chunk, r0, nu, m, wr, wc):
    cols, size = h.shape[-1], m.shape[-1]
    used = _within(nu, size)[..., :, None] & _within(torch.minimum(chunk, last - j + 1), cols)[..., None, :]
    made = torch.matmul(torch.where(_square_within(nu, size), m, 0).mH,
                        torch.where(used, _region_at(h, r0, j, size, cols), 0))
    used = used & _mat(live)
    h = _region_put(h, wr, wc, made, used)
    h = _region_put(h, r0, j, _region_at(h, wr, wc, size, cols), used)
    j = torch.where(live, j + chunk, j)
    return live & (j <= last), h, j


# the reflector loops: REFSUM = X(J,C) + C1 X(J,C+1) [+ C2 X(J,C+2)], X(J,C+I) -= REFSUM T(I) over the rows `rows`
# (_columns_reflected), and the same on rows R.. over the columns `cols` (_rows_reflected)
def _columns_reflected(x, c, coefficients, ts, rows):
    one = torch.ones_like(c)
    block = _region_at(x, one, c, x.shape[-2], len(ts))
    parts = [block[..., i] for i in range(len(ts))]
    refsum = parts[0]
    for coefficient, part in zip(coefficients, parts[1:]):
        refsum = refsum + _vec(coefficient) * part
    made = torch.stack([part - refsum * _vec(t) for part, t in zip(parts, ts)], -1)
    return _region_put(x, one, c, made, rows[..., :, None].expand(*rows.shape, len(ts)))


def _rows_reflected(x, r, coefficients, ts, cols):
    one = torch.ones_like(r)
    block = _region_at(x, r, one, len(ts), x.shape[-1])
    parts = [block[..., i, :] for i in range(len(ts))]
    refsum = parts[0]
    for coefficient, part in zip(coefficients, parts[1:]):
        refsum = refsum + _vec(coefficient) * part
    made = torch.stack([part - refsum * _vec(t) for part, t in zip(parts, ts)], -2)
    return _region_put(x, r, one, made, cols[..., None, :].expand(*cols.shape[:-1], len(ts), cols.shape[-1]))


# DO 50 and DO 60, the 2-reflector V1, V2 on columns C, C+1 of x over the rows `rows`
def _pair_reflected(x, c, v1, v2, rows):
    if not x.is_complex():
        return _columns_reflected(x, c, (v2,), (v1, v1 * v2), rows)
    one = torch.ones_like(c)
    block = _region_at(x, one, c, x.shape[-2], 2)
    first, second = block[..., 0], block[..., 1]
    refsum = _vec(v1) * (first + _vec(v2) * second)
    made = torch.stack([first - refsum, second - refsum * _vec(v2.conj())], -1)
    return _region_put(x, one, c, made, rows[..., :, None].expand(*rows.shape, 2))


# V(1..3, M) of the 3 x NBMPS reflectors, and V(:, M) written where `when`
def _reflector_at(vs, m):
    return [_at(vs[..., i, :], m) for i in range(3)]


def _reflector_put(vs, m, values, when):
    hit = (_counted(vs) == _vec(m)) & _vec(when)
    return torch.where(hit[..., None, :], torch.stack(values, -1)[..., :, None], vs)


# S(2M-1) and S(2M) (complex), or (SR(2M-1), SI(2M-1)) and (SR(2M), SI(2M)), as _laqr1's shift operands
def _shift_pair(shifts, m):
    return tuple(tuple(_at(s, 2 * m - lag) for s in shifts) for lag in (1, 0))


# dlaqr5.f DO 10: I = 1, NSHFTS - 2, 2 in order, SR(I..I+2) and SI(I..I+2) rotated where SI(I) /= -SI(I+1)
def _paired_shifts(sr, si, nshfts, go):
    for i in range(1, sr.shape[-1] - 1, 2):
        swap = go & (i <= nshfts - 2) & (si[..., i - 1] != -si[..., i])
        made = []
        for x in (sr, si):
            a, b, c = x[..., i - 1], x[..., i], x[..., i + 1]
            made.append(_at_put(_at_put(_at_put(x, i, b, swap), i + 1, c, swap), i + 2, a, swap))
        sr, si = made
    return sr, si


# Reference-LAPACK SRC/zlaqr1.f (shifts S1, S2) and dlaqr1.f (SR1 + i SI1, SR2 + i SI2): V, the first column of
# (H - S1 I)(H - S2 I) of the ORDER (2 or 3, static) leading block scaled by S, zero where S = 0; `e(i, j)` that
# block's entries; H(2,1) / S and H(3,1) / S by parts (Fortran's complex by real)
def _laqr1(e, order, first, second):
    h11, h21 = e(1, 1), e(2, 1)
    whole = h11.is_complex()
    if whole:
        (s1,), (s2,) = first, second
        s = _abs1(h11 - s2) + _abs1(h21)
    else:
        (s1, si1), (s2, si2) = first, second
        s = (h11 - s2).abs() + si2.abs() + h21.abs()
    if order == 3:
        s = s + _abs1(e(3, 1))
    vanished = s == 0
    safe = torch.where(vanished, 1, s)
    h21s = _over(h21, safe)
    lead = (h11 - s1) * _over(h11 - s2, safe)
    trace = h11 + e(2, 2) - s1 - s2
    if order == 2:
        first_entry = h21s * e(1, 2) + lead
        if not whole:
            first_entry = first_entry - si1 * (si2 / safe)
        made = [first_entry, h21s * trace]
    else:
        h31s = _over(e(3, 1), safe)
        if not whole:
            lead = lead - si1 * (si2 / safe)
        made = [lead + e(1, 2) * h21s + e(1, 3) * h31s, h21s * trace + e(2, 3) * h31s,
                h31s * (h11 + e(3, 3) - s1 - s2) + h21s * e(3, 2)]
    return [torch.where(vanished, 0, value) for value in made]


# zlaqr5.f / dlaqr5.f's vigilant deflation of H(K+1,K) where `when` (K >= KTOP): TST1's TST1 = 0 additions in order,
# H(K+1,K) = 0 where CABS1(H(K+1,K)) <= MAX(SMLNUM, ULP TST1) and TST2 = 0 or H21 (H12 / SCL) <= MAX(SMLNUM, ULP TST2)
def _vigilant(h, k, ktop, kbot, smlnum, ulp, when):
    sub, corner, hkk, next_ = _entry(h, k + 1, k), _entry(h, k, k + 1), _entry(h, k, k), _entry(h, k + 1, k + 1)
    tst1 = _abs1(hkk) + _abs1(next_)
    vanished = tst1 == 0
    for row, col, within in ((k, k - 1, k >= ktop + 1), (k, k - 2, k >= ktop + 2), (k, k - 3, k >= ktop + 3),
                             (k + 2, k + 1, k <= kbot - 2), (k + 3, k + 1, k <= kbot - 3),
                             (k + 4, k + 1, k <= kbot - 4)):
        tst1 = torch.where(vanished & within, tst1 + _abs1(_entry(h, row, col)), tst1)
    a, b = _abs1(sub), _abs1(corner)
    h12, h21 = _larger(a, b), _smaller(a, b)
    c, d = _abs1(next_), _abs1(hkk - next_)
    h11, h22 = _larger(c, d), _smaller(c, d)
    scl = h11 + h12
    tst2 = h22 * (h11 / scl)
    near = when & (sub != 0) & (a <= _larger(smlnum, ulp * tst1))
    return _entry_put(h, k + 1, k, 0, near & ((tst2 == 0) | (h21 * (h12 / scl) <= _larger(smlnum, ulp * tst2))))


# xLAQR5(WANTT, WANTZ, KACC22, N, KTOP, KBOT, NSHFTS, S, H, ILOZ, IHIZ, Z, V, U, NV, WV, NH, WH) where `proceed`, N
# the member's ORDER (xLAQR4 within xLAQR3 passes JW): returns H, Z (where WANTZ) and the shifts (dlaqr5.f's DO 10
# pairs SR and SI in place)
def _laqr5(wantt, wantz, accum, order, ktop, kbot, nshfts, shifts, h, iloz, ihiz, z, ku, nv, kwv, nh, kwh, proceed):
    limits = torch.finfo(h.real.dtype)
    ulp = limits.eps
    smlnum = limits.tiny * (order.to(h.real.dtype) / ulp)
    go = proceed & (nshfts >= 2) & (ktop < kbot)
    if not h.is_complex():
        shifts = _paired_shifts(*shifts, nshfts, go)
    ns = nshfts - nshfts % 2
    h = _entry_put(h, ktop + 2, ktop, 0, go & (ktop + 2 <= kbot))
    nbmps = ns // 2
    vs = h.new_zeros((*h.shape[:-2], 3, max(1, shifts[0].shape[-1] // 2)))
    incol = ktop - 2 * nbmps + 1
    carried = (go & (incol <= kbot - 2), h.contiguous(), z.contiguous() if wantz else h.new_zeros(()), vs, incol)
    _, h, z, _, _ = while_loop_op(_first_any, functools.partial(_laqr5_columns, wantt, wantz, ulp), carried,
                                  (accum, order, ktop, kbot, nbmps, iloz, ihiz, ku, nv, kwv, nh, kwh, smlnum,
                                   *(s.contiguous() for s in shifts)))
    return h, z, shifts


# DO 180's pass for INCOL: JTOP, NDCOL, U = I (ZLASET, where ACCUM), DO 145, then where ACCUM U into H(KU,1) and
# zlaqr5.f 765-810's DO 150 (_columns_chunk: U(K1,K1)^H H(INCOL+K1,JCOL) into WH, then back), DO 160 and, where
# WANTZ, DO 170 (_rows_chunk: H(JROW,INCOL+K1) and Z(JROW,INCOL+K1) by U(K1,K1) into WV, then back), each chunk by
# chunk in the source's order
def _laqr5_columns(wantt, wantz, ulp, live, h, z, vs, incol, accum, order, ktop, kbot, nbmps, iloz, ihiz, ku, nv,
                   kwv, nh, kwh, smlnum, *shifts):
    ub = 4 * vs.shape[-1]
    kdu = 4 * nbmps
    one = torch.ones_like(ktop)
    jtop = torch.where(accum, torch.maximum(ktop, incol), one if wantt else ktop)
    ndcol = incol + kdu
    u = torch.where(_square_within(kdu, ub), torch.eye(ub, dtype=h.dtype, device=h.device), 0)
    stop = torch.minimum(incol + 2 * nbmps - 1, kbot - 2)
    carried = (live & (incol <= stop), h, z, u, vs, incol.clone())
    _, h, z, u, vs, _ = while_loop_op(_first_any, functools.partial(_laqr5_step, wantt, wantz, ulp), carried,
                                      (stop, incol, accum, order, ktop, kbot, nbmps, jtop, ndcol, iloz, ihiz, smlnum,
                                       *shifts))
    far = live & accum
    jtop, jbot = (one, order) if wantt else (ktop, kbot)
    k1 = torch.clamp(ktop - incol, min=1)
    nu = (kdu - torch.clamp(ndcol - kbot, min=0)) - k1 + 1
    h = _region_put(h, ku, one, u, _square_within(kdu, ub) & _mat(far))
    block = _region_at(u, k1, k1, ub, ub)
    first = torch.minimum(ndcol, kbot) + 1
    _, h, _ = while_loop_op(_first_any, _columns_chunk, (far & (first <= jbot), h.contiguous(), first),
                            (jbot, nh, incol + k1, nu, block, ku, kwh))
    edge = torch.maximum(ktop, incol) - 1
    _, h, _, _ = while_loop_op(_first_any, functools.partial(_rows_chunk, False),
                               (far & (jtop <= edge), h, h.new_zeros(()), jtop.clone()),
                               (edge, nv, incol + k1, nu, block, kwv))
    if wantz:
        _, h, z, _ = while_loop_op(_first_any, functools.partial(_rows_chunk, True),
                                   (far & (iloz <= ihiz), h, z.contiguous(), iloz.clone()),
                                   (ihiz, nv, incol + k1, nu, block, kwv))
    incol = torch.where(live, incol + 2 * nbmps, incol)
    return live & (incol <= kbot - 2), h, z if wantz else z.clone(), vs.clone(), incol


# DO 145's pass for KRCOL: MTOP, MBOT, M22 and BMP22 (_laqr5_pair); DO 80, then JBOT and DO 100, then DO 120 (ACCUM)
# or DO 140 (WANTZ), each M = MBOT, MTOP, -1
def _laqr5_step(wantt, wantz, ulp, live, h, z, u, vs, krcol, stop, incol, accum, order, ktop, kbot, nbmps, jtop,
                ndcol, iloz, ihiz, smlnum, *shifts):
    kdu = 4 * nbmps
    mtop = torch.clamp(torch.div(ktop - krcol, 2, rounding_mode='trunc') + 1, min=1)
    mbot = torch.minimum(nbmps, torch.div(kbot - krcol - 1, 2, rounding_mode='trunc'))
    m22 = mbot + 1
    two = live & (mbot < nbmps) & (krcol + 2 * (m22 - 1) == kbot - 2)
    jbot = torch.where(accum, torch.minimum(ndcol, kbot), order if wantt else kbot)
    h, z, u, vs = _laqr5_pair(wantz, ulp, smlnum, two, h, z, u, vs, krcol + 2 * (m22 - 1), m22, incol, accum, ktop,
                              kbot, jtop, jbot, kdu, iloz, ihiz, shifts)
    bulges = live & (mbot >= mtop)
    _, h, vs, _ = while_loop_op(_first_any, functools.partial(_laqr5_bulge, ulp), (bulges, h, vs, mbot.clone()),
                                (mtop, krcol, ktop, kbot, jtop, smlnum, *shifts))
    jbot = torch.where(accum, torch.minimum(ndcol, kbot), order if wantt else kbot)
    _, h, _ = while_loop_op(_first_any, _laqr5_rows, (bulges.clone(), h, mbot.clone()), (mtop, krcol, ktop, jbot, vs))
    _, u, z, _ = while_loop_op(_first_any, functools.partial(_laqr5_products, wantz),
                               (bulges.clone(), u, z, mbot.clone()),
                               (mtop, mbot, krcol, incol, accum, ktop, kdu, iloz, ihiz, vs))
    krcol = torch.where(live, krcol + 1, krcol)
    return live & (krcol <= stop), h, z if wantz else z.clone(), u.clone(), vs.clone(), krcol


# the BMP22 block where `two`, K = KRCOL + 2 (M22 - 1): xLAQR1(2) and xLARFG(2) at K = KTOP - 1, else xLARFG(2) of
# H(K+1,K), H(K+2,K) (H(K+1,K) = BETA, H(K+2,K) = 0); V(1..2, M22) = TAU, V2; DO 30, DO 40, the vigilant test (K >=
# KTOP), DO 50 on U rows MAX(1, KTOP-INCOL)..KDU (ACCUM) or DO 60 on Z rows ILOZ..IHIZ
def _laqr5_pair(wantz, ulp, smlnum, two, h, z, u, vs, k, m22, incol, accum, ktop, kbot, jtop, jbot, kdu, iloz, ihiz,
                shifts):
    rows = _counted(h)
    intro = two & (k == ktop - 1)
    chase = two & ~intro
    lead = _laqr1(lambda r, c: _entry(h, k + r, k + c), 2, *_shift_pair(shifts, m22))
    beta, tau, (tail,) = _larfg(torch.where(intro, lead[0], _entry(h, k + 1, k)),
                                [torch.where(intro, lead[1], _entry(h, k + 2, k))])
    h = _entry_put(h, k + 1, k, beta, chase)
    h = _entry_put(h, k + 2, k, 0, chase)
    v1, v2 = tau, tail
    vs = _reflector_put(vs, m22, [v1, v2, _at(vs[..., 2, :], m22)], two)
    h = _columns_reflected(h, k + 1, (v2,), (v1, v1 * _cj(v2)),
                           _vec(two) & _span(rows, jtop, torch.minimum(kbot, k + 3)))
    t1 = _cj(v1)
    h = _rows_reflected(h, k + 1, (_cj(v2),), (t1, t1 * v2), _vec(two) & _span(rows, k + 1, jbot))
    h = _vigilant(h, k, ktop, kbot, smlnum, ulp, two & (k >= ktop))
    u = _pair_reflected(u, k - incol + 1, v1, v2, _vec(two & accum) & _span(_counted(u), torch.clamp(
        ktop - incol, min=1), kdu))
    if wantz:
        z = _pair_reflected(z, k + 1, v1, v2, _vec(two & ~accum) & _span(_counted(z), iloz, ihiz))
    return h, z if wantz else z.clone(), u, vs


# DO 80's pass for M, K = KRCOL + 2 (M - 1): xLAQR1(3) of H(KTOP,KTOP) and xLARFG(3) at K = KTOP - 1; else the
# previous V(:,M)'s fill of H(K+3,K..K+2), xLARFG(3) of H(K+1..K+3,K), and where H(K+3,K) = H(K+3,K+1) = 0 and
# H(K+3,K+2) /= 0 the refined VT (xLAQR1(3) of H(K+1,K+1), xLARFG(3)) taken where its H(K+2,K), H(K+3,K) residue is
# at most ULP (CABS1 H(K,K) + H(K+1,K+1) + H(K+2,K+2)); DO 70, the column K+1 update, the vigilant test (K >= KTOP)
def _laqr5_bulge(ulp, live, h, vs, m, mtop, krcol, ktop, kbot, jtop, smlnum, *shifts):
    rows = _counted(h)
    k = krcol + 2 * (m - 1)
    intro = live & (k == ktop - 1)
    chase = live & ~intro
    first, second = _shift_pair(shifts, m)
    lead = _laqr1(lambda r, c: _entry(h, ktop + r - 1, ktop + c - 1), 3, first, second)
    o1, o2, o3 = _reflector_at(vs, m)
    t2, t3 = o1 * _cj(o2), o1 * _cj(o3)
    refsum = o3 * _entry(h, k + 3, k + 2)
    h = _entry_put(h, k + 3, k, -refsum * o1, chase)
    h = _entry_put(h, k + 3, k + 1, -refsum * t2, chase)
    h = _entry_put(h, k + 3, k + 2, _entry(h, k + 3, k + 2) - refsum * t3, chase)
    beta, tau, (x2, x3) = _larfg(torch.where(intro, lead[0], _entry(h, k + 1, k)),
                                 [torch.where(intro, lead[1], _entry(h, k + 2, k)),
                                  torch.where(intro, lead[2], _entry(h, k + 3, k))])
    settled = (_entry(h, k + 3, k) != 0) | (_entry(h, k + 3, k + 1) != 0) | (_entry(h, k + 3, k + 2) == 0)
    trial = _laqr1(lambda r, c: _entry(h, k + r, k + c), 3, first, second)
    _, tau_t, (y2, y3) = _larfg(trial[0], trial[1:])
    w1 = _cj(tau_t)
    w2, w3 = w1 * y2, w1 * y3
    sub, below = _entry(h, k + 1, k), _entry(h, k + 2, k)
    refsum = sub + _cj(y2) * below
    bound = ulp * (_abs1(_entry(h, k, k)) + _abs1(_entry(h, k + 1, k + 1)) + _abs1(_entry(h, k + 2, k + 2)))
    refined = chase & ~settled & ~(_abs1(below - refsum * w2) + _abs1(refsum * w3) > bound)
    h = _entry_put(h, k + 1, k, torch.where(refined, sub - refsum * w1, beta), chase)
    h = _entry_put(h, k + 2, k, 0, chase)
    h = _entry_put(h, k + 3, k, 0, chase)
    v1, v2, v3 = (torch.where(refined, a, b) for a, b in ((tau_t, tau), (y2, x2), (y3, x3)))
    vs = _reflector_put(vs, m, [v1, v2, v3], live)
    h = _columns_reflected(h, k + 1, (v2, v3), (v1, v1 * _cj(v2), v1 * _cj(v3)),
                           _vec(live) & _span(rows, jtop, torch.minimum(kbot, k + 3)))
    t1 = _cj(v1)
    h = _rows_reflected(h, k + 1, (_cj(v2), _cj(v3)), (t1, t1 * v2, t1 * v3), _vec(live) & (rows == _vec(k + 1)))
    h = _vigilant(h, k, ktop, kbot, smlnum, ulp, live & (k >= ktop))
    m = torch.where(live, m - 1, m)
    return live & (m >= mtop), h, vs, m


# DO 100's pass for M: DO 90, rows K+1..K+3 over the columns MAX(KTOP, KRCOL + 2 M)..JBOT
def _laqr5_rows(live, h, m, mtop, krcol, ktop, jbot, vs):
    k = krcol + 2 * (m - 1)
    v1, v2, v3 = _reflector_at(vs, m)
    t1 = _cj(v1)
    h = _rows_reflected(h, k + 1, (_cj(v2), _cj(v3)), (t1, t1 * v2, t1 * v3),
                        _vec(live) & _span(_counted(h), torch.maximum(ktop, krcol + 2 * m), jbot))
    m = torch.where(live, m - 1, m)
    return live & (m >= mtop), h, m


# DO 120's pass for M where ACCUM (DO 110 on U rows I2..I4, columns KMS+1..KMS+3), else DO 140's where WANTZ (DO 130
# on Z rows ILOZ..IHIZ, columns K+1..K+3)
def _laqr5_products(wantz, live, u, z, m, mtop, mbot, krcol, incol, accum, ktop, kdu, iloz, ihiz, vs):
    k = krcol + 2 * (m - 1)
    v1, v2, v3 = _reflector_at(vs, m)
    right = (v1, v1 * _cj(v2), v1 * _cj(v3))
    i2 = torch.maximum(torch.clamp(ktop - incol, min=1), (k - incol) - (krcol - incol) + 1)
    i4 = torch.minimum(kdu, krcol + 2 * (mbot - 1) - incol + 5)
    u = _columns_reflected(u, k - incol + 1, (v2, v3), right, _vec(live & accum) & _span(_counted(u), i2, i4))
    if wantz:
        z = _columns_reflected(z, k + 1, (v2, v3), right, _vec(live & ~accum) & _span(_counted(z), iloz, ihiz))
    m = torch.where(live, m - 1, m)
    return live & (m >= mtop), u, z if wantz else z.clone(), m


# Reference-LAPACK 3.12.1 SRC/zlartg.f90 147-288 (E. Anderson, Algorithm 978, as _dlartg): C, S, R of complex F and
# G, SAFMIN = DLAMCH('S') (la_constants), SAFMAX = 1 / SAFMIN, RTMIN = SQRT(SAFMIN): G = 0; F = 0 by G's parts (a zero
# part, else G1 within (RTMIN, SQRT(SAFMAX / 2)), else by U); F1 and G1 within (RTMIN, RTMAX = SQRT(SAFMAX / 4)), else
# by U, V and W; each with its F2 >= H2 SAFMIN subcase (S = CONJG(G) (F / SQRT(F2 H2)) where F2 > RTMIN and
# H2 < 2 RTMAX, else CONJG(G) (R / H2)) and its other (S = CONJG(G) (F / D)), 206-208, 226, 262-264 and 282. A
# complex by a real divided or multiplied by parts
def _zlartg(f, g):
    limits = torch.finfo(f.real.dtype)
    safmin = limits.tiny
    safmax = 1 / safmin
    rtmin = math.sqrt(safmin)

    def abssq(t):
        return t.real * t.real + t.imag * t.imag

    def bounded(value):
        return value.clamp(safmin, safmax)

    def solved(fs, gs, f2, h2):
        rtmax2 = math.sqrt(safmax / 4) * 2
        cosine = (f2 / h2).sqrt()
        r_first = _over(fs, cosine)
        s_first = torch.where((f2 > rtmin) & (h2 < rtmax2), gs.conj() * _over(fs, (f2 * h2).sqrt()),
                              gs.conj() * _over(r_first, h2))
        d = (f2 * h2).sqrt()
        c_second = f2 / d
        r_second = torch.where(c_second >= safmin, _over(fs, c_second), _scaled(fs, h2 / d))
        s_second = gs.conj() * _over(fs, d)
        first = f2 >= h2 * safmin
        return (torch.where(first, cosine, c_second), torch.where(first, s_first, s_second),
                torch.where(first, r_first, r_second))
    fr, fi, gr, gi = f.real, f.imag, g.real, g.imag
    g1 = torch.maximum(gr.abs(), gi.abs())
    edge = (gr == 0) | (gi == 0)
    r_edge = torch.where(gr == 0, gi.abs(), gr.abs())
    plain = (g1 > rtmin) & (g1 < math.sqrt(safmax / 2))
    u = bounded(g1)
    gs = _over(g, u)
    d_plain, d_scaled = abssq(g).sqrt(), abssq(gs).sqrt()
    s_zero = torch.where(edge, _over(g.conj(), r_edge), torch.where(plain, _over(g.conj(), d_plain),
                                                                      _over(gs.conj(), d_scaled)))
    r_zero = torch.where(edge, r_edge, torch.where(plain, d_plain, d_scaled * u))
    f1 = torch.maximum(fr.abs(), fi.abs())
    rtmax = math.sqrt(safmax / 4)
    inner = (f1 > rtmin) & (f1 < rtmax) & (g1 > rtmin) & (g1 < rtmax)
    c_in, s_in, r_in = solved(f, g, abssq(f), abssq(f) + abssq(g))
    u = bounded(torch.maximum(f1, g1))
    gs = _over(g, u)
    g2 = abssq(gs)
    tiny_f = f1 / u < rtmin
    v = bounded(f1)
    w = torch.where(tiny_f, v / u, torch.ones_like(u))
    fs = torch.where(tiny_f, _over(f, v), _over(f, u))
    f2 = abssq(fs)
    h2 = torch.where(tiny_f, f2 * (w * w) + g2, f2 + g2)
    c_out, s_out, r_out = solved(fs, gs, f2, h2)
    c_out, r_out = c_out * w, _scaled(r_out, u)
    c = torch.where(inner, c_in, c_out)
    s = torch.where(inner, s_in, s_out)
    r = torch.where(inner, r_in, r_out)
    g_zero, f_zero = g == 0, f == 0
    c = torch.where(g_zero, 1, torch.where(f_zero, 0, c))
    s = torch.where(g_zero, 0, torch.where(f_zero, s_zero, s))
    r = torch.where(g_zero, f, torch.where(f_zero, r_zero.to(f.dtype), r))
    return c, s, r


# Reference-LAPACK 3.12.1 SRC/zrot.f DO 30 (drot.f alike, S real): rows A and A+1 of x (`rows`) or its columns A and
# A+1, over the positions `within`: STEMP = C X + S Y, Y = C Y - CONJG(S) X, X = STEMP; the real C's products by parts
def _rotated(x, a, c, s, within, rows):
    k = _counted(x)
    u, w = (_row(x, a), _row(x, a + 1)) if rows else (_column(x, a), _column(x, a + 1))
    c, s = _vec(c), _vec(s)
    first, second = _scaled(u, c) + s * w, _scaled(w, c) - _cj(s) * u
    if rows:
        x = torch.where((k[:, None] == _mat(a)) & within[..., None, :], first[..., None, :], x)
        return torch.where((k[:, None] == _mat(a + 1)) & within[..., None, :], second[..., None, :], x)
    x = torch.where((k == _mat(a)) & within[..., :, None], first[..., :, None], x)
    return torch.where((k == _mat(a + 1)) & within[..., :, None], second[..., :, None], x)


# Reference-LAPACK 3.12.1 SRC/ztrexc.f 182-224 (COMPQ = 'V' where `wantq`) on the leading ORDER x ORDER block of t,
# where `when`: the N <= 1 or IFST = ILST return, then DO 10 K = IFST + M1, ILST + M2, M3 (forward where IFST < ILST,
# else back) as one while_loop: ZLARTG(T(K,K+1), T22 - T11), ZROT of rows K, K+1 over columns K+2..N by CS, SN, ZROT
# of columns K, K+1 over rows 1..K-1 by CS, CONJG(SN), T(K,K) = T22, T(K+1,K+1) = T11, and where WANTQ ZROT of Q's
# columns K, K+1 over rows 1..N. dtrexc.f, the real routine (DLAEXC's 1 x 1 and 2 x 2 blocks), is _dtrexc
def _ztrexc(wantq, order, t, q, ifst, ilst, when):
    forward = ifst < ilst
    step = torch.where(forward, 1, -1)
    carried = (when & (order > 1) & (ifst != ilst), t.contiguous(), q.contiguous() if wantq else t.new_zeros(()),
               torch.where(forward, ifst, ifst - 1))
    _, t, q, _ = while_loop_op(_first_any, functools.partial(_ztrexc_pass, wantq), carried,
                               (torch.where(forward, ilst - 1, ilst), step, order))
    return t, q


def _ztrexc_pass(wantq, live, t, q, k, last, step, order):
    rows = _counted(t)
    t11, t22 = _entry(t, k, k), _entry(t, k + 1, k + 1)
    cs, sn, _ = _zlartg(_entry(t, k, k + 1), t22 - t11)
    t = _rotated(t, k, cs, sn, _vec(live) & (rows >= _vec(k + 2)) & (rows <= _vec(order)), True)
    t = _rotated(t, k, cs, sn.conj(), _vec(live) & (rows <= _vec(k - 1)), False)
    t = _entry_put(t, k, k, t22, live)
    t = _entry_put(t, k + 1, k + 1, t11, live)
    if wantq:
        q = _rotated(q, k, cs, sn.conj(), _vec(live) & (rows <= _vec(order)), False)
    k = torch.where(live, k + step, k)
    return live & ((k - last) * step <= 0), t, q if wantq else q.clone(), k


# Reference-LAPACK 3.12.1 SRC/dlarfx.f 154-155 (the TAU = 0 return) and its 3 x 3 code: 190-206 (SIDE = 'L', M = 3,
# DO 60 J = 1, N: SUM = V1 C(1,J) + V2 C(2,J) + V3 C(3,J), C(I,J) = C(I,J) - SUM TI, TI = TAU VI) and 456-472 (SIDE =
# 'R', N = 3, DO 260 J = 1, M over C(J,1..3)), on C = x(R0.., C0..) where `when`, EXTENT the source's N (`left`) or M;
# V = (V1, V2, V3) batch scalars
def _dlarfx3(left, v, tau, x, r0, c0, extent, when):
    go = when & (tau != 0)
    ts = [tau * vi for vi in v]
    rows, cols = torch.arange(1, x.shape[-2] + 1, device=x.device), _counted(x)
    lines = [_row(x, r0 + i) if left else _column(x, c0 + i) for i in range(3)]
    total = _vec(v[0]) * lines[0] + _vec(v[1]) * lines[1] + _vec(v[2]) * lines[2]
    for i in range(3):
        made = lines[i] - total * _vec(ts[i])
        if left:
            hit = (rows[:, None] == _mat(r0 + i)) & (_span(cols, c0, c0 + extent - 1) & _vec(go))[..., None, :]
            x = torch.where(hit, made[..., None, :], x)
        else:
            hit = (cols == _mat(c0 + i)) & (_span(rows, r0, r0 + extent - 1) & _vec(go))[..., :, None]
            x = torch.where(hit, made[..., :, None], x)
    return x


# Reference-LAPACK 3.12.1 SRC/dlasy2.f 230-475 as dlaexc.f 247-249 calls it (LTRANL = LTRANR = .FALSE., ISGN = -1):
# TL = D(1,1) (N1 x N1), TR = D(N1+1,N1+1) (N2 x N2) and B = D(1,N1+1) (N1 x N2) of the static 4 x 4 `d`, N1, N2 per
# member; EPS = DLAMCH('P'), SMLNUM = DLAMCH('S') / EPS; K = N1 + N1 + N2 - 2 selects 248-264 (K = 1), 270-286 or
# 292-306 then label 40's 312-356 (K = 2, 3: IDAMAX's first largest |TMP|, the LOCU12, LOCL21, LOCU22, XSWPIV and
# BSWPIV tables of 221-224) or 365-475 (K = 4: DO 100's complete pivoting, DO 70 and DO 60 (406-416) taking the last
# ABS(T16(IP,JP)) >= XMAX from XMAX = ZERO, DSWAP of rows (and BTMP) and of columns, DO 90 and DO 80, then DO 120
# and DO 130). On the source's stated domain each I's first comparison ABS(T16(I,I)) >= ZERO holds, so DO 70/60 set
# IPSV and JPSV at every I; the 1 the trace starts them from is unreachable there and selects nothing. Returns SCALE
# and X(1,1), X(2,1), X(1,2), X(2,2) (INFO and XNORM unread by DLAEXC)
def _dlasy2(n1, n2, d):
    limits = torch.finfo(d.dtype)
    eps = limits.eps
    smlnum = limits.tiny / eps
    sgn = -1.0
    one, zero = torch.ones_like(n1), torch.zeros_like(d[..., 0, 0])

    def exchanged(values, at, i):
        made = list(values)
        for r in range(i + 1, len(values) + 1):
            made[i - 1] = torch.where(at == r, values[r - 1], made[i - 1])
            made[r - 1] = torch.where(at == r, values[i - 1], values[r - 1])
        return made
    tl11, tl12, tl21, tl22 = d[..., 0, 0], d[..., 0, 1], d[..., 1, 0], d[..., 1, 1]
    tr11, tr12, tr21, tr22 = (_entry(d, n1 + i, n1 + j) for i, j in ((1, 1), (1, 2), (2, 1), (2, 2)))
    b11, b12, b21, b22 = (_entry(d, one * i, n1 + j) for i, j in ((1, 1), (1, 2), (2, 1), (2, 2)))
    k = n1 + n1 + n2 - 2
    tau1 = tl11 + sgn * tr11
    bet = tau1.abs()
    tiny = bet <= smlnum
    tau1, bet = torch.where(tiny, smlnum, tau1), torch.where(tiny, smlnum, bet)
    gam = b11.abs()
    scale1 = torch.where(smlnum * gam > bet, 1 / gam, torch.ones_like(gam))
    x1 = (b11 * scale1) / tau1
    three = k == 3
    smin = torch.where(three, functools.reduce(_larger, (tr11.abs(), tl11.abs(), tl12.abs(), tl21.abs(), tl22.abs())),
                       functools.reduce(_larger, (tl11.abs(), tr11.abs(), tr12.abs(), tr21.abs(), tr22.abs())))
    smin = _larger(eps * smin, smlnum)
    tmp = [tl11 + sgn * tr11, torch.where(three, tl21, sgn * tr12), torch.where(three, tl12, sgn * tr21),
           torch.where(three, tl22 + sgn * tr11, tl11 + sgn * tr22)]
    btmp1, btmp2 = b11, torch.where(three, b21, b12)
    ipiv, dmax = one, tmp[0].abs()
    for i in range(2, 5):
        bigger = tmp[i - 1].abs() > dmax
        ipiv, dmax = torch.where(bigger, i, ipiv), torch.where(bigger, tmp[i - 1].abs(), dmax)

    def located(table):
        made = tmp[table[0] - 1]
        for p in range(2, 5):
            made = torch.where(ipiv == p, tmp[table[p - 1] - 1], made)
        return made
    u11 = located((1, 2, 3, 4))
    u11 = torch.where(u11.abs() <= smin, smin, u11)
    u12 = located((3, 4, 1, 2))
    l21 = located((2, 1, 4, 3)) / u11
    u22 = located((4, 3, 2, 1)) - u12 * l21
    xswap, bswap = ipiv >= 3, (ipiv == 2) | (ipiv == 4)
    u22 = torch.where(u22.abs() <= smin, smin, u22)
    btmp1, btmp2 = torch.where(bswap, btmp2, btmp1), torch.where(bswap, btmp1 - l21 * btmp2, btmp2 - l21 * btmp1)
    raised = ((2 * smlnum) * btmp2.abs() > u22.abs()) | ((2 * smlnum) * btmp1.abs() > u11.abs())
    scale2 = torch.where(raised, 0.5 / _larger(btmp1.abs(), btmp2.abs()), torch.ones_like(btmp1))
    btmp1, btmp2 = torch.where(raised, btmp1 * scale2, btmp1), torch.where(raised, btmp2 * scale2, btmp2)
    second = btmp2 / u22
    first = btmp1 / u11 - (u12 / u11) * second
    first, second = torch.where(xswap, second, first), torch.where(xswap, first, second)
    smin4 = functools.reduce(_larger, (tr11.abs(), tr12.abs(), tr21.abs(), tr22.abs()))
    smin4 = functools.reduce(_larger, (smin4, tl11.abs(), tl12.abs(), tl21.abs(), tl22.abs()))
    smin4 = _larger(eps * smin4, smlnum)
    t16 = [[zero] * 4 for _ in range(4)]
    t16[0][0], t16[1][1] = tl11 + sgn * tr11, tl22 + sgn * tr11
    t16[2][2], t16[3][3] = tl11 + sgn * tr22, tl22 + sgn * tr22
    t16[0][1], t16[1][0], t16[2][3], t16[3][2] = tl12, tl21, tl12, tl21
    t16[0][2], t16[1][3], t16[2][0], t16[3][1] = sgn * tr21, sgn * tr21, sgn * tr12, sgn * tr12
    btmp = [b11, b21, b12, b22]
    ipsv, jpsv, jpiv = one, one, []
    for i in range(1, 4):
        xmax = zero
        for ip in range(i, 5):
            for jp in range(i, 5):
                taken = t16[ip - 1][jp - 1].abs() >= xmax
                xmax = torch.where(taken, t16[ip - 1][jp - 1].abs(), xmax)
                ipsv, jpsv = torch.where(taken, ip, ipsv), torch.where(taken, jp, jpsv)
        for c in range(4):
            column = exchanged([t16[r][c] for r in range(4)], ipsv, i)
            for r in range(4):
                t16[r][c] = column[r]
        btmp = exchanged(btmp, ipsv, i)
        t16 = [exchanged(row, jpsv, i) for row in t16]
        jpiv.append(jpsv)
        t16[i - 1][i - 1] = torch.where(t16[i - 1][i - 1].abs() < smin4, smin4, t16[i - 1][i - 1])
        for j in range(i + 1, 5):
            t16[j - 1][i - 1] = t16[j - 1][i - 1] / t16[i - 1][i - 1]
            btmp[j - 1] = btmp[j - 1] - t16[j - 1][i - 1] * btmp[i - 1]
            for kk in range(i + 1, 5):
                t16[j - 1][kk - 1] = t16[j - 1][kk - 1] - t16[j - 1][i - 1] * t16[i - 1][kk - 1]
    t16[3][3] = torch.where(t16[3][3].abs() < smin4, smin4, t16[3][3])
    raised = functools.reduce(torch.logical_or, [(8 * smlnum) * btmp[i].abs() > t16[i][i].abs() for i in range(4)])
    scale4 = torch.where(raised, (1 / 8) / functools.reduce(_larger, [b.abs() for b in btmp]), torch.ones_like(zero))
    btmp = [torch.where(raised, b * scale4, b) for b in btmp]
    solved = [zero] * 4
    for i in range(1, 5):
        kk = 5 - i
        temp = 1 / t16[kk - 1][kk - 1]
        solved[kk - 1] = btmp[kk - 1] * temp
        for j in range(kk + 1, 5):
            solved[kk - 1] = solved[kk - 1] - (temp * t16[kk - 1][j - 1]) * solved[j - 1]
    for i in range(1, 4):
        solved = exchanged(solved, jpiv[4 - i - 1], 4 - i)
    single, four = k == 1, k == 4
    return (torch.where(single, scale1, torch.where(four, scale4, scale2)),
            torch.where(single, x1, torch.where(four, solved[0], first)),
            torch.where(four, solved[1], torch.where(three, second, 0)),
            torch.where(four, solved[2], torch.where(k == 2, second, 0)), torch.where(four, solved[3], 0))


# Reference-LAPACK 3.12.1 SRC/dlaexc.f 185-435, DLAEXC(WANTQ, N, T, Q, J1, N1, N2) on the leading ORDER x ORDER block
# of t (q where `wantq`) where `when`, J1, N1, N2 per member: 189-192 the N = 0, N1 = 0, N2 = 0 and J1 + N1 > N
# returns; 198-225 N1 = N2 = 1: DLARTG(T(J1,J2), T22 - T11), DROT of rows J1, J2 over columns J3..N (J3 <= N), of
# columns J1, J2 over rows 1..J1-1 and of Q's columns over 1..N, T(J1,J1) = T22, T(J2,J2) = T11; else 234-243 D =
# T(J1:J1+ND-1, J1:J1+ND-1) (the static 4 x 4, zero past ND), DNORM = DLANGE('M'), THRESH = MAX(TEN EPS DNORM,
# SMLNUM); 247-249 _dlasy2; 253's K = N1 + N1 + N2 - 3 selects label 10 (262-295), 20 (305-337) or 30 (349-394):
# DLARFG(3) of U (U1, U2), DLARFX (_dlarfx3) of D from the left and right, the swap rejected (label 50, INFO = 1, T
# and Q as they were) where the test's largest exceeds THRESH, else DLARFX of T and Q, T's zeros and its corner;
# label 40 (398-426) where N2 = 2, and where N1 = 2 (J3 = J1 + N2): DLANV2 of the 2 x 2 block, its DROTs of rows over
# columns J+2..N, of columns over rows 1..J-1 and of Q's columns. Returns T, Q, INFO
def _dlaexc(wantq, order, t, q, j1, n1, n2, when):
    limits = torch.finfo(t.dtype)
    eps = limits.eps
    smlnum = limits.tiny / eps
    rows, one = _counted(t), torch.ones_like(j1)
    go = when & (order != 0) & (n1 != 0) & (n2 != 0) & (j1 + n1 <= order)
    j2, j3, j4 = j1 + 1, j1 + 2, j1 + 3
    pair = go & (n1 == 1) & (n2 == 1)
    t11, t22 = _entry(t, j1, j1), _entry(t, j2, j2)
    cs, sn = _dlartg(_entry(t, j1, j2), t22 - t11)
    t = _rotated(t, j1, cs, sn, _vec(pair & (j3 <= order)) & _span(rows, j3, order), True)
    t = _rotated(t, j1, cs, sn, _vec(pair) & (rows <= _vec(j1 - 1)), False)
    t = _entry_put(_entry_put(t, j1, j1, t22, pair), j2, j2, t11, pair)
    if wantq:
        q = _rotated(q, j1, cs, sn, _vec(pair) & (_counted(q) <= _vec(order)), False)
    block = go & ~pair
    d = torch.where(_square_within(n1 + n2, 4), _region_at(t, j1, j1, 4, 4), 0)
    thresh = _larger(10 * eps * d.abs().flatten(-2).amax(-1), smlnum)
    scale, x11, x21, x12, x22 = _dlasy2(n1, n2, d)
    unit = torch.ones_like(scale)
    k = n1 + n1 + n2 - 3
    tests = []
    _, tau, (ua, ub) = _larfg(x12, [scale, x11])
    u = (ua, ub, unit)
    t11, case = _entry(t, j1, j1), block & (k == 1)
    d = _dlarfx3(False, u, tau, _dlarfx3(True, u, tau, d, one, one, 3, case), one, one, 3, case)
    taken = case & ~(functools.reduce(_larger, (d[..., 2, 0].abs(), d[..., 2, 1].abs(), (d[..., 2, 2] - t11).abs()))
                     > thresh)
    t = _dlarfx3(True, u, tau, t, j1, j1, order - j1 + 1, taken)
    t = _dlarfx3(False, u, tau, t, one, j1, j2, taken)
    t = _entry_put(_entry_put(_entry_put(t, j3, j1, 0, taken), j3, j2, 0, taken), j3, j3, t11, taken)
    if wantq:
        q = _dlarfx3(False, u, tau, q, one, j1, order, taken)
    tests.append((case, taken))
    _, tau, (ub, uc) = _larfg(-x11, [-x21, scale])
    u = (unit, ub, uc)
    t33, case = _entry(t, j3, j3), block & (k == 2)
    d = _dlarfx3(False, u, tau, _dlarfx3(True, u, tau, d, one, one, 3, case), one, one, 3, case)
    taken = case & ~(functools.reduce(_larger, (d[..., 1, 0].abs(), d[..., 2, 0].abs(), (d[..., 0, 0] - t33).abs()))
                     > thresh)
    t = _dlarfx3(False, u, tau, t, one, j1, j3, taken)
    t = _dlarfx3(True, u, tau, t, j1, j2, order - j1, taken)
    t = _entry_put(_entry_put(_entry_put(t, j1, j1, t33, taken), j2, j1, 0, taken), j3, j1, 0, taken)
    if wantq:
        q = _dlarfx3(False, u, tau, q, one, j1, order, taken)
    tests.append((case, taken))
    _, tau1, (u1b, u1c) = _larfg(-x11, [-x21, scale])
    temp = -tau1 * (x12 + u1b * x22)
    _, tau2, (u2b, u2c) = _larfg(-temp * u1b - x22, [-temp * u1c, scale])
    u1, u2, case = (unit, u1b, u1c), (unit, u2b, u2c), block & (k == 3)
    d = _dlarfx3(False, u1, tau1, _dlarfx3(True, u1, tau1, d, one, one, 4, case), one, one, 4, case)
    d = _dlarfx3(False, u2, tau2, _dlarfx3(True, u2, tau2, d, one + 1, one, 4, case), one, one + 1, 4, case)
    taken = case & ~(functools.reduce(_larger, (d[..., 2, 0].abs(), d[..., 2, 1].abs(), d[..., 3, 0].abs(),
                                                d[..., 3, 1].abs())) > thresh)
    t = _dlarfx3(True, u1, tau1, t, j1, j1, order - j1 + 1, taken)
    t = _dlarfx3(False, u1, tau1, t, one, j1, j4, taken)
    t = _dlarfx3(True, u2, tau2, t, j2, j1, order - j1 + 1, taken)
    t = _dlarfx3(False, u2, tau2, t, one, j2, j4, taken)
    for r, c in ((j3, j1), (j3, j2), (j4, j1), (j4, j2)):
        t = _entry_put(t, r, c, 0, taken)
    if wantq:
        q = _dlarfx3(False, u2, tau2, _dlarfx3(False, u1, tau1, q, one, j1, order, taken), one, j2, order, taken)
    tests.append((case, taken))
    swapped = functools.reduce(torch.logical_or, [taken for _, taken in tests])
    info = torch.where(functools.reduce(torch.logical_or, [case & ~taken for case, taken in tests]), 1, 0)
    for j, standing in ((j1, swapped & (n2 == 2)), (j1 + n2, swapped & (n1 == 2))):
        a, b, c, e, _, _, _, _, cs, sn = _dlanv2(_entry(t, j, j), _entry(t, j, j + 1), _entry(t, j + 1, j),
                                                 _entry(t, j + 1, j + 1))
        for (r, col), value in (((j, j), a), ((j, j + 1), b), ((j + 1, j), c), ((j + 1, j + 1), e)):
            t = _entry_put(t, r, col, value, standing)
        t = _rotated(t, j, cs, sn, _vec(standing & (j + 2 <= order)) & _span(rows, j + 2, order), True)
        t = _rotated(t, j, cs, sn, _vec(standing) & (rows <= _vec(j - 1)), False)
        if wantq:
            q = _rotated(q, j, cs, sn, _vec(standing) & (_counted(q) <= _vec(order)), False)
    return t, q, info


# Reference-LAPACK 3.12.1 SRC/dtrexc.f 207-428 (COMPQ = 'V' where `wantq`) on the leading ORDER x ORDER block of t,
# where `when`: 207-208 the N <= 1 return; 213-234 IFST and ILST moved to their blocks' first rows, NBF and NBL;
# 236-237 the IFST = ILST return; 243-246 ILST adjusted where IFST < ILST; 248 and 337 HERE = IFST, then label 10
# (forward) or label 20 (back) as one while_loop whose pass is one label pass (_dtrexc_pass); 426 ILST = HERE.
# Returns T, Q, ILST (the source's in/out ILST, also where it returns early) and INFO (1 where a checked DLAEXC
# rejected its swap, ILST = HERE then)
def _dtrexc(wantq, order, t, q, ifst, ilst, when):
    go = when & (order > 1)
    ifst = torch.where(go & (ifst > 1) & (_entry(t, ifst, ifst - 1) != 0), ifst - 1, ifst)
    nbf = torch.where((ifst < order) & (_entry(t, ifst + 1, ifst) != 0), 2, 1)
    ilst = torch.where(go & (ilst > 1) & (_entry(t, ilst, ilst - 1) != 0), ilst - 1, ilst)
    nbl = torch.where((ilst < order) & (_entry(t, ilst + 1, ilst) != 0), 2, 1)
    go = go & (ifst != ilst)
    forward = ifst < ilst
    ilst = torch.where(go & forward & (nbf == 2) & (nbl == 1), ilst - 1, ilst)
    ilst = torch.where(go & forward & (nbf == 1) & (nbl == 2), ilst + 1, ilst)
    carried = (go, t.contiguous(), q.contiguous() if wantq else t.new_zeros(()), ifst.clone(), nbf,
               torch.zeros_like(ifst))
    _, t, q, here, _, info = while_loop_op(_first_any, functools.partial(_dtrexc_pass, wantq), carried,
                                           (forward, ilst, order))
    return t, q, torch.where(go, here, ilst), info


# dtrexc.f 250-333 (`forward`) and 338-424, one label 10 or label 20 pass for HERE: where NBF = 1 or 2, NBNEXT of the
# next block (T(HERE+NBF+1,HERE+NBF) /= 0, else T(HERE-1,HERE-2) /= 0), DLAEXC(HERE, NBF, NBNEXT) (back: HERE -
# NBNEXT, NBNEXT, NBF), HERE by NBNEXT and NBF = 3 where the 2 x 2 split (T(HERE+1,HERE) = 0); where NBF = 3, the
# first DLAEXC(HERE + 1, 1, NBNEXT) (back: HERE - NBNEXT, NBNEXT, 1), then where NBNEXT = 1 DLAEXC(HERE, 1, 1) and
# HERE by 1, else NBNEXT = 1 where the next 2 x 2 split (T(HERE+2,HERE+1) = 0, back T(HERE,HERE-1) = 0): where 2
# DLAEXC(HERE, 1, 2) (back: HERE - 1, 2, 1), else DLAEXC(HERE, 1, 1) and DLAEXC(HERE + 1, 1, 1) (back: HERE - 1), HERE
# by 2. A checked DLAEXC's INFO /= 0 ends it (ILST = HERE), the 1 x 1 swaps unchecked as the source leaves them
def _dtrexc_pass(wantq, live, t, q, here, nbf, info, forward, ilst, order):
    one = torch.ones_like(here)
    whole = nbf <= 2
    ahead = torch.where((here + nbf + 1 <= order) & (_entry(t, here + nbf + 1, here + nbf) != 0), 2, 1)
    split = torch.where((here + 3 <= order) & (_entry(t, here + 3, here + 2) != 0), 2, 1)
    behind = torch.where((here >= 3) & (_entry(t, here - 1, here - 2) != 0), 2, 1)
    nbnext = torch.where(forward, torch.where(whole, ahead, split), behind)
    t, q, first = _dlaexc(wantq, order, t, q, torch.where(forward, torch.where(whole, here, here + 1), here - behind),
                          torch.where(forward, torch.where(whole, nbf, 1), behind),
                          torch.where(forward, torch.where(whole, ahead, split), torch.where(whole, nbf, 1)), live)
    failed = live & (first != 0)
    moved = live & ~failed & whole
    shifted = torch.where(forward, here + nbnext, here - nbnext)
    nbf = torch.where(moved & (nbf == 2) & (_entry(t, shifted + 1, shifted) == 0), 3, nbf)
    apart = live & ~failed & ~whole
    lone = nbnext == 1
    paired = ~lone & torch.where(forward, _entry(t, here + 2, here + 1) != 0, _entry(t, here, here - 1) != 0)
    t, q, second = _dlaexc(wantq, order, t, q, torch.where(paired & ~forward, here - 1, here),
                           torch.where(paired & ~forward, 2, 1), torch.where(paired & forward, 2, 1), apart)
    failed = failed | (apart & paired & (second != 0))
    t, q, _ = _dlaexc(wantq, order, t, q, torch.where(forward, here + 1, here - 1), one, one, apart & ~lone & ~paired)
    step = torch.where(lone, 1, 2)
    stepped = torch.where(forward, here + step, here - step)
    here = torch.where(moved, shifted, torch.where(apart & ~failed, stepped, here))
    info = torch.where(failed, 1, info)
    more = torch.where(forward, here < ilst, here > ilst)
    return live & ~failed & more, t, q if wantq else q.clone(), here, nbf, info


# Reference-LAPACK 3.12.1 SRC/zunm2r.f 222-271 (dorm2r.f alike, TAU(I) either way): C = Q C, Q^H C, C Q or C Q^H
# (`left`, `notran`), Q = H(1) ... H(K), on C = c(CR:CR+M-1, CC:CC+N-1), the reflectors' A(1,1) at a(AR, AC) and TAU
# 1-based along `tau`, where `when`: the M = 0, N = 0 or K = 0 return, then DO 10 I = I1, I2, I3 (forward where LEFT
# and not NOTRAN or not LEFT and NOTRAN, else back) as static I over tau's extent where I <= K: MI = M - I + 1 and
# IC = I (left), else NI = N - I + 1 and JC = I; TAUI = TAU(I), or CONJG(TAU(I)) where not NOTRAN; ZLARF1F(SIDE, MI,
# NI, A(I,I), TAUI, C(IC,JC))
def _unm2r(left, notran, m, n, k, a, ar, ac, tau, c, cr, cc, when):
    go = when & (m > 0) & (n > 0) & (k > 0)
    count = tau.shape[-1]
    steps = range(1, count + 1) if left != notran else range(count, 0, -1)
    size = c.shape[-2] if left else c.shape[-1]
    for i in steps:
        mi, ic = (m - i + 1, cr + i - 1) if left else (m, cr)
        ni, jc = (n, cc) if left else (n - i + 1, cc + i - 1)
        taui = tau[..., i - 1] if notran else _cj(tau[..., i - 1])
        column = _column(a, ac + i - 1)
        shift = (ar + i - 1) - (ic if left else jc)
        v = column.gather(-1, (torch.arange(size, device=c.device) + _vec(shift)).clamp(0, column.shape[-1] - 1))
        c = _larf1f(left, mi, ni, v, taui, c, ic, jc, go & (i <= k))
    return c


# Reference-LAPACK 3.12.1 SRC/zunmhr.f 212 and 262-284 (dormhr.f alike): NH = IHI - ILO, the M = 0, N = 0 or NH = 0
# return, then ZUNMQR(SIDE, TRANS, MI, NI, NH, A(ILO+1,ILO), TAU(ILO), C(I1,I2), LWORK) (MI = NH and I1 = ILO + 1
# left, else NI = NH and I2 = ILO + 1) with the caller's LWORK `lwork` (_unmqr)
def _unmhr(left, notran, m, n, ilo, ihi, a, tau, c, when, lwork):
    nh = ihi - ilo
    one = torch.ones_like(ilo)
    mi, ni, i1, i2 = (nh, n, ilo + 1, one) if left else (m, nh, one, ilo + 1)
    index = (torch.arange(tau.shape[-1], device=tau.device) + _vec(ilo - 1)).clamp(0, max(tau.shape[-1] - 1, 0))
    return _unmqr(left, notran, mi, ni, nh, a, ilo + 1, ilo, tau.gather(-1, index), c, i1, i2,
                  when & (m != 0) & (n != 0) & (nh != 0), lwork)


# Reference-LAPACK 3.12.1 SRC/zunmqr.f 207-336 (dormqr.f alike), ZUNMQR(SIDE, TRANS, M, N, K, A, TAU, C) in _unm2r's
# layout with LWORK `lwork`: NQ = M and NW = MAX(1, N) (left), else N and MAX(1, M); NB = MIN(NBMAX = 64,
# ILAENV(1, 'ZUNMQR') = 32, ilaenv.f 412-417, 426-431), LWKOPT = NW NB + TSIZE (TSIZE = LDT NBMAX = 65 64); the M =
# 0, N = 0 or K = 0 return; NBMIN = 2, and where NB > 1, NB < K and LWORK < LWKOPT NB = (LWORK - TSIZE) / LDWORK
# (LDWORK = NW) and NBMIN = MAX(2, ILAENV(2) = 2, ilaenv.f 579-584, 593-598); ZUNM2R where NB < NBMIN or NB >= K,
# else DO 10 I = I1, I2, I3 (_unmqr_block; from 1 by NB where LEFT and not NOTRAN or not LEFT and NOTRAN, else from
# ((K - 1) / NB) NB + 1 by -NB; the loop present where the static extent admits K > 32). 235's LWORK < NW return is
# unreached (xLAQR3's LWORK - JW >= JW = NW)
def _unmqr(left, notran, m, n, k, a, ar, ac, tau, c, cr, cc, when, lwork):
    nq, nw = (m, torch.clamp(n, min=1)) if left else (n, torch.clamp(m, min=1))
    forward, tsize = left != notran, 65 * 64
    go = when & (m != 0) & (n != 0) & (k != 0)
    nb = torch.full_like(k, 32)
    short = (nb > 1) & (nb < k) & (lwork < nw * nb + tsize)
    nb = torch.where(short, torch.div(lwork - tsize, nw, rounding_mode='trunc'), nb)
    blocked = go & ~((nb < 2) | (nb >= k))
    c = _unm2r(left, notran, m, n, k, a, ar, ac, tau, c, cr, cc, go & ~blocked)
    if tau.shape[-1] > 32:
        first = torch.ones_like(k) if forward else torch.div(k - 1, torch.where(blocked, nb, 1),
                                                                   rounding_mode='trunc') * nb + 1
        _, c, _ = while_loop_op(_first_any, functools.partial(_unmqr_block, left, notran), (blocked, c.contiguous(),
                                first), (nb, k, nq, m, n, a, ar, ac, tau, cr, cc))
    return c


# zunmqr.f 306-335, DO 10's pass for I: IB = MIN(NB, K - I + 1); ZLARFT('Forward', 'Columnwise', NQ - I + 1, IB,
# A(I,I), TAU(I), T) (_larft); MI = M - I + 1 and IC = I (left), else NI = N - I + 1 and JC = I; ZLARFB(SIDE, TRANS,
# 'Forward', 'Columnwise', MI, NI, IB, A(I,I), T, C(IC,JC)) (_larfb)
def _unmqr_block(left, notran, live, c, i, nb, k, nq, m, n, a, ar, ac, tau, cr, cc):
    rows, cols = c.shape[-2:]
    ib = torch.minimum(nb, k - i + 1)
    v = _region_at(a, ar + i - 1, ac + i - 1, rows if left else cols, 32)
    t = _larft(nq - i + 1, ib, v, _vec_at(tau, i, 32), live)
    mi, ic, ni, jc = (m - i + 1, cr + i - 1, n, cc) if left else (m, cr, n - i + 1, cc + i - 1)
    made = _larfb(left, not notran, mi, ni, ib, v, t, _region_at(c, ic, jc, rows, cols))
    c = _region_put(c, ic, jc, made, _within(mi, rows)[..., :, None] & _within(ni, cols)[..., None, :] & _mat(live))
    if left != notran:
        i = torch.where(live, i + nb, i)
        return live & (i <= k), c, i
    i = torch.where(live, i - nb, i)
    return live & (i >= 1), c, i


# positions TOP..TOP+SIZE-1 of v (_vec_at; positions past v clamped where unused), and v with `values` placed from
# TOP where `mask` (_vec_put; positions past v unplaced)
def _vec_at(v, top, size):
    return v.gather(-1, ((top - 1)[..., None] + torch.arange(size, device=v.device)).clamp(0, v.shape[-1] - 1))


def _vec_put(v, top, values, mask):
    offset = _counted(v) - _vec(top)
    inside = (offset >= 0) & (offset < values.shape[-1])
    index = offset.clamp(0, values.shape[-1] - 1)
    return torch.where(inside & mask.gather(-1, index), values.gather(-1, index), v)


# zlaqr3.f 466-470 (DO 20 from IFST = I, IFST = J where CABS1(T(J,J)) > CABS1(T(IFST,IFST))) over `keys` where
# `within`: the first position of the largest, a NaN never taken past the first (0-based)
def _first_largest(keys, within):
    start = within.to(torch.int64).argmax(-1)
    first = keys.gather(-1, start[..., None])[..., 0]
    return torch.where(first.isnan(), start, torch.where(within & ~keys.isnan(), keys, -1).argmax(-1))


# Reference-LAPACK 3.12.1 SRC/iparmq.f 259-279: NS of NH = IHI - ILO + 1 for ISPEC 13, 15 and 16, its NH >= 150
# NINT(LOG(REAL(NH)) / LOG(TWO)) in single precision (TWO = 2.0, 247-248)
def _iparmq_shifts(nh):
    ns = torch.where(nh >= 30, 4, 2)
    ns = torch.where(nh >= 60, 10, ns)
    single = nh.to(torch.float32)
    rounded = torch.floor(torch.log(single) / torch.log(torch.full_like(single, 2.0)) + 0.5).to(nh.dtype)
    wide = nh >= 150
    ns = torch.where(wide, torch.clamp(nh // torch.where(wide, rounded, 1), min=10), ns)
    ns = torch.where(nh >= 590, 64, ns)
    ns = torch.where(nh >= 3000, 128, ns)
    ns = torch.where(nh >= 6000, 256, ns)
    return torch.clamp(ns - ns % 2, min=2)


# iparmq.f 260-280 and 305-311's NWR (ISPEC 13) and NSR (ISPEC 15) as xLAQR0 takes them (zlaqr0.f 363-374, dlaqr0.f
# 368-379): NS = _iparmq_shifts(NH), NWR = MIN(IHI - ILO + 1, (N - 1) / 3, MAX(2, NWR)) and NSR = MAX(2, MIN(NSR,
# (N - 3) / 6, IHI - ILO) made even); the one computation both xLAQR0's run (_laqr0) and its query (_laqr0_query) take
def _laqr0_windows(order, ilo, ihi):
    nh = ihi - ilo + 1
    shifts = _iparmq_shifts(nh)
    nwr = torch.minimum(torch.minimum(nh, (order - 1) // 3), torch.clamp(torch.where(nh <= 500, shifts,
                                                                                     3 * shifts // 2), min=2))
    nsr = torch.minimum(torch.minimum(shifts, (order - 3) // 6), ihi - ilo)
    return shifts, nwr, torch.clamp(nsr - nsr % 2, min=2)


# Reference-LAPACK 3.12.1 workspace queries (LWORK = -1) of xLAQR0 and xLAQR4 (zlaqr0.f 322-392, dlaqr0.f 327-397: N = 0
# one, N <= NTINY = 15 LWKOPT = 1, else _laqr0_windows at ILO = 1, IHI = N and MAX(3 NSR / 2, the query of xLAQR3, or
# xLAQR2 where `four`, with NW = NWR + 1, KTOP = 1, KBOT = N), and of xLAQR3 and xLAQR2 (zlaqr3.f 322-348, zlaqr2.f
# 323-342, dlaqr3.f and dlaqr2.f alike: JW = MIN(NW, KBOT - KTOP + 1), JW <= 2 one, else JW + MAX(LWK1, LWK2), for
# xLAQR3 the larger with LWK3, xLAQR4(JW)'s): LWK1 is xGEHRD(JW, 1, JW - 1)'s N NB + TSIZE (zgehrd.f 226-238: NH =
# JW - 1 > 1, NB = MIN(NBMAX = 64, ILAENV(1, 'xGEHRD') = 32, ilaenv.f 340-345), TSIZE = LDT NBMAX = 65 64), LWK2
# xUNMHR('R', 'N', JW, JW, 1, JW - 1)'s NW NB (zunmhr.f 246-255: NW = MAX(1, M) = JW, NB = ILAENV(1, 'xUNMQR') = 32,
# ilaenv.f 405-432)
def _laqr0_query(order, four):
    one = torch.ones_like(order)
    _, nwr, nsr = _laqr0_windows(order, one, order)
    jw = torch.minimum(nwr + 1, order)
    made = jw + torch.maximum(jw * 32 + 65 * 64, jw * 32)
    if not four:
        made = torch.maximum(made, _laqr0_query(jw, True))
    return torch.where(order <= 15, 1, torch.maximum(3 * nsr // 2, torch.where(jw <= 2, 1, made)))


# ATen's lwork for LAPACK_xGEEV(JOBVL = 'N', JOBVR) as the non-CPU method takes it (the CPU route's query is the
# native owner's LAPACK call): BatchLinearAlgebraKernel.cpp 253-257 at torch 2.14.0 git 08187d9e, lapack_work_to_int
# (202-212) of the query's WORK(1), MAX(1, CEIL(NEXTAFTER(WORK(1), +INF))) in the routine's precision `dtype`, of
# zgeev.f 264-300 (`whole`) or dgeev.f 278-321 at ORDER = N: N = 0 one; ZGEEV MAXWRK = N + 32 N (ILAENV(1, 'ZGEHRD')),
# with vectors the larger with N + 32 (N - 1) (ZUNGHR, ilaenv.f 419-425) and N + ZTREVC3's MAX(1, N + 2 N NB), NB =
# ILAENV(1, 'ZTREVC') = 64 (ztrevc3.f 326-327, ilaenv.f 472-477), then MAX(MAXWRK, HSWORK, MINWRK = 2 N); DGEEV MAXWRK
# = 2 N + 32 N, with vectors MINWRK = 4 N and the larger with 2 N + 32 (N - 1) (DORGHR), N + 1, N + HSWORK, N +
# DTREVC3's (dtrevc3.f 301-302) and 4 N, else MINWRK = 3 N and N + 1, N + HSWORK, then MAX(MAXWRK, MINWRK). HSWORK is
# xHSEQR's query (zhseqr.f 391-407, dhseqr.f alike: past its N = 0 return, MAX(xLAQR0's, MAX(1, N))). Its operands
# are the static configuration (ORDER = N = the static extent, JOBVR, the dtype): _geev_routine evaluates it once at
# specialization on 0-d constants (the tracer's constant propagation) and takes the int, the LWORK the method's
# routines then take; the run's own _laqr0_windows over the balanced ILO, IHI is the source's distinct ILAENV call
def _geev_lwork(order, vectors, whole, dtype):
    hswork = torch.maximum(_laqr0_query(order, False), torch.clamp(order, min=1))
    trevc = torch.clamp(order + 2 * order * 64, min=1)
    if whole:
        maxwrk = order + order * 32
        if vectors:
            maxwrk = torch.maximum(torch.maximum(maxwrk, order + (order - 1) * 32), order + trevc)
        maxwrk = torch.maximum(torch.maximum(maxwrk, hswork), 2 * order)
    else:
        maxwrk = torch.maximum(torch.maximum(2 * order + order * 32, order + 1), order + hswork)
        if vectors:
            maxwrk = torch.maximum(torch.maximum(maxwrk, 2 * order + (order - 1) * 32), order + trevc)
            maxwrk = torch.maximum(maxwrk, 4 * order)
        else:
            maxwrk = torch.maximum(maxwrk, 3 * order)
    work = torch.where(order == 0, 1, maxwrk).to(dtype)
    return torch.clamp(torch.ceil(torch.nextafter(work, torch.full_like(work, math.inf))), min=1).to(order.dtype)


# Reference-LAPACK 3.12.1 SRC/zlaqr3.f 361-577, ZLAQR3(WANTT, WANTZ, N, KTOP, KBOT, NW, H, ILOZ, IHIZ, Z, NS, ND, SH,
# V, NH, T, NV, WV) where `proceed`, N the member's ORDER, as zlaqr0.f 516-520 calls it: V = H(KV,1), T = H(KV,KT),
# WV = H(KWV,1), NH = NHO, NV = NVE. ZLAQR3 reaches H(KV,KT) and H(KV,1) only as its T and V: here SIZE x SIZE
# arrays (SIZE = (n - 1) / 3 >= NWMAX >= NW >= JW) taken from those H positions at entry and put back there after
# ZUNMHR, their last write before DO 60; DO 60, DO 70 and DO 80 then write each chunk's xGEMM into H's own WV and T
# positions and copy it out of them (_rows_chunk, _columns_chunk), chunk by chunk in the source's order.
# zlaqr2.f (`nested` False, zlaqr4.f 521's call) differs only at zlaqr3.f 416-425: ZLAHQR for every JW. 361-368 the
# KTOP > KBOT and NW < 1 returns (NS = ND = 0); 372-375 SMLNUM = SAFMIN (N / ULP); 379-385 JW, KWTOP, S; 387-403
# the KBOT = KWTOP window; 411-415 ZLACPY('U') and ZCOPY of the window into T, ZLASET V = I; 416-425 ZLAQR4 where
# JW > NMIN = ILAENV(12, 'ZLAQR3') = 75 (iparmq.f 288) and `nested` (the branch present where SIZE admits JW > 75),
# else ZLAHQR, on T with V; 429-455 DO 10 (the spike tip test, else ZTREXC(IFST = NS, ILST)); 457-458 S = 0 where
# NS = 0; 463-477 DO 30 by ZTREXC where NS < JW; 480-482 DO 40; 485-566 where NS < JW or S = 0: where NS > 1 and
# S /= 0 ZLARFG of CONJG(V(1,1:NS)), ZLASET('L') at T(3,1), ZLARF1F from the left and right on T and from the right
# on V, ZGEHRD(JW, 1, NS, T) (LWORK - JW); H(KWTOP,KWTOP-1) = S CONJG(V(1,1)) where KWTOP > 1; T back into the
# window; ZUNMHR('R', 'N', JW, NS, 1, NS, T, V) (LWORK - JW) where NS > 1 and S /= 0; DO 60 (rows LTOP..KWTOP-1, LTOP
# = 1 where WANTT, else KTOP), DO 70 (WANTT: columns KBOT+1..N) and DO 80 (WANTZ: Z's rows ILOZ..IHIZ); 569-577 ND =
# JW - NS, NS - INFQR. ZGEHRD and ZUNMHR are _gehrd and _unmhr at zlaqr3.f 505-506 and 521-523 (WORK(JW+1) and
# LWORK - JW). Returns H, Z, W (SH), NS, ND. dlaqr3.f and dlaqr2.f, the real routines, are _dlaqr3
def _laqr3(wantt, wantz, nested, size, order, ktop, kbot, nw, h, iloz, ihiz, z, w, kv, kt, nho, kwv, nve, proceed,
           lwork):
    limits = torch.finfo(h.real.dtype)
    ulp = limits.eps
    smlnum = limits.tiny * (order.to(h.real.dtype) / ulp)
    one = torch.ones_like(ktop)
    go = proceed & (ktop <= kbot) & (nw >= 1)
    jw = torch.minimum(nw, kbot - ktop + 1)
    kwtop = kbot - jw + 1
    s = torch.where(kwtop == ktop, 0, _entry(h, kwtop, kwtop - 1))
    corner = _entry(h, kwtop, kwtop)
    single = go & (kbot == kwtop)
    w = _at_put(w, kwtop, corner, single)
    negligible = single & (_abs1(s) <= _larger(smlnum, ulp * _abs1(corner)))
    h = _entry_put(h, kwtop, kwtop - 1, 0, negligible & (kwtop > ktop))
    main = go & (kbot != kwtop)
    ii = torch.arange(1, size + 1, device=h.device)
    within, square = _within(jw, size), _square_within(jw, size)
    upper = (ii[:, None] <= ii + 1) & square
    t = torch.where(upper & _mat(main), _region_at(h, kwtop, kwtop, size, size), _region_at(h, kv, kt, size, size))
    v = torch.where(square & _mat(main), torch.eye(size, dtype=h.dtype, device=h.device),
                    _region_at(h, kv, one, size, size))
    sh = _vec_at(w, kwtop, size)
    small, infqr = main, torch.zeros_like(ktop)
    if nested and size > 75:
        large = main & (jw > 75)
        t, sh, v, infqr = _laqr0(True, True, True, jw, one, jw, t, sh, one, jw, v, large, lwork)
        small = main & ~large
    t, sh, v, info = _zlahqr(True, True, jw, one, jw, t, sh, one, jw, v, small)
    infqr = torch.where(small, info, infqr)
    carried = (main & (infqr + 1 <= jw), t.contiguous(), v.contiguous(), jw.clone(), infqr + 1, infqr + 1)
    _, t, v, ns, _, _ = while_loop_op(_first_any, functools.partial(_laqr3_deflated, ulp), carried, (jw, s, smlnum))
    s = torch.where(ns == 0, 0, s)
    carried = (main & (ns < jw) & (infqr + 1 <= ns), t, v, infqr + 1)
    _, t, v, _ = while_loop_op(_first_any, _laqr3_sorted, carried, (jw, ns))
    sh = torch.where(_span(ii, infqr + 1, jw) & _vec(main), t.diagonal(dim1=-2, dim2=-1), sh)
    update = main & ((ns < jw) | (s == 0))
    spike = update & (ns > 1) & (s != 0)
    lead = _row(v, one).conj()
    beta, tau, made = _larfg_vector(lead[..., 0], torch.where(_span(ii, 2, ns), lead, 0))
    work = torch.where(ii == 1, _vec(beta), torch.where(_span(ii, 2, ns), made, 0))
    tau = torch.where(spike, tau, 0)
    t = torch.where((ii[:, None] >= ii + 2) & square & _mat(spike), 0, t)
    t = _larf1f(True, ns, jw, work, tau.conj(), t, one, one, spike)
    t = _larf1f(False, ns, ns, work, tau, t, one, one, spike)
    v = _larf1f(False, jw, ns, work, tau, v, one, one, spike)
    t, taus = _gehrd(jw, one, torch.where(spike, ns, one), t, lwork - jw)
    h = _entry_put(h, kwtop, kwtop - 1, s * _entry(v, one, one).conj(), update & (kwtop > 1))
    h = _region_put(h, kwtop, kwtop, t, upper & _mat(update))
    v = _unmhr(False, True, jw, ns, one, ns, t, taus, v, spike, lwork - jw)
    h = _region_put(h, kv, kt, t, square & _mat(main))
    h = _region_put(h, kv, one, v, square & _mat(main))
    ltop = one if wantt else ktop
    _, h, _, _ = while_loop_op(_first_any, functools.partial(_rows_chunk, False),
                               (update & (ltop <= kwtop - 1), h.contiguous(), h.new_zeros(()), ltop.clone()),
                               (kwtop - 1, nve, kwtop, jw, v, kwv))
    if wantt:
        _, h, _ = while_loop_op(_first_any, _columns_chunk, (update & (kbot + 1 <= order), h, kbot + 1),
                                (order, nho, kwtop, jw, v, kv, kt))
    if wantz:
        _, h, z, _ = while_loop_op(_first_any, functools.partial(_rows_chunk, True),
                                   (update & (iloz <= ihiz), h, z.contiguous(), iloz.clone()),
                                   (ihiz, nve, kwtop, jw, v, kwv))
    w = _vec_put(w, kwtop, sh, within & _vec(main))
    found = torch.where(single, torch.where(negligible, 0, 1), 0)
    return (h, z, w, torch.where(main, ns - infqr, found), torch.where(main, jw - ns, torch.where(negligible, 1, 0)))


# zlaqr3.f 431-455, DO 10's pass for KNT: FOO = CABS1(T(NS,NS)) (CABS1(S) where 0); NS = NS - 1 where CABS1(S)
# CABS1(V(1,NS)) <= MAX(SMLNUM, ULP FOO), else ZTREXC('V', JW, T, V, IFST = NS, ILST) and ILST = ILST + 1
def _laqr3_deflated(ulp, live, t, v, ns, ilst, knt, jw, s, smlnum):
    foo = _abs1(_entry(t, ns, ns))
    foo = torch.where(foo == 0, _abs1(s), foo)
    held = _abs1(s) * _abs1(_entry(v, torch.ones_like(ns), ns)) <= _larger(smlnum, ulp * foo)
    move = live & ~held
    t, v = _ztrexc(True, jw, t, v, ns, ilst, move)
    knt = torch.where(live, knt + 1, knt)
    return (live & (knt <= jw), t, v, torch.where(live & held, ns - 1, ns), torch.where(move, ilst + 1, ilst), knt)


# zlaqr3.f 465-476, DO 30's pass for I: IFST by DO 20 over I..NS, ZTREXC('V', JW, T, V, IFST, ILST = I) where IFST /= I
def _laqr3_sorted(live, t, v, i, jw, ns):
    ifst = _first_largest(_abs1(t.diagonal(dim1=-2, dim2=-1)), _span(_counted(t), i, ns)) + 1
    t, v = _ztrexc(True, jw, t, v, ifst, i, live & (ifst != i))
    i = torch.where(live, i + 1, i)
    return live & (i <= ns), t, v, i


# Reference-LAPACK 3.12.1 SRC/dlaqr3.f 360-693, DLAQR3(WANTT, WANTZ, N, KTOP, KBOT, NW, H, ILOZ, IHIZ, Z, NS, ND, SR,
# SI, V, NH, T, NV, WV) where `proceed`, as _laqr3 takes zlaqr3.f (dlaqr0.f 520-524's call: T, V and WV the same H
# positions, SIZE the same static extent). dlaqr2.f (`nested` False, dlaqr4.f 526's call) differs only at dlaqr3.f
# 416-423: DLAHQR for every JW (dlaqr2.f 411-412). 360-367 the KTOP > KBOT and NW < 1 returns (NS = ND = 0); 371-374
# SMLNUM = SAFMIN (N / ULP); 378-384 JW, KWTOP, S; 386-403 the KBOT = KWTOP window (SR = H(KWTOP,KWTOP), SI = 0, the
# ABS test); 411-415 DLACPY('U') and DCOPY of the window into T, DLASET V = I; 416-423 DLAQR4 where JW > NMIN =
# ILAENV(12, 'DLAQR3') = 75 and `nested` (the branch present where SIZE admits JW > 75), else DLAHQR, on T with V
# into SR(KWTOP), SI(KWTOP); 427-432 DO 10's clean margin (T(J+2,J) = T(J+3,J) = 0 for J <= JW - 3, T(JW,JW-2) = 0
# where JW > 2); 436-502 NS = JW, ILST = INFQR + 1 and label 20 (_dlaqr3_deflated); 506-507 S = 0 where NS = 0;
# 509-575 where NS < JW, SORTED = .FALSE., I = NS + 1 and label 30 (_dlaqr3_sorted); 579-601 I = JW and label 60
# (_dlaqr3_shifts); 603-681 where NS < JW or S = 0: where NS > 1 and S /= 0 DLARFG of V(1,1:NS), DLASET('L') at
# T(3,1), DLARF1F from the left and right on T and from the right on V, DGEHRD(JW, 1, NS, T) (LWORK - JW);
# H(KWTOP,KWTOP-1) = S V(1,1) where KWTOP > 1; T back into the window; DORMHR('R', 'N', JW, NS, 1, NS, T, V)
# (LWORK - JW) where NS > 1 and S /= 0; DO 70 (rows LTOP..KWTOP-1), DO 80 (WANTT: columns KBOT+1..N, DGEMM('C') the
# transpose) and DO 90 (WANTZ: Z's rows ILOZ..IHIZ) as _laqr3's DO 60, 70 and 80; 685-693 ND = JW - NS, NS - INFQR.
# DGEHRD and DORMHR are _gehrd and _unmhr at dlaqr3.f 622-623 and 638-640 (WORK(JW+1), LWORK - JW).
# Returns H, Z, WR (SR), WI (SI), NS, ND
def _dlaqr3(wantt, wantz, nested, size, order, ktop, kbot, nw, h, iloz, ihiz, z, wr, wi, kv, kt, nho, kwv, nve,
            proceed, lwork):
    limits = torch.finfo(h.dtype)
    ulp = limits.eps
    smlnum = limits.tiny * (order.to(h.dtype) / ulp)
    one = torch.ones_like(ktop)
    go = proceed & (ktop <= kbot) & (nw >= 1)
    jw = torch.minimum(nw, kbot - ktop + 1)
    kwtop = kbot - jw + 1
    s = torch.where(kwtop == ktop, 0, _entry(h, kwtop, kwtop - 1))
    corner = _entry(h, kwtop, kwtop)
    single = go & (kbot == kwtop)
    wr, wi = _at_put(wr, kwtop, corner, single), _at_put(wi, kwtop, 0, single)
    negligible = single & (s.abs() <= _larger(smlnum, ulp * corner.abs()))
    h = _entry_put(h, kwtop, kwtop - 1, 0, negligible & (kwtop > ktop))
    main = go & (kbot != kwtop)
    ii = torch.arange(1, size + 1, device=h.device)
    within, square = _within(jw, size), _square_within(jw, size)
    upper = (ii[:, None] <= ii + 1) & square
    t = torch.where(upper & _mat(main), _region_at(h, kwtop, kwtop, size, size), _region_at(h, kv, kt, size, size))
    v = torch.where(square & _mat(main), torch.eye(size, dtype=h.dtype, device=h.device),
                    _region_at(h, kv, one, size, size))
    sr, si = _vec_at(wr, kwtop, size), _vec_at(wi, kwtop, size)
    small, infqr = main, torch.zeros_like(ktop)
    if nested and size > 75:
        large = main & (jw > 75)
        t, sr, si, v, infqr = _dlaqr0(True, True, True, jw, one, jw, t, sr, si, one, jw, v, large, lwork)
        small = main & ~large
    t, sr, si, v, info = _dlahqr(True, True, jw, one, jw, t, sr, si, one, jw, v, small)
    infqr = torch.where(small, info, infqr)
    below = ii[:, None] - ii
    margin = (((below == 2) | (below == 3)) & (ii <= _mat(jw - 3))) | (
        (ii[:, None] == _mat(jw)) & (ii == _mat(jw - 2)) & _mat(jw > 2))
    t = torch.where(margin & _mat(main), 0, t)
    carried = (main & (infqr + 1 <= jw), t.contiguous(), v.contiguous(), jw.clone(), infqr + 1)
    _, t, v, ns, _ = while_loop_op(_first_any, functools.partial(_dlaqr3_deflated, ulp), carried, (jw, s, smlnum))
    s = torch.where(ns == 0, 0, s)
    carried = (main & (ns < jw), t, v, ns + 1, torch.zeros_like(main))
    _, t, v, _, _ = while_loop_op(_first_any, _dlaqr3_sorted, carried, (jw, ns, infqr))
    carried = (main & (jw >= infqr + 1), sr.contiguous(), si.contiguous(), jw.clone())
    _, sr, si, _ = while_loop_op(_first_any, _dlaqr3_shifts, carried, (t, infqr))
    update = main & ((ns < jw) | (s == 0))
    spike = update & (ns > 1) & (s != 0)
    lead = _row(v, one)
    beta, tau, made = _larfg_vector(lead[..., 0], torch.where(_span(ii, 2, ns), lead, 0))
    work = torch.where(ii == 1, _vec(beta), torch.where(_span(ii, 2, ns), made, 0))
    tau = torch.where(spike, tau, 0)
    t = torch.where((ii[:, None] >= ii + 2) & square & _mat(spike), 0, t)
    t = _larf1f(True, ns, jw, work, tau, t, one, one, spike)
    t = _larf1f(False, ns, ns, work, tau, t, one, one, spike)
    v = _larf1f(False, jw, ns, work, tau, v, one, one, spike)
    t, taus = _gehrd(jw, one, torch.where(spike, ns, one), t, lwork - jw)
    h = _entry_put(h, kwtop, kwtop - 1, s * _entry(v, one, one), update & (kwtop > 1))
    h = _region_put(h, kwtop, kwtop, t, upper & _mat(update))
    v = _unmhr(False, True, jw, ns, one, ns, t, taus, v, spike, lwork - jw)
    h = _region_put(h, kv, kt, t, square & _mat(main))
    h = _region_put(h, kv, one, v, square & _mat(main))
    ltop = one if wantt else ktop
    _, h, _, _ = while_loop_op(_first_any, functools.partial(_rows_chunk, False),
                               (update & (ltop <= kwtop - 1), h.contiguous(), h.new_zeros(()), ltop.clone()),
                               (kwtop - 1, nve, kwtop, jw, v, kwv))
    if wantt:
        _, h, _ = while_loop_op(_first_any, _columns_chunk, (update & (kbot + 1 <= order), h, kbot + 1),
                                (order, nho, kwtop, jw, v, kv, kt))
    if wantz:
        _, h, z, _ = while_loop_op(_first_any, functools.partial(_rows_chunk, True),
                                   (update & (iloz <= ihiz), h, z.contiguous(), iloz.clone()),
                                   (ihiz, nve, kwtop, jw, v, kwv))
    placed = within & _vec(main)
    wr, wi = _vec_put(wr, kwtop, sr, placed), _vec_put(wi, kwtop, si, placed)
    found = torch.where(single, torch.where(negligible, 0, 1), 0)
    return (h, z, wr, wi, torch.where(main, ns - infqr, found),
            torch.where(main, jw - ns, torch.where(negligible, 1, 0)))


# dlaqr3.f 438-502, label 20's pass while ILST <= NS: BULGE = T(NS,NS-1) /= 0 (.FALSE. where NS = 1); FOO =
# ABS(T(NS,NS)), where BULGE plus SQRT(ABS(T(NS,NS-1))) SQRT(ABS(T(NS-1,NS))), ABS(S) where FOO = 0; NS = NS - 1 (2
# where BULGE) where ABS(S V(1,NS)) (where BULGE its MAX with ABS(S V(1,NS-1))) <= MAX(SMLNUM, ULP FOO), else
# DTREXC('V', JW, T, V, IFST = NS, ILST) and ILST = DTREXC's ILST + 1 (+ 2 where BULGE)
def _dlaqr3_deflated(ulp, live, t, v, ns, ilst, jw, s, smlnum):
    one = torch.ones_like(ns)
    bulge = (ns != 1) & (_entry(t, ns, ns - 1) != 0)
    tip = _entry(t, ns, ns).abs()
    foo = torch.where(bulge, tip + _entry(t, ns, ns - 1).abs().sqrt() * _entry(t, ns - 1, ns).abs().sqrt(), tip)
    foo = torch.where(foo == 0, s.abs(), foo)
    spike = (s * _entry(v, one, ns)).abs()
    spike = torch.where(bulge, _larger(spike, (s * _entry(v, one, ns - 1)).abs()), spike)
    held = spike <= _larger(smlnum, ulp * foo)
    move = live & ~held
    t, v, moved, _ = _dtrexc(True, jw, t, v, ns, ilst, move)
    width = torch.where(bulge, 2, 1)
    ns = torch.where(live & held, ns - width, ns)
    ilst = torch.where(move, moved + width, ilst)
    return live & (ilst <= ns), t, v, ns, ilst


# dlaqr3.f 517-573, label 30's pass while not SORTED: SORTED = .TRUE., KEND = I - 1, I = INFQR + 1, K = I + 1 where
# I = NS or T(I+1,I) = 0, else I + 2, then label 40 (_dlaqr3_bubbled) while K <= KEND
def _dlaqr3_sorted(live, t, v, i, sorted_, jw, ns, infqr):
    kend = i - 1
    i = torch.where(live, infqr + 1, i)
    k = torch.where((i == ns) | (_entry(t, i + 1, i) == 0), i + 1, i + 2)
    sorted_ = live | sorted_
    carried = (live & (k <= kend), t, v, i, k, sorted_)
    _, t, v, i, _, sorted_ = while_loop_op(_first_any, _dlaqr3_bubbled, carried, (kend, jw))
    return live & ~sorted_, t, v, i, sorted_


# dlaqr3.f 531-571, label 40's pass for I and K: EVI = ABS(T(I,I)) where K = I + 1, else plus SQRT(ABS(T(I+1,I)))
# SQRT(ABS(T(I,I+1))); EVK of K alike, ABS(T(K,K)) where K = KEND or T(K+1,K) = 0; I = K where EVI >= EVK, else
# SORTED = .FALSE., DTREXC('V', JW, T, V, IFST = I, ILST = K) and I = its ILST where its INFO = 0, else K; then K =
# I + 1 where I = KEND or T(I+1,I) = 0, else I + 2
def _dlaqr3_bubbled(live, t, v, i, k, sorted_, kend, jw):
    tii, tkk = _entry(t, i, i).abs(), _entry(t, k, k).abs()
    evi = torch.where(k == i + 1, tii, tii + _entry(t, i + 1, i).abs().sqrt() * _entry(t, i, i + 1).abs().sqrt())
    evk = torch.where((k == kend) | (_entry(t, k + 1, k) == 0), tkk,
                      tkk + _entry(t, k + 1, k).abs().sqrt() * _entry(t, k, k + 1).abs().sqrt())
    kept = evi >= evk
    move = live & ~kept
    t, v, ilst, info = _dtrexc(True, jw, t, v, i, k, move)
    i = torch.where(live & kept, k, torch.where(move, torch.where(info == 0, ilst, k), i))
    k = torch.where(live, torch.where((i == kend) | (_entry(t, i + 1, i) == 0), i + 1, i + 2), k)
    return live & (k <= kend), t, v, i, k, sorted_ & ~move


# dlaqr3.f 581-600, label 60's pass while I >= INFQR + 1, SR and SI the window's from KWTOP: SR(I) = T(I,I) and SI(I)
# = 0 where I = INFQR + 1 or T(I,I-1) = 0 (I = I - 1), else DLANV2 of T(I-1:I,I-1:I) into SR, SI(I-1) and SR, SI(I)
# (I = I - 2)
def _dlaqr3_shifts(live, sr, si, i, t, infqr):
    lone = (i == infqr + 1) | (_entry(t, i, i - 1) == 0)
    pair, alone = live & ~lone, live & lone
    _, _, _, _, rt1r, rt1i, rt2r, rt2i, _, _ = _dlanv2(_entry(t, i - 1, i - 1), _entry(t, i - 1, i),
                                                       _entry(t, i, i - 1), _entry(t, i, i))
    sr = _at_put(_at_put(_at_put(sr, i, _entry(t, i, i), alone), i - 1, rt1r, pair), i, rt2r, pair)
    si = _at_put(_at_put(_at_put(si, i, 0, alone), i - 1, rt1i, pair), i, rt2i, pair)
    i = torch.where(live, i - torch.where(lone, 1, 2), i)
    return live & (i >= infqr + 1), sr, si, i


# Reference-LAPACK 3.12.1 SRC/zlaqr0.f 323-693, ZLAQR0(WANTT, WANTZ, N, ILO, IHI, H, W, ILOZ, IHIZ, Z) where
# `proceed`, N the member's ORDER within h's static extent n. zlaqr4.f (`four`: zlaqr3.f 418 and zlaqr0.f 573's
# calls) differs only at zlaqr0.f 381 and 516 (ZLAQR2 for ZLAQR3) and 572-580 (ZLAHQR for every NS). 323-327 the
# N = 0 return; 328-335 N <= NTINY = 15 (263), ZLAHQR (alone where n <= 15); 355-372 iparmq.f's NWR (ISPEC 13: NS
# where NH <= KNWSWP = 500, else 3 NS / 2) and NSR (15: NS) of NH = IHI - ILO + 1 (ilaenv.f 773's N2, N3); 397-410
# NMIN = 75 (12), NIBBLE = 14 (14), KACC22 = 2 where its NS >= KACMIN = K22MIN = 14 (16: iparmq.f 384-389 for
# 'LAQR'), else 0, so ACCUM (zlaqr5.f 339) is NS >= 14; 415-422 NWMAX = MIN((N - 1) / 3, LWORK / 2) and NSMAX =
# MIN((N - 3) / 6, 2 LWORK / 3) made even, LWORK the static `lwork`: _geev_lwork's at specialization, passed whole to
# ZHSEQR (zgeev.f 377-379, 410-412 and 420-422: IWRK = ITAU = 1; DHSEQR LWORK - N) and on by zhseqr.f 441-443 and
# 464-465, zlaqr0.f 516-519 and 572-574 and zlaqr3.f 418-419, or NL = 49 on zhseqr.f 479-481's HL; 431 ITMAX =
# MAX(30, 2 KEXSH) MAX(10, NH); 435 KBOT = IHI; 439-690 DO 70 as one while_loop whose pass is one IT; 693 INFO = KBOT
# once IT passes ITMAX (also where that pass left KBOT < ILO). Returns H, W, Z, INFO
def _laqr0(wantt, wantz, four, order, ilo, ihi, h, w, iloz, ihiz, z, proceed, lwork):
    n = h.shape[-1]
    go = proceed & (order > 0)
    h, w, z, info = _zlahqr(wantt, wantz, order, ilo, ihi, h, w, iloz, ihiz, z, go & (order <= 15))
    go = go & (order > 15)
    if n <= 15:
        return h, w, z, info
    nh = ihi - ilo + 1
    shifts, nwr, nsr = _laqr0_windows(order, ilo, ihi)
    nwmax = torch.clamp((order - 1) // 3, max=lwork // 2)
    nsmax = torch.clamp((order - 3) // 6, max=2 * lwork // 3)
    nsmax = nsmax - nsmax % 2
    itmax = 30 * torch.clamp(nh, min=10)
    ones = torch.ones_like(ihi)
    carried = (go & (ihi >= ilo), h.contiguous(), z.contiguous() if wantz else h.new_zeros(()), w.contiguous(),
               ihi.clone(), nwmax.clone(), ones, -ones, ones.clone())
    _, h, z, w, kbot, _, _, _, it = while_loop_op(_first_any, functools.partial(_laqr0_pass, wantt, wantz, four, lwork),
                                                  carried, (order, ilo, iloz, ihiz, itmax, nwr, nsr, nwmax, nsmax,
                                                            shifts >= 14))
    return h, w, z, torch.where(go, torch.where(it > itmax, kbot, 0), info)


# zlaqr0.f 439-690, DO 70's pass for IT: 448-454 KTOP (DO 10); 471-496 NH, NWUPBD, NW (KEXNW = 5) and NDEC; 507-511
# KV, KT, NHO, KWV, NVE; 516-520 ZLAQR3 (ZLAQR2 where `four`; its static SIZE (n - 1) / 3 >= NWMAX); 524 KBOT - LD;
# 528 KS; 536-678 where LD = 0 or 100 LD <= NW NIBBLE and KBOT - KTOP + 1 > MIN(NMIN, NWMAX): 543-544 NS; 552-557
# MOD(NDFL, KEXSH = 6) = 0: KS = KBOT - NS + 1, then DO 30 on H as ZLAQR3 left it (_laqr0_exceptional), else
# 567-605 where KBOT - KS + 1 <= NS / 2 the NS x NS ZLACPY to H(KT,1) (static NSB x NSB, NSB = (n - 3) / 6 made even
# >= NSMAX >= NS) and ZLAQR4 (NS > NMIN, not `four`; present where NSB admits NS > 75) or ZLAHQR of it into W(KS),
# KS + INF, and where KS >= KBOT the 2 x 2 shifts (S, AA..DD = H / S, TR2, DET, RTDISC = SQRT(-DET), KS = KBOT - 1);
# 609-628 where KBOT - KS + 1 > NS DO 50/40's bubble sort; 635-642 KBOT - KS + 1 = 2's nearer shift; 649-668 NS, KS,
# KDU, KU, KWH, NHO, KWV, NVE; 673-676 ZLAQR5; 681-685 NDFL
def _laqr0_pass(wantt, wantz, four, lwork, live, h, z, w, kbot, nw, ndfl, ndec, it, order, ilo, iloz, ihiz, itmax, nwr,
                nsr, nwmax, nsmax, accum):
    n, k, one = h.shape[-1], _counted(h), torch.ones_like(kbot)
    nsb = min((n - 3) // 6, 2 * lwork // 3)
    nsb = max(2, nsb - nsb % 2)
    sub = torch.cat([torch.zeros_like(h[..., :1, 0]), h.diagonal(-1, dim1=-2, dim2=-1)], -1)
    found = torch.where((sub == 0) & (k >= _vec(ilo + 1)) & (k <= _vec(kbot)), k, 0).amax(-1)
    ktop = torch.where(found > 0, found, ilo)
    nh = kbot - ktop + 1
    nwupbd = torch.minimum(nh, nwmax)
    nw = torch.where(ndfl < 5, torch.minimum(nwupbd, nwr), torch.minimum(nwupbd, 2 * nw))
    below = nw < nwmax
    whole = below & (nw >= nh - 1)
    kwtop = kbot - nw + 1
    grow = below & ~whole & (_abs1(_entry(h, kwtop, kwtop - 1)) > _abs1(_entry(h, kwtop - 1, kwtop - 2)))
    nw = torch.where(whole, nh, torch.where(grow, nw + 1, nw))
    bump = (ndfl >= 5) & ((ndec >= 0) | (nw >= nwupbd))
    ndec = torch.where(ndfl < 5, -1, torch.where(bump, ndec + 1, ndec))
    ndec = torch.where(bump & (nw - ndec < 2), 0, ndec)
    nw = torch.where(bump, nw - ndec, nw)
    kt = nw + 1
    h, z, w, ls, ld = _laqr3(wantt, wantz, not four, max(1, min((n - 1) // 3, lwork // 2)), order, ktop, kbot, nw, h,
                             iloz, ihiz, z, w, order - nw + 1, kt, (order - nw - 1) - kt + 1, nw + 2,
                             (order - nw) - (nw + 2) + 1, live, lwork)
    kbot = torch.where(live, kbot - ld, kbot)
    ks = kbot - ls + 1
    sweep = live & ((ld == 0) | ((100 * ld <= nw * 14) & (kbot - ktop + 1 > torch.clamp(nwmax, max=75))))
    ns = torch.minimum(torch.minimum(nsmax, nsr), torch.clamp(kbot - ktop, min=2))
    ns = ns - ns % 2
    exceptional = sweep & (ndfl % 6 == 0)
    computed = sweep & ~exceptional
    first = kbot - ns + 1
    ks = torch.where(exceptional, first, ks)
    _, w, _ = while_loop_op(_first_any, _laqr0_exceptional, (exceptional & (kbot >= ks + 1), w, kbot.clone()),
                            (h, ks + 1))
    few = computed & (kbot - ks + 1 <= ns // 2)
    top = order - ns + 1
    square = _square_within(ns, nsb)
    h = _region_put(h, top, one, _region_at(h, first, first, nsb, nsb), square & _mat(few))
    hs, ws, small, inf = _region_at(h, top, one, nsb, nsb), _vec_at(w, first, nsb), few, torch.zeros_like(kbot)
    if not four and nsb > 75:
        large = few & (ns > 75)
        hs, ws, _, inf = _laqr0(False, False, True, ns, one, ns, hs, ws, one, one, hs.new_zeros(()), large, lwork)
        small = few & ~large
    hs, ws, _, info = _zlahqr(False, False, ns, one, ns, hs, ws, one, one, hs.new_zeros(()), small)
    inf = torch.where(small, info, inf)
    h = _region_put(h, top, one, hs, square & _mat(few))
    w = _vec_put(w, first, ws, _within(ns, nsb) & _vec(few))
    ks = torch.where(few, first + inf, ks)
    pair = few & (ks >= kbot)
    a, b, c, d = (_entry(h, kbot - 1, kbot - 1), _entry(h, kbot - 1, kbot), _entry(h, kbot, kbot - 1),
                  _entry(h, kbot, kbot))
    s = _abs1(a) + _abs1(c) + _abs1(b) + _abs1(d)
    aa, bb, cc, dd = _over(a, s), _over(b, s), _over(c, s), _over(d, s)
    tr2 = _over(aa + dd, 2)
    rtdisc = (-((aa - tr2) * (dd - tr2) - bb * cc)).sqrt()
    w = _at_put(_at_put(w, kbot - 1, _scaled(tr2 + rtdisc, s), pair), kbot, _scaled(tr2 - rtdisc, s), pair)
    ks = torch.where(pair, kbot - 1, ks)
    carried = (computed & (kbot - ks + 1 > ns), w, kbot.clone(), torch.zeros_like(live))
    _, w, _, _ = while_loop_op(_first_any, _laqr0_sorted, carried, (ks,))
    two = sweep & (kbot - ks + 1 == 2)
    last, prior, corner = _at(w, kbot), _at(w, kbot - 1), _entry(h, kbot, kbot)
    nearer = _abs1(last - corner) < _abs1(prior - corner)
    w = _at_put(_at_put(w, kbot - 1, last, two & nearer), kbot, prior, two & ~nearer)
    ns = torch.minimum(ns, kbot - ks + 1)
    ns = ns - ns % 2
    ks = kbot - ns + 1
    kdu = 2 * ns
    h, z, _ = _laqr5(wantt, wantz, accum, order, ktop, kbot, ns, (_vec_at(w, ks, nsb),), h, iloz, ihiz, z,
                     order - kdu + 1, order - kdu - (kdu + 4) + 1, kdu + 4, (order - kdu + 1 - 4) - (kdu + 1) + 1,
                     kdu + 1, sweep)
    ndfl = torch.where(live, torch.where(ld > 0, 1, ndfl + 1), ndfl)
    it = torch.where(live, it + 1, it)
    return (live & (kbot >= ilo) & (it <= itmax), h, z if wantz else z.clone(), w, kbot, nw, ndfl, ndec, it)


# zlaqr0.f 554-557, DO 30's pass for I (KBOT, KS + 1, -2), H the one ZLAQR3 left: W(I) = H(I,I) + WILK1
# CABS1(H(I,I-1)) (WILK1 = 0.75d0, the real product added to H(I,I)'s real part), then W(I-1) = W(I)
def _laqr0_exceptional(live, w, i, h, stop):
    raised = _plus_real(_entry(h, i, i), 0.75 * _abs1(_entry(h, i, i - 1)))
    w = _at_put(_at_put(w, i, raised, live), i - 1, raised, live)
    i = torch.where(live, i - 2, i)
    return live & (i >= stop), w, i


# zlaqr0.f 611-626, DO 50's pass for K (KBOT, KS + 1, -1) until SORTED: SORTED = .TRUE., then DO 40
def _laqr0_sorted(live, w, k, sorted_, ks):
    _, w, _, sorted_ = while_loop_op(_first_any, _laqr0_swapped, (live & (ks <= k - 1), w, ks.clone(),
                                                                  live | sorted_), (k,))
    k = torch.where(live, k - 1, k)
    return live & (k >= ks + 1) & ~sorted_, w, k, sorted_


# DO 40's pass for I (KS..K-1): W(I) and W(I+1) swapped where CABS1(W(I)) < CABS1(W(I+1)), SORTED = .FALSE.
def _laqr0_swapped(live, w, i, sorted_, k):
    first, second = _at(w, i), _at(w, i + 1)
    swap = live & (_abs1(first) < _abs1(second))
    w = _at_put(_at_put(w, i, second, swap), i + 1, first, swap)
    i = torch.where(live, i + 1, i)
    return live & (i <= k - 1), w, i, sorted_ & ~swap


# Reference-LAPACK 3.12.1 SRC/dlaqr0.f 323-730, DLAQR0(WANTT, WANTZ, N, ILO, IHI, H, WR, WI, ILOZ, IHIZ, Z) where
# `proceed`, as _laqr0 takes zlaqr0.f (NWR, NSR, NMIN, NIBBLE, KACC22, NWMAX, NSMAX and ITMAX at dlaqr0.f 368-379,
# 402-427 and 435, the static LWORK alike). dlaqr4.f (`four`: dlaqr3.f 418 and dlaqr0.f 589's calls) differs only at
# dlaqr0.f 385 and 520 (DLAQR2 for DLAQR3, dlaqr4.f 391 and 526) and 588-597 (DLAHQR for every NS, dlaqr4.f 594-596).
# 327-330 the N = 0 return; 332-339 N <= NTINY = 15 (278), DLAHQR (alone where n <= 15); 439 KBOT = IHI; 443-725 DO
# 80 as one while_loop whose pass is one IT (_dlaqr0_pass); 730 INFO = KBOT once IT passes ITMAX (also where that
# pass left KBOT < ILO). Returns H, WR, WI, Z, INFO
def _dlaqr0(wantt, wantz, four, order, ilo, ihi, h, wr, wi, iloz, ihiz, z, proceed, lwork):
    n = h.shape[-1]
    go = proceed & (order > 0)
    h, wr, wi, z, info = _dlahqr(wantt, wantz, order, ilo, ihi, h, wr, wi, iloz, ihiz, z, go & (order <= 15))
    go = go & (order > 15)
    if n <= 15:
        return h, wr, wi, z, info
    nh = ihi - ilo + 1
    shifts, nwr, nsr = _laqr0_windows(order, ilo, ihi)
    nwmax = torch.clamp((order - 1) // 3, max=lwork // 2)
    nsmax = torch.clamp((order - 3) // 6, max=2 * lwork // 3)
    nsmax = nsmax - nsmax % 2
    itmax = 30 * torch.clamp(nh, min=10)
    ones = torch.ones_like(ihi)
    carried = (go & (ihi >= ilo), h.contiguous(), z.contiguous() if wantz else h.new_zeros(()), wr.contiguous(),
               wi.contiguous(), ihi.clone(), nwmax.clone(), ones, -ones, ones.clone())
    _, h, z, wr, wi, kbot, _, _, _, it = while_loop_op(
        _first_any, functools.partial(_dlaqr0_pass, wantt, wantz, four, lwork), carried,
        (order, ilo, iloz, ihiz, itmax, nwr, nsr, nwmax, nsmax, shifts >= 14))
    return h, wr, wi, z, torch.where(go, torch.where(it > itmax, kbot, 0), info)


# dlaqr0.f 443-725, DO 80's pass for IT, as _laqr0_pass (ABS for CABS1): 447-458 KTOP (DO 10); 476-499 NH, NWUPBD, NW
# (KEXNW = 5) and NDEC; 512-516 KV, KT, NHO, KWV, NVE; 520-524 DLAQR3 (DLAQR2 where `four`); 528 KBOT - LD; 532 KS;
# 540-714 where LD = 0 or 100 LD <= NW NIBBLE and KBOT - KTOP + 1 > MIN(NMIN, NWMAX): 547-548 NS; 557-574 MOD(NDFL,
# KEXSH = 6) = 0: KS = KBOT - NS + 1, DO 30 on H as DLAQR3 left it (_dlaqr0_exceptional), then where KS = KTOP
# WR(KS+1) = H(KS+1,KS+1), WI(KS+1) = 0, WR(KS) = WR(KS+1), WI(KS) = WI(KS+1); else 583-614 where KBOT - KS + 1 <=
# NS / 2 the NS x NS DLACPY to H(KT,1) (static NSB x NSB as _laqr0_pass) and DLAQR4 (NS > NMIN, not `four`; present
# where NSB admits NS > 75) or DLAHQR of it into WR(KS), WI(KS), KS + INF, and where KS >= KBOT DLANV2 of
# H(KBOT-1:KBOT,KBOT-1:KBOT) into WR, WI(KBOT-1) and (KBOT), KS = KBOT - 1; 616-643 where KBOT - KS + 1 > NS DO 50/40's
# bubble sort; 651-664 DO 70; 670-679 KBOT - KS + 1 = 2 with WI(KBOT) = 0's nearer real shift; 686-706 NS, KS, KDU,
# KU, KWH, NHO, KWV, NVE; 710-713 DLAQR5 on WR(KS), WI(KS), its DO 10 (dlaqr5.f 330-343) pairing them in place;
# 718-722 NDFL
def _dlaqr0_pass(wantt, wantz, four, lwork, live, h, z, wr, wi, kbot, nw, ndfl, ndec, it, order, ilo, iloz, ihiz,
                 itmax, nwr, nsr, nwmax, nsmax, accum):
    n, k, one = h.shape[-1], _counted(h), torch.ones_like(kbot)
    nsb = min((n - 3) // 6, 2 * lwork // 3)
    nsb = max(2, nsb - nsb % 2)
    sub = torch.cat([torch.zeros_like(h[..., :1, 0]), h.diagonal(-1, dim1=-2, dim2=-1)], -1)
    found = torch.where((sub == 0) & (k >= _vec(ilo + 1)) & (k <= _vec(kbot)), k, 0).amax(-1)
    ktop = torch.where(found > 0, found, ilo)
    nh = kbot - ktop + 1
    nwupbd = torch.minimum(nh, nwmax)
    nw = torch.where(ndfl < 5, torch.minimum(nwupbd, nwr), torch.minimum(nwupbd, 2 * nw))
    below = nw < nwmax
    whole = below & (nw >= nh - 1)
    kwtop = kbot - nw + 1
    grow = below & ~whole & (_entry(h, kwtop, kwtop - 1).abs() > _entry(h, kwtop - 1, kwtop - 2).abs())
    nw = torch.where(whole, nh, torch.where(grow, nw + 1, nw))
    bump = (ndfl >= 5) & ((ndec >= 0) | (nw >= nwupbd))
    ndec = torch.where(ndfl < 5, -1, torch.where(bump, ndec + 1, ndec))
    ndec = torch.where(bump & (nw - ndec < 2), 0, ndec)
    nw = torch.where(bump, nw - ndec, nw)
    kt = nw + 1
    h, z, wr, wi, ls, ld = _dlaqr3(wantt, wantz, not four, max(1, min((n - 1) // 3, lwork // 2)), order, ktop, kbot,
                                   nw, h, iloz, ihiz, z, wr, wi, order - nw + 1, kt, (order - nw - 1) - kt + 1,
                                   nw + 2, (order - nw) - (nw + 2) + 1, live, lwork)
    kbot = torch.where(live, kbot - ld, kbot)
    ks = kbot - ls + 1
    sweep = live & ((ld == 0) | ((100 * ld <= nw * 14) & (kbot - ktop + 1 > torch.clamp(nwmax, max=75))))
    ns = torch.minimum(torch.minimum(nsmax, nsr), torch.clamp(kbot - ktop, min=2))
    ns = ns - ns % 2
    exceptional = sweep & (ndfl % 6 == 0)
    computed = sweep & ~exceptional
    first = kbot - ns + 1
    ks = torch.where(exceptional, first, ks)
    stop = torch.maximum(ks + 1, ktop + 2)
    _, wr, wi, _ = while_loop_op(_first_any, _dlaqr0_exceptional,
                                 (exceptional & (kbot >= stop), wr, wi, kbot.clone()), (h, stop))
    bottom = exceptional & (ks == ktop)
    corner = _entry(h, ks + 1, ks + 1)
    wr = _at_put(_at_put(wr, ks + 1, corner, bottom), ks, corner, bottom)
    wi = _at_put(_at_put(wi, ks + 1, 0, bottom), ks, 0, bottom)
    few = computed & (kbot - ks + 1 <= ns // 2)
    top = order - ns + 1
    square = _square_within(ns, nsb)
    h = _region_put(h, top, one, _region_at(h, first, first, nsb, nsb), square & _mat(few))
    hs, rs, js = _region_at(h, top, one, nsb, nsb), _vec_at(wr, first, nsb), _vec_at(wi, first, nsb)
    small, inf = few, torch.zeros_like(kbot)
    if not four and nsb > 75:
        large = few & (ns > 75)
        hs, rs, js, _, inf = _dlaqr0(False, False, True, ns, one, ns, hs, rs, js, one, one, hs.new_zeros(()), large,
                                     lwork)
        small = few & ~large
    hs, rs, js, _, info = _dlahqr(False, False, ns, one, ns, hs, rs, js, one, one, hs.new_zeros(()), small)
    inf = torch.where(small, info, inf)
    h = _region_put(h, top, one, hs, square & _mat(few))
    copied = _within(ns, nsb) & _vec(few)
    wr, wi = _vec_put(wr, first, rs, copied), _vec_put(wi, first, js, copied)
    ks = torch.where(few, first + inf, ks)
    pair = few & (ks >= kbot)
    _, _, _, _, p1r, p1i, p2r, p2i, _, _ = _dlanv2(_entry(h, kbot - 1, kbot - 1), _entry(h, kbot - 1, kbot),
                                                   _entry(h, kbot, kbot - 1), _entry(h, kbot, kbot))
    wr = _at_put(_at_put(wr, kbot - 1, p1r, pair), kbot, p2r, pair)
    wi = _at_put(_at_put(wi, kbot - 1, p1i, pair), kbot, p2i, pair)
    ks = torch.where(pair, kbot - 1, ks)
    carried = (computed & (kbot - ks + 1 > ns), wr, wi, kbot.clone(), torch.zeros_like(live))
    _, wr, wi, _, _ = while_loop_op(_first_any, _dlaqr0_sorted, carried, (ks,))
    _, wr, wi, _ = while_loop_op(_first_any, _dlaqr0_paired, (computed & (kbot >= ks + 2), wr, wi, kbot.clone()),
                                 (ks,))
    two = sweep & (kbot - ks + 1 == 2) & (_at(wi, kbot) == 0)
    last, before, tip = _at(wr, kbot), _at(wr, kbot - 1), _entry(h, kbot, kbot)
    nearer = (last - tip).abs() < (before - tip).abs()
    wr = _at_put(_at_put(wr, kbot - 1, last, two & nearer), kbot, before, two & ~nearer)
    ns = torch.minimum(ns, kbot - ks + 1)
    ns = ns - ns % 2
    ks = kbot - ns + 1
    kdu = 2 * ns
    h, z, (sr, si) = _laqr5(wantt, wantz, accum, order, ktop, kbot, ns, (_vec_at(wr, ks, nsb), _vec_at(wi, ks, nsb)),
                            h, iloz, ihiz, z, order - kdu + 1, order - kdu - (kdu + 4) + 1, kdu + 4,
                            (order - kdu + 1 - 4) - (kdu + 1) + 1, kdu + 1, sweep)
    paired = _within(ns, nsb) & _vec(sweep)
    wr, wi = _vec_put(wr, ks, sr, paired), _vec_put(wi, ks, si, paired)
    ndfl = torch.where(live, torch.where(ld > 0, 1, ndfl + 1), ndfl)
    it = torch.where(live, it + 1, it)
    return (live & (kbot >= ilo) & (it <= itmax), h, z if wantz else z.clone(), wr, wi, kbot, nw, ndfl, ndec, it)


# dlaqr0.f 559-568, DO 30's pass for I (KBOT, MAX(KS + 1, KTOP + 2), -2), H the one DLAQR3 left: SS = ABS(H(I,I-1))
# + ABS(H(I-1,I-2)), AA = WILK1 SS + H(I,I), BB = SS, CC = WILK2 SS, DD = AA (WILK1 = 0.75d0, WILK2 = -0.4375d0,
# 295), then DLANV2(AA, BB, CC, DD) into WR(I-1), WI(I-1), WR(I), WI(I)
def _dlaqr0_exceptional(live, wr, wi, i, h, stop):
    ss = _entry(h, i, i - 1).abs() + _entry(h, i - 1, i - 2).abs()
    aa = 0.75 * ss + _entry(h, i, i)
    _, _, _, _, rt1r, rt1i, rt2r, rt2i, _, _ = _dlanv2(aa, ss, -0.4375 * ss, aa)
    wr = _at_put(_at_put(wr, i - 1, rt1r, live), i, rt2r, live)
    wi = _at_put(_at_put(wi, i - 1, rt1i, live), i, rt2i, live)
    i = torch.where(live, i - 2, i)
    return live & (i >= stop), wr, wi, i


# dlaqr0.f 623-641, DO 50's pass for K (KBOT, KS + 1, -1) until SORTED: SORTED = .TRUE., then DO 40
def _dlaqr0_sorted(live, wr, wi, k, sorted_, ks):
    _, wr, wi, _, sorted_ = while_loop_op(_first_any, _dlaqr0_swapped, (live & (ks <= k - 1), wr, wi, ks.clone(),
                                                                        live | sorted_), (k,))
    k = torch.where(live, k - 1, k)
    return live & (k >= ks + 1) & ~sorted_, wr, wi, k, sorted_


# DO 40's pass for I (KS..K-1): WR, WI(I) and (I+1) swapped where ABS(WR(I)) + ABS(WI(I)) < ABS(WR(I+1)) +
# ABS(WI(I+1)), SORTED = .FALSE.
def _dlaqr0_swapped(live, wr, wi, i, sorted_, k):
    r1, r2, i1, i2 = _at(wr, i), _at(wr, i + 1), _at(wi, i), _at(wi, i + 1)
    swap = live & (r1.abs() + i1.abs() < r2.abs() + i2.abs())
    wr = _at_put(_at_put(wr, i, r2, swap), i + 1, r1, swap)
    wi = _at_put(_at_put(wi, i, i2, swap), i + 1, i1, swap)
    i = torch.where(live, i + 1, i)
    return live & (i <= k - 1), wr, wi, i, sorted_ & ~swap


# dlaqr0.f 651-664, DO 70's pass for I (KBOT, KS + 2, -2): where WI(I) /= -WI(I-1), WR (and WI) (I), (I-1), (I-2)
# take (I-1), (I-2), (I)
def _dlaqr0_paired(live, wr, wi, i, ks):
    swap = live & (_at(wi, i) != -_at(wi, i - 1))
    made = []
    for x in (wr, wi):
        a, b, c = _at(x, i - 2), _at(x, i - 1), _at(x, i)
        made.append(_at_put(_at_put(_at_put(x, i, b, swap), i - 1, a, swap), i - 2, c, swap))
    wr, wi = made
    i = torch.where(live, i - 2, i)
    return live & (i >= ks + 2), wr, wi, i


# Reference-LAPACK 3.12.1 SRC/zhseqr.f 434-486 past 427-430's ILO = IHI return, where `proceed`, N = n (zgeev.f's
# order): 434-436 NMIN = MAX(NTINY = 15, ILAENV(12) = 75) (iparmq.f 288); 440-443 N > NMIN, ZLAQR0; else 448-450
# ZLAHQR and, where its INFO > 0, 457 KBOT = INFO and ZLAQR0(WANTT, WANTZ, N or NL, ILO, KBOT, ..., ILO, IHI, Z):
# 464-465 on H where N >= NL = 49 (329), else 474-483 on HL, the NL x NL local (ZLACPY of H, HL(N+1,N) = 0 and
# ZLASET of its columns N+1..NL; its rows N+1..NL left of column N otherwise uninitialized in the source and reached
# there only as xLAQR0, xLAQR3 and xLAQR5 workspace written before it is read, zero here), with W and Z themselves
# (ZLAQR0 reaches W(1:KBOT) and Z's rows ILOZ..IHIZ and columns 1..KBOT, KBOT <= N; W and Z padded to NL here for the
# static extent) and WORKL, NL; HL back into H where WANTT or INFO /= 0. LWORK is the caller's (`lwork`, the batch's
# int64 LWORK) on 441-443 and 464-465's H, NL on 479-481's HL. Returns H, W, Z, INFO
def _zhseqr(wantt, wantz, ilo, ihi, h, w, z, proceed, lwork):
    n = h.shape[-1]
    order = torch.full_like(ilo, n)
    if n > 75:
        return _laqr0(wantt, wantz, False, order, ilo, ihi, h, w, ilo, ihi, z, proceed, lwork)
    h, w, z, info = _zlahqr(wantt, wantz, order, ilo, ihi, h, w, ilo, ihi, z, proceed)
    retry = proceed & (info > 0)
    if n >= 49:
        h, w, z, again = _laqr0(wantt, wantz, False, order, ilo, info, h, w, ilo, ihi, z, retry, lwork)
        return h, w, z, torch.where(retry, again, info)
    pad = 49 - n

    def padded(x):
        x = torch.cat([x, x.new_zeros((*x.shape[:-1], pad))], -1)
        return torch.cat([x, x.new_zeros((*x.shape[:-2], pad, 49))], -2)
    hl, wl, zl, again = _laqr0(wantt, wantz, False, torch.full_like(ilo, 49), ilo, info, padded(h),
                               torch.cat([w, w.new_zeros((*w.shape[:-1], pad))], -1), ilo, ihi,
                               padded(z) if wantz else z, retry, 49)
    info = torch.where(retry, again, info)
    back = retry if wantt else retry & (info != 0)
    return (torch.where(_mat(back), hl[..., :n, :n], h), wl[..., :n], zl[..., :n, :n] if wantz else zl, info)


# Reference-LAPACK 3.12.1 SRC/dhseqr.f 451-503 past 443-447's ILO = IHI return, where `proceed`, N = n (dgeev.f's
# order), as _zhseqr takes zhseqr.f: 451-453 NMIN = MAX(NTINY = 15, ILAENV(12) = 75); 457-460 N > NMIN, DLAQR0; else
# 465-467 DLAHQR and, where its INFO > 0, 474 KBOT = INFO and DLAQR0(WANTT, WANTZ, N or NL, ILO, KBOT, ..., ILO, IHI,
# Z): 481-483 on H where N >= NL = 49 (347), else 492-501 on HL as _zhseqr's (WR and WI padded to NL here alike), HL
# back into H where WANTT or INFO /= 0. LWORK is the caller's (`lwork`: dgeev.f's LWORK - N at IWRK = ITAU = N + 1)
# on 458-460 and 481-483's H, NL on 497-499's HL. Returns H, WR, WI, Z, INFO
def _dhseqr(wantt, wantz, ilo, ihi, h, wr, wi, z, proceed, lwork):
    n = h.shape[-1]
    order = torch.full_like(ilo, n)
    if n > 75:
        return _dlaqr0(wantt, wantz, False, order, ilo, ihi, h, wr, wi, ilo, ihi, z, proceed, lwork)
    h, wr, wi, z, info = _dlahqr(wantt, wantz, order, ilo, ihi, h, wr, wi, ilo, ihi, z, proceed)
    retry = proceed & (info > 0)
    if n >= 49:
        h, wr, wi, z, again = _dlaqr0(wantt, wantz, False, order, ilo, info, h, wr, wi, ilo, ihi, z, retry, lwork)
        return h, wr, wi, z, torch.where(retry, again, info)
    pad = 49 - n

    def extended(x):
        return torch.cat([x, x.new_zeros((*x.shape[:-1], pad))], -1)

    def padded(x):
        x = extended(x)
        return torch.cat([x, x.new_zeros((*x.shape[:-2], pad, 49))], -2)
    hl, wrl, wil, zl, again = _dlaqr0(wantt, wantz, False, torch.full_like(ilo, 49), ilo, info, padded(h),
                                      extended(wr), extended(wi), ilo, ihi, padded(z) if wantz else z, retry, 49)
    info = torch.where(retry, again, info)
    back = retry if wantt else retry & (info != 0)
    return (torch.where(_mat(back), hl[..., :n, :n], h), wrl[..., :n], wil[..., :n],
            zl[..., :n, :n] if wantz else zl, info)


# Reference-LAPACK 3.12.1 SRC/zgeev.f 321-513 and dgeev.f 342-542 (JOBVL = 'N', JOBVR = `jobvr`) on a non-CPU
# engine, the results of LAPACK_ZGEEV (w, vr, info) or LAPACK_DGEEV (wr, wi, vr, info) and their shapes (vr
# column-major, or (0,) where JOBVR = 'N'), with LWORK ATen's (_geev_lwork): ANRM = ZLANGE('M') (the largest modulus,
# a NaN kept); INFO = -4 where it is NaN or above overflow, the API's error contract that the CPU route's LAPACK_xGEEV
# returns to _linalg_check_errors and 3.12.1 lacks (Reference-LAPACK master zgeev.reference.f 332-340, dgeev.reference.f
# 349-357, beside this note: DISNAN(ANRM) or ANRM > DLAMCH('Overflow') returns before SCALEA; the member's remaining
# work here its own and unused); SCALEA and CSCALE (SMLNUM = SQRT(SAFMIN) / EPS) and ZLASCL(ANRM -> CSCALE);
# ZGEBAL('B'); ZGEHRD (IWRK = N + 1, LWORK - N; DGEHRD 2 N + 1, LWORK - 2 N, zgeev.f 352-355, dgeev.f 371-374); with
# vectors ZLACPY('L') and ZUNGHR into Z at the same LWORK (zgeev.f 396-404, dgeev.f 414-421) and ZHSEQR('S', 'V'),
# else ZHSEQR('E', 'N') (IWRK = ITAU = 1, LWORK; DHSEQR N + 1, LWORK - N) (zhseqr.f 414-418: W(1:ILO-1) and
# W(IHI+1:N) from H's diagonal, 427-430's ILO = IHI W(ILO) = H(ILO,ILO) return, 434-486 _zhseqr, then 490-491's
# H(3:N,1:N-2) lower part zero where WANTT or INFO /= 0 and N > 2); where INFO = 0 and with vectors ZTREVC3, ZGEBAK('B',
# 'R') and the normalization (VR as ZHSEQR left Z elsewhere); label 50's undo scaling ZLASCL(CSCALE -> ANRM) of
# W(INFO+1:N), and of W(1:ILO-1) where INFO > 0, where SCALEA and INFO >= 0. DHSEQR alike (dhseqr.f 427-434 WR and WI
# = 0 of the isolated, 443-447, 451-503 _dhseqr at LWORK - N, 508-509). The blocked xGEHRD, xUNGQR and xUNMQR paths
# are their LWORK conditions' (_gehrd, _unghr, _unmhr). OPEN, not yet source here: xTREVC3's NB = (LWORK - IWRK + 1 -
# N) / (2 N) (NBMIN 8, NBMAX 128) choice of back-transformation (here VR = Z X for every N); and the BLAS summation
# order, FMA contraction, the products' zero entries past each operand's extent (a BLAS leaves unreferenced entries
# unread; here 0 times a non-finite entry is NaN), and the GPU and ANE precision of each engine
def _geev_routine(a, jobvr):
    n, k, whole, vectors = a.shape[-1], _counted(a), a.is_complex(), jobvr == 'V'
    limits = torch.finfo(a.real.dtype)
    smlnum = math.sqrt(limits.tiny) / limits.eps
    bignum = 1 / smlnum
    anrm = a.abs().flatten(-2).amax(-1)
    bad = anrm.isnan() | (anrm > limits.max)
    low = (anrm > 0) & (anrm < smlnum)
    scalea = ~bad & (low | (anrm > bignum))
    cscale = torch.where(low, smlnum, torch.full_like(anrm, bignum))
    h, scale, perm, ilo, ihi = _gebal(_lascl(a, anrm, cscale, scalea), ~bad)
    lwork = int(_geev_lwork(torch.tensor(n), vectors, whole, a.real.dtype))
    rest = torch.full_like(ilo, lwork - n if whole else lwork - 2 * n)
    h, taus = _gehrd(torch.full_like(ilo, n), ilo, ihi, h, rest)
    z = _unghr(ilo, ihi, torch.tril(h), taus, rest) if vectors else h.new_zeros(())
    isolated = ~_span(k, ilo, ihi) | _vec(ilo == ihi)
    diagonal = h.diagonal(dim1=-2, dim2=-1)
    eigen = (torch.where(isolated, diagonal, 0),) if whole else (torch.where(isolated, diagonal, 0), torch.zeros_like(
        diagonal))
    info = torch.zeros_like(ilo)
    searched = ~bad & (ilo != ihi)
    if n > 1 and whole:
        h, *eigen, z, info = _zhseqr(vectors, vectors, ilo, ihi, h, *eigen, z, searched, lwork)
    elif n > 1:
        h, *eigen, z, info = _dhseqr(vectors, vectors, ilo, ihi, h, *eigen, z, searched, lwork - n)
    if n > 2:
        h = torch.where((k[:, None] - k >= 2) & _mat(searched & (vectors | (info != 0))), 0, h)
    info = torch.where(bad, -4, info)
    if vectors:
        vr = _gebak(_ztrevc(h, z) if whole else _dtrevc(h, z), scale, perm, ilo, ihi)
        vr = _zgeev_normalized(vr) if whole else _dgeev_normalized(vr, eigen[1])
        vr = torch.where(_mat(info == 0), vr, z).mT.contiguous().mT
    else:
        vr = a.new_empty((0,))
    undone = (k >= _vec(info + 1)) | (_vec(info > 0) & (k <= _vec(ilo - 1)))
    eigen = [torch.where(undone, _lascl(x, cscale, anrm, scalea & (info >= 0)), x) for x in eigen]
    return (*eigen, vr, info.to(torch.int32))


def _eigen_geev(self, vectors=True):
    return _eigen(self, vectors, _geev_routine, _geev_routine)


# docs/provider.md#fragment-coded-weights (rdma/mesh_code.py fragment; its codes, tiles and bands uint32, as
# mesh_code.load views them): each output element (i, j) at source row rows[i] and column columns[j], in its 32 x 32
# tile (tr, tc) at (rr, cc). The tile word's bits 12-15 its code width (0: the tile is zero), bits 0-11 with the
# band's low 28 bits its code-word base, its high half its float16 step; the element's code a width-bit field of the
# 16-bit lane half (rr odd: the high half) of the code words at lane `lane`, continued into the next word's (32 on)
# where it crosses 16 bits; its value (2 code + 1 - 2^width) step / 2 in float32 (float64 for a float64 result), times
# row_scale[rows[i]] and col_scale[columns[j]], in `dtype`. Each 32-bit word is read as its unsigned value (an int32
# view widened to int64 and masked), so every field and shift is that of the unsigned word
def _fragment_decoded(codes, tiles, bands, row_scale, col_scale, rows, columns, dtype):
    if not codes.numel():
        return torch.zeros((rows.numel(), columns.numel()), dtype=dtype, device=codes.device)
    t = torch.float64 if dtype is torch.float64 else torch.float32
    row, col = rows.reshape(-1).to(torch.int64)[:, None], columns.reshape(-1).to(torch.int64)[None, :]
    rr, cc, tr, tc = row % 32, col % 32, row // 32, col // 32
    word = _unsigned(tiles.view(torch.int32)[tr, tc])
    width = (word >> 12) & 15
    lane = ((rr // 8) & 1) | ((cc & 3) << 1) | (((rr // 8) & 2) << 2) | ((cc & 4) << 2)
    bit = ((rr & 6) * 2 + cc // 8) * width
    shift, half = bit % 16, (rr & 1) * 16
    at = ((_unsigned(bands.view(torch.int32).reshape(-1)[tr]) & 0x0FFFFFFF) + (word & 0xFFF) + bit // 16) * 32 + lane
    live = width != 0
    crossing = live & (shift + width > 16)
    flat = codes.view(torch.int32).reshape(-1)
    low = torch.where(live, _unsigned(flat[torch.where(live, at, 0)]), 0)
    high = torch.where(crossing, _unsigned(flat[torch.where(crossing, at + 32, 0)]), 0)
    joined = ((low >> half) & 0xFFFF) | (((high >> half) & 0xFFFF) << 16)
    power = torch.ones_like(width) << width
    symbol = (joined >> shift) & (power - 1)
    step = tiles.view(torch.float16)[..., 1::2][tr, tc].to(t)
    value = torch.where(live, (symbol * 2 + 1 - power).to(t) * (0.5 * step), 0)
    value = value * row_scale.reshape(-1)[row].to(t) * col_scale.reshape(-1)[col].to(t)
    return value.to(dtype)


def _unsigned(words):
    return words.to(torch.int64) & 0xFFFFFFFF


FRAGMENT = mesh_code.fragment


# J. K. Salmon, M. A. Moraes, R. O. Dror and D. E. Shaw, Parallel random numbers: as easy as 1, 2, 3, SC 2011, §3.1 and
# §4.2 (Philox4x32-10: its multipliers, Weyl key increments, ten rounds): each draw a counter-based function of its key
# and counter, the key the program's seed, the counter (element, execution, draw site, stream); the generator a STATE
# value {seed low, seed high, execution, 0} the program advances once an execution
PHILOX = (0xD2511F53, 0xCD9E8D57, 0x9E3779B9, 0xBB67AE85)


def seeded(seed):
    words = [seed & 0xFFFFFFFF, (seed >> 32) & 0xFFFFFFFF, 0, 0]
    return torch.tensor([w - (1 << 32) if w >= 1 << 31 else w for w in words], dtype=torch.int32)


# a constructed program's generator: its caller's STATE words, each draw site its source identity (compiler/
# mesh_compiler/export.py read `site`) where supplied, else numbered in construction order from 1, `sites` the count of
# the construction's draws, and their advance once an execution after the program's draws; `counter` the execution
# word's own view (word 2 of `state`), the advance's operand (design/compilation-composition-change-request.md items 6
# and 7: compiler/mesh_compiler/schedule.py analyse's read and write view of the body's state)
class Generator:
    def __init__(self, state):
        self.state, self.sites, self.counter = state, 0, state.narrow(0, 2, 1)

    def site(self, identity=None):
        self.sites += 1
        return self.state, self.sites if identity is None else identity

    def advanced(self):
        return Advance(self.counter, 1) if self.sites else None


# the generator's advance once an execution, after the program's draws, as its consumer's typed operand: `counter` the
# execution word's own view (Generator.counter: word 2 of the state root, the same storage the draws read through
# GENERATOR), `increment` the execution count it adds (Salmon et al., SC 2011, §4.2: each execution's counter word)
Advance = collections.namedtuple('Advance', 'counter increment')


# design/torch-source-lowering-change-request.md item 1: the storage identity of a tensor of the compilation (its
# placed stand-ins, temporaries and views are meta tensors; a constant is a tensor holding its data)
def _key(value):
    return value.untyped_storage()._cdata if value.device.type == 'meta' else value.untyped_storage().data_ptr()


# torch 2.14.0 torch/_tensor_docs.py view_as_real: a storage root's element type, a complex one's real pairs
def _stored(value):
    return torch.view_as_real(value) if value.is_complex() else value


def _metadata(value):
    return (tuple(value.shape), tuple(value.stride()), value.storage_offset(), value.dtype, value.device,
            value.requires_grad) \
        if isinstance(value, torch.Tensor) else value


# design/compilation-composition-change-request.md, numerical lowerer boundary: "Compilation selects the custom
# numerical method explicitly ... A custom numerical or target implementation is supplied there with its explicit
# operands and its cited algorithm (source-lowering items 5/6)". compiler/mesh_compiler/decompose.py selects each
# method and traces its formulation into the placed body: ATen and torch HOPs (while_loop_op, cond_op, switch_op), the
# installed upstream compiler's to lower (torch 2.14.0 torch/_inductor/graph.py GraphLowering call_function,
# lowering.py lowerings and make_fallback), and, on the CPU, the typed LAPACK operators (LAPACK: each one Accelerate
# routine call, never a Torch kernel, its INFO consumed by the body's aten._linalg_check_errors). eig and eigvals off
# the CPU are _eigen_geev's body (_geev_routine: zgeev.f and dgeev.f as ATen and while_loop composition); the outlined
# eigh's step is _qdwh_eigh_pop, its Haar draws reading the body's Generator through GENERATOR
def routine(target):
    return LAPACK.get(getattr(target, 'overloadpacket', target))


# a body's nodes `lookup` finds (its typed LAPACK operators by default), its HOP subgraphs' among them (torch 2.14.0
# torch/_higher_order_ops: each cond, switch and while_loop subgraph a GraphModule its get_attr reads): node -> value
def methods(graph_module, lookup=routine):
    found = {}
    for node in graph_module.graph.nodes:
        held = _get_attr(graph_module, node.target) if node.op == 'get_attr' else None
        if isinstance(held, torch.fx.GraphModule):
            found.update(methods(held, lookup))
        if node.op == 'call_function' and lookup(node.target) is not None:
            found[node] = lookup(node.target)
    return found
