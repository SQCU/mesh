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
# words on site's device) and no eager kernel. The producer binds it to the body's operations.Generator:
# Generator.site(site) records the draw, so the body's Generator.advanced() (its Advance operand) advances the
# execution word once after its draws, and the value is Generator.state's root (design/compilation-composition-change-request.md items 6 and 7),
# never a fresh allocation or Inductor's RandomSeeds
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
# scaled norm (largest modulus, then the root of the sum of squared ratios); CABS1 |Re| + |Im|; DLADIV Smith's
# division (_divided); a real factor of a complex entry multiplies each component (ZDSCAL, Fortran's mixed real and
# complex product); Fortran SIGN(A, B) takes B's sign bit, a -0 B negative

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


# Smith's complex division (xr + i xi) / (yr + i yi), real components
def _divided(xr, xi, yr, yi):
    wide = yr.abs() >= yi.abs()
    first, second = torch.where(wide, yr, yi), torch.where(wide, yi, yr)
    ratio = second / first
    den = first + second * ratio
    return (torch.where(wide, (xr + xi * ratio) / den, (xr * ratio + xi) / den),
            torch.where(wide, (xi - xr * ratio) / den, (xi * ratio - xr) / den))


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


# Reference-LAPACK SRC/zgehrd.f 249-254 and 295-299 (TAU outside ILO..IHI-1 zero; the unblocked code from ILO) and
# zgehd2.f 190-207 (dgehrd.f, dgehd2.f alike), each static column I = 1..n-1 active where ILO <= I <= IHI-1:
# ZLARFG(IHI-I, A(I+1,I), A(I+2,I)) by _reflector over rows I+1..IHI; H(I) from the right to A(1:IHI, I+1:IHI), H(I)^H
# from the left to A(I+1:IHI, I+1:N); beta into A(I+1,I) and v(2:) below it. Returns the packed matrix and TAU
def _gehrd(h, ilo, ihi):
    n, k = h.shape[-1], _counted(h)
    taus = []
    for i in range(1, n):
        active = (ilo <= i) & (i <= ihi - 1)
        below = ((k > i) & (k <= _vec(ihi)))[..., i:]
        v, tau, beta = _reflector(torch.where(below, h[..., i:, i - 1], 0))
        v, tau = torch.where(below, v, 0), torch.where(active, tau, 0)
        region = _mat(active) & (k <= _vec(ihi))[..., :, None] & below[..., None, :]
        h = torch.cat([h[..., :, :i], torch.where(region, _dlarf(h[..., :, i:], v, tau, left=False), h[..., :, i:])],
                      -1)
        region = _mat(active) & below[..., :, None]
        h = _put(h, torch.where(region, _dlarf(h[..., i:, i:], v, tau, adjoint=True), h[..., i:, i:]), i, i)
        packed = torch.cat([beta.to(h.dtype)[..., None], v[..., 1:]], -1)
        h = _put(h, torch.where(_vec(active) & below, packed, h[..., i:, i - 1])[..., :, None], i, i - 1)
        taus.append(tau)
    return h, torch.stack(taus, -1) if taus else h.new_zeros((*h.shape[:-2], 0))


# zunghr.f 198-232 (dorghr.f alike) with zung2r's reflector sequence: Q = H(ILO) ... H(IHI-1) from the identity, each
# H(I) (reverse order) to rows and columns I+1..IHI
def _unghr(h, taus, ilo, ihi):
    n, k = h.shape[-1], _counted(h)
    z = _unit(h).expand(h.shape).contiguous()
    for i in reversed(range(1, n)):
        active = (ilo <= i) & (i <= ihi - 1)
        below = ((k > i) & (k <= _vec(ihi)))[..., i:]
        v = torch.where(below, torch.cat([torch.ones_like(h[..., i:i + 1, i - 1]), h[..., i + 1:, i - 1]], -1), 0)
        region = _mat(active) & below[..., :, None] & below[..., None, :]
        z = _put(z, torch.where(region, _dlarf(z[..., i:, i:], v, taus[..., i - 1]), z[..., i:, i:]), i, i)
    return z


# Reference-LAPACK SRC/dlarfg.f 125-174 (N = len(xs) + 1) on batch scalars: (ALPHA on exit, the beta; TAU; X on
# exit). The label 10 KNT loop (at most 20 passes) and its undo are its static passes; SAFMIN = DLAMCH('S') /
# DLAMCH('E'); the BETA above HUGE / 2 branch scales by SAFMIN once
def _dlarfg(alpha, xs):
    limits = torch.finfo(alpha.dtype)
    safmin = limits.tiny / (limits.eps / 2)
    rsafmn = 1 / safmin
    trivial = _listed_norm(xs) == 0
    beta = -_fsign(_dlapy2(alpha, _listed_norm(xs)), alpha)
    small = beta.abs() < safmin
    large = ~small & (beta.abs() > limits.max / 2)
    running, knt, sb, sa, sx = small, torch.zeros_like(alpha, dtype=torch.int64), beta, alpha, list(xs)
    for _ in range(20):
        knt = torch.where(running, knt + 1, knt)
        sx = [torch.where(running, x * rsafmn, x) for x in sx]
        sb, sa = torch.where(running, sb * rsafmn, sb), torch.where(running, sa * rsafmn, sa)
        running = running & (sb.abs() < safmin)
    fx = [torch.where(small, x, torch.where(large, plain * safmin, plain)) for x, plain in zip(sx, xs)]
    fa = torch.where(small, sa, torch.where(large, alpha * safmin, alpha))
    fb = torch.where(small | large, -_fsign(_dlapy2(fa, _listed_norm(fx)), fa), beta)
    tau, inverse = (fb - fa) / fb, 1 / (fa - fb)
    undone = fb
    for j in range(20):
        undone = torch.where(knt > j, undone * safmin, undone)
    undone = torch.where(large, undone * rsafmn, undone)
    return (torch.where(trivial, alpha, undone), torch.where(trivial, 0, tau),
            [torch.where(trivial, x, made * inverse) for x, made in zip(xs, fx)])


# Reference-LAPACK SRC/zlarfg.f 144-211 with N = 2 on batch scalars: ALPHA and X (complex) to (ALPHA on exit, the
# real beta where TAU is nonzero; TAU; X on exit), as _dlarfg with DLAPY3(ALPHR, ALPHI, XNORM), ALPHA = ZLADIV(1,
# ALPHA - BETA) and ZSCAL
def _zlarfg2(alpha, x):
    limits = torch.finfo(alpha.real.dtype)
    safmin = limits.tiny / (limits.eps / 2)
    rsafmn = 1 / safmin
    ar, ai = alpha.real, alpha.imag
    trivial = (x.abs() == 0) & (ai == 0)
    beta = -_fsign(_dlapy3(ar, ai, x.abs()), ar)
    small = beta.abs() < safmin
    large = ~small & (beta.abs() > limits.max / 2)
    running, knt, sb, sr, si, sx = small, torch.zeros_like(ar, dtype=torch.int64), beta, ar, ai, x
    for _ in range(20):
        knt = torch.where(running, knt + 1, knt)
        sx = torch.where(running, _scaled(sx, rsafmn), sx)
        sb, sr, si = (torch.where(running, v * rsafmn, v) for v in (sb, sr, si))
        running = running & (sb.abs() < safmin)
    fx = torch.where(small, sx, torch.where(large, _scaled(x, safmin), x))
    fr = torch.where(small, sr, torch.where(large, ar * safmin, ar))
    fi = torch.where(small, si, torch.where(large, ai * safmin, ai))
    fb = torch.where(small | large, -_fsign(_dlapy3(fr, fi, fx.abs()), fr), beta)
    tau = torch.complex((fb - fr) / fb, -fi / fb)
    inverse = torch.complex(*_divided(torch.ones_like(fr), torch.zeros_like(fr), fr - fb, fi))
    undone = fb
    for j in range(20):
        undone = torch.where(knt > j, undone * safmin, undone)
    undone = torch.where(large, undone * rsafmn, undone)
    return (torch.where(trivial, alpha, torch.complex(undone, torch.zeros_like(undone))), torch.where(trivial, 0, tau),
            torch.where(trivial, x, inverse * fx))


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


# Reference-LAPACK SRC/zlahqr.f 258-573 (zhseqr.f's call: WANTT = WANTZ = `want`, ILOZ = ILO, IHIZ = IHI): the trash,
# the DO 20 subdiagonal made real (each static I in order, its row and column scalings), then the iteration as one
# while_loop whose pass is one DO 130 iteration (labels 30 and 140 flattened: on deflation W(I) = H(I,I), KDEFL = 0,
# I = L - 1, L = ILO, ITS = 0; INFO = I past ITMAX). Returns H, Z, W (positions ILO..IHI) and INFO
def _zlahqr(h, z, w, ilo, ihi, proceed, want):
    n, k = h.shape[-1], _counted(h)
    h = _cleared(h, ilo, ihi, proceed)
    rows_z = _span(k, ilo, ihi)
    for ii in range(2, n + 1):
        sub = h[..., ii - 1, ii - 2]
        hit = proceed & (ilo + 1 <= ii) & (ii <= ihi) & (sub.imag != 0)
        sc = _over(sub, _abs1(sub))
        sc = _over(sc.conj(), sc.abs())
        h = _entry_put(h, torch.full_like(ilo, ii), torch.full_like(ilo, ii - 1), sub.abs(), hit)
        jhi = n if want else ihi
        cols = _vec(hit) & (k >= ii) & (k <= _vec(jhi))
        h = _row_put(h, ii - 1, torch.where(cols, h[..., ii - 1, :] * _vec(sc), h[..., ii - 1, :]))
        rows = _vec(hit) & (k >= (1 if want else _vec(ilo))) & (k <= (min(n, ii + 1) if want else _vec(
            torch.clamp(ihi, max=ii + 1))))
        h = _column_put(h, ii - 1, torch.where(rows, h[..., :, ii - 1] * _vec(sc.conj()), h[..., :, ii - 1]))
        if want:
            z = _column_put(z, ii - 1, torch.where(_vec(hit) & rows_z, z[..., :, ii - 1] * _vec(sc.conj()),
                                                   z[..., :, ii - 1]))
    smlnum, itmax = _iteration_bounds(h, ilo, ihi)
    zero = torch.zeros_like(ilo)
    carried = (proceed & (ihi >= ilo), h.contiguous(), z.contiguous() if want else h.new_zeros(()), w.contiguous(),
               ihi.clone(), ilo.clone(), zero, zero.clone(), zero.clone())
    _, h, z, w, _, _, _, _, info = while_loop_op(_first_any, functools.partial(
        _zlahqr_pass, want, torch.finfo(h.real.dtype).eps), carried, (ilo, ihi, smlnum, itmax))
    return h, z, w, info


# SMLNUM = SAFMIN (NH / ULP) and ITMAX = 30 MAX(10, NH), NH = IHI - ILO + 1
def _iteration_bounds(h, ilo, ihi):
    limits, nh = torch.finfo(h.real.dtype), ihi - ilo + 1
    return limits.tiny * (nh.to(h.real.dtype) / limits.eps), 30 * torch.clamp(nh, min=10)


def _zlahqr_pass(want, ulp, live, h, z, w, i, l, its, kdefl, info, ilo, ihi, smlnum, itmax):
    n, k = h.shape[-1], _counted(h)
    l = torch.where(live, _deflation(h, i, l, ilo, ihi, smlnum, ulp), l)
    h = _entry_put(h, l, l - 1, 0, live & (l > ilo))
    deflate = live & (l >= i)
    sweep = live & ~deflate
    w = _at_put(w, i, _entry(h, i, i), deflate)
    kdefl = torch.where(sweep, kdefl + 1, kdefl)
    i1, i2 = (torch.ones_like(l), torch.full_like(i, n)) if want else (l, i)
    rows_z = _span(k, ilo, ihi)

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
        v1, t1, v2 = _zlarfg2(v1, v2)
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
        if want:
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
        if want:
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
    if want:
        z = torch.where(_mat(phased) & (k == _mat(i)) & rows_z[..., :, None], z * _mat(temp), z)
    return _iterated(live, deflate, sweep, h, z if want else z.clone(), (w,), i, l, its, kdefl, info, ilo, itmax)


# the pass's ITS, KDEFL, I, L and INFO: a deflation resets ITS and KDEFL, I = L - 1 and L = ILO; a sweep counts
# ITS, the sweep at ITS = ITMAX the last (INFO = I)
def _iterated(live, deflate, sweep, h, z, eigen, i, l, its, kdefl, info, ilo, itmax):
    fail = sweep & (its >= itmax)
    following = torch.where(deflate, l - 1, i)
    return (live & ~fail & (following >= ilo), h, z, *eigen, following, torch.where(deflate, ilo, l),
            torch.where(deflate, 0, torch.where(sweep, its + 1, its)), torch.where(deflate, 0, kdefl),
            torch.where(fail, i, info))


# Reference-LAPACK SRC/dlahqr.f 258-570 as _zlahqr (no DO 20): a deflation of one (WR(I) = H(I,I), WI(I) = 0) or two
# (DLANV2's standardized block and its DROTs of H's rows and columns and Z's columns) at L >= I-1; the 343-379
# exceptional (DAT1 = 3/4, DAT2 = -0.4375) or Francis double shift, the 383-400 M search over L+1..I-2, and the
# 404-481 bulge chase of NR = MIN(3, I-K+1) reflectors
def _dlahqr(h, z, wr, wi, ilo, ihi, proceed, want):
    h = _cleared(h, ilo, ihi, proceed)
    smlnum, itmax = _iteration_bounds(h, ilo, ihi)
    zero = torch.zeros_like(ilo)
    carried = (proceed & (ihi >= ilo), h.contiguous(), z.contiguous() if want else h.new_zeros(()), wr.contiguous(),
               wi.contiguous(), ihi.clone(), ilo.clone(), zero, zero.clone(), zero.clone())
    _, h, z, wr, wi, _, _, _, _, info = while_loop_op(_first_any, functools.partial(
        _dlahqr_pass, want, torch.finfo(h.dtype).eps), carried, (ilo, ihi, smlnum, itmax))
    return h, z, wr, wi, info


# dlahqr.f 387-397: V for the double shift at M, of H(M,M), H(M+1,M), H(M,M+1), H(M+1,M+1) and H(M+2,M+1)
def _double_shifted(hmm, hm1m, hmm1, hm1m1, hm2m1, rt1r, rt1i, rt2r, rt2i):
    s = (hmm - rt2r).abs() + rt2i.abs() + hm1m.abs()
    h21s = hm1m / s
    v1 = h21s * hmm1 + (hmm - rt1r) * ((hmm - rt2r) / s) - rt1i * (rt2i / s)
    v2 = h21s * (hmm + hm1m1 - rt1r - rt2r)
    v3 = h21s * hm2m1
    s = v1.abs() + v2.abs() + v3.abs()
    return v1 / s, v2 / s, v3 / s


def _dlahqr_pass(want, ulp, live, h, z, wr, wi, i, l, its, kdefl, info, ilo, ihi, smlnum, itmax):
    n, k = h.shape[-1], _counted(h)
    l = torch.where(live, _deflation(h, i, l, ilo, ihi, smlnum, ulp), l)
    h = _entry_put(h, l, l - 1, 0, live & (l > ilo))
    deflate = live & (l >= i - 1)
    sweep = live & ~deflate
    rows_z = _span(k, ilo, ihi)

    # 150: one eigenvalue, or the 2 x 2 block's
    single, pair = deflate & (l == i), deflate & (l == i - 1)
    wr, wi = _at_put(wr, i, _entry(h, i, i), single), _at_put(wi, i, 0, single)
    a, b, c, d, rt1r, rt1i, rt2r, rt2i, cs, sn = _dlanv2(_entry(h, i - 1, i - 1), _entry(h, i - 1, i),
                                                         _entry(h, i, i - 1), _entry(h, i, i))
    for row, col, value in ((i - 1, i - 1, a), (i - 1, i, b), (i, i - 1, c), (i, i, d)):
        h = _entry_put(h, row, col, value, pair)
    wr = _at_put(_at_put(wr, i - 1, rt1r, pair), i, rt2r, pair)
    wi = _at_put(_at_put(wi, i - 1, rt1i, pair), i, rt2i, pair)
    if want:
        upper, lower = _row(h, i - 1), _row(h, i)
        cols = (_vec(pair) & (k > _vec(i)))[..., None, :]
        h = torch.where(cols & (k[:, None] == _mat(i - 1)), (_vec(cs) * upper + _vec(sn) * lower)[..., None, :], h)
        h = torch.where(cols & (k[:, None] == _mat(i)), (_vec(cs) * lower - _vec(sn) * upper)[..., None, :], h)
        left, right = _column(h, i - 1), _column(h, i)
        rows = (_vec(pair) & (k <= _vec(i - 2)))[..., :, None]
        h = torch.where(rows & (k == _mat(i - 1)), (_vec(cs) * left + _vec(sn) * right)[..., :, None], h)
        h = torch.where(rows & (k == _mat(i)), (_vec(cs) * right - _vec(sn) * left)[..., :, None], h)
        left, right = _column(z, i - 1), _column(z, i)
        rows = (_vec(pair) & rows_z)[..., :, None]
        z = torch.where(rows & (k == _mat(i - 1)), (_vec(cs) * left + _vec(sn) * right)[..., :, None], z)
        z = torch.where(rows & (k == _mat(i)), (_vec(cs) * right - _vec(sn) * left)[..., :, None], z)

    kdefl = torch.where(sweep, kdefl + 1, kdefl)
    i1, i2 = (torch.ones_like(l), torch.full_like(i, n)) if want else (l, i)

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
        b2, t2_, (x2,) = _dlarfg(v1, [v2])
        if kk + 1 < n:
            b3, t3_, (y2, y3) = _dlarfg(v1, [v2, v3])
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
        if want:
            z = _reflected3(z, kk, None, _vec(act) & rows_z, three, v2, v3, t1, t2, t3)
    return _iterated(live, deflate, sweep, h, z if want else z.clone(), (wr, wi), i, l, its, kdefl, info, ilo, itmax)


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


# Reference-LAPACK SRC/dtrevc.f 342-519 (SIDE = 'R', HOWMNY = 'B'; dtrevc3.f's back-transformed form) on the real
# quasi-triangular Schur form t: column KI of X is real (IP = 0), a pair's second (T(KI,KI-1) /= 0, IP = -1, holding
# the pair's WORK(1+N:) in xr and WORK(1+N2:) in xi) or its first (skipped, IP = 1). All the solves run together in
# the source's block order: each static round J = n-1..1 is the 1 x 1 block at J or the 2 x 2 block ending at J
# (T(J,J-1) /= 0; J the first row of a 2 x 2 block skipped), for every column whose own range J = KI-1..1 (KI-2..1
# for a pair) holds J: DLALN2, the BIGNUM / XNORM rescaling of X and SCALE (a pair's 2 x 2 by REC = 1 / XNORM), DSCAL of
# WORK(1:KI) by SCALE, X into WORK and the DAXPYs above the block. VR = Z X, each pair's columns its real and imaginary
# parts, then by 1 / EMAX (the largest |VR(K,KI)|, a pair's |VR(K,KI-1)| + |VR(K,KI)|)
def _dtrevc(t, z):
    n, k = t.shape[-1], _counted(t)
    limits = torch.finfo(t.dtype)
    ulp = limits.eps
    smlnum = limits.tiny * (n / ulp)
    bignum = (1 - ulp) / smlnum
    pad = torch.zeros_like(t[..., 0, :1])
    sub = torch.cat([pad, t.diagonal(-1, dim1=-2, dim2=-1)], -1)
    sup = torch.cat([pad, t.diagonal(1, dim1=-2, dim2=-1)], -1)
    following = torch.cat([t.diagonal(-1, dim1=-2, dim2=-1), pad], -1)
    second = sub != 0
    first = (following != 0) & ~second
    isreal = ~second & ~first
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
    for j in range(n - 1, 0, -1):
        two = (t[..., j - 1, j - 2] != 0) if j >= 2 else torch.zeros_like(second[..., 0])
        skip = t[..., j, j - 1] != 0
        active = (isreal & (k >= j + 1)) | (second & (k >= j + 2))
        single, double = active & _vec(~two & ~skip), active & _vec(two & ~skip)
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


# Reference-LAPACK SRC/zgeev.f 319-517 and dgeev.f 330-560 (JOBVL = 'N', JOBVR = `jobvr`) on a non-CPU engine,
# the results of LAPACK_ZGEEV (w, vr, info) or LAPACK_DGEEV (wr, wi, vr, info) and their shapes (vr column-major, or
# (0,) where JOBVR = 'N'): ANRM = ZLANGE('M') (the largest modulus, a NaN kept), INFO = -4 where it is NaN or above
# overflow (the member's remaining work its own and unused), SCALEA and CSCALE (SMLNUM = SQRT(SAFMIN) / EPS) and
# ZLASCL(ANRM -> CSCALE); ZGEBAL('B'); ZGEHRD; with vectors ZUNGHR into Z and ZHSEQR('S', 'V'), else ZHSEQR('E', 'N')
# (zhseqr.f 415-430: W(1:ILO-1) and W(IHI+1:N) from H's diagonal, ILO = IHI's W(ILO) = H(ILO,ILO) return, else
# ZLAHQR, then 482-485's H(3:N,1:N-2) lower part zero where WANTT or INFO /= 0 and N > 2); where INFO = 0 and with
# vectors ZTREVC3, ZGEBAK('B', 'R') and the normalization (VR as ZHSEQR left Z elsewhere); label 50's undo scaling
# ZLASCL(CSCALE -> ANRM) of W(INFO+1:N), and of W(1:ILO-1) where INFO > 0, where SCALEA and INFO >= 0. ZHSEQR's ZLAQR0
# (N > NMIN = 75, and its retry where ZLAHQR returns INFO > 0) is ZLAHQR here
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
    h, taus = _gehrd(h, ilo, ihi)
    z = _unghr(h, taus, ilo, ihi) if vectors else h.new_zeros(())
    isolated = ~_span(k, ilo, ihi) | _vec(ilo == ihi)
    diagonal = h.diagonal(dim1=-2, dim2=-1)
    eigen = (torch.where(isolated, diagonal, 0),) if whole else (torch.where(isolated, diagonal, 0), torch.zeros_like(
        diagonal))
    info = torch.zeros_like(ilo)
    searched = ~bad & (ilo != ihi)
    if n > 1:
        h, z, *eigen, info = (_zlahqr if whole else _dlahqr)(h, z, *eigen, ilo, ihi, searched, vectors)
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
