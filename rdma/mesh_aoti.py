# mesh-aoti.cpp; design/compiler-native-assignment-2026-11-13.md;
# output_data/compiler-workgroup-20261113/executor/aoti-binding-contract.md
import ctypes as C
import os
import weakref

import torch

# torch 2.14 _inductor/cpp_builder.py: on Darwin a model links with -undefined dynamic_lookup, so its effect shim calls
# bind at load to this library's exports
LIB = C.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libmesh-aoti.dylib'), mode=C.RTLD_GLOBAL)
P, Q, Z = C.c_void_p, C.c_uint64, C.c_size_t
for name, result, arguments in (
        ('mesh_aoti_load', P, [C.c_char_p, C.c_char_p, C.c_char_p, Z]),
        ('mesh_aoti_run', C.c_int, [P, C.POINTER(P), Z, C.POINTER(P), Z, C.c_char_p, Z]),
        ('mesh_aoti_free', None, [P]),
        ('mesh_aoti_bind', P, [P, Z, C.POINTER(Q), Z, C.POINTER(Q), C.POINTER(Q), Z, C.POINTER(P), C.c_int64, Q, Q,
                               Q, C.c_char_p, Z]),
        ('mesh_aoti_call', C.c_int, [P, C.POINTER(P), Z, C.POINTER(P), Z, C.c_char_p, Z]),
        ('mesh_aoti_unbind', None, [P])):
    function = getattr(LIB, name)
    function.restype, function.argtypes = result, arguments
MESSAGE = 4096
ENTRY = C.cast(LIB.mesh_aoti_run, P).value
# torch 2.14 torch/csrc/inductor/aoti_runner/pybind.cpp: the handles cross as unnamed capsules
CAPSULE = C.pythonapi
CAPSULE.PyCapsule_GetPointer.restype, CAPSULE.PyCapsule_GetPointer.argtypes = P, [C.py_object, C.c_char_p]
CAPSULE.PyCapsule_New.restype, CAPSULE.PyCapsule_New.argtypes = C.py_object, [P, C.c_char_p, P]

# rdma/mesh.h MESH_SEND, MESH_RECEIVE; the invocation operand {mesh_ctx address, plan, execution}
SEND, RECEIVE = 0, 1
INVOCATION = torch.zeros(3, dtype=torch.uint64)
# torch 2.14 torch/library.py Library.define and register_fake; mesh-call.h mesh_host_started, mesh_transfers_start;
# mesh-collective.h mesh_host_publish, mesh_host_wait.  Every effect writes the invocation operand, the transport's
# state, so the effects keep their order through the compiled graph's own dependences; publish writes `sent`, the
# prepared storage it stores `value` into, and wait writes `view`, the prepared storage it makes ready
LIBRARY = torch.library.Library("mesh", "FRAGMENT")
for schema in ("mesh_start(Tensor(a!) invocation) -> ()",
               "mesh_publish(Tensor(a!) invocation, int transfer, Tensor value, Tensor(b!) sent) -> ()",
               "mesh_wait(Tensor(a!) invocation, int transfer, int direction, int delay, Tensor(b!) view) -> ()"):
    LIBRARY.define(schema)
START, PUBLISH, WAIT = (torch.ops.mesh.mesh_start.default, torch.ops.mesh.mesh_publish.default,
                        torch.ops.mesh.mesh_wait.default)
for effect in (START, PUBLISH, WAIT):
    torch.library.register_fake(effect, lambda *arguments: None, lib=LIBRARY)
# torch 2.14 _inductor/config.py aot_inductor.custom_ops_to_c_shims; _inductor/codegen/cpp_wrapper_cpu.py
# get_c_shim_func_name, generate_c_shim_fallback_kernel; _inductor/ir.py FallbackKernel.create, a void result's handle
SHIMS = {
    START: ["AOTITorchError aoti_torch_cpu_mesh_start(AtenTensorHandle invocation, AtenTensorHandle *ret0)"],
    PUBLISH: ["AOTITorchError aoti_torch_cpu_mesh_publish(AtenTensorHandle invocation, int64_t transfer, "
              "AtenTensorHandle value, AtenTensorHandle sent, AtenTensorHandle *ret0)"],
    WAIT: ["AOTITorchError aoti_torch_cpu_mesh_wait(AtenTensorHandle invocation, int64_t transfer, int64_t direction, "
           "int64_t delay, AtenTensorHandle view, AtenTensorHandle *ret0)"]}


# torch 2.14 torch/_C/_aoti unsafe_alloc_void_ptrs_from_tensors, alloc_tensors_by_stealing_from_void_ptrs
def called(call, native, name, count, output_count, arguments):
    if len(arguments) != count:
        raise TypeError(f"{name} takes {count} arguments ({len(arguments)} given)")
    handles = torch._C._aoti.unsafe_alloc_void_ptrs_from_tensors(arguments)
    inputs = (P * count)(*(CAPSULE.PyCapsule_GetPointer(handle, None) for handle in handles))
    outputs, error = (P * output_count)(), C.create_string_buffer(MESSAGE)
    if call(native, inputs, count, outputs, output_count, error, MESSAGE):
        raise RuntimeError(error.value.decode())
    return torch._C._aoti.alloc_tensors_by_stealing_from_void_ptrs(
        [CAPSULE.PyCapsule_New(handle, None, None) for handle in outputs])


class Model:
    def __init__(self, package_path, model_name, input_count, output_count):
        error = C.create_string_buffer(MESSAGE)
        self.context = LIB.mesh_aoti_load(os.fsencode(package_path), model_name.encode(), error, MESSAGE)
        if not self.context:
            raise RuntimeError(error.value.decode())
        weakref.finalize(self, LIB.mesh_aoti_free, self.context)
        self.package_path, self.model_name, self.entry = package_path, model_name, ENTRY
        self.input_count, self.output_count = input_count, output_count

    def __call__(self, *arguments):
        return called(LIB.mesh_aoti_run, self.context, self.model_name, self.input_count, self.output_count, arguments)

    def prepare(self, arguments, views, invocation=None, context=0, plan=0, execution=0):
        return Prepared(self, arguments, views, invocation, context, plan, execution)


# design/compiler-native-assignment-2026-11-13.md, prepared binding: arguments are the AOT input positions of the
# ordinary call arguments; views are (AOT input position, prepared tensors over its solved copies in copy order)
class Prepared:
    def __init__(self, model, arguments, views, invocation, context, plan, execution):
        first = [0]
        for _, copies in views:
            first.append(first[-1] + len(copies))
        handles = torch._C._aoti.unsafe_alloc_void_ptrs_from_tensors([copy for _, copies in views for copy in copies])
        error = C.create_string_buffer(MESSAGE)
        self.binding = LIB.mesh_aoti_bind(
            model.context, model.input_count, (Q * len(arguments))(*arguments), len(arguments),
            (Q * len(views))(*(position for position, _ in views)), (Q * len(first))(*first), len(views),
            (P * len(handles))(*(CAPSULE.PyCapsule_GetPointer(handle, None) for handle in handles)),
            -1 if invocation is None else invocation, context, plan, execution, error, MESSAGE)
        if not self.binding:
            raise RuntimeError(error.value.decode())
        weakref.finalize(self, LIB.mesh_aoti_unbind, self.binding)
        self.model, self.arguments = model, list(arguments)

    def __call__(self, *arguments):
        return called(LIB.mesh_aoti_call, self.binding, self.model.model_name, len(self.arguments),
                      self.model.output_count, arguments)
