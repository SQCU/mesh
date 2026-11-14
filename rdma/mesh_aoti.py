# mesh-aoti.cpp; design/compiler-native-assignment-2026-11-13.md;
# output_data/compiler-workgroup-20261113/executor/aoti-binding-contract.md
import ctypes as C
import os
import weakref

import torch

LIB = C.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libmesh-aoti.dylib'))
P, Z = C.c_void_p, C.c_size_t
for name, result, arguments in (
        ('mesh_aoti_load', P, [C.c_char_p, C.c_char_p, C.c_char_p, Z]),
        ('mesh_aoti_run', C.c_int, [P, C.POINTER(P), Z, C.POINTER(P), Z, C.c_char_p, Z]),
        ('mesh_aoti_free', None, [P])):
    function = getattr(LIB, name)
    function.restype, function.argtypes = result, arguments
MESSAGE = 4096
ENTRY = C.cast(LIB.mesh_aoti_run, P).value
# torch 2.14 torch/csrc/inductor/aoti_runner/pybind.cpp: the handles cross as unnamed capsules
CAPSULE = C.pythonapi
CAPSULE.PyCapsule_GetPointer.restype, CAPSULE.PyCapsule_GetPointer.argtypes = P, [C.py_object, C.c_char_p]
CAPSULE.PyCapsule_New.restype, CAPSULE.PyCapsule_New.argtypes = C.py_object, [P, C.c_char_p, P]


# torch 2.14 torch/_C/_aoti unsafe_alloc_void_ptrs_from_tensors, alloc_tensors_by_stealing_from_void_ptrs
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
        if len(arguments) != self.input_count:
            raise TypeError(f"{self.model_name} takes {self.input_count} arguments ({len(arguments)} given)")
        handles = torch._C._aoti.unsafe_alloc_void_ptrs_from_tensors(arguments)
        inputs = (P * self.input_count)(*(CAPSULE.PyCapsule_GetPointer(handle, None) for handle in handles))
        outputs, error = (P * self.output_count)(), C.create_string_buffer(MESSAGE)
        if LIB.mesh_aoti_run(self.context, inputs, self.input_count, outputs, self.output_count, error, MESSAGE):
            raise RuntimeError(error.value.decode())
        return torch._C._aoti.alloc_tensors_by_stealing_from_void_ptrs(
            [CAPSULE.PyCapsule_New(handle, None, None) for handle in outputs])
