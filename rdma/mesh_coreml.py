# design/the-compiler.md, order 4; mesh-coreml.m
import ctypes as C
import os

LIB = C.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libmesh-coreml.dylib'))
P, U = C.c_void_p, C.c_uint32
MESSAGE, PLAN = 4096, 1 << 20


class View(C.Structure):
    _fields_ = [('name', C.c_char_p), ('pages', P), ('shape', C.POINTER(C.c_int64)), ('stride', C.POINTER(C.c_int64)),
                ('rank', C.c_int32), ('type', C.c_int32)]


for name, result, arguments in (
        ('mesh_coreml_model', P, [C.c_char_p, C.c_char_p, C.c_char_p, C.c_size_t]),
        ('mesh_coreml_release', None, [P]),
        ('mesh_coreml_predict', C.c_int, [P, C.POINTER(View), U, C.POINTER(View), U, C.c_char_p, C.c_size_t]),
        ('mesh_coreml_plan', C.c_int, [C.c_char_p, C.c_char_p, C.c_size_t])):
    function = getattr(LIB, name)
    function.restype, function.argtypes = result, arguments


def model(package, compiled):
    message = C.create_string_buffer(MESSAGE)
    made = LIB.mesh_coreml_model(str(package).encode(), str(compiled).encode(), message, MESSAGE)
    if not made:
        raise RuntimeError(message.value.decode())
    return made


def predict(made, arguments, results):
    message = C.create_string_buffer(MESSAGE)
    written = LIB.mesh_coreml_predict(made, arguments, len(arguments), results, len(results), message, MESSAGE)
    if written < 0:
        raise RuntimeError(message.value.decode())
    return written


def plan(compiled):
    text = C.create_string_buffer(PLAN)
    if LIB.mesh_coreml_plan(str(compiled).encode(), text, PLAN):
        raise RuntimeError(text.value.decode())
    return [tuple(line.split(' ')) for line in text.value.decode().splitlines()]
