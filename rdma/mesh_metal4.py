# design/the-compiler.md, order 4; mesh-metal4.m
import ctypes as C
import hashlib
import json
import math
import os
import pathlib
import subprocess
import tempfile
import weakref

import torch

LIB = C.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libmesh-metal4.dylib'))
P, Q, U = C.c_void_p, C.c_uint64, C.c_uint32


class Dispatch(C.Structure):
    _fields_ = [('pipeline', P), ('table', P), ('address', C.POINTER(Q)), ('addresses', U), ('barrier', U),
                ('groups', U), ('padding', U), ('grid', Q * 3), ('group', Q * 3)]


for name, result, arguments in (
        ('mesh_metal4_pipeline', P, [C.c_char_p, C.c_char_p, C.c_char_p, C.c_size_t]),
        ('mesh_metal4_width', Q, [P]),
        ('mesh_metal4_table', P, [U]),
        ('mesh_metal4_buffer', P, [Q]),
        ('mesh_metal4_view', P, [P, Q]),
        ('mesh_metal4_release', None, [P]),
        ('mesh_metal4_address', Q, [P]),
        ('mesh_metal4_contents', P, [P]),
        ('mesh_metal4_after', None, [P, P, P, P]),
        ('mesh_metal4_fence', P, []),
        ('mesh_metal4_stream', P, [C.POINTER(P), U]),
        ('mesh_metal4_record', P, []),
        ('mesh_metal4_times', None, [P, C.POINTER(C.c_double)]),
        ('mesh_metal4_commit', None, [P, C.POINTER(Dispatch), U, C.POINTER(P), U, C.POINTER(P), U, P, P]),
        ('mesh_metal4_wait', C.c_int, [C.POINTER(P), U, C.c_char_p, C.c_size_t]),
        ('mesh_metal4_reset', None, [P])):
    function = getattr(LIB, name)
    function.restype, function.argtypes = result, arguments
MESSAGE = 4096


def pipeline(source, name):
    message = C.create_string_buffer(MESSAGE)
    made = LIB.mesh_metal4_pipeline(source.encode(), name.encode(), message, MESSAGE)
    if not made:
        raise RuntimeError(message.value.decode())
    return made


def wait(records):
    message = C.create_string_buffer(MESSAGE)
    if LIB.mesh_metal4_wait(records, len(records), message, MESSAGE):
        raise RuntimeError(message.value.decode())


def times(record):
    found = (C.c_double * 2)()
    LIB.mesh_metal4_times(record, found)
    return tuple(found)


# design/compilation-composition-change-request.md, items 5 and 7
COMMIT, WAIT, CHECK, SENT, RECEIVED, COPIED = "commit", "wait", "check", "send", "recv", "copy"
HOSTED = C.CFUNCTYPE(Q, U)
LIBRARIES = {}
PROGRAM = """#include <cstdint>
#include <cstring>
struct mesh_failure { uint32_t failed; char message[4096]; };
typedef uint64_t (*mesh_hosted)(uint32_t);
"""


class Failure(C.Structure):
    _fields_ = [('failed', U), ('message', C.c_char * MESSAGE)]


def native(source, linked=()):
    if source not in LIBRARIES:
        directory = tempfile.TemporaryDirectory()
        stem = pathlib.Path(directory.name) / hashlib.sha256(source.encode()).hexdigest()[:16]
        stem.with_suffix(".cpp").write_text(source)
        built = subprocess.run(["/usr/bin/xcrun", "clang++", "-std=c++20", "-O2", "-dynamiclib", "-framework",
                                "Accelerate", str(stem.with_suffix(".cpp")), *linked, "-o",
                                str(stem.with_suffix(".dylib"))], capture_output=True, text=True)
        if built.returncode:
            raise RuntimeError(built.stderr)
        LIBRARIES[source] = (C.CDLL(str(stem.with_suffix(".dylib"))), directory)
    return LIBRARIES[source][0]


def failed(message, ended):
    return [f"strlcpy(f->message, {message}, sizeof f->message);"] * bool(message) + [
        "f->failed = 1;", *ended, "return;"]


# design/compilation-composition-change-request.md, item 7; design/torch-source-lowering-change-request.md, item 3
def program(entries, storages, begun, ended):
    body = [f"uint64_t *g = a + {storages}, *o = a + {2 * storages}, *m = a + {3 * storages};", *begun]
    for kind, lines, *given in entries:
        if kind is SENT:
            body += [f"host[0]({given[0]});", "if (f->failed) {", *lines, *ended, "return; }"]
        elif kind in (RECEIVED, COPIED):
            k, storage, outstanding = given
            body += [f"a[{storage}] = host[{1 if kind is RECEIVED else 2}]({k});", "if (f->failed) {", *outstanding,
                     *ended, "return; }", *lines]
        elif kind is CHECK:
            kind_, storage, offset, shape, stride, zero, message = given
            steps = [math.prod(shape[d + 1:]) for d in range(len(shape))]
            at = " + ".join([str(offset)] + [f"i / {step} % {n} * {s}" for n, s, step in zip(shape, stride, steps)])
            body += ["{", f"const {kind_} *b = (const {kind_} *)a[{storage}];", "bool bad = false;",
                     f"for (uint64_t i = 0; i < {math.prod(shape)}; i++) bad |= b[{at}] {'!=' if zero else '=='} 0;",
                     "if (bad) {", *lines, *failed(json.dumps(message), ended), "}", "}"]
        elif lines:
            body += ["{", *lines, "}"]
    return ('extern "C" void run(uint64_t *a, void **h, const mesh_hosted *host, mesh_failure *f) {\n    '
            + "\n    ".join([*body, *ended]) + "\n}\n")


def tensor(pages, view):
    _, dtype, shape, stride, offset = view
    if not all(shape):
        return torch.empty(shape, dtype=dtype)
    return pages.as_strided(shape, stride, pages.storage_offset() + offset)


# design/compilation-composition-change-request.md, item 5
class Function:
    def __init__(self, call, storages, handles, static, given, made, owned, sends, receives, copies, results,
                 operators, retained=(), fixed=()):
        self.call, self.storages, self.given, self.made, self.owned = call, storages, given, made, dict(owned)
        self.sends, self.receives, self.copies, self.results = sends, receives, copies, results
        self.send, self.recv, self.copy = operators
        self.a, self.h = (Q * max(4 * storages, 1))(), (P * max(storages + len(handles), 1))()
        self.h[storages:storages + len(handles)] = handles
        for storage, handle, address, device in static:
            self.h[storage], self.a[storage], self.a[storages + storage] = handle, address, device
        call.restype, call.argtypes = None, [C.POINTER(Q), C.POINTER(P), C.POINTER(HOSTED), C.POINTER(Failure)]
        self.failure, self.owner, self.raised = Failure(), {}, []
        self.hosted = (HOSTED * 3)(HOSTED(self.sent), HOSTED(self.received), HOSTED(self.copied))
        self.retained, self.fixed = retained, dict(fixed)
        self.owner = dict(self.fixed)
        for storage, pages in fixed:
            self.hold(storage, pages)

    def pages(self, storage):
        if storage not in self.owner:
            dtype, count = self.owned[storage]
            pages = (C.c_byte * (count * dtype.itemsize)).from_address(self.a[storage])
            weakref.finalize(pages, LIB.mesh_metal4_release, self.h[storage])
            self.owner[storage] = torch.frombuffer(pages, dtype=dtype)
        return self.owner[storage]

    def hold(self, storage, pages):
        self.owner[storage] = pages
        on = pages.device.type == "mps"
        self.a[storage] = pages.untyped_storage().data_ptr() if on else pages.data_ptr()
        self.a[2 * self.storages + storage] = pages.storage_offset() * pages.itemsize if on else 0
        self.a[3 * self.storages + storage] = on
        return self.a[storage]

    def attempt(self, call, *arguments):
        try:
            return call(*arguments) or 0
        except Exception as error:
            self.raised.append(error)
            self.failure.failed = 1
            return 0

    def sent(self, k):
        view, channel = self.sends[k]
        return self.attempt(lambda: self.send(tensor(self.pages(view[0]), view), channel))

    def received(self, k):
        storage, channel = self.receives[k]
        return self.attempt(lambda: self.hold(storage, self.recv(channel)))

    def copied(self, k):
        storage, view, options = self.copies[k]
        return self.attempt(lambda: self.hold(storage, self.copy(tensor(self.pages(view[0]), view), **options)))

    def __call__(self, *arguments):
        self.owner, self.raised, self.failure.failed = dict(self.fixed), [], 0
        for storage, position in self.given:
            self.hold(storage, arguments[position])
        for storage, dtype, count in self.made:
            self.hold(storage, torch.empty(count, dtype=dtype))
        self.call(self.a, self.h, self.hosted, C.byref(self.failure))
        owner = {storage: self.pages(storage) for storage in self.owned} | self.owner
        self.owner = {}
        if self.failure.failed:
            raise (self.raised or [RuntimeError(self.failure.message.decode())])[0]
        return tuple(tensor(owner.get(r[0]), r) if isinstance(r, tuple) else r for r in self.results)
