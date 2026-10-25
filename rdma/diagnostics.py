"""Configured diagnostic entry points backed by the installed mesh library."""
import ctypes
import pathlib
import sys


def _library():
    return ctypes.CDLL(str(pathlib.Path(__file__).resolve().with_name('libmesh.dylib')))


def quiet(arguments=None):
    """Measure one explicitly selected physical wire; no application fallback."""
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    values = [ctypes.create_string_buffer(value.encode()) for value in ['quiet', *arguments]]
    argv = (ctypes.c_char_p * len(values))(*(ctypes.cast(value, ctypes.c_char_p) for value in values))
    call = _library().mesh_diagnostic_quiet
    call.argtypes, call.restype = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)], ctypes.c_int
    return int(call(len(values), argv))


def retire(arguments=None):
    """Observe client-exit retirement through the library's diagnostic API."""
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    if len(arguments) > 1:
        raise ValueError('retire accepts only a round count')
    rounds = int(arguments[0]) if arguments else 40
    if not 0 < rounds < (1 << 31):
        raise ValueError('round count must be a positive int32')
    call = _library().mesh_diagnostic_retire
    call.argtypes, call.restype = [ctypes.c_int], ctypes.c_int
    return int(call(rounds))


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ('quiet', 'retire'):
        raise SystemExit('diagnostics quiet list PEER | quiet run ROLE WIRE ROUNDS GAPS | retire [ROUNDS]')
    raise SystemExit({'quiet': quiet, 'retire': retire}[sys.argv[1]](sys.argv[2:]))


if __name__ == '__main__':
    main()
