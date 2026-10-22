"""The one capacity guard of every writer of the mesh stack (mesh-disk.h; its floor is read from there, where it is
stated).  room(path, nbytes, what): True while the filesystem holding `path` keeps the floor free after `nbytes`
more; else one line on stderr naming what is skipped and why, and False.  write() writes a whole file only with room;
sink() is a file for a child process's output, os.devnull without room at its open (its later writes unchecked).  No
write error of these raises: the computation a record serves never fails for want of disk."""
import os
import re
import sys
from pathlib import Path

FLOOR = int(re.search(r'^#define MESH_DISK_FLOOR_GIB (\d+)', Path(__file__).with_name('mesh-disk.h').read_text(),
                      re.M).group(1)) << 30


def _say(text):
    try:
        print(f'mesh-disk: {text}', file=sys.stderr, flush=True)
    except OSError:
        pass


def free(path):
    """(free bytes, the directory read) of the filesystem holding `path` or its nearest existing ancestor;
    (None, path) where none is found."""
    at = Path(os.path.abspath(path))
    while True:
        try:
            s = os.statvfs(at)
            return s.f_bavail * s.f_frsize, at
        except FileNotFoundError:
            if at == at.parent:
                return None, at
            at = at.parent
        except OSError:
            return None, at


def room(path, nbytes=0, what=None, quiet=False):
    have, at = free(path)
    if have is None or have >= FLOOR + nbytes:
        return True
    if not quiet:
        _say(f'{what or path} not written ({nbytes} bytes): {at} has {have} bytes free, under the floor of '
             f'{FLOOR >> 30} GiB (mesh-disk.h)')
    return False


def write(path, data, what=None, mode='w'):
    """`data` (str or bytes) written to `path` (mode 'w' or 'a') where there is room; True when written."""
    raw = data.encode() if isinstance(data, str) else bytes(data)
    if not room(path, len(raw), what):
        return False
    try:
        with open(path, mode + 'b') as out:
            out.write(raw)
        return True
    except OSError as error:
        _say(f'{what or path} not written: {error}')
        return False


def sink(path, what=None, mode='w'):
    """A file for a child process's stdout: `path` where there is room when it is opened, else os.devnull (the
    child's output is then not kept, said once).  Room is checked at open only: the child's later writes are
    not checked."""
    return open(path if room(path, 0, what or f'the output {path}') else os.devnull, mode)
