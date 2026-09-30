"""The one capacity guard of every writer of the mesh stack (mesh-disk.h; its two numbers are read from there,
where they are stated).  room(path, nbytes, what): True while the filesystem holding `path` keeps the floor free
after `nbytes` more; else one line on stderr naming what is skipped and why, and False.  write() writes a whole
file only with room; stream() is a file each of whose writes is made only with room (the first skipped says so);
sink() is a file for a child process's output, os.devnull without room; cap() moves a log past the cap to
<path>.1.  No write error of these raises: the computation a record serves never fails for want of disk."""
import os
import re
import sys
from pathlib import Path

_STATED = dict((k, int(v)) for k, v in re.findall(r'^#define (MESH_DISK_FLOOR_GIB|MESH_LOG_CAP_MIB) (\d+)',
                                                  Path(__file__).with_name('mesh-disk.h').read_text(), re.M))
FLOOR = _STATED['MESH_DISK_FLOOR_GIB'] << 30
LOG_CAP = _STATED['MESH_LOG_CAP_MIB'] << 20


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


class stream:
    """A file written piece by piece (a run's records), each write made only where there is room: the first
    write skipped says so, and writes go on once there is room again."""

    def __init__(self, path, what=None, mode='a'):
        self.path, self.name, self.what, self.skipping = path, str(path), what or str(path), False
        self.file = open(path, mode) if room(path, 0, self.what) else None

    def write(self, text):
        if self.file is None:
            return
        if not room(self.path, len(text), self.what, quiet=self.skipping):
            self.skipping = True
            return
        self.skipping = False
        try:
            self.file.write(text)
        except OSError as error:
            _say(f'{self.what}: a write not made: {error}')

    def flush(self):
        if self.file is not None:
            try:
                self.file.flush()
            except OSError:
                pass

    def close(self):
        if self.file is not None:
            try:
                self.file.close()
            except OSError:
                pass
            self.file = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def sink(path, what=None, mode='w'):
    """A file for a child process's stdout: `path` where there is room, else os.devnull (the child's output
    is then not kept, said once)."""
    return open(path if room(path, 0, what or f'the output {path}') else os.devnull, mode)


def cap(path):
    """A log at `path` past the cap moved to <path>.1 (the one before dropped); True when moved."""
    try:
        if os.path.getsize(path) > LOG_CAP:
            os.replace(path, f'{path}.1')
            return True
    except OSError:
        pass
    return False
