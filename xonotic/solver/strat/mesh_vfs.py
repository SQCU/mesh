from __future__ import annotations

import mmap
import os
import struct
import zlib
from multiprocessing import shared_memory
from pathlib import Path
from typing import Mapping


MESH_VFS_MAGIC = 0x4D564653
MESH_VFS_VERSION = 1
MESH_VFS_HEADER_SIZE = 64
MESH_VFS_ENTRY_SIZE = 224
MESH_VFS_PATH_MAX = 192

_HEADER_FMT = "<IIQQQIIQQQ"
_ENTRY_FMT = f"<{MESH_VFS_PATH_MAX}sQQIIQ"
assert struct.calcsize(_HEADER_FMT) == MESH_VFS_HEADER_SIZE
assert struct.calcsize(_ENTRY_FMT) == MESH_VFS_ENTRY_SIZE


def _align16(v: int) -> int:
    return (v + 15) & ~15


def _norm_vfs_path(path: str) -> str:
    p = path.replace("\\", "/").lstrip("/")
    if len(p.encode("utf-8")) >= MESH_VFS_PATH_MAX:
        raise ValueError(f"VFS path exceeds {MESH_VFS_PATH_MAX - 1} bytes: {path}")
    return p


class MeshVFSStore:
    """Zero-copy shared-memory virtual filesystem store backed by POSIX SHM (`shm://`) or
    a memory-mapped file, read wait-free by DarkPlaces (`mesh_vfs.h` / `fs.c`) via seqlock."""

    def __init__(
        self,
        *,
        name: str | None = None,
        backing_file: str | Path | None = None,
        size_mb: int = 256,
        capacity: int = 4096,
    ):
        self.total_size = int(size_mb) * 1024 * 1024
        self.capacity = int(capacity)
        self.data_offset = _align16(MESH_VFS_HEADER_SIZE + self.capacity * MESH_VFS_ENTRY_SIZE)
        if self.total_size <= self.data_offset + 1024 * 1024:
            raise ValueError("MeshVFSStore total_size too small for entry table + payload arena")

        self._shm: shared_memory.SharedMemory | None = None
        self._fd: int | None = None
        self._mmap: mmap.mmap | None = None
        self._backing_file: Path | None = None

        if backing_file is not None:
            p = Path(backing_file).expanduser()
            p.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(str(p), os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
            os.ftruncate(fd, self.total_size)
            self._fd = fd
            self._mmap = mmap.mmap(fd, self.total_size, access=mmap.ACCESS_WRITE)
            self._buf = memoryview(self._mmap)
            self._backing_file = p
            self.env_spec = str(p)
        else:
            shm_name = name or f"mesh_vfs_{os.getpid()}_{id(self) & 0xFFFF:04x}"
            if len(shm_name) > 28:
                shm_name = shm_name[:28]
            try:
                old = shared_memory.SharedMemory(name=shm_name, create=False)
                old.close()
                old.unlink()
            except FileNotFoundError:
                pass
            self._shm = shared_memory.SharedMemory(name=shm_name, create=True, size=self.total_size)
            self._buf = self._shm.buf
            self.env_spec = f"shm://{shm_name}"

        self._generation = 0
        self._index: dict[str, int] = {}
        self._count = 0
        self._write_cursor = self.data_offset
        self._write_header()

    def _write_header(self) -> None:
        struct.pack_into(
            _HEADER_FMT,
            self._buf,
            0,
            MESH_VFS_MAGIC,
            MESH_VFS_VERSION,
            self.total_size,
            self.data_offset,
            self._generation,
            self.capacity,
            self._count,
            self._write_cursor,
            0,
            0,
        )

    def _begin_write(self) -> None:
        self._generation += 1
        struct.pack_into("<Q", self._buf, 24, self._generation)

    def _end_write(self) -> None:
        self._generation += 1
        self._write_header()

    def clear(self) -> None:
        self._begin_write()
        self._index.clear()
        self._count = 0
        self._write_cursor = self.data_offset
        self._end_write()

    def put_many(self, files: Mapping[str, bytes | bytearray | memoryview]) -> None:
        norm_items = [(_norm_vfs_path(k), memoryview(v)) for k, v in files.items()]
        needed = sum(_align16(len(v)) for _, v in norm_items)
        if self._write_cursor + needed > self.total_size or (self._count + len(norm_items) > self.capacity):
            self.clear()
            if self._write_cursor + needed > self.total_size:
                raise MemoryError(
                    f"VFS payload ({needed} bytes) exceeds arena capacity ({self.total_size - self.data_offset} bytes)"
                )

        self._begin_write()
        try:
            for path, data in norm_items:
                n = len(data)
                off = _align16(self._write_cursor)
                self._buf[off : off + n] = data
                self._write_cursor = _align16(off + n)
                crc = zlib.crc32(data) & 0xFFFFFFFF
                key = path.lower()
                if key in self._index:
                    slot = self._index[key]
                else:
                    slot = self._count
                    self._index[key] = slot
                    self._count += 1
                entry_off = MESH_VFS_HEADER_SIZE + slot * MESH_VFS_ENTRY_SIZE
                struct.pack_into(
                    _ENTRY_FMT,
                    self._buf,
                    entry_off,
                    path.encode("utf-8"),
                    off,
                    n,
                    crc,
                    1,
                    0,
                )
        finally:
            self._end_write()

    def put(self, path: str, data: bytes | bytearray | memoryview) -> None:
        self.put_many({path: data})

    def list_paths(self) -> list[str]:
        return sorted(self._index.keys())

    def bytes_used(self) -> int:
        return self._write_cursor - self.data_offset

    def close(self, unlink: bool = True) -> None:
        if getattr(self, "_buf", None) is not None:
            try:
                self._buf.release()
            except Exception:
                pass
            self._buf = None
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        if unlink and self._backing_file is not None:
            try:
                self._backing_file.unlink(missing_ok=True)
            except OSError:
                pass
        if self._shm is not None:
            try:
                self._shm.close()
            except Exception:
                pass
            if unlink:
                try:
                    self._shm.unlink()
                except FileNotFoundError:
                    pass
            self._shm = None

    def __enter__(self) -> "MeshVFSStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close(unlink=True)


def pack_vfs_bundle(files: Mapping[str, bytes]) -> bytes:
    parts = [struct.pack("<I", len(files))]
    for path, data in sorted(files.items()):
        pb = _norm_vfs_path(path).encode("utf-8")
        parts.append(struct.pack("<IQ", len(pb), len(data)))
        parts.append(pb)
        parts.append(data)
    return b"".join(parts)


def unpack_vfs_bundle(blob: bytes | memoryview) -> dict[str, bytes]:
    mv = memoryview(blob)
    (n,) = struct.unpack_from("<I", mv, 0)
    off = 4
    out: dict[str, bytes] = {}
    for _ in range(n):
        plen, dlen = struct.unpack_from("<IQ", mv, off)
        off += 12
        path = bytes(mv[off : off + plen]).decode("utf-8")
        off += plen
        out[path] = bytes(mv[off : off + dlen])
        off += dlen
    return out
