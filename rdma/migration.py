"""Compile a changed tensor partition into indexed writes of existing bytes.

This module describes exact indexed ranges and compact affine row masks; it
does not issue transport calls. Offsets refer to each rank's original tensor
layout. Aligned ranges can use direct physical writes. A page-only transport
can explicitly lower sub-page ranges to separate padded ingress frames: the
producer streams the selected bytes and the consumer writes only masked final
positions. Padding never widens a DMA write into neighboring live output.
The mask/address descriptor is compiled, not selected by a received header.
There is no receipt collective or whole-operand join in that realization.

Each source lease remains live until its local readers and all DMA reads of
these ranges have ended. Local DMA-read completion can release it without a
remote acknowledgment; a downstream data-dependency proof can establish the
same fact when one exists. Merely completing an unrelated incoming write is
not such a proof.
"""
from dataclasses import dataclass
import math
from allocate import partition_bounds


@dataclass(frozen=True)
class Write:
    source: int
    destination: int
    source_offset: int
    destination_offset: int
    bytes: int


@dataclass(frozen=True)
class Migration:
    source_shapes: tuple
    destination_shapes: tuple
    writes: tuple[Write, ...]

    def source_uses(self, rank):
        """Indices of the existing-byte readers that own this source lease."""
        return tuple(i for i, write in enumerate(self.writes) if write.source == rank)

    def destination_writes(self, rank):
        return tuple(i for i, write in enumerate(self.writes) if write.destination == rank)

    def unaligned(self, granule):
        """Ranges a backend restricted to whole aligned granules cannot issue.

        These ranges need a different explicit lowering: byte-range support,
        another layout, or affine masked frames in separate ingress storage.
        Padding must never overwrite neighboring live destination bytes.
        """
        if not isinstance(granule, int) or granule <= 0:
            raise ValueError('a transfer granule must be positive')
        return tuple(i for i, w in enumerate(self.writes)
                     if any(value % granule for value in (w.source_offset, w.destination_offset, w.bytes)))


@dataclass(frozen=True)
class RowWrite:
    """One affine byte run per tensor prefix, without enumerating the prefixes.

    Payload byte i maps to base + (i // row_bytes) * stride + i % row_bytes.
    An ingress frame contains a contiguous interval of that payload, padded
    in its own storage. Its final-frame validity mask is i < bytes; padding
    never names a final-destination byte. DMA nevertheless reads the complete
    padded frame and owns that complete source/relay lease through retirement.
    """
    source: int
    destination: int
    source_offset: int
    destination_offset: int
    row_bytes: int
    rows: int
    source_stride: int
    destination_stride: int

    @property
    def bytes(self):
        return self.row_bytes * self.rows

    def source_byte(self, i):
        return self.source_offset + (i // self.row_bytes) * self.source_stride + i % self.row_bytes

    def destination_byte(self, i):
        return self.destination_offset + (i // self.row_bytes) * self.destination_stride + i % self.row_bytes

    def frames(self, frame_bytes):
        if not isinstance(frame_bytes, int) or frame_bytes <= 0:
            raise ValueError('frame bytes must be positive')
        return (self.bytes + frame_bytes - 1) // frame_bytes

    def aligned(self, granule):
        values = (self.source_offset, self.destination_offset, self.row_bytes)
        if self.rows > 1:
            values += (self.source_stride, self.destination_stride)
        return not any(v % granule for v in values)


@dataclass(frozen=True)
class MigrationRows:
    source_shapes: tuple
    destination_shapes: tuple
    runs: tuple[RowWrite, ...]


def _geometry(shape, dim, old_parts, new_parts, element_size):
    shape, old_parts, new_parts = tuple(shape), tuple(old_parts), tuple(new_parts)
    if (not shape or any(not isinstance(n, int) or n < 0 for n in shape) or not isinstance(dim, int) or
            not -len(shape) <= dim < len(shape) or not isinstance(element_size, int) or element_size <= 0):
        raise ValueError('invalid tensor shape, axis, or element size')
    if (not old_parts or len(old_parts) != len(new_parts) or
            any(not isinstance(n, int) or n < 0 for n in old_parts + new_parts) or
            not sum(old_parts) or not sum(new_parts)):
        raise ValueError('partitions must have positive total weights and the same ranks')
    dim %= len(shape)
    prefix, row_bytes = math.prod(shape[:dim]), math.prod(shape[dim + 1:]) * element_size
    before, after = (partition_bounds(parts, shape[dim]) for parts in (old_parts, new_parts))
    old, new = (tuple(b - a for a, b in zip(bounds, bounds[1:])) for bounds in (before, after))
    return shape, dim, old, new, before, after, prefix, row_bytes


def compile_rows(shape, dim, old_parts, new_parts, element_size):
    """Compact affine masks, one per nonempty partition intersection, AOT.

    Descriptor count depends on the partitions, not the number of prefix rows
    or pages. No bitmap or temporary prefix-sized list is built. All offsets,
    strides and row sizes are bytes; payload is copied bit-for-bit, with no
    floating conversion. Local retention and remote transfer use the same map.
    """
    shape, dim, old, new, before, after, prefix, row_bytes = _geometry(shape, dim, old_parts, new_parts, element_size)
    runs = []
    for source in range(len(old)):
        for destination in range(len(new)):
            first, last = max(before[source], after[destination]), min(before[source + 1], after[destination + 1])
            if first >= last or not row_bytes or not prefix:
                continue
            runs.append(RowWrite(source, destination, (first - before[source]) * row_bytes,
                                 (first - after[destination]) * row_bytes, (last - first) * row_bytes, prefix,
                                 old[source] * row_bytes, new[destination] * row_bytes))
    shapes = lambda parts: tuple(shape[:dim] + (part,) + shape[dim + 1:] for part in parts)
    return MigrationRows(shapes(old), shapes(new), tuple(runs))


def compile_migration(shape, dim, old_parts, new_parts, element_size):
    """Every retained or transferred range for a partition change, AOT.

    Parts are units of the same logical axis, including zero-share ranks. Rank
    and prefix order are deterministic; adjacent compatible ranges coalesce.
    The result covers each destination byte exactly once and reads each
    original logical element once. Same-rank ranges can be realized by page
    aliases when their layout permits; otherwise they remain local writes.
    """
    shape, dim, old, new, before, after, prefix, row_bytes = _geometry(shape, dim, old_parts, new_parts, element_size)
    writes = []
    for source in range(len(old)):
        for destination in range(len(new)):
            first = max(before[source], after[destination])
            last = min(before[source + 1], after[destination + 1])
            if first >= last or row_bytes == 0:
                continue
            for p in range(prefix):
                current = Write(source, destination, (p * old[source] + first - before[source]) * row_bytes,
                                (p * new[destination] + first - after[destination]) * row_bytes,
                                (last - first) * row_bytes)
                previous = writes[-1] if writes else None
                if (previous is not None and previous.source == source and previous.destination == destination and
                        previous.source_offset + previous.bytes == current.source_offset and
                        previous.destination_offset + previous.bytes == current.destination_offset):
                    writes[-1] = Write(source, destination, previous.source_offset, previous.destination_offset,
                                       previous.bytes + current.bytes)
                else:
                    writes.append(current)
    shapes = lambda parts: tuple(shape[:dim] + (part,) + shape[dim + 1:] for part in parts)
    return Migration(shapes(old), shapes(new), tuple(writes))
