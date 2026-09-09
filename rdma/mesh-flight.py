import argparse
import json
from pathlib import Path
import struct
import time


HEADER = struct.Struct("<4I7Q1024s")
EVENT = struct.Struct("<5QqiI")
PENDING = -(1 << 63)


def read(path):
    data = Path(path).read_bytes()
    magic, version, size, capacity, pid, node, span, sequence, heartbeat, _, _, names = HEADER.unpack_from(data)
    if (magic, version, size) != (0x4D464C54, 1, EVENT.size):
        raise ValueError(f"unsupported flight format: {magic:x}/{version}/{size}")
    names = names.decode().rstrip("\0").split("\0")
    events = []
    for offset in range(HEADER.size, HEADER.size + capacity * size, size):
        seq, stamp, obj, arg, count, result, error, operation = EVENT.unpack_from(data, offset)
        if not seq or seq > sequence:
            continue
        events.append(dict(sequence=seq, time_us=stamp, operation=names[operation],
                           object=hex(obj), argument=arg, bytes=count,
                           result=None if result == PENDING else result, errno=error,
                           pending=result == PENDING))
    events.sort(key=lambda event: event["sequence"])
    return dict(path=str(path), pid=pid, node=node, region_bytes=span, sequence=sequence,
                heartbeat_us=heartbeat, heartbeat_age_ms=max(0, (time.time_ns() // 1000 - heartbeat) // 1000),
                events=events)


def main():
    parser = argparse.ArgumentParser(description="Read bridge flight records without opening a verbs device")
    parser.add_argument("paths", type=Path, nargs="+")
    parser.add_argument("--last", type=int, default=32)
    args = parser.parse_args()
    for path in args.paths:
        report = read(path)
        report["events"] = report["events"][-args.last:]
        print(json.dumps(report))


if __name__ == "__main__":
    main()
