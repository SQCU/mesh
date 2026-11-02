#!/usr/bin/env python3
"""A preparation's cost, stage by stage and call by call, from the lines nccl-mesh.c prepare_record writes in
MESH_PREPARE_RECORD's directory (a file a process: prepare-<rank>-<pid>.jsonl). It prints; it changes nothing.

    python3 extras/prepare_report.py FILE_OR_DIRECTORY... [--top N]

For each file: each stage's total microseconds and count (routing, write_binding, call, plan_calls, schedule_group,
order_transfers, schedule_consumption, definition), then the N slowest calls with their kind (what, target), elements,
tile bytes, the steps they made and their group's solved trees, routing demands and catalog trees.
"""
import argparse
import collections
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('paths', nargs='+')
    parser.add_argument('--top', type=int, default=12)
    args = parser.parse_args()
    files = sorted(f for p in map(Path, args.paths) for f in (p.rglob('prepare-*.jsonl') if p.is_dir() else [p]))
    for f in files:
        rows = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
        print(f'== {f} ({len(rows)} lines)')
        totals = collections.defaultdict(lambda: [0.0, 0])
        for r in rows:
            totals[r['stage']][0] += r['us']
            totals[r['stage']][1] += 1
        for stage, (us, n) in sorted(totals.items(), key=lambda item: -item[1][0]):
            print(f'   {stage:22} {us / 1e6:10.3f} s  {n:7} lines  {us / n:12.1f} us each')
        calls = sorted((r for r in rows if r['stage'] == 'call'), key=lambda r: -r['us'])
        for r in calls[:args.top]:
            print(f"   group {r['group']:5} call {r['call']:4} what {r['what']:2} target {r['target']:2} count {r['count']:10} "
                  f"tile {r['tile_bytes']:8} steps {r['steps']:9} trees {r['trees']:8} demands {r['demands']:6} "
                  f"catalog {r['catalog']:8} {r['us'] / 1e3:10.2f} ms")


if __name__ == '__main__':
    main()
