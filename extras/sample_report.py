#!/usr/bin/env python3
"""A macOS sample(1) report read three ways, for a rank sampled while it hung or ran long:

    python3 extras/sample_report.py SAMPLE.txt [--under FUNCTION] [--top N]

  - native calls the interpreter made (the first frame outside python3.12 below a python3.12 frame, main thread): where
    a Python program's time went inside extensions and libraries;
  - entries into libnccl-mesh (the first frame of that library below a caller outside it);
  - with --under FUNCTION, the samples of each direct callee beneath every frame named FUNCTION.
Counts are samples (sample(1) takes one a millisecond by default)."""
import argparse
import collections
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'rdma'))
from mesh_graph import preorder_parents  # noqa: E402  (the one graph module, design/rubrics.md C9)

FRAME = re.compile(r'^([ +!:|]*)(\d+) (.*?)  \(in ([^)]+)\)')


def frames(path):
    """[(depth, count, function, library)] of the main thread's call graph, in order."""
    text = open(path, errors='replace').read()
    graph = text.split('Call graph:')[1].split('Total number in stack')[0].splitlines()
    rows, seen_thread = [], False
    for line in graph:
        if re.match(r'^\s+\d+ Thread_', line):
            if seen_thread:
                break
            seen_thread = True
            continue
        m = FRAME.match(line)
        if m:
            rows.append((len(m.group(1)), int(m.group(2)), re.sub(r' \+ \d+.*', '', m.group(3)), m.group(4)))
    return rows


def entries(rows, parents, inside, outside):
    """Samples of each frame whose library satisfies `inside` called from a frame whose library satisfies `outside`."""
    totals = collections.Counter()
    for (depth, count, name, library), parent in zip(rows, parents):
        if parent is not None and inside(library) and outside(rows[parent][3]):
            totals[f'{library}: {name[:110]}'] += count
    return totals


def under(rows, parents, function):
    """Samples of each direct callee of every frame named `function`."""
    totals = collections.Counter()
    for (depth, count, name, library), parent in zip(rows, parents):
        if parent is not None and rows[parent][2] == function:
            totals[name] += count
    return totals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('sample')
    parser.add_argument('--under')
    parser.add_argument('--top', type=int, default=15)
    args = parser.parse_args()
    rows = frames(args.sample)
    parents = preorder_parents([row[0] for row in rows])
    sections = [('native calls from Python', entries(rows, parents, lambda l: l != 'python3.12', lambda l: l == 'python3.12')),
                ('entries into libnccl-mesh', entries(rows, parents, lambda l: l == 'libnccl-mesh.dylib', lambda l: l != 'libnccl-mesh.dylib'))]
    if args.under:
        sections.append((f'beneath {args.under}', under(rows, parents, args.under)))
    for title, totals in sections:
        print(f'== {title}')
        for name, n in totals.most_common(args.top):
            print(f'{n:8} {name}')


if __name__ == '__main__':
    main()
