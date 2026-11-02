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

FRAME = re.compile(r'^([ +!:|]*)(\d+) (.*?)  \(in ([^)]+)\)')


def frames(path):
    """(depth, count, function, library) of the main thread's call graph, in order."""
    text = open(path, errors='replace').read()
    graph = text.split('Call graph:')[1].split('Total number in stack')[0].splitlines()
    seen_thread = False
    for line in graph:
        if re.match(r'^\s+\d+ Thread_', line):
            if seen_thread:
                return
            seen_thread = True
            continue
        m = FRAME.match(line)
        if m:
            yield len(m.group(1)), int(m.group(2)), re.sub(r' \+ \d+.*', '', m.group(3)), m.group(4)


def entries(path, inside, outside):
    """Samples of each frame whose library satisfies `inside` called from a frame whose library satisfies `outside`."""
    totals, stack = collections.Counter(), []
    for depth, count, name, library in frames(path):
        while stack and stack[-1][0] >= depth:
            stack.pop()
        parent = stack[-1][1] if stack else None
        stack.append((depth, library))
        if inside(library) and parent is not None and outside(parent):
            totals[f'{library}: {name[:110]}'] += count
    return totals


def under(path, function):
    totals, inside, child = collections.Counter(), None, None
    for depth, count, name, library in frames(path):
        if inside is not None and depth <= inside:
            inside = None
        if inside is None and name == function:
            inside, child = depth, None
            continue
        if inside is not None:
            if child is None:
                child = depth
            if depth == child:
                totals[name] += count
    return totals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('sample')
    parser.add_argument('--under')
    parser.add_argument('--top', type=int, default=15)
    args = parser.parse_args()
    sections = [('native calls from Python', entries(args.sample, lambda l: l != 'python3.12', lambda l: l == 'python3.12')),
                ('entries into libnccl-mesh', entries(args.sample, lambda l: l == 'libnccl-mesh.dylib', lambda l: l != 'libnccl-mesh.dylib'))]
    if args.under:
        sections.append((f'beneath {args.under}', under(args.sample, args.under)))
    for title, totals in sections:
        print(f'== {title}')
        for name, n in totals.most_common(args.top):
            print(f'{n:8} {name}')


if __name__ == '__main__':
    main()
