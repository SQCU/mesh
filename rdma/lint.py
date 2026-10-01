#!/usr/bin/env python3
"""Fails a build (or a run) whose mesh source has a crash motif the verified source does not.

    lint.py ROOT BASELINE GLOB...      exit 1 and name each new site; exit 0 otherwise
    lint.py --emit ROOT GLOB...        print ROOT's sites in BASELINE's format

A site is a source line (whitespace collapsed; comment-only lines skipped) that matches a motif. The
baseline lists the verified source's sites with their counts (mesh 753676b rdma = 3981ca8, bridge
source d6dceac; metal-microbench c3ef08b = 10ec60a). A line not in the baseline, or present more often
than there, is a new site. Each motif is a construct that crashed or wedged these machines:

  verbs-teardown  2026-10-01 15:20: the bridge deregistered its memory regions twice at shutdown (more
                  callers of down_pair than the verified one); both kernels panicked in
                  AppleThunderboltRDMA / IOThunderboltFamily. 2026-08-27: SIGKILLed verbs processes
                  never released the device.
  host-gpu-wait   2026-10-01 14:03: a host thread waited on a command buffer whose GPU wait needed a
                  word the same thread posts later; WindowServer starved and was killed three times.
  spin            2026-09-30 and 2026-10-01 14:03: GPU waits on words nobody wrote, spinning with no
                  owner process.
  kill-launchd    2026-08-27 SIGKILL wedging the provider; 2026-09-30 launchd ExitTimeOut 0 SIGKILLing
                  a bridge; 2026-10-01 stale KeepAlive LaunchAgents starting Sep 17 bridges at login.
  tcp             2026-09-28 to 2026-10-01: TCP control messages on the transfer path.
"""
import collections
import glob
import os
import re
import sys

MOTIFS = [
    ('verbs-teardown', re.compile(r'\bibv_(dereg_mr|destroy_qp|destroy_cq|dealloc_pd|close_device|destroy_comp_channel|destroy_srq)\s*\(|\b(down_pair|down_device)\s*\(')),
    ('host-gpu-wait', re.compile(r'\bwaitUntil(Completed|Scheduled)\b|\bwaitForEvent\b|\bwaitUntilSignaledValue\b')),
    ('spin', None),
    ('kill-launchd', re.compile(r'SIGKILL|\bkill\s+-(9|KILL)\b|\bpkill\s+-(9|KILL)\b|\bkill\s*\([^)]*,\s*9\s*\)|ExitTimeOut|KeepAlive|LaunchAgents|LaunchDaemons|launchctl\s+(bootstrap|kill|kickstart)')),
    ('tcp', re.compile(r'SOCK_STREAM|\b(connect|accept|listen)\s*\(|TCP_NODELAY|\bsetsockopt\s*\(')),
]
COMMENT = re.compile(r'^\s*(#|//|/\*|\*)')
LOOP = re.compile(r'\bwhile\s*\(')
LOOP_TAIL = re.compile(r'\s*(\{\s*\}|FENCE\s*;|;)')
OTHER_SPIN = re.compile(r'\}\s*while\s*\(|\bfor\s*\(\s*;\s*;\s*\)')


def spins(text):
    """A while whose body is empty (`;`, `{}`, `FENCE;`), a do-while, or for(;;): a loop only another party ends."""
    for start in LOOP.finditer(text):
        depth = 0
        for i in range(start.end() - 1, len(text)):
            depth += {'(': 1, ')': -1}.get(text[i], 0)
            if depth == 0:
                if LOOP_TAIL.match(text, i + 1):
                    return True
                break
    return bool(OTHER_SPIN.search(text))


def sites(root, patterns):
    found = collections.Counter()
    paths = sorted({p for pattern in patterns for p in glob.glob(os.path.join(root, pattern), recursive=True)})
    for path in paths:
        if not os.path.isfile(path) or os.path.basename(path) == 'lint.py':
            continue
        relative = os.path.relpath(path, root)
        with open(path, errors='replace') as source:
            for line in source:
                if COMMENT.match(line):
                    continue
                text = ' '.join(line.split())
                for name, motif in MOTIFS:
                    if (spins(text) if motif is None else motif.search(text)):
                        found[(name, relative, text)] += 1
    return found


def read_baseline(path):
    baseline = collections.Counter()
    with open(path) as lines:
        for line in lines:
            if line.strip() and not line.startswith('#'):
                name, relative, count, text = line.rstrip('\n').split('\t', 3)
                baseline[(name, relative, text)] = int(count)
    return baseline


def main(argv):
    if argv[:1] == ['--emit']:
        for (name, relative, text), count in sorted(sites(argv[1], argv[2:]).items()):
            print(f'{name}\t{relative}\t{count}\t{text}')
        return 0
    root, baseline_path, patterns = argv[0], argv[1], argv[2:]
    baseline = read_baseline(baseline_path)
    new = [(key, count - baseline[key]) for key, count in sorted(sites(root, patterns).items()) if count > baseline[key]]
    for (name, relative, text), extra in new:
        print(f'mesh lint: {name}: {relative}: {extra} new: {text[:160]}', file=sys.stderr)
    if new:
        print(f'mesh lint: {len(new)} new crash-motif site(s) against {baseline_path} (the verified source); '
              f'see rdma/lint.py for the incident behind each motif', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
