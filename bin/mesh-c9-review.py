"""C9's review of each commit to mesh or metal-microbench (design/rubrics.md:422, the 10-30 pair): the commit, its
changed functions whole, read by Claude Haiku with no tools. A Claude Code PostToolUse payload on stdin (each
repository's .claude/settings.json); what the reader finds goes to stderr with exit 2, which Claude Code hands back to
the committing session."""
import json
import os
import re
import subprocess
import sys
import tempfile
import time

REPOSITORIES = ('/Users/mdot/dox/mesh/.git', '/Users/mdot/metal-microbench/.git')
READER = ['claude', '-p', '--model', 'claude-haiku-5-5', '--tools', '', '--setting-sources', 'user',
          '--no-session-persistence', '--strict-mcp-config']
PROMPT = """You review one git commit, given on stdin with each changed function whole (git show -W), for C9 of the
mesh rubric: one primitive, offered as a header. A graph algorithm computes over a graph's vertices, edges, weights or
samples: shortest paths, traversals and worklists, reachability, topological orders, spanning trees, arborescences,
flows, cuts, packings (multiplicative weights), union-find, components, cycle checks, path enumeration, walks over
parent or next-hop tables, tree heights. Its one home is mesh's graph module (rdma/mesh-graph.h, rdma/mesh-graph.c,
rdma/mesh_graph.py), whose functions take the consumer's weights and visits as operands; metal-microbench reaches it
through libmesh and writes none of its own. Code in that module, and code that only calls it or builds its inputs,
passes. Report each function the commit adds or changes (a + or - line inside it) that carries such an algorithm
anywhere else, one line each: `path:line function(): the algorithm; it moves into the graph module`, the line from the
hunk's new-file numbering. If there is none, answer exactly PASS. Nothing else: no style, correctness or naming remarks."""
SKIPPED = [':(exclude)*.md', ':(exclude)*.json', ':(exclude)*.jsonl', ':(exclude)*.csv', ':(exclude)*.txt',
           ':(exclude)*.svg', ':(exclude)*.html']


def git(repo, *args):
    return subprocess.run(['git', '-C', repo, *args], capture_output=True, text=True).stdout.strip()


def made(payload):
    """(repository, sha) of each commit the Bash call made: the `[branch sha]` lines it printed, else (a quiet commit)
    its repository's HEAD where that was committed in the last few minutes"""
    command = (payload.get('tool_input') or {}).get('command') or ''
    # git's commit subcommand, any options before it (the `if` filter lets through commands it cannot parse)
    if not re.search(r'\bgit(?:\s+-[Cc]\s+\S+|\s+--?[\w-]+(?:=\S+)?)*\s+commit\b', command):
        return []
    response = payload.get('tool_response')
    printed = response if isinstance(response, str) else '\n'.join(
        str((response or {}).get(k) or '') for k in ('stdout', 'stderr')) if isinstance(response, dict) else ''
    shas = re.findall(r'^\[(?:detached HEAD|[^\]\s]+)(?: \(root-commit\))? ([0-9a-f]{7,40})\]', printed, re.M)
    cwd = payload.get('cwd') or os.getcwd()
    dirs = [os.path.join(cwd, os.path.expanduser(d.strip('\'"'))) for d in re.findall(r'(?:\s-C|\bcd)\s+(\S+)', command)] + [cwd]
    dirs = [d for d in dict.fromkeys(dirs) if os.path.isdir(d)]
    found = []
    for sha in dict.fromkeys(shas):
        for d in dirs:
            if git(d, 'rev-parse', '--verify', '--quiet', sha + '^{commit}'):
                found.append((d, sha))
                break
    if not shas:
        for d in dirs:
            when = git(d, 'log', '-1', '--format=%ct')
            if when.isdigit() and time.time() - int(when) < 300:
                found.append((d, git(d, 'rev-parse', 'HEAD')))
                break
    return [(d, sha) for d, sha in found if git(d, 'rev-parse', '--path-format=absolute', '--git-common-dir') in REPOSITORIES]


def main():
    findings = []
    for repo, sha in made(json.load(sys.stdin)):
        commit = git(repo, 'show', '-W', '--no-color', '--format=commit %H%n%s%n', sha, '--', '.', *SKIPPED)
        if '\ndiff --git' not in '\n' + commit:
            continue
        read = subprocess.run([*READER, '--system-prompt', PROMPT, 'Review this commit.'], input=commit[:400000],
                              capture_output=True, text=True, cwd=tempfile.gettempdir(), timeout=170)
        said = read.stdout.strip()
        if read.returncode:
            print(f'C9 review of {sha[:10]} in {repo}: the reader failed: {read.stderr.strip()[:500]}', file=sys.stderr)
            return 1
        if said != 'PASS':
            findings.append(f'C9 (design/rubrics.md:422) in {sha[:10]} of {repo}, read by Claude Haiku:\n{said}')
    if findings:
        print('\n\n'.join(findings), file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
