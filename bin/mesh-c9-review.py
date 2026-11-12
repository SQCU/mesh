"""design/rubrics.md:422; design/commentary-review.md#post-commit-findings"""
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
hunk's new-file numbering.
Report also each function the commit adds or changes that is a second producer of a value one module already produces:
(a) where a value's bytes lie in a region (its pages, page lists, hop lists, routes between machines, byte or page
offsets, copies, lifetimes), whose one producer is mesh_plan (rdma/mesh-collective.c), reached from the compiler's
schedule stage (compiler/mesh_compiler/schedule.py);
(b) the fields or bytes of a record a shared header declares (rdma/*.h: struct mesh_state, mesh_hop, mesh_section,
mesh_declared, mesh_planned and the like), computed anywhere but in the rdma C file that implements that header;
(c) which functions, commands or nodes of a program must run before which (a dependence relation over accesses, or
an execution order), whose one home is the graph module;
(d) a record of a program's synchronization edges (struct mesh_graph_signature), built anywhere but through the graph
module's signature functions.
These pass: a declaration that only mirrors a record's layout so that operands can be passed to the library; code that
calls the producer; a tensor's shard sizes, slices and concatenations; an operator's rule for which dimension of its
result follows which of its operands; names and counters. Report only what the changed lines plainly do. One line
each: `path:line function(): what it produces a second time; its one producer is <the producer>`. If there is nothing
of either kind, answer exactly PASS. Nothing else: no style, correctness or naming remarks."""
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
    # the directories the command names (cd, git -C), with variables it assigns itself and the environment's expanded;
    # where it names any, those alone: a quiet commit in a directory it named but this cannot resolve is left
    # unreviewed rather than taken for whatever this hook's own directory last committed (another session's)
    assigned = dict(re.findall(r'(?:^|[;&|\s])([A-Za-z_][A-Za-z0-9_]*)=([^\s;&|]+)', command))
    def expanded(path):
        path = re.sub(r'\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?', lambda m: assigned.get(m.group(1), os.environ.get(m.group(1), m.group(0))),
                      path.strip('\'"'))
        return os.path.join(cwd, os.path.expanduser(path))
    named = re.findall(r'(?:\s-C|\bcd)\s+(\S+)', command)
    dirs = [expanded(d) for d in named] if named else [cwd]
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


def feedback(messages):
    if messages:
        message = '\n\n'.join(messages)
        print(json.dumps({'systemMessage': message,
                          'hookSpecificOutput': {'hookEventName': 'PostToolUse', 'additionalContext': message}}))
    return 0


def main():
    findings = []
    for repo, sha in made(json.load(sys.stdin)):
        commit = git(repo, 'show', '-W', '--no-color', '--format=commit %H%n%s%n', sha, '--', '.', *SKIPPED)
        if '\ndiff --git' not in '\n' + commit:
            continue
        try:
            read = subprocess.run([*READER, '--system-prompt', PROMPT, 'Review this commit.'], input=commit[:400000],
                                  capture_output=True, text=True, cwd=tempfile.gettempdir(), timeout=170)
        except (subprocess.TimeoutExpired, OSError) as error:
            findings.append(f'C9 review of {sha[:10]} in {repo}: the reader failed: {error}')
            continue
        said = read.stdout.strip()
        if read.returncode:
            findings.append(f'C9 review of {sha[:10]} in {repo}: the reader exited {read.returncode}: '
                            f'{read.stderr.strip()[:500]}')
            continue
        if not said:
            findings.append(f'C9 review of {sha[:10]} in {repo}: the reader gave no result')
            continue
        if said != 'PASS':
            findings.append(f'C9 (design/rubrics.md:422) in {sha[:10]} of {repo}, read by Claude Haiku:\n{said}')
    return feedback(findings)


if __name__ == '__main__':
    sys.exit(main())
