"""design/torch-capture-change-request.md U199; design/change-requests.md"""
import datetime
import importlib.util
import json
import os
import subprocess
import sys
import tempfile

MESH = '/Users/mdot/dox/mesh'
REGISTERED = os.path.join(MESH, '.claude', 'change-requests-registered.json')
STANDARD = 'design/change-requests.md'
spec = importlib.util.spec_from_file_location('c9', os.path.join(MESH, 'bin', 'mesh-c9-review.py'))
c9 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c9)
READER = ['claude', '-p', '--model', 'opus', '--tools', '', '--setting-sources', 'user', '--no-session-persistence',
          '--strict-mcp-config']
PROMPT = """You review one git commit against the change requests registered for the agent that made it. On stdin come
the repository's standard for change requests (design/change-requests.md), each registered change request at the
revision it was registered at, and the commit: its message, then its diff with each changed function whole (git show
-W).

A change request is the whole instruction to its implementer. Its turns are the operator's words, quoted. Each of its
requirements names the code it replaces, quotes a source, and states what is written: the source's method, as the
source states it. Code is changed only to adhere to a requirement's source. The turns bind the commit as the
requirements do.

The commit's message states which registered change request and which item each change implements, and how. Report
each of these, one line each:
- a change the message relates to no item of a registered change request;
- a change that no item states: a statement, branch, field, loop, wait, synchronization, commit or submission, copy or
  allocation that the item and its quoted source do not contain;
- a change that does what an item states otherwise than its source states it;
- a change that contradicts the change request's turns;
- an item the message says the commit implements, which the diff does not do as the item states it.
Each line: `path:line: the finding; the change request, its item, and the sentence the change departs from`, the line
from the hunk's new-file numbering, or `message:` for the message. If there is none, answer exactly PASS. Nothing else:
no style or naming remarks."""


def registrations():
    try:
        with open(REGISTERED) as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def register(documents):
    revision = c9.git(MESH, 'rev-parse', 'HEAD')
    entries = registrations()
    for document in documents:
        if subprocess.run(['git', '-C', MESH, 'cat-file', '-e', f'{revision}:{document}']).returncode:
            print(f'{document} is not in mesh at {revision[:10]}', file=sys.stderr)
            return 1
        entries.append({'document': document, 'revision': revision,
                        'at': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')})
    with open(REGISTERED, 'w') as f:
        json.dump(entries, f, indent=1)
    print('\n'.join(f"{e['document']} at {e['revision'][:10]}" for e in entries))
    return 0


def main():
    if sys.argv[1:2] == ['register']:
        return register(sys.argv[2:])
    if sys.argv[1:2] == ['clear']:
        with open(REGISTERED, 'w') as f:
            json.dump([], f)
        return 0
    entries = registrations()
    findings = []
    for repo, sha in c9.made(json.load(sys.stdin)):
        commit = c9.git(repo, 'show', '-W', '--no-color', '--format=commit %H%n%B', sha, '--', '.', *c9.SKIPPED)
        if '\ndiff --git' not in '\n' + commit:
            continue
        if not entries:
            findings.append(f'{sha[:10]} of {repo}: no change request is registered ({REGISTERED}); a commit answers a '
                            f'registered change request, as its message states.')
            continue
        documents = [f'=== {STANDARD}\n' + c9.git(MESH, 'show', f'HEAD:{STANDARD}')]
        documents += [f"=== {e['document']} at mesh {e['revision']}\n" + c9.git(MESH, 'show', f"{e['revision']}:{e['document']}")
                      for e in entries]
        given = '\n\n'.join(documents) + '\n\n=== the commit\n' + commit[:600000]
        read = subprocess.run([*READER, '--system-prompt', PROMPT, 'Review this commit.'], input=given,
                              capture_output=True, text=True, cwd=tempfile.gettempdir(), timeout=880)
        said = read.stdout.strip()
        if read.returncode:
            print(f'Change request review of {sha[:10]} in {repo}: the reader failed: {read.stderr.strip()[:500]}',
                  file=sys.stderr)
            return 1
        if said != 'PASS':
            named = ', '.join(f"{e['document']} at {e['revision'][:10]}" for e in entries)
            findings.append(f'Change request review of {sha[:10]} in {repo}, against {named}:\n{said}')
    if findings:
        print('\n\n'.join(findings), file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
