"""design/torch-capture-change-request.md U199; design/change-requests.md"""
import datetime
import fcntl
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

MESH = '/Users/mdot/dox/mesh'
REGISTERED = os.path.join(MESH, '.claude', 'change-requests-registered.json')
REVIEWS = '/Users/mdot/metal-microbench/output_data/change-request-reviews.jsonl'
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


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')


def registrations():
    try:
        with open(REGISTERED) as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def present(revision, document):
    return not subprocess.run(['git', '-C', MESH, 'cat-file', '-e', f'{revision}:{document}'],
                              capture_output=True).returncode


def register(documents):
    revision = c9.git(MESH, 'rev-parse', 'HEAD')
    entries = registrations()
    for document in documents:
        if not present(revision, document):
            print(f'{document} is not in mesh at {revision[:10]}', file=sys.stderr)
            return 1
        entries.append({'document': document, 'revision': revision, 'at': now()})
    with open(REGISTERED, 'w') as f:
        json.dump(entries, f, indent=1)
    print('\n'.join(f"{e['document']} at {e['revision'][:10]}" for e in entries))
    return 0


def cited(repo, sha, named=()):
    message = c9.git(repo, 'log', '-1', '--format=%B', sha) + ''.join(f' {n}' for n in named)
    when = c9.git(repo, 'log', '-1', '--format=%cI', sha)
    at = re.findall(r'\bat mesh ([0-9a-f]{7,40})\b', message)
    if os.path.realpath(repo) == os.path.realpath(MESH):
        before = c9.git(MESH, 'rev-parse', f'{sha}^')
    else:
        before = c9.git(MESH, 'log', '-1', '--format=%H', f'--before={when}')
    entries = []
    for name in dict.fromkeys(re.findall(r'design/([A-Za-z0-9._-]+-change-request\.md)', message)):
        document = 'design/' + name
        revision = next((r for r in at if present(r, document)), None) or (before if present(before, document) else None)
        if revision:
            entries.append({'document': document, 'revision': c9.git(MESH, 'rev-parse', revision), 'at': when,
                            'by': 'the commit message'})
    return entries


def append(record):
    with open(REVIEWS, 'a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(record) + '\n')
        f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)


def review(repo, sha, entries):
    commit = c9.git(repo, 'show', '-W', '--no-color', '--format=commit %H%n%B', sha, '--', '.', *c9.SKIPPED)
    if '\ndiff --git' not in '\n' + commit:
        return None
    record = {'at': now(), 'repository': repo, 'commit': c9.git(repo, 'rev-parse', sha),
              'subject': c9.git(repo, 'log', '-1', '--format=%s', sha), 'registered': entries}
    if not entries:
        record.update(outstanding=True, findings=[f'no change request is registered ({REGISTERED}); a commit answers a '
                                                  f'registered change request, as its message states.'])
    else:
        documents = [f'=== {STANDARD}\n' + c9.git(MESH, 'show', f'HEAD:{STANDARD}')]
        documents += [f"=== {e['document']} at mesh {e['revision']}\n" + c9.git(MESH, 'show', f"{e['revision']}:{e['document']}")
                      for e in entries]
        given = '\n\n'.join(documents) + '\n\n=== the commit\n' + commit[:600000]
        try:
            read = subprocess.run([*READER, '--system-prompt', PROMPT, 'Review this commit.'], input=given,
                                  capture_output=True, text=True, cwd=tempfile.gettempdir(), timeout=880)
            said = read.stdout.strip()
            failed = (f'exit {read.returncode}: {read.stderr.strip()[:500]}' if read.returncode else
                      'no result' if not said else '')
        except subprocess.TimeoutExpired:
            said, failed = '', 'the reader did not answer in 880 s'
        except OSError as error:
            said, failed = '', str(error)
        if failed:
            record.update(outstanding=None, findings=[f'the reader failed: {failed}'])
        else:
            record.update(outstanding=said != 'PASS', findings=[] if said == 'PASS' else said.splitlines())
    append(record)
    return record


def outstanding():
    latest = {}
    with open(REVIEWS) as f:
        for line in f:
            record = json.loads(line)
            latest[record['repository'], record['commit'], record.get('file')] = record
    for record in latest.values():
        if 'style' in record:
            print(f"{record['commit'][:10]} of {record['repository']}, {record['file']}: {record['style']}\n")
        elif record['outstanding'] is not False:
            named = ', '.join(f"{e['document']} at {e['revision'][:10]}" for e in record['registered']) or 'none'
            print(f"{record['commit'][:10]} of {record['repository']}, against {named}: {record['subject'][:160]}\n"
                  + '\n'.join(record['findings']) + '\n')
    return 0


def main():
    if sys.argv[1:2] == ['register']:
        return register(sys.argv[2:])
    if sys.argv[1:2] == ['clear']:
        with open(REGISTERED, 'w') as f:
            json.dump([], f)
        return 0
    if sys.argv[1:2] == ['outstanding']:
        return outstanding()
    if sys.argv[1:2] == ['review-commits']:
        jobs = []
        for argument in sys.argv[2:]:
            commit, _, named = argument.partition('=')
            repo, sha = commit.rsplit(':', 1)
            jobs.append((repo, sha, cited(repo, sha, [n for n in named.split(',') if n])))
        with ThreadPoolExecutor(6) as pool:
            for record in pool.map(lambda job: review(*job), jobs):
                if record:
                    state = {True: 'outstanding', False: 'pass', None: 'failed'}[record['outstanding']]
                    print(f"{record['commit'][:10]} {state}: {record['subject'][:120]}", flush=True)
        return 0
    entries = registrations()
    findings = []
    for repo, sha in c9.made(json.load(sys.stdin)):
        record = review(repo, sha, entries)
        if not record or record['outstanding'] is False:
            continue
        named = ', '.join(f"{e['document']} at {e['revision'][:10]}" for e in entries) or 'no registered change request'
        state = 'outstanding' if record['outstanding'] else 'incomplete (reader failure)'
        findings.append(f'Change request review of {sha[:10]} in {repo}, against {named}, recorded as {state} in '
                        f'{REVIEWS}:\n' + '\n'.join(record['findings']))
    return c9.feedback(findings)


if __name__ == '__main__':
    sys.exit(main())
