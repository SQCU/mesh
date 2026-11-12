"""design/commentary-review.md#investigative-review"""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
import mesh_review as harness

MESH = harness.MESH
REGISTERED = os.path.join(MESH, '.claude', 'change-requests-registered.json')
REVIEWS = harness.REVIEWS
c9 = harness.c9
TASK = """Review this commit's implementation and documentary changes against the applicable operator instructions
and cited methods. Investigate the governing documents as well as the code, including a possible change in their
scope or authority. Registered documents are shared context hints, not a complete or authoritative assignment.
Locate the relevant instructions even if registration is empty. Follow integration through the affected call paths
and companion changes, and use existing measurements or synchronization evidence where relevant.
Apply rules 4, 5 and 6 of mesh design/review-warnings.md: the clauses design/requirements-by-stage.md lists for each
stage the commit changes, a second producer of a value another stage or module produces, and each clause of anything
the commit says is done."""


def registrations():
    try:
        with open(REGISTERED) as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def register(documents):
    revision = harness.git(MESH, 'rev-parse', 'HEAD')
    entries = registrations()
    for document in documents:
        if subprocess.run(['git', '-C', MESH, 'cat-file', '-e', f'{revision}:{document}'],
                          capture_output=True).returncode:
            print(f'{document} is not in mesh at {revision[:10]}', file=sys.stderr)
            return 1
        entries.append({'document': document, 'revision': revision, 'at': harness.now()})
    with open(REGISTERED, 'w') as f:
        json.dump(entries, f, indent=1)
    print('\n'.join(f"{e['document']} at {e['revision'][:10]}" for e in entries))
    return 0


def review(repo, sha, entries, transcript=None):
    return harness.review(repo, sha, TASK, entries, transcript)


def outstanding():
    latest = {}
    with open(REVIEWS) as f:
        for line in f:
            record = json.loads(line)
            latest[record['repository'], record['commit'], record.get('kind', 'change-request'), record.get('file')] = record
    for record in latest.values():
        if 'style' in record:
            print(f"{record['commit'][:10]} of {record['repository']}, {record['file']}: {record['style']}\n")
        elif record['outstanding'] is not False:
            print(f"{record['commit'][:10]} of {record['repository']}, {record.get('kind', 'change-request')}: "
                  f"{record['subject'][:160]}\n" +
                  (harness.render(record) if 'trace' in record else '\n'.join(record['findings'])) + '\n')
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
            jobs.append((repo, sha, [{'document': n, 'by': 'explicit CLI context'} for n in named.split(',') if n]))
        with ThreadPoolExecutor(6) as pool:
            for record in pool.map(lambda job: review(*job), jobs):
                state = {True: 'findings', False: 'no findings', None: 'incomplete'}[record['outstanding']]
                print(f"{record['commit'][:10]} {state}: {record['subject'][:120]}\n{harness.render(record)}", flush=True)
        return 0
    findings = []
    try:
        entries = registrations()
        payload = json.load(sys.stdin)
        for repo, sha in c9.made(payload):
            record = review(repo, sha, entries, payload.get('transcript_path'))
            if record['outstanding'] is not False:
                findings.append(f'Change request review of {sha[:10]} in {repo}, recorded in {REVIEWS}:\n' +
                                harness.render(record))
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        findings.append(f'Change request review incomplete: {error}')
    return c9.feedback(findings)


if __name__ == '__main__':
    sys.exit(main())
