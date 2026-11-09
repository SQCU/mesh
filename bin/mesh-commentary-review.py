"""design/commentary-review.md#investigative-review"""
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import traceback
import mesh_review as harness

TASK = """Review the documents under design/ changed by this commit, including deletions and renames. Investigate
what the changes do to their meaning, scope and authority. Follow their sources, original turns, code and revision
history as needed. Distinguish requirements, qualitative review standards, implementation descriptions and status
claims using their actual context. Write commentary that helps subsequent review and revision, including evidence
that supports the change as well as any concrete defects. Do not infer documentary requirements from a filename,
heading grammar, vocabulary overlap or missing local quotation."""


def write(repo, sha, transcript=None):
    root = harness.git(repo, 'rev-parse', '--show-toplevel')
    sha = harness.git(root, 'rev-parse', '--verify', '--end-of-options', sha + '^{commit}')
    changed = harness.git(root, 'diff-tree', '--root', '--no-commit-id', '--name-only', '-r', sha, '--', 'design/')
    if not changed:
        return None
    record = harness.review(root, sha, TASK, transcript=transcript, kind='commentary', model='sonnet')
    store = pathlib.Path(harness.git(root, 'rev-parse', '--path-format=absolute', '--git-common-dir')) / 'review-commentary'
    target = store / f'{sha}.md'
    text = (f'# Commentary on {root} {sha}\n\n{record["at"]}\n\n{record["subject"]}\n\n' + harness.render(record) + '\n')
    with tempfile.NamedTemporaryFile(mode='w', dir=store, delete=False) as staged:
        staged.write(text)
    os.replace(staged.name, target)
    with open(store / 'INDEX.md', 'a') as index:
        index.write(f'- {record["at"]} {sha}: [{record["subject"]}]({sha}.md); trace {record["trace"]}\n')
    return str(target)


def note(error):
    store = pathlib.Path(harness.MESH) / '.git/review-commentary'
    store.mkdir(exist_ok=True)
    with open(store / 'log', 'a') as f:
        f.write(f'{harness.now()} {error}\n')


def detached():
    try:
        os.setpriority(os.PRIO_DARWIN_PROCESS, 0, os.PRIO_DARWIN_BG)
    except (AttributeError, OSError):
        os.nice(10)
    try:
        payload = json.load(sys.stdin)
        for repo, sha in harness.c9.made(payload):
            try:
                write(repo, sha, payload.get('transcript_path'))
            except Exception:
                note(traceback.format_exc())
    except Exception:
        note(traceback.format_exc())
    return 0


def hook():
    try:
        with tempfile.TemporaryFile() as spool:
            spool.write(sys.stdin.buffer.read())
            spool.seek(0)
            subprocess.Popen([sys.executable, '-B', os.path.abspath(__file__), '--detached'], stdin=spool,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd='/', start_new_session=True)
    except Exception:
        note(traceback.format_exc())
    return 0


def main():
    args = sys.argv[1:]
    if args == ['--detached']:
        return detached()
    if args[:1] == ['--commit'] and len(args) == 3:
        print(write(harness.REPOSITORIES.get(args[1], os.path.abspath(args[1])), args[2]) or 'No design document changed.')
        return 0
    if args:
        print('mesh-commentary-review.py --commit <mesh|metal-microbench|path> <rev>', file=sys.stderr)
        return 1
    return hook()


if __name__ == '__main__':
    sys.exit(main())
