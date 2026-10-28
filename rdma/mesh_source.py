"""Git checkouts used by installation and process launch, before numerical work."""
import argparse
import contextlib
import os
from pathlib import Path
import shlex
import subprocess
import time


def git(root, *args):
    return subprocess.run(['git', '-C', str(root), *args], check=True, capture_output=True).stdout


def revision(root, ref='HEAD'):
    return git(root, 'rev-parse', '--verify', f'{ref}^{{commit}}').decode().strip()


def upstream(root):
    return git(root, 'remote', 'get-url', 'origin').decode().strip()


def verify(root):
    if git(root, 'status', '--porcelain').strip():
        raise RuntimeError(f'{root}: commit source changes before installation')
    return revision(root)


def checkout_command(origin, commit, destination):
    """The node fetches Git objects from origin; the caller supplies only operands."""
    def path(value):
        value = str(value)
        return '"$HOME"/' + shlex.quote(value[2:]) if value.startswith('~/') else shlex.quote(value)
    repository = Path(destination).parent / 'repository.git'
    return '\n'.join((
        'set -eu',
        'export GIT_TERMINAL_PROMPT=0',
        f'repository={path(repository)}',
        f'destination={path(destination)}',
        'if [ ! -d "$repository" ]; then',
        f'  git clone --quiet --bare --single-branch --branch main {shlex.quote(origin)} "$repository"',
        'fi',
        f'[ "$(git --git-dir="$repository" config --get remote.origin.url)" = {shlex.quote(origin)} ]',
        'if ! git --git-dir="$repository" merge-base --is-ancestor ' + shlex.quote(commit) + ' refs/remotes/origin/main 2>/dev/null; then',
        '  git --git-dir="$repository" fetch --quiet --no-tags origin main:refs/remotes/origin/main',
        'fi',
        f'git --git-dir="$repository" merge-base --is-ancestor {shlex.quote(commit)} refs/remotes/origin/main',
        'if [ ! -e "$destination" ]; then',
        f'  git --git-dir="$repository" worktree add --quiet --detach "$destination" {shlex.quote(commit)}',
        'fi',
        '[ -z "$(git -C "$destination" status --porcelain)" ]',
        f'git -C "$destination" checkout --quiet --detach {shlex.quote(commit)}',
    ))


def checkout(origin, commit, destination, timeout=None):
    result = subprocess.run(['/bin/sh', '-c', checkout_command(origin, commit, destination)],
                            capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f'Git checkout {commit} from {origin} failed ({result.returncode}): {result.stderr.strip()}')


def files(root, relative):
    return git(Path(root) / relative, 'ls-files', '-z').decode().rstrip('\0').split('\0')


def prune(root, kept, seconds):
    """Remove the per-commit checkouts past the `kept` most recently used (`<name>.used`), never one used within
    `seconds` (a run may still load it) or the running bridge's."""
    root, active = Path(root), Path.home() / '.cache/mesh/active-source'
    active = active.read_text().strip() if active.exists() else ''
    used = lambda path: (root / f'{path.name}.used').stat().st_mtime if (root / f'{path.name}.used').exists() else 0
    for path in sorted((p for p in root.iterdir() if (p / '.git').is_file()), key=used, reverse=True)[int(kept):]:
        if str(path) != active and time.time() - used(path) > float(seconds):
            with contextlib.suppress(RuntimeError, subprocess.CalledProcessError):
                verify(path)
                git(root / 'repository.git', 'worktree', 'remove', '--force', str(path))
                (root / f'{path.name}.used').unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('files', 'revision', 'verify', 'prune'))
    parser.add_argument('root', type=Path)
    parser.add_argument('relative', nargs='*', default=['.'])
    args = parser.parse_args()
    if args.action == 'files':
        import sys
        sys.stdout.buffer.write(b''.join(os.fsencode(p) + b'\0' for p in files(args.root, *args.relative)))
    elif args.action == 'revision':
        print(revision(args.root))
    elif args.action == 'prune':
        prune(args.root, *args.relative)
    else:
        print(verify(args.root))


if __name__ == '__main__':
    main()
