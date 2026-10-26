"""Git checkouts used by installation and process launch, before numerical work."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess


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
        'git --git-dir="$repository" fetch --quiet --no-tags origin main',
        f'git --git-dir="$repository" merge-base --is-ancestor {shlex.quote(commit)} FETCH_HEAD',
        'if [ ! -e "$destination" ]; then',
        f'  git --git-dir="$repository" worktree add --quiet --detach "$destination" {shlex.quote(commit)}',
        'fi',
        f'[ "$(git -C "$destination" rev-parse HEAD)" = {shlex.quote(commit)} ]',
        '[ -z "$(git -C "$destination" status --porcelain)" ]',
    ))


def checkout(origin, commit, destination):
    result = subprocess.run(['/bin/sh', '-c', checkout_command(origin, commit, destination)],
                            capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f'Git checkout {commit} from {origin} failed ({result.returncode}): {result.stderr.strip()}')


def files(root, relative):
    return git(Path(root) / relative, 'ls-files', '-z').decode().rstrip('\0').split('\0')


def prune(root, keep):
    """Remove superseded build worktrees after the new installation is active."""
    root = Path(root)
    for path in root.iterdir():
        if path.name != keep and (path / '.git').is_file():
            verify(path)
            git(root / keep, 'worktree', 'remove', '--force', str(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('files', 'revision', 'verify', 'prune'))
    parser.add_argument('root', type=Path)
    parser.add_argument('relative', nargs='?', default='.')
    args = parser.parse_args()
    if args.action == 'files':
        import sys
        sys.stdout.buffer.write(b''.join(os.fsencode(p) + b'\0' for p in files(args.root, args.relative)))
    elif args.action == 'revision':
        print(revision(args.root))
    elif args.action == 'prune':
        prune(args.root, args.relative)
    else:
        print(verify(args.root))


if __name__ == '__main__':
    main()
