"""Git checkouts used by installation and process launch, before numerical work."""
import argparse
from pathlib import Path
import shlex
import os
import subprocess


def git(root, *args):
    """git's output; its failure raised with git's own message (an hourly's ledger keeps only the error's text)."""
    done = subprocess.run(['git', '-C', str(root), *args], capture_output=True, env={**os.environ, 'GIT_TERMINAL_PROMPT': '0'})
    if done.returncode:
        raise RuntimeError(f'git {" ".join(args)} in {root} failed ({done.returncode}): {done.stderr.decode(errors="replace").strip()}')
    return done.stdout


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
        # another run's preparation may be fetching the same ref at once: the ancestry check below decides
        '  git --git-dir="$repository" fetch --quiet --no-tags origin main:refs/remotes/origin/main || true',
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


def prune(root, kept):
    """Remove the per-commit checkouts past the `kept` most recently used, each marked by `<name>.used`."""
    for used in sorted(Path(root).glob('*.used'), key=lambda path: path.stat().st_mtime, reverse=True)[int(kept):]:
        git(Path(root) / 'repository.git', 'worktree', 'remove', '--force', str(used.with_suffix('')))
        used.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('revision', 'verify', 'prune'))
    parser.add_argument('root', type=Path)
    parser.add_argument('kept', nargs='?', type=int)
    args = parser.parse_args()
    if args.action == 'revision':
        print(revision(args.root))
    elif args.action == 'prune':
        prune(args.root, args.kept)
    else:
        print(verify(args.root))


if __name__ == '__main__':
    main()
