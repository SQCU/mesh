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


def checkout_command(origin, commit, destination, paths=()):
    """The node fetches Git objects from origin; the caller supplies only operands. With `paths`, only those
    directories are checked out, and only their files' contents fetched (a blobless clone, a sparse checkout)."""
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
        f'  git clone --quiet --bare --single-branch --branch main {"--filter=blob:none " if paths else ""}'
        f'{shlex.quote(origin)} "$repository"',
        'fi',
        f'[ "$(git --git-dir="$repository" config --get remote.origin.url)" = {shlex.quote(origin)} ]',
        'if ! git --git-dir="$repository" merge-base --is-ancestor ' + shlex.quote(commit) + ' refs/remotes/origin/main 2>/dev/null; then',
        # another run's preparation may be fetching the same ref at once: the ancestry check below decides
        '  git --git-dir="$repository" fetch --quiet --no-tags origin main:refs/remotes/origin/main || true',
        'fi',
        f'git --git-dir="$repository" merge-base --is-ancestor {shlex.quote(commit)} refs/remotes/origin/main',
        'if [ ! -e "$destination" ]; then',
        f'  git --git-dir="$repository" worktree add --quiet --detach {"--no-checkout " if paths else ""}"$destination" '
        f'{shlex.quote(commit)}',
        # a worktree made without its files: the sparse paths set, then its index and files read from its commit
        *([f'  git -C "$destination" sparse-checkout set --cone {" ".join(map(shlex.quote, paths))}',
           '  git -C "$destination" reset --quiet --hard'] if paths else []),
        'fi',
        '[ -z "$(git -C "$destination" status --porcelain)" ]',
        f'git -C "$destination" checkout --quiet --detach {shlex.quote(commit)}',
    ))


def checkout(origin, commit, destination, timeout=None, paths=()):
    result = subprocess.run(['/bin/sh', '-c', checkout_command(origin, commit, destination, paths)],
                            capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f'Git checkout {commit} from {origin} failed ({result.returncode}): {result.stderr.strip()}')


def main():
    """revision ROOT, verify ROOT, or checkout ORIGIN DESTINATION [COMMIT]: DESTINATION (a node's one checkout,
    etc/mesh-nodes.json `checkouts`/mesh/current) made or moved to COMMIT, by default main as ORIGIN has it now, as the
    hourly moves it (a worktree of a bare clone beside it, never a second clone)."""
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('revision', 'verify', 'checkout'))
    parser.add_argument('operands', nargs='+')
    args = parser.parse_args()
    if args.action == 'checkout':
        origin, destination, *commit = args.operands
        commit = commit[0] if commit else git(Path.home(), 'ls-remote', origin, 'refs/heads/main').split()[0].decode()
        checkout(origin, commit, Path(destination).expanduser(), 600)
        print(f'{destination} at {commit}')
    elif args.action == 'revision':
        print(revision(Path(args.operands[0])))
    else:
        print(verify(Path(args.operands[0])))


if __name__ == '__main__':
    main()
