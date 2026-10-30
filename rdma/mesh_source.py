"""A repository's one clone on a node (~/mesh, ~/metal-microbench): its revision, and its move to a commit."""
import argparse
from pathlib import Path
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


def move(root, origin, commit, timeout=None):
    """root, the one clone of origin (~/mesh, ~/metal-microbench; cloned where there is none), at commit, detached:
    main fetched, then a checkout, refused where its tracked source has changes."""
    root = Path(root)
    env = {**os.environ, 'GIT_TERMINAL_PROMPT': '0'}

    def run(*args):
        done = subprocess.run(['git', *map(str, args)], capture_output=True, text=True, timeout=timeout, env=env)
        if done.returncode:
            raise RuntimeError(f'git {" ".join(map(str, args))} failed ({done.returncode}): {done.stderr.strip()}')
        return done.stdout
    if not (root / '.git').exists():
        run('clone', '-q', origin, root)
    run('-C', root, 'fetch', '-q', 'origin', 'main')
    if run('-C', root, 'status', '--porcelain', '--untracked-files=no').strip():
        raise RuntimeError(f'{root}: its source has changes, so it is not moved')
    run('-C', root, 'checkout', '-q', '--detach', commit)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('revision', 'verify'))
    parser.add_argument('operands', nargs=1)
    args = parser.parse_args()
    if args.action == 'revision':
        print(revision(Path(args.operands[0])))
    else:
        print(verify(Path(args.operands[0])))


if __name__ == '__main__':
    main()
