"""A repository's one clone on a node (~/mesh, ~/metal-microbench): its revision, verified before installation."""
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


def verify(root):
    if git(root, 'status', '--porcelain').strip():
        raise RuntimeError(f'{root}: commit source changes before installation')
    return revision(root)


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
