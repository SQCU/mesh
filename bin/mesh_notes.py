"""design/review-warnings.md#14-one-record-for-each-commit"""
import fcntl
import subprocess


def append(repo, ref, message, revision):
    lock = subprocess.check_output(['git', '-C', repo, 'rev-parse', '--path-format=absolute', '--git-path',
                                    f'mesh-{ref}.lock'], text=True).strip()
    with open(lock, 'a') as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        subprocess.run(['git', '-C', repo, 'notes', '--ref', ref, 'append', '-m', message, revision],
                       capture_output=True, check=True)
