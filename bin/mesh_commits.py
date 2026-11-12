"""design/review-warnings.md#7-a-commit-adds-no-assertion-and-no-test"""
import fcntl
import os
import subprocess

REPOSITORIES = ('/Users/mdot/dox/mesh', '/Users/mdot/metal-microbench')
READER = ['claude', '-p', '--model', 'claude-haiku-5-5', '--tools', '', '--setting-sources', 'user',
          '--no-session-persistence', '--strict-mcp-config']


def git(repo, *args):
    return subprocess.run(['git', '-C', repo, *args], capture_output=True, text=True).stdout.strip()


def marker(repo, name):
    path = os.path.join(git(repo, 'rev-parse', '--path-format=absolute', '--git-common-dir'), 'review-commentary', name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def advance(repo, name, sha):
    path = marker(repo, name)
    with open(path + '.new', 'w') as f:
        f.write(sha)
    os.replace(path + '.new', path)


def held(repo, name):
    lock = open(marker(repo, name) + '.lock', 'w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return None
    return lock


def new_commits(repo, name):
    head = git(repo, 'rev-parse', 'HEAD')
    path = marker(repo, name)
    seen = open(path).read().strip() if os.path.exists(path) else ''
    if not seen or not git(repo, 'rev-parse', '--verify', '--quiet', seen + '^{commit}'):
        advance(repo, name, head)
        return []
    return git(repo, 'rev-list', '--reverse', f'{seen}..{head}').split()


def appended(log, record):
    with open(log, 'a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(record + '\n')
