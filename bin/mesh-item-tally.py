"""design/review-warnings.md#15-who-was-given-what-and-what-an-item-has-had"""
import json
import subprocess
import sys

REPOSITORIES = ('/Users/mdot/dox/mesh', '/Users/mdot/metal-microbench')


def git(repo, *args):
    return subprocess.run(['git', '-C', repo, *args], capture_output=True, text=True).stdout.strip()


def main():
    document, name = sys.argv[1], sys.argv[2]
    measure = f'{document}, item {name}'
    print(f'{measure}: historical review records; these do not establish current implementation or coverage.')
    for repo in REPOSITORIES:
        for line in filter(None, git(repo, 'notes', '--ref', 'measurements', 'list').split('\n')):
            sha = line.split()[1]
            for entry in git(repo, 'notes', '--ref', 'measurements', 'show', sha).splitlines():
                record = json.loads(entry) if entry.strip() else {}
                if record.get('measure') == measure:
                    print(json.dumps({'repository': repo, 'commit': sha, 'record': record}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
