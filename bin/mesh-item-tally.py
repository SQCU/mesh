"""design/review-warnings.md#15-who-was-given-what-and-what-an-item-has-had"""
import json
import re
import subprocess
import sys

REPOSITORIES = ('/Users/mdot/dox/mesh', '/Users/mdot/metal-microbench')


def git(repo, *args):
    return subprocess.run(['git', '-C', repo, *args], capture_output=True, text=True).stdout.strip()


def main():
    document, name = sys.argv[1], sys.argv[2]
    measure = f'{document}, item {name}'
    carried, sentences, commits = set(), 0, []
    for repo in REPOSITORIES:
        for line in filter(None, git(repo, 'notes', '--ref', 'measurements', 'list').split('\n')):
            sha = line.split()[1]
            for entry in git(repo, 'notes', '--ref', 'measurements', 'show', sha).splitlines():
                record = json.loads(entry) if entry.strip() else {}
                if record.get('measure') == measure and record.get('item_found'):
                    carried.update(record['carried'])
                    sentences = max(sentences, record['sentences'])
                    commits.append(f"{sha[:10]} carried {record['carried']} not carried {record['not_carried']}")
    print(f'{measure}: {len(carried)} of {sentences} sentences carried by {len(commits)} commit(s)')
    for line in commits:
        print(line)
    print('never carried:', sorted(set(range(1, sentences + 1)) - carried))
    lines = git(REPOSITORIES[0], 'show', f'HEAD:{document}').splitlines()
    start = next((at for at, line in enumerate(lines) if re.match(rf'#{{1,6}}\s+{re.escape(name)}[.:\s]', line)), None)
    if start is not None:
        depth = len(lines[start]) - len(lines[start].lstrip('#'))
        end = next((at for at in range(start + 1, len(lines)) if re.match(rf'#{{1,{depth}}}\s', lines[at])), len(lines))
        for named in sorted(set(re.findall(r'`([A-Za-z_][\w.:/-]*)`', ' '.join(lines[start:end])))):
            held = [f'{repo}: {n}' for repo in REPOSITORIES
                    for n in [len(git(repo, 'grep', '-l', '-F', named.split('/')[-1], '--', ':!*.md').split())] if n]
            print(f'`{named}` is in ' + ('; '.join(f'{h} file(s)' for h in held) or 'no source file'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
