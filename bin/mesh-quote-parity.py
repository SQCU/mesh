"""design/review-warnings.md#9-a-quoted-turn-is-found-where-it-was-typed"""
import datetime
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mesh_commits

LOG = '/Users/mdot/metal-microbench/output_data/quote-parity-commits.jsonl'
TURNS = '/Users/mdot/metal-microbench/output_data/typed-turns.txt'
READ = '/Users/mdot/metal-microbench/output_data/typed-turns.read.json'
SEEN = 'quote-parity.head'
DOCUMENTS = ['*.md', '*.txt', '*.rst']
TRANSCRIPTS = [os.path.expanduser('~/.claude-personal/projects/-Users-mdot-dox-mesh/*.jsonl'),
               os.path.expanduser('~/.claude-personal/projects/-Users-mdot-metal-microbench/*.jsonl'),
               os.path.expanduser('~/.codex/history.jsonl')]
SHORTEST = 40


def squeezed(text):
    return re.sub(r'\s+', ' ', text).strip()


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for inner in value.values():
            yield from strings(inner)
    elif isinstance(value, list):
        for inner in value:
            yield from strings(inner)


def typed():
    read = json.load(open(READ)) if os.path.exists(READ) else {}
    with open(TURNS, 'a') as out:
        for path in sorted(p for pattern in TRANSCRIPTS for p in glob.glob(pattern)):
            start = read.get(path, 0)
            if os.path.getsize(path) <= start:
                continue
            codex = path.endswith('history.jsonl')
            with open(path, 'rb') as f:
                f.seek(start)
                for raw in f:
                    if not raw.endswith(b'\n'):
                        break
                    start += len(raw)
                    if not codex and b'"kind":"human"' not in raw:
                        continue
                    try:
                        record = json.loads(raw)
                    except ValueError:
                        continue
                    held = [record.get('text')] if codex else [record.get('message'), record.get('attachment')]
                    for text in strings(held):
                        out.write(squeezed(text) + '\n')
            read[path] = start
    with open(READ + '.new', 'w') as f:
        json.dump(read, f)
    os.replace(READ + '.new', READ)
    return open(TURNS).read()


def quoted(repo, sha):
    diff = mesh_commits.git(repo, 'show', '--no-color', '--unified=0', '--format=', '--diff-filter=AMR', sha, '--',
                            *DOCUMENTS)
    quotes, held = [], []
    for line in diff.splitlines() + ['']:
        block = re.match(r'\+\s*>\s?(.*)', line)
        if block and block.group(1).strip():
            held.append(block.group(1))
        elif held:
            quotes.append(squeezed(' '.join(held)))
            held = []
    return quotes


def missing(quote, turns):
    pieces = [squeezed(piece) for piece in re.split(r'\[\.\.\.\]|\.\.\.|…', quote)]
    return [piece for piece in pieces if len(piece) >= SHORTEST and piece not in turns]


def main():
    try:
        json.load(sys.stdin)
    except ValueError:
        pass
    messages = []
    for repo in mesh_commits.REPOSITORIES:
        lock = mesh_commits.held(repo, SEEN)
        if lock is None:
            continue
        turns = None
        for sha in mesh_commits.new_commits(repo, SEEN):
            quotes = quoted(repo, sha)
            if quotes:
                turns = typed() if turns is None else turns
                absent = [piece for quote in quotes for piece in missing(quote, turns)]
                mesh_commits.appended(LOG, json.dumps({
                    'at': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
                    'repository': repo, 'commit': sha,
                    'subject': mesh_commits.git(repo, 'show', '-s', '--format=%s', sha),
                    'quotes': len(quotes), 'not_typed': absent}))
                if absent:
                    messages.append(f'WARNING: {sha[:10]} in {repo} adds {len(absent)} block-quoted passage(s) that no '
                                    f'typed turn contains (design/review-warnings.md rule 9); the first begins '
                                    f'{absent[0][:120]!r}')
            mesh_commits.advance(repo, SEEN, sha)
    if messages:
        message = '\n'.join(messages) + f'\nEach commit is appended to {LOG}.'
        print(json.dumps({'systemMessage': message,
                          'hookSpecificOutput': {'hookEventName': 'PostToolUse', 'additionalContext': message}}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
