"""design/review-warnings.md#14-one-record-for-each-commit"""
import datetime
import glob
import json
import os
import re
import subprocess
import sys
import tempfile

MESH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX = 'design/change-requests-index.md'
NOTES = 'measurements'
TURNS = os.path.expanduser('~/.cache/mesh-typed-turns.txt')
READ = TURNS + '.read.json'
TRANSCRIPTS = [os.path.expanduser('~/.claude-personal/projects/*/*.jsonl'), os.path.expanduser('~/.codex/history.jsonl')]
NAMED = re.compile(r'^Change-Request: (\S+), item ([A-Z]{0,2}\.?\d+(?:\.\d+)*)\s*$', re.M)
LIMIT = 300000
HAIKU = ['claude', '-p', '--model', 'claude-haiku-5-5', '--tools', '', '--setting-sources', 'user',
         '--no-session-persistence', '--strict-mcp-config']
SONNET = [part if part != 'claude-haiku-5-5' else 'sonnet' for part in HAIKU]
PROCESS = os.path.basename(os.environ.get('CLAUDE_CODE_MESSAGING_SOCKET', '')).split('.')[0] or None
ENVIRONMENT = {name: value for name, value in os.environ.items() if not name.startswith('GIT_')}
COMMIT = """You are given one git commit on stdin: its message, then its diff. Measure only whether each thing is
present in what the commit ADDS (its message, and the diff lines that begin with '+') or, where a question says so,
REMOVES (lines that begin with '-'). Judge from what the text is, in any file of any name. Do not consider why it was
done, whether it is justified, or what the commit says about itself.

ASSERTION: an added line is an assertion: an assert statement, call or macro in any language, or a function the
commit defines to play that part. Raising an error that refuses an unsupported input is not one.
TEST: the commit adds a test: a file, function, case, harness, check, sample-and-compare case, golden or expected
output, or script whose purpose is to run code and compare what it produces with expected or reference values.
FACT: an added line states, in the writer's own words and not inside a quotation, a claim of fact about the
repository: what some code or file is, does, holds, has or lacks; what a commit did; what is finished, covered,
outstanding, passing or failing; a count or a measurement. This includes prose in documents, comments and
docstrings. The commit's own message describing its own change is not counted.
REQUIREMENT: an added line states, in the writer's own words and not inside a quotation, a claim of requirement:
what must, shall, should, has to, needs to or is to be done, written, kept, deleted or satisfied; a rule; an order of
work; a condition; a definition of when something is done.
REQUIREMENT-REMOVED: a removed line states a requirement that no added line states in the same words.
SOURCE: the commit changes what a program does when it runs: source code of the software the repository builds.
Changes only to documents, to review hooks and their settings, or to data are not this.
HOOK: the commit changes a review hook, the questions a hook asks, a gate on tool calls, or the settings that say
which hooks run.

Then list every passage that the commit's added text presents as the words of the user or the operator, each copied
exactly as the commit writes it, one per line as SAID: "<passage>". If there is none, write SAID: NONE.

Then name each file in which the commit adds a claim of fact or of requirement, one per line as CLAIMS: <path as the
diff gives it>. If there is none, write CLAIMS: NONE.

Answer with exactly these lines first, each ending in YES or NO, then the SAID lines, then the CLAIMS lines, and nothing else:
ASSERTION:
TEST:
FACT:
REQUIREMENT:
REQUIREMENT-REMOVED:
SOURCE:
HOOK:"""
BITS = ('ASSERTION', 'TEST', 'FACT', 'REQUIREMENT', 'REQUIREMENT-REMOVED', 'SOURCE', 'HOOK')
SOURCE = """You are given one git commit's diff on stdin, each changed function whole. Measure only whether each
thing is present in the lines the commit adds or changes, in any file of any name. Do not advise.

GRAPH: a function the commit adds or changes carries a graph algorithm (shortest paths, traversals and worklists,
reachability, topological orders, spanning trees, flows, cuts, packings, union-find, components, cycle checks, path
enumeration, walks over parent or next-hop tables, tree heights) anywhere but in the graph module (mesh-graph.h,
mesh-graph.c, mesh_graph.py). Code that only calls that module or builds its inputs is not this.
SECOND-PRODUCER: a function the commit adds or changes computes again a value one module already produces: where a
value's bytes lie in a region, the fields of a record a shared header declares, which functions must run before
which, or a record of synchronization edges. Code that calls the producer, or only mirrors a record's layout to pass
operands, is not this.
COMMENT: an added comment or docstring in source code contains anything besides citations of documents (a document
path and section, a bibliographic reference, an API header and line, a specification's URL).

Answer with exactly these lines, each ending in YES or NO, then for each YES one or more lines
AT: <path>:<line> <function or first words>, and nothing else:
GRAPH:
SECOND-PRODUCER:
COMMENT:"""
NEW = """You are given one new file on stdin. Answer two questions from the file's text alone.

PROSE: is the file a document written for people to read (prose, a specification, a note, a guide), as opposed to
data, configuration, generated output or a listing?
GUIDE: does the file name, by a path or a Markdown link, a document it says it follows as its style guide, standard
or format? Give that path or link target exactly as the file writes it, or NONE.

Answer with exactly two lines and nothing else:
PROSE: YES or PROSE: NO
GUIDE: <path> or GUIDE: NONE"""
ADHERES = """You are given a style guide, then one new file that says it follows that guide. You have nothing else
and need nothing else. Measure only; do not advise.

Answer with exactly one line and nothing else:
ADHERES: YES or ADHERES: NO"""
ITEM = """You are given one item of a change request, then one git commit's diff. You have nothing else and need
nothing else. Measure; do not advise, explain or recommend.

You are given the item's sentences numbered. For each number give one line:
<number>: YES (the diff does what the sentence says)
<number>: NO (the diff changes the place the sentence is about and does not do what it says)
<number>: NOT ADDRESSED (the diff does not touch what the sentence is about, or the sentence states nothing to write)

Then one line:
ADDED: YES (the diff adds behaviour, a rule, a check, a guard, a wait, a copy or an interface that no sentence of the
item states) or ADDED: NO

Nothing else."""


def git(repo, *args, text=True):
    done = subprocess.run(['git', '-C', repo, *args], capture_output=True, env=ENVIRONMENT)
    return done.stdout.decode(errors='replace').strip() if text else done.stdout


def asked(reader, prompt, given):
    read = subprocess.run([*reader, '--system-prompt', prompt, 'Answer.'], input=given, capture_output=True, text=True,
                          cwd=tempfile.gettempdir(), timeout=600, env=ENVIRONMENT)
    if read.returncode:
        raise RuntimeError(f'the reader exited {read.returncode}: {read.stderr.strip()[:200]}')
    return read.stdout


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
    os.makedirs(os.path.dirname(TURNS), exist_ok=True)
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


def commit(repo, sha):
    shown = git(repo, 'show', '--no-color', '--format=%B', sha)
    answer = asked(HAIKU, COMMIT, shown[:LIMIT])
    bits = dict(re.findall(r'^(' + '|'.join(BITS) + r'): (YES|NO)\s*$', answer, re.M))
    if len(bits) != len(BITS):
        raise RuntimeError(f'the reader answered {sorted(bits)} of {len(BITS)} questions')
    removed = [line[1:].strip() for line in shown.splitlines() if line.startswith('-') and not line.startswith('---')]
    added = {line[1:].strip() for line in shown.splitlines() if line.startswith('+') and not line.startswith('+++')}
    said = [squeezed(passage) for passage in re.findall(r'^SAID: "(.*)"\s*$', answer, re.M)]
    turns = typed() if said else ''
    pieces = [squeezed(piece) for passage in said for piece in re.split(r'\[\.\.\.\]|\.\.\.|…', passage)]
    named = NAMED.findall(git(repo, 'show', '-s', '--format=%B', sha))
    quoting, path = set(), None
    for line in shown.splitlines():
        if line.startswith('+++ '):
            path = line[6:] if line[4:6] == 'b/' else None
        elif path and re.match(r'\+\s*>', line):
            quoting.add(path)
    claiming = [name.strip() for name in re.findall(r'^CLAIMS: (.+?)\s*$', answer, re.M) if name.strip() != 'NONE']
    record = {name.lower().replace('-', '_'): value == 'YES' for name, value in bits.items()}
    record.update(claims_without_quote=[name for name in claiming if name not in quoting],
                  quoted_line_removed=any(line.startswith('>') and line not in added for line in removed),
                  said=len(said), not_typed=[piece for piece in pieces if len(piece) >= 12 and piece not in turns],
                  names_item=[f'{document}, item {name}' for document, name in named],
                  truncated=len(shown) > LIMIT)
    record['source_without_item'] = record['source'] and not named
    record['process'] = PROCESS
    given = given_items(repo, PROCESS) if PROCESS else []
    record['launched_on'] = given
    record['names_item_not_launched_on'] = [name for name in record['names_item']
                                            if PROCESS and not any(name.split(', item ')[0] in g and
                                                                   g.endswith('item ' + name.split(', item ')[1]) for g in given)]
    return record, named


def source(repo, sha):
    shown = git(repo, 'show', '-W', '--no-color', '--format=', sha)
    answer = asked(HAIKU, SOURCE, shown[:LIMIT])
    bits = dict(re.findall(r'^(GRAPH|SECOND-PRODUCER|COMMENT): (YES|NO)\s*$', answer, re.M))
    if len(bits) != 3:
        raise RuntimeError(f'the reader answered {sorted(bits)} of 3 questions')
    return {'graph_algorithm_outside_module': bits['GRAPH'] == 'YES', 'second_producer': bits['SECOND-PRODUCER'] == 'YES',
            'comment_not_a_citation': bits['COMMENT'] == 'YES',
            'at': re.findall(r'^AT: (.+?)\s*$', answer, re.M)[:20], 'truncated': len(shown) > LIMIT}


def given_items(repo, process):
    listed = git(MESH, 'notes', '--ref', 'briefs', 'list').split('\n')
    found = set()
    for line in filter(None, listed):
        for entry in git(MESH, 'notes', '--ref', 'briefs', 'show', line.split()[1]).splitlines():
            try:
                record = json.loads(entry)
            except ValueError:
                continue
            if record.get('process') == process:
                found.update(record.get('given', []))
    return sorted(found)


def new_file(repo, sha, path):
    raw = git(repo, 'show', f'{sha}:{path}', text=False)
    try:
        text = raw.decode()
    except UnicodeDecodeError:
        return {'binary': True}
    answer = asked(HAIKU, NEW, text[:LIMIT])
    prose = re.search(r'^PROSE: (YES|NO)\s*$', answer, re.M)
    cited = re.search(r'^GUIDE: (.+?)\s*$', answer, re.M)
    if not prose or not cited:
        raise RuntimeError('the reader did not answer PROSE and GUIDE')
    record = {'prose': prose.group(1) == 'YES', 'guide': None if cited.group(1) == 'NONE' else cited.group(1)}
    if not record['prose'] or record['guide'] is None:
        return record
    name = record['guide'].split('#')[0]
    places = ((repo, sha, os.path.normpath(os.path.join(os.path.dirname(path), name))), (repo, sha, name),
              (MESH, 'HEAD', name))
    guide = next((git(root, 'show', f'{revision}:{at}') for root, revision, at in places
                  if git(root, 'cat-file', '-t', f'{revision}:{at}') == 'blob'), None)
    record['guide_found'] = guide is not None
    if guide is not None:
        answer = re.search(r'^ADHERES: (YES|NO)\s*$',
                           asked(SONNET, ADHERES, f'THE GUIDE:\n{guide}\n\nTHE NEW FILE ({path}):\n{text}'[:LIMIT]), re.M)
        if not answer:
            raise RuntimeError('the reader did not answer ADHERES')
        record['adheres'] = answer.group(1) == 'YES'
    return record


def item(repo, sha, document, name):
    index = git(MESH, 'show', f'HEAD:{INDEX}')
    if document not in {'design/' + linked for linked in re.findall(r'\]\(([^)\s]+\.md)\)', index)}:
        return {'linked': False}
    revision = sha if git(repo, 'cat-file', '-t', f'{sha}:{document}') == 'blob' else None
    lines = git(repo if revision else MESH, 'show', f'{revision or "HEAD"}:{document}').splitlines()
    found = next(((at, len(heading.group(1))) for at, line in enumerate(lines)
                  for heading in [re.match(rf'(#{{1,6}})\s+{re.escape(name)}[.:\s]', line)] if heading), None)
    if found is None:
        return {'linked': True, 'item_found': False}
    at, depth = found
    end = next((later for later in range(at + 1, len(lines)) if re.match(rf'#{{1,{depth}}}\s', lines[later])), len(lines))
    sentences = [s for s in re.split(r'(?<=[.:;?!"”])\s+', squeezed(' '.join(lines[at + 1:end]))) if len(s) > 20]
    diff = git(repo, 'show', '--no-color', '--format=', sha)
    given = (f'THE ITEM ({document}, item {name}):\n' + '\n'.join(f'{n}. {s}' for n, s in enumerate(sentences, 1)) +
             f'\n\nTHE COMMIT:\n{diff[:LIMIT]}')
    answer = asked(SONNET, ITEM, given)
    states = dict(re.findall(r'^(\d+): (YES|NO|NOT ADDRESSED)\s*$', answer, re.M))
    added = re.search(r'^ADDED: (YES|NO)', answer, re.M)
    if not added or len(states) != len(sentences):
        raise RuntimeError(f'the reader answered {len(states)} of {len(sentences)} sentences')
    return {'linked': True, 'item_found': True, 'sentences': len(sentences),
            'carried': [int(n) for n, s in states.items() if s == 'YES'],
            'not_carried': [int(n) for n, s in states.items() if s == 'NO'],
            'added': added.group(1) == 'YES', 'truncated': len(diff) > LIMIT}


def noted(repo, sha, measure, made):
    try:
        record = made()
    except Exception as error:
        record = {'measured': False, 'reason': f'{type(error).__name__}: {error}'[:300]}
    line = json.dumps({'at': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
                       'measure': measure, **record})
    subprocess.run(['git', '-C', repo, 'notes', '--ref', NOTES, 'append', '-m', line, sha], capture_output=True,
                   env=ENVIRONMENT)
    return record


def main():
    repo, sha = sys.argv[1], sys.argv[2]
    named = []

    def whole():
        record, found = commit(repo, sha)
        named.extend(found)
        return record
    noted(repo, sha, 'commit', whole)
    noted(repo, sha, 'source', lambda: source(repo, sha))
    raw = git(repo, 'diff-tree', '--root', '--no-commit-id', '-r', '--diff-filter=A', '--name-only', '-z', sha)
    for path in filter(None, raw.split('\0')):
        noted(repo, sha, f'new file {path}', lambda path=path: new_file(repo, sha, path))
    for document, name in named:
        noted(repo, sha, f'{document}, item {name}', lambda document=document, name=name: item(repo, sha, document, name))
    return 0


if __name__ == '__main__':
    sys.exit(main())
