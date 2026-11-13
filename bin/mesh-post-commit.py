"""design/review-warnings.md#14-one-record-for-each-commit"""
import datetime
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.request
from mesh_notes import append

MESH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTES = 'measurements'
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
docstrings, and claims in prompts given to reviewers or implementers. Not counted: the commit's own message
describing its own change, or an ordinary runtime message that states no fact about the repository.
REQUIREMENT: an added line states, in the writer's own words and not inside a quotation, a claim of requirement:
what must, shall, should, has to, needs to or is to be done, written, kept, deleted or satisfied; a rule; an order of
work; a condition; a definition of when something is done. Include prompts directing reviewers or implementers.
REQUIREMENT-REMOVED: a removed line states a requirement that no added line states in the same words. Measure
that textual change, not whether the requirement has been withdrawn or whether its removal is justified.
SOURCE: the commit changes source code of the software the repository builds and ships. Scripts that review,
measure or gate commits and tool calls, their settings, documents and data are not that software; a commit that
changes only those is NO.
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
COMMENT: an added comment or docstring in source code contains natural-language explanation or requirements
besides citations of documents (a document path and section, a bibliographic reference, an API header and line,
a specification's URL). Formal proofs/algebra, machine directives, runtime text and model prompts are not commentary.

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
ITEM = """You are given a named item, its complete change request document, then one git commit's diff.
Measure the named item only; do not advise, explain or recommend. Preserve the item's qualifications and citation
roles. The document's claims about code are not execution evidence, and this review does not establish that its
requirements are authentic. You have no execution results or external references beyond the supplied text.

For each clause of the named item that states something to write, change or remove, quote the clause exactly:
CLAUSE: "<clause>" | CARRIED: YES (the diff carries out the clause, as far as the supplied source shows)
or CARRIED: NO (the diff changes the place the clause is about and does not carry it out)
or CARRIED: NOT ADDRESSED (the diff does not address it, or the supplied material cannot establish it)

Then one line:
ADDED: YES (the diff adds behaviour, a rule, a check, a guard, a wait, a copy or an interface that no sentence of the
item states) or ADDED: NO. This measures presence, not whether that addition is justified or forbidden.

Nothing else."""


def git(repo, *args, text=True):
    done = subprocess.run(['git', '-C', repo, *args], capture_output=True, env=ENVIRONMENT)
    return done.stdout.decode(errors='replace').strip() if text else done.stdout


def asked(reader, prompt, given):
    read = subprocess.run([*reader, '--system-prompt', prompt, 'Answer.'], input=given, capture_output=True, text=True,
                          cwd=tempfile.gettempdir(), timeout=600, env=ENVIRONMENT)
    if read.returncode:
        raise RuntimeError(f'the reader exited {read.returncode}: {(read.stderr.strip() or read.stdout.strip())[:200]}')
    return read.stdout


def squeezed(text):
    return re.sub(r'\s+', ' ', text).strip()


def typed():
    turns = []
    for path in sorted(p for pattern in TRANSCRIPTS for p in glob.glob(pattern)):
        codex = path.endswith('history.jsonl')
        with open(path) as stream:
            for raw in stream:
                try:
                    record = json.loads(raw)
                except ValueError:
                    continue
                if not codex and (record.get('type') != 'user' or record.get('origin', {}).get('kind') != 'human'):
                    continue
                content = record.get('text', '') if codex else record.get('message', {}).get('content', '')
                text = content if isinstance(content, str) else '\n'.join(
                    block.get('text', '') for block in content if block.get('type') == 'text')
                turns.append(squeezed(text))
    return turns


def commit(repo, sha, named):
    shown = git(repo, 'show', '--no-color', '--format=%B', sha)
    answer = asked(HAIKU, COMMIT, shown[:LIMIT])
    bits = dict(re.findall(r'^(' + '|'.join(BITS) + r'): (YES|NO)\s*$', answer, re.M))
    if len(bits) != len(BITS):
        raise RuntimeError(f'the reader answered {sorted(bits)} of {len(BITS)} questions')
    removed = [line[1:].strip() for line in shown.splitlines() if line.startswith('-') and not line.startswith('---')]
    added = {line[1:].strip() for line in shown.splitlines() if line.startswith('+') and not line.startswith('+++')}
    said = [squeezed(passage) for passage in re.findall(r'^SAID: "(.*)"\s*$', answer, re.M)]
    turns = typed() if said else []
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
                  said=len(said), not_typed=[passage for passage in said if not any(passage in turn for turn in turns)],
                  transcript_sources=TRANSCRIPTS,
                  names_item=[f'{document}, item {name}' for document, name in named],
                  truncated=len(shown) > LIMIT)
    record['source_without_item'] = record['source'] and not named
    record['process'] = PROCESS
    given = given_items(repo, PROCESS) if PROCESS else []
    record['launched_on'] = given
    return record


def source(repo, sha):
    shown = git(repo, 'show', '-W', '--no-color', '--format=', sha)
    answer = asked(HAIKU, SOURCE, shown[:LIMIT])
    bits = dict(re.findall(r'^(GRAPH|SECOND-PRODUCER|COMMENT): (YES|NO)\s*$', answer, re.M))
    if len(bits) != 3:
        raise RuntimeError(f'the reader answered {sorted(bits)} of 3 questions')
    return {'graph_algorithm_outside_module': bits['GRAPH'] == 'YES', 'second_producer': bits['SECOND-PRODUCER'] == 'YES',
            'comment_not_a_citation': bits['COMMENT'] == 'YES',
            'places': re.findall(r'^AT: (.+?)\s*$', answer, re.M)[:20], 'truncated': len(shown) > LIMIT}


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
    record = {'prose': prose.group(1) == 'YES', 'guide': None if cited.group(1) == 'NONE' else cited.group(1),
              'truncated': len(text) > LIMIT}
    if not record['prose'] or record['guide'] is None:
        return record
    name = record['guide'].split('#')[0]
    if name.startswith(('http://', 'https://')):
        try:
            with urllib.request.urlopen(name, timeout=30) as response:
                guide = response.read().decode(response.headers.get_content_charset() or 'utf-8')
                record['guide_source'] = response.url
        except Exception as error:
            return {**record, 'guide_found': False, 'reason': f'{type(error).__name__}: {error}'[:300]}
    else:
        places = ((repo, sha, os.path.normpath(os.path.join(os.path.dirname(path), name))), (repo, sha, name),
                  (MESH, git(MESH, 'rev-parse', 'HEAD'), name))
        located = next(((root, revision, at) for root, revision, at in places
                        if git(root, 'cat-file', '-t', f'{revision}:{at}') == 'blob'), None)
        guide = git(located[0], 'show', f'{located[1]}:{located[2]}') if located else None
        record['guide_source'] = located
    record['guide_found'] = guide is not None
    if guide is not None:
        record['guide_sha256'] = hashlib.sha256(guide.encode()).hexdigest()
        given = f'THE GUIDE:\n{guide}\n\nTHE NEW FILE ({path}):\n{text}'
        record['truncated'] = len(given) > LIMIT
        answer = re.search(r'^ADHERES: (YES|NO)\s*$',
                           asked(SONNET, ADHERES, given[:LIMIT]), re.M)
        if not answer:
            raise RuntimeError('the reader did not answer ADHERES')
        record['adheres'] = answer.group(1) == 'YES'
    return record


def item(repo, sha, document, name, mesh_revision):
    root, revision = ((repo, sha) if git(repo, 'cat-file', '-t', f'{sha}:{document}') == 'blob'
                      else (MESH, mesh_revision))
    blob = git(root, 'rev-parse', '--verify', f'{revision}:{document}')
    if not blob:
        return {'document_found': False, 'document_revision': revision}
    text = git(root, 'show', blob)
    diff = git(repo, 'show', '--no-color', '--format=', sha)
    given = f'THE NAMED ITEM: {document}, item {name}\nTHE DOCUMENT:\n{text}\n\nTHE COMMIT:\n{diff}'
    answer = asked(SONNET, ITEM, given[:LIMIT])
    added = re.search(r'^ADDED: (YES|NO)', answer, re.M)
    if not added:
        raise RuntimeError('the reader did not answer ADDED')
    clauses = [{'clause': clause, 'carried': state, 'in_document': squeezed(clause) in squeezed(text)}
               for clause, state in re.findall(r'^CLAUSE: "(.*)" \| CARRIED: (YES|NO|NOT ADDRESSED)\s*$', answer, re.M)]
    return {'document_found': True, 'document_repository': root, 'document_revision': revision,
            'document_blob': blob, 'clauses': clauses, 'response': answer,
            'added': added.group(1) == 'YES', 'truncated': len(given) > LIMIT}


def noted(repo, sha, measure, made):
    try:
        record = made()
    except Exception as error:
        record = {'measured': False, 'reason': f'{type(error).__name__}: {error}'[:300]}
    line = json.dumps({'at': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
                       'measure': measure, **record})
    append(repo, NOTES, line, sha)
    return record


def main():
    repo, sha = sys.argv[1], sys.argv[2]
    named = NAMED.findall(git(repo, 'show', '-s', '--format=%B', sha))
    mesh_revision = git(MESH, 'rev-parse', 'HEAD')

    noted(repo, sha, 'commit', lambda: commit(repo, sha, named))
    noted(repo, sha, 'source', lambda: source(repo, sha))
    raw = git(repo, 'diff-tree', '--root', '--no-commit-id', '-r', '--diff-filter=A', '--name-only', '-z', sha)
    for path in filter(None, raw.split('\0')):
        noted(repo, sha, f'new file {path}', lambda path=path: new_file(repo, sha, path))
    for document, name in named:
        noted(repo, sha, f'{document}, item {name}',
              lambda document=document, name=name: item(repo, sha, document, name, mesh_revision))
    return 0


if __name__ == '__main__':
    sys.exit(main())
