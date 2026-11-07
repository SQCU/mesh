"""The change-request review of each commit to mesh (design/change-requests.md): each change request the commit touches,
read whole by Claude Sonnet with no tools against the standard, after string checks of its quoted turns against the
session transcripts and of its quoted words against the retrieval records. A Claude Code PostToolUse payload on stdin
(each repository's .claude/settings.json), or change-request paths as arguments for a review before a commit; what the
review finds goes to stderr with exit 2, which Claude Code hands back to the committing session."""
import fnmatch
import glob
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile

MESH = '/Users/mdot/dox/mesh'
DESIGN = os.path.join(MESH, 'design')
TRANSCRIPTS = ['/Users/mdot/.claude-personal/projects/-Users-mdot-metal-microbench',
               '/Users/mdot/.claude-personal/projects/-Users-mdot-dox-mesh']
RECORDS = ('*-sources-*.md', 'replay-grounding-*.md', 'packing-and-coflows-*.md', 'view-sources-*.md',
           'final-lowering-sources-*.md', 'lowering-sources-*.md')
TURN_RECORD = 'operator-turns-*.md'
READER = ['claude', '-p', '--model', 'sonnet', '--tools', '', '--setting-sources', 'user',
          '--no-session-persistence', '--strict-mcp-config']
CACHE = os.path.join(tempfile.gettempdir(), 'mesh-cr-review')

_c9 = importlib.util.spec_from_file_location('mesh_c9_review', os.path.join(MESH, 'bin', 'mesh-c9-review.py'))
c9 = importlib.util.module_from_spec(_c9)
_c9.loader.exec_module(c9)

RULINGS = """The operator's rulings on wording, verbatim:
- U163 (2026-11-06 14:33): "if all of these are citable, none of these ever need to be called 'plan' ever again, then,
  don't they? we can say 'aot compiled memory plan' 'aot compiled static link schedule' 'aot compiled persistent
  requests' 'aot compiled memory schedule' 'aot allocated page table'... the big isseu here is merely that the user
  noticed several different AOT IRs or AOT serializations were being called 'plan', and that this is clearly polysemous
  in a way that departs from the literature... i don't care if something is called mesh_plan as a function btw"
- U186 (2026-11-07): "cite xternal algorithms, data structure, and computer science literature, both to describe what
  code already does and what the code probably *should* do"
- U187: "when code does osmething nonsensical that doesn't come from a citation we can say that the c urrent approach is
  idiomatic and we need to implement the referenced approach."
- U191: "per threadgroup pages is a made up idea that has nothing to do with a citation. purely idiomatic, purely
  fictional problem."
- U192: "a garnish added after something was authorized on the premise it was only a list of citations and plain
  statements to implement feature q through algorithm a_q?"
- 2026-11-07, on "a page task": "a page is a region of memory. everything else here is idiomatic and i do not want it
  described again in the immediate session in anything but canonical data structures, algorithms, and citations."
- 2026-11-07, on "batch of 1": "this is the idiomaticity that the current owrk is cleraly nad literally and ever
  emphatically *correcting* by grounding things that sound self-evident by weight of repetition with *sources* that
  *jsutify* their history"
"""

PROMPT = """You review one change request of the mesh repository. On stdin come, in order: the repository's standard for
change requests (design/change-requests.md), the request's own rules ("How it is implemented"), the record of the names
of the AOT compiler's outputs ("Terms"), the operator's rulings on wording, the string checks already made, and the
request itself with line numbers.

Read every sentence of the request outside its "## The turns" section and outside quotation marks. Each must be one of:
- O: the operator's words, quoted, with the turn named;
- S: a published source's statement, quoted, naming the source and where it was read;
- C: the location of code the request replaces or keeps (a commit, or a file and line), stated as what that code is;
- P: a stop, which names what no quoted source says, and writes nothing for it;
- A: a sentence that applies an O or S of its own item to a C of its own item, and adds nothing to them;
- H: a heading, a list label, or a sentence that only introduces an O, S, C or P.

Report:
- X: every sentence that is none of these, and every A sentence that adds anything its item's O and S don't contain: a
  structure, a mechanism, a term, a constant, a scope, an example, a reason or a problem. Name what it adds.
- D: every document under design/ written by Claude used as a source. The only documents that may be cited are the
  retrieval records (design/*-sources-*.md, replay-grounding-*.md, packing-and-coflows-*.md, view-sources-*.md,
  final-lowering-sources-*.md, lowering-sources-*.md), as where a quoted source was read, and design/operator-turns-*.md,
  for turn numbers. A retrieval record's own judgment (a "verdict", a "deviates") is not a source either.
- I: every term for a structure, mechanism or problem that neither the operator's quoted words nor a quoted source
  uses: the code's own vocabulary, or Claude's. "plan" used bare for any structure is one; `mesh_plan` as a function's
  name, and the five names in "Terms", are not.

One line per finding, in line order: `line N: X|D|I "the words": what it adds, or which rule`, N the line where the
sentence begins. Don't repeat a string check's finding. If there is nothing to report, answer exactly PASS. Nothing
else: no summary, no praise, no advice."""


def flat(text):
    """whitespace, quotes, dashes, markdown emphasis, line-leading quote and comment marks and AsciiDoc macros made
    uniform, for string search"""
    for a, b in (('‘', "'"), ('’', "'"), ('“', '"'), ('”', '"'), ('–', '-'), ('—', '-'),
                 ('−', '-'), (' ', ' ')):
        text = text.replace(a, b)
    text = re.sub(r'(?m)^[ \t]*(?:>[ \t]*)*(?://+|#+|\*+(?!\*)|-(?= ))?[ \t]?', '', text)
    text = re.sub(r'\b(?:ename|slink|pname|flink|sname|elink|basetype|dlink|tlink):', '', text)
    return re.sub(r'\s+', ' ', re.sub(r'[*_`]', '', text)).strip()


def quotations(text):
    """the quoted spans of a text, its quotation marks paired in order"""
    text = text.replace('“', '"').replace('”', '"')
    marks = [m.start() for m in re.finditer('"', text)]
    return [text[a + 1:b] for a, b in zip(marks[0::2], marks[1::2])]


def operator_turns():
    """every operator turn in the transcripts, one string; each transcript's turns cached under its size and mtime"""
    os.makedirs(CACHE, exist_ok=True)
    texts = []
    for directory in TRANSCRIPTS:
        for path in sorted(glob.glob(os.path.join(directory, '*.jsonl'))):
            stat = os.stat(path)
            key = os.path.join(CACHE, re.sub(r'[^\w.-]', '_', path) + '.json')
            try:
                with open(key) as f:
                    cached = json.load(f)
                if cached['size'] == stat.st_size and cached['mtime'] == stat.st_mtime:
                    texts.extend(cached['turns'])
                    continue
            except (OSError, ValueError, KeyError):
                pass
            turns = []
            with open(path, encoding='utf-8', errors='replace') as f:
                for line in f:
                    if '"type":"user"' not in line or '"tool_result"' in line:
                        continue
                    try:
                        record = json.loads(line)
                    except ValueError:
                        continue
                    if record.get('type') != 'user' or record.get('isMeta') or record.get('isCompactSummary'):
                        continue
                    content = (record.get('message') or {}).get('content')
                    if isinstance(content, list):
                        content = '\n'.join(b.get('text', '') for b in content if isinstance(b, dict) and
                                            b.get('type') == 'text')
                    if isinstance(content, str) and content and not content.startswith('<task-notification'):
                        turns.append(flat(content))
            with open(key, 'w') as f:
                json.dump({'size': stat.st_size, 'mtime': stat.st_mtime, 'turns': turns}, f)
            texts.extend(turns)
    return '\n'.join(texts)


def records():
    """the retrieval records' text and the two repositories' source text, one string: a quotation is found in a record
    where it was read, or in the code it states a fact of"""
    paths = sorted({p for pattern in RECORDS for p in glob.glob(os.path.join(DESIGN, pattern))})
    for root, patterns in ((os.path.join(MESH, 'rdma'), ('*.c', '*.h', '*.m', '*.py')),
                           ('/Users/mdot/metal-microbench', ('*.m', '*.h', '*.metal', 'tools/*.m', 'tools/*.py'))):
        paths += sorted(p for pattern in patterns for p in glob.glob(os.path.join(root, pattern)))
    return '\n'.join(flat(open(p, encoding='utf-8', errors='replace').read()) for p in paths)


def units(lines):
    """(heading, first line number, text) for each paragraph or list item: a line and the lines that continue it"""
    heading, out, current = '', [], None
    for number, line in enumerate(lines, 1):
        if line.startswith('#'):
            if current:
                out.append(current)
            current = None
            heading = line.lstrip('#').strip()
            out.append((heading, number, line))
            continue
        starts = (not line.strip() or re.match(r'\s*(?:[-*]|\d+\.)\s', line) or line.startswith('>') or
                  (current and current[2].startswith('>') and not line.startswith('>')))
        if starts or current is None:
            if current:
                out.append(current)
            current = (heading, number, line) if line.strip() else None
        else:
            current = (current[0], current[1], current[2] + '\n' + line)
    if current:
        out.append(current)
    return out


def fragments(quoted):
    """a quotation's parts between its marks of omission"""
    return [f for f in (flat(p) for p in re.split(r'\[\s*(?:\.\.\.|…)\s*\]|…|\.\.\.', quoted)) if len(f) >= 20]


def checks(path, text):
    """the string checks: quoted turns not in a transcript, quoted words not in a record or turn, bare "plan", and
    design documents other than the records cited under "Source" """
    found = []
    lines = text.split('\n')
    turns = operator_turns()
    kept = records()
    own = []
    for heading, number, unit in units(lines):
        if heading == 'The turns' and unit.startswith('>'):
            body = ' '.join(l[1:].strip() for l in unit.split('\n') if l.startswith('>'))
            own.append(flat(body))
            for part in fragments(body):
                if part not in turns:
                    found.append(f'line {number}: T "{part[:100]}": not in any operator turn of the transcripts')
    own = '\n'.join(own)
    in_source = False
    for heading, number, unit in units(lines):
        if heading == 'The turns' or unit.startswith('#'):
            in_source = False
            continue
        if re.match(r'\s*- \*\*(?:Code|Change|Source)', unit):
            in_source = unit.lstrip().startswith('- **Source')
        for quoted in quotations(unit):
            for part in fragments(quoted):
                if part not in kept and part not in turns and part not in own:
                    found.append(f'line {number}: Q "{part[:100]}": not in any retrieval record, operator turn or '
                                 'source file')
        bare = re.sub(r'`[^`]*`', ' ', unit)
        words = re.sub(r'"[^"]*"|“[^”]*”', ' ', bare.replace('\n', ' '))
        words = re.sub(r'\S+\.(?:md|c|h|m|mm|py|metal|jsonl?)\b|\bmesh_plan\b|AOT compiled memory plan', ' ', words,
                       flags=re.I)
        if re.search(r'\bplans?\b|\bplan\'s\b|\bplans\'', words, re.I):
            found.append(f'line {number}: W "{flat(unit)[:100]}": "plan" bare (problems-of-14b9a757-change-request.md'
                         ', "## Terms"; U163)')
        if in_source:
            for name in re.findall(r'design/([\w.-]+\.md)', unit):
                allowed = any(fnmatch.fnmatch(name, p) for p in RECORDS + (TURN_RECORD,))
                if not allowed:
                    found.append(f'line {number}: D "design/{name}": a document Claude wrote, cited as a source')
    return found


def standard():
    """what the reader is given before the request: the standard, the request's rules, the Terms record, the rulings"""
    parts = [open(os.path.join(DESIGN, 'change-requests.md'), encoding='utf-8').read()]
    terms = open(os.path.join(DESIGN, 'problems-of-14b9a757-change-request.md'), encoding='utf-8').read()
    match = re.search(r'^## Terms\n.*?(?=^## )', terms, re.M | re.S)
    if match:
        parts.append(match.group(0))
    parts.append(RULINGS)
    return '\n\n'.join(parts)


def review(path, text):
    """the string checks, then the reader's findings, as lines"""
    found = checks(path, text)
    lines = text.split('\n')
    rules = re.search(r'^## How it is implemented\n.*?(?=^## )', text, re.M | re.S)
    numbered = '\n'.join(f'{n:4} {l}' for n, l in enumerate(lines, 1))
    given = '\n\n'.join([standard(), rules.group(0) if rules else '', 'String checks already made:\n' +
                         ('\n'.join(found) or 'none'), f'The request, {path}:\n' + numbered])
    read = subprocess.run([*READER, '--system-prompt', PROMPT, 'Review this change request.'], input=given,
                          capture_output=True, text=True, cwd=tempfile.gettempdir(), timeout=1100)
    if read.returncode:
        found.append(f'the reader failed: {read.stderr.strip()[:500]}')
    elif read.stdout.strip() != 'PASS':
        found.extend(read.stdout.strip().split('\n'))
    return found


def requests(paths):
    return [p for p in paths if re.search(r'(?:^|/)design/[^/]*change-request\.md$', p)]


def main():
    reports = []
    if sys.argv[1:]:
        for path in requests([os.path.relpath(os.path.abspath(p), MESH) for p in sys.argv[1:]]):
            found = review(path, open(os.path.join(MESH, path), encoding='utf-8').read())
            if found:
                reports.append(f'Change-request review of {path} (the working tree):\n' + '\n'.join(found))
    else:
        for repo, sha in c9.made(json.load(sys.stdin)):
            changed = c9.git(repo, 'show', '--name-only', '--format=', sha).split('\n')
            for path in requests(changed):
                text = c9.git(repo, 'show', f'{sha}:{path}')
                if not text:
                    continue
                found = review(path, text)
                if found:
                    reports.append(f'Change-request review of {path} in {sha[:10]} of {repo} (design/change-requests.md'
                                   '), read by Claude Sonnet:\n' + '\n'.join(found))
    if reports:
        print('\n\n'.join(reports), file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
