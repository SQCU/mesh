# design/commentary-review.md
import collections
import datetime
import fnmatch
import functools
import glob
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import traceback

sys.dont_write_bytecode = True

MESH = '/Users/mdot/dox/mesh'
MICROBENCH = '/Users/mdot/metal-microbench'
NAMES = {'mesh': MESH, 'metal-microbench': MICROBENCH}
TRANSCRIPTS = ('/Users/mdot/.claude-personal/projects/-Users-mdot-metal-microbench',
               '/Users/mdot/.claude-personal/projects/-Users-mdot-dox-mesh')
STORE = 'review-commentary'
DOCUMENT = re.compile(r'^design/.*\.md$')
STANDARD = 'design/change-requests.md'
RECORDS = ('*-sources-*.md', 'replay-grounding-*.md', 'packing-and-coflows-*.md')
KINDS = (('*change-request.md', 'change request'), *((p, 'retrieval record') for p in RECORDS),
         ('operator-turns-*.md', 'record of operator turns'), ('*', 'design document'))
SOURCES = ('.c', '.h', '.m', '.mm', '.py', '.metal', '.swift', '.sh', '.cpp', '.hpp')
UNREAD = re.compile(r'(?:^|/)(?:vendor|xonotic|third_party|keys)/|(?:^|/)(?:tailscale\.key|NODE-STANDUPS\.md)$|'
                    r'\.(?:key|pem|p12)$|^docs/facts\.md$')
UNEXAMINED = {'source', 'sources', 'replaces', 'keeps', 'stop', 'stops', 'code', 'repo', 'invoked'}
SET_APART = ('The turns', 'Stops')
SHORTEST = 20
READING_TIME = 600
LARGEST_READ = 200000
GIT = {**os.environ, 'GIT_OPTIONAL_LOCKS': '0'}
READER = ['-p', '--model', 'sonnet', '--tools', '', '--setting-sources', 'user', '--no-session-persistence',
          '--strict-mcp-config']
PROMPT = """You write commentary on one document of the mesh repository, for the operator and for the next session
that works on it. Your commentary has no authority: it is not a verdict, a gate, a rejection or an instruction, and
nothing acts on it. The work that made the commit goes on whatever you write.

On stdin come the repository's standard for change requests (design/change-requests.md), the observations string
checks have already made, and the document with its line numbers.

Write observations of three kinds, one a line, each beginning `line N:` with N the line it points at:
- a sentence that states something no quoted source in its own item states: say what it states that the item's
  quotations don't;
- a requirement that seems to ask for work beyond what the operator's quoted turns ask: name the work, and the turns
  you compared it with;
- a quotation that seems attributed to the wrong source, or to no source.

A sentence that joins two of its item's quoted sources at their interface, in their words, adds nothing; don't remark
on it. Don't repeat an observation the string checks made. Write each observation as information: what the line says,
and what the quotations or turns say. No verdicts, no PASS or FAIL, no counts, no ranking, no advice, no instructions
and no praise. If you have no observation, answer exactly: No observations."""
GROUPS = (('quotations', 'Quotations'), ('statements', 'Statements and the quotations of their section'),
          ('locations', 'Code locations and commits'), ('requirements', "Requirements and the operator's turns"))
COMMON = set("""about above across after again against all almost along already also although always among and another
any are around because been before being below between both but can cannot could did does doing done down during each
either else enough even ever every few for from further had has have having here how however into its itself just
least less many may might more most much must neither never nor not now off once one only onto other others otherwise
our ours out over own per rather same shall should since some such than that the their theirs them then there these
they this those though through thus too under until upon very via was were what whatever when where whether which
while who whom whose why will with within without would yet you your item items requirement requirements write writes
written source sources code quoted quote quotes quotation operator line lines stop stops replace replaces keep keeps
change request document documents section sections turn turns commit commits name names named say says said state
states stated take takes taken give gives given make makes made get gets put puts call calls called use uses
used inside together isn't aren't doesn't don't can't won't wasn't weren't it's that's there's""".split())
ENDINGS = ('ational', 'ations', 'ation', 'ments', 'ment', 'ings', 'ing', 'ities', 'ity', 'ies', 'ied', 'ed', 'es',
           'ly', 's', 'e', 'y')
WORD = re.compile(r"[A-Za-z][A-Za-z'’-]*[A-Za-z]")
CODE = re.compile(r'(`+)[^`\n]*?\1')
LIST = re.compile(r'(\s*)(?:[-*+]|\d+[.)])\s+')
FENCE = re.compile(r'\s*(`{3,}|~{3,})')
HEADING = re.compile(r'(#{1,6})\s+(.*?)\s*#*\s*$')
LABEL = re.compile(r'\s*(?:[-*+]|\d+[.)])\s+\*\*([A-Za-z]+)')
PATH = re.compile(r'(?<![\w/.@+-])((?:\.\.?/)?(?:[\w@+-][\w.@+-]*/)*[\w@+-][\w.@+-]*\.(?:c|h|m|mm|py|metal|swift|sh|'
                  r'cpp|hpp|md|json|txt|toml|ya?ml|conf|plist))(?::(\d+)(?:\s*[–-]\s*(\d+))?)?(?![\w/])')
BARE = re.compile(r'(?:(?<=[\s(,])|^):(\d+)(?:\s*[–-]\s*:?(\d+))?(?!\d)')
SYMBOL = re.compile(r'`(?:(?:struct|enum|union|class|typedef|@interface|@implementation|@protocol)\s+)?'
                    r'([A-Za-z_]\w*(?:(?:\.|::)[A-Za-z_]\w*)*:?)(?:\(\))?`')
COMMIT = re.compile(r'\b(?:(before|at)\s+)?(mesh|metal-microbench)\s+([0-9a-f]{7,40})(?![\w-])')
LOOSE = re.compile(r'(?:\b(at|before)\s+)?(?<![\w/.-])((?=[0-9a-f]*\d)(?=[0-9a-f]*[a-f])[0-9a-f]{7,40})(?![\w/-])')
PATHLIKE = re.compile(r'(?<![\w/.@+-])(?:[\w@+-][\w.@+-]*/)+[\w@+-][\w.@+-]*\.[A-Za-z]\w{0,7}(?![\w/])')
HEXLIKE = re.compile(r'\b(?=[0-9a-f]*\d)[0-9a-f]{7,40}\b')
LINK = re.compile(r'\]\([^)\s]*\)')
STATED = re.compile(r'\((\d{4}-\d\d-\d\d)[ T](\d\d:\d\d)')
Block = collections.namedtuple('Block', 'kind first last text heads label')


def git(root, *args, binary=False):
    done = subprocess.run(['git', '-C', root, '-c', 'core.quotePath=false', *args], capture_output=True, env=GIT)
    return done.stdout if binary else done.stdout.decode('utf-8', 'replace').strip()


def blobs(root, specs):
    if not specs:
        return {}
    out = subprocess.run(['git', '-C', root, 'cat-file', '--batch'], input=''.join(s + '\n' for s in specs).encode(),
                         capture_output=True, env=GIT).stdout
    found, at = {}, 0
    for spec in specs:
        end = out.find(b'\n', at)
        if end < 0:
            break
        parts, at = out[at:end].decode('utf-8', 'replace').rsplit(' ', 2), end + 1
        if len(parts) == 3 and parts[2].isdigit():
            size = int(parts[2])
            if parts[1] == 'blob':
                found[spec] = out[at:at + size].decode('utf-8', 'replace')
            at += size + 1
    return found


def now():
    return datetime.datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %z')


def local(stamp):
    try:
        return datetime.datetime.fromisoformat(stamp.replace('Z', '+00:00')).astimezone()
    except (AttributeError, ValueError):
        return None


def flat(text):
    for a, b in (('‘', "'"), ('’', "'"), ('“', '"'), ('”', '"'), ('–', '-'), ('—', '-'), ('−', '-'), ('\u00a0', ' ')):
        text = text.replace(a, b)
    text = re.sub(r'(?m)^[ \t]*(?:>[ \t]*)*(?://+|#+|\*+(?!\*)|-(?= ))?[ \t]?', '', text)
    text = re.sub(r'\b(?:ename|slink|pname|flink|sname|elink|basetype|dlink|tlink):', '', text)
    return re.sub(r'\s+', ' ', re.sub(r'[*_`]', '', text)).strip()


def short(text, n=100):
    return text if len(text) <= n else text[:n - 1].rstrip() + '…'


def stem(word):
    word = re.sub(r"['’]s$", '', word.lower())
    for ending in ENDINGS:
        if word.endswith(ending) and len(word) - len(ending) >= 3:
            return word[:-len(ending)]
    return word


def bare(word):
    return re.sub(r"['’]s$", '', word.lower())


def vocabulary(text):
    return {stem(p) for w in WORD.findall(text) for p in [w, *w.split('-')] if p}


def absent(word, quoted):
    return stem(word) not in quoted and not all(stem(p) in quoted for p in word.split('-') if p)


def content(text):
    return [w for w in WORD.findall(text) if len(bare(w)) >= 3 and bare(w) not in COMMON]


def kind_of(path):
    return next(k for p, k in KINDS if fnmatch.fnmatch(os.path.basename(path), p))


def quotations(text):
    marks = CODE.sub(lambda m: ' ' * len(m.group()), text).replace('“', '"').replace('”', '"')
    found = [m.start() for m in re.finditer('"', marks)]
    return [(a, b + 1, text[a + 1:b]) for a, b in zip(found[0::2], found[1::2])]


def fragments(quoted):
    return [f for f in (flat(p) for p in re.split(r'\[\s*(?:\.\.\.|…)\s*\]|…|\.\.\.', quoted)) if len(f) >= SHORTEST]


def hidden(text, everything=True):
    kept = list(text)
    spans = [m.span() for m in CODE.finditer(text)] if everything else []
    spans += [(a, b) for a, b, _ in quotations(text)]
    if everything:
        spans += [m.span() for pattern in (PATH, COMMIT, HEXLIKE, LINK) for m in pattern.finditer(text)]
    for a, b in spans:
        for i in range(a, b):
            if kept[i] != '\n':
                kept[i] = ' '
    return ''.join(kept)


def parse(text):
    lines, found, heads, labels, k = text.split('\n'), [], (), [], 0
    while k < len(lines):
        line = lines[k]
        fence, heading = FENCE.match(line), HEADING.match(line)
        if fence:
            end = next((j for j in range(k + 1, len(lines)) if lines[j].strip().startswith(fence.group(1))),
                       len(lines))
            found.append(Block('fence', k + 1, min(end + 1, len(lines)), '\n'.join(lines[k + 1:end]), heads, None))
            k, labels = end + 1, []
        elif heading:
            level = len(heading.group(1))
            heads = tuple(h for h in heads if h[0] < level) + ((level, heading.group(2), k + 1),)
            found.append(Block('heading', k + 1, k + 1, heading.group(2), heads, None))
            k, labels = k + 1, []
        elif not line.strip():
            k += 1
        elif line.lstrip().startswith('|'):
            found.append(Block('row', k + 1, k + 1, line, heads, None))
            k, labels = k + 1, []
        elif line.lstrip().startswith('>'):
            end = k
            while end < len(lines) and lines[end].lstrip().startswith('>'):
                end += 1
            body = '\n'.join(re.sub(r'^\s*>\s?', '', l) for l in lines[k:end])
            found.append(Block('quote', k + 1, end, body, heads, None))
            k, labels = end, []
        else:
            end = k + 1
            while end < len(lines) and lines[end].strip() and not (
                    LIST.match(lines[end]) or FENCE.match(lines[end]) or HEADING.match(lines[end]) or
                    lines[end].lstrip().startswith(('>', '|'))):
                end += 1
            item = LIST.match(line)
            if item:
                indent = len(item.group(1))
                labels = [(i, l) for i, l in labels if i < indent]
                own = LABEL.match(line)
                label = own.group(1).lower() if own else labels[-1][1] if labels else None
                labels.append((indent, label))
            else:
                label, labels = None, []
            found.append(Block('item' if item else 'paragraph', k + 1, end, '\n'.join(lines[k:end]), heads, label))
            k = end
    return found


def titles(block):
    return [h[1] for h in block.heads]


def apart(block):
    return any(t in SET_APART for t in titles(block))


def line_of(block, offset):
    return block.first + block.text[:offset].count('\n')


def sentences(block):
    shown = hidden(block.text)
    shown = re.sub(r'(?m)^(\s*(?:[-*+]|\d+[.)])\s+)(\*\*[^*\n]+\*\*[:,.]?)?', lambda m: ' ' * len(m.group()), shown)
    found, start = [], 0
    for m in re.finditer(r'(?<=[.!?])\s+(?=[A-Z(\[])|\Z', shown):
        piece = shown[start:m.start()]
        if piece.strip():
            lead = len(piece) - len(piece.lstrip())
            found.append((block.first + shown[:start + lead].count('\n'), piece.strip()))
        start = m.end()
    return found


def placed(lines, symbol, first, last):
    name = re.split(r'\.|::', symbol.rstrip(':'))[-1] + (':' if symbol.endswith(':') else '')
    word = re.compile(rf'(?<!\w){re.escape(name)}' + ('' if name.endswith(':') else r'(?!\w)'))
    if any(word.search(lines[k - 1]) for k in range(first, min(last, len(lines)) + 1)):
        return True, []
    escaped = re.escape(name.rstrip(':'))
    shapes = (rf'^\s*(?:async\s+)?(?:def|class)\s+{escaped}\b',
              rf'^(?:[A-Za-z_][\w\s*]*[\s*])?{escaped}\s*\((?!.*;\s*$)', rf'^[-+]\s*\([^)]*\)\s*{escaped}\b',
              rf'^(?:typedef\s+)?(?:struct|enum|union)\s+{escaped}\b',
              rf'^@(?:interface|implementation|protocol)\s+{escaped}\b')
    defined = [k for k, l in enumerate(lines, 1) if any(re.search(s, l) for s in shapes)]
    for d in defined:
        if d - 1 <= first <= d + 3 or d <= first and last <= ends(lines, d):
            return True, defined
    return False, defined or [k for k, l in enumerate(lines, 1) if word.search(l)]


def ends(lines, d):
    opening = lines[d - 1]
    if re.match(r'\s*(?:async\s+)?(?:def|class)\s', opening):
        indent, end = len(opening) - len(opening.lstrip()), d
        for k in range(d, len(lines)):
            if lines[k].strip() and len(lines[k]) - len(lines[k].lstrip()) <= indent:
                break
            end = k + 1
        return end
    return next((k for k in range(d + 1, len(lines) + 1) if lines[k - 1].startswith('}')), d)


def spans(numbers):
    numbers, out = sorted(numbers), []
    for n in numbers:
        if out and n == out[-1][1] + 1:
            out[-1][1] = n
        else:
            out.append([n, n])
    return ', '.join(str(a) if a == b else f'{a}–{b}' for a, b in out)


@functools.lru_cache(maxsize=None)
def spoken():
    turns, said = [], []
    for directory in TRANSCRIPTS:
        for path in sorted(glob.glob(os.path.join(directory, '*.jsonl'))):
            try:
                with open(path, 'rb') as f:
                    for line in f:
                        user = b'"type":"user"' in line and b'"tool_result"' not in line
                        if not user and not (b'"type":"assistant"' in line and b'"type":"text"' in line):
                            continue
                        try:
                            record = json.loads(line)
                        except ValueError:
                            continue
                        body = (record.get('message') or {}).get('content')
                        if record.get('type') == 'assistant' and isinstance(body, list):
                            said += [flat(b.get('text', '')) for b in body if isinstance(b, dict) and
                                     b.get('type') == 'text']
                        if record.get('type') != 'user' or record.get('isMeta') or record.get('isCompactSummary'):
                            continue
                        if isinstance(body, list):
                            body = '\n'.join(b.get('text', '') for b in body if isinstance(b, dict) and
                                             b.get('type') == 'text')
                        if isinstance(body, str) and body and not body.startswith('<task-notification'):
                            turns.append((local(record.get('timestamp')), flat(body)))
            except OSError:
                continue
    return turns, '\n'.join(t for _, t in turns), '\n'.join(said)


class Reading:
    def __init__(self, root, sha):
        self.root, self.sha = root, sha
        self.name = next((n for n, r in NAMES.items() if os.path.realpath(r) == os.path.realpath(root)), root)
        self.roots = {**NAMES, self.name: root}
        when = git(root, 'show', '-s', '--format=%ct', sha)
        self.heads = {n: sha if n == self.name else git(r, 'rev-list', '-1', f'--before={when}', 'HEAD') or
                      git(r, 'rev-parse', 'HEAD') for n, r in self.roots.items()}
        self.trees, self.texts, self.known = {}, {}, {}

    def commit(self, name, rev):
        if (name, rev) not in self.known:
            self.known[(name, rev)] = git(self.roots[name], 'rev-parse', '--verify', '--quiet', rev + '^{commit}')
        return self.known[(name, rev)]

    def tree(self, name, rev):
        if (name, rev) not in self.trees:
            listing = git(self.roots[name], 'ls-tree', '-r', '-z', '--name-only', rev, binary=True)
            paths = {p for p in listing.decode('utf-8', 'replace').split('\0') if p}
            bases = collections.defaultdict(list)
            for p in paths:
                bases[os.path.basename(p)].append(p)
            self.trees[(name, rev)] = paths, bases, {p.split('/')[0] for p in paths if '/' in p}
        return self.trees[(name, rev)]

    def read(self, name, rev, paths):
        wanted = [p for p in paths if (name, rev, p) not in self.texts and not UNREAD.search(p)]
        for spec, text in blobs(self.roots[name], [f'{rev}:{p}' for p in wanted]).items():
            self.texts[(name, rev, spec.split(':', 1)[1])] = text
        for p in wanted:
            self.texts.setdefault((name, rev, p), None)
        return {p: self.texts.get((name, rev, p)) for p in paths}

    def changes(self):
        listing = git(self.root, 'show', '--format=', '--name-status', '-M', '--diff-merges=first-parent', self.sha,
                      '--', 'design')
        return [(l.split('\t')[0], l.split('\t')[-1]) for l in listing.split('\n') if
                '\t' in l and DOCUMENT.match(l.split('\t')[-1])]

    def changed(self, path):
        patch = git(self.root, 'show', '--format=', '-U0', '--no-color', '--no-ext-diff', '--no-textconv', '-M',
                    '--diff-merges=first-parent', self.sha, '--', path)
        lines, number = set(), 0
        for line in patch.split('\n'):
            hunk = re.match(r'@@ -\d+(?:,\d+)? \+(\d+)', line)
            if hunk:
                number = int(hunk.group(1))
            elif line.startswith('+') and not line.startswith('+++') and number:
                lines.add(number)
                number += 1
        return lines

    @functools.cached_property
    def corpus(self):
        paths, _, _ = self.tree(self.name, self.sha)
        texts = self.read(self.name, self.sha, sorted(p for p in paths if DOCUMENT.match(p)))
        records = {p: flat(t) for p, t in texts.items() if t and (p == STANDARD or kind_of(p) == 'retrieval record')}
        others = {p: flat(t) for p, t in texts.items() if t and p not in records}
        code = []
        for name in self.roots:
            listed, _, _ = self.tree(name, self.heads[name])
            chosen = sorted(p for p in listed if p.endswith(SOURCES) and not UNREAD.search(p))
            code += [flat(t) for t in self.read(name, self.heads[name], chosen).values() if t]
        return records, others, '\n'.join(code)

    def where(self, part, path, own):
        turns, spoken_text, said = spoken()
        records, others, code = self.corpus
        if part in spoken_text or part in own:
            return 'turn'
        if any(part in t for p, t in records.items() if p != path) or part in code:
            return 'record'
        if part in said:
            return 'said'
        return next((p for p, t in others.items() if p != path and part in t), None)

    def comment(self, status, path):
        if status.startswith('D'):
            return {'path': path, 'status': status, 'deleted': True}
        text = self.read(self.name, self.sha, [path])[path] or ''
        kind, blocks = kind_of(path), parse(text)
        seen = {key: [] for key, _ in GROUPS}
        self.quotes(path, kind, blocks, seen)
        self.statements(kind, blocks, seen)
        cited = self.locations(kind, blocks, seen)
        if kind == 'change request':
            self.requirements(blocks, seen)
        return {'path': path, 'status': status, 'kind': kind, 'text': text, 'lines': len(text.rstrip('\n').split('\n')),
                'changed': self.changed(path), 'seen': seen, 'cited': cited, 'deleted': False}

    def quotes(self, path, kind, blocks, seen):
        if kind == 'retrieval record':
            return
        own = '\n'.join(flat(b.text) for b in blocks if b.kind == 'quote' and 'The turns' in titles(b))
        for k, b in enumerate(blocks):
            if b.kind in ('fence', 'heading'):
                continue
            if b.kind == 'quote' and 'The turns' in titles(b):
                self.turn(b, blocks[k + 1:k + 2], seen)
                continue
            for start, _, inner in [(0, 0, b.text)] if b.kind == 'quote' else quotations(b.text):
                for part in fragments(inner):
                    found = self.where(part, path, own)
                    if found in ('turn', 'record') or found == 'said' and kind != 'change request':
                        continue
                    said_by = ("; it is in Claude's own text in the transcripts" if found == 'said' else
                               f'; it is in {found}, a document Claude wrote' if found else '')
                    seen['quotations'].append((line_of(b, start), f'"{short(part)}" is in none of: the operator\'s '
                                               'turns in the mesh and metal-microbench transcripts, the retrieval '
                                               f'records, design/change-requests.md, the tracked sources{said_by}.'))

    def turn(self, block, after, seen):
        turns, spoken_text, said = spoken()
        stated = next((STATED.match(a.text.strip()) for a in after if a.kind == 'paragraph'), None)
        times = None
        for offset, line in enumerate(block.text.split('\n')):
            for part in fragments(line):
                if part in spoken_text:
                    at = {t for t, text in turns if t and part in text}
                    times = at if times is None else times & at or times | at
                    continue
                note = " It is in Claude's text in the transcripts." if part in said else ''
                seen['quotations'].append((block.first + offset, f'the quoted turn "{short(part)}" is not in any '
                                           'operator turn of the mesh and metal-microbench transcripts.'
                                           f'{note}'))
        if stated and times:
            given = datetime.datetime.strptime(f'{stated.group(1)} {stated.group(2)}', '%Y-%m-%d %H:%M')
            if not any(abs(t.replace(tzinfo=None) - given) <= datetime.timedelta(minutes=1) for t in times):
                when = ', '.join(sorted(t.strftime('%Y-%m-%d %H:%M') for t in times)[:3])
                seen['quotations'].append((block.first, f'the quoted turn is the operator\'s of {when} local in the '
                                           f'transcripts; the document gives {stated.group(1)} {stated.group(2)}.'))

    def statements(self, kind, blocks, seen):
        sections = collections.OrderedDict()
        for b in blocks:
            sections.setdefault(b.heads[-1] if b.heads else (0, '', 1), []).append(b)
        unquoted = []
        for key, members in sections.items():
            if apart(members[0]):
                continue
            quoted = set()
            for b in members:
                if b.kind in ('quote', 'fence'):
                    quoted |= vocabulary(b.text)
                elif b.kind in ('paragraph', 'item', 'row'):
                    for _, _, inner in quotations(b.text):
                        quoted |= vocabulary(inner)
            claims = []
            for b in members:
                if b.kind == 'heading' and key[0] >= 3:
                    claims.append((b.first, re.sub(r'^\d+\.\s*', '', hidden(b.text))))
                elif b.kind in ('paragraph', 'item', 'row') and b.label not in UNEXAMINED:
                    claims += sentences(b)
            claims = [(n, words) for n, s in claims if (words := content(s))]
            if not claims:
                continue
            if not quoted:
                unquoted.append(key)
                continue
            missing = collections.OrderedDict()
            for n, words in claims:
                missing.setdefault(n, {}).update((bare(w), 0) for w in words if absent(w, quoted))
            if kind == 'change request':
                for n, words in missing.items():
                    if words:
                        seen['statements'].append((n, f'in {self.place(key)}, words that no quotation of the '
                                                   f'{self.place(key).split()[0]} has: {", ".join(words)}.'))
                continue
            lines = [n for n, words in missing.items() if words]
            if lines:
                seen['statements'].append((lines[0], f'in {self.place(key)}, {len(lines)} of its {len(missing)} '
                                           'lines of statements have words that no quotation of the section has: '
                                           f'lines {spans(lines)}.'))
        if unquoted:
            listed = ', '.join(f'line {k[2]} (“{short(k[1], 60)}”)' for k in unquoted)
            seen['statements'].append((unquoted[0][2], f'no quotation in these sections, so their sentences stand on '
                                       f'none within them: {listed}.'))

    def place(self, key):
        number = re.match(r'(\d+)\.\s', key[1])
        return f'item {number.group(1)}' if number and key[0] >= 3 else f'section “{short(key[1], 50)}”'

    def locations(self, kind, blocks, seen):
        cited = {}
        for b in blocks:
            if b.kind == 'paragraph' and re.search(r'\bcited at\b', b.text):
                for m in COMMIT.finditer(b.text):
                    full = self.commit(m.group(2), m.group(3))
                    if full:
                        cited.setdefault(m.group(2), full)
                break
        history = []
        for b in blocks:
            if b.kind in ('fence', 'quote', 'heading') or 'The turns' in titles(b):
                continue
            text = hidden(b.text, everything=False)
            revs = self.named(b, text, cited, seen)
            bounds = [m.start() for m in re.finditer(r'\|', text)] if b.kind == 'row' else []
            scopes = list(zip([0] + [k + 1 for k in bounds], bounds + [len(text)]))
            mentioned = []
            for a, z in scopes:
                here = self.scope(kind, b, text, a, z, revs, [] if b.kind == 'row' else history, seen)
                mentioned += [h for h in here if h not in mentioned]
            if b.kind != 'row':
                history = [h for h in mentioned if isinstance(h, tuple)] + [h for h in history if h not in
                                                                             mentioned][:4]
        return cited

    def named(self, b, text, cited, seen):
        revs = {n: cited.get(n) or self.heads[n] for n in self.roots}
        named, before, last, taken = set(), False, 0, []
        for m in COMMIT.finditer(text):
            word, name, sha = m.groups()
            before = word == 'before' or word is None and before and text[last:m.start()].strip() == 'and'
            last = m.end()
            taken.append(m.span())
            full = self.commit(name, sha + ('^' if before else ''))
            if not full:
                seen['locations'].append((line_of(b, m.start()), f'names {name} {sha}, which is not a commit of '
                                          f'{name}.'))
            elif name not in named:
                revs[name], _ = full, named.add(name)
        taken += [m.span() for m in PATHLIKE.finditer(text)]
        for m in LOOSE.finditer(text):
            if any(a <= m.start(2) < z for a, z in taken):
                continue
            owners = [(n, self.commit(n, m.group(2) + ('^' if m.group(1) == 'before' else ''))) for n in self.roots]
            owners = [(n, full) for n, full in owners if full]
            if len(owners) == 1 and owners[0][0] not in named:
                revs[owners[0][0]], _ = owners[0][1], named.add(owners[0][0])
        return revs

    def scope(self, kind, b, text, a, z, revs, history, seen):
        found = [(m.start(), 'path', m) for m in PATH.finditer(text, a, z)]
        taken = [m.span() for _, _, m in found]
        found += [(m.start(), 'other', m) for m in PATHLIKE.finditer(text, a, z) if
                  not any(x <= m.start() < y for x, y in taken)]
        found += [(m.start(), 'bare', m) for m in BARE.finditer(text, a, z)]
        found += [(m.start(), 'symbol', m) for m in SYMBOL.finditer(text, a, z)]
        here, symbol, previous = [], None, None
        for at, sort, m in sorted(found, key=lambda e: e[0]):
            if sort == 'symbol':
                symbol = m.group(1), m.end()
                continue
            if sort == 'other':
                here.insert(0, 'external')
                symbol = None
                continue
            gap = text[symbol[1]:at] if symbol else ''
            name = (symbol[0] if symbol and len(gap) <= 40 and not re.search(r'[;,.]\s', gap) else
                    previous[1] if previous and len(text[previous[0]:at]) <= 8 and
                    not re.search(r'[);]', text[previous[0]:at]) else None)
            if sort == 'path':
                resolved = self.resolve(m.group(1), text, at, revs)
                if resolved == 'external' or resolved[0] == 'missing' and kind == 'retrieval record':
                    here.insert(0, 'external')
                elif resolved[0] == 'missing':
                    here.insert(0, 'external')
                    used = resolved[2]
                    places = [f'{n} at {used[n][:8]}' for n in resolved[1]]
                    gone = [f'{n} {g} deleted it' for n in resolved[1] if (g := git(
                        self.roots[n], 'log', '-1', '--format=%h', '--diff-filter=D', used[n], '--',
                        os.path.normpath(m.group(1))))]
                    seen['locations'].append((line_of(b, at), f'{m.group(1)} is ' + (
                        f'not in {places[0]}' if len(places) == 1 else f'in neither {" nor ".join(places)}') +
                        (f'; {", ".join(gone)}.' if gone else '.')))
                else:
                    here.insert(0, resolved)
                    if m.group(2):
                        self.check(b, at, [resolved], 1, int(m.group(2)), int(m.group(3) or m.group(2)), name,
                                   seen, m.group(1))
            else:
                self.check(b, at, here + [h for h in history if h not in here], len(here), int(m.group(1)),
                           int(m.group(2) or m.group(1)), name, seen)
            previous, symbol = (m.end(), name), None
        return here

    def resolve(self, given, text, at, revs):
        revs = revs if given.endswith(SOURCES) else self.heads
        before = text[max(0, at - 48):at]
        hint = re.search(r'\b(metal-microbench|mesh)\s+(?:(?:at\s+)?[0-9a-f]{7,40}\s+)?$', before)
        if not hint and re.search(r'[A-Z][\w.+-]*(?: [A-Z][\w.+-]*)*,\s*$', before):
            return 'external'
        names = [hint.group(1)] if hint else [self.name] + [n for n in self.roots if n != self.name]
        path = os.path.normpath(given)
        for name in names:
            paths, bases, _ = self.tree(name, revs[name])
            if path in paths:
                return name, revs[name], path
            if '/' not in path and len(bases.get(path, ())) == 1:
                return name, revs[name], bases[path][0]
        if hint or '/' in path and any(path.split('/')[0] in self.tree(n, revs[n])[2] for n in names):
            return 'missing', names, revs
        return 'external'

    def check(self, block, at, candidates, local, first, last, symbol, seen, given=None):
        if not candidates or candidates[0] == 'external':
            return
        holding, short_of, missing = [], [], []
        for k, candidate in enumerate(candidates):
            if candidate == 'external':
                continue
            name, rev, path = candidate
            text = self.read(name, rev, [path])[path]
            if text is None:
                continue
            lines, where = text.rstrip('\n').split('\n'), f'{path} at {name} {rev[:8]}'
            within = first <= last <= len(lines)
            if not symbol:
                if within:
                    return
                if given or k < local:
                    short_of.append(f'{where} has {len(lines)} lines')
                continue
            fits, places = placed(lines, symbol, first, last)
            if fits and within:
                return
            if places:
                holding.append(f'{where} has `{symbol}` at :' + ', :'.join(map(str, places[:4])))
            elif given or k < local:
                missing.append(f'{where} has no `{symbol}`')
        reasons = holding[:1] or missing or short_of
        if reasons:
            where = (f'{given}:' if given else '(:') + f'{first}' + (f'–{last}' if last != first else '') + (
                '' if given else ')')
            read = '' if given else f', read in the file{"s" if len(reasons) > 1 else ""} named before it'
            seen['locations'].append((line_of(block, at), (f'`{symbol}` ' if symbol else '') + f'{where}{read}: ' +
                                      '; '.join(reasons) + '.'))

    def requirements(self, blocks, seen):
        turns = []
        for k, b in enumerate(blocks):
            if b.kind == 'quote' and 'The turns' in titles(b):
                stated = next((STATED.match(a.text.strip()) for a in blocks[k + 1:k + 2] if a.kind == 'paragraph'),
                              None)
                turns.append((b, vocabulary(b.text), f'{stated.group(1)} {stated.group(2)}' if stated else ''))
        own = '\n'.join(flat(b.text) for b, _, _ in turns)
        items = collections.OrderedDict()
        for b in blocks:
            second, third = [h for h in b.heads if h[0] == 2], [h for h in b.heads if h[0] == 3]
            if second and second[0][1] == 'Requirements' and third:
                items.setdefault(third[0], []).append(b)
        for key, members in items.items():
            place = self.place(key)
            spoken_by = [(line_of(b, s), inner) for b in members if b.kind == 'item' and b.label == 'source' and
                         re.match(r'\s*[-*+]\s+the operator\b', b.text, re.I) for s, _, inner in quotations(b.text)]
            for n, inner in spoken_by:
                for part in fragments(inner):
                    if part not in own:
                        seen['requirements'].append((n, f'{place} quotes the operator\'s "{short(part)}", which is '
                                                     'not among the turns quoted in "## The turns".'))
            if spoken_by or not turns:
                continue
            asked = {}
            for b in members:
                if b.kind == 'heading' or b.label in ('write', 'written'):
                    for w in content(hidden(re.sub(r'^\d+\.\s*', '', b.text))):
                        asked.setdefault(stem(w), bare(w))
            best = max(turns, key=lambda t: len(asked.keys() & t[1]))
            shared = [asked[s] for s in asked if s in best[1]]
            if shared:
                said = f'the turn {"of " + best[2] + " " if best[2] else ""}at line {best[0].first}'
                seen['requirements'].append((key[2], f'{place} quotes no operator turn. Of the turns in "## The '
                                             f'turns", {said} shares the most of the words of its heading and its '
                                             f'Write: {", ".join(shared[:10])}.'))
            else:
                seen['requirements'].append((key[2], f'{place} quotes no operator turn, and no turn in "## The '
                                             'turns" shares a word with its heading or its Write.'))

    def model(self, path, text, observed):
        shims = [p for p in os.environ.get('PATH', '').split(os.pathsep) if p and 'cmux-cli-shims' not in p]
        claude = shutil.which('claude', path=os.pathsep.join(shims))
        if not claude:
            return None, 'no claude CLI on PATH'
        env = {k: v for k, v in os.environ.items() if not k.startswith(('CLAUDE', 'CMUX')) or k == 'CLAUDE_CONFIG_DIR'}
        env['PATH'] = os.pathsep.join(shims)
        standard = self.read(self.name, self.sha, [STANDARD])[STANDARD] or ''
        numbered = '\n'.join(f'{n:5} {l}' for n, l in enumerate(text.split('\n'), 1))
        given = (f'{STANDARD}:\n{standard}\n\nObservations already made by string checks:\n{observed or "none"}\n\n'
                 f'The document, {path}, with line numbers:\n{numbered}')
        try:
            done = subprocess.run([claude, *READER, '--system-prompt', PROMPT, 'Write your commentary on this '
                                   'document.'], input=given, capture_output=True, text=True,
                                  cwd=tempfile.gettempdir(), env=env, timeout=READING_TIME)
        except subprocess.TimeoutExpired:
            return None, f'the read ran past {READING_TIME} s'
        except OSError as error:
            return None, f'the read could not start: {error}'
        if done.returncode:
            return None, f'the read exited with status {done.returncode}: {done.stderr.strip()[:400]}'
        return done.stdout.strip(), None


def observations(part):
    return [(n, key, said) for key, _ in GROUPS for n, said in sorted(set(part['seen'][key]))]


def render(reading, parts, state):
    subject = git(reading.root, 'show', '-s', '--format=%ad%x00%s', '--date=format:%Y-%m-%d %H:%M %z', reading.sha)
    when, _, subject = subject.partition('\0')
    version = git(MESH, 'log', '-1', '--format=%h', '--', 'bin/mesh-commentary-review.py') or 'uncommitted'
    out = [f'# Commentary on {reading.name} {reading.sha[:10]}', '',
           f'- **Commit:** {reading.sha}, {when}: {subject}',
           f'- **Read:** {now()}, by mesh bin/mesh-commentary-review.py at {version} '
           '(mesh design/commentary-review.md)',
           '', 'Observations for the operator and the next session. Each points at a line of the document as this '
           'commit has it, and each is information, not a verdict: nothing reads it, and the work that made the commit '
           'goes on unaffected.', '']
    if not parts:
        out.append('The commit changes no document under design/.')
    for part in parts:
        out += [f'## {part["path"]}', '']
        if part['deleted']:
            out += ['The commit deletes this document.', '']
            continue
        changed = part['changed']
        span = ('the commit adds it' if part['status'].startswith('A') else
                f'the commit changes line{"s" if len(changed) > 1 else ""} {spans(changed)}' if changed else
                'the commit changes no line of it')
        cited = ' and '.join(f'{n} {s[:8]}' for n, s in part['cited'].items())
        out += [f'A {part["kind"]} of {part["lines"]} lines; {span}. Observations on lines the commit changed are '
                'marked "(this commit)".' + (f' Code is read at {cited}, as the document states, where a line names '
                                             'no other commit.' if cited else ''), '']
        for key, title in GROUPS:
            listed = sorted(set(part['seen'][key]))
            if key == 'requirements' and part['kind'] != 'change request':
                continue
            out += [f'### {title}', '']
            out += [f'- line {n}{" (this commit)" if n in changed else ""}: {said}' for n, said in listed] or [
                'No observation of this kind.']
            out.append('')
        if part['kind'] == 'change request':
            out += ['### Model commentary', '']
            if state == 'pending':
                out += ['Claude Sonnet is reading the document; its commentary replaces this line when it ends.', '']
            elif part.get('model'):
                fence = '`' * max(3, 1 + max((len(r) for r in re.findall(r'`+', part['model'])), default=0))
                out += ['Claude Sonnet (`claude -p --model sonnet`, no tools, no saved session) read the standard, the '
                        'document and the observations above. Its text follows verbatim. A model\'s reading varies '
                        'from run to run.', '', fence + 'text', part['model'], fence, '']
            else:
                out += [f'No model commentary: {part.get("failure") or "the read was not run"}.', '']
    return '\n'.join(out).rstrip('\n') + '\n'


def put(path, text):
    staged = f'{path}.{os.getpid()}.part'
    with open(staged, 'w', encoding='utf-8') as f:
        f.write(text)
    os.replace(staged, path)


def append(path, line, header=''):
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        if header and not os.fstat(fd).st_size:
            os.write(fd, header.encode())
        os.write(fd, line.encode())
    finally:
        os.close(fd)


def note(root, what):
    try:
        home = next((r for r in NAMES.values() if os.path.realpath(root or '/').startswith(r)), MESH)
        store = os.path.join(home, '.git', STORE)
        os.makedirs(store, exist_ok=True)
        append(os.path.join(store, 'log'), f'{now()} {what.rstrip()}\n')
    except BaseException:
        pass


def write(repo, sha, model=True):
    root = git(repo, 'rev-parse', '--show-toplevel')
    full = git(root, 'rev-parse', '--verify', '--quiet', sha + '^{commit}') if root else ''
    if not full:
        raise ValueError(f'{sha} is not a commit of {repo}')
    reading = Reading(root, full)
    store = os.path.join(git(root, 'rev-parse', '--path-format=absolute', '--git-common-dir'), STORE)
    os.makedirs(store, exist_ok=True)
    parts = [reading.comment(status, path) for status, path in reading.changes()]
    target = os.path.join(store, f'{reading.sha}.md')
    asks = [p for p in parts if not p['deleted'] and p['kind'] == 'change request']
    if parts:
        put(target, render(reading, parts, 'pending' if model and asks else 'done'))
    for part in asks if model else []:
        if len(part['text']) > LARGEST_READ:
            part['failure'] = f'the document is longer than {LARGEST_READ} characters'
            continue
        observed = '\n'.join(f'line {n}: {said}' for n, _, said in observations(part))
        part['model'], part['failure'] = reading.model(part['path'], part['text'], observed)
        if part['failure']:
            note(root, f'{reading.name} {reading.sha[:10]} {part["path"]}: the model read failed: {part["failure"]}')
    if parts:
        put(target, render(reading, parts, 'done'))
    count = sum(len(observations(p)) for p in parts if not p['deleted'])
    subject = git(root, 'show', '-s', '--format=%s', reading.sha)
    said = (f'{len(parts)} document{"s" if len(parts) != 1 else ""} under design/, {count} observation'
            f'{"s" if count != 1 else ""} → {reading.sha}.md' if parts else 'no document under design/')
    append(os.path.join(store, 'INDEX.md'), f'- {now()[:16]} {reading.sha[:10]}: {said}. “{short(subject, 90)}”\n',
           f'# Commentary on the commits of {reading.name}\n\nWritten after each commit by mesh '
           'bin/mesh-commentary-review.py (mesh design/commentary-review.md), one line per commit read, newest last. '
           'Each commit\'s observations are in the file its line names.\n\n')
    return target if parts else os.path.join(store, 'INDEX.md')


def detached():
    try:
        os.setpriority(os.PRIO_DARWIN_PROCESS, 0, os.PRIO_DARWIN_BG)
    except (AttributeError, OSError):
        os.nice(10)
    try:
        payload = json.loads(sys.stdin.buffer.read() or b'{}')
        spec = importlib.util.spec_from_file_location('mesh_c9_review', os.path.join(MESH, 'bin', 'mesh-c9-review.py'))
        c9 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(c9)
        made = c9.made(payload)
    except BaseException:
        note(MESH, 'the hook payload could not be read:\n' + traceback.format_exc())
        return 0
    for repo, sha in made:
        try:
            write(repo, sha)
        except BaseException:
            note(repo, f'the review of {sha} failed:\n' + traceback.format_exc())
    return 0


def hook():
    try:
        payload = sys.stdin.buffer.read()
        with tempfile.TemporaryFile() as spool:
            spool.write(payload)
            spool.flush()
            spool.seek(0)
            subprocess.Popen([sys.executable, '-B', os.path.abspath(__file__), '--detached'], stdin=spool,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd='/', start_new_session=True)
    except BaseException:
        note(MESH, 'the hook could not start the review:\n' + traceback.format_exc())
    return 0


def main():
    args = sys.argv[1:]
    if args[:1] == ['--detached']:
        return detached()
    if args[:1] == ['--commit'] and len(args) in (3, 4) and args[3:] in ([], ['--strings']):
        print(write(NAMES.get(args[1], os.path.abspath(args[1])), args[2], model=not args[3:]))
        return 0
    if args:
        print('mesh-commentary-review.py --commit <mesh|metal-microbench|path> <rev> [--strings]', file=sys.stderr)
        return 1
    return hook()


if __name__ == '__main__':
    sys.exit(main())
