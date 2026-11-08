import ast
import bisect
import concurrent.futures
import functools
import glob
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import tokenize

MESH = '/Users/mdot/dox/mesh'
MICROBENCH = '/Users/mdot/metal-microbench'
NAMES = {'mesh': MESH, 'metal-microbench': MICROBENCH}

_c9 = importlib.util.spec_from_file_location('mesh_c9_review', os.path.join(MESH, 'bin', 'mesh-c9-review.py'))
c9 = importlib.util.module_from_spec(_c9)
_c9.loader.exec_module(c9)

RULE = ("can we delete comments entirely where they're trying to assert a spec or something similar thanks. comments "
        "can cite an external document but anything beyond that is sounding really idiomatic.")
PROMPT = f"""You read source comments for one rule. The operator's words: "{RULE}"

So a comment passes only if it contains nothing but citations of external documents:
- a repository document path and its section or anchor: design/plan-to-bridge-change-request.md R4,
  design/joint-compilation.md §3, row 3, ../dox/mesh/design/algorithm-sources.md#native-metal-program;
- a short bibliographic reference: R. L. Graham, Bell System Technical Journal 45, 1966; C.-Y. Chen, arXiv 2205.02474,
  2022;
- an API header and line: MTL4ComputeCommandEncoder.h:109, or a library source path such as torch/export/__init__.py;
- a specification's URL.
Several citations may be joined by ";". Anything else in a comment is a finding:
- prose, explanation, rationale, contracts and specifications;
- quotations;
- dated revision notes, "What must stay" clauses, TODOs.

The comments come on stdin, one a line, numbered. Does each contain anything besides citations of external documents?
Answer one line for each, in order: `N: PASS`, or `N: FAIL` followed by its non-citation text, verbatim, cut at 25
words. Nothing else."""
QUESTION = 'Read these comments.'
BATCH = 40
READERS = 4

SOURCES = ('.c', '.h', '.m', '.mm', '.py', '.swift', '.metal', '.sh')
SKIPPED = re.compile(r'(?:^|/)(?:\.build|third_party|xonotic|output_data|vendor|node_modules|\.venv|venv|site-packages)/'
                     r'|(?:^|/)tools/quant_search/data_cache/|generated', re.I)
LICENSE = re.compile(r'SPDX-License-Identifier|Copyright\s*(?:\(c\)|©|\d{4})|Licensed under the|'
                     r'Permission is hereby granted|All rights reserved', re.I)
DIRECTIVE = {
    '#': re.compile(r'#+\s*(?:(?i:noqa)\b|type:|pragma\b|pylint:|fmt:|isort:|mypy:|pyright:|ruff:|nosec\b|yapf:|pyre-|'
                    r'shellcheck\s|-\*-|vim?:|flake8\b)'),
    'C': re.compile(r'/[/*][*!/<]?\s*(?:clang-format\b|NOLINT|swiftlint:|swift-format-ignore|swift-tools-version|'
                    r'MARK:|(?i:fallthrough)\b|NOTREACHED\b|ARGSUSED\b|-\*-|vim?:|@generated\b|IWYU pragma:|'
                    r'cppcheck-suppress)'),
}
MARKERS = ((re.compile(r'(?:^|[\s(;,])(?:19|20)\d\d-\d\d-\d\d\b'), 'a dated revision note'),
           (re.compile(r'\bWhat must stay\b', re.I), 'a "What must stay" clause'),
           (re.compile(r'\b(?:TODO|FIXME|XXX)\b'), 'a TODO'))

DASH = r'[-–—]'
NUMBER = r'\(?(?:[A-Z]?\d+(?:\.\d+)*(?:p\d+|[a-z])?|[A-Z](?:\.\d+)*|[IVXL]+)(?!\w)\)?'
SPAN = rf'{NUMBER}(?:\s*(?:{DASH}|to(?!\w))\s*{NUMBER})?'
LABEL = r'[A-Z]{1,3}(?:-?\d+)?(?:\.\d+)*[a-z]?(?!\w)'
EQUATION = r'\(\d+[a-z]?\)'
WORD = (r'(?i:appendix|appendices|app\.|annex|section|sections|sec\.|chapter|chapters|chap\.|ch\.|part|parts|item|items|'
        r'row|rows|column|columns|col\.|table|tables|tab\.|figure|figures|fig\.|figs\.|equation|equations|eq\.|eqs\.|'
        r'step|steps|line|lines|ll?\.|page|pages|pp?\.|paragraph|paragraphs|para\.|note|notes|footnote|footnotes|'
        r'theorem|theorems|thm\.|lemma|lemmas|corollary|cor\.|proposition|prop\.|definition|definitions|def\.|'
        r'algorithm|algorithms|alg\.|example|examples|exercise|exercises|listing|volume|vol\.|no\.|clause|clauses|'
        r'rule|rules|requirement|requirements|article|art\.|slide|slides|question|claim|remark|property|axiom|point|'
        r'points|problem|problems)(?!\w)')
LOCATOR = (rf'(?:[§¶]+\s*{SPAN}(?:\s*(?:,|and|&)\s*[§¶]*\s*{SPAN})*|(?:{WORD}\s*)+(?:{SPAN}(?:\s*(?:,|and|&)\s*{SPAN})*)?|'
           rf'(?:at\s+)?(?:mesh|metal-microbench)\s+(?P<sha>[0-9a-f]{{7,40}})(?!\w)|{LABEL}(?:\s*(?:{DASH}|to(?!\w))\s*{LABEL})?|'
           rf'{EQUATION}(?:\s*(?:{DASH}|to(?!\w))\s*{EQUATION})?|\d+(?:\.\d+)+(?:p\d+)?(?!\w))')
PATH = (r'(?:\.\.?/)*[\w@+-][\w.@+-]*(?:/[\w.@+-]+)*\.[A-Za-z][A-Za-z0-9]{0,7}|'
        r'[\w@+-][\w.@+-]*(?:/[\w.@+-]+)+\.\d')
WHERE = rf'(?:#[\w.:%~+-]+|:\d+(?:\s*{DASH}\s*\d+)?(?:\s*,\s*:?\d+(?:\s*{DASH}\s*\d+)?)*)?'
LINK = r'<?(?:https?://|doi:\s?)[^\s<>"]*[^\s<>".,;:)\]]>?'
TAIL = re.compile(rf'\s*(?:,?\s*(?:and|&)(?!\w)\s*|[,:]\s*|\(\s*)?(?:(?P<quote>"[^"]+"|“[^”]+”)|{LINK}|'
                  rf'(?P<also>{PATH}){WHERE}|{LOCATOR}|`(?P<code>[A-Za-z_][\w.:]*(?:\(\))?)`|'
                  r'(?P<symbol>(?:(?:struct|enum|union|class|typedef|@interface|@protocol)\s+)?(?:[-+](?=[a-z]))?'
                  r'[A-Za-z_][\w:]*(?:\(\))?(?!\w)))\s*\)?')
LOCATORS = re.compile(rf'{LOCATOR}(?:\s*(?:,|and(?!\w)|&)?\s*{LOCATOR})*'.replace('(?P<sha>', '(?:'))
STOP = (r'(?:The|This|That|These|Those|It|Its|Each|Every|All|Any|Some|Where|When|If|And|Or|But|Not|Is|Are|Was|Were|'
        r'Has|Have|Had|Uses|Use|Returns|Calls|Takes|Makes|Gives|Holds|Keeps|Reads|Writes|Here|There|Now|Only|Also|'
        r'Both|One|First|Last|Next|With|Without|From|For|In|On|At|By|To|As|After|Before|Under|Over|Since|Until)(?!\w)')
NAME = r'(?:[A-Z][A-Z0-9+-]+(?![a-z])|[A-Z]{2,}[a-z]\w*|[A-Z][a-z]+(?:[A-Z][\w+-]*)?)'
PROJECT = rf'(?:(?!{STOP}){NAME}(?:,?\s+{NAME})*|[a-z][a-z0-9]*(?:-[a-z0-9]+)+|[a-z][\w-]*(?=\s+v?\d))'
VERSION = r'v?\d+(?:\.\d+)+'
CITED = re.compile(rf'(?:(?P<project>{PROJECT})\s+)?(?P<version>{VERSION},?\s+)?(?P<path>{PATH}){WHERE}(?P<rest>.*)')
FOREIGN = re.compile(rf'(?:[A-Z][a-z]+[A-Z][\w+-]*|[A-Z]{{2,}}[\w+-]*|\w*\d\w*[A-Za-z]\w*|[a-z][a-z0-9]*(?:-[a-z0-9]+)+|'
                     rf'{VERSION})')
WORK = re.compile(rf'{PROJECT}(?:\s+{VERSION})?(?P<rest>,?\s*{LOCATOR}.*)'.replace('(?P<sha>', '(?:'))
PATHS = re.compile(rf'(?<![\w/.@+-])((?:\.\.?/)*[\w@+-][\w.@+-]*(?:/[\w.@+-]+)+\.[A-Za-z][A-Za-z0-9]{{0,7}})(?![\w/])')
URL = re.compile(rf'(?:{PROJECT},?\s+)?\(?{LINK}\)?(?P<rest>.*)')
STANDARD = re.compile(r'(?:RFC\s?\d+|ISO(?:/IEC)?\s?\d+(?:[-:]\d+)*|IEEE\s(?:Std\s)?\d+(?:\.\d+)*(?:-\d{4})?|PEP\s?\d+|'
                      r'ECMA-\d+|C(?:89|99|11|17|23)|C\+\+(?:11|14|17|20|23)|arXiv:?\s?\d{4}\.\d{4,5}(?:v\d+)?)'
                      r'(?![\w.])(?P<rest>.*)')
INITIALS = r'[A-Z][a-z]?\.(?:\s?-?\s?[A-Z][a-z]?\.)*'
SURNAME = r"(?:(?:van|von|de|der|den|du|da|di|del|della|la|le|dos|das|ter|ten|zu)\s+)*[A-Z][\w'’]*(?:-[A-Z][\w'’]*)?"
AUTHORS = re.compile(rf'(?:{INITIALS}\s?{SURNAME}(?:(?:\s*,\s*(?:and\s+)?|\s+and\s+|\s*&\s*){INITIALS}\s?{SURNAME})*'
                     rf'(?:,?\s+et\s+al\.?)?|{SURNAME}\s+et\s+al\.?)\s*,\s*')
ARXIV = re.compile(r'\d{4}\.\d{4,5}(?:v\d+)?')
VENUE = re.compile(r"[A-Z][\w'’.:+&/-]*|\(?\d+[A-Z]?\)?(?:\(\d+\))?(?:[:–-]\d+)?|arXiv:?|\d{4}\.\d{4,5}(?:v\d+)?|doi:\S+|of|on|"
                   r'and|for|in|the|to|a|an|de|des|du|der|für|und|et|la|le|thesis|dissertation|report|preprint|'
                   r'manuscript|draft|edition|ed\.|edn\.|eds\.|vol\.|no\.|pp?\.|tech\.|rep\.|proc\.|trans\.')
YEAR = re.compile(r'\(?(?:1[5-9]|20)\d\d[a-z]?\)?')
SHAPED = re.compile(r'[A-Za-z0-9]_|_[A-Za-z0-9]|[a-z][A-Z]|::|\(\)$')
KEYWORD = re.compile(r'(?:struct|enum|union|class|typedef|@interface|@protocol)\s+')
C_START = re.compile(r'//|/\*|(?<!\w)(?:u8|[uUL])?R"|"|\'')
SWIFT_START = re.compile(r'//|/\*|#+"|"')
C_END = {'"': re.compile(r'\\.|"|\n', re.S), "'": re.compile(r"\\.|'|\n", re.S)}
RAW = re.compile(r'#+"')
HEREDOC = re.compile(r'(?<!<)<<(?!<)-?\s*([\'"]?)([A-Za-z_]\w*)\1')


def run(*args):
    return subprocess.run(args, capture_output=True).stdout.decode('utf-8', 'replace')


def swift_string(text, s):
    n, hashes = len(text), 0
    while s + hashes < n and text[s + hashes] == '#':
        hashes += 1
    multi = text.startswith('"""', s + hashes)
    close = ('"""' if multi else '"') + '#' * hashes
    escape = '\\' + '#' * hashes
    j = s + hashes + (3 if multi else 1)
    while j < n:
        if text.startswith(escape, j):
            k = j + len(escape)
            j = swift_code(text, k + 1) if text.startswith('(', k) else k + 1
        elif text.startswith(close, j):
            return j + len(close)
        elif text[j] == '\n' and not multi:
            return j
        else:
            j += 1
    return n


def swift_code(text, j):
    n, depth = len(text), 1
    while j < n:
        if text[j] == '"' or RAW.match(text, j):
            j = swift_string(text, j)
            continue
        if text[j] == '(':
            depth += 1
        elif text[j] == ')':
            depth -= 1
            if not depth:
                return j + 1
        j += 1
    return n


def c_family(text, swift):
    found, n, pos = [], len(text), 0
    newlines = [k for k, c in enumerate(text) if c == '\n']

    def where(offset):
        k = bisect.bisect_left(newlines, offset)
        return k + 1, newlines[k - 1] + 1 if k else 0

    while True:
        m = (SWIFT_START if swift else C_START).search(text, pos)
        if not m:
            return found
        s, token = m.start(), m.group()
        if token in ('//', '/*'):
            e = s
            if token == '//':
                while True:
                    e = text.find('\n', e + 1)
                    if e < 0:
                        e = n
                        break
                    if swift or text[e - 1] != '\\':
                        break
            else:
                e, depth = s + 2, 1
                while e < n and depth:
                    if text.startswith('*/', e):
                        depth, e = depth - 1, e + 2
                    elif swift and text.startswith('/*', e):
                        depth, e = depth + 1, e + 2
                    else:
                        e += 1
            first, start = where(s)
            found.append((token, first, where(max(s, e - 1))[0], s - start, text[s:e], not text[start:s].strip()))
            pos = e
        elif swift:
            pos = swift_string(text, s)
        elif token.endswith('R"'):
            opened = text.find('(', m.end(), m.end() + 17)
            delimiter = text[m.end():opened] if opened >= 0 else None
            if delimiter is None or not re.fullmatch(r'[^\s()\\]*', delimiter):
                pos = m.end()
            else:
                closed = text.find(')' + delimiter + '"', opened)
                pos = n if closed < 0 else closed + len(delimiter) + 2
        elif token == "'" and re.search(r'(?<![\w.])\d[\w.]*$', text[max(0, s - 64):s]):
            pos = s + 1
        else:
            pos = s + 1
            while True:
                q = C_END[token].search(text, pos)
                if not q:
                    pos = n
                    break
                if q.group() == '\n':
                    pos = q.start()
                    break
                pos = q.end()
                if q.group() == token:
                    break


def python(text):
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, SyntaxError):
        return shell(text)
    documented = any(t.type == tokenize.NAME and t.string == '__doc__' for t in tokens)
    found, logical, first = [], [], True
    for t in tokens:
        if t.type == tokenize.COMMENT:
            row, col = t.start
            if not (row == 1 and t.string.startswith('#!')) and not (row <= 2 and re.match(r'#.*coding[:=]', t.string)):
                found.append(('#', row, row, col, t.string, not t.line[:col].strip()))
        elif t.type in (tokenize.NEWLINE, tokenize.ENDMARKER):
            if logical and all(x.type == tokenize.STRING for x in logical) and not (first and documented):
                found.append(('doc', logical[0].start[0], logical[-1].end[0], logical[0].start[1],
                              ' '.join(x.string for x in logical), False))
            first = first and not logical
            logical = []
        elif t.type not in (tokenize.NL, tokenize.INDENT, tokenize.DEDENT, tokenize.ENCODING):
            logical.append(t)
    return found


def shell(text):
    found, quote, pending = [], '', []
    for number, line in enumerate(text.split('\n'), 1):
        if pending:
            word, tabs = pending[0]
            if (line.lstrip('\t') if tabs else line).rstrip('\r') == word:
                pending.pop(0)
            continue
        if number == 1 and line.startswith('#!'):
            continue
        i, cut = 0, len(line)
        while i < len(line):
            c = line[i]
            if quote:
                if c == '\\' and quote != "'":
                    i += 2
                    continue
                if c == quote:
                    quote = ''
            elif c == '\\':
                i += 2
                continue
            elif c in '\'"`':
                quote = c
            elif c == '#' and (i == 0 or line[i - 1] in ' \t;|&()'):
                found.append(('#', number, number, i, line[i:], not line[:i].strip()))
                cut = i
                break
            i += 1
        pending = [(m.group(2), m.group(0).startswith('<<-')) for m in HEREDOC.finditer(line[:cut])]
    return found


def words(kind, raw):
    if kind == 'doc':
        try:
            value = ast.literal_eval(raw)
            raw = value.decode('utf-8', 'replace') if isinstance(value, bytes) else str(value)
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            pass
        lines = raw.split('\n')
    elif kind == '/*':
        body = re.sub(r'\*+/\Z', '', re.sub(r'\A/\*[*!]?', '', raw))
        lines = [re.sub(r'^\s*\*+(?=\s|$)', '', line) for line in body.split('\n')]
    elif kind == '//':
        lines = [re.sub(r'^\s*//+[!<]?', '', line) for line in raw.split('\n')]
    else:
        lines = [re.sub(r'^\s*#+', '', line) for line in raw.split('\n')]
    return [line for line in (re.sub(r'\s+', ' ', line).strip() for line in lines) if line]


def joined(lines):
    said = ''
    for line in lines:
        glued = re.search(r'\w[-/]\Z', said) and re.match(r'\w', line)
        said = line if not said else said + ('' if glued else ' ') + line
    return said


def comments(path, text):
    suffix = os.path.splitext(path)[1]
    found = python(text) if suffix == '.py' else shell(text) if suffix == '.sh' else c_family(text, suffix == '.swift')
    found = [c for c in found if c[0] == 'doc' or not DIRECTIVE['#' if c[0] == '#' else 'C'].match(c[4].strip())]
    merged = []
    for c in found:
        kind, first, last, col, raw, full = c
        p = merged[-1] if merged else None
        if p and kind in ('//', '#') and full and p[5] and p[0] == kind and p[3] == col and p[2] == first - 1:
            merged[-1] = (kind, p[1], last, col, p[4] + '\n' + raw, True)
        else:
            merged.append(c)
    said = [(first, last, words(kind, raw)) for kind, first, last, col, raw, full in merged]
    return [(first, last, joined(lines), lines) for first, last, lines in said
            if re.search(r'[^\W_]', joined(lines)) and not LICENSE.search(joined(lines))]


def split(text, mark=';'):
    parts, depth, quoted, start = [], 0, '', 0
    for i, c in enumerate(text):
        if quoted:
            quoted = '' if c == quoted else quoted
        elif c in '"“':
            quoted = '"' if c == '"' else '”'
        elif c in '([':
            depth += 1
        elif c in ')]':
            depth = max(0, depth - 1)
        elif c == mark and not depth:
            parts.append(text[start:i])
            start = i + 1
    return [p.strip() for p in parts + [text[start:]]]


def flat(text):
    for a, b in (('‘', "'"), ('’', "'"), ('“', '"'), ('”', '"'), ('–', '-'), ('—', '-')):
        text = text.replace(a, b)
    return re.sub(r'\s+', ' ', re.sub(r'[*_`]', '', text)).strip().lower()


@functools.lru_cache(maxsize=None)
def content(full):
    try:
        with open(full, encoding='utf-8', errors='replace') as f:
            return f.read()
    except OSError:
        return ''


@functools.lru_cache(maxsize=None)
def headings(full):
    found = []
    for line in content(full).split('\n'):
        label = re.match(r'\s*(?:[-*+]\s+|\d+\.\s+)?\*\*(.+?)\*\*', line)
        if line.startswith('#'):
            found.append(flat(line.lstrip('#')))
        elif label:
            found.append(flat(label.group(1)).rstrip('.:'))
    return tuple(found)


@functools.lru_cache(maxsize=None)
def tracked(root):
    names = {}
    for path in run('git', '-C', root, 'ls-files', '-z').split('\0'):
        if path and not SKIPPED.search(path):
            names.setdefault(os.path.basename(path), []).append(os.path.join(root, path))
    return names


@functools.lru_cache(maxsize=None)
def committed(root, sha):
    return run('git', '-C', root, 'cat-file', '-t', sha).strip() == 'commit'


def heading(full, quoted):
    pieces = [flat(p) for p in re.split(r'\.\.\.|…', quoted) if p.strip()]
    if not full or not full.endswith('.md') or sum(map(len, pieces)) < 4:
        return False
    for h in headings(full):
        at = 0
        for k, piece in enumerate(pieces):
            m = re.compile(rf'(?<!\w){re.escape(piece)}(?!\w)').search(h, at)
            if not m or not k and m.start() and len(piece) < 10:
                break
            at = m.end()
        else:
            return True
    return False


@functools.lru_cache(maxsize=None)
def stripped(full):
    text = content(full)
    if os.path.splitext(full)[1] in SOURCES:
        for c in sorted(python(text) if full.endswith('.py') else shell(text) if full.endswith('.sh') else
                        c_family(text, full.endswith('.swift')), key=lambda c: -len(c[4])):
            text = text.replace(c[4], ' ')
    return text


@functools.lru_cache(maxsize=None)
def sdk():
    try:
        path = run('xcrun', '--show-sdk-path').strip()
    except OSError:
        return ''
    return path if os.path.isdir(path) else ''


@functools.lru_cache(maxsize=None)
def header(path):
    if not sdk() or not path.endswith('.h'):
        return None
    found = [os.path.join(sdk(), 'usr', 'include', path)]
    found += glob.glob(os.path.join(glob.escape(sdk()), 'System', 'Library', 'Frameworks', '*.framework', 'Headers',
                                    glob.escape(path)))
    return next((f for f in found if os.path.isfile(f)), None)


def symbol(name, full, code=False):
    bare = KEYWORD.sub('', name, count=1)
    shaped = code or bare != name or re.match(r'[-+][a-z]', bare) or SHAPED.search(bare) or bare.endswith(':')
    bare = bare.lstrip('-+').removesuffix('()').split(':')[0]
    if not (full and os.path.isfile(full)):
        return bool(shaped)
    if full.endswith('.md'):
        return bool(shaped) and bare in content(full) or heading(full, name)
    return re.search(rf'(?<!\w){re.escape(bare)}(?!\w)', stripped(full)) is not None


def located(path, place):
    root, here, tree = place
    if '/' not in path:
        for base in (here, root):
            if os.path.isfile(os.path.join(base, path)):
                return False, os.path.join(base, path)
        for repository in dict.fromkeys((root, MESH, MICROBENCH)):
            if len(tracked(repository).get(path, ())) == 1:
                return False, tracked(repository)[path][0]
        return False, header(path)
    if path.startswith(('./', '../')):
        bases = (root, here, MESH, MICROBENCH)
    else:
        top = path.split('/')[0]
        bases = [b for b in dict.fromkeys((root, MESH, MICROBENCH)) if os.path.isdir(os.path.join(b, top))]
        if not bases:
            return False, header(path)
    for base in dict.fromkeys(bases):
        full = os.path.normpath(os.path.join(base, path))
        if os.path.exists(full) or tree and os.path.relpath(full, root) in tree:
            return True, full
    return True, None


def tail(rest, full, place):
    cites, i = [], 0
    while rest[i:].strip(' .,'):
        m = TAIL.match(rest, i)
        if not m or m.end() == i:
            return None
        if m.group('quote') and not heading(full, m.group('quote')[1:-1]):
            return None
        if m.group('symbol') and not symbol(m.group('symbol'), full):
            return None
        if m.group('code') and not symbol(m.group('code'), full, code=True):
            return None
        if m.group('sha') and not committed(MICROBENCH if 'metal-microbench' in m.group(0) else MESH, m.group('sha')):
            return None
        if m.group('also'):
            repository, full = located(m.group('also'), place)
            cites.append((m.group('also'), repository and not full))
        i = m.end()
    return cites


def bibliographic(part):
    m = AUTHORS.match(part)
    if not m:
        return False
    dated = numbered = pointed = False
    for segment in split(part[m.end():], ','):
        if not segment:
            return False
        if LOCATORS.fullmatch(segment):
            pointed = True
            continue
        tokens = segment.split()
        if not all(VENUE.fullmatch(t) for t in tokens):
            return False
        dated = dated or any(YEAR.fullmatch(t) or ARXIV.fullmatch(t) for t in tokens)
        numbered = numbered or any(t[:1].isdigit() for t in tokens)
    return dated or numbered and pointed


def citation(part, place):
    for form in (URL, STANDARD):
        m = form.fullmatch(part)
        cites = tail(m.group('rest'), None, place) if m else None
        if cites is not None:
            return cites
    if bibliographic(part):
        return []
    m = CITED.fullmatch(part)
    if m:
        abroad = m.group('version') or m.group('project') and named(m.group('project').split()[0].rstrip(','))
        repository, full = (False, header(m.group('path'))) if abroad else located(m.group('path'), place)
        cites = tail(m.group('rest'), full, place)
        if cites is not None:
            return [(m.group('path'), repository and not full)] + cites
    m = WORK.fullmatch(part)
    return tail(m.group('rest'), None, place) if m else None


def named(token):
    return token not in NAMES and bool(FOREIGN.fullmatch(token)) and not re.fullmatch(LABEL, token)


def foreign(text, start):
    token = re.search(r'(\S+) \Z', text[:start])
    return bool(token) and named(token.group(1))


def balanced(text):
    depth = 0
    for c in text:
        depth += (c in '([') - (c in ')]')
        if depth < 0:
            return False
    return depth == 0


def cited(part, place):
    while True:
        m = re.fullmatch(r'\((.*)\)|\[(.*)\]', part)
        inner = m and (m.group(1) if m.group(1) is not None else m.group(2))
        if not inner or not balanced(inner):
            break
        part = inner.strip()
    for candidate in (part, part[:-1].rstrip()) if part.endswith('.') else (part,):
        found = citation(candidate, place)
        if found is not None:
            return found
    return None


def missing(paths):
    return [f'cites {p}, which is in neither repository' for p in dict.fromkeys(paths)]


def only(text, place):
    cites = []
    for part in split(text):
        found = cited(part, place) if part else []
        if found is None:
            return None
        cites += found
    return cites


def citations(text, lines, place):
    cites = only(text, place)
    if cites is not None or len(lines) < 2:
        return cites
    cites, j = [], 0
    while j < len(lines):
        for k in range(min(len(lines), j + 4), j, -1):
            found = only(joined(lines[j:k]), place)
            if found is not None:
                cites, j = cites + found, k
                break
        else:
            return None
    return cites


def settle(text, lines, place):
    cites = citations(text, lines, place)
    if cites is not None:
        gone = missing(p for p, absent in cites if absent)
        return ('fail', '; '.join(gone)) if gone else ('pass', '')
    gone = missing(m.group(1) for m in PATHS.finditer(text)
                   if not foreign(text, m.start()) and (lambda r: r[0] and not r[1])(located(m.group(1), place)))
    unquoted = re.sub(r'"[^"]*"|“[^”]*”', '""', text)
    marks = [reason for pattern, reason in MARKERS if pattern.search(unquoted)]
    return 'fail' if marks else 'read', '; '.join(marks + gone)


def review(shown, text, changed, place):
    found = []
    for first, last, said, lines in comments(shown, text):
        if changed is None or any(k in changed for k in range(first, last + 1)):
            found.append([shown, first, said, *settle(said, lines, place)])
    return found


def added(root, sha):
    patch = run('git', '-C', root, '-c', 'core.quotePath=false', 'show', '--format=', '-U0', '--no-color', '--no-ext-diff',
                '--no-textconv', '-M', '--diff-merges=first-parent', '--src-prefix=a/', '--dst-prefix=b/', sha)
    lines, path, preamble, number = {}, None, False, 0
    for line in patch.split('\n'):
        if line.startswith('diff --git '):
            path, preamble = None, True
        elif preamble and line.startswith('+++ '):
            name = line[4:].rstrip('\t')
            if name.startswith('"'):
                name = ast.literal_eval(name).encode('latin-1').decode('utf-8', 'replace')
            path = name[2:] if name.startswith('b/') else None
            if path:
                lines.setdefault(path, set())
        elif line.startswith('@@ '):
            preamble = False
            m = re.match(r'@@ -\d+(?:,\d+)? \+(\d+)', line)
            number = int(m.group(1)) if m else 0
        elif path and not preamble and line.startswith('+'):
            lines[path].add(number)
            number += 1
        elif path and not preamble and line.startswith(' '):
            number += 1
    return lines


def commit(repo, sha):
    root = c9.git(repo, 'rev-parse', '--show-toplevel')
    tree = set(run('git', '-C', root, 'ls-tree', '-r', '--name-only', '-z', sha).split('\0'))
    found = []
    for path, changed in sorted(added(root, sha).items()):
        if changed and path.endswith(SOURCES) and not SKIPPED.search(path):
            blob = subprocess.run(['git', '-C', root, 'show', f'{sha}:{path}'], capture_output=True).stdout
            if b'\0' not in blob:
                place = (root, os.path.join(root, os.path.dirname(path)), tree)
                found += review(path, blob.decode('utf-8', 'replace'), changed, place)
    return found


def working(paths):
    found = []
    for given in paths:
        full = os.path.abspath(given)
        base = full if os.path.isdir(full) else os.path.dirname(full)
        root = c9.git(base, 'rev-parse', '--show-toplevel') or base
        if os.path.isdir(full):
            names = [os.path.join(root, p) for p in run('git', '-C', full, 'ls-files', '-z', '--full-name', '.').split('\0')
                     if p.endswith(SOURCES) and not SKIPPED.search(p)]
        else:
            names = [full] if full.endswith(SOURCES) else []
        for name in names:
            found += review(os.path.relpath(name), content(name), None, (root, os.path.dirname(name), None))
    return found


def ask(batch):
    given = '\n'.join(f'{k}: {said[:4000]}' for k, (_, said) in enumerate(batch, 1))
    try:
        read = subprocess.run([*c9.READER, '--system-prompt', PROMPT, QUESTION], input=given, capture_output=True,
                              text=True, cwd=tempfile.gettempdir(), timeout=300)
    except subprocess.TimeoutExpired:
        return {}, 'the reader timed out'
    except OSError as error:
        return {}, f'the reader failed: {error}'
    if read.returncode:
        return {}, f'the reader failed: {read.stderr.strip()[:500]}'
    verdicts = {}
    for m in re.finditer(r'^[^\w\n]*(\d+)[^\w\n]*(PASS|FAIL)\b[ \t:—–-]*(.*)$', read.stdout, re.M):
        k = int(m.group(1))
        if 1 <= k <= len(batch) and batch[k - 1][0] not in verdicts:
            verdicts[batch[k - 1][0]] = (m.group(2), m.group(3).strip())
    return verdicts, None


def reads(items):
    verdicts, failures = {}, []
    for _ in range(2):
        left = [item for item in items if item[0] not in verdicts]
        if not left or failures:
            break
        batches = [left[k:k + BATCH] for k in range(0, len(left), BATCH)]
        with concurrent.futures.ThreadPoolExecutor(READERS) as pool:
            for found, failure in pool.map(ask, batches):
                verdicts.update(found)
                if failure:
                    failures.append(failure)
    unread = len([item for item in items if item[0] not in verdicts])
    if unread and not failures:
        failures.append(f'the reader gave no verdict on {unread} comment(s)')
    return verdicts, failures


def short(text, n):
    return text if len(text) <= n else text[:n - 1].rstrip() + '…'


def report(groups):
    undecided = [((g, k), f[2]) for g, (_, found) in enumerate(groups) for k, f in enumerate(found) if f[3] == 'read']
    verdicts, failures = reads(undecided) if undecided else ({}, [])
    reports = []
    for g, (label, found) in enumerate(groups):
        lines = []
        for k, (shown, first, said, verdict, reason) in enumerate(found):
            read = verdicts.get((g, k))
            if verdict == 'read' and read:
                if read[0] == 'FAIL':
                    said_by = f'not a citation: "{short(read[1], 160)}"' if read[1] else 'not a citation'
                    reason = '; '.join(r for r in (said_by, reason) if r)
                verdict = 'fail' if reason else 'pass'
            if verdict == 'fail':
                lines.append(f'{shown}:{first}: {short(said, 100)} — {reason}')
        if lines:
            reports.append(f'Comment review of {label} (a comment may only cite external documents; what the '
                           'citation grammar cannot settle is read by Claude Haiku):\n' + '\n'.join(lines))
    return reports, failures


def main():
    args = sys.argv[1:]
    if args[:1] == ['--commit']:
        if len(args) != 3:
            print('mesh-comment-review.py --commit <repo> <rev>, or mesh-comment-review.py <paths...>', file=sys.stderr)
            return 1
        repo = NAMES.get(args[1], os.path.abspath(args[1]))
        sha = c9.git(repo, 'rev-parse', '--verify', '--quiet', args[2] + '^{commit}')
        if not sha:
            print(f'{args[2]}: no such commit in {repo}', file=sys.stderr)
            return 1
        groups = [(f'{sha[:10]} in {repo}', commit(repo, sha))]
    elif args:
        groups = [('the working tree', working(args))]
    else:
        groups = [(f'{sha[:10]} in {repo}', commit(repo, sha)) for repo, sha in c9.made(json.load(sys.stdin))]
    reports, failures = report(groups)
    said = reports + (['Comment review: ' + '; '.join(dict.fromkeys(failures))] if failures else [])
    if said:
        print('\n\n'.join(said), file=sys.stderr)
    return 2 if reports else 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
