#!/usr/bin/env python3
"""Waits, and what is shaped like them, in the kernel sources the page-effects compiler wrote around each numerical
kernel (metal-page-effects.m MetalPageEffectsSource; torch_mesh/page_effects.py _source_record writes each distinct one
to MESH_PREPARE_RECORD's directory as sources-<pid>/NNN-<kind>-<function>.metal). It prints a description for people
revising that compiler; it judges, rejects and changes nothing.

    python3 extras/wait_report.py DIRECTORY... [--member NAME] [--no-compile]

A DIRECTORY is a run's (each <member>-prepare/ beneath it) or one member's. For each member, each source:
  - its bytes; the mesh_step_observe calls of its live text (the text through the Metal preprocessor, so that only the
    kernel definition its own #if keeps is read); the words they wait on and the waits on each word in one prologue (a
    landed count only grows, so one wait at a word's greatest threshold implies the others: the waits past one a word
    are `extra`); threshold tables named by more than one wait;
  - the addressing of its kernel body: each load and store's index expression, with the integer divisions and modulos
    it needs (its own, those of the variables it is computed from and of the index helpers it calls: mesh_storage_N,
    floor_divide), the mesh_storage_ wrapper calls, and the system-coherent accesses (coherent(system) operands,
    mesh_payload and mesh_step_read loads), those in a loop counted apart;
  - its compiled form: the source compiled after the library's prelude (`xcrun metal -S`, AIR as LLVM text), and for
    its kernel, as if every call were inlined: instructions, spin loops (a loop whose exit depends on a coherent or
    atomic load of an address the loop does not change), fences, barriers, coherent and atomic loads, atomic updates and
    integer divisions, those in loops counted apart.
The prelude is read from this checkout, as layout.pipeline assembles it (torch_mesh/native.py command_records): the
record holds no layout declarations, so every binding compiles as the identity layout (LAYOUT below).
"""
import argparse
import codecs
import collections
import os
import re
import subprocess
import sys
from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'rdma'))
from mesh_graph import ancestors, descendants, preorder_parents  # noqa: E402  (the one graph module, design/rubrics.md C9)

SOURCE = re.compile(r'(\d+)-([^-]+)-(.+)\.metal$')
MARK = '#pragma mesh_wait_report_source'
# The record holds no layout declarations (layout.py pipeline: index_source or index_constants, made from each command's
# views at run time), so every binding compiles here as the identity layout: metal-recording.m's mesh_index template,
# no binding affine. The divisions counted are the source's own, never a layout's.
LAYOUT = ('template<uint B> constexpr bool mesh_affine() { return false; }\n'
          'template<uint B> inline long mesh_position(ulong i) { return long(i); }\n')


# ---- the text a source is compiled after, and the compiler

STRING = re.compile(r'\s*@?"((?:[^"\\\n]|\\.)*)"')


def literal(text, marker):
    """The string literal after the regular expression `marker` in C or Objective-C text, its adjacent pieces joined."""
    at, pieces = re.search(marker, text).end(), []
    while (m := STRING.match(text, at)):
        pieces.append(codecs.escape_decode(m.group(1).encode())[0].decode())
        at = m.end()
    return ''.join(pieces)


def prelude():
    """nccl-mesh-metal.m metal_prelude (metal-capabilities.h MetalCompilationPrelude, rdma/mesh.metal, nccl-mesh-kernels.h
    STEPS, these two through the C preprocessor), metal-recording.m MetalViewIndexPrelude, then LAYOUT."""
    rdma = ROOT / 'rdma'
    expanded = subprocess.run(['xcrun', 'clang', '-E', '-P', '-x', 'objective-c', '-I', str(rdma), '-'],
                              input='#import "nccl-mesh-kernels.h"\n#include "metal-capabilities.h"\n',
                              capture_output=True, text=True, check=True).stdout
    recording = (rdma / 'metal-recording.m').read_text()
    return (literal(expanded, r'MetalCompilationPrelude\(void\)\s*\{\s*return') + (rdma / 'mesh.metal').read_text()
            + '\n' + literal(expanded, r'NSString \*const STEPS =')
            + literal(recording, r'MetalViewIndexPrelude\(void\)\s*\{\s*return \[')
            + literal(recording, r'MetalViewIndexTemplates\(void\)\s*\{\s*return') + LAYOUT + '#endif\n')


class Compiler:
    """The Metal toolchain the Makefile compiles with (rdma/Makefile XCODE: DEVELOPER_DIR's, else the last
    /Applications/Xcode*.app), at etc/compute/metal.mk's language."""
    def __init__(self):
        found = sorted(Path('/Applications').glob('Xcode*.app'))
        developer = os.environ.get('DEVELOPER_DIR') or (str(found[-1] / 'Contents/Developer') if found else None)
        language = re.search(r'^MESH_METAL_LANGUAGE\s*\??=\s*(\S+)', (ROOT / 'etc/compute/metal.mk').read_text(), re.M)
        self.env = dict(os.environ, DEVELOPER_DIR=developer or '', SDKROOT='macosx')
        self.command = ['xcrun', '-sdk', 'macosx', 'metal', f'-std=metal{language.group(1)}', '-w', '-x', 'metal', '-',
                        '-o', '-']
        self.head = prelude() + MARK + '\n'
        rev = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True)
        self.describe = (f"{' '.join(self.command[3:5])} -E (live text) and -S (AIR), {developer}; prelude of "
                         f"{ROOT} {rev.stdout.strip()}; layouts the identity")

    def run(self, text):
        """(live text, AIR, error) of a source: what follows MARK through the preprocessor, and the module."""
        outputs = []
        for flag in ('-E', '-S'):
            done = subprocess.run(self.command + [flag] + (['-P'] if flag == '-E' else []), input=self.head + text,
                                  capture_output=True, text=True, env=self.env)
            if done.returncode:
                error = next((line for line in done.stderr.splitlines() if 'error' in line), done.stderr.strip())
                return (outputs[0] if outputs else None), None, re.sub(r'^<stdin>:', 'line ', error)[:160]
            outputs.append(done.stdout.split(MARK, 1)[1] if flag == '-E' else done.stdout)
        return outputs[0], outputs[1], None


# ---- reading source text

def masked(text):
    """text with its comments and quoted literals blanked, the same length, so that brackets and operators are code."""
    return re.sub(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'',
                  lambda m: re.sub(r'\S', ' ', m.group()), text, flags=re.S)


def closing(plain, at):
    """The index past the bracket that closes the one at `at`."""
    depth = 0
    for i in range(at, len(plain)):
        if plain[i] in '([{':
            depth += 1
        elif plain[i] in ')]}':
            depth -= 1
            if not depth:
                return i + 1
    return len(plain)


def arguments(text, plain, at):
    """The argument texts of the call or list whose bracket is at `at`, and the index past its close."""
    end, out, depth, start = closing(plain, at), [], 0, at + 1
    for i in range(at + 1, end - 1):
        if plain[i] in '([{':
            depth += 1
        elif plain[i] in ')]}':
            depth -= 1
        elif plain[i] == ',' and not depth:
            out.append(text[start:i].strip())
            start = i + 1
    out.append(text[start:end - 1].strip())
    return out, end


LOOPS = ('for', 'while', 'do')


def opener(plain, at):
    """The keyword whose body opens at `at` (for, while, do, if, else, switch; a function's or block's: its name or '')."""
    j = at - 1
    while j >= 0 and plain[j].isspace():
        j -= 1
    if j >= 0 and plain[j] == ')':
        depth = 0
        while j >= 0:
            depth += plain[j] == ')'
            depth -= plain[j] == '('
            if not depth:
                break
            j -= 1
        j -= 1
    m = re.search(r'(\w+)\s*$', plain[max(0, j - 60):j + 1])
    return m.group(1) if m else ''


class Scopes:
    """The scopes of a text, in order of their opening: each braced block and each loop's unbraced statement, its span,
    the keyword it is the body of, its parent (mesh_graph preorder_parents) and whether a loop holds it."""
    def __init__(self, plain):
        found, open_ = [], []
        for m in re.finditer(r'[{}]', plain):
            if m.group() == '{':
                open_.append(len(found))
                found.append([m.start(), len(plain), opener(plain, m.start())])
            elif open_:
                found[open_.pop()][1] = m.end()
        for m in re.finditer(r'\b(for|while)\s*\(', plain):
            end = closing(plain, m.end() - 1)
            rest = plain[end:end + 200].lstrip()
            if rest and rest[0] not in '{;':
                found.append([end, plain.find(';', end) + 1 or len(plain), m.group(1)])
        found.sort(key=lambda s: (s[0], -s[1]))
        depths, held = [], []
        for first, end, _ in found:
            while held and held[-1] <= first:
                held.pop()
            depths.append(len(held))
            held.append(end)
        self.spans, self.opens = found, [s[0] for s in found]
        self.parents = preorder_parents(depths)
        self.looped = []
        for k, (_, _, keyword) in enumerate(found):
            parent = self.parents[k]
            self.looped.append(keyword in LOOPS or (parent is not None and self.looped[parent]))

    def at(self, position):
        """The innermost scope holding position, or None."""
        k = bisect_right(self.opens, position) - 1
        while k is not None and k >= 0 and not (self.spans[k][0] < position < self.spans[k][1]):
            k = self.parents[k]
        return None if k is None or k < 0 else k

    def outermost(self, position):
        k = self.at(position)
        while k is not None and self.parents[k] is not None:
            k = self.parents[k]
        return k

    def in_loop(self, position):
        k = self.at(position)
        return k is not None and self.looped[k]


TYPES = r'(?:auto|bool|char|uchar|short|ushort|int|uint|long|ulong|size_t|u?int(?:8|16|32|64)_t)'
DEFINITION = re.compile(rf'\b{TYPES}\s+(\w+)\s*=\s*([^;{{}}]*);')
HELPER = re.compile(r'\binline\s+(?:long|ulong|int|uint)\s+(\w+)\s*\(([^()]*)\)\s*\{([^{}]*)\}')
DIVISION = re.compile(r'\bfloor_divide\s*\(|/(?!=)|%(?!=)')
CALLED = re.compile(r'\b(\w+)\s*(?:<[^<>()]*>)?\s*\(')
NAMED = re.compile(r'\b([A-Za-z_]\w*)\b(?!\s*(?:<[^<>()]*>)?\s*\()')
PUBLICATIONS = ('mesh_graph_coverage', 'mesh_graph_settle', 'mesh_publish', 'mesh_step_release', 'mesh_payload_store',
                'mesh_step_write')


def source_counts(text, function):
    """A source's waits and addressing, from its live text (or its raw text where that could not be had)."""
    plain = masked(text)
    scopes = Scopes(plain)
    out = collections.Counter()
    # the waits: mesh_step_observe(word, threshold, violation, polls, channel, entry), metal-page-effects.m
    prologues = collections.defaultdict(list)
    for m in re.finditer(r'\bmesh_step_observe\s*\(', plain):
        args, _ = arguments(text, plain, m.end() - 1)
        if len(args) == 6:
            prologues[scopes.outermost(m.start())].append(args)
    words, tables, per_word = set(), 0, {}
    for k, waits in prologues.items():
        same = collections.Counter(re.sub(r'\s+', '', a[0]) for a in waits)
        words.update(same)
        for word, n in same.items():
            per_word[word] = max(per_word.get(word, 0), n)
        out['waits'] += len(waits)
        out['extra'] += len(waits) - len(same)
        out['guarded'] += sum('mesh_effect_group' in a[1] for a in waits)
        named = collections.Counter(re.sub(r'\s+', '', re.sub(r'\[[^\[\]]*\]$', '', a[1])) for a in waits)
        tables += sum(1 for n in named.values() if n > 1)
    out['words'], out['tables'], out['prologues'] = len(words), tables, len(prologues)
    out['most'] = max(per_word.values(), default=0)
    entry = re.search(rf'\bkernel\s+void\s+{re.escape(function)}\s*\(', plain)
    if not entry:
        return out, per_word, None
    parameters, end = arguments(text, plain, entry.end() - 1)
    pointers = {}
    for p in parameters:
        name = re.search(r'(\w+)\s*(?:\[\[.*\]\])?\s*$', p)
        if '*' in p and name and name.group(1) != 'mesh_effects':
            pointers[name.group(1)] = 'coherent(system)' in p
    first = plain.find('{', end)
    last = closing(plain, first)
    body, at = plain[first:last], first
    # the index expressions: each subscript of a pointer operand, the divisions its value needs
    helpers = {m.group(1): m.group(3) for m in HELPER.finditer(plain, 0, first)}
    defined = {m.group(1): m.group(2) for m in DEFINITION.finditer(body)}

    def refs(expression):
        return ([('h', h) for h in CALLED.findall(expression) if h in helpers]
                + [('v', v) for v in NAMED.findall(expression) if v in defined])

    def inputs(node):
        return refs(helpers[node[1]] if node[0] == 'h' else defined[node[1]])

    def divisions(node):
        return len(DIVISION.findall(helpers[node[1]] if node[0] == 'h' else defined[node[1]]))

    known, needed = {}, []
    for m in re.finditer(r'\b(\w+)\s*\[', body):
        if m.group(1) not in pointers:
            continue
        index, _ = arguments(body, body, m.end() - 1)
        expression = index[0]
        nodes = set(refs(expression))
        for node in list(nodes):
            nodes |= ancestors(inputs, node, known)
        needed.append(len(DIVISION.findall(expression)) + sum(divisions(n) for n in nodes))
        out['accesses'] += 1
        if pointers[m.group(1)]:
            out['coherent'] += 1
            out['coherent loop'] += scopes.in_loop(at + m.start())
    out['divisions'] = sum(needed)
    out['most divisions'] = max(needed, default=0)
    out['wrappers'] = len(re.findall(r'\bmesh_storage_\d+\s*\(', body))
    for m in re.finditer(r'\b(mesh_payload|mesh_step_read)\s*(?:<[^<>()]*>)?\s*\(', body):
        tail = body[closing(body, body.index('(', m.end() - 1)):][:8]
        if m.group(1) == 'mesh_payload' and re.match(r'\s*!=\s*0', tail):
            out['predicates'] += 1
            continue
        out['coherent'] += 1
        out['coherent loop'] += scopes.in_loop(at + m.start())
    out['publications'] = len(re.findall(rf"\b(?:{'|'.join(PUBLICATIONS)})\s*(?:<[^<>()]*>)?\s*\(", body))
    return out, per_word, len(body)


# ---- reading AIR

DEFINE = re.compile(r'^define\b[^@]*@"?([^"(\s]+)"?\(')
CALLEE = re.compile(r'\bcall\b[^@]*@"?([^"(\s]+)"?\(')
TARGET = re.compile(r'label %([\w.$-]+)')
VALUE = re.compile(r'%([\w.$-]+)')
DIVIDE = re.compile(r'= (udiv|sdiv|urem|srem) (?:exact )?(?:<\d+ x )?i(\d+)[^,]*,\s*(\S+)')
UNCOUNTED = ('llvm.lifetime', 'llvm.dbg', 'llvm.experimental.noalias')


def functions(air):
    """{name: [[block, [instruction, ...]], ...]} of each function the module defines."""
    out, name = {}, None
    for line in air.splitlines():
        if name is None:
            m = DEFINE.match(line)
            if m:
                name, blocks = m.group(1), [['', []]]
            continue
        if line.startswith('}'):
            out[name] = [b for b in blocks if b[0] or b[1]]
            name = None
            continue
        m = re.match(r'^([\w.$-]+):', line)
        if m:
            blocks.append([m.group(1), []])
        elif line.strip() and not line.strip().startswith(';'):
            blocks[-1][1].append(line.strip())
    return out


def classify(line, callee):
    if callee:
        if callee.startswith('air.load.system_coherent'):
            return 'coherent loads'
        if re.match(r'air\.atomic\.\w+\.load', callee):
            return 'atomic loads'
        if callee.startswith('air.atomic.fence'):
            return 'fences'
        if 'barrier' in callee:
            return 'barriers'
        if re.match(r'air\.atomic\.\w+\.(xchg|cmpxchg|add|sub|max|min|and|or|xor)', callee):
            return 'atomic updates'
        if callee.startswith('air.store.system_coherent'):
            return 'coherent stores'
    elif re.match(r'(%\S+ = )?load volatile', line):
        return 'coherent loads'
    return None


def function_counts(blocks, defined):
    """One function's own counts (each also as 'loop <name>' where a loop holds it), its spin loops, and its calls:
    [(callee, whether a loop holds the call)]. A block lies in a loop where it reaches itself (mesh_graph ancestors)."""
    succ = {b: sorted({t for line in lines for t in TARGET.findall(line)}) for b, lines in blocks}
    reach = {}
    looped = {b for b, _ in blocks if b in ancestors(lambda v: succ.get(v, ()), b, reach)}
    defs, place = {}, {}
    for b, lines in blocks:
        for line in lines:
            m = re.match(r'%([\w.$-]+) = (.*)', line)
            if m:
                defs[m.group(1)], place[m.group(1)] = m.group(2), b
    own, calls, loads, ssa = collections.Counter(), [], [], {}
    for b, lines in blocks:
        for line in lines:
            if re.match(r'(i\d+ |\])', line):
                continue  # a switch's case lines
            callee = CALLEE.search(line)
            callee = callee.group(1) if callee else None
            if callee and callee.startswith(UNCOUNTED):
                continue
            names = [('instructions', 1)]
            what = classify(line, callee)
            if what:
                names.append((what, 1))
                if what in ('coherent loads', 'atomic loads'):
                    m = re.match(r'%([\w.$-]+) = ', line)
                    address = VALUE.findall(line[line.index('(') + 1:] if callee else line.split(',', 1)[-1])
                    if m and address:
                        loads.append((m.group(1), address[0], b))
            d = DIVIDE.search(line)
            if d:
                names += [('divisions', 1), ('64-bit divisions', d.group(2) == '64'),
                          ('variable divisions', not re.fullmatch(r'-?\d+', d.group(3)))]
            for name, n in names:
                own[name] += n
                own['loop ' + name] += n * (b in looped)
            if callee and callee in defined:
                calls.append((callee, b in looped))
    # a spin loop: a loop holding a coherent or atomic load of an address made outside it, on which one of its exits
    # depends (mesh.metal mesh_wait_bounded's; a scan of cells whose address moves, mesh_graph_prefix's, is none)
    def operands(v):
        return [w for w in VALUE.findall(defs.get(v, '')) if w in defs]

    spins = set()
    for value, address, b in loads:
        loop = {v for v in reach[b] if b in reach.get(v, ())} if b in looped else set()
        if not loop or place.get(address) in loop:
            continue
        for x, lines in blocks:
            if x not in loop or not lines:
                continue
            exit_ = re.match(r'br i1 %([\w.$-]+), label %([\w.$-]+), label %([\w.$-]+)', lines[-1])
            if exit_ and ({exit_.group(2), exit_.group(3)} - loop):
                if value == exit_.group(1) or value in ancestors(operands, exit_.group(1), ssa):
                    spins.add(min(loop))
                    break
    own['spins'] = len(spins)
    return own, calls



def kernel_counts(air, function):
    """The kernel's counts as if each call were inlined: its own and those of every callee once a call path
    (mesh_graph descendants over call sites), a callee's whole count a loop's where a loop holds a call on its path."""
    defined = functions(air)
    if function not in defined:
        return None
    each = {name: function_counts(blocks, defined) for name, blocks in defined.items()}
    total = collections.Counter(each[function][0])
    for name, looped in descendants((function, False), lambda v: [(c, v[1] or held) for c, held in each[v[0]][1]]):
        own = each[name][0]
        for key, n in own.items():
            if not key.startswith('loop '):
                total[key] += n
                total['loop ' + key] += n if looped else own['loop ' + key]
    return total


# ---- records

class Source:
    def __init__(self, path):
        m = SOURCE.match(path.name)
        self.path, self.index, self.kind, self.function = path, m.group(1), m.group(2), m.group(3)
        self.text = path.read_text()
        self.bytes = len(self.text.encode())


def records(path, member=None):
    """Each member's record beneath path: (member, its sources in index order, its parts file or None)."""
    path = Path(path)
    found = []
    for directory in sorted([path] if path.name.startswith('sources-') else path.rglob('sources-*')):
        if not directory.is_dir():
            continue
        name = directory.parent.name.removesuffix('-prepare')
        if member and name != member:
            continue
        parts = directory.parent / f"parts-{directory.name.split('-', 1)[1]}.jsonl"
        sources = sorted((Source(f) for f in directory.iterdir() if SOURCE.match(f.name)), key=lambda s: s.index)
        found.append((name, directory, sources, parts if parts.exists() else None))
    return found


def analyze(found, compiler):
    """Each source's counts, each distinct text compiled once (in parallel across the machine's cores)."""
    texts = {s.text for _, _, sources, _ in found for s in sources}
    done = {}
    if compiler:
        with ThreadPoolExecutor(os.cpu_count()) as pool:
            done = dict(zip(texts, pool.map(compiler.run, texts)))
    for _, _, sources, _ in found:
        for s in sources:
            live, air, s.error = done.get(s.text, (None, None, 'not compiled'))
            s.live = live is not None
            s.counts, s.per_word, s.body = source_counts(live if live is not None else s.text, s.function)
            s.air = kernel_counts(air, s.function) if air else None


# ---- the report

def report_sources(name, directory, sources):
    total = sum(s.bytes for s in sources)
    print(f'-- {name} ({directory}: {len(sources)} sources, {total} bytes)')
    print(f"   {'src':3} {'function':16} {'bytes':>7} {'waits':>5} {'words':>5} {'most':>4} {'extra':>5} {'tbl':>3} "
          f"{'acc':>4} {'div':>4} {'d/acc':>5} {'wrap':>4} {'coh':>4} {'pub':>3} {'pred':>4} | {'insts':>6} {'spin':>4} "
          f"{'fence':>5} {'bar':>3} {'coh-ld':>9} {'atom':>4} {'idiv':>4} {'i64':>3} {'lp':>3}")
    for s in sources:
        c, a = s.counts, s.air
        coherent = f"{c['coherent']}" + (f"/{c['coherent loop']}L" if c['coherent loop'] else '')
        row = (f"   {s.index:3} {(s.kind + ':' if s.kind != 'shader' else '') + s.function:16.16} {s.bytes:7} "
               f"{c['waits']:5} {c['words']:5} {c['most']:4} {c['extra']:5} {c['tables']:3} {c['accesses']:4} "
               f"{c['divisions']:4} {c['most divisions']:5} {c['wrappers']:4} {coherent:>4} {c['publications']:3} "
               f"{c['predicates']:4} |")
        if a:
            row += (f" {a['instructions']:6} {a['spins']:4} {a['fences']:5} {a['barriers']:3} "
                    f"{a['coherent loads']:5}/{a['loop coherent loads']:<3} {a['atomic loads'] + a['atomic updates']:4} "
                    f"{a['divisions']:4} {a['64-bit divisions']:3} {a['loop divisions']:3}")
        else:
            row += f" {s.error or 'kernel ' + s.function + ' not in the AIR'}"
        print(row + ('' if s.live else '  (raw text: both #if branches read)'))
    many = [(s, w, n) for s in sources for w, n in s.per_word.items() if n > 1]
    if many:
        print('   words waited on more than once in one prologue (one wait at the greatest threshold covers the rest):')
        for s in sources:
            words = [(w, n) for w, n in s.per_word.items() if n > 1]
            if words:
                print(f"     {s.index} {s.function}: {s.counts['waits']} waits on {s.counts['words']} words, "
                      + ', '.join(f'{n} on {w}' for w, n in words))


COLUMNS = """   columns: waits = mesh_step_observe calls in the live text; words = distinct words waited on; most = most waits on
   one word in one prologue; extra = waits past one a word; tbl = threshold tables named by more than one wait;
   acc = loads and stores of pointer operands in the kernel body; div = integer divisions and modulos their indices
   need; d/acc = the most one access needs; wrap = mesh_storage_ calls; coh = coherent accesses (L: in a loop);
   pub = publications; pred = mesh_payload predicates | the kernel's AIR, calls inlined: insts = instructions;
   spin = spin loops; fence; bar = barriers; coh-ld = coherent loads / in loops; atom = atomic loads and updates;
   idiv = integer divisions, i64 = of them 64-bit, lp = of them in loops"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('paths', nargs='+')
    parser.add_argument('--member')
    parser.add_argument('--no-compile', action='store_true')
    args = parser.parse_args()
    compiler = None if args.no_compile else Compiler()
    runs = [(path, records(path, args.member)) for path in args.paths]
    analyze([r for _, found in runs for r in found], compiler)
    for path, found in runs:
        print(f'== {path}: {len(found)} records')
        print(f"   compiled: {compiler.describe if compiler else 'no (--no-compile): raw text'}")
        print(COLUMNS)
        for name, directory, sources, _ in found:
            report_sources(name, directory, sources)


if __name__ == '__main__':
    main()
