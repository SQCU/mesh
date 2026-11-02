#!/usr/bin/env python3
"""Waits, and what is shaped like them, in the kernel sources the page-effects compiler wrote around each numerical
kernel (metal-page-effects.m MetalPageEffectsSource; torch_mesh/page_effects.py _source_record writes each distinct one
to MESH_PREPARE_RECORD's directory as sources-<pid>/NNN-<kind>-<function>.metal), beside each dispatch's GPU time
(metal-recording.m MetalScheduleCompletedRecord: parts-<pid>.jsonl). It prints a description for people revising that
compiler; it judges, rejects and changes nothing.

    python3 extras/wait_report.py DIRECTORY... [--member NAME] [--no-compile]
    python3 extras/wait_report.py --diff BEFORE AFTER [--member NAME]

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
    integer divisions, those in loops counted apart;
  - its stamped dispatches: how many, their median and their share of the member's GPU part time (which dispatch ran
    which source is inferred from command order: align);
then the occurrences of one declaration (its text less its waits) whose bytes or waits grow with their index, the
step's dispatches by place (before its repetition, each place of a tick) with their sources, waits and times, the
step's link dispatches (report_links: those recorded later than their earliest order, and the waits on their words),
and a line a member. --diff: per member and function the change in sources, bytes, waits a step and GPU time, and
each place's. The prelude is read from this checkout, as layout.pipeline assembles it (torch_mesh/native.py
command_records): the record holds no layout declarations, so every binding compiles as the identity layout (LAYOUT).
"""
import argparse
import codecs
import collections
import json
import math
import os
import re
import statistics
import subprocess
import sys
from bisect import bisect_right
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'rdma'))
from mesh_graph import ancestors, cyclic, descendants, preorder_parents, recorded_slack  # noqa: E402  (the one graph module, design/rubrics.md C9)

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
        parents = preorder_parents(depths)
        self.enclosing, self.known = (lambda k: () if parents[k] is None else (parents[k],)), {}
        self.looped = [any(found[j][2] in LOOPS for j in self.chain(k)) for k in range(len(found))]

    def chain(self, k):
        """Scope k and every scope that holds it (mesh_graph ancestors over the parent relation)."""
        return {k} | ancestors(self.enclosing, k, self.known)

    def at(self, position):
        """The innermost scope holding position, or None: the latest-opened of the scopes holding the last one opened
        before it."""
        k = bisect_right(self.opens, position) - 1
        held = [j for j in (self.chain(k) if k >= 0 else ()) if self.spans[j][0] < position < self.spans[j][1]]
        return max(held, default=None)

    def outermost(self, position):
        k = self.at(position)
        return None if k is None else min(self.chain(k))

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
        out.update({f'table {n}': 1 for n in map(int, re.findall(r'mesh_effects\s*\+\s*(\d+)ul', ' '.join(named)))})
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
    pred = collections.defaultdict(list)
    for b, targets in succ.items():
        for t in targets:
            pred[t].append(b)
    forward, backward = {}, {}

    def loop_of(b):
        return cyclic(lambda v: succ.get(v, ()), lambda v: pred.get(v, ()), b, forward, backward)

    looped = {b for b, _ in blocks if b in loop_of(b)}
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
        loop = loop_of(b)
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
        self.declaration = declaration(self.text)
        self.durations = []


def declaration(text):
    """A source's text but for its waits: the lines that call mesh_step_observe or only close a brace go, metadata offsets
    unnumbered. One numerical declaration's occurrences (one tick's and the next's) share it."""
    kept = [line for line in text.splitlines() if 'mesh_step_observe' not in line and line.strip() != '}']
    return re.sub(r'mesh_effects\s*\+\s*\d+ul', 'mesh_effects', '\n'.join(kept))


class Record:
    """One member's record: its sources in index order and its GPU parts (metal-recording.m
    MetalScheduleCompletedRecord: schedule, part, dispatches, gpu_start, gpu_end, stamped [name, start us, us])."""
    def __init__(self, name, directory, parts):
        self.name, self.directory, self.parts_file = name, directory, parts
        self.sources = sorted((Source(f) for f in directory.iterdir() if SOURCE.match(f.name)), key=lambda s: s.index)
        self.parts = [json.loads(line) for line in parts.read_text().splitlines() if line.strip()] if parts else []
        self.functions = {s.function for s in self.sources}


def records(path, member=None):
    """Each member's record beneath path (a run's directory, a member's, or a sources-<pid>/ itself)."""
    path = Path(path)
    found = []
    for directory in sorted([path] if path.name.startswith('sources-') else path.rglob('sources-*')):
        name = directory.parent.name.removesuffix('-prepare')
        if directory.is_dir() and (not member or name == member):
            parts = directory.parent / f"parts-{directory.name.split('-', 1)[1]}.jsonl"
            found.append(Record(name, directory, parts if parts.exists() else None))
    return found


def examine(compiler, text, function):
    """A source's counts: its live text's, its raw text's where it has none, and its kernel's AIR's."""
    live, air, error = compiler.run(text) if compiler else (None, None, 'not compiled')
    counts, per_word, body = source_counts(live if live is not None else text, function)
    return live is not None, counts, per_word, body, kernel_counts(air, function) if air else None, error


def analyze(found, compiler):
    """Each source's counts, each distinct text examined once, the texts across the machine's cores."""
    jobs = sorted({(s.text, s.function) for r in found for s in r.sources})
    with ProcessPoolExecutor(os.cpu_count()) as pool:
        done = dict(zip(jobs, pool.map(examine, [compiler] * len(jobs), *zip(*jobs), chunksize=1)))
    for r in found:
        for s in r.sources:
            s.live, s.counts, s.per_word, s.body, s.air, s.error = done[s.text, s.function]
        align(r)


# ---- the stamped dispatches and their sources

def stamp_key(name):
    """A stamp by its place: a kernel's name and index in its replay; the collective's unlabeled pipelines, numbered
    across the whole schedule, by their label alone."""
    label = name.rsplit('#', 1)[0]
    return label if label == 'dispatch' else name


def period(keys):
    """(first, length) of the repetition running from the earliest place to the end of keys (an unrolled step's ticks),
    the shortest there; (len(keys), 0) where nothing repeats twice."""
    n = len(keys)
    for first in range(n):
        for length in range(1, (n - first) // 2 + 1):
            if all(keys[i] == keys[i + length] for i in range(first, n - length)):
                return first, length
    return n, 0


def align(record):
    """Each stamped dispatch's source. A stamp names its pipeline and its index in its replay, not its source; but the
    sources were written in command order at their first occurrence and the dispatches encoded in command order
    (page_effects.py Plan.append). A kernel dispatch with no earlier counterpart (before the repetition, or in its first
    tick) takes the next unwritten source where its function is that source's; a later one (the same place a tick
    before, or the same kernel dispatch of an earlier part whose kernel dispatches are these, whatever its collective
    ones) takes it only where it is the counterpart's declaration, and otherwise runs the counterpart's text again.
    None: an earlier text the record does not name."""
    at, assigned, seen, record.shapes = 0, {}, [], []
    for q, part in enumerate(record.parts):
        stamps = part.get('stamped') or []
        keys = [stamp_key(name) for name, _, _ in stamps]
        first, length = period(keys)
        kernels = [i for i, (name, _, _) in enumerate(stamps) if name.rsplit('#', 1)[0] in record.functions]
        signature = [keys[i] for i in kernels]
        earlier = next((r for r, (k, _, _) in enumerate(seen) if k == keys), None)
        like = next((r for r, (_, k, _) in enumerate(seen) if k == signature and signature), None)
        seen.append((keys, signature, kernels))
        record.shapes.append((q if earlier is None else earlier, first, length))
        for ordinal, i in enumerate(kernels):
            name, _, duration = stamps[i]
            if like is not None:
                counterpart = (like, seen[like][2][ordinal])
            else:
                counterpart = (q, i - length) if length and i >= first + length else None
            prior = assigned.get(counterpart)
            label = name.rsplit('#', 1)[0]
            following = record.sources[at] if at < len(record.sources) and record.sources[at].function == label else None
            if following is not None and (counterpart is None or (prior is not None and
                                                                 following.declaration == prior.declaration)):
                assigned[q, i] = following
                at += 1
            else:
                assigned[q, i] = prior
            if assigned[q, i] is not None:
                assigned[q, i].durations.append(duration)
    record.assigned = assigned


def places(record):
    """{pattern: (its parts, {place: entry})}: the stamped dispatches of each set of parts with the same dispatches, by
    place: (region, stamp key, its occurrence among the region's), the region 'before' the repetition or a 'tick' of it,
    so that runs whose steps differ only in collective dispatches still meet. Each entry: its first stamp name, its
    position (at), durations, sources (None: not named) and the stamps' (part, index)."""
    found = {}
    for q, part in enumerate(record.parts):
        pattern, first, length = record.shapes[q]
        parts, table = found.setdefault(pattern, ([], {}))
        parts.append(q)
        stamps = part.get('stamped') or []
        keys = [stamp_key(name) for name, _, _ in stamps]
        for i, (name, _, duration) in enumerate(stamps):
            tick = bool(length) and i >= first
            local = (i - first) % length if tick else i
            region = keys[first:first + length][:local] if tick else keys[:i]
            place = ('tick' if tick else 'before', keys[i], region.count(keys[i]))
            entry = table.setdefault(place, dict(name=name, at=(tick, local), durations=[], sources=[], stamps=[]))
            entry['durations'].append(duration)
            entry['stamps'].append((q, i))
            if name.rsplit('#', 1)[0] in record.functions:
                entry['sources'].append(record.assigned.get((q, i)))
    return found


def span(part):
    return (part['gpu_end'] - part['gpu_start']) * 1e6


def median(values):
    return statistics.median(values) if values else 0.0


def named(sources):
    """Distinct sources by index, in order, '?' for one the record does not name, long lists elided."""
    if not sources:
        return '(collective)'
    order = list(dict.fromkeys('?' if s is None else s.index for s in sources))
    return ' '.join(order) if len(order) <= 5 else f"{' '.join(order[:2])} .. {order[-1]} ({len(order)})"


def spread(values):
    values = sorted(set(values))
    return '-' if not values else f'{values[0]}' if len(values) == 1 else f'{values[0]}-{values[-1]}'


def main_pattern(record):
    """The pattern of parts with the most kernel dispatches (the step), or None."""
    shapes = places(record)
    best = max(shapes.items(), key=lambda item: sum(len(e['sources']) for e in item[1][1].values()), default=None)
    return (best[0], *best[1]) if best and any(e['sources'] for e in best[1][1].values()) else None


# ---- the report

def report_sources(record):
    sources, gpu = record.sources, sum(span(p) for p in record.parts)
    print(f"-- {record.name} ({record.directory}: {len(sources)} sources, {sum(s.bytes for s in sources)} bytes; "
          f"{len(record.parts)} GPU parts, {gpu / 1e3:.1f} ms)")
    print(f"   {'src':3} {'function':16} {'bytes':>7} {'waits':>5} {'words':>5} {'most':>4} {'extra':>5} {'tbl':>3} "
          f"{'acc':>4} {'div':>4} {'d/acc':>5} {'wrap':>4} {'coh':>4} {'pub':>3} {'pred':>4} | {'insts':>6} {'spin':>4} "
          f"{'fence':>5} {'bar':>3} {'coh-ld':>9} {'atom':>4} {'idiv':>4} {'i64':>3} {'lp':>3} | {'n':>3} {'med us':>9} "
          f"{'share':>6}")
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
                    f"{a['divisions']:4} {a['64-bit divisions']:3} {a['loop divisions']:3} |")
        else:
            row += f" {(s.error or 'kernel ' + s.function + ' not in the AIR')[:60]:60} |"
        row += (f" {len(s.durations):3} {median(s.durations):9.1f} {100 * sum(s.durations) / gpu:5.1f}%" if s.durations
                else f" {0:3} {'-':>9} {'-':>6}")
        print(row + ('' if s.live else '  (raw text: both #if branches read)'))
    for s in sources:
        words = [(w, n) for w, n in s.per_word.items() if n > 1]
        if words:
            print(f"   {s.index} {s.function}: {s.counts['waits']} waits on {s.counts['words']} words in one prologue, "
                  + ', '.join(f'{n} on {w}' for w, n in words) + ' (one at its greatest threshold covers each word)')
    chains = collections.defaultdict(list)
    for s in sources:
        chains[s.kind, s.function, s.declaration].append(s)
    for (kind, function, _), same in chains.items():
        if len(same) > 1:
            sizes, waits = [s.bytes for s in same], [s.counts['waits'] for s in same]
            grows = [name for name, v in (('bytes', sizes), ('waits', waits)) if all(x < y for x, y in zip(v, v[1:]))]
            print(f"   {function}: one declaration in {len(same)} sources {named(same)}: bytes {sizes[0]} .. {sizes[-1]}, "
                  f"waits {' '.join(map(str, waits))}" + (f"; {' and '.join(grows)} grow at every occurrence" if grows else ''))


def report_dispatches(record):
    if not record.parts:
        print('   no parts file beside the sources')
        return
    print(f"   stamped dispatches ({record.parts_file.name}):")
    for pattern, (parts, table) in places(record).items():
        _, first, length = record.shapes[pattern]
        spans = [span(record.parts[q]) for q in parts]
        stamps = len(record.parts[pattern].get('stamped') or [])
        shape = f'{first} before a repetition of {length}, {(stamps - first) // length} times' if length else 'no repetition'
        if not any(e['sources'] for e in table.values()):
            print(f"     {len(parts)} part(s) of {stamps} collective dispatches, median part {median(spans):.1f} us")
            continue
        print(f"     {len(parts)} part(s) of {stamps} dispatches ({shape}), median part {median(spans) / 1e3:.2f} ms:")
        print(f"       {'place':10} {'dispatch':20} {'sources':22} {'waits':>7} {'extra':>7} {'n':>4} {'median us':>10} "
              f"{'total ms':>9} {'share':>6}")
        for e in sorted(table.values(), key=lambda e: e['at']):
            known = [s for s in e['sources'] if s is not None]
            print(f"       {('tick ' if e['at'][0] else 'before ') + str(e['at'][1]):10} {e['name']:20.20} {named(e['sources']) if e['sources'] else '(collective)':22.22} "
                  f"{spread(s.counts['waits'] for s in known):>7} {spread(s.counts['extra'] for s in known):>7} "
                  f"{len(e['durations']):4} {median(e['durations']):10.1f} {sum(e['durations']) / 1e3:9.2f} "
                  f"{100 * sum(e['durations']) / sum(spans):5.1f}%")


def report_links(record):
    """The step's link dispatches (nccl-mesh.c zinth: per channel with landings, its landing pages declared written,
    then metal_open_gate's publish, recorded at the collective's own place), each run of them beside the first later
    kernel that waits. The record holds no page reaches, so a command is taken to touch the landing pages where it
    observes a word that kernel observes (words named by their place in each command's metadata), and a kernel whose
    source is not named as one that may. A run recorded after kernels that touch none of them could have been recorded
    that much earlier: its receive posted before them, not after."""
    found = main_pattern(record)
    if not found:
        return
    _, parts, _ = found
    # a word by its offset in its page: a composed field lies at its page's place in the metadata plus its own offset in
    # that page (metal-page-effects.m MetalPageFields compose), the page width the one that divides every threshold
    # table's place (each table a whole buffer of its own, so at a page's start)
    width = math.gcd(*(int(k.split()[1]) for s in record.sources for k in s.counts if k.startswith('table ')))

    def words(s):
        return {int(n) % width if width else int(n) for w in s.per_word for n in re.findall(r'(\d+)ul', w)}

    runs = {}
    for q in parts:
        stamps = record.parts[q].get('stamped') or []
        _, first, length = record.shapes[q]

        def at(i):
            return ('tick ' if length and i >= first else 'before ') + str((i - first) % length if length and i >= first else i)

        def source(k):
            return record.assigned.get((q, k), False)  # False: a collective dispatch, None: a kernel not named

        i = 0
        while i < len(stamps):
            if source(i) is not False:
                i += 1
                continue
            j = i
            while j < len(stamps) and source(j) is False:
                j += 1
            waiting = next((k for k in range(j, len(stamps)) if source(k) and source(k).counts['waits']), None)
            if waiting is None:
                i = j
                continue
            landing = words(source(waiting))

            def touches(a, b):
                # the run at i touches what observes a word of its landing; a kernel not named may touch it
                other = source(b if a == i else a)
                return other is None or (other is not False and bool(landing & words(other)))

            following = next((k for k in range(j, len(stamps)) if source(k) is False), len(stamps))
            toucher, before, reached = recorded_slack(range(len(stamps)), touches, i, following)
            between = [k for k in before if source(k) is not False]
            later = [k for k in reached if k > waiting and source(k)]
            entry = runs.setdefault((at(i), at(waiting), None if toucher is None else at(toucher)), dict(
                names=[stamps[k][0] for k in range(i, j)], waiting=stamps[waiting][0], between=between and [
                    (at(k), stamps[k][0], source(k)) for k in between], toucher=None if toucher is None else
                stamps[toucher][0], gaps=[], spent=[], waited=[], sources=[], touchers=[], n=0, later={}))
            for k in later:
                held = entry['later'].setdefault(at(k), [stamps[k][0], [], []])
                held[1].append(source(k))
                held[2].append(stamps[k][2])
            entry['n'] += 1
            entry['sources'].append(source(waiting))
            entry['waited'].append(stamps[waiting][2])
            entry['spent'].append(sum(stamps[k][2] for k in between))
            if toucher is not None:
                entry['touchers'].append(source(toucher))
                entry['gaps'].append(stamps[i][1] - stamps[toucher][1] - stamps[toucher][2])
            i = j
    if not runs:
        return
    print(f'   link dispatches before a kernel that waits (a command touches the landing pages where it observes a word '
          f'that kernel observes, a word named by its offset in its {width}-byte page):')
    for (place, waiting_place, toucher_place), e in runs.items():
        line = (f"     {place} {' '.join(e['names'])} -> {waiting_place} {e['waiting']} ({named(e['sources'])}, "
                f"{spread(s.counts['waits'] for s in e['sources'])} waits, median {median(e['waited']):.1f} us), "
                f"{e['n']} time(s): ")
        line += ('no earlier kernel of the step touches them' if toucher_place is None else
                 f"last touched by {toucher_place} {e['toucher']} ({named(e['touchers'])})")
        if e['between']:
            line += (f"; recorded {len(e['between'])} kernel dispatch(es) later than its earliest order ("
                     + ', '.join(f"{p} {n} {named([s])}" for p, n, s in e['between'])
                     + f": median {median(e['spent']):.1f} us of GPU time)")
        else:
            line += '; recorded at its earliest order'
        if e['gaps']:
            line += f"; the toucher's end to the link's start: median {median(e['gaps']):.1f} us"
        if e['later']:
            line += ('; later waits on its words before the next link: ' + ', '.join(
                f"{p} {n} ({named(s)}, median {median(d):.1f} us)" for p, (n, s, d) in e['later'].items()))
        print(line)


def step(record):
    """Per part of the step (main_pattern): its GPU us, waits and extra waits run, kernel dispatches whose source is not
    named, and per function its GPU us; with the step's places."""
    found = main_pattern(record)
    if not found:
        return None
    pattern, parts, table = found
    each = []
    for q in parts:
        row = collections.Counter(span=span(record.parts[q]))
        for i, (name, _, duration) in enumerate(record.parts[q].get('stamped') or []):
            label = name.rsplit('#', 1)[0]
            row['us ' + label] += duration
            row['n ' + label] += 1
            if label in record.functions:
                s = record.assigned.get((q, i))
                row['unnamed'] += s is None
                if s is not None:
                    row['waits'] += s.counts['waits']
                    row['extra'] += s.counts['extra']
                    row['waits ' + label] += s.counts['waits']
                    row['extra ' + label] += s.counts['extra']
        each.append(row)
    return collections.Counter({key: median([row[key] for row in each]) for key in set().union(*each)}), table


def report_summary(found):
    print('   summary, a step being one part of the most kernel dispatches (medians over its parts):')
    print(f"     {'member':12} {'sources':>7} {'bytes':>9} {'waits':>6} {'extra':>6} {'unnamed':>7} {'step ms':>8}  slowest place")
    for r in found:
        made = step(r)
        if not made:
            print(f"     {r.name:12} {len(r.sources):7} {sum(s.bytes for s in r.sources):9}  (no stamped kernel dispatches)")
            continue
        values, table = made
        e = max(table.values(), key=lambda e: median(e['durations']))
        known = [s for s in e['sources'] if s is not None]
        print(f"     {r.name:12} {len(r.sources):7} {sum(s.bytes for s in r.sources):9} {values['waits']:6.0f} "
              f"{values['extra']:6.0f} {values['unnamed']:7.0f} {values['span'] / 1e3:8.2f}  "
              f"{('tick ' if e['at'][0] else 'before ') + str(e['at'][1])} {e['name']} "
              f"{median(e['durations']):.0f} us, sources {named(e['sources']) if e['sources'] else '(collective)'}, "
              f"waits {spread(s.counts['waits'] for s in known)}")


def report_diff(paths, before, after):
    print(f'== {paths[0]} -> {paths[1]}: per member and function, a step being one part of the most kernel dispatches')
    print(f"   {'member':12} {'function':18} {'sources':>9} {'bytes':>17} {'waits/step':>11} {'extra/step':>11} "
          f"{'GPU ms/step':>15} {'median us':>17}")
    pairs = [(a, b) for a in before for b in after if a.name == b.name]
    for a, b in pairs:
        made = step(a), step(b)
        for function in sorted(a.functions | b.functions) + ['dispatch']:
            row = []
            for r, m in zip((a, b), made):
                values = m[0] if m else collections.Counter()
                mine = [s for s in r.sources if s.function == function]
                durations = [d for s in mine for d in s.durations] if function != 'dispatch' else \
                    [d for part in r.parts for name, _, d in part.get('stamped') or [] if name.startswith('dispatch#')]
                row.append((len(mine), sum(s.bytes for s in mine), values.get('waits ' + function, 0),
                            values.get('extra ' + function, 0), values.get('us ' + function, 0) / 1e3, median(durations)))
            (n0, b0, w0, e0, g0, m0), (n1, b1, w1, e1, g1, m1) = row
            print(f"   {a.name:12} {function if function != 'dispatch' else '(collective)':18} {n0:4}->{n1:<4} "
                  f"{b0:8}->{b1:<8} {w0:5.0f}->{w1:<5.0f} {e0:5.0f}->{e1:<5.0f} {g0:7.2f}->{g1:<7.2f} {m0:8.1f}->{m1:<8.1f}")
        if made[0] and made[1]:
            print(f"   {'':12} {'step':18} {'':9} {'':17} {made[0][0]['waits']:5.0f}->{made[1][0]['waits']:<5.0f} "
                  f"{made[0][0]['extra']:5.0f}->{made[1][0]['extra']:<5.0f} "
                  f"{made[0][0]['span'] / 1e3:7.2f}->{made[1][0]['span'] / 1e3:<7.2f}")
    print('   by place in the step (the same kernel or collective dispatch, counted in its region, in both runs):')
    print(f"   {'member':12} {'place':9} {'dispatch':20} {'sources':>24} {'waits':>13} {'median us':>21}")
    for a, b in pairs:
        made = step(a), step(b)
        if not (made[0] and made[1]):
            continue
        both = [p for p in made[0][1] if p in made[1][1]]
        for place in sorted(both, key=lambda p: made[0][1][p]['at']):
            e0, e1 = made[0][1][place], made[1][1][place]
            if not (e0['sources'] or e1['sources']) and median(e0['durations']) < 100 and median(e1['durations']) < 100:
                continue
            w = [spread(s.counts['waits'] for s in e['sources'] if s is not None) for e in (e0, e1)]
            src = [named(e['sources']) if e['sources'] else 'coll' for e in (e0, e1)]
            at = ('tick ' if e0['at'][0] else 'before ') + str(e0['at'][1])
            print(f"   {a.name:12} {at:9} {e0['name']:20.20} {src[0]:>11.11}->{src[1]:<11.11} "
                  f"{w[0]:>6}->{w[1]:<6} {median(e0['durations']):9.1f}->{median(e1['durations']):<9.1f}")
        alone = len(made[0][1]) + len(made[1][1]) - 2 * len(both)
        if alone:
            print(f"   {a.name:12} {alone} place(s) in one run's step only")


COLUMNS = """   columns: waits = mesh_step_observe calls in the live text; words = distinct words waited on; most = most waits on
   one word in one prologue; extra = waits past one a word; tbl = threshold tables named by more than one wait;
   acc = loads and stores of pointer operands in the kernel body; div = integer divisions and modulos their indices
   need; d/acc = the most one access needs; wrap = mesh_storage_ calls; coh = coherent accesses (L: in a loop);
   pub = publications; pred = mesh_payload predicates | the kernel's AIR, calls inlined: insts = instructions;
   spin = spin loops; fence; bar = barriers; coh-ld = coherent loads / in loops; atom = atomic loads and updates;
   idiv = integer divisions, i64 = of them 64-bit, lp = of them in loops | its stamped dispatches (align: which
   dispatch ran which source is inferred from command order): n, median us, share of the member's GPU part time"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('paths', nargs='+')
    parser.add_argument('--member')
    parser.add_argument('--no-compile', action='store_true')
    parser.add_argument('--diff', action='store_true', help='two runs: the per-function change from the first')
    args = parser.parse_args()
    if args.diff and len(args.paths) != 2:
        parser.error('--diff takes two runs')
    compiler = None if args.no_compile else Compiler()
    runs = [(path, records(path, args.member)) for path in args.paths]
    analyze([r for _, found in runs for r in found], compiler)
    if args.diff:
        report_diff(args.paths, runs[0][1], runs[1][1])
        return
    for path, found in runs:
        print(f'== {path}: {len(found)} records')
        print(f"   compiled: {compiler.describe if compiler else 'no (--no-compile): raw text'}")
        print(COLUMNS)
        for record in found:
            report_sources(record)
            report_dispatches(record)
            report_links(record)
        report_summary(found)


if __name__ == '__main__':
    main()
