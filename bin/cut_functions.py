"""Delete C or Objective-C top-level definitions by name, with their leading comments and their prototypes.

usage: cut_functions.py FILE NAME...   (struct:NAME for a struct definition)

A function definition starts at a column-0 line naming NAME( and runs until its braces balance (braces in strings,
character literals and comments not counted); a column-0 prototype of NAME (a statement ending in ';' before any '{')
goes too. A struct definition starts at a column-0 'struct NAME {' and runs until its braces balance and its ';'. The
comment block directly above a definition goes with it. The file is rewritten in place; each name nothing was found for
is printed, so a typo is never silent.
"""
import re
import sys


def _depths(lines):
    """the brace depth at the end of each line, outside strings, character literals and comments"""
    depth, state, out = 0, None, []
    for k, line in enumerate(lines):
        # a C++ linkage block (a header's extern "C" { under #ifdef __cplusplus, and its lone closing brace) is no
        # nesting: its declarations are top-level
        if line.strip() == 'extern "C" {' or (line.strip() == '}' and k and lines[k - 1].strip() == '#ifdef __cplusplus'):
            out.append(depth)
            continue
        i = 0
        while i < len(line):
            c, d = line[i], line[i:i + 2]
            if state == 'block':
                if d == '*/':
                    state, i = None, i + 1
            elif state in ('"', "'"):
                if c == '\\':
                    i += 1
                elif c == state:
                    state = None
            elif d == '//':
                break
            elif d == '/*':
                state, i = 'block', i + 1
            elif c in '"\'':
                state = c
            elif c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
            i += 1
        if state in ('"', "'"):
            state = None
        out.append(depth)
    return out


def cut(text, names):
    lines = text.split('\n')
    depth = _depths(lines)
    before = lambda i: depth[i - 1] if i else 0
    dead = [False] * len(lines)
    found = set()

    def kill(start, end):
        while start > 0 and lines[start - 1].lstrip().startswith(('/*', '*', '//')) and not dead[start - 1] \
                and before(start - 1) == 0:
            start -= 1
        for k in range(start, end + 1):
            dead[k] = True

    for name in names:
        if name.startswith('struct:'):
            pattern = re.compile(rf'struct {re.escape(name[len("struct:"):])}\s*\{{')
        else:
            pattern = re.compile(rf'\b{re.escape(name)}\s*\(')
        for i, line in enumerate(lines):
            if dead[i] or before(i) != 0 or not line or line[0] in ' \t#/*}' or not pattern.search(line):
                continue
            end = i
            while end < len(lines) and '{' not in lines[end] and not lines[end].rstrip().endswith(';'):
                end += 1
            if end < len(lines) and '{' in lines[end]:
                while end < len(lines) - 1 and depth[end] != 0:
                    end += 1
            elif name.startswith('struct:'):
                continue
            kill(i, end)
            found.add(name)
    kept = [line for line, gone in zip(lines, dead) if not gone]
    return '\n'.join(kept), [name for name in names if name not in found]


if __name__ == '__main__':
    path, names = sys.argv[1], sys.argv[2:]
    text, missing = cut(open(path).read(), names)
    open(path, 'w').write(text)
    for name in missing:
        print(f'{path}: no {name}', file=sys.stderr)
