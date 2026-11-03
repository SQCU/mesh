#!/usr/bin/env mesh-python
import ast
import io
import os
import re
import subprocess
import sys
import tokenize
from pathlib import Path

C_SUFFIXES = {".c", ".h", ".cc", ".cpp", ".m", ".mm", ".qc", ".qh", ".inc", ".glsl", ".vert", ".frag", ".css", ".rc", ".def", ".xpm", ".cfg"}
PY_SUFFIXES = {".py"}
SHELL_SUFFIXES = {".sh", ".zsh", ".bash", ".pl", ".mk", ".toml", ".conf", ".example", ".ini"}
XML_SUFFIXES = {".xml", ".plist", ".vcxproj", ".vcproj", ".props", ".pbxproj", ".xcworkspacedata", ".cbp", ".dev"}
MAKE_NAMES = {"Makefile", "makefile", "BSDmakefile", "Doxyfile"}

def tidy(value):
    value = re.sub(r"[ \t]+(?=\r?$)", "", value, flags=re.M)
    value = re.sub(r"\n{3,}", "\n\n", value)
    value = value.rstrip("\r\n")
    return value + ("\n" if value else "")

def redundant_passes(value):
    try:
        tree = ast.parse(value)
    except SyntaxError:
        return value
    lines = value.splitlines(keepends=True)
    remove = set()
    for node in ast.walk(tree):
        for _, sequence in ast.iter_fields(node):
            if not isinstance(sequence, list):
                continue
            statements = [item for item in sequence if isinstance(item, ast.stmt)]
            if len(statements) < 2:
                continue
            for item in statements:
                if isinstance(item, ast.Pass) and lines[item.lineno - 1].strip() == "pass":
                    remove.add(item.lineno)
    return "".join(line for number, line in enumerate(lines, 1) if number not in remove)

def gap(value):
    return "".join("\n" if char == "\n" else " " for char in value)

def c_text(value):
    out = []
    i = 0
    state = "code"
    quote = ""
    while i < len(value):
        char = value[i]
        following = value[i + 1] if i + 1 < len(value) else ""
        if state == "line":
            if char == "\n":
                out.append(char)
                state = "code"
            else:
                out.append(" ")
            i += 1
            continue
        if state == "block":
            if char == "\\" and following == "\n":
                out.extend((char, following))
                i += 2
                continue
            if char == "*" and following == "/":
                out.extend((" ", " "))
                i += 2
                state = "code"
            else:
                out.append("\n" if char == "\n" else " ")
                i += 1
            continue
        if state == "string":
            out.append(char)
            if char == "\\" and following:
                out.append(following)
                i += 2
                continue
            if char == quote:
                state = "code"
            i += 1
            continue
        if char in ("'", '"', "`"):
            state = "string"
            quote = char
            out.append(char)
            i += 1
            continue
        if char == "/" and following == "/":
            out.extend((" ", " "))
            i += 2
            state = "line"
            continue
        if char == "/" and following == "*":
            out.extend((" ", " "))
            i += 2
            state = "block"
            continue
        out.append(char)
        i += 1
    return tidy("".join(out))

def python_text(value):
    tokens = []
    expect_document = True
    depth = 0
    try:
        stream = tokenize.generate_tokens(io.StringIO(value).readline)
        for token in stream:
            kind, text, start, end, line = token
            if kind == tokenize.INDENT:
                depth += 1
                expect_document = True
                tokens.append(token)
                continue
            if kind == tokenize.DEDENT:
                depth = max(0, depth - 1)
                expect_document = False
                tokens.append(token)
                continue
            if kind == tokenize.COMMENT:
                if start[0] == 1 and text.startswith("#!"):
                    tokens.append(token)
                continue
            if expect_document and kind in (tokenize.NL, tokenize.NEWLINE, tokenize.DEDENT):
                tokens.append(token)
                continue
            if expect_document and kind == tokenize.STRING:
                if depth:
                    tokens.append(tokenize.TokenInfo(tokenize.NAME, "pass", start, end, line))
                expect_document = False
                continue
            if kind not in (tokenize.ENCODING, tokenize.ENDMARKER):
                expect_document = False
            tokens.append(token)
        return tidy(redundant_passes(tokenize.untokenize(tokens)))
    except (IndentationError, SyntaxError, tokenize.TokenError):
        return shell_text(value)

def shell_text(value):
    output = []
    quote = ""
    heredoc = None
    for number, line in enumerate(value.splitlines(keepends=True), 1):
        bare = line.rstrip("\r\n")
        ending = line[len(bare):]
        if heredoc is not None:
            output.append(line)
            if bare.lstrip("\t") == heredoc:
                heredoc = None
            continue
        if number == 1 and line.startswith("#!"):
            output.append(line)
            continue
        chars = []
        i = 0
        while i < len(bare):
            char = bare[i]
            following = bare[i + 1] if i + 1 < len(bare) else ""
            if quote:
                chars.append(char)
                if quote != "'" and char == "\\" and following:
                    chars.append(following)
                    i += 2
                    continue
                if char == quote:
                    quote = ""
                i += 1
                continue
            if char in ("'", '"', "`"):
                quote = char
                chars.append(char)
                i += 1
                continue
            if char == "#" and (i == 0 or bare[i - 1] in " \t;|&()"):
                chars.extend(" " for _ in bare[i:])
                break
            chars.append(char)
            i += 1
        rebuilt = "".join(chars).rstrip(" \t") + ending
        output.append(rebuilt)
        match = re.search(r"<<-?\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?", "".join(chars))
        if match:
            heredoc = match.group(1)
    return "".join(output)

def html_text(value):
    value = re.sub(r"<!--.*?-->", lambda match: gap(match.group(0)), value, flags=re.S)
    expression = re.compile(r"(<(script|style)\b[^>]*>)(.*?)(</\2\s*>)", re.I | re.S)
    return tidy(expression.sub(lambda match: match.group(1) + c_text(match.group(3)) + match.group(4), value))

def xml_text(value):
    return tidy(re.sub(r"<!--.*?-->", lambda match: gap(match.group(0)), value, flags=re.S))

def source_kind(path, value):
    suffix = path.suffix.lower()
    if suffix in PY_SUFFIXES:
        return python_text
    if suffix in C_SUFFIXES:
        return c_text
    if suffix in SHELL_SUFFIXES or path.name in MAKE_NAMES:
        return shell_text
    if suffix == ".html":
        return html_text
    if suffix in {".js", ".ts"}:
        return c_text
    if suffix in XML_SUFFIXES:
        return xml_text
    if value.startswith("#!"):
        return python_text if "python" in value.splitlines()[0] else shell_text
    return None

def paths():
    raw = subprocess.check_output(["git", "ls-files", "-co", "--exclude-standard", "-z"])
    return [Path(os.fsdecode(item)) for item in raw.split(b"\0") if item]

def main():
    check = "--check" in sys.argv[1:]
    changed = []
    for path in paths():
        try:
            value = path.read_text()
        except (OSError, UnicodeError):
            continue
        transform = source_kind(path, value)
        if transform is None:
            continue
        replacement = transform(value)
        if replacement == value:
            continue
        changed.append(str(path))
        if not check:
            path.write_text(replacement)
    print("source_comment_file_mass=%d" % len(changed))
    for path in changed:
        print(path)
    return int(check and bool(changed))

if __name__ == "__main__":
    raise SystemExit(main())
