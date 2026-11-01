"""Preparation-only edits to retained Metal entry declarations."""
import re


def body_scope(body):
    """Keep numerical early returns inside the emitted numerical expression."""
    nested = []
    for match in re.finditer(r'\[[^\]]*\]\s*(?:\([^)]*\))?\s*(?:->[^{};]+)?\s*\{', body):
        cursor, depth = match.end(), 1
        while depth:
            depth += (body[cursor] == '{') - (body[cursor] == '}')
            cursor += 1
        nested.append((match.start(), cursor))
    returns = [match for match in re.finditer(r'\breturn\b', body)
               if not any(first <= match.start() < last for first, last in nested)]
    if not returns:
        return body
    # Metal requires threadgroup storage at the kernel's lexical scope. Its
    # constant declarations stay there too; only numerical control is inlined.
    declarations = []
    for match in re.finditer(r'(?m)^\s*(?:threadgroup|constexpr)\s+[^;{}]+;', body):
        if body[:match.start()].count('{') != body[:match.start()].count('}'):
            continue
        declarations.append(match)
    prefix = ''.join(match.group() for match in declarations)
    for match in reversed(declarations):
        body = body[:match.start()] + body[match.end():]
    return prefix + '[&]() {' + body + '}();'


def entries(source, function, transform):
    """Apply one edit to every source definition of the declared entry.

    Conditional source variants and explicitly instantiated templates retain
    the same declaration. Callers change parameters and the body together;
    they do not parse or select another implementation at execution.
    """
    instance = re.search(r'host_name\("' + re.escape(function) + r'"\).*?decltype\((\w+)\s*<', source)
    definition = instance.group(1) if instance is not None else function
    matches = tuple(re.finditer(r'kernel\s+void\s+' + re.escape(definition) + r'\s*\(', source))
    if not matches:
        raise ValueError('a declared shader entry must retain its source definition')
    for match in reversed(matches):
        start, cursor, depth = match.end(), match.end(), 1
        while depth:
            depth += (source[cursor] == '(') - (source[cursor] == ')')
            cursor += 1
        end = cursor - 1
        opening = source.index('{', cursor)
        closing, depth = opening + 1, 1
        while depth:
            depth += (source[closing] == '{') - (source[closing] == '}')
            closing += 1
        parameters, body = transform(source[start:end].rstrip(), source[opening + 1:closing - 1])
        template = re.search(r'template\s*<[^;{}]*>\s*$', source[:match.start()])
        declaration = template.start() if template is not None else match.start()
        source = (source[:declaration] + '#pragma METAL internals : enable\n' + source[declaration:start] +
                  parameters + source[end:opening + 1] + body + source[closing - 1:])
    return source
