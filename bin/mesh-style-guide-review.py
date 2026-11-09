"""design/change-requests.md"""
import datetime
import fnmatch
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile

MESH = '/Users/mdot/dox/mesh'
REVIEWS = '/Users/mdot/metal-microbench/output_data/change-request-reviews.jsonl'
spec = importlib.util.spec_from_file_location('c9', os.path.join(MESH, 'bin', 'mesh-c9-review.py'))
c9 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c9)
CODE = {'.c', '.h', '.m', '.mm', '.cc', '.cpp', '.cxx', '.hpp', '.swift', '.metal', '.py', '.pyi', '.pyx', '.sh',
        '.bash', '.zsh', '.js', '.mjs', '.ts', '.mk', '.s', '.S', '.ll', '.go', '.rs'}
NAMES = {'Makefile', 'GNUmakefile'}
KINDS = {'*-change-request.md': os.path.join(MESH, 'design', 'change-requests.md')}
READER = ['claude', '-p', '--model', 'opus', '--tools', '', '--setting-sources', 'user', '--no-session-persistence',
          '--strict-mcp-config']
PROMPT = """On stdin come one file a commit adds to a repository, then documents it cites or that its kind names. A style
guide for the file is one of these documents that states how files of the file's kind are written: one the file cites
as the standard it follows, or one whose stated subject is files of its kind. If none of the documents is a style guide
for the file, answer exactly NONE. If one is, report each place the file departs from it, one line each:
`line: the departure; the guide's sentence it departs from`, or answer exactly PASS if there is none. Nothing else."""
WARNING = 'Warning: Claudes are doing decadent and deceptive things as usual.'


def added(repo, sha):
    raw = c9.git(repo, 'diff-tree', '--root', '--no-commit-id', '-r', '--diff-filter=A', sha)
    for line in raw.splitlines():
        meta, _, path = line.partition('\t')
        mode = meta.split()[1] if meta.startswith(':') else ''
        name = os.path.basename(path)
        if mode != '100755' and os.path.splitext(name)[1] not in CODE and name not in NAMES:
            yield path


def candidates(repo, sha, path, text):
    found = [g for pattern, g in KINDS.items() if fnmatch.fnmatch(os.path.basename(path), pattern)]
    for cited in re.findall(r'[\w./-]+\.md\b', text):
        for base in (os.path.dirname(os.path.join(repo, path)), repo, MESH):
            full = os.path.normpath(os.path.join(base, cited))
            if os.path.isfile(full) and full != os.path.normpath(os.path.join(repo, path)):
                found.append(full)
                break
    return list(dict.fromkeys(found))


def review(repo, sha, path):
    text = c9.git(repo, 'show', f'{sha}:{path}')
    guides = candidates(repo, sha, path, text)
    if not guides:
        return 'no style guide: it cites none, and no guide names its kind'
    given = f'=== the file {path}\n{text[:200000]}\n\n' + '\n\n'.join(
        f'=== {g}\n{open(g, errors="replace").read()[:60000]}' for g in guides)
    try:
        read = subprocess.run([*READER, '--system-prompt', PROMPT, 'Review this file.'], input=given[:600000],
                              capture_output=True, text=True, cwd=tempfile.gettempdir(), timeout=880)
    except subprocess.TimeoutExpired:
        return 'the reader did not answer in 880 s'
    said = read.stdout.strip()
    if read.returncode:
        return f'the reader failed: {read.stderr.strip()[:300]}'
    if said == 'PASS':
        return None
    if said == 'NONE':
        return 'no style guide: none of the documents it cites is one'
    return 'departs from its style guide:\n' + said


def main():
    findings = []
    for repo, sha in c9.made(json.load(sys.stdin)):
        for path in added(repo, sha):
            found = review(repo, sha, path)
            if found:
                findings.append(f'{path}, added in {sha[:10]} of {repo}: {found}')
                with open(REVIEWS, 'a') as f:
                    f.write(json.dumps({'at': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
                                        'repository': repo, 'commit': c9.git(repo, 'rev-parse', sha), 'file': path,
                                        'style': found}) + '\n')
    if findings:
        message = WARNING + '\n' + '\n\n'.join(findings)
        print(json.dumps({'systemMessage': message,
                          'hookSpecificOutput': {'hookEventName': 'PostToolUse', 'additionalContext': message}}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
