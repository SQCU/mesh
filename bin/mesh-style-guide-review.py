"""design/review-warnings.md#13-the-measurement-of-a-new-files-style-guide"""
import datetime
import json
import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mesh_commits

LOG = '/Users/mdot/metal-microbench/output_data/new-file-style-guides.jsonl'
SEEN = 'new-file-style-guides.head'
CODE = {'.c', '.h', '.m', '.mm', '.cc', '.cpp', '.cxx', '.hpp', '.swift', '.metal', '.py', '.pyi', '.pyx', '.sh',
        '.bash', '.zsh', '.js', '.mjs', '.ts', '.mk', '.s', '.S', '.ll', '.go', '.rs'}
NAMES = {'Makefile', 'GNUmakefile'}
WARNING = 'Warning: Claudes are doing decadent and deceptive things as usual.'
CITES = """You are given one new file on stdin. Answer one question from the file's text alone. Does the file name,
by a path or a Markdown link, a document it says it follows as its style guide, standard or format?

Answer with exactly one line and nothing else:
GUIDE: <the path or link target exactly as the file writes it>
or
GUIDE: NONE"""
ADHERES = """You are given a style guide, then one new file that says it follows that guide. You have nothing else
and need nothing else. Measure only; do not advise.

Answer with exactly one line and nothing else:
ADHERES: YES or ADHERES: NO"""


def added(repo, sha):
    raw = mesh_commits.git(repo, 'diff-tree', '--root', '--no-commit-id', '-r', '--diff-filter=A', '-z', sha)
    fields = raw.split('\0')
    for meta, path in zip(fields[::2], fields[1::2]):
        mode = meta.split()[1] if meta.startswith(':') else ''
        name = os.path.basename(path)
        if mode != '100755' and os.path.splitext(name)[1] not in CODE and name not in NAMES:
            yield path


def asked(prompt, given, model):
    reader = [model if part == 'claude-haiku-5-5' else part for part in mesh_commits.READER]
    read = subprocess.run([*reader, '--system-prompt', prompt, 'Answer the question.'], input=given[:400000],
                          capture_output=True, text=True, cwd=tempfile.gettempdir(), timeout=300)
    if read.returncode:
        raise RuntimeError(f'the reader exited {read.returncode}: {read.stderr.strip()[:200]}')
    return read.stdout


def measured(repo, sha, path):
    text = mesh_commits.git(repo, 'show', f'{sha}:{path}')
    cited = re.search(r'^GUIDE: (.+?)\s*$', asked(CITES, text, 'claude-haiku-5-5'), re.M)
    if not cited:
        raise RuntimeError('the reader named no guide and did not say NONE')
    if cited.group(1) == 'NONE':
        return {'cites_guide': False}
    target = os.path.normpath(os.path.join(os.path.dirname(path), cited.group(1).split('#')[0]))
    guide = next((mesh_commits.git(root, 'show', f'{revision}:{name}')
                  for root, revision, name in ((repo, sha, target), (repo, sha, cited.group(1).split('#')[0]),
                                               (mesh_commits.REPOSITORIES[0], 'HEAD', cited.group(1).split('#')[0]))
                  if mesh_commits.git(root, 'cat-file', '-t', f'{revision}:{name}') == 'blob'), None)
    if guide is None:
        return {'cites_guide': True, 'guide': cited.group(1), 'guide_found': False}
    answer = re.search(r'^ADHERES: (YES|NO)\s*$',
                       asked(ADHERES, f'THE GUIDE:\n{guide}\n\nTHE NEW FILE ({path}):\n{text}', 'sonnet'), re.M)
    if not answer:
        raise RuntimeError('the reader did not answer ADHERES')
    return {'cites_guide': True, 'guide': cited.group(1), 'guide_found': True, 'adheres': answer.group(1) == 'YES'}


def main():
    try:
        json.load(sys.stdin)
    except ValueError:
        pass
    messages = []
    for repo in mesh_commits.REPOSITORIES:
        lock = mesh_commits.held(repo, SEEN)
        if lock is None:
            continue
        for sha in mesh_commits.new_commits(repo, SEEN):
            try:
                for path in added(repo, sha):
                    bits = measured(repo, sha, path)
                    mesh_commits.appended(LOG, json.dumps({
                        'at': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
                        'repository': repo, 'commit': sha, 'file': path, **bits}))
                    if not bits['cites_guide']:
                        messages.append(f'{WARNING} {sha[:10]} in {repo} adds {path}, which names no style guide.')
                    elif not bits['guide_found']:
                        messages.append(f"{WARNING} {sha[:10]} in {repo} adds {path}, which names {bits['guide']}, "
                                        f"and no such file is in the repository.")
                    elif not bits['adheres']:
                        messages.append(f"{WARNING} {sha[:10]} in {repo} adds {path}, which names {bits['guide']} "
                                        f"and does not adhere to it.")
            except (RuntimeError, OSError, subprocess.SubprocessError) as error:
                messages.append(f'New files of {sha[:10]} in {repo} were not measured: {error}')
                break
            mesh_commits.advance(repo, SEEN, sha)
    if messages:
        message = '\n'.join(messages) + f'\nEach file is appended to {LOG} (design/review-warnings.md rule 13).'
        print(json.dumps({'systemMessage': message,
                          'hookSpecificOutput': {'hookEventName': 'PostToolUse', 'additionalContext': message}}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
