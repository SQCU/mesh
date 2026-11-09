"""design/commentary-review.md#investigative-review"""
import json
import os
import subprocess
import sys
import mesh_review as harness

CODE = {'.c', '.h', '.m', '.mm', '.cc', '.cpp', '.cxx', '.hpp', '.swift', '.metal', '.py', '.pyi', '.pyx', '.sh',
        '.bash', '.zsh', '.js', '.mjs', '.ts', '.mk', '.s', '.S', '.ll', '.go', '.rs'}
NAMES = {'Makefile', 'GNUmakefile'}
WARNING = 'Warning: Claudes are doing decadent and deceptive things as usual.'
TASK = """Review the new non-executable files listed below against the style guides that actually govern their kinds.
Find the applicable guides in the repository and its history, including guides not explicitly linked from a file.
Read the files and guides and investigate their provenance where needed. Report an absent guide or a substantive
failure to adhere to the applicable guide with evidence. Do not equate a missing Markdown link or unfamiliar filename
with an absent guide. Existing files are grandfathered for this review; read them as context, not as new violations.
The review of implementation correctness is separate from this assignment."""


def added(repo, sha):
    raw = harness.git(repo, 'diff-tree', '--root', '--no-commit-id', '-r', '--diff-filter=A', '-z', sha)
    fields = raw.split('\0')
    for meta, path in zip(fields[::2], fields[1::2]):
        mode = meta.split()[1] if meta.startswith(':') else ''
        name = os.path.basename(path)
        if mode != '100755' and os.path.splitext(name)[1] not in CODE and name not in NAMES:
            yield path


def main():
    messages = []
    try:
        payload = json.load(sys.stdin)
        for repo, sha in harness.c9.made(payload):
            paths = list(added(repo, sha))
            if not paths:
                continue
            record = harness.review(repo, sha, TASK + '\n\nNew files:\n' + json.dumps(paths),
                                    transcript=payload.get('transcript_path'), kind='style')
            if record['outstanding'] is not False:
                prefix = WARNING if record['findings'] else 'Style review incomplete.'
                messages.append(prefix + '\n' + harness.render(record))
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        messages.append(f'Style review incomplete: {error}')
    return harness.c9.feedback(messages)


if __name__ == '__main__':
    sys.exit(main())
