"""design/review-warnings.md#12-the-measurement-of-a-change-to-a-design-document"""
import datetime
import json
import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mesh_commits

LOG = '/Users/mdot/metal-microbench/output_data/design-document-changes.jsonl'
SEEN = 'design-document-changes.head'
PROMPT = """You are given, as a diff on stdin, what one git commit changes in one document. Answer one yes/no
question about the REMOVED lines (those that begin with '-'). Measure only whether the thing is present. Do not
consider why, or what the commit says about itself.

REQUIREMENT-REMOVED: does a removed line state what must, shall, should, has to, needs to or is to be done, written,
kept, deleted or satisfied, where no added line (beginning with '+') states the same thing in the same words?

Answer with exactly one line and nothing else:
REQUIREMENT-REMOVED: YES or REQUIREMENT-REMOVED: NO"""


def measured(repo, sha):
    paths = mesh_commits.git(repo, 'show', '--format=', '--name-only', '--diff-filter=MD', sha, '--',
                             'design/').split('\n')
    documents = {}
    for path in filter(None, paths):
        diff = mesh_commits.git(repo, 'show', '--no-color', '--unified=0', '--format=', sha, '--', path)
        removed = [line[1:].strip() for line in diff.splitlines() if line.startswith('-') and not line.startswith('---')]
        added = {line[1:].strip() for line in diff.splitlines() if line.startswith('+') and not line.startswith('+++')}
        if not removed:
            continue
        read = subprocess.run([*mesh_commits.READER, '--system-prompt', PROMPT, 'Answer the question for this commit.'],
                              input=diff[:400000], capture_output=True, text=True, cwd=tempfile.gettempdir(), timeout=170)
        answer = re.search(r'^REQUIREMENT-REMOVED: (YES|NO)\s*$', read.stdout, re.M)
        if read.returncode or not answer:
            raise RuntimeError(f'the reader exited {read.returncode} and said {read.stdout.strip()[:200]!r}')
        documents[path] = {'requirement_removed': answer.group(1) == 'YES',
                           'quoted_turn_removed': any(line.startswith('>') and line not in added for line in removed)}
    return documents or None


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
                documents = measured(repo, sha)
            except (RuntimeError, OSError, subprocess.SubprocessError) as error:
                messages.append(f'Design document changes of {sha[:10]} in {repo} were not measured: {error}')
                break
            if documents:
                mesh_commits.appended(LOG, json.dumps({
                    'at': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
                    'repository': repo, 'commit': sha,
                    'subject': mesh_commits.git(repo, 'show', '-s', '--format=%s', sha), 'documents': documents}))
                for path, bits in documents.items():
                    found = ' and '.join(name for name, key in (('a requirement', 'requirement_removed'),
                                                                ('a block-quoted line', 'quoted_turn_removed')) if bits[key])
                    if found:
                        messages.append(f'WARNING: {sha[:10]} in {repo} removes {found} from {path} '
                                        f'(design/review-warnings.md rule 12). Appended to {LOG}.')
            mesh_commits.advance(repo, SEEN, sha)
    if messages:
        message = '\n'.join(messages)
        print(json.dumps({'systemMessage': message,
                          'hookSpecificOutput': {'hookEventName': 'PostToolUse', 'additionalContext': message}}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
