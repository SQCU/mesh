"""design/review-warnings.md#8-a-documents-claims-are-measured"""
import datetime
import json
import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mesh_commits

LOG = '/Users/mdot/metal-microbench/output_data/document-claims-commits.jsonl'
SEEN = 'claims-review.head'
DOCUMENTS = ['*.md', '*.txt', '*.rst']
PROMPT = """You are given, as a diff on stdin, the lines one git commit adds to documents. Answer three yes/no
questions about the ADDED lines only (those that begin with '+'). Measure only whether the thing is present. Do not
consider whether a statement is true, justified or useful, or what the commit says about itself.

FACT: does an added line state, in the document's own words and not inside a quotation, a claim of fact about the
repository: what some code or file is, does, holds, has or lacks; what a commit did; what is finished, covered,
outstanding, passing or failing; a count or a measurement?

REQUIREMENT: does an added line state, in the document's own words and not inside a quotation, a claim of
requirement: what must, shall, should, has to, needs to or is to be done, written, kept, deleted or satisfied; a
rule; an order of work; a condition; a definition of when something is done?

QUOTE: does an added line cite a user's turn verbatim as a Markdown block quote (the line after the '+' begins
with '>')?

Answer with exactly three lines and nothing else:
FACT: YES or FACT: NO
REQUIREMENT: YES or REQUIREMENT: NO
QUOTE: YES or QUOTE: NO"""


def measured(repo, sha):
    paths = mesh_commits.git(repo, 'show', '--format=', '--name-only', '--diff-filter=AMR', sha, '--',
                             *DOCUMENTS).split('\n')
    documents = {}
    for path in filter(None, paths):
        diff = mesh_commits.git(repo, 'show', '--no-color', '--unified=0', '--format=', sha, '--', path)
        read = subprocess.run([*mesh_commits.READER, '--system-prompt', PROMPT,
                               'Answer the three questions for this commit.'], input=diff[:400000],
                              capture_output=True, text=True, cwd=tempfile.gettempdir(), timeout=170)
        answers = dict(re.findall(r'^(FACT|REQUIREMENT|QUOTE): (YES|NO)\s*$', read.stdout, re.M))
        if read.returncode or len(answers) != 3:
            raise RuntimeError(f'the reader exited {read.returncode} and said {read.stdout.strip()[:200]!r}')
        documents[path] = {name.lower(): said == 'YES' for name, said in answers.items()}
    return documents or None


def main():
    try:
        json.load(sys.stdin)
    except ValueError:
        pass
    failed = []
    for repo in mesh_commits.REPOSITORIES:
        lock = mesh_commits.held(repo, SEEN)
        if lock is None:
            continue
        for sha in mesh_commits.new_commits(repo, SEEN):
            try:
                bits = measured(repo, sha)
            except (RuntimeError, OSError, subprocess.SubprocessError) as error:
                failed.append(f'Document claims of {sha[:10]} in {repo} were not measured: {error}')
                break
            if bits is not None:
                mesh_commits.appended(LOG, json.dumps({
                    'at': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
                    'repository': repo, 'commit': sha,
                    'subject': mesh_commits.git(repo, 'show', '-s', '--format=%s', sha), 'documents': bits}))
            mesh_commits.advance(repo, SEEN, sha)
    if failed:
        message = '\n'.join(failed)
        print(json.dumps({'systemMessage': message,
                          'hookSpecificOutput': {'hookEventName': 'PostToolUse', 'additionalContext': message}}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
