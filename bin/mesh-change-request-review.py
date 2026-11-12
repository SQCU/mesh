"""design/review-warnings.md#11-the-change-request-measurement"""
import datetime
import json
import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mesh_commits

MESH = '/Users/mdot/dox/mesh'
INDEX = 'design/change-requests-index.md'
LOG = '/Users/mdot/metal-microbench/output_data/change-request-measurements.jsonl'
SEEN = 'change-request-measurement.head'
NAMED = re.compile(r'^Change-Request: (\S+), item ([A-Z]{0,2}\.?\d+(?:\.\d+)*)\s*$', re.M)
READER = ['claude', '-p', '--model', 'sonnet', '--tools', '', '--setting-sources', 'user', '--no-session-persistence',
          '--strict-mcp-config']
PROMPT = """You are given one item of a change request, then one git commit's diff. You have nothing else and need
nothing else. Measure; do not advise, explain or recommend.

For each sentence of the item that states something to be written, changed or removed, give one line:
CLAUSE: "<the sentence, copied exactly from the item>" | CARRIED: YES
or CARRIED: NO (the diff changes the place the sentence is about and does not do what it says)
or CARRIED: NOT ADDRESSED (the diff does not touch what the sentence is about).

Then one line:
ADDED: YES (the diff adds behaviour, a rule, a check, a guard, a wait, a copy or an interface that no sentence of the
item states) or ADDED: NO

Nothing else: no summary, no reasons, no suggestions."""


def squeezed(text):
    return re.sub(r'\s+', ' ', text).strip()


def linked():
    index = mesh_commits.git(MESH, 'show', f'HEAD:{INDEX}')
    return {'design/' + name for name in re.findall(r'\]\(([^)\s]+\.md)\)', index)}


def item(document, name, revision):
    text = mesh_commits.git(MESH, 'show', f'{revision}:{document}')
    lines = text.splitlines()
    for at, line in enumerate(lines):
        heading = re.match(rf'(#{{1,6}})\s+{re.escape(name)}[.:\s]', line)
        if heading:
            depth = len(heading.group(1))
            end = next((later for later in range(at + 1, len(lines))
                        if re.match(rf'#{{1,{depth}}}\s', lines[later])), len(lines))
            return '\n'.join(lines[at:end])
    return None


def measured(repo, sha, document, name):
    revision = sha if os.path.samefile(repo, MESH) else 'HEAD'
    text = item(document, name, revision)
    if text is None:
        return {'item_found': False}
    diff = mesh_commits.git(repo, 'show', '--no-color', '--format=', sha)
    given = f'THE ITEM ({document}, item {name}):\n{text}\n\nTHE COMMIT:\n{diff[:300000]}'
    read = subprocess.run([*READER, '--system-prompt', PROMPT, 'Measure this commit against this item.'], input=given,
                          capture_output=True, text=True, cwd=tempfile.gettempdir(), timeout=600)
    added = re.search(r'^ADDED: (YES|NO)', read.stdout, re.M)
    if read.returncode or not added:
        raise RuntimeError(f'the reader exited {read.returncode} and said {read.stdout.strip()[:200]!r}')
    whole = squeezed(text)
    clauses = [{'clause': clause, 'carried': carried, 'in_item': squeezed(clause) in whole}
               for clause, carried in re.findall(r'^CLAUSE: "(.*)" \| CARRIED: (YES|NO|NOT ADDRESSED)\s*$', read.stdout, re.M)]
    return {'item_found': True, 'clauses': clauses, 'added': added.group(1) == 'YES'}


def told(sha, repo, document, name, result):
    if not result['item_found']:
        return f'Change request measurement of {sha[:10]} in {repo}: {document} has no item {name}.'
    counted = {state: sum(1 for c in result['clauses'] if c['in_item'] and c['carried'] == state)
               for state in ('YES', 'NO', 'NOT ADDRESSED')}
    return (f"Change request measurement of {sha[:10]} in {repo} against {document}, item {name}: "
            f"{counted['YES']} sentences carried, {counted['NO']} not carried, {counted['NOT ADDRESSED']} not addressed; "
            f"{sum(1 for c in result['clauses'] if not c['in_item'])} quoted sentences are not in the item; "
            f"the diff adds something the item does not state: {'yes' if result['added'] else 'no'}. "
            f"The sentences are in {LOG}. This is a measurement, not an instruction; the item is the instruction.")


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
            named = NAMED.findall(mesh_commits.git(repo, 'show', '-s', '--format=%B', sha))
            try:
                for document, name in named:
                    if document not in linked():
                        continue
                    result = measured(repo, sha, document, name)
                    mesh_commits.appended(LOG, json.dumps({
                        'at': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
                        'repository': repo, 'commit': sha, 'document': document, 'item': name, **result}))
                    messages.append(told(sha, repo, document, name, result))
            except (RuntimeError, OSError, subprocess.SubprocessError) as error:
                messages.append(f'Change request measurement of {sha[:10]} in {repo} was not made: {error}')
                break
            mesh_commits.advance(repo, SEEN, sha)
    if messages:
        message = '\n'.join(messages)
        print(json.dumps({'systemMessage': message,
                          'hookSpecificOutput': {'hookEventName': 'PostToolUse', 'additionalContext': message}}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
