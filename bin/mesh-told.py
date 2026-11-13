"""design/review-warnings.md#14-one-record-for-each-commit"""
import json
import hashlib
import os
import subprocess
import sys
from mesh_notes import append

REPOSITORIES = ('/Users/mdot/dox/mesh', '/Users/mdot/metal-microbench')
PROCESS = os.path.basename(os.environ.get('CLAUDE_CODE_MESSAGING_SOCKET', '')).split('.')[0] or None
WARNING = 'Warning: Claudes are doing decadent and deceptive things as usual.'


def git(repo, *args):
    return subprocess.run(['git', '-C', repo, *args], capture_output=True, text=True).stdout.strip()


def warnings(record):
    measure = record.get('measure', '')
    if record.get('measured') is False:
        return [f"{measure}: not measured ({record.get('reason')})"]
    found = []
    if measure == 'commit':
        if record.get('assertion') or record.get('test'):
            found.append('NONCOMPLIANT and OUTSTANDING: adds ' + ' and '.join(
                name for name, key in (('an assertion', 'assertion'), ('a test', 'test')) if record.get(key)))
        if record.get('claims_without_quote'):
            found.append('adds claims of fact or of requirement, with no block-quoted user turn, to '
                         + ', '.join(record['claims_without_quote']))
        if record.get('not_typed'):
            found.append(f"{len(record['not_typed'])} quoted passage(s) were not located in the searched local "
                         f"transcripts; the first begins {record['not_typed'][0][:100]!r}. "
                         'This does not establish that the user never typed them.')
        if record.get('requirement_removed') or record.get('quoted_line_removed'):
            found.append('the diff removes text identified as ' + ' and '.join(name for name, key in (
                ('a requirement', 'requirement_removed'), ('a block-quoted line', 'quoted_line_removed')) if record.get(key))
                + '; this does not establish semantic withdrawal or require restoration')
        if record.get('source_without_item'):
            found.append('changes program source and names no change request item; this alone establishes no violation')
        if record.get('hook'):
            found.append('changes a review hook, its questions, a gate or the hook settings')
    elif measure == 'source':
        for name, key in (('a graph algorithm outside the graph module', 'graph_algorithm_outside_module'),
                          ('a second producer of a value one module produces', 'second_producer'),
                          ('a source comment that is more than a citation', 'comment_not_a_citation')):
            if record.get(key):
                found.append(f"the diff reader flagged {name}: {'; '.join(record.get('places', [])[:4])}")
    elif measure.startswith('new file '):
        if record.get('prose') and record.get('guide') is None:
            found.append(f'{WARNING} {measure} names no style guide')
        elif record.get('guide_found') is False:
            found.append(f"{measure}: cited guide {record['guide']} was not retrieved; adherence was not measured")
        elif record.get('adheres') is False:
            found.append(f"{WARNING} the reader judged {measure} not to adhere to {record['guide']}")
    elif 'clauses' in record:
        clauses = record['clauses']
        found.append(f"{measure} at document {record['document_revision']}: "
                     + '; '.join(f"{state}: {sum(c['carried'] == state for c in clauses)}"
                                 for state in ('YES', 'NO', 'NOT ADDRESSED'))
                     + f"; {sum(not c['in_document'] for c in clauses)} quotations not found in the supplied document. "
                     f"Adds something not stated: {'yes' if record['added'] else 'no'}. "
                     'These are diff judgments, not current coverage, execution evidence, or authority for requirements.')
    elif record.get('item_found'):
        found.append(f'{measure}: historical sentence-index judgment; inspect its measurement note. '
                     'It does not establish current coverage.')
    elif record.get('linked') is False:
        found.append(f'{measure}: historical reviewer skipped the document because of its former index restriction')
    elif record.get('document_found') is False or record.get('item_found') is False:
        found.append(f'{measure}: the requested document or item was not located in the supplied revision')
    if record.get('truncated'):
        found.append(f'{measure}: the reader received truncated input; the findings do not cover the omitted text')
    return found


def main():
    try:
        json.load(sys.stdin)
    except ValueError:
        pass
    messages = []
    for repo in REPOSITORIES:
        for entry in git(repo, 'notes', '--ref', 'measurements', 'list').splitlines():
            sha = entry.split()[1]
            measured = git(repo, 'notes', '--ref', 'measurements', 'show', sha)
            digest = hashlib.sha256(measured.encode()).hexdigest()
            delivered = git(repo, 'notes', '--ref', 'warnings', 'show', sha)
            if not measured or f'Measurements-SHA256: {digest}' in delivered:
                continue
            first = json.loads(measured.splitlines()[0])
            if first.get('process') not in (None, PROCESS) and os.path.exists(f"/tmp/cc-socks/{first['process']}.sock"):
                continue
            found = [w for line in measured.splitlines() if line.strip() for w in warnings(json.loads(line))]
            append(repo, 'warnings', f'Measurements-SHA256: {digest}\n' + ('\n'.join(found) or 'none'), sha)
            messages += [f'{sha[:10]} in {repo}: {w}' for w in found]
    if messages:
        message = '\n'.join(messages) + '\n(design/review-warnings.md rule 14; git notes --ref measurements show <commit>)'
        print(json.dumps({'systemMessage': message,
                          'hookSpecificOutput': {'hookEventName': 'PostToolUse', 'additionalContext': message}}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
