"""design/review-warnings.md#14-one-record-for-each-commit"""
import json
import os
import subprocess
import sys

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
        if record['assertion'] or record['test']:
            found.append('NONCOMPLIANT and OUTSTANDING: adds ' + ' and '.join(
                name for name, key in (('an assertion', 'assertion'), ('a test', 'test')) if record[key]))
        if record['claims_without_quote']:
            found.append('adds claims of fact or of requirement, with no block-quoted user turn, to '
                         + ', '.join(record['claims_without_quote']))
        if record['not_typed']:
            found.append(f"presents {len(record['not_typed'])} passage(s) as the user's words that no typed turn "
                         f"contains; the first begins {record['not_typed'][0][:100]!r}")
        if record['requirement_removed'] or record['quoted_line_removed']:
            found.append('removes ' + ' and '.join(name for name, key in (
                ('a requirement', 'requirement_removed'), ('a block-quoted line', 'quoted_line_removed')) if record[key]))
        if record['source_without_item']:
            found.append('changes program source and names no change request item')
        if record['hook']:
            found.append('changes a review hook, its questions, a gate or the hook settings')
        if record.get('names_item_not_launched_on'):
            found.append('names an item no agent of this process was launched on: '
                         + ', '.join(record['names_item_not_launched_on']))
    elif measure == 'source':
        for name, key in (('a graph algorithm outside the graph module', 'graph_algorithm_outside_module'),
                          ('a second producer of a value one module produces', 'second_producer'),
                          ('a source comment that is more than a citation', 'comment_not_a_citation')):
            if record[key]:
                found.append(f"has {name}: {'; '.join(record['at'][:4])}")
    elif measure.startswith('new file '):
        if record.get('prose') and record.get('guide') is None:
            found.append(f'{WARNING} {measure} names no style guide')
        elif record.get('guide_found') is False:
            found.append(f"{WARNING} {measure} names {record['guide']}, and no such file is in the repository")
        elif record.get('adheres') is False:
            found.append(f"{WARNING} {measure} names {record['guide']} and does not adhere to it")
    elif record.get('item_found'):
        found.append(f"{measure}: {len(record['carried'])} of {record['sentences']} sentences carried, "
                     f"{len(record['not_carried'])} not carried; adds what the item does not state: "
                     f"{'yes' if record['added'] else 'no'}. A measurement; the item is the instruction.")
    elif record.get('linked') is False or record.get('item_found') is False:
        found.append(f'{measure}: no such linked document or item')
    return found


def main():
    try:
        json.load(sys.stdin)
    except ValueError:
        pass
    messages = []
    for repo in REPOSITORIES:
        for sha in git(repo, 'log', '-40', '--format=%H').split():
            measured = git(repo, 'notes', '--ref', 'measurements', 'show', sha)
            if not measured or git(repo, 'notes', '--ref', 'warnings', 'show', sha):
                continue
            first = json.loads(measured.splitlines()[0])
            if first.get('process') not in (None, PROCESS) and os.path.exists(f"/tmp/cc-socks/{first['process']}.sock"):
                continue
            found = [w for line in measured.splitlines() if line.strip() for w in warnings(json.loads(line))]
            subprocess.run(['git', '-C', repo, 'notes', '--ref', 'warnings', 'add', '-m',
                            '\n'.join(found) or 'none', sha], capture_output=True)
            messages += [f'{sha[:10]} in {repo}: {w}' for w in found]
    if messages:
        message = '\n'.join(messages) + '\n(design/review-warnings.md rule 14; git notes --ref measurements show <commit>)'
        print(json.dumps({'systemMessage': message,
                          'hookSpecificOutput': {'hookEventName': 'PostToolUse', 'additionalContext': message}}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
