"""design/review-warnings.md#8-a-documents-claims-are-measured"""
import datetime
import fcntl
import json
import os
import sys

MEASURED = '/Users/mdot/metal-microbench/output_data/document-claims-commits.jsonl'
WARNINGS = '/Users/mdot/metal-microbench/output_data/document-claims-warnings.jsonl'


def main():
    try:
        json.load(sys.stdin)
    except ValueError:
        pass
    if not os.path.exists(MEASURED):
        return 0
    messages = []
    with open(WARNINGS, 'a+') as out:
        try:
            fcntl.flock(out, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return 0
        out.seek(0)
        warned = {json.loads(line)['commit'] for line in out if line.strip()}
        for line in open(MEASURED):
            record = json.loads(line) if line.strip() else None
            if not record or record['commit'] in warned or record['quote'] or not (record['fact'] or record['requirement']):
                continue
            claims = ' and '.join(name for name in ('fact', 'requirement') if record[name])
            warning = (f"WARNING: {record['commit'][:10]} in {record['repository']} adds claims of {claims} to a "
                       f"document and cites no verbatim user turn (design/review-warnings.md rule 8, read by Claude "
                       f"Haiku): {record['subject'][:160]}")
            out.write(json.dumps({'at': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
                                  'repository': record['repository'], 'commit': record['commit'],
                                  'fact': record['fact'], 'requirement': record['requirement'], 'quote': False,
                                  'warning': warning}) + '\n')
            out.flush()
            warned.add(record['commit'])
            messages.append(warning)
    if messages:
        message = '\n'.join(messages) + f'\nEach is appended to {WARNINGS}.'
        print(json.dumps({'systemMessage': message,
                          'hookSpecificOutput': {'hookEventName': 'PostToolUse', 'additionalContext': message}}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
