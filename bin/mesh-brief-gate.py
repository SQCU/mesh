"""design/review-warnings.md#10-a-brief-is-a-direction-to-read-a-file"""
import datetime
import fcntl
import json
import os
import re
import sys

LOG = '/Users/mdot/metal-microbench/output_data/agent-brief-warnings.jsonl'
DIRECTION = re.compile(r'Read (/\S+?)(, (item|section) [A-Z]{0,2}\.?\d+(\.\d+)*|, lines \d+-\d+)? and act on it\.')
FIELDS = {'Agent': ('prompt',), 'Task': ('prompt',), 'SendMessage': ('message',), 'Workflow': ('script', 'args')}


def given(payload):
    held = payload.get('tool_input') or {}
    parts = [held.get(field) for field in FIELDS.get(payload.get('tool_name'), ())]
    return '\n'.join(part if isinstance(part, str) else json.dumps(part) for part in parts if part)


def main():
    payload = json.load(sys.stdin)
    if payload.get('tool_name') not in FIELDS:
        return 0
    text = given(payload)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    inlined = [line for line in lines
               if not (DIRECTION.fullmatch(line) and os.path.isfile(DIRECTION.fullmatch(line).group(1)))]
    if not inlined and lines:
        return 0
    if not lines and payload.get('tool_name') in ('Workflow', 'SendMessage'):
        return 0
    reason = (f"{payload.get('tool_name')} was given {len(inlined)} line(s) that are not a direction to read a file "
              f"(design/review-warnings.md rule 10). A brief is lines of the form 'Read /absolute/path and act on "
              f"it.', the path an existing file, with ', item N', ', section N' or ', lines N-M' allowed after it, and nothing else; put anything more in a committed file and name the file. First such line: "
              f"{(inlined or [''])[0][:160]!r}")
    with open(LOG, 'a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps({'at': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
                            'session': payload.get('session_id'), 'tool': payload.get('tool_name'),
                            'inlined_lines': len(inlined), 'characters': len(text), 'first': (inlined or [''])[0][:300],
                            'warning': reason}) + '\n')
    print(json.dumps({'hookSpecificOutput': {'hookEventName': 'PreToolUse', 'permissionDecision': 'deny',
                                             'permissionDecisionReason': reason}}))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as error:
        print(json.dumps({'hookSpecificOutput': {'hookEventName': 'PreToolUse', 'permissionDecision': 'deny',
                                                 'permissionDecisionReason': f'the brief gate failed: {error}'}}))
        sys.exit(0)
