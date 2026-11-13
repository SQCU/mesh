"""design/review-warnings.md#10-a-brief-is-a-direction-to-read-a-file"""
import datetime
import json
import os
import re
import sys
from mesh_notes import append

MESH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIRECTION = re.compile(r'Read (/.+?)(, (item|section) [A-Z]{0,2}\.?\d+(\.\d+)*|, lines \d+-\d+)? and act on it\.')
FIELDS = {'Agent': ('prompt',), 'Task': ('prompt',), 'SendMessage': ('message',), 'Workflow': ('script', 'args')}


def given(payload):
    held = payload.get('tool_input') or {}
    parts = [held.get(field) for field in FIELDS.get(payload.get('tool_name'), ())]
    return '\n'.join(part if isinstance(part, str) else json.dumps(part) for part in parts if part)


def given_lines(payload):
    return [line.strip() for line in given(payload).splitlines() if line.strip()]


def refusal(payload):
    tool, held = payload.get('tool_name'), payload.get('tool_input') or {}
    if tool == 'Workflow' and held.get('scriptPath') and not given(payload):
        return None
    text = given(payload)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None if tool == 'SendMessage' else 'the call has no text'
    matches = [DIRECTION.fullmatch(line) for line in lines]
    inlined = [line for line, match in zip(lines, matches) if not (match and os.path.isfile(match.group(1)))]
    if inlined:
        return (f"{len(inlined)} line(s) are not a direction to read an existing file. A brief is lines of "
                f"the form 'Read /absolute/path and act on it.', with ', item N', ', section N' or ', lines N-M' "
                f"allowed after the path, and nothing else. First such line: {inlined[0][:160]!r}")
    return None


def main():
    payload = json.load(sys.stdin)
    if payload.get('tool_name') not in FIELDS:
        return 0
    why = refusal(payload)
    if why is None:
        held = payload.get('tool_input') or {}
        given = [f"{m.group(1)}{m.group(2) or ''}" for m in map(DIRECTION.fullmatch, given_lines(payload)) if m]
        line = json.dumps({'at': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
                           'session': payload.get('session_id'), 'agent': payload.get('agent_id'),
                           'process': os.path.basename(os.environ.get('CLAUDE_CODE_MESSAGING_SOCKET', '')).split('.')[0] or None,
                           'tool': payload.get('tool_name'), 'given': given or [held.get('scriptPath')]})
        append(MESH, 'briefs', line, 'HEAD')
        return 0
    reason = f"{payload.get('tool_name')} refused (design/review-warnings.md rule 10): {why}"
    line = json.dumps({'at': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
                       'session': payload.get('session_id'), 'tool': payload.get('tool_name'), 'warning': reason})
    append(MESH, 'brief-refusals', line, 'HEAD')
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
