"""design/review-warnings.md#10-a-brief-is-a-direction-to-read-a-file"""
import datetime
import json
import os
import re
import subprocess
import sys

MESH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIRECTION = re.compile(r'Read (/\S+?)(, (item|section) [A-Z]{0,2}\.?\d+(\.\d+)*|, lines \d+-\d+)? and act on it\.')
FIELDS = {'Agent': ('prompt',), 'Task': ('prompt',), 'SendMessage': ('message',), 'Workflow': ('script', 'args')}


def given(payload):
    held = payload.get('tool_input') or {}
    parts = [held.get(field) for field in FIELDS.get(payload.get('tool_name'), ())]
    return '\n'.join(part if isinstance(part, str) else json.dumps(part) for part in parts if part)


def committed(path):
    folder = os.path.dirname(path)
    inside = subprocess.run(['git', '-C', folder, 'ls-files', '--error-unmatch', path], capture_output=True)
    clean = subprocess.run(['git', '-C', folder, 'diff', '--quiet', 'HEAD', '--', path], capture_output=True)
    return os.path.isfile(path) and inside.returncode == 0 and clean.returncode == 0


def refusal(payload):
    tool, held = payload.get('tool_name'), payload.get('tool_input') or {}
    if tool == 'Workflow':
        script = held.get('scriptPath')
        if held.get('script') or held.get('name') or not (script and committed(script)):
            return 'a Workflow runs only from a scriptPath that is a committed, unmodified file'
        return None
    text = given(payload)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None if tool == 'SendMessage' else 'the call has no text'
    matches = [DIRECTION.fullmatch(line) for line in lines]
    inlined = [line for line, match in zip(lines, matches) if not (match and committed(match.group(1)))]
    if inlined:
        return (f"{len(inlined)} line(s) are not a direction to read a committed, unmodified file. A brief is lines of "
                f"the form 'Read /absolute/path and act on it.', with ', item N', ', section N' or ', lines N-M' "
                f"allowed after the path, and nothing else. First such line: {inlined[0][:160]!r}")
    named = os.path.basename(matches[0].group(1))
    if tool in ('Agent', 'Task') and held.get('description') != named:
        return f"the description must be the file's name, {named!r}"
    return None


def main():
    payload = json.load(sys.stdin)
    if payload.get('tool_name') not in FIELDS:
        return 0
    why = refusal(payload)
    if why is None:
        return 0
    reason = f"{payload.get('tool_name')} refused (design/review-warnings.md rule 10): {why}"
    line = json.dumps({'at': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
                       'session': payload.get('session_id'), 'tool': payload.get('tool_name'), 'warning': reason})
    subprocess.run(['git', '-C', MESH, 'notes', '--ref', 'brief-refusals', 'append', '-m', line, 'HEAD'],
                   capture_output=True)
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
