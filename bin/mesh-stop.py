"""design/review-warnings.md#16-what-an-agent-leaves-and-says"""
import datetime
import json
import os
import re
import subprocess
import sys
import tempfile

REPOSITORIES = ('/Users/mdot/dox/mesh', '/Users/mdot/metal-microbench')
PROCESS = os.path.basename(os.environ.get('CLAUDE_CODE_MESSAGING_SOCKET', '')).split('.')[0] or None
HAIKU = ['claude', '-p', '--model', 'claude-haiku-5-5', '--tools', '', '--setting-sources', 'user',
         '--no-session-persistence', '--strict-mcp-config']
REPORT = """You are given the last message an agent wrote when it stopped. Measure only whether each thing is
present. Do not consider whether it is justified.

UNDONE: the message says part of what the agent was given was not done, not started, left, deferred or stopped.
ASKS: the message asks or tells its reader to do something, or says what it needs from someone else.
CLAIMS: the message states what code does or what holds without naming the commit that shows it.

Answer with exactly three lines, each ending in YES or NO, and nothing else:
UNDONE:
ASKS:
CLAIMS:"""


def git(repo, *args):
    return subprocess.run(['git', '-C', repo, *args], capture_output=True, text=True).stdout.strip()


def said(payload):
    if isinstance(payload.get('last_assistant_message'), str):
        return payload['last_assistant_message']
    path = payload.get('agent_transcript_path') or payload.get('transcript_path')
    last = None
    if path and os.path.exists(path):
        for line in open(path):
            if '"type":"assistant"' not in line:
                continue
            try:
                content = json.loads(line)['message']['content']
            except (ValueError, KeyError, TypeError):
                continue
            text = ' '.join(c.get('text', '') for c in content if isinstance(c, dict) and c.get('type') == 'text')
            last = text or last
    return last


def main():
    payload = json.load(sys.stdin)
    record = {'at': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
              'event': payload.get('hook_event_name'), 'process': PROCESS, 'session': payload.get('session_id'),
              'agent': payload.get('agent_id') or payload.get('agent_type'), 'given_fields': sorted(payload)}
    record['uncommitted'] = {repo: git(repo, 'diff', '--name-only', 'HEAD').splitlines()
                             for repo in REPOSITORIES}
    try:
        text = said(payload) if record['event'] == 'SubagentStop' else None
        if text:
            read = subprocess.run([*HAIKU, '--system-prompt', REPORT, 'Answer.'], input=text[:200000],
                                  capture_output=True, text=True, cwd=tempfile.gettempdir(), timeout=120)
            bits = dict(re.findall(r'^(UNDONE|ASKS|CLAIMS): (YES|NO)\s*$', read.stdout, re.M))
            record['report'] = ({name.lower(): value == 'YES' for name, value in bits.items()} if len(bits) == 3
                                else {'measured': False, 'reason': read.stdout.strip()[:120]})
        elif record['event'] == 'SubagentStop':
            record['report'] = {'measured': False, 'reason': 'no last message found'}
    except Exception as error:
        record['report'] = {'measured': False, 'reason': f'{type(error).__name__}: {error}'[:200]}
    subprocess.run(['git', '-C', REPOSITORIES[0], 'notes', '--ref', 'stops', 'append', '-m', json.dumps(record), 'HEAD'],
                   capture_output=True)
    found = [f"{len(paths)} tracked file(s) uncommitted in {repo}: {', '.join(paths[:6])}"
             for repo, paths in record['uncommitted'].items() if paths]
    report = record.get('report') or {}
    found += [text for key, text in (('undone', 'its last message says part of its work was not done'),
                                     ('asks', 'its last message asks its reader for something'),
                                     ('claims', 'its last message states what holds without naming a commit'))
              if report.get(key)]
    if found and record['event'] == 'SubagentStop':
        print(json.dumps({'systemMessage': 'An agent stopped (design/review-warnings.md rule 16): ' + '; '.join(found)}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
