"""design/review-warnings.md#16-what-an-agent-leaves-and-says"""
import json
import os
import sys

PROCESS = os.path.basename(os.environ.get('CLAUDE_CODE_MESSAGING_SOCKET', '')).split('.')[0]


def main():
    payload = json.load(sys.stdin)
    if PROCESS:
        folder = os.path.expanduser('~/.cache/mesh-agent')
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, PROCESS + '.new'), 'w') as f:
            f.write(str(payload.get('agent_id') or payload.get('agent_type') or 'main'))
        os.replace(os.path.join(folder, PROCESS + '.new'), os.path.join(folder, PROCESS))
    return 0


if __name__ == '__main__':
    sys.exit(main())
