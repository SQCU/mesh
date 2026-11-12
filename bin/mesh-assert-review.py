"""design/review-warnings.md#7-a-commit-adds-no-assertion-and-no-test"""
import datetime
import fcntl
import json
import os
import re
import subprocess
import sys
import tempfile

REPOSITORIES = ('/Users/mdot/dox/mesh', '/Users/mdot/metal-microbench')
LOG = '/Users/mdot/metal-microbench/output_data/assert-test-commits.jsonl'
SEEN = 'review-commentary/assert-review.head'
READER = ['claude', '-p', '--model', 'claude-haiku-5-5', '--tools', '', '--setting-sources', 'user',
          '--no-session-persistence', '--strict-mcp-config']
PROMPT = """You are given one git commit as a diff on stdin. Answer two yes/no questions about what the commit ADDS:
the lines that begin with '+', and the files it creates. Measure only whether the thing is present. Do not consider
why it was added, whether it is justified, or what any message, comment or document says about it.

ASSERT: does an added line contain an assertion? An assertion is an assert statement, call or macro in any language
(assert, _Static_assert, static_assert, NSAssert, XCTAssert..., torch._assert, numpy.testing..., unittest or pytest
assertions, or a function the commit defines to play that part). Raising an error that refuses an unsupported input
is not an assertion.

TEST: does the commit add a test? A test is a test file, test function, test case, sample case, harness, check
function, golden or expected-output file, or script whose purpose is to run code and compare what it produces with
expected or reference values.

Answer with exactly two lines and nothing else:
ASSERT: YES or ASSERT: NO
TEST: YES or TEST: NO"""


def git(repo, *args):
    return subprocess.run(['git', '-C', repo, *args], capture_output=True, text=True).stdout.strip()


def marker(repo):
    path = os.path.join(git(repo, 'rev-parse', '--path-format=absolute', '--git-common-dir'), SEEN)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def new_commits(repo):
    head = git(repo, 'rev-parse', 'HEAD')
    path = marker(repo)
    seen = open(path).read().strip() if os.path.exists(path) else ''
    if not seen or not git(repo, 'rev-parse', '--verify', '--quiet', seen + '^{commit}'):
        advance(repo, head)
        return []
    return git(repo, 'rev-list', '--reverse', f'{seen}..{head}').split()


def advance(repo, sha):
    path = marker(repo)
    with open(path + '.new', 'w') as f:
        f.write(sha)
    os.replace(path + '.new', path)


def measured(repo, sha):
    diff = git(repo, 'show', '--no-color', '--unified=0', '--format=', '--diff-filter=AMR', sha)
    if not diff:
        return False, False
    read = subprocess.run([*READER, '--system-prompt', PROMPT, 'Answer the two questions for this commit.'],
                          input=diff[:400000], capture_output=True, text=True, cwd=tempfile.gettempdir(), timeout=170)
    answers = dict(re.findall(r'^(ASSERT|TEST): (YES|NO)\s*$', read.stdout, re.M))
    if read.returncode or len(answers) != 2:
        raise RuntimeError(f'the reader exited {read.returncode} and said {read.stdout.strip()[:200]!r}')
    return answers['ASSERT'] == 'YES', answers['TEST'] == 'YES'


def logged(repo, sha, asserted, tested):
    record = {'at': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'), 'repository': repo,
              'commit': sha, 'subject': git(repo, 'show', '-s', '--format=%s', sha), 'assertion': asserted,
              'test': tested, 'status': 'noncompliant', 'outstanding': True}
    with open(LOG, 'a+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(record) + '\n')
        f.flush()
        f.seek(0)
        return sum(1 for line in f if line.strip())


def main():
    try:
        json.load(sys.stdin)
    except ValueError:
        pass
    messages = []
    for repo in REPOSITORIES:
        held = open(marker(repo) + '.lock', 'w')
        try:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            continue
        for sha in new_commits(repo):
            try:
                asserted, tested = measured(repo, sha)
            except (RuntimeError, OSError, subprocess.SubprocessError) as error:
                messages.append(f'Assertion and test measurement of {sha[:10]} in {repo} was not made: {error}')
                break
            advance(repo, sha)
            if asserted or tested:
                found = ' and '.join(name for name, present in (('an assertion', asserted), ('a test', tested)) if present)
                count = logged(repo, sha, asserted, tested)
                messages.append(f'NONCOMPLIANT and OUTSTANDING: {sha[:10]} in {repo} adds {found} '
                                f'(design/review-warnings.md rule 7, read by Claude Haiku). {count} such commits are '
                                f'logged in {LOG}.')
    if messages:
        message = '\n'.join(messages)
        print(json.dumps({'systemMessage': message,
                          'hookSpecificOutput': {'hookEventName': 'PostToolUse', 'additionalContext': message}}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
