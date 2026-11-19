"""External observation of a bound public Metal invocation around its native issue span."""

from collections import Counter


def body_guard(name, probe):
    if name == 'queue_wait':
        return ' && !self->handoff'
    if name == 'host_notify':
        return ' && !self->release'
    if 'waitUntilSignaledValue' in probe:
        return ' && !self->ending'
    return ''


def probes(contract):
    lines = []

    def probe(where, name, values=(), when=''):
        fmt = 'mesh-public %d ' + name + ' %d' * len(values) + r'\n'
        predicate = ' /' + when + '/' if when else ''
        lines.append(
            '%s%s { printf("%s", timestamp%s); }'
            % (where, predicate, fmt, ''.join(', (uint64_t)(' + v + ')' for v in values))
        )

    for where, flag in (
        ('pid$target::*WaitForMPSStream*:', 'handoff'),
        ('pid$target::MetalExternalRelease:', 'release'),
        ('pid$target::MetalProgramRunArguments:', 'issuing'),
        ('pid$target::MetalProgramEnd:', 'ending'),
    ):
        lines.extend(
            (f'{where}entry {{ self->{flag}++; }}', f'{where}return /self->{flag}/ {{ self->{flag}--; }}')
        )
    for selector, name, values in (
        ('encodeSignalEvent?value?', 'mps_signal', ('arg0', 'arg2', 'arg3')),
        ('commit', 'mps_commit', ('arg0',)),
    ):
        where = 'objc$target:*CommandBuffer*:-' + selector + ':'
        probe(where + 'entry', name, values, 'self->inapi == 0')
        lines.extend((where + 'entry { self->inapi++; }', where + 'return /self->inapi/ { self->inapi--; }'))
    where = 'objc$target:*CommandQueue*:-commandBuffer:'
    probe(where + 'entry', 'checkout', ('arg0',), 'self->handoff && !self->checkout')
    lines.append(where + 'entry /self->handoff/ { self->checkout++; }')
    probe(where + 'return', 'checked_out', ('arg1',), 'self->checkout == 1')
    lines.append(where + 'return /self->checkout/ { self->checkout--; }')
    probe('pid$target::dispatch_semaphore_wait:entry', 'checkout_wait', ('arg0', 'arg1'), 'self->checkout')
    probe('pid$target::dispatch_semaphore_wait:return', 'checkout_result', ('arg1',), 'self->checkout')
    probe(
        'objc$target::-waitForEvent?value?:entry',
        'ingress',
        ('arg0', 'arg2', 'arg3'),
        'self->handoff && !self->inapi',
    )
    probe(
        'pid$target::MetalProgramRunArguments:entry',
        'issue',
        ('arg0', 'arg2', 'arg3', 'arg4', 'arg7', '(uint32_t)arg8'),
    )
    probe(
        'objc$target::-signalEvent?value?:entry',
        'completion_signal',
        ('arg0', 'arg2', 'arg3'),
        f'!self->inapi && arg2 == {int(contract["ended"])}',
    )
    probe(
        'objc$target::-setSignaledValue?:entry',
        'completion_store',
        ('arg0', 'arg2'),
        f'!self->inapi && arg0 == {int(contract["ended"])}',
    )
    probe('pid$target::MetalProgramRunArguments:return', 'issued', ('arg1',))
    probe(
        'objc$target::-notifyListener?atValue?block?:entry',
        'residency',
        ('arg0', 'arg3'),
        'self->release && !self->inapi',
    )
    probe('pid$target::MetalProgramEnd:entry', 'end', ('arg0', 'arg1'))
    probe('pid$target::MetalEventWait:entry', 'completion', ('arg0', 'arg1'), 'self->ending')
    where = 'objc$target::-signaledValue:'
    lines.append(where + 'entry /self->ending/ { self->reading++; self->readEvent = arg0; }')
    probe(where + 'return', 'value', ('self->readEvent', 'arg1'), 'self->reading == 1')
    lines.append(where + 'return /self->reading/ { self->reading--; }')
    where = 'objc$target::-waitUntilSignaledValue?timeoutMS?:'
    probe(where + 'entry', 'host_wait', ('arg0', 'arg2', 'arg3'), 'self->ending && !self->inapi')
    probe(where + 'return', 'wait_result', ('arg1',), 'self->ending && self->inapi == 1')
    group = int(contract['notifications'])
    for action in ('enter', 'leave'):
        probe(f'pid$target::dispatch_group_{action}:entry', 'group_' + action, ('arg0',), f'arg0 == {group}')
    probe('pid$target::dispatch_group_wait:entry', 'group_wait', ('arg0',), 'self->ending || self->issuing')
    probe('pid$target::MetalProgramEnd:return', 'ended', ('arg1',))
    where = 'pid$target::MetalWaitPrefixes:'
    probe(where + 'entry', 'prefixes', ('arg0', 'arg1', 'arg2', 'arg3'))
    probe('pid$target::MetalWaitPrefix:entry', 'prefix', ('arg0', 'arg1'))
    probe('pid$target::MetalWaitPrefix:return', 'acquired', ('arg1',))
    probe(where + 'return', 'prefix_done', ('arg1',))
    where = 'pid$target::mesh_host_wait_fields:'
    probe(where + 'entry', 'fields', ('arg1', 'arg2', 'arg3', 'arg4', 'arg5'), '!self->issuing')
    probe(where + 'return', 'field_done', ('arg1',), '!self->issuing')
    return lines


def compare(contract, lines, submits):
    events = sorted(
        (int(row[1]), row[2], tuple(map(int, row[3:])))
        for line in lines
        if (row := line.split()) and row[0] == 'mesh-public'
    )
    counts = Counter(name for _, name, _ in events)
    failures = []
    groups = [r for r in events if r[1] in ('group_enter', 'group_leave')]
    enters = leaves = 0
    for _, name, args in groups:
        enters += name == 'group_enter'
        leaves += name == 'group_leave'
        if args != (contract['notifications'],) or leaves > enters:
            failures.append('completion callback group identity or balance differs')
    expected = submits + int(contract['residency'])
    if enters != expected or leaves != expected:
        failures.append(f'callback group entered {enters} times and left {leaves} times, expected {expected} each')
    rows = iter(r for r in events if r[1] not in ('group_enter', 'group_leave'))
    current = next(rows, None)

    def take(name, expected=None):
        nonlocal current
        if current is None or current[1] != name:
            raise ValueError(f'expected {name}, observed {current and current[1:]}')
        args = current[2]
        current = next(rows, None)
        if expected is not None and args != tuple(expected):
            raise ValueError(f'{name} expected {tuple(expected)}, observed {args}')
        return args

    ended = contract['ended']
    first = contract['first']
    target = first + 1

    def checkouts():
        buffer = None
        while current and current[1] == 'checkout':
            take('checkout')
            while current and current[1] == 'checkout_wait':
                take('checkout_wait')
                take('checkout_result', (0,))
            (buffer,) = take('checked_out')
        return buffer

    try:
        if contract['ingress']:
            buffer = checkouts()
            command, event, value = take('mps_signal')
            if buffer is not None and command != buffer:
                raise ValueError('MPS signalled a different command buffer from its checkout')
            checkouts()
            take('mps_commit', (command,))
            take('ingress', (contract['queue'], event, value))
        take(
            'issue', (contract['program'], first, 1, ended, contract['argument_table'], len(contract['arguments']))
        )
        if current and current[1] == 'completion_store':
            take('completion_store', (ended, target))
        else:
            take('completion_signal', (contract['queue'], ended, target))
        if contract['residency']:
            take('residency', (ended, target))
        take('issued', (0,))
        take('end', (contract['program'], ended))
        take('completion', (ended, target))
        previous = -1
        while True:
            observed, seen = take('value')
            if observed != ended:
                raise ValueError('completion value read from a different shared event')
            if seen <= previous:
                raise ValueError('completion wait retried without progress')
            if seen >= target:
                break
            take('host_wait', (ended, target, 5000))
            if take('wait_result')[0]:
                break
            previous = seen
        take('group_wait', (contract['notifications'],))
        if groups and current and groups[-1][0] > current[0]:
            raise ValueError('completion callback group released after MetalProgramEnd returned')
        take('ended', (0,))
        if contract['prefixes']:
            take('prefixes', (contract['prefix_base'], contract['prefix_table'], len(contract['prefixes']), first))
            for word, base, stride in contract['prefixes']:
                take('prefix', (contract['prefix_base'] + word, base + stride * first))
                take('acquired', (0,))
            take('prefix_done', (0,))
        for plan, transfer, generation, address, mask in contract['fields']:
            take('fields', (plan, transfer, generation, len(mask), address))
            take('field_done', (0,))
        if current:
            raise ValueError(f'unexpected {current[1:]} after the last public operation')
    except (StopIteration, ValueError) as error:
        failures.append('public invocation mismatch: ' + str(error))
    return dict(counts), failures
