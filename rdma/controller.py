"""One complete-function controller for placement and engine shares.

It never selects a measured winner or turns a failed function into a synthetic timing. The immutable proposal is the
evidence identity; a caller must not change its operands inside a function. Routes are the bridges' own, per page, and
are never a decision here.
"""
import copy
from whole import FunctionFlow


class WholeFunctionController:
    VERSION = 'whole-function-controller-v2'

    def __init__(self, decisions, workload, schedule=None, calibrate=True, history_limit=64, **options):
        if not decisions:
            raise ValueError('a function controller needs declared decisions')
        self.workload = copy.deepcopy(workload)
        self.schedule = copy.deepcopy(schedule)
        self.flow = FunctionFlow(decisions, **options)
        self.history_limit = max(4, int(history_limit))
        self.failures = []
        self._proposal = self.flow.ask(explore=not calibrate)

    def proposal(self, explore=None):
        if explore is False:
            return copy.deepcopy(self.flow.ask(explore=False))
        return copy.deepcopy(self._proposal)

    ask = proposal

    @property
    def parts(self):
        return self.proposal()['parts']

    @property
    def round(self):
        return self.flow.round

    @property
    def history(self):
        return self.flow.history

    def observe(self, observation):
        expected = self._proposal
        if (observation.get('id') != expected['id'] or observation.get('parts') != expected['parts']
                or observation.get('workload') != self.workload):
            raise ValueError('whole-call evidence must name the exact realized proposal and workload')
        record = {**observation, 'phase': expected['phase']}
        self._proposal = self.flow.observe(record)
        self.flow.history[:] = self.flow.history[-self.history_limit:]
        return self.proposal()

    def completed(self, seconds, proposal=None, **evidence):
        """Submit one actually completed whole invocation."""
        proposal = self.proposal() if proposal is None else proposal
        return self.observe({**evidence, 'id': proposal['id'], 'parts': proposal['parts'],
                             'scope': 'function', 'complete': True,
                             'workload': self.workload, 'seconds': seconds})

    def failed(self, reason, proposal=None):
        """Record a structural/runtime failure. No rollback and no invented cost."""
        proposal = self.proposal() if proposal is None else proposal
        self.failures.append({'proposal': copy.deepcopy(proposal), 'reason': str(reason)})
        self.failures[:] = self.failures[-self.history_limit:]
        return self.proposal()

    def state(self):
        return {'version': self.VERSION, 'workload': copy.deepcopy(self.workload),
                'schedule': copy.deepcopy(self.schedule), 'history_limit': self.history_limit,
                'flow': self.flow.state(), 'proposal': self.proposal(), 'failures': copy.deepcopy(self.failures)}

    @classmethod
    def of(cls, state):
        if state.get('version') != cls.VERSION:
            raise ValueError('unsupported whole-function controller state')
        obj = cls.__new__(cls)
        obj.workload, obj.schedule = copy.deepcopy(state['workload']), copy.deepcopy(state.get('schedule'))
        obj.history_limit = int(state.get('history_limit', 64))
        obj.flow = FunctionFlow.of(state['flow'])
        obj._proposal, obj.failures = copy.deepcopy(state['proposal']), copy.deepcopy(state.get('failures', []))
        return obj
