"""Complete-function feedback keeps delayed GPU evidence tied to its operands."""
import ast
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

RDMA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RDMA))
from controller import WholeFunctionController


class Controller(WholeFunctionController):
    def __init__(self):
        super().__init__({'tp': {'parts': [4, 2], 'grain': 1, 'low': [1, 1], 'high': [5, 5]}},
                         workload={'function': 'tp', 'units': 6})
        self.completed_records = []

    def completed(self, seconds, **evidence):
        self.completed_records.append((seconds, evidence))
        return super().completed(seconds, **evidence)


class Timer:
    def __init__(self, events):
        self.done, self.marks, self.events = False, 0, events

    def mark(self):
        self.marks += 1
        self.events.append(('marker', self.marks))

    def ready(self):
        return self.done

    def read(self):
        if not self.done:
            raise AssertionError('read must never wait for unfinished work')
        return [10.0, 10.03]


def environment():
    path = RDMA / 'torch-mesh/torch_mesh/partition.py'
    class Clean(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            return None
    names = {'evaluations', '_balance', 'evaluation_state'}
    tree = ast.Module(body=[Clean().visit(n) for n in ast.parse(path.read_text()).body
                           if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[])
    events, timers, state, controller = [], [], {}, Controller()
    group = SimpleNamespace(plan_active=lambda active: events.append(('active', active)), comm='native-comm')
    mesh = SimpleNamespace(get_group=lambda _: group, device_type='mps')
    # Objects used as a registry key need identity hashing.
    mesh = type('Mesh', (), dict(get_group=lambda self, _: group, device_type='mps'))()
    operand = SimpleNamespace(capacity=[4, 2], units=6, parts=SimpleNamespace(tolist=lambda: [4, 2]))
    def timing(_):
        timer = Timer(events)
        timers.append(timer)
        return timer
    def shared(held, selected, words):
        events.append(('feedback', tuple(words)))
        return [words, [words[0], words[1] * 2]]
    env = dict(WholeFunctionController=lambda *a, **kw: controller, _BALANCING=state,
               _PLANS={(mesh, 'tp'): object()}, _PREPARATIONS={}, _function_timings=timing,
               operand=lambda *_: operand, _shared=shared,
               _attach=lambda *args: events.append(('attach', args[-1])),
               _relay=lambda *_: events.append(('relay',)), _replan=lambda *_: events.append(('replan',)))
    exec(compile(tree, str(path), 'exec'), env)
    return env, mesh, controller, timers, events


class FunctionFeedbackTests(unittest.TestCase):
    def test_complete_function_maximum_includes_communication_and_labels_pending_calls(self):
        env, mesh, controller, timers, events = environment()
        calls = env['evaluations'](mesh, 'tp', 4)
        next(calls)
        next(calls)  # first call submitted but still running: no feedback timing
        self.assertEqual(controller.completed_records, [])
        timers[0].done = True
        next(calls)  # observes call zero, advances the whole-function proposal
        self.assertEqual(len(controller.completed_records), 1)
        seconds, record = controller.completed_records[0]
        self.assertAlmostEqual(seconds, .06)
        self.assertEqual(record['proposal']['id'], 0)
        self.assertEqual(record['sequence'], 0)
        timers[1].done = True  # issued under old proposal; cannot train proposal one
        next(calls)
        self.assertEqual(len(controller.completed_records), 1)
        self.assertEqual(env['evaluation_state'](mesh, 'tp')['stale_completed'], 1)
        calls.close()
        self.assertEqual(timers[-1].marks, 1, 'abandoned function must not manufacture completion')
        self.assertEqual(events[-1], ('active', False))

    def test_plan_scope_excludes_control_exchange_and_closes_on_abandonment(self):
        env, mesh, _, _, events = environment()
        calls = env['evaluations'](mesh, 'tp', 2)
        next(calls)
        calls.close()
        self.assertEqual(events, [('active', False), ('feedback', (-1, 0.0)), ('active', True), ('marker', 1), ('active', False)])
        self.assertEqual(env['evaluation_state'](mesh, 'tp')['pending'], 0)

    def test_different_workload_cannot_share_feedback_history(self):
        env, mesh, _, _, _ = environment()
        calls = env['evaluations'](mesh, 'tp', 1)
        next(calls)
        calls.close()
        with self.assertRaisesRegex(ValueError, 'different whole-function'):
            next(env['evaluations'](mesh, 'tp', 1, workload='unrelated DNN'))


if __name__ == '__main__':
    unittest.main()
