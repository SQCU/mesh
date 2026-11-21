"""Whole-sequence state survives service processes and never trains on ticks."""
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'torch-mesh')]
from controller import WholeFunctionController
from torch_mesh import Recording
from torch_mesh.invocation import Invocation


class Store:
    def __init__(self):
        self.values = {}

    def set(self, key, value):
        self.values[key] = value

    def get(self, key):
        return self.values.get(key, b'0.03' if key.endswith('/seconds/1') else None)


class InvocationState(unittest.TestCase):
    def test_sequence_observation_persists_and_restores_before_operand_construction(self):
        from torch_mesh import invocation as module
        events, store = [], Store()
        group = types.SimpleNamespace(rank=lambda: 0, size=lambda: 2, group_name='full', comm='native',
                    fabric=types.SimpleNamespace(cables=[]),
                    reset_fused_configuration=lambda: events.append('retire-generations'),
                    plan_active=lambda active: events.append(('active', active)),
                    plan_demands=lambda: {(0, 1): 12})
        mesh = types.SimpleNamespace(get_group=lambda _: group)
        operand = types.SimpleNamespace(units=6, parts=types.SimpleNamespace(tolist=lambda: [4, 2]))
        workload = {'function': 'entire-lstm', 'steps': 17, 'layers': 3}
        controller = WholeFunctionController({'tp': {'parts': [4, 2], 'grain': 1, 'low': [0, 0], 'high': [6, 6]}},
                                             workload=workload, topology={'nodes': 2, 'cables': []})
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(module.dist.distributed_c10d, '_get_default_store', return_value=store), \
             patch.object(module.partition, 'operand', return_value=operand), \
             patch.object(module.partition, '_attach', side_effect=lambda *_args: events.append(('attach', _args[-1]))), \
             patch.object(module.partition, 'plan', side_effect=lambda *_args: events.append(('declare', _args[-1]))), \
             patch.object(module.partition, '_replan', side_effect=lambda *_: events.append('redeclare')), \
             patch.object(WholeFunctionController, 'from_comm', return_value=controller):
            state = Path(directory) / 'sequence.json'
            first = Invocation(mesh, 'tp', workload, state)
            first.realize(['input-broadcast', 'all-17-ticks'], prepare=lambda: events.append('prepared'))
            decisions = WholeFunctionController.from_comm.call_args.args[0]['tp']
            self.assertEqual(decisions['low'], [0, 0])
            self.assertEqual(decisions['high'], [6, 6])
            self.assertEqual(len(controller.history), 0)
            first.begin()
            self.assertFalse(state.exists(), 'no per-tick proposal/checkpoint may be manufactured')
            summary = first.completed(.01, useful_work_bytes=12)
            self.assertEqual(summary['seconds'], .03)
            self.assertEqual(controller.flow.serial, 1)
            self.assertEqual(controller.flow.metric_contract[:2], ['logical_output_bytes', 12.0])
            saved = json.loads(state.read_text())
            second = Invocation(mesh, 'tp', workload, state)
            self.assertEqual(events[-1], ('attach', saved['proposal']['parts']['tp']))
            self.assertEqual(second.controller.state(), saved)
            self.assertEqual(events[:4], [('declare', ['input-broadcast', 'all-17-ticks']),
                                         'retire-generations', 'redeclare', ('active', False)])
            with self.assertRaisesRegex(ValueError, 'another complete function'):
                Invocation(mesh, 'tp', {**workload, 'steps': 18}, state)

    def test_restored_relay_only_rank_has_zero_compute_share(self):
        from torch_mesh import invocation as module
        store, attached = Store(), []
        group = types.SimpleNamespace(rank=lambda: 0, size=lambda: 2, group_name='zero')
        mesh = types.SimpleNamespace(get_group=lambda _: group)
        controller = WholeFunctionController({'tp': {'parts': [0, 6], 'grain': 1, 'low': [0, 0], 'high': [6, 6]}},
                                             workload='sequence', topology={'nodes': 2, 'cables': []})
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(module.dist.distributed_c10d, '_get_default_store', return_value=store), \
             patch.object(module.partition, 'operand', return_value=types.SimpleNamespace(units=6)), \
             patch.object(module.partition, '_attach', side_effect=lambda *_args: attached.append(_args[-1])):
            state = Path(directory) / 'zero.json'
            state.write_text(json.dumps(controller.state()))
            restored = Invocation(mesh, 'tp', 'sequence', state)
            self.assertEqual(attached, [[0, 6]])
            self.assertEqual(restored.controller.proposal()['parts']['tp'], [0, 6])

    def test_recorded_schedule_preserves_full_chunks_and_remainder(self):
        recording = Recording(lambda: None, 1, 3)
        recording.declarations = {3: ({'full': [('three', 3)]}, ['shape-three']),
                                  1: ({'full': ['one']}, ['shape-one'])}
        group = types.SimpleNamespace(group_name='full')
        calls, forms = recording.schedule(group, 7)
        self.assertEqual(calls, [('three', 3), ('three', 3), 'one'])
        self.assertEqual(forms, ['shape-three', 'shape-three', 'shape-one'])
        with self.assertRaisesRegex(ValueError, 'must be compiled'):
            recording.schedule(group, 8)


if __name__ == '__main__':
    unittest.main()
