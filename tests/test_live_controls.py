from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import tempfile
import uuid
from unittest.mock import patch

import numpy as np
import torch
from aiohttp import web

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parents[1]), str(ROOT / 'vendor' / 'ardy')]
server = SimpleNamespace(PromptServer=SimpleNamespace(instance=SimpleNamespace(routes=web.RouteTableDef())))
previous_server = sys.modules.get('server')
sys.modules['server'] = server
try:
    import preview
finally:
    if previous_server is None:
        sys.modules.pop('server')
    else:
        sys.modules['server'] = previous_server


class LiveControlTests(unittest.TestCase):
    def test_saved_take_skips_model_loading(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(preview.folder_paths, 'get_output_directory', return_value=folder):
            control_id = str(uuid.uuid4())
            path = preview.recording_path(control_id)
            path.parent.mkdir(parents=True)
            np.savez(path, frames=2, fps=20, parents=[-1], joint_names=['root'],
                     posed_joints=np.zeros((2, 1, 3), dtype=np.float32))
            node = preview.ArdyLivePreview()
            self.assertEqual(node.check_lazy_status(control_id), [])
            result = node.execute(None, 'changed prompt', 7, 10, 2, 1, 5, control_id, '3', '[]')
            self.assertEqual(result['result'][0]['frames'], 2)
            self.assertEqual(result['result'][0]['joint_names'], ['root'])
            self.assertEqual(node.check_lazy_status(str(uuid.uuid4())), ['model'])
            with self.assertRaises(ValueError):
                preview.recording_path('../outside')

    def test_append_preserves_pending_route_without_revisiting_consumed_points(self):
        controls = preview.LiveControls('walk', 1, [[1, 0], [2, 0]])
        controls.advance_waypoints([[1, 0], [2, 0]], np.array([[1, 0]]))
        controls.update({'append_waypoints': [[3, 0], [4, 0]]})
        self.assertEqual(controls.snapshot()['waypoints'], [[2, 0], [3, 0], [4, 0]])

    def test_blank_waypoints_clear_route(self):
        for value in ('', ' \t\r\n', '[]'):
            with self.subTest(value=value):
                self.assertEqual(preview.parse_waypoints(value), [])
                controls = preview.LiveControls('walk', 1, [[1, 2]])
                controls.update({'waypoints': value})
                self.assertEqual(controls.snapshot()['waypoints'], [])

    def test_route_crosses_corner_without_returning_to_old_waypoint(self):
        model = SimpleNamespace(gen_horizon_len=4, num_frames_per_token=1, device='cpu', skeleton=None,
            motion_rep=SimpleNamespace(fps=2, create_conditions_from_constraints=lambda *a, **k: (torch.zeros(1), torch.zeros(1))))
        with patch.object(preview, 'Root2DConstraintSet') as constraint:
            preview.waypoint_conditions(model, 0, [0, 0], None, 1, [[1, 0], [1, 1]])
            np.testing.assert_allclose(constraint.call_args.args[2].numpy(), [[.5, 0], [1, 0], [1, .5], [1, 1]])
        controls = preview.LiveControls('walk', 1, [[1, 0], [1, 1]])
        controls.advance_waypoints([[1, 0], [1, 1]], np.array([[.5, 0], [1, 0], [1, 1]]))
        self.assertEqual(controls.snapshot()['waypoints'], [[1, 1]])

    def test_new_route_is_not_consumed_by_previous_chunk(self):
        controls = preview.LiveControls('walk', 1, [[1, 0], [1, 1]])
        controls.update({'waypoints': [[5, 5], [6, 6]]})
        controls.advance_waypoints([[1, 0], [1, 1]], np.array([[1, 0], [1, 1]]))
        self.assertEqual(controls.snapshot()['waypoints'], [[5, 5], [6, 6]])

    def test_keyboard_direction_and_validation(self):
        controls = preview.LiveControls('walk', 1)
        controls.update({'direction': [1, -1]})
        self.assertEqual(controls.snapshot()['direction'], [1, -1])
        controls.update({'direction': None})
        self.assertIsNone(controls.snapshot()['direction'])
        for points in ('{}', '[[1]]', '[[0, NaN]]'):
            with self.assertRaises(ValueError):
                preview.parse_waypoints(points)
