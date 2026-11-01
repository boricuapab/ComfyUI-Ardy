import json
import sys
from pathlib import Path
import unittest
import tempfile
import uuid
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT.parents[1]), str(ROOT)]
from render_preview import ArdyCameraRecorder, ArdyLoadCamera, camera_at, parse_camera_track


class CameraTests(unittest.TestCase):
    def test_saved_take_renders_without_connected_live_motion(self):
        with tempfile.TemporaryDirectory() as folder, patch('render_preview.folder_paths.get_output_directory', return_value=folder), \
             patch('render_preview.folder_paths.get_temp_directory', return_value=folder):
            identifier = str(uuid.uuid4())
            path = Path(folder) / 'ardy' / 'recordings' / f'{identifier}.npz'
            path.parent.mkdir(parents=True)
            np.savez(path, frames=2, fps=20, mesh_vertices=np.zeros((2, 3, 3), dtype=np.float32), mesh_faces=[[0, 1, 2]])
            node = ArdyCameraRecorder()
            self.assertEqual(node.check_lazy_status(recording_id=identifier), [])
            with patch.object(node, '_render', return_value={'ui': {}, 'result': ()}) as render:
                result = node.render(recording_id=identifier)
                self.assertEqual(render.call_args.args[0]['frames'], 2)
                self.assertEqual(result['ui']['ardy_recording_id'], [identifier])
            with self.assertRaises(ValueError):
                node.render(recording_id='../outside')

    def test_camera_loader_preserves_timed_track(self):
        track = [{'time': t, 'position': {'x': t, 'y': 2, 'z': 5},
                  'target': {'x': 0, 'y': 1, 'z': 0}, 'fov': 40, 'zoom': 1} for t in (0, 2)]
        camera, = ArdyLoadCamera().load(json.dumps(track))
        self.assertEqual(camera['ardy_track'], track)
        self.assertEqual(camera['position'], track[0]['position'])

    def test_24fps_render_preserves_duration_and_interpolates_motion(self):
        vertices = np.zeros((40, 3, 3), dtype=np.float32)
        vertices[:, :, 0] = np.arange(40)[:, None]
        motion = {'mesh_vertices': vertices, 'mesh_faces': np.array([[0, 1, 2]]), 'fps': 20}
        positions = []
        def render(mesh, *args, **kwargs):
            positions.append(float(mesh.vertices[0, 0, 0]))
            return SimpleNamespace(result=(torch.zeros(1, 2, 2, 3), torch.zeros(1, 2, 2)))
        with patch('render_preview.RenderMesh.execute', side_effect=render), \
             patch('render_preview.UI.ImageSaveHelper.save_animated_webp'), \
             patch('render_preview.UI.SavedImages'):
            result = ArdyCameraRecorder()._render(motion, 2, 2, 'solid', '#000000', 0, 10, 1, None, output_fps=24)
        images, masks, fps = result['result']
        self.assertEqual((len(images), len(masks), fps), (48, 48, 24))
        self.assertAlmostEqual(positions[1], 20 / 24)
        self.assertEqual(positions[-1], 39)

    def test_camera_interpolation_and_endpoint_hold(self):
        track = [{'time': time, 'position': {'x': x, 'y': 2, 'z': 5},
                  'target': {'x': 0, 'y': 1, 'z': 0}, 'fov': fov, 'zoom': 1}
                 for time, x, fov in [(0, 0, 40), (2, 4, 60)]]
        track = parse_camera_track(json.dumps(track))
        camera = camera_at(track, [0, 2], 1)
        self.assertEqual(camera['position']['x'], 2)
        self.assertEqual(camera['fov'], 50)
        self.assertEqual(camera_at(track, [0, 2], -1), track[0])
        self.assertEqual(camera_at(track, [0, 2], 4), track[1])
        self.assertEqual(parse_camera_track('  '), [])
        with self.assertRaises(ValueError):
            parse_camera_track(json.dumps(track[::-1]))
