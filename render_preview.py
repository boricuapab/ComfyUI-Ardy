"""Render recorded ARDY motion with ComfyUI's native mesh renderer."""
import math
import bisect
import json
import struct
import tempfile
import uuid
from pathlib import Path
import numpy as np
import torch
import folder_paths

from comfy_api.latest import IO, UI, Types
import comfy.model_management
import comfy.utils
from comfy_extras.nodes_mesh_postprocess import RenderMesh


def parse_camera_track(value):
    track = json.loads(value.strip() or '[]')
    if not isinstance(track, list):
        raise ValueError('Camera track must be a list of timed camera positions.')
    for key in track:
        values = [key['time'], key['fov'], key.get('zoom', 1)]
        values += [key[name][axis] for name in ('position', 'target') for axis in ('x', 'y', 'z')]
        if not all(math.isfinite(float(v)) for v in values):
            raise ValueError('Camera values must be finite.')
        if not 0 < key['fov'] < 180 or key.get('zoom', 1) <= 0:
            raise ValueError('Camera FOV must be between 0 and 180, and zoom must be positive.')
    if any(b['time'] <= a['time'] for a, b in zip(track, track[1:])):
        raise ValueError('Camera keyframe times must increase.')
    return track


def camera_at(track, times, time):
    index = bisect.bisect_right(times, time)
    if index == 0:
        return track[0]
    if index == len(track):
        return track[-1]
    a, b = track[index - 1:index + 1]
    alpha = (time - a['time']) / (b['time'] - a['time'])
    result = {name: {axis: a[name][axis] + (b[name][axis] - a[name][axis]) * alpha
                    for axis in ('x', 'y', 'z')} for name in ('position', 'target')}
    for name, default in (('fov', 40), ('zoom', 1)):
        result[name] = a.get(name, default) + (b.get(name, default) - a.get(name, default)) * alpha
    return result


class ArdyPreviewAdvanced:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "motion": ("ARDY_MOTION",),
            "width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
            "height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
            "mode": (["solid", "normal", "depth"],),
            "background": ("STRING", {"default": "#000000"}),
            "yaw": ("FLOAT", {"default": 0, "min": -180, "max": 180}),
            "elevation": ("FLOAT", {"default": 10, "min": -89, "max": 89}),
            "frame_step": ("INT", {"default": 1, "min": 1, "max": 1000}),
        }, "optional": {"camera_info": ("LOAD3D_CAMERA",)}}

    RETURN_TYPES = ("IMAGE", "MASK", "FLOAT")
    RETURN_NAMES = ("images", "mask", "fps")
    FUNCTION = "render"
    CATEGORY = "Ardy"
    OUTPUT_NODE = True

    def render(self, motion, width, height, mode, background, yaw, elevation, frame_step, camera_info=None):
        return self._render(motion, width, height, mode, background, yaw, elevation, frame_step, camera_info)

    def _render(self, motion, width, height, mode, background, yaw, elevation, frame_step, camera_info, track=(), output_fps=0):
        if camera_info and 'ardy_track' in camera_info:
            track = parse_camera_track(json.dumps(camera_info['ardy_track']))
        vertices = motion.get('mesh_vertices')
        if vertices is None:
            raise ValueError('ARDY Preview Advanced requires recorded or generated body mesh motion.')
        color = background.lstrip('#')
        if len(color) != 6 or any(c not in '0123456789abcdefABCDEF' for c in color):
            raise ValueError('Background must be a color such as #000000.')
        if camera_info is None:
            low, high = vertices.min(axis=(0, 1)), vertices.max(axis=(0, 1))
            center = (low + high) / 2
            radius = max(float(np.linalg.norm(high - low)) / 2, .1)
            distance = radius / math.sin(math.radians(20)) * max(1, height / width) * 1.05
            azimuth, pitch = math.radians(yaw), math.radians(elevation)
            eye = center + distance * np.array([math.sin(azimuth) * math.cos(pitch), math.sin(pitch), math.cos(azimuth) * math.cos(pitch)])
            camera_info = {'position': dict(zip(('x', 'y', 'z'), eye.tolist())),
                           'target': dict(zip(('x', 'y', 'z'), center.tolist())), 'fov': 40, 'zoom': 1}
        fps = float(output_fps or motion['fps']) / frame_step
        count = math.ceil(len(vertices) / motion['fps'] * fps)
        images = torch.empty((count, height, width, 3), dtype=torch.float32)
        masks = torch.empty((count, height, width), dtype=torch.float32)
        faces = torch.as_tensor(motion['mesh_faces'])[None]
        progress = comfy.utils.ProgressBar(count)
        times = [key['time'] for key in track]
        for index in range(count):
            comfy.model_management.throw_exception_if_processing_interrupted()
            sample = min(index / fps * motion['fps'], len(vertices) - 1)
            frame = int(sample)
            next_frame = min(frame + 1, len(vertices) - 1)
            posed = vertices[frame:frame + 1] + (vertices[next_frame:next_frame + 1] - vertices[frame:frame + 1]) * (sample - frame)
            mesh = Types.MESH(torch.from_numpy(posed), faces)
            camera = camera_at(track, times, index / fps) if track else camera_info
            image, mask = RenderMesh.execute(mesh, mode, width, height, background, camera_info=camera).result
            images[index] = image[0].cpu()
            masks[index] = mask[0].cpu()
            progress.update(1)
        saved = UI.ImageSaveHelper.save_animated_webp(images, filename_prefix='ardy_preview',
            folder_type=IO.FolderType.temp, cls=None, fps=fps, lossless=False, quality=85, method=0)
        return {'ui': UI.SavedImages([saved], is_animated=count > 1).as_dict(), 'result': (images, masks, fps)}


class ArdyCameraRecorder(ArdyPreviewAdvanced):
    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs['required'].pop('motion')
        inputs['optional'] = {'motion': ('ARDY_MOTION', {'lazy': True}), **inputs['optional']}
        inputs['optional']['camera_track'] = ('STRING', {'default': '[]', 'multiline': True})
        inputs['optional']['output_fps'] = ('FLOAT', {'default': 24, 'min': 1, 'max': 120,
            'tooltip': 'Resample motion without changing its speed. MiniMax H3 reference video uses 24 fps.'})
        inputs['optional']['recording_id'] = ('STRING', {'default': ''})
        return inputs

    def check_lazy_status(self, recording_id='', motion=None, **kwargs):
        return ['motion'] if not recording_id and motion is None else []

    def render(self, motion=None, width=512, height=512, mode='solid', background='#000000', yaw=0, elevation=10, frame_step=1, camera_info=None, camera_track='[]', output_fps=24, recording_id=''):
        if recording_id:
            path = Path(folder_paths.get_output_directory()) / 'ardy' / 'recordings' / f'{uuid.UUID(recording_id)}.npz'
            if not path.is_file():
                raise ValueError('Saved take is missing. Load an NPZ or choose Use connected motion.')
            with np.load(path, allow_pickle=False) as saved:
                motion = {key: saved[key].tolist() if key in ('fps', 'frames', 'parents', 'joint_names') else saved[key] for key in saved.files}
            motion['recording_id'] = recording_id
        if motion is None:
            raise ValueError('Load an ARDY recording NPZ or connect recorded motion.')
        track = parse_camera_track(camera_track)
        result = self._render(motion, width, height, mode, background, yaw, elevation, frame_step, camera_info, track, output_fps)
        vertices = np.asarray(motion['mesh_vertices'], dtype='<f4')
        faces = np.asarray(motion['mesh_faces'], dtype='<u4')
        header = json.dumps({'frames': len(vertices), 'vertices': vertices.shape[1], 'faces': faces.size,
                             'fps': motion['fps'], 'width': width, 'height': height,
                             'camera_track': camera_info.get('ardy_track', []) if camera_info else []}).encode()
        header += b' ' * (-len(header) % 4)
        with tempfile.NamedTemporaryFile(dir=folder_paths.get_temp_directory(), prefix='ardy_camera_', suffix='.bin', delete=False) as file:
            file.write(struct.pack('<I', len(header)))
            file.write(header)
            file.write(faces.tobytes())
            file.write(vertices.tobytes())
        result['ui']['ardy_camera'] = [{'filename': Path(file.name).name, 'type': 'temp', 'subfolder': ''}]
        result['ui']['ardy_recording_id'] = [motion.get('recording_id', '')]
        return result


class ArdyLoadCamera:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'camera_track': ('STRING', {'default': '[]', 'multiline': True})}}

    RETURN_TYPES = ('LOAD3D_CAMERA',)
    RETURN_NAMES = ('camera_info',)
    FUNCTION = 'load'
    CATEGORY = 'Ardy'

    def load(self, camera_track):
        track = parse_camera_track(camera_track)
        if not track:
            raise ValueError('Load a saved camera JSON file first.')
        return ({**track[0], 'ardy_track': track},)
