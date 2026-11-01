"""Local controls and execution-scoped streaming for the ARDY preview."""
import math
import json
from pathlib import Path
import threading
import time
import uuid
import tempfile

from aiohttp import web
import numpy as np
from scipy.spatial.transform import Rotation
import torch
import folder_paths

import comfy.model_management
from server import PromptServer
from ardy.constraints import Root2DConstraintSet
from ardy.viz.core_skin import CoreSkin
from comfy_execution.graph_utils import ExecutionBlocker


# Only small control messages live here. Models, history and embeddings belong to execute().
_controls = {}


def recording_path(control_id):
    return Path(folder_paths.get_output_directory()) / 'ardy' / 'recordings' / f'{uuid.UUID(control_id)}.npz'


def load_recording(path):
    with np.load(path, allow_pickle=False) as saved:
        motion = {key: saved[key].tolist() if key in ('fps', 'frames', 'parents', 'joint_names') else saved[key]
                  for key in saved.files}
    motion['recording_id'] = Path(path).stem
    return motion


@PromptServer.instance.routes.post('/ardy/recording/upload')
async def upload_recording(request):
    reader = await request.multipart()
    field = await reader.next()
    if field is None or not (field.filename or '').lower().endswith('.npz'):
        raise web.HTTPBadRequest(text='Choose an ARDY recording NPZ file.')
    identifier = str(uuid.uuid4())
    path = recording_path(identifier)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile() as file:
        while chunk := await field.read_chunk():
            file.write(chunk)
        file.seek(0)
        try:
            with np.load(file, allow_pickle=False) as saved:
                motion = {key: saved[key] for key in saved.files}
                if not {'mesh_vertices', 'mesh_faces', 'fps', 'frames', 'posed_joints', 'parents', 'joint_names'} <= motion.keys():
                    raise ValueError('Choose a recorded ARDY motion NPZ, not a model or skin file.')
                if motion['mesh_vertices'].ndim != 3 or motion['mesh_vertices'].shape[-1] != 3 or float(motion['fps']) <= 0:
                    raise ValueError('Invalid ARDY recording geometry or frame rate.')
        except (ValueError, OSError) as error:
            raise web.HTTPBadRequest(text=str(error)) from error
    np.savez(path, **motion)
    return web.json_response({'recording_id': identifier})


class LiveControls:
    def __init__(self, prompt, speed, waypoints=None):
        self.lock = threading.Lock()
        self.values = {"prompt": prompt, "speed": speed, "target": None, "paused": False,
                       "stop": False, "ready": False, "played": 0, "seen": time.monotonic(),
                       "waypoints": waypoints or [], "direction": None}

    def update(self, payload):
        changes = {}
        for name in ("paused", "stop", "ready"):
            if name in payload:
                if not isinstance(payload[name], bool):
                    raise ValueError(f"{name} must be a boolean")
                changes[name] = payload[name]
        for name in ("speed", "played"):
            if name in payload:
                value = float(payload[name])
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"{name} must be a finite nonnegative number")
                changes[name] = value
        if "prompt" in payload:
            if not isinstance(payload["prompt"], str):
                raise ValueError("prompt must be text")
            changes["prompt"] = payload["prompt"]
        if "target" in payload:
            target = payload["target"]
            if target is not None:
                if not isinstance(target, list) or len(target) != 2:
                    raise ValueError("target must contain ground coordinates [x, z]")
                target = [float(x) for x in target]
                if not all(math.isfinite(x) for x in target):
                    raise ValueError("target coordinates must be finite")
            changes["target"] = target
            changes["waypoints"] = [] if target is None else [target]
        if "waypoints" in payload:
            changes["waypoints"] = parse_waypoints(payload['waypoints'])
        if "direction" in payload:
            direction = payload['direction']
            changes['direction'] = None if direction is None else parse_waypoints([direction])[0]
        appended = parse_waypoints(payload.get('append_waypoints', []))
        with self.lock:
            self.values.update(changes, seen=time.monotonic())
            if appended:
                self.values['waypoints'] = self.values['waypoints'] + appended

    def snapshot(self):
        with self.lock:
            return self.values.copy()

    def steering(self, root):
        with self.lock:
            points = self.values['waypoints']
            while len(points) > 1 and math.dist(root, points[0]) < .25:
                points = points[1:]
            self.values['waypoints'] = points
            return self.values.copy()

    def advance_waypoints(self, route, root_positions):
        with self.lock:
            if self.values['waypoints'] != route:
                return
            points = route
            start = 0
            while len(points) > 1:
                distances = np.linalg.norm(root_positions[start:] - points[0], axis=-1)
                reached = np.flatnonzero(distances < .35)
                if not len(reached):
                    break
                start += int(reached[0])
                points = points[1:]
            self.values['waypoints'] = points


def parse_waypoints(value):
    points = json.loads(value.strip() or '[]') if isinstance(value, str) else value
    if not isinstance(points, list):
        raise ValueError('Waypoints must be a JSON list of [x, z] coordinates.')
    result = []
    for point in points:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError('Each waypoint must contain [x, z].')
        point = [float(v) for v in point]
        if not all(math.isfinite(v) for v in point):
            raise ValueError('Waypoint coordinates must be finite.')
        result.append(point)
    return result


@PromptServer.instance.routes.post('/ardy/live/{control_id}')
async def update_live_controls(request):
    controls = _controls.get(request.match_info['control_id'])
    if controls is None:
        return web.json_response({"error": "This ARDY session has ended. Run the node to start again."}, status=404)
    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise ValueError("Expected a control object")
        controls.update(payload)
    except (ValueError, TypeError, OverflowError) as error:
        return web.json_response({"error": str(error)}, status=400)
    return web.json_response({"ok": True})


def waypoint_conditions(model, history_length, root, target, speed, waypoints=None, direction=None):
    horizon = model.gen_horizon_len
    total = history_length + horizon
    route = waypoints or ([] if target is None else [target])
    if direction is not None and math.hypot(*direction) > 0:
        length = math.hypot(*direction)
        route = [[root[i] + direction[i] / length * speed * horizon / model.motion_rep.fps for i in range(2)]]
    if not route:
        return None, None
    offsets = list(range(model.num_frames_per_token - 1, horizon, model.num_frames_per_token))
    points = []
    for offset in offsets:
        remaining = speed * (offset + 1) / model.motion_rep.fps
        point = list(root)
        for destination in route:
            distance = math.dist(point, destination)
            if distance > remaining:
                point = [point[i] + (destination[i] - point[i]) * remaining / distance for i in range(2)]
                break
            remaining -= distance
            point = list(destination)
        points.append(point)
    constraint = Root2DConstraintSet(model.skeleton,
        torch.tensor([history_length + i for i in offsets], device=model.device),
        torch.tensor(points, dtype=torch.float32, device=model.device))
    observed, mask = model.motion_rep.create_conditions_from_constraints(
        [constraint], total, to_normalize=True, device=model.device)
    return observed[None], mask[None]


class ArdyLivePreview:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("ARDY_MODEL", {"lazy": True}),
            "prompt": ("STRING", {"default": "A person walks forward", "multiline": True}),
            "seed": ("INT", {"default": 42, "min": 0, "max": 2**32}),
            "denoising_steps": ("INT", {"default": 10, "min": 1, "max": 200}),
            "cfg_weight": ("FLOAT", {"default": 2.0, "min": 0, "max": 10}),
            "speed": ("FLOAT", {"default": 1.2, "min": 0, "max": 10, "step": 0.1}),
            "run_seconds": ("INT", {"default": 120, "min": 0, "max": 3600,
                "tooltip": "Live session duration. Zero runs until Stop or the viewer disconnects."}),
            "control_id": ("STRING", {"default": ""}),
        }, "optional": {"waypoints": ("STRING", {"default": "[]", "multiline": True,
            "tooltip": "Ordered ground coordinates [[x,z],...]. Click to append; Clear route removes all points. Start new take applies edits."})},
            "hidden": {"unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = ("ARDY_MOTION",)
    RETURN_NAMES = ("recorded_motion",)
    FUNCTION = "execute"
    CATEGORY = "Ardy"
    OUTPUT_NODE = True

    def check_lazy_status(self, control_id, model=None, **kwargs):
        return [] if control_id and recording_path(control_id).is_file() else ['model'] if model is None else []

    def execute(self, model, prompt, seed, denoising_steps, cfg_weight, speed, run_seconds,
                control_id, unique_id, waypoints="[]"):
        control_id = str(uuid.UUID(control_id)) if control_id else str(uuid.uuid4())
        cache_path = recording_path(control_id)
        if cache_path.is_file():
            motion = load_recording(cache_path)
            return {"ui": {"ardy_status": [f"Cached take · {motion['frames']} frames"]}, "result": (motion,)}
        ardy = model.model
        skin_path = Path(ardy.skeleton.folder) / 'skin_standard.npz'
        if not skin_path.is_file():
            raise ValueError("ARDY Live Preview currently requires a Core checkpoint with skin_standard.npz.")
        controls = LiveControls(prompt, speed, parse_waypoints(waypoints))
        _controls[control_id] = controls

        def send(kind, **data):
            PromptServer.instance.send_sync('ardy.live', {
                "node": str(unique_id), "control_id": control_id, "kind": kind, **data})

        try:
            model.load_for_execution()
            with np.load(skin_path) as skin:
                body = {k: skin[k].tolist() for k in (
                    'bind_vertices', 'faces', 'bind_rig_transform', 'lbs_indices', 'lbs_weights')}
            parents = [-1 if parent is None else ardy.skeleton.bone_index[parent]
                       for _, parent in ardy.skeleton.bone_order_names_with_parents]
            send('start', body=body, parents=parents, fps=ardy.motion_rep.fps, prompt=prompt, speed=speed)
            torch.manual_seed(seed)
            horizon = ardy.gen_horizon_len
            patch = ardy.num_frames_per_token
            history_limit = max(patch, ((10 * ardy.motion_rep.fps - horizon) // patch) * patch)
            history = None
            text_prompt = None
            text_feat = text_mask = None
            root = [0.0, 0.0]
            generated = 0
            recorded_positions = []
            recorded_rotations = []
            while not run_seconds or controls.snapshot()['played'] < run_seconds * ardy.motion_rep.fps - 1:
                comfy.model_management.throw_exception_if_processing_interrupted()
                state = controls.steering(root)
                if state['stop'] or time.monotonic() - state['seen'] > 30:
                    break
                # A single chunk ahead keeps clicks responsive and bounds GPU/CPU memory.
                if not state['ready'] or state['paused'] or generated - state['played'] > horizon / 2:
                    time.sleep(0.05)
                    continue
                if state['prompt'] != text_prompt:
                    send('status', message='Encoding prompt…')
                    text_feat, text_mask = ardy._encode_text([state['prompt']])
                    model.load_for_execution()
                    text_prompt = state['prompt']
                history_length = 0 if history is None else history.shape[1]
                observed, mask = waypoint_conditions(ardy, history_length, root, None, state['speed'],
                                                     state['waypoints'], state['direction'])
                samples = ardy.autoregressive_step(
                    num_frames=history_length + horizon,
                    num_denoising_steps=min(denoising_steps, ardy.diffusion.num_base_steps),
                    motion_mask=mask, observed_motion=observed, cfg_weight=(cfg_weight, 2.0),
                    text_feat=text_feat, text_pad_mask=text_mask, init_history_sequence=history,
                )
                decoded = ardy.motion_rep.inverse(samples[:, -horizon:], is_normalized=True)
                positions = decoded['posed_joints'][0].detach().cpu().numpy()
                rotations = decoded['global_rot_mats'][0].detach().cpu().numpy()
                recorded_positions.append(positions.copy())
                recorded_rotations.append(rotations.copy())
                quaternions = Rotation.from_matrix(rotations.reshape(-1, 3, 3)).as_quat().reshape(horizon, -1, 4)
                root = positions[-1, ardy.skeleton.root_idx, [0, 2]].tolist()
                if state['direction'] is None:
                    controls.advance_waypoints(state['waypoints'], positions[:, ardy.skeleton.root_idx][:, [0, 2]])
                send('chunk', start=generated, positions=positions.tolist(), rotations=quaternions.tolist(),
                     target=state['target'], waypoints=controls.snapshot()['waypoints'])
                generated += horizon
                history = samples[:, -history_limit:].clone()
            if not recorded_positions:
                return {"ui": {"ardy_status": ["No motion recorded"]}, "result": (ExecutionBlocker(None),)}
            frames = min(generated, int(controls.snapshot()['played']) + 1)
            if run_seconds:
                frames = min(frames, int(run_seconds * ardy.motion_rep.fps))
            positions = np.concatenate(recorded_positions)[:frames].copy()
            rotations = np.concatenate(recorded_rotations)[:frames].copy()
            send('status', message=f'Exporting {frames} recorded frames…')
            skin = CoreSkin(ardy.skeleton)
            vertices = np.empty((frames, len(skin.bind_vertices), 3), dtype=np.float32)
            for start in range(0, frames, 8):
                comfy.model_management.throw_exception_if_processing_interrupted()
                vertices[start:start + 8] = skin.skin(
                    torch.from_numpy(rotations[start:start + 8]).to(ardy.device),
                    torch.from_numpy(positions[start:start + 8]).to(ardy.device),
                    rot_is_global=True).detach().cpu().numpy()
            motion = {"posed_joints": positions, "joint_names": list(ardy.skeleton.bone_order_names),
                      "parents": parents, "fps": ardy.motion_rep.fps, "frames": frames,
                      "mesh_vertices": vertices, "mesh_faces": skin.faces.cpu().numpy().copy()}
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_suffix('.tmp.npz')
            np.savez(temporary, **motion)
            temporary.replace(cache_path)
            motion['recording_id'] = control_id
            return {"ui": {"ardy_status": [f"Recorded {frames} frames"]}, "result": (motion,)}
        finally:
            model.offload()
            _controls.pop(control_id, None)
            send('stop')
