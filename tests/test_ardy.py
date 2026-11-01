import importlib.util
import json
from pathlib import Path
import struct
import sys
import unittest
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "ardy"))


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ArdyTests(unittest.TestCase):
    def test_mesh_morph_targets_reconstruct_frames(self):
        frames = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                           [[0, 0, 1], [1, 0, 2], [0, 1, 1]]], dtype=np.float32)
        result = load("exporters").motion_to_glb({"posed_joints": frames[:, :1],
            "parents": [-1], "joint_names": ["root"], "fps": 20,
            "mesh_vertices": frames, "mesh_faces": np.array([[0, 1, 2]])}).getvalue()
        size = struct.unpack_from("<I", result, 12)[0]
        doc = json.loads(result[20:20 + size])
        binary = result[28 + size:]

        def values(index):
            a = doc["accessors"][index]
            view = doc["bufferViews"][a["bufferView"]]
            dtype = "<u4" if a["componentType"] == 5125 else "<f4"
            return np.frombuffer(binary, dtype=dtype, count=view["byteLength"] // 4,
                                 offset=view["byteOffset"]).reshape(a["count"], -1)

        body = doc["meshes"][doc["nodes"][0]["mesh"]]["primitives"][0]
        np.testing.assert_array_equal(values(body["indices"]).ravel(), [0, 1, 2])
        for frame, target in zip(frames, body["targets"]):
            np.testing.assert_allclose(values(body["attributes"]["POSITION"]) + values(target["POSITION"]), frame)
        weights = values(doc["animations"][0]["samplers"][0]["output"])
        np.testing.assert_array_equal(weights.reshape(2, 2), np.eye(2))

    def test_glb_animation_preserves_joint_and_bone_endpoints(self):
        positions = np.array([[[0, 1, 0], [0, 2, 0]],
                              [[1, 1, 0], [1, 0, 0]],
                              [[2, 1, 0], [2, 1, 0]]], dtype=np.float32)
        result = load("exporters").motion_to_glb({"posed_joints": positions,
            "parents": [-1, 0], "joint_names": ["root", "child"], "fps": 20}).getvalue()
        magic, version, size = struct.unpack_from("<4sII", result)
        self.assertEqual((magic, version, size), (b"glTF", 2, len(result)))
        json_size = struct.unpack_from("<I", result, 12)[0]
        doc = json.loads(result[20:20 + json_size])
        binary = result[28 + json_size:]

        def values(index):
            a = doc["accessors"][index]
            view = doc["bufferViews"][a["bufferView"]]
            return np.frombuffer(binary, dtype="<f4", count=view["byteLength"] // 4,
                                 offset=view["byteOffset"]).reshape(a["count"], -1)

        anim = doc["animations"][0]
        tracks = {(c["target"]["node"], c["target"]["path"]): values(anim["samplers"][c["sampler"]]["output"])
                  for c in anim["channels"]}
        np.testing.assert_allclose(values(anim["samplers"][0]["input"])[:, 0], [0, .05, .1])
        for joint in range(2):
            np.testing.assert_allclose(tracks[joint, "translation"], positions[:, joint])
        center = tracks[2, "translation"]
        extent = Rotation.from_quat(tracks[2, "rotation"]).apply(tracks[2, "scale"] * [0, 1, 0])
        np.testing.assert_allclose(center - extent, positions[:, 0], atol=1e-6)
        np.testing.assert_allclose(center + extent, positions[:, 1], atol=1e-6)

    def test_generation_decodes_and_caps_unsupported_steps(self):
        class Model:
            device = "cpu"
            diffusion = SimpleNamespace(num_base_steps=10)
            num_frames_per_token = 4
            gen_horizon_len = 40
            skeleton = SimpleNamespace(name="test", bone_order_names=["root"],
                bone_order_names_with_parents=[("root", None)], bone_index={"root": 0})
            motion_rep = SimpleNamespace(fps=20,
                inverse=lambda x, is_normalized: {"posed_joints": torch.zeros(1, 12, 1, 3)})
            def __call__(self, **kwargs):
                self.inputs = kwargs
                return torch.zeros(1, kwargs["num_frames"], 8)
            def _encode_text(self, texts):
                return torch.zeros(1, 1, 8), torch.ones(1, 1)
        model = Model()
        result = load("ardy_runner").ArdyRunner(SimpleNamespace(model=model, load_for_execution=lambda: None)).generate(
            "walking", seed=2, frames=11, fps=25, denoising_steps=50, cfg_weight=3)
        self.assertEqual(model.inputs["num_frames"], 11)
        self.assertEqual(model.inputs["num_denoising_steps"], 10)
        self.assertEqual(model.inputs["cfg_weight"], 3)
        self.assertEqual(model.inputs["crop_history_length"], 160)
        self.assertIsNone(model.inputs["motion_mask"])
        self.assertEqual(result["posed_joints"].shape, (11, 1, 3))
        self.assertEqual(result["fps"], 25)


if __name__ == "__main__":
    unittest.main()
