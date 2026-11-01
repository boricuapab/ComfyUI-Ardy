"""Animated body and skeleton GLB export in ARDY's Y-up space."""
import io
import json
import struct

import numpy as np


def motion_to_glb(motion, display_mode="mesh"):
    joints = np.asarray(motion["posed_joints"], dtype=np.float32)
    if joints.ndim != 3 or joints.shape[-1] != 3 or len(joints) == 0:
        raise ValueError("ARDY export requires joint positions with shape (frames, joints, 3).")
    if not np.isfinite(joints).all():
        raise ValueError("ARDY motion contains non-finite joint positions.")
    fps = motion["fps"]
    if fps <= 0:
        raise ValueError("ARDY export FPS must be positive.")
    doc = {"asset": {"version": "2.0", "generator": "ComfyUI-Ardy"}, "scene": 0,
           "scenes": [{"nodes": []}], "nodes": [], "meshes": [],
           "buffers": [], "bufferViews": [], "accessors": [],
           "materials": [{"pbrMetallicRoughness": {"baseColorFactor": [0.15, 0.65, 1.0, 1.0],
                          "metallicFactor": 0, "roughnessFactor": 0.7}, "doubleSided": True}],
           "animations": [{"name": "ARDY Motion", "samplers": [], "channels": []}]}
    binary = bytearray()

    def accessor(values, kind, dtype="<f4"):
        values = np.ascontiguousarray(values, dtype=dtype)
        offset = len(binary)
        binary.extend(values.tobytes())
        view = len(doc["bufferViews"])
        doc["bufferViews"].append({"buffer": 0, "byteOffset": offset, "byteLength": values.nbytes})
        entry = {"bufferView": view, "componentType": 5125 if dtype == "<u4" else 5126, "count": len(values), "type": kind}
        entry["min"] = values.reshape(len(values), -1).min(axis=0).tolist()
        entry["max"] = values.reshape(len(values), -1).max(axis=0).tolist()
        doc["accessors"].append(entry)
        return len(doc["accessors"]) - 1

    # A shared octahedron is scaled into joints and bones.
    vertices = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
    faces = [[2, 4, 0], [2, 1, 4], [2, 5, 1], [2, 0, 5],
             [3, 0, 4], [3, 4, 1], [3, 1, 5], [3, 5, 0]]
    positions = vertices[np.array(faces)].reshape(-1, 3)
    normals = positions.reshape(-1, 3, 3).mean(axis=1)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    doc["meshes"].append({"primitives": [{"attributes": {
        "POSITION": accessor(positions, "VEC3"),
        "NORMAL": accessor(np.repeat(normals, 3, axis=0), "VEC3")}, "material": 0}]})
    times = accessor(np.arange(len(joints), dtype=np.float32) / fps, "SCALAR")
    animation = doc["animations"][0]

    def node(name, translation, scale, rotation=None):
        index = len(doc["nodes"])
        item = {"name": name, "mesh": 0, "translation": translation[0].tolist(), "scale": scale[0].tolist()}
        tracks = [("translation", translation, "VEC3"), ("scale", scale, "VEC3")]
        if rotation is not None:
            item["rotation"] = rotation[0].tolist()
            tracks.append(("rotation", rotation, "VEC4"))
        doc["nodes"].append(item)
        doc["scenes"][0]["nodes"].append(index)
        for path, values, kind in tracks:
            sampler = len(animation["samplers"])
            animation["samplers"].append({"input": times, "output": accessor(values, kind), "interpolation": "LINEAR"})
            animation["channels"].append({"sampler": sampler, "target": {"node": index, "path": path}})

    show_bones = display_mode != "mesh" or "mesh_vertices" not in motion
    for joint, parent in enumerate(motion["parents"] if show_bones else []):
        name = motion["joint_names"][joint]
        node(name, joints[:, joint], np.full((len(joints), 3), 0.025))
        if parent < 0:
            continue
        delta = joints[:, joint] - joints[:, parent]
        length = np.linalg.norm(delta, axis=1)
        direction = np.divide(delta, length[:, None], out=np.zeros_like(delta), where=length[:, None] > 0)
        rotation = np.column_stack((direction[:, 2], np.zeros(len(joints)), -direction[:, 0], 1 + direction[:, 1]))
        norm = np.linalg.norm(rotation, axis=1)
        rotation[norm < 1e-6] = [1, 0, 0, 0]
        rotation /= np.linalg.norm(rotation, axis=1, keepdims=True)
        for frame in range(1, len(rotation)):
            if np.dot(rotation[frame - 1], rotation[frame]) < 0:
                rotation[frame] *= -1
        scale = np.column_stack((np.full(len(joints), 0.015), length / 2, np.full(len(joints), 0.015)))
        node(name + " bone", (joints[:, joint] + joints[:, parent]) / 2, scale, rotation)

    if display_mode != "skeleton" and "mesh_vertices" in motion:
        frames = motion["mesh_vertices"]
        faces = motion["mesh_faces"]

        def normals(vertices):
            triangles = vertices[faces]
            face_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
            result = np.zeros_like(vertices)
            for corner in range(3):
                np.add.at(result, faces[:, corner], face_normals)
            lengths = np.linalg.norm(result, axis=1, keepdims=True)
            return np.divide(result, lengths, out=result, where=lengths > 0)

        base = frames[0]
        base_normals = normals(base)
        # Morph targets preserve all five skin influences in viewers that only support four joint weights.
        targets = [{"POSITION": accessor(frame - base, "VEC3"),
                    "NORMAL": accessor(normals(frame) - base_normals, "VEC3")}
                   for frame in frames]
        mesh_index = len(doc["meshes"])
        doc["meshes"].append({"name": "ARDY Body", "weights": [0.0] * len(frames), "primitives": [{
            "attributes": {"POSITION": accessor(base, "VEC3"),
                           "NORMAL": accessor(base_normals, "VEC3")},
            "indices": accessor(faces.reshape(-1), "SCALAR", "<u4"),
            "material": 0, "targets": targets}]})
        body_index = len(doc["nodes"])
        doc["nodes"].append({"name": "ARDY Body", "mesh": mesh_index})
        doc["scenes"][0]["nodes"].append(body_index)
        sampler = len(animation["samplers"])
        animation["samplers"].append({"input": times, "output": accessor(np.eye(len(frames)).reshape(-1), "SCALAR"),
                                      "interpolation": "LINEAR"})
        animation["channels"].append({"sampler": sampler, "target": {"node": body_index, "path": "weights"}})

    doc["buffers"] = [{"byteLength": len(binary)}]
    header = json.dumps(doc, separators=(",", ":"), allow_nan=False).encode("utf-8")
    header += b" " * (-len(header) % 4)
    binary.extend(b"\0" * (-len(binary) % 4))
    total = 12 + 8 + len(header) + 8 + len(binary)
    return io.BytesIO(struct.pack("<4sII", b"glTF", 2, total)
                      + struct.pack("<I4s", len(header), b"JSON") + header
                      + struct.pack("<I4s", len(binary), b"BIN\0") + binary)
