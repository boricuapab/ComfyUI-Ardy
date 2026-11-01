import torch
from ardy.viz.core_skin import CoreSkin


class ArdyRunner:
    def __init__(self, ardy_model):
        self.owner = ardy_model
        self.model = ardy_model.model

    def generate(self, prompt, seed=0, frames=160, fps=20, denoising_steps=50, cfg_weight=2.0):
        torch.manual_seed(seed)
        model = self.model
        device = model.device
        text_feat, text_mask = model._encode_text([prompt])
        self.owner.load_for_execution()
        steps = max(1, min(denoising_steps, model.diffusion.num_base_steps))
        patch = model.num_frames_per_token
        history = ((int(10 * model.motion_rep.fps) - model.gen_horizon_len) // patch) * patch
        features = model(
            texts=[prompt],
            text_feat=text_feat,
            text_pad_mask=text_mask,
            num_frames=frames,
            num_denoising_steps=steps,
            pad_mask=torch.ones(1, frames, dtype=torch.bool, device=device),
            first_heading_angle=torch.zeros(1, device=device),
            motion_mask=None,
            observed_motion=None,
            cfg_weight=cfg_weight,
            crop_history_length=history,
        )
        decoded = model.motion_rep.inverse(features, is_normalized=True)
        result = {
            "motion": features.detach().cpu(),
            "posed_joints": decoded["posed_joints"][0, :frames].detach().cpu().numpy().copy(),
            "joint_names": list(model.skeleton.bone_order_names),
            "parents": [
                -1 if parent is None else model.skeleton.bone_index[parent]
                for _, parent in model.skeleton.bone_order_names_with_parents
            ],
            "fps": fps,
            "frames": frames,
            "prompt": prompt,
        }
        if model.skeleton.name == "cskel27":
            skin = CoreSkin(model.skeleton)
            vertices = []
            for start in range(0, frames, 8):
                vertices.append(skin.skin(
                    decoded["global_rot_mats"][0, start:start + 8],
                    decoded["posed_joints"][0, start:start + 8], rot_is_global=True,
                ).detach().cpu())
            result["mesh_vertices"] = torch.cat(vertices).numpy()
            result["mesh_faces"] = skin.faces.cpu().numpy().copy()
        return result
