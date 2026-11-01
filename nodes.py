import torch

from .folder_paths import get_model_list, get_model_path
from .ardy_loader import load_model, load_text_encoder
from .exporters import motion_to_glb
from comfy_api.latest import Types
from .preview import ArdyLivePreview
from .render_preview import ArdyPreviewAdvanced, ArdyCameraRecorder, ArdyLoadCamera

class ArdyGenerateMotion:


    @classmethod
    def INPUT_TYPES(cls):

        return {

            "required":

            {

                "model":
                (
                    "ARDY_MODEL",
                ),


                "prompt":
                (
                    "STRING",
                    {
                        "multiline": True,
                        "default":
                        "A person walks forward"
                    }
                ),


                "seed":
                (
                    "INT",
                    {
                        "default":0,
                        "min":0,
                        "max":2**32
                    }
                ),

                "denoising_steps":
                (
                    "INT",
                    {
                        "default":50,
                        "min":1,
                        "max":200
                    }
                ),

                "cfg_weight":
                (
                    "FLOAT",
                    {
                        "default":2.0,
                        "min":0,
                        "max":10
                    }
                ),


                "frames":
                (
                    "INT",
                    {
                        "default":160,
                        "min":8,
                        "max":500
                    }
                ),


                "fps":
                (
                    "INT",
                    {
                        "default":20,
                        "min":1,
                        "max":120
                    }
                )

            }
        }



    RETURN_TYPES = (
        "ARDY_MOTION",
    )


    FUNCTION = "generate"


    CATEGORY = "Ardy"



    def generate(
        self,
        model,
        prompt,
        seed,
        denoising_steps,
        cfg_weight,
        frames,
        fps
    ):

        model.load_for_execution()
        try:
            motion = model.runner.generate(
                prompt=prompt,
                seed=seed,
                denoising_steps=denoising_steps,
                cfg_weight=cfg_weight,
                frames=frames,
                fps=fps
            )
        finally:
            model.offload()


        return (
            motion,
        )


class LoadArdyModel:


    @classmethod
    def INPUT_TYPES(cls):

        models = get_model_list()


        if not models:

            models = [
                "No ARDY models installed"
            ]


        return {

            "required": {

                "model": (
                    models,
                ),


                "device": (
                    [
                        "auto",
                        "cuda",
                        "cpu"
                    ],
                ),

                "text_encoder": ("ARDY_TEXT_ENCODER",),
            }

        }



    RETURN_TYPES = (
        "ARDY_MODEL",
    )

    FUNCTION = "load"

    CATEGORY = "Ardy"



    def load(
        self,
        model,
        device,
        text_encoder
    ):



        if device == "auto":

            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )


        device = torch.device('cuda', torch.cuda.current_device()) if device == 'cuda' else torch.device(device)



        path = get_model_path(model)


        loaded = load_model(
            path,
            device,
            text_encoder
        )


        return (
            loaded,
        )


class LoadArdyTextEncoder:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "text_encoder": (["auto", "int8_convrot", "bf16"],),
            "device": (["cuda:0", "cpu"],),
        }}

    RETURN_TYPES = ("ARDY_TEXT_ENCODER",)
    FUNCTION = "load"
    CATEGORY = "Ardy/loaders"

    def load(self, text_encoder, device):
        return (load_text_encoder(text_encoder, device),)


class ArdyInfo:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("STRING",)

    FUNCTION = "run"

    CATEGORY = "Ardy"

    def run(self):

        return (
            "ARDY Extension Loaded",
        )


class ArdyMotionToGLB:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"motion": ("ARDY_MOTION",)},
                "optional": {"display_mode": (["mesh", "mesh and skeleton", "skeleton"],)}}

    RETURN_TYPES = ("FILE_3D_GLB",)
    RETURN_NAMES = ("model_3d",)
    FUNCTION = "export"
    CATEGORY = "Ardy"

    def export(self, motion, display_mode="mesh"):
        return (Types.File3D(motion_to_glb(motion, display_mode), "glb"),)


NODE_CLASS_MAPPINGS = {
    "ArdyLoadCamera": ArdyLoadCamera,
    "ArdyCameraRecorder": ArdyCameraRecorder,
    "ArdyPreviewAdvanced": ArdyPreviewAdvanced,
    "LoadArdyTextEncoder": LoadArdyTextEncoder,
    "ArdyLivePreview": ArdyLivePreview,
    "ArdyMotionToGLB": ArdyMotionToGLB,

    "LoadArdyModel": LoadArdyModel,

    "ArdyGenerateMotion": ArdyGenerateMotion,

    "ArdyInfo": ArdyInfo,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArdyLoadCamera": "Load ARDY Camera",
    "ArdyCameraRecorder": "ARDY Camera Recorder",
    "ArdyPreviewAdvanced": "ARDY Preview Advanced · Driving Frames",
    "LoadArdyTextEncoder": "Load ARDY Text Encoder",
    "ArdyLivePreview": "ARDY Live Preview",
    "ArdyMotionToGLB": "ARDY Motion to GLB",

    "LoadArdyModel": "Load ARDY Model",

    "ArdyGenerateMotion": "Generate ARDY Motion",

    "ArdyInfo": "ARDY Info",

}
