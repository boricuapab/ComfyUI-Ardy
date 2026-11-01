"""Load the motion checkpoint and its local Llama/LLM2Vec text encoder."""
from pathlib import Path

import torch
import comfy.model_management as mm
from comfy.model_patcher import ModelPatcher
from peft import PeftModel
from transformers import AutoTokenizer

from .ardy_import import setup_ardy_path
from .folder_paths import get_ardy_directory
from .model_downloads import ensure_text_encoder

setup_ardy_path()

from .ardy_runner import ArdyRunner
from ardy.model import load_model as load_ardy_model
from ardy.model.llm2vec.llm2vec import LLM2Vec
from ardy.model.llm2vec.models.bidirectional_llama import LlamaBiModel
from .quantized_encoder import QuantizedTextEncoder


class LocalTextEncoder:
    def __init__(self, device="cuda:0"):
        self.device = device
        root = get_ardy_directory() / "text_encoder"
        base = root / "Meta-Llama-3-8B-Instruct"
        mntp = root / "LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"
        supervised = root / "LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised"
        for path in (base, mntp, supervised):
            if not path.is_dir():
                raise FileNotFoundError(f"ARDY text encoder is missing: {path}")
        tokenizer = AutoTokenizer.from_pretrained(str(mntp), local_files_only=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model = LlamaBiModel.from_pretrained(str(base), dtype=torch.bfloat16, local_files_only=True)
        model = PeftModel.from_pretrained(model, str(mntp), local_files_only=True).merge_and_unload()
        model = PeftModel.from_pretrained(model, str(supervised), local_files_only=True)
        model.config._name_or_path = "meta-llama/Meta-Llama-3-8B-Instruct"
        self.model = LLM2Vec(model, tokenizer).eval()
        self.patcher = ModelPatcher(model, load_device=torch.device(device), offload_device=torch.device('cpu'))

    def __call__(self, texts):
        mm.load_models_gpu([self.patcher], force_full_load=True)
        embeddings = []
        try:
            for text in texts:
                prepared = self.model.prepare_for_tokenization(self.model._convert_to_str('', text))
                features = {k: v.to(self.patcher.load_device) for k, v in self.model.tokenize([prepared]).items()}
                embeddings.append(self.model(features).detach().float().cpu())
            return torch.cat(embeddings)[:, None], [1] * len(texts)
        finally:
            mm.unload_model_and_clones(self.patcher)


class ArdyModel:
    def __init__(self, model_dir, device, text_encoder):
        self.model_dir = Path(model_dir)
        self.device = device
        self.model, self.config = load_ardy_model(
            self.model_dir.name,
            device='cpu',
            checkpoints_dir=str(self.model_dir.parent),
            text_encoder=False,
            return_config=True,
        )
        self.model.text_encoder = text_encoder
        self.patcher = ModelPatcher(self.model, load_device=device, offload_device=torch.device('cpu'))
        self.runner = ArdyRunner(self)

    def load_for_execution(self):
        mm.load_models_gpu([self.patcher], force_full_load=True)

    def offload(self):
        mm.unload_model_and_clones(self.patcher)


def load_model(model_path, device, text_encoder):
    return ArdyModel(model_path, device, text_encoder)


def load_text_encoder(text_encoder="auto", device="cuda:0"):
    if device not in ("cuda:0", "cpu"):
        raise ValueError(f"Unknown ARDY text encoder device: {device}")
    if text_encoder not in ("auto", "int8_convrot", "bf16"):
        raise ValueError(f"Unknown ARDY text encoder: {text_encoder}")
    root = get_ardy_directory() / "text_encoder"
    quantized = root / "Meta-Llama-3-8B-Instruct_Int8_Convrot" / "meta_llama_3_8B_Instruct_comfy_int8_convrot.safetensors"
    use_quantized = text_encoder in ("auto", "int8_convrot")
    ensure_text_encoder(root, use_quantized)
    return QuantizedTextEncoder(quantized, root, device) if use_quantized else LocalTextEncoder(device)
