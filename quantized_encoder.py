"""ComfyUI quantized Llama with LLM2Vec's bidirectional attention and pooling."""
import json

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

import comfy.lora
import comfy.model_management as mm
import comfy.model_patcher
import comfy.ops
import comfy.utils
from comfy.ldm.modules.attention import optimized_attention_for_device
from comfy.text_encoders.llama import Llama2_, Llama2Config
from comfy.weight_adapter.bypass import BypassForwardHook
from comfy.patcher_extension import PatcherInjection
from comfy.weight_adapter.lora import LoRAAdapter
from ardy.model.llm2vec.llm2vec import LLM2Vec


class BidirectionalLlama(Llama2_):
    def encode_bidirectional(self, input_ids, attention_mask):
        x = self.embed_tokens(input_ids, out_dtype=torch.bfloat16)
        positions = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        freqs = self.compute_freqs_cis(positions, x.device)
        mask = torch.zeros((x.shape[0], 1, 1, x.shape[1]), device=x.device, dtype=x.dtype)
        mask.masked_fill_(~attention_mask[:, None, None, :].bool(), torch.finfo(x.dtype).min)
        attention = optimized_attention_for_device(x.device, mask=True, small_input=True)
        for layer in self.layers:
            x, _ = layer(x, attention_mask=mask, freqs_cis=freqs, optimized_attention=attention)
        return self.norm(x)


class EncoderAdapterHook(BypassForwardHook):
    def __init__(self, module, adapter, device):
        super().__init__(module, adapter)
        self.compute_device = device

    def _move_adapter_weights_to_device(self, device, dtype=None):
        super()._move_adapter_weights_to_device(self.compute_device, dtype)

    def eject(self):
        super().eject()
        super()._move_adapter_weights_to_device(torch.device('cpu'))


class QuantizedTextEncoder:
    def __init__(self, checkpoint, root, device="cuda:0"):
        device = torch.device(device)
        with safe_open(str(checkpoint), framework="pt", device="cpu") as file:
            state = {key.removeprefix("model."): file.get_tensor(key)
                     for key in file.keys() if key.startswith("model.")}
        quant = comfy.utils.detect_layer_quantization(state, "")
        if quant is None:
            raise ValueError(f"No ComfyUI quantization metadata in {checkpoint.name}")
        config = Llama2Config(vocab_size=state["embed_tokens.weight"].shape[0])
        config._name_or_path = "meta-llama/Meta-Llama-3-8B-Instruct"
        model = BidirectionalLlama(config, device=torch.device("cpu"), dtype=torch.bfloat16,
                                   ops=comfy.ops.mixed_precision_ops(quant, torch.bfloat16))
        model.load_state_dict(state, strict=True)
        del state
        self.patcher = comfy.model_patcher.ModelPatcher(model,
            load_device=device, offload_device=torch.device("cpu"))
        combined = {}
        for name in ("mntp", "mntp-supervised"):
            folder = root / f"LLM2Vec-Meta-Llama-3-8B-Instruct-{name}"
            adapter_config = json.loads((folder / "adapter_config.json").read_text())
            weights = comfy.utils.load_torch_file(str(folder / "adapter_model.safetensors"), safe_load=True)
            mapping = {key.removesuffix(".lora_A.weight"): key.removesuffix(".lora_A.weight").removeprefix("base_model.model.") + ".weight"
                       for key in weights if key.endswith(".lora_A.weight")}
            patches = comfy.lora.load_lora(weights, mapping)
            if len(patches) != config.num_hidden_layers * 7:
                raise ValueError(f"Incomplete ARDY LLM2Vec adapter: {folder}")
            for key, adapter in patches.items():
                model.get_submodule(key.removesuffix(".weight"))
                up, down = adapter.weights[:2]
                scale = adapter_config["lora_alpha"] / adapter_config["r"]
                combined.setdefault(key, []).append((up * scale, down))
        hooks = []
        for key, pairs in combined.items():
            up = torch.cat([pair[0] for pair in pairs], dim=1)
            down = torch.cat([pair[1] for pair in pairs], dim=0)
            hooks.append(EncoderAdapterHook(model.get_submodule(key.removesuffix(".weight")),
                LoRAAdapter(set(), (up, down, None, None, None, None)), device))
        def inject(patcher):
            for hook in hooks:
                hook.inject()
        def eject(patcher):
            for hook in hooks:
                hook.eject()
        self.patcher.set_injections("llm2vec", [PatcherInjection(inject=inject, eject=eject)])
        tokenizer = AutoTokenizer.from_pretrained(
            str(root / "LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"), local_files_only=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        self.encoder = LLM2Vec(model, tokenizer).eval()

    def __call__(self, texts):
        mm.load_models_gpu([self.patcher])
        try:
            return self._encode(texts)
        finally:
            mm.unload_model_and_clones(self.patcher)

    def _encode(self, texts):
        device = self.patcher.load_device
        embeddings = []
        for text in texts:
            prepared = self.encoder.prepare_for_tokenization(self.encoder._convert_to_str("", text))
            features = {key: value.to(device) for key, value in self.encoder.tokenize([prepared]).items()}
            hidden = self.patcher.model.encode_bidirectional(features["input_ids"], features["attention_mask"])
            embeddings.append(self.encoder.get_pooling(features, hidden).detach().float().cpu())
        return torch.cat(embeddings)[:, None], [1] * len(texts)
