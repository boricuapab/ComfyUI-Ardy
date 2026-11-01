"""Download missing model artifacts when a loader executes."""
import os
from pathlib import Path

os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
from huggingface_hub import snapshot_download

CORE_MODEL = 'ARDY-Core-RP-20FPS-Horizon40'
INT8_FILE = 'meta_llama_3_8B_Instruct_comfy_int8_convrot.safetensors'


def ensure_model(folder, repo, patterns, required):
    folder = Path(folder)
    if all((folder / name).is_file() for name in required):
        return
    snapshot_download(repo_id=repo, local_dir=str(folder), allow_patterns=patterns)
    missing = [name for name in required if not (folder / name).is_file()]
    if missing:
        raise FileNotFoundError(f'{repo} is missing required files: {missing}')


def ensure_text_encoder(root, quantized):
    for suffix in ('mntp', 'mntp-supervised'):
        name = f'LLM2Vec-Meta-Llama-3-8B-Instruct-{suffix}'
        required = ['adapter_config.json', 'adapter_model.safetensors']
        if suffix == 'mntp':
            required += ['tokenizer.json', 'tokenizer_config.json']
        ensure_model(root / name, f'McGill-NLP/{name}',
                     ['*.json', '*.safetensors', '*.model'], required)
    if quantized:
        ensure_model(root / 'Meta-Llama-3-8B-Instruct_Int8_Convrot',
                     'boricuapab/meta_llama_3_8B_Instruct_comfy_int8_convrot',
                     [INT8_FILE], [INT8_FILE])
    else:
        ensure_model(root / 'Meta-Llama-3-8B-Instruct', 'meta-llama/Meta-Llama-3-8B-Instruct',
                     ['*.json', '*.safetensors', '*.model'],
                     ['config.json', 'model.safetensors.index.json',
                      *[f'model-0000{i}-of-00004.safetensors' for i in range(1, 5)]])
