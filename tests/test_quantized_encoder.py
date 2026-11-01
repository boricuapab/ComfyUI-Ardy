from pathlib import Path
import sys
import unittest

import torch
from transformers import LlamaConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parents[1]), str(ROOT / "vendor" / "ardy")]

import comfy.ops
from comfy.text_encoders.llama import Llama2Config
from ardy.model.llm2vec.models.bidirectional_llama import LlamaBiModel
from quantized_encoder import BidirectionalLlama


class BidirectionalEncoderTests(unittest.TestCase):
    def test_native_blocks_match_llm2vec_attention(self):
        torch.manual_seed(42)
        config = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            rms_norm_eps=1e-5, rope_theta=500000.0)
        reference = LlamaBiModel(config).to(dtype=torch.bfloat16).eval()
        native_config = Llama2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)
        native_config.head_dim = 8
        native = BidirectionalLlama(native_config, device="cpu", dtype=torch.bfloat16,
                                    ops=comfy.ops.manual_cast)
        native.load_state_dict(reference.state_dict(), strict=True)
        ids = torch.tensor([[0, 0, 3, 4, 5], [1, 2, 3, 4, 5]])
        mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
        with torch.set_grad_enabled(False):
            # Transformers 5 bypasses upstream's _update_causal_mask override.
            full_mask = torch.zeros(2, 1, 5, 5, dtype=torch.bfloat16)
            full_mask.masked_fill_(~mask[:, None, None, :].bool(), torch.finfo(torch.bfloat16).min)
            expected = reference(input_ids=ids, attention_mask=full_mask).last_hidden_state
            actual = native.encode_bidirectional(ids, mask)
            torch.testing.assert_close(actual[mask.bool()], expected[mask.bool()], atol=.04, rtol=.04)
            changed = ids.clone()
            changed[:, -1] = 12
            changed_output = native.encode_bidirectional(changed, mask)
            self.assertGreater((changed_output[:, 2] - actual[:, 2]).abs().max().item(), .01)
