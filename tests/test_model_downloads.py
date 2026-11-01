from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_downloads import ensure_model, ensure_text_encoder


class DownloadTests(unittest.TestCase):
    def test_complete_model_stays_offline_and_partial_model_resumes(self):
        with tempfile.TemporaryDirectory() as folder, patch('model_downloads.snapshot_download') as download:
            path = Path(folder)
            (path / 'config.json').write_text('{}')
            ensure_model(path, 'example/model', ['*.json'], ['config.json'])
            download.assert_not_called()
            download.side_effect = lambda **kwargs: (path / 'weights.safetensors').touch()
            ensure_model(path, 'example/model', ['*.safetensors'], ['config.json', 'weights.safetensors'])
            download.assert_called_once()

    def test_int8_downloads_adapters_and_ungated_weights_only(self):
        with patch('model_downloads.ensure_model') as download:
            ensure_text_encoder(Path('models'), True)
            self.assertEqual(len(download.call_args_list), 3)
            self.assertEqual(download.call_args_list[-1].args[1],
                             'boricuapab/meta_llama_3_8B_Instruct_comfy_int8_convrot')
