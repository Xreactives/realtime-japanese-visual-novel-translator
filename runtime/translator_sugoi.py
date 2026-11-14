from collections import OrderedDict
import os
from pathlib import Path
import ctranslate2
import sentencepiece as spm

from runtime.device_utils import can_use_cuda_for_ctranslate2


class SugoiV4Translator:
    def __init__(
        self,
        ct2_model_dir: str | Path = "weights/sugoi-v4-ja-en-ctranslate2",
        ja_spm_path: str | Path = "weights/sugoi-v4-ja-en-ctranslate2/spm/spm.ja.nopretok.model",
        en_spm_path: str | Path = "weights/sugoi-v4-ja-en-ctranslate2/spm/spm.en.nopretok.model",
        device: str = "auto",
        compute_type: str | None = None,
        cache_size: int = 512,
    ):
        self.ct2_model_dir = str(ct2_model_dir)
        self.sp_ja = spm.SentencePieceProcessor(model_file=str(ja_spm_path))
        self.sp_en = spm.SentencePieceProcessor(model_file=str(en_spm_path))
        self.cache: OrderedDict[str, str] = OrderedDict()
        self.cache_size = cache_size

        if device == "auto":
            device = "cuda" if can_use_cuda_for_ctranslate2() else "cpu"

        self._init_engine(device, compute_type)

    def _init_engine(self, device: str, compute_type: str | None = None) -> None:
        if compute_type is None:
            compute_type = "float16" if device == "cuda" else "int8"

        cpu_threads = max(2, (os.cpu_count() or 4) // 2) if device == "cpu" else 0
        self.device = device
        self.translator = ctranslate2.Translator(
            self.ct2_model_dir,
            device=device,
            compute_type=compute_type,
            intra_threads=cpu_threads,
        )

    def _run_ct2(self, tokens: list[str]) -> list[str]:
        results = self.translator.translate_batch(
            [tokens],
            beam_size=1 if self.device == "cpu" else 2,
            max_decoding_length=256,
            no_repeat_ngram_size=3,
        )
        return results[0].hypotheses[0]

    def translate(self, text: str) -> str:
        text = text.strip()
        if not text:
            return ""

        if text in self.cache:
            self.cache.move_to_end(text)
            return self.cache[text]

        tokens = self.sp_ja.encode(text, out_type=str)

        try:
            out_tokens = self._run_ct2(tokens)
        except RuntimeError as err:
            if self.device == "cuda" and "cublas" in str(err).lower():
                print("[Info] cuBLAS failed during inference; switching Sugoi V4 to CPU INT8...")
                self._init_engine(device="cpu", compute_type="int8")
                out_tokens = self._run_ct2(tokens)
            else:
                raise

        translated = self.sp_en.decode(out_tokens).strip()

        self.cache[text] = translated
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)

        return translated