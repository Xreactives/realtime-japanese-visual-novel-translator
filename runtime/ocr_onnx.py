import json
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort


class ONNXVNRecognizer:
    def __init__(
        self,
        onnx_path: str | Path = "weights/crnn_vn.onnx",
        vocab_path: str | Path = "model/vocab.json",
        use_gpu: bool = True,
    ):
        with open(vocab_path, encoding="utf-8") as f:
            raw_vocab = json.load(f)

        if isinstance(raw_vocab, dict) and "idx_to_char" in raw_vocab:
            self.idx_to_char: dict[int, str] = {
                int(idx): ch for idx, ch in raw_vocab["idx_to_char"].items()
            }
        elif isinstance(raw_vocab, dict) and "char_to_idx" in raw_vocab:
            self.idx_to_char = {
                int(idx): ch for ch, idx in raw_vocab["char_to_idx"].items()
            }
        elif isinstance(raw_vocab, list):
            self.idx_to_char = {idx: ch for idx, ch in enumerate(raw_vocab)}
        else:
            self.idx_to_char = {int(idx): ch for ch, idx in raw_vocab.items()}

        self.input_height = 32

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if use_gpu
            else ["CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(str(onnx_path), providers=providers)

    def _preprocess_batch(self, line_crops_bgr: list[np.ndarray]) -> np.ndarray:
        resized_imgs = []
        max_w = 16  # minimum safe width for CNN pooling

        for img in line_crops_bgr:
            h, w = img.shape[:2]
            scale = self.input_height / max(1, h)
            new_w = max(16, int(round(w * scale)))
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
            resized = cv2.resize(rgb, (new_w, self.input_height), interpolation=interp)
            resized_imgs.append(resized)
            if new_w > max_w:
                max_w = new_w

        batch = np.zeros((len(resized_imgs), 3, self.input_height, max_w), dtype=np.float32)
        for i, img in enumerate(resized_imgs):
            # Normalize [0, 255] -> [-1.0, 1.0] matching VNTextDataset
            norm = (img.astype(np.float32) / 255.0 - 0.5) / 0.5
            # HWC -> CHW; right-side padding stays 0.0 (= mid-gray 128)
            batch[i, :, :, : img.shape[1]] = np.transpose(norm, (2, 0, 1))

        return batch

    def _ctc_greedy_decode(self, logits: np.ndarray) -> list[str]:
        # logits: (B, T, C)
        pred_indices = np.argmax(logits, axis=-1)
        results = []
        for seq in pred_indices:
            chars = []
            prev = 0
            for idx in seq:
                idx_int = int(idx)
                if idx_int != 0 and idx_int != prev:
                    chars.append(self.idx_to_char.get(idx_int, ""))
                prev = idx_int
            results.append("".join(chars))
        return results

    def recognize(self, line_crops_bgr: list[np.ndarray]) -> list[str]:
        if not line_crops_bgr:
            return []
        batch_tensor = self._preprocess_batch(line_crops_bgr)
        logits = self.session.run(["logits"], {"input": batch_tensor})[0]
        return self._ctc_greedy_decode(logits)