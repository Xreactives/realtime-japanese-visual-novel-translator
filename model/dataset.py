import csv
import json
from pathlib import Path
import random
import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import Dataset


def load_vocab(vocab_path: Path) -> tuple[dict[str, int], dict[int, str]]:
    with open(vocab_path, encoding="utf-8") as f:
        payload = json.load(f)
    char_to_idx = payload["char_to_idx"]
    idx_to_char = {int(k): v for k, v in payload["idx_to_char"].items()}
    return char_to_idx, idx_to_char


def ctc_greedy_decode(logits: torch.Tensor, idx_to_char: dict[int, str]) -> list[str]:
    """Decodes (B, T, C) logits using greedy CTC collapsing."""
    pred_ids = torch.argmax(logits, dim=-1).cpu().tolist()
    decoded_texts = []
    for seq in pred_ids:
        chars = []
        prev = None
        for idx in seq:
            if idx != 0 and idx != prev:  # 0 = <blank>
                ch = idx_to_char.get(idx, "")
                if ch not in {"<blank>", "<unk>"}:
                    chars.append(ch)
            prev = idx
        decoded_texts.append("".join(chars))
    return decoded_texts


def compute_cer(preds: list[str], targets: list[str]) -> float:
    """Computes average Character Error Rate (Levenshtein distance / target length)."""
    total_edits = 0
    total_chars = 0
    for pred, target in zip(preds, targets):
        total_chars += max(1, len(target))
        # Dynamic programming edit distance
        dp = list(range(len(target) + 1))
        for i, p_ch in enumerate(pred, 1):
            prev_diag = dp[0]
            dp[0] = i
            for j, t_ch in enumerate(target, 1):
                temp = dp[j]
                cost = 0 if p_ch == t_ch else 1
                dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev_diag + cost)
                prev_diag = temp
        total_edits += dp[-1]
    return total_edits / max(1, total_chars)


class VNTextDataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        img_dir: Path,
        char_to_idx: dict[str, int],
        input_height: int = 32,
        augment: bool = False,
        rows_override: list[dict] | None = None,
    ):
        self.img_dir = Path(img_dir)
        self.char_to_idx = char_to_idx
        self.unk_idx = char_to_idx.get("<unk>", 1)
        self.input_height = input_height
        self.augment = augment

        if rows_override is not None:
            raw_rows = rows_override
        else:
            with open(csv_path, encoding="utf-8") as f:
                raw_rows = list(csv.DictReader(f))

        self.samples: list[tuple[Path, str, list[int]]] = []
        for row in raw_rows:
            label = row["label"].strip()
            img_path = self.img_dir / row["filename"]
            if not label or not img_path.exists():
                continue
            encoded = [self.char_to_idx.get(ch, self.unk_idx) for ch in label]
            self.samples.append((img_path, label, encoded))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label, encoded = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        # Resize proportionally to height = 32
        orig_w, orig_h = img.size
        scale = self.input_height / max(1, orig_h)
        new_w = max(16, int(round(orig_w * scale)))
        img = img.resize((new_w, self.input_height), Image.Resampling.BILINEAR)

        if self.augment:
            if random.random() < 0.5:
                img = ImageEnhance.Brightness(img).enhance(random.uniform(0.85, 1.15))
            if random.random() < 0.5:
                img = ImageEnhance.Contrast(img).enhance(random.uniform(0.85, 1.15))

        arr = np.asarray(img, dtype=np.float32) / 255.0
        if self.augment and random.random() < 0.35:
            arr += np.random.normal(0.0, 0.02, arr.shape).astype(np.float32)
            arr = np.clip(arr, 0.0, 1.0)

        # Normalize to [-1, 1] and convert (H, W, 3) -> (3, H, W)
        arr = (arr - 0.5) / 0.5
        tensor_img = torch.from_numpy(arr).permute(2, 0, 1)
        target_tensor = torch.tensor(encoded, dtype=torch.long)

        return tensor_img, target_tensor, label


def ctc_collate_fn(batch):
    """
    Pads variable-width images to the longest width in the batch (rounded up to
    a multiple of 8, min 64px) and ensures T >= max_target_len for CTCLoss.
    """
    imgs, targets, labels = zip(*batch)
    max_target_len = max(len(t) for t in targets)

    # CRNN outputs T = (W // 8) - 3; ensure T >= max_target_len + 2
    min_required_w = (max_target_len + 5) * 8
    raw_max_w = max(max(img.shape[2] for img in imgs), min_required_w, 64)
    padded_w = ((raw_max_w + 7) // 8) * 8

    batch_size = len(imgs)
    padded_imgs = torch.zeros((batch_size, 3, 32, padded_w), dtype=torch.float32)
    input_lengths = torch.full((batch_size,), (padded_w // 8) - 3, dtype=torch.long)
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)

    for i, img in enumerate(imgs):
        w = img.shape[2]
        padded_imgs[i, :, :, :w] = img

    concat_targets = torch.cat(targets)
    return padded_imgs, concat_targets, input_lengths, target_lengths, list(labels)