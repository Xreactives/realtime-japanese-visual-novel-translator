import csv
from pathlib import Path
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from model.crnn import CRNN
from model.dataset import VNTextDataset, compute_cer, ctc_collate_fn, ctc_greedy_decode, load_vocab

# --- Paths ---
VOCAB_PATH = Path("model/vocab.json")
SYNTH_CSV = Path("data/synthetic_trdg/labels.csv")
SYNTH_IMG_DIR = Path("data/synthetic_trdg/images")
REAL_CSV = Path("data/verified_labels.csv")
REAL_IMG_DIR = Path("data/line_crops")
WEIGHTS_DIR = Path("weights")
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate(
    model: nn.Module, loader: DataLoader, idx_to_char: dict[int, str], device: torch.device
) -> tuple[float, float]:
    model.eval()
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    total_loss = 0.0
    all_preds: list[str] = []
    all_targets: list[str] = []

    with torch.inference_mode():
        for imgs, targets, input_lengths, target_lengths, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(imgs)  # (B, T, C)
                log_probs = F.log_softmax(logits.float(), dim=-1).permute(1, 0, 2)
                loss = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)

            total_loss += loss.item() * len(labels)
            preds = ctc_greedy_decode(logits, idx_to_char)
            all_preds.extend(preds)
            all_targets.extend(labels)

    avg_loss = total_loss / max(1, len(all_targets))
    cer = compute_cer(all_preds, all_targets)
    return avg_loss, cer


def train_phase(
    phase_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    idx_to_char: dict[int, str],
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    grad_accum_steps: int,
    save_path: Path,
):
    print(f"\n{'=' * 65}")
    print(
        f"Starting {phase_name} | Epochs: {epochs} | LR: {lr} | "
        f"Accum: {grad_accum_steps}x | Device: {device}"
    )
    print(f"{'=' * 65}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.05)
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_cer = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        sample_count = 0
        optimizer.zero_grad(set_to_none=True)

        for step, (imgs, targets, input_lengths, target_lengths, labels) in enumerate(train_loader, start=1):
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(imgs)  # (B, T, C)
                log_probs = F.log_softmax(logits.float(), dim=-1).permute(1, 0, 2)
                loss = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)
                scaled_loss = loss / grad_accum_steps

            scaler.scale(scaled_loss).backward()

            if step % grad_accum_steps == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item() * len(labels)
            sample_count += len(labels)

        scheduler.step()
        train_loss = running_loss / max(1, sample_count)
        val_loss, val_cer = evaluate(model, val_loader, idx_to_char, device)

        marker = ""
        if val_cer < best_cer:
            best_cer = val_cer
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_height": 32,
                    "num_classes": len(idx_to_char),
                    "val_cer": val_cer,
                },
                save_path,
            )
            marker = "  <-- Saved Best"

        print(
            f"[{phase_name}] Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val CER: {val_cer * 100:.2f}%{marker}"
        )

    # Preview 5 validation predictions at the end of the phase
    model.eval()
    imgs, _, _, _, labels = next(iter(val_loader))
    with torch.inference_mode():
        preds = ctc_greedy_decode(model(imgs[:5].to(device)), idx_to_char)
    print(f"\nSample predictions after {phase_name}:")
    for gt, pr in zip(labels[:5], preds):
        print(f"  GT  : {gt}")
        print(f"  PRED: {pr}\n")


def main(
    phase1_batch_size: int = 16,
    phase1_grad_accum: int = 4,
    phase1_epochs: int = 12,
    phase1_lr: float = 1e-3,
    phase2_batch_size: int = 16,
    phase2_grad_accum: int = 2,
    phase2_epochs: int = 35,
    phase2_lr: float = 2e-4,
    weight_decay: float = 1e-4,
    skip_phase1: bool = False,
    num_workers: int = 2,
    seed: int = 42,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    char_to_idx, idx_to_char = load_vocab(VOCAB_PATH)
    num_classes = len(char_to_idx)
    print(f"Loaded vocabulary: {num_classes} classes")

    model = CRNN(input_height=32, num_classes=num_classes).to(device)
    synth_ckpt = WEIGHTS_DIR / "crnn_pretrained_synth.pt"
    best_vn_ckpt = WEIGHTS_DIR / "crnn_vn_best.pt"

    # --- PHASE 1: Synthetic Pretraining ---
    if not skip_phase1:
        full_synth = VNTextDataset(SYNTH_CSV, SYNTH_IMG_DIR, char_to_idx, input_height=32, augment=False)
        val_size = max(500, int(0.05 * len(full_synth)))
        train_size = len(full_synth) - val_size
        synth_train, synth_val = random_split(
            full_synth, [train_size, val_size], generator=torch.Generator().manual_seed(seed)
        )

        synth_train_loader = DataLoader(
            synth_train,
            batch_size=phase1_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=ctc_collate_fn,
        )
        synth_val_loader = DataLoader(
            synth_val,
            batch_size=phase1_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=ctc_collate_fn,
        )

        train_phase(
            phase_name="Phase 1 (Synthetic Pretrain)",
            model=model,
            train_loader=synth_train_loader,
            val_loader=synth_val_loader,
            idx_to_char=idx_to_char,
            device=device,
            epochs=phase1_epochs,
            lr=phase1_lr,
            weight_decay=weight_decay,
            grad_accum_steps=phase1_grad_accum,
            save_path=synth_ckpt,
        )

        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Load best synthetic weights before Phase 2
    if synth_ckpt.exists():
        ckpt = torch.load(synth_ckpt, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded synthetic checkpoint ({synth_ckpt}) with Val CER: {ckpt['val_cer'] * 100:.2f}%")

    # --- PHASE 2: Real Visual Novel Fine-Tuning ---
    with open(REAL_CSV, encoding="utf-8") as f:
        real_rows = list(csv.DictReader(f))
    random.Random(seed).shuffle(real_rows)

    split_idx = max(1, int(0.9 * len(real_rows)))
    real_train = VNTextDataset(
        REAL_CSV, REAL_IMG_DIR, char_to_idx, input_height=32, augment=True, rows_override=real_rows[:split_idx]
    )
    real_val = VNTextDataset(
        REAL_CSV, REAL_IMG_DIR, char_to_idx, input_height=32, augment=False, rows_override=real_rows[split_idx:]
    )

    real_train_loader = DataLoader(
        real_train,
        batch_size=min(phase2_batch_size, len(real_train)),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=ctc_collate_fn,
    )
    real_val_loader = DataLoader(
        real_val,
        batch_size=min(phase2_batch_size, len(real_val)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=ctc_collate_fn,
    )

    train_phase(
        phase_name="Phase 2 (Real VN Fine-Tune)",
        model=model,
        train_loader=real_train_loader,
        val_loader=real_val_loader,
        idx_to_char=idx_to_char,
        device=device,
        epochs=phase2_epochs,
        lr=phase2_lr,
        weight_decay=weight_decay,
        grad_accum_steps=phase2_grad_accum,
        save_path=best_vn_ckpt,
    )

    print(f"\nTraining complete! Production weights saved to {best_vn_ckpt.resolve()}")


if __name__ == "__main__":
    main(
        phase1_batch_size=32,
        phase1_grad_accum=1,
        phase1_epochs=12,
        phase1_lr=1e-3,
        phase2_batch_size=16,
        phase2_grad_accum=1,
        phase2_epochs=35,
        phase2_lr=2e-4,
        weight_decay=1e-4,
        skip_phase1=False,
        num_workers=2,
        seed=42,
    )