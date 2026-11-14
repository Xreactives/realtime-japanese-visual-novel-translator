import csv
from pathlib import Path
import sys
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import torch
from torch.utils.data import DataLoader

# Ensure project root is importable when running from pipeline/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.crnn import CRNN
from model.dataset import VNTextDataset, compute_cer, ctc_collate_fn, ctc_greedy_decode, load_vocab

VOCAB_PATH = PROJECT_ROOT / "model" / "vocab.json"
WEIGHTS_PATH = PROJECT_ROOT / "weights" / "crnn_vn_best.pt"
REAL_CSV = PROJECT_ROOT / "data" / "verified_labels.csv"
REAL_IMG_DIR = PROJECT_ROOT / "data" / "line_crops"


def collect_mismatches() -> tuple[list[dict], list[dict]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    char_to_idx, idx_to_char = load_vocab(VOCAB_PATH)

    ckpt = torch.load(WEIGHTS_PATH, map_location=device, weights_only=True)
    model = CRNN(input_height=ckpt["input_height"], num_classes=ckpt["num_classes"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with open(REAL_CSV, encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))

    dataset = VNTextDataset(
        REAL_CSV, REAL_IMG_DIR, char_to_idx, input_height=32, augment=False, rows_override=all_rows
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=ctc_collate_fn)

    mismatches = []
    sample_idx = 0

    with torch.inference_mode():
        for imgs, _, _, _, labels in loader:
            preds = ctc_greedy_decode(model(imgs.to(device)), idx_to_char)
            for gt, pred in zip(labels, preds):
                img_path = dataset.samples[sample_idx][0]
                sample_idx += 1
                if gt != pred:
                    cer = compute_cer([pred], [gt])
                    mismatches.append(
                        {
                            "cer": cer,
                            "filename": img_path.name,
                            "img_path": img_path,
                            "csv_label": gt,
                            "crnn_pred": pred,
                        }
                    )

    mismatches.sort(key=lambda x: x["cer"], reverse=True)
    return all_rows, mismatches


class NoisyLabelAuditGUI:
    def __init__(self, root: tk.Tk, all_rows: list[dict], mismatches: list[dict]):
        self.root = root
        self.root.title("CRNN Post-Training Label Auditor")
        self.root.geometry("920x460")
        self.root.configure(bg="#1e1e24")

        self.all_rows = all_rows
        self.row_map = {r["filename"]: r for r in all_rows}
        self.mismatches = mismatches
        self.idx = 0
        self.photo_ref = None

        # Header status
        self.status_var = tk.StringVar()
        tk.Label(
            root, textvariable=self.status_var, font=("Segoe UI", 11, "bold"), fg="#9cdcfe", bg="#1e1e24"
        ).pack(pady=(12, 6))

        # Image preview canvas
        self.img_label = tk.Label(root, bg="#111116", bd=2, relief="solid")
        self.img_label.pack(pady=8, padx=20)

        # CRNN prediction display (read-only + quick copy button)
        pred_frame = tk.Frame(root, bg="#1e1e24")
        pred_frame.pack(fill="x", padx=24, pady=(8, 4))
        tk.Label(pred_frame, text="CRNN Pred:", font=("Segoe UI", 10, "bold"), fg="#4ec9b0", bg="#1e1e24", width=11, anchor="w").pack(side="left")
        self.pred_var = tk.StringVar()
        tk.Entry(
            pred_frame, textvariable=self.pred_var, font=("Yu Gothic UI", 13), state="readonly",
            readonlybackground="#2a2a32", fg="#dcdcaa", relief="flat"
        ).pack(side="left", fill="x", expand=True, padx=(0, 8))
        tk.Button(
            pred_frame, text="Use CRNN Pred", command=self.use_crnn_pred,
            bg="#2d5a4c", fg="white", font=("Segoe UI", 9, "bold"), relief="flat", padx=10
        ).pack(side="right")

        # Editable CSV ground-truth entry
        gt_frame = tk.Frame(root, bg="#1e1e24")
        gt_frame.pack(fill="x", padx=24, pady=(4, 12))
        tk.Label(gt_frame, text="CSV Label:", font=("Segoe UI", 10, "bold"), fg="#ce9178", bg="#1e1e24", width=11, anchor="w").pack(side="left")
        self.gt_var = tk.StringVar()
        self.gt_entry = tk.Entry(
            gt_frame, textvariable=self.gt_var, font=("Yu Gothic UI", 13),
            bg="#2d2d36", fg="white", insertbackground="white", relief="flat"
        )
        self.gt_entry.pack(side="left", fill="x", expand=True)
        self.gt_entry.bind("<Return>", lambda _: self.save_and_next())

        # Action buttons
        btn_frame = tk.Frame(root, bg="#1e1e24")
        btn_frame.pack(pady=10)

        tk.Button(btn_frame, text="← Prev", command=self.prev_item, width=10, bg="#3a3a44", fg="white", relief="flat").pack(side="left", padx=5)
        tk.Button(btn_frame, text="Keep CSV (Skip)", command=self.next_item, width=14, bg="#3a3a44", fg="white", relief="flat").pack(side="left", padx=5)
        tk.Button(btn_frame, text="Save Edit & Next (Enter)", command=self.save_and_next, width=22, bg="#0e639c", fg="white", font=("Segoe UI", 9, "bold"), relief="flat").pack(side="left", padx=5)
        tk.Button(btn_frame, text="Delete Bad Crop", command=self.delete_row, width=15, bg="#a1260d", fg="white", relief="flat").pack(side="left", padx=5)

        tk.Label(
            root, text="Tip: Click 'Use CRNN Pred' if the model read the image right and your CSV had a typo.",
            font=("Segoe UI", 9), fg="#858585", bg="#1e1e24"
        ).pack(pady=(6, 0))

        self.load_current()

    def load_current(self):
        if not self.mismatches:
            messagebox.showinfo("Done", "No mismatches remaining!")
            self.root.destroy()
            return

        self.idx = max(0, min(self.idx, len(self.mismatches) - 1))
        item = self.mismatches[self.idx]

        self.status_var.set(
            f"Mismatch {self.idx + 1} / {len(self.mismatches)}   |   "
            f"File: {item['filename']}   |   Disagreement CER: {item['cer'] * 100:.1f}%"
        )
        self.pred_var.set(item["crnn_pred"])
        self.gt_var.set(item["csv_label"])

        # Upscale crop slightly for comfortable reading while preserving aspect ratio
        img = Image.open(item["img_path"]).convert("RGB")
        target_h = 64
        scale = target_h / max(1, img.height)
        new_w = min(860, max(60, int(round(img.width * scale))))
        img = img.resize((new_w, target_h), Image.Resampling.BILINEAR)
        self.photo_ref = ImageTk.PhotoImage(img)
        self.img_label.configure(image=self.photo_ref)
        self.gt_entry.focus_set()

    def use_crnn_pred(self):
        self.gt_var.set(self.pred_var.get())

    def save_csv(self):
        if not self.all_rows:
            return
        fieldnames = list(self.all_rows[0].keys())
        tmp_path = REAL_CSV.with_suffix(".csv.tmp")
        with open(tmp_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.all_rows)
        tmp_path.replace(REAL_CSV)

    def save_and_next(self):
        item = self.mismatches[self.idx]
        new_label = self.gt_var.get().strip()
        if not new_label:
            messagebox.showwarning("Empty Label", "Label cannot be empty. Use 'Delete Bad Crop' to remove it.")
            return

        if item["filename"] in self.row_map:
            self.row_map[item["filename"]]["label"] = new_label
            item["csv_label"] = new_label
            self.save_csv()

        self.next_item()

    def delete_row(self):
        item = self.mismatches[self.idx]
        fname = item["filename"]
        self.all_rows = [r for r in self.all_rows if r["filename"] != fname]
        self.row_map.pop(fname, None)
        self.save_csv()

        self.mismatches.pop(self.idx)
        self.load_current()

    def next_item(self):
        if self.idx + 1 < len(self.mismatches):
            self.idx += 1
            self.load_current()
        else:
            messagebox.showinfo("Audit Complete", "Reached the end of the mismatch list. verified_labels.csv is saved!")

    def prev_item(self):
        if self.idx > 0:
            self.idx -= 1
            self.load_current()


def main():
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Trained weights not found at {WEIGHTS_PATH}. Finish Phase 2 training first.")

    print("Running CRNN inference over verified_labels.csv to rank label disagreements...")
    all_rows, mismatches = collect_mismatches()
    print(f"Found {len(mismatches)} mismatches out of {len(all_rows)} rows. Launching GUI...")

    root = tk.Tk()
    NoisyLabelAuditGUI(root, all_rows, mismatches)
    root.mainloop()


if __name__ == "__main__":
    main()