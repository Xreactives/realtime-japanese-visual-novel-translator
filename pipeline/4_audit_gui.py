import csv
from difflib import SequenceMatcher
from pathlib import Path
import tkinter as tk
from tkinter import messagebox
import unicodedata
from PIL import Image, ImageTk

# --- Paths & Config ---
CROPS_DIR = Path("../data/line_crops")
RAW_CSV = Path("../data/raw_labels.csv")
VERIFIED_CSV = Path("../data/verified_labels.csv")
DISCARDED_LOG = Path("../data/discarded_crops.txt")

VERIFIED_FIELDS = ["filename", "source_img", "role", "label"]
PROTECTED_BRACKETS = set("「」【】『』")


def strip_all_whitespace(text: str) -> str:
    """Removes all spaces/newlines while keeping all symbols and characters intact."""
    return "".join(ch for ch in text if not ch.isspace())


def normalize_for_comparison(text: str) -> str:
    """
    Strips spaces and non-bracket punctuation/symbols for comparison only,
    so minor symbol discrepancies default to Qwen3-VL's symbol choice, while
    discrepancies in '「」【】『』' or actual characters trigger a disagreement.
    """
    cleaned = []
    for ch in text:
        if ch.isspace():
            continue
        if ch in PROTECTED_BRACKETS:
            cleaned.append(ch)
            continue
        cat = unicodedata.category(ch)
        if cat.startswith(("P", "S")):
            continue
        cleaned.append(ch)
    return "".join(cleaned)


def is_effective_agreement(vlm_text: str, easy_text: str) -> bool:
    """
    Returns True if Qwen3-VL and EasyOCR agree on all characters and protected
    brackets ('「」【】『』'). Any non-bracket symbol discrepancy is treated as
    an agreement and defaults to Qwen3-VL's symbols.
    """
    vlm_clean = strip_all_whitespace(vlm_text)
    easy_clean = strip_all_whitespace(easy_text)
    if not vlm_clean or not easy_clean:
        return False
    return normalize_for_comparison(vlm_clean) == normalize_for_comparison(easy_clean)


def append_verified_rows(rows: list[dict]):
    if not rows:
        return
    file_exists = VERIFIED_CSV.exists()
    with open(VERIFIED_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=VERIFIED_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


class DualOCRAuditApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("VN Japanese OCR Audit Tool — Qwen3-VL vs EasyOCR")
        self.root.geometry("1280x680")
        self.root.configure(bg="#18181b")

        self.records, self.auto_approved_count, self.previously_done_count = self._prepare_queue()
        self.index = 0
        self.history: list[tuple[int, str, dict]] = []

        self._build_ui()
        self._bind_shortcuts()
        self.show_record()

    def _prepare_queue(self) -> tuple[list[dict], int, int]:
        if not RAW_CSV.exists():
            raise FileNotFoundError(f"raw_labels.csv not found at {RAW_CSV.resolve()}. Run 3_prelabel.py first.")

        already_done = set()
        if VERIFIED_CSV.exists():
            with open(VERIFIED_CSV, encoding="utf-8") as f:
                already_done.update(r["filename"] for r in csv.DictReader(f))
        if DISCARDED_LOG.exists():
            already_done.update(
                line.strip() for line in DISCARDED_LOG.read_text(encoding="utf-8").splitlines() if line.strip()
            )

        with open(RAW_CSV, encoding="utf-8") as f:
            raw_rows = list(csv.DictReader(f))

        disagreements = []
        auto_verified_batch = []

        for row in raw_rows:
            fname = row["filename"]
            if fname in already_done:
                continue

            vlm_raw = row.get("vlm_pred", "")
            easy_raw = row.get("easyocr_pred", "")

            if is_effective_agreement(vlm_raw, easy_raw):
                auto_verified_batch.append({
                    "filename": fname,
                    "source_img": row["source_img"],
                    "role": row["role"],
                    "label": strip_all_whitespace(vlm_raw),
                })
            else:
                disagreements.append(row)

        append_verified_rows(auto_verified_batch)

        disagreements.sort(key=lambda r: r["filename"])
        return disagreements, len(auto_verified_batch), len(already_done)

    def _build_ui(self):
        # Top Progress & Metadata Bar
        self.status_lbl = tk.Label(
            self.root,
            text="",
            font=("Segoe UI", 12, "bold"),
            fg="#e4e4e7",
            bg="#27272a",
            pady=10,
        )
        self.status_lbl.pack(fill="x")

        # Taller Cropped Image Display Area
        img_frame = tk.Frame(self.root, bg="#09090b", height=190, bd=1, relief="solid")
        img_frame.pack(fill="x", padx=28, pady=(16, 12))
        img_frame.pack_propagate(False)

        self.img_panel = tk.Label(img_frame, bg="#09090b")
        self.img_panel.pack(expand=True)

        # Diff Comparison Panel
        diff_frame = tk.Frame(self.root, bg="#18181b")
        diff_frame.pack(fill="x", padx=28, pady=6)

        tk.Button(
            diff_frame,
            text="[F1] Use Qwen3-VL",
            font=("Segoe UI", 12, "bold"),
            bg="#2563eb",
            fg="white",
            activebackground="#1d4ed8",
            width=18,
            pady=6,
            command=self.pick_vlm,
        ).grid(row=0, column=0, padx=(0, 12), pady=6, sticky="w")

        self.vlm_text_widget = tk.Text(
            diff_frame,
            height=1,
            font=("Yu Gothic UI", 20),
            bg="#27272a",
            fg="#f4f4f5",
            bd=0,
            padx=10,
            pady=6,
            wrap="none",
        )
        self.vlm_text_widget.grid(row=0, column=1, sticky="ew", pady=6)
        self.vlm_text_widget.tag_configure("diff", background="#7f1d1d", foreground="#fecaca")
        self.vlm_text_widget.config(state="disabled")

        tk.Button(
            diff_frame,
            text="[F2] Use EasyOCR",
            font=("Segoe UI", 12, "bold"),
            bg="#4f46e5",
            fg="white",
            activebackground="#4338ca",
            width=18,
            pady=6,
            command=self.pick_easy,
        ).grid(row=1, column=0, padx=(0, 12), pady=6, sticky="w")

        self.easy_text_widget = tk.Text(
            diff_frame,
            height=1,
            font=("Yu Gothic UI", 20),
            bg="#27272a",
            fg="#f4f4f5",
            bd=0,
            padx=10,
            pady=6,
            wrap="none",
        )
        self.easy_text_widget.grid(row=1, column=1, sticky="ew", pady=6)
        self.easy_text_widget.tag_configure("diff", background="#7f1d1d", foreground="#fecaca")
        self.easy_text_widget.config(state="disabled")

        diff_frame.columnconfigure(1, weight=1)

        # Ground Truth Input Box
        lbl_gt = tk.Label(
            self.root,
            text="Final Ground Truth Label (Whitespace auto-stripped on save, symbols preserved):",
            font=("Segoe UI", 12, "bold"),
            fg="#a1a1aa",
            bg="#18181b",
        )
        lbl_gt.pack(anchor="w", padx=28, pady=(14, 4))

        self.entry_var = tk.StringVar()
        self.text_entry = tk.Entry(
            self.root,
            textvariable=self.entry_var,
            font=("Yu Gothic UI", 22),
            bg="#27272a",
            fg="#ffffff",
            insertbackground="#ffffff",
            relief="flat",
        )
        self.text_entry.pack(fill="x", padx=28, ipady=10)

        # Action Buttons
        ctrl_frame = tk.Frame(self.root, bg="#18181b")
        ctrl_frame.pack(pady=20)

        tk.Button(
            ctrl_frame,
            text="[Enter] Save & Next",
            font=("Segoe UI", 12, "bold"),
            bg="#16a34a",
            fg="white",
            width=22,
            pady=8,
            command=self.save_current,
        ).grid(row=0, column=0, padx=10)

        tk.Button(
            ctrl_frame,
            text="[Del] Discard Bad Crop",
            font=("Segoe UI", 12, "bold"),
            bg="#dc2626",
            fg="white",
            width=20,
            pady=8,
            command=self.discard_current,
        ).grid(row=0, column=1, padx=10)

        tk.Button(
            ctrl_frame,
            text="[Ctrl+Z] Undo Previous",
            font=("Segoe UI", 12),
            bg="#3f3f46",
            fg="white",
            width=18,
            pady=8,
            command=self.undo_last,
        ).grid(row=0, column=2, padx=10)

        hint_lbl = tk.Label(
            self.root,
            text="Shortcuts:  F1 = Qwen3-VL   |   F2 = EasyOCR   |   Enter = Save   |   Del = Discard   |   Ctrl+Z = Undo",
            font=("Segoe UI", 11),
            fg="#a1a1aa",
            bg="#18181b",
        )
        hint_lbl.pack(side="bottom", pady=10)

    def _bind_shortcuts(self):
        self.root.bind("<F1>", lambda e: self.pick_vlm())
        self.root.bind("<F2>", lambda e: self.pick_easy())
        self.root.bind("<Control-Return>", lambda e: self.save_current())
        self.root.bind("<Return>", lambda e: self.save_current())
        self.root.bind("<Delete>", lambda e: self.discard_current())
        self.root.bind("<Control-z>", lambda e: self.undo_last())

    def _render_diff_boxes(self, vlm_str: str, easy_str: str):
        for widget in (self.vlm_text_widget, self.easy_text_widget):
            widget.config(state="normal")
            widget.delete("1.0", tk.END)

        matcher = SequenceMatcher(None, vlm_str, easy_str)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                self.vlm_text_widget.insert(tk.END, vlm_str[i1:i2])
                self.easy_text_widget.insert(tk.END, easy_str[j1:j2])
            else:
                if i1 < i2:
                    self.vlm_text_widget.insert(tk.END, vlm_str[i1:i2], "diff")
                if j1 < j2:
                    self.easy_text_widget.insert(tk.END, easy_str[j1:j2], "diff")

        for widget in (self.vlm_text_widget, self.easy_text_widget):
            widget.config(state="disabled")

    def show_record(self):
        if self.index >= len(self.records):
            self.status_lbl.config(
                text=(
                    f"Audit Complete! Auto-approved: {self.auto_approved_count} | "
                    f"Previously done: {self.previously_done_count}"
                )
            )
            self.img_panel.config(image="")
            self.entry_var.set("")
            messagebox.showinfo(
                "Audit Complete",
                f"All disagreements resolved!\n"
                f"Auto-approved agreements (VLM): {self.auto_approved_count}\n"
                f"Saved to: {VERIFIED_CSV.resolve()}",
            )
            return

        rec = self.records[self.index]
        self.status_lbl.config(
            text=(
                f"Disagreement {self.index + 1} / {len(self.records)}   |   "
                f"Auto-Approved: {self.auto_approved_count}   |   "
                f"Resumed Skip: {self.previously_done_count}   |   "
                f"Role: {rec['role']}   |   File: {rec['filename']}"
            )
        )

        img_path = CROPS_DIR / rec["filename"]
        if img_path.exists():
            pil_img = Image.open(img_path)
            scale = min(1180 / max(1, pil_img.width), 170 / max(1, pil_img.height), 3.5)
            new_size = (max(1, int(pil_img.width * scale)), max(1, int(pil_img.height * scale)))
            pil_img = pil_img.resize(new_size, Image.Resampling.LANCZOS)
            self.tk_img = ImageTk.PhotoImage(pil_img)
            self.img_panel.config(image=self.tk_img)
        else:
            self.img_panel.config(image="", text="[Missing Image File]", fg="red", font=("Segoe UI", 14, "bold"))

        vlm_pred = strip_all_whitespace(rec.get("vlm_pred", ""))
        easy_pred = strip_all_whitespace(rec.get("easyocr_pred", ""))
        self._render_diff_boxes(vlm_pred, easy_pred)

        self.entry_var.set(vlm_pred or easy_pred)
        self.text_entry.focus_set()
        self.text_entry.icursor(tk.END)

    def pick_vlm(self):
        if self.index < len(self.records):
            self.entry_var.set(strip_all_whitespace(self.records[self.index].get("vlm_pred", "")))
            self.text_entry.focus_set()

    def pick_easy(self):
        if self.index < len(self.records):
            self.entry_var.set(strip_all_whitespace(self.records[self.index].get("easyocr_pred", "")))
            self.text_entry.focus_set()

    def save_current(self):
        if self.index >= len(self.records):
            return

        rec = self.records[self.index]
        final_label = strip_all_whitespace(self.entry_var.get())
        if not final_label:
            messagebox.showwarning("Empty Label", "Label is empty. Enter text or press [Del] to discard this crop.")
            return

        append_verified_rows([{
            "filename": rec["filename"],
            "source_img": rec["source_img"],
            "role": rec["role"],
            "label": final_label,
        }])

        self.history.append((self.index, "save", rec))
        self.index += 1
        self.show_record()

    def discard_current(self):
        if self.index >= len(self.records):
            return

        rec = self.records[self.index]
        with open(DISCARDED_LOG, "a", encoding="utf-8") as f:
            f.write(rec["filename"] + "\n")

        self.history.append((self.index, "discard", rec))
        self.index += 1
        self.show_record()

    def undo_last(self):
        if not self.history:
            return

        prev_idx, action, rec = self.history.pop()
        if action == "save" and VERIFIED_CSV.exists():
            with open(VERIFIED_CSV, encoding="utf-8") as f:
                rows = [r for r in csv.DictReader(f) if r["filename"] != rec["filename"]]
            with open(VERIFIED_CSV, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=VERIFIED_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
        elif action == "discard" and DISCARDED_LOG.exists():
            lines = [ln for ln in DISCARDED_LOG.read_text(encoding="utf-8").splitlines() if ln != rec["filename"]]
            DISCARDED_LOG.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        self.index = prev_idx
        self.show_record()


if __name__ == "__main__":
    app_root = tk.Tk()
    DualOCRAuditApp(app_root)
    app_root.mainloop()