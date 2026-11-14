import ctypes
import sys

if sys.platform == "win32":
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        ctypes.windll.user32.SetProcessDPIAware()

from runtime.device_utils import register_optional_cuda_dlls

register_optional_cuda_dlls()

import queue
import threading
import time
import tkinter as tk
import cv2
import mss
import numpy as np
import re
import argparse

from runtime.detector_onnx import ONNXVNDetector
from runtime.ocr_onnx import ONNXVNRecognizer
from runtime.postprocess import VNTextPostProcessor
from runtime.translator_sugoi import SugoiV4Translator

WDA_EXCLUDEFROMCAPTURE = 0x00000011


class SubtitleOverlay:
    def __init__(self, ui_queue: queue.Queue, shared_rect: dict[str, int], allow_capture: bool = False):
        self.ui_queue = ui_queue
        self.shared_rect = shared_rect
        self.allow_capture = allow_capture

        self.root = tk.Tk()
        self.root.title("VN Real-Time Translator")
        self.root.geometry("1100x110+410+60")
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", 0.88)
        self.root.configure(bg="#121216")

        self.root.update_idletasks()
        if not self.allow_capture:
            self._exclude_from_capture()
        self._sync_rect()

        self._drag_x = 0
        self._drag_y = 0
        self.root.bind("<ButtonPress-1>", self._start_move)
        self.root.bind("<B1-Motion>", self._on_move)
        self.root.bind("<Escape>", lambda _: self.root.destroy())

        self.jp_var = tk.StringVar(value="Loading Detector + OCR + Sugoi V4...")
        self.en_var = tk.StringVar(value="Drag this bar to reposition | Press ESC to quit")

        tk.Label(
            self.root,
            textvariable=self.jp_var,
            font=("Yu Gothic UI", 12),
            fg="#9cdcfe",
            bg="#121216",
            wraplength=1060,
            justify="left",
            anchor="w",
        ).pack(fill="x", padx=18, pady=(12, 4))

        tk.Label(
            self.root,
            textvariable=self.en_var,
            font=("Segoe UI", 14, "bold"),
            fg="#ffffff",
            bg="#121216",
            wraplength=1060,
            justify="left",
            anchor="w",
        ).pack(fill="x", padx=18, pady=(2, 12))

        self.root.after(40, self._poll_queue)

    def _exclude_from_capture(self) -> None:
        """Makes the overlay window invisible to mss / Windows screen capture."""
        if sys.platform != "win32":
            return
        try:
            hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id())
            if not hwnd:
                hwnd = self.root.winfo_id()
            ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)
        except Exception:
            pass

    def _sync_rect(self) -> None:
        """Tracks live overlay position so the worker can also black-mask its region."""
        self.shared_rect["x"] = self.root.winfo_x()
        self.shared_rect["y"] = self.root.winfo_y()
        self.shared_rect["w"] = self.root.winfo_width()
        self.shared_rect["h"] = self.root.winfo_height()

    def _start_move(self, event):
        self._drag_x = event.x
        self._drag_y = event.y

    def _on_move(self, event):
        x = self.root.winfo_x() + (event.x - self._drag_x)
        y = self.root.winfo_y() + (event.y - self._drag_y)
        self.root.geometry(f"+{x}+{y}")
        self._sync_rect()

    def _poll_queue(self):
        self._sync_rect()
        try:
            while True:
                jp_text, en_text = self.ui_queue.get_nowait()
                self.jp_var.set(jp_text)
                self.en_var.set(en_text)
                self.root.update_idletasks()
                req_h = max(110, self.root.winfo_reqheight())
                self.root.geometry(f"1100x{req_h}+{self.root.winfo_x()}+{self.root.winfo_y()}")
        except queue.Empty:
            pass
        self.root.after(40, self._poll_queue)

    def run(self):
        self.root.mainloop()


def ocr_translation_worker(ui_queue: queue.Queue, shared_rect: dict[str, int]):
    detector = ONNXVNDetector(
        onnx_path="weights/vn_detector.onnx",
        imgsz=1280,
        conf_threshold=0.30,
        pad_x=8,
        pad_y=4,
    )
    ocr = ONNXVNRecognizer(
        onnx_path="weights/crnn_vn.onnx",
        vocab_path="model/vocab.json",
        use_gpu=False,
    )
    post = VNTextPostProcessor()
    translator = SugoiV4Translator(
        ct2_model_dir="weights/sugoi-v4-ja-en-ctranslate2",
        ja_spm_path="weights/sugoi-v4-ja-en-ctranslate2/spm/spm.ja.nopretok.model",
        en_spm_path="weights/sugoi-v4-ja-en-ctranslate2/spm/spm.en.nopretok.model",
        device="auto",
    )
    ui_queue.put((
        f"Ready ({translator.device.upper()} mode) — full-screen detection active",
        "Advance text in your Visual Novel",
    ))

    prev_roi = None
    last_translated_roi = None
    stable_count = 0
    pending_new_frame = False
    last_jp_text = ""

    with mss.MSS() as sct:
        monitor = sct.monitors[1]
        mon_left, mon_top = monitor["left"], monitor["top"]

        while True:
            t0 = time.perf_counter()
            raw = np.array(sct.grab(monitor))
            frame_bgr = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)

            # Mask out overlay window
            ox = max(0, shared_rect["x"] - mon_left)
            oy = max(0, shared_rect["y"] - mon_top)
            ox2 = min(frame_bgr.shape[1], ox + shared_rect["w"])
            oy2 = min(frame_bgr.shape[0], oy + shared_rect["h"])
            if ox2 > ox and oy2 > oy:
                frame_bgr[oy:oy2, ox:ox2] = 0

            # Focus change detection on lower 75% (covers both bottom dialogue & center choices)
            h, w = frame_bgr.shape[:2]
            roi_gray = cv2.cvtColor(frame_bgr[int(h * 0.25):, :], cv2.COLOR_BGR2GRAY)
            roi_small = cv2.resize(roi_gray, (320, 135))

            if prev_roi is not None:
                step_diff = float(np.mean(np.abs(roi_small.astype(np.float32) - prev_roi.astype(np.float32))))
                if step_diff > 0.45:
                    pending_new_frame = True
                    stable_count = 0
                else:
                    stable_count += 1
            else:
                pending_new_frame = True
                stable_count = 1

            # Also mark dirty if screen changed while we were busy translating the previous line
            if not pending_new_frame and last_translated_roi is not None:
                drift_diff = float(np.mean(np.abs(roi_small.astype(np.float32) - last_translated_roi.astype(np.float32))))
                if drift_diff > 0.75:
                    pending_new_frame = True
                    stable_count = 1

            prev_roi = roi_small

            # Trigger as soon as text settles for 1 poll (~30ms) OR max 4 polls (~120ms) if background animates
            if pending_new_frame and (stable_count >= 1 or stable_count >= 4):
                pending_new_frame = False
                last_translated_roi = roi_small.copy()

                detections = detector.detect(frame_bgr)
                if not detections:
                    continue

                cls_names = [cls_name for cls_name, _ in detections]
                crops = [crop for _, crop in detections]
                raw_lines = ocr.recognize(crops)

                name_raw: list[str] = []
                dialogue_raw: list[str] = []
                choice_raw: list[str] = []

                for cls_name, text in zip(cls_names, raw_lines):
                    if not text.strip():
                        continue
                    if "name" in cls_name or "speaker" in cls_name:
                        name_raw.append(text)
                    elif "choice" in cls_name or "option" in cls_name:
                        choice_raw.append(text)
                    else:
                        dialogue_raw.append(text)

                speaker_jp = ""
                if name_raw:
                    speaker_cleaned = post.process_box(name_raw)
                    speaker_jp = re.sub(r"^[【\[「『]+|[】\]」』]+$", "", speaker_cleaned["text"]).strip()

                dialogue_cleaned = post.process_box(dialogue_raw) if dialogue_raw else None
                if not speaker_jp and dialogue_cleaned and dialogue_cleaned["speaker"]:
                    speaker_jp = dialogue_cleaned["speaker"]

                dialogue_jp = dialogue_cleaned["dialogue"] if dialogue_cleaned else ""
                dialogue_en = translator.translate(dialogue_jp) if dialogue_jp else ""

                choices_jp: list[str] = []
                choices_en: list[str] = []
                for c_text in choice_raw:
                    c_clean = post.process_box([c_text])["dialogue"]
                    if c_clean:
                        choices_jp.append(c_clean)
                        choices_en.append(translator.translate(c_clean))

                jp_blocks: list[str] = []
                en_blocks: list[str] = []

                if dialogue_jp or speaker_jp:
                    if speaker_jp:
                        speaker_en = translator.translate(speaker_jp)
                        jp_blocks.append(f"{speaker_jp}: {dialogue_jp}".strip())
                        en_blocks.append(f"{speaker_en}: {dialogue_en}".strip())
                    else:
                        jp_blocks.append(dialogue_jp)
                        en_blocks.append(dialogue_en)

                if choices_jp:
                    if jp_blocks:
                        jp_blocks.append("")
                        en_blocks.append("")
                    jp_blocks.extend(choices_jp)
                    en_blocks.extend(choices_en)

                jp_display = "\n".join(jp_blocks).strip()
                en_display = "\n".join(en_blocks).strip()

                if jp_display and jp_display != last_jp_text:
                    last_jp_text = jp_display
                    latency_ms = int((time.perf_counter() - t0) * 1000)
                    ui_queue.put((f"{jp_display}  [{latency_ms}ms]", en_display))

            time.sleep(0.03)


def main():
    parser = argparse.ArgumentParser(description="Real-Time VN Translator")
    parser.add_argument(
        "--record-demo",
        action="store_true",
        help="Allow screen recorders (OBS/ScreenToGif) to capture the overlay window",
    )
    args = parser.parse_args()
    ui_queue: queue.Queue = queue.Queue()
    #shared_rect = {"x": 410, "y": 60, "w": 1100, "h": 145}
    shared_rect = {"x": 410, "y": 60, "w": 1100, "h": 110}
    worker = threading.Thread(
        target=ocr_translation_worker,
        args=(ui_queue, shared_rect),
        daemon=True,
    )
    worker.start()

    overlay = SubtitleOverlay(ui_queue, shared_rect, allow_capture=args.record_demo)
    overlay.run()


if __name__ == "__main__":
    main()