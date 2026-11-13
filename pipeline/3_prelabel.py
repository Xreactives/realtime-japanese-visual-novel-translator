import csv
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

# --- Paths & Config ---
CROPS_DIR = Path("../data/line_crops")
CLEAN_MANIFEST_CSV = Path("../data/clean_manifest.csv")
OUT_CSV = Path("../data/raw_labels.csv")
QWEN_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"

CSV_FIELDS = [
    "filename",
    "source_img",
    "role",
    "vlm_pred",
    "easyocr_pred",
    "easyocr_conf",
    "auto_agreed",
]


def load_qwen3_vl(model_id: str = QWEN_MODEL_ID):
    print(f"Loading {model_id} in 4-bit mode...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def query_qwen3_vl(crop_bgr: np.ndarray, model, processor) -> str:
    pil_img = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_img},
                {
                    "type": "text",
                    "text": (
                        "Transcribe the exact Japanese text in this single-line visual novel crop. "
                        "Keep every kanji, hiragana, katakana, and punctuation mark (「」『』！？…・ー、。) exact. "
                        "Do NOT translate, explain, or add quotes. Output ONLY the Japanese text."
                    ),
                },
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=96, do_sample=False)

    trimmed_ids = generated_ids[0][len(inputs.input_ids[0]) :]
    raw_text = processor.decode(trimmed_ids, skip_special_tokens=True)
    return "".join(raw_text.splitlines()).strip()


def load_done_filenames() -> set[str]:
    if not OUT_CSV.exists():
        return set()
    with open(OUT_CSV, encoding="utf-8") as f:
        return {row["filename"] for row in csv.DictReader(f)}


def main():
    if not CLEAN_MANIFEST_CSV.exists():
        raise FileNotFoundError(f"Clean manifest not found at {CLEAN_MANIFEST_CSV.resolve()}. Run 2_deduplicate.py first.")

    done_files = load_done_filenames()
    with open(CLEAN_MANIFEST_CSV, encoding="utf-8") as f:
        pending_rows = [r for r in csv.DictReader(f) if r["filename"] not in done_files]

    if not pending_rows:
        print("All unique crops have already been labeled!")
        return

    print(f"Labeling {len(pending_rows)} unique crops with Qwen3-VL-4B...")
    qwen_model, qwen_processor = load_qwen3_vl()

    file_exists = OUT_CSV.exists()
    with open(OUT_CSV, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()

        agreed_count = 0
        for idx, row in enumerate(pending_rows, 1):
            crop_path = CROPS_DIR / row["filename"]
            crop_bgr = cv2.imread(str(crop_path))
            if crop_bgr is None:
                continue

            easy_text = row["easyocr_pred"].strip()
            qwen_text = query_qwen3_vl(crop_bgr, qwen_model, qwen_processor)
            is_match = (qwen_text == easy_text) and bool(qwen_text)

            writer.writerow({
                "filename": row["filename"],
                "source_img": row["source_img"],
                "role": row["role"],
                "vlm_pred": qwen_text,
                "easyocr_pred": easy_text,
                "easyocr_conf": row["easyocr_conf"],
                "auto_agreed": is_match,
            })

            if is_match:
                agreed_count += 1

            if idx % 20 == 0 or idx == len(pending_rows):
                csv_file.flush()
                print(f"[{idx}/{len(pending_rows)}] Labeled | Exact Qwen3-VL & EasyOCR matches: {agreed_count}")

    print(f"\nStep 3 complete! Labeled dataset saved to {OUT_CSV}")


if __name__ == "__main__":
    main()