import csv
from difflib import SequenceMatcher
from pathlib import Path
import cv2
import easyocr

# --- Paths & Config ---
CROPS_DIR = Path("../data/line_crops")
MANIFEST_CSV = Path("../data/crops_manifest.csv")
CLEAN_MANIFEST_CSV = Path("../data/clean_manifest.csv")
RAW_LABELS_CSV = Path("../data/raw_labels.csv")

SIMILARITY_THRESHOLD = 0.90  # 90% character match filters minor OCR/VLM punctuation jitter

CLEAN_FIELDS = ["filename", "source_img", "role", "easyocr_pred", "easyocr_conf"]
RAW_LABEL_FIELDS = [
    "filename",
    "source_img",
    "role",
    "vlm_pred",
    "easyocr_pred",
    "easyocr_conf",
    "auto_agreed",
]


def recognize_crop(crop_bgr, reader: easyocr.Reader) -> tuple[str, float]:
    """Runs EasyOCR recognition on a single cropped line."""
    results = reader.readtext(crop_bgr, detail=1, paragraph=False)
    if not results:
        return "", 0.0
    results.sort(key=lambda r: min(pt[0] for pt in r[0]))
    text = "".join(r[1].strip() for r in results)
    avg_conf = round(sum(float(r[2]) for r in results) / len(results), 3)
    return text, avg_conf


def is_duplicate_text(candidate: str, kept_texts: list[str]) -> bool:
    """Checks for exact matches, prefix truncations, or high fuzzy similarity."""
    for existing in kept_texts:
        # 1. Exact match (catches repeated character names & static UI lines)
        if candidate == existing:
            return True

        # 2. Prefix match (catches mid-typewriter screenshot captures)
        if len(candidate) >= 4 and existing.startswith(candidate):
            return True

        # 3. Fuzzy match only on longer lines (avoids dropping distinct short 2-3 char names)
        if len(candidate) >= 6 and len(existing) >= 6:
            ratio = SequenceMatcher(None, candidate, existing).ratio()
            if ratio >= SIMILARITY_THRESHOLD:
                return True

    return False


def deduplicate_from_raw_labels():
    """Post-VLM mode: Deduplicates using existing Qwen3-VL + EasyOCR predictions in raw_labels.csv."""
    with open(RAW_LABELS_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"Found {RAW_LABELS_CSV.name} — deduplicating {len(rows)} labeled crops using VLM/EasyOCR text...")

    valid_items = []
    empty_count = 0

    for row in rows:
        crop_path = CROPS_DIR / row["filename"]
        if not crop_path.exists():
            empty_count += 1
            continue

        # Prefer Qwen3-VL transcription as canonical text, fallback to EasyOCR
        vlm_text = row.get("vlm_pred", "").strip()
        easy_text = row.get("easyocr_pred", "").strip()
        canonical_text = vlm_text or easy_text

        if not canonical_text:
            crop_path.unlink(missing_ok=True)
            empty_count += 1
            continue

        row["_canonical"] = canonical_text
        valid_items.append(row)

    # Prioritize: 1) Longest text (beats typewriter fragments), 2) Qwen+EasyOCR agreement, 3) EasyOCR confidence
    valid_items.sort(
        key=lambda r: (
            len(r["_canonical"]),
            r.get("auto_agreed") == "True",
            float(r.get("easyocr_conf") or 0.0),
        ),
        reverse=True,
    )

    kept_texts: list[str] = []
    clean_rows: list[dict] = []
    dup_count = 0

    for item in valid_items:
        if is_duplicate_text(item["_canonical"], kept_texts):
            (CROPS_DIR / item["filename"]).unlink(missing_ok=True)
            dup_count += 1
        else:
            kept_texts.append(item["_canonical"])
            clean_rows.append({k: item.get(k, "") for k in RAW_LABEL_FIELDS})

    # Restore original filename order
    clean_rows.sort(key=lambda r: r["filename"])

    with open(RAW_LABELS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_LABEL_FIELDS)
        writer.writeheader()
        writer.writerows(clean_rows)

    print("\nPost-VLM deduplication complete!")
    print(f"  Original labeled rows : {len(rows)}")
    print(f"  Empty/missing removed : {empty_count}")
    print(f"  Duplicate lines cut   : {dup_count}")
    print(f"  Unique rows kept      : {len(clean_rows)} -> updated {RAW_LABELS_CSV}")


def deduplicate_from_manifest():
    """Pre-VLM mode: Deduplicates crops_manifest.csv via EasyOCR before running Qwen3-VL."""
    if not MANIFEST_CSV.exists():
        raise FileNotFoundError(
            f"Neither {RAW_LABELS_CSV.resolve()} nor {MANIFEST_CSV.resolve()} found. Run 1_crop_lines.py first."
        )

    with open(MANIFEST_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"Running EasyOCR text deduplication on {len(rows)} crops...")
    reader = easyocr.Reader(["ja"], gpu=True)

    recognized_items = []
    empty_count = 0

    for idx, row in enumerate(rows, 1):
        crop_path = CROPS_DIR / row["filename"]
        if not crop_path.exists():
            continue

        img = cv2.imread(str(crop_path))
        if img is None:
            crop_path.unlink(missing_ok=True)
            empty_count += 1
            continue

        text, conf = recognize_crop(img, reader)
        if not text:
            crop_path.unlink(missing_ok=True)
            empty_count += 1
            continue

        recognized_items.append({
            "filename": row["filename"],
            "source_img": row["source_img"],
            "role": row["role"],
            "easyocr_pred": text,
            "easyocr_conf": conf,
        })

        if idx % 100 == 0 or idx == len(rows):
            print(f"  Scanned {idx}/{len(rows)} crops...")

    recognized_items.sort(key=lambda item: len(item["easyocr_pred"]), reverse=True)

    kept_texts: list[str] = []
    clean_rows: list[dict] = []
    dup_count = 0

    for item in recognized_items:
        if is_duplicate_text(item["easyocr_pred"], kept_texts):
            (CROPS_DIR / item["filename"]).unlink(missing_ok=True)
            dup_count += 1
        else:
            kept_texts.append(item["easyocr_pred"])
            clean_rows.append(item)

    clean_rows.sort(key=lambda item: item["filename"])

    with open(CLEAN_MANIFEST_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CLEAN_FIELDS)
        writer.writeheader()
        writer.writerows(clean_rows)

    print("\nPre-VLM deduplication complete!")
    print(f"  Original crops     : {len(rows)}")
    print(f"  Empty/unreadable   : {empty_count}")
    print(f"  Duplicate lines    : {dup_count}")
    print(f"  Unique crops kept  : {len(clean_rows)} -> saved to {CLEAN_MANIFEST_CSV}")


def main():
    if RAW_LABELS_CSV.exists():
        deduplicate_from_raw_labels()
    else:
        deduplicate_from_manifest()


if __name__ == "__main__":
    main()