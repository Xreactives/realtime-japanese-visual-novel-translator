import csv
from pathlib import Path
import cv2
import easyocr
from ultralytics import YOLO

# --- Paths & Config ---
RAW_DIR = Path("../data/raw_screenshots")
CROPS_DIR = Path("../data/line_crops")
MANIFEST_CSV = Path("../data/crops_manifest.csv")
YOLO_WEIGHTS = Path("../weights/vn_detector.pt")

CROPS_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_CSV.parent.mkdir(parents=True, exist_ok=True)

MANIFEST_FIELDS = ["filename", "source_img", "role"]


def merge_horizontal_boxes(boxes: list[list[int]], y_overlap_ratio: float = 0.5) -> list[list[int]]:
    """Merges text detection boxes on the same horizontal row so single lines stay intact."""
    if not boxes:
        return []

    # boxes format from easyocr.detect: [[x_min, x_max, y_min, y_max], ...]
    items = [{"x1": int(b[0]), "x2": int(b[1]), "y1": int(b[2]), "y2": int(b[3])} for b in boxes]
    items.sort(key=lambda d: d["y1"])

    lines: list[list[dict]] = []
    for item in items:
        placed = False
        for line in lines:
            ref = line[0]
            overlap = min(ref["y2"], item["y2"]) - max(ref["y1"], item["y1"])
            min_h = max(1, min(ref["y2"] - ref["y1"], item["y2"] - item["y1"]))
            if (overlap / min_h) >= y_overlap_ratio:
                line.append(item)
                placed = True
                break
        if not placed:
            lines.append([item])

    merged = []
    for line in lines:
        x1 = min(d["x1"] for d in line)
        y1 = min(d["y1"] for d in line)
        x2 = max(d["x2"] for d in line)
        y2 = max(d["y2"] for d in line)
        merged.append([x1, y1, x2, y2])

    return merged


def main():
    if not YOLO_WEIGHTS.exists():
        raise FileNotFoundError(f"YOLO weights not found at {YOLO_WEIGHTS.resolve()}")

    detector = YOLO(str(YOLO_WEIGHTS))
    # Initialize EasyOCR for fast line-box detection
    easy_reader = easyocr.Reader(["ja"], gpu=True)

    image_files = sorted(
        p for p in RAW_DIR.glob("*.*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    )
    if not image_files:
        print(f"No screenshots found in {RAW_DIR.resolve()}")
        return

    print(f"Cropping lines from {len(image_files)} screenshots...")
    crop_idx = 0

    with open(MANIFEST_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()

        for img_num, img_path in enumerate(image_files, 1):
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            h_img, w_img = img.shape[:2]
            res = detector.predict(img, conf=0.5, verbose=False)[0]
            sorted_boxes = sorted(res.boxes, key=lambda b: float(b.xyxy[0][1]))

            for box in sorted_boxes:
                role = detector.names[int(box.cls[0])]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_img, x2), min(h_img, y2)

                ui_crop = img[y1:y2, x1:x2]
                if ui_crop.size == 0:
                    continue

                # Detect line boxes only (horizontal_list[0])
                horizontal_list, _ = easy_reader.detect(ui_crop)
                raw_boxes = horizontal_list[0] if horizontal_list else []
                merged_lines = merge_horizontal_boxes(raw_boxes)

                for lx1, ly1, lx2, ly2 in merged_lines:
                    pad_x, pad_y = 6, 4
                    cx1 = max(0, lx1 - pad_x)
                    cy1 = max(0, ly1 - pad_y)
                    cx2 = min(ui_crop.shape[1], lx2 + pad_x)
                    cy2 = min(ui_crop.shape[0], ly2 + pad_y)

                    line_crop = ui_crop[cy1:cy2, cx1:cx2]
                    if line_crop.shape[0] < 10 or line_crop.shape[1] < 10:
                        continue

                    crop_name = f"vn_line_{crop_idx:06d}.png"
                    cv2.imwrite(str(CROPS_DIR / crop_name), line_crop)

                    writer.writerow({
                        "filename": crop_name,
                        "source_img": img_path.name,
                        "role": role,
                    })
                    crop_idx += 1

            if img_num % 25 == 0 or img_num == len(image_files):
                print(f"[{img_num}/{len(image_files)}] Total line crops saved: {crop_idx}")

    print(f"\nStep 1 complete! Saved {crop_idx} crops to {CROPS_DIR} and manifest to {MANIFEST_CSV}")


if __name__ == "__main__":
    main()