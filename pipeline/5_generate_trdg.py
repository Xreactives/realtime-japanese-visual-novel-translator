import csv
import json
from pathlib import Path
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# --- Paths & Config ---
VERIFIED_CSV = Path("../data/verified_labels.csv")
CUSTOM_FONTS_DIR = Path("../fonts")
VOCAB_PATH = Path("../model/vocab.json")
SYNTH_DIR = Path("../data/synthetic_trdg")
SYNTH_IMG_DIR = SYNTH_DIR / "images"
SYNTH_CSV = SYNTH_DIR / "labels.csv"

NUM_SYNTH_SAMPLES = 20000
RANDOM_SEED = 42

VOCAB_PATH.parent.mkdir(parents=True, exist_ok=True)
SYNTH_IMG_DIR.mkdir(parents=True, exist_ok=True)

HIRAGANA = (
    "ぁあぃいぅうぇえぉおかがきぎくぐけげこごさざしじすずせぜそぞ"
    "ただちぢっつづてでとどなにぬねのはばぱひびぴふぶぷへべぺほぼぽ"
    "まみむめもゃやゅゆょよらりるれろゎわゐゑをんゔ"
)
KATAKANA = (
    "ァアィイゥウェエォオカガキギクグケゲコゴサザシジスズセゼソゾ"
    "タダチヂッツヅテデトドナニヌネノハバパヒビピフブプヘベペホボポ"
    "マミムメモャヤュユョヨラリルレロヮワヰヱヲンヴヵヶー"
)
VN_SYMBOLS = "「」『』【】（）！？…・、。〜～－―：；0123456789０１２３４５６７８９"


def get_all_japanese_fonts() -> list[str]:
    """Collects both downloaded fonts in ../fonts/ and built-in Windows Japanese fonts."""
    font_paths: list[str] = []

    # 1. Downloaded custom fonts in ../fonts/ (recursive in case of extracted subfolders)
    if CUSTOM_FONTS_DIR.exists():
        valid_exts = {".ttf", ".otf", ".ttc"}
        custom_fonts = sorted(
            str(p.resolve())
            for p in CUSTOM_FONTS_DIR.rglob("*")
            if p.is_file() and p.suffix.lower() in valid_exts
        )
        font_paths.extend(custom_fonts)
        print(f"Found {len(custom_fonts)} custom font(s) in {CUSTOM_FONTS_DIR.resolve()}")

    # 2. Windows built-in Japanese fonts
    win_dir = Path("C:/Windows/Fonts")
    win_candidates = [
        "meiryo.ttc",
        "meiryob.ttc",
        "YuGothR.ttc",
        "YuGothM.ttc",
        "YuGothB.ttc",
        "yumin.ttf",
        "yumindb.ttf",
        "msgothic.ttc",
        "msmincho.ttc",
        "UDDigiKyokashoN-R.ttc",
        "UDDigiKyokashoN-B.ttc",
    ]
    win_fonts = [str(win_dir / f) for f in win_candidates if (win_dir / f).exists()]
    font_paths.extend(win_fonts)
    print(f"Found {len(win_fonts)} Windows system Japanese font(s) in {win_dir}")

    if not font_paths:
        raise FileNotFoundError("No Japanese fonts found in ../fonts or C:/Windows/Fonts.")

    return font_paths


def load_font_variants(font_paths: list[str], sizes: tuple[int, ...] = (28, 32, 36)) -> list[ImageFont.FreeTypeFont]:
    """Loads FreeType font objects across sizes, skipping any corrupted or unsupported font files."""
    font_objs: list[ImageFont.FreeTypeFont] = []
    for fp in font_paths:
        for sz in sizes:
            try:
                font_objs.append(ImageFont.truetype(fp, size=sz, index=0))
            except OSError as e:
                print(f"Skipping unreadable font {Path(fp).name}: {e}")
                break
    if not font_objs:
        raise RuntimeError("Failed to load any valid FreeType fonts.")
    return font_objs


def build_vocab(verified_labels: list[str]) -> dict:
    unique_chars = set(HIRAGANA + KATAKANA + VN_SYMBOLS)
    for text in verified_labels:
        unique_chars.update(ch for ch in text if not ch.isspace())

    sorted_chars = sorted(unique_chars)
    char_to_idx = {"<blank>": 0, "<unk>": 1}
    for idx, ch in enumerate(sorted_chars, start=2):
        char_to_idx[ch] = idx

    idx_to_char = {str(v): k for k, v in char_to_idx.items()}
    vocab_payload = {
        "num_classes": len(char_to_idx),
        "char_to_idx": char_to_idx,
        "idx_to_char": idx_to_char,
    }

    with open(VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump(vocab_payload, f, ensure_ascii=False, indent=2)

    print(f"Saved vocab.json with {len(char_to_idx)} classes -> {VOCAB_PATH}")
    return char_to_idx


def synthesize_training_strings(verified_labels: list[str], char_to_idx: dict, count: int) -> list[str]:
    rng = random.Random(RANDOM_SEED)
    valid_chars = [ch for ch in char_to_idx if ch not in {"<blank>", "<unk>"}]
    clean_corpus = ["".join(ch for ch in line if ch in char_to_idx) for line in verified_labels]
    clean_corpus = [line for line in clean_corpus if len(line) >= 2]

    strings = []
    for i in range(count):
        if i % 2 == 0 and clean_corpus:
            base = rng.choice(clean_corpus)
            if len(base) > 4:
                start = rng.randint(0, len(base) - 3)
                end = rng.randint(start + 2, len(base))
                snippet = base[start:end]
            else:
                snippet = base

            if rng.random() < 0.35:
                snippet = f"「{snippet}」"
            strings.append(snippet[:28])
        else:
            length = rng.randint(4, 22)
            strings.append("".join(rng.choices(valid_chars, k=length)))

    return strings


def create_vn_background(width: int, height: int, rng: random.Random) -> Image.Image:
    """Generates a realistic visual novel textbox background (gradient + subtle noise)."""
    palettes = [
        ((15, 20, 35), (35, 45, 70)),
        ((18, 18, 22), (42, 42, 50)),
        ((28, 16, 32), (55, 32, 62)),
        ((12, 26, 28), (28, 52, 56)),
        ((235, 235, 240), (210, 215, 225)),
    ]
    c1, c2 = rng.choice(palettes)
    grad = np.linspace(0, 1, height, dtype=np.float32)[:, None, None]
    bg = np.array(c1, dtype=np.float32) * (1 - grad) + np.array(c2, dtype=np.float32) * grad
    bg = np.broadcast_to(bg, (height, width, 3)).copy()

    noise = rng.randint(2, 8)
    bg += np.random.default_rng(rng.randint(0, 999999)).normal(0, noise, bg.shape)
    bg = np.clip(bg, 0, 255).astype(np.uint8)
    return Image.fromarray(bg, mode="RGB")


def render_vn_line(text: str, font_objs: list[ImageFont.FreeTypeFont], rng: random.Random) -> Image.Image:
    font = rng.choice(font_objs)
    pad_x = rng.randint(6, 14)
    pad_y = rng.randint(4, 10)
    stroke_w = rng.choice([1, 2, 2])

    dummy = Image.new("RGB", (10, 10))
    draw_dummy = ImageDraw.Draw(dummy)
    bbox = draw_dummy.textbbox((0, 0), text, font=font, stroke_width=stroke_w)
    text_w = max(12, bbox[2] - bbox[0])
    text_h = max(16, bbox[3] - bbox[1])

    img_w = text_w + pad_x * 2
    img_h = text_h + pad_y * 2

    img = create_vn_background(img_w, img_h, rng)
    draw = ImageDraw.Draw(img)

    avg_luma = np.mean(np.array(img))
    if avg_luma < 140:
        fill_color = rng.choice([(255, 255, 255), (245, 247, 255), (255, 245, 190), (210, 240, 255)])
        stroke_color = (10, 10, 15)
    else:
        fill_color = rng.choice([(20, 20, 25), (30, 20, 50)])
        stroke_color = (250, 250, 252)

    draw_x = pad_x - bbox[0]
    draw_y = pad_y - bbox[1]

    if rng.random() < 0.6:
        draw.text((draw_x + 2, draw_y + 2), text, font=font, fill=(0, 0, 0))

    draw.text(
        (draw_x, draw_y),
        text,
        font=font,
        fill=fill_color,
        stroke_width=stroke_w,
        stroke_fill=stroke_color,
    )

    if rng.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 0.7)))

    return img


def main():
    if not VERIFIED_CSV.exists():
        raise FileNotFoundError(f"verified_labels.csv not found at {VERIFIED_CSV.resolve()}.")

    with open(VERIFIED_CSV, encoding="utf-8") as f:
        verified_rows = list(csv.DictReader(f))

    verified_labels = [r["label"].strip() for r in verified_rows if r.get("label", "").strip()]
    if not verified_labels:
        raise ValueError("verified_labels.csv is empty.")

    print(f"Loaded {len(verified_labels)} verified ground-truth labels.")
    char_to_idx = build_vocab(verified_labels)

    font_paths = get_all_japanese_fonts()
    font_objs = load_font_variants(font_paths, sizes=(28, 32, 36))
    print(f"Loaded {len(font_objs)} total font/size variants.")

    synth_strings = synthesize_training_strings(verified_labels, char_to_idx, NUM_SYNTH_SAMPLES)
    rng = random.Random(RANDOM_SEED)

    print(f"Generating {NUM_SYNTH_SAMPLES} synthetic Japanese VN images...")
    with open(SYNTH_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "label"])
        writer.writeheader()

        for idx, text in enumerate(synth_strings):
            img = render_vn_line(text, font_objs, rng)
            fname = f"synth_{idx:06d}.png"
            img.save(SYNTH_IMG_DIR / fname)
            writer.writerow({"filename": fname, "label": text})

            if (idx + 1) % 2500 == 0 or (idx + 1) == NUM_SYNTH_SAMPLES:
                print(f"  Generated {idx + 1} / {NUM_SYNTH_SAMPLES} images...")

    print(f"\nStep 5 complete! Synthetic dataset ready at {SYNTH_DIR.resolve()}")


if __name__ == "__main__":
    main()