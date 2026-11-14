from pathlib import Path
from huggingface_hub import snapshot_download

WEIGHTS_DIR = Path("weights")
SUGOI_DIR = WEIGHTS_DIR / "sugoi-v4-ja-en-ctranslate2"


def main():
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Sugoi V4 (CTranslate2) to {SUGOI_DIR.resolve()} ...")
    snapshot_download(
        repo_id="entai2965/sugoi-v4-ja-en-ctranslate2",
        local_dir=str(SUGOI_DIR),
    )

    required_files = [
        SUGOI_DIR / "model.bin",
        SUGOI_DIR / "spm" / "spm.ja.nopretok.model",
        SUGOI_DIR / "spm" / "spm.en.nopretok.model",
    ]
    for path in required_files:
        assert path.exists(), f"Missing required translation file: {path}"

    print("Sugoi V4 download complete and verified!")

    # Verify custom vision models (place downloaded release assets in weights/)
    custom_models = [
        WEIGHTS_DIR / "vn_detector.onnx",
        WEIGHTS_DIR / "crnn_vn.onnx",
    ]
    missing = [str(p) for p in custom_models if not p.exists()]
    if missing:
        print(
            "\n[Action Needed] Place your custom ONNX weights in weights/ "
            "(download from GitHub Releases or export via export_recognizer_onnx.py & export_detector_onnx.py):"
        )
        for m in missing:
            print(f"  - {m}")
    else:
        print("All custom ONNX vision weights verified!")


if __name__ == "__main__":
    main()