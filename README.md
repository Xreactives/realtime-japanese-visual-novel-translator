
```
visual-novel-text-recognizer/
├── assets/
│   └── audit_gui_preview.png           # Screenshot of Qwen3-VL vs EasyOCR audit GUI
├── data/                               # Gitignored local datasets
│   ├── raw_screenshots/                # Source VN screenshots
│   ├── line_crops/                     # Cropped single-line text strips
│   ├── raw_labels.csv                  # Qwen3-VL + EasyOCR dual predictions
│   ├── verified_labels.csv             # Human-audited ground truth
│   └── synthetic_trdg/                 # Phase 1 TRDG generated images + labels
├── weights/
│   ├── yolo_vn_detector.pt             # Trained YOLO26n weights
│   ├── crnn_phase1_synth.pth           # Phase 1: Pretrained on TRDG synthetic Japanese
│   ├── crnn_phase2_final.pth           # Phase 2: Fine-tuned on real VN crops
│   └── sugoi-v4-ja-en-ctranslate2/     # Offline Sugoi V4 CTranslate2 + SentencePiece models
├── pipeline/
│   ├── 1_crop_and_prelabel.py          # YOLO26 UI crop -> line slice -> Qwen3-VL + EasyOCR
│   ├── 2_audit_gui.py                  # Tkinter Audit GUI (Qwen3-VL vs EasyOCR)
│   └── 3_generate_trdg.py              # Builds vocab.json & runs TRDG synthetic generator
├── model/
│   ├── crnn.py                         # Your custom CRNN architecture
│   ├── dataset.py                      # Japanese vocab encoder, CTC collate, augmentations
│   └── vocab.json                      # Unified Japanese character dictionary
├── train.py                            # 2-Phase CRNN training (Phase 1 TRDG -> Phase 2 Real)
├── realtime_translator.py              # Live screen watcher: YOLO26 -> CRNN -> Sugoi V4
├── requirements.txt
└── README.md
```