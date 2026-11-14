from pathlib import Path
import numpy as np
from ultralytics import YOLO


class ONNXVNDetector:
    def __init__(
        self,
        onnx_path: str | Path = "weights/vn_detector.onnx",
        imgsz: int = 1280,
        conf_threshold: float = 0.30,
        iou_threshold: float = 0.45,
        target_classes: set[str] | None = None,
        pad_x: int = 8,
        pad_y: int = 4,
    ):
        self.model = YOLO(str(onnx_path), task="detect")
        self.imgsz = imgsz
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.target_classes = target_classes
        self.pad_x = pad_x
        self.pad_y = pad_y

    def detect(self, frame_bgr: np.ndarray) -> list[tuple[str, np.ndarray]]:
        results = self.model.predict(
            frame_bgr,
            imgsz=self.imgsz,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device="cpu",
            verbose=False,
        )[0]

        if results.boxes is None or len(results.boxes) == 0:
            return []

        names = results.names
        h, w = frame_bgr.shape[:2]
        detections: list[tuple[float, float, str, np.ndarray]] = []

        for box in results.boxes:
            cls_name = str(names[int(box.cls[0])]).strip().lower()
            if self.target_classes and cls_name not in self.target_classes:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            px1 = max(0, int(round(x1)) - self.pad_x)
            py1 = max(0, int(round(y1)) - self.pad_y)
            px2 = min(w, int(round(x2)) + self.pad_x)
            py2 = min(h, int(round(y2)) + self.pad_y)

            if (px2 - px1) >= 12 and (py2 - py1) >= 10:
                crop = frame_bgr[py1:py2, px1:px2]
                detections.append((y1, x1, cls_name, crop))

        detections.sort(key=lambda item: (round(item[0] / 12), item[1]))
        return [(cls_name, crop) for _, _, cls_name, crop in detections]