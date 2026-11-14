from ultralytics import YOLO

model = YOLO("../weights/vn_detector.pt")
model.export(format="onnx", imgsz=1280, dynamic=False, simplify=True)