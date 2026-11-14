from pathlib import Path
import numpy as np
import onnxruntime as ort
import torch

from model.crnn import CRNN

WEIGHTS_PATH = Path("weights/crnn_vn_best.pt")
ONNX_PATH = Path("weights/crnn_vn.onnx")


def main():
    device = torch.device("cpu")
    ckpt = torch.load(WEIGHTS_PATH, map_location=device, weights_only=True)

    input_height = ckpt["input_height"]
    num_classes = ckpt["num_classes"]

    model = CRNN(input_height=input_height, num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dummy_input = torch.randn(1, 3, input_height, 256, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        str(ONNX_PATH),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch_size", 3: "width"},
            "logits": {0: "batch_size", 1: "seq_len"},
        },
    )
    print(f"Exported ONNX model -> {ONNX_PATH.resolve()}")

    ort_session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])

    for test_w in [128, 320, 640]:
        test_x = torch.randn(2, 3, input_height, test_w, dtype=torch.float32)
        with torch.inference_mode():
            pt_out = model(test_x).numpy()

        ort_out = ort_session.run(["logits"], {"input": test_x.numpy()})[0]
        max_diff = float(np.max(np.abs(pt_out - ort_out)))
        print(f"Width {test_w:3d}px -> Output shape {ort_out.shape} | Max diff vs PyTorch: {max_diff:.6f}")


if __name__ == "__main__":
    main()