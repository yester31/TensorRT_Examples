# Convert pytorch to tensorrt (fp16)

## How to Run

1. train base model with imagenet100 dataset
    - [Train Base Model (resnet18)](tmo/base_model/README.md)

2. export onnx model
    ```
    python onnx_export.py
    ```

3. generate tensorrt model
    ```
    python onnx2trt.py
    ```
- fp16
- Gpu Mem: 138M
- [TRT_E] Test Top-1 Accuracy: 84.54%
- [TRT_E] Test Top-5 Accuracy: 97.20%
- [TRT_E] Inference FPS: 476.77 samples/sec