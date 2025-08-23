# NVIDIA 2:4 Sparsity

## How to Run

1. train base model with imagenet100 dataset
- [Train Base Model (resnet18)](tmo/base_model/README.md)

2. Sparsify and export sparsity onnx
    ```
    python onnx_export_sparsity.py
    ```
- fine tuning 
- sparsity_mode: "sparsegpt" or "sparse_magnitude"

3. generate tensorrt model
    ```
    python onnx2trt.py
    ```
- fp16 sparse_magnitude
- Gpu Mem: 138M
- [TRT_E] Test Top-1 Accuracy: 83.28%
- [TRT_E] Test Top-5 Accuracy: 96.72%
- [TRT_E] Inference FPS: 681.17 samples/sec

## Reference

- [TensorRT-Model-Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer)
