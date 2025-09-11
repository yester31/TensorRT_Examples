# Pruning

## How to Run

1. train base model with imagenet100 dataset
    - [Train Base Model (resnet18)](tmo/base_model/README.md)

2. Pruning and export pruned onnx
    ```
    python onnx_export_qat.py
    ```
    - fine tuning

3. generate tensorrt model
    ```
    python onnx2trt.py
    ```
- fp16 pruned flops 80% (fine tuning)   
    Gpu Mem: 130M   
    [TRT_E] Test Top-1 Accuracy: 82.76%   
    [TRT_E] Test Top-5 Accuracy: 96.42%   
    [TRT_E] 10000 iterations time: 6.3565 [sec]   
    [TRT_E] Average FPS: 1573.20 [fps]   
    [TRT_E] Average inference time: 0.64 [msec]   
## Reference

- [TensorRT-Model-Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer)
