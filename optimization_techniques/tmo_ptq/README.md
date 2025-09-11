# Post-training quantization (PTQ)

## How to Run

1. train base model with imagenet100 dataset
    - [Train Base Model (resnet18)](tmo/base_model/README.md)

2. Post-training quantization (PTQ) and export ptq onnx
    ```
    python onnx_export_ptq.py
    ```
    - no train 
    - calibration

3. generate tensorrt model
    ```
    python onnx2trt.py
    ```
- int8 ptq (Explicit)   
    Gpu Mem: 124M   
    [TRT_E] Test Top-1 Accuracy: 84.20%   
    [TRT_E] Test Top-5 Accuracy: 97.06%   
    [TRT_E] 10000 iterations time: 6.4837 [sec]   
    [TRT_E] Average FPS: 1542.34 [fps]   
    [TRT_E] Average inference time: 0.65 [msec]   

## Reference

- [TensorRT-Model-Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer)
