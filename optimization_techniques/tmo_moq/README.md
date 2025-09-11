# ONNX Post-training quantization

## How to Run

1. train base model with imagenet100 dataset
    - [Train Base Model (resnet18)](tmo/base_model/README.md)

2. export input onnx
    - [Export input onnx](tmo/base_trt/README.md)

3. ONNX Post-training quantization and export moq onnx
    ```
    python onnx_export_moq.py
    ```
    - no train 
    - calibration

4. generate tensorrt model
    ```
    python onnx2trt.py
    ```
- int8 onnx ptq (Explicit)  
    Gpu Mem: 124M   
    [TRT_E] Test Top-1 Accuracy: 84.50%   
    [TRT_E] Test Top-5 Accuracy: 97.00%   
    [TRT_E] 10000 iterations time: 5.2702 [sec]   
    [TRT_E] Average FPS: 1897.46 [fps]   
    [TRT_E] Average inference time: 0.53 [msec]   

## Reference

- [TensorRT-Model-Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer)
