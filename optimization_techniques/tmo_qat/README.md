# Quantization-aware training (QAT)

## How to Run

1. train base model with imagenet100 dataset
    - [Train Base Model (resnet18)](tmo/base_model/README.md)

2. Quantization-aware training (QAT) and export qat onnx
    ```
    python onnx_export_qat.py
    ```
    - fine tuning (We recommend QAT for 10% of the original training epochs)

3. generate tensorrt model
    ```
    python onnx2trt.py
    ```
- int8 qat (Explicit)   
    Gpu Mem: 138M   
    [TRT_E] Test Top-1 Accuracy: 84.42%   
    [TRT_E] Test Top-5 Accuracy: 97.10%   
    [TRT_E] 10000 iterations time: 6.3580 [sec]  
    [TRT_E] Average FPS: 1572.81 [fps]   
    [TRT_E] Average inference time: 0.64 [msec]   

## Reference

- [TensorRT-Model-Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer)
