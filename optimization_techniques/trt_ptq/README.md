# TesnorRT implict Post-training quantization

## How to Run

1. train base model with imagenet100 dataset
    - [Train Base Model (resnet18)](tmo/base_model/README.md)

2. export input onnx
    - [Export input onnx](tmo/base_trt/README.md)

3. gen calibration data
    ```
    python gen_calib_data.py
    ```

4. generate tensorrt model (TesnorRT implict Post-training quantization)
    ```
    python onnx2trt.py
    ```
    - calibration

- int8 tensorrt ptq (Implict)   
    Gpu Mem: 124M   
    [TRT_E] Test Top-1 Accuracy: 84.34%   
    [TRT_E] Test Top-5 Accuracy: 97.10%   
    [TRT_E] 10000 iterations time: 5.2218 [sec]   
    [TRT_E] Average FPS: 1915.04 [fps]   
    [TRT_E] Average inference time: 0.52 [msec]   

## Reference

- [TensorRT-implicit-quantization](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html#implicit-quantization)
