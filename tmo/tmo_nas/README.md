# NAS(Neural Architecture Search)

## How to Run

1. train base model with imagenet100 dataset
    - [Train Base Model (resnet18)](tmo/base_model/README.md)

2. NAS(Neural Architecture Search) and export nas onnx
    ```
    python onnx_export_nas.py
    ```
    1. Convert Model
    2. NAS traing
    3. Subnet architecture search 
    4. Fine-tuning

3. generate tensorrt model
    ```
    python onnx2trt.py
    ```
- fp16 nas
- Gpu Mem: M

## Reference

- [TensorRT-Model-Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer)
