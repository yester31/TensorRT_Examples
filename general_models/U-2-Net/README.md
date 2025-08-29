# Semantic Segmentation

## How to Run

1. download onnx 
    mkdir -p onnx
    wget https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx -P onnx

2. generate tensorrt model
    ```
    python onnx2trt.py
    ```

- [Sky-Segmentation-and-Post-processing](https://github.com/xiongzhu666/Sky-Segmentation-and-Post-processing)

