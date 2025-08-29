# Semantic Segmentation (MODNet: Trimap-Free Portrait Matting in Real Time)

## How to Run

1. download onnx
    ```
    mkdir -p onnx
    wget -O onnx/MODNet.onnx https://huggingface.co/onnx-community/modnet-webnn/resolve/main/onnx/model.onnx
    ```

2. generate tensorrt model
    ```
    python onnx2trt.py
    ```
- [TRT_E] 100 iterations time: 0.9260 [sec]
- [TRT_E] Average FPS: 108.00 [fps]
- [TRT_E] Average inference time: 9.26 [msec]


- [MODNet: Real-Time Trimap-Free Portrait Matting via Objective Decomposition](https://arxiv.org/abs/2011.11961)
- [MODNet official GitHub](https://github.com/ZHKKKe/MODNet)

