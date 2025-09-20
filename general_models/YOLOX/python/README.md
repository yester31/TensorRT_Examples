# Run YOLOX python demo

## How to Run

1. check pytorch model inference
    ```
    python infer.py
    ```

2. generate onnx file
    ```
    python onnx_export.py
    ```

3. generate tensorrt model
    ```
    python onnx2trt.py
    ```
- fp16   
    [TRT_E] 1000 iterations time: 3.3544 [sec]   
    [TRT_E] Average FPS: 298.12 [fps]   
    [TRT_E] Average inference time: 3.35 [msec]      
    GPU mem : 174M      