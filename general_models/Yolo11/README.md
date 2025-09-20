# object_detection1

## How to Run

1. set target model (yolo11n)
    ```
    git clone https://github.com/ultralytics/ultralytics
    cd ultralytics
    pip install ultralytics
    ```

2. generate .onnx from timm model
    ```
    cd ..
    python onnx_export.py     # pytorch model -> onnx(fp16 or fp32)
    ```

3. build tensorrt model and run
    ```
    python onnx2trt.py
    ```

- fp16   
    [TRT_E] 1000 iterations time: 3.4358 [sec]   
    [TRT_E] Average FPS: 291.05 [fps]   
    [TRT_E] Average inference time: 3.44 [msec]      
    GPU mem : 164M     

- [ultralytics](https://github.com/ultralytics/ultralytics)