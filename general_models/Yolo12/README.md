# Face Detection (Yolo12)

## How to Run

1. set up a virtual environment.
    ```
    pip install ultralytics
    pip install datasets
    ```

2. check pytorch model inference
    ```
    cd ..
    python infer.py
    ```

## How to Run (TensorRT)

1. generate onnx file
    ```
    python onnx_export.py
    ```

2. generate tensorrt model
    ```
    python onnx2trt.py
    ```
    
- fp16 (yolo12n)    
    [TRT_E] 1000 iterations time: 4.4858 [sec]  
    [TRT_E] Average FPS: 222.92 [fps]   
    [TRT_E] Average inference time: 4.49 [msec]     
    GPU mem : 164M      


- [Yolo12 official GitHub](https://github.com/sunsmarterjie/yolov12)

