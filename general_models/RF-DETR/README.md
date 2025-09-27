# RF-DETR: SOTA Real-Time Object Detection Model

## How to Run

1. set up a virtual environment.
    ```
    git clone https://github.com/roboflow/rf-detr.git
    cd rf-detr
    conda create -n rfdetr -y python=3.11
    conda activate rfdetr
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
    pip install rfdetr
    pip install opencv-python
    pip install onnx
    pip install onnxscript
    pip install onnxsim
    ```

2. check pytorch model inference
    ```
    cd ..
    python infer.py
    ```

## How to Run (TensorRT)

1. generate onnx file
    ```
    python onnx_export_new_ln.py
    ```

2. generate tensorrt model
    ```
    python onnx2trt.py
    ```

- fp32 (rf_detr_nano)   
    [TRT_E] 1000 iterations time: 7.9187 [sec]
    [TRT_E] Average FPS: 126.28 [fps]
    [TRT_E] Average inference time: 7.92 [msec]
    GPU mem : 250M   
- fp16 (rf_detr_nano)   
    [TRT_E] 1000 iterations time: 3.5675 [sec]
    [TRT_E] Average FPS: 280.31 [fps]
    [TRT_E] Average inference time: 3.57 [msec]
    GPU mem : 190M   

- [RF-DETR: SOTA Real-Time Object Detection Model](https://blog.roboflow.com/rf-detr/)
- [RF-DETR official GitHub](https://github.com/roboflow/rf-detr)

