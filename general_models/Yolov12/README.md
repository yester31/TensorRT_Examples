# Face Detection (Yolov12)

## How to Run

1. set up a virtual environment.
    ```
    pip install ultralytics
    pip install datasets
    ```

2. download pretrained checkpoints.
    ```
    mkdir -p checkpoints
    wget https://github.com/YapaLab/yolo-face/releases/download/v0.0.0/yolov12n-face.pt -P checkpoints
    ```

3. check pytorch model inference
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

3. post train qauntization (TensorRT Model Optimzation)
    ```
    python onnx_export_ptq.py
    python onnx2trt.py
    ```
    - precision = "int8" in onnx2trt.py
    - quant_method = "ptq" in onnx2trt.py

4. post train qauntization (ONNX)
    ```
    conda create -n moq python=3.11
    conda activate moq
    pip install onnx==1.16.0 nvidia-modelopt[onnx]
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
    pip install datasets

    python onnx_export_moq.py
    python onnx2trt.py
    ```
    - precision = "int8" in onnx2trt.py
    - quant_method = "moq" in onnx2trt.py

- [Yolo-Face official GitHub](https://github.com/YapaLab/yolo-face)

