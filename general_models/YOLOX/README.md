# YOLOX: Exceeding YOLO Series in 2021

## How to Run

1. set up a virtual environment.
    ```
    git clone https://github.com/Megvii-BaseDetection/YOLOX.git
    cd YOLOX
    conda create -n yolox -y python=3.11
    conda activate yolox
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129

    # comment line 18 in requirements.txt (# onnx-simplifier==0.4.10)
    pip install -v -e .
    pip install opencv-python
    pip install onnx
    pip install onnxscript
    pip install onnxsim
    ```

2. download pretrained checkpoints.
    ```
    mkdir -p pretrained
    wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth -P pretrained
    ```
3. run demo from original repository
    ```
    python tools/demo.py image -n yolox-s -c checkpoints/yolox_s.pth --path assets/dog.jpg --conf 0.25 --nms 0.45 --tsize 640 --save_result --device gpu
    ```

4. check pytorch model inference
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
- fp16   
    [TRT_E] 1000 iterations time: 3.3544 [sec]   
    [TRT_E] Average FPS: 298.12 [fps]   
    [TRT_E] Average inference time: 3.35 [msec]      
    GPU mem : 174M      


- [YOLOX: Exceeding YOLO Series in 2021](https://arxiv.org/pdf/2107.08430)
- [YOLOX official GitHub](https://github.com/Megvii-BaseDetection/YOLOX)

- TODO   
    - Replace EfficientNMS with INMSLayer   
        - Define a custom nn.Module using torchvision.ops.nms, then export it to ONNX.   
        - Build a standalone network using the INMSLayer API.   