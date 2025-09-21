# YOLOX: Exceeding YOLO Series in 2021

## Preparation steps before running a TensorRT demo

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

## DEMOS   
### 1. [YOLOX Python Version](python/README.md)    
### 2. [YOLOX Cpp Version](cpp/README.md)   
### 3. [YOLOX Cpp Version 2](cpp2/README.md)   

----
- [YOLOX: Exceeding YOLO Series in 2021](https://arxiv.org/pdf/2107.08430)
- [YOLOX official GitHub](https://github.com/Megvii-BaseDetection/YOLOX)
