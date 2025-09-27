# DEIM: DETR with Improved Matching for Fast Convergence

## How to Run

1. set up a virtual environment.
    ```
    git clone https://github.com/Intellindust-AI-Lab/DEIM.git
    cd DEIM
    conda create -n deim -y python=3.11
    conda activate deim
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
    pip install opencv-python
    pip install onnx
    pip install onnxscript
    pip install onnxsim
    pip install gdown
    pip install -r requirements.txt
    ```

2. download pretrained checkpoints.
    ```
    mkdir -p pretrained
    gdown --fuzzy https://drive.google.com/file/d/153_JKff6EpFgiLKaqkJsoDcLal_0ux_F/view?usp=drive_link -O pretrained/
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
    
- fp16 (DEIM-RT-DETRv2-S)   
    [TRT_E] 1000 iterations time: 6.0511 [sec]  
    [TRT_E] Average FPS: 165.26 [fps]   
    [TRT_E] Average inference time: 6.05 [msec]     
    GPU mem : 236M   

- [DEIM: DETR with Improved Matching for Fast Convergence](https://arxiv.org/pdf/2412.04234)
- [DEIM official GitHub](https://github.com/Intellindust-AI-Lab/DEIM)

