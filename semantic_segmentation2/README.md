# Semantic Segmentation (BEN2:Background Erase Network)

## How to Run

1. set up a virtual environment.
    ```
    git clone https://github.com/PramaLLC/BEN2.git
    cd BEN2
    conda create -n ben2 -y python=3.11
    conda activate ben2
    pip install git+https://github.com/PramaLLC/BEN2.git
    pip install onnxsim
    pip install onnx
    pip install onnxscript
    pip install opencv-python
    ```

2. download checkpoint
    ```
    mkdir -p checkpoint
    wget https://huggingface.co/PramaLLC/BEN2/resolve/main/BEN2_Base.pth -P checkpoint
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
- [TRT_E] 100 iterations time: 3.6876 [sec]
- [TRT_E] Average FPS: 27.12 [fps]
- [TRT_E] Average inference time: 36.88 [msec]


- [BEN: Using Confidence-Guided Matting for Dichotomous Image Segmentation](https://arxiv.org/abs/2501.06230)
- [BEN2 official GitHub](https://github.com/PramaLLC/BEN2)

