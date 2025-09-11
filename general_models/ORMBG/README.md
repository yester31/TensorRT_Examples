# Semantic Segmentation (ormbg(Open Remove Background Model))

## How to Run

1. set up a virtual environment.
    ```
    git clone https://github.com/schirrmacher/ormbg.git
    cd ormbg
    conda create -n ormbg -y python=3.11
    conda activate ormbg
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    pip install onnxsim
    pip install onnx
    pip install onnxscript
    pip install opencv-python
    pip install scikit-image
    ```

2. download checkpoint
    ```
    mkdir -p checkpoint
    wget https://huggingface.co/schirrmacher/ormbg/resolve/main/models/ormbg.pth -P checkpoint
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
- input size : 512   
    [TRT_E] 100 iterations time: 0.6925 [sec]   
    [TRT_E] Average FPS: 144.40 [fps]   
    [TRT_E] Average inference time: 6.93 [msec]   
- input size : 1024   
    [TRT_E] 100 iterations time: 1.8420 [sec]   
    [TRT_E] Average FPS: 54.29 [fps]   
    [TRT_E] Average inference time: 18.42 [msec]   

- [ormbg official GitHub](https://github.com/schirrmacher/ormbg)

