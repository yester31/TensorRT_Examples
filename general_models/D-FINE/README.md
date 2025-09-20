# D-FINE

## How to Run

1. set up a virtual environment.
    ```
    git clone https://github.com/Peterande/D-FINE.git
    cd D-FINE
    conda create -n dfine -y python=3.11
    conda activate dfine
    pip install -r requirements.txt
    pip install opencv-python
    pip install onnx
    pip install onnxscript
    pip install onnxsim
    ```

2. download pretrained checkpoints.
    ```
    mkdir -p checkpoints
    wget https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_n_coco.pth -P checkpoints
    wget https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_s_obj2coco.pth -P checkpoints
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

- fp16 (dfine_hgnetv2_s)   
    [TRT_E] 1000 iterations time: 4.7764 [sec]   
    [TRT_E] Average FPS: 209.36 [fps]   
    [TRT_E] Average inference time: 4.78 [msec]   
    GPU mem : 186M   

- [D-FINE: Redefine Regression Task of DETRs as Fine‑grained Distribution Refinement](https://arxiv.org/pdf/2410.13842)
- [D-FINE official GitHub](https://github.com/Peterande/D-FINE)

