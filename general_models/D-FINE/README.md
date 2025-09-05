# D-FINE

## How to Run

1. set up a virtual environment.
    ```
    git clone https://github.com/Peterande/D-FINE.git
    cd D-FINE
    conda create -n dfine -y python=3.11
    conda activate dfine
    pip install -r requirements.txt
    ```

mkdir -p checkpoints
wget https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_n_coco.pth -P checkpoints
wget https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_s_obj2coco.pth -P checkpoints

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

- [D-FINE: Redefine Regression Task of DETRs as Fine‑grained Distribution Refinement](https://arxiv.org/pdf/2410.13842)
- [D-FINE official GitHub](https://github.com/Peterande/D-FINE)

