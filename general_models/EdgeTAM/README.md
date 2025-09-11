# EdgeTAM: On-Device Track Anything Model

## How to Run

1. set up a virtual environment.
    ```
    git clone https://github.com/facebookresearch/EdgeTAM.git
    cd EdgeTAM
    conda create -n edgetam -y python=3.12
    conda activate edgetam
    pip install -e .
    pip install -e ".[notebooks]"
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

- [EdgeTAM: On-Device Track Anything Model](https://arxiv.org/pdf/2501.07256)
- [EdgeTAM official GitHub](https://github.com/facebookresearch/EdgeTAM)

