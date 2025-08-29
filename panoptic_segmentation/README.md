# Panoptic Segmentation (EoMT)

## How to Run

1. set up a virtual environment.
    ```
    conda create -n eomt -y python=3.11
    conda activate eomt
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
    pip install transformers
    pip install onnxsim
    pip install onnx
    pip install onnxscript
    pip install opencv-python
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

- [Your ViT is Secretly an Image Segmentation Model](https://arxiv.org/pdf/2503.19108)
- [EoMT official GitHub](https://github.com/tue-mps/eomt)

