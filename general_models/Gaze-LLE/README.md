# Gaze Target Estimation (Gaze-LLE)

## How to Run

1. set up a virtual environment.
    ```
    git clone https://github.com/fkryan/gazelle.git
    cd gazelle

    conda create -n gazelle -y python=3.11
    conda activate gazelle
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
    pip install transformers
    pip install onnxsim
    pip install onnx
    pip install onnxscript
    pip install opencv-python
    # pip install -e .
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

- [Gaze-LLE: Gaze Target Estimation via Large-Scale Learned Encoders](https://arxiv.org/abs/2412.09586)
- [Gaze-LLE official GitHub](https://github.com/fkryan/gazelle)

