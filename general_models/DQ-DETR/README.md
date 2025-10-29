# DQ-DETR: DETR with Dynamic Query for Tiny Object Detection

## How to Run

1. set up a virtual environment.
    ```
    git clone https://github.com/hoiliu-0801/DQ-DETR.git
    cd DQ-DETR
    conda create -n dqdetr -y python=3.11
    conda activate dqdetr
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
    pip install opencv-python
    pip install onnx
    pip install onnxscript
    pip install onnxsim
    pip install gdown
    pip install pycocotools

    pip install cython
    pip install -r requirements.txt
    conda install -c conda-forge libstdcxx-ng

    pip install cython==0.29.36
    pip install "git+https://github.com/jwwangchn/cocoapi-aitod.git#subdirectory=aitodpycocotools"

    cd models/dqdetr/ops
    python setup.py build install
    # unit test (should see all checking is True)
    python test.py
    cd ../../..
    ```

2. download pretrained checkpoints.
    ```
    mkdir -p pretrained
    gdown --fuzzy https://drive.google.com/file/d/1zqN9xBBGt60wn3KB8kmfX0KALj1JwAGR/view?usp=sharing -O pretrained/
    gdown --fuzzy https://drive.google.com/file/d/1mF-nZURBOKJeeZXd_CsY174rDw8Brimn/view?usp=sharing -O pretrained/
    ```

2. check pytorch model inference
    ```
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
    
- fp16 ( -S)   
    
    GPU mem :  M   

- [DQ-DETR: DETR with Dynamic Query for Tiny Object Detection](https://arxiv.org/pdf/2404.03507)
- [DQ-DETR official GitHub](https://github.com/hoiliu-0801/DQ-DETR)

