# depth_estimation_trt

## How to Run

1. set target model

```
git clone https://github.com/apple/ml-depth-pro
cd ml-depth-pro
conda create -n depth-pro -y python=3.9
conda activate depth-pro
pip install -e .
source get_pretrained_models.sh   # Files will be downloaded to `checkpoints` directory.
python infer.py
```

2. generate .onnx from timm model

```
cd ..
pip install onnx
pip install cuda-python
pip install tensorrt

python onnx_export.py
// a file 'dinov2l16_384_cuda.onnx' will be generated in onnx directory.
```

3. build tensorrt model and run

```
python onnx2trt.py
// a file 'dinov2l16_384_fp16.engine' will be generated in engine directory.
```

https://github.com/apple/ml-depth-pro
