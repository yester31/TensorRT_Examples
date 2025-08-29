# timm_to_trt_python1

## How to Run

1. generate .onnx from timm model

```
pip install torch
pip install onnx
pip install timm
pip install cuda-python
pip install tensorrt
pip install opencv-python

python onnx_export.py
// a file 'resnet18_cuda.onnx' will be generated in onnx directory.
```

2. build tensorrt model and run

```
python onnx2trt.py
// a file 'resnet18_fp16.engine' will be generated in engine directory.
```
