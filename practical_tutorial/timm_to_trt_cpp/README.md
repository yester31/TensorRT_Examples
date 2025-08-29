# timm_to_trt_cpp

## How to Run

1. generate .onnx from timm model

```
pip install torch
pip install onnx
pip install timm
python onnx_export.py
// a file 'resnet18_cuda.onnx' will be generated in onnx directory.
```

2. build tensorrt model and run

```
mkdir build
cd build
cmake ..
make
./timm_to_trt_cpp
// a file 'resnet18_16.engine' will be generated in engine directory.
```
