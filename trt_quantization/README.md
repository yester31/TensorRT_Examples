# trt_quantization

- implicit quantization (PTQ) TensorRT example

## How to Run

1. generate .onnx from timm model

```
pip install torch
pip install onnx
pip install timm
pip install cuda-python
pip install tensorrt
pip install onnx-simplifier

python onnx_export.py
// a file 'resnet18_cuda.onnx' will be generated in onnx directory.
```

2. prepare calibration datas

```
mkdir calib_data
// insert 3-500 calib datas for model (no need label)
```

3. build tensorrt model and run

```
python onnx2trt.py
// a file 'resnet18_int8.engine' will be generated in engine directory.
```
