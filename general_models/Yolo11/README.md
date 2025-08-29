# object_detection1

## How to Run

1. set target model (yolo11l)

```
git clone https://github.com/ultralytics/ultralytics
cd ultralytics
pip install ultralytics
```

2. generate .onnx from timm model

```
cd ..
python onnx_export.py     # pytorch model -> onnx(fp16 or fp32)
python moq_onnx_export.py # ptq, pytorch model -> onnx(int8)
python ptq_onnx_export.py # ptq, onnx(fp16 or fp32, only static input shape) -> onnx(int8)
// a file 'yolo11l_cuda.onnx', 'yolo11l_cuda_ptq.onnx', or 'yolo11l_cuda_moq.onnx' will be generated in onnx directory.
```

3. build tensorrt model and run

```
python onnx2trt.py
// a file 'yolo11l_fp16_.engine', 'yolo11l_int8_ptq.engine', or 'yolo11l_int8_moq.engine' will be generated in engine directory.
```

https://github.com/ultralytics/ultralytics
