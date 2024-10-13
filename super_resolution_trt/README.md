# super_resolution_trt

## How to Run

1. set target model

```
git clone https://github.com/xinntao/Real-ESRGAN.git
cd Real-ESRGAN
# Install basicsr - https://github.com/xinntao/BasicSR
# We use BasicSR for both training and inference
pip install basicsr
# facexlib and gfpgan are for face enhancement
pip install facexlib
pip install gfpgan
pip install -r requirements.txt
python setup.py develop
```

2. generate .onnx from timm model

```
cd ..
pip install onnx
pip install cuda-python
pip install tensorrt
pip install onnx-simplifier

python onnx_export.py
// a file 'RealESRGAN_x4plus_cuda.onnx' will be generated in onnx directory.
```

3. build tensorrt model and run

```
python onnx2trt.py
// a file 'resnet18_fp16.engine' will be generated in engine directory.
```

https://github.com/xinntao/Real-ESRGAN
