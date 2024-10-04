# gradcam tensorrt

## How to Run

1. generate .onnx from timm model and extract fc_weights

```
pip install torch
pip install onnx
pip install timm
pip install cuda-python
pip install tensorrt
pip install onnx-simplifier

python onnx_export.py
// a file 'resnet18_cuda.onnx' and 'resnet18_fc_weights.bin' will be generated in onnx directory.
```

2. modify 'resnet18_cuda.onnx'

```
pip install onnx_graphsurgeon

python gs.py
// a file 'resnet18_cuda_modified.onnx' will be generated in onnx directory.
```

3. build tensorrt model and run

```
python onnx2trt.py
// a file 'resnet18_fp16.engine' will be generated in engine directory.
// panda0_heatmap.jpg and panda0_mix_cam_ori.jpg will be generated in results directory.
```
