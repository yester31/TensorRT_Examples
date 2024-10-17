# tmo_moq

- no dynamic input shape

## How to Run

1. prepare imagenet100 dataset

```
cd ..
mkdir datasets

// download imagenet100 dataset from kaggle (see below)
```

2. train resnet18 with imagenet100 dataset

```
cd base_model
python train.py
// 'best_model.pth' will be generated in checkpoint directory.
```

3. generate onnx file

```
cd tmo_moq
python onnx_export.py
// a file 'resnet18_cuda_bf.onnx' will be generated in onnx directory.
```

4. ONNX Post-training quantization (PTQ)

```
python moq_onnx_export.py
// a file 'resnet18_moq.onnx' will be generated in onnx directory.
python onnx2trt.py
// a file 'resnet18_int8_moq.engine' will be generated in engine directory.
```

## Reference

- [TensorRT-Model-Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer)
- [imagenet100](https://www.kaggle.com/datasets/ambityga/imagenet100)
