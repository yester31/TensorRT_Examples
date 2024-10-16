# tmo_qat

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

3. Quantization-aware training (QAT)

```
cd tmo_qat
python qat_onnx_export.py
python onnx2trt.py
// a file 'resnet18_int8_qat_bf.engine' will be generated in engine directory.
```

## Reference

- [TensorRT-Model-Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer)
- [imagenet100](https://www.kaggle.com/datasets/ambityga/imagenet100)
