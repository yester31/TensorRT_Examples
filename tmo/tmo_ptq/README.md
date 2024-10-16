# tmo_ptq

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

3. Post-training quantization (PTQ)

```
cd tmo_ptq
python ptq_onnx_export.py
python onnx2trt.py
// a file 'resnet18_int8_ptq_bf.engine' will be generated in engine directory.
```

## Reference

- [TensorRT-Model-Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer)
- [imagenet100](https://www.kaggle.com/datasets/ambityga/imagenet100)
