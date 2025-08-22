# tmo_ptq

## How to Run

1. train base model with imagenet100 dataset
- [Train Base Model (resnet18)](tmo/base_model/README.md)

2. Post-training quantization (PTQ)

```

python ptq_onnx_export.py
python onnx2trt.py
// a file 'resnet18_int8_ptq_bf.engine' will be generated in engine directory.
```

## Reference

- [TensorRT-Model-Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer)
