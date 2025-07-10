# Examples of TensorRT models using ONNX

All useful sample codes of TensorRT models using ONNX


## 0. Development Environment

- RTX3060 (notebook)
- WSL 
- Ubuntu 22.04.5 LTS
- cuda 12.8


```
conda create -n trte python=3.12 --yes 
conda activate trte

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install cuda-python
pip install tensorrt
pip install onnx

```

## 1. Basic step

1. Generation TensorRT Model by using ONNX  
   [1.1 TensorRT CPP API](timm_to_trt_cpp/README.md)  
   [1.2 TensorRT Python API](timm_to_trt_python1/README.md)  
   [1.3 Polygraphy](timm_to_trt_python2/README.md)

2. Dynamic shapes for TensorRT  
   [2.1 Dynamic batch](dynamic_batch_trt/README.md)  
   [2.2 Dynamic input size](dynamic_input_size_trt/README.md)

## 2. Intermediate step

3. Custom Plugin  
   [3.1 Adding a pre-processing layer by cuda](custom_layer/README.md)

4. Modifying an ONNX graph by ONNX GraphSurgeon  
   [4.1 Extracting a feature map of the last Conv for Grad-Cam](gradcam_trt/README.md)  
   4.2 Generating a TensorRT model with a custom plugin and ONNX

5. TensorRT Model Optimizer  
   [5.1 Explict Quantization (PTQ)](tmo/tmo_ptq/README.md)  
   [5.2 Explict Quantization (QAT)](tmo/tmo_qat/README.md)  
   [5.3 Explict Quantization (ONNX PTQ)](tmo/tmo_moq/README.md)  
   [5.4 Implicit Quantization (TensorRT PTQ)](trt_quantization/README.md)  
   [5.5 Sparsity (2:4 sparsity pattern)](tmo/tmo_sparsity/README.md)  
   5.6 Pruning  
   5.7 Distillation  
   5.8 NAS(Neural Architecture Search)  
   5.9 Combinations multi-method

## 3. Advanced step

6. Super Resolution  
   [6.1 Real-ESRGAN](super_resolution_trt/README.md)
7. Object Detection  
   [7.1 yolo11](object_detection1/README.md)
8. Instance Segmentation
9. Semantic Segmentation
10. Depth Estimation  
     [10.1 Depth Pro](depth_estimation_trt/README.md) (
    "It is under repair due to an accuracy issue.")

## 4. reference

- [TensorRT](https://github.com/NVIDIA/TensorRT)
- [TensorRT-Model-Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer)
