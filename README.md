# Examples of TensorRT models using ONNX

All useful sample codes of TensorRT models using ONNX

## 0. Dependency

- cuda 12.2
- tensorrt 10.5.0
- torch 2.4.1

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
   5.2 Explict Quantization (QAT)  
   5.3 Sparsity (2:4 sparsity pattern)

## 3. Advanced step

6. Super Resolution  
   [6.1 Real-ESRGAN](super_resolution_trt/README.md)
7. Object Detection
8. Instance Segmentation
9. Semantic Segmentation
10. Depth Estimation  
     [10.1 Depth Pro](depth_estimation_trt/README.md) (
    "It is under repair due to an accuracy issue.")

## 4. reference

- [TensorRT-Github](https://github.com/NVIDIA/TensorRT)
