# Examples of TensorRT models using ONNX

All useful sample codes of TensorRT models using ONNX


## 0. Development Environment

- RTX3060 (notebook)
- WSL 
- Ubuntu 22.04.5 LTS
- cuda 12.8

conda deactivate 
conda env remove -n trte -y 
```
conda create -n trte python=3.11 --yes 
conda activate trte

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
pip install cuda-python==12.9.2
pip install tensorrt-cu12
pip install onnx
pip install opencv-python
pip install timm
pip install matplotlib

pip install -U "nvidia-modelopt[all]"

# Check installation 
python -c "import modelopt; print(modelopt.__version__)"
python -c "import modelopt.torch.quantization.extensions as ext; ext.precompile()"
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
   [5.0 Train Base Model (resnet18)](tmo/base_model/README.md)  
   [5.1 Base TensorRT (fp16)](tmo/base_trt/README.md)  
   [5.2 Explict Quantization (PTQ)](tmo/tmo_ptq/README.md)  
   [5.3 Explict Quantization (QAT)](tmo/tmo_qat/README.md)  
   [5.4 Explict Quantization (ONNX PTQ)](tmo/tmo_moq/README.md)  
   [5.5 Implicit Quantization (TensorRT PTQ)](tmo/trt_ptq/README.md)  
   [5.6 Sparsity (2:4 sparsity)](tmo/tmo_sparsity/README.md)  
   [5.7 Pruning](tmo/tmo_pruning/README.md)  
   [5.8 NAS(work in progress...)](tmo/tmo_nas/README.md)  
   5.9 Multiple Optimizations Techniques      
      5.9.1 (Pruning + Sparsity)   
      5.9.2 (Pruning + Sparsity + Quantization(QAT))   


<table border="1" cellspacing="0" cellpadding="4">
  <thead>
    <tr>
      <th>Framework</th>
      <th>PyTorch</th>
      <th>TensorRT</th>
      <th>TensorRT</th>
      <th>TensorRT</th>
      <th>TensorRT</th>
      <th>TensorRT</th>
      <th>TensorRT</th>
      <th>TensorRT</th>
    </tr>
  </thead>
  <tbody>
      <tr>
      <td>Opti Technique</td>
      <td>-</td>
      <td>-</td>
      <td>trt ptq (Implicit)</td>
      <td>onnx ptq (Explict)</td>
      <td>tmo ptq (Explict)</td>
      <td>tmo qat (Explict)</td>
      <td>tmo sparsity</td>
      <td>tmo pruning (flops 80%)</td>
    </tr>
    <tr>
      <td>Precision</td>
      <td>fp16</td>
      <td>fp16</td>
      <td>int8</td>
      <td>int8</td>
      <td>int8</td>
      <td>int8</td>
      <td>fp16</td>
      <td>fp16</td>
    </tr>
    <tr>
      <td>Top-1 Acc [%]</td>
      <td>84.58</td>
      <td>84.54</td>
      <td>84.34</td>
      <td>84.5</td>
      <td>84.2</td>
      <td>84.42</td>
      <td>83.28</td>
      <td>82.76</td>
    </tr>
    <tr>
      <td>Top-5 Acc [%]</td>
      <td>97.2</td>
      <td>97.2</td>
      <td>97.1</td>
      <td>97</td>
      <td>97.06</td>
      <td>97.1</td>
      <td>96.72</td>
      <td>96.42</td>
    </tr>
    <tr>
      <td>FPS [Frame/sec]</td>
      <td>406.27</td>
      <td>1463.45</td>
      <td>1915.04</td>
      <td>1897.46</td>
      <td>1542.34</td>
      <td>1572.81</td>
      <td>1483.85</td>
      <td>1573.2</td>
    </tr>
    <tr>
      <td>Avg Latency [ms]</td>
      <td>2.46</td>
      <td>0.68</td>
      <td>0.52</td>
      <td>0.53</td>
      <td>0.65</td>
      <td>0.64</td>
      <td>0.67</td>
      <td>0.64</td>
    </tr>
    <tr>
      <td>GPU Mem [MB]</td>
      <td>286</td>
      <td>138</td>
      <td>124</td>
      <td>124</td>
      <td>124</td>
      <td>138</td>
      <td>138</td>
      <td>130</td>
    </tr>
  </tbody>
</table>

## 3. Advanced step

6. Super Resolution  
   [6.1 Real-ESRGAN](super_resolution_trt/README.md)
7. Object Detection  
   [7.1 yolo11](object_detection1/README.md)
8. Instance Segmentation
9. Semantic Segmentation
10. Depth Estimation  
     [10.1 Depth Pro](depth_estimation_trt/README.md)

## 4. reference

- [TensorRT](https://github.com/NVIDIA/TensorRT)
- [TensorRT-Model-Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer)
