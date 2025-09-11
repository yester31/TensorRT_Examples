# by yhpark 2024-10-16
# Object Detection Example with TensorRT Model Optimization
import modelopt.torch.quantization as mtq

from ultralytics import YOLO
import torch.onnx
import onnx
import math
import os 
import sys
import numpy as np 
import copy

import torchvision.transforms as transforms
from calib_data import dataset_load

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Print version information for debugging purposes
print(f"PyTorch version: {torch.__version__}")
print(f"ONNX version: {onnx.__version__}")

def checker_onnx(export_model_path):
    try:
        onnx_model = onnx.load(export_model_path)
        onnx.checker.check_model(onnx_model)
    except Exception as e:
        print(f"[MDET] failed onnx.checker.check_model() : {e}")
    finally:
        onnx.checker.check_model(export_model_path)

    for input in onnx_model.graph.input:
        print(f"[MDET] Input: {input.name}")
        for d in input.type.tensor_type.shape.dim:
            print("[MDET] dim_value:", d.dim_value, "dim_param:", d.dim_param)

    for output in onnx_model.graph.output:
        print(f"[MDET] Output: {output.name}")
        for d in output.type.tensor_type.shape.dim:
            print("[MDET] dim_value:", d.dim_value, "dim_param:", d.dim_param)

def simplify_onnx(export_model_path, export_model_sim_path):
    print("[MDET] Simplify exported onnx model")
    onnx_model = onnx.load(export_model_path)
    try:
        model_simplified, check = simplify(onnx_model)
        if not check:
            raise RuntimeError("[MDET] Simplified model is invalid.")
        onnx.save(model_simplified, export_model_sim_path)
        print(f"[MDET] simplified onnx model saved to: {export_model_sim_path}")
    except Exception as e:
        print(f"[MDET] simplification failed: {e}")
    checker_onnx(export_model_sim_path)

def main():
    
    batch_size = 1
    input_h, input_w = 640, 640 
    model_name = "yolov12n-face"
    model = YOLO(f'{CUR_DIR}/checkpoints/{model_name}.pt').model.to(DEVICE)  # load a pretrained YOLOv8n detection model
    model = model.eval() # set evaluation mode

    dynamic = False  # True or False 
    onnx_sim = True # True or False 
    model_name = f"{model_name}_{input_h}x{input_w}"
    model_name = f"{model_name}_dynamic" if dynamic else model_name
    model_name = f"{model_name}_ptq"
    export_model_path = os.path.join(CUR_DIR, 'onnx', f'{model_name}.onnx')

    transform_test = transforms.Compose([
        transforms.Resize((input_h, input_w)),             
        transforms.ToTensor(),
    ])
    test_loader = dataset_load(batch_size, transform_test, 'test')

    # Quantize the model
    print("Starting quantization...")
    # Select quantization config
    config = mtq.INT8_DEFAULT_CFG

    # Define forward_loop. Please wrap the data loader in the forward_loop
    def calibrate_fn(model):
        seen = 0
        for i, (image, target) in enumerate(test_loader):
            model(image.to(DEVICE))
            seen += image.size(0)
            if seen >= 1024:
                break

    # Quantize the model and perform calibration (PTQ)
    model = mtq.quantize(model, config, calibrate_fn)
    
    # Print quantization summary after successfully quantizing the model with mtq.quantize
    # This will show the quantizers inserted in the model and their configurations
    mtq.print_quant_summary(model)
        
    dummy_input = torch.randn((batch_size, 3, input_h, input_w), requires_grad=True).to(DEVICE)

    dynamic_axes = None 
    if dynamic:
        dynamic_axes={
            "input": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "anchors"}
            } 

    # generate onnx file
    with torch.no_grad():
        torch.onnx.export(model,                   # pytorch model
                        dummy_input,               # model dummy input
                        export_model_path,         # onnx model path
                        opset_version=20,          # the version of the opset
                        input_names=['input'],     # input name
                        output_names=['output'],   # output name
                        dynamic_axes=dynamic_axes
                        )  

        print("ONNX Model exported at ", export_model_path)

    print("[MDET] Validate exported onnx model")
    checker_onnx(export_model_path)
    if onnx_sim :
        export_model_sim_path = os.path.join(CUR_DIR, 'onnx', f'{model_name}_sim.onnx')
        simplify_onnx(export_model_path, export_model_sim_path)

if __name__ == '__main__':
    main()