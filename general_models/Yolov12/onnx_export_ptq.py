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
from infer import * 
from onnx_export import * 

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

def main():
    
    batch_size = 1
    input_h, input_w = 640, 640 
    # load model
    model_name = "yolov12n-face"
    checkpoint_path = f'{CUR_DIR}/checkpoints/{model_name}.pt'
    class_count = 1
    model = YOLOv12(checkpoint_path, class_count)
    model = model.eval().to(DEVICE)

    onnx_sim = True # True or False 
    model_name = f"{model_name}_{input_h}x{input_w}"
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

    # generate onnx file
    with torch.no_grad():
        torch.onnx.export(model,                   # pytorch model
                        dummy_input,               # model dummy input
                        export_model_path,         # onnx model path
                        opset_version=20,          # the version of the opset
                        input_names=['input'],     # input name
                        output_names=["boxes", "scores"], 
                        )  

        print("ONNX Model exported at ", export_model_path)

    print("[MDET] Validate exported onnx model")
    checker_onnx(export_model_path)
    if onnx_sim :
        export_model_sim_path = os.path.join(CUR_DIR, 'onnx', f'{model_name}_sim.onnx')
        simplify_onnx(export_model_path, export_model_sim_path)

    max_output_boxes = 300
    iou_threshold = 0.7
    score_threshold = 0.01
    yolo_insert_nms(
        path=export_model_sim_path,
        score_threshold=score_threshold,
        iou_threshold=iou_threshold,
        max_output_boxes=max_output_boxes,
    )

if __name__ == '__main__':
    main()