import torch
import random
import numpy as np
import cv2
import os
from datetime import datetime

import segmentation_models_pytorch as smp
import torch.onnx
import onnx


def check_device():
    """
    GPU가 사용 가능한지 확인하는 유틸리티 함수
    :return: 사용 가능한 디바이스 ('cuda' 또는 'cpu')
    """
    print("[check device]")
    if torch.cuda.is_available():
        print(f"Torch gpu available : {torch.cuda.is_available()}")
        print(f"The number of gpu device : {torch.cuda.device_count()}")
        for g_idx in range(torch.cuda.device_count()):
            print(f"{g_idx} device name : {torch.cuda.get_device_name(g_idx)}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu:0")
    print(f"device : {device} is available")
    return device


if __name__ == '__main__':
    
    # 0. 기본 세팅
    current_file_path = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file_path)
    print(f"current file path: {current_file_path}")
    print(f"current directory: {current_directory}")
    device = check_device() # 디바이스 확인
    
    # 1. 모델 초기화
    model_name = 'UnetPlusPlus'
    model_path = os.path.join(current_directory, 'checkpoints', model_name, 'seg_model_best.pth.tar')
    checkpoint = torch.load(model_path, map_location=device)
    model = smp.create_model(
        arch=model_name,                # name of the architecture, e.g. 'Unet'/ 'FPN' / etc. Case INsensitive!
        encoder_name=checkpoint['encoder_name'],      # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights= checkpoint['encoder_weights'],     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=1,                      # model output channels (number of classes in your dataset)
    ).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval().to(device)

    export_model_path = os.path.join(current_directory, 'onnx', '{}_{}_{}.onnx'.format(model_name,checkpoint['encoder_name'], device.type ))

    # Ensure the export directory exists
    os.makedirs(os.path.dirname(export_model_path), exist_ok=True)
    
    # Get model input size from the model configuration
    input_size = [1,256,256]
    dummy_input = torch.randn(input_size, requires_grad=False).unsqueeze(0).to(device)  # Create a dummy input

    # Export the model to ONNX format
    with torch.no_grad():  # Disable gradients for efficiency
        torch.onnx.export(
            model, 
            dummy_input, 
            export_model_path, 
            opset_version=19, 
            input_names=["input"], 
            output_names=["output"], 
            #dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}  # Allow variable batch size
        )
        print(f"ONNX model exported to: {export_model_path}")

    # Verify the exported ONNX model
    onnx_model = onnx.load(export_model_path)
    onnx.checker.check_model(onnx_model)  # Perform a validity check
    print("ONNX model validation successful!")