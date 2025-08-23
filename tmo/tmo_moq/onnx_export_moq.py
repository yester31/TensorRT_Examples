# by yhpark 2024-10-17
# by yhpark 2025-8-23
# ONNX PTQ example
import modelopt.onnx.quantization as moq
import torch
import os
import sys
import numpy as np 
import torchvision.transforms as transforms

sys.path.insert(1, os.path.join(sys.path[0], ".."))
from base_model.utils_tmo import *
set_random_seed()

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print('[TRT_E] gpu device count : ', torch.cuda.device_count())
    print('[TRT_E] device_name : ', torch.cuda.get_device_name(0))

def main():
    
    calibration_data_path = f"{CUR_DIR}/calib_datas/calib_data.npy"
    os.makedirs(os.path.dirname(calibration_data_path), exist_ok=True)

    if os.path.exists(calibration_data_path) is False:
        print("generate calib data")
        batch_size = 512  
        transform_test = transforms.Compose([
            transforms.Resize(256),             
            transforms.CenterCrop(224),         
            transforms.ToTensor(),              
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225]),
        ])

        # load dataset 
        test_loader = dataset_load(batch_size, transform_test, 'test')
        batch = next(iter(test_loader))
        calib_data = np.array(batch["image"])
        np.save(calibration_data_path, calib_data)

    calibration_data = np.load(calibration_data_path)

    print("export moq model")
    model_name = 'resnet18'
    input_onnx_path = f"{CUR_DIR}/../base_trt/onnx/{model_name}.onnx"
    output_onnx_path = os.path.join(CUR_DIR, 'onnx', f'{model_name}_optq.onnx')
    os.makedirs(os.path.dirname(output_onnx_path), exist_ok=True)

    moq.quantize(
        onnx_path=input_onnx_path,
        calibration_data=calibration_data,
        output_path=output_onnx_path,
        quantize_mode="int8",
    )

if __name__ == '__main__':
    main()