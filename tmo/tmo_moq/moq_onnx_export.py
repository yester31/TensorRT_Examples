# by yhpark 2024-10-17
# ONNX PTQ example
import modelopt.torch.quantization as mtq
import modelopt.onnx.quantization as moq

import torch
import torch.onnx
import torch.nn as nn
import onnx
import os
import sys
import timm
import numpy as np 
import copy

from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import torchvision.datasets as datasets

sys.path.insert(1, os.path.join(sys.path[0], "../.."))
from utils import *
set_random_seed()

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
print(f"current file path: {current_file_path}")
print(f"current directory: {current_directory}")

def main():
    
    calib_data_dir_path = f'{current_directory}/calib_datas'
    os.makedirs(calib_data_dir_path, exist_ok=True)
    calibration_data_path = f"{calib_data_dir_path}/calib_data.npy"

    if os.path.exists(calibration_data_path) is False:
        batch_size = 512  
        workers = 4
        transform = transforms.Compose([
            transforms.Resize(256),             
            transforms.CenterCrop(224),         
            transforms.ToTensor(),              
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225]),
        ])

        # load dataset 
        calib_size = 512 # typically 128-512 samples
        train_dataset = datasets.ImageFolder(root=f'{current_directory}/../datasets/imagenet100/train', transform=transform)
        # Randomly select the specified number of dataset
        indices = np.random.choice(len(train_dataset), calib_size, replace=False)
        subset_train_dataset = Subset(train_dataset, indices)
        train_loader = DataLoader(dataset=subset_train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, sampler=None)
        class_count = len(train_dataset.classes)



        dataiter = iter(train_loader)
        images, labels = next(dataiter)
        calib_data = np.array(images)

        np.save(calibration_data_path, calib_data)

    calibration_data = np.load(calibration_data_path)

    input_onnx_path = f"{current_directory}/onnx/resnet18_cuda_bf.onnx"
    output_onnx_path = f"{current_directory}/onnx/resnet18_moq.onnx"
    moq.quantize(
        onnx_path=input_onnx_path,
        calibration_data=calibration_data,
        output_path=output_onnx_path,
        quantize_mode="int8",
    )

if __name__ == '__main__':
    main()