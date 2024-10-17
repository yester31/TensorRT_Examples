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

from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import torchvision.datasets as datasets

sys.path.insert(1, os.path.join(sys.path[0], ".."))
from utils import *
set_random_seed()

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
print(f"current file path: {current_file_path}")
print(f"current directory: {current_directory}")

# Print version information for debugging purposes
print(f"PyTorch version: {torch.__version__}")
print(f"ONNX version: {onnx.__version__}")

# Set device to GPU if available, otherwise fallback to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def main():
    
    # 2. load model
    ptq_mode = True
    model_name = "yolo11l.pt"
    model = YOLO(model_name).model    # yolov11 model load
    model.fuse()
    model = model.eval().to(device)              # set evaluation mode
    export_model_path = os.path.join(current_directory, 'onnx', f'{os.path.splitext(model_name)[0]}_{device.type}.onnx')

    if ptq_mode:
        # Quantization need calibration data. Setup calibration data loader
        batch_size = 32  
        workers = 4
        transform = transforms.Compose([
            transforms.Resize((640,640)),             
            transforms.ToTensor(),
        ])

        # load dataset 
        calib_size = 512 # typically 128-512 samples
        train_dataset = datasets.ImageFolder(root=f'{current_directory}/../tmo/datasets/imagenet100/train', transform=transform)
        # Randomly select the specified number of dataset
        indices = np.random.choice(len(train_dataset), calib_size, replace=False)
        subset_train_dataset = Subset(train_dataset, indices)
        train_loader = DataLoader(dataset=subset_train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, sampler=None)
        class_count = len(train_dataset.classes)
    
        # Select quantization config
        config = mtq.INT8_DEFAULT_CFG

        # Define forward_loop. Please wrap the data loader in the forward_loop
        def forward_loop(model):
            for i, (batch, target) in enumerate(train_loader):
                model(batch.to(device))

        # Quantize the model and perform calibration (PTQ)
        model = mtq.quantize(model, config, forward_loop)
        
        # Print quantization summary after successfully quantizing the model with mtq.quantize
        # This will show the quantizers inserted in the model and their configurations
        mtq.print_quant_summary(model)
        
        export_model_path = os.path.join(current_directory, 'onnx', f'{os.path.splitext(model_name)[0]}_{device.type}_ptq.onnx')


    # Ensure the export directory exists
    os.makedirs(os.path.dirname(export_model_path), exist_ok=True)
    
    # 1. input size
    batch_size = 1
    dummy_input = torch.randn(batch_size, 3, 640, 640, requires_grad=True).to(device)

    # 3. generate onnx file
    with torch.no_grad():
        torch.onnx.export(model,                   # pytorch model
                        dummy_input,               # model dummy input
                        export_model_path,         # onnx model path
                        opset_version=19,          # the version of the opset
                        input_names=['input'],     # input name
                        output_names=['output'],   # output name
                        dynamic_axes={
                            "input": {0: "batch_size", 2: "width", 3: "height"},
                            "output0": {0: "batch_size", 2: "anchors"}
                                    }
                        )  

        print("ONNX Model exported at ", export_model_path)

    # 4. check the produced onnx file
    onnx_model = onnx.load(export_model_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX Model check done!")

if __name__ == '__main__':
    main()