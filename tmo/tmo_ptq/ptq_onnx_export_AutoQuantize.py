# by yhpark 2024-10-15
# TensorRT Model Optimization PTQ example
import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.algorithms import QuantRecipe
print(f"Weight compression for INT8_DEFAULT_CFG: {QuantRecipe('INT8_DEFAULT_CFG').compression}")
import modelopt.torch.opt as mto

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

# Print version information for debugging purposes
print(f"PyTorch version: {torch.__version__}")
print(f"ONNX version: {onnx.__version__}")

# Set device to GPU if available, otherwise fallback to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def main():
    
    batchNorm_folding = True
    # Quantization need calibration data. Setup calibration data loader
    batch_size = 32  
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
    
    model_name = "resnet18"
    if batchNorm_folding:
        export_model_path = os.path.join(current_directory, 'onnx', f'{model_name}_{device.type}_ptq_auto_bf.onnx')
    else:
        export_model_path = os.path.join(current_directory, 'onnx', f'{model_name}_{device.type}_ptq_auto.onnx')
    # Ensure the export directory exists
    os.makedirs(os.path.dirname(export_model_path), exist_ok=True)
    
    # Load the pre-trained model
    model = timm.create_model(model_name=model_name, num_classes=class_count, pretrained=True).to(device)
    model_path = f'{current_directory}/../base_model/checkpoint/best_model.pth'
    if torch.cuda.is_available():
        checkpoint = torch.load(model_path, map_location=device)
    else:
        checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    model.eval()  # Set model to evaluation mode
    
    checkpoint_dir_path = f'{current_directory}/checkpoint'  # Directory to save graphs and models
    os.makedirs(os.path.dirname(checkpoint_dir_path), exist_ok=True)

    modelopt_state_path = f"{checkpoint_dir_path}/modelopt_state.pt"
    if os.path.exists(modelopt_state_path):
        # Restore the previously saved modelopt state followed by model weights
        mto.restore_from_modelopt_state(model, torch.load(modelopt_state_path))  # Restore modelopt state
        model.load_state_dict(torch.load(f"{checkpoint_dir_path}/model_weights.pt"), ...)  # Load the model weights
    else :
        # Define loss function
        loss_f = nn.CrossEntropyLoss()
        def loss_func(outputs, inputs):
            return loss_f(outputs, inputs[1].to(device))
        
        data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, sampler=None)

        # Perform AutoQuantize
        model, search_state_dict = mtq.auto_quantize(
            model,
            data_loader,
            loss_func = loss_func,
            constraints = {"weight_compression": 0.50},
            # supported quantization formats are listed in `modelopt.torch.quantization.config.choices`
            quantization_formats = ["INT8_DEFAULT_CFG", None],
            )
        # Save the modelopt state and model weights separately
        torch.save(mto.modelopt_state(model), f"{checkpoint_dir_path}/modelopt_state.pt") # Save the modelopt state
        torch.save(model.state_dict(), f"{checkpoint_dir_path}/model_weights.pt") # Save the model weights
    
    if batchNorm_folding :
        # batch folding(conv + bn -> fused conv)
        model = fuse_bn_recursively(model) 
    model.to(device)  # Move the model to the chosen device

    # Print quantization summary after successfully quantizing the model with mtq.quantize
    # This will show the quantizers inserted in the model and their configurations
    mtq.print_quant_summary(model)
    
    # Get model input size from the model configuration
    input_size = model.pretrained_cfg["input_size"]
    dummy_input = torch.randn(input_size, requires_grad=False).unsqueeze(0).to(device)  # Create a dummy input

    # Export the model to ONNX format
    with torch.no_grad():  # Disable gradients for efficiency
        torch.onnx.export(
            model, 
            dummy_input, 
            export_model_path, 
            opset_version=19, 
            input_names=["input"], 
            output_names=["output"]
        )
        print(f"ONNX model exported to: {export_model_path}")

    # Verify the exported ONNX model
    onnx_model = onnx.load(export_model_path)
    onnx.checker.check_model(onnx_model)  # Perform a validity check
    print("ONNX model validation successful!")
    

# Run the main function
if __name__ == "__main__":
    main()